r"""Contract two MULTI-STRUCTURED operands block-wise — the ``B@B``-multi case.

This module owns the vertex-elimination contraction where BOTH operands carry
meta-block-diagonal (``DiagonalIndex`` / "B") structure on their contracted
sides, over POSSIBLY SEVERAL contracted pairs at once, optionally coexisting
with plain-Dense ("D") contracted pairs and Dense<->Block ("D_B") contracted
pairs.  The existing pairwise kernels each own ONE structured contracted pair
(``contract_B_B`` is a single ``B@B`` pair; ``contract_dense_multi_block_diagonal``
requires ONE operand fully dense).  The case where EACH operand carries >=1
block-diagonal pair on the contracted side (the measured ViT / attention
signatures ``('B','B','B','B')``, ``('B','D','B','D')``, ...) fell through to the
densifying ``composed_dense`` fallback, which materializes BOTH operands and
loses all block-diagonal sparsity.  This kernel keeps it sparse.

================================================================================
1. MATH
================================================================================
Vertex elimination contracts ``lhs.primal_dims`` against ``rhs.out_dims`` over a
set of aligned (by ``_align_contract_dims``) pairs.  Each contracted pair is one
of:

  * ``B@B`` — lhs carries a DiagonalIndex pair (SURVIVING free out-side dim,
    meta ``N_p`` block ``Po_p``  <->  CONTRACTED primal-side dim, meta ``N_p``
    block ``Kc_p``) and rhs carries a DiagonalIndex pair (CONTRACTED out-side,
    meta ``N_p`` block ``Kc_p``  <->  SURVIVING free primal-side, meta ``N_p``
    block ``Qf_p``).  Both block-diagonal: a contracted position
    ``k = g_p*Kc_p + i_p`` is non-zero on the LHS only inside meta ``g_p`` and on
    the RHS only inside meta ``g_p`` (the diagonal restriction shares ``g_p``).
  * ``dense`` — both sides plain Dense over a shared logical axis ``Kd``; an
    ordinary contraction index.
  * ``D_B`` — one side Dense, the other a DiagonalIndex; the Block side's free
    partner SURVIVES (as a DENSE axis ``N*Bf`` — the ``D@B``/``B@D`` closure).

KEY CLOSURE FACT.  A ``B@B`` pair's two surviving partners (lhs out block
``Po_p`` and rhs primal block ``Qf_p``) BOTH index meta ``g_p``, and the
diagonal restriction forces them to the SAME ``g_p`` even when a coexisting
``dense`` / ``D_B`` pair is contracted jointly (that pair only sums WITHIN the
shared meta).  So **each ``B@B`` pair survives meta-block-diagonal in the
result** (meta ``N_p``, blocks ``Po_p`` x ``Qf_p``) — verified: the off-meta
cells of the result are exactly zero.  A ``D_B`` survivor and any free dense dim
are plain dense; ``dense`` pairs vanish.

The whole contraction is one batched einsum: BATCH over every shared ``B@B`` meta
``g_p`` (it is the SAME index on lhs out, lhs con, rhs con, rhs primal),
CONTRACT every ``B@B`` block axis ``i_p``, every ``dense`` axis ``k_d`` and every
``D_B`` contracted axis.  Cost ``~ nnz`` — never the dense outer product.

================================================================================
2. ALGORITHM
================================================================================
Pack each operand's ``val`` (read via ``.axis`` / ``.block_axis`` indirection —
the physical layout is implementation-defined) into the einsum-subscript order,
then ONE ``jnp.einsum``.  Per ``B@B`` pair we assign a shared meta letter ``g_p``
(BATCH — appears on both operands AND the output, once per side), a contract
letter ``i_p`` (both operands, not output), and the surviving block letters
``r_p`` (lhs out) / ``c_p`` (rhs primal) on the output.  The output keeps each
``B@B`` pair PACKED (``g_p`` once, ``r_p``, ``c_p``) so the result is genuinely
block-diagonal, never the dense block-diagonal matrix.  ``dense`` / ``D_B``
contracted axes get a shared contract letter; ``D_B`` survivors and free dense
dims ride through as plain dense letters.

Output dims are emitted in NATIVE order — lhs's surviving out dims (in
``lhs.out_dims`` order) then rhs's surviving out dims, then lhs's surviving
primal then rhs's surviving primal — with ids renumbered ``0..n_out-1`` (out) /
``n_out..`` (primal).  This matches the dispatcher's composed-dense fallback for
the contractions that actually occur (validated against it in the unit tests and
end-to-end cosine).

================================================================================
3. SCOPE / FALLBACK
================================================================================
Returns ``None`` (dispatcher keeps the always-correct ``composed_dense``
fallback) when: a ``B@B`` pair has MISMATCHED meta (``N_p != M_p``); a contracted
dim is implicit/compressed; a surviving dim is a free ride-through DiagonalIndex
pair (both sides survive); or an axis can't be located in ``val``.  Both operands
are assumed zero-fill (the structured Jacobian case); the dispatcher reroutes
non-zero fills before reaching here.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

import jax.numpy as jnp

from graphax.sparse.dtype_compute import _scaled_mul
from graphax.sparse.elemental._common import is_block_diagonal
from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.ops.utils import _compute_dtype

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


# --------------------------------------------------------------------------- #
# Pair / dim resolution
# --------------------------------------------------------------------------- #
def _partner(st: "SparseTensor", dim):
    """The dim of ``st`` coupled with ``dim`` via ``other_id`` (its block-diagonal
    partner), or ``None`` if absent."""
    for d in st.dims:
        if d.id == dim.other_id:
            return d
    return None


def _on_out_side(st: "SparseTensor", dim) -> bool:
    return any(d.id == dim.id for d in st.out_dims)


def contract_multi_structured(lhs, rhs, pairs, kinds):
    r"""Contract ``lhs @ rhs`` where BOTH operands carry block-diagonal structure
    on their contracted sides, over the already-resolved contracted ``pairs``
    (list of ``(lhs_dim, rhs_dim)``) with parallel ``kinds`` (from the dispatcher's
    ``_classify_pair``).

    Stays sparse via a SINGLE batched ``einsum`` over the shared meta axes (see
    the module docstring); each ``B@B`` pair survives meta-block-diagonal in the
    result.  Returns ``None`` for anything outside scope so the dispatcher keeps
    the always-correct ``composed_dense`` fallback.
    """
    from graphax.sparse.ops.utils import _is_zero_fill

    if not (_is_zero_fill(lhs) and _is_zero_fill(rhs)):
        return None
    if not pairs:
        return None

    out_dtype = _compute_dtype(lhs.dtype, rhs.dtype)

    # ----- classify each contracted pair ----------------------------------- #
    bb_pairs = []   # (lhs_con, lhs_free, rhs_con, rhs_free, N, Kc, Po, Qf)
    d_pairs = []    # (lhs_con, rhs_con, K)  — plain-dense contracted axes
    db_pairs = []   # (b_st, b_con, b_free, d_st, d_con, side_of_b, N, Kc, Bf)
    contracted_lhs_ids = set()
    contracted_rhs_ids = set()
    for (ld, rd), k in zip(pairs, kinds):
        contracted_lhs_ids.add(ld.id)
        contracted_rhs_ids.add(rd.id)
        if k == "B_B":
            l_free = _partner(lhs, ld)
            r_free = _partner(rhs, rd)
            if l_free is None or r_free is None:
                return None
            if not _on_out_side(lhs, l_free):
                return None
            if _on_out_side(rhs, r_free):
                return None
            N, M = ld.size, rd.size
            if N != M:
                return None  # mismatched meta — fallback
            if (ld.block_size or 1) != (rd.block_size or 1):
                return None
            Po = l_free.block_size or 1
            Qf = r_free.block_size or 1
            bb_pairs.append((ld, l_free, rd, r_free, N, ld.block_size or 1, Po, Qf))
        elif k == "dense":
            if int(ld.logical_size) != int(rd.logical_size):
                return None
            d_pairs.append((ld, rd, int(ld.logical_size)))
        elif k == "D_B":
            # One side block-diagonal (its free partner survives DENSE), the other
            # plain dense (contracts away). Resolve which side is the block.
            if is_block_diagonal(ld) and not is_block_diagonal(rd):
                b_st, b_con, d_con, b_is_lhs = lhs, ld, rd, True
            elif is_block_diagonal(rd) and not is_block_diagonal(ld):
                b_st, b_con, d_con, b_is_lhs = rhs, rd, ld, False
            else:
                return None
            b_free = _partner(b_st, b_con)
            if b_free is None:
                return None
            N = b_con.size
            Kc = b_con.block_size or 1
            Bf = b_free.block_size or 1
            db_pairs.append((b_st, b_con, b_free, d_con, b_is_lhs, N, Kc, Bf))
        else:
            return None  # implicit — not this kernel's scope

    if not bb_pairs:
        # No genuine B@B pair — the single-pair / multi-D_B / fallback owns it.
        return None

    # ----- surviving free dims (per operand; ids unique only WITHIN a tensor) - #
    lhs_bb_free_ids = {lf.id for (lc, lf, rc, rf, *_r) in bb_pairs}
    rhs_bb_free_ids = {rf.id for (lc, lf, rc, rf, *_r) in bb_pairs}
    # D_B survivors are the block side's free partners.
    db_lhs_free_ids = {bf.id for (bs, bc, bf, dc, bil, *_r) in db_pairs if bil}
    db_rhs_free_ids = {bf.id for (bs, bc, bf, dc, bil, *_r) in db_pairs if not bil}

    skip_lhs = contracted_lhs_ids | lhs_bb_free_ids | db_lhs_free_ids
    skip_rhs = contracted_rhs_ids | rhs_bb_free_ids | db_rhs_free_ids

    lhs_free_out = [d for d in lhs.out_dims if d.id not in skip_lhs]
    lhs_free_primal = [d for d in lhs.primal_dims if d.id not in skip_lhs]
    rhs_free_out = [d for d in rhs.out_dims if d.id not in skip_rhs]
    rhs_free_primal = [d for d in rhs.primal_dims if d.id not in skip_rhs]

    # A surviving dim that is itself block-diagonal but NOT a matched B@B partner
    # is a free ride-through diagonal pair — out of scope.
    for d in lhs_free_out + lhs_free_primal + rhs_free_out + rhs_free_primal:
        if is_block_diagonal(d):
            return None

    # ----- einsum letters --------------------------------------------------- #
    pool = iter(string.ascii_letters)

    def fresh():
        try:
            return next(pool)
        except StopIteration:
            return None

    # B@B pair: g (shared meta, batch), i (contract block), r (lhs out block),
    # c (rhs primal block).
    pair_syms = []
    for (lc, lf, rc, rf, N, Kc, Po, Qf) in bb_pairs:
        g, i, r, c = fresh(), fresh(), fresh(), fresh()
        if None in (g, i, r, c):
            return None
        pair_syms.append((g, i, r, c, lc, lf, rc, rf, N, Kc, Po, Qf))

    d_syms = [(fresh(), ld, rd, K) for (ld, rd, K) in d_pairs]
    # D_B: g (block meta, becomes a SURVIVING dense (g,j) axis), i (contract
    # block, on the block operand), j (free block, survives dense). The dense
    # operand carries the contracted axis as (g, i) reshaped.
    db_syms = []
    for (bs, bc, bf, dc, bil, N, Kc, Bf) in db_pairs:
        g, i, j = fresh(), fresh(), fresh()
        if None in (g, i, j):
            return None
        db_syms.append((g, i, j, bs, bc, bf, dc, bil, N, Kc, Bf))

    lfo_syms = [(fresh(), d) for d in lhs_free_out]
    lfp_syms = [(fresh(), d) for d in lhs_free_primal]
    rfo_syms = [(fresh(), d) for d in rhs_free_out]
    rfp_syms = [(fresh(), d) for d in rhs_free_primal]
    if any(s is None for (s, _d) in lfo_syms + lfp_syms + rfo_syms + rfp_syms):
        return None

    # ----- pack each operand -------------------------------------------------#
    lhs_factor, lhs_sub = _pack_operand(
        lhs, out_dtype, side="lhs", pair_syms=pair_syms, d_syms=d_syms,
        db_syms=db_syms, free_out_syms=lfo_syms, free_primal_syms=lfp_syms,
    )
    if lhs_factor is None:
        return None
    rhs_factor, rhs_sub = _pack_operand(
        rhs, out_dtype, side="rhs", pair_syms=pair_syms, d_syms=d_syms,
        db_syms=db_syms, free_out_syms=rfo_syms, free_primal_syms=rfp_syms,
    )
    if rhs_factor is None:
        return None

    # ----- output subscript (packed; native dim order) ---------------------- #
    # We lay the packed val as a flat axis sequence, then build dims pointing at
    # those axes via .axis / .block_axis. Order chosen so the LOGICAL out dims are
    # [lhs out survivors..., rhs out survivors...] and primal [lhs primal
    # survivors..., rhs primal survivors...], matching the fallback's native order.
    spec = _OutputSpec(lhs, rhs, pair_syms, db_syms, lfo_syms, lfp_syms,
                       rfo_syms, rfp_syms)
    out_sub = spec.subscript()

    eq = f"{lhs_sub},{rhs_sub}->{out_sub}"
    out = jnp.einsum(eq, lhs_factor, rhs_factor)

    scalar = _scaled_mul(lhs.scalar_mult, rhs.scalar_mult).astype(out_dtype)
    out = _scaled_mul(out.astype(out_dtype), scalar)

    return spec.wrap(out)


# --------------------------------------------------------------------------- #
# Operand packing
# --------------------------------------------------------------------------- #
def _pack_operand(st, out_dtype, *, side, pair_syms, d_syms, db_syms,
                  free_out_syms, free_primal_syms):
    """Transpose ``st.val`` into the einsum-subscript axis order and return
    ``(array, subscript)`` (or ``(None, None)`` when an axis can't be located).

    ``want`` is the ordered list of ``(physical_axis_or_None, [sizes], [syms],
    split)`` slots: most slots are a single ``(size, sym)`` physical axis, but a
    dense operand's ``D_B`` contracted axis carries the block operand's
    ``(meta, block)`` FLATTENED into one physical axis — it is SPLIT into the two
    letters ``(g, i)`` by a reshape so it lines up with the block operand's
    separate ``g``/``i`` axes.  Absent block axes become broadcast singletons;
    ``val is None`` -> an all-ones packed buffer."""
    # Each want entry: (physical_axis_or_None, [(size, sym), ...]).  A single
    # physical axis usually maps to one (size, sym); a split D_B dense axis maps
    # to two (the flattened (g-size, i-size) it reshapes into).
    want = []

    for (g, i, r, c, lc, lf, rc, rf, N, Kc, Po, Qf) in pair_syms:
        if side == "lhs":
            con_dim, free_dim, free_sym, free_blk = lc, lf, r, Po
        else:
            con_dim, free_dim, free_sym, free_blk = rc, rf, c, Qf
        meta_axis = con_dim.axis if con_dim.axis is not None else free_dim.axis
        want.append((meta_axis, [(N, g)]))
        want.append((con_dim.block_axis, [(Kc, i)]))
        want.append((free_dim.block_axis, [(free_blk, free_sym)]))

    for (sym, ld, rd, K) in d_syms:
        dim = ld if side == "lhs" else rd
        want.append((dim.axis, [(K, sym)]))

    for (g, i, j, bs, bc, bf, dc, bil, N, Kc, Bf) in db_syms:
        b_is_this = (bil and side == "lhs") or ((not bil) and side == "rhs")
        if b_is_this:
            meta_axis = bc.axis if bc.axis is not None else bf.axis
            want.append((meta_axis, [(N, g)]))
            want.append((bc.block_axis, [(Kc, i)]))
            want.append((bf.block_axis, [(Bf, j)]))
        else:
            # Dense operand: contracted axis is (g, i) flattened into ONE axis;
            # split it so it aligns with the block operand's g and i.
            want.append((dc.axis, [(N, g), (Kc, i)]))

    for (sym, d) in free_out_syms + free_primal_syms:
        want.append((d.axis, [(int(d.logical_size), sym)]))

    sizes = [sz for (_ax, slots) in want for (sz, _s) in slots]
    sub = "".join(s for (_ax, slots) in want for (_sz, s) in slots)

    if st.val is None:
        return jnp.ones(tuple(sizes), dtype=out_dtype), sub

    val = st.val
    present = [ax for (ax, _slots) in want if ax is not None]
    if len(present) != len(set(present)):
        return None, None
    if any(ax >= val.ndim for ax in present):
        return None, None  # a wanted axis is out of the val's physical range
    leftover = [a for a in range(val.ndim) if a not in present]
    perm = present + leftover
    v = jnp.transpose(val, perm) if perm != list(range(val.ndim)) else val

    n_present = len(present)
    if v.ndim > n_present:
        import math as _m

        if _m.prod(v.shape[n_present:]) != 1:
            return None, None
        v = v.reshape(v.shape[:n_present])

    # Re-expand: present physical axes consume v's axes, SPLITTING a flattened
    # D_B dense axis into its ``(g, i)`` sizes when the physical axis carries the
    # full ``N*Kc`` extent; a broadcast (size-1) physical axis or an absent axis
    # becomes singleton(s) and is broadcast up at the end.
    final_shape = []
    src = iter(range(n_present))
    for (ax, slots) in want:
        if ax is not None:
            phys_sz = v.shape[next(src)]
            prod = 1
            for (sz, _s) in slots:
                prod *= sz
            if phys_sz == prod:
                for (sz, _s) in slots:
                    final_shape.append(sz)
            elif phys_sz == 1:
                for (_sz, _s) in slots:
                    final_shape.append(1)
            else:
                return None, None  # unexpected packed extent
        else:
            for (_sz, _s) in slots:
                final_shape.append(1)
    v = v.reshape(tuple(final_shape))
    if v.shape != tuple(sizes):
        v = jnp.broadcast_to(v, tuple(sizes))
    return v.astype(out_dtype), sub


# --------------------------------------------------------------------------- #
# Output spec — subscript + wrap, in native dim order
# --------------------------------------------------------------------------- #
class _OutputSpec:
    """Builds the einsum output subscript and wraps the result.

    PHYSICAL val layout (the einsum output): each B@B pair contributes its shared
    meta ``g`` ONCE plus block axes ``r`` (lhs out) and ``c`` (rhs primal); each
    D_B survivor a ``(g, j)`` axis pair; each free dim one axis.  (A B@B pair's
    meta is SHARED between its out and primal dim, so ``g`` must appear once — a
    repeated output letter is illegal in einsum.)

    LOGICAL dim order (ids / native order) is tracked SEPARATELY: out dims are
    lhs's surviving out dims (in ``lhs.out_dims`` order) then rhs's surviving out
    dims; primal likewise.  Each logical dim records which physical axes it owns,
    so a B@B pair's out and primal dims both point at the shared meta axis."""

    def __init__(self, lhs, rhs, pair_syms, db_syms, lfo_syms, lfp_syms,
                 rfo_syms, rfp_syms):
        self.lhs, self.rhs = lhs, rhs
        self.pair_syms = pair_syms
        self.db_syms = db_syms
        free = lfo_syms + lfp_syms + rfo_syms + rfp_syms
        self._free_by_obj = {id(fd): sym for (sym, fd) in free}

        # Physical axis sequence (letters) — each shared B@B meta emitted ONCE,
        # then its two block axes; each D_B survivor as adjacent (g, j); each free
        # dim one axis. ``wrap`` maps letter -> axis from this order.
        self._phys = []
        for (g, i, r, c, lc, lf, rc, rf, N, Kc, Po, Qf) in pair_syms:
            self._phys += [g, r, c]
        for (g, i, j, bs, bc, bf, dc, bil, N, Kc, Bf) in db_syms:
            self._phys += [g, j]
        for (sym, fd) in free:
            self._phys.append(sym)

        # Logical dim entries in native order.
        self.out_entries = (self._side(lhs, "out", "lhs")
                            + self._side(rhs, "out", "rhs"))
        self.primal_entries = (self._side(lhs, "primal", "lhs")
                              + self._side(rhs, "primal", "rhs"))

    def _side(self, st, which, side):
        dims = st.out_dims if which == "out" else st.primal_dims
        entries = []
        for d in dims:
            ent = self._entry_for_dim(d, side)
            if ent is not None:
                entries.append((d, ent))
        return entries

    def _entry_for_dim(self, d, side):
        for (g, i, r, c, lc, lf, rc, rf, N, Kc, Po, Qf) in self.pair_syms:
            if side == "lhs" and d.id == lf.id:
                return ("bb", g, r, N, Po)
            if side == "rhs" and d.id == rf.id:
                return ("bb", g, c, N, Qf)
        for (g, i, j, bs, bc, bf, dc, bil, N, Kc, Bf) in self.db_syms:
            if (bil and side == "lhs" and d.id == bf.id) or \
               ((not bil) and side == "rhs" and d.id == bf.id):
                return ("db", g, j, N, Bf)
        sym = self._free_by_obj.get(id(d))
        if sym is not None:
            return ("free", sym, int(d.logical_size))
        return None

    def subscript(self):
        return "".join(self._phys)

    def wrap(self, out):
        from graphax.sparse.tensor import SparseTensor

        # The einsum output axes are in ``self._phys`` order. A D_B survivor's
        # (g, j) adjacent axes are merged into ONE dense axis of size N*Bf; B@B
        # (g, r, c) and free axes stay as-is. We build the final physical layout
        # and a letter->axis map, then dims index it.
        ax_of = {letter: k for k, letter in enumerate(self._phys)}

        # Merge each D_B (g, j) adjacent pair into a single axis via reshape.
        # Process from the back so earlier axis indices stay valid.
        db_merges = sorted(
            [(ax_of[g], ax_of[j]) for (g, i, j, *_r) in self.db_syms],
            key=lambda t: t[0], reverse=True,
        )
        shape = list(out.shape)
        # Merges are of adjacent (g, j) where j == g+1 by construction.
        for (ga, ja) in db_merges:
            assert ja == ga + 1, "D_B (g, j) must be adjacent in physical layout"
            merged = shape[ga] * shape[ja]
            out = out.reshape(tuple(shape[:ga] + [merged] + shape[ja + 1:]))
            shape = shape[:ga] + [merged] + shape[ja + 1:]
        # Recompute axis-of-letter after merges: each merge removed one axis.
        # Rebuild ax_of by walking the original _phys, skipping the j of each
        # merged pair (j collapses into g).
        merged_j = {ax_of[j] for (g, i, j, *_r) in self.db_syms}
        new_ax_of = {}
        cur = 0
        for k, letter in enumerate(self._phys):
            if k in merged_j:
                continue  # collapsed into the preceding meta axis
            new_ax_of[letter] = cur
            cur += 1
        ax_of = new_ax_of

        n_out = len(self.out_entries)
        out_ids = list(range(n_out))
        pri_ids = list(range(n_out, n_out + len(self.primal_entries)))

        # Build dims; collect B@B g-letter -> (out_id, primal_id) for other_id.
        bb_link = {}
        for k, (_d, ent) in enumerate(self.out_entries):
            if ent[0] == "bb":
                bb_link.setdefault(ent[1], {})["out"] = out_ids[k]
        for k, (_d, ent) in enumerate(self.primal_entries):
            if ent[0] == "bb":
                bb_link.setdefault(ent[1], {})["primal"] = pri_ids[k]

        def make(mid, ent, side):
            kind = ent[0]
            if kind == "bb":
                _k, g, blk, N, B = ent
                partner = bb_link[g]["primal" if side == "out" else "out"]
                return DiagonalIndex(
                    mid, N, axis=ax_of[g], other_id=partner,
                    block_size=B if B > 1 else None,
                    block_axis=ax_of[blk] if B > 1 else None)
            if kind == "db":
                _k, g, j, N, Bf = ent
                return DenseIndex(mid, N * Bf, axis=ax_of[g])
            _k, sym, sz = ent
            return DenseIndex(mid, sz, axis=ax_of[sym])

        out_dims = [make(out_ids[k], ent, "out")
                    for k, (_d, ent) in enumerate(self.out_entries)]
        primal_dims = [make(pri_ids[k], ent, "primal")
                       for k, (_d, ent) in enumerate(self.primal_entries)]

        # The einsum emits a block axis even for size-1 blocks (whose dim carries
        # ``block_size=None`` and so does NOT reference that axis). Squeeze every
        # size-1 physical axis NOT referenced by a dim, and remap the surviving
        # ``axis`` / ``block_axis`` so the result is a clean, fully-described val
        # (``dense()`` transposes by the dims' axes — an orphan axis breaks it).
        from dataclasses import replace as _dc_replace

        referenced = set()
        for d in out_dims + primal_dims:
            if d.axis is not None:
                referenced.add(d.axis)
            if getattr(d, "block_axis", None) is not None:
                referenced.add(d.block_axis)
        drop = [a for a in range(out.ndim)
                if a not in referenced and out.shape[a] == 1]
        if drop:
            keep = [a for a in range(out.ndim) if a not in drop]
            out = jnp.squeeze(out, axis=tuple(drop))
            remap = {old: new for new, old in enumerate(keep)}

            def reax(d):
                if d.is_sparse:
                    return _dc_replace(
                        d, axis=remap[d.axis],
                        block_axis=(remap[d.block_axis]
                                    if d.block_axis is not None else None))
                return _dc_replace(d, axis=remap[d.axis])

            out_dims = [reax(d) for d in out_dims]
            primal_dims = [reax(d) for d in primal_dims]

        return SparseTensor(
            tuple(out_dims), tuple(primal_dims), out,
            fill_value=None, check_consistency=False,
        )
