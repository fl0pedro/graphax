"""Tiled block-sparse matmul: the one contraction engine.

Pipeline (zero-fill fast path):
    1. Classify each dim pair as ``contract``/``batch_*``/``spatial_*``.
    2. ``_prepare_physical_arrays`` — bring both ``val`` buffers into the canonical
       (outer, block, shared_block, *leftover) layout via the shared transpose
       primitive, building ONLY the slots a val axis backs and returning a
       ``_Slots`` map that says where each nominal slot went.
    3. ``_execute_block_sparse_contraction`` — shrink every frame slot no operand
       stores, then emit ONE einsum over integer sublists.
    4. ``_build_output_tensor`` — re-emit ``SparseTensor`` ``out_dims`` / ``primal_dims``.

EMISSION (owner ruling D1, revised 2026-09-08). Every contraction leaves this
module as ``jnp.einsum`` over INTEGER sublists, never a ``lax.dot_general``
this module wrote itself and never a hand-written multiply-then-reduce.

Be precise about what that buys. ``jnp.einsum`` still lowers to ``dot_general``
in the jaxpr; it is a jnp-level API, not a different XLA primitive. What
changes is WHO picks the dimension numbers and what has to happen to the
operands first. Stating the contraction lets einsum choose, and it chooses
better than the frame did: an axis one operand does not store is given a
PRIVATE label, so einsum emits a ``reduce_sum`` over the storing side instead
of a ``broadcast_in_dim`` growing the other side up to it. MEASURED on a
meta-32 diagonal pair whose rhs stores no meta axis: the broadcast
(96 -> 3072 elements) is gone and the compiled temp goes 12 288 B -> 0 B, with
identical flops. Integer labels also lift the letter form's 52-symbol cap.
See ``_frame_sublists``.

Late-densification escape hatch (non-zero fill_value):
    The tiled algorithm assumes implicit positions are zero. When ``_is_zero_fill``
    returns ``False`` for either operand, ``matmul`` reroutes through
    ``_matmul_via_densify`` — which materializes both sides via the fusion-friendly
    ``dense_for_matmul``. Densification stays as a JAX expression so XLA can fold
    it into the contraction kernel (SMEM, not HBM).

MISALIGNED (least-common-multiple) GRIDS materialize NEITHER operand. The lcm
refinement is carved out of each operand's own block axis by a reshape, so a
meta-a and a meta-b factoring of one contracted axis meet at meta lcm(a, b) for
free. The cost sits after the contraction: ``_reduce_grid`` folds the refined
metas into the output's ``(gcd, a/gcd, b/gcd)`` band grid through a constant
one-hot contraction, and that grid holds cells that are structurally zero.
MEASURED on meta 16 against meta 24: 3 072 stored in, 49 152 stored out
against a structural support of 32 768 (1.50x), temp 196 608 B, two
contractions and 10 HLO fusions, against 0 B, one contraction and 4 fusions
for the aligned control. The class that held that support exactly was
``BandedIndex``, deleted by the two-class ruling of 2026-09-07, so 1.50x is
the floor for the classes that remain.

There is ONE engine and no environment switch. The structure-lowering planner
(``sparse/lower/``), its compact-frame and spill entry points, the verbatim
legacy tiled executor, and the race knobs that selected between them were
deleted on 2026-09-08 (ticket dsnn-3qm.72); what they proved is recorded in
the docstrings that carry it.
"""

# pyright: reportImportCycles=false
from __future__ import annotations

import builtins
import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from graphax.sparse.indexes import DenseIndex, Index, DiagonalIndex

from .dense import dense_for_matmul
from .layout import generate_block_permutation, generate_grouped_permutation
from .utils import check_nominal_order
from .utils import (
    _arr2st,
    _copy,
    _is_sparse,
    _is_zero_fill,
    _prepare_physical_array,
    _scaled_fill,
    _val_or_one,
)

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


AXES_PER_PAIR = 3  # (outer, block, shared_block) per dimension pair
SPLIT_AXES = 4
GRID_AXES_PER_PAIR = 3

PairingType: TypeAlias = Literal[
    "batch_sparse",
    "batch_out",
    "batch_primal",
    "spatial_sparse_lhs",
    "spatial_out_lhs",
    "spatial_primal_lhs",
    "spatial_sparse_rhs",
    "spatial_out_rhs",
    "spatial_primal_rhs",
    "contract",
]


class PairData(NamedTuple):
    outer_len: int
    block_len: int
    shared_block_len: int
    outer_axis: int | None = None
    block_axis: int | None = None
    shared_block_axis: int | None = None
    dim: Index | None = None
    shared_dim: Index | None = None


class Pair(NamedTuple):
    pairing_type: PairingType
    logical_element_count: int
    lhs: PairData
    rhs: PairData


class Ctx(NamedTuple):
    lhs: "SparseTensor"
    rhs: "SparseTensor"
    pairs: list["Pair"]
    rhs_id_offset: int


class CRes(NamedTuple):
    grid: Array
    shared_factors: list[int]
    lhs_block_lens: list[int]
    rhs_block_lens: list[int]
    scalar_mult: float
    # --- lazy frame (ticket dsnn-3qm.67) ---------------------------------
    # ``grid`` is built on the EFFECTIVE frame: every extent that neither
    # operand stores is 1 there instead of broadcast to its logical length.
    # The three fields above therefore describe the BUFFER. The output DIMS
    # need the logical lengths, which is what these carry; ``eff_pairs`` is
    # the Pair list the buffer was built from and ``lazy`` says which of each
    # pair's three output slots has no physical axis. All None on the eager
    # frame, where buffer and logic agree.
    eff_pairs: "list[Pair] | None" = None
    lazy: "list[_Lazy] | None" = None
    true_shared_factors: "list[int] | None" = None
    true_lhs_block_lens: "list[int] | None" = None
    true_rhs_block_lens: "list[int] | None" = None


class _Lazy(NamedTuple):
    """Which of a Pair's three output slots stayed symbolic: the shared
    (meta / diagonal) slot, the lhs (out-side) slot, the rhs (primal-side)
    slot. A True slot keeps its logical extent and gets ``axis=None`` —
    the planner's ``out:implicit_kept`` / ``out:pair_retained``."""

    shared: bool = False
    lhs: bool = False
    rhs: bool = False


_NO_LAZY = _Lazy()


# --- Topology resolution ---------------------------------------------------
def _dim_vals(dim, is_outer=False):
    """(length, axis) for one logical axis of a Index. is_outer=True picks sparse-pair length."""
    if not dim:
        return 1, None
    if not dim.is_sparse:
        return (1, None) if is_outer else (dim.size, dim.axis)
    if is_outer:
        return dim.size, dim.axis
    return (dim.block_size if dim.block_size is not None else 1), dim.block_axis


def _is_blocked_dense(dim) -> bool:
    """``Index.is_blocked_dense``, tolerating the ``None`` dims a Pair carries.

    It matters all over this module because the frame reads a DiagonalIndex's
    meta off its PARTNER (``_full_pair_data``'s ``lo`` / ``ri``), and a blocked
    dense dim has no partner: its meta has to come from the dim itself."""
    return dim is not None and dim.is_blocked_dense


def _reject_blocked_dense(where: str, *dims) -> None:
    """Raise on a BLOCKED DENSE dim in a role this module cannot express yet.

    Project rule: an unimplemented path raises, it never silently passes. The
    two BATCH pairings multiply the two operands POSITION BY POSITION along the
    shared dim, so an implicit block would have to be materialized on the side
    that carries one — exactly the densification the blocked form exists to
    avoid. There is no correct answer to give here, so say which dim and which
    pairing rather than reading ``size`` as the whole extent and returning a
    wrong-shaped product."""
    for d in dims:
        if _is_blocked_dense(d):
            raise ValueError(
                f"matmul: a blocked dense dim (id={d.id} size={d.size} "
                f"block_size={d.block_size} logical_size={d.logical_size}) is an "
                f"operand of a '{where}' pairing, which is not implemented "
                f"(ticket dsnn-3qm.62). The pairing would have to materialize "
                f"the implicit block to align the two sides position by "
                f"position. Densify the operand before the matmul, or contract "
                f"the dim instead of batching it."
            )


def _outer_v(dim, sibling):
    """Outer axis for `dim`, falling back to its sibling's axis if dim itself is unmaterialized."""
    if dim is None:
        return None
    if dim.axis is not None:
        return dim.axis
    if (
        sibling is not None
        and dim.is_sparse
        and sibling.is_sparse
        and dim.other_id == sibling.id
    ):
        return sibling.axis
    return None


def _full_pair_data(lo, li, ro, ri, swap_rhs=False):
    """Construct (lhs_PairData, rhs_PairData) for a 2-dim-per-side pairing (sparse pair or contract)."""
    l_outer, _ = _dim_vals(lo, True)
    r_outer, _ = _dim_vals(ri if swap_rhs else ro, True)
    l_block, l_block_v = _dim_vals(lo, False)
    l_shared, l_shared_v = _dim_vals(li, False)
    r_block, r_block_v = _dim_vals(ro, False)
    r_shared, r_shared_v = _dim_vals(ri, False)
    l_outer_v = _outer_v(lo, li)
    r_outer_v = _outer_v(ri if swap_rhs else ro, ro if swap_rhs else ri)
    # A BLOCKED DENSE dim being CONTRACTED (``li`` on the lhs, ``ro`` on the rhs)
    # factors its own contracted extent into ``size`` stored blocks x an IMPLICIT
    # ``block_size`` (ticket dsnn-3qm.62). That is the same shape of statement a
    # DiagonalIndex's meta makes, but it has no partner dim to read the meta off,
    # so ``_dim_vals(lo, True)`` / ``_dim_vals(ri, True)`` see nothing: the meta
    # comes from the contracted dim itself and its block becomes the (unstored)
    # contracted slot, from where the frame lcm-refines it against the other
    # side's factoring of the same extent without materializing anything.
    if _is_blocked_dense(li):
        l_outer, l_outer_v = li.size, li.axis
        l_shared, l_shared_v = li.block_size, li.block_axis
    if _is_blocked_dense(ro):
        r_outer, r_outer_v = ro.size, ro.axis
        r_block, r_block_v = ro.block_size, ro.block_axis
    return (
        PairData(
            l_outer, l_block, l_shared, l_outer_v, l_block_v, l_shared_v, lo, li
        ),
        PairData(
            r_outer,
            r_block,
            r_shared,
            r_outer_v,
            r_block_v,
            r_shared_v,
            ro,
            ri,
        ),
    )


def _matched_pair(lout, lprimal, rout, rprimal):
    """Pair of dims that exists on both sides (post-id-alignment)."""
    # NB the message used to print `.id` while the CHECK is on `.logical_size`, so
    # "Batch mismatch: 1 vs 4" read like "batch size 1 vs 4" (a broadcast case) when it
    # actually meant "dim id 1 vs dim id 4" and told you NOTHING about the sizes.
    def _d(d):
        return (f"id={d.id} logical_size={d.logical_size} size={d.size} "
                f"block={getattr(d, 'block_size', None)} axis={getattr(d, 'axis', None)} "
                f"sparse={d.is_sparse}")
    if lout and rout and lout.logical_size != rout.logical_size:
        raise ValueError(
            f"Batch mismatch (out side): lhs[{_d(lout)}] vs rhs[{_d(rout)}] — "
            f"logical_size {lout.logical_size} != {rout.logical_size}"
        )
    if lprimal and rprimal and lprimal.logical_size != rprimal.logical_size:
        raise ValueError(
            f"Batch mismatch (primal side): lhs[{_d(lprimal)}] vs rhs[{_d(rprimal)}] — "
            f"logical_size {lprimal.logical_size} != {rprimal.logical_size}"
        )
    if lout and lprimal and rout and rprimal:
        return Pair(
            "batch_sparse",
            1,
            *_full_pair_data(lout, lprimal, rout, rprimal, swap_rhs=True),
        )
    if lout and rout:
        _reject_blocked_dense("batch_out", lout, rout)
        l_len, l_v = _dim_vals(lout, False)
        r_len, r_v = _dim_vals(rout, False)
        return Pair(
            "batch_out",
            1,
            PairData(l_len, 1, 1, l_v, None, None, lout),
            PairData(r_len, 1, 1, r_v, None, None, rout),
        )
    if lprimal and rprimal:
        _reject_blocked_dense("batch_primal", lprimal, rprimal)
        l_len, l_v = _dim_vals(lprimal, False)
        r_len, r_v = _dim_vals(rprimal, False)
        return Pair(
            "batch_primal",
            1,
            PairData(l_len, 1, 1, None, None, l_v, None, lprimal),
            PairData(r_len, 1, 1, None, None, r_v, None, rprimal),
        )
    return None


def _unmatched_pair(out_dim, primal_dim, on_left):
    """Pair of dims that exists only on one side (carried through as spatial)."""
    if not (out_dim or primal_dim):
        return None
    if out_dim and primal_dim:
        outer, _ = _dim_vals(out_dim, True)
        block, block_v = _dim_vals(out_dim, False)
        shared, shared_v = _dim_vals(primal_dim, False)
        side = PairData(
            outer,
            block,
            shared,
            _outer_v(out_dim, primal_dim),
            block_v,
            shared_v,
            out_dim,
            primal_dim,
        )
        kind = "sparse"
    elif out_dim:
        ln, v = _dim_vals(out_dim, False)
        side = PairData(1, ln, 1, None, v, None, out_dim)
        kind = "out"
    else:
        ln, v = _dim_vals(primal_dim, False)
        side = PairData(1, 1, ln, None, None, v, None, primal_dim)
        kind = "primal"
    # ``ptype`` is one of nine literals from PairingType; cast() avoids
    # the f-string returning ``LiteralString`` instead of the narrow union.
    from typing import cast

    ptype = cast(PairingType, f"spatial_{kind}_{'lhs' if on_left else 'rhs'}")
    empty = PairData(1, 1, 1)
    return Pair(ptype, 1, side, empty) if on_left else Pair(ptype, 1, empty, side)


def _align_tensor_ids(lhs, rhs):
    rhs_id_offset = builtins.max([d.id for d in lhs.dims] + [-1]) + 1

    def offset(d):
        kw: dict[str, Any] = {"id": d.id + rhs_id_offset}
        if d.is_sparse:
            kw["other_id"] = d.other_id + rhs_id_offset
        return replace(d, **kw)

    return (
        tuple(offset(d) for d in rhs.out_dims),
        tuple(offset(d) for d in rhs.primal_dims),
        rhs_id_offset,
    )


def _unprocessed_topos(dims, dim_map, processed, target_list):
    """List of (out_dim, primal_dim) topo pairs for dims not yet consumed."""
    target_ids = {d.id for d in target_list}

    def info(d):
        if d.id in processed:
            return (None, None), -1
        if not d.is_sparse:
            return ((d, None) if d.id in target_ids else (None, d)), d.id
        other = dim_map.get(d.other_id)
        if not other or other.id in processed:
            return ((d, None) if d.id in target_ids else (None, d)), d.id
        return ((d, other) if d.id in target_ids else (other, d)), other.id

    out, seen = [], set()
    for d in dims:
        if d.id in seen:
            continue
        topo, extra = info(d)
        if topo != (None, None):
            out.append(topo)
            if extra != -1:
                seen.add(extra)
        seen.add(d.id)
    return out


def _resolve_contract_pair(lp, ro, lhs_out_map, rhs_primal_map):
    if lp.logical_size != ro.logical_size:
        raise ValueError(
            f"Contraction size mismatch: {lp.logical_size} vs {ro.logical_size}"
        )
    lo = (
        lhs_out_map.get(getattr(lp, "other_id", -1))
        if lp.is_sparse
        else None
    )
    rp = (
        rhs_primal_map.get(getattr(ro, "other_id", -1))
        if ro.is_sparse
        else None
    )
    lhs_ids = [lp.id]
    if lo:
        lhs_ids.append(lo.id)
    rhs_ids = [ro.id]
    if rp:
        rhs_ids.append(rp.id)
    return (
        Pair(
            # logical_element_count = elements per contraction unit: the block
            # size, or the dim size when the dim carries no block (block_size is
            # present-but-None for PLAIN dense/scalar contracting dims, so a
            # getattr default never fires — use ``or`` to fall through), or 1.
            # A BLOCKED DENSE contracting dim (dsnn-3qm.62) has a block_size too,
            # and ``block_size`` is the right reading there as well: its meta is
            # the block COUNT, so the unit is still one block.
            "contract",
            (getattr(lp, "block_size", None) or getattr(lp, "size", None) or 1),
            *_full_pair_data(lo, lp, ro, rp, swap_rhs=True),
        ),
        lhs_ids,
        rhs_ids,
    )


def _resolve_broadcast_topos(lhs_topos, rhs_topos, offset):
    def _deferred_broadcast(a, b):
        """A size-1 IMPLICIT dim that ``_align_contract_dims`` deliberately skipped and,
        per its own docstring, "falls through to ``_resolve_broadcast_topos`` as a
        free/broadcast dim". It must therefore NOT be re-captured here as a MATCHED
        batch pair: ``_matched_pair`` compares logical_size and would raise
        "Batch mismatch" on the very broadcast the skip deferred (ViT layer_norm:
        mean(keepdims=True) gives a size-1 seq axis against x's size-8 -> 1 vs 8).

        The producer (_align_contract_dims, 0a07b41) got the _is_implicit_block_dim
        gate; this consumer (_matched_pair/_resolve_broadcast_topos, 6a6afa47) predates
        it and never did — the deferral had no receiver.

        This does NOT weaken the 1-vs-N guard: a size-1 dim WITH a physical axis is not
        _is_implicit_block_dim, so it is still matched and still raises the genuine
        "Contraction size mismatch" in _resolve_contract_pair."""
        if a is None or b is None:
            return False
        if int(a.logical_size) == int(b.logical_size):
            return False
        return _is_implicit_block_dim(a) or _is_implicit_block_dim(b)

    def find_match(lout, lprimal, candidates):
        for i, (rout, rprimal) in enumerate(candidates):
            if lout and rout and lout.id == rout.id - offset:
                if _deferred_broadcast(lout, rout):
                    continue                     # free/broadcast dim -> _unmatched_pair
                return i
            if lprimal and rprimal and lprimal.id == rprimal.id - offset:
                if _deferred_broadcast(lprimal, rprimal):
                    continue                     # free/broadcast dim -> _unmatched_pair
                return i
        return -1

    pairs, remaining = [], list(rhs_topos)
    for lout, lprimal in lhs_topos:
        idx = find_match(lout, lprimal, remaining)
        if idx != -1:
            rout, rprimal = remaining.pop(idx)
            meta = _matched_pair(lout, lprimal, rout, rprimal)
        else:
            meta = _unmatched_pair(lout, lprimal, on_left=True)
        if meta:
            pairs.append(meta)
    for rout, rprimal in remaining:
        meta = _unmatched_pair(rout, rprimal, on_left=False)
        if meta:
            pairs.append(meta)
    return pairs


def _align_contract_indices(lhs_primal, rhs_out, *, embed):
    """Right-align ``lhs.primal_dims`` with ``rhs.out_dims`` into contraction
    INDEX pairs ``(li, rj)``. The two lists describe the same contracted vertex
    axes, but one side may carry an extra size-1 axis the other lacks, so a
    blind ``zip(lhs[-n:], rhs[-n:])`` pairs the wrong axes (batch vs class).

    The two consuming paths need DIFFERENT handling of an implicit-block size-1
    dim (size-1 with no physical axis), so the behaviour is selected by ``embed``:

    * ``embed=False`` (TILED path): skip EVERY implicit-block size-1 — the tiled
      kernel contracts neither a stray broadcast NOR a metadata embed (a real
      embed is routed to the densify path *before* it reaches the tiled builder;
      were it paired here the kernel would contract ``1`` against ``N`` and read
      past the val buffer). A skipped dim becomes a free/broadcast output axis.
    * ``embed=True`` (DENSIFY path / dispatch gates / FLOP depth): skip only a
      STRAY size-1 (its size-N counterpart has an equal-size partner elsewhere,
      so the size-1 is the extra one — a ``(C, 1)`` bias / ``reshape(-1, 1)``
      head under vmap); a METADATA EMBED (size-1 with no equal-size partner) is
      kept as a contraction pair that ``_matmul_via_densify`` zero-pads up to N.

    Either way a size-1 dim that carries a physical axis is never implicit-block,
    so a genuine ``1 vs N`` mismatch still pairs through and raises in
    ``_resolve_contract_pair``. Equal-length lists with no stray implicit-block
    dim reproduce the old positional ``[-n:]`` pairing exactly."""

    def _has_equal(size, dims, upto):
        return any(int(dims[k].logical_size) == size for k in range(upto))

    i, j = len(lhs_primal) - 1, len(rhs_out) - 1
    out = []
    while i >= 0 and j >= 0:
        lp, ro = lhs_primal[i], rhs_out[j]
        ls, rs = int(lp.logical_size), int(ro.logical_size)
        if ls == rs:
            out.append((i, j))
            i -= 1
            j -= 1
        elif _is_implicit_block_dim(lp) and (not embed or _has_equal(rs, lhs_primal, i)):
            i -= 1                       # implicit-block / stray size-1 on lhs -> free dim
        elif _is_implicit_block_dim(ro) and (not embed or _has_equal(ls, rhs_out, j)):
            j -= 1                       # implicit-block / stray size-1 on rhs -> free dim
        else:
            out.append((i, j))           # equal, kept embed (embed=True), or genuine mismatch
            i -= 1
            j -= 1
    out = out[::-1]
    # size-aware re-pair only when positional alignment left a size mismatch
    if all(int(lhs_primal[a].logical_size) == int(rhs_out[b].logical_size) for a, b in out):
        return out
    lhs_idxs = [a for a, _ in out]; rhs_idxs = [b for _, b in out]
    used = set(); repaired = []
    for b in rhs_idxs:
        rs = int(rhs_out[b].logical_size)
        for a in lhs_idxs:
            if a not in used and int(lhs_primal[a].logical_size) == rs:
                used.add(a); repaired.append((a, b)); break
    if len(repaired) == len(out):
        repaired.sort(); return repaired
    # The re-pair above only searches the positionally-SELECTED indices, so a dim
    # the positional walk DROPPED is invisible to it (the ViT seq<->embed swap:
    # lhs.primal=[8,17,1] vs rhs.out=[1,8,17] drops the size-8 dim, leaving 17
    # paired against 8). Re-pair by a full UNAMBIGUOUS size-bijection over BOTH
    # complete lists: a size-N lhs dim contracts the size-N rhs dim regardless of
    # position. Bail (keep the positional result) if the lists aren't equal-length
    # or a size repeats within a side — size alone can't disambiguate then, and a
    # wrong contraction is worse than the loud size-mismatch raise.
    if len(lhs_primal) == len(rhs_out):
        used_r, full = set(), []
        for a in range(len(lhs_primal)):
            sa = int(lhs_primal[a].logical_size)
            cand = [
                k
                for k in range(len(rhs_out))
                if k not in used_r and int(rhs_out[k].logical_size) == sa
            ]
            if len(cand) != 1:
                # Duplicate SIZE-1 dims are interchangeable for value
                # purposes — pair positionally (first unused) instead of
                # bailing. Without this, a permuted primal side like
                # (1,17,1) vs nominal (1,1,17) hits two ambiguous size-1
                # candidates, the repair breaks, and the positional walk's
                # 1-vs-17 pairing raises "Contraction size mismatch"
                # (the ViT layer_norm div case). Sizes > 1 still bail —
                # a wrong contraction is worse than the loud raise.
                if sa != 1 or not cand:
                    break
            used_r.add(cand[0])
            full.append((a, cand[0]))
        if len(full) == len(lhs_primal):
            full.sort()
            return full
    # UNEQUAL-LENGTH size bijection (ticket dsnn-3qm.67). The two sides factor
    # the same logical extent into a different NUMBER of dims once the tiled
    # path stops collapsing a one-sided implicit dim to 1 (finding 61 verdict
    # 6): the Hessian of the vmapped MLP reaches here with lhs.primal sizes
    # (1,8,1,1,1,1,16,1,1) against rhs.out (16,8,1,1,1,1,1,1). Neither the
    # positional walk nor the equal-length repair above can pair the 16.
    # Every size > 1 dim needs exactly one partner of its own size; the size-1
    # dims then pair from the right and the surplus rides through as a free
    # dim. Only reachable when the positional result already carries a size
    # mismatch, which raises today — so this can only turn a raise into a
    # correct pairing, never change a working one. ``embed=True`` callers keep
    # their mismatches, which is how they route a metadata embed to densify.
    if not embed:
        l_big = [a for a in range(len(lhs_primal))
                 if int(lhs_primal[a].logical_size) != 1]
        r_big = [b for b in range(len(rhs_out))
                 if int(rhs_out[b].logical_size) != 1]
        if len(l_big) == len(r_big):
            taken, big, ok = set(), [], True
            for a in l_big:
                sa = int(lhs_primal[a].logical_size)
                cand = [b for b in r_big
                        if b not in taken and int(rhs_out[b].logical_size) == sa]
                if len(cand) != 1:
                    ok = False
                    break
                taken.add(cand[0])
                big.append((a, cand[0]))
            if ok:
                l_ones = [a for a in range(len(lhs_primal)) if a not in l_big]
                r_ones = [b for b in range(len(rhs_out)) if b not in r_big]
                big += list(zip(l_ones[::-1], r_ones[::-1]))
                big.sort()
                return big
    return out


def _align_contract_dims(lhs_primal, rhs_out, *, embed):
    """Dim-pair view of :func:`_align_contract_indices` (see its docstring)."""
    return [
        (lhs_primal[i], rhs_out[j])
        for i, j in _align_contract_indices(lhs_primal, rhs_out, embed=embed)
    ]


def _build_matmul_topology(lhs, rhs_out_dims, rhs_primal_dims, rhs_id_offset):
    lhs_out_map = {d.id: d for d in lhs.out_dims}
    rhs_primal_map = {d.id: d for d in rhs_primal_dims}
    rhs_dims = rhs_out_dims + rhs_primal_dims
    pairs, processed_l, processed_r = [], set(), set()
    for lp, ro in _align_contract_dims(lhs.primal_dims, rhs_out_dims, embed=False):
        meta, lhs_ids, rhs_ids = _resolve_contract_pair(
            lp, ro, lhs_out_map, rhs_primal_map
        )
        pairs.append(meta)
        processed_l.update(lhs_ids)
        processed_r.update(rhs_ids)
        if meta.lhs.dim:
            processed_l.add(meta.lhs.dim.id)
        if meta.rhs.shared_dim:
            processed_r.add(meta.rhs.shared_dim.id)
    lhs_topos = _unprocessed_topos(
        lhs.dims, {d.id: d for d in lhs.dims}, processed_l, lhs.out_dims
    )
    rhs_topos = _unprocessed_topos(
        rhs_dims, {d.id: d for d in rhs_dims}, processed_r, rhs_out_dims
    )
    pairs.extend(_resolve_broadcast_topos(lhs_topos, rhs_topos, rhs_id_offset))
    return pairs


# --- Physical array preparation -------------------------------------------
class _Slots(NamedTuple):
    """Where each NOMINAL frame slot landed on a prepared operand.

    The frame names three slots per contraction pair — ``3 * i + k`` for pair
    ``i``, role ``k`` (0 outer/meta, 1 block, 2 shared block). Only the slots a
    ``val`` axis actually backs are BUILT; ``pos[3 * i + k]`` is the physical
    axis of a built slot and ``None`` for one that was not built. So the
    prepared rank is the number of backed slots, not ``3 * N``, and the
    leftovers start at ``rank``.

    Nothing is lost by not building a slot: an unbacked slot could only ever be
    extent 1, and it is extent 1 because no axis of the buffer is behind it.
    Read a not-built slot's extent with ``_slot_len``.
    """

    pos: tuple
    rank: int


def _slot_len(val, slots, k):
    """Physical extent of nominal frame slot ``k`` — 1 when it was not built."""
    p = slots.pos[k]
    return 1 if p is None else int(val.shape[p])


def _prepare_physical_arrays(lhs_val, rhs_val, pairs):
    """Canonical per-pair slot order, building ONLY the slots a val axis backs.

    The frame used to pad the prepared buffer out to the full ``3 * N`` nominal
    slots, which costs a ``reshape`` to append the missing size-1 axes and turns
    the canonicalizing ``transpose`` into a non-identity permutation even when
    the source axes were already in order. Both are pure bookkeeping: an
    unbacked slot is extent 1 and carries nothing. Building the compact layout
    instead hands ``_prepare_physical_array`` a permutation of real axes only,
    which is a no-op whenever they are already ascending.
    """

    def prep(val, sides):
        flat = [
            a for s in sides for a in (s.outer_axis, s.block_axis, s.shared_block_axis)
        ]
        # A slot is BACKED when its source axis exists on this buffer. A nominal
        # axis at or beyond ``val.ndim`` belongs to an implicit dim and backs
        # nothing — the same test ``_prepare_physical_array`` applies internally.
        built = [k for k, a in enumerate(flat) if a is not None and a < val.ndim]
        pos = [None] * len(flat)
        for new, k in enumerate(built):
            pos[k] = new
        return (
            _prepare_physical_array(val, [flat[k] for k in built]),
            _Slots(tuple(pos), len(built)),
        )

    lhs_out, lhs_slots = prep(lhs_val, [p.lhs for p in pairs])
    rhs_out, rhs_slots = prep(rhs_val, [p.rhs for p in pairs])
    return lhs_out, rhs_out, lhs_slots, rhs_slots


# --- Tiled contraction core -----------------------------------------------
def _contraction_factors(pairs):
    shared, total, split, scalar = [], [], [], 1.0
    for p in pairs:
        if (
            p.pairing_type == "contract"
            and p.lhs.outer_len == 1
            and p.rhs.outer_len == 1
            and p.lhs.shared_block_len == 1
            and p.rhs.block_len == 1
        ):
            scalar *= float(max(p.logical_element_count, 1))
        gcd_len = math.gcd(p.lhs.outer_len, p.rhs.outer_len)
        lcm_len = math.lcm(p.lhs.outer_len, p.rhs.outer_len)
        shared.append(gcd_len)
        total.append(lcm_len)
        s = 1
        if p.lhs.shared_block_len > 1:
            _den = lcm_len // p.lhs.outer_len
            if _den == 0 or p.lhs.shared_block_len % _den != 0:
                raise ValueError(
                    f"Non-divisible contraction split: lhs.shared_block_len "
                    f"{p.lhs.shared_block_len} not a multiple of {_den}."
                )
            s = p.lhs.shared_block_len // _den
        elif p.rhs.block_len > 1:
            _den = lcm_len // p.rhs.outer_len
            if _den == 0 or p.rhs.block_len % _den != 0:
                raise ValueError(
                    f"Non-divisible contraction split: rhs.block_len "
                    f"{p.rhs.block_len} not a multiple of {_den}."
                )
            s = p.rhs.block_len // _den
        split.append(s)
    return shared, total, split, scalar


def _contraction_perms(N):
    perm_l = (
        generate_block_permutation(N, SPLIT_AXES, [0, 2])
        + generate_grouped_permutation(N, SPLIT_AXES, [1])
        + generate_grouped_permutation(N, SPLIT_AXES, [3])
    )
    perm_r = (
        generate_block_permutation(N, SPLIT_AXES, [0, 1])
        + generate_grouped_permutation(N, SPLIT_AXES, [2])
        + generate_grouped_permutation(N, SPLIT_AXES, [3])
    )
    return perm_l, perm_r


def _as_shape(view, target_shape, *, mode):
    """No-op if ``view`` already has ``target_shape``; otherwise apply the
    requested transformation. ``mode`` is ``"broadcast"`` (introduce missing
    1-len axes via ``jnp.broadcast_to``) or ``"reshape"`` (collapse axes the
    sizes already line up for)."""
    target = tuple(target_shape)
    if view.shape == target:
        return view
    return (
        jnp.broadcast_to(view, target) if mode == "broadcast" else view.reshape(target)
    )


class _FrameOperands(NamedTuple):
    """The two operands of the frame einsum, plus the slot bookkeeping.

    ``lhs`` / ``rhs`` are COMPACT: a nominal merged slot that carries nothing
    was never built. ``lhs_shape`` / ``rhs_shape`` are the NOMINAL
    ``3 * N + leftovers`` shapes and ``keep_l`` / ``keep_r`` say which of those
    slots survived, so ``_frame_sublists`` can label the nominal layout — the
    only layout the labels are defined on — and the emission site then drops the
    same slots from the label lists.
    """

    lhs: Any
    rhs: Any
    lhs_shape: list
    rhs_shape: list
    keep_l: list
    keep_r: list
    lhs_leftover: list
    rhs_leftover: list


def _merged_keep(N, pairs, lhs_shape, rhs_shape):
    """Which NOMINAL merged slots the frame has to build.

    A slot carries nothing when its extent is 1 AND the output does not carry
    its label. The einsum sums a label absent from the output, and a sum over
    one element is the identity; where the partner carries the same label at
    extent N the einsum sums the PARTNER instead, which is the same product. So
    the number is the same whether or not the axis exists.

    A label the output does carry stays even at extent 1, because einsum has to
    produce it from somewhere: that is both block groups, both leftovers, the
    meta of every pair, and the split of every pair that rides through.
    """
    keep_l = [True] * len(lhs_shape)
    keep_r = [True] * len(rhs_shape)
    for i, p in enumerate(pairs):
        # The meta rides through, so only the side that stores nothing along it
        # is droppable, and only against a partner that stores something.
        el, er = int(lhs_shape[i]), int(rhs_shape[i])
        if el == 1 and er != 1:
            keep_l[i] = False
        elif er == 1 and el != 1:
            keep_r[i] = False
        if p.pairing_type != "contract":
            continue        # a riding-through split is in the output
        if int(lhs_shape[2 * N + i]) == 1:
            keep_l[2 * N + i] = False
        if int(rhs_shape[N + i]) == 1:
            keep_r[N + i] = False
    return keep_l, keep_r


def _prepare_contraction_views(
    lhs_val,
    rhs_val,
    pairs,
    shared,
    total,
    split,
    slots_l,
    slots_r,
    keep_l=None,
    keep_r=None,
    keep_sl=None,
    keep_sr=None,
):
    """``keep_l[i]`` False means the lhs stores nothing along pair ``i``'s meta
    axis, so that axis stays at length 1 on the lhs instead of being broadcast
    to ``total[i]`` (ticket dsnn-3qm.67). ``keep_sl`` / ``keep_sr`` do the same
    for the CONTRACTED axis. Either way the einsum then gives the size-1 axis a
    private label and sums it away, which is free (see ``_frame_sublists``).
    With every keep True this is the incumbent frame.

    Any ``_as_shape(mode="broadcast")`` that still grows a buffer here is the
    genuine least-common-multiple tiling of a misaligned grid, which no
    labelling can avoid."""
    N = len(pairs)
    keep_l = [True] * N if keep_l is None else keep_l
    keep_r = [True] * N if keep_r is None else keep_r
    keep_sl = [True] * N if keep_sl is None else keep_sl
    keep_sr = [True] * N if keep_sr is None else keep_sr
    split_l = [split[i] if keep_sl[i] else 1 for i in range(N)]
    split_r = [split[i] if keep_sr[i] else 1 for i in range(N)]
    lhs_leftover, rhs_leftover = (
        list(lhs_val.shape[slots_l.rank :]),
        list(rhs_val.shape[slots_r.rank :]),
    )

    def split_shape(val, slots, pairs_side, lens, is_lhs):
        out = []
        for i, p in enumerate(pairs):
            ax0, ax1, ax2 = (
                _slot_len(val, slots, 3 * i),
                _slot_len(val, slots, 3 * i + 1),
                _slot_len(val, slots, 3 * i + 2),
            )
            ps = pairs_side[i]
            if is_lhs:
                tail = (1, 1) if ax2 == 1 else (total[i] // ps.outer_len, split[i])
                out.extend([ax0, ax1, *tail])
            else:
                tail = (1, 1) if ax1 == 1 else (total[i] // ps.outer_len, split[i])
                out.extend([ax0, *tail, ax2])
        return out + lens

    lhs_split = split_shape(
        lhs_val, slots_l, [p.lhs for p in pairs], lhs_leftover, True
    )
    rhs_split = split_shape(
        rhs_val, slots_r, [p.rhs for p in pairs], rhs_leftover, False
    )
    perm_l, perm_r = _contraction_perms(N)
    perm_l.extend(range(4 * N, len(lhs_split)))
    perm_r.extend(range(4 * N, len(rhs_split)))

    def reshape_transpose(val, split_list, perm):
        v = val.reshape(split_list) if tuple(split_list) != val.shape else val
        if perm != list(range(len(split_list))):
            v = v.transpose(perm)
        return v

    lhs_view = reshape_transpose(lhs_val, lhs_split, perm_l)
    rhs_view = reshape_transpose(rhs_val, rhs_split, perm_r)
    lhs_unmerged = (
        [
            v
            for i, p in enumerate(pairs)
            for v in (
                (p.lhs.outer_len, total[i] // p.lhs.outer_len) if keep_l[i] else (1, 1)
            )
        ]
        + [p.lhs.block_len for p in pairs]
        + split_l
        + lhs_leftover
    )
    rhs_unmerged = (
        [
            v
            for i, p in enumerate(pairs)
            for v in (
                (p.rhs.outer_len, total[i] // p.rhs.outer_len) if keep_r[i] else (1, 1)
            )
        ]
        + split_r
        + [p.rhs.shared_block_len for p in pairs]
        + rhs_leftover
    )
    lhs_view = _as_shape(lhs_view, lhs_unmerged, mode="broadcast")
    rhs_view = _as_shape(rhs_view, rhs_unmerged, mode="broadcast")
    lhs_meta = [total[i] if keep_l[i] else 1 for i in range(N)]
    rhs_meta = [total[i] if keep_r[i] else 1 for i in range(N)]
    lhs_merged = lhs_meta + [p.lhs.block_len for p in pairs] + split_l + lhs_leftover
    rhs_merged = (
        rhs_meta + split_r + [p.rhs.shared_block_len for p in pairs] + rhs_leftover
    )
    # The merge from the 4N split layout down to the 3N slot layout is a reshape
    # that always happens, so the compact slot layout costs nothing extra: give
    # the reshape the compact target and the unit slots are never built.
    keep_ml, keep_mr = _merged_keep(N, pairs, lhs_merged, rhs_merged)
    lhs_view = _as_shape(
        lhs_view, [d for d, k in zip(lhs_merged, keep_ml) if k], mode="reshape"
    )
    rhs_view = _as_shape(
        rhs_view, [d for d, k in zip(rhs_merged, keep_mr) if k], mode="reshape"
    )
    lhs_bc, rhs_bc = [], []
    for i, p in enumerate(pairs):
        lhs_bc.extend(
            [p.lhs.outer_len, p.lhs.block_len, (total[i] // p.lhs.outer_len) * split[i]]
        )
        rhs_bc.extend(
            [
                p.rhs.outer_len,
                (total[i] // p.rhs.outer_len) * split[i],
                p.rhs.shared_block_len,
            ]
        )
    frame = _FrameOperands(
        lhs_view,
        rhs_view,
        lhs_merged,
        rhs_merged,
        keep_ml,
        keep_mr,
        lhs_leftover,
        rhs_leftover,
    )
    return frame, lhs_bc, rhs_bc


def _tiled_index(p, gcd_len, lcm_len):
    a, b = p.lhs.outer_len, p.rhs.outer_len
    if gcd_len == lcm_len:
        return np.arange(lcm_len), lcm_len
    r = np.arange(lcm_len)
    idx = (
        (r // (lcm_len // gcd_len)) * ((a // gcd_len) * (b // gcd_len))
        + ((r // (lcm_len // a)) % (a // gcd_len)) * (b // gcd_len)
        + ((r // (lcm_len // b)) % (b // gcd_len))
    )
    return idx, gcd_len * (a // gcd_len) * (b // gcd_len)


def _reduce_grid(res_view, pairs, shared, total, lhs_block_lens, rhs_block_lens):
    N = len(pairs)
    per_idx, per_num = [], []
    for i, p in enumerate(pairs):
        idx, num = _tiled_index(p, shared[i], total[i])
        per_idx.append(idx)
        per_num.append(num)
    flat_idx = np.zeros(tuple(total), dtype=np.int32)
    for i in range(N):
        shape = [1] * N
        shape[i] = total[i]
        flat_idx += per_idx[i].reshape(shape) * (
            math.prod(per_num[i + 1 :]) if i + 1 < N else 1
        )
    extra = lhs_block_lens + rhs_block_lens
    flat_arr = flat_idx.flatten()
    if np.array_equal(flat_arr, np.arange(len(flat_arr))):
        return res_view.reshape(*per_num, *extra)
    # Note: a "pure permutation" branch (unique non-identity) is structurally unreachable
    # with the current ``_tiled_index`` formula — every misaligned (gcd < lcm) case
    # produces collisions in adjacent r values, and the mixed-radix combination across
    # multiple pairs preserves those collisions. ``segment_sum`` handles both pure-
    # permutation and true-collision cases correctly, so we always fall through here.
    # ONE-HOT DOT (#51, revived 2026-08-04): ``segment_sum`` is an unsorted
    # scatter-add — GPU atomics, a fusion barrier — driven here by a STATIC
    # numpy index map. Contract against a constant one-hot instead:
    # identical groups, deterministic order, and the dot fuses. VALUE-
    # identical within float reordering (the first attempt passed every
    # allclose); the _PIN_LAYOUT byte-identity replicas in
    # explicit_matmul_test.py mirror this same reduction. A >4M-entry
    # safety valve keeps segment_sum for pathological grids.
    _n_src = math.prod(total)
    _n_seg = math.prod(per_num)
    if _n_src * _n_seg <= 4_000_000:
        _onehot = np.zeros((_n_seg, _n_src), dtype=np.float32)
        _onehot[flat_arr, np.arange(_n_src)] = 1.0
        _rv = res_view.reshape(_n_src, math.prod(extra))
        _oh = jnp.asarray(_onehot, dtype=_rv.dtype)
        res = jnp.einsum(_oh, [0, 1], _rv, [1, 2], [0, 2])
    else:
        res = jax.ops.segment_sum(
            res_view.reshape(math.prod(total), math.prod(extra)),
            jnp.array(flat_arr),
            num_segments=math.prod(per_num),
        )
    return res.reshape(*per_num, *extra)


def _frame_sublists(N, pairs, lhs_shape, rhs_shape, n_ll, n_rl):
    """Integer sublists for the frame contraction, plus the output order.

    Called on the NOMINAL slot layout — three slots per pair — which is the only
    layout the labels are defined on. The operands themselves build only the
    slots that carry data (``_merged_keep``), and ``_frame_contract`` drops the
    rest from the two lists it gets back. ``out_sub`` is unaffected: every label
    it carries is built by at least one operand.

    The nominal layout is:

      lhs  ``[meta_i] + [lhs block_i] + [split_i] + lhs leftover``
      rhs  ``[meta_i] + [split_i] + [rhs shared block_i] + rhs leftover``

    Every slot gets an integer label. A label on both operands means the two
    axes meet: the meta axes ride through to the output, and a ``split`` axis
    is contracted when its pair contracts and rides through otherwise.

    An axis that its operand does not store sits at extent 1 against a partner
    of extent N. That is the case the incumbent frame handled by broadcasting
    the size-1 side up to N before a ``dot_general``, which is the tiled path's
    single largest materialization (finding 61: 56.9 MB against 378 KB on
    TLM/CPU). Here the size-1 axis is given a PRIVATE label instead. It is then
    absent from the output, so einsum sums it, and summing an extent-1 axis is
    the identity. The partner's full axis carries the result:

      * a meta axis only one side stores rides as that side's free axis,
      * a contracted axis only one side stores becomes a plain sum over that
        side, which is the same number the broadcast dot produced.

    No buffer is written for either. Whether XLA re-introduces a broadcast is
    XLA's decision, per shape and per device, which is the point of stating the
    contraction rather than pinning one lowering.

    The output order is the canonical one the rest of the pipeline reads:
    metas, the ridden-through splits, the lhs blocks, the rhs shared blocks,
    then the two leftovers.
    """
    meta = list(range(N))                       # M_i
    split = [N + i for i in range(N)]           # S_i
    blk_l = [2 * N + i for i in range(N)]       # B_i, lhs only
    blk_r = [3 * N + i for i in range(N)]       # F_i, rhs only
    left_l = [4 * N + k for k in range(n_ll)]
    left_r = [4 * N + n_ll + k for k in range(n_rl)]
    nxt = 4 * N + n_ll + n_rl                   # private labels start here

    lhs_sub = meta + blk_l + split + left_l
    rhs_sub = meta + split + blk_r + left_r
    for i in range(N):
        # (lhs slot, rhs slot) of the two labels the two operands share.
        for la, ra in ((i, i), (2 * N + i, N + i)):
            el, er = int(lhs_shape[la]), int(rhs_shape[ra])
            if el == er:
                continue
            if el == 1:
                lhs_sub[la] = nxt
            elif er == 1:
                rhs_sub[ra] = nxt
            else:
                raise ValueError(
                    f"frame contraction pair {i}: extents {el} and {er} meet "
                    "on the same axis and neither is 1"
                )
            nxt += 1
    ride = [i for i, p in enumerate(pairs) if p.pairing_type != "contract"]
    out_sub = (
        meta
        + [split[i] for i in ride]
        + blk_l
        + blk_r
        + left_l
        + left_r
    )
    return lhs_sub, rhs_sub, out_sub


def _frame_contract(frame: "_FrameOperands", pairs):
    """The one emission site of the tiled frame contraction: ONE einsum.

    The labels are assigned on the NOMINAL slot layout, the only layout they are
    defined on, and then the slots the operands do not build (see
    ``_merged_keep``) are dropped from the two label lists. ``out_sub`` is
    untouched: every label it carries is built by at least one operand."""
    lhs_sub, rhs_sub, out_sub = _frame_sublists(
        len(pairs),
        pairs,
        frame.lhs_shape,
        frame.rhs_shape,
        len(frame.lhs_leftover),
        len(frame.rhs_leftover),
    )
    lhs_sub = [s for s, k in zip(lhs_sub, frame.keep_l) if k]
    rhs_sub = [s for s, k in zip(rhs_sub, frame.keep_r) if k]
    return _emit_einsum(frame.lhs, lhs_sub, frame.rhs, rhs_sub, out_sub)


def _final_grid(N, shared, lhs_bc, rhs_bc, lhs_block_lens, rhs_block_lens):
    grid = []
    for i in range(N):
        grid.extend(
            [
                shared[i],
                lhs_bc[AXES_PER_PAIR * i] // shared[i],
                rhs_bc[AXES_PER_PAIR * i] // shared[i],
            ]
        )
    grid.extend(lhs_block_lens + rhs_block_lens)
    return grid


def _finalize_output(
    N, res_raw, total, pairs, split, shared, lhs_bc, rhs_bc, lhs_leftover, rhs_leftover
):
    non_contract = [i for i, p in enumerate(pairs) if p.pairing_type != "contract"]
    d_ls = [p.lhs.block_len for p in pairs]
    f_ls = [p.rhs.shared_block_len for p in pairs]
    ss_out = [split[i] if i in non_contract else 1 for i in range(N)]
    expanded = list(total) + ss_out + d_ls + f_ls + lhs_leftover + rhs_leftover
    res = res_raw.reshape(expanded) if res_raw.shape != tuple(expanded) else res_raw
    fast_perm = (
        list(range(N))
        + [
            ax
            for i in range(N)
            for ax in (
                (2 * N + i, N + i)
                if pairs[i].pairing_type == "spatial_sparse_rhs"
                else (2 * N + i,)
            )
        ]
        + [
            ax
            for i in range(N)
            for ax in (
                (3 * N + i,)
                if pairs[i].pairing_type == "spatial_sparse_rhs"
                else (N + i, 3 * N + i)
            )
        ]
        + list(range(4 * N, len(expanded)))
    )
    final_lhs_lens = [
        d_ls[i] * (ss_out[i] if pairs[i].pairing_type == "spatial_sparse_rhs" else 1)
        for i in range(N)
    ]
    final_rhs_lens = [
        f_ls[i] * (ss_out[i] if pairs[i].pairing_type != "spatial_sparse_rhs" else 1)
        for i in range(N)
    ]
    target = (*total, *final_lhs_lens, *final_rhs_lens, *lhs_leftover, *rhs_leftover)
    if fast_perm != list(range(len(fast_perm))):
        res_view = res.transpose(fast_perm)
        if res_view.shape != target:
            res_view = res_view.reshape(target)
    else:
        res_view = res.reshape(target) if res.shape != target else res
    if any(shared[i] != total[i] for i in range(N)):
        res = _reduce_grid(
            res_view,
            pairs,
            shared,
            total,
            final_lhs_lens,
            final_rhs_lens + lhs_leftover + rhs_leftover,
        )
    else:
        res = res_view
    grid = _final_grid(N, shared, lhs_bc, rhs_bc, final_lhs_lens, final_rhs_lens)
    grid.extend(lhs_leftover + rhs_leftover)
    perm_out = (
        generate_grouped_permutation(N, GRID_AXES_PER_PAIR, [0])
        + [
            ax
            for i in range(N)
            for ax in (GRID_AXES_PER_PAIR * i + 1, GRID_AXES_PER_PAIR * N + i)
        ]
        + [
            ax
            for i in range(N)
            for ax in (GRID_AXES_PER_PAIR * i + 2, (GRID_AXES_PER_PAIR + 1) * N + i)
        ]
        + list(range(5 * N, len(grid)))
    )
    if res.shape != tuple(grid):
        res = res.reshape(grid)
    if perm_out != list(range(len(grid))):
        res = res.transpose(perm_out)
    return res, final_lhs_lens, final_rhs_lens


# --- Lazy frame (ticket dsnn-3qm.67) ---------------------------------------
# The physical extents a Pair slot may claim, read off the PREPARED operand
# arrays: slot ``3*i + k`` of a prepared array is 1 exactly when no val axis
# backs it. ``_prepare_contraction_views`` used to broadcast every such 1 up
# to the logical length before the dot, which is where the output structure
# died (finding 61, verdicts 3, 5 and 6) and where the CPU temp went.
_LAZY_PAIRINGS = frozenset(
    {
        "contract",
        "batch_out",
        "batch_primal",
        "batch_sparse",
        "spatial_out_lhs",
        "spatial_out_rhs",
        "spatial_primal_lhs",
        "spatial_primal_rhs",
    }
)


def _slot_phys(val, slots, i):
    return (
        _slot_len(val, slots, 3 * i),
        _slot_len(val, slots, 3 * i + 1),
        _slot_len(val, slots, 3 * i + 2),
    )


def _lazy_frame(lhs_val, rhs_val, pairs, slots_l, slots_r):
    """Shrink every frame slot that neither operand stores.

    Returns ``(pairs_eff, lazy, demote)``.

    * ``pairs_eff`` — the Pair list with each unstored extent set to 1, so the
      whole existing pipeline (``_contraction_factors`` down to
      ``_final_grid``) builds the SMALL physical grid and no
      ``_as_shape(mode="broadcast")`` grows a buffer.
    * ``lazy[i]`` — which of pair ``i``'s three output slots therefore has no
      physical axis. ``_build_pair_dims`` gives those the logical extent and
      ``axis=None``: a free implicit dim stays implicit, a surviving diagonal
      pair stays a pair, a fully implicit result keeps ``val=None``.
    * ``demote[i]`` — the meta axis is stored on exactly one side; it leaves
      the dot's batch list instead of broadcasting the other side.

    A contracted extent neither side stores is an analytic scale: setting both
    of its lens to 1 makes ``_contraction_factors``'s existing scalar rule fold
    ``logical_element_count`` into ``scalar_mult`` — no dot over a broadcast.

    Anything this cannot prove (a genuine LCM grid, a spatial-sparse pair, a
    partially stored extent) keeps the incumbent frame slot for slot.

    The demotion used to be a per-device choice, because a ``dot_general``
    forced one: XLA on GPU fused the broadcast into the batched dot and lost
    the fusion once the axis left the batch list (11 percent on TLM/GPU),
    while XLA on CPU allocated the broadcast instead (56.9 MB against 378 KB,
    finding 61). The einsum emission does not force the choice. The axis is
    stated, never broadcast, and XLA re-introduces the broadcast when it wants
    it — so the demotion is now unconditional."""
    eff, lazy, demote = [], [], []
    for i, p in enumerate(pairs):
        lo_p, lb_p, ls_p = _slot_phys(lhs_val, slots_l, i)
        ro_p, rb_p, rs_p = _slot_phys(rhs_val, slots_r, i)
        l, r = p.lhs, p.rhs
        ol, orr = int(l.outer_len), int(r.outer_len)
        T, G = math.lcm(ol, orr), math.gcd(ol, orr)
        # ``_tiled_index`` is the identity only on an aligned grid; a genuine
        # LCM grid (both sizes > 1 and unequal) keeps the incumbent path.
        aligned = T == G or ol == 1 or orr == 1
        can = p.pairing_type in _LAZY_PAIRINGS
        # What each side physically contributes to the merged meta axis.
        m_l = lo_p * ((T // ol) if ls_p != 1 else 1)
        m_r = ro_p * ((T // orr) if rb_p != 1 else 1)
        meta_lazy = False
        dem = None
        if can and aligned and T > 1:
            if m_l == 1 and m_r == 1:
                meta_lazy = True
            elif m_l == T and m_r == 1:
                dem = "r"       # the rhs stores nothing along this meta axis
            elif m_r == T and m_l == 1:
                dem = "l"
        nl, nr = l, r
        if meta_lazy:
            nl = nl._replace(outer_len=1)
            nr = nr._replace(outer_len=1)
        # Surviving own-side extents nobody stores.
        lhs_lazy = can and int(l.block_len) > 1 and lb_p == 1
        rhs_lazy = can and int(r.shared_block_len) > 1 and rs_p == 1
        if lhs_lazy:
            nl = nl._replace(block_len=1)
        if rhs_lazy:
            nr = nr._replace(shared_block_len=1)
        # The two one-sided pairings park their extent in the OTHER slot of the
        # triple, from where ``_finalize_output`` folds it through ``split`` and
        # ``ss_out`` into the rhs half of the grid.
        if can and p.pairing_type == "spatial_primal_lhs" \
                and ls_p == 1 and int(l.shared_block_len) > 1:
            nl = nl._replace(shared_block_len=1)
            rhs_lazy = True
        if can and p.pairing_type == "spatial_out_rhs" \
                and rb_p == 1 and int(r.block_len) > 1:
            nr = nr._replace(block_len=1)
            rhs_lazy = True
        # A contracted extent no operand stores is an analytic scale, but that
        # is decided on the CONTRACTED axis in the driver, not here — see the
        # ``split`` loop of _execute_block_sparse_contraction.
        np_ = p
        if nl is not l or nr is not r:
            np_ = p._replace(lhs=nl, rhs=nr)
        if meta_lazy:
            # ``_contraction_factors``'s scalar rule reads a contract pair with
            # both outer lens 1 as "the contracted extent exists only in the
            # metadata, fold logical_element_count". Shrinking the meta above
            # makes that condition true by accident: here the meta is a real
            # diagonal RIDING THROUGH to the output, not a summed axis.
            np_ = np_._replace(logical_element_count=1)
        eff.append(np_)
        lazy.append(_Lazy(meta_lazy, lhs_lazy, rhs_lazy))
        demote.append(dem)
    return eff, lazy, demote


def _execute_block_sparse_contraction(
    lhs_val, rhs_val, pairs, ctx: "Ctx", slots_l, slots_r
):
    N = len(pairs)
    true_pairs = pairs
    pairs, lazy, demote = _lazy_frame(lhs_val, rhs_val, pairs, slots_l, slots_r)
    # ``frame_changed`` means the OUTPUT geometry moved, so the band probes
    # (which read that geometry) sit this one out. A demotion or a summed
    # contracted axis leaves the geometry alone and only changes the buffers.
    frame_changed = pairs != true_pairs
    is_lazy = frame_changed or any(demote)
    shared, total, split, scalar = _contraction_factors(pairs)
    if is_lazy:
        # The CONTRACTED length is a property of the operands, not of the frame:
        # shrinking a meta axis must not re-read a diagonal's meta as extra
        # contraction depth. Take it from the logical topology.
        _, _, split_true, _ = _contraction_factors(true_pairs)
        split = [
            split_true[i] if pairs[i].pairing_type == "contract" else split[i]
            for i in range(N)
        ]
    # A contracted axis only ONE operand stores is a plain sum over that
    # operand: the same number the broadcast dot produced, at none of its cost.
    # Keeping the non-storing side at extent 1 is all that is needed — the
    # einsum gives that axis a private label and sums the storing side for us
    # (see ``_frame_sublists``). Stored by NEITHER side, the contraction is an
    # analytic scale: nothing physical is left to sum, and the logical length
    # only exists in the topology, so it is folded into ``scalar`` here.
    keep_sl, keep_sr = [True] * N, [True] * N
    split_fold = 1
    for i, p in enumerate(pairs):
        if p.pairing_type != "contract" or split[i] <= 1:
            continue
        st_l = _slot_len(lhs_val, slots_l, 3 * i + 2) != 1
        st_r = _slot_len(rhs_val, slots_r, 3 * i + 1) != 1
        if st_l and not st_r:
            keep_sr[i] = False
        elif st_r and not st_l:
            keep_sl[i] = False
        elif not st_l and not st_r:
            keep_sl[i] = keep_sr[i] = False
            split_fold *= split[i]
    frame, lhs_bc, rhs_bc = _prepare_contraction_views(
        lhs_val,
        rhs_val,
        pairs,
        shared,
        total,
        split,
        slots_l,
        slots_r,
        keep_l=[d != "l" for d in demote],
        keep_r=[d != "r" for d in demote],
        keep_sl=keep_sl,
        keep_sr=keep_sr,
    )
    if split_fold != 1:
        scalar *= float(split_fold)
        is_lazy = True
    lhs_leftover = frame.lhs_leftover
    rhs_leftover = frame.rhs_leftover
    # ONE einsum, straight into the canonical output order. No broadcast to
    # line the operands up, no squeeze of a demoted twin, no transpose back.
    res_raw = _frame_contract(frame, pairs)
    grid, final_lhs_lens, final_rhs_lens = _finalize_output(
        N,
        res_raw,
        total,
        pairs,
        split,
        shared,
        lhs_bc,
        rhs_bc,
        lhs_leftover,
        rhs_leftover,
    )
    if not is_lazy:
        return grid, shared, final_lhs_lens, final_rhs_lens, scalar, None
    true_shared, true_total, true_split, _ = _contraction_factors(true_pairs)
    true_fll, true_frl = _compact_block_lens(
        true_pairs, true_shared, true_total, true_split
    )
    # A slot may only stay symbolic when its EFFECTIVE grid axis really is 1;
    # otherwise the axis carries content and dropping it would take slice 0 of
    # live data. This is the safety net for every rule above.
    checked = []
    for i in range(N):
        z = lazy[i]
        f_l = (pairs[i].lhs.outer_len // shared[i]) * final_lhs_lens[i]
        f_r = (pairs[i].rhs.outer_len // shared[i]) * final_rhs_lens[i]
        checked.append(
            _Lazy(
                z.shared and shared[i] == 1,
                z.lhs and f_l == 1,
                z.rhs and f_r == 1,
            )
        )
    return (
        grid,
        shared,
        final_lhs_lens,
        final_rhs_lens,
        scalar,
        (pairs, checked, true_shared, true_fll, true_frl),
    )


# --- Output tensor build --------------------------------------------------
def _resolve_output_shape(ctx, res):
    # The grid was built on the EFFECTIVE pairs when the lazy frame fired, so
    # the shape must be read off those, not off the logical topology.
    pairs = ctx.pairs if res.eff_pairs is None else res.eff_pairs
    shape, sh_map, lhs_map, rhs_map, squeeze, ax = [], {}, {}, {}, [], 0
    for i, factor in enumerate(res.shared_factors):
        shape.append(factor)
        sh_map[i] = ax
        ax += 1
    for i, p in enumerate(pairs):
        factor = p.lhs.outer_len // res.shared_factors[i]
        if p.pairing_type == "spatial_sparse_lhs":
            shape.extend([factor, res.lhs_block_lens[i]])
            squeeze.append(sh_map[i])
            sh_map[i] = ax
            ax += 1
            lhs_map[i] = ax
            ax += 1
        else:
            shape.append(factor * res.lhs_block_lens[i])
            lhs_map[i] = ax
            ax += 1
    for i, p in enumerate(pairs):
        factor = p.rhs.outer_len // res.shared_factors[i]
        if p.pairing_type == "spatial_sparse_rhs":
            shape.extend([factor, res.rhs_block_lens[i]])
            squeeze.append(sh_map[i])
            sh_map[i] = ax
            ax += 1
            rhs_map[i] = ax
            ax += 1
        else:
            shape.append(factor * res.rhs_block_lens[i])
            rhs_map[i] = ax
            ax += 1
    shape.extend(res.grid.shape[5 * len(ctx.pairs) :])
    return shape, (sh_map, lhs_map, rhs_map), squeeze


def _build_sparse(
    dim_id, other_id, outer_sz, outer_val, outer_pres, inner_sz, inner_val, inner_pres
):
    if outer_sz == 1:
        return DenseIndex(dim_id, inner_sz, axis=inner_val if inner_pres else None)
    return DiagonalIndex(
        dim_id,
        outer_sz,
        axis=outer_val if outer_pres else None,
        other_id=other_id,
        block_size=inner_sz if inner_sz > 1 else None,
        block_axis=inner_val if inner_pres and inner_sz > 1 else None,
    )


def _dense_survivor(dim_id, logical, phys, axis, pres):
    """The surviving DENSE dim of one side of a contraction.

    ``logical`` is the extent the topology says the dim spans; ``phys`` is what
    the grid axis ``axis`` actually holds. The lazy frame (dsnn-3qm.67) leaves an
    extent at 1 when no operand stores it, so the two differ whenever part of
    this dim rides implicitly, and the job here is to describe which part:

    * nothing physical — one implicit dense dim of the full logical extent,
      ``axis=None``; ``dense()`` broadcasts it back.
    * all of it physical — a plain dense dim.
    * the META physical and the BLOCK implicit — a BLOCKED DENSE dim
      (dsnn-3qm.62): ``size`` is the stored meta, ``block_size`` the implicit
      block, ``block_axis`` None. This is the COMPRESS'd-block Jacobian: ``val``
      keeps one entry per block, the block extent is uniform inside each block,
      and the contraction materializes NONE of it. ``logical % phys == 0`` is
      what makes the statement well formed — ``phys`` blocks of ``logical //
      phys`` positions each, in that order, because ``axis`` is the OUTER
      pointer of an ``Index``.

    The fourth combination — the block physical but the meta implicit — is an
    "implicit outer, explicit inner" dim no ``Index`` can describe, so it raises
    instead of mislabelling the stored axis as the meta."""
    if not pres:
        return DenseIndex(dim_id, logical, axis=None)
    if phys == logical:
        # Including 1 == 1: a size-1 survivor keeps its physical axis, as it
        # always has. Demoting it to implicit here orphans that val axis and
        # takes a compressible axis off the micro-action slot list.
        return DenseIndex(dim_id, logical, axis=axis)
    if phys == 1:
        return DenseIndex(dim_id, logical, axis=None)
    if logical % phys == 0:
        return Index(dim_id, phys, axis, None, logical // phys, None)
    raise ValueError(
        f"matmul: contraction survivor id={dim_id} spans {logical} logical "
        f"positions but its grid axis {axis} holds {phys}, which does not "
        f"divide it. A blocked dense survivor needs the stored extent to be "
        f"the OUTER (block-count) factor of the logical one (dsnn-3qm.62)."
    )


def _with_implicit_block(dim, src):
    """Re-attach a PASS-THROUGH dim's implicit block.

    A spatial (uncontracted) pairing carries the source dim's PHYSICAL extent
    through the frame and nothing else: the contraction never reads, splits or
    sums it. So a blocked dense source dim's ``block_size`` is metadata that
    comes back verbatim — only the ``size`` had to survive the frame, and the
    guard below states exactly that."""
    if not _is_blocked_dense(src) or dim is None:
        return dim
    if int(dim.size) != int(src.size) or dim.block_size is not None:
        raise ValueError(
            f"matmul: pass-through of blocked dense dim id={src.id} "
            f"(size={src.size} block_size={src.block_size}) came out of the "
            f"frame as size={dim.size} block_size={dim.block_size}; the frame "
            f"was expected to carry its stored extent unchanged (dsnn-3qm.62)."
        )
    return replace(dim, block_size=src.block_size, block_axis=None)


class _Expand(NamedTuple):
    """One grid axis a surviving PAIR needs at its full logical extent.

    The axis holds ``(outer_eff, block_eff)`` and the dim it backs spans
    ``(outer, block)``; each ``_eff`` is either the full extent or 1, because the
    lazy frame only ever shrinks an extent NO operand stores — so the missing
    positions are uniform and ``broadcast_to`` is their exact materialization.

    WHY A PAIR MUST MATERIALIZE WHERE A DENSE SURVIVOR DOES NOT. A surviving
    ``DiagonalIndex`` already spends ``size`` on the meta it shares with its
    partner and ``block_size`` on its own extent, so a PARTLY implicit own
    extent needs a THIRD field that ``Index`` does not have (and folding the
    implicit factor into the meta would change the PARTNER's logical extent,
    which is not the same tensor). A dense survivor has ``size`` free and
    therefore states it for nothing — see ``_dense_survivor``."""

    axis: int
    outer_eff: int
    block_eff: int
    outer: int
    block: int


def _apply_expands(values, shape, expands):
    """Materialize each ``_Expand`` on ``values``, rank-preservingly.

    ``(… , outer_eff * block_eff, …)`` -> ``(…, outer_eff, block_eff, …)`` ->
    broadcast -> ``(…, outer * block, …)``. The rank never changes, so no other
    dim's ``axis`` moves and the caller's ``shape`` bookkeeping only has to
    update that one entry."""
    for e in expands:
        pre, post = list(shape[: e.axis]), list(shape[e.axis + 1 :])
        values = values.reshape(pre + [e.outer_eff, e.block_eff] + post)
        values = jnp.broadcast_to(values, pre + [e.outer, e.block] + post)
        values = values.reshape(pre + [e.outer * e.block] + post)
        shape[e.axis] = e.outer * e.block
    return values, shape


def _build_pair_dims(pm, i, sa, la, ra, res, next_id):
    """Build (out_dim, primal_dim, next_id, expands) for one pair, dispatching
    on pairing_type."""
    # Sizes come from the LOGICAL topology; the axis maps come from the buffer.
    # On the eager frame the two agree and ``true_*`` is None.
    sf = (res.shared_factors if res.true_shared_factors is None
          else res.true_shared_factors)[i]
    _lbl = res.lhs_block_lens if res.true_lhs_block_lens is None \
        else res.true_lhs_block_lens
    _rbl = res.rhs_block_lens if res.true_rhs_block_lens is None \
        else res.true_rhs_block_lens
    final_l = (pm.lhs.outer_len // sf) * _lbl[i]
    final_r = (pm.rhs.outer_len // sf) * _rbl[i]
    # The PHYSICAL extents of the same two grid axes, read off the effective
    # (buffer) frame exactly as ``_resolve_output_shape`` read the shape off it.
    # ``final_* != phys_*`` is a part of the dim the buffer does not store.
    _eff = pm if res.eff_pairs is None else res.eff_pairs[i]
    of_l = _eff.lhs.outer_len // res.shared_factors[i]
    of_r = _eff.rhs.outer_len // res.shared_factors[i]
    phys_l = of_l * res.lhs_block_lens[i]
    phys_r = of_r * res.rhs_block_lens[i]
    expands: list[_Expand] = []
    z = _NO_LAZY if res.lazy is None else res.lazy[i]
    pres_shared = pm.lhs.outer_axis is not None or pm.rhs.outer_axis is not None
    pres_lhs = pm.lhs.outer_axis is not None or pm.lhs.block_axis is not None
    pres_rhs = (
        pm.rhs.outer_axis is not None
        or getattr(pm.rhs, "shared_block_axis", None) is not None
    )
    any_val = (
        pres_shared
        or pres_lhs
        or pres_rhs
        or pm.lhs.shared_block_axis is not None
        or getattr(pm.rhs, "block_axis", None) is not None
    )
    if any_val:
        pres_shared = pres_shared or sf > 1
        pres_lhs = pres_lhs or final_l > 1
        pres_rhs = pres_rhs or final_r > 1
    # The lazy frame never built a physical axis for these slots, so the
    # ``any_val`` forcing above must not claim one: the extent stays symbolic
    # (the planner's ``out:implicit_kept`` / ``out:pair_retained``).
    pres_shared = pres_shared and not z.shared
    pres_lhs = pres_lhs and not z.lhs
    pres_rhs = pres_rhs and not z.rhs

    def gen(v):
        nonlocal next_id
        if v == "next":
            n = next_id
            next_id += 1
            return n
        return v

    l_id = gen(pm.lhs.dim.id if pm.lhs.dim else "next")
    r_id = gen(pm.rhs.dim.id if pm.rhs.dim else "next")
    ls_id = gen(pm.lhs.shared_dim.id if pm.lhs.shared_dim else "next")
    rs_id = gen(pm.rhs.shared_dim.id if pm.rhs.shared_dim else "next")
    pt = pm.pairing_type
    out_dim = primal_dim = None
    if pt == "contract":
        if pm.lhs.dim and pm.rhs.shared_dim:
            # The pair SURVIVES as a pair, on both sides. ``_build_sparse`` puts
            # the surviving extent in ``block_size`` and points ``block_axis`` at
            # the grid axis, so a part of it the buffer does not store can only
            # be described by leaving ``block_axis`` None (``inner_pres`` False,
            # which the lazy frame's own ``z`` already does). Anything else is
            # metadata that lies about the buffer — say so here rather than in
            # SparseTensor's topology check.
            if pres_lhs and phys_l != final_l:
                expands.append(_Expand(
                    la, of_l, res.lhs_block_lens[i],
                    pm.lhs.outer_len // sf, _lbl[i]))
            if pres_rhs and phys_r != final_r:
                expands.append(_Expand(
                    ra, of_r, res.rhs_block_lens[i],
                    pm.rhs.outer_len // sf, _rbl[i]))
            out_dim = _build_sparse(
                l_id, rs_id, sf, sa, pres_shared, final_l, la, pres_lhs
            )
            primal_dim = _build_sparse(
                rs_id, l_id, sf, sa, pres_shared, final_r, ra, pres_rhs
            )
        elif pm.lhs.dim:
            out_dim = _dense_survivor(l_id, final_l, phys_l, la, pres_lhs)
        elif pm.rhs.shared_dim:
            primal_dim = _dense_survivor(rs_id, final_r, phys_r, ra, pres_rhs)
    elif pt == "batch_out":
        out_dim = DenseIndex(
            l_id if pm.lhs.dim else ls_id, sf, axis=sa if pres_shared else None
        )
    elif pt == "batch_primal":
        primal_dim = DenseIndex(
            l_id if pm.lhs.dim else ls_id, sf, axis=sa if pres_shared else None
        )
    elif pt == "spatial_out_lhs":
        out_dim = _with_implicit_block(
            DenseIndex(l_id, final_l, axis=la if pres_lhs else None), pm.lhs.dim
        )
    elif pt == "spatial_out_rhs":
        out_pres = getattr(pm.rhs, "block_axis", None) is not None
        out_dim = _with_implicit_block(
            DenseIndex(r_id, final_r, axis=ra if out_pres else None), pm.rhs.dim
        )
    elif pt == "spatial_primal_lhs":
        # A one-sided PRIMAL dim rides in ``PairData.shared_block_len``, which
        # ``_finalize_output`` folds into the RHS half of the grid (``split`` ->
        # ``ss_out`` -> ``final_rhs_lens``); the LHS half is 1 for this pairing.
        # Reading ``final_l`` / ``la`` here collapsed the extent to 1 and took
        # slice 0 of the values — finding 61, verdict 6: NeuralNetwork, reverse
        # order, Reduce on slot ``new``, where the chain ends in ``X @ scalar``
        # and every primal dim of ``X`` is such a pair.
        prim_pres = pm.lhs.shared_block_axis is not None
        primal_dim = _with_implicit_block(
            DenseIndex(ls_id, final_r, axis=ra if prim_pres else None),
            pm.lhs.shared_dim,
        )
    elif pt == "spatial_primal_rhs":
        primal_dim = _with_implicit_block(
            DenseIndex(rs_id, final_r, axis=ra if pres_rhs else None),
            pm.rhs.shared_dim,
        )
    elif pt == "batch_sparse":
        out_dim = _build_sparse(l_id, rs_id, sf, sa, pres_shared, final_l, la, pres_lhs)
        primal_dim = _build_sparse(
            rs_id, l_id, sf, sa, pres_shared, final_r, ra, pres_rhs
        )
    elif pt == "spatial_sparse_lhs":
        out_dim = _build_sparse(
            l_id,
            ls_id,
            pm.lhs.outer_len,
            sa,
            pres_shared,
            pm.lhs.block_len,
            la,
            pres_lhs,
        )
        prim_inner_pres = (
            pm.lhs.shared_block_axis is not None if pm.lhs.outer_len == 1 else pres_rhs
        )
        primal_dim = _build_sparse(
            ls_id,
            l_id,
            pm.lhs.outer_len,
            sa,
            pres_shared,
            pm.lhs.shared_block_len,
            ra,
            prim_inner_pres,
        )
    elif pt == "spatial_sparse_rhs":
        out_inner_pres = getattr(pm.rhs, "block_axis", None) is not None
        out_dim = _build_sparse(
            r_id,
            rs_id,
            pm.rhs.outer_len,
            sa,
            pres_shared,
            pm.rhs.block_len,
            la,
            out_inner_pres,
        )
        primal_dim = _build_sparse(
            rs_id,
            r_id,
            pm.rhs.outer_len,
            sa,
            pres_shared,
            pm.rhs.shared_block_len,
            ra,
            pres_rhs,
        )
    return out_dim, primal_dim, next_id, expands


def _meta_is_summed(pm) -> bool:
    """True when pair ``pm``'s grid axes hold PARTIAL sums the output must add up.

    A contract pair's meta rides through to the output only because a surviving
    DiagonalIndex carries it there — ``lhs.dim`` on the out side, or
    ``rhs.shared_dim`` on the primal side. A BLOCKED DENSE contracted dim
    (dsnn-3qm.62) factors its OWN contracted extent into stored blocks x an
    implicit block and has no such partner, so the meta is part of the
    contraction: the einsum leaves one partial sum per block group and they must
    be added, not sliced.

    With neither side carrying a surviving dim, every pre-.62 pair had meta 1 on
    both sides (a meta > 1 came from a DiagonalIndex, which always survives), so
    this sums a single element and is the incumbent behaviour there."""
    return (
        pm.pairing_type == "contract"
        and pm.lhs.dim is None
        and pm.rhs.shared_dim is None
    )


def _pair_output_dims(ctx, rhs_dims, res):
    """The metadata half of the output build.

    Returns ``(out_dims, primal_dims, shape, squeeze, summed, expands)``: the
    per-pair dims in pair order, the grid shape they were read off, the grid axes
    no dim claims that are UNIFORM (slice 0 is exact), the ones that carry live
    partial sums (``_meta_is_summed``), and the axes a surviving pair needs
    materialized (``_Expand``). ``_build_output_tensor`` and ``_output_dims`` each
    carried a verbatim copy of this; they call it now so the dims a contraction
    REPORTS cannot drift from the ones it BUILDS."""
    shape, (sh_map, lhs_map, rhs_map), squeeze = _resolve_output_shape(ctx, res)
    next_id = (
        builtins.max([d.id for d in ctx.lhs.dims] + [d.id for d in rhs_dims] + [-1]) + 1
    )
    out_dims, primal_dims, expands = [], [], []
    for i, pm in enumerate(ctx.pairs):
        od, pd, next_id, ex = _build_pair_dims(
            pm, i, sh_map[i], lhs_map[i], rhs_map[i], res, next_id
        )
        expands.extend(ex)
        if od:
            out_dims.append(od)
        if pd:
            primal_dims.append(pd)
    used_axes = set()
    for d in out_dims + primal_dims:
        if d.axis is not None:
            used_axes.add(d.axis)
        if getattr(d, "block_axis", None) is not None:
            used_axes.add(d.block_axis)
    summed = []
    for i, pm in enumerate(ctx.pairs):
        for ax in (sh_map[i], lhs_map[i], rhs_map[i]):
            if ax not in used_axes:
                (summed if _meta_is_summed(pm) else squeeze).append(ax)
    return out_dims, primal_dims, shape, squeeze, summed, expands


def _store_narrow(values, lhs_dtype, rhs_dtype):
    # A bf16 x bf16 contraction sums in f32 (_emit_einsum) and stores its
    # result bf16 (owner ruling 2026-09-23: the face Quant is two-sided).
    if (values is not None
            and jnp.dtype(lhs_dtype) == jnp.dtype(jnp.bfloat16)
            and jnp.dtype(rhs_dtype) == jnp.dtype(jnp.bfloat16)
            and values.dtype != jnp.dtype(jnp.bfloat16)):
        return values.astype(jnp.bfloat16)
    return values


def _build_output_tensor(ctx, rhs_dims, res):
    from graphax.sparse.tensor import SparseTensor

    out_dims, primal_dims, shape, squeeze, summed, expands = _pair_output_dims(
        ctx, rhs_dims, res
    )
    grid_view = res.grid.reshape(shape) if res.grid.shape != tuple(shape) else res.grid
    if expands:
        grid_view, shape = _apply_expands(grid_view, list(shape), expands)
    if summed:
        # keepdims so every axis index below still means what it meant.
        grid_view = grid_view.sum(axis=tuple(sorted(set(summed))), keepdims=True)
    squeeze = squeeze + summed
    if squeeze:
        unique_sq = tuple(sorted(set(squeeze)))
        final_shape = [s for i, s in enumerate(shape) if i not in unique_sq]
        if grid_view.size == math.prod(final_shape):
            values = grid_view.reshape(final_shape)
        else:
            # NOTE: a squeezed axis may be size>1 but UNIFORM (all slices
            # equal — e.g. a broadcast factor), so slice-0 is exact here.
            idx = tuple(0 if i in unique_sq else slice(None) for i in range(len(shape)))
            values = grid_view[idx]
            if values.shape != tuple(final_shape):
                values = values.reshape(final_shape)

        def shift(v):
            return None if v is None else v - sum(1 for s in unique_sq if s < v)

        def update(dims):
            return [
                replace(
                    d,
                    axis=shift(d.axis),
                    **(
                        {"block_axis": shift(d.block_axis)}
                        if d.is_sparse
                        else {}
                    ),
                )
                for d in dims
            ]

        out_dims, primal_dims = update(out_dims), update(primal_dims)
    else:
        values = grid_view
    # The operands' nominal order (dsnn-3qm.71): out_dims follow lhs.out_dims,
    # primal_dims follow rhs.primal_dims. A dim that belongs to neither source
    # (a partial contraction: an lhs primal dim or an rhs out dim the other
    # operand does not carry) sits where ``lhs.dense() @ rhs.dense()`` puts
    # it -- rhs-only out dims AFTER lhs's, lhs-only primal dims BEFORE rhs's
    # -- so ``dense()`` of the result is the tensordot of the operands.
    lhs_order = {d.id: i for i, d in enumerate(ctx.lhs.dims)}
    final_out = tuple(sorted(out_dims, key=lambda d: lhs_order.get(d.id, 999)))
    rhs_order = {d.id: i for i, d in enumerate(rhs_dims)}
    final_primal = tuple(sorted(primal_dims, key=lambda d: rhs_order.get(d.id, -1)))
    id_map = {d.id: i for i, d in enumerate(final_out + final_primal)}

    def finalize(d, new_id):
        kw = {"id": new_id}
        if d.is_sparse:
            kw["other_id"] = id_map.get(d.other_id, d.other_id)
        return replace(d, **kw)

    final_out = tuple(finalize(d, i) for i, d in enumerate(final_out))
    n_out = len(final_out)
    final_primal = tuple(finalize(d, n_out + i) for i, d in enumerate(final_primal))
    if values is not None and values.ndim > 1:
        seen = set()
        order = []
        for d in final_out + final_primal:
            for a in (d.axis, getattr(d, "block_axis", None)):
                if a is not None and a not in seen:
                    seen.add(a)
                    order.append(a)
        if sorted(order) == list(range(values.ndim)) and order != list(range(values.ndim)):
            values = values.transpose(order)
            new_of_old = {a: i for i, a in enumerate(order)}

            def _renum(d):
                kw = {}
                if d.axis is not None:
                    kw["axis"] = new_of_old[d.axis]
                if d.is_sparse and d.block_axis is not None:
                    kw["block_axis"] = new_of_old[d.block_axis]
                return replace(d, **kw) if kw else d

            final_out = tuple(_renum(d) for d in final_out)
            final_primal = tuple(_renum(d) for d in final_primal)
    has_val = any(d.axis is not None for d in final_out + final_primal) or any(
        d.is_sparse and d.block_axis is not None
        for d in final_out + final_primal
    )
    # Combine the three scalar_mults through the narrow-dtype-safe promotion
    # (_scaled_mul maps float8/int8/etc. to the common compute dtype) so a Quant'd
    # (float8) operand scalar_mult never trips the JAX implicit-promotion guard
    # here — the seed-vertex adjoint contraction reaches this tiled path with mixed
    # float8/float32 scalar_mults (the pre-op _unify only touches the operand val,
    # not this post-contraction 3-way scalar_mult product).
    from graphax.sparse.dtype_compute import _scaled_mul as _sm_promote
    final_mult = _sm_promote(
        _sm_promote(ctx.lhs.scalar_mult, ctx.rhs.scalar_mult), res.scalar_mult
    )
    if not has_val and values is not None and values.size == 1:
        final_mult = _sm_promote(final_mult, jnp.squeeze(values))
        values = None
    values = _store_narrow(values, ctx.lhs.dtype, ctx.rhs.dtype)
    out_dtype = values.dtype if values is not None else jnp.asarray(final_mult).dtype
    # transforms intentionally not propagated through matmul; callers in
    # core.py unload pre/post transforms before the matmul and reattach
    # fresh ones to the result.
    return SparseTensor(
        final_out,
        final_primal,
        values,
        scalar_mult=jnp.asarray(final_mult).astype(out_dtype),
        fill_value=None,  # tiled path assumes zero fill → statically zero
    )


# --- Metadata-stated single-block contraction -----------------------------
# A contracting dim of size 1 whose ``val`` does NOT physically carry it
# (``axis`` and ``block_axis`` both None) is a single structural block embedded
# in a larger logical axis: the metadata says it occupies one block and the
# rest of the partner's positions are off-structure. Contracting it against a
# size-N partner therefore *zero-pads* (the lone block at index 0, ``fill_value``
# at the remaining N-1 positions) — NOT a replicate-broadcast. (Concretely:
# a val=None 1×1 operand is ones·scalar_mult on its single block, fill off it;
# eye(N) @ [v, fill, …] selects column 0 → [v, fill, …], matching jax.jacfwd.)
# A size-1 dim WITH a physical axis is a genuine size-1 mismatch — never padded.
def _is_implicit_block_dim(d) -> bool:
    return (
        int(d.logical_size) == 1
        and getattr(d, "axis", None) is None
        and getattr(d, "block_axis", None) is None
    )


def _contract_pair_compatible(l, r) -> bool:
    """A contracting pair ``dot_general`` can take after densify: equal sizes,
    or a metadata-stated single-block embed (the size-1 side carries no physical
    axis), which ``_matmul_via_densify`` zero-pads up to the partner size."""
    if int(l.logical_size) == int(r.logical_size):
        return True
    small = l if int(l.logical_size) < int(r.logical_size) else r
    return int(small.logical_size) == 1 and _is_implicit_block_dim(small)


def _has_implicit_block_contraction(lhs, rhs) -> bool:
    """True iff some contracting pair is a metadata-stated size-1↔size-N embed
    — the only size mismatch we route to the densify path (which zero-pads it);
    genuine mismatches keep falling through to the tiled path's strict error."""
    if not (hasattr(lhs, "primal_dims") and hasattr(rhs, "out_dims")):
        return False
    return any(
        int(l.logical_size) != int(r.logical_size) and _contract_pair_compatible(l, r)
        for l, r in _align_contract_dims(lhs.primal_dims, rhs.out_dims, embed=True)
    )


def _pad_axis_to(arr, axis: int, size: int, fill):
    """Zero-pad (with ``fill``) ``arr`` along ``axis`` from its current size up
    to ``size``. The existing block stays at index 0; off-block positions take
    ``fill`` (the operand's off-structure value)."""
    pad_shape = list(arr.shape)
    pad_shape[axis] = size - arr.shape[axis]
    pad = jnp.full(tuple(pad_shape), fill, dtype=arr.dtype)
    return jnp.concatenate([arr, pad], axis=axis)


# --- Both-implicit contracting-pair analytic fold --------------------------
import os as _os


def _dims_to_sublists(lhs_ndim, rhs_ndim, dims):
    """``dot_general`` dimension numbers -> three INTEGER sublists for einsum.

    ``dims`` is ``((lhs_contract, rhs_contract), (lhs_batch, rhs_batch))``, so a
    caller that already holds dot dimension numbers cannot disagree with the
    einsum about which axis meets which. Labels are integers: the interleaved
    sublist form has no 52-symbol alphabet cap that the letter form imposes.

    The output order is ``dot_general``'s own -- batch axes, then the lhs's kept
    axes in order, then the rhs's kept axes in order -- so every index
    calculation downstream of a converted call site stays correct.
    """
    (lc, rc), (lb, rb) = dims
    lhs_sub = [None] * lhs_ndim
    rhs_sub = [None] * rhs_ndim
    nxt = 0
    for a, b in zip(lb, rb):          # batch axes share a label
        lhs_sub[a] = rhs_sub[b] = nxt
        nxt += 1
    n_batch = nxt
    for a, b in zip(lc, rc):          # contracted axes share a label, absent
        lhs_sub[a] = rhs_sub[b] = nxt  # from the output, so einsum sums them
        nxt += 1
    lhs_kept, rhs_kept = [], []
    for a in range(lhs_ndim):
        if lhs_sub[a] is None:
            lhs_sub[a] = nxt
            lhs_kept.append(nxt)
            nxt += 1
    for b in range(rhs_ndim):
        if rhs_sub[b] is None:
            rhs_sub[b] = nxt
            rhs_kept.append(nxt)
            nxt += 1
    out_sub = list(range(n_batch)) + lhs_kept + rhs_kept
    return lhs_sub, rhs_sub, out_sub


def _emit_einsum(a, lhs_sub, b, rhs_sub, out_sub):
    """The contraction as ONE ``jnp.einsum`` over integer sublists.

    The sole emission of the contraction engine (owner ruling D1, revised
    2026-09-08). It states which axes meet and leaves the lowering to XLA,
    which may pick a library GEMM, a fused multiply and reduce, or something
    else, per shape and per device. A ``dot_general`` pins one of those for
    every shape, and a hand-written multiply-then-reduce pins another.

    DTYPE. A bf16 x bf16 contraction accumulates in float32 through
    ``preferred_element_type``. Only ``Quant`` produces a bf16 operand, and a
    mixed ``{bf16, f32}`` pair is upcast by ``dtype_compute`` before it reaches
    here, so this fires exactly when both edges were quantized.

    Measured caveat, carried over from the deleted planner
    (``lower.matmul._einsum_accum_dtype``, 2026-08): ``jnp.einsum`` honours
    ``preferred_element_type`` as a genuine bf16-in / f32-out dot only for the
    plain ``ij,jk->ik`` form. For a form carrying batch labels or size-1 axes
    it instead converts BOTH operands to f32 up front, which deletes the bf16
    dot and adds converts (measured on mlp2 / mlp4 / attn: every bf16 dot gone,
    about 50 percent more converts). The error against the exact f32 Jacobian
    was unchanged either way (relerr 4.207e-3 on mlp2, 4.103e-3 on mlp4),
    because XLA already accumulates a bf16 dot in f32 internally and only
    rounds the output. So the kwarg costs Quant some speed and buys no
    accuracy on the batched forms. It stays because dropping it changes the
    STORED width of the result, and every downstream edge dtype with it.
    Changing that is a deliberate precision decision, not a tidy-up.
    """
    if (jnp.dtype(a.dtype) == jnp.dtype(jnp.bfloat16)
            and jnp.dtype(b.dtype) == jnp.dtype(jnp.bfloat16)):
        return jnp.einsum(a, lhs_sub, b, rhs_sub, out_sub,
                          preferred_element_type=jnp.float32)
    return jnp.einsum(a, lhs_sub, b, rhs_sub, out_sub)


def _gx_einsum(a, b, dims):
    """The contraction stated by ``dot_general`` dimension numbers, as an
    einsum. For call sites that hold dimension numbers rather than the tiled
    frame's slot layout."""
    lhs_sub, rhs_sub, out_sub = _dims_to_sublists(a.ndim, b.ndim, dims)
    return _emit_einsum(a, lhs_sub, b, rhs_sub, out_sub)


def _both_implicit_contract_pairs(lhs, rhs):
    """Contracting pairs where BOTH dims are IMPLICIT (``axis is None``, no stored
    physical axis) and share the same logical size N. Contracting a broadcast axis
    of length N is a reduce of N identical products => a scale-by-N: we drop the
    pair from the dot_general and fold N into the result ``scalar_mult`` (no op).

    Returns ``(pairs, factor)`` where ``pairs`` is the list of matched
    ``(lhs_primal_dim, rhs_out_dim)`` and ``factor`` is the product of their Ns."""
    if not (hasattr(lhs, "primal_dims") and hasattr(rhs, "out_dims")):
        return [], 1
    pairs = []
    factor = 1
    for l, r in _align_contract_dims(lhs.primal_dims, rhs.out_dims, embed=True):
        l_impl = getattr(l, "axis", "x") is None and getattr(l, "block_axis", None) is None
        r_impl = getattr(r, "axis", "x") is None and getattr(r, "block_axis", None) is None
        n_l, n_r = int(l.logical_size), int(r.logical_size)
        if l_impl and r_impl and n_l == n_r and n_l > 1 and not getattr(l, "is_sparse", False) and not getattr(r, "is_sparse", False):
            pairs.append((l, r))
            factor *= n_l
    return pairs, factor


def _fold_both_implicit(lhs, rhs, count):
    """Analytic scale-by-N for every BOTH-IMPLICIT contracting pair (no
    materialization, no dot_general over the broadcast axis). Drop the paired
    implicit dims from ``lhs.primal`` / ``rhs.out`` and multiply the reduced
    contraction's result ``scalar_mult`` by the product of their sizes. Composes
    with any existing (non-unit / quantized) ``scalar_mult`` — it MULTIPLIES.

    Returns the folded result (recurses through ``matmul`` on the reduced
    operands) or ``None`` when there is no both-implicit pair to fold."""
    from graphax.sparse.tensor import SparseTensor

    pairs, factor = _both_implicit_contract_pairs(lhs, rhs)
    if not pairs:
        return None
    drop_l = {id(l) for l, _ in pairs}
    drop_r = {id(r) for _, r in pairs}
    # Rebuild each operand without the dropped implicit dims. Physical axes are
    # unaffected (implicit dims carry none), so ``val`` and every remaining dim's
    # ``axis`` stay valid — only the dim lists shrink.
    new_lhs = SparseTensor(
        lhs.out_dims,
        tuple(d for d in lhs.primal_dims if id(d) not in drop_l),
        lhs.val,
        scalar_mult=lhs.scalar_mult,
        fill_value=lhs.fill_value,
        check_consistency=False,
    )
    new_rhs = SparseTensor(
        tuple(d for d in rhs.out_dims if id(d) not in drop_r),
        rhs.primal_dims,
        rhs.val,
        scalar_mult=rhs.scalar_mult,
        fill_value=rhs.fill_value,
        check_consistency=False,
    )
    # After dropping the both-implicit contracted dims BOTH operands may be
    # 0-rank scalars (a Compress-reduced seed-vertex edge contracting another
    # scalar seed): matmul rejects 0-rank@0-rank, so the mathematically-correct
    # scalar product is an ELEMENTWISE multiply. Route it there instead of
    # recursing into matmul (which would raise). Non-scalar reduced operands
    # keep the normal recursive matmul.
    _both_scalar = (
        new_lhs.out_dims == () and new_lhs.primal_dims == ()
        and new_rhs.out_dims == () and new_rhs.primal_dims == ()
    )
    _lhs_scalar = new_lhs.out_dims == () and new_lhs.primal_dims == ()
    _rhs_scalar = new_rhs.out_dims == () and new_rhs.primal_dims == ()
    if _both_scalar:
        out = new_lhs * new_rhs
        cnt = (0, 1, 0)  # one scalar multiply
    elif _lhs_scalar or _rhs_scalar:
        # Exactly ONE operand folded to rank-0. The caller passed two ranked
        # tensors, so core's scalar routing could not see this; the scale is
        # routed here, where the scalar first exists (ticket dsnn-3qm.68).
        _tn, _sc = (new_rhs, new_lhs) if _lhs_scalar else (new_lhs, new_rhs)
        # The remainder still carries the operand's ids (built above with
        # check_consistency=False). matmul renumbers on its own output path;
        # this path must do the same before the tensor is checked again.
        _dims = tuple(_tn.out_dims) + tuple(_tn.primal_dims)
        _id_map = {d.id: i for i, d in enumerate(_dims)}
        for d in _dims:
            if d.is_sparse and d.other_id not in _id_map:
                raise ValueError(
                    "both-implicit fold: sparse dim "
                    f"{d.id} lost its partner {d.other_id}; the remainder "
                    "cannot be expressed as a SparseTensor (dsnn-3qm.68)."
                )

        def _renum(d):
            kw = {"id": _id_map[d.id]}
            if d.is_sparse:
                kw["other_id"] = _id_map[d.other_id]
            return replace(d, **kw)

        _tn = SparseTensor(
            tuple(_renum(d) for d in _tn.out_dims),
            tuple(_renum(d) for d in _tn.primal_dims),
            _tn.val, scalar_mult=_tn.scalar_mult, fill_value=_tn.fill_value,
        )
        res = scale_by_scalar(_tn, _sc, count=count)
        if count:
            out, cnt = res
        else:
            out = res
    else:
        res = matmul(new_lhs, new_rhs, count=count)
        if count:
            out, cnt = res
        else:
            out = res
    fac = jnp.asarray(factor, dtype=out.scalar_mult.dtype)
    out = out.copy(scalar_mult=out.scalar_mult * fac)
    return (out, cnt) if count else out


# --- Late-densification escape hatch for non-zero fill_value --------------
def _matmul_via_densify(lhs, rhs):
    """Late-densification matmul for SparseTensors with non-zero ``fill_value``.

    The tiled algorithm assumes implicit positions are zero. For non-zero fill,
    materialize each operand via the fusion-friendly ``dense_for_matmul`` and run
    a plain ``dot_general``. Densification stays as a JAX expression so XLA can
    fold it into the matmul kernel — keeps the expansion in SMEM, not HBM.

    Contraction matching: ``dot_general`` requires contracting dim shapes to
    line up pair-by-pair. The densify path runs *before* ``_align_tensor_ids``
    so the lhs.primal/rhs.out dim orders aren't guaranteed to match — we
    pair them up by ``Index.id`` (graphax keeps ids consistent between paired
    dims through the AD pipeline) and fall back to positional pairing among
    any leftovers. Same for batch axes.
    """
    from graphax.sparse.tensor import SparseTensor

    n_lhs_dims = len(lhs.dims)
    n_lhs_out = len(lhs.out_dims)
    n_rhs_out = len(rhs.out_dims)

    # Deferred scale (#49): densify UNSCALED so the dense builders fuse
    # straight into the GEMM (an eager ``* scalar_mult`` producer cannot fuse
    # into the cuBLAS custom call and materializes scaled HBM copies of both
    # operands); ``s_l * s_r`` is folded into the result's ``scalar_mult``
    # below, mirroring the tiled path. Bool operands keep the legacy eager
    # fold (``&``/``|`` semantics do not scale linearly).
    _defer = not (jnp.issubdtype(jnp.dtype(lhs.dtype), jnp.bool_)
                  or jnp.issubdtype(jnp.dtype(rhs.dtype), jnp.bool_))
    lhs_dense = dense_for_matmul(lhs, defer_scale=_defer)
    rhs_dense = dense_for_matmul(rhs, defer_scale=_defer)

    # Contraction axes: the size-1-aware alignment (``_align_contract_indices``)
    # decides which ``lhs.primal`` axis pairs with which ``rhs.out`` axis — the
    # SAME source of truth as the tiled path's ``_build_matmul_topology`` and the
    # ``_densify_is_safe`` gate, so the three never disagree. A blind positional
    # ``[-n:]`` zip would mispair when a stray ``(C, 1)``-style size-1 axis
    # offsets the alignment (contracting batch against class). A skipped stray
    # size-1 simply becomes a kept (free) output axis below, like on the tiled
    # path. We deliberately don't id-match across operands: lhs and rhs use
    # independent id spaces here (``_align_tensor_ids`` only runs on the tiled
    # path), so raw-id matching is meaningless.
    contract_idx = _align_contract_indices(lhs.primal_dims, rhs.out_dims, embed=True)
    lhs_contract: list[int] = [n_lhs_out + li for li, _ in contract_idx]
    rhs_contract: list[int] = [rj for _, rj in contract_idx]

    # Batch axes: dims with the same id on both sides (excluding contract
    # axes). Batching collapses two same-id dims into one output dim, so we
    # only do it when the matched lhs/rhs dims live on the same side of the
    # out/primal split — otherwise an accidental id collision (lhs and rhs
    # use independent id spaces here, ``_align_tensor_ids`` only runs on the
    # tiled path) would silently turn an outer product into an element-wise
    # product. Restricting to (out↔out) or (primal↔primal) matches the
    # standard batched-matmul convention and is what the tiled path's
    # topology resolver yields after id alignment.
    rhs_id_to_axis = {d.id: i for i, d in enumerate(rhs.dims) if i not in rhs_contract}
    lhs_batch, rhs_batch = [], []
    for i, d in enumerate(lhs.dims):
        if i in lhs_contract:
            continue
        j = rhs_id_to_axis.get(d.id)
        if j is None or lhs.dims[i].logical_size != rhs.dims[j].logical_size:
            continue
        lhs_is_out = i < n_lhs_out
        rhs_is_out = j < n_rhs_out
        if lhs_is_out != rhs_is_out:
            continue  # cross-side id collision: not a real batch dim
        lhs_batch.append(i)
        rhs_batch.append(j)
        del rhs_id_to_axis[d.id]

    # Metadata-stated single-block embed: a contracting dim whose val doesn't
    # carry it (``axis`` None, size 1) is one structural block in a larger
    # logical axis. ``dot_general`` needs the physical contracting shapes to
    # line up, so zero-pad the size-1 axis up to its size-N partner here (block
    # at index 0, ``fill_value`` elsewhere) — gated strictly on the metadata
    # (``_is_implicit_block_dim``) so a genuine size-1 is never silently padded
    # (it reaches dot_general mismatched and raises, as before).
    for la, ra, (li, rj) in zip(lhs_contract, rhs_contract, contract_idx):
        ld, rd = lhs.primal_dims[li], rhs.out_dims[rj]
        ls, rs = lhs_dense.shape[la], rhs_dense.shape[ra]
        if ls == rs:
            continue
        if ls == 1 and _is_implicit_block_dim(ld):
            lhs_dense = _pad_axis_to(
                lhs_dense, la, rs,
                lhs._eff_fill if _defer else _scaled_fill(lhs))
        elif rs == 1 and _is_implicit_block_dim(rd):
            rhs_dense = _pad_axis_to(
                rhs_dense, ra, ls,
                rhs._eff_fill if _defer else _scaled_fill(rhs))

    # EMISSION. One einsum, stated by the dimension numbers (owner ruling D1,
    # revised 2026-09-08).
    _dn = (
        (tuple(lhs_contract), tuple(rhs_contract)),
        (tuple(lhs_batch), tuple(rhs_batch)),
    )
    result = _store_narrow(_gx_einsum(lhs_dense, rhs_dense, _dn),
                           lhs_dense.dtype, rhs_dense.dtype)

    # The result axes are laid out as: batch, then lhs's kept (in order), then rhs's
    # kept (in order). Build the output sizes/slot tags in that same order.
    lhs_kept = [
        i for i in range(n_lhs_dims) if i not in lhs_contract and i not in lhs_batch
    ]
    rhs_kept = [
        i for i in range(len(rhs.dims)) if i not in rhs_contract and i not in rhs_batch
    ]

    sizes_slots = (
        [
            (lhs.dims[i].logical_size, "out") for i in lhs_batch
        ]  # batched dims become out_dims
        + [
            (lhs.dims[i].logical_size, "out" if i < n_lhs_out else "primal")
            for i in lhs_kept
        ]
        + [
            (rhs.dims[j].logical_size, "out" if j < n_rhs_out else "primal")
            for j in rhs_kept
        ]
    )
    out_axes = [i for i, (_, s) in enumerate(sizes_slots) if s == "out"]
    primal_axes = [i for i, (_, s) in enumerate(sizes_slots) if s == "primal"]
    perm = out_axes + primal_axes
    if perm != list(range(len(perm))):
        result = jnp.transpose(result, perm)

    # Output dim ids are ``range(0, n_out + n_primal)`` — the same convention
    # ``_build_output_tensor`` finalizes the tiled path to (it allocates
    # ``next_id = max(input_ids) + 1`` only intermediately, then
    # ``finalize`` renumbers everything to ``range(...)``). Keeping both
    # paths on the same convention means downstream code joining by id sees
    # one topology regardless of which fast path fired.
    out_dims = tuple(
        DenseIndex(i, sizes_slots[a][0], axis=i) for i, a in enumerate(out_axes)
    )
    primal_dims = tuple(
        DenseIndex(len(out_axes) + i, sizes_slots[a][0], axis=len(out_axes) + i)
        for i, a in enumerate(primal_axes)
    )
    # transforms intentionally not propagated through matmul; callers in
    # core.py unload pre/post transforms before the matmul and reattach
    # fresh ones to the result.
    if _defer:
        # Fold the deferred operand scales into the output, exactly like the
        # tiled path's _build_output_tensor. Promote mixed scalar dtypes to
        # their common compute dtype first (a narrow-Quant sm has no implicit
        # promotion path against f32).
        _sl, _sr = lhs.scalar_mult, rhs.scalar_mult
        if isinstance(_sl, (int, float)) and isinstance(_sr, (int, float)):
            _sm_out = _sl * _sr
        else:
            from graphax.sparse.dtype_compute import _compute_dtype
            _a, _b = jnp.asarray(_sl), jnp.asarray(_sr)
            _cdt = _compute_dtype(_a.dtype, _b.dtype)
            _sm_out = _a.astype(_cdt) * _b.astype(_cdt)
        return SparseTensor(
            out_dims,
            primal_dims,
            result,
            fill_value=None,  # densified output is fully dense → no fill cells
            scalar_mult=_sm_out,
            check_consistency=False,
        )
    return SparseTensor(
        out_dims,
        primal_dims,
        result,
        fill_value=None,  # densified output is fully dense → no fill cells
        check_consistency=False,
    )


# --- Path tracing (test-only) ---------------------------------------------
# Re-exports from ``_path_tracking``. ``record_path(name)`` is a no-op when
# no tracker is active (the default in production); tests opt in via either
# the ``track_paths()`` context manager or the ``TRACK_PATHS=1`` env var.
# See ``ops._path_tracking`` for the full design.
from ._path_tracking import record_path as _record_path  # noqa: E402


# ``last_path`` is exposed as a module attribute for backward compatibility
# (and convenience under TRACK_PATHS=1). Reads forward to the shared mirror.
def __getattr__(name: str):
    if name == "last_path":
        from . import _path_tracking

        return _path_tracking.last_path
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Main dispatcher ------------------------------------------------------
def _reconcile_permuted_val(tensor):
    """Restore the layout invariant ``val.shape[dim.axis] == dim.size`` (and
    ``val.shape[dim.block_axis] == dim.block_size``) when an upstream
    Diag / Compress / transpose left a tensor whose dim METADATA disagrees with
    its physical ``val`` axis order — the ViT seq(17)/embed(32) swap (bug-doc
    Class 3b: an ``Incompatible types for broadcasting`` raised in the tiled
    contraction kernel / elemental densify because the kernel lays ``val`` out
    by ``dim.axis`` yet builds its target shape from ``dim.size``).

    ``dim.id`` / ``dim.size`` are the LOGICAL truth the id-based contraction and
    the result dims rely on; ``dim.axis`` is only a pointer into ``val``. When a
    pointer lands on a val axis whose extent != the dim's size, the data the dim
    OWNS lives at the (unique) val axis whose extent DOES match — a permutation
    of the materialized axes. We transpose ``val`` so each dim's ``axis`` again
    holds its own data (matching by extent), keeping every ``dim.axis`` pointer
    valid so the downstream ``_prepare_physical_array`` gather lines up.

    No-op (returns ``tensor`` unchanged, so byte-identical) whenever the
    invariant already holds — which is always true on the EXACT-AD path
    (``transforms=()`` never permutes dims). Bails (returns unchanged) when the
    mismatched axes' extents are not an UNAMBIGUOUS permutation of the dims'
    expected extents (a repeated size among the unmatched axes can't be
    disambiguated by extent alone — better to let the existing strict shape
    check fire than to risk a mis-laid-out Jacobian)."""
    val = tensor.val
    if val is None:
        return tensor
    ndim = val.ndim
    # Desired extent at each materialized val axis, from the dim metadata.
    want = {}
    ok = True
    for d in tensor.dims:
        if d.axis is not None and d.axis < ndim:
            want[d.axis] = int(d.size)
            if int(val.shape[d.axis]) != int(d.size):
                ok = False
        if d.is_sparse and d.block_axis is not None and d.block_axis < ndim:
            bs = int(d.block_size or 1)
            want[d.block_axis] = bs
            if int(val.shape[d.block_axis]) != bs:
                ok = False
    if ok:
        return tensor  # invariant already holds — no-op (EXACT path included)
    # Constrained axes whose extents already match keep their position; the
    # mismatched ones are re-matched to the source axis carrying their extent.
    perm = list(range(ndim))
    used, pending = set(), []
    for a in want:
        if int(val.shape[a]) == want[a]:
            perm[a] = a
            used.add(a)
        else:
            pending.append(a)
    # A size that repeats among the still-unmatched axes can't be disambiguated
    # by extent alone — bail rather than risk a wrong (mis-laid-out) transpose.
    pend_sizes = [want[t] for t in pending]
    if len(set(pend_sizes)) != len(pend_sizes):
        return tensor
    free = [a for a in want if a not in used]
    for t in pending:
        match = next(
            (s for s in free if s not in used and int(val.shape[s]) == want[t]), None
        )
        if match is None:
            return tensor  # not a clean permutation — leave for the strict check
        perm[t] = match
        used.add(match)
    if perm == list(range(ndim)):
        return tensor
    return tensor.copy(val=val.transpose(perm))


def _normalize_inputs(lhs, rhs):
    """Convert array operands to ``SparseTensor``. Pure shape / structure prep —
    no actual matmul work happens here."""
    if not _is_sparse(lhs):
        lhs = _arr2st(lhs, out_ndim=lhs.ndim - len(rhs.out_dims))
    if not _is_sparse(rhs):
        rhs = _arr2st(rhs, out_ndim=len(lhs.primal_dims))
    # Class 3b: an approx transform can leave a dim whose physical ``val`` axis
    # order disagrees with its ``size`` metadata (the ViT seq/embed swap). Both
    # the tiled kernel and the elemental densify lay ``val`` out by ``dim.axis``
    # but size it by ``dim.size``, so a swap broadcasts mismatched operands.
    # Reconcile here, before either path — a strict no-op on the EXACT-AD path.
    lhs = _reconcile_permuted_val(lhs)
    rhs = _reconcile_permuted_val(rhs)
    # Mixed-precision upcast: a narrow (Quant) val and a float val have no
    # implicit promotion path, so dot_general would raise; combine both at
    # their highest common compute dtype. No-op when dtypes already match.
    # DELIBERATE (2026-08-04): a {bf16, f32} MIXED pair UPCASTS -- quantizing
    # one edge must never silently approximate its exact partner. The bf16
    # narrow GEMM engages only when BOTH operands were made bf16 (the policy
    # quantizes both incident edges); see _emit_einsum.
    from graphax.sparse.dtype_compute import _unify_operand_dtypes
    lhs, rhs = _unify_operand_dtypes(lhs, rhs)
    return lhs, rhs


def _compact_block_lens(pairs, shared, total, split):
    """Pure replica of _finalize_output's final_lhs_lens / final_rhs_lens."""
    N = len(pairs)
    non_contract = [i for i, p in enumerate(pairs) if p.pairing_type != "contract"]
    d_ls = [p.lhs.block_len for p in pairs]
    f_ls = [p.rhs.shared_block_len for p in pairs]
    ss_out = [split[i] if i in non_contract else 1 for i in range(N)]
    fll = [d_ls[i] * (ss_out[i] if pairs[i].pairing_type == "spatial_sparse_rhs" else 1)
           for i in range(N)]
    frl = [f_ls[i] * (ss_out[i] if pairs[i].pairing_type != "spatial_sparse_rhs" else 1)
           for i in range(N)]
    return fll, frl


def _output_dims(ctx, rhs_dims, res):
    """Canonical output dims (ids/sizes/axis) — the pure metadata half of
    _build_output_tensor, derived from res.grid.shape (no val touched). Both
    halves read the same ``_pair_output_dims``, so they cannot disagree."""
    # ``expands`` is a VAL action only: the dims it makes representable are
    # already the ones ``_build_pair_dims`` emitted, so the metadata half needs
    # nothing from it.
    out_dims, primal_dims, _shape, squeeze, summed, _expands = _pair_output_dims(
        ctx, rhs_dims, res
    )
    squeeze = squeeze + summed
    if squeeze:
        unique_sq = tuple(sorted(set(squeeze)))

        def shift(v):
            return None if v is None else v - sum(1 for s in unique_sq if s < v)

        def update(dims):
            return [
                replace(
                    d,
                    axis=shift(d.axis),
                    **({"block_axis": shift(d.block_axis)} if d.is_sparse else {}),
                )
                for d in dims
            ]

        out_dims, primal_dims = update(out_dims), update(primal_dims)
    # The operands' nominal order (dsnn-3qm.71): out_dims follow lhs.out_dims,
    # primal_dims follow rhs.primal_dims. A dim that belongs to neither source
    # (a partial contraction: an lhs primal dim or an rhs out dim the other
    # operand does not carry) sits where ``lhs.dense() @ rhs.dense()`` puts
    # it -- rhs-only out dims AFTER lhs's, lhs-only primal dims BEFORE rhs's
    # -- so ``dense()`` of the result is the tensordot of the operands.
    lhs_order = {d.id: i for i, d in enumerate(ctx.lhs.dims)}
    final_out = tuple(sorted(out_dims, key=lambda d: lhs_order.get(d.id, 999)))
    rhs_order = {d.id: i for i, d in enumerate(rhs_dims)}
    final_primal = tuple(sorted(primal_dims, key=lambda d: rhs_order.get(d.id, -1)))
    id_map = {d.id: i for i, d in enumerate(final_out + final_primal)}

    def finalize(d, new_id):
        kw = {"id": new_id}
        if d.is_sparse:
            kw["other_id"] = id_map.get(d.other_id, d.other_id)
        return replace(d, **kw)

    final_out = tuple(finalize(d, i) for i, d in enumerate(final_out))
    n_out = len(final_out)
    final_primal = tuple(finalize(d, n_out + i) for i, d in enumerate(final_primal))
    return final_out, final_primal


def _execute_tiled(ctx, rhs_dims):
    """Fallback: full tiled algorithm. Handles every case the fast paths
    bail on, including LCM-mismatched outer sizes, spatial sparse pairs,
    and broadcast / unmaterialized val axes."""
    lhs_val, rhs_val = _val_or_one(ctx.lhs), _val_or_one(ctx.rhs)
    lhs_val, rhs_val, slots_l, slots_r = _prepare_physical_arrays(
        lhs_val, rhs_val, ctx.pairs
    )
    grid, shared, lhs_lens, rhs_lens, scalar, lazy_info = (
        _execute_block_sparse_contraction(
            lhs_val, rhs_val, ctx.pairs, ctx, slots_l, slots_r
        )
    )
    res = CRes(
        grid=grid,
        shared_factors=shared,
        lhs_block_lens=lhs_lens,
        rhs_block_lens=rhs_lens,
        scalar_mult=scalar,
    )
    if lazy_info is not None:
        eff_pairs, lazy, t_shared, t_fll, t_frl = lazy_info
        res = res._replace(
            eff_pairs=eff_pairs,
            lazy=lazy,
            true_shared_factors=t_shared,
            true_lhs_block_lens=t_fll,
            true_rhs_block_lens=t_frl,
        )
    out = _build_output_tensor(ctx, rhs_dims, res)
    if hasattr(out, "out_dims") and hasattr(ctx.lhs, "out_dims") and hasattr(ctx.rhs, "primal_dims"):
        if len(out.out_dims) == len(ctx.lhs.out_dims) and len(out.primal_dims) == len(ctx.rhs.primal_dims):
            check_nominal_order(out, ctx.lhs, ctx.rhs, 'matmul._execute_tiled')
    return out




class ScalarMatmul(ValueError):
    """``matmul`` was called with a rank-0 operand (ticket dsnn-3qm.68).

    A scalar has no axes, so there is nothing to contract. The operation the
    caller means is an elementwise scale. Use :func:`scale_by_scalar`.

    This RAISES rather than rerouting silently (owner ruling 2026-09-07):
    accepting ``X @ scalar`` hides the call site that is wrong, and every
    module with a real matmul rejects it for the same reason.
    """


def scale_by_scalar(tensor, scalar, count: bool = False):
    """``scalar * tensor``, folded into ``scalar_mult``. No per-element work.

    This is what a caller means when it reaches for ``X @ scalar``. The scalar
    operand's effective value -- its 0-d ``val`` (``1`` when ``val is None``)
    times its own ``scalar_mult`` -- is folded into the tensor operand's
    ``scalar_mult``, leaving the tensor's ``val``, ``fill_value`` and dims
    untouched. The scale is deferred, exactly like the both-implicit fold.

    NOT ``scalar._stored_val()``: that returns a FLAT array sized
    ``_structural_val_size`` (shape ``(N,)`` even for ``N == 1``), which
    silently added a size-1 axis to ``scalar_mult`` on every fold and
    compounded to ``(1, 1)`` on a second scale in the same chain. A rank-0
    tensor's own ``val``, when present, is genuinely 0-d.
    """
    from graphax.sparse.dtype_compute import _scaled_mul

    sval = (scalar.val if scalar.val is not None
            else jnp.ones((), dtype=scalar.scalar_mult.dtype))
    factor = _scaled_mul(sval, scalar.scalar_mult)
    factor = jnp.asarray(factor, dtype=tensor.scalar_mult.dtype)
    out = tensor.copy(scalar_mult=_scaled_mul(tensor.scalar_mult, factor))
    if count:
        return out, (0, out.size, 0)
    return out


def matmul(lhs, rhs, count: bool = False):
    """Sparse matmul dispatcher. Runs a cascade of paths, first-applicable
    wins, falling back to the general tiled algorithm; each path either
    handles the operands or declines (returns ``None`` / predicate false) and
    control passes to the next.

    Dispatch order (first applicable wins), in body order:
      1. ``dense_dense``        -- both operands are plain arrays (no SparseTensor).
      2. ``scalar_elementwise`` -- either operand is a 0-rank (scalar)
                                  SparseTensor (one or both), routed through
                                  ``*``: no shared dim to contract, so the
                                  contraction is a scale.
      3. ``both_implicit_fold`` -- both contracted dims implicit: analytic
                                  scale-by-N folded into ``scalar_mult``.
      4. ``elemental``          -- structured (diagonal/block) kernels,
                                  active only under ``GRAPHAX_ELEMENTAL=1``
                                  (default off, so this path never fires).
      5. ``densify``            -- non-zero ``fill_value`` OR an implicit-block
                                  contraction, when ``_densify_is_safe``:
                                  materialize via ``dense_for_matmul`` and
                                  contract (tiled assumes implicit positions
                                  are zero, wrong when fill != 0). Runs LATE,
                                  after the structured paths above.
      6. ``tiled``              -- general LCM/topology/finalize pipeline; the
                                  sparsity-preserving engine for everything else.

    A 0-rank (scalar) operand RAISES :class:`ScalarMatmul`, whether one side or
    both. A scalar has no axes, so there is nothing to contract: the caller
    means a scale (owner ruling, dsnn-3qm.68). Use :func:`scale_by_scalar`.

    With ``count=True`` returns ``(result, (adds, muls, fmas))``. Per output
    element the dot product decomposes into 1 plain multiply (no
    accumulator yet) plus ``K-1`` fused multiply-adds, so:

      * ``muls = output_size``           (one initial mul per output element)
      * ``fmas = output_size * (K - 1)`` (accumulating multiply-adds)
      * ``adds = 0``                     (folded into the FMAs)

    where ``K`` is the contraction depth. Computed from static
    shape/topology — pure Python ints, jit-friendly. When ``K <= 1``
    (scalar matmul / outer product), ``muls = output_size`` and ``fmas = 0``.
    """
    # Edge-level LowRank (L3): unwrap FIRST — the factored closure lives in
    # graphax.sparse.lowrank and every path below assumes SparseTensor.
    if getattr(lhs, "_is_lowrank", False) or getattr(rhs, "_is_lowrank", False):
        from graphax.sparse.lowrank import lowrank_matmul

        return lowrank_matmul(lhs, rhs, count=count)
    _record_path(None)
    if not _is_sparse(lhs) and not _is_sparse(rhs):
        _record_path("dense_dense")
        out = jnp.matmul(lhs, rhs)
        if count:
            return out, _compute_matmul_count(lhs, rhs, out)
        return out
    lhs, rhs = _normalize_inputs(lhs, rhs)
    _lhs_scalar = (
        getattr(lhs, "out_dims", ()) == () and getattr(lhs, "primal_dims", ()) == ()
    )
    _rhs_scalar = (
        getattr(rhs, "out_dims", ()) == () and getattr(rhs, "primal_dims", ()) == ()
    )
    # Two 0-rank (scalar) SparseTensors: the contraction is a scalar product =
    # an ELEMENTWISE multiply (scalar . X == scale). This is what the AGGREGATION
    # Two 0-rank (scalar) SparseTensors: the product of two scalars is a
    # scale, not a contraction. The chain-rule site in core.py routes these to
    # ``scale_by_scalar`` itself; anything that still arrives here is a caller
    # that has not been fixed, so say so instead of quietly multiplying.
    if _lhs_scalar and _rhs_scalar:
        raise ScalarMatmul(
            "matmul of two 0-rank SparseTensors: a scalar has no axes to "
            "contract. The operation meant here is a scale. Call "
            "graphax.sparse.ops.matmul.scale_by_scalar(tensor, scalar) or "
            "``lhs * rhs`` (ticket dsnn-3qm.68)."
        )
    # Exactly ONE 0-rank (scalar) operand: ``X @ scalar`` (or ``scalar @ X``)
    # has no shared dimension to contract. It is a SCALE, and asking matmul for
    # it is a caller error, so it RAISES (owner ruling 2026-09-07, ticket
    # dsnn-3qm.68). The earlier fix rerouted it into ``scalar_mult`` silently,
    # which produced the right number and hid the wrong call site.
    # :func:`scale_by_scalar` is that fold, now public, for callers to use.
    if _lhs_scalar or _rhs_scalar:
        which = "lhs" if _lhs_scalar else "rhs"
        other = rhs if _lhs_scalar else lhs
        raise ScalarMatmul(
            f"matmul got a rank-0 (scalar) {which} against an operand of shape "
            f"{getattr(other, 'shape', None)}. A scalar has no axes, so there "
            "is nothing to contract: this is a scale, not a matmul. Call "
            "graphax.sparse.ops.matmul.scale_by_scalar(tensor, scalar). Fix "
            "the caller (ticket dsnn-3qm.68); this used to be rerouted "
            "silently."
        )
    # Elemental fast path (Phase: bridge-cse): route a STRUCTURED contraction
    # (block-diagonal / implicit contracted dims) through the
    # composed elemental kernels. Returns None for a pure-dense contraction, so
    # the EXACT-AD (transforms=()) edge never enters here and stays byte-
    # identical. Built with the canonical output-id convention so a downstream
    # multi-edge / all-vertices contraction aligns by id.
    # Both-implicit contracting pair -> analytic scale-by-N folded into
    # scalar_mult (nothing is contracted over the broadcast axis, nothing is
    # materialized).
    _folded = _fold_both_implicit(lhs, rhs, count)
    if _folded is not None:
        _record_path("both_implicit_fold")
        _out = _folded[0] if count else _folded
        if hasattr(_out, "out_dims") and hasattr(lhs, "out_dims") and hasattr(rhs, "primal_dims"):
            check_nominal_order(_out, lhs, rhs, 'matmul.matmul')
        return _folded

    from graphax.sparse.elemental.dispatch import try_elemental_matmul

    _elem = try_elemental_matmul(lhs, rhs, count=count)
    if _elem is not None:
        _record_path("elemental")
        _out = _elem[0] if count else _elem
        if hasattr(_out, "out_dims") and hasattr(lhs, "out_dims") and hasattr(rhs, "primal_dims"):
            check_nominal_order(_out, lhs, rhs, 'matmul.matmul')
        return _elem
    # Densify path: handles non-zero ``fill_value`` correctly (the tiled
    # path's contraction assumes implicit positions are zero, which is wrong
    # for non-zero fills). Only safe when contracting dim sizes pair up
    # positionally — graphax's AD pipeline can produce permuted dim orders
    # that need the tiled path's id-aware topology resolver. ``_is_zero_fill``
    # is the static ``fill_value is None`` test (None lives in the treedef) so
    # this stays jit-friendly.
    has_nonzero_fill = not _is_zero_fill(lhs) or not _is_zero_fill(rhs)
    # The densify path also owns metadata-stated single-block contractions
    # (size-1↔size-N where the size-1 side carries no physical axis): the tiled
    # path's ``_resolve_contract_pair`` rejects the size mismatch, but the embed
    # is well-defined and ``_matmul_via_densify`` zero-pads it.
    need_block_embed = _has_implicit_block_contraction(lhs, rhs)
    if has_nonzero_fill or need_block_embed:
        if _densify_is_safe(lhs, rhs):
            _record_path("densify")
            out = _matmul_via_densify(lhs, rhs)
            if hasattr(out, "out_dims") and hasattr(lhs, "out_dims") and hasattr(rhs, "primal_dims"):
                check_nominal_order(out, lhs, rhs, 'matmul.matmul')
            if count:
                return out, _compute_matmul_count(lhs, rhs, out)
            return out
        if has_nonzero_fill:
            # Tiled / aligned-pair / dot_general fast paths assume zero fill;
            # falling through silently mislabels the result as zero-fill.
            raise NotImplementedError(
                "matmul of operands with non-zero fill_value and incompatible "
                "logical sizes is not supported; reorder dim ids first"
            )
        # Zero-fill broadcast that isn't densify-safe (a permuted dim order):
        # fall through to the tiled path, which raises the strict size error.
    # Tiled path: the sole sparse contraction engine. It preserves output
    # sparsity and resolves factored / block-diagonal contracted mids through
    # its id-aware topology resolver (upstream apply_block_diagonal /
    # _subdivide_coupled_blockdiag reconcile the two operands' factorings
    # BEFORE the contraction). A genuinely irreconcilable size mismatch raises
    # ValueError here, loudly. The former pure-diagonal / block-diagonal
    # densifying fast-path fallbacks were removed (Phase bridge-cse): they only
    # fired when tiled raised, and no live trajectory (ViT / ConvNet / MoE,
    # exact + approx) reaches that fallback any more.
    # Misaligned meta grids: meet at the gcd, which IS the result's own frame,
    # instead of at the lcm, which the tiled path has to fold back afterwards.
    _bx = _expand_blocked_against_meta(lhs, rhs)
    if _bx is not None:
        _record_path("expand_blocked_meta")
        lhs, rhs = _bx
    _rf = _reframe_misaligned_contraction(lhs, rhs)
    if _rf is not None:
        _record_path("reframe_gcd")
        lhs, rhs = _rf
    rhs_out_dims, rhs_primal_dims, rhs_id_offset = _align_tensor_ids(lhs, rhs)
    rhs_dims = rhs_out_dims + rhs_primal_dims
    pairs = _build_matmul_topology(lhs, rhs_out_dims, rhs_primal_dims, rhs_id_offset)
    ctx = Ctx(lhs=lhs, rhs=rhs, pairs=pairs, rhs_id_offset=rhs_id_offset)
    _record_path("tiled")
    out = _execute_tiled(ctx, rhs_dims)
    if hasattr(out, "out_dims") and hasattr(lhs, "out_dims") and hasattr(rhs, "primal_dims"):
        if len(out.out_dims) == len(lhs.out_dims) and len(out.primal_dims) == len(rhs.primal_dims):
            check_nominal_order(out, lhs, rhs, 'matmul.matmul')
    if count:
        return out, _compute_matmul_count(lhs, rhs, out)
    return out


def _mm_locate(t, dim_id):
    """``(is_out, rel_index, dim)`` for ``dim_id`` in ``t``, or ``None``."""
    for rel, d in enumerate(t.out_dims):
        if d.id == dim_id:
            return True, rel, d
    for rel, d in enumerate(t.primal_dims):
        if d.id == dim_id:
            return False, rel, d
    return None


def _coarsen_pair_to(t, dim_id, other_id, meta):
    from graphax.sparse.tensor import _coarsen_coupled_blockdiag
    l1, l2 = _mm_locate(t, dim_id), _mm_locate(t, other_id)
    if l1 is None or l2 is None:
        return None
    if l1[2].size == meta:
        return t
    return _coarsen_coupled_blockdiag(
        t, l1[0], l1[1], l1[2], l2[0], l2[1], l2[2], meta)


def _materialize_blocked_dim(t, dim):
    # Write out a blocked dense dim's implicit block as dense() would: size
    # blocks of block_size uniform positions, in that order, rank preserved.
    n = int(dim.size) * int(dim.block_size)
    new = DenseIndex(dim.id, n, dim.axis)

    def swap(dims):
        return tuple(new if d.id == dim.id else d for d in dims)

    if dim.axis is None:
        return _copy(t, out_dims=swap(t.out_dims), primal_dims=swap(t.primal_dims))
    for d in t.dims:
        if d.id == dim.id:
            continue
        if d.axis == dim.axis or getattr(d, "block_axis", None) == dim.axis:
            raise ValueError(
                f"matmul: blocked dense dim id={dim.id} shares val axis "
                f"{dim.axis} with dim id={d.id}; its implicit block cannot be "
                f"materialized without moving the other dim's storage."
            )
    shape = list(t.val.shape)
    val, _ = _apply_expands(
        t.val, shape,
        [_Expand(dim.axis, int(dim.size), 1, int(dim.size), int(dim.block_size))],
    )
    return _copy(t, val=val, out_dims=swap(t.out_dims),
                 primal_dims=swap(t.primal_dims))


def _expand_blocked_against_meta(lhs, rhs):
    # dsnn-tsl. A blocked dense contracted dim gives the pair its meta (the
    # block COUNT); a diagonal partner on the other side gives the pair the
    # SURVIVOR's whole extent as its meta; shared_factors takes the gcd, so the
    # block count is divided out of the survivor and _build_pair_dims can name
    # only the quotient. The diagonal varies inside the block, so the block has
    # to be written out here; every other blocked dense contraction keeps meta
    # 1 or no survivor and is left alone.
    lhs_out = {d.id: d for d in lhs.out_dims}
    rhs_primal = {d.id: d for d in rhs.primal_dims}
    l_hit, r_hit = None, None
    for lp, ro in _align_contract_dims(lhs.primal_dims, rhs.out_dims, embed=False):
        lo = lhs_out.get(getattr(lp, "other_id", -1)) if lp.is_sparse else None
        rp = rhs_primal.get(getattr(ro, "other_id", -1)) if ro.is_sparse else None
        l_outer = int(lp.size) if _is_blocked_dense(lp) else _dim_vals(lo, True)[0]
        r_outer = int(ro.size) if _is_blocked_dense(ro) else _dim_vals(rp, True)[0]
        if math.gcd(int(l_outer), int(r_outer)) == 1:
            continue
        if _is_blocked_dense(lp) and rp is not None:
            l_hit = lp
        if _is_blocked_dense(ro) and lo is not None:
            r_hit = ro
    if l_hit is None and r_hit is None:
        return None
    if l_hit is not None:
        lhs = _materialize_blocked_dim(lhs, l_hit)
    if r_hit is not None:
        rhs = _materialize_blocked_dim(rhs, r_hit)
    return lhs, rhs


def _reframe_misaligned_contraction(lhs, rhs):
    """``(lhs, rhs)`` re-cut onto ONE meta grid before the tiled path, or
    ``None`` to leave the operands alone.

    The contracted axis has one logical extent ``L`` and the two operands can
    disagree about how it is cut: the lhs block-diagonal at meta ``a`` with
    blocks ``K1``, the rhs at meta ``b`` with blocks ``K2``, ``a*K1 == b*K2 ==
    L``. The tiled path meets them on the least-common-multiple grid, which
    costs nothing going in -- the refinement is carved out of each operand's
    own block axis by reshape -- but costs afterwards, in ``_reduce_grid``,
    which folds the refined grid back with a one-hot whose size is quadratic in
    the lcm.

    Where the answer lives says the grid should be the gcd instead. Result
    position ``(i, k)`` is live when some ``j`` has both operands live, that is
    when the ``K1``-block of ``i`` and the ``K2``-block of ``k`` overlap. The
    finest block-diagonal holding all of those is meta ``gcd(a, b)``, with
    blocks ``L / gcd(a, b) == lcm(K1, K2)``. So coarsening BOTH operands to meta
    ``gcd(a, b)`` lands directly in the result's own frame: the contraction is
    then one batched einsum over the meta axis and ``_reduce_grid`` has nothing
    to fold.

    Coarsening is NOT always cheaper, so it is not unconditional. It trades the
    fold away for a longer contraction, and the size rule at the loop below
    decides which is smaller. Declined when ``gcd(a, b) == 1``, where the
    "block-diagonal" container is the dense form and coarsening is just an
    early densify of both operands. Declined on non-zero fill, because
    coarsening is defined for structural zeros only.
    """
    if not _is_zero_fill(lhs) or not _is_zero_fill(rhs):
        return None
    if len(lhs.primal_dims) != len(rhs.out_dims):
        return None
    plan = []
    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if not (ld.is_sparse and rd.is_sparse):
            continue
        if ld.size == rd.size:
            continue
        if ld.logical_size != rd.logical_size:
            return None
        a, b = int(ld.size), int(rd.size)
        g = math.gcd(a, b)
        if g <= 1:
            return None
        # Which frame is cheaper. The lcm route pays a fold, and that fold
        # emits ``a*b/g`` segments per unit of the surrounding block extents.
        # The gcd route pays a longer contraction instead: coarsened blocks are
        # ``L/g`` wide, so its reduction runs ``L/g`` deep. Coarsening is worth
        # it exactly when the fold it removes is the bigger of the two, that is
        # when ``a*b/g > L/g``, that is when ``a*b > L``.
        #
        # MEASURED on 9 misaligned shapes against the incumbent lcm path, on an
        # RTX 3090. The rule agrees with the measurement on all 9. Where it says
        # coarsen, the gcd route runs 2.0x, 6.9x and 13.8x fewer flops. Where it
        # says do not, the gcd route would have cost 1.5x to 4.1x MORE -- an
        # unconditional coarsening loses on more shapes than it wins, so the
        # rule is not optional. Note this contradicts an earlier hand-written
        # A/B of the two plans, which reported the gcd route ahead everywhere;
        # that comparison modelled the routes rather than running them.
        if a * b <= int(ld.logical_size):
            return None
        plan.append((ld, rd, g))
    if not plan:
        return None
    new_l, new_r = lhs, rhs
    for ld, rd, g in plan:
        new_l = _coarsen_pair_to(new_l, ld.id, ld.other_id, g)
        new_r = _coarsen_pair_to(new_r, rd.id, rd.other_id, g)
        if new_l is None or new_r is None:
            return None
    return new_l, new_r


def _densify_is_safe(lhs, rhs) -> bool:
    """``_matmul_via_densify`` contracts the size-1-aware aligned pairs
    (``_align_contract_dims``): equal sizes pair directly and a metadata embed
    (size-1 side, no physical axis) is zero-padded up to its partner before
    ``dot_general``. Safe iff every aligned pair is so contractible. A genuine
    size mismatch — e.g. a transposed Jacobian whose dim *order* doesn't line up
    — yields a non-compatible pair, so we return False and let the caller fall
    through to the tiled path, which permutes dims by id via ``_align_tensor_ids``.
    """
    return all(
        _contract_pair_compatible(l, r)
        for l, r in _align_contract_dims(lhs.primal_dims, rhs.out_dims, embed=True)
    )


def _logical_size(t) -> int:
    """Product of dims for a SparseTensor, or of ``shape`` for an Array. 0-d → 1."""
    shape = getattr(t, "shape", None)
    if shape is None:
        return 1
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _matmul_contraction_depth(lhs, rhs) -> int:
    """Length of the dot-product reduction (``K`` in ``(M,K) @ (K,N)``).

    The contracting dims are the size-1-aware aligned pairs
    (``_align_contract_dims`` — the same set the kernel actually contracts, so
    the FLOP count can't drift from the topology); depth is the product of their
    *logical* sizes (a ``DiagonalIndex`` of size N with block_size B contributes
    ``N*B``, matching what ``dot_general`` reduces over after densification).
    """
    if hasattr(lhs, "primal_dims") and hasattr(rhs, "out_dims"):
        K = 1
        for l, r in _align_contract_dims(lhs.primal_dims, rhs.out_dims, embed=True):
            # The contraction runs over the broadcast (max) size: a
            # metadata-stated size-1 embed against a size-N partner reduces over
            # N, not 1. A stray size-1 is already dropped by the alignment.
            K *= max(int(l.logical_size), int(r.logical_size))
        return K
    # Both inputs are plain arrays (the dense_dense path).
    if hasattr(lhs, "shape") and hasattr(rhs, "shape"):
        if lhs.ndim == 0 or rhs.ndim == 0:
            return 1
        if lhs.ndim == 1 and rhs.ndim == 1:
            return int(lhs.shape[0])
        return int(lhs.shape[-1])
    return 1


def _compute_matmul_count(lhs, rhs, out) -> tuple[int, int, int]:
    """Exact ``(adds, muls, fmas)`` for a matmul, summed across the depth.

    Each output element is a length-``K`` dot product. Counted in the
    fused-multiply-add convention: the first contracting step is a plain
    multiply (nothing to accumulate into yet), every subsequent step is one
    fused multiply-add — so per output element you get 1 mul + (K-1) FMAs:

    * ``muls = output_size``           (the initial mul of each dot product)
    * ``adds = 0``                     (folded into the FMAs)
    * ``fmas = output_size * (K - 1)`` (accumulating multiply-adds)

    where ``output_size`` is the logical product of all kept dims and ``K``
    is the contraction depth (a ``DiagonalIndex(size=N, block_size=B)``
    contributes ``N*B``, matching what ``dot_general`` actually contracts
    after densification).

    When ``K <= 1`` (scalar matmul / outer product / identity contraction)
    there's no accumulation at all: ``muls = output_size`` and ``fmas = 0``.

    Computed from static shape / topology — pure Python, no tracing.
    """
    out_size = _logical_size(out)
    K = _matmul_contraction_depth(lhs, rhs)
    if K <= 1:
        return (0, out_size, 0)
    return (0, out_size, out_size * (K - 1))
