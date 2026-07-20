"""Tiled block-sparse matmul.

Pipeline (zero-fill fast path):
    1. Classify each dim pair as ``contract``/``batch_*``/``spatial_*``.
    2. ``_prepare_physical_arrays`` — bring both ``val`` buffers into the canonical
       (outer, block, shared_block, *leftover) layout via the shared transpose primitive.
    3. ``_execute_block_sparse_contraction`` — split / broadcast / dot_general / reduce.
    4. ``_build_output_tensor`` — re-emit ``SparseTensor`` ``out_dims`` / ``primal_dims``.

Late-densification escape hatch (non-zero fill_value):
    The tiled algorithm assumes implicit positions are zero. When ``_is_zero_fill``
    returns ``False`` for either operand, ``matmul`` reroutes through
    ``_matmul_via_densify`` — which materializes both sides via the fusion-friendly
    ``dense_for_matmul`` and runs a plain ``jax.lax.dot_general``. Densification
    stays as a JAX expression so XLA can fold it into the matmul kernel (SMEM, not HBM).
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

from .block_storage import _band_axis_select
from .dense import dense_for_matmul
from .layout import generate_block_permutation, generate_grouped_permutation
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
    banded_geom: "BandedLayout | MultiAxisBandedLayout | None" = None


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
    return (
        PairData(
            l_outer, l_block, l_shared, _outer_v(lo, li), l_block_v, l_shared_v, lo, li
        ),
        PairData(
            r_outer,
            r_block,
            r_shared,
            _outer_v(ri if swap_rhs else ro, ro if swap_rhs else ri),
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
    def _full(tag, lo, li, ro, ri):
        import sys as _s
        f = lambda d: "None" if d is None else _d(d)
        print(f"[PAIRDUMP] {tag}", file=_s.stderr)
        print(f"[PAIRDUMP]   lhs.out ={f(lo)}", file=_s.stderr)
        print(f"[PAIRDUMP]   lhs.prim={f(li)}", file=_s.stderr)
        print(f"[PAIRDUMP]   rhs.out ={f(ro)}", file=_s.stderr)
        print(f"[PAIRDUMP]   rhs.prim={f(ri)}", file=_s.stderr)
    if lout and rout and lout.logical_size != rout.logical_size:
        raise ValueError(
            f"Batch mismatch (out side): lhs[{_d(lout)}] vs rhs[{_d(rout)}] — "
            f"logical_size {lout.logical_size} != {rout.logical_size}"
        )
    if lprimal and rprimal and lprimal.logical_size != rprimal.logical_size:
        _full("PRIMAL-SIDE MISMATCH", lout, lprimal, rout, rprimal)
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
        l_len, l_v = _dim_vals(lout, False)
        r_len, r_v = _dim_vals(rout, False)
        return Pair(
            "batch_out",
            1,
            PairData(l_len, 1, 1, l_v, None, None, lout),
            PairData(r_len, 1, 1, r_v, None, None, rout),
        )
    if lprimal and rprimal:
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
            # present-but-None for dense/scalar contracting dims, so a getattr
            # default never fires — use ``or`` to fall through), or 1.
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
                break
            used_r.add(cand[0])
            full.append((a, cand[0]))
        if len(full) == len(lhs_primal):
            full.sort()
            return full
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
def _prepare_physical_arrays(lhs_val, rhs_val, pairs):
    def flat_axes(sides):
        return [
            a for s in sides for a in (s.outer_axis, s.block_axis, s.shared_block_axis)
        ]

    return (
        _prepare_physical_array(lhs_val, flat_axes([p.lhs for p in pairs])),
        _prepare_physical_array(rhs_val, flat_axes([p.rhs for p in pairs])),
    )


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
            s = p.lhs.shared_block_len // (lcm_len // p.lhs.outer_len)
        elif p.rhs.block_len > 1:
            s = p.rhs.block_len // (lcm_len // p.rhs.outer_len)
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


def _prepare_contraction_views(lhs_val, rhs_val, pairs, shared, total, split):
    N = len(pairs)
    lhs_leftover, rhs_leftover = (
        list(lhs_val.shape[3 * N :]),
        list(rhs_val.shape[3 * N :]),
    )

    def split_shape(val, pairs_side, lens, is_lhs):
        out = []
        for i, p in enumerate(pairs):
            ax0, ax1, ax2 = val.shape[3 * i], val.shape[3 * i + 1], val.shape[3 * i + 2]
            ps = pairs_side[i]
            if is_lhs:
                tail = (1, 1) if ax2 == 1 else (total[i] // ps.outer_len, split[i])
                out.extend([ax0, ax1, *tail])
            else:
                tail = (1, 1) if ax1 == 1 else (total[i] // ps.outer_len, split[i])
                out.extend([ax0, *tail, ax2])
        return out + lens

    lhs_split = split_shape(lhs_val, [p.lhs for p in pairs], lhs_leftover, True)
    rhs_split = split_shape(rhs_val, [p.rhs for p in pairs], rhs_leftover, False)
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
            for v in (p.lhs.outer_len, total[i] // p.lhs.outer_len)
        ]
        + [p.lhs.block_len for p in pairs]
        + split
        + lhs_leftover
    )
    rhs_unmerged = (
        [
            v
            for i, p in enumerate(pairs)
            for v in (p.rhs.outer_len, total[i] // p.rhs.outer_len)
        ]
        + split
        + [p.rhs.shared_block_len for p in pairs]
        + rhs_leftover
    )
    lhs_view = _as_shape(lhs_view, lhs_unmerged, mode="broadcast")
    rhs_view = _as_shape(rhs_view, rhs_unmerged, mode="broadcast")
    lhs_merged = list(total) + [p.lhs.block_len for p in pairs] + split + lhs_leftover
    rhs_merged = (
        list(total) + split + [p.rhs.shared_block_len for p in pairs] + rhs_leftover
    )
    lhs_view = _as_shape(lhs_view, lhs_merged, mode="reshape")
    rhs_view = _as_shape(rhs_view, rhs_merged, mode="reshape")
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
    return lhs_view, rhs_view, lhs_bc, rhs_bc


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
    res = jax.ops.segment_sum(
        res_view.reshape(math.prod(total), math.prod(extra)),
        jnp.array(flat_arr),
        num_segments=math.prod(per_num),
    )
    return res.reshape(*per_num, *extra)


def _dot_general_axes(N, pairs):
    contract_l, contract_r = [], []
    batch_l, batch_r = list(range(N)), list(range(N))
    for i, p in enumerate(pairs):
        if p.pairing_type == "contract":
            contract_l.append(2 * N + i)
            contract_r.append(N + i)
        else:
            batch_l.append(2 * N + i)
            batch_r.append(N + i)
    return ((contract_l, contract_r), (batch_l, batch_r))


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


def _execute_block_sparse_contraction(lhs_val, rhs_val, pairs, ctx: "Ctx"):
    N = len(pairs)
    shared, total, split, scalar = _contraction_factors(pairs)
    lhs_view, rhs_view, lhs_bc, rhs_bc = _prepare_contraction_views(
        lhs_val, rhs_val, pairs, shared, total, split
    )
    lhs_leftover = list(lhs_view.shape[3 * N :])
    rhs_leftover = list(rhs_view.shape[3 * N :])
    res_raw = jax.lax.dot_general(lhs_view, rhs_view, _dot_general_axes(N, pairs))
    nc_len = sum(1 for p in pairs if p.pairing_type != "contract")
    dg_perm = (
        list(range(2 * N + nc_len))
        + list(
            range(
                2 * N + nc_len + len(lhs_leftover), 3 * N + nc_len + len(lhs_leftover)
            )
        )
        + list(range(2 * N + nc_len, 2 * N + nc_len + len(lhs_leftover)))
        + list(range(3 * N + nc_len + len(lhs_leftover), res_raw.ndim))
    )
    if dg_perm != list(range(len(dg_perm))):
        res_raw = jnp.transpose(res_raw, dg_perm)
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
    banded_geom = _should_emit_block_banded(
        ctx,
        pairs,
        shared,
        total,
        final_lhs_lens,
        final_rhs_lens,
        lhs_leftover,
        rhs_leftover,
    )
    if banded_geom is None:
        # Fall through to the K>1 multi-axis probe when the K=1 single-pair
        # probe didn't fire (typically because ``len(pairs) != 1``).
        banded_geom = _should_emit_multi_axis_banded(
            ctx,
            pairs,
            shared,
            total,
            final_lhs_lens,
            final_rhs_lens,
            lhs_leftover,
            rhs_leftover,
        )
    return grid, shared, final_lhs_lens, final_rhs_lens, scalar, banded_geom


# --- Output tensor build --------------------------------------------------
def _resolve_output_shape(ctx, res):
    shape, sh_map, lhs_map, rhs_map, squeeze, ax = [], {}, {}, {}, [], 0
    for i, factor in enumerate(res.shared_factors):
        shape.append(factor)
        sh_map[i] = ax
        ax += 1
    for i, p in enumerate(ctx.pairs):
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
    for i, p in enumerate(ctx.pairs):
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


def _build_pair_dims(pm, i, sa, la, ra, res, next_id):
    """Build (out_dim, primal_dim, next_id) for one pair, dispatching on pairing_type."""
    sf = res.shared_factors[i]
    final_l = (pm.lhs.outer_len // sf) * res.lhs_block_lens[i]
    final_r = (pm.rhs.outer_len // sf) * res.rhs_block_lens[i]
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
            out_dim = _build_sparse(
                l_id, rs_id, sf, sa, pres_shared, final_l, la, pres_lhs
            )
            primal_dim = _build_sparse(
                rs_id, l_id, sf, sa, pres_shared, final_r, ra, pres_rhs
            )
        elif pm.lhs.dim:
            out_dim = DenseIndex(l_id, final_l, axis=la if pres_lhs else None)
        elif pm.rhs.shared_dim:
            primal_dim = DenseIndex(rs_id, final_r, axis=ra if pres_rhs else None)
    elif pt == "batch_out":
        out_dim = DenseIndex(
            l_id if pm.lhs.dim else ls_id, sf, axis=sa if pres_shared else None
        )
    elif pt == "batch_primal":
        primal_dim = DenseIndex(
            l_id if pm.lhs.dim else ls_id, sf, axis=sa if pres_shared else None
        )
    elif pt == "spatial_out_lhs":
        out_dim = DenseIndex(l_id, final_l, axis=la if pres_lhs else None)
    elif pt == "spatial_out_rhs":
        out_pres = getattr(pm.rhs, "block_axis", None) is not None
        out_dim = DenseIndex(r_id, final_r, axis=ra if out_pres else None)
    elif pt == "spatial_primal_lhs":
        prim_pres = pm.lhs.shared_block_axis is not None
        primal_dim = DenseIndex(ls_id, final_l, axis=la if prim_pres else None)
    elif pt == "spatial_primal_rhs":
        primal_dim = DenseIndex(rs_id, final_r, axis=ra if pres_rhs else None)
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
    return out_dim, primal_dim, next_id


def _build_output_tensor(ctx, rhs_dims, res):
    from graphax.sparse.tensor import SparseTensor

    shape, (sh_map, lhs_map, rhs_map), squeeze = _resolve_output_shape(ctx, res)
    next_id = (
        builtins.max([d.id for d in ctx.lhs.dims] + [d.id for d in rhs_dims] + [-1]) + 1
    )
    out_dims, primal_dims = [], []
    for i, pm in enumerate(ctx.pairs):
        od, pd, next_id = _build_pair_dims(
            pm, i, sh_map[i], lhs_map[i], rhs_map[i], res, next_id
        )
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
    for i in range(len(ctx.pairs)):
        for ax in (sh_map[i], lhs_map[i], rhs_map[i]):
            if ax not in used_axes:
                squeeze.append(ax)
    grid_view = res.grid.reshape(shape) if res.grid.shape != tuple(shape) else res.grid
    if squeeze:
        unique_sq = tuple(sorted(set(squeeze)))
        final_shape = [s for i, s in enumerate(shape) if i not in unique_sq]
        if grid_view.size == math.prod(final_shape):
            values = grid_view.reshape(final_shape)
        else:
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
    final_out = tuple(sorted(out_dims, key=lambda d: d.id))
    final_primal = tuple(sorted(primal_dims, key=lambda d: d.id))
    id_map = {d.id: i for i, d in enumerate(final_out + final_primal)}

    def finalize(d, new_id):
        kw = {"id": new_id}
        if d.is_sparse:
            kw["other_id"] = id_map.get(d.other_id, d.other_id)
        return replace(d, **kw)

    final_out = tuple(finalize(d, i) for i, d in enumerate(final_out))
    n_out = len(final_out)
    final_primal = tuple(finalize(d, n_out + i) for i, d in enumerate(final_primal))
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
    # Banded emission: when ``_should_emit_block_banded`` (run upstream in
    # ``_execute_block_sparse_contraction``) finds a band-storage form
    # strictly tighter than the natural dense output, pack ``values`` into
    # the extended ``BlockBanded`` pytree via broadcast+where+sum (gather-
    # free; XLA fuses with the producing dot_general). The probe gates on
    # 2-D single-contract-pair geometry with no leftover, so ``values`` is
    # always 2-D dense at this point.
    if res.banded_geom is not None and values is not None:
        from graphax.sparse.indexes import BandedIndex

        layout = res.banded_geom
        out_id = final_out[0].id if final_out else 0
        primal_id = final_primal[0].id if final_primal else (out_id + 1)

        # Multi-axis (K≥2) banded output: pack into the interleaved band
        # buffer and emit K BandedIndex pairs describing each axis-pair's band.
        if isinstance(layout, MultiAxisBandedLayout):
            band_data = _pack_dense_to_multi_axis_banded(values, layout)
            K = len(layout.per_axis)
            out_dims_new = []
            primal_dims_new = []
            for i, ax in enumerate(layout.per_axis):
                is_row_primary = ax.primary_axis == 0
                M_row = ax.m_primary if is_row_primary else ax.n_secondary
                M_col = ax.n_secondary if is_row_primary else ax.m_primary
                # ``size`` is the META count (logical_size = size*block_size),
                # matching DiagonalIndex convention.
                out_dims_new.append(BandedIndex(
                    id=out_id + i, size=ax.n_meta * M_row,
                    axis=i, other_id=primal_id + i,
                    block_size=ax.block_row, block_axis=K + i,
                    band_width=ax.band_width, offset=ax.offset,
                    primary=is_row_primary, n_secondary=ax.n_secondary,
                    n_meta=ax.n_meta,
                ))
                primal_dims_new.append(BandedIndex(
                    id=primal_id + i, size=ax.n_meta * M_col,
                    axis=K + i, other_id=out_id + i,
                    block_size=ax.block_col, block_axis=3 * K + i,
                    band_width=ax.band_width, offset=ax.offset,
                    primary=is_row_primary, n_secondary=ax.n_secondary,
                    n_meta=ax.n_meta,
                ))
            return SparseTensor(
                tuple(out_dims_new), tuple(primal_dims_new), band_data,
                scalar_mult=jnp.asarray(final_mult).astype(values.dtype),
                fill_value=None,  # tiled path assumes zero fill → statically zero
                check_consistency=False,
            )

        # K=1 banded output: band buffer in val + a single BandedIndex pair.
        band_data = _pack_dense_to_banded(values, layout)
        is_row_primary = layout.primary_axis == 0
        M_row = layout.m_primary if is_row_primary else layout.n_secondary
        M_col = layout.n_secondary if is_row_primary else layout.m_primary
        # ``size`` is the META count (logical_size = size*block_size).
        # ``axis`` / ``block_axis`` are NOMINAL for a BandedIndex — densify
        # reconstructs the layout from ``val.shape`` + the band params, never
        # from these fields — but we keep them distinct per side and matching
        # the K>=2 convention (out: axis=i, block_axis=K+i; primal: axis=K+i,
        # block_axis=3K+i, here K=1) so no consumer conflates the two sides.
        out_ix = BandedIndex(
            id=out_id, size=layout.n_meta * M_row,
            axis=0, other_id=primal_id, block_size=layout.block_row, block_axis=1,
            band_width=layout.band_width, offset=layout.offset,
            primary=is_row_primary, n_secondary=layout.n_secondary,
            n_meta=layout.n_meta,
        )
        primal_ix = BandedIndex(
            id=primal_id, size=layout.n_meta * M_col,
            axis=1, other_id=out_id, block_size=layout.block_col, block_axis=3,
            band_width=layout.band_width, offset=layout.offset,
            primary=is_row_primary, n_secondary=layout.n_secondary,
            n_meta=layout.n_meta,
        )
        return SparseTensor(
            (out_ix,), (primal_ix,), band_data,
            scalar_mult=jnp.asarray(final_mult).astype(values.dtype),
            fill_value=None,  # tiled path assumes zero fill → statically zero
            check_consistency=False,
        )
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


class MultiAxisBandedLayout(NamedTuple):
    """Multi-axis (K>1) banded layout returned by the multi-contract probe.

    Each axis-pair carries its own :class:`BandedLayout`-equivalent
    metadata (orientation, band width, sub-block dims, offset). Currently
    populated for K=2; the structure naturally extends to K>2 once the
    emission kernel supports it.
    """

    per_axis: tuple["BandedLayout", ...]


class BandedLayout(NamedTuple):
    """Static layout for emitting a ``BlockBanded`` matmul output.

    Computed by :func:`_should_emit_block_banded` from input metadata alone
    (no traced array shapes). The new fields mirror the extended
    :class:`~graphax.sparse.ops.block_storage.BlockBanded` pytree:

      * ``primary_axis``: ``0`` = row-primary (band traverses cols), ``1`` =
        col-primary (band traverses rows). The probe picks whichever
        orientation packs tighter for the geometry at hand.
      * ``m_primary``: meta-blocks along the primary axis (``M_row`` for
        row-primary, ``M_col`` for col-primary). Equals ``data.shape[0]``.
      * ``n_secondary``: meta-blocks along the non-primary axis.
      * ``band_width``: ``W`` = max in-band sub-blocks per primary slot.
        Equals ``data.shape[1]``.
      * ``block_row`` / ``block_col``: sub-block dims. Equal
        ``lhs.out_dims[0].block_size`` / ``rhs.primal_dims[0].block_size``.
      * ``offset``: per-primary integer offsets along the secondary axis.

    Replaces the legacy ``BandedGeom`` (which mirrored a dead emission gate
    and never fired in practice). The new probe fires for misaligned-
    contract cases (``B_a_w`` and ``B_b_h`` non-divisible), which the
    pre-Phase-5d matmul stored as a fully-dense output.
    """

    primary_axis: int
    m_primary: int
    n_secondary: int
    band_width: int
    block_row: int
    block_col: int
    offset: tuple[int, ...]
    n_meta: int = 1


def _row_band_spans(
    M_a: int, B_a_w: int, B_b_h: int, M_b: int
) -> list[tuple[int, int]]:
    """For each output meta-row ``a in [0, M_a)``, return
    ``(b_lo, b_hi)`` — the inclusive-exclusive range of overlapping
    rhs meta-cols. ``b_lo`` is the smallest ``b`` with overlap, ``b_hi``
    one past the largest.

    The overlap condition (mathematician's derivation): row ``a`` covers
    contracting range ``[a*B_a_w, (a+1)*B_a_w)``; col ``b`` covers
    ``[b*B_b_h, (b+1)*B_b_h)``. Overlap iff
    ``a*B_a_w < (b+1)*B_b_h ∧ b*B_b_h < (a+1)*B_a_w``.
    """
    spans: list[tuple[int, int]] = []
    for a in range(M_a):
        a_lo, a_hi = a * B_a_w, (a + 1) * B_a_w
        # b_lo: smallest b such that (b+1)*B_b_h > a_lo, i.e., b >= a_lo // B_b_h.
        b_lo = a_lo // B_b_h
        # b_hi: smallest b such that b*B_b_h >= a_hi, i.e., b >= ceil(a_hi/B_b_h).
        b_hi = -(-a_hi // B_b_h)
        b_lo = builtins.max(0, b_lo)
        b_hi = builtins.min(M_b, b_hi)
        spans.append((b_lo, b_hi))
    return spans


def _should_emit_block_banded(
    ctx: "Ctx",
    pairs: list["Pair"],
    shared: list[int],
    total: list[int],
    final_lhs_lens: list[int],
    final_rhs_lens: list[int],
    lhs_leftover: list[int],
    rhs_leftover: list[int],
) -> BandedLayout | None:
    """Static probe: detect whether the matmul output has a band-sparse
    structure tighter than the fully-dense form, and choose the
    row-primary vs col-primary orientation that packs tighter.

    Returns a :class:`BandedLayout` when emission is justified
    (band-storage strictly less than the dense alternative), else ``None``
    (matmul falls through to the legacy ``val=values`` path).

    Gates (static, all decidable from operand metadata):

      * 2-D single-contract-pair matmul on block-sparse inputs.
      * Equal logical contracting size: ``M_a * B_a_w == M_b * B_b_h``.
      * Band width > 1 OR row/col counts differ — i.e., something is
        actually being compressed (pure-aligned cases stay on the existing
        ``val=(M, B_h, B_w)`` storage, no BlockBanded wrap).
    """
    lhs, rhs = ctx.lhs, ctx.rhs
    if len(lhs.dims) != 2 or len(rhs.dims) != 2:
        return None
    if not all(d.is_sparse for d in (*lhs.dims, *rhs.dims)):
        return None
    if len(pairs) != 1 or pairs[0].pairing_type != "contract":
        return None
    if lhs_leftover or rhs_leftover:
        return None
    if len(total) != 1 or len(final_lhs_lens) != 1 or len(final_rhs_lens) != 1:
        return None

    M_a = lhs.out_dims[0].size
    B_a_h = lhs.out_dims[0].block_size or 1
    B_a_w = lhs.primal_dims[0].block_size or 1
    M_b = rhs.primal_dims[0].size
    B_b_h = rhs.out_dims[0].block_size or 1
    B_b_w = rhs.primal_dims[0].block_size or 1
    if M_a * B_a_w != M_b * B_b_h:
        return None  # logical contract sizes don't align — outside scope

    # Divisibility gate: when one contract block divides the other, the natural
    # matmul output preserves meta-block structure (stored as ``val=(M, B_h, B_w)``
    # — tight already). BlockBanded only helps when neither divides the other
    # — that's when the LCM-grid expansion forces the natural output to be
    # 2-D dense, losing the meta-block dim.
    if B_a_w % B_b_h == 0 or B_b_h % B_a_w == 0:
        return None

    # When ``gcd(M_a, M_b) > 1`` the matmul output splits into ``N = gcd``
    # independent banded meta-blocks stacked along the meta-diagonal (Case A
    # of the mathematician's taxonomy). The per-batch geometry is the same
    # as a smaller ``(M_per_a, B_a) × (M_per_b, B_b)`` matmul; ``BlockBanded``'s
    # ``n_meta`` field carries the outer batch count.
    N = math.gcd(M_a, M_b)
    M_per_a = M_a // N
    M_per_b = M_b // N

    B_row, B_col = B_a_h, B_b_w

    # Compute both orientations' band spans at *per-batch* granularity.
    row_spans = _row_band_spans(M_per_a, B_a_w, B_b_h, M_per_b)
    col_spans = _row_band_spans(M_per_b, B_b_h, B_a_w, M_per_a)

    # If any row has empty span (shouldn't happen given the logical-contract
    # check above, but bail defensively) - no band, fall through.
    if any(lo >= hi for lo, hi in row_spans):
        return None

    W_rp = builtins.max(hi - lo for lo, hi in row_spans)
    W_cp = builtins.max(hi - lo for lo, hi in col_spans)
    offset_rp = tuple(lo for lo, _ in row_spans)
    offset_cp = tuple(lo for lo, _ in col_spans)

    # Storage footprint per orientation. ``N`` factors out — pick whichever
    # per-batch slot count is smaller. Compare against the natural per-batch
    # dense (which is what the eager path materializes when N>1).
    rp_cost = N * M_per_a * W_rp * B_row * B_col
    cp_cost = N * M_per_b * W_cp * B_row * B_col
    dense_cost = N * (M_per_a * B_row) * (M_per_b * B_col)

    # Fall through when no compression possible — keeps existing val=values path
    # for aligned (divisor) cases that the natural meta-block storage already
    # handles tightly.
    if rp_cost >= dense_cost and cp_cost >= dense_cost:
        return None

    # Also fall through when the per-batch band is the trivial identity
    # diagonal (``W=1``, identity offset, square per-batch counts). The
    # existing 3-D ``val=(N, M_per*B_h, M_per*B_w)`` storage is equivalent to
    # a per-batch pure block-diagonal and the wrapping doesn't add value at
    # such granularity.
    if (
        W_rp == 1
        and offset_rp == tuple(range(M_per_a))
        and M_per_a == M_per_b
        and W_cp == 1
        and offset_cp == tuple(range(M_per_b))
    ):
        return None

    # Pick the tighter orientation.
    if cp_cost < rp_cost:
        return BandedLayout(
            primary_axis=1,
            m_primary=M_per_b,
            n_secondary=M_per_a,
            band_width=W_cp,
            block_row=B_row,
            block_col=B_col,
            offset=offset_cp,
            n_meta=N,
        )
    return BandedLayout(
        primary_axis=0,
        m_primary=M_per_a,
        n_secondary=M_per_b,
        band_width=W_rp,
        block_row=B_row,
        block_col=B_col,
        offset=offset_rp,
        n_meta=N,
    )


def _should_emit_multi_axis_banded(
    ctx: "Ctx",
    pairs: list["Pair"],
    shared: list[int],
    total: list[int],
    final_lhs_lens: list[int],
    final_rhs_lens: list[int],
    lhs_leftover: list[int],
    rhs_leftover: list[int],
) -> MultiAxisBandedLayout | None:
    """K=2 multi-contract probe: detect when both contract pairs are
    misaligned and emission as a multi-axis block-banded output strictly
    beats the dense 4-D output.

    K=2 only for now; K>2 would mirror the same per-pair logic and emit
    a higher-rank multi-axis block-banded output once the K-axis
    ``to_dense`` kernel extends to K>2.
    """
    K = len(pairs)
    if K < 2:
        return None  # K=1 handled by single-axis probe upstream.
    if any(p.pairing_type != "contract" for p in pairs):
        return None
    if lhs_leftover or rhs_leftover:
        return None

    lhs, rhs = ctx.lhs, ctx.rhs
    if len(lhs.out_dims) != K or len(lhs.primal_dims) != K:
        return None
    if len(rhs.out_dims) != K or len(rhs.primal_dims) != K:
        return None
    if not all(d.is_sparse for d in (*lhs.dims, *rhs.dims)):
        return None

    per_axis: list[BandedLayout] = []
    for pair_i in range(K):
        M_a = lhs.out_dims[pair_i].size
        B_a_h = lhs.out_dims[pair_i].block_size or 1
        B_a_w = lhs.primal_dims[pair_i].block_size or 1
        M_b = rhs.primal_dims[pair_i].size
        B_b_h = rhs.out_dims[pair_i].block_size or 1
        B_b_w = rhs.primal_dims[pair_i].block_size or 1
        if M_a * B_a_w != M_b * B_b_h:
            return None
        # Per-axis divisibility gate (same as K=1 single-axis probe).
        if B_a_w % B_b_h == 0 or B_b_h % B_a_w == 0:
            return None
        # K=2 multi-axis with per-axis n_meta>1 is a follow-up (would
        # require multi-axis block-banded ``to_dense`` per-batch handling).
        if math.gcd(M_a, M_b) > 1:
            return None

        row_spans = _row_band_spans(M_a, B_a_w, B_b_h, M_b)
        if any(lo >= hi for lo, hi in row_spans):
            return None
        W_rp = builtins.max(hi - lo for lo, hi in row_spans)
        offset_rp = tuple(lo for lo, _ in row_spans)
        # K=2 multi-axis: row-primary only for now (col-primary follows the
        # same swap+transpose pattern but the kernel doesn't yet implement it).
        per_axis.append(
            BandedLayout(
                primary_axis=0,
                m_primary=M_a,
                n_secondary=M_b,
                band_width=W_rp,
                block_row=B_a_h,
                block_col=B_b_w,
                offset=offset_rp,
                n_meta=1,
            )
        )

    # Storage check: combined K=2 compressed < combined dense.
    # Compressed = prod over axes of (M_p * W * B_row * B_col).
    # Dense     = prod over axes of (M_p * n_sec * B_row * B_col).
    # Ratio = prod(W_i / n_sec_i). Compression iff prod(W_i) < prod(n_sec_i).
    compressed_factor = 1
    dense_factor = 1
    for ax in per_axis:
        compressed_factor *= ax.band_width
        dense_factor *= ax.n_secondary
    if compressed_factor >= dense_factor:
        return None

    return MultiAxisBandedLayout(per_axis=tuple(per_axis))


def _pack_dense_to_banded(values: Array, layout: BandedLayout) -> Array:
    """Pack a dense ``(M_row*B_row, M_col*B_col)`` matmul output into
    ``BlockBanded`` data shape ``(m_primary, W, B_row, B_col)`` via
    broadcast+where+sum — no gather, XLA-fusable with the producing
    dot_general so the dense intermediate stays in SMEM, not HBM.

    Layout:
      * ``values``: dense output of the matmul.
      * ``layout``: computed by ``_should_emit_block_banded``; determines
        primary-axis orientation, band width, sub-block sizes, offsets.

    Algorithm (uniform for row- and col-primary):

      1. Reshape dense ``(M_row * B_row, M_col * B_col)`` to
         ``(M_row, B_row, M_col, B_col)``.
      2. Transpose so the primary-axis becomes axis-0:
         ``(M_primary, M_secondary, B_row, B_col)``.
      3. Insert a W-axis via broadcast: ``(M_p, W, M_s, B_row, B_col)``.
      4. Build a one-hot mask ``m == offset[p] + w`` of shape ``(M_p, W, M_s)``
         and ``jnp.where(mask, ., 0).sum(axis=2)`` to collapse the secondary
         axis. Exactly one ``m`` matches per ``(p, w)``, so the sum acts as a
         per-cell select — XLA fuses into a single ``kLoop`` pass.
    """
    M_p = layout.m_primary
    N_s = layout.n_secondary
    W = layout.band_width
    B_row = layout.block_row
    B_col = layout.block_col
    N = layout.n_meta

    if layout.primary_axis == 0:
        M_row, M_col = M_p, N_s
    else:
        M_row, M_col = N_s, M_p

    # Step 1: reshape to ``(N, M_row, B_row, M_col, B_col)``. For ``N=1`` values
    # arrives as 2-D dense ``(M_row*B_row, M_col*B_col)``; for ``N>1`` it arrives
    # 3-D ``(N, M_row*B_row, M_col*B_col)`` (the matmul writes the per-batch
    # diagonals into a leading axis already).
    if N == 1:
        grid_5d = values.reshape(1, M_row, B_row, M_col, B_col)
    else:
        grid_5d = values.reshape(N, M_row, B_row, M_col, B_col)

    # Step 2: permute so axis-1 is the primary, axis-2 is the secondary.
    # (Axis-0 stays as the batch axis.)
    if layout.primary_axis == 0:
        # Row-primary: M_row at axis-1, M_col at axis-3 → bring M_col to axis-2.
        grid_t = grid_5d.transpose(0, 1, 3, 2, 4)  # (N, M_p, N_s, B_row, B_col)
    else:
        # Col-primary: M_col at axis-3 (= M_p), M_row at axis-1 (= N_s).
        grid_t = grid_5d.transpose(0, 3, 1, 2, 4)  # (N, M_p, N_s, B_row, B_col)

    # Step 3: insert W axis via broadcast (pure broadcast, zero-copy).
    grid_b = jnp.broadcast_to(
        grid_t[:, :, None, ...],  # (N, M_p, 1, N_s, B_row, B_col)
        (N, M_p, W, N_s, B_row, B_col),
    )

    # Step 4: one-hot band mask + sum collapse. Shares ``_band_axis_select`` with
    # the inverse densify kernel (provably inverse). The selector yields
    # ``(M_p, N_s, W)``; transpose to this pack's ``(M_p, W, N_s)`` axis order.
    mask = _band_axis_select(N_s, W, tuple(layout.offset)).transpose(0, 2, 1)
    mask = mask[None, ..., None, None]  # (1, M_p, W, N_s, 1, 1)
    out = jnp.where(mask, grid_b, 0).sum(axis=3)  # (N, M_p, W, B_row, B_col)
    # Flatten the leading ``(N, M_p)`` into ``(N * M_p)`` — the layout
    # BlockBanded expects on its ``data`` axis-0 (``n_meta * M_per_primary``).
    return out.reshape(N * M_p, W, B_row, B_col)


def _pack_dense_to_multi_axis_banded(
    values: Array, layout: MultiAxisBandedLayout
) -> Array:
    """Pack a dense ``(M_row_0*B_row_0, ..., M_row_{K-1}*B_row_{K-1},
    M_col_0*B_col_0, ..., M_col_{K-1}*B_col_{K-1}, *L)`` K-axis matmul
    output into multi-axis block-banded data shape
    ``(M_p_0, W_0, ..., M_p_{K-1}, W_{K-1}, B_row_0, ..., B_row_{K-1},
       B_col_0, ..., B_col_{K-1}, *L)`` via per-axis broadcast+where+sum.

    Generalizes the K=1 packing kernel by adding one ``(M_p, M_s, W)``
    prefix triple per axis, combined via AND of per-axis one-hot masks.
    Both per-axis ``M_s`` reductions happen in one fused pass.
    """
    K = len(layout.per_axis)
    M_p = [ax.m_primary for ax in layout.per_axis]
    M_s = [ax.n_secondary for ax in layout.per_axis]
    W = [ax.band_width for ax in layout.per_axis]
    B_row = [ax.block_row for ax in layout.per_axis]
    B_col = [ax.block_col for ax in layout.per_axis]
    offsets = [ax.offset for ax in layout.per_axis]
    L = values.shape[2 * K :]

    # Step 1: reshape dense to ``(M_p_0, B_row_0, ..., M_p_{K-1}, B_row_{K-1},
    # M_s_0, B_col_0, ..., M_s_{K-1}, B_col_{K-1}, *L)`` — split each
    # output axis into (meta, sub-block).
    split_shape = []
    for i in range(K):
        split_shape += [M_p[i], B_row[i]]
    for i in range(K):
        split_shape += [M_s[i], B_col[i]]
    split_shape += list(L)
    grid = values.reshape(*split_shape)

    # Step 2: permute to group per-axis (M_p_i, M_s_i, B_row_i, B_col_i):
    # Target order: M_p_0, M_s_0, M_p_1, M_s_1, ..., M_p_{K-1}, M_s_{K-1},
    # B_row_0, B_row_1, ..., B_row_{K-1}, B_col_0, ..., B_col_{K-1}, *L.
    perm: list[int] = []
    for i in range(K):
        perm.append(2 * i)              # M_p_i (rows split)
        perm.append(2 * K + 2 * i)      # M_s_i (cols split)
    for i in range(K):
        perm.append(2 * i + 1)          # B_row_i
    for i in range(K):
        perm.append(2 * K + 2 * i + 1)  # B_col_i
    perm += list(range(4 * K, 4 * K + len(L)))
    grid = grid.transpose(perm)
    # Shape now: (M_p_0, M_s_0, M_p_1, M_s_1, ..., M_p_{K-1}, M_s_{K-1},
    #             B_row_0, ..., B_row_{K-1}, B_col_0, ..., B_col_{K-1}, *L)

    # Step 3: insert W axes via singleton broadcast. Each (M_p_i, M_s_i)
    # gets a W_i axis right after the pair.
    indexer = []
    for i in range(K):
        indexer.append(slice(None))  # M_p_i
        indexer.append(slice(None))  # M_s_i
        indexer.append(None)          # W_i (inserted)
    indexer += [slice(None)] * (2 * K + len(L))  # B_row, B_col, L
    grid_b = grid[tuple(indexer)]
    # Broadcast W axes to their actual sizes.
    expanded = []
    for i in range(K):
        expanded += [M_p[i], M_s[i], W[i]]
    expanded += B_row
    expanded += B_col
    expanded += list(L)
    grid_b = jnp.broadcast_to(grid_b, tuple(expanded))

    # Step 4: build per-axis selection masks ``w_idx == b - offset[a]``
    # and AND them. Shares ``_band_axis_select`` with the inverse densify kernel
    # so pack / densify stay provably in-band-consistent.
    combined_mask = None
    for i in range(K):
        sel = _band_axis_select(M_s[i], W[i], offsets[i])  # (M_p_i, M_s_i, W_i)
        sel_shape = [1] * len(expanded)
        sel_shape[3 * i] = M_p[i]
        sel_shape[3 * i + 1] = M_s[i]
        sel_shape[3 * i + 2] = W[i]
        sel_r = sel.reshape(*sel_shape)
        combined_mask = sel_r if combined_mask is None else (combined_mask & sel_r)

    # Step 5: where + sum over all M_s_i axes (positions 1, 4, 7, ...) in
    # one fused reduction.
    Ms_axes = tuple(3 * i + 1 for i in range(K))
    out = jnp.where(combined_mask, grid_b, 0).sum(axis=Ms_axes)
    # Shape: (M_p_0, W_0, M_p_1, W_1, ..., B_row_0, ..., B_col_{K-1}, *L)
    # — matches the multi-axis block-banded data layout.
    return out


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


# --- Both-implicit contracting-pair analytic fold (GRAPHAX_KEEP_BLOCKDIAG) -----
import os as _os
_KEEP_BLOCKDIAG_MM = _os.environ.get("GRAPHAX_KEEP_BLOCKDIAG", "1") != "0"


def _struct_lower_enabled() -> bool:
    """Structure-lowering layer gate (GRAPHAX_STRUCT_LOWER, default OFF).
    Read per call so tests / the differential harness can toggle it without
    re-importing the module."""
    return _os.environ.get("GRAPHAX_STRUCT_LOWER", "0") != "0"


def _einsum_general_enabled() -> bool:
    """New einsum general-path gate (GRAPHAX_EINSUM_GENERAL, default OFF).
    Read per call so the differential harness can toggle it without
    re-importing the module. When OFF, ``matmul`` is byte-identical to the
    incumbent waterfall."""
    return _os.environ.get("GRAPHAX_EINSUM_GENERAL", "0") != "0"
# Two-scalar matmul -> elementwise multiply (seed-vertex aggregation). Default on.
_SEED_SCALAR_MM = _os.environ.get("GRAPHAX_SEED_VERTICES_SCALAR_MM", "1") != "0"


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
    if _both_scalar:
        out = new_lhs * new_rhs
        cnt = (0, 1, 0)  # one scalar multiply
    else:
        res = matmul(new_lhs, new_rhs, count=count)
        if count:
            out, cnt = res
        else:
            out = res
    fac = jnp.asarray(factor, dtype=out.scalar_mult.dtype)
    out = out.copy(scalar_mult=out.scalar_mult * fac)
    return (out, cnt) if count else out


# --- New einsum general path (GRAPHAX_EINSUM_GENERAL, default OFF) ----------
def _carries_sparse_pair(st) -> bool:
    """True iff any dim of ``st`` is a member of a DiagonalIndex pair
    (``is_sparse`` == ``other_id is not None``) — a plain diagonal, a
    block-diagonal, or a spatial-sparse pair. Admits plain-diagonal
    contractions (``block_size`` None) to the einsum lowering, which keeps the
    surviving diagonal a pair instead of densifying it."""
    return any(
        getattr(d, "is_sparse", False)
        for d in (*st.out_dims, *st.primal_dims)
    )


def _einsum_matmul_general(lhs, rhs, count: bool = False):
    """Sparsity-retaining general contraction path.

    Emits ONE ``jnp.einsum`` over the operands' PHYSICAL axes only; every
    implicit dim (``axis is None``, logical>1 = extent stored once) contributes
    an einsum letter but is NEVER materialized into a physical buffer:

      * an implicit-vs-physical contraction lowers to a ``jnp.sum`` reduction
        over the physical operand's axis (XLA already fuses the broadcast),
      * a both-implicit contraction folds analytically to a scale-by-N into
        ``scalar_mult`` (no compute at all),
      * a surviving free implicit dim stays ``axis=None`` in the output; a
        surviving diagonal / block pair stays a ``DiagonalIndex`` pair;
        ``val=None`` stays ``val=None``.

    The pairing (what contracts / batches / rides through) is NOT re-derived:
    it consumes the incumbent ``_align_tensor_ids`` / ``_build_matmul_topology``
    ``Pair`` list, so it can never disagree with the tiled path about topology
    — it only changes HOW the physical buffers combine. Output ids are the
    canonical ``_build_output_tensor`` numbering so a downstream multi-edge
    contraction aligns by id.

    Returns the contracted ``SparseTensor`` (or ``(result, (adds, muls, fmas))``
    with ``count=True``), or ``None`` on any case it cannot yet prove correct —
    the caller then falls through to the existing path UNCHANGED. Returning
    ``None`` (fall through) is always the safe choice.

    The planner/executor is the shared einsum implementation in
    ``graphax.sparse.lower.matmul`` (the working prototype); this entry point
    applies the correctness firewall and routes it under the new flag. On
    normalized inputs (``matmul`` always normalizes before this hook) the
    differential harness proves the planner byte-exact vs the incumbent for
    every structured contraction except a rank-0 (scalar) operand, which is the
    one genuine structural disagreement and is excluded below.
    Gate: ``GRAPHAX_EINSUM_GENERAL`` (read by the caller) plus the guards below.
    """
    # EXACT-AD firewall: only an elimination carrying a Diag/Compress/Quant
    # transform (approx_active) may be re-associated. An exact-AD contraction
    # (transforms=()) never enters here, so the exact path is byte-identical
    # whether the flag is on or off.
    from graphax.sparse.elemental.dispatch import approx_active

    if not approx_active():
        return None
    from graphax.sparse.ops.utils import _is_approx, _is_zero_fill

    # The einsum path assumes zero fill on the implicit positions (a broadcast
    # of the stored extent). A non-zero fill is owned by the densify path.
    if not (_is_zero_fill(lhs) and _is_zero_fill(rhs)):
        return None
    # Compressed dims are materialized by ``_normalize_inputs`` before this hook;
    # this is a defensive guard for any direct caller.
    if any(getattr(d, "is_compressed", False) for d in (*lhs.dims, *rhs.dims)):
        return None
    if lhs.dtype == jnp.bool_ or rhs.dtype == jnp.bool_:
        return None
    # Only take cases that actually carry lowerable structure — an implicit dim
    # (Compress), a rectangular block (Diag), a PLAIN-DIAGONAL pair, or a
    # pure-structure (``val=None``) operand. The plain-diagonal case is the big
    # one: the incumbent general/elemental paths SKIP a diagonal-⊗-diagonal
    # contraction ('no_lowerable_structure') and densify it O(N²), but a
    # diagonal composed with a diagonal IS a diagonal — a repeated einsum index
    # (elementwise, O(N)) with the surviving pair kept symbolic. ``is_sparse``
    # (== ``other_id is not None``) detects any DiagonalIndex-pair member,
    # including a plain diagonal (``block_size`` None). A plain-dense contraction
    # carries no such structure and stays on the incumbent path.
    if not (
        _is_approx(lhs)
        or _is_approx(rhs)
        or lhs.val is None
        or rhs.val is None
        or _carries_sparse_pair(lhs)
        or _carries_sparse_pair(rhs)
    ):
        return None
    # Metadata-stated size-1↔size-N embeds are owned by the densify path.
    if _has_implicit_block_contraction(lhs, rhs):
        return None

    # A RANK-0 (scalar) operand has no dim to contract: ``X @ scalar`` is a
    # broadcast/scale, not a contraction, and the two paths genuinely DISAGREE on
    # its semantics — the incumbent collapses the unpaired contracted dim to
    # size 1, the einsum planner rides it through — so this is not float noise
    # but a real structural mismatch. The incumbent path owns this degenerate
    # case (a Compress-fully-reduced scalar edge contracting a matrix), so fall
    # through. (Two-scalar matmul is already handled earlier in ``matmul``.)
    # This is the ONLY family the differential harness proves ``_lower`` gets
    # WRONG; every structured contraction — plain/block diagonal, implicit,
    # block-refine, forced-broadcast — is byte-exact vs the incumbent on CPU.
    # (On GPU those structured cases can differ from the tiled path by ~1e-3 in
    # float32 — normal reduction-order non-associativity between two correct
    # implementations, well inside the approximation regime this runs in and
    # confirmed by the cosine-based anchors — so they are NOT gated.)
    if (not lhs.out_dims and not lhs.primal_dims) or (
        not rhs.out_dims and not rhs.primal_dims
    ):
        return None

    from graphax.sparse.lower.matmul import _NoRule, _lower

    try:
        out, counts, _tags = _lower(lhs, rhs)
    except _NoRule:
        # No rule yet for this geometry — fall through to the existing path.
        return None
    except Exception:
        # Any unexpected planner error is a fall-through, never a wrong result.
        return None
    if count:
        return out, counts
    return out


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

    lhs_dense, rhs_dense = dense_for_matmul(lhs), dense_for_matmul(rhs)

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
            lhs_dense = _pad_axis_to(lhs_dense, la, rs, _scaled_fill(lhs))
        elif rs == 1 and _is_implicit_block_dim(rd):
            rhs_dense = _pad_axis_to(rhs_dense, ra, ls, _scaled_fill(rhs))

    result = jax.lax.dot_general(
        lhs_dense,
        rhs_dense,
        (
            (tuple(lhs_contract), tuple(rhs_contract)),
            (tuple(lhs_batch), tuple(rhs_batch)),
        ),
    )

    # `dot_general` lays out result axes as: batch, then lhs's kept (in order), then rhs's
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
    """Convert array operands to ``SparseTensor`` and pre-densify any
    compressed Index dims. Pure shape / structure prep — no actual matmul
    work happens here. Materializing compressed storage is fused into the
    consuming kernel by XLA (SMEM, not HBM)."""
    if not _is_sparse(lhs):
        lhs = _arr2st(lhs, out_ndim=lhs.ndim - len(rhs.out_dims))
    if not _is_sparse(rhs):
        rhs = _arr2st(rhs, out_ndim=len(lhs.primal_dims))
    # Phase 8: pre-densify any compressed Index dims (BandedIndex / SetIndex)
    # to DiagonalIndex / DenseIndex — the tiled matmul consumes only those.
    # No-op when the operand has no compressed dims.
    from .utils import _materialize_for_op
    lhs = _materialize_for_op(lhs)
    rhs = _materialize_for_op(rhs)
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
    from graphax.sparse.dtype_compute import _unify_operand_dtypes
    lhs, rhs = _unify_operand_dtypes(lhs, rhs)
    return lhs, rhs


def _execute_tiled(ctx, rhs_dims):
    """Fallback: full tiled algorithm. Handles every case the fast paths
    bail on, including LCM-mismatched outer sizes, spatial sparse pairs,
    and broadcast / unmaterialized val axes."""
    lhs_val, rhs_val = _val_or_one(ctx.lhs), _val_or_one(ctx.rhs)
    lhs_val, rhs_val = _prepare_physical_arrays(lhs_val, rhs_val, ctx.pairs)
    grid, shared, lhs_lens, rhs_lens, scalar, banded_geom = (
        _execute_block_sparse_contraction(lhs_val, rhs_val, ctx.pairs, ctx)
    )
    res = CRes(
        grid=grid,
        shared_factors=shared,
        lhs_block_lens=lhs_lens,
        rhs_block_lens=rhs_lens,
        scalar_mult=scalar,
        banded_geom=banded_geom,
    )
    return _build_output_tensor(ctx, rhs_dims, res)



def matmul(lhs, rhs, count: bool = False):
    """Sparse matmul dispatcher. Runs a cascade of paths, first-applicable
    wins, falling back to the general tiled algorithm; each path either
    handles the operands or declines (returns ``None`` / predicate false) and
    control passes to the next.

    Dispatch order (first applicable wins), in body order:
      1. ``dense_dense``        -- both operands are plain arrays (no SparseTensor).
      2. ``scalar_elementwise`` -- both 0-rank scalars, routed through ``*``.
      3. ``einsum_general``     -- opt-in (``GRAPHAX_EINSUM_GENERAL``): opt_einsum
                                  lowering that retains sparsity.
      4. ``struct_lower``       -- opt-in (``GRAPHAX_STRUCT_LOWER``): structure-
                                  lowering contraction planner.
      5. ``both_implicit_fold`` -- both contracted dims implicit: analytic
                                  scale-by-N folded into ``scalar_mult``.
      6. ``elemental``          -- structured (diagonal/block/banded) kernels,
                                  active only under ``approx_active()``.
      7. ``densify``            -- non-zero ``fill_value`` OR an implicit-block
                                  contraction, when ``_densify_is_safe``:
                                  materialize via ``dense_for_matmul`` +
                                  ``dot_general`` (tiled assumes implicit
                                  positions are zero, wrong when fill != 0).
                                  Runs LATE, after the structured paths above.
      8. ``tiled``              -- general LCM/topology/finalize pipeline; the
                                  sparsity-preserving fallback for everything else.

    Scalar @ scalar (both 0-rank SparseTensors) is rejected — use
    ``lhs * rhs`` (elementwise) instead. Vertex elimination routes scalar
    edges through ``*`` since core-v2.

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
    _record_path(None)
    if not _is_sparse(lhs) and not _is_sparse(rhs):
        _record_path("dense_dense")
        out = jnp.matmul(lhs, rhs)
        if count:
            return out, _compute_matmul_count(lhs, rhs, out)
        return out
    lhs, rhs = _normalize_inputs(lhs, rhs)
    # Two 0-rank (scalar) SparseTensors: the contraction is a scalar product =
    # an ELEMENTWISE multiply (scalar . X == scale). This is what the AGGREGATION
    # step needs when a Compress-reduced / seed-vertex scalar edge contracts
    # another scalar edge (the --seed-vertices sentinel driver). The chain-rule
    # site in core.py guards this too, but routes that leak through (the folded
    # both-implicit reduction, a merge of two scalar edges) land here; do the
    # mathematically-correct multiply rather than raise. Set
    # GRAPHAX_SEED_VERTICES_SCALAR_MM=0 to restore the legacy raise.
    if (
        getattr(lhs, "out_dims", ()) == ()
        and getattr(lhs, "primal_dims", ()) == ()
        and getattr(rhs, "out_dims", ()) == ()
        and getattr(rhs, "primal_dims", ()) == ()
    ):
        if _SEED_SCALAR_MM:
            out = lhs * rhs
            _record_path("scalar_elementwise")
            if count:
                return out, (0, 1, 0)
            return out
        raise ValueError(
            "matmul of two 0-rank SparseTensors is not supported; "
            "use ``lhs * rhs`` (elementwise) instead"
        )
    # New einsum general path (GRAPHAX_EINSUM_GENERAL, default OFF): the
    # sparsity-retaining contraction. Tried FIRST after normalize/scalar, before
    # the elemental cascade and the tiled path. Emits ONE einsum over physical
    # axes, keeps implicit dims / block pairs / val=None symbolic, and folds a
    # both-implicit contraction into scalar_mult. Returns None on any case it
    # cannot yet prove correct, falling through to the EXISTING path UNCHANGED —
    # so nothing regresses while this path is built out. Gated on
    # ``approx_active()`` inside, so the EXACT-AD path never enters and stays
    # byte-identical whether the flag is on or off.
    if _einsum_general_enabled():
        _ein = _einsum_matmul_general(lhs, rhs, count=count)
        if _ein is not None:
            _record_path("einsum_general")
            return _ein
    # Structure-lowering layer (GRAPHAX_STRUCT_LOWER, default OFF): compile
    # the minimal physical computation for a structured contraction (ONE
    # einsum over physical axes only) and build the output structure
    # SYMBOLICALLY — a free implicit dim stays implicit, a surviving
    # block-diagonal pair stays a pair, val=None stays val=None. Returns None
    # on any case without a rule; every miss is counted in
    # ``lower.matmul.LOWER_STATS`` (no silent behavior change). Gated on
    # ``approx_active()`` inside, so the EXACT-AD path never enters here and
    # stays byte-identical whether the env flag is on or off.
    if _struct_lower_enabled():
        from graphax.sparse.lower.matmul import try_lower_matmul

        _low = try_lower_matmul(lhs, rhs, count=count)
        if _low is not None:
            _record_path("struct_lower")
            return _low
    # Elemental fast path (Phase: bridge-cse): route a STRUCTURED contraction
    # (block-diagonal / implicit / compressed contracted dims) through the
    # composed elemental kernels. Returns None for a pure-dense contraction, so
    # the EXACT-AD (transforms=()) edge never enters here and stays byte-
    # identical. Built with the canonical output-id convention so a downstream
    # multi-edge / all-vertices contraction aligns by id.
    # Both-implicit contracting pair -> analytic scale-by-N folded into
    # scalar_mult (no dot_general over the broadcast axis, no materialization).
    if _KEEP_BLOCKDIAG_MM:
        _folded = _fold_both_implicit(lhs, rhs, count)
        if _folded is not None:
            _record_path("both_implicit_fold")
            return _folded

    from graphax.sparse.elemental.dispatch import try_elemental_matmul

    _elem = try_elemental_matmul(lhs, rhs, count=count)
    if _elem is not None:
        _record_path("elemental")
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
    rhs_out_dims, rhs_primal_dims, rhs_id_offset = _align_tensor_ids(lhs, rhs)
    rhs_dims = rhs_out_dims + rhs_primal_dims
    pairs = _build_matmul_topology(lhs, rhs_out_dims, rhs_primal_dims, rhs_id_offset)
    ctx = Ctx(lhs=lhs, rhs=rhs, pairs=pairs, rhs_id_offset=rhs_id_offset)
    _record_path("tiled")
    out = _execute_tiled(ctx, rhs_dims)
    if count:
        return out, _compute_matmul_count(lhs, rhs, out)
    return out


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
