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

from graphax.sparse.indexes import DenseIndex, Index, SparseIndex

from .dense import dense_for_matmul
from .layout import generate_block_permutation, generate_grouped_permutation
from .utils import (
    _arr2st,
    _copy,
    _is_sparse,
    _is_zero_fill,
    _materialize_compressed,
    _prepare_physical_array,
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


# --- Topology resolution ---------------------------------------------------
def _dim_vals(dim, is_outer=False):
    """(length, axis) for one logical axis of a Index. is_outer=True picks sparse-pair length."""
    if not dim:
        return 1, None
    if isinstance(dim, DenseIndex):
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
        and isinstance(dim, SparseIndex)
        and isinstance(sibling, SparseIndex)
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
    if lout and rout and lout.logical_size != rout.logical_size:
        raise ValueError(f"Batch mismatch: {lout.id} vs {rout.id}")
    if lprimal and rprimal and lprimal.logical_size != rprimal.logical_size:
        raise ValueError(f"Batch mismatch: {lprimal.id} vs {rprimal.id}")
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
        if isinstance(d, SparseIndex):
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
        if not isinstance(d, SparseIndex):
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
        if isinstance(lp, SparseIndex)
        else None
    )
    rp = (
        rhs_primal_map.get(getattr(ro, "other_id", -1))
        if isinstance(ro, SparseIndex)
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
            "contract",
            getattr(lp, "block_size", getattr(lp, "size", 1)),
            *_full_pair_data(lo, lp, ro, rp, swap_rhs=True),
        ),
        lhs_ids,
        rhs_ids,
    )


def _resolve_broadcast_topos(lhs_topos, rhs_topos, offset):
    def find_match(lout, lprimal, candidates):
        for i, (rout, rprimal) in enumerate(candidates):
            if lout and rout and lout.id == rout.id - offset:
                return i
            if lprimal and rprimal and lprimal.id == rprimal.id - offset:
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


def _build_matmul_topology(lhs, rhs_out_dims, rhs_primal_dims, rhs_id_offset):
    lhs_out_map = {d.id: d for d in lhs.out_dims}
    rhs_primal_map = {d.id: d for d in rhs_primal_dims}
    rhs_dims = rhs_out_dims + rhs_primal_dims
    n_contract = min(len(lhs.primal_dims), len(rhs_out_dims))
    lhs_contract = list(lhs.primal_dims[-n_contract:] if n_contract > 0 else [])
    rhs_contract = list(rhs_out_dims[-n_contract:] if n_contract > 0 else [])
    pairs, processed_l, processed_r = [], set(), set()
    for lp, ro in zip(lhs_contract, rhs_contract):
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


def _execute_block_sparse_contraction(lhs_val, rhs_val, pairs):
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
    return grid, shared, final_lhs_lens, final_rhs_lens, scalar


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
    return SparseIndex(
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
                        if isinstance(d, SparseIndex)
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
        if isinstance(d, SparseIndex):
            kw["other_id"] = id_map.get(d.other_id, d.other_id)
        return replace(d, **kw)

    final_out = tuple(finalize(d, i) for i, d in enumerate(final_out))
    n_out = len(final_out)
    final_primal = tuple(finalize(d, n_out + i) for i, d in enumerate(final_primal))
    has_val = any(d.axis is not None for d in final_out + final_primal) or any(
        isinstance(d, SparseIndex) and d.block_axis is not None
        for d in final_out + final_primal
    )
    final_mult = ctx.lhs.scalar_mult * ctx.rhs.scalar_mult * res.scalar_mult
    if not has_val and values is not None and values.size == 1:
        final_mult = final_mult * jnp.squeeze(values)
        values = None
    # Lazy compressed form: when the output is meta-block-diagonal-square AND
    # a tighter BlockBanded (smaller B + bandwidth w) exists, repack the val
    # into ``compressed_val=BlockBanded`` to drop the structured-zero portion
    # of the meta-block-diagonal storage.
    cv_st = _try_compressed_block_banded(
        ctx,
        final_out,
        final_primal,
        values,
        final_mult,
    )
    if cv_st is not None:
        return cv_st
    out_dtype = values.dtype if values is not None else jnp.asarray(final_mult).dtype
    # transforms intentionally not propagated through matmul; callers in
    # core.py unload pre/post transforms before the matmul and reattach
    # fresh ones to the result.
    return SparseTensor(
        final_out,
        final_primal,
        values,
        scalar_mult=jnp.asarray(final_mult).astype(out_dtype),
        sort_val=True,
        zero_fill=True,
    )


def _block_banded_geometry(
    M_eager: int,
    B_eager: int,
    B_x_h: int,
    B_x_w: int,
    B_y_h: int,
    B_y_w: int,
) -> tuple[int, int, int] | None:
    """Compute the BlockBanded repacking geometry for an output that's
    written in eager ``(M_eager, B_eager, B_eager)`` form. Returns
    ``(M_new, B_new, w_band)`` or ``None`` if no granularity gain is
    possible. ``B_new`` is the natural square sub-block size that contains
    each output cell (``max(B_x_h, B_y_w)``); ``M_new = M_eager *
    (B_eager / B_new)``; ``w_band`` is the half-bandwidth (max ``|a-b|``
    among sub-block index pairs whose contraction ranges overlap).
    """
    B_new = builtins.max(B_x_h, B_y_w)
    if B_eager % B_new != 0 or B_new == B_eager:
        return None  # Either can't repack or no granularity gain.
    M_new = M_eager * (B_eager // B_new)
    # Sub-block (a, b) is non-zero iff x's contraction range from rows
    # [a·B_new, a·B_new+B_new) overlaps y's contraction range from cols
    # [b·B_new, b·B_new+B_new). Enumerate one period; track max |a-b|.
    w_band = 0
    for a in range(M_new):
        x_lo = ((a * B_new) // B_x_h) * B_x_w
        x_hi = (((a * B_new + B_new - 1) // B_x_h) + 1) * B_x_w
        for b in range(M_new):
            y_lo = ((b * B_new) // B_y_h) * B_y_w
            y_hi = (((b * B_new + B_new - 1) // B_y_h) + 1) * B_y_w
            if x_lo < y_hi and y_lo < x_hi:
                w_band = builtins.max(w_band, abs(b - a))
    W = 2 * w_band + 1
    if M_new * W * B_new * B_new >= M_eager * B_eager * B_eager:
        return None  # Eager is already as tight or tighter.
    return M_new, B_new, w_band


def _gather_banded_data(
    values: Array, M_eager: int, B_eager: int, M_new: int, B_new: int, w_band: int
) -> Array:
    """Extract in-band ``(B_new, B_new)`` sub-blocks from the eager
    ``(M_eager, B_eager, B_eager)`` val into ``(M_new, W, B_new, B_new)``
    data — one fused gather XLA folds with the matmul output write.

    Output sub-block (a, w_idx) sits at column ``b = a + (w_idx - w_band)``
    in the banded layout. Out-of-band positions are zeroed via
    ``jnp.where``; only sub-blocks that share an eager meta-block are
    physically present (the eager output is meta-block-diagonal).
    """
    K = B_eager // B_new
    W = 2 * w_band + 1
    v5 = values.reshape(M_eager, K, B_new, K, B_new)
    a_idx = np.arange(M_new)[:, None]  # (M_new, 1)
    w_idx_arr = np.arange(W)[None, :]  # (1, W)
    b_idx = a_idx + (w_idx_arr - w_band)  # (M_new, W)
    in_band = (b_idx >= 0) & (b_idx < M_new)
    eager_meta = a_idx // K  # (M_new, 1)
    sub_a = a_idx % K  # (M_new, 1)
    sub_b = np.where(in_band, b_idx % K, 0)  # (M_new, W) — clipped
    same_meta = (b_idx // K) == eager_meta
    keep = in_band & same_meta
    em = jnp.broadcast_to(jnp.asarray(eager_meta), (M_new, W))
    sa = jnp.broadcast_to(jnp.asarray(sub_a), (M_new, W))
    sb = jnp.asarray(sub_b)
    # ``v5[em, sa, :, sb, :]`` returns ``Array`` for our index shape, but
    # the type stub annotates a wider ``Array | tuple[Array, ...]``. Cast
    # at the boundary so downstream consumers see a single tensor.
    from typing import cast

    data = cast(Array, v5[em, sa, :, sb, :])  # (M_new, W, B_new, B_new)
    return jnp.where(
        jnp.asarray(keep)[..., None, None], data, jnp.zeros((), dtype=data.dtype)
    )


def _try_compressed_block_banded(ctx, final_out, final_primal, values, final_mult):
    """Repack a meta-block-diagonal-square matmul output ``(M, B_eager,
    B_eager)`` into a tighter ``BlockBanded(M_new, w, B_new, B_new)`` form
    when the input geometry produces a narrow band. For nearly-coprime
    block ratios (e.g. 5/11) ``w = 1`` (2-3 cells visible per row) and
    BlockBanded is strictly tighter; for divisor / equal blocks the eager
    form is already optimal and we fall through.

    Conditions: 2-D single-sparse-pair on both inputs, square output
    (``o.block_size == p.block_size``, ``o.size == p.size``), eager val
    in canonical ``(M, B, B)`` shape, and the geometry must yield
    ``M_new·W·B_new² < M_eager·B_eager²``.
    """
    if values is None:
        return None
    if len(final_out) != 1 or len(final_primal) != 1:
        return None
    o, p = final_out[0], final_primal[0]
    if not isinstance(o, SparseIndex) or not isinstance(p, SparseIndex):
        return None
    if o.block_size != p.block_size or o.size != p.size:
        return None  # Not meta-block-diagonal-square.
    if o.block_size is None:  # narrow Optional for the type checker
        return None
    M_eager, B_eager = o.size, o.block_size
    if values.shape != (M_eager, B_eager, B_eager):
        return None  # Output wasn't written in (M, B, B) form.
    lhs, rhs = ctx.lhs, ctx.rhs
    if len(lhs.dims) != 2 or len(rhs.dims) != 2:
        return None
    if not all(isinstance(d, SparseIndex) for d in (*lhs.dims, *rhs.dims)):
        return None
    geom = _block_banded_geometry(
        M_eager,
        B_eager,
        B_x_h=lhs.out_dims[0].block_size or 1,
        B_x_w=lhs.primal_dims[0].block_size or 1,
        B_y_h=rhs.out_dims[0].block_size or 1,
        B_y_w=rhs.primal_dims[0].block_size or 1,
    )
    if geom is None:
        return None
    M_new, B_new, w_band = geom
    data = _gather_banded_data(values, M_eager, B_eager, M_new, B_new, w_band)

    from graphax.sparse.tensor import SparseTensor

    from .block_storage import BlockBanded

    bb = BlockBanded(data=data, fill_value=jnp.zeros((), dtype=values.dtype))
    # ``BlockBanded(w>0)`` can't be expressed as a single sparse pair, so the
    # SparseTensor wraps it behind two full-size ``DenseIndex``s — same dim
    # structure ``from_compressed`` uses for the BlockBanded fallback path.
    full = M_new * B_new
    # transforms intentionally not propagated through matmul; callers in
    # core.py unload pre/post transforms before the matmul and reattach
    # fresh ones to the result.
    return SparseTensor(
        (DenseIndex(o.id, full, axis=0),),
        (DenseIndex(p.id, full, axis=1),),
        val=None,
        compressed_val=bb,
        scalar_mult=jnp.asarray(final_mult).astype(values.dtype),
        sort_val=False,
        check_consistency=False,
        zero_fill=True,
    )


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

    n_contract = min(len(lhs.primal_dims), len(rhs.out_dims))
    n_lhs_dims = len(lhs.dims)
    n_lhs_out = len(lhs.out_dims)
    n_rhs_out = len(rhs.out_dims)

    lhs_dense, rhs_dense = dense_for_matmul(lhs), dense_for_matmul(rhs)

    # Contraction axes: pair the last ``n_contract`` of ``lhs.primal_dims``
    # with the last ``n_contract`` of ``rhs.out_dims`` positionally — same
    # convention as the tiled path's ``_resolve_contract_pair``. The
    # dispatcher's ``_densify_is_safe`` gate has already verified the paired
    # sizes match, so ``dot_general`` accepts this directly. We deliberately
    # don't try id-based pairing across operands here: lhs and rhs use
    # independent id spaces at this point (``_align_tensor_ids`` only runs
    # on the tiled path), so matching by raw id is meaningless and can pair
    # mismatched-size dims together.
    lhs_contract: list[int] = []
    rhs_contract: list[int] = []
    if n_contract > 0:
        n_lhs_pri = len(lhs.primal_dims)
        lhs_contract = [n_lhs_out + p for p in range(n_lhs_pri - n_contract, n_lhs_pri)]
        rhs_contract = list(range(n_rhs_out - n_contract, n_rhs_out))

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
        fill_value=jnp.array(0, dtype=result.dtype),
        sort_val=False,
        check_consistency=False,
        zero_fill=True,
    )


# --- Fast paths (tried in priority order by ``matmul()``) ----------------
def _try_aligned_pair_matmul(lhs, rhs):
    """Aligned single-pair matmul fast path.

    Both ``lhs`` and ``rhs`` are 2-D ``SparseTensor``s with one sparse pair on
    each side. The contracting axis is fully aligned (same outer ``N`` *and*
    same block size on ``lhs.primal[0]`` / ``rhs.out[0]``), so the operation
    reduces to a batched matmul ``lhs.val @ rhs.val`` over the shared outer
    axis. Output: a single sparse pair of size ``N`` with block
    ``(lhs.out[0].block_size, rhs.primal[0].block_size)`` and
    ``val.shape = (N, B_a_h, B_b_w)``.

    Saves ~30% HLO and a couple of tracing-time reshape/transpose ops vs the
    full tiled algorithm — the contraction is already canonical, so all the
    LCM / topology work in ``_execute_block_sparse_contraction`` is wasted
    effort here. Returns ``None`` when conditions don't apply.
    """
    if not _is_zero_fill(lhs) or not _is_zero_fill(rhs):
        return None  # implicit positions contribute under non-zero fill
    if lhs.val is None or rhs.val is None:
        return None
    if len(lhs.dims) != 2 or len(rhs.dims) != 2:
        return None
    a_o, a_p = lhs.out_dims[0], lhs.primal_dims[0]
    b_o, b_p = rhs.out_dims[0], rhs.primal_dims[0]
    if not all(isinstance(d, SparseIndex) for d in (a_o, a_p, b_o, b_p)):
        return None
    # Each side: out_dim and primal_dim form a sibling pair.
    if a_o.other_id != a_p.id or a_p.other_id != a_o.id:
        return None
    if b_o.other_id != b_p.id or b_p.other_id != b_o.id:
        return None
    # Aligned contraction: same outer count *and* same block size on both
    # sides of the contraction.
    if a_p.size != b_o.size:
        return None
    if (a_p.block_size or 1) != (b_o.block_size or 1):
        return None
    # All axiss should be 0 (the sparse-pair outer) — anything else means
    # a less-canonical layout and we'd need to permute first.
    if any(d.axis != 0 for d in (a_o, a_p, b_o, b_p)):
        return None
    # Need actual block axes on the val (not just the implicit sib outer).
    # When both sides are no-block sib pairs (val is 1-D), ``jnp.matmul``
    # computes an inner product, which is wrong here — the operation is
    # diagonal × diagonal = diagonal. Defer to the general dot_general
    # fast path which handles that correctly.
    if a_p.block_size is None or b_o.block_size is None:
        return None
    # All four sib-pair sides must have their block axes materialized in val
    # (block_axis set). When any side has block_axis=None the block axis
    # is an implicit broadcast — ``jnp.matmul`` would see a smaller rank than
    # expected and pick the wrong contracting axis. Defer to the tiled path,
    # which handles broadcast / unmaterialized block dims correctly via
    # ``_prepare_physical_array``.
    if any(
        d.block_axis is None for d in (a_o, a_p, b_o, b_p) if d.block_size is not None
    ):
        return None

    N = a_o.size
    B_a_h = a_o.block_size or 1
    B_b_w = b_p.block_size or 1

    # ``a.val`` is ``(N, B_a_h, B_a_w)``; ``b.val`` is ``(N, B_a_w, B_b_w)``.
    # ``jnp.matmul`` batches over leading axes, so this is the right shape.
    # Defer scalar_mult to the output's ``scalar_mult`` field — saves two
    # elementwise multiplies that XLA can't always fold into the matmul kernel.
    result = jnp.matmul(lhs.val, rhs.val)

    # Build output dim ids: keep ``lhs``'s out_dim id; allocate a fresh primal
    # id (mirroring what ``_align_tensor_ids`` would do for the rhs side).
    out_id = a_o.id
    primal_id = builtins.max(a_o.id, a_p.id, b_o.id, b_p.id) + 1
    new_out = SparseIndex(
        out_id, N, axis=0, other_id=primal_id, block_size=B_a_h, block_axis=1
    )
    new_primal = SparseIndex(
        primal_id, N, axis=0, other_id=out_id, block_size=B_b_w, block_axis=2
    )
    s_mult = (lhs.scalar_mult * rhs.scalar_mult).astype(result.dtype)
    from graphax.sparse.tensor import SparseTensor

    # transforms intentionally not propagated through matmul; callers in
    # core.py unload pre/post transforms before the matmul and reattach
    # fresh ones to the result.
    return SparseTensor(
        (new_out,),
        (new_primal,),
        result,
        scalar_mult=s_mult,
        sort_val=False,
        check_consistency=False,
        zero_fill=True,
    )


def _dot_general_fast_path_eligible(ctx: Ctx) -> bool:
    """True iff ``ctx`` describes a matmul whose pairs are
    ``contract`` / ``batch_*`` / non-sparse ``spatial_*`` with no LCM
    mismatch and whose val arrays are fully materialized (no implicit /
    broadcast axes, no leftovers). These are the cases where ``dot_general``
    on the raw val arrays produces the same result as the tiled algorithm
    after XLA folds the reshape/transpose chain.
    """
    if ctx.lhs.val is None or ctx.rhs.val is None:
        return False
    pairs = ctx.pairs
    if not pairs:
        return False
    for p in pairs:
        if p.pairing_type.startswith("spatial_sparse_"):
            return False  # sib-pair on one side; size-1 placeholder layout misses it
        if p.lhs.outer_len != p.rhs.outer_len:
            return False  # LCM mismatch — defer to tiled
        if p.pairing_type == "contract":
            if p.lhs.shared_block_len != p.rhs.block_len:
                return False  # contract sizes must match exactly
        elif p.pairing_type.startswith("batch_"):
            if p.lhs.block_len != 1 or p.lhs.shared_block_len != 1:
                return False
            if p.rhs.block_len != 1 or p.rhs.shared_block_len != 1:
                return False
        # Implicit / unmaterialized dims: defer (the tiled broadcast path
        # handles them; ``dot_general`` here would see lower-rank vals).
        for side in (p.lhs, p.rhs):
            if (
                side.outer_len > 1
                and side.outer_axis is None
                and (
                    # contract sib pairs are allowed if the partner carries axis.
                    p.pairing_type != "contract"
                    or side.dim is None
                    or side.shared_dim is None
                )
            ):
                return False
            if side.block_len > 1 and side.block_axis is None:
                return False
            if side.shared_block_len > 1 and side.shared_block_axis is None:
                return False

    # No leftover val axes (every val axis must be referenced by some pair).
    def _used(side_attr):
        return {
            v
            for p in pairs
            for v in (
                getattr(p, side_attr).outer_axis,
                getattr(p, side_attr).block_axis,
                getattr(p, side_attr).shared_block_axis,
            )
            if v is not None
        }

    if _used("lhs") != set(range(ctx.lhs.val.ndim)):
        return False
    if _used("rhs") != set(range(ctx.rhs.val.ndim)):
        return False
    return True


def _build_dot_general_axes(
    pairs: list[Pair],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]] | None:
    """Collect (batch_l, batch_r, contract_l, contract_r) for the
    ``dot_general`` call, sorted by ``lhs`` axis so the result lays out
    axes in the same order as ``lhs.val`` — matches what the manual
    references emit and lets XLA pick the canonical kernel layout.

    Each batch entry tracks ``(lhs_axis, rhs_axis, pair_idx)``; each
    contract entry tracks ``(lhs_axis, rhs_axis)``. Returns ``None`` if
    any pair-side has missing axiss for a non-trivial axis.
    """
    batch_entries: list[tuple[int, int, int]] = []
    contract_entries: list[tuple[int, int]] = []
    for i, p in enumerate(pairs):
        if p.lhs.outer_len > 1:
            if p.lhs.outer_axis is None or p.rhs.outer_axis is None:
                return None
            batch_entries.append((p.lhs.outer_axis, p.rhs.outer_axis, i))
        if p.pairing_type == "contract" and p.lhs.shared_block_len > 1:
            if p.lhs.shared_block_axis is None or p.rhs.block_axis is None:
                return None
            contract_entries.append((p.lhs.shared_block_axis, p.rhs.block_axis))
    batch_entries.sort()
    contract_entries.sort()
    if not batch_entries and not contract_entries:
        return None  # pure outer product — uncommon, bail to tiled
    return batch_entries, contract_entries


def _reshape_to_canonical_layout(
    result: Array,
    pairs: list[Pair],
    batch_entries: list[tuple[int, int, int]],
    contract_l: list[int],
    contract_r: list[int],
    lhs_ndim: int,
    rhs_ndim: int,
) -> Array | None:
    """Permute + reshape ``result`` from ``dot_general``'s output layout
    (``[batch_l-order, lhs-kept-sorted, rhs-kept-sorted]``) to the
    pre-finalize layout ``_build_output_tensor`` expects:
    ``[*shared_factors, *lhs_factors, *rhs_factors]`` with size-1
    placeholders for axes absent from the dot_general result.

    Returns the reshaped tensor or ``None`` if the result rank doesn't
    match expectations (sanity check).
    """
    batch_l = [e[0] for e in batch_entries]
    lhs_kept_v = sorted(set(range(lhs_ndim)) - set(contract_l) - set(batch_l))
    rhs_kept_v = sorted(
        set(range(rhs_ndim)) - set(contract_r) - {e[1] for e in batch_entries}
    )
    lhs_kept_pos = {ax: pos for pos, ax in enumerate(lhs_kept_v)}
    rhs_kept_pos = {ax: pos for pos, ax in enumerate(rhs_kept_v)}
    n_batch, n_lhs_kept = len(batch_entries), len(lhs_kept_v)
    N = len(pairs)
    # Map each pair to its position in the dot_general result (or None for
    # absent axes that need a size-1 placeholder).
    pair_outer_ax: list[int | None] = [None] * N
    pair_lhs_block_ax: list[int | None] = [None] * N
    pair_rhs_shared_ax: list[int | None] = [None] * N
    for k, (_, _, pi) in enumerate(batch_entries):
        pair_outer_ax[pi] = k
    for i, p in enumerate(pairs):
        if (
            p.pairing_type in ("contract", "spatial_out_lhs")
            and p.lhs.block_len > 1
            and p.lhs.block_axis is not None
        ):
            pair_lhs_block_ax[i] = n_batch + lhs_kept_pos[p.lhs.block_axis]
        if (
            p.pairing_type in ("contract", "spatial_primal_rhs")
            and p.rhs.shared_block_len > 1
            and p.rhs.shared_block_axis is not None
        ):
            pair_rhs_shared_ax[i] = (
                n_batch + n_lhs_kept + rhs_kept_pos[p.rhs.shared_block_axis]
            )
    perm = [
        a
        for a in (pair_outer_ax + pair_lhs_block_ax + pair_rhs_shared_ax)
        if a is not None
    ]
    if len(perm) != result.ndim:
        return None
    if perm != list(range(result.ndim)):
        result = jnp.transpose(result, perm)
    target_shape = (
        [p.lhs.outer_len for p in pairs]
        + [p.lhs.block_len for p in pairs]
        + [p.rhs.shared_block_len for p in pairs]
    )
    if tuple(target_shape) != result.shape:
        result = result.reshape(target_shape)
    return result


def _try_dot_general_fast_path(ctx, rhs_dims):
    """Direct ``dot_general`` bypass for fully-aligned matmul.

    Conditions encoded in ``_dot_general_fast_path_eligible`` (the validation)
    plus the axis-collection check inside ``_build_dot_general_axes``:
      - All pairs are ``contract`` / ``batch_*`` / non-sparse ``spatial_*``.
      - No LCM mismatch (matched outer sizes per pair).
      - Both vals fully materialized; no leftover or implicit axes.

    When eligible, emits a single ``dot_general`` on the raw val arrays —
    same path the ``manual_13`` / ``manual_15`` / ``manual_16`` /
    ``manual_19`` references take. The result is reshaped into the
    pre-finalize layout that ``_build_output_tensor`` expects.

    ``_build_output_tensor`` accumulates ``lhs.scalar_mult * rhs.scalar_mult
    * res.scalar_mult`` for the output, so we leave the inputs unscaled and
    pass ``res.scalar_mult=1`` — XLA may not fold pre-multiplied scalars
    into the dot_general kernel.
    """
    if not _dot_general_fast_path_eligible(ctx):
        return None
    axes = _build_dot_general_axes(ctx.pairs)
    if axes is None:
        return None
    batch_entries, contract_entries = axes
    contract_l = [e[0] for e in contract_entries]
    contract_r = [e[1] for e in contract_entries]
    batch_l = [e[0] for e in batch_entries]
    batch_r = [e[1] for e in batch_entries]
    result = jax.lax.dot_general(
        ctx.lhs.val,
        ctx.rhs.val,
        ((tuple(contract_l), tuple(contract_r)), (tuple(batch_l), tuple(batch_r))),
    )
    result = _reshape_to_canonical_layout(
        result,
        ctx.pairs,
        batch_entries,
        contract_l,
        contract_r,
        ctx.lhs.val.ndim,
        ctx.rhs.val.ndim,
    )
    if result is None:
        return None
    res = CRes(
        grid=result,
        shared_factors=[p.lhs.outer_len for p in ctx.pairs],
        lhs_block_lens=[p.lhs.block_len for p in ctx.pairs],
        rhs_block_lens=[p.rhs.shared_block_len for p in ctx.pairs],
        scalar_mult=1.0,
    )
    return _build_output_tensor(ctx, rhs_dims, res)


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
def _normalize_inputs(lhs, rhs):
    """Convert array operands to ``SparseTensor`` and materialize any
    ``compressed_val`` storage. Pure shape / structure prep — no actual
    matmul work happens here. Materializing compressed storage is fused
    into the consuming kernel by XLA (SMEM, not HBM)."""
    if not _is_sparse(lhs):
        lhs = _arr2st(lhs, out_ndim=lhs.ndim - len(rhs.out_dims))
    if not _is_sparse(rhs):
        rhs = _arr2st(rhs, out_ndim=len(lhs.primal_dims))
    if getattr(lhs, "compressed_val", None) is not None:
        lhs = _copy(lhs, val=_materialize_compressed(lhs))
    if getattr(rhs, "compressed_val", None) is not None:
        rhs = _copy(rhs, val=_materialize_compressed(rhs))
    return lhs, rhs


def _execute_tiled(ctx, rhs_dims):
    """Fallback: full tiled algorithm. Handles every case the fast paths
    bail on, including LCM-mismatched outer sizes, spatial sparse pairs,
    and broadcast / unmaterialized val axes."""
    lhs_val, rhs_val = _val_or_one(ctx.lhs), _val_or_one(ctx.rhs)
    lhs_val, rhs_val = _prepare_physical_arrays(lhs_val, rhs_val, ctx.pairs)
    grid, shared, lhs_lens, rhs_lens, scalar = _execute_block_sparse_contraction(
        lhs_val, rhs_val, ctx.pairs
    )
    res = CRes(
        grid=grid,
        shared_factors=shared,
        lhs_block_lens=lhs_lens,
        rhs_block_lens=rhs_lens,
        scalar_mult=scalar,
    )
    return _build_output_tensor(ctx, rhs_dims, res)


def matmul(lhs, rhs, count: bool = False):
    """Sparse matmul dispatcher. Tries fast paths in priority order, falls
    back to the tiled algorithm. Each fast path is a function that returns
    ``None`` when its conditions don't apply.

    Dispatch order (first applicable wins):
      1. ``dense_dense``      — both operands are arrays (no SparseTensor).
      2. ``scalar``           — both operands are 0-rank SparseTensors (no
                                contracting axes). Vertex elimination
                                composes scalar edges this way.
      3. ``densify``          — non-zero ``fill_value`` on either side
                                *and* contracting dim sizes line up
                                positionally (``_densify_is_safe``). The
                                tiled path assumes implicit positions are
                                zero, which is wrong when ``fill ≠ 0``;
                                we materialize via ``dense_for_matmul`` +
                                ``dot_general`` instead. When the safety
                                check fails (graphax's AD can produce
                                permuted dim orders), we skip this and let
                                the tiled path handle it via id-aware
                                topology resolution.
      4. ``aligned_pair``     — 2-D single-sparse-pair tensors with matching
                                contracting block size; emits ``jnp.matmul``.
      5. ``dot_general_fast`` — fully-aligned multi-pair: emits ``dot_general``
                                directly on the raw val arrays. Same shape as
                                the manual_13/15/16/19 references.
      6. ``tiled``            — full LCM/topology/finalize pipeline; handles
                                everything else.

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
    # Scalar @ scalar — both sides have no contracting axes. Vertex
    # elimination composes scalar edges this way (e.g. tan'(x) @ (-1)·sin'(x))
    # and the matmul pipeline below assumes at least one pair to contract on.
    if (
        getattr(lhs, "out_dims", ()) == ()
        and getattr(lhs, "primal_dims", ()) == ()
        and getattr(rhs, "out_dims", ()) == ()
        and getattr(rhs, "primal_dims", ()) == ()
    ):
        _record_path("scalar")
        out = _scalar_matmul(lhs, rhs)
        if count:
            return out, _compute_matmul_count(lhs, rhs, out)
        return out
    # Densify path: handles non-zero ``fill_value`` correctly (the tiled
    # path's contraction assumes implicit positions are zero, which is wrong
    # for non-zero fills). Only safe when contracting dim sizes pair up
    # positionally — graphax's AD pipeline can produce permuted dim orders
    # that need the tiled path's id-aware topology resolver. ``_is_zero_fill``
    # checks the static ``_zero_fill`` flag (set at ``SparseTensor`` ctor time)
    # so this stays jit-friendly.
    has_nonzero_fill = not _is_zero_fill(lhs) or not _is_zero_fill(rhs)
    if has_nonzero_fill:
        if _densify_is_safe(lhs, rhs):
            _record_path("densify")
            out = _matmul_via_densify(lhs, rhs)
            if count:
                return out, _compute_matmul_count(lhs, rhs, out)
            return out
        # Tiled / aligned-pair / dot_general fast paths assume zero fill;
        # falling through silently mislabels the result as ``zero_fill=True``.
        raise NotImplementedError(
            "matmul of operands with non-zero fill_value and incompatible "
            "logical sizes is not supported; reorder dim ids first"
        )
    aligned = _try_aligned_pair_matmul(lhs, rhs)
    if aligned is not None:
        _record_path("aligned_pair")
        if count:
            return aligned, _compute_matmul_count(lhs, rhs, aligned)
        return aligned
    # Below this point we need the topology pairs computed; both remaining
    # paths share that work.
    rhs_out_dims, rhs_primal_dims, rhs_id_offset = _align_tensor_ids(lhs, rhs)
    rhs_dims = rhs_out_dims + rhs_primal_dims
    pairs = _build_matmul_topology(lhs, rhs_out_dims, rhs_primal_dims, rhs_id_offset)
    ctx = Ctx(lhs=lhs, rhs=rhs, pairs=pairs, rhs_id_offset=rhs_id_offset)
    fast = _try_dot_general_fast_path(ctx, rhs_dims)
    if fast is not None:
        _record_path("dot_general_fast")
        if count:
            return fast, _compute_matmul_count(lhs, rhs, fast)
        return fast
    _record_path("tiled")
    out = _execute_tiled(ctx, rhs_dims)
    if count:
        return out, _compute_matmul_count(lhs, rhs, out)
    return out


def _densify_is_safe(lhs, rhs) -> bool:
    """``_matmul_via_densify`` pairs contracting dims positionally (last
    n_contract of lhs.primal ↔ last n_contract of rhs.out) — that's all
    ``dot_general`` accepts directly. Graphax's AD pipeline can produce
    operands whose dim *orders* don't match (a transposed Jacobian feeds a
    sub-elimination), so positional pairing yields a shape mismatch and
    blows up. Detect that ahead of time and let the caller fall through to
    the tiled path, which permutes dims by id through ``_align_tensor_ids``.
    """
    n_contract = min(len(lhs.primal_dims), len(rhs.out_dims))
    if n_contract == 0:
        return True
    lhs_pri = lhs.primal_dims[-n_contract:]
    rhs_out = rhs.out_dims[-n_contract:]
    return all(
        int(l.logical_size) == int(r.logical_size) for l, r in zip(lhs_pri, rhs_out)
    )


def _scalar_matmul(lhs, rhs):
    """``SparseTensor((), (), x) @ SparseTensor((), (), y)`` => ``x*y``.

    Vertex-elimination composes scalar edges via this path. The general
    matmul pipeline assumes at least one contracting pair, so we short-
    circuit here. ``val=None`` denotes structural-identity *value* (the
    edge has no per-element data), but ``scalar_mult`` is still a real
    multiplier on the implicit identity and must compose through the
    matmul (e.g. ``-1 * sin'(x)`` ∘ another scalar Jacobian)."""
    from graphax.sparse.tensor import SparseTensor

    lv = lhs.val
    rv = rhs.val
    composed_mult = lhs.scalar_mult * rhs.scalar_mult
    if lv is None and rv is None:
        new_val = None
        new_mult = composed_mult
    elif lv is None:
        new_val = rv * composed_mult
        new_mult = jnp.array(1.0, dtype=new_val.dtype)
    elif rv is None:
        new_val = lv * composed_mult
        new_mult = jnp.array(1.0, dtype=new_val.dtype)
    else:
        new_val = (lv * lhs.scalar_mult) * (rv * rhs.scalar_mult)
        new_mult = jnp.array(1.0, dtype=new_val.dtype)

    fill = lhs.fill_value * rhs.fill_value
    return SparseTensor(
        (),
        (),
        new_val,
        scalar_mult=new_mult,
        fill_value=fill,
        sort_val=False,
        check_consistency=False,
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

    For sparse operands the contracting dims are the last ``min(len(lhs.primal),
    len(rhs.out))`` of each side; depth is the product of their *logical*
    sizes (a ``SparseIndex`` of size N with block_size B contributes ``N*B``,
    matching what ``dot_general`` actually contracts over after densification).
    """
    if hasattr(lhs, "primal_dims") and hasattr(rhs, "out_dims"):
        n_contract = min(len(lhs.primal_dims), len(rhs.out_dims))
        K = 1
        for i in range(n_contract):
            d = lhs.primal_dims[-1 - i]
            K *= int(d.logical_size)
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
    is the contraction depth (a ``SparseIndex(size=N, block_size=B)``
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
