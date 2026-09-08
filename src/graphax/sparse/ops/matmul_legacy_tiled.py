"""RACE-ONLY: a verbatim copy of the incumbent tiled executor of graphax
``wip/t62-20260905`` (1f3d404), reachable only under ``GRAPHAX_TILED_LEGACY=1``.

Ticket dsnn-3qm.67 (the .28 race, lane B).  The live tiled path in
``ops/matmul.py`` is being made lazy: implicit axes, uniform values and
surviving diagonal pairs stay symbolic instead of being broadcast into the
physical grid.  The landing test has to pair the candidate against the
UNTOUCHED incumbent inside one process, so the incumbent lives here, byte for
byte, with only the module it is copied from changed (the shared helpers it
does not modify are imported from ``ops/matmul.py``).

Nothing imports this module unless ``GRAPHAX_TILED_LEGACY`` is set.  Step 3 of
the .28 design note DELETES this file together with the losing engine.
"""
# pyright: reportImportCycles=false
from __future__ import annotations

import builtins
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from graphax.sparse.indexes import DenseIndex, DiagonalIndex

from .utils import _prepare_physical_array, _val_or_one
from .matmul import (
    AXES_PER_PAIR,
    GRID_AXES_PER_PAIR,
    CRes,
    _as_shape,
    _build_sparse,
    _contraction_factors,
    _contraction_perms,
    _final_grid,
    _gx_dot_general,
    _reduce_grid,
)
from .layout import generate_grouped_permutation


def _prepare_physical_arrays(lhs_val, rhs_val, pairs):
    def flat_axes(sides):
        return [
            a for s in sides for a in (s.outer_axis, s.block_axis, s.shared_block_axis)
        ]

    return (
        _prepare_physical_array(lhs_val, flat_axes([p.lhs for p in pairs])),
        _prepare_physical_array(rhs_val, flat_axes([p.rhs for p in pairs])),
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
    res_raw = _gx_dot_general(lhs_view, rhs_view, _dot_general_axes(N, pairs))
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



def _execute_tiled(ctx, rhs_dims):
    """Fallback: full tiled algorithm. Handles every case the fast paths
    bail on, including LCM-mismatched outer sizes, spatial sparse pairs,
    and broadcast / unmaterialized val axes."""
    lhs_val, rhs_val = _val_or_one(ctx.lhs), _val_or_one(ctx.rhs)
    lhs_val, rhs_val = _prepare_physical_arrays(lhs_val, rhs_val, ctx.pairs)
    grid, shared, lhs_lens, rhs_lens, scalar = (
        _execute_block_sparse_contraction(lhs_val, rhs_val, ctx.pairs, ctx)
    )
    res = CRes(
        grid=grid,
        shared_factors=shared,
        lhs_block_lens=lhs_lens,
        rhs_block_lens=rhs_lens,
        scalar_mult=scalar,
    )
    return _build_output_tensor(ctx, rhs_dims, res)



