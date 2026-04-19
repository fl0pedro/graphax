from __future__ import annotations

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import Array
from typing import TYPE_CHECKING, Sequence
from dataclasses import replace

from ..dimensions import Dimension, SparseDimension, DenseDimension

if TYPE_CHECKING:
    from ..tensor import SparseTensor


def _is_dimension_implicit(dim, id_to_idx):
    needs_val = dim.val_dim is None
    needs_block = (
        isinstance(dim, SparseDimension)
        and dim.block_size
        and dim.block_val_dim is None
    )
    if needs_val or needs_block:
        to_add = {id_to_idx[dim.id]}
        if isinstance(dim, SparseDimension):
            to_add.add(id_to_idx[dim.other_id])
        return True, to_add
    return False, set()


def _get_implicit_indices(
    tensor: SparseTensor, requested_axes: set[int], hard: bool
) -> set[int]:
    id_to_idx = {dim.id: i for i, dim in enumerate(tensor.dims)}
    implicit = set()
    for i in requested_axes:
        is_impl, indices = _is_dimension_implicit(tensor.dims[i], id_to_idx)
        if is_impl:
            implicit.update(indices)
    return implicit


def _calculate_target_shape(val_shape, dims):
    target = list(val_shape)
    for dim in dims:
        if dim.val_dim is not None:
            target[dim.val_dim] = max(target[dim.val_dim], dim.size)
        if isinstance(dim, SparseDimension) and dim.block_val_dim is not None:
            target[dim.block_val_dim] = max(
                target[dim.block_val_dim], dim.block_size or 1
            )
    return tuple(target)


def _append_primary_dimension(
    i, dim, implicit_indices, current_ndim, dims_to_append, sparse_pair_val_dim_map
):
    if i in implicit_indices and dim.val_dim is None:
        if isinstance(dim, SparseDimension):
            pair_key = tuple(sorted((dim.id, dim.other_id)))
            if pair_key in sparse_pair_val_dim_map:
                new_idx = sparse_pair_val_dim_map[pair_key]
            else:
                new_idx = current_ndim + len(dims_to_append)
                dims_to_append.append(dim.size)
                sparse_pair_val_dim_map[pair_key] = new_idx
        else:
            new_idx = current_ndim + len(dims_to_append)
            dims_to_append.append(dim.size)
        return replace(dim, val_dim=new_idx)
    return dim


def _append_block_dimension(i, dim, implicit_indices, current_ndim, dims_to_append):
    if (
        i in implicit_indices
        and isinstance(dim, SparseDimension)
        and dim.block_val_dim is None
        and dim.block_size is not None
    ):
        new_idx = current_ndim + len(dims_to_append)
        dims_to_append.append(dim.block_size)
        return replace(dim, block_val_dim=new_idx)
    return dim


def _broadcast_and_append_dimensions(
    tensor: SparseTensor, values: Array, implicit_indices: set[int]
) -> tuple[Array, list[Dimension]]:
    if tensor.val is not None:
        target = _calculate_target_shape(values.shape, tensor.dims)
        if target != values.shape:
            values = jnp.broadcast_to(values, target)

    dims_to_append, sparse_pair_map, current_ndim = [], {}, values.ndim
    new_dims = [
        _append_primary_dimension(
            i, d, implicit_indices, current_ndim, dims_to_append, sparse_pair_map
        )
        for i, d in enumerate(tensor.dims)
    ]
    new_dims = [
        _append_block_dimension(i, d, implicit_indices, current_ndim, dims_to_append)
        for i, d in enumerate(new_dims)
    ]

    if dims_to_append:
        values = lax.broadcast_in_dim(
            values, values.shape + tuple(dims_to_append), tuple(range(current_ndim))
        )
    return values, new_dims


def _collect_scatter_indices(logical_indices, updated_dims, id_to_idx):
    scatter_logical = set()
    for i in logical_indices:
        scatter_logical.add(i)
        if isinstance(updated_dims[i], SparseDimension):
            scatter_logical.add(id_to_idx[updated_dims[i].other_id])
    return scatter_logical


def dense(
    tensor: SparseTensor, axes: Sequence[int] | None = None, hard: bool = False
) -> SparseTensor:
    logical_indices = set(range(tensor.ndim)) if axes is None else set(axes)
    id_to_idx = {dim.id: i for i, dim in enumerate(tensor.dims)}
    implicit_indices = _get_implicit_indices(tensor, logical_indices, hard)
    if not hard:
        logical_indices -= implicit_indices

    values = jnp.array(1.0, dtype=tensor.dtype) if tensor.val is None else tensor.val
    values, updated_dims = _broadcast_and_append_dimensions(
        tensor, values, implicit_indices
    )

    actual_scatter = _collect_scatter_indices(logical_indices, updated_dims, id_to_idx)
    phys_to_scatter = sorted(
        list(
            {
                updated_dims[i].val_dim
                for i in actual_scatter
                if isinstance(updated_dims[i], SparseDimension)
                and updated_dims[i].val_dim is not None
            }
        )
    )

    values, result_dims = _apply_dense_scattering(
        values, tensor.fill_value, updated_dims, actual_scatter, phys_to_scatter
    )
    from ..tensor import SparseTensor

    return SparseTensor(
        tuple(result_dims[: len(tensor.out_dims)]),
        tuple(result_dims[len(tensor.out_dims) :]),
        values,
        tensor.scalar_mult,
        tensor.fill_value,
        sort_val=False,
        check_consistency=False,
    )


def _prepare_values_for_scattering(
    values: Array, scatter_axes: list[int], fill_value: Array
) -> tuple[Array, dict[int, int | tuple[int, int]]]:
    if not scatter_axes:
        return values, {idx: idx for idx in range(values.ndim)}

    other_axes = [i for i in range(values.ndim) if i not in scatter_axes]
    unique_perm = []
    seen = set()
    for ax in scatter_axes + other_axes:
        if ax not in seen:
            unique_perm.append(ax)
            seen.add(ax)

    values = values.transpose(unique_perm)
    num_scatter = len(scatter_axes)
    s_shape, t_shape = values.shape[:num_scatter], values.shape[num_scatter:]

    values = _densify_diagonal_scatter(values.reshape((-1,) + t_shape), fill_value)
    values = values.reshape(s_shape + s_shape + t_shape)

    phys_map = {old: (i, num_scatter + i) for i, old in enumerate(scatter_axes)}
    for i, old in enumerate(other_axes):
        phys_map[old] = 2 * num_scatter + i
    return values, phys_map


def _apply_dense_scattering(
    values: Array,
    fill_value: Array,
    logical_dims: list[Dimension],
    scatter_logical_indices: set[int],
    scatter_phys_axes: list[int],
) -> tuple[Array, list[Dimension]]:
    values, phys_map = _prepare_values_for_scattering(
        values, scatter_phys_axes, fill_value
    )
    unique_perm, final_val_shape, logical_to_physical = _compute_physical_layout_fixed(
        values.ndim, logical_dims, scatter_logical_indices, phys_map
    )
    redundant = set(range(len(scatter_phys_axes), 2 * len(scatter_phys_axes)))
    leftover = [i for i in range(values.ndim) if i not in unique_perm and i not in redundant]
    full_perm = unique_perm + leftover
    
    values = values.transpose(full_perm).reshape(
        tuple(final_val_shape) + tuple(values.shape[i] for i in leftover)
    )
    return values, _reconstruct_logical_dimensions(logical_dims, logical_to_physical)


def _process_scatter_dim(
    dim, phys_map, visited_scatter_pairs, final_val_perm, active_axes, curr_final_idx
):
    pair_key = tuple(sorted((dim.id, dim.other_id)))
    idx = 1 if pair_key in visited_scatter_pairs else 0
    visited_scatter_pairs.add(pair_key)
    phys_idx = phys_map[dim.val_dim][idx]
    final_val_perm.append(phys_idx)
    active_axes.add(phys_idx)
    l_size = dim.size
    if dim.block_val_dim is not None:
        b_phys = phys_map[dim.block_val_dim]
        final_val_perm.append(b_phys)
        active_axes.add(b_phys)
        l_size *= dim.block_size
    return (curr_final_idx, None, l_size), curr_final_idx + 1


def _process_remain_sparse(
    dim,
    phys_map,
    visited_sparse_pairs,
    final_val_perm,
    active_axes,
    curr_f_idx,
    sparse_phys_to_final,
):
    pair_key = tuple(sorted((dim.id, dim.other_id)))
    if pair_key not in visited_sparse_pairs:
        if dim.val_dim is not None:
            phys_idx = phys_map[dim.val_dim]
            final_val_perm.append(phys_idx)
            active_axes.add(phys_idx)
            sparse_phys_to_final[dim.val_dim] = curr_f_idx
            curr_f_idx += 1
        visited_sparse_pairs.add(pair_key)

    new_val_dim = sparse_phys_to_final.get(dim.val_dim)
    new_block_dim = None
    if dim.block_val_dim is not None:
        b_phys = phys_map[dim.block_val_dim]
        final_val_perm.append(b_phys)
        active_axes.add(b_phys)
        new_block_dim = curr_f_idx
        curr_f_idx += 1
    return (new_val_dim, new_block_dim, None), curr_f_idx


def _process_remain_dense(dim, phys_map, final_val_perm, active_axes, curr_f_idx):
    new_val_dim = None
    if dim.val_dim is not None:
        phys_idx = phys_map[dim.val_dim]
        final_val_perm.append(phys_idx)
        active_axes.add(phys_idx)
        new_val_dim = curr_f_idx
        curr_f_idx += 1
    return (new_val_dim, None, None), curr_f_idx


def _compute_physical_layout_fixed(
    values_ndim: int,
    logical_dims: list[Dimension],
    scatter_logical_indices: set[int],
    phys_map: dict[int, int | tuple[int, int]],
):
    final_val_perm, final_val_shape, active_axes = [], [], set()
    curr_f_idx, visited_scatter, visited_sparse, sparse_phys_map, log_to_phys = (
        0,
        set(),
        set(),
        {},
        {},
    )
    for i, dim in enumerate(logical_dims):
        if i in scatter_logical_indices and isinstance(dim, SparseDimension):
            res, curr_f_idx = _process_scatter_dim(
                dim, phys_map, visited_scatter, final_val_perm, active_axes, curr_f_idx
            )
            final_val_shape.append(res[2])
        elif isinstance(dim, SparseDimension):
            res, curr_f_idx = _process_remain_sparse(
                dim,
                phys_map,
                visited_sparse,
                final_val_perm,
                active_axes,
                curr_f_idx,
                sparse_phys_map,
            )
            if res[0] is not None:
                final_val_shape.append(dim.size)
            if res[1] is not None:
                final_val_shape.append(dim.block_size)
        else:
            res, curr_f_idx = _process_remain_dense(
                dim, phys_map, final_val_perm, active_axes, curr_f_idx
            )
            if res[0] is not None:
                final_val_shape.append(dim.size)
        log_to_phys[i] = res

    # unique_perm should only contain axes associated with logical dimensions

    unique_perm = []
    seen = set()
    for p in final_val_perm:
        if p not in seen:
            unique_perm.append(p)
            seen.add(p)
    return unique_perm, final_val_shape, log_to_phys


def _reconstruct_logical_dimensions(
    logical_dims: list[Dimension], logical_to_physical: dict[int, tuple]
):
    res = []
    for i, dim in enumerate(logical_dims):
        v_ax, b_ax, d_size = logical_to_physical[i]
        if d_size is not None:
            res.append(DenseDimension(dim.id, d_size, v_ax))
        elif isinstance(dim, SparseDimension):
            res.append(replace(dim, val_dim=v_ax, block_val_dim=b_ax))
        else:
            res.append(replace(dim, val_dim=v_ax))
    return res


def _densify_diagonal_scatter(val: Array, fill_value: Array) -> Array:
    B = val.shape[0]
    out = jnp.full((B, B) + val.shape[1:], fill_value, dtype=val.dtype)
    idx = jnp.arange(B)
    return out.at[idx, idx].set(val)
