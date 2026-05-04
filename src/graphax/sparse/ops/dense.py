from __future__ import annotations

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import Array
from typing import TYPE_CHECKING, Sequence
from dataclasses import replace

from graphax.sparse.indexes import Index, SparseIndex, DenseIndex

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


def _is_dimension_implicit(dim, id_to_idx):
    needs_val = dim.axis is None
    needs_block = (
        isinstance(dim, SparseIndex)
        and dim.block_size
        and dim.block_axis is None
    )
    if needs_val or needs_block:
        to_add = {id_to_idx[dim.id]}
        if isinstance(dim, SparseIndex):
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
        if dim.axis is not None:
            target[dim.axis] = max(target[dim.axis], dim.size)
        if isinstance(dim, SparseIndex) and dim.block_axis is not None:
            target[dim.block_axis] = max(
                target[dim.block_axis], dim.block_size or 1
            )
    return tuple(target)


def _append_primary_dimension(
    i, dim, implicit_indices, current_ndim, dims_to_append, sparse_pair_axis_map
):
    if i in implicit_indices and dim.axis is None:
        if isinstance(dim, SparseIndex):
            pair_key = tuple(sorted((dim.id, dim.other_id)))
            if pair_key in sparse_pair_axis_map:
                new_idx = sparse_pair_axis_map[pair_key]
            else:
                new_idx = current_ndim + len(dims_to_append)
                dims_to_append.append(dim.size)
                sparse_pair_axis_map[pair_key] = new_idx
        else:
            new_idx = current_ndim + len(dims_to_append)
            dims_to_append.append(dim.size)
        return replace(dim, axis=new_idx)
    return dim


def _append_block_dimension(i, dim, implicit_indices, current_ndim, dims_to_append):
    if (
        i in implicit_indices
        and isinstance(dim, SparseIndex)
        and dim.block_axis is None
        and dim.block_size is not None
    ):
        new_idx = current_ndim + len(dims_to_append)
        dims_to_append.append(dim.block_size)
        return replace(dim, block_axis=new_idx)
    return dim


def _broadcast_and_append_dimensions(
    tensor: SparseTensor, values: Array, implicit_indices: set[int]
) -> tuple[Array, list[Index]]:
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
        if isinstance(updated_dims[i], SparseIndex):
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
                updated_dims[i].axis
                for i in actual_scatter
                if isinstance(updated_dims[i], SparseIndex)
                and updated_dims[i].axis is not None
            }
        )
    )

    values, result_dims = _apply_dense_scattering(
        values, tensor.fill_value, updated_dims, actual_scatter, phys_to_scatter
    )
    from graphax.sparse.tensor import SparseTensor

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
    unique_perm = list(dict.fromkeys(scatter_axes + other_axes))

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
    logical_dims: list[Index],
    scatter_logical_indices: set[int],
    scatter_phys_axes: list[int],
) -> tuple[Array, list[Index]]:
    values, phys_map = _prepare_values_for_scattering(
        values, scatter_phys_axes, fill_value
    )

    final_perm, final_shape, active_axes = [], [], set()
    log_to_phys, visited_pairs, sparse_phys_to_final = {}, set(), {}
    curr_f_idx = 0

    for i, dim in enumerate(logical_dims):
        pair_key = (
            tuple(sorted((dim.id, dim.other_id)))
            if isinstance(dim, SparseIndex)
            else (dim.id,)
        )
        if i in scatter_logical_indices and isinstance(dim, SparseIndex):
            idx = 1 if pair_key in visited_pairs else 0
            visited_pairs.add(pair_key)
            p_idx = phys_map[dim.axis][idx]
            final_perm.append(p_idx)
            active_axes.add(p_idx)
            l_size = dim.size
            if dim.block_axis is not None:
                b_p = phys_map[dim.block_axis]
                final_perm.append(b_p)
                active_axes.add(b_p)
                l_size *= dim.block_size
            final_shape.append(l_size)
            log_to_phys[i] = (curr_f_idx, None, l_size)
            curr_f_idx += 1
        elif isinstance(dim, SparseIndex):
            new_v, new_b = None, None
            if pair_key not in visited_pairs:
                if dim.axis is not None:
                    p_idx = phys_map[dim.axis]
                    final_perm.append(p_idx)
                    active_axes.add(p_idx)
                    sparse_phys_to_final[dim.axis] = curr_f_idx
                    final_shape.append(dim.size)
                    curr_f_idx += 1
                visited_pairs.add(pair_key)
            new_v = sparse_phys_to_final.get(dim.axis)
            if dim.block_axis is not None:
                b_p = phys_map[dim.block_axis]
                final_perm.append(b_p)
                active_axes.add(b_p)
                new_b = curr_f_idx
                final_shape.append(dim.block_size)
                curr_f_idx += 1
            log_to_phys[i] = (new_v, new_b, None)
        else:
            new_v = None
            if dim.axis is not None:
                p_idx = phys_map[dim.axis]
                final_perm.append(p_idx)
                active_axes.add(p_idx)
                new_v = curr_f_idx
                final_shape.append(dim.size)
                curr_f_idx += 1
            log_to_phys[i] = (new_v, None, None)

    final_perm.extend(i for i in range(values.ndim) if i not in active_axes)
    unique_perm = list(dict.fromkeys(final_perm))

    values = values.transpose(unique_perm).reshape(
        tuple(final_shape) + values.shape[len(unique_perm) :]
    )
    return values, _reconstruct_logical_dimensions(logical_dims, log_to_phys)


def _reconstruct_logical_dimensions(
    logical_dims: list[Index], logical_to_physical: dict[int, tuple]
):
    res = []
    for i, dim in enumerate(logical_dims):
        v_ax, b_ax, d_size = logical_to_physical[i]
        if d_size is not None:
            res.append(DenseIndex(dim.id, d_size, v_ax))
        elif isinstance(dim, SparseIndex):
            res.append(replace(dim, axis=v_ax, block_axis=b_ax))
        else:
            res.append(replace(dim, axis=v_ax))
    return res


def _densify_diagonal_scatter(val: Array, fill_value: Array) -> Array:
    B = val.shape[0]
    out = jnp.full((B, B) + val.shape[1:], fill_value, dtype=val.dtype)
    idx = jnp.arange(B)
    return out.at[idx, idx].set(val)
