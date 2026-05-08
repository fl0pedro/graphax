"""Densification of ``SparseTensor`` axes.

* ``dense(tensor, axes=None, hard=False)`` — materialize sparse pairs into dense axes,
  returning a new ``SparseTensor`` whose ``val`` carries the requested axes in dense form.
  ``hard=True`` also materializes implicit pairs (``axis is None``).

* ``dense_for_matmul(tensor) -> Array`` — a fusion-friendly densifier that returns a raw
  ``Array`` ready to feed into ``jax.lax.dot_general``. For tensors with simple structure
  (fully-dense, or a single sparse pair), it emits a single broadcast+select fusion that
  XLA can fold into the consuming matmul kernel — keeping the expansion in SMEM rather
  than spilling to L2 / HBM. Falls back to the scatter-based ``dense()`` otherwise.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
from dataclasses import replace

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import Array

from graphax.sparse.indexes import Index, SparseIndex, DenseIndex

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


# --- Public API ----------------------------------------------------------
def dense(tensor: SparseTensor, axes: Sequence[int] | None = None, hard: bool = False) -> SparseTensor:
    # Compressed-storage path: materialize only as much as the requested ``axes``
    # demand.
    #   * No axes asked / axes covering every meta-block-diagonal pair → expand
    #     fully (``to_dense()`` for banded, ``to_meta_blocks()`` then standard
    #     densify for the meta-block-diag case — same end result either way).
    #   * Axes only target *outside* the meta-block-diagonal pair → leave
    #     ``compressed_val`` untouched and densify only the requested outer axes.
    #     The pair stays compressed (M× tighter HBM footprint).
    # ``_materialize_compressed`` picks the structure-preserving form when it's
    # available (``to_meta_blocks``); both expressions are fused broadcast/select
    # chains XLA folds into the consumer.
    cv = getattr(tensor, "compressed_val", None)
    if cv is not None:
        from .utils import _copy, _materialize_compressed, _has_meta_block_diag_dims
        meta = getattr(cv, "meta_block_shape", None)
        is_meta_diag = (meta is not None
                        and _has_meta_block_diag_dims(tensor, meta))
        # The compressed pair occupies the FIRST two axes (out_dim / primal_dim
        # built by elementwise's compressed fast path or ``from_compressed``).
        compressed_axes = {0, 1} if is_meta_diag else set(range(tensor.ndim))
        requested = set(range(tensor.ndim)) if axes is None else set(axes)
        if compressed_axes & requested:
            # Materialize the compressed pair. Meta-block-diag → expand to
            # ``(M, H, W)`` so the rest of the function emits a regular block-
            # diagonal densify; banded → full ``to_dense``.
            tensor = _copy(tensor, val=_materialize_compressed(tensor))
        else:
            # Requested axes don't touch the compressed pair → keep
            # ``compressed_val`` as-is. For the 2-D case there are no other axes
            # to densify; just return the tensor unchanged. For higher-rank
            # cases this preserves the M× compression on the pair and lets the
            # caller densify only the leftover dims via a future recursion.
            return tensor

    logical_indices = set(range(tensor.ndim)) if axes is None else set(axes)
    id_to_idx = {dim.id: i for i, dim in enumerate(tensor.dims)}
    implicit = _get_implicit_indices(tensor, logical_indices, hard)
    if not hard:
        logical_indices -= implicit

    values = jnp.array(1.0, dtype=tensor.dtype) if tensor.val is None else tensor.val
    values, updated_dims = _broadcast_and_append_dimensions(tensor, values, implicit)

    actual_scatter = _collect_scatter_indices(logical_indices, updated_dims, id_to_idx)
    phys_to_scatter = sorted({
        updated_dims[i].axis
        for i in actual_scatter
        if isinstance(updated_dims[i], SparseIndex) and updated_dims[i].axis is not None
    })

    values, result_dims = _apply_dense_scattering(
        values, tensor.fill_value, updated_dims, actual_scatter, phys_to_scatter
    )
    from graphax.sparse.tensor import SparseTensor
    return SparseTensor(
        tuple(result_dims[: len(tensor.out_dims)]),
        tuple(result_dims[len(tensor.out_dims):]),
        values,
        scalar_mult=tensor.scalar_mult,
        fill_value=tensor.fill_value,
        sort_val=False, check_consistency=False,
        zero_fill=getattr(tensor, "_zero_fill", None),
    )


def dense_for_matmul(tensor: SparseTensor) -> Array:
    """Fusion-friendly dense form (broadcast+select; no scatter / no gather where possible).

    See module docstring. Falls back to the scatter-based ``dense()`` for shapes the
    simple builder doesn't cover.
    """
    # Fast path: fully-dense tensor — val IS the dense form (modulo permutation).
    if all(isinstance(d, DenseIndex) for d in tensor.dims):
        if tensor.val is None:
            return jnp.broadcast_to(tensor.fill_value * tensor.scalar_mult, tensor.shape)
        v = tensor.val
        perm = [d.axis for d in tensor.dims if d.axis is not None]
        if perm and sorted(perm) == list(range(len(perm))) and perm != list(range(len(perm))):
            v = v.transpose(perm)
        v = v * tensor.scalar_mult
        # Broadcast back up to the logical shape: a SparseTensor can carry a
        # rank-0 (or otherwise rank-reduced) ``val`` while its dims advertise
        # a larger structural shape (e.g. concat-transformed Jacobians where
        # the val landed as a scalar but the structure says ``(N,)``).
        if v.ndim != len(tensor.shape):
            v = jnp.broadcast_to(v, tensor.shape)
        return v

    # Single-sparse-pair fast path: emit a where over a 1-fusion dense form.
    sparse_dims = [d for d in tensor.dims if isinstance(d, SparseIndex)]
    if (tensor.val is not None and len(sparse_dims) == 2
            and sparse_dims[0].other_id == sparse_dims[1].id
            and sparse_dims[1].other_id == sparse_dims[0].id
            and sparse_dims[0].axis == sparse_dims[1].axis):
        d_o, d_i = sparse_dims
        B_o, B_i = d_o.block_size or 1, d_i.block_size or 1
        N = d_o.size
        logical_outer, logical_inner = N * B_o, N * B_i
        gather_axes = [d_o.axis, d_o.block_axis, d_i.block_axis]
        if all(a is not None for a in gather_axes):
            v = tensor.val * tensor.scalar_mult
            leftover = [a for a in range(v.ndim) if a not in gather_axes]
            v = v.transpose(gather_axes + leftover)
            # v.shape: (N, B_o, B_i, *leftover_sizes). Collapse (N, B_o) → logical_outer
            # so that v_2d[i, k, *l] == v[i // B_o, i % B_o, k, *l].
            leftover_sizes = list(v.shape[3:])
            v_2d = v.reshape(logical_outer, B_i, *leftover_sizes)
            # Tile across the inner axis via broadcast+reshape (pure shape ops, fold into
            # the consuming kernel). gathered[i, j, *l] == v_2d[i, j % B_i, *l].
            v_3d = jnp.broadcast_to(v_2d[:, None, ...], (logical_outer, N, B_i, *leftover_sizes))
            gathered = v_3d.reshape(logical_outer, logical_inner, *leftover_sizes)
            blk_o = jnp.arange(logical_outer) // B_o
            blk_i = jnp.arange(logical_inner) // B_i
            mask = blk_o[:, None] == blk_i[None, :]
            mask_b = mask[(..., *((None,) * len(leftover_sizes)))]
            dense_pair = jnp.where(mask_b, gathered, tensor.fill_value * tensor.scalar_mult)
            # Reorder dense_pair's axes to match tensor.dims order. ``target_axes[i]`` is
            # the dense_pair axis that should land at result position ``i``, so the
            # transpose permutation is ``target_axes`` directly (NOT its inverse).
            leftover_iter = iter(range(2, 2 + len(leftover_sizes)))
            target_axes = [
                0 if d is d_o else 1 if d is d_i else next(leftover_iter)
                for d in tensor.dims
            ]
            return dense_pair.transpose(target_axes)

    return dense(tensor, hard=True).val * tensor.scalar_mult


# --- Internals -----------------------------------------------------------
def _is_dimension_implicit(dim, id_to_idx):
    needs_val = dim.axis is None
    needs_block = (isinstance(dim, SparseIndex) and dim.block_size and dim.block_axis is None)
    if needs_val or needs_block:
        to_add = {id_to_idx[dim.id]}
        if isinstance(dim, SparseIndex):
            to_add.add(id_to_idx[dim.other_id])
        return True, to_add
    return False, set()


def _get_implicit_indices(tensor, requested_axes, hard):
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
            target[dim.block_axis] = max(target[dim.block_axis], dim.block_size or 1)
    return tuple(target)


def _append_primary_dimension(i, dim, implicit, current_ndim, dims_to_append, sparse_pair_map):
    if i in implicit and dim.axis is None:
        if isinstance(dim, SparseIndex):
            pair_key = tuple(sorted((dim.id, dim.other_id)))
            if pair_key in sparse_pair_map:
                new_idx = sparse_pair_map[pair_key]
            else:
                new_idx = current_ndim + len(dims_to_append)
                dims_to_append.append(dim.size)
                sparse_pair_map[pair_key] = new_idx
        else:
            new_idx = current_ndim + len(dims_to_append)
            dims_to_append.append(dim.size)
        return replace(dim, axis=new_idx)
    return dim


def _append_block_dimension(i, dim, implicit, current_ndim, dims_to_append):
    if (i in implicit and isinstance(dim, SparseIndex)
            and dim.block_axis is None and dim.block_size is not None):
        new_idx = current_ndim + len(dims_to_append)
        dims_to_append.append(dim.block_size)
        return replace(dim, block_axis=new_idx)
    return dim


def _broadcast_and_append_dimensions(tensor, values, implicit):
    if tensor.val is not None:
        target = _calculate_target_shape(values.shape, tensor.dims)
        if target != values.shape:
            values = jnp.broadcast_to(values, target)
    dims_to_append, sparse_pair_map, current_ndim = [], {}, values.ndim
    new_dims = [_append_primary_dimension(i, d, implicit, current_ndim, dims_to_append, sparse_pair_map)
                for i, d in enumerate(tensor.dims)]
    new_dims = [_append_block_dimension(i, d, implicit, current_ndim, dims_to_append)
                for i, d in enumerate(new_dims)]
    if dims_to_append:
        values = lax.broadcast_in_dim(values, values.shape + tuple(dims_to_append), tuple(range(current_ndim)))
    return values, new_dims


def _collect_scatter_indices(logical_indices, updated_dims, id_to_idx):
    scatter_logical = set()
    for i in logical_indices:
        scatter_logical.add(i)
        if isinstance(updated_dims[i], SparseIndex):
            scatter_logical.add(id_to_idx[updated_dims[i].other_id])
    return scatter_logical


def _prepare_values_for_scattering(values, scatter_axes, fill_value):
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


def _apply_dense_scattering(values, fill_value, logical_dims, scatter_logical_indices, scatter_phys_axes):
    values, phys_map = _prepare_values_for_scattering(values, scatter_phys_axes, fill_value)
    final_perm, final_shape, active_axes = [], [], set()
    log_to_phys, visited_pairs, sparse_phys_to_final = {}, set(), {}
    curr_f_idx = 0

    for i, dim in enumerate(logical_dims):
        pair_key = (tuple(sorted((dim.id, dim.other_id)))
                    if isinstance(dim, SparseIndex) else (dim.id,))
        if i in scatter_logical_indices and isinstance(dim, SparseIndex):
            idx = 1 if pair_key in visited_pairs else 0
            visited_pairs.add(pair_key)
            p_idx = phys_map[dim.axis][idx]
            final_perm.append(p_idx); active_axes.add(p_idx)
            l_size = dim.size
            if dim.block_axis is not None:
                b_p = phys_map[dim.block_axis]
                final_perm.append(b_p); active_axes.add(b_p)
                l_size *= dim.block_size
            final_shape.append(l_size)
            log_to_phys[i] = (curr_f_idx, None, l_size)
            curr_f_idx += 1
        elif isinstance(dim, SparseIndex):
            new_v, new_b = None, None
            if pair_key not in visited_pairs:
                if dim.axis is not None:
                    p_idx = phys_map[dim.axis]
                    final_perm.append(p_idx); active_axes.add(p_idx)
                    sparse_phys_to_final[dim.axis] = curr_f_idx
                    final_shape.append(dim.size)
                    curr_f_idx += 1
                visited_pairs.add(pair_key)
            new_v = sparse_phys_to_final.get(dim.axis)
            if dim.block_axis is not None:
                b_p = phys_map[dim.block_axis]
                final_perm.append(b_p); active_axes.add(b_p)
                new_b = curr_f_idx
                final_shape.append(dim.block_size)
                curr_f_idx += 1
            log_to_phys[i] = (new_v, new_b, None)
        else:
            new_v = None
            if dim.axis is not None:
                p_idx = phys_map[dim.axis]
                final_perm.append(p_idx); active_axes.add(p_idx)
                new_v = curr_f_idx
                final_shape.append(dim.size)
                curr_f_idx += 1
            log_to_phys[i] = (new_v, None, None)

    final_perm.extend(i for i in range(values.ndim) if i not in active_axes)
    unique_perm = list(dict.fromkeys(final_perm))
    values = values.transpose(unique_perm).reshape(tuple(final_shape) + values.shape[len(unique_perm):])
    return values, _reconstruct_logical_dimensions(logical_dims, log_to_phys)


def _reconstruct_logical_dimensions(logical_dims, logical_to_physical):
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
    """Place ``val[i, ...]`` on the diagonal of a ``(B, B, *trailing)`` grid, with
    ``fill_value`` everywhere off-diagonal. Implemented as a single broadcast+select
    fusion (no scatter) so XLA can fold it into a downstream consumer."""
    B = val.shape[0]
    fv = jnp.asarray(fill_value, dtype=val.dtype)
    eye_mask = jnp.eye(B, dtype=jnp.bool_)
    eye_mask = eye_mask[(slice(None), slice(None)) + (None,) * (val.ndim - 1)]
    val_b = jnp.broadcast_to(val[:, None], (B, B) + val.shape[1:])
    return jnp.where(eye_mask, val_b, fv)
