from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
from dataclasses import replace

import jax.numpy as jnp

from graphax.sparse.ops.dense import dense

from graphax.sparse.indexes import Index, SparseIndex, DenseIndex

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor
from graphax.sparse.indexes import Index, SparseIndex, DenseIndex


def _get_full_permutation(
    num_out: int,
    num_primal: int,
    out_axes: Sequence[int] | None = None,
    primal_axes: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], list[int], list[int]]:
    """Computes the full permutation vector from out and primal axes."""
    total_ndim = num_out + num_primal

    if out_axes is None and primal_axes is None:
        # Default transpose flips all dimensions
        full_permutation = tuple(range(total_ndim - 1, -1, -1))
        out_axes = list(full_permutation[:num_primal])
        primal_axes = list(full_permutation[num_primal:])
    else:
        out_axes = list(out_axes) if out_axes is not None else list(range(num_out))
        primal_axes = (
            list(primal_axes)
            if primal_axes is not None
            else list(range(num_out, total_ndim))
        )

    # Normalize negative indices
    out_axes = [i % total_ndim for i in out_axes]
    primal_axes = [i % total_ndim for i in primal_axes]
    full_permutation = tuple(out_axes) + tuple(primal_axes)

    if len(full_permutation) != total_ndim or len(set(full_permutation)) != total_ndim:
        raise ValueError(
            f"Invalid transpose permutation: {full_permutation} for ndim {total_ndim}"
        )

    return full_permutation, out_axes, primal_axes


def _ensure_valid_sparsity(
    tensor: SparseTensor, out_axes: Sequence[int], primal_axes: Sequence[int]
) -> SparseTensor:
    """Identify and densify sparse dimensions that would end up on the same side."""
    dim_id_to_index = {d.id: i for i, d in enumerate(tensor.dims)}
    out_set = set(out_axes)
    primal_set = set(primal_axes)

    axes_to_densify = []
    for i, dim in enumerate(tensor.dims):
        if isinstance(dim, SparseIndex):
            other_pos = dim_id_to_index[dim.other_id]
            # If both ends of a sparse pair end up on the same side (both out or both primal),
            # they must be densified to maintain the invariant that sparse dimensions
            # are split between the out and primal groups.
            if (i in out_set and other_pos in out_set) or (
                i in primal_set and other_pos in primal_set
            ):
                axes_to_densify.append(i)

    if axes_to_densify:
        tensor = dense(tensor, axes=tuple(axes_to_densify), hard=True)

    return tensor


def transpose(
    tensor: SparseTensor,
    out_axes: Sequence[int] | None = None,
    primal_axes: Sequence[int] | None = None,
) -> SparseTensor:
    """Transpose a SparseTensor by reordering its out and primal dimensions."""
    from graphax.sparse.ops.utils import _sort_val, _copy

    # 1. Normalize and validate the permutation
    full_permutation, new_out_axes, new_primal_axes = _get_full_permutation(
        len(tensor.out_dims), len(tensor.primal_dims), out_axes, primal_axes
    )

    # 2. Ensure sparse dimensions remain split across sides; densify if not.
    tensor = _ensure_valid_sparsity(tensor, new_out_axes, new_primal_axes)

    # 3. Reorder dimensions and update IDs
    all_dims = tensor.dims
    reordered_dims = [all_dims[i] for i in full_permutation]
    updated_dims = reordered_dims

    n_out = len(new_out_axes)
    new_out_dims, new_primal_dims = updated_dims[:n_out], updated_dims[n_out:]

    # 4. Sort values and return
    new_out_dims, new_primal_dims, new_val = _sort_val(
        new_out_dims, new_primal_dims, tensor.val
    )

    return _copy(
        tensor, val=new_val, out_dims=new_out_dims, primal_dims=new_primal_dims
    )
