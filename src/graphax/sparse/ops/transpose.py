from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
from dataclasses import replace

import jax.numpy as jnp

from .dense import dense

from ..dimensions import Dimension, SparseDimension, DenseDimension

if TYPE_CHECKING:
    from ..tensor import SparseTensor


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
        if isinstance(dim, SparseDimension):
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


def _remap_logical_dimensions(
    tensor: SparseTensor,
    full_permutation: tuple[int, ...],
    out_axes: Sequence[int],
    primal_axes: Sequence[int],
) -> tuple[list[Dimension], list[Dimension]]:
    """Update logical dimension metadata (IDs and other_IDs) after transpose."""
    dims = tensor.dims
    # Map from old dimension ID to its new position (ID) in the full permutation
    old_to_new_id = {
        dims[old_idx].id: new_idx for new_idx, old_idx in enumerate(full_permutation)
    }

    def update_dimension_metadata(dim: Dimension, new_id: int) -> Dimension:
        if isinstance(dim, SparseDimension):
            return replace(dim, id=new_id, other_id=old_to_new_id[dim.other_id])
        return replace(dim, id=new_id)

    new_out_dims = [
        update_dimension_metadata(dims[i], idx) for idx, i in enumerate(out_axes)
    ]
    new_primal_dims = [
        update_dimension_metadata(dims[i], len(out_axes) + idx)
        for idx, i in enumerate(primal_axes)
    ]

    return new_out_dims, new_primal_dims


def transpose(
    tensor: SparseTensor,
    out_axes: Sequence[int] | None = None,
    primal_axes: Sequence[int] | None = None,
) -> SparseTensor:
    """Transpose a SparseTensor by reordering its out and primal dimensions."""
    from .utils import _sort_val  # Late import to avoid cycle

    num_out = len(tensor.out_dims)
    num_primal = len(tensor.primal_dims)

    # 1. Normalize and validate the permutation
    full_permutation, new_out_axes, new_primal_axes = _get_full_permutation(
        num_out, num_primal, out_axes, primal_axes
    )

    # 2. Ensure sparse dimensions remain split across sides; densify if not.
    tensor = _ensure_valid_sparsity(tensor, new_out_axes, new_primal_axes)

    # 3. Update logical dimension metadata
    new_out_dims, new_primal_dims = _remap_logical_dimensions(
        tensor, full_permutation, new_out_axes, new_primal_axes
    )

    # 4. Sort the values based on new dimensions and return
    new_out_dims, new_primal_dims, new_val = _sort_val(
        new_out_dims, new_primal_dims, tensor.val
    )

    from ..tensor import SparseTensor

    return SparseTensor(
        new_out_dims,
        new_primal_dims,
        new_val,
        tensor.scalar_mult,
        tensor.fill_value,
        sort_val=False,
        check_consistency=False,
    )
