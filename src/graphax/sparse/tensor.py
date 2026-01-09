from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class


@dataclass(frozen=True)
class Dimension(ABC):
    """Base class for all dimension types."""

    id: int
    size: int
    val_dim: int | None

    def __post_init__(self):
        if self.size < 0:
            raise ValueError(f"Dimension size must be non-negative, got {self.size}")

    @property
    @abstractmethod
    def logical_size(self) -> int:
        """Total size of the dimension including block structure."""
        pass


@dataclass(frozen=True)
class DenseDimension(Dimension):
    """Represents a dense dimension in the tensor."""

    @property
    def logical_size(self) -> int:
        return self.size


@dataclass(frozen=True)
class SparseDimension(Dimension):
    """Represents a sparse (diagonal) dimension in the tensor."""

    other_id: int
    block_size: int = 1
    block_val_dim: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.block_size <= 0:
            raise ValueError(
                f"SparseDimension block_size must be positive, got {self.block_size}"
            )

    @property
    def logical_size(self) -> int:
        return self.size * self.block_size


Transform = Callable[
    ["BlockSparseTensor", "BlockSparseTensor", Array], "BlockSparseTensor"
]


@register_pytree_node_class
@dataclass(init=False, frozen=True)
class BlockSparseTensor:
    """
    Represents a tensor with block-sparse structure, optimizing memory and computation
    by storing only non-zero blocks.

    Attributes:
        out_dims: Dimensions corresponding to output (rows).
        primal_dims: Dimensions corresponding to input (columns).
        val: The underlying compact data array containing the blocks.
        scalar_mult: Scalar multiplier applied to the tensor.
    """

    out_dims: tuple[Dimension, ...]
    primal_dims: tuple[Dimension, ...]
    val: Array | None
    scalar_mult: Array
    pre_transforms: tuple[Transform, ...]
    post_transforms: tuple[Transform, ...]

    def tree_flatten(self):
        children = (self.val, self.scalar_mult)
        aux_data = (
            self.out_dims,
            self.primal_dims,
            self.pre_transforms,
            self.post_transforms,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        val, scalar_mult = children
        out_dims, primal_dims, pre, post = aux_data
        obj = cls.__new__(cls)
        _immutable_assign_bst(obj, out_dims, primal_dims, val, scalar_mult, pre, post)
        return obj

    def __init__(
        self,
        out_dims: Sequence[Dimension],
        primal_dims: Sequence[Dimension],
        val: Array | None,
        scalar_mult: Array | None = None,
        pre_transforms: Sequence[Callable] = (),
        post_transforms: Sequence[Callable] = (),
        pre: Sequence[Callable] = None,
        post: Sequence[Callable] = None,
    ):
        """
        Initializes a BlockSparseTensor.

        Args:
            out_dims: Row dimensions.
            primal_dims: Column dimensions.
            val: Data array containing block values.
            scalar_mult: Global scalar multiplier.
            pre_transforms: List of pre-computation transforms.
            post_transforms: List of post-computation transforms.
        """
        if scalar_mult is None:
            scalar_mult = jnp.array(1.0)

        if pre is not None:
            pre_transforms = pre
        if post is not None:
            post_transforms = post

        canon_out_dims, canon_primal_dims, canon_val = _sort_val(
            out_dims, primal_dims, val
        )
        _immutable_assign_bst(
            self,
            canon_out_dims,
            canon_primal_dims,
            canon_val,
            scalar_mult,
            pre_transforms,
            post_transforms,
        )
        _assert_consistency(self)

    @property
    def dims(self) -> tuple[Dimension, ...]:
        return self.out_dims + self.primal_dims

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(d.logical_size for d in self.dims)

    @property
    def batch_size(self) -> int:
        """Returns the size of the sparse batch dimension, or 1 if none exists."""
        for d in self.dims:
            if isinstance(d, SparseDimension) and d.val_dim is not None:
                return d.size
        return 1

    @property
    def size(self) -> int:
        return jnp.prod(jnp.array(self.shape))

    def __array__(self, *args, **kwargs):
        import numpy as np

        return np.array(jnp.array(self), *args, **kwargs)

    def __jax_array__(self):
        """Materializes the tensor to a standard dense JAX array."""
        return _to_jax_array(self)

    def dense(
        self, axes: tuple[int, ...] | None = None, hard: bool = False
    ) -> BlockSparseTensor:
        """
        Partially or fully materializes the tensor.

        Args:
            axes: Indices of dimensions to check for materialization overlap.
                  If None, considers all dimensions.
            hard: If True, forces materialization of sparse dimensions into dense ones
                  (diagonalization). Implicit dimensions are always materialized if targeted.

        Returns:
            A new BlockSparseTensor with specified improvements to density.
        """
        return _dense(self, axes, hard)

    @property
    def T(self) -> BlockSparseTensor:
        """Transpose of the tensor. Equivalent to transpose()."""
        return self.transpose()

    def swapdims(self) -> BlockSparseTensor:
        """Swaps output and primal dimensions. Equivalent to transpose()."""
        return self.transpose()
        # TODO placeholder, transpose should reverse the dims, swap dims flips out_dims <-> primal_dims

    def transpose(
        self,
        out_perm: Sequence[int] | None = None,
        primal_perm: Sequence[int] | None = None,
    ) -> BlockSparseTensor:
        return _transpose(self, out_perm, primal_perm)

    def copy(self, val: Array | None = None, scalar_mult: Array | None = None):
        return _copy(self, val, scalar_mult)

    def __matmul__(self, other: Any) -> BlockSparseTensor | Array:
        return _matmul(self, other)

    def __rmatmul__(self, other: Any) -> BlockSparseTensor | Array:
        return _rmatmul(self, other)

    def __copy__(self, val: Array | None = None, scalar_mult: Array | None = None):
        return _copy(self, val, scalar_mult)

    def __deepcopy__(self):
        return _copy(self, deep=True)

    def join_blocks(self, factor: int) -> BlockSparseTensor:
        """Joins adjacent blocks into larger blocks."""
        return _join_blocks(self, factor)

    def split_blocks(self, factor: int = 1, force: bool = False) -> BlockSparseTensor:
        """Splits blocks into smaller sub-blocks."""
        return _split_blocks(self, factor, force)

    def __add__(self, other: Any) -> BlockSparseTensor:
        return _bst_add(self, other)

    def __iadd__(self, other: Any) -> BlockSparseTensor:
        return _bst_add(self, other)

    def __mul__(self, other: Any) -> BlockSparseTensor:
        return _bst_mul(self, other)


def _immutable_assign_bst(
    bst: BlockSparseTensor,
    out_dims,
    primal_dims,
    val,
    scalar_mult,
    pre_transforms,
    post_transforms,
):
    object.__setattr__(bst, "out_dims", out_dims)
    object.__setattr__(bst, "primal_dims", primal_dims)
    object.__setattr__(bst, "val", val)
    object.__setattr__(bst, "scalar_mult", scalar_mult)
    object.__setattr__(bst, "pre_transforms", pre_transforms)
    object.__setattr__(bst, "post_transforms", post_transforms)


def _rmatmul(rhs: BlockSparseTensor, lhs: Any):
    if isinstance(lhs, (Array, jnp.ndarray)):
        lhs_shape = lhs.shape
        n_contract = len(rhs.out_dims)
        if len(lhs_shape) < n_contract:
            raise ValueError(
                f"Cannot contract: rhs has {n_contract} out dims, lhs has rank {len(lhs_shape)}"
            )

        new_out_dims = []
        next_id = max((d.id for d in rhs.dims), default=0) + 1
        val_dim_offset = 0
        for i in range(len(lhs_shape) - n_contract):
            new_out_dims.append(DenseDimension(next_id, lhs_shape[i], val_dim_offset))
            next_id += 1
            val_dim_offset += 1

        new_primal_dims = []
        for i in range(len(lhs_shape) - n_contract, len(lhs_shape)):
            d_rhs = rhs.out_dims[i - (len(lhs_shape) - n_contract)]
            if lhs_shape[i] != d_rhs.logical_size:
                raise ValueError(
                    f"Shape mismatch: dim {d_rhs.id} expects total size {d_rhs.logical_size}, but array has {lhs_shape[i]}"
                )
            new_primal_dims.append(
                DenseDimension(d_rhs.id, lhs_shape[i], val_dim_offset)
            )
            val_dim_offset += 1

        lhs_bst = BlockSparseTensor(new_out_dims, new_primal_dims, lhs)
        return _bst_matmul(lhs_bst, rhs)

    if not isinstance(lhs, BlockSparseTensor):
        if (
            hasattr(lhs, "val")
            and hasattr(lhs, "out_dims")
            and hasattr(lhs, "primal_dims")
        ):
            try:
                lhs = BlockSparseTensor.from_sparse_tensor(lhs)
            except (ValueError, AttributeError):
                return NotImplemented
        else:
            return NotImplemented
    return _bst_matmul(lhs, rhs)


def _update_indices_for_sort(
    dims: Sequence[Dimension], old_to_new: dict[int, int]
) -> tuple[Dimension, ...]:
    """
    Updates dimension indices based on a new sorting permutation.

    Args:
        dims (Sequence[Dimension]): List of dimensions to update.
        old_to_new (dict[int, int]): Mapping from old index to new index.

    Returns:
        tuple[Dimension, ...]: Tuple of updated dimensions.
    """
    updated_dims = []
    for dimension in dims:
        new_val_dim = (
            old_to_new[dimension.val_dim] if dimension.val_dim is not None else None
        )
        new_block_dim = None
        if isinstance(dimension, SparseDimension):
            new_block_dim = (
                old_to_new[dimension.block_val_dim]
                if dimension.block_val_dim is not None
                else None
            )
            updated_dims.append(
                replace(dimension, val_dim=new_val_dim, block_val_dim=new_block_dim)
            )
        else:
            updated_dims.append(replace(dimension, val_dim=new_val_dim))
    return tuple(updated_dims)


def _sort_val(  # TODO add an extension to this to also remove the remainding dims in val.
    out_dims: Sequence[Dimension], primal_dims: Sequence[Dimension], val: Array | None
) -> tuple[tuple[Dimension, ...], tuple[Dimension, ...], Array | None]:
    """
    Canonicalizes the storage of the 'val' array by reordering axes.

    The expected order of axes is:
    1. Sparse Axes (val_dim from SparseDimension)
    2. Dense Axes (val_dim from DenseDimension)
    3. Block Value Axes (block_val_dim from SparseDimension)
    4. Remaining Axes (any other dimensions in val), i.e., the rest is ignored

    Args:
        out_dims: Sequence of output dimensions (row dimensions).
        primal_dims: Sequence of primal dimensions (column dimensions).
        val: The underlying data array, or None if empty.

    Returns:
        A tuple containing:
        - Updated out_dims with canonicalized val_dim/block_val_dim indices.
        - Updated primal_dims with canonicalized val_dim/block_val_dim indices.
        - The reordered val array.
    """
    if val is None:
        return tuple(out_dims), tuple(primal_dims), None

    sparse_axes = {}
    dense_axes = {}
    block_val_axes = {}

    for dim in list(out_dims) + list(primal_dims):
        if isinstance(dim, SparseDimension):
            if dim.val_dim is not None:
                sparse_axes[dim.val_dim] = dim.val_dim
            if dim.block_val_dim is not None:
                block_val_axes[dim.block_val_dim] = dim.block_val_dim
        elif dim.val_dim is not None:
            dense_axes[dim.val_dim] = dim.val_dim

    sorted_sparse_axes = sorted(list(sparse_axes.keys()))
    sorted_dense_axes = sorted(list(set(dense_axes.keys()) - set(sorted_sparse_axes)))
    sorted_block_val_axes = sorted(
        list(
            set(block_val_axes.keys())
            - set(sorted_sparse_axes)
            - set(sorted_dense_axes)
        )
    )

    permutation = sorted_sparse_axes + sorted_dense_axes + sorted_block_val_axes
    permutation += sorted(list(set(range(val.ndim)) - set(permutation)))

    sorted_val = (
        val if permutation == list(range(val.ndim)) else jnp.transpose(val, permutation)
    )
    old_to_new_indices = {old: new for new, old in enumerate(permutation)}

    return (
        _update_indices_for_sort(out_dims, old_to_new_indices),
        _update_indices_for_sort(primal_dims, old_to_new_indices),
        sorted_val,
    )


def _validate_val_range(bst: BlockSparseTensor) -> None:
    """
    Validates that dimension indices do not exceed the bounds of the 'val' array.

    Args:
        bst (BlockSparseTensor): The tensor to validate.

    Raises:
        ValueError: If a dimension refers to an axis outside `val.ndim`.
    """
    if bst.val is not None:
        max_axis = -1
        for dim in bst.dims:
            if dim.val_dim is not None:
                max_axis = max(max_axis, dim.val_dim)
            if isinstance(dim, SparseDimension) and dim.block_val_dim is not None:
                max_axis = max(max_axis, dim.block_val_dim)

        if bst.val.ndim <= max_axis:
            raise ValueError(
                f"Dimension refers to axis {max_axis}, but val.ndim is {bst.val.ndim}"
            )


def _validate_sparse_partners(bst: BlockSparseTensor) -> None:
    """
    Validates that sparse dimensions are correctly paired.

    Args:
        bst (BlockSparseTensor): The tensor to validate.

    Raises:
        ValueError: If a sparse dimension is paired with a dense dimension or if pairing is asymmetric.
    """
    id_to_dim = {d.id: d for d in bst.dims}
    for dim in bst.dims:
        if isinstance(dim, SparseDimension) and dim.other_id in id_to_dim:
            other = id_to_dim[dim.other_id]
            if not isinstance(other, SparseDimension):
                raise ValueError(
                    f"Asymmetric: sparse dim {dim.id} paired with dense {other.id}"
                )
            if other.other_id != dim.id:
                raise ValueError(
                    f"Asymmetric pairing: {dim.id}->{other.id}, but {other.id}->{other.other_id}"
                )


def _validate_dimension_consistency(bst: BlockSparseTensor) -> None:
    """
    Validates consistency of shared dimensions and block sizes.

    Args:
        bst (BlockSparseTensor): The tensor to validate.

    Raises:
        ValueError: If shared dimensions have different sizes or mismatch the 'val' shape.
    """
    val_dim_to_dims: dict[int, list[Dimension]] = {}
    for d in bst.dims:
        if d.val_dim is not None:
            val_dim_to_dims.setdefault(d.val_dim, []).append(d)

    for dim in bst.dims:
        if dim.val_dim is not None:
            for sharing_dim in val_dim_to_dims[dim.val_dim]:
                if sharing_dim.size != dim.size:
                    raise ValueError(
                        f"Inconsistent size for shared val_dim {dim.val_dim}"
                    )
            if bst.val is not None:
                val_size = bst.val.shape[dim.val_dim]
                if val_size != dim.size and val_size != 1:
                    raise ValueError(
                        f"Size mismatch: dim {dim.id} has size {dim.size} but val.shape[{dim.val_dim}] is {val_size}"
                    )

        if (
            isinstance(dim, SparseDimension)
            and dim.block_val_dim is not None
            and bst.val is not None
        ):
            if bst.val.shape[dim.block_val_dim] != dim.block_size:
                raise ValueError(
                    f"Block size mismatch for dim {dim.id}: "
                    f"{bst.val.shape=}, "
                    f"{dim.block_val_dim=}, "
                    f"{dim.block_size=}"
                )


def _assert_consistency(block_sparse_tensor: BlockSparseTensor) -> None:
    """
    Verifies internal consistency of the BlockSparseTensor metadata and data.
    """
    _validate_val_range(block_sparse_tensor)
    _validate_sparse_partners(block_sparse_tensor)
    _validate_dimension_consistency(block_sparse_tensor)


def _update_dims_join(dims: Sequence[Dimension], factor: int) -> tuple[Dimension, ...]:
    """
    Updates dimension metadata when joining blocks.

    Args:
        dims (Sequence[Dimension]): List of dimensions.
        factor (int): Factor by which to join blocks (divide size, multiply block_size).

    Returns:
        tuple[Dimension, ...]: Updated dimensions.
    """
    updated = []
    for d in dims:
        if isinstance(d, SparseDimension) and d.val_dim is not None:
            updated.append(
                replace(d, size=d.size // factor, block_size=d.block_size * factor)
            )
        else:
            updated.append(d)
    return tuple(updated)


def _join_vals_1d(values: Array, new_batch: int, factor: int) -> Array:
    """
    Reshape and diagonalize 1D values for block joining.

    Args:
        values (Array): Input 1D value array.
        new_batch (int): New batch size.
        factor (int): Joining factor.

    Returns:
        Array: 3D array of joined 1D values (diagonalized).
    """
    reshaped_values = values.reshape(new_batch, factor)
    return jax.vmap(jnp.diag)(reshaped_values)


def _join_vals_nd(values: Array, new_batch: int, factor: int) -> Array:
    """
    Rearrange ND values into larger blocks.

    Args:
        values (Array): Input value array.
        new_batch (int): New batch size.
        factor (int): The factor by which blocks are joined.

    Returns:
        Array: New value array with merged blocks.
    """
    block_height, block_width = values.shape[-2], values.shape[-1]
    reshaped_values = values.reshape(
        (new_batch, factor) + values.shape[1:-2] + (block_height, block_width)
    )
    new_block_height, new_block_width = factor * block_height, factor * block_width
    final_values = jnp.zeros(
        (new_batch,) + values.shape[1:-2] + (new_block_height, new_block_width),
        dtype=values.dtype,
    )
    for i in range(factor):
        final_values = final_values.at[
            ...,
            i * block_height : (i + 1) * block_height,
            i * block_width : (i + 1) * block_width,
        ].set(reshaped_values[:, i, ...])
    return final_values


# TODO make this specific to a certain set of axes.
def _join_blocks(tensor: BlockSparseTensor, factor: int) -> BlockSparseTensor:
    """Joins adjacent blocks into larger blocks."""
    if factor == 1:
        return tensor
    current_batch = tensor.batch_size
    if current_batch % factor != 0:
        raise ValueError(f"Cannot join {current_batch} blocks by factor {factor}")
    new_batch = current_batch // factor

    new_out = _update_dims_join(tensor.out_dims, factor)
    new_primal = _update_dims_join(tensor.primal_dims, factor)

    if tensor.val is None:
        return BlockSparseTensor(new_out, new_primal, None, tensor.scalar_mult)
    values = tensor.val

    if values.ndim == 1:
        joined_values = _join_vals_1d(values, new_batch, factor)
    else:
        joined_values = _join_vals_nd(values, new_batch, factor)

    return BlockSparseTensor(new_out, new_primal, joined_values, tensor.scalar_mult)


def _update_dims_split(dims: Sequence[Dimension], factor: int) -> tuple[Dimension, ...]:
    """
    Updates dimension metadata when splitting blocks.

    Args:
        dims (Sequence[Dimension]): List of dimensions.
        factor (int): Factor by which to split blocks (multiply size, divide block_size).

    Returns:
        tuple[Dimension, ...]: Updated dimensions.
    """
    updated = []
    for d in dims:
        if isinstance(d, SparseDimension):
            updated.append(
                replace(d, size=d.size * factor, block_size=d.block_size // factor)
            )
        else:
            updated.append(d)
    return tuple(updated)


def _split_vals_nd(values: Array, factor: int, current_batch: int) -> Array:
    """
    Splits ND values into smaller blocks.

    Args:
        values (Array): Input value array.
        factor (int): Split factor.
        current_batch (int): Current batch size.
        force (bool): If True, ignores non-zero off-diagonal elements.

    Returns:
        Array: New value array with split blocks.

    Raises:
        ValueError: If off-diagonal sub-blocks are non-zero and force=False.
    """
    block_height, block_width = values.shape[-2], values.shape[-1]
    sub_block_height, sub_block_width = (
        block_height // factor,
        block_width // factor,
    )
    reshaped_values = values.reshape(
        values.shape[:-2] + (factor, sub_block_height, factor, sub_block_width)
    )
    new_values_parts = [reshaped_values[..., i, :, i, :] for i in range(factor)]
    stacked = jnp.stack(new_values_parts, axis=1)
    new_shape = (
        (current_batch * factor,)
        + values.shape[1:-2]
        + (sub_block_height, sub_block_width)
    )
    return stacked.reshape(new_shape)


# TODO make this specific to a certain set of axes.
def _split_blocks(
    tensor: BlockSparseTensor, factor: int = 1, force: bool = False
) -> BlockSparseTensor:
    """Splits blocks into smaller sub-blocks."""
    if factor == 1:
        return tensor
    current_batch = tensor.batch_size
    for d in tensor.dims:
        if isinstance(d, SparseDimension) and d.block_size % factor != 0:
            raise ValueError(
                f"Block size {d.block_size} not divisible by factor {factor} for dim {d.id}"
            )

    new_out = _update_dims_split(tensor.out_dims, factor)
    new_primal = _update_dims_split(tensor.primal_dims, factor)

    if tensor.val is None:
        return BlockSparseTensor(new_out, new_primal, None, tensor.scalar_mult)
    values = tensor.val

    if values.ndim == 1:
        split_values = values.reshape(values.shape[0], factor).flatten()
    else:
        split_values = _split_vals_nd(values, factor, current_batch, force)

    return BlockSparseTensor(new_out, new_primal, split_values, tensor.scalar_mult)


def _infer_block_dims(val: Array | None, dims: Sequence[Dimension]) -> dict[int, int]:
    """
    Infers which physical axes correspond to the block component of SparseDimensions.

    Args:
        val (Array | None): The value array.
        dims (Sequence[Dimension]): List of dimensions.

    Returns:
        dict[int, int]: Mapping from dimension ID to inferred block axis index in `val`.
    """
    inferred = {}
    if val is not None:
        used = {d.val_dim for d in dims if d.val_dim is not None}
        used |= {
            d.block_val_dim
            for d in dims
            if isinstance(d, SparseDimension) and d.block_val_dim is not None
        }
        unused = sorted(list(set(range(val.ndim)) - used))
        curr_unused = 0
        for d in dims:
            if (
                isinstance(d, SparseDimension)
                and d.block_size > 1
                and d.block_val_dim is None
            ):
                for i in range(curr_unused, len(unused)):
                    ax = unused[i]
                    if val.shape[ax] == d.block_size:
                        inferred[d.id], curr_unused = ax, i + 1
                        break
    return inferred


def _build_axis_mapping(
    dims: Sequence[Dimension], inferred_block_dims: dict[int, int]
) -> dict[int, set[tuple[str, int]]]:
    """
    Builds a mapping from physical axes to logical dimensions.

    Args:
        dims (Sequence[Dimension]): List of logical dimensions.
        inferred_block_dims (dict[int, int]): Inferred block axes.

    Returns:
        dict[int, set[tuple[str, int]]]: Mapping from physical axis index to set of (type, dim_index) tuples.
    """
    mapping = {}
    for i, d in enumerate(dims):
        if d.val_dim is not None:
            mapping.setdefault(d.val_dim, set()).add(("v", i))
        if isinstance(d, SparseDimension):
            b_dim = (
                d.block_val_dim
                if d.block_val_dim is not None
                else inferred_block_dims.get(d.id)
            )
            if b_dim is not None:
                mapping.setdefault(b_dim, set()).add(("b", i))
    return mapping


def _update_dims_for_densify(
    dims: Sequence[Dimension],
    off: int,
    to_broadcast: list[Dimension],
    to_broadcast_map: dict[int, int],
    id_map: dict[int, Dimension],
    to_broadcast_ids: list[int],
) -> tuple[Dimension, ...]:
    """
    Updates dimension indices after densification (broadcasting).

    Args:
        dims (Sequence[Dimension]): List of dimensions to update.
        off (int): Offset for new dimensions (number of broadcasted dims).
        to_broadcast (list[Dimension]): List of dimensions being broadcasted.
        to_broadcast_map (dict[int, int]): Map from dim ID to new index in broadcasted part.
        id_map (dict[int, Dimension]): Map of all dimension IDs to objects.
        to_broadcast_ids (list[int]): List of IDs being broadcasted.

    Returns:
        tuple[Dimension, ...]: Updated dimensions.
    """
    res = []
    for d in dims:
        nv = d.val_dim + off if d.val_dim is not None else None
        nb = (
            d.block_val_dim + off
            if isinstance(d, SparseDimension) and d.block_val_dim is not None
            else None
        )

        if nv is None:
            match_idx = -1
            if d.id in to_broadcast_map:
                match_idx = to_broadcast_map[d.id]
            else:
                if isinstance(d, SparseDimension):
                    if d.other_id in to_broadcast_map:
                        match_idx = to_broadcast_map[d.other_id]
                    else:
                        match = next(
                            (
                                i
                                for i, tid in enumerate(to_broadcast_ids)
                                if tid == d.other_id
                            ),
                            -1,
                        )
                        if match != -1:
                            match_idx = match

            if match_idx != -1:
                nv = match_idx
            elif isinstance(d, SparseDimension):
                p = id_map.get(d.other_id)
                if p and p.val_dim is not None:
                    nv = p.val_dim + off

        if isinstance(d, SparseDimension):
            res.append(replace(d, val_dim=nv, block_val_dim=nb))
        else:
            res.append(replace(d, val_dim=nv))
    return tuple(res)


def _densify_implicits(
    bst: BlockSparseTensor, targets: Sequence[Dimension]
) -> BlockSparseTensor:
    """Materializes implicit dimensions by broadcasting 'val'."""
    to_broadcast, seen_ids, id_map = [], set(), {d.id: d for d in bst.dims}
    for dim in targets:
        if dim.id in seen_ids:
            continue
        if isinstance(dim, SparseDimension):
            partner = id_map.get(dim.other_id)
            if partner and partner.val_dim is not None:
                seen_ids.add(dim.id)
                continue
            if partner:
                seen_ids.add(partner.id)
        to_broadcast.append(dim)
        seen_ids.add(dim.id)

    val = bst.val if bst.val is not None else jnp.array(1.0)
    b_sizes = tuple(d.size for d in to_broadcast) + val.shape
    b_axes = tuple(range(len(to_broadcast), len(to_broadcast) + val.ndim))
    new_val = jax.lax.broadcast_in_dim(val, b_sizes, b_axes)
    off = len(to_broadcast)

    to_broadcast_map = {d.id: i for i, d in enumerate(to_broadcast)}
    to_broadcast_ids = [d.id for d in to_broadcast]

    new_out = _update_dims_for_densify(
        bst.out_dims, off, to_broadcast, to_broadcast_map, id_map, to_broadcast_ids
    )
    new_primal = _update_dims_for_densify(
        bst.primal_dims, off, to_broadcast, to_broadcast_map, id_map, to_broadcast_ids
    )

    return BlockSparseTensor(new_out, new_primal, new_val, bst.scalar_mult)


def _identify_diagonal_partners(
    bst: BlockSparseTensor, targets: Sequence[SparseDimension]
) -> tuple[list[tuple[SparseDimension, SparseDimension | None]], set[int]]:
    """
    Identifies diagonal partners for list of sparse dimensions.

    Args:
        bst (BlockSparseTensor): The tensor containing dimensions.
        targets (Sequence[SparseDimension]): List of dimensions to find partners for.

    Returns:
        tuple: A tuple containing:
            - List of pairs (dimension, partner).
            - Set of all seen dimension IDs.
    """
    seen = set()
    pairs = []

    id_map = {d.id: d for d in bst.dims}

    for t in bst.dims:
        if t in targets and t.id not in seen:
            partner = None
            if isinstance(t, SparseDimension):
                cand = id_map.get(t.other_id)
                if cand:
                    if not isinstance(cand, SparseDimension) or cand.other_id == t.id:
                        partner = cand

            if partner:
                pairs.append((t, partner) if t in bst.out_dims else (partner, t))
                seen.update([t.id, t.other_id])
            else:
                pairs.append((t, None))
                seen.add(t.id)
    return pairs, seen


def _expand_shared_axis_diagonal(
    new_v: Array,
    amap: dict[int, set[tuple[str, int]]],
    r_ax: int,
    r_idx: int,
    c_idx: int,
    bst: BlockSparseTensor,
) -> tuple[Array, dict[int, set[tuple[str, int]]]]:
    """
    Expands a shared axis into two axes (diagonal expansion).

    Args:
        new_v (Array): Current value array.
        amap (dict): Axis mapping.
        r_ax (int): Range axis index.
        r_idx (int): Row dimension index.
        c_idx (int): Column dimension index.

    Returns:
        tuple[Array, dict]: Updated value array and axis mapping.
    """
    size = bst.dims[r_idx].size
    new_v = jnp.expand_dims(new_v, r_ax + 1)
    ident = jnp.eye(size)
    resh_id = ident.reshape([1] * r_ax + [size, size] + [1] * (new_v.ndim - r_ax - 2))
    new_v *= resh_id

    n_map = {}
    for ax in sorted(amap.keys()):
        n_ax = ax if ax <= r_ax else ax + 1
        if ax == r_ax:
            r_its = {it for it in amap[ax] if it[1] == r_idx}
            c_its = {it for it in amap[ax] if it[1] == c_idx} if c_idx != -1 else set()
            others = amap[ax] - r_its - c_its
            n_map[r_ax], n_map[r_ax + 1] = r_its | others, c_its
        else:
            n_map[n_ax] = amap[ax]
    return new_v, n_map


def _mask_separate_axes_diagonal(
    new_v: Array, r_ax: int, c_ax: int, r_idx: int, c_idx: int, bst: BlockSparseTensor
) -> Array:
    """
    Masks off-diagonal elements for separate axes.

    Args:
        new_v (Array): Current value array.
        r_ax (int): Row axis index.
        c_ax (int): Column axis index.

    Returns:
        Array: Masked value array.
    """
    size = bst.dims[r_idx].size
    idx_r = jnp.arange(size).reshape(
        [1] * r_ax + [size] + [1] * (new_v.ndim - r_ax - 1)
    )
    idx_c = jnp.arange(size).reshape(
        [1] * c_ax + [size] + [1] * (new_v.ndim - c_ax - 1)
    )
    new_v *= idx_r == idx_c
    return new_v


def _apply_diagonal_expansion(
    val: Array,
    amap: dict[int, set[tuple[str, int]]],
    pairs: list[tuple[SparseDimension, SparseDimension | None]],
    bst: BlockSparseTensor,
) -> tuple[Array, dict[int, set[tuple[str, int]]]]:
    new_v = val
    for row_d, col_d in pairs:
        r_idx = bst.dims.index(row_d)
        c_idx = bst.dims.index(col_d) if col_d is not None else -1

        r_ax = next((ax for ax, its in amap.items() if ("v", r_idx) in its), None)
        c_ax = next((ax for ax, its in amap.items() if ("v", c_idx) in its), None)

        if r_ax is not None and r_ax == c_ax:
            new_v, amap = _expand_shared_axis_diagonal(
                new_v, amap, r_ax, r_idx, c_idx, bst
            )
        elif r_ax is not None and c_ax is not None:
            new_v = _mask_separate_axes_diagonal(new_v, r_ax, c_ax, r_idx, c_idx, bst)
    return new_v, amap


def _split_axis(
    val: Array,
    amap: dict[int, set[tuple[str, int]]],
    ax: int,
    keep_entries: set[tuple[str, int]],
    bst: BlockSparseTensor,
) -> tuple[Array, dict[int, set[tuple[str, int]]]]:
    """
    Splits an axis into two using diagonal expansion logic.
    Entries in `keep_entries` stay at `ax`. Others move to `ax + 1`.
    """
    it = next(iter(keep_entries))
    size = bst.dims[it[1]].size
    ident = jnp.eye(size)
    sh = list(val.shape)
    sh.insert(ax + 1, 1)
    resh_id = ident.reshape([1] * ax + [size, size] + [1] * (len(sh) - ax - 2))
    new_v = val.reshape(sh) * resh_id
    n_map = {}
    for x in sorted(amap.keys()):
        n_x = x if x <= ax else x + 1
        if x == ax:
            kept = {it for it in amap[ax] if it in keep_entries}
            moved = amap[ax] - kept
            n_map[ax] = kept
            n_map[ax + 1] = moved
        else:
            n_map[n_x] = amap[x]

    return new_v, n_map


def _ensure_axis_exclusive(
    val: Array, amap: dict, dim_idx: int, type_tag: str, bst: BlockSparseTensor
) -> tuple[Array, dict]:
    """Ensures the physical axis for (type_tag, dim_idx) is not shared with others."""
    ax = next((a for a, its in amap.items() if (type_tag, dim_idx) in its), None)
    if ax is None:
        return val, amap

    entries = amap[ax]
    target = (type_tag, dim_idx)

    if len(entries) > 1:
        val, amap = _split_axis(val, amap, ax, {target}, bst)

    return val, amap


def _perform_single_merge(
    new_v: Array, amap: dict, merges: list[tuple[int, int, int]]
) -> tuple[Array, dict, list[tuple[int, int, int]]]:
    """
    Performs a single merge of a value axis and a block axis.

    Args:
        new_v (Array): Current value array.
        amap (dict): Axis mapping.
        merges (list): List of pending merges (dim_idx, val_ax, block_ax).

    Returns:
        tuple[Array, dict, list]: Tuple of updated value array, axis mapping, and remaining merges.
    """
    m_idx, v_ax, b_ax = max(merges, key=lambda x: max(x[1], x[2]))
    merges.remove((m_idx, v_ax, b_ax))
    mx, mn = max(v_ax, b_ax), min(v_ax, b_ax)

    perm = list(range(new_v.ndim))
    perm.remove(mx)
    perm.insert(mn + 1, mx)
    new_v = new_v.transpose(perm)

    sh = list(new_v.shape)
    m_size = sh[mn] * sh[mn + 1]
    new_v = new_v.reshape(sh[:mn] + [m_size] + sh[mn + 2 :])

    amap[mn] |= amap[mx]
    del amap[mx]
    amap = {(ax if ax < mx else ax - 1): its for ax, its in amap.items()}

    new_merges = []
    for mi, mv, mb in merges:
        new_merges.append((mi, mv if mv < mx else mv - 1, mb if mb < mx else mb - 1))
    return new_v, amap, new_merges


def _merge_diagonal_block_dims(
    val: Array,
    amap: dict[int, set[tuple[str, int]]],
    bst: BlockSparseTensor,
    seen_ids: set[int],
) -> tuple[Array, dict[int, set[tuple[str, int]]]]:
    new_v = val
    dims = bst.dims

    for i, d in enumerate(dims):
        if d.id in seen_ids:
            new_v, amap = _ensure_axis_exclusive(new_v, amap, i, "v", bst)
            new_v, amap = _ensure_axis_exclusive(new_v, amap, i, "b", bst)

    merges = []
    for i, d in enumerate(dims):
        if d.id in seen_ids:
            v_ax = next((ax for ax, its in amap.items() if ("v", i) in its), None)
            b_ax = next((ax for ax, its in amap.items() if ("b", i) in its), None)
            if v_ax is not None and b_ax is not None:
                merges.append((i, v_ax, b_ax))

    while merges:
        new_v, amap, merges = _perform_single_merge(new_v, amap, merges)
    return new_v, amap


def _reconstruct_diagonalized_dims(
    dims: Sequence[Dimension],
    offset: int,
    seen_ids: set[int],
    amap: dict[int, set[tuple[str, int]]],
) -> tuple[Dimension, ...]:
    """
    Reconstructs dimension objects after diagonalization.

    Args:
        dims (Sequence[Dimension]): Original dimensions.
        offset (int): Offset for indexing (e.g. for primal dims).
        seen_ids (set[int]): Set of IDs involved in diagonalization.
        amap (dict): Axis mapping.

    Returns:
        tuple[Dimension, ...]: Updated dimensions.
    """
    res = []
    for i, d in enumerate(dims):
        idx = i + offset
        if d.id in seen_ids:
            v_s = [ax for ax, its in amap.items() if ("v", idx) in its]
            if not v_s:
                v_s = [
                    ax for ax, its in amap.items() if any(it[1] == idx for it in its)
                ]
            res.append(DenseDimension(d.id, d.size * d.block_size, v_s[0]))
        elif d.val_dim is not None:
            v_s = [ax for ax, its in amap.items() if ("v", idx) in its]
            res.append(replace(d, val_dim=v_s[0]) if v_s else d)
        else:
            res.append(d)
    return tuple(res)


def _diagonalize_sparse_dims(
    bst: BlockSparseTensor, targets: Sequence[SparseDimension]
) -> BlockSparseTensor:
    """Converts sparse dimensions to dense ones by expanding with identity matrices."""
    bst = _dense(bst, axes=tuple(bst.dims.index(t) for t in targets), hard=False)

    pairs, seen = _identify_diagonal_partners(bst, targets)

    inf = _infer_block_dims(bst.val, bst.dims)
    amap = _build_axis_mapping(bst.dims, inf)

    new_v, amap = _apply_diagonal_expansion(bst.val, amap, pairs, bst)
    new_v, amap = _merge_diagonal_block_dims(new_v, amap, bst, seen)

    return BlockSparseTensor(
        _reconstruct_diagonalized_dims(bst.out_dims, 0, seen, amap),
        _reconstruct_diagonalized_dims(bst.primal_dims, len(bst.out_dims), seen, amap),
        new_v,
        bst.scalar_mult,
    )


def _transpose(
    tensor: BlockSparseTensor,
    out_perm: Sequence[int] | None = None,
    primal_perm: Sequence[int] | None = None,
) -> BlockSparseTensor:
    """
    Transposes the BlockSparseTensor.

    Args:
        tensor (BlockSparseTensor): The tensor to transpose.
        out_perm (Sequence[int] | None): Permutation for output dimensions.
        primal_perm (Sequence[int] | None): Permutation for primal dimensions.

    Returns:
        BlockSparseTensor: Transposed tensor.

    Raises:
        ValueError: If only one permutation is provided.
    """
    n_out = len(tensor.out_dims)
    n_primal = len(tensor.primal_dims)

    if out_perm is None and primal_perm is None:
        out_perm = tuple(range(n_out, n_out + n_primal))
        primal_perm = tuple(range(n_out))
    elif out_perm is None or primal_perm is None:
        raise ValueError("Must provide both out_perm and primal_perm, or neither")

    all_dims = tensor.out_dims + tensor.primal_dims
    new_out = tuple(all_dims[i] for i in out_perm)
    new_primal = tuple(all_dims[i] for i in primal_perm)

    new_out_c, new_primal_c, new_val = _sort_val(new_out, new_primal, tensor.val)
    return BlockSparseTensor(new_out_c, new_primal_c, new_val, tensor.scalar_mult)


def _dense(
    tensor: BlockSparseTensor, axes: tuple[int, ...] | None = None, hard: bool = False
) -> BlockSparseTensor:
    """Partially or fully materializes the tensor."""
    target_indices = axes if axes is not None else range(len(tensor.dims))
    target_ids = {tensor.dims[i].id for i in target_indices}

    implicit_targets = [
        d for d in tensor.dims if d.id in target_ids and d.val_dim is None
    ]
    current_tensor = tensor
    if implicit_targets:
        current_tensor = _densify_implicits(current_tensor, implicit_targets)

    if hard:
        sparse_targets = [  # TODO maybe we can make this a function/property of Sparse/DenseDimensions
            d
            for d in current_tensor.dims
            if d.id in target_ids and isinstance(d, SparseDimension)
        ]
        if sparse_targets:
            partner_ids = {d.other_id for d in sparse_targets}
            implicit_partners = [
                i
                for i, d in enumerate(current_tensor.dims)
                if d.id in partner_ids and d.val_dim is None
            ]
            if implicit_partners:
                current_tensor = _dense(
                    current_tensor, tuple(implicit_partners), hard=False
                )
                sparse_targets = [
                    d
                    for d in current_tensor.dims
                    if d.id in target_ids and isinstance(d, SparseDimension)
                ]

            current_tensor = _diagonalize_sparse_dims(current_tensor, sparse_targets)
    return current_tensor


# TODO break this apart
def _to_jax_array(tensor: BlockSparseTensor) -> Array:
    """Materializes the tensor to a standard dense JAX array."""
    has_sparse = any(isinstance(d, SparseDimension) for d in tensor.dims)
    has_implicits = any(d.val_dim is None for d in tensor.dims)

    shared_phys = False
    phys_to_ids = {}
    for d in tensor.dims:
        if d.val_dim is not None:
            if d.val_dim in phys_to_ids:
                shared_phys = True
                break
            phys_to_ids[d.val_dim] = d.id

    dense_tensor = tensor
    if has_implicits or has_sparse or shared_phys:
        dense_tensor = _dense(tensor, hard=True)

    val = (
        dense_tensor.val if dense_tensor.val is not None else jnp.array(1.0)
    ) * dense_tensor.scalar_mult

    active_phys_axes = []
    for d in dense_tensor.dims:
        if d.val_dim is not None and d.val_dim not in active_phys_axes:
            active_phys_axes.append(d.val_dim)

    all_phys = list(range(val.ndim))
    unused = [ax for ax in all_phys if ax not in active_phys_axes]

    new_val = val.transpose(active_phys_axes + unused)

    curr_shape = []
    phys_idx = 0
    for d in dense_tensor.dims:
        if d.val_dim is not None:
            if d.val_dim in active_phys_axes:
                if active_phys_axes.index(d.val_dim) == phys_idx:
                    curr_shape.append(d.logical_size)
                    phys_idx += 1
                else:
                    curr_shape.append(1)
            else:
                curr_shape.append(1)
        else:
            curr_shape.append(1)

    try:
        if new_val.size == dense_tensor.size:
            return new_val.reshape(dense_tensor.shape)

        res_val = new_val.reshape(curr_shape)
        return jnp.broadcast_to(res_val, dense_tensor.shape)
    except (ValueError, TypeError) as e:
        return jnp.broadcast_to(new_val, dense_tensor.shape)


def _matmul(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor | Array
) -> BlockSparseTensor:
    if isinstance(rhs, Array):
        rhs_shape = rhs.shape
        contracting_dims = lhs.primal_dims
        n_contract = len(contracting_dims)
        if len(rhs_shape) < n_contract:
            raise ValueError(
                f"Cannot contract: lhs has {n_contract} primal dims, rhs has rank {len(rhs_shape)}"
            )
        new_out_dims = []
        target_shape = []
        val_dim_offset = 0

        for i in range(n_contract):
            d_lhs = contracting_dims[i]
            orig_len = rhs_shape[i]

            expected_size = d_lhs.logical_size
            if orig_len != expected_size:
                raise ValueError(
                    f"Shape mismatch: dim {d_lhs.id} expects total size {expected_size}, but array has {orig_len}"
                )

            target_shape.append(orig_len)
            new_out_dims.append(DenseDimension(d_lhs.id, orig_len, val_dim_offset))
            val_dim_offset += 1

        new_primal_dims = []
        used_ids = {d.id for d in lhs.dims}
        next_id = max(used_ids) + 1 if used_ids else 0

        for i in range(n_contract, len(rhs_shape)):
            target_shape.append(rhs_shape[i])
            new_primal_dims.append(
                DenseDimension(next_id, rhs_shape[i], val_dim_offset)
            )
            val_dim_offset += 1
            next_id += 1

        rhs_reshaped = rhs.reshape(target_shape)
        rhs_bst = BlockSparseTensor(new_out_dims, new_primal_dims, rhs_reshaped)
        return _bst_matmul(lhs, rhs_bst)
    if not isinstance(rhs, BlockSparseTensor):
        # if (
        #     hasattr(rhs, "val")
        #     and hasattr(rhs, "out_dims")
        #     and hasattr(rhs, "primal_dims")
        # ):
        #     try:
        #         rhs = BlockSparseTensor.from_sparse_tensor(rhs)
        #     except (ValueError, AttributeError):
        #         return NotImplemented
        # else:
        #     return NotImplemented
        raise ValueError(
            "Invalid type: rhs must be either an Array or a BlockSparseTensor"
        )
    return _bst_matmul(lhs, rhs)


def _align_blocks(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor
) -> tuple[BlockSparseTensor, BlockSparseTensor]:
    """
    Simulates block alignment by densifying or joining blocks to ensure compatible batch sizes.

    Args:
        lhs (BlockSparseTensor): Left-hand side tensor.
        rhs (BlockSparseTensor): Right-hand side tensor.

    Returns:
        tuple[BlockSparseTensor, BlockSparseTensor]: A tuple of (lhs, rhs) with compatible block batch sizes.
    """
    lhs_batch, rhs_batch = lhs.batch_size, rhs.batch_size
    if lhs_batch == rhs_batch:
        return lhs, rhs
    if lhs_batch > 1 and rhs_batch == 1:
        return lhs, rhs.dense((), False)
    if rhs_batch > 1 and lhs_batch == 1:
        return lhs.dense((), False), rhs
    if lhs_batch > rhs_batch and lhs_batch % rhs_batch == 0:
        return lhs.join_blocks(lhs_batch // rhs_batch), rhs
    if rhs_batch > lhs_batch and rhs_batch % lhs_batch == 0:
        return lhs, rhs.join_blocks(rhs_batch // lhs_batch)
    return lhs, rhs


def _find_shared_contracting_ids(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor
) -> set[int]:
    """
    Finds common dimension IDs with matching logical sizes.

    Args:
        lhs (BlockSparseTensor): Left-hand side tensor.
        rhs (BlockSparseTensor): Right-hand side tensor.

    Returns:
        set[int]: Set of shared dimension IDs valid for contraction.
    """
    lhs_primal_map = {d.id: d for d in lhs.primal_dims}
    rhs_out_map = {d.id: d for d in rhs.out_dims}
    contracting_ids = set(lhs_primal_map.keys()) & set(rhs_out_map.keys())

    valid_contracting = set()
    for cid in contracting_ids:
        if lhs_primal_map[cid].logical_size == rhs_out_map[cid].logical_size:
            valid_contracting.add(cid)
    return valid_contracting


def _remap_rhs_dimensions(
    rhs: BlockSparseTensor, id_map: dict[int, int], shift: int
) -> BlockSparseTensor:
    """
    Remaps rhs dimension IDs according to id_map.

    Args:
        rhs (BlockSparseTensor): rhs tensor.
        id_map (dict[int, int]): Map from old ID to new ID.
        shift (int): Shift value for non-mapped IDs.

    Returns:
        BlockSparseTensor: rhs tensor with remapped IDs.
    """

    def remap_dims(dims):
        new_dims = []
        for d in dims:
            new_id = id_map.get(d.id, d.id)
            if isinstance(d, SparseDimension):
                new_dims.append(
                    replace(
                        d,
                        id=new_id,
                        other_id=id_map.get(d.other_id, d.other_id + shift),
                    )
                )
            else:
                new_dims.append(replace(d, id=new_id))
        return tuple(new_dims)

    return BlockSparseTensor(
        remap_dims(rhs.out_dims),
        remap_dims(rhs.primal_dims),
        rhs.val,
        rhs.scalar_mult,
    )


def _infer_positional_contracting(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor
) -> tuple[set[int], BlockSparseTensor]:
    """
    Attempts to infer contracting dimensions by position (suffix of lhs matches prefix of rhs).

    Args:
        lhs (BlockSparseTensor): lhs tensor.
        rhs (BlockSparseTensor): rhs tensor.

    Returns:
        tuple[set[int], BlockSparseTensor]: Tuple of (contracting IDs, updated rhs tensor with remapped IDs).
    """
    if not lhs.primal_dims or not rhs.out_dims:
        return set(), rhs

    lhs_primal, rhs_out = lhs.primal_dims, rhs.out_dims
    max_k = min(len(lhs_primal), len(rhs_out))
    match_k = 0

    for k in range(max_k, 0, -1):
        if tuple(d.logical_size for d in lhs_primal[-k:]) == tuple(
            d.logical_size for d in rhs_out[:k]
        ):
            match_k = k
            break

    if match_k > 0:
        max_lhs_id = max(d.id for d in lhs.dims)
        shift = max_lhs_id + 1000
        id_map = {}
        target_ids = [d.id for d in lhs_primal[-match_k:]]

        for i, d in enumerate(rhs.out_dims):
            new_id = target_ids[i] if i < match_k else d.id + shift
            id_map[d.id] = new_id

        for d in rhs.primal_dims:
            if d.id not in id_map:
                id_map[d.id] = d.id + shift

        rhs = _remap_rhs_dimensions(rhs, id_map, shift)
        return set(target_ids), rhs

    return set(), rhs


def _validate_contraction_compatibility(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor, contracting_ids: set[int]
) -> None:
    """
    Validates size and block size consistency for contracting dimensions.

    Args:
        lhs (BlockSparseTensor): lhs tensor.
        rhs (BlockSparseTensor): rhs tensor.
        contracting_ids (set[int]): Set of IDs to contract.

    Raises:
        ValueError: If size or block size mismatch found.
        KeyError: If contracting ID missing in rhs.
    """
    lhs_primal_map = {d.id: d for d in lhs.primal_dims}
    rhs_out_map = {d.id: d for d in rhs.out_dims}

    for cid in contracting_ids:
        l_d, r_d = lhs_primal_map[cid], rhs_out_map.get(cid)
        if r_d is None:
            raise KeyError(f"Contracting ID {cid} not found in rhs after mapping")
        if l_d.logical_size != r_d.logical_size:
            raise ValueError(f"Size mismatch: {cid}")
        if (
            isinstance(l_d, SparseDimension)
            and isinstance(r_d, SparseDimension)
            and l_d.block_size != r_d.block_size
        ):
            raise ValueError(f"Block size mismatch: {cid}")


def _get_contracting_ids(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor
) -> tuple[set[int], BlockSparseTensor]:
    """
    Identifies dimensions to contract between lhs and rhs tensors.

    Decomposed into:
    1. Explicit ID matching.
    2. Positional fallback (if no explicit match).
    3. Validation.

    Args:
        lhs (BlockSparseTensor): lhs tensor.
        rhs (BlockSparseTensor): rhs tensor.

    Returns:
        tuple[set[int], BlockSparseTensor]: Tuple of (contracting IDs, rhs tensor).
    """
    contracting_ids = _find_shared_contracting_ids(lhs, rhs)

    if not contracting_ids:
        contracting_ids, rhs = _infer_positional_contracting(lhs, rhs)

    _validate_contraction_compatibility(lhs, rhs, contracting_ids)

    return contracting_ids, rhs


def _densify_mixed_sparsity(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor, c_ids: set[int]
) -> tuple[BlockSparseTensor, BlockSparseTensor]:
    """
    Densifies dimensions if there is a mismatch between Sparse/Dense types on shared IDs.

    Args:
        lhs (BlockSparseTensor): Left-hand side tensor.
        rhs (BlockSparseTensor): Right-hand side tensor.
        c_ids (set[int]): Set of contracting dimension IDs.

    Returns:
        tuple[BlockSparseTensor, BlockSparseTensor]: A tuple of (lhs, rhs) where shared dimensions have consistent sparsity types.
    """
    shared_ids = set(d.id for d in lhs.dims) & set(d.id for d in rhs.dims)
    for sid in shared_ids:
        l_d = next(d for d in lhs.dims if d.id == sid)
        r_d = next(d for d in rhs.dims if d.id == sid)
        if isinstance(l_d, SparseDimension) != isinstance(r_d, SparseDimension):
            if isinstance(l_d, SparseDimension):
                lhs = lhs.dense((lhs.dims.index(l_d),), True)
            else:
                rhs = rhs.dense((rhs.dims.index(r_d),), True)
    if lhs.val is None and rhs.val is not None:
        lhs = lhs.dense(hard=False)
    elif rhs.val is None and lhs.val is not None:
        rhs = rhs.dense(hard=False)
    return lhs, rhs


def _handle_none_val_matmul(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor, c_ids: set[int]
) -> BlockSparseTensor:
    """
    Optimized path for matrix multiplication when both operands have implicitly defined values (val=None).

    In this case, the result is also implicit (val=None), and the operation reduces to
    updating the scalar multiplier and managing dimension metadata (pairings).

    Args:
        lhs (BlockSparseTensor): Left-hand side tensor (val=None).
        rhs (BlockSparseTensor): Right-hand side tensor (val=None).
        c_ids (set[int]): Set of contracting dimension IDs.

    Returns:
        BlockSparseTensor: A new BlockSparseTensor representing the result with val=None.
    """
    l_s = lhs.scalar_mult if lhs.scalar_mult is not None else 1.0
    r_s = rhs.scalar_mult if rhs.scalar_mult is not None else 1.0
    mult, tp = l_s * r_s, {}
    l_p_map, r_o_map = (
        {d.id: d for d in lhs.primal_dims},
        {d.id: d for d in rhs.out_dims},
    )
    for cid in c_ids:
        l_d, r_d = l_p_map[cid], r_o_map[cid]
        if not (isinstance(l_d, SparseDimension) and isinstance(r_d, SparseDimension)):
            mult *= l_d.logical_size
        else:
            lp, rp = l_d.other_id, r_d.other_id
            tp[lp], tp[rp] = rp, lp

    def update(dims):
        res = []
        for d in dims:
            if d.id in c_ids:
                continue
            res.append(
                replace(d, other_id=tp[d.id])
                if isinstance(d, SparseDimension) and d.id in tp
                else d
            )
        return tuple(res)

    return BlockSparseTensor(update(lhs.out_dims), update(rhs.primal_dims), None, mult)


def _align_physical_axes(
    target: BlockSparseTensor, source: BlockSparseTensor
) -> BlockSparseTensor:
    """
    Aligns the physical layout of 'source' to match 'target'.
    Assumes fully compatible logical structure (and both materialized).
    """
    if target.val is None or source.val is None:
        return source

    target_phys_to_key = {}
    for d in target.dims:
        if d.val_dim is not None:
            target_phys_to_key[d.val_dim] = (d.id, 0)
        if isinstance(d, SparseDimension) and d.block_val_dim is not None:
            target_phys_to_key[d.block_val_dim] = (d.id, 1)

    source_key_to_phys = {}
    for d in source.dims:
        if d.val_dim is not None:
            source_key_to_phys[(d.id, 0)] = d.val_dim
        if isinstance(d, SparseDimension) and d.block_val_dim is not None:
            source_key_to_phys[(d.id, 1)] = d.block_val_dim

    perm = []
    for i in range(target.val.ndim):
        key = target_phys_to_key.get(i)
        if key is None:
            raise ValueError(
                f"Target physical axis {i} has no corresponding dimension ID."
            )

        src_idx = source_key_to_phys.get(key)
        if src_idx is None:
            raise ValueError(
                f"Source missing physical axis corresponding to ID {key[0]} type {key[1]}"
            )
        perm.append(src_idx)

    if perm == list(range(target.val.ndim)):
        return source

    new_val = source.val.transpose(perm)

    target_id_map = {d.id: d for d in target.dims}

    new_out = []
    for d in source.out_dims:
        t_d = target_id_map[d.id]
        if isinstance(d, SparseDimension):
            new_out.append(
                replace(d, val_dim=t_d.val_dim, block_val_dim=t_d.block_val_dim)
            )
        else:
            new_out.append(replace(d, val_dim=t_d.val_dim))

    new_primal = []
    for d in source.primal_dims:
        t_d = target_id_map[d.id]
        if isinstance(d, SparseDimension):
            new_primal.append(
                replace(d, val_dim=t_d.val_dim, block_val_dim=t_d.block_val_dim)
            )
        else:
            new_primal.append(replace(d, val_dim=t_d.val_dim))

    return BlockSparseTensor(
        tuple(new_out), tuple(new_primal), new_val, source.scalar_mult
    )


def _validate_batch_compatibility(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor, b_ids: set[int]
) -> None:
    """
    Validates that shared batch dimensions have matching logical sizes.

    Args:
        lhs (BlockSparseTensor): lhs tensor.
        rhs (BlockSparseTensor): rhs tensor.
        b_ids (set[int]): Set of batch dimension IDs.

    Raises:
        ValueError: If size mismatch for shared batch dimensions.
    """
    l_map = {d.id: d for d in lhs.dims}
    r_map = {d.id: d for d in rhs.dims}
    for bid in b_ids:
        l_d, r_d = l_map[bid], r_map[bid]
        if l_d.logical_size != r_d.logical_size:
            raise ValueError(
                f"Incompatible batch sizes for shared dim {bid}: "
                f"{l_d.logical_size} vs {r_d.logical_size}"
            )


def _identify_physical_conflicts(
    dims: Sequence[Dimension], c_ids: set[int]
) -> set[int]:
    """
    Identifies physical axes used by contracting dimensions.

    Args:
        dims (Sequence[Dimension]): List of dimensions to check.
        c_ids (set[int]): Contracting dimension IDs.

    Returns:
        set[int]: Set of physical axes (val_dim/block_val_dim) indices.
    """
    conflicts = set()
    for d in dims:
        if d.id in c_ids:
            if d.val_dim is not None:
                conflicts.add(d.val_dim)
            if isinstance(d, SparseDimension) and d.block_val_dim is not None:
                conflicts.add(d.block_val_dim)
    return conflicts


def _densify_conflicting_dimensions(
    tensor: BlockSparseTensor, target_ids: set[int], conflict_axes: set[int]
) -> BlockSparseTensor:
    """
    Densifies dimensions that use conflicting physical axes.

    Args:
        tensor (BlockSparseTensor): The tensor to process.
        target_ids (set[int]): Set of IDs to consider for densification.
        conflict_axes (set[int]): Set of physical axes that are conflicted.

    Returns:
        BlockSparseTensor: Processed tensor with necessary densifications.
    """
    idxs_to_densify = [
        i
        for i, d in enumerate(tensor.dims)
        if d.id in target_ids
        and (
            d.val_dim in conflict_axes
            or (isinstance(d, SparseDimension) and d.block_val_dim in conflict_axes)
        )
    ]
    if idxs_to_densify:
        return tensor.dense(tuple(idxs_to_densify), hard=True)
    return tensor


def _align_implicit_dimensions(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor, ids: set[int]
) -> tuple[BlockSparseTensor, BlockSparseTensor]:
    """
    Aligns implicit/explicit status for shared dimensions.

    Args:
        lhs (BlockSparseTensor): lhs tensor.
        rhs (BlockSparseTensor): rhs tensor.
        ids (set[int]): Set of dimension IDs to align.

    Returns:
        tuple[BlockSparseTensor, BlockSparseTensor]: Tuple of aligned lhs and rhs tensors.
    """
    for sid in ids:
        l_map = {d.id: d for d in lhs.dims}
        r_map = {d.id: d for d in rhs.dims}
        l_d, r_d = l_map.get(sid), r_map.get(sid)

        if l_d and r_d and (l_d.val_dim is None) != (r_d.val_dim is None):
            if l_d.val_dim is None:
                lhs = lhs.dense((lhs.dims.index(l_d),), hard=False)
            else:
                rhs = rhs.dense((rhs.dims.index(r_d),), hard=False)
    return lhs, rhs


def _compute_implicit_scalar_correction(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor, c_ids: set[int]
) -> float:
    """
    Computes scalar correction for purely implicit contractions.

    Args:
        lhs (BlockSparseTensor): lhs tensor.
        rhs (BlockSparseTensor): rhs tensor.
        c_ids (set[int]): Contracting dimension IDs.

    Returns:
        float: Scalar multiplier correction.
    """
    scalar = 1.0
    l_p_map = {d.id: d for d in lhs.primal_dims}
    r_o_map = {d.id: d for d in rhs.out_dims}
    for cid in c_ids:
        if l_p_map[cid].val_dim is None and r_o_map[cid].val_dim is None:
            scalar *= l_p_map[cid].logical_size
    return float(scalar)


def _resolve_densification_and_batching(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor, c_ids: set[int]
) -> tuple[BlockSparseTensor, BlockSparseTensor, set[int], float]:
    """
    Resolves conflicts between implicit/materialized axes and ensures batch consistency.

    Decomposed into elementary steps:
    1. Validation
    2. Conflict Identification
    3. Densification
    4. Implicit Alignment
    5. Scalar Correction

    Args:
        lhs (BlockSparseTensor): lhs tensor.
        rhs (BlockSparseTensor): rhs tensor.
        c_ids (set[int]): Contracting IDs.

    Returns:
        tuple: (Updated lhs, Updated rhs, Batch IDs, Scalar Correction)
    """
    all_lhs_ids = set(d.id for d in lhs.dims)
    all_rhs_ids = set(d.id for d in rhs.dims)
    b_ids = (all_lhs_ids & all_rhs_ids) - c_ids

    _validate_batch_compatibility(lhs, rhs, b_ids)

    lhs, rhs = _align_implicit_dimensions(lhs, rhs, b_ids | c_ids)

    l_conflicts = _identify_physical_conflicts(lhs.primal_dims, c_ids)
    r_conflicts = _identify_physical_conflicts(rhs.out_dims, c_ids)

    lhs = _densify_conflicting_dimensions(
        lhs, b_ids | {d.id for d in lhs.out_dims}, l_conflicts
    )
    rhs = _densify_conflicting_dimensions(
        rhs, b_ids | {d.id for d in rhs.primal_dims}, r_conflicts
    )

    scalar = _compute_implicit_scalar_correction(lhs, rhs, c_ids)

    return lhs, rhs, b_ids, scalar


def _identify_matmul_axes(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor, c_ids: set[int], b_ids: set[int]
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Identifies the physical axes (indices) in `val` arrays corresponding to batch and contracting dimensions.

    Args:
        lhs (BlockSparseTensor): Left-hand side tensor.
        rhs (BlockSparseTensor): Right-hand side tensor.
        c_ids (set[int]): Set of contracting dimension IDs.
        b_ids (set[int]): Set of shared batch dimension IDs.

    Returns:
        tuple[list[int], list[int], list[int], list[int]]:
            - lhs contracting axes
            - rhs contracting axes
            - lhs batch axes
            - rhs batch axes
    """
    l_c, r_c, l_b, r_b, l_seen, r_seen, l_c_s, r_c_s = (
        [],
        [],
        [],
        [],
        set(),
        set(),
        set(),
        set(),
    )
    l_m, r_m = {d.id: d for d in lhs.dims}, {d.id: d for d in rhs.dims}
    l_p, r_o = {d.id: d for d in lhs.primal_dims}, {d.id: d for d in rhs.out_dims}

    for bid in sorted(list(b_ids)):
        l_d, r_d = l_m[bid], r_m[bid]
        if l_d.val_dim is not None and r_d.val_dim is not None:
            l_b.append(l_d.val_dim)
            r_b.append(r_d.val_dim)
            l_seen.add(l_d.val_dim)
            r_seen.add(r_d.val_dim)
            if isinstance(l_d, SparseDimension) and l_d.block_val_dim is not None:
                l_b.append(l_d.block_val_dim)
                r_b.append(r_d.block_val_dim)
                l_seen.add(l_d.block_val_dim)
                r_seen.add(r_d.block_val_dim)

    for cid in sorted(list(c_ids)):
        l_d, r_d = l_p[cid], r_o[cid]
        if isinstance(l_d, SparseDimension) and l_d.val_dim is not None:
            if l_d.val_dim not in l_seen:
                l_b.append(l_d.val_dim)
                r_b.append(r_d.val_dim)
                l_seen.add(l_d.val_dim)
                r_seen.add(r_d.val_dim)
        elif l_d.val_dim is not None and l_d.val_dim not in l_c_s:
            l_c.append(l_d.val_dim)
            r_c.append(r_d.val_dim)
            l_c_s.add(l_d.val_dim)
            r_c_s.add(r_d.val_dim)
        if (
            isinstance(l_d, SparseDimension)
            and l_d.block_val_dim is not None
            and l_d.block_val_dim not in l_c_s
        ):
            l_c.append(l_d.block_val_dim)
            r_c.append(r_d.block_val_dim)
            l_c_s.add(l_d.block_val_dim)
            r_c_s.add(r_d.block_val_dim)
    return l_c, r_c, l_b, r_b


def _prepare_values_for_dot(
    lhs_val: Array | None,
    rhs_val: Array | None,
    l_c: list[int],
    r_c: list[int],
    l_b: list[int],
    r_b: list[int],
) -> tuple[Array, Array]:
    """
    Prepares value arrays for `dot_general` by handling broadcasting logic.

    Args:
        lhs_val (Array | None): Value array of lhs (or None).
        rhs_val (Array | None): Value array of rhs (or None).
        l_c (list[int]): List of lhs contracting axes.
        r_c (list[int]): List of rhs contracting axes.
        l_b (list[int]): List of lhs batch axes.
        r_b (list[int]): List of rhs batch axes.

    Returns:
        tuple[Array, Array]: A tuple of (lhs_val, rhs_val) ready for contraction.
    """
    lv, rv = (
        (lhs_val if lhs_val is not None else jnp.array(1.0)),
        (rhs_val if rhs_val is not None else jnp.array(1.0)),
    )
    for l_ax, r_ax in zip(l_c + l_b, r_c + r_b):
        ls, rs = lv.shape[l_ax], rv.shape[r_ax]
        if ls != rs:
            if ls == 1:
                lv = jnp.broadcast_to(
                    lv, lv.shape[:l_ax] + (rs,) + lv.shape[l_ax + 1 :]
                )
            elif rs == 1:
                rv = jnp.broadcast_to(
                    rv, rv.shape[:r_ax] + (ls,) + rv.shape[r_ax + 1 :]
                )
    return lv, rv


def _get_transitive_pairing(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor, c_ids: set[int]
) -> dict[int, int]:
    """
    Computes transitive pairings for sparse dimensions that are preserved through contraction.

    Args:
        lhs (BlockSparseTensor): Left-hand side tensor.
        rhs (BlockSparseTensor): Right-hand side tensor.
        c_ids (set[int]): Set of contracting dimension IDs.

    Returns:
        dict[int, int]: A dictionary mapping dimension IDs to their new transitive partner IDs.
    """
    tp, l_p, r_o = (
        {},
        {d.id: d for d in lhs.primal_dims},
        {d.id: d for d in rhs.out_dims},
    )
    for cid in c_ids:
        l_d, r_d = l_p[cid], r_o[cid]
        if isinstance(l_d, SparseDimension) and isinstance(r_d, SparseDimension):
            lp, rp = l_d.other_id, r_d.other_id
            tp[lp], tp[rp] = rp, lp
    return tp


def _map_physical_axes(
    lhs: BlockSparseTensor,
    rhs: BlockSparseTensor,
    l_b: list[int],
    r_b: list[int],
    l_c: list[int],
    r_c: list[int],
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Maps physical axes from input tensors to their positions in the resulting contracted tensor.

    Args:
        lhs (BlockSparseTensor): Left-hand side tensor.
        rhs (BlockSparseTensor): Right-hand side tensor.
        l_b (list[int]): lhs batch axes.
        r_b (list[int]): rhs batch axes.
        l_c (list[int]): lhs contracting axes.
        r_c (list[int]): rhs contracting axes.

    Returns:
        tuple: A tuple of (lhs_axis_map, rhs_axis_map).
    """
    l_map, r_map, curr = {}, {}, 0
    for i in range(len(l_b)):
        l_map[l_b[i]] = r_map[r_b[i]] = curr
        curr += 1
    u_l, u_r = set(l_b) | set(l_c), set(r_b) | set(r_c)
    for i in range(lhs.val.ndim):
        if i not in u_l:
            l_map[i] = curr
            curr += 1
    for i in range(rhs.val.ndim):
        if i not in u_r:
            r_map[i] = curr
            curr += 1
    return l_map, r_map


def _reconstruct_matmul_dims(
    lhs: BlockSparseTensor,
    rhs: BlockSparseTensor,
    l_map: dict,
    r_map: dict,
    c_ids: set[int],
    b_ids: set[int],
    tp: dict,
) -> tuple[tuple[Dimension, ...], tuple[Dimension, ...], list[tuple[int, int]]]:
    """
    Reconstructs the logical dimensions of the result tensor and identifies axes that need merging.

    Args:
        lhs (BlockSparseTensor): Left-hand side tensor.
        rhs (BlockSparseTensor): Right-hand side tensor.
        l_map (dict): Map from lhs physical axes to result axes.
        r_map (dict): Map from rhs physical axes to result axes.
        c_ids (set[int]): Set of contracting dimension IDs.
        b_ids (set[int]): Set of shared batch dimension IDs.
        tp (dict): Transitive pairing dictionary.

    Returns:
        tuple: (New out dims, New primal dims, Pairs of axes to merge).
    """
    to_f, r_ids = (
        [],
        (set(d.id for d in lhs.dims) | set(d.id for d in rhs.dims)) - c_ids,
    )

    def recon(dims, amap):
        res = []
        for d in dims:
            if d.id in c_ids:
                continue
            v_p = amap.get(d.val_dim) if d.val_dim is not None else None
            if isinstance(d, SparseDimension):
                pid, b_p = (
                    tp.get(d.id, d.other_id),
                    amap.get(d.block_val_dim) if d.block_val_dim is not None else None,
                )
                partner_exists = pid in r_ids or pid == d.id
                if not partner_exists or pid in c_ids:
                    res.append(DenseDimension(d.id, d.size * d.block_size, v_p))
                    if v_p is not None and b_p is not None:
                        to_f.append((v_p, b_p))
                else:
                    res.append(replace(d, val_dim=v_p, block_val_dim=b_p, other_id=pid))
            else:
                res.append(replace(d, val_dim=v_p))
        return res

    new_o = tuple(_deduplicate(recon(lhs.out_dims, l_map) + recon(rhs.out_dims, r_map)))
    new_p = tuple(
        _deduplicate(recon(lhs.primal_dims, l_map) + recon(rhs.primal_dims, r_map))
    )
    return new_o, new_p, to_f


def _deduplicate(dims: list[Dimension]) -> list[Dimension]:
    """
    Deduplicates a list of dimensions based on ID.

    Args:
        dims (list[Dimension]): The dimensions to deduplicate.

    Returns:
        list[Dimension]: Deduplicated list.
    """
    seen, final = set(), []
    for d in dims:
        if d.id not in seen:
            final.append(d)
            seen.add(d.id)
    return final


def _group_axes_to_merge(
    val_ndim: int, to_f: list[tuple[int, int]]
) -> dict[int, list[int]]:
    """
    Groups axes that need to be merged using Union-Find.

    Args:
        val_ndim (int): Number of dimensions in the value array.
        to_f (list[tuple[int, int]]): List of pairs of axes to merge.

    Returns:
        dict[int, list[int]]: Mapping from root axis to list of axes in the group.
    """
    parent = list(range(val_ndim))

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            next_i = parent[i]
            parent[i] = root
            i = next_i
        return root

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for v, b in to_f:
        union(v, b)
    groups = {}
    for i in range(val_ndim):
        groups.setdefault(find(i), []).append(i)

    return {root: sorted(axes) for root, axes in groups.items() if len(axes) > 1}


def _flatten_grouped_axes(
    val: Array, groups: dict[int, list[int]]
) -> tuple[Array, dict[int, int]]:
    """
    Flattens grouped axes in the value array.

    Args:
        val (Array): Input value array.
        groups (dict[int, list[int]]): Groups of axes to merge.

    Returns:
        tuple[Array, dict[int, int]]: Flattened array and mapping from old to new axis indices.
    """
    used_in_merge = {ax for axes in groups.values() for ax in axes}

    perm, ops = [], []
    for i in range(val.ndim):
        if i in used_in_merge:
            if i in groups:
                perm.extend(groups[i])
                ops.append(len(groups[i]))
            continue
        perm.append(i)
        ops.append(1)

    val = val.transpose(perm)
    flat, curr = [], 0
    for count in ops:
        if count > 1:
            size = 1
            for _ in range(count):
                size *= val.shape[curr]
                curr += 1
            flat.append(size)
        else:
            flat.append(val.shape[curr])
            curr += 1
    val = val.reshape(flat)

    o2n, k, cn = {}, 0, 0
    for count in ops:
        for _ in range(count):
            o2n[perm[k]] = cn
            k += 1
        cn += 1
    return val, o2n


def _update_dims_after_flattening(
    dims: tuple[Dimension, ...], o2n: dict[int, int]
) -> tuple[Dimension, ...]:
    """
    Updates dimension physical axis indices after flattening.

    Args:
        dims (tuple[Dimension, ...]): Dimensions to update.
        o2n (dict[int, int]): Old to new axis index mapping.

    Returns:
        tuple[Dimension, ...]: Updated dimensions.
    """
    res = []
    for d in dims:
        nv = o2n.get(d.val_dim) if d.val_dim is not None else None
        nb = (
            o2n.get(d.block_val_dim)
            if isinstance(d, SparseDimension) and d.block_val_dim is not None
            else None
        )
        res.append(
            replace(d, val_dim=nv, block_val_dim=nb)
            if isinstance(d, SparseDimension)
            else replace(d, val_dim=nv)
        )
    return tuple(res)


def _flatten_result_axes(
    val: Array,
    to_f: list[tuple[int, int]],
    out: tuple[Dimension, ...],
    primal: tuple[Dimension, ...],
) -> tuple[Array, tuple[Dimension, ...], tuple[Dimension, ...]]:
    """
    Flattens (merges) physical axes in the result tensor that correspond to diagonal sparse blocks.

    Args:
        val (Array): The raw result array from `dot_general`.
        to_f (list[tuple[int, int]]): List of axis pairs to merge.
        out (tuple[Dimension, ...]): Preliminary output dimensions.
        primal (tuple[Dimension, ...]): Preliminary primal dimensions.

    Returns:
        tuple: (Flattened value array, Updated output dims, Updated primal dims).
    """
    groups = _group_axes_to_merge(val.ndim, to_f)
    val, o2n = _flatten_grouped_axes(val, groups)

    return (
        val,
        _update_dims_after_flattening(out, o2n),
        _update_dims_after_flattening(primal, o2n),
    )


def _copy(bst, val=None, scalar_mult=None, deep=False):
    if deep:
        out_dims = copy.deepcopy(bst.out_dims)
        primal_dims = copy.deepcopy(bst.primal_dims)
        pre = copy.deepcopy(bst.pre_transforms)
        post = copy.deepcopy(bst.post_transforms)

        if val is None:
            val = copy.deepcopy(bst.val)

        if scalar_mult is None:
            scalar_mult = copy.deepcopy(bst.scalar_mult)
    else:
        out_dims = bst.out_dims
        primal_dims = bst.primal_dims
        pre = bst.pre_transforms
        post = bst.post_transforms

        if val is None:
            val = bst.val

        if scalar_mult is None:
            scalar_mult = bst.scalar_mult

    return BlockSparseTensor(out_dims, primal_dims, val, scalar_mult, pre, post)


def _is_pure_dense_interface(
    c_ids: set[int], l_map: dict[int, Dimension], r_map: dict[int, Dimension]
) -> bool:
    """Checks if contraction involves only DenseDimensions."""
    for cid in c_ids:
        if isinstance(l_map[cid], SparseDimension) or isinstance(
            r_map[cid], SparseDimension
        ):
            return False
    return True


def _is_pure_sparse_contraction(
    c_ids: set[int], l_map: dict[int, Dimension], r_map: dict[int, Dimension]
) -> bool:
    """Checks if contraction preserves diagonal structure (Sparse @ Sparse)."""
    for cid in c_ids:
        if not (
            isinstance(l_map[cid], SparseDimension)
            and isinstance(r_map[cid], SparseDimension)
        ):
            return False
    return True


def _block_pure_dot_product_mul(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor, contracting_ids: set[int]
) -> BlockSparseTensor:
    """Optimized path for Dense @ Dense contraction."""
    lhs_implicits = [
        d for d in lhs.dims if d.id in contracting_ids and d.val_dim is None
    ]
    if lhs_implicits:
        lhs = _densify_implicits(lhs, lhs_implicits)

    rhs_implicits = [
        d for d in rhs.dims if d.id in contracting_ids and d.val_dim is None
    ]
    if rhs_implicits:
        rhs = _densify_implicits(rhs, rhs_implicits)

    l_c, r_c = [], []
    l_map = {d.id: d for d in lhs.primal_dims}
    r_map = {d.id: d for d in rhs.out_dims}

    for cid in contracting_ids:
        l_c.append(l_map[cid].val_dim)
        r_c.append(r_map[cid].val_dim)

    l_b, r_b = [], []

    lhs_val = lhs.val if lhs.val is not None else jnp.ones((1,), dtype=float)
    rhs_val = rhs.val if rhs.val is not None else jnp.ones((1,), dtype=float)

    final_scalar = lhs.scalar_mult * rhs.scalar_mult

    out_val = lax.dot_general(lhs_val, rhs_val, ((l_c, r_c), (l_b, r_b)))

    new_out_dims = []
    new_primal_dims = []

    lhs_kept_axes = [i for i in range(lhs_val.ndim) if i not in l_c]
    rhs_kept_axes = [i for i in range(rhs_val.ndim) if i not in r_c]

    l_remap = {old: new for new, old in enumerate(lhs_kept_axes)}
    r_remap = {old: new + len(lhs_kept_axes) for new, old in enumerate(rhs_kept_axes)}

    for d in lhs.out_dims:
        new_val_dim = l_remap.get(d.val_dim) if d.val_dim is not None else None
        new_block_val_dim = None
        if isinstance(d, SparseDimension) and d.block_val_dim is not None:
            new_block_val_dim = l_remap.get(d.block_val_dim)

        if isinstance(d, SparseDimension):
            new_out_dims.append(
                replace(d, val_dim=new_val_dim, block_val_dim=new_block_val_dim)
            )
        else:
            new_out_dims.append(replace(d, val_dim=new_val_dim))

    for d in rhs.primal_dims:
        new_val_dim = r_remap.get(d.val_dim) if d.val_dim is not None else None
        new_block_val_dim = None
        if isinstance(d, SparseDimension) and d.block_val_dim is not None:
            new_block_val_dim = r_remap.get(d.block_val_dim)

        if isinstance(d, SparseDimension):
            new_primal_dims.append(
                replace(d, val_dim=new_val_dim, block_val_dim=new_block_val_dim)
            )
        else:
            new_primal_dims.append(replace(d, val_dim=new_val_dim))

    return BlockSparseTensor(
        tuple(new_out_dims), tuple(new_primal_dims), out_val, final_scalar
    )


def _block_pure_broadcast_mul(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor, contracting_ids: set[int]
) -> BlockSparseTensor:
    """Optimized path for Sparse @ Sparse contraction."""

    l_c, r_c = [], []
    l_b, r_b = [], []

    l_map = {d.id: d for d in lhs.primal_dims}
    r_map = {d.id: d for d in rhs.out_dims}

    for cid in contracting_ids:
        ld = l_map[cid]
        rd = r_map[cid]

        if ld.val_dim is not None and rd.val_dim is not None:
            if ld.val_dim not in l_b:
                l_b.append(ld.val_dim)
            if rd.val_dim not in r_b:
                r_b.append(rd.val_dim)

        if isinstance(ld, SparseDimension):
            if ld.block_val_dim is not None:
                l_c.append(ld.block_val_dim)
        elif ld.val_dim is not None and ld.val_dim not in l_b:
            l_c.append(ld.val_dim)

        if isinstance(rd, SparseDimension):
            if rd.block_val_dim is not None:
                r_c.append(rd.block_val_dim)
        elif rd.val_dim is not None and rd.val_dim not in r_b:
            r_c.append(rd.val_dim)

    if lhs.val is None and rhs.val is None:
        return _handle_none_val_matmul(lhs, rhs, contracting_ids)

    if lhs.val is None or rhs.val is None:
        if lhs.val is None:
            lhs = lhs.dense(hard=False)
        if rhs.val is None:
            rhs = rhs.dense(hard=False)

    lhs_val, rhs_val = _prepare_values_for_dot(lhs.val, rhs.val, l_c, r_c, l_b, r_b)

    final_scalar = lhs.scalar_mult * rhs.scalar_mult

    out_val = lax.dot_general(lhs_val, rhs_val, ((l_c, r_c), (l_b, r_b)))

    new_out_dims = []
    new_primal_dims = []
    transitive = {}

    num_batch = len(l_b)
    lhs_kept_phys = [i for i in range(lhs_val.ndim) if i not in l_c and i not in l_b]
    rhs_kept_phys = [i for i in range(rhs_val.ndim) if i not in r_c and i not in r_b]

    batch_remap = {old: new for new, old in enumerate(l_b)}

    for cid in contracting_ids:
        ld = l_map[cid]
        rd = r_map[cid]
        transitive[ld.other_id] = rd.other_id
        transitive[rd.other_id] = ld.other_id

    for d in lhs.out_dims:
        if d.id in transitive:
            new_v = batch_remap.get(d.val_dim)
            new_bv = None
            if d.block_val_dim is not None:
                try:
                    idx = lhs_kept_phys.index(d.block_val_dim)
                    new_bv = num_batch + idx
                except ValueError:
                    pass
            new_other = transitive[d.id]
            new_out_dims.append(
                replace(d, val_dim=new_v, block_val_dim=new_bv, other_id=new_other)
            )
        else:
            new_v = None
            if d.val_dim is not None:
                if d.val_dim in l_b:
                    new_v = batch_remap[d.val_dim]
                elif d.val_dim in lhs_kept_phys:
                    new_v = num_batch + lhs_kept_phys.index(d.val_dim)
            new_out_dims.append(replace(d, val_dim=new_v))

    rhs_batch_map = {old: new for new, old in enumerate(r_b)}
    for d in rhs.primal_dims:
        if d.id in transitive:
            new_v = rhs_batch_map.get(d.val_dim)

            new_bv = None
            if d.block_val_dim is not None:
                try:
                    idx = rhs_kept_phys.index(d.block_val_dim)
                    new_bv = num_batch + len(lhs_kept_phys) + idx
                except ValueError:
                    pass
            new_other = transitive[d.id]
            new_primal_dims.append(
                replace(d, val_dim=new_v, block_val_dim=new_bv, other_id=new_other)
            )
        else:
            new_v = None
            if d.val_dim is not None:
                if d.val_dim in r_b:
                    new_v = rhs_batch_map[d.val_dim]
                elif d.val_dim in rhs_kept_phys:
                    new_v = (
                        num_batch + len(lhs_kept_phys) + rhs_kept_phys.index(d.val_dim)
                    )
            new_primal_dims.append(replace(d, val_dim=new_v))

    return BlockSparseTensor(
        tuple(new_out_dims), tuple(new_primal_dims), out_val, final_scalar
    )


def _bst_matmul(lhs: BlockSparseTensor, rhs: BlockSparseTensor) -> BlockSparseTensor:
    """
    Performs block-sparse matrix multiplication with optimized dispatch.
    """
    lhs, rhs = _align_blocks(lhs, rhs)
    c_ids, rhs = _get_contracting_ids(lhs, rhs)
    l_map = {d.id: d for d in lhs.primal_dims}
    r_map = {d.id: d for d in rhs.out_dims}

    if c_ids and _is_pure_dense_interface(c_ids, l_map, r_map):
        return _block_pure_dot_product_mul(lhs, rhs, c_ids)

    if c_ids and _is_pure_sparse_contraction(c_ids, l_map, r_map):
        return _block_pure_broadcast_mul(lhs, rhs, c_ids)

    return _block_mixed_mul(lhs, rhs, c_ids)


def _block_mixed_mul(
    lhs: BlockSparseTensor, rhs: BlockSparseTensor, c_ids: set[int]
) -> BlockSparseTensor:
    """
    Performs block-sparse matrix multiplication.

    Args:
        lhs (BlockSparseTensor): Left-hand side tensor.
        rhs (BlockSparseTensor): Right-hand side tensor.
        c_ids (set[int]): Contracting IDs.

    Returns:
        BlockSparseTensor: The result of the matrix multiplication.
    """
    lhs, rhs = _densify_mixed_sparsity(lhs, rhs, c_ids)
    if lhs.val is None and rhs.val is None:
        return _handle_none_val_matmul(lhs, rhs, c_ids)

    lhs, rhs, b_ids, scalar_adjustment = _resolve_densification_and_batching(
        lhs, rhs, c_ids
    )
    l_c, r_c, l_b, r_b = _identify_matmul_axes(lhs, rhs, c_ids, b_ids)
    l_val, r_val = _prepare_values_for_dot(lhs.val, rhs.val, l_c, r_c, l_b, r_b)

    contracted_val = jax.lax.dot_general(l_val, r_val, ((l_c, r_c), (l_b, r_b)))

    transitive_pairing = _get_transitive_pairing(lhs, rhs, c_ids)
    l_map, r_map = _map_physical_axes(lhs, rhs, l_b, r_b, l_c, r_c)

    new_out, new_primal, axes_to_merge = _reconstruct_matmul_dims(
        lhs, rhs, l_map, r_map, c_ids, b_ids, transitive_pairing
    )

    final_val, final_out, final_primal = _flatten_result_axes(
        contracted_val, axes_to_merge, new_out, new_primal
    )

    final_scalar = lhs.scalar_mult * rhs.scalar_mult * jnp.array(scalar_adjustment)

    return BlockSparseTensor(final_out, final_primal, final_val, final_scalar)


def _bst_add(lhs: BlockSparseTensor, rhs: BlockSparseTensor) -> BlockSparseTensor:
    lhs, rhs = _align_blocks(lhs, rhs)
    all_ids = set(d.id for d in lhs.dims) | set(d.id for d in rhs.dims)
    lhs, rhs = _align_implicit_dimensions(lhs, rhs, all_ids)

    if lhs.val is not None and rhs.val is not None:
        rhs = _align_physical_axes(lhs, rhs)

    if lhs.val is None:
        return BlockSparseTensor(
            lhs.out_dims, lhs.primal_dims, None, lhs.scalar_mult + rhs.scalar_mult
        )

    v1 = lhs.val * lhs.scalar_mult
    v2 = rhs.val * rhs.scalar_mult
    return BlockSparseTensor(lhs.out_dims, lhs.primal_dims, v1 + v2)


def _bst_mul(lhs: BlockSparseTensor, rhs: BlockSparseTensor) -> BlockSparseTensor:
    lhs, rhs = _align_blocks(lhs, rhs)
    all_ids = set(d.id for d in lhs.dims) | set(d.id for d in rhs.dims)
    lhs, rhs = _align_implicit_dimensions(lhs, rhs, all_ids)

    if lhs.val is not None and rhs.val is not None:
        rhs = _align_physical_axes(lhs, rhs)

    if lhs.val is None:
        return BlockSparseTensor(
            lhs.out_dims, lhs.primal_dims, None, lhs.scalar_mult * rhs.scalar_mult
        )

    v1 = lhs.val * lhs.scalar_mult
    v2 = rhs.val * rhs.scalar_mult
    return BlockSparseTensor(lhs.out_dims, lhs.primal_dims, v1 * v2)


SparseTensor = BlockSparseTensor  # Compatability
