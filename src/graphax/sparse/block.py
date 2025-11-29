import copy
import operator
from dataclasses import KW_ONLY, dataclass
from functools import partial, reduce
from itertools import count
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    NamedTuple,
    Sequence,
    TypeAlias,
)

import jax
import jax.numpy as jnp
import numpy as np

# import numpy as np
from chex import Array
from jax import jit, lax
from jax._src.dispatch import Backend
from jax.tree_util import register_pytree_node_class

from .tensor import SparseTensor

# TODO: make parent class, or inherit sparse tensor ??


class DenseDimension(NamedTuple):
    id: int
    size: int
    val_dim: int | None
    # val_axis: int = None


# val_axes: These are two axis of the blocks that we would like to apply the diagonal on.
#   By default pick the first two (0, 1) as the axis). TODO: make negatives work like indeces
class SparseDimension(NamedTuple):
    id: int
    size: int
    val_dim: int
    other_id: int
    block_size: int = None
    # val_axis: int = None


# TODO TODO TODO TODO TODO, the new idea is to setup Sparse dim, such that if two different val_dims are set for a pair,
# then its sparse block...


Dimension = DenseDimension | SparseDimension


# TODO (somewhere) blocks of blocks with neighbors can be combined to one block
# I believe that tensors with d diagonal then there will be 2d blocks maximum (the rest can be merged into individual blocks)
@register_pytree_node_class
class BlockSparseTensor:
    out_dims: Any
    primal_dims: Any
    out_shape: tuple[int, ...]
    primal_shape: tuple[int, ...]
    shape: tuple[int, ...]
    size: int
    ndim: int
    blocks: Array | None
    pre_transforms: Array
    post_transforms: Array
    sparse_dims: int
    block_shape: tuple[int, ...]
    block_size: int
    _sparse_dim_order: list[tuple[int, int]]

    def __init__(
        self,
        out_dims: Sequence[Dimension],
        primal_dims: Sequence[Dimension],
        blocks: Array | None,
        pre_transforms: Sequence[Callable] = None,
        post_transforms: Sequence[Callable] = None,
    ) -> None:
        if pre_transforms is None:
            pre_transforms = []
        if post_transforms is None:
            post_transforms = []

        sparse_dims = sum(isinstance(d, SparseDimension) for d in out_dims)
        assert sparse_dims == sum(isinstance(d, SparseDimension) for d in primal_dims)

        # TODO: assertions

        # assert all(d == i for d, i in zip(sorted_val_dims, list(range(n)))), \
        #    "Value dimensions should be continuous"

        # print(sorted_val_dims)
        # print(
        #    {d.val_axis for d in out_dims if isinstance(d, DenseDimension)}
        #    | {d.val_axis for d in primal_dims}
        # )
        # assert n == len(
        #        {d.val_axis for d in out_dims if isinstance(d, DenseDimension)}
        #        | {d.val_axis for d in primal_dims}
        #    ), "Value axis should be unique"

        # assert blocks.ndim > sparse_dims # <-- breaks jit stuff
        block_shape = blocks.shape[sparse_dims:]
        # TODO add ones for non mentioned areas

        # print(blocks.shape)
        # print(block_shape)

        out_shape = tuple(
            x.size if isinstance(x, DenseDimension) else x.size * block_shape[x.val_dim]
            for x in out_dims
        )
        primal_shape = tuple(
            x.size if isinstance(x, DenseDimension) else x.size * block_shape[x.val_dim]
            for x in primal_dims
        )

        self.out_dims = out_dims if isinstance(out_dims, tuple) else tuple(out_dims)
        self.primal_dims = (
            primal_dims if isinstance(primal_dims, tuple) else tuple(primal_dims)
        )

        self.sparse_dims = sparse_dims
        self.block_shape = blocks.shape[self.sparse_dims :]
        self.block_size = reduce(operator.mul, self.block_shape)

        self.out_shape = tuple(out_shape)

        self.primal_shape = tuple(primal_shape)

        # self.out_shape = get_block_shape(out_dims)
        # self.primal_shape = get_block_shape(primal_dims)

        self.shape = tuple(self.out_shape + self.primal_shape)  # isn't quite right
        # TODO: _get_fully_materialized_shape
        self.size = reduce(operator.mul, self.shape)
        self.ndim = len(self.shape)

        self.blocks = blocks

        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms

        self._sparse_dim_order = [
            (d.id, d.other_id) for d in self.out_dims if isinstance(d, SparseDimension)
        ]  # may not be necessary to include, but must be mentioned in the docs

    def tree_flatten(self):
        return (
            (self.blocks,),
            (
                self.out_dims,
                self.primal_dims,
                self.pre_transforms,
                self.post_transforms,
            ),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (blocks,) = children
        out_dims, primal_dims, pre_transforms, post_transforms = aux_data
        return cls(out_dims, primal_dims, blocks, pre_transforms, post_transforms)

    def __repr__(self) -> str:
        def map_str(a: Sequence) -> Generator:
            return (str(s) for s in a)

        def multiline_seq(s: Sequence, brackets: str) -> str:
            lb, rb, *_ = brackets
            if s:
                res = f"{lb}\n    " + ",\n    ".join(map_str(s)) + f",\n  {rb}"
            else:
                res = lb + rb
            return res

        str_out_shape = ", ".join(map_str(self.out_shape))
        str_primal_shape = ", ".join(map_str(self.primal_shape))

        multiline_out_dims = multiline_seq(self.out_dims, "()")
        multiline_primal_dims = multiline_seq(self.primal_dims, "()")
        multiline_pre_transform = multiline_seq(self.pre_transforms, "[]")
        multiline_post_transform = multiline_seq(self.post_transforms, "[]")

        return (
            f"BlockSparseTensor(\n"
            f"  shape = ({str_out_shape} | {str_primal_shape}),\n"
            f"  out_dims = {multiline_out_dims},\n"
            f"  primal_dims = {multiline_primal_dims},\n"
            f"  val = Array(shape={self.blocks.shape}, dtype={self.blocks.dtype}),\n"
            f"  pre_transforms = {multiline_pre_transform},\n"
            f"  post_transforms = {multiline_post_transform}\n"
            f")"
        )

    # This is not a transpose like w/ normal tensors. The order should be completely reversed.
    # testcase: st == st.T.T
    def transpose(self, *args, **kwargs):
        return _transpose(self, *args, **kwargs)

    def block_until_ready(self):
        self.blocks.block_until_ready()
        return self

    @property
    def T(self):
        return self.transpose()

    @property
    def dtype(self):
        return self.blocks.dtype

    def dense(self) -> Array:
        return _dense(self, method="multiplication")

    def __eq__(lhs, rhs):
        return _eq(lhs, rhs)

    def __req__(rhs, lhs):
        return _eq(rhs, lhs)

    def __add__(lhs, rhs):
        return _add(lhs, rhs)

    def __radd__(rhs, lhs):
        return _add(rhs, lhs)

    def __mul__(lhs, rhs):
        return _mul(lhs, rhs)

    def __rmul__(rhs, lhs):
        return _mul(rhs, lhs)

    def __matmul__(lhs, rhs):
        return _matmul(lhs, rhs)

    def __rmatmul__(rhs, lhs):
        return _matmul(rhs.T, lhs.T).T

    def __copy__(self, val=None):
        return _copy(self, val)

    def __deepcopy__(self):
        return _copy(self, deep=True)

    def _apply_to_val(self, fn, *args, **kwargs):
        return self.__copy__(fn(self.blocks, *args, **kwargs))

    def _apply_to_two_vals(lhs, rhs, fn, *args, **kwargs):
        return self.__copy__(fn(lhs.blocks, rhs.blocks, *args, **kwargs))


def _transpose(bst, out_transpose=None, primal_transpose=None):
    if out_transpose is None and primal_transpose is None:
        n_out = len(bst.out_dims)
        n_primal = len(bst.primal_dims)
        out_transpose = tuple(range(n_out + n_primal - 1, n_out - 1, -1))
        primal_transpose = tuple(range(n_out - 1, -1, -1))

    full_transpose = tuple(out_transpose) + tuple(primal_transpose)

    if len(full_transpose) != bst.ndim or len(set(full_transpose)) != bst.ndim:
        raise TypeError(
            f"transpose permutation isn't a permutation of operand dimensions, "
            f"got permutation ({str(out_transpose)[1:-1]} | {str(primal_transpose)[1:-1]}) "
            f"for operand with {bst.ndim} dimensions."
        )

    dims = bst.out_dims + bst.primal_dims

    inverse_full_transpose = np.argsort(full_transpose)

    c = count()

    def remap_permuted_dim(old_idx, new_val_dim):
        old_dim = dims[old_idx]
        new_id = next(c)
        if isinstance(old_dim, SparseDimension):
            new_other_id = inverse_full_transpose[old_dim.other_id]
            return old_dim._replace(
                id=new_id, other_id=new_other_id, val_dim=new_val_dim
            )
        else:
            return old_dim._replace(id=new_id, val_dim=new_val_dim)

    new_out_dims = [
        remap_permuted_dim(i, new_val_dim)
        for new_val_dim, i in enumerate(out_transpose)
    ]
    new_primal_dims = [
        remap_permuted_dim(i, new_val_dim + len(out_transpose))
        for new_val_dim, i in enumerate(primal_transpose)
    ]

    val_dim_perm = np.argsort(
        [
            dims[i].val_dim
            for i in full_transpose
            if isinstance(dims[i], SparseDimension)
        ]
    )

    axes = list(range(bst.sparse_dims)) + [bst.sparse_dims + i for i in val_dim_perm]
    new_blocks = jnp.transpose(bst.blocks, axes=axes)

    return BlockSparseTensor(new_out_dims, new_primal_dims, new_blocks)


def _dense(
    st: BlockSparseTensor,
):
    if st.sparse_dims == 0:
        return jnp.broadcast_to(st.blocks, st.shape)

    sparse_shape_grid = st.blocks.shape[: st.sparse_dims]
    dense_shape_block = st.blocks.shape[st.sparse_dims :]
    num_blocks = np.prod(sparse_shape_grid).item()

    blocks_flat = st.blocks.reshape(num_blocks, *dense_shape_block)
    diag_blocks_flat = jnp.einsum(
        "i...,ij->ij...", blocks_flat, jnp.eye(num_blocks, dtype=st.blocks.dtype)
    )

    diag_tensor_full_axes = diag_blocks_flat.reshape(
        *sparse_shape_grid, *sparse_shape_grid, *dense_shape_block
    )

    processed_pairs = {
        tuple(sorted((d.id, d.other_id)))
        for d in st.out_dims + st.primal_dims
        if isinstance(d, SparseDimension)
    }
    sparse_pair_to_abstract_grid_idx = {
        pair: i for i, pair in enumerate(sorted(list(processed_pairs)))
    }

    permutation = []
    block_dense_val_dim_to_source_axis = {
        idx: 2 * st.sparse_dims + idx for idx in range(len(dense_shape_block))
    }

    for d in st.out_dims:
        if isinstance(d, SparseDimension):
            abstract_grid_idx = sparse_pair_to_abstract_grid_idx[
                tuple(sorted((d.id, d.other_id)))
            ]
            permutation.append(abstract_grid_idx)
        permutation.append(block_dense_val_dim_to_source_axis[d.val_dim])

    for d in st.primal_dims:
        if isinstance(d, SparseDimension):
            abstract_grid_idx = sparse_pair_to_abstract_grid_idx[
                tuple(sorted((d.id, d.other_id)))
            ]
            permutation.append(st.sparse_dims + abstract_grid_idx)
        permutation.append(block_dense_val_dim_to_source_axis[d.val_dim])

    all_source_axes_used = set(permutation)
    expected_source_axes = set(range(2 * st.sparse_dims + len(dense_shape_block)))
    if len(all_source_axes_used) != len(expected_source_axes):
        unmapped_block_internal_axes = sorted(
            list(expected_source_axes - all_source_axes_used)
        )
        permutation.extend(unmapped_block_internal_axes)

    transposed_tensor = jnp.transpose(diag_tensor_full_axes, axes=permutation)

    return transposed_tensor.reshape(st.shape)


def _copy(bst, val=None, deep=False):
    if val is None:
        if deep:
            val = copy.deepcopy(bst.blocks)
        else:
            val = bst.blocks

    return BlockSparseTensor(
        bst.out_dims, bst.primal_dims, val, bst.pre_transforms, bst.post_transforms
    )


def _eq(lhs, rhs):
    return (
        isinstance(rhs, BlockSparseTensor)
        and lhs.out_dims == rhs.out_dims
        and lhs.primal_dims == rhs.primal_dims
        and jnp.all(lhs.blocks == rhs.blocks)
    )


# TODO TODO TODO TODO TODO
# the None cases!!!


def _add(lhs, rhs):
    assert rhs.shape == lhs.shape, "Tensors must be of equal shape"
    if isinstance(rhs, BlockSparseTensor):
        if rhs.blocks is None:
            if (
                rhs.shape == lhs.shape
                and rhs.primal_dims == lhs.primal_dims
                and rhs.out_dims == lhs.out_dims
            ):
                res = lhs.copy()
                res.blocks = res.blocks + 1
                return res
            else:
                lhs.dense() + rhs.dense()  # worst case ...
        elif (
            rhs.shape == lhs.shape
            and rhs.primal_dims == lhs.primal_dims
            and rhs.out_dims == lhs.out_dims
        ):
            return BlockSparseTensor(
                lhs.out_dims, lhs.primal_dims, lhs.blocks + rhs.blocks
            )
    elif isinstance(rhs, SparseTensor):
        pass
    elif isinstance(rhs, Array):
        return lhs.dense() + rhs
    else:
        raise TypeError(
            "Expected to add with type BlockSparseTensor, SparseTensor, or Array"
        )


def _mul(lhs, rhs):
    assert lhs.shape == rhs.shape, "Tensors must be of equal shape"
    if isinstance(rhs, BlockSparseTensor):
        if rhs.blocks is None:
            if (
                rhs.shape == lhs.shape
                and rhs.primal_dims == lhs.primal_dims
                and rhs.out_dims == lhs.out_dims
            ):
                return lhs.copy()
            else:
                lhs.dense() * rhs.dense()  # worst case ...
        elif (
            lhs.shape == rhs.shape
            and lhs.primal_dims == rhs.primal_dims
            and lhs.out_dims == rhs.out_dims
        ):
            return BlockSparseTensor(
                lhs.out_dims, lhs.primal_dims, lhs.blocks * rhs.blocks
            )
    elif isinstance(rhs, SparseTensor):
        pass
    elif isinstance(rhs, Array):
        return lhs.dense() + rhs
    else:
        raise TypeError(
            "Expected to add with type BlockSparseTensor, SparseTensor, or Array"
        )


def _matmul(lhs, rhs):
    # TODO assert something
    if isinstance(rhs, BlockSparseTensor):
        if lhs.blocks is None:
            return copy.copy(rhs)
        elif rhs.blocks is None:
            return copy.copy(lhs)
        elif (
            isinstance(lhs.blocks, Array)
            and isinstance(rhs.blocks, Array)
            and lhs.primal_shape == rhs.out_shape
            and lhs.sparse_dims == rhs.sparse_dims
        ):
            block_idxs = tuple(range(lhs.sparse_dims))
            lhs_val_dims = [x.val_dim + lhs.sparse_dims for x in lhs.primal_dims]
            rhs_val_dims = [x.val_dim + rhs.sparse_dims for x in rhs.out_dims]
            dim_nums = (
                (lhs_val_dims, rhs_val_dims),
                (block_idxs, block_idxs),
            )

            val = lax.dot(lhs.blocks, rhs.blocks, dimension_numbers=dim_nums)

            out_dims = [d._replace(val_dim=i) for i, d in enumerate(lhs.out_dims)]

            primal_dims = [
                d._replace(
                    id=d.id - len(rhs.out_dims) + len(lhs.out_dims),
                    val_dim=i + len(lhs.out_dims),
                )
                for i, d in enumerate(rhs.primal_dims)
            ]

            return BlockSparseTensor(out_dims, primal_dims, val)

    elif isinstance(rhs, SparseTensor):
        pass
    elif isinstance(rhs, Array):  # TODO: Fix default check
        block_nums = lhs.blocks.shape[: lhs.sparse_dims]
        block_sizes = [
            d.block_size for d in lhs.primal_dims if isinstance(d, SparseDimension)
        ]

        rhs = rhs.reshape(*block_nums, *block_sizes, *rhs.shape[lhs.sparse_dims :])

        block_idxs = tuple(range(lhs.sparse_dims))
        lhs_val_dims = [d.val_dim + lhs.sparse_dims for d in lhs.primal_dims]
        rhs_val_dims = [i + lhs.sparse_dims for i in range(len(lhs_val_dims))]
        dim_nums = ((lhs_val_dims, rhs_val_dims), (block_idxs, block_idxs))

        res = lax.dot(lhs.blocks, rhs, dimension_numbers=dim_nums)
        return res.reshape(
            lhs.out_shape + rhs.shape[len(lhs.primal_dims) + lhs.sparse_dims :]
        )
    else:
        raise TypeError(
            "Expected to matmul with type BlockSparseTensor, SparseTensor, or Array"
        )
