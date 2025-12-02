import copy
import operator
from functools import reduce
from itertools import count
from math import prod
from typing import (
    Any,
    Callable,
    Generator,
    NamedTuple,
    Sequence,
)

import jax.numpy as jnp
from chex import Array
from jax import lax
from jax.tree_util import register_pytree_node_class

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
    val_dim: int | None
    other_id: int
    block_size: int = None
    # val_axis: int = None


Dimension = DenseDimension | SparseDimension


# TODO function like tensor._swap_axes where the val_dims match the out_/primal_dims


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
        # block_shape = blocks.shape[sparse_dims:]

        # print(blocks.shape)
        # print(block_shape)

        out_shape = tuple(
            d.size * d.block_size if hasattr(d, "block_size") else d.size
            for d in out_dims
        )
        primal_shape = tuple(
            d.size * d.block_size if hasattr(d, "block_size") else d.size
            for d in primal_dims
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

        self.shape = tuple(self.out_shape + self.primal_shape)  # isn't quite right
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

        multiline_out_dims = multiline_seq(self.out_dims, "()")
        multiline_primal_dims = multiline_seq(self.primal_dims, "()")
        multiline_pre_transform = multiline_seq(self.pre_transforms, "[]")
        multiline_post_transform = multiline_seq(self.post_transforms, "[]")

        return (
            f"BlockSparseTensor(\n"
            f"  shape = ({str(list(self.out_shape))[1:-1]} | {str(list(self.primal_shape))[1:-1]}),\n"
            f"  out_dims = {multiline_out_dims},\n"
            f"  primal_dims = {multiline_primal_dims},\n"
            f"  val = Array(shape={self.blocks.shape}, dtype={self.blocks.dtype}),\n"
            f"  pre_transforms = {multiline_pre_transform},\n"
            f"  post_transforms = {multiline_post_transform}\n"
            f")"
        )

    def transpose(
        self,
        out_transpose: Sequence[int] | None = None,
        primal_transpose: Sequence[int] | None = None,
    ):
        return _transpose(self, out_transpose, primal_transpose)

    def swapdims(self):
        print([d.id for d in self.primal_dims], [d.id for d in self.out_dims])
        return self.transpose(
            [d.id for d in self.primal_dims], [d.id for d in self.out_dims]
        )

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
        return _dense(self)

    def all(self):
        return self.blocks.all()

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


def _process_empty_transpose(
    n_out_dims, n_primal_dims, out_transpose, primal_transpose
):
    if out_transpose is None and primal_transpose is None:
        full_transpose = tuple(range(n_out_dims + n_primal_dims - 1, -1, -1))
        out_transpose = full_transpose[:n_primal_dims]
        primal_transpose = full_transpose[n_primal_dims:]
    else:
        if out_transpose is None:
            out_transpose = range(n_out_dims)
        if primal_transpose is None:
            primal_transpose = range(n_out_dims, n_primal_dims + n_out_dims)

        out_transpose = tuple(out_transpose)
        primal_transpose = tuple(primal_transpose)
        full_transpose = out_transpose + primal_transpose

    assert len(full_transpose) == n_out_dims + n_primal_dims
    assert len(set(full_transpose)) == n_out_dims + n_primal_dims
    return full_transpose, out_transpose, primal_transpose


def _sparse_transpose(sparse_dims, out_transpose, out_dims, primal_dims):
    c = count()
    sparse_id_to_pos = {
        d.id: next(c) for d in out_dims if isinstance(d, SparseDimension)
    }

    new_id = 0
    axes = [0] * sparse_dims
    dims = out_dims + primal_dims
    for i in out_transpose:
        d = dims[i]
        if isinstance(d, SparseDimension):
            id = d.id if d.id in sparse_id_to_pos else d.other_id
            axes[sparse_id_to_pos[id]] = new_id
            new_id += 1

    return axes


def _dense_transpose_and_new_dims(sparse_dims, transpose, dims):
    axes = []
    new_dims = []
    for new_id, old_id in enumerate(transpose):
        dim = dims[old_id]  # can also be new_id.
        if isinstance(dim, SparseDimension):
            dim = dim._replace(other_id=transpose.index(dim.other_id))
        dim = dim._replace(id=new_id, val_dim=new_id)
        axes.append(old_id + sparse_dims)
        new_dims.append(dim)

    return axes, new_dims


def _transpose(
    st: BlockSparseTensor,
    out_transpose: Sequence[int] | None = None,
    primal_transpose: Sequence[int] | None = None,
):
    full_transpose, out_transpose, primal_transpose = _process_empty_transpose(
        len(st.out_dims), len(st.primal_dims), out_transpose, primal_transpose
    )

    sparse_axes = _sparse_transpose(
        st.sparse_dims, out_transpose, st.out_dims, st.primal_dims
    )
    dense_axes, new_dims = _dense_transpose_and_new_dims(
        st.sparse_dims, full_transpose, st.out_dims + st.primal_dims
    )

    return BlockSparseTensor(
        new_dims[: len(out_transpose)],
        new_dims[len(out_transpose) :],
        st.blocks.transpose(sparse_axes + dense_axes),
        st.pre_transforms,
        st.post_transforms,
    )


def _block_diag_raw_dense(blocks, sparse_shape, dense_shape):
    num_blocks = prod(sparse_shape)

    blocks_flat = blocks.reshape(num_blocks, *dense_shape)
    diag_blocks_flat = jnp.einsum(
        "i...,ij->ij...", blocks_flat, jnp.eye(num_blocks, dtype=blocks.dtype)
    )

    return diag_blocks_flat.reshape(*sparse_shape, *sparse_shape, *dense_shape)


def _transpose_dense(dims: tuple[Dimension, ...], sparse_dims: int):
    axes = []
    sparse_id = 0
    for d in dims:
        if isinstance(d, SparseDimension):
            axes.append(sparse_id)
            sparse_id += 1
        axes.append(sparse_dims * 2 + d.val_dim)
    return axes


def _dense(
    st: BlockSparseTensor,
):
    if st.sparse_dims == 0:
        return jnp.broadcast_to(st.blocks, st.shape)

    # TODO for the Nones cases, generate the actual block values, this will make things easier and should still be efficient.

    sparse_shape = st.blocks.shape[: st.sparse_dims]
    dense_shape = st.blocks.shape[st.sparse_dims :]

    block_diag = _block_diag_raw_dense(st.blocks, sparse_shape, dense_shape)
    axes = _transpose_dense(st.out_dims + st.primal_dims, st.sparse_dims)
    transposed_tensor = block_diag.transpose(axes)

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
    if not isinstance(rhs, BlockSparseTensor):
        raise ValueError("Cannot compare BlockSparseTensor with non-BlockSparseTensor")
    elif lhs.out_dims != rhs.out_dims or lhs.primal_dims != rhs.primal_dims:
        raise ValueError("Cannot compare BlockSparseTensors with different dimensions")
    else:
        return _copy(lhs, lhs.blocks == rhs.blocks)


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
    elif isinstance(rhs, Array):
        return lhs.dense() + rhs
    else:
        raise TypeError(
            "Expected to add with type BlockSparseTensor, SparseTensor, or Array"
        )


def _matmul(lhs, rhs):
    if isinstance(rhs, BlockSparseTensor):
        if lhs.blocks is None:
            return _copy(rhs, deep=True)
        elif rhs.blocks is None:
            return _copy(lhs, deep=True)
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
        else:
            raise ValueError(
                f"Can't multiply shapes: ({str(list(lhs.out_shape))[1:-1]} | {str(list(lhs.primal_shape))[1:-1]}) and ({str(list(rhs.out_shape))[1:-1]} | {str(list(rhs.primal_shape))[1:-1]})"
            )

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
