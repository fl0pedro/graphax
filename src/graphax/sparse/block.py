import copy
from collections.abc import Iterator, Sequence
from itertools import chain, count
from math import prod
from typing import (
    Callable,
    NamedTuple,
)

import jax.numpy as jnp
from jax import Array, lax
from jax.tree_util import register_pytree_node_class
from jax.typing import DTypeLike
from numpy import ndarray


class DenseDimension(NamedTuple):
    id: int
    size: int
    val_dim: int | None


class SparseDimension(NamedTuple):
    id: int
    size: int
    val_dim: int | None
    other_id: int
    block_size: int | None = None
    block_dim: int | None = None


Dimension = DenseDimension | SparseDimension
Block = Array | None
Transform = Callable[[Array], Array]
SparseTensorAux = tuple[
    tuple[Dimension, ...], tuple[Dimension, ...], list[Transform], list[Transform]
]


def _is_sparse_tensor(
    out_dims: Sequence[Dimension], primal_dims: Sequence[Dimension]
) -> bool:
    has_sparse = False
    for d in chain(out_dims, primal_dims):
        if isinstance(d, SparseDimension):
            has_sparse = True
            if d.block_size not in (None, 1):
                return False
    return has_sparse


@register_pytree_node_class
class BlockSparseTensor:
    out_dims: tuple[Dimension, ...]
    primal_dims: tuple[Dimension, ...]
    val: Block
    pre_transforms: list[Transform]
    post_transforms: list[Transform]

    def __new__(
        cls,
        out_dims: Sequence[Dimension],
        primal_dims: Sequence[Dimension],
        *args,
        **kwargs,
    ) -> "BlockSparseTensor | SparseTensor":
        if cls is BlockSparseTensor and _is_sparse_tensor(out_dims, primal_dims):
            return super().__new__(SparseTensor)  # pyright: ignore[reportArgumentType]

        return super().__new__(cls)

    def __init__(
        self,
        out_dims: Sequence[Dimension],
        primal_dims: Sequence[Dimension],
        val: Array | ndarray | None,
        pre_transforms: Sequence[Transform] | None = None,
        post_transforms: Sequence[Transform] | None = None,
    ) -> None:
        if pre_transforms is None:
            pre_transforms = []
        if post_transforms is None:
            post_transforms = []

        self.out_dims = tuple(
            d._replace(block_size=1)
            if isinstance(d, SparseDimension) and d.block_size is None
            else d
            for d in out_dims
        )
        self.primal_dims = tuple(
            d._replace(block_size=1)
            if isinstance(d, SparseDimension) and d.block_size is None
            else d
            for d in primal_dims
        )

        self.val = None if val is None else jnp.array(val)
        self.pre_transforms = list(pre_transforms)
        self.post_transforms = list(post_transforms)

    @property
    def dims(self) -> tuple[Dimension, ...]:
        return self.out_dims + self.primal_dims

    @property
    def out_shape(self) -> tuple[int, ...]:
        return tuple(
            d.size * d.block_size if isinstance(d, SparseDimension) else d.size
            for d in self.out_dims
        )

    @property
    def primal_shape(self) -> tuple[int, ...]:
        return tuple(
            d.size * d.block_size if isinstance(d, SparseDimension) else d.size
            for d in self.primal_dims
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.out_shape + self.primal_shape

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def sparse_pairs(self) -> list[tuple[int, int]]:
        return [
            (d.id, d.other_id) for d in self.out_dims if isinstance(d, SparseDimension)
        ]

    @property
    def sparse_shape(self) -> tuple[int, ...]:
        return tuple(self.out_dims[i].size for i, _ in self.sparse_pairs)

    @property
    def sparse_size(self) -> int:
        return prod(self.sparse_shape)

    @property
    def sparse_ndim(self) -> int:
        return len(self.sparse_shape)

    @property
    def dense_shape(self) -> tuple[int, ...]:
        return tuple(  # pyright: ignore[reportReturnType]
            d.block_size if isinstance(d, SparseDimension) else d.size
            for d in self.dims
        )

    @property
    def dense_size(self) -> int:
        return prod(self.dense_shape)

    @property
    def dense_ndim(self) -> int:
        return len(self.dense_shape)

    def tree_flatten(
        self,
    ) -> tuple[tuple[Block], SparseTensorAux]:
        return (
            (self.val,),
            (
                self.out_dims,
                self.primal_dims,
                self.pre_transforms,
                self.post_transforms,
            ),
        )

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: SparseTensorAux,
        children: tuple[Block],
    ) -> "BlockSparseTensor":
        (blocks,) = children
        out_dims, primal_dims, pre_transforms, post_transforms = aux_data
        return cls(out_dims, primal_dims, blocks, pre_transforms, post_transforms)

    def __repr__(self) -> str:
        def map_str(a: Sequence[Dimension | Transform]) -> Iterator[str]:
            return (str(s) for s in a)

        def multiline_seq(s: Sequence, brackets: str) -> str:
            lb, rb, *_ = brackets
            res = f"{lb}\n    " + ",\n    ".join(map_str(s)) + f",\n  {rb}"
            return res if s else lb + rb

        out_dims = multiline_seq(self.out_dims, "()")
        primal_dims = multiline_seq(self.primal_dims, "()")
        pre_transform = multiline_seq(self.pre_transforms, "[]")
        post_transform = multiline_seq(self.post_transforms, "[]")

        return (
            f"BlockSparseTensor(\n"
            f"  shape = {self._repr_shape()},\n"
            f"  out_dims = {out_dims},\n"
            f"  primal_dims = {primal_dims},\n"
            f"  val = {self._repr_val()},\n"
            f"  pre_transforms = {pre_transform},\n"
            f"  post_transforms = {post_transform}\n"
            f")"
        )

    def _repr_val(self) -> str:
        if self.val is not None:
            return f"Array(shape=({str(list(self.sparse_shape))[1:-1]}{'; ' if self.sparse_shape else ''}{str(list(self.dense_shape))[1:-1]}), dtype={self.dtype})"
        return "None"

    def _repr_shape(self) -> str:
        return f"({str(list(self.out_shape))[1:-1]} | {str(list(self.primal_shape))[1:-1]})"

    def transpose(
        self,
        out_transpose: Sequence[int] | None = None,
        primal_transpose: Sequence[int] | None = None,
    ) -> "BlockSparseTensor":
        return _transpose(self, out_transpose, primal_transpose)

    def swapdims(self) -> "BlockSparseTensor":
        return self.transpose(
            [d.id for d in self.primal_dims], [d.id for d in self.out_dims]
        )

    def block_until_ready(self) -> "BlockSparseTensor":
        if self.val is not None:
            self.val.block_until_ready()
        return self

    @property
    def T(self) -> "BlockSparseTensor":
        return self.transpose()

    @property
    def dtype(self) -> DTypeLike:
        if self.val is None:
            return type(None)
        return self.val.dtype

    def dense(self) -> "Array":
        return _dense(self)

    def all(self) -> Array:
        if self.val is None:
            return jnp.array(False)
        return self.val.all()

    def __eq__(self, rhs) -> "BlockSparseTensor":  # pyright: ignore[reportIncompatibleMethodOverride]
        return _eq(self, rhs)

    def __req__(self, lhs):
        return _eq(self, lhs)

    def __add__(self, rhs):
        return _add(self, rhs)

    def __radd__(self, lhs):
        return _add(self, lhs)

    def __mul__(self, rhs):
        return _mul(self, rhs)

    def __rmul__(self, lhs):
        return _mul(self, lhs)

    def __matmul__(self, rhs):
        return _matmul(self, rhs)

    def __rmatmul__(self, lhs):
        return _matmul(self.T, lhs.T).T

    def __copy__(self, val=None):
        return _copy(self, val)

    def __deepcopy__(self):
        return _copy(self, deep=True)


class SparseTensor(BlockSparseTensor):
    def __new__(
        cls,
        out_dims: Sequence[Dimension],
        primal_dims: Sequence[Dimension],
        *args,
        **kwargs,
    ) -> "BlockSparseTensor | SparseTensor":
        if cls is SparseTensor and not _is_sparse_tensor(out_dims, primal_dims):
            return super().__new__(
                BlockSparseTensor,  # pyright: ignore[reportArgumentType]
                out_dims,
                primal_dims,
                *args,
                **kwargs,
            )

        return super().__new__(cls, out_dims, primal_dims, *args, **kwargs)

    def __init__(self, out_dims, primal_dims, val, *args, **kwargs):
        super().__init__(out_dims, primal_dims, val, *args, **kwargs)

    def __repr__(self):
        res = super().__repr__()[5:]
        return res.replace(", block_size=None, block_dim=None", "")


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
    bst: BlockSparseTensor,
    out_transpose: Sequence[int] | None = None,
    primal_transpose: Sequence[int] | None = None,
):
    full_transpose, out_transpose, primal_transpose = _process_empty_transpose(
        len(bst.out_dims), len(bst.primal_dims), out_transpose, primal_transpose
    )

    sparse_axes = _sparse_transpose(
        bst.sparse_ndim, out_transpose, bst.out_dims, bst.primal_dims
    )
    dense_axes, new_dims = _dense_transpose_and_new_dims(
        bst.sparse_ndim, full_transpose, bst.out_dims + bst.primal_dims
    )

    return BlockSparseTensor(
        new_dims[: len(out_transpose)],
        new_dims[len(out_transpose) :],
        bst.val.transpose(sparse_axes + dense_axes),
        bst.pre_transforms,
        bst.post_transforms,
    )


def _block_diag_raw_dense(blocks, sparse_shape, dense_shape):
    assert len(sparse_shape) > 0
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
    bst: BlockSparseTensor,
):
    if bst.val is None:
        bst.val = jnp.ones(1)
    if bst.sparse_ndim == 0:
        return jnp.broadcast_to(bst.val, bst.shape)

    # TODO for the Nones cases, generate the actual block values, this will make things easier and should still be efficient.

    sparse_shape = bst.val.shape[: bst.sparse_ndim]
    dense_shape = bst.val.shape[bst.sparse_ndim :]

    block_diag = _block_diag_raw_dense(bst.val, sparse_shape, dense_shape)
    axes = _transpose_dense(bst.out_dims + bst.primal_dims, bst.sparse_ndim)
    transposed_tensor = block_diag.transpose(axes)

    return transposed_tensor.reshape(bst.shape)


def _copy(bst, val=None, deep=False):
    if val is None:
        if deep:
            val = copy.deepcopy(bst.val)
        else:
            val = bst.val

    return BlockSparseTensor(
        bst.out_dims, bst.primal_dims, val, bst.pre_transforms, bst.post_transforms
    )


def _eq(lhs, rhs):
    if not isinstance(rhs, BlockSparseTensor):
        raise ValueError("Cannot compare BlockSparseTensor with non-BlockSparseTensor")
    elif lhs.out_dims != rhs.out_dims or lhs.primal_dims != rhs.primal_dims:
        raise ValueError("Cannot compare BlockSparseTensors with different dimensions")
    else:
        return _copy(lhs, lhs.val == rhs.val)


# TODO TODO TODO TODO TODO
# the None cases!!!


def _add(lhs, rhs):
    assert rhs.shape == lhs.shape, "Tensors must be of equal shape"
    if isinstance(rhs, BlockSparseTensor):
        if rhs.val is None:
            if (
                rhs.shape == lhs.shape
                and rhs.primal_dims == lhs.primal_dims
                and rhs.out_dims == lhs.out_dims
            ):
                res = lhs.copy()
                res.val = res.val + 1
                return res
            else:
                return lhs.dense() + rhs.dense()  # worst case ...
        elif (
            rhs.shape == lhs.shape
            and rhs.primal_dims == lhs.primal_dims
            and rhs.out_dims == lhs.out_dims
        ):
            return BlockSparseTensor(lhs.out_dims, lhs.primal_dims, lhs.val + rhs.val)
    elif isinstance(rhs, Array):
        return lhs.dense() + rhs
    else:
        raise TypeError(
            "Expected to add with type BlockSparseTensor, SparseTensor, or Array"
        )


def _mul(lhs, rhs):
    assert lhs.shape == rhs.shape, "Tensors must be of equal shape"
    if isinstance(rhs, BlockSparseTensor):
        if rhs.val is None:
            if (
                rhs.shape == lhs.shape
                and rhs.primal_dims == lhs.primal_dims
                and rhs.out_dims == lhs.out_dims
            ):
                return lhs.copy()
            else:
                return lhs.dense() * rhs.dense()  # worst case ...
        elif (
            lhs.shape == rhs.shape
            and lhs.primal_dims == rhs.primal_dims
            and lhs.out_dims == rhs.out_dims
        ):
            return BlockSparseTensor(lhs.out_dims, lhs.primal_dims, lhs.val * rhs.val)
    elif isinstance(rhs, Array):
        return lhs.dense() + rhs
    else:
        raise TypeError(
            "Expected to add with type BlockSparseTensor, SparseTensor, or Array"
        )


def _matmul(lhs, rhs):
    if isinstance(rhs, BlockSparseTensor):
        if lhs.val is None:
            return _copy(rhs, deep=True)
        elif rhs.val is None:
            return _copy(lhs, deep=True)
        elif (
            isinstance(lhs.val, Array)
            and isinstance(rhs.val, Array)
            and lhs.primal_shape == rhs.out_shape
            and lhs.sparse_ndim == rhs.sparse_ndim
        ):
            lhs_sparse_idxs = tuple(range(lhs.sparse_ndim))
            lhs_sparse_idxs = tuple(range(rhs.sparse_ndim))
            # if they are different you need to make one dense in one axis
            lhs_val_dims = [x.val_dim + lhs.sparse_ndim for x in lhs.primal_dims]
            rhs_val_dims = [x.val_dim + rhs.sparse_ndim for x in rhs.out_dims]
            dim_nums = (
                (lhs_val_dims, rhs_val_dims),
                (lhs_sparse_idxs, lhs_sparse_idxs),
            )

            val = lax.dot(lhs.val, rhs.val, dimension_numbers=dim_nums)

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
                f"Can't multiply shapes: {lhs._repr_shape()} and {rhs._repr_shape()}"
            )

    elif isinstance(rhs, Array):  # TODO: Fix default check
        block_nums = lhs.val.shape[: lhs.sparse_ndim]
        block_sizes = [
            d.block_size for d in lhs.primal_dims if isinstance(d, SparseDimension)
        ]

        rhs = rhs.reshape(*block_nums, *block_sizes, *rhs.shape[lhs.sparse_ndim :])

        block_idxs = tuple(range(lhs.sparse_ndim))
        lhs_val_dims = [d.val_dim + lhs.sparse_ndim for d in lhs.primal_dims]
        rhs_val_dims = [i + lhs.sparse_ndim for i in range(len(lhs_val_dims))]
        dim_nums = ((lhs_val_dims, rhs_val_dims), (block_idxs, block_idxs))

        res = lax.dot(lhs.val, rhs, dimension_numbers=dim_nums)
        return res.reshape(
            lhs.out_shape + rhs.shape[len(lhs.primal_dims) + lhs.sparse_ndim :]
        )
    else:
        raise TypeError(
            "Expected to matmul with type BlockSparseTensor, SparseTensor, or Array"
        )
