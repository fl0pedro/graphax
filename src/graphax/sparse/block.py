from typing import Any, Callable, Sequence, Generator, TypeAlias, Iterable, NamedTuple

# import numpy as np
from chex import Array

import jax
from jax import lax, jit
import jax.numpy as jnp
from .tensor import SparseTensor
import operator
from dataclasses import dataclass, KW_ONLY
import copy
from functools import reduce, partial
import numpy as np
from jax.tree_util import register_pytree_node_class

# TODO: make parent class, or inherit sparse tensor ??

class DenseDimension(NamedTuple):
    id: int
    size: int
    val_dim: int | None
    #val_axis: int = None

# val_axes: These are two axis of the blocks that we would like to apply the diagonal on.
#   By default pick the first two (0, 1) as the axis). TODO: make negatives work like indeces
class SparseDimension(NamedTuple):
    id: int
    size: int
    val_dim: int
    other_id: int
    block_size: int = None
    #val_axis: int = None

# TODO TODO TODO TODO TODO, the new idea is to setup Sparse dim, such that if two different val_dims are set for a pair,
# then its sparse block...


Dimension = DenseDimension | SparseDimension

# find a better name for this.
MultiSparseDimensionBlocks: TypeAlias = Sequence[Array] | Sequence['MultiSparseDimensionBlocks']

# TODO (somewhere) blocks of blocks with neighbors can be combined to one block
# I believe that tensors with d diagonal then there will be 2d blocks maximum (the rest can be merged into individual blocks)
@register_pytree_node_class
class BlockSparseTensor:
    out_dims: Any
    primal_dims: Any
    out_shape: tuple[int, ...]
    primal_shape:  tuple[int, ...]
    shape:  tuple[int, ...]
    size: int
    ndim: int
    blocks: MultiSparseDimensionBlocks | Array | None
    pre_transforms: Array
    post_transforms: Array
    sparse_dims: int
    block_shape: tuple[int, ...]
    block_size: int
    _sparse_dim_order: list[tuple[int, int]]

    def __init__(self,
        out_dims: Sequence[Dimension],
        primal_dims: Sequence[Dimension],
        blocks: MultiSparseDimensionBlocks | Array | None,
        pre_transforms: Sequence[Callable] = None,
        post_transforms: Sequence[Callable] = None
        ) -> None:

        if pre_transforms is None:
            pre_transforms = []
        if post_transforms is None:
            post_transforms = []

        sparse_dims = sum(isinstance(d, SparseDimension) for d in out_dims)
        assert sparse_dims == sum(isinstance(d, SparseDimension) for d in primal_dims)

        #assert all(d == i for d, i in zip(sorted_val_dims, list(range(n)))), \
        #    "Value dimensions should be continuous"
        
        #print(sorted_val_dims)
        #print(
        #    {d.val_axis for d in out_dims if isinstance(d, DenseDimension)} 
        #    | {d.val_axis for d in primal_dims}
        #)
        #assert n == len(
        #        {d.val_axis for d in out_dims if isinstance(d, DenseDimension)} 
        #        | {d.val_axis for d in primal_dims}
        #    ), "Value axis should be unique"

        #assert blocks.ndim > sparse_dims # <-- breaks jit stuff
        block_shape = blocks.shape[sparse_dims:]
        # TODO add ones for non mentioned areas

        #print(blocks.shape)
        #print(block_shape)

        out_shape = tuple(
            x.size if isinstance(x, DenseDimension)
            else x.size * block_shape[x.val_dim]
            for x in out_dims
        )
        primal_shape = tuple(
            x.size if isinstance(x, DenseDimension)
            else x.size * block_shape[x.val_dim]
            for x in primal_dims
        )

        self.out_dims = out_dims if isinstance(out_dims, tuple) else tuple(out_dims)
        self.primal_dims = primal_dims if isinstance(primal_dims, tuple) else tuple(primal_dims)

        self.sparse_dims = sparse_dims
        self.block_shape = blocks.shape[self.sparse_dims:]
        self.block_size = reduce(operator.mul, self.block_shape)

        self.out_shape = tuple(out_shape)

        self.primal_shape = tuple(primal_shape)

        # self.out_shape = get_block_shape(out_dims)
        # self.primal_shape = get_block_shape(primal_dims)

        self.shape = tuple(self.out_shape + self.primal_shape) # isn't quite right
        self.size = reduce(operator.mul, self.shape)
        self.ndim = len(self.shape)

        self.blocks = blocks

        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms

        self._sparse_dim_order = [(d.id, d.other_id) for d in self.out_dims if isinstance(d, SparseDimension)] # may not be necessary to include, but must be mentioned in the docs

    def tree_flatten(self):
        return ((self.blocks,), (self.out_dims, self.primal_dims, self.pre_transforms, self.post_transforms))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        blocks, = children
        out_dims, primal_dims, pre_transforms, post_transforms = aux_data
        return cls(out_dims, primal_dims, blocks, pre_transforms, post_transforms)

    def __repr__(self) -> str:
        def map_str(a: Sequence) -> Generator:
            return (str(s) for s in a)

        def multiline_seq(s: Sequence, brackets: str) -> str:
            lb, rb, *_ = brackets
            if s:
                res = f'{lb}\n    ' + ',\n    '.join(map_str(s)) + f',\n  {rb}'
            else:
                res = lb + rb
            return res

        str_out_shape = ', '.join(map_str(self.out_shape))
        str_primal_shape = ', '.join(map_str(self.primal_shape))

        multiline_out_dims = multiline_seq(self.out_dims, '()')
        multiline_primal_dims = multiline_seq(self.primal_dims, '()')
        multiline_pre_transform = multiline_seq(self.pre_transforms, '[]')
        multiline_post_transform = multiline_seq(self.post_transforms, '[]')

        return f"BlockSparseTensor(\n" \
               f"  shape = ({str_out_shape} | {str_primal_shape}),\n" \
               f"  out_dims = {multiline_out_dims},\n" \
               f"  primal_dims = {multiline_primal_dims},\n" \
               f"  val = Array(shape={self.blocks.shape}, dtype={self.blocks.dtype}),\n" \
               f"  sparse_dims = {self.sparse_dims},\n" \
               f"  pre_transforms = {multiline_pre_transform},\n" \
               f"  post_transforms = {multiline_post_transform}\n" \
               f")"

    # This is not a transpose like w/ normal tensors. The order should be completely reversed.
    def transpose(self, *args):
        if len(args) > 0:
            pass # regular transpose, warn on breaking of sparsity
        else:
            out_dims = [
                d._replace(
                    id=d.id - len(self.out_dims),
                    other_id=d.other_id + len(self.primal_dims),
                )
                if isinstance(d,SparseDimension)
                else d._replace(id=d.id - len(self.out_dims))
                for d in self.primal_dims
            ]
            primal_dims = [
                d._replace(
                    id=d.id + len(self.primal_dims),
                    other_id=d.other_id - len(self.out_dims),
                )
                if isinstance(d, SparseDimension) 
                else d._replace(id=d.id + len(self.primal_dims))
                for d in self.out_dims
            ]
            return BlockSparseTensor(
                out_dims,
                primal_dims,
                self.blocks,
                self.sparse_dims
            )

    def block_until_ready(self):
        self.blocks.block_until_ready()
        return self

    @property
    def T(self):
        return self.transpose()

    def dense(self) -> Array:
        return _dense(self)

    def __add__(rhs, lhs):
        return _add(rhs, lhs)

    def __mul__(rhs, lhs):
        return _mul(rhs, lhs)

    def __matmul__(rhs, lhs):
        return _matmul(rhs, lhs)

def ndindex(*shape):
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    return jnp.indices(shape).reshape(len(shape),-1).T

def _dense(bst: BlockSparseTensor) -> Array: # really slow on GPU :(

    # check efficient cases, already dense and blocksize = 1

    dense_tensor = jnp.zeros(bst.shape, dtype=bst.blocks.dtype)

    val_shape = bst.blocks.shape[bst.sparse_dims:]
    blocks = bst.blocks.reshape(-1, *val_shape).transpose(
        0, *(d.val_dim+1 for d in bst.out_dims+bst.primal_dims)
    )

    increments = jnp.empty(len(bst.shape), dtype=jnp.int8)
    for dim1 in bst.out_dims:
        if isinstance(dim1, SparseDimension):
            dim2 = bst.primal_dims[dim1.other_id-len(bst.out_dims)]
            increments = increments.at[dim1.id].set(dim1.block_size)
            increments = increments.at[dim2.id].set(dim2.block_size)

    start_coords = jax.vmap(lambda idxs: jnp.where(increments > 0, idxs*increments, 0))(
        ndindex(*bst.blocks.shape[:bst.sparse_dims])
    )

    def calc(dense_tensor, args):
        return lax.dynamic_update_slice(dense_tensor, *args), None

    dense_tensor, _ = lax.scan(calc, dense_tensor, (blocks, start_coords))

    return dense_tensor

def _add(lhs, rhs):
    assert rhs.shape == lhs.shape, "Tensors must be of equal shape"
    if isinstance(lhs, BlockSparseTensor):
        if lhs.blocks is None:
            pass
        elif isinstance(rhs.blocks, Array) and isinstance(lhs.blocks, Array):
            if rhs.shape == lhs.shape and rhs.primal_dims == lhs.primal_dims and rhs.out_dims == lhs.out_dims:
                return BlockSparseTensor(rhs.out_dims, rhs.primal_dims, rhs.blocks + lhs.blocks)
        elif all(b1.shape == b2.shape for b1, b2 in zip(rhs.blocks, lhs.blocks)):
            pass
    elif isinstance(lhs, SparseTensor):
        pass
    elif isinstance(lhs, Array):
        rhs.dense() + lhs
    else:
        raise TypeError("Expected to add with type BlockSparseTensor, SparseTensor, or Array")

def _mul(lhs, rhs):
    assert lhs.shape == rhs.shape, "Tensors must be of equal shape"
    if isinstance(rhs, BlockSparseTensor):
        if rhs.blocks is None:
            pass
        elif isinstance(lhs.blocks, Array) and isinstance(rhs.blocks, Array):
            if lhs.shape == rhs.shape and lhs.primal_dims == rhs.primal_dims and lhs.out_dims == rhs.out_dims:
                return BlockSparseTensor(lhs.out_dims, lhs.primal_dims, lhs.blocks * rhs.blocks)
        elif all(b1.shape == b2.shape for b1, b2 in zip(lhs.blocks, rhs.blocks)):
            pass
    elif isinstance(rhs, SparseTensor):
        pass
    elif isinstance(rhs, Array):
        lhs.dense() + rhs
    else:
        raise TypeError("Expected to add with type BlockSparseTensor, SparseTensor, or Array")

def _matmul(lhs, rhs):
    # TODO assert something
    if isinstance(rhs, BlockSparseTensor):
        if lhs.blocks is None:
            return copy.copy(rhs)
        elif rhs.blocks is None:
            return copy.copy(lhs)
        elif isinstance(lhs.blocks, Array) and isinstance(rhs.blocks, Array) \
                and lhs.out_shape == rhs.primal_shape and rhs.sparse_dims == lhs.sparse_dims:
            out_dims = [
                d._replace(val_dim=i) 
                for i, d in enumerate(lhs.out_dims)
            ]
            
            primal_dims = [
                d._replace(
                    id=d.id - len(rhs.out_dims) + len(lhs.out_dims),
                    val_dim=i+len(lhs.out_dims)
                )
                for i, d in enumerate(rhs.primal_dims)
            ]
            
            return BlockSparseTensor(
                out_dims,
                primal_dims,
                lax.dot_general(lhs.blocks, rhs.blocks, (([x.val_dim + lhs.sparse_dims for x in lhs.primal_dims], [x.val_dim + rhs.sparse_dims for x in rhs.out_dims]), (list(range(lhs.sparse_dims)),)*2)),
            )
    elif isinstance(rhs, SparseTensor):
        pass
    elif isinstance(rhs, Array):  # TODO: Fix default check
        block_nums = lhs.blocks.shape[:lhs.sparse_dims]
        block_idxs = tuple(range(lhs.sparse_dims))
        block_sizes = [d.block_size for d in lhs.primal_dims if isinstance(d, SparseDimension)]
        val_dims = tuple(d.val_dim+lhs.sparse_dims for d in lhs.primal_dims )
        
        #print(f"{lhs.shape=}, {rhs.shape=}")
        
        #print(f"{block_nums} + {block_sizes}")
        #print(f"{lhs.sparse_dims=} -> {rhs.shape[lhs.sparse_dims:]}")
        #print(f"{len(lhs.primal_dims)=} -> {rhs.shape[-len(lhs.primal_dims):]}")
        rhs = rhs.reshape(*block_nums, *block_sizes, *rhs.shape[lhs.sparse_dims:]) #*rhs.shape[-len(lhs.primal_dims):])
        
        #print(f"{lhs.blocks.shape=}, {rhs.shape=}")
        #print(f"{val_dims=}, {block_nums=}")
        
        rhs_val_dims = tuple(i+lhs.sparse_dims for i in range(len(val_dims)))
        dim_nums = (
            (val_dims, rhs_val_dims),
            (block_idxs, block_idxs)
        )
        
        #print(f"{dim_nums=}")

        res = lax.dot(lhs.blocks, rhs, dimension_numbers=dim_nums)
        
        #print(f"{lhs.out_shape=}")
        # print(f"{res.shape=}")
        # print(f"{lhs.sparse_dims=}")
        # print(lhs.out_shape, "+", rhs.shape[:len(lhs.primal_dims)])

        #print(res.reshape(*[x for i, x in enumerate(lhs.blocks.shape) if i not in val_dims]+*[x for i, x in enumerate(rhs.shape) if i-lhs.sparse_dims not in ...))
        #res = res.reshape(
        res = res.reshape(lhs.out_shape + rhs.shape[len(lhs.primal_dims)+lhs.sparse_dims:])
        #print(f"{res.shape=}")
        return res
    else:
        raise TypeError("Expected to matmul with type BlockSparseTensor, SparseTensor, or Array")

def _rmatmul(lhs, rhs):
    if isinstance(lhs, SparseTensor):
        pass
    elif isinstance(lhs, Array):
        block_nums = rhs.blocks.shape[:rhs.sparse_dims]
        block_idxs = tuple(range(rhs.sparse_dims))
        block_sizes = [d.block_size for d in rhs.primal_dims if isinstance(d, SparseDimension)]
        val_dims = [d.val_dim+rhs.sparse_dims for d in rhs.primal_dims if isinstance(d, SparseDimension)]

        # print(rhs.shape, lhs.shape)

        # note the -1 for arbitrary last dimension (*)
        lhs = lhs.reshape(*block_nums, *block_sizes, -1)
        transposed_axes = [rhs.primal_dims[d.other_id-len(rhs.out_dims)].val_dim for d in rhs.out_dims] \
                            + [rhs.out_dims[d.other_id].val_dim for d in rhs.primal_dims]
        # print(transposed_axes)
        rhs.rhs = rhs.blocks.transpose(0,*[x+1 for x in transposed_axes])

        # print(rhs.blocks.shape, lhs.shape)
        # print(val_dims, block_nums)

        dim_nums = (
            # both axis are the same for the two tensors
            # the -1 is for being one left of the last dimension (*)
            ([x-1 for x in val_dims],)*2,
            (block_idxs,)*2
        )

        res = lax.dot(rhs.blocks, lhs, dimension_numbers=dim_nums)
        
        #print(res.shape)
        return res.reshape(lhs.out_shape + rhs.shape[-lhs.sparse_dims:]).transpose(transposed_axes)
    else:
        raise TypeError("Expected to matmul SparseTensor or Array and BlockSparseTensor")
