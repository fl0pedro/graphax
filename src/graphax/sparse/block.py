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
    blocks: Array | None
    pre_transforms: Array
    post_transforms: Array
    sparse_dims: int
    block_shape: tuple[int, ...]
    block_size: int
    _sparse_dim_order: list[tuple[int, int]]

    def __init__(self,
        out_dims: Sequence[Dimension],
        primal_dims: Sequence[Dimension],
        blocks: Array | None,
        pre_transforms: Sequence[Callable] = None,
        post_transforms: Sequence[Callable] = None
        ) -> None:

        if pre_transforms is None:
            pre_transforms = []
        if post_transforms is None:
            post_transforms = []

        sparse_dims = sum(isinstance(d, SparseDimension) for d in out_dims)
        assert sparse_dims == sum(isinstance(d, SparseDimension) for d in primal_dims)

        # TODO: assertions 

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
        # TODO: _get_fully_materialized_shape
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
               f"  pre_transforms = {multiline_pre_transform},\n" \
               f"  post_transforms = {multiline_post_transform}\n" \
               f")"

    # This is not a transpose like w/ normal tensors. The order should be completely reversed.
    # testcase: st == st.T.T
    def transpose(self, *args, force=False):
        if len(args) == 2 and len(args[0])+len(args[1]) == self.ndim:
            # blegh TODO
            #all_idxs, _ = zip(*sorted(enumerate(args[0]+args[1]), key=lambda x: x[1])) #TODO the assert is ignored by this...
            all_idxs = args[0]+args[1]
            all_idxs = [all_idxs.index(i) for i in all_idxs]
            assert len(all_idxs) == len(set(all_idxs)), "All listed idxs must be unique"
            # TODO assert sparse dims have to be in opposite, otherwise force them to be dense. (or new dead sparse dimension, which is not treated as a sparse dimension unless transposed again)
            if force:
                raise NotImplementedError("Keyword argument `force` has not yet been implemented")
                pass # make it so that pairs of sparse dimensions in one primal_/out_dims are made dense.
            sorted_dims = sorted(zip(all_idxs, self.out_dims+self.primal_dims))

            out_dims = [
                d._replace(
                    id=j, other_id=all_idxs.index(d.other_id)
                )
                if isinstance(d, SparseDimension) 
                else d._replace(id=j)
                for j, d in sorted_dims[:len(args[0])]
            ]
            primal_dims = [
                d._replace(
                    id=j, other_id=all_idxs.index(d.other_id)
                )
                if isinstance(d, SparseDimension) 
                else d._replace(id=j)
                for j, d in sorted_dims[len(args[0]):]
            ]

            return BlockSparseTensor(
                out_dims,
                primal_dims,
                self.blocks
            )
        elif len(args) == 0: # should we transpose the values too?
            base = len(self.out_dims)+len(self.primal_dims) - 1
            out_dims = [
                d._replace(
                    id=base-d.id,
                    other_id=base-d.other_id
                )
                if isinstance(d,SparseDimension)
                else d._replace(id=base-d.id)
                for d in self.primal_dims[::-1]
            ]
            primal_dims = [
                d._replace(
                    id=base-d.id,
                    other_id=base-d.other_id
                )
                if isinstance(d, SparseDimension) 
                else d._replace(id=base-d.id)
                for d in self.out_dims[::-1]
            ]
            return BlockSparseTensor(
                out_dims,
                primal_dims,
                self.blocks
            )
        else:
            raise TypeError(f"transpose permutation isn't a permutation of operand dimensions, got permutation {args} for operand shape {self.shape}.")

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

    # missing is copy, but we will use the same logic as for tensor
    # I don't love the weird val arg

def ndindex(*shape):
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    return jnp.indices(shape).reshape(len(shape),-1).T

def _dense(bst: BlockSparseTensor) -> Array: # really slow on GPU :(

    # TODO what if val_dim is None
    # check efficient cases, already dense and blocksize = 1

    dense_tensor = jnp.zeros(bst.shape, dtype=bst.blocks.dtype)

    val_shape = bst.blocks.shape[bst.sparse_dims:]
    blocks = bst.blocks.reshape(-1, *val_shape).transpose(
        0, *(d.val_dim+1 for d in bst.out_dims+bst.primal_dims)
    )

    increments = jnp.empty(len(bst.shape), dtype=jnp.int8)
    mapping = jnp.empty((bst.sparse_dims*2,), dtype=jnp.int8)

    i = 0
    for dim in bst.out_dims+bst.primal_dims:
        if isinstance(dim, SparseDimension):
            increments = increments.at[dim.id].set(dim.block_size)
            mapping = mapping.at[i].set(dim.id)
            i+=1

    start_coords = jax.vmap(lambda idxs: jnp.where(increments > 0, idxs*increments, 0))(
            ndindex(*bst.blocks.shape[:bst.sparse_dims]).repeat(2, axis=1)[:, mapping]
    )

    def calc(dense_tensor, args):
        return lax.dynamic_update_slice(dense_tensor, *args), None

    dense_tensor, _ = lax.scan(calc, dense_tensor, (blocks, start_coords))

    return dense_tensor

def _eq(lhs, rhs):
    return (
        isinstance(rhs, BlockSparseTensor) and
        lhs.out_dims == rhs.out_dims and
        lhs.primal_dims == rhs.primal_dims and
        jnp.all(lhs.blocks == rhs.blocks)
    )

# TODO TODO TODO TODO TODO
# the None cases!!!

def _add(lhs, rhs):
    assert rhs.shape == lhs.shape, "Tensors must be of equal shape"
    if isinstance(rhs, BlockSparseTensor):
        if rhs.blocks is None:
            if rhs.shape == lhs.shape and rhs.primal_dims == lhs.primal_dims and rhs.out_dims == lhs.out_dims:
                res = lhs.copy()
                res.blocks = res.blocks + 1
                return res
            else:
                lhs.dense() + rhs.dense() # worst case ... 
        elif rhs.shape == lhs.shape and rhs.primal_dims == lhs.primal_dims and rhs.out_dims == lhs.out_dims:
            return BlockSparseTensor(lhs.out_dims, lhs.primal_dims, lhs.blocks + rhs.blocks)
    elif isinstance(rhs, SparseTensor):
        pass
    elif isinstance(rhs, Array):
        return lhs.dense() + rhs
    else:
        raise TypeError("Expected to add with type BlockSparseTensor, SparseTensor, or Array")

def _mul(lhs, rhs):
    assert lhs.shape == rhs.shape, "Tensors must be of equal shape"
    if isinstance(rhs, BlockSparseTensor):
        if rhs.blocks is None:
            if rhs.shape == lhs.shape and rhs.primal_dims == lhs.primal_dims and rhs.out_dims == lhs.out_dims:
                return lhs.copy()
            else:
                lhs.dense() * rhs.dense() # worst case ...
        elif lhs.shape == rhs.shape and lhs.primal_dims == rhs.primal_dims and lhs.out_dims == rhs.out_dims:
            return BlockSparseTensor(lhs.out_dims, lhs.primal_dims, lhs.blocks * rhs.blocks)
    elif isinstance(rhs, SparseTensor):
        pass
    elif isinstance(rhs, Array):
        return lhs.dense() + rhs
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
                and lhs.primal_shape == rhs.out_shape and lhs.sparse_dims == rhs.sparse_dims:
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
        
        rhs = rhs.reshape(*block_nums, *block_sizes, *rhs.shape[lhs.sparse_dims:])
        
        rhs_val_dims = tuple(i+lhs.sparse_dims for i in range(len(val_dims)))
        dim_nums = (
            (val_dims, rhs_val_dims),
            (block_idxs, block_idxs)
        )
        
        res = lax.dot(lhs.blocks, rhs, dimension_numbers=dim_nums)
        return res.reshape(lhs.out_shape + rhs.shape[len(lhs.primal_dims)+lhs.sparse_dims:])
    else:
        raise TypeError("Expected to matmul with type BlockSparseTensor, SparseTensor, or Array")
