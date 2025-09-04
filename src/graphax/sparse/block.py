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
                 out_shape: Sequence[int],
                 primal_shape: Sequence[int],
                 blocks: MultiSparseDimensionBlocks | Array | None,
                 sparse_dims: int,
                 pre_transforms: Sequence[Callable] = None,
                 post_transforms: Sequence[Callable] = None) -> None:

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

    def __repr__(self) -> str:
        def map_str(a: Sequence) -> Generator:
            return (str(s) for s in a)

        def multiline_seq(s: Sequence, brackets: str) -> str:
            lb, rb, *_ = brackets
            if s:
                res = f'{lb}\n\t\t' + ',\n\t\t'.join(map_str(s)) + f',\n\t{rb}'
            else:
                res = lb + rb
            return res

        str_out_shape = ', '.join(map_str(self.out_shape))
        str_primal_shape = ', '.join(map_str(self.primal_shape))

        multiline_out_dims = multiline_seq(self.out_dims, '()')
        multiline_primal_dims = multiline_seq(self.primal_dims, '()')
        multiline_pre_transform = multiline_seq(self.pre_transforms, '[]')
        multiline_post_transform = multiline_seq(self.post_transforms, '[]')

        return f"""BlockSparseTensor(
    shape = ({str_out_shape} | {str_primal_shape}),
    out_dims = {multiline_out_dims},
    primal_dims = {multiline_primal_dims},
    blocks = Array(shape={self.blocks.shape}, dtype={self.blocks.dtype}),
    sparse_dims = {self.sparse_dims},
    pre_transforms = {multiline_pre_transform},
    post_transforms = {multiline_post_transform}
)"""

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
                self.primal_shape,
                self.out_shape,
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

    def __add__(lhs, rhs):
        return _add(lhs, rhs)

    def __mul__(lhs, rhs):
        return _mul(lhs, rhs)

    def __matmul__(lhs, rhs):
        return _matmul(lhs, rhs)

def has_equal_depth(node: MultiSparseDimensionBlocks, depth=0):
    if isinstance(node, Array):
        return True, depth
    else:
        equality, depth = zip(*[has_equal_depth(child, depth+1) for child in node])
        local_equality = all(d == depth[0] for d in depth)
        return all(equality) and local_equality, max(depth)

# TODO, this will be quite complex...
def has_equal_shapes(node: MultiSparseDimensionBlocks, dims = Sequence[Dimension], shape=None): 
    ...

def new_block_sparse_tensor(
    out_dims: Sequence[Dimension],
    primal_dims: Sequence[Dimension],
    blocks: MultiSparseDimensionBlocks | Array | None,
    pre_transforms: Sequence[Callable] = None,
    post_transforms: Sequence[Callable] = None
) -> BlockSparseTensor:

    if pre_transforms is None:
        pre_transforms = []
    if post_transforms is None:
        post_transforms = []

    n = sum(isinstance(d, SparseDimension) for d in out_dims)
    assert n == sum(isinstance(d, SparseDimension) for d in primal_dims)

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

    # TODO add checks between primal and out dims.

    if isinstance(blocks, Array):
        assert blocks.ndim > n
        block_shape = blocks.shape[n:]
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
        #print(out_shape, primal_shape)
    else:
        raise TypeError("blocks as MultiSparseDimensionBlocks is not yet implemented")
        assert has_equal_depth(blocks) == (True, n)
        

    return BlockSparseTensor(
        out_dims,
        primal_dims,
        out_shape,
        primal_shape,
        blocks,
        n,
        pre_transforms,
        post_transforms
    )

def get_ienumerated_blocks(seq: Sequence, cur_idx: list[int] = None) -> Iterable[tuple[list[int], Array]]:
    if cur_idx is None:
        cur_idx = []
    for i, elem in enumerate(seq):
        if isinstance(elem, Sequence):
            for res in get_ienumerated_blocks(elem, cur_idx + [i]):
                yield res
        elif isinstance(elem, Array):
            yield cur_idx + [i], elem

def _dense(bst: BlockSparseTensor) -> Array:
    shape = bst.out_shape + bst.primal_shape
    dense_tensor = jnp.zeros(shape, dtype=bst.blocks.dtype)
    start_coords = [0] * len(shape)
    blocks = bst.blocks.transpose(
        *range(bst.sparse_dims), 
        *(d.val_dim+bst.sparse_dims 
          for d in bst.out_dims+bst.primal_dims)
    ) # not great but good enough
    
    # TODO: blocks break when the shapes are different across the different dimensions, and you have val_dims different not in the same order as id...
    # ^ this breaks Transpose.
    
    # print(f"{val_id=}")
    # print(f"{bst.sparse_dims=}")
    # print(f"{bst.block_shape=}")
    # print("dims = (", *(bst.primal_dims+bst.out_dims), sep='\n  ', end="\n)\n")
    for idxs in np.ndindex(bst.blocks.shape[:bst.sparse_dims]):
        i = 0
        for dim1 in bst.out_dims:
            if isinstance(dim1, SparseDimension):
                dim2 = bst.primal_dims[dim1.other_id-len(bst.out_dims)]
    
                start_coords[dim1.id] = dim1.block_size*idxs[i]
                start_coords[dim2.id] = dim2.block_size*idxs[i]
    
                i += 1
        # print(f"{idxs=}, {start_coords=}")
        # print(f"{blocks[idxs].shape=}")
        dense_tensor = lax.dynamic_update_slice(
            dense_tensor,
            blocks[idxs],
            start_coords
        )
        # print(blocks[idxs])
    # print()
    return dense_tensor

# @partial(jit, static_argnames=('rhs', 'lhs'))
def _add(rhs, lhs):
    assert lhs.shape == rhs.shape, "Tensors must be of equal shape"
    if isinstance(rhs, BlockSparseTensor):
        if rhs.blocks is None:
            pass
        elif isinstance(lhs.blocks, Array) and isinstance(rhs.blocks, Array):
            if lhs.shape == rhs.shape and lhs.primal_dims == rhs.primal_dims and lhs.out_dims == rhs.out_dims:
                return BlockSparseTensor(lhs.out_dims, lhs.primal_dims, lhs.out_shape, lhs.primal_shape, lhs.blocks + rhs.blocks, lhs.sparse_dims)
        elif all(b1.shape == b2.shape for b1, b2 in zip(lhs.blocks, rhs.blocks)):
            pass
    elif isinstance(rhs, SparseTensor):
        pass
    elif isinstance(rhs, Array):
        lhs.dense() + rhs
    else:
        raise TypeError("Expected to add with type BlockSparseTensor, SparseTensor, or Array")

# @partial(jit, static_argnames=('rhs', 'lhs'))
def _mul(rhs, lhs):
    assert rhs.shape == lhs.shape, "Tensors must be of equal shape"
    if isinstance(lhs, BlockSparseTensor):
        if lhs.blocks is None:
            pass
        elif isinstance(rhs.blocks, Array) and isinstance(lhs.blocks, Array):
            if rhs.shape == lhs.shape and rhs.primal_dims == lhs.primal_dims and rhs.out_dims == lhs.out_dims:
                return BlockSparseTensor(rhs.out_dims, rhs.primal_dims, rhs.out_shape, rhs.primal_shape, rhs.blocks * lhs.blocks, rhs.sparse_dims)
        elif all(b1.shape == b2.shape for b1, b2 in zip(rhs.blocks, lhs.blocks)):
            pass
    elif isinstance(lhs, SparseTensor):
        pass
    elif isinstance(lhs, Array):
        rhs.dense() + lhs
    else:
        raise TypeError("Expected to add with type BlockSparseTensor, SparseTensor, or Array")

# @partial(jit, static_argnames=('rhs', 'lhs'))
def _matmul(rhs, lhs):
    # TODO assert something
    if isinstance(lhs, BlockSparseTensor):
        if rhs.blocks is None:
            return copy.copy(lhs)
        elif lhs.blocks is None:
            return copy.copy(rhs)
        elif isinstance(rhs.blocks, Array) and isinstance(lhs.blocks, Array):
            if rhs.out_shape == lhs.primal_shape and lhs.sparse_dims == rhs.sparse_dims:
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
                    lhs.out_shape,
                    rhs.primal_shape,
                    lax.dot_general(lhs.blocks, rhs.blocks, (([x.val_dim + lhs.sparse_dims for x in lhs.primal_dims], [x.val_dim + rhs.sparse_dims for x in rhs.out_dims]), (list(range(lhs.sparse_dims)),)*2)),
                    lhs.sparse_dims
                )
        elif all(b1.shape == b2.shape for b1, b2 in zip(rhs.blocks, lhs.blocks)):
            pass
    elif isinstance(lhs, SparseTensor):
        pass
    elif isinstance(lhs, Array):  # TODO: Fix default check
        return rhs.dense() @ lhs
    else:
        raise TypeError("Expected to add with type BlockSparseTensor, SparseTensor, or Array")
