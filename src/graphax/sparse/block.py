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
import numpy as np

# TODO: make parent class, or inherit sparse tensor ??

class DenseDimension(NamedTuple):
    id: int
    size: int
    val_dim: int | None
    val_axis: int = None

# val_axes: These are two axis of the blocks that we would like to apply the diagonal on.
#   By default pick the first two (0, 1) as the axis). TODO: make negatives work like indeces
class SparseDimension(NamedTuple):
    id: int
    size: int
    val_dim: int
    other_id: int
    val_axis: int = None

Dimension = DenseDimension | SparseDimension

MultiSparseDimensionBlocks: TypeAlias = Sequence[Array] | Sequence['MultiSparseDimensionBlocks']

class BlockSparseTensor:
    out_dims: Any
    primal_dims: Any
    out_shape: tuple[int]
    primal_shape:  tuple[int]
    shape:  tuple[int]
    size: int
    ndim: int
    blocks: MultiSparseDimensionBlocks | Array | None
    pre_transforms: Array
    post_transforms: Array
    elementary_block_idx: int
    block_shape: tuple[int]
    block_size: int

    def __init__(self,
                 out_dims: Sequence[Dimension],
                 primal_dims: Sequence[Dimension],
                 out_shape: Sequence[int],
                 primal_shape: Sequence[int],
                 blocks: MultiSparseDimensionBlocks | Array | None,
                 elementary_block_idx: int,
                 pre_transforms: Sequence[Callable],
                 post_transforms: Sequence[Callable]) -> None:

        self.out_dims = out_dims if isinstance(out_dims, tuple) else tuple(out_dims)
        self.primal_dims = primal_dims if isinstance(primal_dims, tuple) else tuple(primal_dims)

        self.elementary_block_idx = elementary_block_idx
        self.block_shape = blocks.shape[self.elementary_block_idx:]
        self.block_size = jnp.prod(jnp.array(self.block_shape))

        self.out_shape = tuple(out_shape)

        self.primal_shape = tuple(primal_shape)

        # self.out_shape = get_block_shape(out_dims)
        # self.primal_shape = get_block_shape(primal_dims)

        self.shape = tuple(self.out_shape + self.primal_shape) # isn't quite right
        self.size = jnp.prod(jnp.array(self.shape))
        self.ndim = len(self.shape)

        if all(b1.shape == b2.shape for b1, b2 in zip(blocks, blocks[1:])):
            self.blocks = jnp.array(blocks)
        else:
            self.blocks = blocks

        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms

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
    blocks = {self.blocks},
    elementary_block_idx = {self.elementary_block_idx},
    pre_transforms = {multiline_pre_transform},
    post_transforms = {multiline_post_transform}
)"""

    # @property
    # def out_shape(self):
    #     return tuple(self.out_shape)
    # @out_shape.setter
    # def out_shape(self, value):
    #     self.out_shape = jnp.array(value)
    #
    # @property
    # def primal_shape(self):
    #     return tuple(self.primal_shape)
    # @primal_shape.setter
    # def primal_shape(self, value):
    #     self.primal_shape = jnp.array(value)
    #
    # @property
    # def shape(self):
    #     return tuple(self.shape)
    # @shape.setter
    # def shape(self, value):
    #     self.shape = jnp.array(value)
    #
    # @property
    # def block_shape(self):
    #     return tuple(self.block_shape)
    # @block_shape.setter
    # def block_shape(self, value):
    #     self.block_shape = jnp.array(value)

    def dense(self):
        if all(isinstance(d, DenseDimension) for d in self.out_dims + self.primal_dims):
            assert isinstance(self.blocks, Array), \
                "Must be able to return a valid Array, not a Sequence"
            return self.blocks
        else:
            res = jnp.zeros(self.shape)
            if isinstance(self.blocks, Array):
                # _dense_array(self.blocks, self.primal_dims + self.out_dims, self.shape)
                res = jnp.zeros(self.shape)
                index = jnp.zeros(self.elementary_block_idx + 1)
                index_offset = index.copy()
                dim_pointer = self.elementary_block_idx
                dims = self.primal_dims + self.out_dims
                non_block_shape = self.blocks.shape[:self.elementary_block_idx]
                non_block_size = jnp.prod(jnp.array(non_block_shape))
                non_block_ndim = len(non_block_shape)

                # the shape for each dimensions regarding one block
                block_shape_per_dim = jnp.array([
                    self.block_shape[dim.val_dim]
                    if dim.val_dim is not None else 1
                    for dim in self.primal_dims + self.out_dims
                ])

                assert jnp.prod(block_shape_per_dim) == self.block_size, \
                        "block_shape_per_dim should of the same size as blocks"

                # def dense_step(carry, index):
                #     res = carry
                #     # this only works for arrays, for list of arrays the blocks could be different sizes
                #     # as such the index_offset should be accounted for in carry
                #     scaled_index = index * block_shape_per_dim
                #     reshaped_block = self.blocks[tuple(index)].reshape(block_shape_per_dim)
                #     res = lax.dynamic_update_slice(res, reshaped_block, scaled_index)
                #     return res, None

                indices = jnp.indices(non_block_shape).transpose(*range(1, non_block_ndim + 1), 0)
                flattened_indices = indices.reshape(non_block_size, non_block_ndim)
                for index in flattened_indices:
                    # this only works for arrays, for list of arrays the blocks could be different sizes
                    # as such the index_offset should be accounted for in carry
                    scaled_index = index * block_shape_per_dim
                    reshaped_block = self.blocks[*index].reshape(block_shape_per_dim)
                    res = lax.dynamic_update_slice(res, reshaped_block, scaled_index)
                # res, _ = lax.scan(dense_step, res, flattened_indices)

            else:
                # this is no longer of type Sequence but rather MultiSparseDimensionBlocks
                # use get_ienumerated_blocks
                # first val_dim has to be sparse

                # pointer = (0,) * self.blocks.ndim
                for i, block in get_ienumerated_blocks(self.blocks):
                    pass
            return res


    def __add__(lhs, rhs):
        return _add(lhs, rhs)

    def __mul__(lhs, rhs):
        return _mul(lhs, rhs)

    def __matmul__(lhs, rhs):
        return _matmul(lhs, rhs)


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

    assert all(b1.ndim == b2.ndim for b1, b2 in zip(blocks, blocks[1:])), \
        "Block dimensions should all be equal"

    if not isinstance(blocks, Array):
        raise NotImplementedError("MultiSparseDimensionBlocks is WIP")

    out_dims = [
        x if not None
        else x._replace(val_axis=1)
        for x in out_dims
    ]
    primal_dims = [
        x if not None
        else x._replace(val_axis=0)
        for x in primal_dims
    ]

    val_dims = jnp.array(
        [x.val_dim for x in out_dims] + [x.val_dim for x in primal_dims if isinstance(x, DenseDimension)]
    )

    cur_val_dim = 0
    for val_dim in sorted(val_dims):
        if val_dim is not None:
            assert val_dim == cur_val_dim, "Value dimensions should be continuous"
            cur_val_dim += 1

    elementary_block_idx = cur_val_dim + 1

    block_shape = blocks.shape[elementary_block_idx:]

    out_shape = tuple(
        x.size if isinstance(x, DenseDimension)
        else x.size * block_shape[x.val_axis]
        for x in out_dims
    )
    primal_shape = tuple(
        x.size if isinstance(x, DenseDimension)
        else x.size * block_shape[x.val_axis]
        for x in out_dims
    )

    print(out_shape, type(out_shape))
    print(primal_shape, type(primal_shape))

    return BlockSparseTensor(
        out_dims,
        primal_dims,
        out_shape,
        primal_shape,
        blocks,
        elementary_block_idx,
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


def _dense_array(blocks, sparse_dims, shape):
    """
    Recreate a dense tensor from sparse blocks.

    Args:
        blocks: List of dense blocks (jnp.ndarray).
        sparse_dims: List of SparseDimensions.
        shape: Shape of the resulting dense tensor.

    Returns:
        A dense tensor recreated from sparse blocks.
    """
    dense_result = jnp.zeros(shape, dtype=jnp.float32)

    sparse_ids = jnp.array([dim.id for dim in sparse_dims])
    sparse_sizes = jnp.array([dim.size for dim in sparse_dims])
    sparse_val_dims = jnp.array([dim.val_dim if dim.val_dim is not None else -1
                                 for dim in sparse_dims])

    def get_indices(idx, block_shape):
        start_indices = jnp.where(
            sparse_val_dims == -1,
            0,
            idx[sparse_val_dims]
        )
        end_indices = jnp.where(
            sparse_val_dims == -1,
            sparse_sizes,
            start_indices + block_shape
        )
        return start_indices, end_indices

    def dense_step(carry, block):
        dense, idx = carry
        start, end = get_indices(idx, jnp.array(block.shape))

        slices = tuple(slice(start, end) for start, end in zip(start, end))
        print(block)
        print(slices)
        print(dense.at[*slices].get())
        dense = dense.at[slices].set(block)
        new_idx = jnp.where(sparse_val_dims == -1, idx, idx + sparse_sizes)

        return (dense, new_idx), None

    initial_idx = jnp.zeros(len(shape), dtype=int)

    (dense_result, _), _ = lax.scan(dense_step, (dense_result, initial_idx), blocks)

    return dense_result

# @partial(jit, static_argnames=('rhs', 'lhs'))
def _add(rhs, lhs):
    assert lhs.shape == rhs.shape, "Tensors must be of equal shape"
    if isinstance(rhs, BlockSparseTensor):
        if rhs.blocks is None:
            pass
        elif isinstance(lhs.blocks, Array) and isinstance(rhs.blocks, Array):
            if lhs.shape == rhs.shape and lhs.primal_dims == rhs.primal_dims and lhs.out_dims == rhs.out_dims:
                return BlockSparseTensor(lhs.primal_dims, lhs.out_dims, lhs.blocks + rhs.blocks, lhs.elementary_block_idx)
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
                return BlockSparseTensor(rhs.primal_dims, rhs.out_dims, rhs.blocks * lhs.blocks, rhs.elementary_block_idx)
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
            print("type rhs.out_shape", type(rhs.out_shape))
            print("type rhs.primal_shape", type(rhs.primal_shape))
            print("type lhs.out_shape", type(lhs.out_shape))
            print("type lhs.primal_shape", type(lhs.primal_shape))
            if True: #rhs.out_shape == lhs.primal_shape \
                    #and rhs.primal_dims == lhs.primal_dims \
                    #and rhs.out_dims == lhs.out_dims:
                non_block_shape = jnp.array(rhs.blocks.shape[:rhs.elementary_block_idx])
                non_block_size = jnp.prod(non_block_shape)

                def flatten_blocks(blocks):
                    extra_dims = [1] * max(2 - len(rhs.block_shape), 0)
                    return blocks.reshape((non_block_size, *rhs.block_shape, *extra_dims))

                def _calc(rb, lb):
                    block_mul = jax.vmap(lambda a, b: a @ b, in_axes=(0, 0))

                    return block_mul(
                        flatten_blocks(rb),
                        flatten_blocks(lb)
                    )

                new_blocks = _calc(rhs.blocks, lhs.blocks)
                print("mat mul res:", new_blocks)
                # rhs.elementary_block_idx may not be general
                blocksparse_tensor = BlockSparseTensor(rhs.primal_dims, lhs.out_dims, new_blocks, rhs.elementary_block_idx)
                print("obj", blocksparse_tensor)
                return blocksparse_tensor
        elif all(b1.shape == b2.shape for b1, b2 in zip(rhs.blocks, lhs.blocks)):
            pass
    elif isinstance(lhs, SparseTensor):
        pass
    elif isinstance(lhs, Array):  # TODO: Fix default check
        rhs.dense() @ lhs
    else:
        raise TypeError("Expected to add with type BlockSparseTensor, SparseTensor, or Array")