from typing import Any, Callable, Sequence, Generator, TypeAlias, Iterable

# import numpy as np
from chex import Array

import jax
from jax import lax, jit
import jax.numpy as jnp
from .tensor import SparseTensor
import operator
from dataclasses import dataclass, KW_ONLY
import copy
from functools import reduce
import numpy as np

# TODO: make parent class, or inherit sparse tensor ??

@dataclass
class DenseDimension:
    id: int
    size: int
    val_dim: int | None
    _: KW_ONLY
    val_axis: int = None

# val_axes: These are two axis of the blocks that we would like to apply the diagonal on.
#   By default pick the first two (0, 1) as the axis). TODO: make negatives work like indeces
@dataclass
class SparseDimension:
    id: int
    size: int
    val_dim: int
    other_id: int
    _: KW_ONLY
    val_axis: int = None

Dimension = DenseDimension | SparseDimension

MultiSparseDimensionBlocks: TypeAlias = Sequence[Array] | Sequence['MultiSparseDimensionBlocks']

class BlockSparseTensor:
    out_dims: Any
    primal_dims: Any
    out_shape: tuple[int]
    primal_shape: tuple[int]
    shape: tuple[int]
    size: int
    ndim: int
    blocks: MultiSparseDimensionBlocks | Array | None
    pre_transforms: tuple[Callable]
    post_transforms: tuple[Callable]
    elementary_block_idx: int
    block_shape: tuple[int,...]
    block_size: int

    def __init__(self,
                 out_dims: Sequence[Dimension],
                 primal_dims: Sequence[Dimension],
                 blocks: MultiSparseDimensionBlocks | Array | None,
                 pre_transforms: Sequence[Callable] = None,
                 post_transforms: Sequence[Callable] = None) -> None:

        if pre_transforms is None:
            pre_transforms = []
        if post_transforms is None:
            post_transforms = []

        assert all(b1.ndim == b2.ndim for b1, b2 in zip(blocks, blocks[1:])), \
            "Block dimensions should all be equal"

        for x in primal_dims:
            if x.val_axis is None:
                x.val_axis = 0
        for x in out_dims:
            if x.val_axis is None:
                x.val_axis = 1

        assert all(b1.ndim == b2.ndim for b1, b2 in zip(blocks, blocks[1:])), \
            "Block dimensions should all be equal"

        self.out_dims = out_dims if isinstance(out_dims, tuple) else tuple(out_dims)
        self.primal_dims = primal_dims if isinstance(primal_dims, tuple) else tuple(primal_dims)


        val_dims = [x.val_dim for x in out_dims] + [x.val_dim for x in primal_dims if isinstance(x, DenseDimension)]

        if not isinstance(blocks, Array):
            raise NotImplementedError("MultiSparseDimensionBlocks is WIP")

        checked_max_val = reduce(lambda a,b: b if a+1==b else -1, sorted(val_dims))
        assert checked_max_val != -1 and checked_max_val <= blocks.ndim, \
            "Value dimensions should include all till the maximum (<= max block dimension)"
        self.block_shape = blocks.shape[checked_max_val+1:]
        self.block_size = reduce(operator.mul, self.block_shape)

        self.elementary_block_idx = checked_max_val+1
        self.primal_shape = [
            x.size if isinstance(x, DenseDimension)
            else x.size*self.block_shape[x.val_axis]
            for x in primal_dims
        ]
        self.out_shape = [
            x.size if isinstance(x, DenseDimension)
            else x.size*self.block_shape[x.val_axis]
            for x in out_dims
        ]

        # self.out_shape = get_block_shape(out_dims)
        # self.primal_shape = get_block_shape(primal_dims)

        self.shape = tuple(self.out_shape + self.primal_shape) # isn't quite right
        self.size = reduce(operator.mul, self.shape)
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

        return f"""SparseTensor(
    shape = ({str_out_shape} | {str_primal_shape}),
    out_dims = {multiline_out_dims},
    primal_dims = {multiline_primal_dims},
    blocks = {self.blocks},
    pre_transforms = {multiline_pre_transform},
    post_transforms = {multiline_post_transform}
)"""

    def dense(self):
        if all(isinstance(d, DenseDimension) for d in self.out_dims + self.primal_dims):
            assert isinstance(self.blocks, Array), \
                "Must be able to return a valid Array, not a Sequence"
            return self.blocks
        else:
            res = jnp.zeros(self.shape)
            if isinstance(self.blocks, Array):
                # THIS IS HARD CODED AND BAD
                offset = self.blocks.shape[self.elementary_block_idx]
                # jax.scan loop, rollout length, if elemnts = 100 ~ pick 10
                for i, elem in np.ndenumerate(self.blocks):
                    index = jnp.array(i[1:]) + jnp.array([i[0]*offset] * (len(i)-1))
                    res = res.at[*index].set(elem)
            else:
                # this is no longer of type Sequence but rather MultiSparseDimensionBlocks
                # use get_ienumerated_blocks
                # first val_dim has to be sparse

                # pointer = (0,) * self.blocks.ndim
                for i, block in get_ienumerated_blocks(self.blocks):
                    pass
            return res


    def __add__(self, other):
        assert self.shape == other.shape, "Tensors must be of equal shape"
        if isinstance(other, BlockSparseTensor):
            if other.blocks is None:
                pass
            elif isinstance(self.blocks, Array) and isinstance(other.blocks, Array):
                if self.shape == other.shape and self.primal_dims == other.primal_dims and self.out_dims == other.out_dims:
                    return BlockSparseTensor(self.primal_dims, self.out_dims, self.blocks + other.blocks)
            elif all(b1.shape == b2.shape for b1, b2 in zip(self.blocks, other.blocks)):
                pass
        elif isinstance(other, SparseTensor):
            pass
        elif isinstance(other, Array):
            self.dense() + other
        else:
            raise TypeError("Expected to add with type BlockSparseTensor, SparseTensor, or Array")

    def __mul__(self, other):
        assert self.shape == other.shape, "Tensors must be of equal shape"
        if isinstance(other, BlockSparseTensor):
            if other.blocks is None:
                pass
            elif isinstance(self.blocks, Array) and isinstance(other.blocks, Array):
                if self.shape == other.shape and self.primal_dims == other.primal_dims and self.out_dims == other.out_dims:
                    return BlockSparseTensor(self.primal_dims, self.out_dims, self.blocks * other.blocks)
            elif all(b1.shape == b2.shape for b1, b2 in zip(self.blocks, other.blocks)):
                pass
        elif isinstance(other, SparseTensor):
            pass
        elif isinstance(other, Array):
            self.dense() + other
        else:
            raise TypeError("Expected to add with type BlockSparseTensor, SparseTensor, or Array")

    def __matmul__(self, other):
        # TODO assert something
        if isinstance(other, BlockSparseTensor):
            if self.blocks is None:
                return copy.copy(other)
            elif other.blocks is None:
                return copy.copy(self)
            elif isinstance(self.blocks, Array) and isinstance(other.blocks, Array):
                if self.out_shape == other.primal_shape \
                        and self.primal_dims == other.primal_dims \
                        and self.out_dims == other.out_dims:

                    non_block_shape = self.blocks.shape[:self.elementary_block_idx]
                    non_block_size = reduce(operator.mul, non_block_shape)

                    block_mul = jit(jax.vmap(lambda a,b: a@b, in_axes=(0, 0)))

                    def flatten_blocks(blocks):
                        #
                        # if len(self.block_shape) <= 2:
                        #     extra_dims = [1] * self.block_shape
                        # else:
                        #     extra_dims = []
                        extra_dims = [1]*max(2-len(self.block_shape),0)
                        return blocks.reshape((non_block_size, *self.block_shape, *extra_dims))

                    new_blocks = block_mul(
                        flatten_blocks(self.blocks),
                        flatten_blocks(other.blocks)
                    )

                    return BlockSparseTensor(self.primal_dims, other.out_dims, new_blocks)
            elif all(b1.shape == b2.shape for b1, b2 in zip(self.blocks, other.blocks)):
                pass
        elif isinstance(other, SparseTensor):
            pass
        elif isinstance(other, Array): # TODO: Fix default check
            self.dense() @ other
        else:
            raise TypeError("Expected to add with type BlockSparseTensor, SparseTensor, or Array")

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
