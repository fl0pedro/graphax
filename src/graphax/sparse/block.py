from typing import Any, Callable, Sequence, Generator
from chex import Array

import jax
import jax.numpy as jnp

from tensor import SparseTensor
from base import SparseBase, SparseDimension, DenseDimension, Dimension


class BlockSparseTensor(SparseBase[Sequence[Array]]):
    def __add__(self, _tensor: Any):
        _add(self, _tensor)

    def __mul__(self, _tensor: Any):
        _mul(self, _tensor)

    def dense(self, iota: Array) -> Array:
        pass

    def copy(self):
        pass

    def assert_value_consistency(self):
        pass # not currently implemented

def _add(lhs: BlockSparseTensor, rhs: BlockSparseTensor) -> BlockSparseTensor:
    """
    Function that multiplies two `BlockSparseTensor` objects together. The function
    first performs a sequence of checks to guarantee the integrity of both
    `BlockSparseTensor` objects. It then proceeds to add both tensors, thereby
    possibly materializing certain sparse dimensions.

    Args:
        lhs (BlockSparseTensor): The left-hand side `SparseTensor` object.
        rhs (BlockSparseTensor): The right-hand side `SparseTensor` object.

    Returns:
        BlockSparseTensor: The resulting `BlockSparseTensor` object.
    """
    lhs.assert_sparse_object_consistency()
    rhs.assert_sparse_object_consistency()

    assert lhs.shape == rhs.shape, \
        f"{lhs.shape} and {rhs.shape} not compatible for addition!"

    res = _sparse_add(lhs, rhs)

    res.assert_sparse_object_consistency()
    return res