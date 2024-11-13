from typing import Any, Callable, Sequence, Generator
from chex import Array

import jax
import jax.numpy as jnp

from tensor import SparseTensor
from base import SparseBase, SparseDimension, DenseDimension, Dimension


class BlockSparseTensor(SparseBase):
    blocks: Sequence[Array]

    def __init__(self, blocks: Sequence[Array], **kwargs):
        super().__init__(**kwargs)
        self.blocks = blocks
    ...
    

