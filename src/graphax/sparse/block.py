from typing import Any, Callable, Sequence, Generator
from chex import Array

import jax
import jax.numpy as jnp

from tensor import SparseTensor
from base import SparseBase, SparseDimension, DenseDimension, Dimension


class BlockSparseTensor(SparseBase[Sequence[Array]]):
    ...
