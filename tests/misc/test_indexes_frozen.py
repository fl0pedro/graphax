"""Pin the immutability and validation contract of the Index hierarchy.

Index/DenseIndex/DiagonalIndex flow as static aux_data through SparseTensor
pytrees (see `tensor.py:312`). Mutating an instance after the pytree has
been registered would silently desync the structural cache key from the
object's actual state — a classic JAX footgun. Switching the dataclasses
from `slots=True` to `frozen=True` makes that mutation impossible at the
language level. These tests pin that contract.

The second test covers a previously-duplicated validation: DiagonalIndex
used to re-implement the `size < 0` check inline instead of calling
`super().__post_init__()`, so any future Index-level validation would
silently fail to fire on DiagonalIndex instances. The fix is a `super()`
call — this test confirms the chain works end-to-end.
"""

import dataclasses

import pytest

from graphax.sparse.indexes import DenseIndex, Index, DiagonalIndex


def test_index_instances_are_frozen():
    """Mutating any Index/DenseIndex/DiagonalIndex field raises FrozenInstanceError."""
    idx = Index(0, 5, 0)
    didx = DenseIndex(1, 4, 1)
    sidx = DiagonalIndex(2, 3, 0, other_id=1)

    with pytest.raises(dataclasses.FrozenInstanceError):
        idx.id = 99
    with pytest.raises(dataclasses.FrozenInstanceError):
        didx.size = 99
    with pytest.raises(dataclasses.FrozenInstanceError):
        sidx.other_id = 99


def test_sparse_index_post_init_propagates_index_validation():
    """DiagonalIndex.__post_init__ must call super().__post_init__()."""
    # Index-level rule: size must be non-negative. If DiagonalIndex re-implements
    # the check inline, future Index-level tightening won't propagate; using
    # super() guarantees it does. Here we hit the existing rule via DiagonalIndex
    # to confirm the super() call is wired up.
    with pytest.raises(ValueError, match="size must be non-negative"):
        DiagonalIndex(0, -1, 0, other_id=1)

    # DiagonalIndex-specific rule (block_size > 0) still fires after super().
    with pytest.raises(ValueError, match="block_size must be positive"):
        DiagonalIndex(0, 5, 0, other_id=1, block_size=0)


def test_index_base_class_is_constructable():
    """Index is no longer ABC; bare construction must succeed."""
    # The previous `Index(ABC)` declaration had no abstract methods, so
    # `Index(...)` actually worked but pretended to be abstract. Dropping ABC
    # makes the class's role honest: it's a base for shared fields.
    idx = Index(0, 5, 0)
    assert idx.id == 0
    assert idx.size == 5
    assert idx.axis == 0
    assert idx.logical_size == 5
    assert idx.shape == (5,)
