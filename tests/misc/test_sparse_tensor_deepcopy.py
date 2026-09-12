"""``copy.deepcopy(SparseTensor)`` used to call ``self.copy(deep=True)`` even
though ``SparseTensor.copy`` doesn't accept a ``deep`` kwarg — every deepcopy
raised ``TypeError``. After the fix ``__deepcopy__`` calls plain ``self.copy()``
(which already returns a fresh SparseTensor with the same JAX-array
references — those are immutable from the user's perspective).
"""
import copy

import jax.numpy as jnp

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor


def test_deepcopy_with_val():
    val = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    st = SparseTensor(
        (DenseIndex(0, 2, axis=0),),
        (DenseIndex(1, 3, axis=1),),
        val,
    )
    cp = copy.deepcopy(st)
    assert isinstance(cp, SparseTensor)
    assert cp.shape == st.shape
    assert jnp.array_equal(cp.val, st.val)


def test_deepcopy_with_val_none():
    st = SparseTensor(
        (DenseIndex(0, 4, axis=None),),
        (DenseIndex(1, 4, axis=None),),
        val=None,
    )
    cp = copy.deepcopy(st)
    assert isinstance(cp, SparseTensor)
    assert cp.val is None
    assert cp.shape == st.shape


def test_deepcopy_with_block_diagonal_index():
    """Smoke: deepcopy of a tensor carrying a meta-block-diagonal pair does not
    raise and preserves the pair. This used to exercise a ``BandedIndex`` pair;
    that class is gone (ruling 2026-09-07), so the surviving structured form is
    the DiagonalIndex pair.
    """
    M, B = 2, 3
    data = jnp.zeros((M, B, B), dtype=jnp.float32)
    out = (DiagonalIndex(0, M, 0, 1, B, 1),)
    primal = (DiagonalIndex(1, M, 0, 0, B, 2),)
    st = SparseTensor(out, primal, data)

    cp = copy.deepcopy(st)
    assert isinstance(cp, SparseTensor)
    assert cp.out_dims[0].is_sparse
    assert jnp.array_equal(cp.val, st.val)
