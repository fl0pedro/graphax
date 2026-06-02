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


def test_deepcopy_with_compressed_index():
    """Smoke: deepcopy of a tensor carrying a compressed ``BandedIndex`` pair
    (band buffer in ``val``) doesn't raise and preserves the compressed dims.
    """
    from graphax.sparse.indexes import BandedIndex

    M, W, B = 2, 3, 3   # half-bandwidth = (W-1)/2 = 1 > 0
    data = jnp.zeros((M, W, B, B), dtype=jnp.float32)
    out = (BandedIndex(id=0, size=M, axis=0, other_id=1,
                       block_size=B, block_axis=1, band_width=W, offset=(),
                       primary=True, n_secondary=M, n_meta=1),)
    primal = (BandedIndex(id=1, size=M, axis=0, other_id=0,
                          block_size=B, block_axis=2, band_width=W, offset=(),
                          primary=False, n_secondary=M, n_meta=1),)
    st = SparseTensor(out, primal, data, check_consistency=False)

    cp = copy.deepcopy(st)
    assert isinstance(cp, SparseTensor)
    assert cp.out_dims[0].is_compressed
    assert jnp.array_equal(cp.val, st.val)
