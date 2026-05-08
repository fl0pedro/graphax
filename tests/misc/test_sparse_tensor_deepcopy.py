"""``copy.deepcopy(SparseTensor)`` used to call ``self.copy(deep=True)`` even
though ``SparseTensor.copy`` doesn't accept a ``deep`` kwarg — every deepcopy
raised ``TypeError``. After the fix ``__deepcopy__`` calls plain ``self.copy()``
(which already returns a fresh SparseTensor with the same JAX-array
references — those are immutable from the user's perspective).
"""
import copy

import jax.numpy as jnp

from graphax.sparse.indexes import DenseIndex, SparseIndex
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


def test_deepcopy_with_compressed_val():
    """Smoke: deepcopy of a compressed-storage SparseTensor doesn't raise.

    Use ``BlockBanded(w>0)`` which from_compressed wraps via ``compressed_val=...``
    rather than the meta-block-diagonal val path.
    """
    from graphax.sparse.ops.block_storage import BlockBanded

    M, W, B = 2, 3, 3   # half-bandwidth = (W-1)/2 = 1 > 0
    data = jnp.zeros((M, W, B, B), dtype=jnp.float32)
    bb = BlockBanded(data=data, fill_value=jnp.array(0.0))

    st = SparseTensor.from_compressed(bb)
    cp = copy.deepcopy(st)
    assert isinstance(cp, SparseTensor)
    assert cp.compressed_val is not None
