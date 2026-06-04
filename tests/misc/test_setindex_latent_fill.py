"""Latent-case regressions for the compressed-Index densify boundary.

L2 — ``BandedIndex.reduces_to_diagonal`` only self-determines squareness on the
PRIMARY side: on a col-primary index ``size // n_meta`` is the secondary meta
count, so the ``!= n_secondary`` test is vacuous and would mislabel a non-square
band as diagonal. A non-primary index must conservatively report ``False``.

L3 — when ``_densify_compressed_dims`` materializes a ``SetIndex`` pair, the
result's implicit cells hold the op-combined fill ``op(fill_lhs, fill_rhs)``
(what ``densify_axis`` / ``to_meta_blocks`` stitch into the data), so the
re-wrapped tensor's ``fill_value`` must be that combined value — not the raw
per-side fill. Reachable only with a non-zero fill, which the elementwise
emitter never produces (it bails to the dense path), so this is a latent guard.
"""
import jax.numpy as jnp
import numpy as np

from graphax.sparse.tensor import SparseTensor
from graphax.sparse.indexes import BandedIndex
from graphax.sparse.ops.elementwise import elementwise
from graphax.sparse.ops.utils import _densify_compressed_dims
from graphax.sparse.indexes import DiagonalIndex
import jax.random as jr


def _n(shape, key=0):
    return jr.normal(jr.PRNGKey(key), shape).astype(jnp.float32)


# --- L2 ---------------------------------------------------------------------

def test_col_primary_band_not_vacuously_diagonal():
    # Col-primary W=1 identity band: size//n_meta == n_secondary, so the old
    # squareness test (n_secondary != size//n_meta) was vacuously satisfied →
    # wrongly reported True. The col side can't see the row count, so: False.
    col = BandedIndex(id=1, size=3, axis=0, other_id=0, block_size=2,
                      block_axis=1, band_width=1, offset=(), primary=False,
                      n_secondary=3, n_meta=1)
    assert col.reduces_to_diagonal() is False


def test_primary_band_squareness_still_decided():
    sq = BandedIndex(id=0, size=3, axis=0, other_id=1, block_size=2,
                     block_axis=1, band_width=1, offset=(), primary=True,
                     n_secondary=3, n_meta=1)
    nonsq = BandedIndex(id=0, size=4, axis=0, other_id=1, block_size=2,
                        block_axis=1, band_width=1, offset=(), primary=True,
                        n_secondary=3, n_meta=1)
    assert sq.reduces_to_diagonal() is True
    assert nonsq.reduces_to_diagonal() is False


# --- L3 ---------------------------------------------------------------------

def _nonzero_fill_setindex(semantic, op, fill):
    """A real misaligned 5/11 SetIndex pair, re-emitted with a non-zero fill."""
    a = SparseTensor((DiagonalIndex(0, 11, 0, 1, 5, 1),),
                     (DiagonalIndex(1, 11, 0, 0, 5, 2),), _n((11, 5, 5), 1))
    b = SparseTensor((DiagonalIndex(0, 5, 0, 1, 11, 1),),
                     (DiagonalIndex(1, 5, 0, 0, 11, 2),), _n((5, 11, 11), 2))
    t = elementwise(a, b, op, is_intersection=(semantic == "intersection"))
    # Rebuild with a non-zero fill (the emitter only ever makes zero-fill ones).
    return SparseTensor(t.out_dims, t.primal_dims, t.val,
                        fill_value=jnp.array(fill, dtype=jnp.float32),
                        check_consistency=False)


def test_setindex_result_fill_is_op_combined_union():
    # union → op = add; per-side fill 0.5 each → combined implicit fill 1.0.
    t = _nonzero_fill_setindex("union", jnp.add, 0.5)
    compact = _densify_compressed_dims(t, compact=True)
    dense = _densify_compressed_dims(t, compact=False)
    np.testing.assert_allclose(float(compact.fill_value), 1.0, atol=1e-6)
    np.testing.assert_allclose(float(dense.fill_value), 1.0, atol=1e-6)


def test_setindex_result_fill_is_op_combined_intersection():
    # intersection → op = multiply; per-side fill 0.5 each → combined 0.25.
    t = _nonzero_fill_setindex("intersection", jnp.multiply, 0.5)
    compact = _densify_compressed_dims(t, compact=True)
    dense = _densify_compressed_dims(t, compact=False)
    np.testing.assert_allclose(float(compact.fill_value), 0.25, atol=1e-6)
    np.testing.assert_allclose(float(dense.fill_value), 0.25, atol=1e-6)


def test_setindex_compact_and_dense_fill_agree():
    # The two materialization paths must agree on the result fill.
    t = _nonzero_fill_setindex("union", jnp.add, -0.3)
    compact = _densify_compressed_dims(t, compact=True)
    dense = _densify_compressed_dims(t, compact=False)
    np.testing.assert_allclose(float(compact.fill_value),
                               float(dense.fill_value), atol=1e-6)
