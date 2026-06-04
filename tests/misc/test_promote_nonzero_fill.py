"""B3 regression: ``_promote_to_unified`` must scatter each operand's blocks
onto the LCM grid with the OFF-diagonal positions holding that operand's
post-scaled fill — not a literal 0.

When two block-diagonal operands with mismatched block sizes are combined,
the smaller-block side is promoted to the LCM block grid: its sub-blocks land
on the grid diagonal and the off-diagonal sub-blocks are positions where the
operand has no explicit block, i.e. its implicit fill. Writing a literal 0
there was correct only for zero-fill operands; for a non-zero fill it dropped
the fill contribution, so ``(a + b).dense()`` disagreed with
``a.dense() + b.dense()``. The non-zero-fill misaligned case routes through
the general promote path (the SetIndex / divisor-remainder emitters bail when
either fill is non-zero), so this path is reachable.
"""
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from graphax.sparse.tensor import SparseTensor
from graphax.sparse.indexes import DiagonalIndex


def _blockdiag(M, B, key, fill):
    """``M`` blocks of ``B×B`` on the diagonal of an ``(M*B)`` square, with
    a non-zero implicit fill off the meta-diagonal."""
    val = jr.normal(jr.PRNGKey(key), (M, B, B)).astype(jnp.float32)
    return SparseTensor(
        (DiagonalIndex(0, M, 0, 1, B, 1),),
        (DiagonalIndex(1, M, 0, 0, B, 2),),
        val,
        fill_value=jnp.array(fill, dtype=jnp.float32),
    )


def test_promote_nonzero_fill_add_matches_dense():
    # B=1 (6 unit blocks) + B=2 (3 2x2 blocks): LCM block = 2, so the B=1 side
    # is promoted exp=2 onto the 2x2 grid — exercises the off-diagonal fill.
    a = _blockdiag(6, 1, 1, fill=0.5)
    b = _blockdiag(3, 2, 2, fill=-0.3)
    got = np.asarray((a + b).dense())
    ref = np.asarray(a.dense()) + np.asarray(b.dense())
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


def test_promote_nonzero_fill_multiply_matches_dense():
    a = _blockdiag(6, 1, 3, fill=0.5)
    b = _blockdiag(3, 2, 4, fill=-0.3)
    got = np.asarray((a * b).dense())
    ref = np.asarray(a.dense()) * np.asarray(b.dense())
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)
