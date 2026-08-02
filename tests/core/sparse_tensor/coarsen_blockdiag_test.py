"""Meta coarsening — the EXACT dual of `_subdivide_coupled_blockdiag`.

`_coarsen_coupled_blockdiag` re-factors a coupled block-diagonal pair from
meta ``N`` to a coarser meta ``G`` (``G | N``): each new ``G``-block holds its
``N/G`` finer blocks on the sub-diagonal, the off-sub-diagonal positions
become explicit stored zeros. This is what makes two commensurable factorings
of the same logical axis contractible without densifying either operand (the
``GRAPHAX_RECONCILE_BLOCKDIAG_METAS`` matmul-entry reconciliation, default
OFF — see the measured trade-off note at the matmul entry).

These tests assert (1) the dense form is BIT-identical across coarsening
(incl. ``val is None`` and the meta→1 edge), and (2) the reconciled
mismatched-meta B@B contraction matches the dense oracle.
"""
import unittest

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.tensor import SparseTensor, _coarsen_coupled_blockdiag
from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.ops.matmul import _reconcile_blockdiag_metas, matmul


def _n(shape, key=0):
    return jr.normal(jr.PRNGKey(key), shape).astype(jnp.float32)


def _diag16(extra=10):
    """(16, e) x (16, e): pure diagonal pair (meta 16, block 1) + dense dims."""
    od = (DiagonalIndex(0, 16, 0, 2, None, None), DenseIndex(1, extra, 1))
    pd = (DiagonalIndex(2, 16, 0, 0, None, None), DenseIndex(3, extra, 2))
    return SparseTensor(od, pd, _n((16, extra, extra), 1),
                        check_consistency=False)


def _block2(extra=784):
    """(16,10) x (256, e): meta-2 pair, blocks 8/128 (a Diag(factor) edge)."""
    od = (DiagonalIndex(0, 2, 0, 2, 8, 1), DenseIndex(1, 10, 2))
    pd = (DiagonalIndex(2, 2, 0, 0, 128, 3), DenseIndex(3, extra, 4))
    return SparseTensor(od, pd, _n((2, 8, 10, 128, extra), 2),
                        check_consistency=False)


class TestCoarsenExactness(unittest.TestCase):
    def _coarsen(self, t, g):
        return _coarsen_coupled_blockdiag(
            t, True, 0, t.out_dims[0], False, 0, t.primal_dims[0], g)

    def test_meta16_to_2_dense_identical(self):
        t = _diag16()
        before = np.asarray(t.dense())
        after = np.asarray(self._coarsen(t, 2).dense())
        self.assertEqual(np.abs(before - after).max(), 0.0)

    def test_meta16_to_1_dense_identical(self):
        t = _diag16()
        before = np.asarray(t.dense())
        after = np.asarray(self._coarsen(t, 1).dense())
        self.assertEqual(np.abs(before - after).max(), 0.0)

    def test_val_none_uniform_ones(self):
        od = (DiagonalIndex(0, 16, None, 2, None, None),
              DenseIndex(1, 10, None))
        pd = (DiagonalIndex(2, 16, None, 0, None, None),
              DenseIndex(3, 10, None))
        t = SparseTensor(od, pd, None, check_consistency=False)
        before = np.asarray(t.dense())
        after = np.asarray(self._coarsen(t, 4).dense())
        self.assertEqual(np.abs(before - after).max(), 0.0)

    def test_noop_when_meta_equal(self):
        t = _diag16()
        self.assertIs(self._coarsen(t, 16), t)

    def test_indivisible_meta_raises(self):
        t = _diag16()
        with self.assertRaises(ValueError):
            self._coarsen(t, 3)


class TestMismatchedMetaContraction(unittest.TestCase):
    def test_reconciled_matches_dense_oracle(self):
        lhs, rhs = _diag16(10), _block2(784)
        Ld = np.asarray(lhs.dense()).reshape(16, 10, 16, 10)
        Rd = np.asarray(rhs.dense()).reshape(16, 10, 256, 784)
        oracle = np.einsum("abkl,klcd->abcd", Ld, Rd)
        l2, r2 = _reconcile_blockdiag_metas(lhs, rhs)
        # reconcile coarsened the pure diagonal to the shared meta 2
        self.assertEqual(l2.out_dims[0].size, 2)
        got = np.asarray(matmul(l2, r2).dense()).reshape(oracle.shape)
        tol = 1e-4 * max(1.0, float(np.abs(oracle).max()))
        self.assertLess(float(np.abs(oracle - got).max()), tol)


if __name__ == "__main__":
    unittest.main()
