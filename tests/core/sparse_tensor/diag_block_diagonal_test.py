"""Exact-pattern coverage for ``Diag`` block-diagonalisation (``apply_diag``).

Restores the EXACT-output assertions the legacy ``apply_dynamic_sparsity`` tests
carried (removed together with that function in the Phase-8 sparse refactor). The
live successor — a ``Diag(i, j, factor)`` micro-action applied via ``apply_diag``
— previously had only FINITENESS coverage (``test_sparsity_map_factors``), so a
wrong block structure would have gone unnoticed. Here we build a fully-dense
``(N, N)`` SparseTensor pair and check ``apply_diag`` materialises the exact
block-diagonal mask of the input ``val``.

Note: ``Diag.factor`` is the block COUNT (``DiagonalIndex.size``); ``block_size =
N // factor``. The legacy ``-1`` / ``0`` gcd-collapse sentinels are no longer
accepted (pass the explicit divisor), so only positive divisors are covered.
"""
import jax.numpy as jnp
import numpy as np

from graphax.sparse.indexes import DenseIndex
from graphax.sparse.micro_actions import Diag, apply_diag
from graphax.sparse.tensor import SparseTensor
from utils import assert_structure


def _dense_pair_st(n: int):
    """A SparseTensor whose single (out, primal) pair is a fully-dense (n, n)
    val with distinct nonzero entries (so block masking is observable)."""
    val = jnp.arange(n * n, dtype=jnp.float32).reshape(n, n) + 1.0
    out = (DenseIndex(id=0, size=n, axis=0),)
    primal = (DenseIndex(id=1, size=n, axis=1),)
    return SparseTensor(out, primal, val), np.asarray(val)


def test_diag_factor_one_is_noop():
    """factor == 1 -> single block == the whole matrix (apply_diag returns st)."""
    st, val = _dense_pair_st(4)
    out = apply_diag(st, Diag(0, 1, 1))
    np.testing.assert_array_equal(np.asarray(out.dense()), val)


def test_diag_factor_two_is_block_diagonal():
    """factor == 2 on a 4x4 -> two 2x2 diagonal blocks, off-block zeroed."""
    st, val = _dense_pair_st(4)
    out = apply_diag(st, Diag(0, 1, 2))
    # STRUCTURE: apply_diag must yield a block-diagonal PAIR, stored compactly (not dense 4x4)
    assert_structure(out, expect_block=True, msg='diag factor-2')
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[0:2, 0:2] = val[0:2, 0:2]
    expected[2:4, 2:4] = val[2:4, 2:4]
    np.testing.assert_array_equal(np.asarray(out.dense()), expected)


def test_diag_full_factor_is_pure_diagonal():
    """factor == N -> block_size 1 -> pure diagonal (only val[k, k] survive)."""
    st, val = _dense_pair_st(4)
    out = apply_diag(st, Diag(0, 1, 4))
    expected = np.diag(np.diag(val)).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(out.dense()), expected)


def test_diag_non_square_block_diagonal():
    """factor must divide BOTH sizes; a 6x6 with factor 3 -> three 2x2 blocks."""
    st, val = _dense_pair_st(6)
    out = apply_diag(st, Diag(0, 1, 3))
    expected = np.zeros((6, 6), dtype=np.float32)
    for b in range(3):
        s = slice(2 * b, 2 * b + 2)
        expected[s, s] = val[s, s]
    np.testing.assert_array_equal(np.asarray(out.dense()), expected)
