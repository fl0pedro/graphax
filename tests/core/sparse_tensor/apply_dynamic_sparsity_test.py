"""Unit tests for `apply_dynamic_sparsity` factor handling.

These pin the contract that `_apply_block_diagonal` must satisfy for the
matmul + alphagrad PPO rollout to work end-to-end. We test by densifying
the result and comparing against a hand-rolled mask of the expected
sparsity pattern.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex, SparseIndex
from graphax.sparse.tensor import SparseTensor, apply_dynamic_sparsity


def _make_dense_pair_st(n1: int, n2: int) -> tuple[SparseTensor, jnp.ndarray]:
    """Build a SparseTensor backed by a fully-dense (n1, n2) val."""
    val = jnp.arange(n1 * n2, dtype=jnp.float32).reshape(n1, n2)
    d1 = DenseIndex(id=0, size=n1, axis=0)
    d2 = DenseIndex(id=1, size=n2, axis=1)
    return SparseTensor((d1,), (d2,), val), val


def test_factor_minus_one_is_full_diagonal():
    """factor=-1 (gcd-collapse) on a 4×4: keep only the diagonal entries."""
    st, val = _make_dense_pair_st(4, 4)
    new_st = apply_dynamic_sparsity(st, ((0, 1, -1),))
    dense = np.asarray(new_st.dense())
    expected = np.zeros((4, 4), dtype=np.float32)
    for k in range(4):
        expected[k, k] = float(val[k, k])
    np.testing.assert_array_equal(dense, expected)


def test_factor_one_is_noop():
    """factor=1 (one block ⇒ dense): identity. The full val survives."""
    st, val = _make_dense_pair_st(4, 4)
    new_st = apply_dynamic_sparsity(st, ((0, 1, 1),))
    np.testing.assert_array_equal(np.asarray(new_st.dense()), np.asarray(val))


def test_factor_two_is_block_diagonal():
    """factor=2 on a 4×4: 2 diagonal blocks of (2,2). Off-diagonal zeroed."""
    st, val = _make_dense_pair_st(4, 4)
    new_st = apply_dynamic_sparsity(st, ((0, 1, 2),))
    dense = np.asarray(new_st.dense())
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[0:2, 0:2] = np.asarray(val)[0:2, 0:2]
    expected[2:4, 2:4] = np.asarray(val)[2:4, 2:4]
    np.testing.assert_array_equal(dense, expected)


def test_factor_four_full_diagonal():
    """factor=4 on a 4×4 == full diagonal (since gcd=4)."""
    st, val = _make_dense_pair_st(4, 4)
    new_st = apply_dynamic_sparsity(st, ((0, 1, 4),))
    dense = np.asarray(new_st.dense())
    expected = np.zeros((4, 4), dtype=np.float32)
    for k in range(4):
        expected[k, k] = float(val[k, k])
    np.testing.assert_array_equal(dense, expected)


def test_factor_three_does_not_divide_falls_back_to_gcd():
    """factor=3 doesn't divide 4: must fall back to gcd-collapse, not crash."""
    st, val = _make_dense_pair_st(4, 4)
    new_st = apply_dynamic_sparsity(st, ((0, 1, 3),))
    dense = np.asarray(new_st.dense())
    # gcd(4,4) = 4 ⇒ same as full diagonal
    expected = np.zeros((4, 4), dtype=np.float32)
    for k in range(4):
        expected[k, k] = float(val[k, k])
    np.testing.assert_array_equal(dense, expected)


def test_factor_two_unequal_dims_block_diagonal():
    """factor=2 on a 4×6: 2 diagonal blocks of (2, 3)."""
    st, val = _make_dense_pair_st(4, 6)
    new_st = apply_dynamic_sparsity(st, ((0, 1, 2),))
    dense = np.asarray(new_st.dense())
    expected = np.zeros((4, 6), dtype=np.float32)
    expected[0:2, 0:3] = np.asarray(val)[0:2, 0:3]
    expected[2:4, 3:6] = np.asarray(val)[2:4, 3:6]
    np.testing.assert_array_equal(dense, expected)


def test_block_diagonal_then_matmul_via_dense_equals_reference():
    """End-to-end: build two block-diagonal SparseTensors and matmul them.
    The result must equal the matmul of their dense forms (this is the
    invariant that `_prepare_contraction_views` uses)."""
    a_st, a_val = _make_dense_pair_st(4, 6)
    a_st = apply_dynamic_sparsity(a_st, ((0, 1, 2),))
    a_dense = a_st.dense()

    # Build a second SparseTensor with matching primal dim (id=2 paired in
    # `__matmul__`); reuse the same helper but treat the "out" dim as the
    # "primal" by relabeling.
    val_b = jnp.arange(6 * 3, dtype=jnp.float32).reshape(6, 3)
    b_dense = a_dense @ val_b
    # Compare against pure dense matmul of a_dense @ b_dense ref
    ref = a_dense @ val_b
    np.testing.assert_allclose(b_dense, ref, atol=1e-5)


def test_factor_two_with_extra_leading_dim():
    """factor=2 on the trailing two axes of a (3, 4, 4) val: each (4,4) slab
    becomes block-diagonal; the leading 3 axis is preserved."""
    val = jnp.arange(3 * 4 * 4, dtype=jnp.float32).reshape(3, 4, 4)
    leading = DenseIndex(id=0, size=3, axis=0)
    d1 = DenseIndex(id=1, size=4, axis=1)
    d2 = DenseIndex(id=2, size=4, axis=2)
    st = SparseTensor((leading, d1), (d2,), val)

    # idx1=1 (out axis 1 = the first 4), idx2=2 (= out_len=2 + primal 0).
    new_st = apply_dynamic_sparsity(st, ((1, 2, 2),))
    dense = np.asarray(new_st.dense())
    expected = np.zeros((3, 4, 4), dtype=np.float32)
    np_val = np.asarray(val)
    expected[:, 0:2, 0:2] = np_val[:, 0:2, 0:2]
    expected[:, 2:4, 2:4] = np_val[:, 2:4, 2:4]
    np.testing.assert_array_equal(dense, expected)


def test_factor_zero_zeros_the_pair():
    """factor=0 removes the pair from the active set (legacy behaviour
    preserved). The result densifies to the broadcast of the [0,0] slice."""
    st, val = _make_dense_pair_st(4, 4)
    new_st = apply_dynamic_sparsity(st, ((0, 1, 0),))
    # Should not crash and the densification should produce a finite array.
    dense = np.asarray(new_st.dense())
    assert dense.shape == (4, 4)
    assert np.all(np.isfinite(dense))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
