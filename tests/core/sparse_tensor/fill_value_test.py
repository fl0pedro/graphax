"""Correctness tests for non-zero ``fill_value`` semantics.

The tiled block-sparse algorithms in ``ops/matmul.py`` and ``ops/elementwise.py``
both have to be aware of ``fill_value`` because structurally-implicit positions
contribute to the result whenever the fill is not zero:

  * For elementwise ``op(lhs, rhs)``, an unstored position on either side
    contributes ``op(fill_lhs, val_rhs)`` or ``op(val_lhs, fill_rhs)`` (or
    ``op(fill_lhs, fill_rhs)`` if both are unstored).
  * For matmul ``A @ B``, an unstored position contributes its fill value to the
    contraction sum exactly the same way a stored zero would have *failed* to.

These tests pin both behaviors down by comparing each op's output against a
reference computed on the densified operands.
"""
import unittest

import jax
import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.tensor import SparseTensor
from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.ops.matmul import matmul
from graphax.sparse.ops.elementwise import elementwise


def _n(shape, key_idx, dtype=jnp.float32):
    return jr.normal(jr.PRNGKey(key_idx), shape).astype(dtype)


class TestNonZeroFillElementwise(unittest.TestCase):
    """Elementwise must propagate ``op(fill_lhs, fill_rhs)`` to unstored positions."""

    def _check(self, lhs, rhs, op, atol=1e-5):
        for use_jit in (False, True):
            with self.subTest(jit=use_jit):
                fn = jax.jit(lambda a, b: elementwise(a, b, op)) if use_jit else (
                    lambda a, b: elementwise(a, b, op)
                )
                got = fn(lhs, rhs)
                expected = op(lhs.dense(), rhs.dense())
                self.assertTrue(
                    jnp.allclose(got.dense(), expected, atol=atol),
                    f"elementwise({op.__name__}) mismatch: max diff "
                    f"{float(jnp.max(jnp.abs(got.dense() - expected)))}",
                )

    def test_dense_dense_nonzero_fill_add(self):
        # Both fully-dense tensors with mismatched non-zero fills.
        lhs = SparseTensor(
            (DenseIndex(0, 4, 0),), (DenseIndex(1, 4, 1),),
            _n((4, 4), 1), fill_value=jnp.array(2.0, dtype=jnp.float32),
        )
        rhs = SparseTensor(
            (DenseIndex(0, 4, 0),), (DenseIndex(1, 4, 1),),
            _n((4, 4), 2), fill_value=jnp.array(3.0, dtype=jnp.float32),
        )
        self._check(lhs, rhs, jax.lax.add)

    def test_sparse_pair_nonzero_fill_mul(self):
        lhs = SparseTensor(
            (DiagonalIndex(0, 4, axis=0, other_id=1),),
            (DiagonalIndex(1, 4, axis=0, other_id=0),),
            _n((4,), 1), fill_value=jnp.array(0.5, dtype=jnp.float32),
        )
        rhs = SparseTensor(
            (DiagonalIndex(0, 4, axis=0, other_id=1),),
            (DiagonalIndex(1, 4, axis=0, other_id=0),),
            _n((4,), 2), fill_value=jnp.array(1.5, dtype=jnp.float32),
        )
        self._check(lhs, rhs, jax.lax.mul)

    def test_blocked_sparse_nonzero_fill_max(self):
        # Sparse pair with block_size on both sides.
        N, B = 3, 2
        lhs = SparseTensor(
            (DiagonalIndex(0, N, axis=0, other_id=1, block_size=B, block_axis=1),),
            (DiagonalIndex(1, N, axis=0, other_id=0, block_size=B, block_axis=2),),
            _n((N, B, B), 1), fill_value=jnp.array(0.25, dtype=jnp.float32),
        )
        rhs = SparseTensor(
            (DiagonalIndex(0, N, axis=0, other_id=1, block_size=B, block_axis=1),),
            (DiagonalIndex(1, N, axis=0, other_id=0, block_size=B, block_axis=2),),
            _n((N, B, B), 2), fill_value=jnp.array(-0.5, dtype=jnp.float32),
        )
        self._check(lhs, rhs, jax.lax.max)


class TestNonZeroFillMatmul(unittest.TestCase):
    """Matmul must include ``fill_value`` contributions in the contraction sum."""

    def _check(self, lhs, rhs, atol=1e-4):
        for use_jit in (False, True):
            with self.subTest(jit=use_jit):
                fn = jax.jit(matmul) if use_jit else matmul
                got = fn(lhs, rhs)
                expected = jnp.matmul(lhs.dense(), rhs.dense())
                self.assertTrue(
                    jnp.allclose(got.dense(), expected, atol=atol),
                    f"matmul mismatch (jit={use_jit}): max diff "
                    f"{float(jnp.max(jnp.abs(got.dense() - expected)))}",
                )

    def test_dense_dense_nonzero_fill(self):
        # Even fully-dense SparseTensors carry fill_value semantics in principle.
        lhs = SparseTensor(
            (DenseIndex(0, 4, 0),), (DenseIndex(1, 4, 1),),
            _n((4, 4), 1), fill_value=jnp.array(0.5, dtype=jnp.float32),
        )
        rhs = SparseTensor(
            (DenseIndex(0, 4, 0),), (DenseIndex(1, 4, 1),),
            _n((4, 4), 2), fill_value=jnp.array(0.0, dtype=jnp.float32),
        )
        self._check(lhs, rhs)

    def test_sparse_lhs_nonzero_fill_dense_rhs(self):
        # Block-diagonal LHS with non-zero fill, dense RHS.
        N, B = 3, 2
        lhs = SparseTensor(
            (DiagonalIndex(0, N, axis=0, other_id=1, block_size=B, block_axis=1),),
            (DiagonalIndex(1, N, axis=0, other_id=0, block_size=B, block_axis=2),),
            _n((N, B, B), 1), fill_value=jnp.array(0.5, dtype=jnp.float32),
        )
        rhs = SparseTensor(
            (DenseIndex(0, N * B, 0),), (DenseIndex(1, N * B, 1),),
            _n((N * B, N * B), 2),
        )
        self._check(lhs, rhs)

    def test_both_sides_nonzero_fill(self):
        N = 4
        lhs = SparseTensor(
            (DiagonalIndex(0, N, axis=0, other_id=1),),
            (DiagonalIndex(1, N, axis=0, other_id=0),),
            _n((N,), 1), fill_value=jnp.array(0.7, dtype=jnp.float32),
        )
        rhs = SparseTensor(
            (DenseIndex(0, N, 0),), (DenseIndex(1, N, 1),),
            _n((N, N), 2), fill_value=jnp.array(-0.3, dtype=jnp.float32),
        )
        self._check(lhs, rhs)


if __name__ == "__main__":
    unittest.main()
