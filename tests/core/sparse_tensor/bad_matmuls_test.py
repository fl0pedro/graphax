import unittest
import jax.numpy as jnp
import jax.random as jr
from graphax.sparse.tensor import SparseTensor, DenseIndex, SparseIndex
from graphax.sparse.ops.matmul import matmul
from graphax.sparse.ops.elementwise import elementwise


class TestBroadcastingFailures(unittest.TestCase):
    def setUp(self):
        self.key = jr.PRNGKey(42)

    def _n(self, shape, key_idx=0):
        return jr.normal(jr.PRNGKey(key_idx), shape)

    def test_elementwise_dense_shape_mismatch(self):
        """Fails if matched dense dimensions have different sizes."""
        a = SparseTensor((DenseIndex(0, 4, 0),), (), self._n((4,), 1))
        b = SparseTensor((DenseIndex(0, 3, 0),), (), self._n((3,), 2))
        with self.assertRaises((ValueError, AssertionError, TypeError)):
            res = a + b

    def test_elementwise_sparse_shape_mismatch(self):
        """Fails if matched sparse dimensions have different outer sizes."""
        a = SparseTensor(
            (SparseIndex(0, 4, axis=0, other_id=1),),
            (SparseIndex(1, 4, axis=0, other_id=0),),
            self._n((4,), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, 5, axis=0, other_id=1),),
            (SparseIndex(1, 5, axis=0, other_id=0),),
            self._n((5,), 2),
        )
        with self.assertRaises((ValueError, AssertionError, TypeError)):
            res = a * b

    def test_elementwise_sparse_logical_size_mismatch(self):
        """Fails if matched sparse dimensions compute to different total logical sizes."""
        a = SparseTensor(
            (
                SparseIndex(
                    0, 4, axis=0, other_id=1, block_size=2, block_axis=1
                ),
            ),
            (
                SparseIndex(
                    1, 4, axis=0, other_id=0, block_size=2, block_axis=2
                ),
            ),
            self._n((4, 2, 2), 1),
        )  # Logical Size: 8
        b = SparseTensor(
            (
                SparseIndex(
                    0, 4, axis=0, other_id=1, block_size=3, block_axis=1
                ),
            ),
            (
                SparseIndex(
                    1, 4, axis=0, other_id=0, block_size=3, block_axis=2
                ),
            ),
            self._n((4, 3, 3), 2),
        )  # Logical Size: 12
        with self.assertRaises((ValueError, AssertionError, TypeError)):
            res = elementwise(a, b, jnp.maximum)

    def test_matmul_batch_dense_mismatch(self):
        """Fails if implicit batch dimensions (same ID) have different sizes."""
        a = SparseTensor(
            (DenseIndex(0, 6, 0),), (DenseIndex(1, 5, 1),), self._n((6, 5), 1)
        )
        b = SparseTensor(
            (DenseIndex(0, 2, 0), DenseIndex(1, 5, 1)),
            (DenseIndex(2, 4, 2),),
            self._n((2, 5, 4), 2),
        )
        with self.assertRaises((ValueError, AssertionError, TypeError)):
            res = a @ b

    def test_matmul_batch_sparse_mismatch(self):
        """Fails if implicit sparse batch dimensions have different sizes."""
        a = SparseTensor(
            (SparseIndex(0, 4, axis=0, other_id=1), DenseIndex(2, 5, 1)),
            (SparseIndex(1, 4, axis=0, other_id=0),),
            self._n((4, 5), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, 3, axis=0, other_id=1), DenseIndex(2, 5, 1)),
            (SparseIndex(1, 3, axis=0, other_id=0),),
            self._n((3, 5), 2),
        )
        with self.assertRaises((ValueError, AssertionError, TypeError)):
            res = a @ b

    def test_matmul_contraction_dense_dense_mismatch(self):
        """Fails if contraction dimensions have different dense sizes."""
        a = SparseTensor((), (DenseIndex(0, 4, 0),), self._n((4,), 1))
        b = SparseTensor((DenseIndex(0, 5, 0),), (), self._n((5,), 2))
        with self.assertRaises((ValueError, AssertionError, TypeError)):
            res = a @ b

    def test_matmul_contraction_sparse_dense_mismatch(self):
        """Fails if sparse LHS contracts against differently sized dense RHS."""
        a = SparseTensor(
            (SparseIndex(0, 4, axis=0, other_id=1),),
            (SparseIndex(1, 4, axis=0, other_id=0),),
            self._n((4,), 1),
        )  # Logical contraction size: 4
        b = SparseTensor(
            (DenseIndex(0, 5, 0),), (), self._n((5,), 2)
        )  # Logical contraction size: 5
        with self.assertRaises((ValueError, AssertionError, TypeError)):
            res = a @ b

    def test_matmul_contraction_dense_sparse_mismatch(self):
        """Fails if dense LHS contracts against differently sized sparse RHS."""
        a = SparseTensor(
            (), (DenseIndex(0, 4, 0),), self._n((4,), 1)
        )  # Logical contraction size: 4
        b = SparseTensor(
            (SparseIndex(0, 5, axis=0, other_id=1),),
            (SparseIndex(1, 5, axis=0, other_id=0),),
            self._n((5,), 2),
        )  # Logical contraction size: 5
        with self.assertRaises((ValueError, AssertionError, TypeError)):
            res = a @ b

    def test_matmul_contraction_sparse_sparse_outer_mismatch(self):
        """Fails if LHS/RHS sparse contractions have different outer sizes."""
        a = SparseTensor(
            (SparseIndex(0, 4, axis=0, other_id=1),),
            (SparseIndex(1, 4, axis=0, other_id=0),),
            self._n((4,), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, 5, axis=0, other_id=1),),
            (SparseIndex(1, 5, axis=0, other_id=0),),
            self._n((5,), 2),
        )
        with self.assertRaises((ValueError, AssertionError, TypeError)):
            res = a @ b

    def test_matmul_contraction_sparse_sparse_block_mismatch(self):
        """Fails if LHS/RHS sparse contractions compute to different logical sizes."""
        a = SparseTensor(
            (
                SparseIndex(
                    0, 4, axis=0, other_id=1, block_size=2, block_axis=1
                ),
            ),
            (
                SparseIndex(
                    1, 4, axis=0, other_id=0, block_size=2, block_axis=2
                ),
            ),
            self._n((4, 2, 2), 1),
        )  # Logical contraction size: 8
        b = SparseTensor(
            (
                SparseIndex(
                    0, 4, axis=0, other_id=1, block_size=3, block_axis=1
                ),
            ),
            (
                SparseIndex(
                    1, 4, axis=0, other_id=0, block_size=3, block_axis=2
                ),
            ),
            self._n((4, 3, 3), 2),
        )  # Logical contraction size: 12
        with self.assertRaises((ValueError, AssertionError, TypeError)):
            res = a @ b


if __name__ == "__main__":
    unittest.main()
