import unittest
import jax.numpy as jnp
from graphax.sparse.tensor import _sort_val, DenseDimension, SparseDimension, SparseTensor


class TestSortVal(unittest.TestCase):
    def test_sort_val_flipped_dense(self):
        """
        Replicates the failure condition in test_simple_None_dense where
        physical axes mapped inversely to logical out/primal definitions.
        """
        val = jnp.arange(6).reshape(2, 3)

        out_dims = [DenseDimension(id=0, size=3, val_dim=1)]
        primal_dims = [DenseDimension(id=1, size=2, val_dim=0)]

        new_out, new_primal, new_val = _sort_val(out_dims, primal_dims, val)

        self.assertEqual(new_val.shape, (3, 2))
        self.assertEqual(new_out[0].val_dim, 0)
        self.assertEqual(new_primal[0].val_dim, 1)
        self.assertTrue(jnp.array_equal(new_val, val.T))

    def test_sort_val_mixed(self):
        """
        Validates the strict permutation hierarchy:
        [sparse out -> sparse primal -> dense/block out -> dense/block primal].
        """
        val = jnp.arange(2 * 5 * 3 * 4).reshape(2, 5, 3, 4)

        # Original physical layout:
        # 0: dense out, 1: sparse primal, 2: block out, 3: sparse out
        out_dims = [
            SparseDimension(
                id=0, size=4, val_dim=3, other_id=2, block_size=3, block_val_dim=2
            ),
            DenseDimension(id=1, size=2, val_dim=0),
        ]
        primal_dims = [SparseDimension(id=2, size=5, val_dim=1, other_id=0)]

        new_out, new_primal, new_val = _sort_val(out_dims, primal_dims, val)

        # Expected Permutation: [3, 1, 2, 0]
        self.assertEqual(new_val.shape, (4, 5, 3, 2))

        # Check physical re-indexing
        self.assertEqual(new_out[0].val_dim, 0)
        self.assertEqual(new_primal[0].val_dim, 1)
        self.assertEqual(new_out[0].block_val_dim, 2)
        self.assertEqual(new_out[1].val_dim, 3)

    def test_sort_val_none(self):
        """Validates the bypass when val is unmaterialized."""
        out_dims = [DenseDimension(id=0, size=3, val_dim=0)]
        primal_dims = [DenseDimension(id=1, size=2, val_dim=1)]

        new_out, new_primal, new_val = _sort_val(out_dims, primal_dims, None)

        self.assertIsNone(new_val)
        self.assertEqual(new_out, tuple(out_dims))
        self.assertEqual(new_primal, tuple(primal_dims))

    # TODO make the ones bellow the general init tests
    def test_bad_sort_val_one_sparse(self):
        d0 = SparseDimension(id=0, size=2, val_dim=0, other_id=1)
        d1 = SparseDimension(id=0, size=2, val_dim=0, other_id=1)
        with self.assertRaises(AssertionError):
            SparseTensor((d0,), (d1,), jnp.ones((2,)))

    def test_bad_sort_val_two_sparse(self):
        d0 = SparseDimension(id=0, size=2, val_dim=0, other_id=1)
        d1 = SparseDimension(id=1, size=2, val_dim=0, other_id=0)
        d2 = SparseDimension(id=0, size=2, val_dim=1, other_id=3)
        d3 = SparseDimension(id=3, size=2, val_dim=1, other_id=0)
        with self.assertRaises(AssertionError):
            SparseTensor((d0, d1), (d2, d3), jnp.ones((2, 2)))

    def test_bad_sort_val_mixed(self):
        d0 = DenseDimension(id=0, size=2, val_dim=0)
        d1 = SparseDimension(id=0, size=2, val_dim=1, other_id=2)
        d2 = SparseDimension(id=2, size=2, val_dim=1, other_id=0)
        with self.assertRaises(AssertionError):
            SparseTensor((d0,), (d1, d2), jnp.ones((2, 2)))

    # WIP
    def test_bad_val_dim_none(self):
        return
        stb = SparseTensor(
            stb.out_dims, stb.primal_dims, jnp.ones((4, 5, 5))
        )  # assert error


if __name__ == "__main__":
    unittest.main()
