import unittest
from itertools import combinations, count

import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.sparse.tensor import (
    DenseIndex,
    SparseIndex,
    SparseTensor,
)


class TestDenseNones(unittest.TestCase):
    def get_keys(self, seed=42):
        return jrand.split(jrand.PRNGKey(seed), 10)

    def test_arr2st_densification(self):
        cases = [
            ((4, 5), 1),
            ((4, 5), 2),
            ((3, 4, 5), 1),
            ((3, 4, 5), 2),
            ((3, 4, 5), 3),
            ((2, 3, 4, 5), 1),
            ((2, 3, 4, 5), 2),
            ((2, 3, 4, 5), 3),
            ((2, 3, 4, 5), 4),
        ]
        keys = self.get_keys()

        for shape, nones in cases:
            with self.subTest(shape=shape, nones=nones):
                n_c = len(shape) // 2
                for c in combinations(range(len(shape)), nones):
                    dims = []
                    axis = count()
                    for i, s in enumerate(shape):
                        dims.append(
                            DenseIndex(i, s, None if i in c else next(axis))
                        )

                    none_shape = tuple(d.size for d in dims if d.axis is not None)

                    val = jrand.normal(keys[0], none_shape) if none_shape else None
                    st = SparseTensor(
                        dims[:n_c], dims[n_c:], val if none_shape else None
                    )

                    if val is None:
                        dense_ref = jnp.ones(shape)
                    else:
                        broadcast_dims = tuple(
                            i for i in range(len(shape)) if i not in c
                        )
                        dense_ref = jax.lax.broadcast_in_dim(val, shape, broadcast_dims)

                    recovered = st.dense()
                    self.assertEqual(recovered.shape, dense_ref.shape)
                    self.assertTrue(jnp.allclose(recovered, dense_ref))

    def test_block_diagonal_2d_densification_first_block_none(self):
        keys = self.get_keys()
        val = jrand.normal(keys[0], (2, 4))

        d0 = SparseIndex(
            id=0, size=2, axis=0, other_id=1, block_size=3, block_axis=None
        )
        d1 = SparseIndex(
            id=1, size=2, axis=0, other_id=0, block_size=4, block_axis=1
        )

        st = SparseTensor((d0,), (d1,), val)

        val = jax.lax.broadcast_in_dim(val, (2, 3, 4), (0, 2))

        dense_ref = jnp.zeros((6, 8))
        dense_ref = dense_ref.at[0:3, 0:4].set(val[0])
        dense_ref = dense_ref.at[3:6, 4:8].set(val[1])

        recovered = st.dense()
        self.assertEqual(recovered.shape, dense_ref.shape)
        self.assertTrue(jnp.allclose(recovered, dense_ref))

    def test_block_diagonal_2d_densification_second_block_none(self):
        keys = self.get_keys()
        val = jrand.normal(keys[0], (2, 3))

        d0 = SparseIndex(
            id=0, size=2, axis=0, other_id=1, block_size=3, block_axis=1
        )
        d1 = SparseIndex(
            id=1, size=2, axis=0, other_id=0, block_size=4, block_axis=None
        )

        st = SparseTensor((d0,), (d1,), val)

        val = jax.lax.broadcast_in_dim(val, (2, 3, 4), (0, 1))

        dense_ref = jnp.zeros((6, 8))
        dense_ref = dense_ref.at[0:3, 0:4].set(val[0])
        dense_ref = dense_ref.at[3:6, 4:8].set(val[1])

        recovered = st.dense()
        self.assertEqual(recovered.shape, dense_ref.shape)
        self.assertTrue(jnp.allclose(recovered, dense_ref))

    def test_block_diagonal_2d_densification_both_blocks_nones(self):
        keys = self.get_keys()
        val = jrand.normal(keys[0], (2,))

        d0 = SparseIndex(
            id=0, size=2, axis=0, other_id=1, block_size=3, block_axis=None
        )
        d1 = SparseIndex(
            id=1, size=2, axis=0, other_id=0, block_size=4, block_axis=None
        )

        st = SparseTensor((d0,), (d1,), val)

        val = jax.lax.broadcast_in_dim(val, (2, 3, 4), (0,))

        dense_ref = jnp.zeros((6, 8))
        dense_ref = dense_ref.at[0:3, 0:4].set(val[0])
        dense_ref = dense_ref.at[3:6, 4:8].set(val[1])

        recovered = st.dense()
        self.assertEqual(recovered.shape, dense_ref.shape)
        self.assertTrue(jnp.allclose(recovered, dense_ref))

    def test_block_diagonal_2d_densification_sparse_none(self):
        keys = self.get_keys()
        val = jrand.normal(keys[0], (3, 4))

        d0 = SparseIndex(
            id=0, size=2, axis=None, other_id=1, block_size=3, block_axis=0
        )
        d1 = SparseIndex(
            id=1, size=2, axis=None, other_id=0, block_size=4, block_axis=1
        )

        st = SparseTensor((d0,), (d1,), val)

        val = jnp.broadcast_to(val, (2, 3, 4))
        dense_ref = jnp.zeros((6, 8))
        dense_ref = dense_ref.at[0:3, 0:4].set(val[0])
        dense_ref = dense_ref.at[3:6, 4:8].set(val[1])

        recovered = st.dense()
        self.assertEqual(recovered.shape, dense_ref.shape)
        self.assertTrue(jnp.allclose(recovered, dense_ref))

    def test_block_diagonal_2d_densification_all_none(self):
        d0 = SparseIndex(
            id=0, size=2, axis=None, other_id=1, block_size=3, block_axis=None
        )
        d1 = SparseIndex(
            id=1, size=2, axis=None, other_id=0, block_size=4, block_axis=None
        )

        st = SparseTensor((d0,), (d1,), None)

        block = jnp.ones((3, 4))
        dense_ref = jnp.zeros((6, 8))
        dense_ref = dense_ref.at[0:3, 0:4].set(block)
        dense_ref = dense_ref.at[3:6, 4:8].set(block)

        recovered = st.dense()
        self.assertEqual(recovered.shape, dense_ref.shape)
        self.assertTrue(jnp.allclose(recovered, dense_ref))

    def test_pure_diagonal_2d_densification(self):
        d0 = SparseIndex(id=0, size=2, axis=None, other_id=1)
        d1 = SparseIndex(id=1, size=2, axis=None, other_id=0)

        st = SparseTensor((d0,), (d1,), None)

        dense_ref = jnp.eye(2)

        recovered = st.dense()
        self.assertEqual(recovered.shape, dense_ref.shape)
        self.assertTrue(jnp.allclose(recovered, dense_ref))

    def test_block_diagonal_3d_densification(self):
        keys = self.get_keys()
        val = jrand.normal(keys[0], (2, 4))

        d0 = SparseIndex(
            id=0, size=2, axis=0, other_id=2, block_size=3, block_axis=None
        )
        d1 = DenseIndex(id=1, size=3, axis=None)
        d2 = SparseIndex(
            id=2, size=2, axis=0, other_id=0, block_size=4, block_axis=1
        )

        st = SparseTensor((d0, d1), (d2,), val)

        b_val = jax.lax.broadcast_in_dim(val, (2, 3, 4), (0, 2))
        dense_ref = jnp.zeros((6, 3, 8))
        dense_ref = dense_ref.at[0:3, :, 0:4].set(b_val[0])
        dense_ref = dense_ref.at[3:6, :, 4:8].set(b_val[1])

        recovered = st.dense()
        self.assertEqual(recovered.shape, dense_ref.shape)
        self.assertTrue(jnp.allclose(recovered, dense_ref))

    def test_mixed_diagonal_4d_densification(self):
        keys = self.get_keys()
        val = jrand.normal(keys[0], (2, 3))

        d0 = SparseIndex(
            id=0, size=2, axis=0, other_id=2, block_size=2, block_axis=None
        )
        d1 = SparseIndex(id=1, size=3, axis=1, other_id=3)
        d2 = SparseIndex(
            id=2, size=2, axis=0, other_id=0, block_size=3, block_axis=None
        )
        d3 = SparseIndex(id=3, size=3, axis=1, other_id=1)

        st = SparseTensor((d0, d1), (d2, d3), val)

        dense_ref = jnp.zeros((4, 3, 6, 3))
        for i in range(2):
            for j in range(3):
                b_val = jax.lax.broadcast_in_dim(val[i, j], (2, 3), ())
                dense_ref = dense_ref.at[
                    i * 2 : (i + 1) * 2, j, i * 3 : (i + 1) * 3, j
                ].set(b_val)

        recovered = st.dense()
        self.assertEqual(recovered.shape, dense_ref.shape)
        self.assertTrue(jnp.allclose(recovered, dense_ref))

    def test_mixed_diagonal_5d_densification(self):
        d0 = SparseIndex(
            id=0, size=2, axis=None, other_id=3, block_size=2, block_axis=None
        )
        d1 = SparseIndex(id=1, size=3, axis=None, other_id=4)
        d2 = DenseIndex(id=2, size=4, axis=None)
        d3 = SparseIndex(
            id=3, size=2, axis=None, other_id=0, block_size=2, block_axis=None
        )
        d4 = SparseIndex(id=4, size=3, axis=None, other_id=1)

        st = SparseTensor((d0, d1, d2), (d3, d4), None)

        dense_ref = jnp.zeros((4, 3, 4, 4, 3))
        for i in range(2):
            for j in range(3):
                dense_ref = dense_ref.at[
                    i * 2 : (i + 1) * 2, j, :, i * 2 : (i + 1) * 2, j
                ].set(1.0)

        recovered = st.dense()
        self.assertEqual(recovered.shape, dense_ref.shape)
        self.assertTrue(jnp.allclose(recovered, dense_ref))

    def test_mixed_diagonal_6d_densification(self):
        d0 = SparseIndex(
            id=0, size=2, axis=None, other_id=3, block_size=2, block_axis=None
        )
        d1 = SparseIndex(
            id=1, size=2, axis=None, other_id=4, block_size=3, block_axis=None
        )
        d2 = SparseIndex(id=2, size=3, axis=None, other_id=5)
        d3 = SparseIndex(
            id=3, size=2, axis=None, other_id=0, block_size=4, block_axis=None
        )
        d4 = SparseIndex(
            id=4, size=2, axis=None, other_id=1, block_size=2, block_axis=None
        )
        d5 = SparseIndex(id=5, size=3, axis=None, other_id=2)

        st = SparseTensor((d0, d1, d2), (d3, d4, d5), None)

        dense_ref = jnp.zeros((4, 6, 3, 8, 4, 3))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    dense_ref = dense_ref.at[
                        i * 2 : (i + 1) * 2,
                        j * 3 : (j + 1) * 3,
                        k,
                        i * 4 : (i + 1) * 4,
                        j * 2 : (j + 1) * 2,
                        k,
                    ].set(1.0)

        recovered = st.dense()
        self.assertEqual(recovered.shape, dense_ref.shape)
        self.assertTrue(jnp.allclose(recovered, dense_ref))


if __name__ == "__main__":
    unittest.main()
