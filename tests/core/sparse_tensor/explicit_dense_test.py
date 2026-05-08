import unittest

import jax.numpy as jnp
import jax.random as jrand

from graphax.sparse.tensor import (
    DenseIndex,
    SparseIndex,
    SparseTensor,
    _arr2st,
    dense,
)


class TestDenseDiags(unittest.TestCase):
    def get_keys(self, seed=42):
        return jrand.split(jrand.PRNGKey(seed), 10)

    def test_densify_arr2st(self):
        cases = [
            ((4, 5), 0),
            ((4, 5), 1),
            ((4, 5), 2),
            ((3, 4, 5), 0),
            ((3, 4, 5), 1),
            ((3, 4, 5), 2),
            ((3, 4, 5), 3),
            ((2, 3, 4, 5), 0),
            ((2, 3, 4, 5), 1),
            ((2, 3, 4, 5), 2),
            ((2, 3, 4, 5), 3),
            ((2, 3, 4, 5), 4),
        ]
        keys = self.get_keys()
        for shape, out_ndim in cases:
            with self.subTest(shape=shape, out_ndim=out_ndim):
                arr = jrand.normal(keys[0], shape)
                st = _arr2st(arr, out_ndim=out_ndim)
                st1 = SparseTensor(st.out_dims, st.primal_dims, st.val)
                st2 = SparseTensor(st.out_dims, st.primal_dims, st.val)
                self.assertTrue((st == st1).all())
                self.assertTrue((st1 == st2).all())
                self.assertEqual(len(st.out_dims), out_ndim)
                self.assertTrue(jnp.allclose(st.dense(), arr))

    def test_densify_block_2d(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 4))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=1, block_size=3, block_axis=1)
        d1 = SparseIndex(id=1, size=2, axis=0, other_id=0, block_size=4, block_axis=2)
        st = SparseTensor(
            (d0,),
            (d1,),
            val,
        )
        st2 = SparseTensor((d0,), (d1,), val)
        self.assertTrue((st == st2).all())
        dense_ref = jnp.zeros((6, 8))
        dense_ref = dense_ref.at[0:3, 0:4].set(val[0])
        dense_ref = dense_ref.at[3:6, 4:8].set(val[1])
        self.assertTrue(jnp.allclose(st.dense(), dense_ref))

    def test_densify_block_3d(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 4, 5))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=1, block_size=3, block_axis=1)
        d1 = SparseIndex(id=1, size=2, axis=0, other_id=0, block_size=4, block_axis=2)
        d2 = DenseIndex(id=2, size=5, axis=3)
        st = SparseTensor((d0,), (d1, d2), val)
        st2 = SparseTensor((d0,), (d1, d2), val)
        self.assertTrue((st == st2).all())
        dense_ref = jnp.zeros((6, 8, 5))
        dense_ref = dense_ref.at[0:3, 0:4, :].set(val[0])
        dense_ref = dense_ref.at[3:6, 4:8, :].set(val[1])
        self.assertTrue(jnp.allclose(st.dense(), dense_ref))

    def test_densify_block_4d_1pair(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 5, 4, 6))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2, block_size=3, block_axis=1)
        d1 = DenseIndex(id=1, size=5, axis=2)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0, block_size=4, block_axis=3)
        d3 = DenseIndex(id=3, size=6, axis=4)
        st = SparseTensor((d0, d1), (d2, d3), val)
        st2 = SparseTensor((d0, d1), (d2, d3), val)
        self.assertTrue((st == st2).all())
        dense_ref = jnp.zeros((6, 5, 8, 6))
        for i in range(2):
            block = val[i]
            for b0 in range(3):
                for b1 in range(4):
                    dense_ref = dense_ref.at[i * 3 + b0, :, i * 4 + b1, :].set(
                        block[b0, :, b1, :]
                    )
        self.assertTrue(jnp.allclose(st.dense(), dense_ref))

    def test_densify_block_4d_2pairs(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 4, 6, 5, 7))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2, block_size=4, block_axis=2)
        d1 = SparseIndex(id=1, size=3, axis=1, other_id=3, block_size=6, block_axis=3)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0, block_size=5, block_axis=4)
        d3 = SparseIndex(id=3, size=3, axis=1, other_id=1, block_size=7, block_axis=5)
        st = SparseTensor((d0, d1), (d2, d3), val)
        st2 = SparseTensor((d0, d1), (d2, d3), val)
        self.assertTrue((st == st2).all())
        dense_ref = jnp.zeros((8, 18, 10, 21))
        for i in range(2):
            for j in range(3):
                block = val[i, j]
                for b0 in range(4):
                    for b1 in range(6):
                        for b2 in range(5):
                            for b3 in range(7):
                                dense_ref = dense_ref.at[
                                    i * 4 + b0, j * 6 + b1, i * 5 + b2, j * 7 + b3
                                ].set(block[b0, b1, b2, b3])
        self.assertTrue(jnp.allclose(st.dense(), dense_ref))

    def test_densify_pure_2d(self):
        val = jrand.normal(self.get_keys()[0], (2,))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=1)
        d1 = SparseIndex(id=1, size=2, axis=0, other_id=0)
        st = SparseTensor((d0,), (d1,), val)
        st2 = SparseTensor((d0,), (d1,), val)
        self.assertTrue((st == st2).all())
        dense_ref = jnp.zeros((2, 2))
        dense_ref = dense_ref.at[0, 0].set(val[0])
        dense_ref = dense_ref.at[1, 1].set(val[1])
        self.assertTrue(jnp.allclose(st.dense(), dense_ref))

    def test_densify_pure_3d(self):
        val = jrand.normal(self.get_keys()[0], (2, 3))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2)
        d1 = DenseIndex(id=1, size=3, axis=1)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0)
        st = SparseTensor((d0,), (d2, d1), val)
        st2 = SparseTensor((d0,), (d2, d1), val)
        self.assertTrue((st == st2).all())
        dense_ref = jnp.zeros((2, 2, 3))
        dense_ref = dense_ref.at[0, 0, :].set(val[0])
        dense_ref = dense_ref.at[1, 1, :].set(val[1])
        self.assertTrue(jnp.allclose(st.dense(), dense_ref))

    def test_densify_pure_4d_1pair(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 4))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2)
        d1 = DenseIndex(id=1, size=3, axis=1)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0)
        d3 = DenseIndex(id=3, size=4, axis=2)
        st = SparseTensor((d0, d1), (d2, d3), val)
        st2 = SparseTensor((d0, d1), (d2, d3), val)
        self.assertTrue((st == st2).all())
        dense_ref = jnp.zeros((2, 3, 2, 4))
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    dense_ref = dense_ref.at[i, j, i, k].set(val[i, j, k])
        self.assertTrue(jnp.allclose(st.dense(), dense_ref))

    def test_densify_pure_4d_2pairs(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 4))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2)
        d1 = DenseIndex(id=1, size=3, axis=1)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0)
        d3 = DenseIndex(id=3, size=4, axis=2)
        st = SparseTensor((d0, d1), (d2, d3), val)
        st2 = SparseTensor((d0, d1), (d2, d3), val)
        self.assertTrue((st == st2).all())
        dense_ref = jnp.zeros((2, 3, 2, 4))
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    dense_ref = dense_ref.at[i, j, i, k].set(val[i, j, k])
        self.assertTrue(jnp.allclose(st.dense(), dense_ref))

    def test_densify_mixed_4d_1pair(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 4))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2)
        d1 = SparseIndex(id=1, size=3, axis=1, other_id=3)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0)
        d3 = SparseIndex(id=3, size=3, axis=1, other_id=1, block_size=4, block_axis=2)
        st = SparseTensor((d0, d1), (d2, d3), val)
        st2 = SparseTensor((d0, d1), (d2, d3), val)
        self.assertTrue((st == st2).all())
        dense_ref = jnp.zeros((2, 3, 2, 12))
        for i in range(2):
            for j in range(3):
                block = val[i, j]
                for b0 in range(4):
                    dense_ref = dense_ref.at[i, j, i, j * 4 + b0].set(block[b0])
        self.assertTrue(jnp.allclose(st.dense(), dense_ref))

    def test_densify_mixed_4d_2pairs_v1(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 5, 4))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2)
        d1 = SparseIndex(id=1, size=3, axis=1, other_id=3, block_size=5, block_axis=2)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0, block_size=4, block_axis=3)
        d3 = SparseIndex(id=3, size=3, axis=1, other_id=1)
        st = SparseTensor((d0, d1), (d2, d3), val)
        st2 = SparseTensor((d0, d1), (d2, d3), val)
        self.assertTrue((st == st2).all())
        dense_ref = jnp.zeros((2, 15, 8, 3))
        for i in range(2):
            for j in range(3):
                block = val[i, j]
                for b0 in range(5):
                    for b1 in range(4):
                        dense_ref = dense_ref.at[i, j * 5 + b0, i * 4 + b1, j].set(
                            block[b0, b1]
                        )
        self.assertTrue(jnp.allclose(st.dense(), dense_ref))

    def test_densify_mixed_4d_2pairs_v2(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 5, 4, 6))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2)
        d1 = SparseIndex(id=1, size=3, axis=1, other_id=3, block_size=5, block_axis=2)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0, block_size=4, block_axis=3)
        d3 = SparseIndex(id=3, size=3, axis=1, other_id=1, block_size=6, block_axis=4)
        st = SparseTensor((d0, d1), (d2, d3), val)
        st2 = SparseTensor((d0, d1), (d2, d3), val)
        self.assertTrue((st == st2).all())
        dense_ref = jnp.zeros((2, 15, 8, 18))
        for i in range(2):
            for j in range(3):
                block = val[i, j]
                for b0 in range(5):
                    for b1 in range(4):
                        for b2 in range(6):
                            dense_ref = dense_ref.at[
                                i, j * 5 + b0, i * 4 + b1, j * 6 + b2
                            ].set(block[b0, b1, b2])
        self.assertTrue(jnp.allclose(st.dense(), dense_ref))

    def test_partial_pure_4d_axis0(self):
        val = jrand.normal(self.get_keys()[0], (2, 3))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2)
        d1 = SparseIndex(id=1, size=3, axis=1, other_id=3)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0)
        d3 = SparseIndex(id=3, size=3, axis=1, other_id=1)
        st = SparseTensor((d0, d1), (d2, d3), val)
        st_p = dense(st, axes=(0,))
        self.assertTrue((st_p == dense(st, axes=(0, 2))).all())
        self.assertTrue(
            isinstance(st_p.out_dims[0], DenseIndex)
            and isinstance(st_p.primal_dims[0], DenseIndex)
        )
        self.assertTrue(jnp.allclose(st_p.dense(), st.dense()))

    def test_partial_pure_4d_axis1(self):
        val = jrand.normal(self.get_keys()[0], (2, 3))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2)
        d1 = SparseIndex(id=1, size=3, axis=1, other_id=3)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0)
        d3 = SparseIndex(id=3, size=3, axis=1, other_id=1)
        st = SparseTensor((d0, d1), (d2, d3), val)
        st_p = dense(st, axes=(1,))
        self.assertTrue((st_p == dense(st, axes=(1, 3))).all())
        self.assertTrue(
            isinstance(st_p.out_dims[1], DenseIndex)
            and isinstance(st_p.primal_dims[1], DenseIndex)
        )
        self.assertTrue(jnp.allclose(st_p.dense(), st.dense()))

    def test_partial_block_4d_axis0(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 4, 5))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2, block_size=4, block_axis=2)
        d1 = SparseIndex(id=1, size=3, axis=1, other_id=3, block_size=5, block_axis=3)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0)
        d3 = SparseIndex(id=3, size=3, axis=1, other_id=1)
        st = SparseTensor((d0, d1), (d2, d3), val)
        st_p = dense(st, axes=(0,))
        self.assertTrue((st_p == dense(st, axes=(0, 2))).all())
        self.assertEqual(st_p.out_dims[0].size, 8)
        self.assertEqual(st_p.primal_dims[0].size, 2)
        self.assertTrue(jnp.allclose(st_p.dense(), st.dense()))

    def test_partial_block_4d_axis1(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 4, 5))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2, block_size=4, block_axis=2)
        d1 = SparseIndex(id=1, size=3, axis=1, other_id=3, block_size=5, block_axis=3)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0)
        d3 = SparseIndex(id=3, size=3, axis=1, other_id=1)
        st = SparseTensor((d0, d1), (d2, d3), val)
        st_p = dense(st, axes=(1,))
        self.assertTrue((st_p == dense(st, axes=(1, 3))).all())
        self.assertEqual(st_p.out_dims[1].size, 15)
        self.assertEqual(st_p.primal_dims[1].size, 3)
        self.assertTrue(jnp.allclose(st_p.dense(), st.dense()))

    def test_partial_mixed_4d_axis0(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 5))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2)
        d1 = SparseIndex(id=1, size=3, axis=1, other_id=3, block_size=5, block_axis=2)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0)
        d3 = SparseIndex(id=3, size=3, axis=1, other_id=1)
        st = SparseTensor((d0, d1), (d2, d3), val)
        st_p = dense(st, axes=(0,))
        self.assertTrue((st_p == dense(st, axes=(0, 2))).all())
        self.assertTrue(
            isinstance(st_p.out_dims[0], DenseIndex)
            and isinstance(st_p.primal_dims[0], DenseIndex)
        )
        self.assertTrue(jnp.allclose(st_p.dense(), st.dense()))

    def test_partial_mixed_4d_axis1(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 5))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=2)
        d1 = SparseIndex(id=1, size=3, axis=1, other_id=3, block_size=5, block_axis=2)
        d2 = SparseIndex(id=2, size=2, axis=0, other_id=0)
        d3 = SparseIndex(id=3, size=3, axis=1, other_id=1)
        st = SparseTensor((d0, d1), (d2, d3), val)
        st_p = dense(st, axes=(1,))
        self.assertTrue((st_p == dense(st, axes=(1, 3))).all())
        self.assertTrue(
            isinstance(st_p.out_dims[1], DenseIndex)
            and isinstance(st_p.primal_dims[1], DenseIndex)
        )
        self.assertTrue(jnp.allclose(st_p.dense(), st.dense()))

    def test_partial_pure_5d_axis0(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 4))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=3)
        d1 = SparseIndex(id=1, size=3, axis=1, other_id=4)
        d2 = DenseIndex(id=2, size=4, axis=2)
        d3 = SparseIndex(id=3, size=2, axis=0, other_id=0)
        d4 = SparseIndex(id=4, size=3, axis=1, other_id=1)
        st = SparseTensor((d0, d1, d2), (d3, d4), val)
        st_p = dense(st, axes=(0,))
        self.assertTrue((st_p == dense(st, axes=(0, 3))).all())
        self.assertTrue(
            isinstance(st_p.out_dims[0], DenseIndex)
            and isinstance(st_p.primal_dims[0], DenseIndex)
        )
        self.assertTrue(jnp.allclose(st_p.dense(), st.dense()))

    def test_partial_pure_5d_axis1(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 4))
        d0 = SparseIndex(id=0, size=2, axis=0, other_id=3)
        d1 = SparseIndex(id=1, size=3, axis=1, other_id=4)
        d2 = DenseIndex(id=2, size=4, axis=2)
        d3 = SparseIndex(id=3, size=2, axis=0, other_id=0)
        d4 = SparseIndex(id=4, size=3, axis=1, other_id=1)
        st = SparseTensor((d0, d1, d2), (d3, d4), val)
        st_p = dense(st, axes=(1,))
        self.assertTrue((st_p == dense(st, axes=(1, 4))).all())
        self.assertTrue(
            isinstance(st_p.out_dims[1], DenseIndex)
            and isinstance(st_p.primal_dims[1], DenseIndex)
        )
        self.assertTrue(jnp.allclose(st_p.dense(), st.dense()))

    def test_partial_pure_6d_single_axis(self):
        cases = [0, 1, 2]
        for axis in cases:
            with self.subTest(axis=axis):
                val = jrand.normal(self.get_keys()[0], (2, 3, 4))
                d0 = SparseIndex(id=0, size=2, axis=0, other_id=3)
                d1 = SparseIndex(id=1, size=3, axis=1, other_id=4)
                d2 = SparseIndex(id=2, size=4, axis=2, other_id=5)
                d3 = SparseIndex(id=3, size=2, axis=0, other_id=0)
                d4 = SparseIndex(id=4, size=3, axis=1, other_id=1)
                d5 = SparseIndex(id=5, size=4, axis=2, other_id=2)
                st = SparseTensor((d0, d1, d2), (d3, d4, d5), val)
                st_p = dense(st, axes=(axis,))
                self.assertTrue((st_p == dense(st, axes=(axis, axis + 3))).all())
                self.assertTrue(
                    isinstance(st_p.out_dims[axis], DenseIndex)
                    and isinstance(st_p.primal_dims[axis], DenseIndex)
                )
                self.assertTrue(jnp.allclose(st_p.dense(), st.dense()))

    def test_partial_pure_6d_multiple_axes(self):
        cases = [(0, 1), (0, 2), (1, 2)]
        for axes in cases:
            with self.subTest(axes=axes):
                val = jrand.normal(self.get_keys()[0], (2, 3, 4))
                d0 = SparseIndex(id=0, size=2, axis=0, other_id=3)
                d1 = SparseIndex(id=1, size=3, axis=1, other_id=4)
                d2 = SparseIndex(id=2, size=4, axis=2, other_id=5)
                d3 = SparseIndex(id=3, size=2, axis=0, other_id=0)
                d4 = SparseIndex(id=4, size=3, axis=1, other_id=1)
                d5 = SparseIndex(id=5, size=4, axis=2, other_id=2)
                st = SparseTensor((d0, d1, d2), (d3, d4, d5), val)
                st_p = dense(st, axes=axes)
                f_axes = tuple(axes) + tuple(a + 3 for a in axes)
                self.assertTrue((st_p == dense(st, axes=f_axes)).all())
                for a in axes:
                    self.assertTrue(
                        isinstance(st_p.out_dims[a], DenseIndex)
                        and isinstance(st_p.primal_dims[a], DenseIndex)
                    )
                self.assertTrue(jnp.allclose(st_p.dense(), st.dense()))

    def test_partial_block_5d(self):
        cases = [(0,), (1,)]
        for axes in cases:
            with self.subTest(axes=axes):
                val = jrand.normal(self.get_keys()[0], (2, 3, 6, 4, 5, 8, 9))
                d0 = SparseIndex(
                    id=0, size=2, axis=0, other_id=3, block_size=6, block_axis=2
                )
                d1 = SparseIndex(
                    id=1, size=3, axis=1, other_id=4, block_size=4, block_axis=3
                )
                d2 = DenseIndex(id=2, size=5, axis=4)
                d3 = SparseIndex(
                    id=3, size=2, axis=0, other_id=0, block_size=8, block_axis=5
                )
                d4 = SparseIndex(
                    id=4, size=3, axis=1, other_id=1, block_size=9, block_axis=6
                )
                st = SparseTensor((d0, d1, d2), (d3, d4), val)
                st_p = dense(st, axes=axes)
                f_axes = tuple(axes) + tuple(a + 3 for a in axes)
                self.assertTrue((st_p == dense(st, axes=f_axes)).all())
                for a in axes:
                    self.assertTrue(
                        isinstance(st_p.out_dims[a], DenseIndex)
                        and isinstance(st_p.primal_dims[a], DenseIndex)
                    )
                self.assertTrue(jnp.allclose(st_p.dense(), st.dense()))

    # this is toooo large, takes over 5 minutes!
    # def test_partial_block_6d(self):
    #     cases = [(1, 2), (0, 1), (0, 2), (0, 1, 2)]
    #     for axes in cases:
    #         with self.subTest(axes=axes):
    #             val = jrand.normal(self.get_keys()[0], (2, 3, 4, 6, 7, 8, 9, 10, 11))
    #             d0 = SparseIndex(
    #                 id=0, size=2, axis=0, other_id=3, block_size=6, block_axis=3
    #             )
    #             d1 = SparseIndex(
    #                 id=1, size=3, axis=1, other_id=4, block_size=7, block_axis=4
    #             )
    #             d2 = SparseIndex(
    #                 id=2, size=4, axis=2, other_id=5, block_size=8, block_axis=5
    #             )
    #             d3 = SparseIndex(
    #                 id=3, size=2, axis=0, other_id=0, block_size=9, block_axis=6
    #             )
    #             d4 = SparseIndex(
    #                 id=4, size=3, axis=1, other_id=1, block_size=10, block_axis=7
    #             )
    #             d5 = SparseIndex(
    #                 id=5, size=4, axis=2, other_id=2, block_size=11, block_axis=8
    #             )
    #             st = SparseTensor((d0, d1, d2), (d3, d4, d5), val)
    #             st_p = dense(st, axes=axes)
    #             f_axes = tuple(axes) + tuple(a + 3 for a in axes)
    #             self.assertTrue((st_p == dense(st, axes=f_axes)).all())
    #             for a in axes:
    #                 self.assertTrue(
    #                     isinstance(st_p.out_dims[a], DenseIndex)
    #                     and isinstance(st_p.primal_dims[a], DenseIndex)
    #                 )
    #             self.assertTrue(jnp.allclose(st_p.dense(), st.dense()))

    def test_shuffled_4d(self):
        val = jrand.normal(self.get_keys()[0], (2, 3))
        d0, d1 = SparseIndex(0, 2, 0, 2), SparseIndex(1, 3, 1, 3)
        d2, d3 = SparseIndex(2, 2, 0, 0), SparseIndex(3, 3, 1, 1)
        st = SparseTensor((d0, d1), (d3, d2), val)
        self.assertTrue((st == SparseTensor((d0, d1), (d3, d2), val)).all())
        dense_ref = jnp.zeros((2, 3, 3, 2))
        for i in range(2):
            for j in range(3):
                dense_ref = dense_ref.at[i, j, j, i].set(val[i, j])
        self.assertTrue(jnp.allclose(st.dense(), dense_ref))
        st_p = dense(st, axes=(0,))
        self.assertTrue(
            isinstance(st_p.out_dims[0], DenseIndex)
            and isinstance(st_p.primal_dims[1], DenseIndex)
        )
        self.assertTrue(jnp.allclose(st_p.dense(), dense_ref))

    def test_shuffled_6d(self):
        val = jrand.normal(self.get_keys()[0], (2, 3, 4))
        d0, d1, d2 = (
            SparseIndex(0, 2, 0, 3),
            SparseIndex(1, 3, 1, 4),
            SparseIndex(2, 4, 2, 5),
        )
        d3, d4, d5 = (
            SparseIndex(3, 2, 0, 0),
            SparseIndex(4, 3, 1, 1),
            SparseIndex(5, 4, 2, 2),
        )
        st = SparseTensor((d0, d1, d2), (d3, d5, d4), val)
        self.assertTrue((st == SparseTensor((d0, d1, d2), (d3, d5, d4), val)).all())
        dense_ref = jnp.zeros((2, 3, 4, 2, 4, 3))
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    dense_ref = dense_ref.at[i, j, k, i, k, j].set(val[i, j, k])
        self.assertTrue(jnp.allclose(st.dense(), dense_ref))
        st_p = dense(st, axes=(1,))
        self.assertTrue(
            isinstance(st_p.out_dims[1], DenseIndex)
            and isinstance(st_p.primal_dims[2], DenseIndex)
        )
        self.assertTrue(jnp.allclose(st_p.dense(), dense_ref))


if __name__ == "__main__":
    unittest.main()
