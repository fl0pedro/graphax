import unittest
from itertools import combinations, count

import jax.numpy as jnp

from graphax.sparse.tensor import (
    DenseIndex,
    DiagonalIndex,
    SparseTensor,
    dense,
)


class TestSelfDenseNones(unittest.TestCase):
    def st_2d(self):
        return SparseTensor(
            (DiagonalIndex(0, 100, None, 1),),
            (DiagonalIndex(1, 100, None, 0),),
            None,
        )

    def bst_2d(self):
        return SparseTensor(
            (DiagonalIndex(0, 4, None, 1, 5),),
            (DiagonalIndex(1, 4, None, 0, 5),),
            None,
        )

    def bst_3d(self):
        return SparseTensor(
            (DiagonalIndex(0, 3, None, 2, 4),),
            (DenseIndex(1, 5, None), DiagonalIndex(2, 3, None, 0, 6)),
            None,
        )

    def bst_3d_val_dense(self):
        x = jnp.arange(5)
        return SparseTensor(
            (DiagonalIndex(0, 3, None, 2, 4),),
            (DenseIndex(1, 5, 0), DiagonalIndex(2, 3, None, 0, 6)),
            x,
        )

    def st_4d(self):
        return SparseTensor(
            (DiagonalIndex(0, 3, None, 2), DiagonalIndex(1, 4, None, 3)),
            (DiagonalIndex(2, 3, None, 0), DiagonalIndex(3, 4, None, 1)),
            None,
        )

    def st_4d_val_sparse_1(self):
        x = jnp.arange(3).reshape(3, 1, 1)
        return SparseTensor(
            (DiagonalIndex(0, 3, 0, 2), DiagonalIndex(1, 4, None, 3)),
            (DiagonalIndex(2, 3, 0, 0), DiagonalIndex(3, 4, None, 1)),
            x,
        )

    def st_4d_val_sparse_2(self):
        x = jnp.arange(4).reshape(4, 1, 1)
        return SparseTensor(
            (DiagonalIndex(0, 3, None, 2), DiagonalIndex(1, 4, 0, 3)),
            (DiagonalIndex(2, 3, None, 0), DiagonalIndex(3, 4, 0, 1)),
            x,
        )

    def bst_4d_1(self):
        return SparseTensor(
            (DiagonalIndex(0, 3, None, 2, 5), DiagonalIndex(1, 4, None, 3, 6)),
            (DiagonalIndex(2, 3, None, 0, 7), DiagonalIndex(3, 4, None, 1, 8)),
            None,
        )

    def bst_4d_1_val_sparse_1(self):
        x = jnp.arange(3 * 5 * 7).reshape(3, 5, 7)
        return SparseTensor(
            (
                DiagonalIndex(0, 3, 0, 2, 5, 1),
                DiagonalIndex(1, 4, None, 3, 6, None),
            ),
            (
                DiagonalIndex(2, 3, 0, 0, 7, 2),
                DiagonalIndex(3, 4, None, 1, 8, None),
            ),
            x,
        )

    def bst_4d_1_val_sparse_2(self):
        x = jnp.arange(4 * 6 * 8).reshape(4, 6, 8)
        return SparseTensor(
            (
                DiagonalIndex(0, 3, None, 2, 5, None),
                DiagonalIndex(1, 4, 0, 3, 6, 1),
            ),
            (
                DiagonalIndex(2, 3, None, 0, 7, None),
                DiagonalIndex(3, 4, 0, 1, 8, 2),
            ),
            x,
        )

    def bst_4d_2(self):
        return SparseTensor(
            (DiagonalIndex(0, 3, None, 2, 4), DenseIndex(1, 5, None)),
            (DiagonalIndex(2, 3, None, 0, 6), DenseIndex(3, 7, None)),
            None,
        )

    def bst_4d_2_val_sparse(self):
        x = jnp.arange(3 * 4 * 6).reshape(3, 4, 6)
        return SparseTensor(
            (DiagonalIndex(0, 3, 0, 2, 4, 1), DenseIndex(1, 5, None)),
            (DiagonalIndex(2, 3, 0, 0, 6, 2), DenseIndex(3, 7, None)),
            x,
        )

    def bst_4d_2_val_dense_1(self):
        x = jnp.arange(5)
        return SparseTensor(
            (DiagonalIndex(0, 3, None, 2, 4), DenseIndex(1, 5, 0)),
            (DiagonalIndex(2, 3, None, 0, 6), DenseIndex(3, 7, None)),
            x,
        )

    def bst_4d_2_val_dense_2(self):
        x = jnp.arange(7)
        return SparseTensor(
            (DiagonalIndex(0, 3, None, 2, 4), DenseIndex(1, 5, None)),
            (DiagonalIndex(2, 3, None, 0, 6), DenseIndex(3, 7, 0)),
            x,
        )

    def bst_4d_2_val_dense_3(self):
        x = jnp.arange(5 * 7).reshape(5, 7)
        return SparseTensor(
            (DiagonalIndex(0, 3, None, 2, 4), DenseIndex(1, 5, 0)),
            (DiagonalIndex(2, 3, None, 0, 6), DenseIndex(3, 7, 1)),
            x,
        )

    def bst_4d_2_val_sparse_and_dense(self):
        x = jnp.arange(3 * 4 * 6 * 7).reshape(3, 4, 6, 7)
        return SparseTensor(
            (DiagonalIndex(0, 3, 0, 2, 4, 1), DenseIndex(1, 5, None)),
            (DiagonalIndex(2, 3, 0, 0, 6, 2), DenseIndex(3, 7, 3)),
            x,
        )

    def get_all_fixtures(self):
        return [
            self.st_2d,
            self.bst_2d,
            self.bst_3d,
            self.bst_3d_val_dense,
            self.st_4d,
            self.st_4d_val_sparse_1,
            self.st_4d_val_sparse_2,
            self.bst_4d_1,
            self.bst_4d_1_val_sparse_1,
            self.bst_4d_1_val_sparse_2,
            self.bst_4d_2,
            self.bst_4d_2_val_sparse,
            self.bst_4d_2_val_dense_1,
            self.bst_4d_2_val_dense_2,
            self.bst_4d_2_val_dense_3,
            self.bst_4d_2_val_sparse_and_dense,
        ]

    def test_dense_once(self):
        for fixture_fn in self.get_all_fixtures():
            with self.subTest(fixture=fixture_fn.__name__):
                st = fixture_fn()
                result = dense(st)
                self.assertEqual(st.shape, result.shape)
                self.assertEqual(len(st.dims), len(result.dims))

    def test_dense_once_hard(self):
        for fixture_fn in self.get_all_fixtures():
            with self.subTest(fixture=fixture_fn.__name__):
                st = fixture_fn()
                result = dense(st, hard=True)
                self.assertEqual(st.shape, result.shape)
                self.assertEqual(len(st.dims), len(result.dims))
                if result.val is not None:
                    self.assertEqual(st.shape, result.val.shape)
                self.assertTrue(all(not d.is_sparse for d in result.dims))

    def test_dense_twice(self):
        for fixture_fn in self.get_all_fixtures():
            with self.subTest(fixture=fixture_fn.__name__):
                st = fixture_fn()
                result = dense(st, hard=True)
                ddense = dense(dense(st))
                self.assertTrue(jnp.allclose(result.dense(), ddense.dense()))

    def test_dense_axis(self):
        for fixture_fn in self.get_all_fixtures():
            with self.subTest(fixture=fixture_fn.__name__):
                st = fixture_fn()
                axes = [d.id for d in st.dims if d.axis is None]

                for r in range(len(axes)):
                    for axis in combinations(axes, r):
                        result = dense(st, axes=axis)
                        self.assertEqual(st.shape, result.shape)
                        self.assertEqual(len(st.dims), len(result.dims))

                self.assertTrue(
                    jnp.allclose(dense(st, axes=tuple(axes)).dense(), dense(st).dense())
                )

    def test_dense_axis_hard(self):
        for fixture_fn in self.get_all_fixtures():
            with self.subTest(fixture=fixture_fn.__name__):
                st = fixture_fn()
                axes = [d.id for d in st.dims if d.axis is None]
                for r in range(len(axes)):
                    for axis in combinations(axes, r):
                        result = dense(st, axes=axis, hard=True)
                        self.assertEqual(st.shape, result.shape)
                        self.assertEqual(len(st.dims), len(result.dims))

                self.assertTrue(
                    jnp.allclose(
                        dense(st, axes=tuple(axes), hard=True).dense(),
                        dense(st, hard=True).dense(),
                    )
                )


if __name__ == "__main__":
    unittest.main()
