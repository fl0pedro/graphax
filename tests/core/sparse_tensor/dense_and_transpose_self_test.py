import unittest

import jax.numpy as jnp

from graphax.sparse.tensor import (
    DenseIndex,
    SparseIndex,
    SparseTensor,
)
from utils import matmul_reference
from dataclasses import replace
from graphax.sparse.ops import dense as _dense


class TestSelfDenseAndTranspose(unittest.TestCase):
    def st_2d(self):
        return SparseTensor(
            (SparseIndex(0, 100, None, 1),),
            (SparseIndex(1, 100, None, 0),),
            None,
        )

    def bst_2d(self):
        return SparseTensor(
            (SparseIndex(0, 4, None, 1, 5),),
            (SparseIndex(1, 4, None, 0, 5),),
            None,
        )

    def bst_3d(self):
        return SparseTensor(
            (SparseIndex(0, 3, None, 2, 4),),
            (DenseIndex(1, 5, None), SparseIndex(2, 3, None, 0, 6)),
            None,
        )

    def bst_3d_val_dense(self):
        x = jnp.arange(5)
        return SparseTensor(
            (SparseIndex(0, 3, None, 2, 4),),
            (DenseIndex(1, 5, 0), SparseIndex(2, 3, None, 0, 6)),
            x,
        )

    def st_4d(self):
        return SparseTensor(
            (SparseIndex(0, 3, None, 2), SparseIndex(1, 4, None, 3)),
            (SparseIndex(2, 3, None, 0), SparseIndex(3, 4, None, 1)),
            None,
        )

    def st_4d_val_sparse_1(self):
        x = jnp.arange(3).reshape(3, 1, 1)
        return SparseTensor(
            (SparseIndex(0, 3, 0, 2), SparseIndex(1, 4, None, 3)),
            (SparseIndex(2, 3, 0, 0), SparseIndex(3, 4, None, 1)),
            x,
        )

    def st_4d_val_sparse_2(self):
        x = jnp.arange(4).reshape(4, 1, 1)
        return SparseTensor(
            (SparseIndex(0, 3, None, 2), SparseIndex(1, 4, 0, 3)),
            (SparseIndex(2, 3, None, 0), SparseIndex(3, 4, 0, 1)),
            x,
        )

    def bst_4d_1(self):
        return SparseTensor(
            (SparseIndex(0, 3, None, 2, 5), SparseIndex(1, 4, None, 3, 6)),
            (SparseIndex(2, 3, None, 0, 7), SparseIndex(3, 4, None, 1, 8)),
            None,
        )

    def bst_4d_1_val_sparse_1(self):
        x = jnp.arange(3 * 5 * 7).reshape(3, 5, 7)
        return SparseTensor(
            (
                SparseIndex(0, 3, 0, 2, 5, 1),
                SparseIndex(1, 4, None, 3, 6, None),
            ),
            (
                SparseIndex(2, 3, 0, 0, 7, 2),
                SparseIndex(3, 4, None, 1, 8, None),
            ),
            x,
        )

    def bst_4d_1_val_sparse_2(self):
        x = jnp.arange(4 * 6 * 8).reshape(4, 6, 8)
        return SparseTensor(
            (
                SparseIndex(0, 3, None, 2, 5, None),
                SparseIndex(1, 4, 0, 3, 6, 1),
            ),
            (
                SparseIndex(2, 3, None, 0, 7, None),
                SparseIndex(3, 4, 0, 1, 8, 2),
            ),
            x,
        )

    def bst_4d_2(self):
        return SparseTensor(
            (SparseIndex(0, 3, None, 2, 4), DenseIndex(1, 5, None)),
            (SparseIndex(2, 3, None, 0, 6), DenseIndex(3, 7, None)),
            None,
        )

    def bst_4d_2_val_sparse(self):
        x = jnp.arange(3 * 4 * 6).reshape(3, 4, 6)
        return SparseTensor(
            (SparseIndex(0, 3, 0, 2, 4, 1), DenseIndex(1, 5, None)),
            (SparseIndex(2, 3, 0, 0, 6, 2), DenseIndex(3, 7, None)),
            x,
        )

    def bst_4d_2_val_dense_1(self):
        x = jnp.arange(5)
        return SparseTensor(
            (SparseIndex(0, 3, None, 2, 4), DenseIndex(1, 5, 0)),
            (SparseIndex(2, 3, None, 0, 6), DenseIndex(3, 7, None)),
            x,
        )

    def bst_4d_2_val_dense_2(self):
        x = jnp.arange(7)
        return SparseTensor(
            (SparseIndex(0, 3, None, 2, 4), DenseIndex(1, 5, None)),
            (SparseIndex(2, 3, None, 0, 6), DenseIndex(3, 7, 0)),
            x,
        )

    def bst_4d_2_val_dense_3(self):
        x = jnp.arange(5 * 7).reshape(5, 7)
        return SparseTensor(
            (SparseIndex(0, 3, None, 2, 4), DenseIndex(1, 5, 0)),
            (SparseIndex(2, 3, None, 0, 6), DenseIndex(3, 7, 1)),
            x,
        )

    def bst_4d_2_val_sparse_and_dense(self):
        x = jnp.arange(3 * 4 * 6 * 7).reshape(3, 4, 6, 7)
        return SparseTensor(
            (SparseIndex(0, 3, 0, 2, 4, 1), DenseIndex(1, 5, None)),
            (SparseIndex(2, 3, 0, 0, 6, 2), DenseIndex(3, 7, 3)),
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

    def test_block_diagonal_transpose(self):
        stc = self.bst_3d_val_dense()
        d1 = _dense(stc, hard=True).dense().T
        d2 = _dense(stc.T, hard=True).dense()
        self.assertTrue(jnp.allclose(d1, d2))

    def test_transpose_one_sparse(self):
        stb = self.bst_2d()
        # need some val to check shape
        stb = _dense(stb, hard=False)
        n, x, y = stb.val.shape
        self.assertEqual(stb.T.val.shape, (n, y, x))
        self.assertTrue(jnp.allclose(stb.dense().T, stb.T.dense()))

    def test_transpose_mixed(self):
        stc = self.bst_4d_2_val_sparse_and_dense()
        n, x, a, y = stc.val.shape
        self.assertEqual(stc.T.val.shape, (n, y, a, x))
        self.assertTrue(jnp.allclose(stc.dense().T, stc.T.dense()))

    def test_transpose_two_sparse(self):
        ste = SparseTensor(
            (SparseIndex(0, 3, 0, 2, 5, 2), SparseIndex(1, 4, 1, 3, 6, 3)),
            (SparseIndex(2, 3, 0, 0, 7, 4), SparseIndex(3, 4, 1, 1, 8, 5)),
            jnp.ones((3, 4, 5, 6, 7, 8)),
        )
        n, m, x, a, y, b = ste.val.shape
        self.assertEqual(ste.T.val.shape, (m, n, b, y, a, x))
        self.assertTrue(jnp.allclose(ste.dense().T, ste.T.dense()))

    def test_transpose_invariance(self):
        for fixture_fn in self.get_all_fixtures():
            with self.subTest(fixture=fixture_fn.__name__):
                st = fixture_fn()
                st_tt = st.T.T
                self.assertEqual(st.shape, st_tt.shape)
                self.assertEqual(len(st.dims), len(st_tt.dims))
                self.assertTrue(jnp.allclose(st.dense(), st_tt.dense()))

    def test_dense_transpose(self):
        for fixture_fn in self.get_all_fixtures():
            with self.subTest(fixture=fixture_fn.__name__):
                st = fixture_fn()
                arr = st.dense()
                arr_t = st.T.dense()
                self.assertTrue(jnp.allclose(arr.T, arr_t))

    def test_matmul_transpose(self):
        for fixture_fn in self.get_all_fixtures():
            with self.subTest(fixture=fixture_fn.__name__):
                st = fixture_fn()
                if len(st.primal_dims) <= 1:
                    res = st @ st.T
                    expected = matmul_reference(st, st.T)
                    self.assertTrue(jnp.allclose(res.dense(), expected.dense()))
                res = st @ st.swapdims()
                expected = matmul_reference(st, st.swapdims())
                self.assertTrue(jnp.allclose(res.dense(), expected.dense()))

    def test_2d_block_consistency(self):
        stb_none = self.bst_2d()  # Has val=None

        # Build an equivalent explicit tensor with val=ones dynamically mapped
        dims = []
        val_shape = []
        v_idx = 0
        seen_sparse = {}

        for d in stb_none.out_dims + stb_none.primal_dims:
            if isinstance(d, SparseIndex):
                pair_key = tuple(sorted((d.id, d.other_id)))
                if pair_key not in seen_sparse:
                    seen_sparse[pair_key] = v_idx
                    val_shape.append(d.size)
                    v_idx += 1

                vd = seen_sparse[pair_key]
                bd = None
                if d.block_size:
                    bd = v_idx
                    val_shape.append(d.block_size)
                    v_idx += 1
                dims.append(replace(d, axis=vd, block_axis=bd))
            else:
                dims.append(replace(d, axis=v_idx))
                val_shape.append(d.size)
                v_idx += 1

        out_dims = dims[: len(stb_none.out_dims)]
        primal_dims = dims[len(stb_none.out_dims) :]

        stb_ones = SparseTensor(out_dims, primal_dims, jnp.ones(val_shape))

        self.assertTrue(jnp.allclose(stb_none.dense(), stb_ones.dense()))


if __name__ == "__main__":
    unittest.main()
