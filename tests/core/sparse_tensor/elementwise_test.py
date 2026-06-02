import os
import unittest
import operator
import jax.numpy as jnp
import jax.random as jrand
from utils import generate_tensors, idfn
from graphax.sparse.indexes import DiagonalIndex, DenseIndex
from graphax.sparse.tensor import SparseTensor
from graphax.sparse.ops.elementwise import elementwise, _arr2st

EXHAUSTIVE = os.getenv("EXHAUSTIVE", "0") == "1"


def get_valid_elementwise_pairs(blocks_a, blocks_b, ndim):
    tensors_a = list(generate_tensors(blocks_a, ndim))
    tensors_b = list(generate_tensors(blocks_b, ndim))

    pairs = []
    for ta in tensors_a:
        for tb in tensors_b:
            if ta.shape == tb.shape:
                pairs.append((ta, tb))
    return pairs


def generate_elementwise_cases(
    seed, n_blocks_a, base_shape_a, n_blocks_b, base_shape_b
):
    cases = []
    for ndim in [2, 3]:
        key = jrand.PRNGKey(seed)
        k1, k2 = jrand.split(key)

        shape_a = (base_shape_a,) * ndim
        shape_b = (base_shape_b,) * ndim

        blocks_a = jrand.normal(k1, (n_blocks_a,) + shape_a)
        blocks_b = jrand.normal(k2, (n_blocks_b,) + shape_b)

        for ta, tb in get_valid_elementwise_pairs(blocks_a, blocks_b, ndim):
            cases.append((ndim, ta, tb))
    return cases


class TestElementwise(unittest.TestCase):
    def _run_elementwise_test(self, ndim, ta, tb, op):
        res_sparse = op(ta, tb)
        res_dense_ref = op(ta.dense(), tb.dense())
        self.assertTrue(jnp.allclose(res_sparse.dense(), res_dense_ref, atol=1e-5))

    @unittest.skipUnless(EXHAUSTIVE, "set EXHAUSTIVE=1 to run exhaustive sweeps")
    def test_exhaustive_aligned_blocks(self):
        cases = generate_elementwise_cases(
            seed=0, n_blocks_a=4, base_shape_a=2, n_blocks_b=4, base_shape_b=2
        )
        for ndim, ta, tb in cases:
            with self.subTest(ndim=ndim, op="add", ta=idfn(ta), tb=idfn(tb)):
                self._run_elementwise_test(ndim, ta, tb, operator.add)
            with self.subTest(ndim=ndim, op="mul", ta=idfn(ta), tb=idfn(tb)):
                self._run_elementwise_test(ndim, ta, tb, operator.mul)

    @unittest.skipUnless(EXHAUSTIVE, "set EXHAUSTIVE=1 to run exhaustive sweeps")
    def test_exhaustive_perfect_intersect_blocks(self):
        cases = generate_elementwise_cases(
            seed=1, n_blocks_a=4, base_shape_a=2, n_blocks_b=2, base_shape_b=4
        )
        for ndim, ta, tb in cases:
            with self.subTest(ndim=ndim, op="add", ta=idfn(ta), tb=idfn(tb)):
                self._run_elementwise_test(ndim, ta, tb, operator.add)
            with self.subTest(ndim=ndim, op="mul", ta=idfn(ta), tb=idfn(tb)):
                self._run_elementwise_test(ndim, ta, tb, operator.mul)

    @unittest.skipUnless(EXHAUSTIVE, "set EXHAUSTIVE=1 to run exhaustive sweeps")
    def test_exhaustive_unaligned_diff_blocks(self):
        cases = generate_elementwise_cases(
            seed=2, n_blocks_a=4, base_shape_a=3, n_blocks_b=3, base_shape_b=4
        )
        for ndim, ta, tb in cases:
            with self.subTest(ndim=ndim, op="add", ta=idfn(ta), tb=idfn(tb)):
                self._run_elementwise_test(ndim, ta, tb, operator.add)
            with self.subTest(ndim=ndim, op="mul", ta=idfn(ta), tb=idfn(tb)):
                self._run_elementwise_test(ndim, ta, tb, operator.mul)

    def test_aligned_blocks(self):
        d0 = DiagonalIndex(0, 2, axis=0, other_id=1, block_size=2, block_axis=1)
        d1 = DiagonalIndex(1, 2, axis=0, other_id=0, block_size=2, block_axis=2)

        val_a = jnp.ones((2, 2, 2))
        val_b = jnp.full((2, 2, 2), 2.0)

        ta = SparseTensor((d0,), (d1,), val_a)
        tb = SparseTensor((d0,), (d1,), val_b)

        expected_val_add = val_a + val_b
        expected_val_mul = val_a * val_b

        self.assertTrue(
            jnp.allclose(ta.copy(val=expected_val_add).dense(), (ta + tb).dense())
        )
        self.assertTrue(
            jnp.allclose(ta.copy(val=expected_val_mul).dense(), (ta * tb).dense())
        )

    def test_perfect_intersect_blocks(self):
        d0_a = DiagonalIndex(
            0, 1, axis=0, other_id=1, block_size=4, block_axis=1
        )
        d1_a = DiagonalIndex(
            1, 1, axis=0, other_id=0, block_size=4, block_axis=2
        )
        val_a = jnp.arange(16.0).reshape(1, 4, 4)
        ta = SparseTensor((d0_a,), (d1_a,), val_a)

        d0_b = DiagonalIndex(
            0, 2, axis=0, other_id=1, block_size=2, block_axis=1
        )
        d1_b = DiagonalIndex(
            1, 2, axis=0, other_id=0, block_size=2, block_axis=2
        )
        val_b = jnp.ones((2, 2, 2))
        tb = SparseTensor((d0_b,), (d1_b,), val_b)

        expected_val_add = val_a.copy()
        expected_val_add = expected_val_add.at[0, 0:2, 0:2].add(val_b[0])
        expected_val_add = expected_val_add.at[0, 2:4, 2:4].add(val_b[1])

        expected_val_mul = jnp.zeros_like(val_b)
        expected_val_mul = expected_val_mul.at[0].set(val_a[0, 0:2, 0:2] * val_b[0])
        expected_val_mul = expected_val_mul.at[1].set(val_a[0, 2:4, 2:4] * val_b[1])

        self.assertTrue(jnp.allclose(expected_val_add.reshape(4, 4), (ta + tb).dense()))

        dense_mul_ref = jnp.zeros((4, 4))
        dense_mul_ref = dense_mul_ref.at[0:2, 0:2].set(expected_val_mul[0])
        dense_mul_ref = dense_mul_ref.at[2:4, 2:4].set(expected_val_mul[1])
        self.assertTrue(jnp.allclose(dense_mul_ref, (ta * tb).dense()))

    def test_unaligned_diff_blocks(self):
        d0_a = DiagonalIndex(
            0, 2, axis=0, other_id=1, block_size=3, block_axis=1
        )
        d1_a = DiagonalIndex(
            1, 2, axis=0, other_id=0, block_size=3, block_axis=2
        )
        val_a = jnp.ones((2, 3, 3)) * 2
        ta = SparseTensor((d0_a,), (d1_a,), val_a)

        d0_b = DiagonalIndex(
            0, 3, axis=0, other_id=1, block_size=2, block_axis=1
        )
        d1_b = DiagonalIndex(
            1, 3, axis=0, other_id=0, block_size=2, block_axis=2
        )
        val_b = jnp.ones((3, 2, 2)) * 3
        tb = SparseTensor((d0_b,), (d1_b,), val_b)

        expected_val_mul_blocks = jnp.zeros((6, 6))
        expected_val_mul_blocks = expected_val_mul_blocks.at[0:2, 0:2].set(6.0)
        expected_val_mul_blocks = expected_val_mul_blocks.at[2, 2].set(6.0)
        expected_val_mul_blocks = expected_val_mul_blocks.at[3, 3].set(6.0)
        expected_val_mul_blocks = expected_val_mul_blocks.at[4:6, 4:6].set(6.0)

        expected_val_add_dense = jnp.zeros((6, 6))
        for i in range(2):
            expected_val_add_dense = expected_val_add_dense.at[
                i * 3 : i * 3 + 3, i * 3 : i * 3 + 3
            ].add(2.0)
        for i in range(3):
            expected_val_add_dense = expected_val_add_dense.at[
                i * 2 : i * 2 + 2, i * 2 : i * 2 + 2
            ].add(3.0)

        self.assertTrue(jnp.allclose(expected_val_mul_blocks, (ta * tb).dense()))
        self.assertTrue(jnp.allclose(expected_val_add_dense, (ta + tb).dense()))

    def test_elementwise_shape_mismatch(self):
        st1 = _arr2st(jnp.arange(6).reshape(2, 3))
        st2 = _arr2st(jnp.arange(8).reshape(2, 4))
        with self.assertRaises(ValueError) as cm:
            elementwise(st1, st2, jnp.add)
        self.assertIn("Shape mismatch", str(cm.exception))

    def test_elementwise_dense_lhs(self):
        st_rhs = _arr2st(jnp.arange(4).reshape(2, 2))
        lhs_dense = jnp.arange(4).reshape(2, 2)
        res = elementwise(lhs_dense, st_rhs, jnp.add)
        self.assertTrue(jnp.allclose(res.dense(), lhs_dense + st_rhs.dense()))

    def test_elementwise_dense_rhs(self):
        st_lhs = _arr2st(jnp.arange(4).reshape(2, 2))
        rhs_dense = jnp.arange(4).reshape(2, 2)
        res = elementwise(st_lhs, rhs_dense, jnp.add)
        self.assertTrue(jnp.allclose(res.dense(), st_lhs.dense() + rhs_dense))

    def test_elementwise_intersection_summing(self):
        d0_l = DiagonalIndex(
            0, 4, axis=0, other_id=1, block_size=2, block_axis=1
        )
        d1_l = DiagonalIndex(
            1, 4, axis=0, other_id=0, block_size=2, block_axis=2
        )
        st_l = SparseTensor((d0_l,), (d1_l,), jnp.arange(16).reshape(4, 2, 2))

        d0_r = DiagonalIndex(
            0, 2, axis=0, other_id=1, block_size=4, block_axis=1
        )
        d1_r = DiagonalIndex(
            1, 2, axis=0, other_id=0, block_size=4, block_axis=2
        )
        st_r = SparseTensor((d0_r,), (d1_r,), jnp.arange(32).reshape(2, 4, 4))

        res = elementwise(st_l, st_r, jnp.add, is_intersection=True)
        self.assertEqual(res.shape, (8, 8))


if __name__ == "__main__":
    unittest.main()
