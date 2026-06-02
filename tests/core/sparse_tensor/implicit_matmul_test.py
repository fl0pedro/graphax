import unittest
import jax
import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.indexes import DiagonalIndex, DenseIndex
from graphax.sparse.tensor import SparseTensor, _arr2st
from graphax.sparse.ops.matmul import matmul


def dense_ref(A_full, B_full):
    return jax.lax.dot_general(A_full, B_full, (((2,), (1,)), ((0,), (0,))))


class TestImplicit(unittest.TestCase):
    def setUp(self):
        self.rng_key = jr.PRNGKey(42)

    # All have idxs 0, 1, 3 instead of 0, 1, 2...
    def test_implicit_a_b_c(self):
        rng_key = self.rng_key
        # ([a], b, c) @ (a, c, d)
        a, b, c, d = 2, 3, 4, 5
        x, y = 0.24, 1.32
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (b, c))
        B = jr.normal(k2, (a, c, d))

        A_st = SparseTensor(
            (DenseIndex(0, a, None), DenseIndex(1, b, 0)),
            (DenseIndex(2, c, 1),),
            A,
            scalar_mult=jnp.array(x),
        )
        B_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, c, 1)),
            (DenseIndex(2, d, 2),),
            B,
            scalar_mult=jnp.array(y),
        )

        A_dense = jnp.broadcast_to(A, (a, b, c))
        R_dense_ref = dense_ref(A_dense * x, B * y)

        # Manual val logic
        R = jax.lax.dot_general(A, B, (((1,), (1,)), ((), ()))).transpose(1, 0, 2)
        R_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, b, 1)),
            (DenseIndex(2, d, 2),),
            R,
            scalar_mult=jnp.array(x * y),
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(R_ref.dense(), R_st.dense())
        assert (R_st == R_ref).all()
        assert jnp.allclose(R * (x * y), R_ref.val * R_ref.scalar_mult)

    def test_a_implicit_b_c(self):
        rng_key = self.rng_key
        # (a, [b], c) @ (a, c, d)
        a, b, c, d = 2, 3, 4, 5
        x, y = 0.24, 1.32
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a, c))
        B = jr.normal(k2, (a, c, d))

        A_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, b, None)),
            (DenseIndex(2, c, 1),),
            A,
            scalar_mult=jnp.array(x),
        )
        B_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, c, 1)),
            (DenseIndex(2, d, 2),),
            B,
            scalar_mult=jnp.array(y),
        )

        A_dense = jnp.broadcast_to(jnp.expand_dims(A, 1), (a, b, c))
        R_dense_ref = dense_ref(A_dense * x, B * y)

        # Manual val logic
        R = jax.lax.dot_general(A, B, (((1,), (1,)), ((0,), (0,))))
        R_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, b, None)),
            (DenseIndex(2, d, 1),),
            R,
            scalar_mult=jnp.array(x * y),
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        # TODO create these operations for calculating jnp.allclose: absolute(a - b) <= (atol + rtol * absolute(b))
        assert jnp.allclose(R_ref.dense(), R_st.dense())
        # assert (R_st == R_ref).all()
        assert jnp.allclose(R_st.val, R_ref.val)
        assert jnp.allclose(R * (x * y), R_ref.val * R_ref.scalar_mult)

    def test_a_b_implicit_c(self):
        rng_key = self.rng_key
        # (a, b, [c]) @ (a, c, d)
        a, b, c, d = 2, 3, 4, 5
        x, y = 0.24, 1.32
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a, b))
        B = jr.normal(k2, (a, c, d))

        A_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, b, 1)),
            (DenseIndex(2, c, None),),
            A,
            scalar_mult=jnp.array(x),
        )
        B_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, c, 1)),
            (DenseIndex(2, d, 2),),
            B,
            scalar_mult=jnp.array(y),
        )

        A_dense = jnp.broadcast_to(jnp.expand_dims(A, 2), (a, b, c))
        R_dense_ref = dense_ref(A_dense * x, B * y)

        # Manual val logic
        R = jnp.expand_dims(A, 2) * jnp.expand_dims(B.sum(axis=1), 1)
        R_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, b, 1)),
            (DenseIndex(2, d, 2),),
            R,
            scalar_mult=jnp.array(x * y),
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(R_ref.dense(), R_st.dense())
        # assert (R_st == R_ref).all()
        assert jnp.allclose(R_st.val, R_ref.val)
        assert jnp.allclose(R * (x * y), R_ref.val * R_ref.scalar_mult)

    def test_a_b_c_implicit_c_d(self):
        rng_key = self.rng_key
        # (a, b, c) @ (a, [c], d)
        a, b, c, d = 2, 3, 4, 5
        x, y = 0.24, 1.32
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a, b, c))
        B = jr.normal(k2, (a, d))

        A_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, b, 1)),
            (DenseIndex(2, c, 2),),
            A,
            scalar_mult=jnp.array(x),
        )
        B_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, c, None)),
            (DenseIndex(2, d, 1),),
            B,
            scalar_mult=jnp.array(y),
        )

        B_dense = jnp.broadcast_to(jnp.expand_dims(B, 1), (a, c, d))
        R_dense_ref = dense_ref(A * x, B_dense * y)

        # Manual val logic
        R = jnp.expand_dims(A.sum(axis=2), 2) * jnp.expand_dims(B, 1)
        R_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, b, 1)),
            (DenseIndex(2, d, 2),),
            R,
            scalar_mult=jnp.array(x * y),
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(R_ref.dense(), R_st.dense())
        # assert (R_st == R_ref).all()
        assert jnp.allclose(R_st.val, R_ref.val)
        assert jnp.allclose(R * (x * y), R_ref.val * R_ref.scalar_mult)

    def test_implicit_a_b_c_a_c_d(self):
        rng_key = self.rng_key
        # ([a], [b], c) @ (a, c, d)
        a, b, c, d = 2, 3, 4, 5
        x, y = 0.24, 1.32
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (c,))
        B = jr.normal(k2, (a, c, d))

        A_st = SparseTensor(
            (DenseIndex(0, a, None), DenseIndex(1, b, None)),
            (DenseIndex(2, c, 0),),
            A,
            scalar_mult=jnp.array(x),
        )
        B_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, c, 1)),
            (DenseIndex(2, d, 2),),
            B,
            scalar_mult=jnp.array(y),
        )

        A_dense = jnp.broadcast_to(A, (a, b, c))
        R_dense_ref = dense_ref(A_dense * x, B * y)

        # Manual val logic
        R = jax.lax.dot_general(A, B, (((0,), (1,)), ((), ())))
        R_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, b, None)),
            (DenseIndex(2, d, 1),),
            R,
            scalar_mult=jnp.array(x * y),
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(R_ref.dense(), R_st.dense())
        assert (R_st == R_ref).all()
        assert jnp.allclose(R * (x * y), R_ref.val * R_ref.scalar_mult)

    def test_all_implicit_a_c_d(self):
        rng_key = self.rng_key
        # ([a], [b], [c]) @ (a, [c], [d])
        a, b, c, d = 2, 3, 4, 5
        x, y = 0.24, 1.32
        _, k1, k2 = jr.split(rng_key, 3)

        B = jr.normal(k2, (a,))

        A_st = SparseTensor(
            (DenseIndex(0, a, None), DenseIndex(1, b, None)),
            (DenseIndex(2, c, None),),
            None,
            scalar_mult=jnp.array(x),
        )
        B_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, c, None)),
            (DenseIndex(2, d, None),),
            B,
            scalar_mult=jnp.array(y),
        )

        A_dense = jnp.ones((a, b, c))
        B_dense = jnp.broadcast_to(jnp.expand_dims(B, (1, 2)), (a, c, d))
        R_dense_ref = dense_ref(A_dense * x, B_dense * y)

        # Manual val logic
        R = B
        R_st = SparseTensor(
            (DenseIndex(0, a, 0), DenseIndex(1, b, None)),
            (DenseIndex(2, d, None),),
            R,
            scalar_mult=jnp.array(c * x * y),
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(R_ref.dense(), R_st.dense())
        assert (R_st == R_ref).all()
        assert jnp.allclose(R * (c * x * y), R_ref.val * R_ref.scalar_mult)

    def test_all_implicit_all_implicit(self):
        # ([a], [b], [c]) @ ([a], [c], [d])
        a, b, c, d = 2, 3, 4, 5
        x, y = 0.24, 1.32

        A_st = SparseTensor(
            (DenseIndex(0, a, None), DenseIndex(1, b, None)),
            (DenseIndex(2, c, None),),
            None,
            scalar_mult=jnp.array(x),
        )
        B_st = SparseTensor(
            (DenseIndex(0, a, None), DenseIndex(1, c, None)),
            (DenseIndex(2, d, None),),
            None,
            scalar_mult=jnp.array(y),
        )

        A_dense = jnp.ones((a, b, c))
        B_dense = jnp.ones((a, c, d))
        R_dense_ref = dense_ref(A_dense * x, B_dense * y)

        R_st = SparseTensor(
            (DenseIndex(0, a, None), DenseIndex(1, b, None)),
            (DenseIndex(2, d, None),),
            None,
            scalar_mult=jnp.array(c * x * y),
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(R_ref.dense(), R_st.dense())
        assert (R_st == R_ref).all()
        assert jnp.allclose(c * x * y, R_ref.scalar_mult)
        assert R_ref.val is None

    def test_matmul_implicit_value(self):
        """Target cases with unmaterialized (None) values."""
        st1 = _arr2st(jnp.ones((2, 2)))
        st1.val = None
        st2 = _arr2st(jnp.ones((2, 2)))
        res = matmul(st1, st2)
        self.assertTrue(res.val is not None)


if __name__ == "__main__":
    unittest.main()
