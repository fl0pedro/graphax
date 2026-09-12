import os
import unittest
import jax
import jax.numpy as jnp
import jax.random as jr

# Layout pins compare against HAND-BUILT tiled-layout references. There is one
# engine now (the planner was deleted 2026-09-08, ticket dsnn-3qm.72), so the
# pins always hold. The flag stays as a named constant because 18 asserts read
# it and it documents which of them are layout pins rather than value checks.
_PIN_LAYOUT = True

def _engine_reduce(view, flat_idx, num_segments):
    """Mirror of the engine's _reduce_grid reduction (#51): constant one-hot
    dot with the same <=4M valve, so the _PIN_LAYOUT byte-identity asserts
    keep comparing like-for-like emissions. Accepts (src, *trailing)."""
    import numpy as _np
    flat_idx = _np.asarray(flat_idx)
    n_src = int(view.shape[0])
    if n_src * int(num_segments) <= 4_000_000:
        oh = _np.zeros((int(num_segments), n_src), dtype=_np.float32)
        oh[flat_idx, _np.arange(n_src)] = 1.0
        v2 = view.reshape(n_src, -1)
        res = jnp.einsum(
            jnp.asarray(oh, dtype=v2.dtype), [0, 1], v2, [1, 2], [0, 2])
        return res.reshape((int(num_segments),) + tuple(view.shape[1:]))
    return jax.ops.segment_sum(view, jnp.asarray(flat_idx),
                               num_segments=num_segments)


from graphax.sparse.indexes import DiagonalIndex, DenseIndex
from graphax.sparse.tensor import SparseTensor, _arr2st
import math


def assert_meta_grid_storage(R_ref, dense_ref, support, stored):  # noqa: D401
    """Storage contract of a misaligned-contract matmul (ticket dsnn-3qm.28.5).

    ``support`` is the number of structurally non-zero entries of the dense
    product. ``stored`` is the element count the engine asks for: the meta grid
    at the least common multiple of the two operands' block sizes, coarsened to
    ``gcd`` many meta blocks.

    The BandedIndex form used to store exactly ``support``, the smallest honest
    buffer. That class is gone (ruling 2026-09-07: SparseTensor has exactly two
    index classes, DenseIndex and DiagonalIndex), so the meta grid is the
    fallback and it stores more. Both numbers are hand-written; neither is read
    off the result."""
    got = 0 if R_ref.val is None else int(R_ref.val.size)
    ref_support = int(jnp.sum(jnp.abs(dense_ref) > 1e-9))
    assert ref_support == support, (
        f"the dense product has {ref_support} non-zeros, not the {support} "
        f"this test claims")
    assert got == stored, (
        f"stored {got} != the {stored}-element meta grid; the misaligned "
        f"contraction changed how much it materialises")
    assert stored >= support


def get_routing_idx(b, d, l, g):
    k = jnp.arange(l)
    w = k // (l // g)
    i = (k // (l // b)) % (b // g)
    j = (k // (l // d)) % (d // g)
    idx = w * ((b // g) * (d // g)) + i * (d // g) + j
    num_blocks = g * (b // g) * (d // g)
    return idx, num_blocks


class TestExplicit(unittest.TestCase):
    def setUp(self):
        self.rng_key = jr.PRNGKey(42)

    def test_pure_pure(self):
        rng_key = self.rng_key
        a = 4
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a,))
        B = jr.normal(k2, (a,))

        A_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1),), (DiagonalIndex(1, a, 0, 0),), A
        )
        B_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1),), (DiagonalIndex(1, a, 0, 0),), B
        )

        R = A * B

        R_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1),), (DiagonalIndex(1, a, 0, 0),), R
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_dense_dense(self):
        rng_key = self.rng_key
        a, b, c = 2, 3, 4
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a, b))
        B = jr.normal(k2, (b, c))

        A_st = _arr2st(A)
        B_st = _arr2st(B)

        R = jax.lax.dot_general(A, B, (((1,), (0,)), ((), ())))

        R_st = _arr2st(R)

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_pure_dense(self):
        rng_key = self.rng_key
        a, b = 3, 4
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a,))
        B = jr.normal(k2, (a, b))

        A_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1),), (DiagonalIndex(1, a, 0, 0),), A
        )
        B_st = _arr2st(B)

        R = jnp.expand_dims(A, 1) * B

        R_st = _arr2st(R)

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_dense_pure(self):
        rng_key = self.rng_key
        a, b = 3, 4
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a, b))
        B = jr.normal(k2, (b,))

        A_st = _arr2st(A)
        B_st = SparseTensor(
            (DiagonalIndex(0, b, 0, 1),), (DiagonalIndex(1, b, 0, 0),), B
        )

        R = A * jnp.expand_dims(B, 0)

        R_st = _arr2st(R)

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_block_block_equal(self):
        rng_key = self.rng_key
        a, b, c, d = 2, 3, 4, 5
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a, b, c))
        B = jr.normal(k2, (a, c, d))

        A_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, b, 1),),
            (DiagonalIndex(1, a, 0, 0, c, 2),),
            A,
        )
        B_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, c, 1),),
            (DiagonalIndex(1, a, 0, 0, d, 2),),
            B,
        )

        R = jax.lax.dot_general(A, B, (((2,), (1,)), ((0,), (0,))))

        R_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, b, 1),),
            (DiagonalIndex(1, a, 0, 0, d, 2),),
            R,
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_block_dense(self):
        rng_key = self.rng_key
        a, b, c, d = 2, 3, 4, 5
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a, b, c))
        B = jr.normal(k2, (a * c, d))

        A_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, b, 1),),
            (DiagonalIndex(1, a, 0, 0, c, 2),),
            A,
        )
        B_st = _arr2st(B)

        B_prime = B.reshape(a, c, d)
        R_inter = jax.lax.dot_general(A, B_prime, (((2,), (1,)), ((0,), (0,))))
        R = R_inter.reshape(a * b, d)

        R_st = SparseTensor(
            (DenseIndex(0, a * b, 0),), (DenseIndex(1, d, 1),), R
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_dense_block(self):
        rng_key = self.rng_key
        a, b, c, d = 2, 3, 4, 5
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a, b * c))
        B = jr.normal(k2, (b, c, d))

        A_st = _arr2st(A)
        B_st = SparseTensor(
            (DiagonalIndex(0, b, 0, 1, c, 1),),
            (DiagonalIndex(1, b, 0, 0, d, 2),),
            B,
        )

        A_prime = A.reshape(a, b, c)
        R_inter = jax.lax.dot_general(A_prime, B, (((2,), (1,)), ((1,), (0,))))
        R = R_inter.transpose(1, 0, 2).reshape(a, b * d)

        R_st = SparseTensor(
            (DenseIndex(0, a, 0),), (DenseIndex(1, b * d, 1),), R
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_block_pure(self):
        rng_key = self.rng_key
        a, b, c = 2, 3, 4
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a, b, c))
        B = jr.normal(k2, (a * c,))

        A_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, b, 1),),
            (DiagonalIndex(1, a, 0, 0, c, 2),),
            A,
        )
        B_st = SparseTensor(
            (DiagonalIndex(0, a * c, 0, 1),), (DiagonalIndex(1, a * c, 0, 0),), B
        )

        B_prime = B.reshape(a, c)
        R = A * jnp.expand_dims(B_prime, 1)  # broadcast_mul

        R_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, b, 1),),
            (DiagonalIndex(1, a, 0, 0, c, 2),),
            R,
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_pure_block(self):
        rng_key = self.rng_key
        a, b, c = 2, 3, 4
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a * b,))
        B = jr.normal(k2, (a, b, c))

        A_st = SparseTensor(
            (DiagonalIndex(0, a * b, 0, 1),), (DiagonalIndex(1, a * b, 0, 0),), A
        )
        B_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, b, 1),),
            (DiagonalIndex(1, a, 0, 0, c, 2),),
            B,
        )

        A_prime = A.reshape(a, b)
        R = jnp.expand_dims(A_prime, 2) * B  # broadcast_mul

        R_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, b, 1),),
            (DiagonalIndex(1, a, 0, 0, c, 2),),
            R,
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_block_block_factor_a(self):
        rng_key = self.rng_key
        a, b, c, d, e, f = 2, 3, 4, 5, 6, 7
        k = e // b  # 2
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a, d, e))
        B = jr.normal(k2, (c, b, f))

        A_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, d, 1),),
            (DiagonalIndex(1, a, 0, 0, e, 2),),
            A,
        )
        B_st = SparseTensor(
            (DiagonalIndex(0, c, 0, 1, b, 1),),
            (DiagonalIndex(1, c, 0, 0, f, 2),),
            B,
        )

        A_prime = A.reshape(a, d, k, b).transpose(0, 2, 1, 3).reshape(c, d, b)
        R_inter = jax.lax.dot_general(A_prime, B, (((2,), (1,)), ((0,), (0,))))
        R = R_inter.reshape(a, k, d, f).transpose(0, 2, 1, 3).reshape(a, d, k * f)

        R_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, d, 1),),
            (DiagonalIndex(1, a, 0, 0, k * f, 2),),
            R,
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_block_block_factor_b(self):
        rng_key = self.rng_key
        a, b, c, d, e, f = 4, 6, 2, 5, 3, 7
        k = b // e  # 2
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a, d, e))
        B = jr.normal(k2, (c, b, f))

        A_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, d, 1),),
            (DiagonalIndex(1, a, 0, 0, e, 2),),
            A,
        )
        B_st = SparseTensor(
            (DiagonalIndex(0, c, 0, 1, b, 1),),
            (DiagonalIndex(1, c, 0, 0, f, 2),),
            B,
        )

        B_prime = B.reshape(a, e, f)
        R_inter = jax.lax.dot_general(A, B_prime, (((2,), (1,)), ((0,), (0,))))
        R = R_inter.reshape(c, k * d, f)

        R_st = SparseTensor(
            (DiagonalIndex(0, c, 0, 1, k * d, 1),),
            (DiagonalIndex(1, c, 0, 0, f, 2),),
            R,
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_block_block_gcd(self):
        rng_key = self.rng_key
        a, b, c, d, e, f = 4, 6, 2, 5, 3, 7
        k = math.gcd(a, b)
        l = math.lcm(a, b)
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a, d, e))
        B = jr.normal(k2, (b, c, f))

        A_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, d, 1),),
            (DiagonalIndex(1, a, 0, 0, e, 2),),
            A,
        )
        B_st = SparseTensor(
            (DiagonalIndex(0, b, 0, 1, c, 1),),
            (DiagonalIndex(1, b, 0, 0, f, 2),),
            B,
        )

        A_view = (
            A.reshape(a, d, l // a, e // (l // a))
            .transpose(0, 2, 1, 3)
            .reshape(l, d, -1)
        )
        B_view = B.reshape(b, l // b, c // (l // b), f).reshape(l, -1, f)

        C_view = jax.lax.dot_general(A_view, B_view, (((2,), (1,)), ((0,), (0,))))

        flat_idx, num_seg = get_routing_idx(a, b, l, k)
        C_reduced = _engine_reduce(
            C_view.reshape(l, -1), flat_idx, num_seg
        )

        R = (
            C_reduced.reshape(k, a // k, b // k, d, f)
            .transpose(0, 1, 3, 2, 4)
            .reshape((k, (a // k) * d, (b // k) * f))
        )

        R_st = SparseTensor(
            (DiagonalIndex(0, k, 0, 1, (a // k) * d, 1),),
            (DiagonalIndex(1, k, 0, 0, (b // k) * f, 2),),
            R,
        )

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        # gcd(4, 6) = 2 meta blocks of (4/2)*5 = 10 rows by (6/2)*7 = 21
        # cols, so 2 * 10 * 21 = 420. The band form stored the 280-element
        # support exactly; that class is gone, so the grid stores 1.50x it.
        assert_meta_grid_storage(R_ref, R_st.dense(), 280, 420)

        # Compare densified forms.
        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )

    def test_block_block_coprime_a_greater_b(self):
        rng_key = self.rng_key
        a, b, c, d, e, f = 2, 3, 4, 5, 6, 7
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (a, d, e))
        B = jr.normal(k2, (b, c, f))

        A_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, d, 1),),
            (DiagonalIndex(1, a, 0, 0, e, 2),),
            A,
        )
        B_st = SparseTensor(
            (DiagonalIndex(0, b, 0, 1, c, 1),),
            (DiagonalIndex(1, b, 0, 0, f, 2),),
            B,
        )

        A_dense = A_st.dense()
        A_prime = A_dense.reshape(a * d, b, c)

        R_inter = jax.lax.dot_general(A_prime, B, (((2,), (1,)), ((1,), (0,))))
        R = R_inter.transpose(1, 0, 2).reshape(a * d, b * f)

        R_st = _arr2st(R)

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        # gcd(2, 3) = 1, so the grid is a single 10 x 21 = 210 meta block.
        # The band form stored the 140-element support exactly; that class
        # is gone, so the grid stores 1.50x it.
        assert_meta_grid_storage(R_ref, R_st.dense(), 140, 210)

        # Compare densified forms; structural shape / dim count is still
        # preserved.
        assert jnp.allclose(
            R_st.dense(), R_ref.dense(), rtol=1e-4, atol=1e-6
        )
        assert R_st.shape == R_ref.shape
        assert len(R_st.dims) == len(R_ref.dims)

    def test_block_block_coprime_a_less_b(self):
        rng_key = self.rng_key
        a, b, c, d, e, f = 2, 3, 4, 5, 6, 7
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (b, f, c))
        B = jr.normal(k2, (a, e, d))

        A_st = SparseTensor(
            (DiagonalIndex(0, b, 0, 1, f, 1),),
            (DiagonalIndex(1, b, 0, 0, c, 2),),
            A,
        )
        B_st = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, e, 1),),
            (DiagonalIndex(1, a, 0, 0, d, 2),),
            B,
        )

        B_dense = B_st.dense()
        B_prime = B_dense.reshape(b, c, a * d)

        R_inter = jax.lax.dot_general(A, B_prime, (((2,), (1,)), ((0,), (0,))))
        R = R_inter.reshape(b * f, a * d)

        R_st = _arr2st(R)

        assert jnp.allclose(R_st.dense(), A_st.dense() @ B_st.dense())

        R_ref = A_st @ B_st

        # gcd(2, 3) = 1, so the grid is a single 10 x 21 = 210 meta block.
        # The band form stored the 140-element support exactly; that class
        # is gone, so the grid stores 1.50x it.
        assert_meta_grid_storage(R_ref, R_st.dense(), 140, 210)

        # Compare densified forms; structural shape / dim count is still
        # preserved.
        assert jnp.allclose(
            R_st.dense(), R_ref.dense(), rtol=1e-4, atol=1e-6
        )
        assert R_st.shape == R_ref.shape
        assert len(R_st.dims) == len(R_ref.dims)

    def test_pure_block_dense_pure(self):
        rng_key = self.rng_key
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (4, 9, 3))
        B = jr.normal(k2, (2, 9, 2))

        A_st = SparseTensor(
            (DiagonalIndex(0, 4, 0, 1, 3, 2),),
            (
                DiagonalIndex(1, 4, 0, 0),
                DenseIndex(2, 9, 1),
            ),
            A,
        )

        B_st = SparseTensor(
            (
                DiagonalIndex(0, 2, 0, 2, 2, 2),
                DiagonalIndex(1, 9, 1, 3),
            ),
            (
                DiagonalIndex(2, 2, 0, 0),
                DiagonalIndex(3, 9, 1, 1),
            ),
            B,
        )

        A_prime = A.reshape(2, 2, 9, 3)
        B_prime = B.transpose(0, 2, 1)

        R_inter = A_prime * jnp.expand_dims(B_prime, -1)

        R = R_inter.transpose(0, 1, 3, 2).reshape(2, 6, 9)

        R_st = SparseTensor(
            (DiagonalIndex(0, 2, 0, 1, 6, 1),),
            (
                DiagonalIndex(1, 2, 0, 0),
                DenseIndex(2, 9, 2),
            ),
            R,
        )

        assert jnp.allclose(
            R_st.dense(),
            jnp.tensordot(A_st.dense(), B_st.dense(), axes=([1, 2], [0, 1])),
        )

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_pure_block_block_block_gcd(self):
        rng_key = self.rng_key
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (4, 4, 3, 2, 6))
        B = jr.normal(k2, (2, 3, 2, 8, 2))

        A_st = SparseTensor(
            (DiagonalIndex(0, 4, 0, 2, 3, 2), DiagonalIndex(1, 4, 1, 3, 2, 3)),
            (DiagonalIndex(2, 4, 0, 0), DiagonalIndex(3, 4, 1, 1, 6, 4)),
            A,
        )
        B_st = SparseTensor(
            (DiagonalIndex(0, 2, 0, 2, 2, 2), DiagonalIndex(1, 3, 1, 3, 8, 3)),
            (DiagonalIndex(2, 2, 0, 0), DiagonalIndex(3, 3, 1, 1, 2, 4)),
            B,
        )

        A_aligned = A.reshape(4, 4, 3, 2, 3, 2)
        A_aligned = jnp.expand_dims(A_aligned, 4)
        A_aligned = A_aligned.transpose(0, 4, 1, 5, 2, 3, 6).reshape(4, 12, 3, 2, 2)
        A_aligned = jnp.expand_dims(A_aligned, 4)

        B_aligned = B.reshape(2, 3, 2, 1, 4, 2, 2)
        B_aligned = B_aligned.transpose(0, 2, 1, 4, 3, 5, 6).reshape(4, 12, 1, 2, 2)

        C_view = jax.lax.dot_general(
            A_aligned, B_aligned, (((4, 5), (2, 3)), ((0, 1), (0, 1)))
        )

        idx1, num_seg1 = get_routing_idx(4, 2, 4, 2)
        idx2, num_seg2 = get_routing_idx(4, 3, 12, 1)

        C_view_flat = C_view.reshape(4 * 12, *C_view.shape[2:])
        idx_flat = idx1[:, None] * num_seg2 + idx2[None, :]

        C_reduced_flat = _engine_reduce(
            C_view_flat, idx_flat.flatten(), num_seg1 * num_seg2
        )
        C_reduced = C_reduced_flat.reshape(num_seg1, num_seg2, *C_view.shape[2:])

        C_grid = C_reduced.reshape(2, 2, 1, 1, 4, 3, 3, 2, 2)

        R = C_grid.transpose(0, 3, 1, 6, 4, 7, 2, 5, 8).reshape(2, 6, 8, 6)

        R_st = SparseTensor(
            (DiagonalIndex(0, 2, 0, 2, 6, 1), DiagonalIndex(1, 1, None, 3, 8, 2)),
            (DiagonalIndex(2, 2, 0, 0), DiagonalIndex(3, 1, None, 1, 6, 3)),
            R,
        )
        assert jnp.allclose(
            R_st.dense(),
            jnp.tensordot(A_st.dense(), B_st.dense(), axes=([2, 3], [0, 1])),
            atol=1e-6,
            rtol=1e-4,
        )

        R_ref = A_st @ B_st

        assert jnp.allclose(R_ref.dense(), R_st.dense(), atol=1e-6, rtol=1e-4)
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_block_block_gcd_block_dense(self):
        rng_key = self.rng_key
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (4, 3, 4, 2, 3, 2))
        B = jr.normal(k2, (6, 6, 2, 5))

        A_st = SparseTensor(
            (DiagonalIndex(0, 4, 0, 2, 4, 2), DiagonalIndex(1, 3, 1, 3, 2, 3)),
            (DiagonalIndex(2, 4, 0, 0, 3, 4), DiagonalIndex(3, 3, 1, 1, 2, 5)),
            A,
        )
        B_st = SparseTensor(
            (DiagonalIndex(0, 6, 0, 2, 2, 2), DenseIndex(1, 6, 1)),
            (DiagonalIndex(2, 6, 0, 0, 5, 3),),
            B,
        )

        A_aligned = A.reshape(4, 3, 4, 2, 3, 1, 2)
        A_aligned = A_aligned.transpose(0, 4, 1, 2, 3, 5, 6).reshape(12, 3, 4, 2, 1, 2)

        B_aligned = B.reshape(6, 3, 2, 2, 1, 5)
        B_aligned = B_aligned.transpose(0, 3, 1, 4, 2, 5).reshape(12, 3, 1, 2, 5)

        C_view = jax.lax.dot_general(
            A_aligned, B_aligned, (((4, 5), (2, 3)), ((0, 1), (0, 1)))
        )

        idx1, num_seg1 = get_routing_idx(4, 6, 12, 2)
        C_reduced = _engine_reduce(C_view, idx1, num_seg1)

        C_grid = C_reduced.reshape(2, 2, 3, 3, 4, 2, 5)
        R = C_grid.transpose(0, 1, 4, 3, 5, 2, 6).reshape(2, 8, 6, 15)

        R_st = SparseTensor(
            (DiagonalIndex(0, 2, 0, 2, 8, 1), DenseIndex(1, 6, 2)),
            (DiagonalIndex(2, 2, 0, 0, 15, 3),),
            R,
        )
        assert jnp.allclose(
            R_st.dense(),
            jnp.tensordot(A_st.dense(), B_st.dense(), axes=([2, 3], [0, 1])),
            atol=1e-6,
            rtol=1e-4,
        )

        R_ref = A_st @ B_st

        # VALUES are the contract, and they hold: the hand-built reference above
        # and the engine agree on the dense form.
        assert jnp.allclose(R_ref.dense(), R_st.dense(), atol=1e-6, rtol=1e-4)

        # LAYOUT is not the reference's any more, and must not be pinned to it.
        # The contracted pair is meta 4 against meta 6 over one logical extent
        # 12, so ``a*b = 24 > 12`` and the engine now meets them on the gcd grid
        # instead of the lcm grid. The reference above is hand-built on the lcm
        # route (``get_routing_idx`` + ``_engine_reduce``), so it is a different
        # factoring of the same numbers and byte-identity no longer holds.
        #
        # Pin what IS the contract now: the result lives in the gcd frame.
        # meta gcd(4, 6) = 2, and the two block sides carry the whole logical
        # extent divided by that meta.
        got_meta = {d.size for d in R_ref.dims if d.is_sparse}
        assert got_meta == {2}, (
            f"expected the gcd frame (meta 2), got meta {got_meta}")
        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )

    def test_pure_block_dense_pure_pure_pure(self):
        rng_key = self.rng_key
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (4, 9, 3, 3))
        B = jr.normal(k2, (2, 9, 3, 2))

        A_st = SparseTensor(
            (
                DiagonalIndex(0, 4, 0, 2, 3, 3),
                DiagonalIndex(1, 3, 2, 4),
            ),
            (
                DiagonalIndex(2, 4, 0, 0),
                DenseIndex(3, 9, 1),
                DiagonalIndex(4, 3, 2, 1),
            ),
            A,
        )

        B_st = SparseTensor(
            (
                DiagonalIndex(0, 2, 0, 3, 2, 3),
                DiagonalIndex(1, 9, 1, 4),
                DiagonalIndex(2, 3, 2, 5),
            ),
            (
                DiagonalIndex(3, 2, 0, 0),
                DiagonalIndex(4, 9, 1, 1),
                DiagonalIndex(5, 3, 2, 2),
            ),
            B,
        )

        A_prime = A.reshape(2, 2, 9, 3, 3)
        B_prime = B.transpose(0, 3, 1, 2)

        R_inter = A_prime * jnp.expand_dims(B_prime, -1)

        R = R_inter.transpose(0, 3, 1, 4, 2).reshape(2, 3, 6, 9)

        R_st = SparseTensor(
            (
                DiagonalIndex(0, 2, 0, 2, 6, 2),
                DiagonalIndex(1, 3, 1, 4),
            ),
            (
                DiagonalIndex(2, 2, 0, 0),
                DenseIndex(3, 9, 3),
                DiagonalIndex(4, 3, 1, 1),
            ),
            R,
        )

        assert jnp.allclose(
            R_st.dense(),
            jnp.tensordot(A_st.dense(), B_st.dense(), axes=([2, 3, 4], [0, 1, 2])),
        )

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert (R_st == R_ref).all()
            assert jnp.allclose(R, R_ref.val)
        else:
            # engine value-identity up to float32 reduction reordering
            assert jnp.allclose(
                R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
            )

    def test_pure_block_block_block_gcd_block_dense(self):
        rng_key = self.rng_key
        _, k1, k2 = jr.split(rng_key, 3)

        A = jr.normal(k1, (4, 4, 3, 3, 2, 3, 3, 3))
        B = jr.normal(k2, (2, 3, 2, 4, 9, 2))

        A_st = SparseTensor(
            (
                DiagonalIndex(0, 4, 0, 3, 3, 3),
                DiagonalIndex(1, 4, 1, 4, 2, 4),
                DiagonalIndex(2, 3, 2, 5, 3, 6),
            ),
            (
                DiagonalIndex(3, 4, 0, 0),
                DiagonalIndex(4, 4, 1, 1, 3, 5),
                DiagonalIndex(5, 3, 2, 2, 3, 7),
            ),
            A,
        )
        B_st = SparseTensor(
            (
                DiagonalIndex(0, 2, 0, 3, 2, 2),
                DiagonalIndex(1, 3, 1, 4, 4, 3),
                DenseIndex(2, 9, 4),
            ),
            (DiagonalIndex(3, 2, 0, 0), DiagonalIndex(4, 3, 1, 1, 2, 5)),
            B,
        )

        A_aligned = A.reshape(4, 4, 3, 3, 2, 3, 1, 3, 3)
        A_aligned = jnp.expand_dims(A_aligned, 6)
        A_aligned = A_aligned.transpose(0, 6, 1, 5, 2, 3, 4, 7, 8, 9).reshape(
            4, 12, 3, 3, 2, 1, 3, 3
        )
        A_aligned = jnp.expand_dims(A_aligned, 5)

        B_aligned = B.reshape(2, 3, 2, 1, 4, 1, 3, 3, 2)
        B_aligned = B_aligned.transpose(0, 2, 1, 4, 6, 3, 5, 7, 8).reshape(
            4, 12, 3, 1, 1, 3, 2
        )

        C_view = jax.lax.dot_general(
            A_aligned, B_aligned, (((5, 6, 8), (3, 4, 5)), ((0, 1, 2), (0, 1, 2)))
        )

        idx1, num_seg1 = get_routing_idx(4, 2, 4, 2)
        idx2, num_seg2 = get_routing_idx(4, 3, 12, 1)

        C_view_flat = C_view.reshape(4 * 12, *C_view.shape[2:])
        idx_flat = idx1[:, None] * num_seg2 + idx2[None, :]

        C_reduced_flat = _engine_reduce(
            C_view_flat, idx_flat.flatten(), num_seg1 * num_seg2
        )
        C_reduced = C_reduced_flat.reshape(num_seg1, num_seg2, *C_view.shape[2:])

        C_grid = C_reduced.reshape(2, 2, 1, 1, 4, 3, 3, 3, 2, 3, 2)

        R = C_grid.transpose(0, 1, 7, 3, 4, 8, 6, 9, 2, 5, 10).reshape(2, 6, 8, 9, 6)

        R_st = SparseTensor(
            (
                DiagonalIndex(0, 2, 0, 3, 6, 1),
                DiagonalIndex(1, 1, None, 4, 8, 2),
                DenseIndex(2, 9, 3),
            ),
            (DiagonalIndex(3, 2, 0, 0), DiagonalIndex(4, 1, None, 1, 6, 4)),
            R,
        )

        assert jnp.allclose(
            R_st.dense(),
            jnp.tensordot(A_st.dense(), B_st.dense(), axes=([3, 4, 5], [0, 1, 2])),
            atol=1e-5,
        )

        R_ref = A_st @ B_st

        assert jnp.allclose(
            R_ref.dense(), R_st.dense(), rtol=1e-4, atol=1e-6
        )
        assert jnp.allclose(
            R_st.dense(), R_ref.dense(), rtol=1e-4, atol=1e-6
        )
        if _PIN_LAYOUT:
            assert jnp.allclose(R, R_ref.val)


if __name__ == "__main__":
    unittest.main()
