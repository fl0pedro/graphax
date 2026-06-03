"""Code-review follow-up — value-semantic consumers of a SparseTensor must
densify compressed (`BandedIndex` / `SetIndex`) dims first.

A raw band buffer carries out-of-band padding slots and a set buffer stores the
pre-combination per-side blocks, so their element multiset does NOT match the
dense form. Reductions (sum/prod/max/min/all/any/mean) and the non-linear unary
ops (abs/round) therefore route through `SparseTensor._materialize_compressed`
before reading `val`. These tests build real compressed outputs from
elementwise / matmul and assert agreement with the dense form.
"""
import unittest

import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.tensor import SparseTensor
from graphax.sparse.indexes import DiagonalIndex, BandedIndex
from graphax.sparse.ops.elementwise import elementwise
from graphax.sparse.ops.matmul import matmul


def _n(shape, key=0):
    return jr.normal(jr.PRNGKey(key), shape).astype(jnp.float32)


def _set_output(semantic="union", is_intersection=False):
    """A misaligned 5/11 elementwise output → a SetIndex pair (compressed)."""
    a = SparseTensor(
        (DiagonalIndex(0, 11, 0, 1, 5, 1),), (DiagonalIndex(1, 11, 0, 0, 5, 2),),
        _n((11, 5, 5), 1),
    )
    b = SparseTensor(
        (DiagonalIndex(0, 5, 0, 1, 11, 1),), (DiagonalIndex(1, 5, 0, 0, 11, 2),),
        _n((5, 11, 11), 2),
    )
    op = jnp.multiply if is_intersection else jnp.add
    return elementwise(a, b, op, is_intersection=is_intersection)


def _banded_output_W3():
    """A genuine width-3 band tensor (out-of-band padding slots in val)."""
    M, W, B = 4, 3, 2
    data = _n((M, W, B, B), 7)
    out = (BandedIndex(id=0, size=M, axis=0, other_id=1, block_size=B,
                       block_axis=1, band_width=W, offset=(), primary=True,
                       n_secondary=M, n_meta=1),)
    primal = (BandedIndex(id=1, size=M, axis=0, other_id=0, block_size=B,
                          block_axis=3, band_width=W, offset=(), primary=False,
                          n_secondary=M, n_meta=1),)
    return SparseTensor(out, primal, data, check_consistency=False)


def _k2_banded_output():
    """A K=2 misaligned-contract matmul → 4 BandedIndex dims (compressed)."""
    a = SparseTensor(
        (DiagonalIndex(0, 11, 0, 2, 5, 2), DiagonalIndex(1, 3, 1, 3, 4, 4)),
        (DiagonalIndex(2, 11, 0, 0, 5, 3), DiagonalIndex(3, 3, 1, 1, 2, 5)),
        _n((11, 3, 5, 5, 4, 2), 1),
    )
    b = SparseTensor(
        (DiagonalIndex(0, 5, 0, 2, 11, 2), DiagonalIndex(1, 2, 1, 3, 3, 4)),
        (DiagonalIndex(2, 5, 0, 0, 7, 3), DiagonalIndex(3, 2, 1, 1, 9, 5)),
        _n((5, 2, 11, 7, 3, 9), 2),
    )
    return matmul(a, b)


class TestCompressedReductions(unittest.TestCase):
    def _check(self, t):
        self.assertTrue(any(d.is_compressed for d in t.dims))
        ref = t.dense()
        self.assertTrue(jnp.allclose(t.sum(), ref.sum(), atol=1e-3), "sum")
        self.assertTrue(jnp.allclose(t.max(), ref.max(), atol=1e-4), "max")
        self.assertTrue(jnp.allclose(t.min(), ref.min(), atol=1e-4), "min")
        self.assertTrue(jnp.allclose(t.mean(), ref.mean(), atol=1e-4), "mean")
        self.assertEqual(bool(t.any()), bool(ref.any()))
        self.assertEqual(bool(t.all()), bool(jnp.all(ref != 0)))

    def test_setindex_union_reductions(self):
        self._check(_set_output("union"))

    def test_setindex_intersection_reductions(self):
        self._check(_set_output("intersection", is_intersection=True))

    def test_banded_W3_reductions(self):
        # The headline case: sum previously summed out-of-band padding slots.
        self._check(_banded_output_W3())

    def test_k2_banded_reductions(self):
        self._check(_k2_banded_output())

    def test_max_is_actually_wrong_without_densify(self):
        # Guard against regressing to the raw-buffer reduction: the dense max of
        # a union output exceeds the raw buffer max (overlapping lhs+rhs cells
        # sum), so a correct max() must be strictly above the buffer max here.
        t = _set_output("union")
        self.assertGreater(float(t.max()), float(jnp.max(t.val)) + 1e-3)


class TestCompressedUnaryOps(unittest.TestCase):
    def test_abs_matches_dense(self):
        for t in (_set_output("union"), _banded_output_W3()):
            self.assertTrue(
                jnp.allclose(abs(t).dense(), jnp.abs(t.dense()), atol=1e-4)
            )

    def test_neg_matches_dense(self):
        # __neg__ is linear → stays lazy, but must still equal the dense negate.
        t = _set_output("union")
        self.assertTrue(jnp.allclose((-t).dense(), -t.dense(), atol=1e-4))

    def test_round_matches_dense(self):
        # round() with no ndigits passes None to jnp.round and fails for ANY
        # tensor (pre-existing, unrelated to compressed storage), so pass 2.
        t = _set_output("union")
        self.assertTrue(
            jnp.allclose(round(t, 2).dense(), jnp.round(t.dense(), 2), atol=1e-4)
        )


if __name__ == "__main__":
    unittest.main()
