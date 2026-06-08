import unittest

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from graphax.sparse.indexes import DiagonalIndex
from graphax.sparse.tensor import SparseTensor
from utils import assert_matmul_result


class TestSparseMatmul(unittest.TestCase):
    def test_simple_broadcast(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4,))
        y = jrand.normal(ykey, (4,))
        res = jnp.diag(x) @ jnp.diag(y)

        stx = SparseTensor(
            [DiagonalIndex(0, 4, 0, 1)], [DiagonalIndex(1, 4, 0, 0)], x
        )
        sty = SparseTensor(
            [DiagonalIndex(0, 4, 0, 1)], [DiagonalIndex(1, 4, 0, 0)], y
        )
        stres = stx @ sty

        assert_matmul_result(stres, res, (4,), (4,), (4,))

    def test_simple_Nones(self):
        _x = jnp.eye(3)
        _y = jnp.eye(3)

        res = _x @ _y

        stx = SparseTensor(
            [DiagonalIndex(0, 3, None, 1)], [DiagonalIndex(1, 3, None, 0)], None
        )
        sty = SparseTensor(
            [DiagonalIndex(0, 3, None, 1)], [DiagonalIndex(1, 3, None, 0)], None
        )
        stres = stx @ sty

        assert_matmul_result(stres, res, (3,), (3,), None)

    ### Tests for 4d tensors
    def test_4d_sparse_double_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)

        x = jrand.normal(xkey, (3, 5))
        d = jnp.eye(15, 15)
        d = d.reshape(3, 5, 3, 5)
        _x = jnp.einsum("ij,ijkl->ijkl", x, d)

        y = jrand.normal(ykey, (3, 5))
        d = jnp.eye(15, 15)
        d = d.reshape(3, 5, 3, 5)
        _y = jnp.einsum("ij,ijkl->ijkl", y, d)

        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)

        stx = SparseTensor(
            [DiagonalIndex(0, 3, 0, 2), DiagonalIndex(1, 5, 1, 3)],
            [DiagonalIndex(2, 3, 0, 0), DiagonalIndex(3, 5, 1, 1)],
            x,
        )
        sty = SparseTensor(
            [DiagonalIndex(0, 3, 0, 2), DiagonalIndex(1, 5, 1, 3)],
            [DiagonalIndex(2, 3, 0, 0), DiagonalIndex(3, 5, 1, 1)],
            y,
        )
        stres = stx @ sty

        assert_matmul_result(stres, res, (3, 5), (3, 5), (3, 5))

    def test_4d_only_Nones(self):
        d1 = jnp.eye(3)
        d2 = jnp.eye(4)
        _x = jnp.einsum("ij,kl->ikjl", d1, d2)

        d1 = jnp.eye(3)
        d2 = jnp.eye(4)
        _y = jnp.einsum("ij,kl->iklj", d1, d2)

        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)

        stx = SparseTensor(
            [DiagonalIndex(0, 3, None, 2), DiagonalIndex(1, 4, None, 3)],
            [DiagonalIndex(2, 3, None, 0), DiagonalIndex(3, 4, None, 1)],
            None,
        )
        sty = SparseTensor(
            [DiagonalIndex(0, 3, None, 3), DiagonalIndex(1, 4, None, 2)],
            [DiagonalIndex(2, 4, None, 1), DiagonalIndex(3, 3, None, 0)],
            None,
        )
        stres = stx @ sty

        assert_matmul_result(stres, res, (3, 4), (4, 3), None)


if __name__ == "__main__":
    unittest.main()
