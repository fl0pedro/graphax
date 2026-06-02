import unittest

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor
from utils import assert_matmul_result


class TestReplicationMatmul(unittest.TestCase):
    ### Replication tests
    def test_simple_replication(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4,))
        _x = jnp.expand_dims(x, 1)
        _x = jnp.tile(_x, (1, 3))
        y = jrand.normal(ykey, (3, 2))
        res = _x @ y

        stx = SparseTensor([DenseIndex(0, 4, 0)], [DenseIndex(1, 3, None)], x)
        sty = SparseTensor([DenseIndex(0, 3, 0)], [DenseIndex(1, 2, 1)], y)
        stres = stx @ sty

        assert_matmul_result(stres, res, (4,), (2,), (4, 2))

    def test_double_replication(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4,))
        _x = jnp.expand_dims(x, 1)
        _x = jnp.tile(_x, (1, 3))

        y = jrand.normal(ykey, (2,))
        _y = jnp.expand_dims(y, 0)
        _y = jnp.tile(_y, (3, 1))
        res = _x @ _y

        stx = SparseTensor([DenseIndex(0, 4, 0)], [DenseIndex(1, 3, None)], x)
        sty = SparseTensor([DenseIndex(0, 3, None)], [DenseIndex(1, 2, 0)], y)
        stres = stx @ sty

        assert_matmul_result(stres, res, (4,), (2,), (4, 2))

    def test_replication_2d(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)

        x = jrand.normal(xkey, (4,))
        _x = jnp.eye(4) * x
        _x = jnp.expand_dims(_x, 2)
        _x = jnp.tile(_x, (1, 1, 5))

        y = jrand.normal(ykey, (4, 5))
        _y = jnp.einsum("ij,jk->ijk", y, jnp.eye(5))

        res = jnp.einsum("ijk,jkl->il", _x, _y)

        stx = SparseTensor(
            [DiagonalIndex(0, 4, 0, 1)],
            [DiagonalIndex(1, 4, 0, 0), DenseIndex(2, 5, None)],
            x,
        )
        sty = SparseTensor(
            [DenseIndex(0, 4, 0), DiagonalIndex(1, 5, 1, 2)],
            [DiagonalIndex(2, 5, 1, 1)],
            y,
        )
        stres = stx @ sty

        assert_matmul_result(stres, res, (4,), (5,), (4, 5))

    def test_replication_2d_2nd(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)

        x = jrand.normal(xkey, (4, 5))
        _x = jnp.einsum("ij,ik->ikj", x, jnp.eye(4))

        y = jrand.normal(ykey, (5,))
        _y = jnp.eye(5) * y
        _y = jnp.expand_dims(_y, 0)
        _y = jnp.tile(_y, (4, 1, 1))

        res = jnp.einsum("ijk,jkl->il", _x, _y)

        stx = SparseTensor(
            [DiagonalIndex(0, 4, 0, 1)],
            [DiagonalIndex(1, 4, 0, 0), DenseIndex(2, 5, 1)],
            x,
        )
        sty = SparseTensor(
            [DenseIndex(0, 4, None), DiagonalIndex(1, 5, 0, 2)],
            [DiagonalIndex(2, 5, 0, 1)],
            y,
        )
        stres = stx @ sty

        assert_matmul_result(stres, res, (4,), (5,), (4, 5))

    def test_4d_replication(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (3, 4, 5))
        d = jnp.eye(4)
        _x = jnp.einsum("ijk,jl->ijlk", x, d)

        y = jrand.normal(ykey, (5, 2))
        _y = jnp.expand_dims(y, 0)
        _y = jnp.tile(_y, (4, 1, 1))
        res = jnp.einsum("ijkl,klm->ijm", _x, _y)

        stx = SparseTensor(
            [DenseIndex(0, 3, 0), DiagonalIndex(1, 4, 1, 2)],
            [DiagonalIndex(2, 4, 1, 1), DenseIndex(3, 5, 2)],
            x,
        )
        sty = SparseTensor(
            [DenseIndex(0, 4, None), DenseIndex(1, 5, 0)],
            [DenseIndex(2, 2, 1)],
            y,
        )
        stres = stx @ sty

        assert_matmul_result(stres, res, (3, 4), (2,), (4, 3, 2))


if __name__ == "__main__":
    unittest.main()
