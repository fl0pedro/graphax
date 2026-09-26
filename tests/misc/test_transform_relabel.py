# The transpose and squeeze relabels move no data (dsnn-dfw.252).
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import jacve
from graphax.primitives.transforms import _squeeze_elementals, _transpose_elementals
from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor

KEY = jax.random.PRNGKey(4)


def _names(jaxpr):
    return [e.primitive.name for e in jaxpr.eqns]


def _dense_edge(val, n_out):
    dims = tuple(DenseIndex(i, s, i) for i, s in enumerate(val.shape))
    return SparseTensor(dims[:n_out], dims[n_out:], val)


@pytest.mark.parametrize("perm", list(itertools.permutations(range(3))))
def test_the_transpose_relabel_permutes_the_dims_and_moves_no_data(perm):
    x = jax.random.normal(KEY, (2, 3, 4), jnp.float32)
    y = jnp.transpose(x, perm)
    t = _transpose_elementals((x,), y, permutation=perm)[0].pre_transforms[0]
    w = jax.random.normal(jax.random.fold_in(KEY, 1), (5,) + tuple(y.shape), jnp.float32)
    post = _dense_edge(w, 1)
    names = _names(jax.make_jaxpr(lambda v: t.inverse_transform(_dense_edge(v, 1)).val)(w))
    assert "transpose" not in names, names
    got = t.inverse_transform(post)
    want = jnp.transpose(w, (0,) + tuple(1 + int(p) for p in np.argsort(perm)))
    assert got.shape == (5,) + tuple(x.shape)
    np.testing.assert_array_equal(np.asarray(got.dense()), np.asarray(want))
    pre = _dense_edge(jax.random.normal(jax.random.fold_in(KEY, 2), tuple(x.shape) + (6,), jnp.float32), 3)
    fwd = t.transform(pre)
    assert fwd.shape == tuple(y.shape) + (6,)
    np.testing.assert_array_equal(np.asarray(fwd.dense()),
                                  np.asarray(jnp.transpose(pre.val, tuple(perm) + (3,))))


def test_the_squeeze_relabel_adds_its_unit_dim_implicitly():
    x = jax.random.normal(KEY, (3, 1, 4), jnp.float32)
    y = jnp.squeeze(x, axis=1)
    t = _squeeze_elementals((x,), y, dimensions=(1,))[0].pre_transforms[0]
    w = jax.random.normal(jax.random.fold_in(KEY, 3), (5, 3, 4), jnp.float32)
    names = _names(jax.make_jaxpr(lambda v: t.inverse_transform(_dense_edge(v, 1)).val)(w))
    assert not [n for n in names if n in ("broadcast_in_dim", "reshape", "transpose")], names
    got = t.inverse_transform(_dense_edge(w, 1))
    assert got.shape == (5, 3, 1, 4)
    np.testing.assert_array_equal(np.asarray(got.dense()), np.asarray(w)[:, :, None, :])


def test_a_function_with_transposes_and_squeezes_keeps_its_gradient():
    w1 = jax.random.normal(KEY, (4, 3), jnp.float32)

    def f(x):
        h = jnp.tanh(x @ w1).T
        return jnp.sum(jnp.squeeze(h[:, None, :], axis=1) ** 2)

    x = jax.random.normal(jax.random.fold_in(KEY, 5), (2, 4), jnp.float32)
    got = jacve(f, "rev", argnums=(0,))(x)
    np.testing.assert_allclose(np.asarray(got), np.asarray(jax.grad(f)(x)), rtol=1e-5, atol=1e-6)
