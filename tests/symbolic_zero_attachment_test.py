"""A value attached through ``x - stop_gradient(x)`` keeps its edge and loses
its forward: the elimination never binds the zero-valued product."""

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.interpreters import partial_eval as pe

from graphax import jacve


def attached(w, J):
    d = w - jax.lax.stop_gradient(w)
    return jnp.tanh(jnp.sum(J * d, axis=-1) + 0.5)


def plain(w, c, J):
    return jnp.tanh(jnp.sum(J * (w - c), axis=-1) + 0.5)


def _dced_names(fn, *args):
    closed = jax.make_jaxpr(fn)(*args)
    jx, _ = pe.dce_jaxpr(closed.jaxpr, [True] * len(closed.jaxpr.outvars))
    return [e.primitive.name for e in jx.eqns]


def test_the_zero_valued_attachment_keeps_its_edge_and_loses_its_forward():
    key = jax.random.PRNGKey(0)
    w = jax.random.normal(key, (5, 7))
    J = jax.random.normal(jax.random.fold_in(key, 1), (5, 7))
    jac = jacve(attached, "rev", argnums=(0,))
    names = _dced_names(jac, w, J)
    assert "sub" not in names, names
    assert "stop_gradient" not in names, names
    want = jax.jacrev(attached)(w, J)
    np.testing.assert_allclose(np.asarray(jac(w, J)), np.asarray(want),
                               rtol=1e-6, atol=1e-6)


def test_a_difference_of_two_inputs_is_still_computed():
    key = jax.random.PRNGKey(1)
    w = jax.random.normal(key, (5, 7))
    c = jax.random.normal(jax.random.fold_in(key, 1), (5, 7))
    J = jax.random.normal(jax.random.fold_in(key, 2), (5, 7))
    jac = jacve(plain, "rev", argnums=(0,))
    names = _dced_names(jac, w, c, J)
    assert "sub" in names, names
    want = jax.jacrev(plain)(w, c, J)
    np.testing.assert_allclose(np.asarray(jac(w, c, J)), np.asarray(want),
                               rtol=1e-6, atol=1e-6)
