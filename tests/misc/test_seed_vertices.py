"""Tangent / adjoint seed VERTICES (graphax.seed_vertices).

The seeds enter as ordinary jaxpr equations, so they become eliminable vertices
that the elimination ORDER (a learned alphagrad order, passed as `order=`) can
schedule. Correctness must hold for ANY order: an adjoint seed gives the VJP,
a tangent seed gives the JVP — both equal to jax, under fwd / rev / random.
"""
import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import pytest

import graphax.seed_vertices as sv
from graphax import jacve


def f(W, x):
    return jnp.tanh(W @ x)   # (3,) vector output


KEY = jrand.PRNGKey(0)
W = jrand.normal(KEY, (3, 4))
X = jrand.normal(jrand.split(KEY)[1], (4,))
YBAR = jrand.normal(KEY, (3,))
XD = (jrand.normal(KEY, (3, 4)), jrand.normal(jrand.split(KEY)[1], (4,)))


def _orders():
    nv = len(jax.make_jaxpr(sv.with_adjoint_seed(f, YBAR))(W, X).jaxpr.eqns)
    yield "rev"
    yield "fwd"
    for s in range(3):
        yield [int(v) for v in np.random.default_rng(s).permutation(np.arange(1, nv + 1))]


@pytest.mark.parametrize("order", list(_orders()))
def test_adjoint_seed_is_vjp(order):
    _, vjp_fn = jax.vjp(f, W, X)
    ref = vjp_fn(YBAR)
    got = sv.seed_vjp(f, YBAR, order=order, argnums=(0, 1))(W, X)
    # rtol-based: different elimination orders reassociate the same float32 sum
    # differently, so a pure atol=1e-5 is too tight for some orders on some BLAS
    # backends (e.g. `order4` failed only on the pgi15 cluster, passed locally).
    # rtol=1e-4 tolerates reassociation while still catching a real divergence.
    assert all(
        bool(jnp.allclose(a, b, rtol=1e-4, atol=1e-5)) for a, b in zip(got, ref)
    )


@pytest.mark.parametrize("order", ["fwd", "rev"])
def test_tangent_seed_is_jvp(order):
    ref = jax.jvp(f, (W, X), XD)[1]
    got = sv.seed_jvp(f, XD, (W, X), order=order)
    assert bool(jnp.allclose(jnp.ravel(got), ref, atol=1e-5))


def test_seeds_add_eliminable_vertices():
    """The seed enters as real vertices (action space), not a special case."""
    base = len(jax.make_jaxpr(lambda W, x: jnp.sum(f(W, x)))(W, X).jaxpr.eqns)
    adj = len(jax.make_jaxpr(sv.with_adjoint_seed(f, YBAR))(W, X).jaxpr.eqns)
    tan = len(jax.make_jaxpr(sv.with_tangent_seed(f, XD))(jnp.zeros(()), W, X).jaxpr.eqns)
    assert adj >= base and tan >= base
