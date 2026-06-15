"""lax.cond / lax.switch differentiation.

The branch index is concrete during graphax's forward pass, so cond/switch are
differentiated through the TAKEN branch (recursively, like jax). The old rule
returned ``[]`` — a silently-zero gradient for ALL control flow. The branch
recursion bypasses the id-keyed eliminator cache (fresh_eliminator) so repeated
cond calls in one process can't cross-contaminate.

Note: branches that self-alias an input (``lambda v: v*v``) are avoided here —
graphax under-counts the aliased self-multiply (returns v, not 2v) at every
level, a separate pre-existing limitation, not a cond issue.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import jacve, tree_allclose

X = jnp.array([1.0, 2.0, 3.0])


def _check(fn, argnums=(0,), args=(X,)):
    ref = jax.jacrev(fn, argnums)(*args)
    for mode in ("fwd", "rev"):
        assert tree_allclose(jacve(fn, mode, argnums)(*args), ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("pred", [True, False], ids=["true", "false"])
def test_cond_branches(pred):
    _check(lambda z: jax.lax.cond(z[0] > 0 if pred else z[0] < 0,
                                  lambda v: jnp.sin(v), lambda v: 3.0 * v, z))


def test_cond_scalar_output():
    _check(lambda z: jax.lax.cond(z[0] > 0, lambda v: jnp.sum(jnp.sin(v)),
                                  lambda v: jnp.sum(2 * v), z))


def test_cond_multi_output():
    _check(lambda z: jax.lax.cond(z[0] > 0, lambda v: (jnp.sin(v), jnp.exp(v)),
                                  lambda v: (v, 2 * v), z)[0])


def test_cond_two_operands():
    _check(lambda a, b: jax.lax.cond(a[0] > 0, lambda p, q: p * q, lambda p, q: p + q, a, b),
           argnums=(0, 1), args=(X, 2 * X))


def test_cond_composed_in_graph():
    _check(lambda z: jnp.sum(jnp.cos(jax.lax.cond(z[0] > 0, jnp.sin, lambda v: 2 * v, z))))


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_switch(idx):
    _check(lambda z: jax.lax.switch(idx, [lambda v: v * 2, lambda v: jnp.sin(v),
                                          lambda v: v + 1.0], z))


def test_many_conds_no_cache_contamination():
    # Several distinct cond/switch graphs in one process must all stay correct
    # (the branch recursion bypasses the id-keyed eliminator cache).
    fns = [
        lambda z: jax.lax.cond(z[0] > 0, jnp.sin, lambda v: 3 * v, z),
        lambda z: jax.lax.cond(z[0] > 0, jnp.exp, lambda v: v + 1, z),
        lambda z: jax.lax.switch(2, [lambda v: v * 2, jnp.sin, lambda v: v + 1], z),
        lambda z: jax.lax.switch(1, [lambda v: v * 2, jnp.cos, lambda v: v + 1], z),
    ]
    for fn in fns:
        _check(fn)
