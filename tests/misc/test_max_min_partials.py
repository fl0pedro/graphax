import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose


X = jnp.array([-2.0, 0.5, 4.0, 7.0])


def _check(f, x):
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
    revres = jax.jit(jax.jacrev(f, argnums=(0,)))(x)
    assert tree_allclose(veres, revres)


def test_maximum_x_literal():
    _check(lambda x: jnp.maximum(x, 0.0), X)


def test_maximum_literal_x():
    _check(lambda x: jnp.maximum(0.0, x), X)


def test_minimum_x_literal():
    _check(lambda x: jnp.minimum(x, 5.0), X)


def test_minimum_literal_x():
    _check(lambda x: jnp.minimum(5.0, x), X)


def test_max_plus_min_combo():
    # Original reproducer: expected diag([1, 2, 2, 1]).
    f = lambda x: jnp.maximum(x, 0.0) + jnp.minimum(x, 5.0)
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(X)
    expected = jnp.diag(jnp.array([1.0, 2.0, 2.0, 1.0]))
    assert tree_allclose(veres, (expected,))


