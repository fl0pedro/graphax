"""Bug: rev-mode through concat hit the DiagonalIndex-with-axis=None branch.

In rev mode, the accumulated Jacobian at the concat node is the full identity
tensor, represented as a DiagonalIndex pair with `axis=None`. This exercises
the else-branch of `inverse_concatenate_transform` — the same one that did
positional `SparseTensor(...)` construction and was therefore broken.

Pinned here as a focused regression test (the existing
`tests/core/primitive_test.py` covers the same scenario but does a lot more,
this one is the minimum surface area for a fast regression catch).
"""

import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose


def test_concat_then_tan_rev():
    """tan(concat(sin(x), log(y))) — sparse-with-axis=None inverse_concatenate path."""

    def f(x, y):
        z, w = jnp.sin(x), jnp.log(y)
        return jnp.tan(jnp.concatenate([z, w], axis=0))

    x = jnp.ones((2,))
    y = jnp.ones((3,))
    veres = jax.jit(jacve(f, order="rev", argnums=(0, 1)))(x, y)
    refres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)
    assert bool(tree_allclose(veres, refres))


def test_concat_then_tan_fwd():
    """Same in forward mode (different code path through concatenate_transform)."""

    def f(x, y):
        z, w = jnp.sin(x), jnp.log(y)
        return jnp.tan(jnp.concatenate([z, w], axis=0))

    x = jnp.ones((2,))
    y = jnp.ones((3,))
    veres = jax.jit(jacve(f, order="fwd", argnums=(0, 1)))(x, y)
    refres = jax.jit(jax.jacfwd(f, argnums=(0, 1)))(x, y)
    assert bool(tree_allclose(veres, refres))
