"""Bug: convert_element_type transform crashed when applied to a val=None edge.

After several composition steps, an edge can have val=None (it's a pure
structural identity, e.g. a sparse Kronecker). `inverse_convert_element_type_transform`
unconditionally called `lax.convert_element_type(post.val, new_dtype)` which
raised `ValueError: Invalid argument to dtype: None.`

The fix: short-circuit and return `post.copy()` when `post.val is None`. This
matters in any function with a Python-scalar (e.g. `0.5`) used together with
sigmoid/log/exp paths — JAX inserts a `convert_element_type` node, and the
gradient path eventually composes that transform with a structural edge.
"""

import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose


def test_convert_element_type_in_chain():
    """Triggers the convert_element_type transform on a val=None edge."""

    def f(state, alpha):
        new_state = alpha * state - alpha
        return jnp.sum(new_state)

    state = jnp.ones(8)
    alpha = jnp.array(0.95)

    veres = jax.jit(jacve(f, order="rev", argnums=(0, 1)))(state, alpha)
    refres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(state, alpha)
    assert bool(tree_allclose(veres, refres))


def test_squared_loss_with_scalar_factor():
    """The reduced repro of the SNN failure: 0.5 * sigmoid(...) ** 2."""

    def f(U, alpha):
        s = jax.nn.sigmoid(alpha * U)
        return 0.5 * s

    U = jnp.zeros(4)
    veres = jax.jit(jacve(f, order="rev", argnums=(0, 1)))(U, jnp.array(0.95))
    refres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(U, jnp.array(0.95))
    assert bool(tree_allclose(veres, refres))
