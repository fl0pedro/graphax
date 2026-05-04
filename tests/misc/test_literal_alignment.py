"""Bug: `_build_graph` mis-indexed elementals when an eqn had Literal inputs.

For an eqn like `mul 0.5 x` (Literal first, Var second), `eqn.invars` has 2
entries but only 1 is a Var. The elemental rule returns 2 elementals (one per
primal). The old code zipped `invars` (filtered to Vars) with the
unfiltered elemental list — so the Var `x` got the elemental at index 0
(which was for the literal 0.5, e.g. broadcast scaling), not the elemental at
index 1 (which was 0.5 itself).

The fix: track each Var's position in the original `eqn.invars` and index
into the elemental list with that position.

Manifested as wildly wrong Jacobians for any function that used scalar
literals like `0.5 * x` or `2.0 * y`.
"""

import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose


def test_literal_first_then_var():
    """`0.5 * x` — Literal at position 0, Var at position 1."""
    def f(x):
        return 0.5 * x

    x = jnp.array([1.0, 2.0, 3.0])
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
    refres = jax.jit(jax.jacrev(f, argnums=(0,)))(x)
    assert bool(tree_allclose(veres, refres))


def test_var_first_then_literal():
    """`x * 2.0` — Var at position 0, Literal at position 1."""
    def f(x):
        return x * 2.0

    x = jnp.array([1.0, 2.0, 3.0])
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
    refres = jax.jit(jax.jacrev(f, argnums=(0,)))(x)
    assert bool(tree_allclose(veres, refres))


def test_literal_in_chain():
    """Compose `0.5 * sigmoid(x)` so the wrong-elemental bug shows up via composition."""
    def f(x):
        return 0.5 * jax.nn.sigmoid(x)

    x = jnp.array([0.1, 0.2, 0.3])
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
    refres = jax.jit(jax.jacrev(f, argnums=(0,)))(x)
    assert bool(tree_allclose(veres, refres))


def test_literal_via_division():
    """`x / 2.0` — literal at position 1 of a binary op."""
    def f(x):
        return x / 2.0

    x = jnp.array([1.0, 2.0, 3.0])
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
    refres = jax.jit(jax.jacrev(f, argnums=(0,)))(x)
    assert bool(tree_allclose(veres, refres))


def test_two_literals_in_chain():
    """`(0.5 * x) + 1.0` — literal in two separate ops."""
    def f(x):
        return (0.5 * x) + 1.0

    x = jnp.array([1.0, 2.0, 3.0])
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
    refres = jax.jit(jax.jacrev(f, argnums=(0,)))(x)
    assert bool(tree_allclose(veres, refres))
