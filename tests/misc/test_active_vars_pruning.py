"""Feature pin: `_build_graph(..., argnums=...)` skips dead branches.

When `argnums` is supplied to `_build_graph`, edges should only be emitted
along paths reachable forward from the differentiable inputs. Branches
reachable only from non-differentiable inputs (e.g. extra args used solely
for primal computation) shouldn't pollute the graph or get composed during
elimination.

This both verifies correctness (the Jacobian is unaffected) and that the
pruning actually reduces the graph (fewer edges than the unpruned build).
"""

import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose
from graphax.core import _build_graph


def _count_edges(graph) -> int:
    return sum(len(inner) for inner in graph.values())


def test_active_vars_reduces_edges():
    """A function with a non-differentiable arg has fewer edges with active_vars."""

    def f(x, y, z):
        # x, y are differentiated; z is used for primal but argnums=(0,1).
        a = x * y
        b = a + jnp.sin(z)  # depends on z
        return b.sum()

    closed = jax.make_jaxpr(f)(jnp.ones(3), jnp.ones(3), jnp.ones(3))
    args = (jnp.ones(3), jnp.ones(3), jnp.ones(3))

    _, g_full, _, _ = _build_graph(closed.jaxpr, args, closed.literals, argnums=None)
    _, g_pruned, _, _ = _build_graph(
        closed.jaxpr, args, closed.literals, argnums=(0, 1)
    )

    assert _count_edges(g_pruned) < _count_edges(g_full), (
        "active_vars pruning should reduce edges when there are dead branches"
    )


def test_active_vars_correctness_preserved():
    """Pruning must not change the computed Jacobian."""

    def f(x, y, z):
        a = x * y
        b = a + jnp.sin(z)
        return b.sum()

    args = (jnp.array([1.0, 2.0, 3.0]),
            jnp.array([4.0, 5.0, 6.0]),
            jnp.array([7.0, 8.0, 9.0]))
    veres = jax.jit(jacve(f, order="rev", argnums=(0, 1)))(*args)
    refres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(*args)
    assert bool(tree_allclose(veres, refres))


def test_active_vars_no_argnums_treats_all_as_active():
    """With argnums=None, all invars are active (= old behaviour)."""

    def f(x, y):
        return (x * y).sum()

    closed = jax.make_jaxpr(f)(jnp.ones(2), jnp.ones(2))
    args = (jnp.ones(2), jnp.ones(2))
    _, g, _, _ = _build_graph(closed.jaxpr, args, closed.literals, argnums=None)
    # mul x*y -> at least 1 edge per active invar.
    assert _count_edges(g) >= 1
