"""Bug regression: `_checkify_order` rejected partial orders + JAX scalars.

The post-merge `_checkify_order` raised

    ValueError: Supplied order is missing vertices {...}

if the user supplied an explicit elimination order that didn't cover *every*
eliminable vertex. The original (b0589be) behaviour was to filter the
supplied order — keeping only valid vertex IDs in the supplied relative
order — and leave any unlisted eliminable vertex un-eliminated. This is
useful for triplet / staged elimination strategies.

It also failed when ``order`` was a JAX scalar array (e.g. result of
``jnp.array([3, 1, 2])``) because element-wise comparisons against a Python
``set`` blew up. b0589be coerced via ``order.tolist()`` and ``int(o)``.

This test file pins both behaviours.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax import jacve, tree_allclose
from graphax.core import _checkify_order


def _simple_chain():
    """A 4-eqn chain. Eliminable vertices are {1, 2, 3} — vertex 4
    (reduce_sum) produces the final output, so it stays."""
    def f(x):
        a = jnp.sin(x)
        b = jnp.cos(a)
        c = a * b
        return jnp.sum(c)

    return f, jnp.array([0.1, 0.2, 0.3])


def test_full_explicit_order_unchanged():
    """An order covering every eliminable vertex is returned as-is."""
    f, x = _simple_chain()
    closed = jax.make_jaxpr(f)(x)
    # Only intermediates {1, 2, 3} are eliminable; vertex 4 is the output.
    full = [1, 2, 3]
    out = _checkify_order(full, closed.jaxpr, vo_vertices=set())
    assert out == full


def test_partial_order_filters_to_listed_vertices():
    """Partial order keeps only the listed vertices, in the same relative order."""
    f, x = _simple_chain()
    closed = jax.make_jaxpr(f)(x)
    out = _checkify_order([3, 1], closed.jaxpr, vo_vertices=set())
    assert out == [3, 1], f"expected [3, 1], got {out}"


def test_partial_order_drops_invalid_ids():
    """IDs not in the eliminable set are dropped silently."""
    f, x = _simple_chain()
    closed = jax.make_jaxpr(f)(x)
    # 99 isn't a real vertex; 4 is the output (non-eliminable).
    # Both should be filtered, the eliminable 2 preserved.
    out = _checkify_order([2, 99, 4], closed.jaxpr, vo_vertices=set())
    assert out == [2]


def test_jax_array_order_is_accepted():
    """Numeric arrays (e.g. from a learned policy) are coerced to ints."""
    f, x = _simple_chain()
    closed = jax.make_jaxpr(f)(x)
    arr_order = jnp.array([3, 2, 1])
    out = _checkify_order(arr_order, closed.jaxpr, vo_vertices=set())
    assert out == [3, 2, 1]
    assert all(isinstance(o, int) for o in out)


def test_partial_order_via_jacve_matches_full_when_valid():
    """End-to-end: a partial order that covers all eliminable vertices matches
    the equivalent full elimination."""
    def f(x):
        a = jnp.sin(x)
        return (a * a).sum()

    x = jnp.array([0.1, 0.2, 0.3])
    full = jacve(f, order="rev", argnums=(0,))(x)
    # Explicitly pass an order with all eliminable vertices.
    closed = jax.make_jaxpr(f)(x)
    n = len(closed.jaxpr.eqns)
    partial = jacve(f, order=list(range(n, 0, -1)), argnums=(0,))(x)
    assert bool(tree_allclose(full, partial))


def test_partial_order_with_array_input_via_jacve():
    """jacve accepts a JAX array as the order argument."""
    def f(x):
        return jnp.sin(x).sum()

    x = jnp.array([1.0, 2.0])
    closed = jax.make_jaxpr(f)(x)
    n = len(closed.jaxpr.eqns)
    arr_order = jnp.arange(n, 0, -1)  # JAX array, descending
    res_arr = jacve(f, order=arr_order, argnums=(0,))(x)
    res_str = jacve(f, order="rev", argnums=(0,))(x)
    assert bool(tree_allclose(res_arr, res_str))


def test_empty_order_does_no_elimination():
    """An empty order leaves the graph untouched — Jacobian still computed
    via the post-elimination densify step (which materializes uneliminated
    paths from inputs to outputs)."""
    def f(x):
        return jnp.sin(x).sum()

    x = jnp.array([0.1, 0.2, 0.3])
    # With an empty order the Jacobian should still be correct as long as
    # the densify step at the end of vertex_elimination_jaxpr handles
    # uneliminated-but-reachable edges. If this test fails, that's a
    # downstream bug worth flagging — but `_checkify_order` itself must
    # still accept the empty list without raising.
    out = _checkify_order([], jax.make_jaxpr(f)(x).jaxpr, vo_vertices=set())
    assert out == []
