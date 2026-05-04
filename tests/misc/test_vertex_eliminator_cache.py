"""Feature pin: `VertexEliminator` shares prefix state across elimination orders.

`VertexEliminator` keeps a tree of `GraphState` nodes keyed by
`(vertex, sp_rules)`. When a second `eliminate(...)` call shares a prefix
with the first, the cached states along that prefix are reused — only the
suffix is re-executed. This is what makes the immutables.Map graph
representation worth the bookkeeping cost.
"""

import jax
import jax.numpy as jnp

from graphax.core import _get_eliminator, _build_graph, _checkify_order


def _make():
    def f(x, y):
        return jnp.sin(x * y).sum()

    args = (jnp.ones(4), jnp.ones(4))
    closed = jax.make_jaxpr(f)(*args)
    return closed.jaxpr, args, closed.literals


def test_eliminator_returns_same_object_per_jaxpr():
    """`_get_eliminator` is `pytree_hash_cache`d — same inputs -> same instance."""
    jaxpr, args, consts = _make()
    e1 = _get_eliminator(jaxpr, args, consts, (0, 1))
    e2 = _get_eliminator(jaxpr, args, consts, (0, 1))
    assert e1 is e2


def test_eliminator_caches_prefix_states():
    """Run twice with the same order — second run hits children at every step."""
    jaxpr, args, consts = _make()
    eliminator = _get_eliminator(jaxpr, args, consts, (0, 1))

    _, _, _, vo_vertices = _build_graph(jaxpr, args, consts, (0, 1))
    order = _checkify_order("rev", jaxpr, vo_vertices)

    # First run populates the cache.
    eliminator.eliminate(order, jaxpr, None, vo_vertices, count_ops=False)

    # Walk the GraphState tree along `order` — every step should be a child.
    node = eliminator.root
    for v in order:
        key = (v, ())  # no sparsity rules
        assert key in node.children, (
            f"vertex {v} not cached after first elimination"
        )
        node = node.children[key]


def test_eliminator_correctness():
    """Second invocation of jacve with the same plan still produces the right Jac."""
    from graphax import jacve, tree_allclose

    def f(x, y):
        return jnp.sin(x * y).sum()

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    jit_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    res1 = jit_fn(x, y)
    res2 = jit_fn(x, y)  # Cache hit
    refres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)
    assert bool(tree_allclose(res1, refres))
    assert bool(tree_allclose(res2, refres))
