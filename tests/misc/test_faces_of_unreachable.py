"""Unit test for ticket dsnn-3qm.74: faces_of prunes unreachable / None LazyEdges.

faces_of used to list faces whose edges forced to None (stop_gradient, select_n
predicate, etc.), over-reporting faces that _eliminate_vertex never visited.
"""
import jax
import jax.lax as lax
import jax.numpy as jnp
import pytest

from graphax import faces_of
from graphax.incremental import IncrementalJaxpr


def test_stop_gradient_edge_pruned_from_faces_of():
    def f(x):
        y = jnp.sin(x)
        z = lax.stop_gradient(y)
        return z * 2.0

    x = jnp.ones((4,))
    cj = jax.make_jaxpr(f)(x)
    ij = IncrementalJaxpr(cj.jaxpr, [0], list(cj.literals), [x], track_faces=False)

    # Find vertex corresponding to stop_gradient
    sg_v = None
    for i, eqn in enumerate(cj.jaxpr.eqns, 1):
        if eqn.primitive is lax.stop_gradient_p:
            sg_v = i
            break
    assert sg_v is not None, "stop_gradient eqn not found"

    # faces_of at stop_gradient must be empty: no differentiable face passes through it
    keys = faces_of(ij.graph, ij.tgraph, sg_v, cj.jaxpr)
    assert keys == [], f"Expected [] but got {keys}"


def test_select_n_predicate_edge_pruned_from_faces_of():
    def f(pred, a, b):
        return lax.select_n(pred, a, b)

    pred = jnp.array(0, dtype=jnp.int32)
    a = jnp.ones((4,))
    b = jnp.zeros((4,))
    cj = jax.make_jaxpr(f)(pred, a, b)
    ij = IncrementalJaxpr(cj.jaxpr, [0, 1, 2], list(cj.literals), [pred, a, b], track_faces=False)

    # Vertex 1 is select_n
    keys = faces_of(ij.graph, ij.tgraph, 1, cj.jaxpr)
    # The in-edges should only be a and b (indices 1 and 2), NOT pred (index 0)
    # pred is non-differentiable (NO_EDGE)
    in_indices = {k[0] for k in keys}
    assert 0 not in in_indices, f"Non-differentiable pred var (index 0) should not appear in in_edges: {keys}"


def test_every_enumerated_face_is_visited_in_mlp():
    """Verify that in an MLP, every face returned by faces_of has non-None operands."""
    def loss_fn(w, x):
        return jnp.sum(jnp.tanh(x @ w))

    w = jnp.ones((4, 4))
    x = jnp.ones((2, 4))
    cj = jax.make_jaxpr(loss_fn)(w, x)
    ij = IncrementalJaxpr(cj.jaxpr, [0], list(cj.literals), [w, x], track_faces=False)

    for v in range(1, len(cj.jaxpr.eqns) + 1):
        keys = faces_of(ij.graph, ij.tgraph, v, cj.jaxpr)
        # Check that for every key, the edges actually exist and are non-None
        eqn = cj.jaxpr.eqns[v - 1]
        for central_var in eqn.outvars:
            if central_var not in ij.graph:
                continue
            for out_edge, out_e in ij.graph[central_var].items():
                val_out = out_e.value if hasattr(out_e, "value") else out_e
                assert val_out is not None, f"Edge {central_var}->{out_edge} forced to None!"
        ij.eliminate(v, (), None)
