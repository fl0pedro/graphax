"""The vmapped batch axis stays a diagonal pair of every accumulated Jacobian
(dsnn-dfw.140).

The graph is VmappedTransformerLM: the two-block TLM of ``graphax.examples``
under ``jax.vmap`` over x (B, S, D) and y (B, S, V), loss the mean over batch
and positions, differentiated for the weights. The Jacobian of one batched
intermediate with respect to another is a B x B block structure of which only
the diagonal is nonzero; storing it dense is what made four times the batch
cost seven times the memory (probe job 67696).

Three statements. (1) No stored edge of the minimum-Markowitz elimination
carries the batch axis as anything but a diagonal pair. (2) The compiled
program's temporaries grow linearly with B. (3) The Jacobian is exact.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrand
import pytest
from jax._src import core as jcore

from graphax import inline_call_primitives, jacve, tree_allclose
from graphax.core import _force
from graphax.examples.deep_learning import encoder_block, softmax_cross_entropy
from graphax.incremental import IncrementalJaxpr

ARGNUMS = tuple(range(2, 17))


def _tlm(x, y, WQ1, WK1, WV1, W1, b1, g0, be0, WQ2, WK2, WV2, W2, b2, g1, be1, Wout):
    z1 = encoder_block(x, WQ1, WK1, WV1, W1, b1, g0, be0)
    z2 = encoder_block(z1, WQ2, WK2, WV2, W2, b2, g1, be1)
    return softmax_cross_entropy(z2 @ Wout, y)


def vmapped_tlm_loss():
    f = jax.vmap(_tlm, in_axes=(0, 0) + (None,) * 15)
    return lambda *a: jnp.mean(f(*a))


def make_args(key, B, S, D, V):
    ks = jrand.split(key, 6)
    x = jrand.normal(ks[0], (B, S, D))
    y = jax.nn.one_hot(jrand.randint(ks[1], (B, S), 0, V), V)
    sc = 1.0 / jnp.sqrt(jnp.float32(D))

    def blk(k):
        kk = jrand.split(k, 4)
        return [jrand.normal(kk[0], (D, D)) * sc, jrand.normal(kk[1], (D, D)) * sc,
                jrand.normal(kk[2], (D, D)) * sc, jrand.normal(kk[3], (D, D)) * sc,
                jnp.zeros((D,)), jnp.ones((D,)), jnp.zeros((D,))]

    return [x, y, *blk(ks[2]), *blk(ks[3]), jrand.normal(ks[4], (D, V)) * sc]


def traced_inlined(fn, xs):
    cj = jax.make_jaxpr(fn)(*xs)
    jx, consts = inline_call_primitives(cj.jaxpr, cj.literals)
    return jx, list(consts)


def markowitz_order(jaxpr, consts, args):
    # The greedy static minimum Markowitz degree order alphagrad pins a run
    # to (common/order.py): |preds| x |succs| on the current graph, ties to
    # the lowest id, every vertex that is not a pure output.
    ij = IncrementalJaxpr(jaxpr, ARGNUMS, consts, args, track_faces=False)
    eliminable = {i for i, e in enumerate(jaxpr.eqns, 1)
                  if e.outvars[0] not in jaxpr.outvars}
    order = []
    while eliminable:
        scores = {}
        for v in eliminable:
            v_var = jaxpr.eqns[v - 1].outvars[0]
            preds = [u for u in ij.graph if v_var in ij.graph[u]]
            scores[v] = len(preds) * len(ij.graph.get(v_var, {}))
        best = min(scores, key=lambda v: (scores[v], v))
        order.append(best)
        eliminable.remove(best)
        ij.eliminate(best, (), None)
    return order


def _batch_axis_is_dense(t, B) -> bool:
    if getattr(t, "_is_deferred_output", False) or not hasattr(t, "out_dims"):
        return False
    out_b = [d for d in t.out_dims if d.logical_size == B]
    pri_b = [d for d in t.primal_dims if d.logical_size == B]
    if not out_b or not pri_b:
        return False
    for o in out_b:
        if not o.is_sparse or o.block_size not in (None, 1):
            return True
        if not any(p.id == o.other_id for p in pri_b):
            return True
    return False


def test_batch_axis_stays_diagonal_on_every_stored_edge():
    B, S, D, V = 3, 5, 8, 7
    xs = make_args(jrand.PRNGKey(0), B, S, D, V)
    loss = vmapped_tlm_loss()
    jaxpr, consts = traced_inlined(loss, xs)
    order = markowitz_order(jaxpr, consts, xs)
    ij = IncrementalJaxpr(jaxpr, ARGNUMS, consts, xs, track_faces=False)
    dense_edges = []
    seen = {}
    for v in order:
        ij.eliminate(v, (), None)
        with jcore.set_current_trace(ij.trace):
            for u, row in ij.graph.items():
                for w, e in row.items():
                    t = _force(e)
                    if seen.get((u, w)) is t:
                        continue
                    seen[(u, w)] = t
                    if _batch_axis_is_dense(t, B):
                        dense_edges.append((v, jaxpr.eqns[v - 1].primitive.name,
                                            t.out_dims, t.primal_dims))
    assert not dense_edges, dense_edges


def _temp_bytes(loss, xs, order, jaxpr, consts) -> int:
    fn = jacve(loss, order, argnums=ARGNUMS, jaxpr=jaxpr, consts=consts)
    return int(jax.jit(fn).lower(*xs).compile().memory_analysis().temp_size_in_bytes)


def test_markowitz_temporaries_grow_linearly_with_batch():
    S, D, V = 32, 128, 1024
    loss = vmapped_tlm_loss()
    temps = {}
    for B in (1, 4, 32):
        xs = make_args(jrand.PRNGKey(0), B, S, D, V)
        jaxpr, consts = traced_inlined(loss, xs)
        order = markowitz_order(jaxpr, consts, xs)
        temps[B] = _temp_bytes(loss, xs, order, jaxpr, consts)
    assert temps[4] <= 1.2 * 4 * temps[1], temps
    assert temps[32] <= 1.2 * 32 * temps[1], temps


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_vmapped_jacve_equals_jacrev(B, seed):
    S, D, V = 8, 16, 32
    xs = make_args(jrand.PRNGKey(seed), B, S, D, V)
    loss = vmapped_tlm_loss()
    jaxpr, consts = traced_inlined(loss, xs)
    order = markowitz_order(jaxpr, consts, xs)
    ref = jax.jit(jax.jacrev(loss, argnums=ARGNUMS))(*xs)
    for o in (order, "rev"):
        got = jax.jit(jacve(loss, o, argnums=ARGNUMS, jaxpr=jaxpr, consts=consts))(*xs)
        assert tree_allclose(got, ref, atol=1e-5, rtol=1e-4)
