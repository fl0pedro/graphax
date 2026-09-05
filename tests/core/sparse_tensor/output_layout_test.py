"""The output-layout contract of ``jacve`` on a CPU toy (ticket dsnn-3qm.62).

A 2-layer MLP with a scalar MSE loss, eliminated on the static minimum
Markowitz degree order (the fixed order of the campaign) and on the reverse
order, under BOTH contraction engines (tiled: GRAPHAX_EINSUM_GENERAL=0;
planner: GRAPHAX_EINSUM_GENERAL=1 GRAPHAX_PLANNER_EXACT=1), with and without a
face transform. Before the contract the tiled engine returned ``Wout`` with its
val transposed (axes (1, 0)) and the planner did not, so the exact and the
approximated gradient had different pytree structure (finding 60). Now:

  * every returned SparseTensor is in parameter layout (axis == position),
  * the two engines return the SAME pytree structure for the same plan,
  * the exact gradient equals ``jax.grad`` (the dense oracle, owner Q12),
  * an approximated gradient equals its own ``sparse_representation=False``
    run (the dense oracle for approximations).
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import faces_of, jacve
from graphax.incremental import IncrementalJaxpr
from graphax.sparse.micro_actions import quant, compress
from graphax.sparse.ops.output_layout import is_parameter_layout
from graphax.sparse.tensor import SparseTensor

B, DIN, H, V = 4, 8, 8, 16
KEY = jax.random.PRNGKey(0)
KS = jax.random.split(KEY, 6)
X = jax.random.normal(KS[0], (B, DIN))
Y = jax.random.normal(KS[1], (B, V))
W1 = jax.random.normal(KS[2], (DIN, H)) * 0.3
B1 = jax.random.normal(KS[3], (H,)) * 0.1
WOUT = jax.random.normal(KS[4], (H, V)) * 0.3
ARGS = (X, Y, W1, B1, WOUT)
ARGNUMS = (2, 3, 4)


def loss_fn(x, y, w1, b1, wout):
    h = jnp.tanh(x @ w1 + b1)
    return jnp.mean((h @ wout - y) ** 2)


ENGINES = {
    "tiled": {"GRAPHAX_EINSUM_GENERAL": "0", "GRAPHAX_PLANNER_EXACT": "0"},
    "planner": {"GRAPHAX_EINSUM_GENERAL": "1", "GRAPHAX_PLANNER_EXACT": "1"},
}


def _graph():
    cj = jax.make_jaxpr(loss_fn)(*ARGS)
    jaxpr, consts = cj.jaxpr, cj.literals
    outvars = set(map(id, jaxpr.outvars))
    valid = [i + 1 for i, e in enumerate(jaxpr.eqns)
             if not any(id(o) in outvars for o in e.outvars)]
    return jaxpr, consts, valid


def _markowitz(jaxpr, consts, valid):
    ij = IncrementalJaxpr(jaxpr, ARGNUMS, list(consts), list(ARGS), track_faces=False)
    left = set(valid)
    order = []
    while left:
        deg = {}
        for v in left:
            var = jaxpr.eqns[v - 1].outvars[0]
            preds = [u for u in ij.graph if var in ij.graph[u]]
            succs = list(ij.graph.get(var, {}).keys())
            deg[v] = len(preds) * len(succs)
        best = min(left, key=lambda v: (deg[v], v))
        order.append(best)
        left.remove(best)
        ij.eliminate(best, (), None)
    return order


def _first_face(jaxpr, consts, order):
    ij = IncrementalJaxpr(jaxpr, ARGNUMS, list(consts), list(ARGS), track_faces=False)
    for v in order:
        keys = faces_of(ij.graph, ij.tgraph, v, jaxpr)
        if keys:
            return v, keys[0]
        ij.eliminate(v, (), None)
    raise AssertionError("no live face")


JAXPR, CONSTS, VALID = _graph()
ORDERS = {"markowitz": _markowitz(JAXPR, CONSTS, VALID),
          "reverse": sorted(VALID, reverse=True)}


def _plans(order):
    v0, key0 = _first_face(JAXPR, CONSTS, order)
    return {
        "exact": None,
        "quant_lhs": {v0: {key0: (quant("bfloat16"), None, None)}},
        "compress_lhs": {v0: {key0: (compress("mean", 0), None, None)}},
    }


def _run(order, ft, engine, sparse):
    saved = {k: os.environ.get(k) for k in ENGINES[engine]}
    os.environ.update(ENGINES[engine])
    try:
        fn = jacve(loss_fn, list(order), argnums=ARGNUMS,
                   sparse_representation=sparse,
                   transforms=[], face_transforms=ft)
        return jax.jit(fn)(*ARGS)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _dense(out):
    return [np.asarray(t.dense() if isinstance(t, SparseTensor) else t, np.float64)
            for t in out]


def _structure(out):
    return jax.tree_util.tree_structure(out)


@pytest.mark.parametrize("order_name", sorted(ORDERS))
@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize("plan", ["exact", "quant_lhs", "compress_lhs"])
def test_every_returned_gradient_is_in_parameter_layout(order_name, engine, plan):
    order = ORDERS[order_name]
    ft = _plans(order)[plan]
    out = _run(order, ft, engine, True)
    assert len(out) == len(ARGNUMS)
    for t in out:
        assert isinstance(t, SparseTensor)
        assert is_parameter_layout(t), (order_name, engine, plan, t.dims, t.val.shape)
        dense_dims = [d for d in t.dims if d.axis is not None]
        for pos, d in enumerate(t.dims):
            if d.axis is not None and len(dense_dims) == len(t.dims):
                assert d.axis == pos


@pytest.mark.parametrize("order_name", sorted(ORDERS))
@pytest.mark.parametrize("plan", ["exact", "quant_lhs"])
def test_both_engines_return_the_same_pytree_structure(order_name, plan):
    order = ORDERS[order_name]
    ft = _plans(order)[plan]
    a = _run(order, ft, "tiled", True)
    b = _run(order, ft, "planner", True)
    assert _structure(a) == _structure(b)


@pytest.mark.parametrize("order_name", sorted(ORDERS))
@pytest.mark.parametrize("plan", ["exact", "quant_lhs"])
def test_exact_and_approximated_share_one_structure(order_name, plan):
    """The reward path compares the plan's gradient with the same-order exact
    gradient by tree_map; that needs one structure (finding 60's failure)."""
    order = ORDERS[order_name]
    for engine in ENGINES:
        exact = _run(order, None, engine, True)
        appr = _run(order, _plans(order)[plan], engine, True)
        assert _structure(exact) == _structure(appr), (order_name, engine, plan)
        jax.tree_util.tree_map(lambda e, a: None, exact, appr)


@pytest.mark.parametrize("order_name", sorted(ORDERS))
@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_exact_gradient_equals_jax_grad(order_name, engine):
    order = ORDERS[order_name]
    ref = [np.asarray(g, np.float64)
           for g in jax.grad(loss_fn, argnums=ARGNUMS)(*ARGS)]
    for sparse in (True, False):
        got = _dense(_run(order, None, engine, sparse))
        for g, r in zip(got, ref):
            assert g.shape == r.shape
            np.testing.assert_allclose(g, r, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("order_name", sorted(ORDERS))
@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize("plan", ["quant_lhs", "compress_lhs"])
def test_approximated_sparse_equals_its_dense_oracle(order_name, engine, plan):
    order = ORDERS[order_name]
    ft = _plans(order)[plan]
    sp = _dense(_run(order, ft, engine, True))
    dn = _dense(_run(order, ft, engine, False))
    for a, b in zip(sp, dn):
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
