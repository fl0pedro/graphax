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

import math
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


def test_a_diagonal_pair_output_passes_the_contract():
    """A full Jacobian (non-scalar target) of an elementwise op is a plain
    diagonal pair: both members point to val axis 0. The contract counts a
    pair's shared axis once, so the tensor is in parameter layout as it is."""
    def f(x):
        return jnp.tanh(x) * 2.0
    x = jnp.arange(4.0) + 0.5
    for engine in ENGINES:
        saved = {k: os.environ.get(k) for k in ENGINES[engine]}
        os.environ.update(ENGINES[engine])
        try:
            out = jax.jit(jacve(f, [1, 2], argnums=(0,), sparse_representation=True))(x)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        t = out[0]
        assert isinstance(t, SparseTensor)
        assert is_parameter_layout(t), (engine, t.dims, t.val.shape)
        np.testing.assert_allclose(np.asarray(t.dense()), np.asarray(jax.jacfwd(f)(x)), rtol=1e-6)


# --- Growing-broadcast census (ticket dsnn-3qm.28.1, deliverable b) --------
# Turns lane B's ad hoc probe (.scratch/trustworthy-approx-search/probes/
# t28b/broadcast_census.py) into an asserted test. That probe wraps
# ``matmul._as_shape`` and counts every ``mode="broadcast"`` call whose
# target has more elements than its input — the exact site
# ``_prepare_contraction_views`` uses to materialize an implicit axis (F2,
# grill2). A whole-program ``jax.make_jaxpr`` walk (as ``analyze_and_smoke_
# test.py``'s helper does for one isolated matmul call) is too blunt here:
# ``loss_fn`` itself broadcasts (the bias-add ``x @ w1 + b1``), and that
# broadcast has nothing to do with the tiled contraction frame. Wrapping
# ``_as_shape`` counts only what the frame itself asks XLA to materialize.
import importlib as _importlib

_mm = _importlib.import_module("graphax.sparse.ops.matmul")
_mm_legacy = _importlib.import_module("graphax.sparse.ops.matmul_legacy_tiled")


def _as_shape_growth_census(order, ft, *, tiled_legacy, lazy_rules="nodemote"):
    saved = {k: os.environ.get(k) for k in
             ("GRAPHAX_TILED_LEGACY", "GRAPHAX_TILED_LAZY",
              "GRAPHAX_EINSUM_GENERAL", "GRAPHAX_PLANNER_EXACT")}
    os.environ["GRAPHAX_TILED_LEGACY"] = "1" if tiled_legacy else "0"
    os.environ["GRAPHAX_TILED_LAZY"] = lazy_rules
    os.environ["GRAPHAX_EINSUM_GENERAL"] = "0"
    os.environ["GRAPHAX_PLANNER_EXACT"] = "0"
    orig_as_shape = _mm._as_shape
    grew = {"calls": 0, "elems": 0}

    def _wrapped(view, target_shape, *, mode):
        out = orig_as_shape(view, target_shape, mode=mode)
        if mode == "broadcast":
            in_n = math.prod(view.shape) if view.shape else 1
            target = tuple(target_shape)
            out_n = math.prod(target) if target else 1
            if out_n > in_n:
                grew["calls"] += 1
                grew["elems"] += out_n - in_n
        return out

    # ``matmul_legacy_tiled.py`` does ``from .matmul import (..., _as_shape,
    # ...)`` — a name binding taken at import time. Patching
    # ``matmul._as_shape`` alone does not touch that already-bound name in
    # the legacy module, so the incumbent path (GRAPHAX_TILED_LEGACY=1)
    # would silently read 0 calls. Patch both module-level names to the same
    # wrapper so either engine's calls are counted.
    _mm._as_shape = _wrapped
    _mm_legacy._as_shape = _wrapped
    try:
        fn = jacve(loss_fn, list(order), argnums=ARGNUMS,
                   sparse_representation=True,
                   transforms=[], face_transforms=ft)
        jax.eval_shape(fn, *ARGS)
    finally:
        _mm._as_shape = orig_as_shape
        _mm_legacy._as_shape = orig_as_shape
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return grew["calls"], grew["elems"]


@pytest.mark.parametrize("order_name", sorted(ORDERS))
@pytest.mark.parametrize("plan", ["exact", "quant_lhs", "compress_lhs"])
def test_lazy_tiled_frame_full_rules_has_zero_growing_broadcasts(order_name, plan):
    """``GRAPHAX_TILED_LAZY=full`` (the demote rule ON, ticket dsnn-3qm.67)
    reaches ZERO growing ``_as_shape(mode="broadcast")`` calls on this MLP
    toy, on both orders and every plan (exact, Quant, Reduce) — the same
    claim T28B-RESULT.md section 3 made for NeuralNetwork/TransformerLM. This
    is the leaner-CPU setting; the LANDED DEFAULT is ``nodemote`` (demote
    OFF, chosen for GPU fusion — see the next test), which does not reach
    zero here. Both are "the lazy frame"; only one env value differs."""
    order = ORDERS[order_name]
    ft = _plans(order)[plan]
    calls, elems = _as_shape_growth_census(order, ft, tiled_legacy=False, lazy_rules="full")
    assert calls == 0, (
        f"{order_name}/{plan}: GRAPHAX_TILED_LAZY=full has {calls} growing "
        f"_as_shape(mode='broadcast') call(s) ({elems} elements grown), expected zero"
    )


@pytest.mark.parametrize("order_name", sorted(ORDERS))
@pytest.mark.parametrize("plan", ["exact", "quant_lhs", "compress_lhs"])
def test_lazy_tiled_frame_default_rules_grow_fewer_than_the_incumbent(order_name, plan):
    """The LANDED DEFAULT (``GRAPHAX_TILED_LAZY=nodemote``, demote OFF — the
    device-dependent tradeoff of T28B-RESULT.md section 4: nodemote keeps the
    GPU fusion, at the cost of the CPU broadcast this test measures) does
    NOT reach zero growing broadcasts on this toy, unlike ``full`` (previous
    test). It IS a strict improvement over the incumbent on every (order,
    plan) cell. Do not read "zero" into this test name — the zero claim
    belongs to ``full``, not to the landed default. See
    findings/62-implicit-axis-small-case.md for the full count table."""
    order = ORDERS[order_name]
    ft = _plans(order)[plan]
    lazy_calls, lazy_elems = _as_shape_growth_census(
        order, ft, tiled_legacy=False, lazy_rules="nodemote")
    legacy_calls, legacy_elems = _as_shape_growth_census(order, ft, tiled_legacy=True)
    assert lazy_calls < legacy_calls, (
        f"{order_name}/{plan}: nodemote (default) has {lazy_calls} growing "
        f"_as_shape(mode='broadcast') call(s) ({lazy_elems} elements grown); "
        f"incumbent has {legacy_calls} ({legacy_elems} elements grown) — "
        "expected the default lazy frame to grow strictly fewer, even though not zero"
    )
