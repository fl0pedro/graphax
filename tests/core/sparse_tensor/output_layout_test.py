"""The output-layout contract of ``jacve`` on a CPU toy (ticket dsnn-3qm.62).

A 2-layer MLP with a scalar MSE loss, eliminated on the static minimum
Markowitz degree order (the fixed order of the campaign) and on the reverse
order, with and without a face transform. Before the contract the tiled engine
returned ``Wout`` with its val transposed (axes (1, 0)) and the deleted planner
did not, so the exact and the approximated gradient had different pytree
structure (finding 60). Now:

  * every returned SparseTensor is in parameter layout (axis == position),
  * the exact and the approximated plan return ONE pytree structure,
  * the exact gradient equals ``jax.grad`` (the one oracle),
  * an approximated gradient equals its own ``sparse_representation=False``
    run. That flag changes the RETURN form only (core.py:453, :3199), so
    this is an output-packing check, not an oracle (grill 2026-09-06).

There is one engine since 2026-09-08 (ticket dsnn-3qm.72), so the tests that
used to run each cell twice run it once.
"""
from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import SKIP_FACE, faces_of, jacve
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
        # The most aggressive thing a face can carry: the whole contraction is
        # dropped. Needed here because it is the one class that can leave a
        # gradient with NO tensor at all (dsnn-3qm.72).
        "skip_face": {v0: {key0: SKIP_FACE}},
    }


def _run(order, ft, sparse):
    fn = jacve(loss_fn, list(order), argnums=ARGNUMS,
               sparse_representation=sparse,
               transforms=[], face_transforms=ft)
    return jax.jit(fn)(*ARGS)


def _dense(out):
    return [np.asarray(t.dense() if isinstance(t, SparseTensor) else t, np.float64)
            for t in out]


def _structure(out):
    return jax.tree_util.tree_structure(out)


@pytest.mark.parametrize("order_name", sorted(ORDERS))
@pytest.mark.parametrize("plan", ["exact", "quant_lhs", "compress_lhs"])
def test_every_returned_gradient_is_in_parameter_layout(order_name, plan):
    order = ORDERS[order_name]
    ft = _plans(order)[plan]
    out = _run(order, ft, True)
    assert len(out) == len(ARGNUMS)
    for t in out:
        assert isinstance(t, SparseTensor)
        assert is_parameter_layout(t), (order_name, plan, t.dims, t.val.shape)
        dense_dims = [d for d in t.dims if d.axis is not None]
        for pos, d in enumerate(t.dims):
            if d.axis is not None and len(dense_dims) == len(t.dims):
                assert d.axis == pos


@pytest.mark.parametrize("order_name", sorted(ORDERS))
@pytest.mark.parametrize("plan", ["exact", "quant_lhs"])
def test_exact_and_approximated_share_one_structure(order_name, plan):
    """The reward path compares the plan's gradient with the same-order exact
    gradient by tree_map; that needs one structure (finding 60's failure)."""
    order = ORDERS[order_name]
    exact = _run(order, None, True)
    appr = _run(order, _plans(order)[plan], True)
    assert _structure(exact) == _structure(appr), (order_name, plan)
    jax.tree_util.tree_map(lambda e, a: None, exact, appr)


@pytest.mark.parametrize("order_name", sorted(ORDERS))
def test_exact_gradient_equals_jax_grad(order_name):
    order = ORDERS[order_name]
    ref = [np.asarray(g, np.float64)
           for g in jax.grad(loss_fn, argnums=ARGNUMS)(*ARGS)]
    for sparse in (True, False):
        got = _dense(_run(order, None, sparse))
        for g, r in zip(got, ref):
            assert g.shape == r.shape
            np.testing.assert_allclose(g, r, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("order_name", sorted(ORDERS))
@pytest.mark.parametrize("plan", ["quant_lhs", "compress_lhs"])
def test_approximated_sparse_equals_its_dense_return_form(order_name, plan):
    order = ORDERS[order_name]
    ft = _plans(order)[plan]
    sp = _dense(_run(order, ft, True))
    dn = _dense(_run(order, ft, False))
    for a, b in zip(sp, dn):
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("order_name", sorted(ORDERS))
@pytest.mark.parametrize("plan", ["exact", "quant_lhs", "compress_lhs",
                                  "skip_face"])
def test_the_logical_gradient_is_recoverable_for_every_plan_class(
        order_name, plan):
    """THE ONLY COMPARISON A CONSUMER MAY MAKE (ticket dsnn-3qm.72).

    ``test_exact_and_approximated_share_one_structure`` above covers exact and
    quant only, and that is not an oversight: a Compress changes ``val``'s rank
    and a SKIP_FACE can leave a gradient with no tensor at all, so the sparse
    return's PYTREE STRUCTURE is not a function of the logical tensor and a
    consumer cannot ``tree_map`` the pair. Measured on nn256/mnist with twelve
    random per-face plans per order: the structures differed in 1 of 12 plans
    on fwd and 11 of 12 on rev once SKIP_FACE was in the family, and comparing
    the raw pytree CHILDREN scored 11 of those 12 plans at the worst possible
    quality while their true grad-cosines ran up to 0.965.

    What IS guaranteed, and what the reward path must therefore use, is this:
    the LOGICAL tensor is always recoverable and always matches the dense
    return form -- for every class, including the two that break the
    structure. ``None`` means a structurally zero gradient, the same thing the
    dense branch spells ``zeros_like``."""
    order = ORDERS[order_name]
    ft = _plans(order)[plan]
    sp = _run(order, ft, True)
    dn = _run(order, ft, False)
    assert len(sp) == len(dn) == len(ARGNUMS)
    for t, d in zip(sp, dn):
        ref = np.asarray(d, np.float64)
        if t is None:
            # the sparse branch's spelling of a structurally zero gradient
            got = np.zeros_like(ref)
        else:
            assert isinstance(t, SparseTensor)
            got = np.asarray(t.dense(), np.float64)
        assert got.shape == ref.shape, (order_name, plan, got.shape, ref.shape)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_a_diagonal_pair_output_passes_the_contract():
    """A full Jacobian (non-scalar target) of an elementwise op is a plain
    diagonal pair: both members point to val axis 0. The contract counts a
    pair's shared axis once, so the tensor is in parameter layout as it is."""
    def f(x):
        return jnp.tanh(x) * 2.0
    x = jnp.arange(4.0) + 0.5
    out = jax.jit(jacve(f, [1, 2], argnums=(0,), sparse_representation=True))(x)
    t = out[0]
    assert isinstance(t, SparseTensor)
    assert is_parameter_layout(t), (t.dims, t.val.shape)
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


def _as_shape_growth_census(order, ft):
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

    _mm._as_shape = _wrapped
    try:
        fn = jacve(loss_fn, list(order), argnums=ARGNUMS,
                   sparse_representation=True,
                   transforms=[], face_transforms=ft)
        jax.eval_shape(fn, *ARGS)
    finally:
        _mm._as_shape = orig_as_shape
    return grew["calls"], grew["elems"]


@pytest.mark.parametrize("order_name", sorted(ORDERS))
@pytest.mark.parametrize("plan", ["exact", "quant_lhs", "compress_lhs"])
def test_the_frame_has_zero_growing_broadcasts(order_name, plan):
    """The contraction frame never grows a buffer to line two operands up.

    An axis one operand does not store is stated in the einsum and summed away
    at extent 1, never broadcast to its partner's extent. Zero on both orders
    and every plan (exact, Quant, Reduce), the same claim T28B-RESULT.md
    section 3 made for NeuralNetwork and TransformerLM.

    This used to hold only under ``GRAPHAX_TILED_LAZY=full``, because a
    ``dot_general`` forced a choice between the broadcast and the GPU fusion.
    The einsum emission does not force it, so the rule is unconditional
    (ticket dsnn-3qm.72)."""
    order = ORDERS[order_name]
    ft = _plans(order)[plan]
    calls, elems = _as_shape_growth_census(order, ft)
    assert calls == 0, (
        f"{order_name}/{plan}: {calls} growing _as_shape(mode='broadcast') "
        f"call(s) ({elems} elements grown), expected zero"
    )
