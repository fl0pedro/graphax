"""Feature pin: the append-only jaxpr (AOJ) as a full measurement target.

:class:`graphax.IncrementalJaxpr` (formerly ``IncrementalJacobian``) builds ONE
persistent jax trace whose jaxpr GROWS as vertices are eliminated. It used to
hand back only a DENSE Jacobian and skipped the transform drain that
``graphax.core.vertex_elimination_jaxpr`` performs before it collects outputs.
This module pins the four properties an RL measurement harness needs from it:

1. THE DRAIN — an output edge carrying a queued pre/post Jacobian transform is
   drained (``_drain_transforms(t.copy(), post_first=False)``, the exact call
   ``vertex_elimination_jaxpr`` makes) before it is returned, so the AOJ agrees
   with ``jacve`` instead of silently yielding a different Jacobian.
2. SPARSE BY DEFAULT — ``jacobian_outputs()`` returns ``SparseTensor``s;
   ``dense=True`` opts into the legacy arrays.
3. (VALUE, JACOBIAN) — the primal outputs are materialized too, and
   ``current_jaxpr(include_value=True)`` emits them ahead of the Jacobian with
   an unambiguous ``(outvar, None)`` label.
4. TRUTHFUL, ALWAYS-ON LOGGING — every dispatched micro-action lands in
   ``ij.xlog`` regardless of ``track_faces``, carrying an ``applied`` flag that
   is ``False`` for a numerical no-op; the drains are logged there too.

The headline invariant is EXTENSIONAL EQUIVALENCE: ``jacve(f, order, ...)`` and
the AOJ driven with the SAME order are equal AS FUNCTIONS. They are not the same
object and need not share a signature -- the AOJ's recovered function takes only
the flat arrays.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import graphax
from graphax import IncrementalJacobian, IncrementalJaxpr, faces_of, jacve
from graphax.core import (
    _build_graph, _checkify_order, _drain_transforms, _force,
    _inline_call_primitives,
)
from graphax.sparse.micro_actions import Diag, Quant
from graphax.sparse.tensor import SparseTensor

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
_A = jnp.asarray(np.arange(12, dtype=np.float32).reshape(4, 3) / 11.0)
_W = jnp.asarray(np.arange(20, dtype=np.float32).reshape(5, 4) / 19.0)
_B = jnp.asarray(np.arange(8, dtype=np.float32).reshape(2, 4) / 7.0)
_M = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0 + 0.1)
_W53 = jnp.asarray(np.arange(15, dtype=np.float32).reshape(5, 3) / 14.0 + 0.1)

_X3 = jnp.asarray(np.linspace(0.1, 0.9, 3, dtype=np.float32))
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))
_X5 = jnp.asarray(np.linspace(0.1, 0.9, 5, dtype=np.float32))
_Y4 = jnp.asarray(np.linspace(-0.4, 0.6, 4, dtype=np.float32))
# Helmholtz needs ``1 - sum(x) > 0`` or its log goes NaN.
_XH = jnp.asarray(np.array([0.05, 0.1, 0.15, 0.2], dtype=np.float32))


def _helmholtz(x):
    return x * jnp.log(x / (1.0 + -jnp.sum(x)))


def _fanout(x):
    """ONE in-edge, TWO out-edges -> a vertex with two faces."""
    e = _A @ x
    return _W @ e, _B @ e


def _mlp2in(x, y):
    h = jnp.tanh(_M @ x + y)
    return jnp.sum(h * h), _M @ h


def _mixed(x):
    a = jnp.sin(x)
    b = jnp.exp(a)
    c = a * b
    return jnp.sum(c) * c


def _sliced(x):
    """A SLICE on the input path leaves a queued ``pre_transform`` on the output
    edge -- the drain case. Without the drain the "Jacobian" comes out at the
    head-sliced shape ``(5, 3)`` instead of the true ``(5, 5)``."""
    return _W53 @ x[1:4]


def _structural(x):
    """The reshape/matmul pair whose ``rhs`` face operand is a STRUCTURAL
    (``val is None``) Jacobian -- ``apply_quant`` returns it unchanged, so a
    quant there is a genuine no-op."""
    return jnp.reshape(jnp.sin(x), (2, 2)) @ jnp.ones((2, 2), jnp.float32)


CASES = [
    ("helmholtz", _helmholtz, (_XH,), (0,)),
    ("fanout", _fanout, (_X3,), (0,)),
    ("mlp2in", _mlp2in, (_X4, _Y4), (0, 1)),
    ("mixed", _mixed, (_X4,), (0,)),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _make(fn, args, argnums, track_faces=False):
    """Build an :class:`IncrementalJaxpr` over the SAME jaxpr ``jacve`` uses
    (``make_jaxpr`` + call-primitive inlining)."""
    flat, _ = jax.tree_util.tree_flatten(args)
    closed = jax.make_jaxpr(fn)(*flat)
    jaxpr, consts = _inline_call_primitives(closed.jaxpr, closed.literals)
    ij = IncrementalJaxpr(jaxpr, argnums, list(consts), list(args),
                          track_faces=track_faces)
    return jaxpr, consts, ij


def _numeric_order(jaxpr, consts, args, order):
    """``order`` ("fwd"/"rev"/a sequence) as the numeric order ``jacve`` runs."""
    _, _, _, vo = _build_graph(jaxpr, list(args), list(consts))
    return [int(v) for v in _checkify_order(order, jaxpr, vo)]


def _drive(fn, args, argnums, order, face_transforms=None, transforms=None,
           track_faces=False):
    """Eliminate every vertex of ``order``; ``face_transforms`` is a callable
    ``(ij, vertex) -> map | None`` invoked immediately before each vertex (the
    only point at which its face keys are valid)."""
    jaxpr, consts, ij = _make(fn, args, argnums, track_faces)
    tmap = {int(v): tuple(ts) for v, ts in (transforms or ())}
    for v in _numeric_order(jaxpr, consts, args, order):
        ft = face_transforms(ij, v) if face_transforms else None
        ij.eliminate(v, tmap.get(v, ()), ft)
    return ij


def _recover(ij, args, include_value=False):
    """The AOJ's recovered FUNCTION applied to ``args`` -> list of arrays."""
    jaxpr, consts, labels = ij.current_jaxpr(include_value=include_value)
    outs = jax.core.eval_jaxpr(jaxpr, consts, *args)
    return [np.asarray(o) for o in outs], labels


def _jacve_flat(fn, args, argnums, order, transforms=None):
    out = jacve(fn, order, argnums=argnums, transforms=transforms)(*args)
    return [np.asarray(l) for l in jax.tree_util.tree_leaves(out)]


def _relerr(got, ref):
    got = np.asarray(got, np.float64)
    ref = np.asarray(ref, np.float64)
    assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
    scale = np.abs(ref).max()
    diff = np.abs(got - ref).max()
    return float(diff / scale) if scale > 0 else float(diff)


ORDERS = ["fwd", "rev"]


# ---------------------------------------------------------------------------
# 0. naming / exports
# ---------------------------------------------------------------------------
def test_incremental_jaxpr_is_exported_under_both_names():
    """The class was renamed ``IncrementalJacobian`` -> ``IncrementalJaxpr``;
    alphagrad still imports the old name, so BOTH must be exported and be the
    same object."""
    assert IncrementalJacobian is IncrementalJaxpr
    assert graphax.IncrementalJaxpr is IncrementalJaxpr
    assert graphax.IncrementalJacobian is IncrementalJaxpr
    assert IncrementalJaxpr.__name__ == "IncrementalJaxpr"
    # faces_of is the companion enumerator and must stay importable too.
    assert graphax.faces_of is faces_of


# ---------------------------------------------------------------------------
# 1. THE HEADLINE INVARIANT: jacve == AOJ as FUNCTIONS
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,fn,args,argnums", CASES,
                         ids=[c[0] for c in CASES])
@pytest.mark.parametrize("order", ORDERS)
def test_aoj_is_extensionally_equal_to_jacve(name, fn, args, argnums, order):
    """``jacve(f, order, argnums=...)`` and the AOJ driven with the SAME order
    agree on the same inputs. They are NOT the same object -- the AOJ's
    recovered function takes only the flat arrays."""
    ref = _jacve_flat(fn, args, argnums, order)
    ij = _drive(fn, args, argnums, order)
    got, labels = _recover(ij, args)

    assert len(got) == len(ref) == len(labels)
    for g, r in zip(got, ref):
        assert _relerr(g, r) <= 1e-6


@pytest.mark.parametrize("order", [[1, 2, 3, 4, 5], [5, 3, 1, 2, 4],
                                   [3, 1, 5, 4, 2]])
def test_aoj_equals_jacve_under_custom_orders(order):
    """Not just fwd/rev: an arbitrary full permutation agrees too."""
    ref = _jacve_flat(_helmholtz, (_XH,), (0,), order)
    ij = _drive(_helmholtz, (_XH,), (0,), order)
    got, _ = _recover(ij, (_XH,))
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert _relerr(g, r) <= 1e-6


@pytest.mark.parametrize("name,fn,args,argnums", CASES,
                         ids=[c[0] for c in CASES])
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("micro", [Quant(dtype="bfloat16"), Diag(0, 1, 2)],
                         ids=["quant", "diag"])
def test_aoj_equals_jacve_with_approximations(name, fn, args, argnums, order,
                                              micro):
    """Same equivalence with an APPROXIMATION applied: the per-vertex transform
    spec ``jacve`` accepts, driven identically through both paths."""
    spec = [(3, [micro])]
    ref = _jacve_flat(fn, args, argnums, order, transforms=spec)
    ij = _drive(fn, args, argnums, order, transforms=spec)
    got, _ = _recover(ij, args)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert _relerr(g, r) <= 1e-6


@pytest.mark.parametrize("name,fn,args,argnums", CASES,
                         ids=[c[0] for c in CASES])
@pytest.mark.parametrize("order", ORDERS)
def test_aoj_equals_reference_with_per_face_approximations(
        name, fn, args, argnums, order):
    """PER-FACE approximations: ``jacve`` has no per-face API, so the reference
    is ``core.vertex_elimination_jaxpr`` with the same ``face_transforms``
    injected at the same vertex, and the AOJ must reproduce it."""
    import graphax.core as gcore

    target = 3
    micro = Quant(dtype="bfloat16")

    def _slots(keys):
        return {k: (None, None, micro) for k in keys}

    jaxpr, consts, _ = _make(fn, args, argnums)
    orig = gcore._eliminate_vertex

    def _patched(vertex, jx, graph, tgraph, vo, count_ops=False,
                 transforms=(), face_transforms=None):
        ft = (_slots(faces_of(graph, tgraph, int(vertex), jx))
              if int(vertex) == target else None)
        return orig(vertex, jx, graph, tgraph, vo, count_ops,
                    transforms=transforms, face_transforms=ft)

    gcore._eliminate_vertex = _patched
    try:
        out = gcore.vertex_elimination_jaxpr(
            jaxpr, order, consts, *args, argnums=argnums,
            fresh_eliminator=True)
    finally:
        gcore._eliminate_vertex = orig
    ref = [np.asarray(l) for l in jax.tree_util.tree_leaves(out)]

    ij = _drive(fn, args, argnums, order,
                face_transforms=lambda b, v: (_slots(b.faces(v))
                                              if v == target else None))
    got, _ = _recover(ij, args)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert _relerr(g, r) <= 1e-6


# ---------------------------------------------------------------------------
# 2. THE DRAIN
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("order", ORDERS)
def test_output_edge_with_queued_transform_is_drained(order):
    """REGRESSION. ``_sliced``'s output edge carries a queued ``pre_transform``
    (the slice embed). Draining it is what grows the head-sliced free axis back
    to the full input size -- WITHOUT the drain the AOJ returns a ``(5, 3)``
    array where the true Jacobian is ``(5, 5)``."""
    args, argnums = (_X5,), (0,)
    ij = _drive(_sliced, args, argnums, order)

    # the edge really does carry a queued transform (else this pins nothing)
    queued = 0
    for ov in ij.jaxpr.outvars:
        for ii in ij.argnums:
            inner = ij.graph.get(ij.jaxpr.invars[ii])
            edge = inner.get(ov) if inner is not None else None
            t = _force(edge) if edge is not None else None
            if t is not None:
                queued += len(t.pre_transforms) + len(t.post_transforms)
    assert queued > 0, "model no longer exercises the drain"

    ref = _jacve_flat(_sliced, args, argnums, order)
    got, _ = _recover(ij, args)
    assert [g.shape for g in got] == [r.shape for r in ref]
    assert got[0].shape == (5, 5)
    for g, r in zip(got, ref):
        assert _relerr(g, r) <= 1e-6


def test_drain_is_recorded_in_the_transform_log():
    """Each output-edge drain lands in the always-on log as a ``kind="drain"``
    record naming WHAT was drained and WHICH edge it was on."""
    ij = _drive(_sliced, (_X5,), (0,), "rev")
    assert ij.drains_recorded() == []          # nothing until outputs are taken
    ij.jacobian_outputs()

    drains = ij.drains_recorded()
    assert len(drains) == 1
    d = drains[0]
    assert d.kind == "drain"
    assert d.atype == "DRAIN"
    assert d.slot == "output"
    assert d.vertex is None
    assert d.applied is True
    # WHAT was drained: a non-empty pre/post transform list, by class name.
    assert d.params["post_first"] is False
    assert len(d.params["pre"]) + len(d.params["post"]) > 0
    # WHICH edge: the (invar, outvar) pair.
    assert d.in_edge is ij.jaxpr.invars[0]
    assert d.out_edge is ij.jaxpr.outvars[0]


def test_no_drain_recorded_when_no_transforms_are_queued():
    """A clean output edge is not reported as drained (truthful record)."""
    ij = _drive(_helmholtz, (_XH,), (0,), "rev")
    ij.jacobian_outputs()
    assert ij.drains_recorded() == []


def test_drain_does_not_mutate_the_live_graph():
    """The builder stays live after outputs are taken: the drain works on a
    ``copy()`` and is NOT written back, so repeated calls (and any further
    elimination) see the same un-drained edges and agree."""
    ij = _drive(_sliced, (_X5,), (0,), "rev")
    first, _ = ij.jacobian_outputs(dense=True)
    second, _ = ij.jacobian_outputs(dense=True)
    res = ij.trace.to_jaxpr(list(first) + list(second), ij.dbg, ij.si)
    outs = [np.asarray(o) for o in jax.core.eval_jaxpr(res[0], res[1], _X5)]
    n = len(first)
    for a, b in zip(outs[:n], outs[n:]):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 3. SPARSE BY DEFAULT
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,fn,args,argnums", CASES,
                         ids=[c[0] for c in CASES])
def test_jacobian_outputs_are_sparse_by_default(name, fn, args, argnums):
    """Intermediates live in the graph as ``SparseTensor``s; the accessor keeps
    them that way. ``dense=True`` is the opt-in for arrays."""
    ij = _drive(fn, args, argnums, "rev")
    sparse_outs, sparse_labels = ij.jacobian_outputs()
    dense_outs, dense_labels = ij.jacobian_outputs(dense=True)

    assert sparse_outs, "model produced no Jacobian edges"
    assert all(isinstance(t, SparseTensor) for t in sparse_outs)
    assert not any(isinstance(t, SparseTensor) for t in dense_outs)
    assert sparse_labels == dense_labels


@pytest.mark.parametrize("name,fn,args,argnums", CASES,
                         ids=[c[0] for c in CASES])
def test_sparse_and_dense_outputs_agree_numerically(name, fn, args, argnums):
    """Densifying the sparse result reproduces the dense result exactly."""
    ij = _drive(fn, args, argnums, "rev")
    sparse_outs, _ = ij.jacobian_outputs()
    dense_outs, _ = ij.jacobian_outputs(dense=True)
    probes = list(dense_outs)
    import jax._src.core as jcore
    with jcore.set_current_trace(ij.trace):
        probes += [t.dense() for t in sparse_outs]
    res = ij.trace.to_jaxpr(probes, ij.dbg, ij.si)
    outs = [np.asarray(o) for o in jax.core.eval_jaxpr(res[0], res[1], *args)]
    n = len(dense_outs)
    for d, s in zip(outs[:n], outs[n:]):
        assert d.shape == s.shape
        np.testing.assert_allclose(d, s, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. (VALUE, JACOBIAN)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,fn,args,argnums", CASES,
                         ids=[c[0] for c in CASES])
def test_value_and_jacobian_match_jax(name, fn, args, argnums):
    """The primal outputs are the function's own outputs (``jax.jvp``'s primal)
    and the Jacobian is ``jax.jacrev``'s."""
    ij = _drive(fn, args, argnums, "rev")
    got, labels = _recover(ij, args, include_value=True)

    n_val = sum(1 for _, a in labels if a is None)
    values, jacs = got[:n_val], got[n_val:]

    primal_ref, _ = jax.jvp(fn, tuple(args), tuple(jnp.zeros_like(a)
                                                  for a in args))
    primal_ref = [np.asarray(l) for l in jax.tree_util.tree_leaves(primal_ref)]
    assert len(values) == len(primal_ref)
    for v, r in zip(values, primal_ref):
        assert _relerr(v, r) <= 1e-6

    jac_ref = jax.jacrev(fn, argnums=argnums)(*args)
    jac_ref = [np.asarray(l) for l in jax.tree_util.tree_leaves(jac_ref)]
    assert len(jacs) == len(jac_ref)
    for g, r in zip(jacs, jac_ref):
        assert _relerr(g, r) <= 1e-6


def test_value_labels_are_unambiguous_and_ordered_first():
    """``(outvar, None)`` marks a VALUE output, ``(outvar, argnum)`` a Jacobian;
    with ``include_value=True`` every value precedes every Jacobian."""
    ij = _drive(_mlp2in, (_X4, _Y4), (0, 1), "rev")
    _, _, labels = ij.current_jaxpr(include_value=True)

    kinds = [a is None for _, a in labels]
    assert kinds == sorted(kinds, reverse=True), "values must come first"
    n_val = sum(kinds)
    assert n_val == len(ij.jaxpr.outvars)
    assert [ov for ov, _ in labels[:n_val]] == list(ij.jaxpr.outvars)
    assert all(a in ij.argnums for _, a in labels[n_val:])


def test_current_jaxpr_default_is_unchanged():
    """Backward compat: the default ``current_jaxpr()`` still returns the
    3-tuple ``(jaxpr, consts, labels)`` with JACOBIAN-ONLY outputs and
    ``(outvar, argnum)`` labels."""
    ij = _drive(_fanout, (_X3,), (0,), "rev")
    res = ij.current_jaxpr()
    assert isinstance(res, tuple) and len(res) == 3
    jaxpr, consts, labels = res
    assert len(jaxpr.outvars) == len(labels)
    assert all(a is not None for _, a in labels)
    ref = _jacve_flat(_fanout, (_X3,), (0,), "rev")
    got = [np.asarray(o) for o in jax.core.eval_jaxpr(jaxpr, consts, _X3)]
    for g, r in zip(got, ref):
        assert _relerr(g, r) <= 1e-6


def test_value_and_jacobian_accessor_labels_cover_both_lists():
    vals, jacs, labels = _drive(_fanout, (_X3,), (0,), "rev").value_and_jacobian()
    assert len(labels) == len(vals) + len(jacs)
    assert all(a is None for _, a in labels[:len(vals)])
    assert all(isinstance(t, SparseTensor) for t in jacs)


# ---------------------------------------------------------------------------
# 5. TRUTHFUL, ALWAYS-ON LOGGING
# ---------------------------------------------------------------------------
def _quant_at(vertex, slot, dtype):
    idx = {"lhs": 0, "rhs": 1, "res": 2}[slot]

    def _build(ij, v):
        if v != vertex:
            return None
        ft = {}
        for k in ij.faces(v):
            slots = [None, None, None]
            slots[idx] = Quant(dtype=dtype)
            ft[k] = tuple(slots)
        return ft
    return _build


@pytest.mark.parametrize("track_faces", [False, True])
def test_face_transforms_are_logged_even_without_track_faces(track_faces):
    """The record is ALWAYS-ON: ``track_faces=False`` still logs the per-face
    micro-actions (the FaceSink is opt-in, the TransformLog is not)."""
    ij = _drive(_helmholtz, (_XH,), (0,), "rev",
                face_transforms=_quant_at(3, "res", "int8"),
                track_faces=track_faces)
    recs = [r for r in ij.transform_records() if r.kind == "transform"]
    assert len(recs) == 1
    r = recs[0]
    assert r.atype == "QUANT"
    assert r.params == {"dtype": "int8"}
    assert r.slot == "res"
    assert r.vertex == 3
    assert r.applied is True
    assert r.end > r.start                      # it emitted equations
    assert r.in_edge is not None and r.out_edge is not None
    # the FaceSink still only exists when asked for
    assert (ij.face_sink is not None) is track_faces


@pytest.mark.parametrize("track_faces", [False, True])
def test_noop_quant_on_structural_edge_is_not_recorded_as_applied(track_faces):
    """``apply_quant`` returns its input UNCHANGED when ``val is None``, so
    quantising a STRUCTURAL edge changes nothing. The log must say so.

    ``_structural``'s vertex 1 under the FORWARD order has the broadcast/matmul
    structural Jacobian in its ``rhs`` slot -- a ``val is None`` tensor."""
    ij = _drive(_structural, (_X4,), (0,), "fwd",
                face_transforms=_quant_at(1, "rhs", "int8"),
                track_faces=track_faces)
    recs = [r for r in ij.transform_records() if r.kind == "transform"]
    assert len(recs) == 1, "the micro-action was dispatched exactly once"
    assert recs[0].atype == "QUANT"
    assert recs[0].applied is False, "a val-is-None quant changes nothing"
    assert recs[0].start == recs[0].end, "and it emits no equations"
    # applied_only filtering hides it, which is the point of the flag
    assert ij.xlog.transforms(applied_only=True) == []
    if track_faces:
        # ... and no empty ``approx`` block is rendered for it
        assert sum(len(f.approx) for f in ij.all_faces()) == 0


@pytest.mark.parametrize("track_faces", [False, True])
def test_noop_quant_to_same_dtype_is_not_recorded_as_applied(track_faces):
    """The other documented ``apply_quant`` no-op: the target dtype already
    matches, so the tensor is returned unchanged."""
    ij = _drive(_helmholtz, (_XH,), (0,), "rev",
                face_transforms=_quant_at(3, "res", "float32"),
                track_faces=track_faces)
    recs = [r for r in ij.transform_records() if r.kind == "transform"]
    assert len(recs) == 1
    assert recs[0].applied is False
    if track_faces:
        assert sum(len(f.approx) for f in ij.all_faces()) == 0


def test_noop_transform_does_not_change_the_jacobian():
    """Cross-check that ``applied=False`` is TRUE: the resulting Jacobian is
    byte-identical to the un-approximated one."""
    plain = _drive(_structural, (_X4,), (0,), "fwd")
    noop = _drive(_structural, (_X4,), (0,), "fwd",
                  face_transforms=_quant_at(1, "rhs", "int8"))
    a, _ = _recover(plain, (_X4,))
    b, _ = _recover(noop, (_X4,))
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_per_vertex_transforms_are_logged_with_the_vertex_slot():
    """The per-vertex ``transforms`` site is logged too, tagged ``slot="vertex"``
    and carrying the edge it was applied to."""
    ij = _drive(_helmholtz, (_XH,), (0,), "rev",
                transforms=[(3, [Quant(dtype="bfloat16")])])
    recs = [r for r in ij.transform_records() if r.kind == "transform"]
    assert recs, "per-vertex transform was not logged"
    assert {r.slot for r in recs} == {"vertex"}
    assert all(r.vertex == 3 for r in recs)
    assert all(r.applied for r in recs)


def test_step_transform_records_are_scoped_to_their_step():
    """``step_transform_records(i)`` slices the log to step ``i`` only."""
    ij = _drive(_helmholtz, (_XH,), (0,), "rev",
                face_transforms=_quant_at(3, "res", "int8"))
    per_step = [ij.step_transform_records(i) for i in range(len(ij.steps))]
    assert sum(len(s) for s in per_step) == len(
        [r for r in ij.transform_records() if r.kind == "transform"])
    hit = [i for i, s in enumerate(per_step) if s]
    assert len(hit) == 1
    assert per_step[hit[0]][0].vertex == 3


# ---------------------------------------------------------------------------
# 6. the "early-vertex quant looks like a no-op" report
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("vertex,expect_change", [(1, False), (2, False),
                                                  (3, True), (4, True),
                                                  (5, True)])
def test_early_vertex_quant_is_lossless_not_lost(vertex, expect_change):
    """DOCUMENTS a reported "the quant is applied but never reaches the compiled
    Jacobian" for Helmholtz vertices 1 and 2.

    It is NOT lost -- it is EXACT. Those vertices' face results are the
    CONSTANT structural Jacobian ``-1`` (``reduce_sum`` -> all-ones, ``neg`` ->
    -1, ``add 1.0`` -> +1). Symmetric per-tensor int8 quantization of a tensor
    whose entries share one magnitude is bit-exact: ``s = absmax/127``,
    ``round(val/s) = ±127`` and ``±127 * s == val``. So an unchanged Jacobian
    (cosine similarity exactly 1.0) is the CORRECT answer there, and the log
    honestly reports ``applied=True`` -- the cast really happened.

    From vertex 3 on the edge is data-dependent, the cast is lossy, and the
    Jacobian moves. See also
    ``test_early_vertex_quant_is_constant_folded_so_flops_are_unchanged``.
    """
    args, argnums = (_XH,), (0,)
    base = _drive(_helmholtz, args, argnums, "fwd")
    ref, _ = _recover(base, args)

    ij = _drive(_helmholtz, args, argnums, "fwd",
                face_transforms=_quant_at(vertex, "res", "int8"))
    got, _ = _recover(ij, args)

    recs = [r for r in ij.transform_records() if r.kind == "transform"]
    assert len(recs) == 1 and recs[0].applied is True, (
        "the cast genuinely fires and genuinely changes the tensor")

    changed = not np.array_equal(got[0], ref[0])
    assert changed is expect_change
    # it stays a Jacobian of the right shape and within int8's error budget
    assert got[0].shape == ref[0].shape
    assert _relerr(got[0], ref[0]) <= 5e-2


def test_early_vertex_quant_is_constant_folded_so_flops_are_unchanged():
    """The second half of the report: unchanged compiled FLOPS for vertices 1
    and 2. The pre-quant edge there does not depend on the input (it is the
    constant ``-1``), so XLA constant-folds the whole quantize/dequantize chain
    away. The equations ARE emitted into the jaxpr -- they simply cost nothing.
    """
    args, argnums = (_XH,), (0,)

    def _flops(vertex):
        ij = _drive(_helmholtz, args, argnums, "fwd",
                    face_transforms=(_quant_at(vertex, "res", "int8")
                                     if vertex else None))
        jaxpr, consts, _ = ij.current_jaxpr()
        fn = jax.jit(lambda a: jax.core.eval_jaxpr(jaxpr, consts, a))
        ca = fn.lower(*args).compile().cost_analysis()
        if isinstance(ca, list):
            ca = ca[0]
        return ca.get("flops"), len(jaxpr.eqns)

    base_flops, base_eqns = _flops(None)
    v1_flops, v1_eqns = _flops(1)
    v3_flops, _ = _flops(3)

    assert v1_eqns > base_eqns, "the quant equations ARE in the jaxpr"
    assert v1_flops == base_flops, "but they constant-fold to nothing"
    assert v3_flops > base_flops, "a data-dependent quant does cost flops"
