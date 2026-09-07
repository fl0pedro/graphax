"""An illegal per-face transform RAISES (ticket dsnn-3qm.70, owner ruling D11).

Before this, `_apply_face_transform` caught the ValueError and returned the
operand unchanged, so a measured plan could differ from the plan the caller
asked for with nothing in the record. On the CPU toy under the static minimum
Markowitz degree order a literal Diag(0, 1) was dropped on every planned face.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from graphax import faces_of, jacve
from graphax.core import FaceTransformIllegal
from graphax.incremental import IncrementalJaxpr
from graphax.sparse.micro_actions import (Diag, action_is_legal,
                                          legal_diag_pairs, quant)

B, DIN, H, V = 4, 8, 8, 16
_KS = jax.random.split(jax.random.PRNGKey(0), 6)
ARGS = (jax.random.normal(_KS[0], (B, DIN)),
        jax.random.normal(_KS[1], (B, V)),
        jax.random.normal(_KS[2], (DIN, H)) * 0.3,
        jax.random.normal(_KS[3], (H,)) * 0.1,
        jax.random.normal(_KS[4], (H, V)) * 0.3)
ARGNUMS = (2, 3, 4)


def loss_fn(x, y, w1, b1, wout):
    h = jnp.tanh(x @ w1 + b1)
    return jnp.mean((h @ wout - y) ** 2)


def _graph():
    cj = jax.make_jaxpr(loss_fn)(*ARGS)
    outv = set(map(id, cj.jaxpr.outvars))
    valid = [i + 1 for i, e in enumerate(cj.jaxpr.eqns)
             if not any(id(o) in outv for o in e.outvars)]
    return cj.jaxpr, cj.literals, valid


JAXPR, CONSTS, VALID = _graph()


def _markowitz():
    ij = IncrementalJaxpr(JAXPR, ARGNUMS, list(CONSTS), list(ARGS),
                          track_faces=False)
    left, order = set(VALID), []
    while left:
        deg = {}
        for v in left:
            var = JAXPR.eqns[v - 1].outvars[0]
            deg[v] = (len([u for u in ij.graph if var in ij.graph[u]])
                      * len(ij.graph.get(var, {})))
        b = min(left, key=lambda v: (deg[v], v))
        order.append(b)
        left.remove(b)
        ij.eliminate(b, (), None)
    return order


ORDER = _markowitz()


def _all_faces(order):
    ij = IncrementalJaxpr(JAXPR, ARGNUMS, list(CONSTS), list(ARGS),
                          track_faces=False)
    out = []
    for v in order:
        for k in faces_of(ij.graph, ij.tgraph, v, JAXPR):
            out.append((v, k))
        ij.eliminate(v, (), None)
    return out


FACES = _all_faces(ORDER)


def _run(plan):
    fn = jacve(loss_fn, list(ORDER), argnums=ARGNUMS,
               sparse_representation=True, transforms=[], face_transforms=plan)
    return jax.jit(fn)(*ARGS)


def test_an_illegal_literal_diag_raises_instead_of_being_skipped():
    """Diag(0, 1) ties two OUT axes, which no diagonal can do. Every planned
    face is illegal, and the run must stop at the first one."""
    plan = {v: {k: (Diag(0, 1, 2), None, None)} for v, k in FACES}
    with pytest.raises(FaceTransformIllegal) as exc:
        _run(plan)
    msg = str(exc.value)
    assert "cannot be applied" in msg
    assert "out/primal" in msg          # names the reason
    assert "action_is_legal" in msg     # tells the caller what to do


def test_the_message_names_the_vertex_and_the_slot():
    v, k = FACES[0]
    plan = {v: {k: (Diag(0, 1, 2), None, None)}}
    with pytest.raises(FaceTransformIllegal) as exc:
        _run(plan)
    assert f"vertex {v}" in str(exc.value)
    assert "slot" in str(exc.value)


def test_a_chooser_that_declines_is_the_legal_way_to_skip():
    """A callable handed the live operand may return None. That is a decline,
    not a fault, and the run completes."""
    seen = []

    def chooser(st):
        seen.append(len(st.dims))
        return None

    plan = {v: {k: (chooser, None, None)} for v, k in FACES}
    out = _run(plan)
    assert len(out) == len(ARGNUMS)
    assert seen, "the chooser was never consulted"


def test_a_chooser_may_pick_a_legal_diag_from_the_mask():
    """`legal_diag_pairs` reads the live operand, so a policy can mask with it.
    Every pair it returns must apply without raising."""
    picked = []

    def chooser(st):
        pairs = legal_diag_pairs(st)
        if not pairs:
            return None
        i, j, f = pairs[0]
        act = Diag(i, j, f)
        assert action_is_legal(st, act), (i, j, f)
        picked.append((i, j, f))
        return act

    plan = {v: {k: (chooser, None, None)} for v, k in FACES}
    out = _run(plan)
    assert len(out) == len(ARGNUMS)
    assert picked, "no legal Diag pair was found on any face"


def test_action_is_legal_agrees_with_the_applier():
    """The predicate and the applier can never disagree: the predicate IS the
    applier, with the exception turned into a bool."""
    agree = []

    def chooser(st):
        act = Diag(0, 1, 2)
        agree.append(action_is_legal(st, act))
        return None

    _run({v: {k: (chooser, None, None)} for v, k in FACES})
    assert agree, "the chooser was never consulted"
    assert not any(agree), "Diag(0, 1) tied two out axes and was called legal"


def test_a_legal_transform_still_applies():
    """Quant fits every operand, so nothing here changes for a legal action."""
    plan = {v: {k: (quant("bfloat16"), None, None)} for v, k in FACES}
    out = _run(plan)
    assert len(out) == len(ARGNUMS)
