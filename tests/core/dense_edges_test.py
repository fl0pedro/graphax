"""The DENSE-CONTRACTION mode, ``jacve(..., dense_edges=True)`` (dsnn-3qm.69).

``sparse_representation`` changes the RETURN form only (``core.py:453``,
``core.py:3199``), so the sparse engine compared against its own dense output
packing proves the packing, not the values (finding 61 verdict 4). This module
pins the second engine that IS a value oracle for an approximated plan: every
edge a plain array, every contraction a ``jnp.tensordot`` over the eliminated
variable's axes, Quant a cast, Reduce a mean broadcast back, Diag a zeroing of
everything outside the blocks.

Target: the 2-layer MLP of ``sparse_tensor/output_layout_test.py``, on the static
minimum Markowitz degree order (the fixed order of the campaign) and on the
reverse and forward orders.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import (
    ActionCensus, CensusMismatch, SKIP_FACE, census_plan, compare_censuses,
    jacve,
)
from graphax.core import (
    _build_graph, _checkify_order, _prune_graph, _stable_var_index,
    prune_enabled,
)
from graphax.core import FaceTransformIllegal
from graphax.dense_edges import DenseBudgetExceeded, unwrap, wrap
from graphax.incremental import IncrementalJaxpr
from graphax.sparse.micro_actions import (
    Compress, Diag, Quant, apply_compress, apply_diag, apply_quant,
)
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

# The tolerance of the value comparison, PER CLASS (owner ruling D10). Both are
# MEASURED on this toy, not chosen.
#
# ``TOL_EXACT`` covers the exact, Reduce and Diag classes, where the two engines
# differ only by float32 reduction order. Measured worst 2026-09-06: 1.0e-6, on
# the FORWARD order with Reduce on every face (the longest accumulation chains
# of the three orders); every other case sits at 1e-7 or below. The
# approximation itself is 0.57 relative, so the margin is five orders wide.
#
# QUANT IS NOT AN ABSOLUTE BOUND. bf16 rounds different intermediates on the two
# engines, and WHICH intermediates depends on the machine: the same plan (Quant
# on slot lhs of every face, forward order) reads 0.0 on one CPU and 2.106e-3 on
# pgi15-cpu2 (job 63832). An absolute ceiling is therefore not a property of the
# engines. The comparable quantity is the disagreement as a FRACTION of the
# approximation's own size, and the bound is that the two engines must not
# disagree by MORE than the approximation itself. Measured worst fraction:
# 0.444 locally (Markowitz, both operands narrow) and about 0.68 on pgi15-cpu2
# (forward, slot lhs). A wrong Jacobian sits at 1e-1 to 1 relative, three orders
# above a bf16 Quant's own 3e-3, so this bound still catches one.
TOL_EXACT = 1e-5
QUANT_MARGIN = 1.0


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
    ij = IncrementalJaxpr(jaxpr, ARGNUMS, list(consts), list(ARGS),
                          track_faces=False)
    left, order = set(valid), []
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


JAXPR, CONSTS, VALID = _graph()
ORDERS = {"markowitz": _markowitz(JAXPR, CONSTS, VALID),
          "reverse": sorted(VALID, reverse=True),
          "forward": sorted(VALID)}


def _face_catalog(order):
    """Every face the elimination visits, in visit order.

    One entry per face: ``(vertex, face key, result shape, result out_ndim)``.
    The result shape is ``out_edge.aval.shape + in_edge.aval.shape``, the
    nominal shape of the edge the face stores, which is what a Diag pair has to
    be legal against.

    Built by replaying the graph rewiring on the SHARED helpers, so the keys are
    the ones both engines look up in ``face_transforms``.
    """
    cj = jax.make_jaxpr(loss_fn)(*ARGS)
    jaxpr, consts = cj.jaxpr, cj.literals
    _, graph, tgraph, vo = _build_graph(jaxpr, list(ARGS), list(consts),
                                        tuple(ARGNUMS))
    if prune_enabled():
        _prune_graph(graph, tgraph, jaxpr, tuple(ARGNUMS))
    vidx = _stable_var_index(jaxpr)
    G = {u: dict(graph[u]) for u in graph}
    T = {v: dict(tgraph[v]) for v in tgraph}
    cat = []
    for vertex in _checkify_order(list(order), jaxpr, vo):
        for central in jaxpr.eqns[int(vertex) - 1].outvars:
            if central not in G:
                continue
            for oe in sorted(G[central], key=lambda k: vidx.get(k, 1 << 30)):
                for ie in sorted(T.get(central, {}),
                                 key=lambda k: vidx.get(k, 1 << 30)):
                    cat.append((int(vertex), (vidx.get(ie), vidx.get(oe)),
                                tuple(oe.aval.shape) + tuple(ie.aval.shape),
                                oe.aval.ndim))
                    G.setdefault(ie, {})[oe] = True
                    T.setdefault(oe, {})[ie] = True
            if central not in vo:
                for iv in list(T.get(central, {})):
                    G.get(iv, {}).pop(central, None)
            for ov in list(G[central]):
                T.get(ov, {}).pop(central, None)
            G.pop(central, None)
            if central not in vo:
                T.pop(central, None)
    return cat


CATALOG = {name: _face_catalog(o) for name, o in ORDERS.items()}
REF = [np.asarray(g, np.float64)
       for g in jax.grad(loss_fn, argnums=ARGNUMS)(*ARGS)]


def _run(order, ft, *, dense, sparse=True):
    kw = {"dense_edges": True} if dense else {"sparse_representation": sparse}
    fn = jacve(loss_fn, list(order), argnums=ARGNUMS, transforms=[],
               face_transforms=ft, **kw)
    return jax.jit(fn)(*ARGS)


def _np(out):
    """The gradient leaves as float64 arrays.

    ``sparse_representation=True`` returns ``None`` for a DEAD path (one a SKIP
    cut), where the gradient IS zero -- the dense mode and the dense return form
    both write the zeros out. Materialize the ``None`` so the two are
    comparable, exactly as ``env._gradient_similarity`` does.
    """
    res = []
    for i, t in enumerate(out):
        if t is None:
            res.append(np.zeros(REF[i].shape, np.float64))
        else:
            res.append(np.asarray(
                t.dense() if isinstance(t, SparseTensor) else t, np.float64))
    return res


def _rel(a, b):
    num = math.sqrt(sum(float(np.sum((x - y) ** 2)) for x, y in zip(a, b)))
    den = math.sqrt(sum(float(np.sum(y ** 2)) for y in b))
    return num / max(den, 1e-30)


def _plan(catalog, slots, faces=None):
    """``face_transforms`` for the listed faces, or for every face.

    ``slots`` is either a dict ``slot index -> action`` used on every selected
    face, or a callable ``(index, catalog entry) -> slots tuple or None`` that
    picks per face.
    """
    ft = {}
    for i, (vertex, key, _shape, _ondim) in enumerate(catalog):
        if faces is not None and i not in faces:
            continue
        if callable(slots):
            entry = slots(i, catalog[i])
            if entry is None:
                continue
            ft.setdefault(vertex, {})[key] = entry
        else:
            ft.setdefault(vertex, {})[key] = tuple(
                slots.get(s) for s in range(3))
    return ft


# ---------------------------------------------------------------------------
# the contracts of the mode
# ---------------------------------------------------------------------------
def test_dense_edges_with_sparse_representation_raises():
    """The dense mode has no SparseTensor to return. Silently ignoring the
    requested return form is how a measurement lies, so it raises."""
    with pytest.raises(ValueError, match="dense_edges=True with sparse"):
        jacve(loss_fn, ORDERS["reverse"], argnums=ARGNUMS,
              dense_edges=True, sparse_representation=True)


def test_count_ops_with_dense_edges_raises():
    """The counts would describe the value oracle, not the engine under test."""
    fn = jacve(loss_fn, ORDERS["reverse"], argnums=ARGNUMS,
               dense_edges=True, count_ops=True)
    with pytest.raises(NotImplementedError, match="count_ops"):
        fn(*ARGS)


def test_the_byte_budget_raises_instead_of_filling_the_node():
    """A dense edge is out_size * primal_size numbers, so the mode is a
    small-shape oracle. Over the ceiling it says so."""
    fn = jacve(loss_fn, ORDERS["reverse"], argnums=ARGNUMS,
               dense_edges=True, dense_max_bytes=1024)
    with pytest.raises(DenseBudgetExceeded, match="value oracle"):
        fn(*ARGS)


def test_every_returned_gradient_is_a_plain_array():
    out = _run(ORDERS["markowitz"], None, dense=True)
    assert len(out) == len(ARGNUMS)
    for got, ref in zip(out, REF):
        assert not isinstance(got, SparseTensor)
        assert tuple(got.shape) == tuple(ref.shape)


@pytest.mark.parametrize("order_name", sorted(ORDERS))
def test_the_exact_plan_equals_jax_grad(order_name):
    got = _np(_run(ORDERS[order_name], None, dense=True))
    for g, r in zip(got, REF):
        np.testing.assert_allclose(g, r, rtol=1e-5, atol=1e-6)


def test_the_adapter_round_trips_a_dense_edge():
    arr = jax.random.normal(KS[5], (3, 4, 5))
    st = wrap(arr, 1)
    assert len(st.out_dims) == 1 and len(st.primal_dims) == 2
    np.testing.assert_array_equal(np.asarray(unwrap(st)), np.asarray(arr))


def test_a_rank_zero_edge_keeps_rank_zero_through_the_adapter():
    """``ops.utils._arr2st`` expands a 0-rank array to ``(1,)``; the mode's own
    wrap must not, or the contraction against a scalar-loss seed changes rank."""
    st = wrap(jnp.asarray(2.0), 0)
    assert st.out_dims == () and st.primal_dims == ()
    assert float(np.asarray(unwrap(st))) == 2.0


# ---------------------------------------------------------------------------
# WHY THE Quant TESTS HERE ARE **NOT** EXEMPTED FROM THE SUITE'S MATMUL PIN
# ---------------------------------------------------------------------------
# ``tests/conftest.py`` pins ``jax_default_matmul_precision = "highest"`` on a
# GPU and offers a ``device_matmul_precision`` marker to opt a test out. The
# obvious thing would be to put that marker on the bfloat16 Quant tests below,
# on the theory that a test measuring what a narrow dtype costs should measure it
# on the device's own arithmetic. It was TRIED (2026-09-10) and it is WRONG here,
# for two independent reasons.
#
# 1. ``REF`` above is a ``jax.grad`` evaluated AT IMPORT, i.e. during pytest's
#    collection phase, which the conftest pins. A marker is a per-test fact and
#    cannot reach back into collection, so an opted-out test would compare an
#    UNPINNED engine against a PINNED ground truth. MEASURED: the
#    ``approximation`` figure for (reverse, Quant on slot lhs) then reads
#    1.652e-4 -- TF32 noise between the two precisions -- instead of the honest
#    1.121e-7, so the ``approximation > 1e-4`` precondition starts PASSING for
#    the one reason it exists to rule out. That is worse than no exemption.
#
# 2. The exemption is not needed, because the pin does not touch a Quant.
#    ``highest`` changes how a dot ACCUMULATES; it cannot restore mantissa bits
#    a cast already threw away. MEASURED on an RTX 3090, one 128x256x64 dot with
#    the lhs rounded to bfloat16: 1.6662e-3 from the exact answer unpinned and
#    1.6555e-3 pinned, i.e. the cast's full cost survives and only the 2.1e-4 of
#    tensor-core noise riding on it is removed. Every number this file measures
#    is byte-identical pinned and unpinned, and identical at the pre-pin commit
#    dc1aabc.
#
# What DOES remove the approximation is XLA:GPU's optimizer deleting the dense
# engine's bf16 casts under ``jax.jit``: ``make_jaxpr`` shows all 10
# ``convert_element_type[bfloat16]`` and the pre-optimisation StableHLO 60 bf16
# mentions, while the OPTIMIZED HLO has ZERO bf16 and one fused ``dot``. The same
# program on XLA:CPU keeps 7 bf16 converts and 11 dots, which is why these cases
# passed while the suite was accidentally running on the CPU. See the long note
# in tests/conftest.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the packing check that is NOT an oracle (finding 61 verdict 4)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("order_name", sorted(ORDERS))
def test_sparse_representation_false_is_only_the_output_packing(order_name):
    """Bit-identical to ``sparse_representation=True`` on an APPROXIMATED plan:
    the same contractions packed two ways. This is why the dense mode exists."""
    ft = _plan(CATALOG[order_name], {0: Quant("bfloat16")})
    sp = _np(_run(ORDERS[order_name], ft, dense=False, sparse=True))
    pk = _np(_run(ORDERS[order_name], ft, dense=False, sparse=False))
    for a, b in zip(sp, pk):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# the oracle itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("order_name", sorted(ORDERS))
@pytest.mark.parametrize("cls", ["quant", "reduce"])
def test_an_approximated_plan_agrees_with_the_sparse_engine(order_name, cls):
    """The real check: the same plan, two independent accumulations.

    The census is compared FIRST (literal actions, so it must match), then the
    values. The plan must also be a REAL approximation, i.e. far from jax.grad.
    """
    catalog = CATALOG[order_name]
    action = Quant("bfloat16") if cls == "quant" else Compress((0,), "mean")
    ft = _plan(catalog, {0: action})
    cs, cd = ActionCensus(), ActionCensus()
    sp = _np(_run(ORDERS[order_name], census_plan(ft, cs), dense=False))
    dn = _np(_run(ORDERS[order_name], census_plan(ft, cd), dense=True))
    compare_censuses(cs, cd, site=f"{order_name}/{cls}")
    assert cs.counts() == cd.counts() and sum(cs.counts().values()) > 0
    disagreement, approximation = _rel(dn, sp), _rel(dn, REF)
    assert approximation > 1e-4, "the plan did not approximate anything"
    bound = (QUANT_MARGIN * approximation) if cls == "quant" else TOL_EXACT
    assert disagreement <= bound, (order_name, cls, disagreement, bound)


def _diag_for(entry):
    """The Diag pair to try on one face's result, or None when none fits.

    A Diag ties one OUT axis to one PRIMAL axis with a factor dividing both
    extents, so the pair is read off the face's own nominal result shape.
    """
    _vertex, _key, shape, ondim = entry
    if ondim < 1 or len(shape) <= ondim:
        return None
    factor = math.gcd(int(shape[0]), int(shape[ondim]))
    if factor <= 1:
        return None
    return Diag(0, ondim, factor)


def _both_engines_apply(order, catalog, index, slots):
    """Does this ONE-face plan apply the same actions on both engines?

    Legality is not the same on the two sides even for a LITERAL action: a
    sparse edge already carries diagonal pairs and implicit axes, so
    ``apply_diag`` can raise there, and ``core._apply_face_transform`` then
    SWALLOWS the ValueError and leaves the operand exact. The dense mode raises
    instead (owner ruling D8). So an oracle site is chosen by probing both
    engines, never assumed.
    """
    ft = _plan(catalog, slots, faces={index})
    cs, cd = ActionCensus(), ActionCensus()
    try:
        _run(order, census_plan(ft, cs), dense=False)
        _run(order, census_plan(ft, cd), dense=True)
    except Exception:
        return False
    return cs.counts() == cd.counts() and sum(cs.counts().values()) > 0


@pytest.mark.parametrize("order_name", sorted(ORDERS))
def test_a_diag_plan_agrees_with_the_sparse_engine(order_name):
    """Diag on the contraction result, on every face both engines accept."""
    order = ORDERS[order_name]
    catalog = CATALOG[order_name]
    sites = {}
    for i, entry in enumerate(catalog):
        rule = _diag_for(entry)
        if rule is None:
            continue
        slots = {2: rule}
        if _both_engines_apply(order, catalog, i, slots):
            sites[i] = rule
    if not sites:
        # Finding 59: Diag on the contraction result applies NOWHERE under the
        # reverse order. That is a property of the order, not of this mode.
        pytest.skip(f"no Diag site both engines accept on the {order_name} order")
    ft = _plan(catalog,
               lambda i, _e: (None, None, sites[i]) if i in sites else None)
    cs, cd = ActionCensus(), ActionCensus()
    sp = _np(_run(order, census_plan(ft, cs), dense=False))
    dn = _np(_run(order, census_plan(ft, cd), dense=True))
    compare_censuses(cs, cd, site=f"{order_name}/diag")
    assert sum(cs.counts().values()) == len(sites)
    assert _rel(dn, sp) <= TOL_EXACT, (order_name, _rel(dn, sp))
    assert _rel(dn, REF) > 1e-4, "the plan did not approximate anything"


def test_a_literal_action_the_sparse_engine_cannot_apply_now_raises():
    """The asymmetry this test was written for is GONE at the source.

    ``core._apply_face_transform`` used to swallow a ``ValueError`` and leave
    the operand exact, so a LITERAL Diag could be dropped on the sparse side
    and applied on the dense one, and only the census made that visible. Ticket
    dsnn-3qm.70 makes the sparse side RAISE instead, so the divergence can no
    longer be produced: the run stops at the face the caller got wrong.

    The census stays valuable for the asymmetries that remain (a MASKED hook,
    ticket .69's original finding). What is pinned here is that the silent-skip
    route is closed.
    """
    order = ORDERS["markowitz"]
    catalog = CATALOG["markowitz"]
    for i, entry in enumerate(catalog):
        rule = _diag_for(entry)
        if rule is None:
            continue
        if _both_engines_apply(order, catalog, i, {2: rule}):
            continue
        # A site the two engines used to disagree about. The sparse engine now
        # raises rather than skipping.
        ft = _plan(catalog, {2: rule}, faces={i})
        with pytest.raises(FaceTransformIllegal):
            _run(order, census_plan(ft, ActionCensus()), dense=False)
        return
    pytest.skip("no literal Diag site the sparse engine declines")
def test_a_skipped_face_is_skipped_in_both_engines():
    """SKIP_FACE drops the face's contraction outright. Both engines must agree
    on the (different) gradient that leaves."""
    order = ORDERS["markowitz"]
    vertex, key = CATALOG["markowitz"][0][:2]
    ft = {vertex: {key: SKIP_FACE}}
    sp = _np(_run(order, ft, dense=False))
    dn = _np(_run(order, ft, dense=True))
    assert _rel(dn, sp) <= TOL_EXACT
    assert _rel(dn, REF) > 1e-4, "the SKIP did not drop anything"


def test_the_quant_tolerance_is_the_measured_bound():
    """The Quant bound is MEASURED here, not picked by hand (ruling D10).

    Quant on ONE contraction operand leaves the contraction in f32. Quant on
    BOTH makes the contraction itself narrow, and bf16 then rounds different
    intermediates on the two engines. The disagreement is reported as a fraction
    of the approximation's own size, because the absolute number is a property
    of the machine, not of the engines (see the ``QUANT_MARGIN`` note above).
    """
    q = Quant("bfloat16")
    worst = 0.0
    where = None
    for order_name in sorted(ORDERS):
        catalog = CATALOG[order_name]
        for slots in ({0: q}, {0: q, 1: q}, {0: q, 1: q, 2: q}):
            ft = _plan(catalog, slots)
            cs, cd = ActionCensus(), ActionCensus()
            sp = _np(_run(ORDERS[order_name], census_plan(ft, cs), dense=False))
            dn = _np(_run(ORDERS[order_name], census_plan(ft, cd), dense=True))
            compare_censuses(cs, cd, site="quant bound")
            approximation = _rel(dn, REF)
            assert approximation > 1e-4
            fraction = _rel(dn, sp) / approximation
            if fraction > worst:
                worst, where = fraction, (order_name, sorted(slots))
    assert worst <= QUANT_MARGIN, (worst, where)
    if worst == 0.0:
        pytest.skip("the two engines agreed bit-for-bit on every Quant plan of "
                    "this machine, so the bound is untested here")


def test_quant_keeps_the_narrow_dtype_only_when_both_operands_are_narrow():
    """The ruling of 2026-09-06: the engine never narrows or widens on its own,
    and a contraction of two narrow operands stays narrow."""
    order = ORDERS["markowitz"]
    catalog = CATALOG["markowitz"]
    one = _run(order, _plan(catalog, {0: Quant("bfloat16")}), dense=True)
    both = _run(order, _plan(catalog, {0: Quant("bfloat16"),
                                       1: Quant("bfloat16")}), dense=True)
    assert all(g.dtype == jnp.float32 for g in one)
    assert all(g.dtype == jnp.bfloat16 for g in both)


# ---------------------------------------------------------------------------
# the census: what makes the value comparison legitimate
# ---------------------------------------------------------------------------
def _masked(rule):
    """A stand-in for alphagrad's ``make_live_masked_hook``: apply the rule
    where it is legal on THIS operand, leave the operand exact where it is
    not. The hook that produced the divergence of ticket .69."""
    def hook(st):
        try:
            if isinstance(rule, Compress):
                return apply_compress(st, rule)
            if isinstance(rule, Diag):
                return apply_diag(st, rule)
            return apply_quant(st, rule)
        except ValueError:
            return st
    return hook


@pytest.mark.parametrize("rule,slot", [(Compress((0,), "mean"), 0),
                                       (Diag(0, 2, 4), 2)])
def test_a_masked_hook_that_diverges_is_a_loud_census_failure(rule, slot):
    """THE POINT OF THE CENSUS (ticket .69, owner ruling D8).

    A sparse edge already carries diagonal pairs and implicit axes; a dense edge
    carries none. So a MASKED hook is legal on different faces on the two sides
    and the two runs are DIFFERENT PLANS. Their values must never be compared
    silently: the census comparison has to raise first.
    """
    order = ORDERS["markowitz"]
    ft = _plan(CATALOG["markowitz"], {slot: _masked(rule)})
    cs, cd = ActionCensus(), ActionCensus()
    sp = _np(_run(order, census_plan(ft, cs), dense=False))
    dn = _np(_run(order, census_plan(ft, cd), dense=True))
    # the two hooks really did apply the rule a different number of times
    assert cs.counts() != cd.counts(), (cs.counts(), cd.counts())
    with pytest.raises(CensusMismatch, match="did not apply the same actions"):
        compare_censuses(cs, cd)
    # and the silent failure it prevents: without the census this reads as a
    # value disagreement between the engines, which it is not.
    assert _rel(dn, sp) >= 0.0


def test_the_census_matches_for_a_literal_plan_and_the_values_are_compared():
    """The other half: a plan of literal decoded micro-actions applies the same
    actions on both engines (ruling D8), so the values ARE comparable."""
    order = ORDERS["markowitz"]
    ft = _plan(CATALOG["markowitz"], {0: Compress((0,), "mean")})
    cs, cd = ActionCensus(), ActionCensus()
    sp = _np(_run(order, census_plan(ft, cs), dense=False))
    dn = _np(_run(order, census_plan(ft, cd), dense=True))
    compare_censuses(cs, cd)
    assert cs.counts() == cd.counts() == {"COMPRESS": len(CATALOG["markowitz"])}
    assert _rel(dn, sp) <= TOL_EXACT


def test_the_census_survives_the_two_op_face_form():
    """``((lhs, rhs, new), (jl, jr, jres))`` is wrapped slot by slot, so a
    two-op plan is censused exactly like its flat equivalent."""
    order = ORDERS["markowitz"]
    vertex, key = CATALOG["markowitz"][0][:2]
    q = Quant("bfloat16")
    ft = {vertex: {key: ((q, None, None), (None, None, None))}}
    cs, cd = ActionCensus(), ActionCensus()
    sp = _np(_run(order, census_plan(ft, cs), dense=False))
    dn = _np(_run(order, census_plan(ft, cd), dense=True))
    compare_censuses(cs, cd)
    assert cs.counts() == {"QUANT": 1}
    assert _rel(dn, sp) <= QUANT_MARGIN * max(_rel(dn, REF), TOL_EXACT)


# ---------------------------------------------------------------------------
# the build-and-densify step, on the structural primitives
# ---------------------------------------------------------------------------
KX = jax.random.normal(jax.random.PRNGKey(3), (4, 6))
KW = jax.random.normal(jax.random.PRNGKey(4), (6, 6))

STRUCTURAL = {
    "reshape": lambda x, w: jnp.sum(jnp.tanh((x @ w).reshape(2, 12)) ** 2),
    "transpose": lambda x, w: jnp.sum(jnp.sin((x @ w).T) ** 2),
    "slice": lambda x, w: jnp.sum(jnp.tanh((x @ w)[:, :3]) ** 2),
    "concatenate": lambda x, w: jnp.sum(
        jnp.tanh(jnp.concatenate([x @ w, x], 1)) ** 2),
    "reduce_sum": lambda x, w: jnp.sum(jnp.tanh(jnp.sum(x @ w, axis=0)) ** 2),
    "broadcast": lambda x, w: jnp.sum(
        jnp.tanh(x @ w + jnp.ones((4, 6))) ** 2),
    "repeated_operand": lambda x, w: jnp.sum((x * x) @ w),
}


@pytest.mark.parametrize("name", sorted(STRUCTURAL))
def test_the_drained_elemental_is_the_right_dense_edge(name):
    """The mode densifies each elemental ONCE, after draining the primitives'
    queued relabels. If that drain were wrong the exact plan would not equal
    jax.grad on these targets."""
    fun = STRUCTURAL[name]
    got = jax.jit(jacve(fun, "rev", argnums=(0, 1), dense_edges=True))(KX, KW)
    ref = jax.grad(fun, argnums=(0, 1))(KX, KW)
    for g, r in zip(got, ref):
        np.testing.assert_allclose(np.asarray(g, np.float64),
                                   np.asarray(r, np.float64),
                                   rtol=1e-5, atol=1e-6)


def test_the_default_stays_the_sparse_engine():
    """``dense_edges`` defaults to False and the default path is the sparse one:
    it returns SparseTensors under ``sparse_representation=True``, which the
    dense mode never does."""
    out = _run(ORDERS["markowitz"], None, dense=False, sparse=True)
    assert all(isinstance(t, SparseTensor) for t in out)
