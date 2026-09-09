"""The NOMINAL-SHAPE invariant, and what an APPROXIMATION does to it.

``core.py`` asserts, for a stored edge, that

    SparseTensor.shape == out_edge.aval.shape + in_edge.aval.shape

``SparseTensor.shape`` is ``tuple(d.logical_size for d in self.dims)`` -- an
ORDERED tuple -- so the assert is a statement about dim ORDER as well as about
extents. The assert is currently gated off whenever an approximation is armed::

    if not _perpath and not _is_approx_cfg and not approx_active():

with the comment "an approximation can leave an edge sparse/permuted".

MEASURED (2026-09-09, probe ``t93_nominal.py``; nn256 =
``VmappedNeuralNetwork``/mnist and TLM = ``TransformerLM``/wikitext at the
campaign shapes, minimum-Markowitz order, THREE random ``Diag`` + THREE
``Compress`` + THREE ``Quant`` per sample on distinct vertices and faces, plus
a variant that adds three ``SKIP_FACE``; 100 nn256 and 46 TLM armed samples,
every stored edge censused by wrapping ``core._set_inner`` and every operand
of the gated assert recorded at the assert's own site):

  * class (c) -- genuinely different extents, or a different rank -- NEVER
    occurred. Not once, on either target, in any plan.
  * class (b) -- a PERMUTATION of nominal -- does occur, but it occurs in
    EXACT AD too, at the SAME vertices with the SAME shapes, and the whole
    census is bit-identical between the exact and the armed run of the same
    order (nn256: 22 (a) + 6 (b) per run, both ways; TLM: 130 (a) + 1 (b) per
    run, both ways). It is dim-ORDER drift -- ticket .62, the same defect as
    the transposed parameter gradients -- not something an approximation does.
  * at the assert's OWN site (the merge branch) both operands were class (a)
    in 1506 recorded armed checks across both targets, and 110 armed samples
    run with the gate FORCED OPEN raised nothing.
  * an approximation CAN change how many edges are stored -- ``SKIP_FACE``
    drops contractions, and ``GRAPHAX_FACTORED_OUTPUTS`` defers an output
    edge in exact AD that is not deferred once the approx config is armed.
    So the pin below is "no NEW non-nominal pattern", not "the same counts".

So the three approximation classes leave the LOGICAL shape alone, exactly as
the API documents: ``Compress`` sets ``axis=None`` and keeps ``size``,
``Diag`` re-factors meta against block, ``Quant`` touches only ``val.dtype``.
The tests below pin both halves of that: the per-class shape preservation
(hand-built, must pass) and the residual (b) drift (xfail on .62).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import graphax.core as gxcore
from graphax import faces_of, inline_call_primitives, jacve
from graphax.core import _checkify_order
from graphax.incremental import IncrementalJaxpr
from graphax.sparse.indexes import DenseIndex
from graphax.sparse.micro_actions import (
    Compress, Diag, Quant, apply_compress, apply_diag, apply_quant)
from graphax.sparse.tensor import SparseTensor


# ===========================================================================
# PART 1 -- hand-built tensors: no approximation class changes .shape
# ===========================================================================
def _nominal_edge(out_shape, primal_shape, dtype=jnp.float32):
    """A dense, fully explicit edge in NOMINAL dim order.

    ``val`` carries one physical axis per logical dim, in the same order, so
    ``shape == out_shape + primal_shape`` by construction -- the state every
    stored edge is supposed to be in.
    """
    dims = list(out_shape) + list(primal_shape)
    val = jnp.asarray(
        np.arange(math.prod(dims), dtype=np.float32).reshape(dims) / 7.0 + 0.1
    ).astype(dtype)
    out_dims = tuple(DenseIndex(i, n, i) for i, n in enumerate(out_shape))
    primal_dims = tuple(
        DenseIndex(len(out_shape) + i, n, len(out_shape) + i)
        for i, n in enumerate(primal_shape)
    )
    return SparseTensor(out_dims, primal_dims, val)


def test_nominal_edge_helper_is_nominal():
    """Guard on the fixture itself -- everything below reads against it."""
    st = _nominal_edge((4, 12), (8,))
    assert st.shape == (4, 12, 8)
    assert st.val.shape == (4, 12, 8)


def test_diag_refactors_meta_against_block_and_keeps_the_logical_shape():
    """``Diag`` moves extent between ``size`` and ``block_size``; the PRODUCT
    -- ``logical_size`` -- is what ``.shape`` reports, and it does not move.

    12 x 8 with factor 4 becomes a meta-4 pair carrying 3 x 2 blocks: the
    storage is block-diagonal, the logical extents are still 12 and 8.
    """
    st = _nominal_edge((12,), (8,))
    assert st.shape == (12, 8)

    out = apply_diag(st, Diag(i=0, j=1, factor=4))

    assert out.shape == (12, 8), (
        f"Diag changed the LOGICAL shape: {st.shape} -> {out.shape}")
    i, j = out.dims
    assert i.is_sparse and j.is_sparse, "Diag must produce a paired index"
    assert i.size == j.size == 4, "the meta grid is the requested factor"
    assert i.size * (i.block_size or 1) == 12
    assert j.size * (j.block_size or 1) == 8


def test_compress_drops_the_axis_pointer_and_keeps_the_logical_size():
    """``Compress`` is IMPLICIT STORAGE, not a rank reduction: the reduced dim
    keeps its ``logical_size`` and only loses its pointer into ``val``."""
    st = _nominal_edge((4,), (6, 5))
    assert st.shape == (4, 6, 5)

    out = apply_compress(st, Compress(axes=(1,), kind="mean"))

    assert out.shape == (4, 6, 5), (
        f"Compress changed the LOGICAL shape: {st.shape} -> {out.shape}")
    assert out.dims[1].axis is None, "the compressed dim must go implicit"
    assert out.val.ndim == st.val.ndim - 1, "one physical axis is gone"
    # the surviving pointers are shifted down, never left dangling
    assert all(d.axis is None or d.axis < out.val.ndim for d in out.dims)


def test_quant_changes_only_the_dtype():
    st = _nominal_edge((4,), (6,))
    out = apply_quant(st, Quant(dtype="bfloat16"))

    assert out.shape == st.shape, (
        f"Quant changed the LOGICAL shape: {st.shape} -> {out.shape}")
    assert out.val.dtype == jnp.bfloat16
    assert [(d.id, d.size, d.axis, d.block_size) for d in out.dims] == \
           [(d.id, d.size, d.axis, d.block_size) for d in st.dims]


@pytest.mark.parametrize(
    "action",
    [
        Diag(i=0, j=2, factor=2),
        Diag(i=1, j=2, factor=3),
        Compress(axes=(0,), kind="mean"),
        Compress(axes=(2,), kind="abs_max"),
        Compress(axes=(0, 2), kind="max"),
        Quant(dtype="bfloat16"),
        Quant(dtype="float16"),
    ],
    ids=lambda a: repr(a),
)
def test_no_approximation_class_changes_the_logical_shape(action):
    """The owner's position, as one parametrized statement: whatever the
    policy picks, ``SparseTensor.shape`` is invariant under it."""
    st = _nominal_edge((4, 6), (12,))
    before = st.shape

    apply = {Diag: apply_diag, Compress: apply_compress,
             Quant: apply_quant}[type(action)]
    out = apply(st, action)

    assert out.shape == before, (
        f"{action!r} changed the LOGICAL shape {before} -> {out.shape}")


# ===========================================================================
# PART 2 -- the (b) pattern, hand-built
# ===========================================================================
# The engine census found exactly one violating pattern, at BOTH targets and
# with or without approximation: an edge out of a SCALAR output (``out_dims``
# empty, so nominal is just ``in_edge.aval.shape``) whose two primal dims are
# listed in REVERSED order, while ``val`` and every ``axis`` pointer stay in
# nominal order. Verbatim from nn256/mnist, exact AD, vertex 7:
#
#   nominal (10, 63)   stored (63, 10)   val (10, 63)
#   dims: id=0 logical=63 axis=1 dense
#         id=1 logical=10 axis=0 dense
#
# The tensor is internally CONSISTENT -- the data is nominal, only the dim
# LIST is transposed -- which is why nothing downstream notices and why the
# defect surfaces as transposed parameter gradients (.62).
def _drifted_edge():
    """The nn256 vertex-7 edge, rebuilt by hand from the recorded dims."""
    val = jnp.asarray(
        np.arange(10 * 63, dtype=np.float32).reshape(10, 63) / 101.0)
    return SparseTensor(
        out_dims=(),
        primal_dims=(DenseIndex(0, 63, 1), DenseIndex(1, 10, 0)),
        val=val,
    )


def test_drifted_edge_carries_nominal_data():
    """The drift is BOOKKEEPING: ``val`` is in nominal order and every dim
    points at the right physical axis. Only the dim list is transposed."""
    st = _drifted_edge()
    assert st.val.shape == (10, 63)                      # nominal order
    assert [d.axis for d in st.dims] == [1, 0]           # pointers agree
    assert [d.logical_size for d in st.dims] == [63, 10]  # list is reversed


@pytest.mark.xfail(
    strict=True,
    reason="dsnn-3qm.62 -- dim-ORDER drift: the primal dims of a "
           "scalar-output edge are listed transposed, so SparseTensor.shape "
           "is a PERMUTATION of out_edge.aval.shape + in_edge.aval.shape. "
           "Present in exact AD as well as under approximation; do NOT "
           "weaken this to a multiset comparison.",
)
def test_stored_edge_shape_is_nominal_ordered():
    """core.py's own invariant, on the recorded drifted edge."""
    nominal = () + (10, 63)          # out_edge.aval.shape + in_edge.aval.shape
    assert _drifted_edge().shape == nominal


def test_drift_is_a_permutation_and_never_a_different_extent():
    """Classification pin: the drift is class (b), never class (c).

    A multiset mismatch or a rank change would mean an approximation lost
    metadata -- that is the serious bug, and it was never observed.
    """
    nominal = (10, 63)
    stored = _drifted_edge().shape
    assert sorted(stored) == sorted(nominal), "class (c): extents were lost"
    assert len(stored) == len(nominal), "class (c): the rank changed"
    assert stored != nominal, "this fixture is supposed to be drifted"


# ===========================================================================
# PART 3 -- the engine: an approximation does not move the census
# ===========================================================================
_W1 = jnp.asarray(np.arange(48, dtype=np.float32).reshape(6, 8) / 47.0 - 0.4)
_W2 = jnp.asarray(np.arange(24, dtype=np.float32).reshape(4, 6) / 23.0 - 0.3)
_X = jnp.asarray(np.arange(40, dtype=np.float32).reshape(5, 8) / 39.0 - 0.5)
_Y = jnp.asarray(np.eye(4, dtype=np.float32)[np.array([0, 1, 2, 3, 0])])
_ARGNUMS = [0, 1]


def _mlp_batch_xent(W1, W2, x, y):
    """The smallest model that reproduces the (b) drift: a VMAPPED two-layer
    MLP under a batch-meaned cross-entropy, i.e. the nn256 shape in miniature.
    Unbatched, or without the softmax/mean head, every stored edge is (a)."""
    logits = jax.vmap(lambda xi: W2 @ jnp.tanh(W1 @ xi))(x)
    return jnp.mean(jnp.sum(-(y * jnp.log(jax.nn.softmax(logits, axis=-1))),
                            axis=-1))


_ARGS = (_W1, _W2, _X, _Y)


def _classify(nominal, stored):
    if stored == nominal:
        return "a"
    if sorted(stored) == sorted(nominal):
        return "b"
    return "c"


# Captured once, at import: ``_store_census`` may run twice inside one test
# and must never wrap its own wrapper (that would double-count every store).
_ORIG_SET_INNER = gxcore._set_inner


def _store_census(monkeypatch, face_transforms=None, order="rev"):
    """Every edge stored by one elimination, classified against nominal.

    ``core._set_inner(graph, in_edge, out_edge, val)`` is immediately followed
    by the mirrored write into the transpose graph with the SAME object and
    the keys swapped; only the first of each pair is counted, so ``k1`` is
    always the in_edge and ``k2`` the out_edge.
    """
    orig = _ORIG_SET_INNER
    recs = []
    last = [None]

    def _wrapped(outer, k1, k2, v):
        same = (id(v) == last[0])
        last[0] = id(v)
        if not same:
            nominal = (tuple(int(n) for n in k2.aval.shape)
                       + tuple(int(n) for n in k1.aval.shape))
            stored = tuple(int(d.logical_size) for d in v.dims)
            recs.append((_classify(nominal, stored), nominal, stored))
        return orig(outer, k1, k2, v)

    monkeypatch.setattr(gxcore, "_set_inner", _wrapped)
    # Every edge is stored while the elimination is TRACED, so tracing is
    # enough -- no XLA compile per census (the sweep below runs one per face).
    jax.make_jaxpr(jacve(_mlp_batch_xent, order, argnums=_ARGNUMS,
                         face_transforms=face_transforms))(*_ARGS)
    return recs


def _counts(recs):
    out = {"a": 0, "b": 0, "c": 0}
    for cls, _, _ in recs:
        out[cls] += 1
    return out


def _all_faces(order_name="rev"):
    """Every ``(vertex, face_key)`` of the order, enumerated the way the env
    does: on a replay that has already eliminated the earlier vertices, since
    a face key is only valid on the graph its vertex is eliminated from.

    ``jacve`` inlines call primitives before it eliminates, so the vertex
    numbering a face key is built against has to come from the INLINED jaxpr
    or the key names a different edge pair.
    """
    closed = jax.make_jaxpr(_mlp_batch_xent)(*_ARGS)
    jaxpr, consts = inline_call_primitives(closed.jaxpr, closed.literals)
    order = _checkify_order(order_name, jaxpr, set())
    ij = IncrementalJaxpr(jaxpr, tuple(_ARGNUMS), list(consts), list(_ARGS))
    out = []
    for v in order:
        v = int(v)
        out += [(v, k) for k in faces_of(ij.graph, ij.tgraph, v, jaxpr)]
        ij.eliminate(v, (), None)
    return out


def _pick_diag(shape):
    """The first index pair of ``shape`` admitting a NON-TRIVIAL block split.

    ``Diag.factor == 1`` is a documented no-op, so a fixed ``factor=2`` would
    silently approximate nothing on an operand whose axes are coprime.
    """
    for i in range(len(shape)):
        for j in range(i + 1, len(shape)):
            g = math.gcd(int(shape[i]), int(shape[j]))
            if g > 1:
                return Diag(i=i, j=j, factor=g)
    return None


def _best_effort(action, log):
    """A per-face hook that applies ``action`` when it fits the operand and
    passes the operand through untouched when it does not -- the same
    drop-on-illegal semantics the policy's live mask has. ``log`` records
    whether the approximation actually landed, so a test cannot pass by
    approximating nothing.

    ``action`` may be the string ``"diag"``, which resolves against the
    operand's own shape at hook time (see :func:`_pick_diag`).
    """
    def _hook(st):
        live = _pick_diag(st.shape) if action == "diag" else action
        if live is None:
            log.append(("skipped", tuple(st.shape)))
            return st
        apply = {Diag: apply_diag, Compress: apply_compress,
                 Quant: apply_quant}[type(live)]
        try:
            out = apply(st, live)
        except Exception:
            log.append(("skipped", tuple(st.shape)))
            return st
        log.append(("applied", repr(live), tuple(st.shape), tuple(out.shape)))
        return out

    return _hook


def _bad_patterns(recs):
    """The DISTINCT non-nominal (nominal, stored) pairs of a census."""
    return {(nominal, stored) for cls, nominal, stored in recs if cls != "a"}


# ``GRAPHAX_FACTORED_OUTPUTS`` changes WHICH edges get stored -- with it on,
# the final contraction onto a pure output head is deferred as a factor pair
# (``DeferredOutputProduct``) and stored through a different site, and that
# deferral is disabled once an approximation is armed. So the two settings do
# not store the same NUMBER of edges, and an equality of raw counts between
# the exact and the armed run is only true with it off (the suite leaks it on
# from tests/core/factored_outputs_test.py, which is how this was found).
# Both settings are pinned here, against the claim that survives either:
# an approximation introduces no NEW non-nominal pattern.
@pytest.mark.parametrize("factored", ["0", "1"], ids=["plain", "factored"])
@pytest.mark.parametrize(
    "action",
    [Quant(dtype="bfloat16"), Compress(axes=(0,), kind="mean"), "diag"],
    ids=["quant", "compress", "diag"],
)
def test_approximation_introduces_no_new_non_nominal_edge(
        monkeypatch, action, factored):
    """THE measurement, as a regression pin.

    ``Compress`` keeps the logical size behind an implicit dim, ``Diag``
    re-factors meta against block, ``Quant`` moves only the dtype -- so no
    approximation may put an edge into the graph at a shape exact AD would
    not also have stored it at. Class (c) may never appear at all.

    Swept over EVERY face of the model, all three slots (``lhs``/``rhs``/
    ``res``) hooked, and checked at every face where the approximation
    actually landed -- a face that declines it proves nothing.
    """
    monkeypatch.setenv("GRAPHAX_FACTORED_OUTPUTS", factored)
    exact_recs = _store_census(monkeypatch)
    exact_bad = _bad_patterns(exact_recs)
    assert _counts(exact_recs)["c"] == 0

    landed = 0
    for v, key in _all_faces():
        log = []
        hook = _best_effort(action, log)
        ft = {v: {key: (hook, hook, hook)}}
        armed_recs = _store_census(monkeypatch, face_transforms=ft)
        if not any(e[0] == "applied" for e in log):
            continue
        landed += 1
        counts = _counts(armed_recs)
        assert counts["c"] == 0, (
            f"{action!r} on face {key} of vertex {v} stored a class-(c) edge "
            f"-- an extent or a rank was LOST: "
            f"{[r for r in armed_recs if r[0] == 'c']}")
        new = _bad_patterns(armed_recs) - exact_bad
        assert not new, (
            f"{action!r} on face {key} of vertex {v} introduced a "
            f"non-nominal stored shape exact AD never produces: {new} "
            f"(exact patterns {exact_bad}, hook log {log})")

    assert landed, (
        f"{action!r} never landed on ANY of the {len(_all_faces())} faces -- "
        f"the sweep approximated nothing, so it would be vacuous")


@pytest.mark.parametrize("factored", ["0", "1"], ids=["plain", "factored"])
def test_no_stored_edge_ever_loses_an_extent(monkeypatch, factored):
    """Class (c) is the serious bug -- an approximation dropping metadata.
    It was never observed on either campaign target, exact or armed."""
    monkeypatch.setenv("GRAPHAX_FACTORED_OUTPUTS", factored)
    fts = [None]
    for v, key in _all_faces():
        for act in (Quant(dtype="bfloat16"), Compress(axes=(0,), kind="mean"),
                    "diag"):
            hook = _best_effort(act, [])
            fts.append({v: {key: (hook, hook, hook)}})
    for transforms in fts:
        recs = _store_census(monkeypatch, face_transforms=transforms)
        bad = [r for r in recs if r[0] == "c"]
        assert not bad, f"class (c) stored edge(s): {bad}"


@pytest.mark.xfail(
    strict=True,
    reason="dsnn-3qm.62 -- dim-ORDER drift: this model stores one edge whose "
           "shape is a PERMUTATION of nominal, in EXACT AD. Deleting the "
           "approximation gate on core.py's nominal-shape assert does not "
           "expose this edge (it never reaches the merge branch), but the "
           "invariant as written is what should hold at the store.",
)
def test_every_stored_edge_matches_nominal_in_exact_ad(monkeypatch):
    # Pinned OFF: with factored outputs on, the drifted edge is stored through
    # the deferred-output site instead, and a leaked setting would make this
    # strict xfail non-deterministic.
    monkeypatch.setenv("GRAPHAX_FACTORED_OUTPUTS", "0")
    recs = _store_census(monkeypatch)
    bad = [r for r in recs if r[0] != "a"]
    assert not bad, f"{len(bad)} of {len(recs)} stored edges are not nominal: {bad}"
