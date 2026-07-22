"""Feature pin: PER-FACE (per local path) Jacobian transforms.

A vertex elimination contracts one FACE per ``in_edge -> central_var ->
out_edge`` path. The per-vertex ``transforms`` argument is applied uniformly to
every one of those faces, so a policy that must choose an approximation PER
PATH cannot express its choice. ``_eliminate_vertex(..., face_transforms=...)``
adds that: a mapping ``(vidx[in_edge], vidx[out_edge]) -> (lhs, rhs, res)``
whose slots are named after the local path ``res = op(lhs, rhs)`` --
``lhs`` is the in_edge Jacobian, ``rhs`` the out_edge Jacobian (both transformed
BEFORE the contraction) and ``res`` the contraction result (transformed at the
per-vertex transform site, immediately after it).

:func:`graphax.faces_of` enumerates a vertex's face keys BEFORE it is
eliminated, in exactly the order the elimination visits them, so the policy's
choice loop can be unrolled statically.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import faces_of
from graphax.core import _stable_var_index
from graphax.incremental import IncrementalJacobian
from graphax.sparse.micro_actions import Compress, Diag, Quant

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
# Distinct, non-square shapes throughout so each of lhs / rhs / res carries a
# DIFFERENT shape and cannot be confused for another slot.
_A = jnp.asarray(np.arange(12, dtype=np.float32).reshape(4, 3) / 11.0)
_W = jnp.asarray(np.arange(20, dtype=np.float32).reshape(5, 4) / 19.0)
_B = jnp.asarray(np.arange(8, dtype=np.float32).reshape(2, 4) / 7.0)

_X3 = jnp.asarray(np.linspace(0.1, 0.9, 3, dtype=np.float32))
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))
_Y4 = jnp.asarray(np.linspace(-0.4, 0.6, 4, dtype=np.float32))


def _fanout(x):
    """ONE in-edge, TWO out-edges -> a vertex with two faces that share their
    ``lhs`` operand. Per-face control is only distinguishable from per-EDGE or
    per-vertex control on a model like this."""
    e = _A @ x                       # vertex 1, central ``e``: (4,) <- x (3,)
    return _W @ e, _B @ e            # two consumers of ``e``


def _grid(x, y):
    """TWO in-edges x TWO out-edges -> a vertex with four faces, for the
    enumeration-order pin."""
    u = jnp.sin(x)
    v = jnp.cos(y)
    e = u * v                        # vertex 3, central ``e``
    return _W @ e, _B @ e


_M = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0 + 0.1)
_P = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 13.0 + 0.2)
_Q = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 17.0 + 0.3)


def _square(x):
    """Same fan-out as ``_fanout`` but square throughout, so a ``Diag`` block
    split actually FITS the operands (4 = 2 x 2)."""
    e = _M @ x
    return _P @ e, _Q @ e


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _build(fn, args, argnums, track_faces=True):
    closed = jax.make_jaxpr(fn)(*args)
    ij = IncrementalJacobian(closed.jaxpr, argnums, list(closed.literals),
                             list(args), track_faces=track_faces)
    return closed.jaxpr, ij


def _run(fn, args, argnums, face_transforms=None, vertex=1, track_faces=True):
    """Eliminate every vertex in order, feeding ``face_transforms`` to
    ``vertex`` only. Returns ``(jaxpr, ij, step)`` where ``step`` indexes that
    vertex's entry in ``ij.steps``.

    ``face_transforms`` may be a callable ``(ij, jaxpr) -> map``; it is invoked
    IMMEDIATELY before the target vertex is eliminated, which is the only point
    where the face keys are valid -- eliminating an earlier vertex rewires the
    graph and therefore changes which faces the target vertex has.
    """
    jaxpr, ij = _build(fn, args, argnums, track_faces)
    step = None
    for v in range(1, len(jaxpr.eqns) + 1):
        if v != vertex:
            ij.eliminate(v)
            continue
        ft = (face_transforms(ij, jaxpr) if callable(face_transforms)
              else face_transforms)
        step = len(ij.steps)
        ij.eliminate(v, (), ft)
    return jaxpr, ij, step


def _eval(ij, args):
    """Materialize + evaluate the Jacobian jaxpr built so far."""
    jaxpr, consts, _ = ij.current_jaxpr()
    return [np.asarray(v) for v in jax.core.eval_jaxpr(jaxpr, consts, *args)]


def _jacobians(fn, args, argnums):
    """Reference Jacobians (``jax.jacrev``), flattened to match
    ``jacobian_outputs`` order: outvar-major, argnum-minor."""
    ref = jax.jacrev(fn, argnums=argnums)(*args)
    return [np.asarray(per_arg) for per_out in ref for per_arg in per_out]


def _recorded_keys(ij, jaxpr, step=0):
    """The face keys the elimination ACTUALLY visited, from the FaceSink."""
    vidx = _stable_var_index(jaxpr)
    return [(vidx[fr.in_edge], vidx[fr.out_edge]) for fr in ij.step_faces(step)]


def _scale(k):
    """A callable slot transform scaling its operand by ``k``.

    Scales through ``scalar_mult`` -- the representation-agnostic knob -- so it
    works on a structural (``val is None``) operand too.
    """
    def _apply(st):
        return st.copy(
            scalar_mult=st.scalar_mult * jnp.asarray(k, st.scalar_mult.dtype))
    return _apply


def _record(slot, log):
    """A callable slot transform that records the operand it was handed."""
    def _apply(st):
        log.append((slot, tuple(st.shape)))
        return st
    return _apply


# ---------------------------------------------------------------------------
# (a) per-face targeting
# ---------------------------------------------------------------------------
def test_face_transform_hits_exactly_one_face_of_two():
    """A Quant on ONE face key shows up in that face's FaceSink record and in
    no other -- even though both faces share the same in-edge."""
    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        assert len(keys) == 2
        return {keys[1]: (None, None, Quant("bfloat16"))}

    jaxpr, ij, step = _run(_fanout, (_X3,), (0,), _ft, vertex=1)
    keys = _recorded_keys(ij, jaxpr, step)
    approx = [fr.approx for fr in ij.step_faces(step)]

    assert len(keys) == 2
    assert approx[0] == [], "untargeted face must stay exact"
    assert len(approx[1]) == 1, "targeted face must carry exactly one approx"
    atype, params, start, end = approx[1][0]
    assert atype == "QUANT"
    assert params == {"dtype": "bfloat16"}
    assert end > start, "the recorded approx range must cover real equations"


def test_face_transform_hits_exactly_one_face_of_four():
    """Same, on a 2x2 (two in-edges x two out-edges) vertex: only the chosen
    (in_edge, out_edge) PAIR is approximated, not the whole row or column."""
    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 3, jaxpr)
        assert len(keys) == 4
        return {keys[2]: (None, None, Quant("bfloat16"))}

    _, ij, step = _run(_grid, (_X4, _Y4), (0, 1), _ft, vertex=3)
    approx = [fr.approx for fr in ij.step_faces(step)]

    assert len(approx) == 4
    assert [len(a) for a in approx] == [0, 0, 1, 0]
    assert approx[2][0][0] == "QUANT"


def test_untargeted_faces_stay_numerically_exact():
    """Approximating one face perturbs only the output that face feeds."""
    ref = _jacobians(_fanout, (_X3,), (0,))

    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        return {keys[1]: (None, None, Quant("bfloat16"))}

    _, ij, _ = _run(_fanout, (_X3,), (0,), _ft, vertex=1)
    out = _eval(ij, (_X3,))

    assert np.array_equal(out[0], ref[0]), "untargeted path must be untouched"
    assert not np.array_equal(out[1], ref[1]), "targeted path must be approximated"
    assert np.allclose(out[1], ref[1], atol=1e-2), "bfloat16 is still close"


# ---------------------------------------------------------------------------
# (b) enumeration order
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fn,args,argnums,vertex,n_faces",
    [(_fanout, (_X3,), (0,), 1, 2),
     (_grid, (_X4, _Y4), (0, 1), 3, 4)],
)
def test_faces_of_matches_open_face_order(fn, args, argnums, vertex, n_faces):
    """``faces_of`` returns the face keys in EXACTLY the order the elimination
    opens them (captured through the FaceSink)."""
    jaxpr, ij = _build(fn, args, argnums, track_faces=True)
    # the keys are only valid for the graph as it stands RIGHT BEFORE this
    # vertex is eliminated, so drain the predecessors first
    for v in range(1, vertex):
        ij.eliminate(v)

    predicted = faces_of(ij.graph, ij.tgraph, vertex, jaxpr)
    assert len(predicted) == n_faces

    step = len(ij.steps)
    ij.eliminate(vertex)
    actual = _recorded_keys(ij, jaxpr, step=step)

    assert predicted == actual


def test_faces_of_is_out_edge_major():
    """The nested product is out_edge-major / in_edge-minor, both ordered by
    first appearance in the jaxpr."""
    jaxpr, ij = _build(_grid, (_X4, _Y4), (0, 1), track_faces=True)
    keys = faces_of(ij.graph, ij.tgraph, 3, jaxpr)

    outs = [k[1] for k in keys]
    ins = [k[0] for k in keys]
    assert outs == sorted(outs), "out_edge index must be non-decreasing (major)"
    assert outs[0] == outs[1] and outs[2] == outs[3]
    assert ins[0] < ins[1] and ins[2] < ins[3], "in_edge index cycles (minor)"
    assert keys[:2] != keys[2:]


def test_faces_of_is_read_only():
    """Enumerating must not force a lazy edge or grow the graph -- the caller
    may call it many times before deciding."""
    jaxpr, ij = _build(_grid, (_X4, _Y4), (0, 1), track_faces=True)
    before_g = {k: set(v) for k, v in ij.graph.items()}
    before_t = {k: set(v) for k, v in ij.tgraph.items()}
    n_eqns = ij._n_eqns()

    first = faces_of(ij.graph, ij.tgraph, 3, jaxpr)
    second = faces_of(ij.graph, ij.tgraph, 3, jaxpr)

    assert first == second
    assert {k: set(v) for k, v in ij.graph.items()} == before_g
    assert {k: set(v) for k, v in ij.tgraph.items()} == before_t
    assert ij._n_eqns() == n_eqns, "enumeration must not emit equations"


def test_faces_of_works_without_face_tracking():
    """The keys do not depend on the FaceSink -- a policy may run untracked."""
    jaxpr_t, ij_t = _build(_grid, (_X4, _Y4), (0, 1), track_faces=True)
    jaxpr_u, ij_u = _build(_grid, (_X4, _Y4), (0, 1), track_faces=False)

    assert (faces_of(ij_t.graph, ij_t.tgraph, 3, jaxpr_t)
            == faces_of(ij_u.graph, ij_u.tgraph, 3, jaxpr_u))


def test_incremental_faces_helper_matches_faces_of():
    jaxpr, ij = _build(_fanout, (_X3,), (0,), track_faces=True)
    assert ij.faces(1) == faces_of(ij.graph, ij.tgraph, 1, jaxpr)


def test_faces_of_is_public():
    import graphax
    assert graphax.faces_of is faces_of


# ---------------------------------------------------------------------------
# (c) the lhs / rhs / res slots reach their intended operand
# ---------------------------------------------------------------------------
def test_slots_receive_the_expected_operands():
    """Every slot of ``_fanout`` carries a distinct shape, so the recorded log
    pins each slot to exactly one operand:

        face 0:  lhs = d(e)/d(x)  (4, 3)
                 rhs = d(o0)/d(e) (5, 4)      res = d(o0)/d(x) (5, 3)
        face 1:  lhs = d(e)/d(x)  (4, 3)      <- the SAME in-edge Jacobian
                 rhs = d(o1)/d(e) (2, 4)      res = d(o1)/d(x) (2, 3)

    ``rhs`` and ``res`` differ per face while ``lhs`` repeats, which is exactly
    the fan-out topology -- and it is visited face by face, not slot by slot.
    """
    log = []

    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        return {k: (_record(f"lhs{i}", log), _record(f"rhs{i}", log),
                    _record(f"res{i}", log))
                for i, k in enumerate(keys)}

    _run(_fanout, (_X3,), (0,), _ft, vertex=1)

    assert log == [
        ("lhs0", (4, 3)), ("rhs0", (5, 4)), ("res0", (5, 3)),
        ("lhs1", (4, 3)), ("rhs1", (2, 4)), ("res1", (2, 3)),
    ]


def test_lhs_slot_is_per_face_not_per_edge():
    """Both faces of ``_fanout`` share the SAME in-edge, so scaling ``lhs`` on
    one face must move ONLY that face's output. A per-edge mechanism would move
    both."""
    ref = _jacobians(_fanout, (_X3,), (0,))

    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        return {keys[0]: (_scale(3.0), None, None)}

    _, ij, _ = _run(_fanout, (_X3,), (0,), _ft, vertex=1)
    out = _eval(ij, (_X3,))

    assert np.allclose(out[0], 3.0 * ref[0], atol=1e-5)
    assert np.allclose(out[1], ref[1], atol=1e-5)


def test_rhs_slot_scales_only_its_face():
    ref = _jacobians(_fanout, (_X3,), (0,))

    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        return {keys[1]: (None, _scale(5.0), None)}

    _, ij, _ = _run(_fanout, (_X3,), (0,), _ft, vertex=1)
    out = _eval(ij, (_X3,))

    assert np.allclose(out[0], ref[0], atol=1e-5)
    assert np.allclose(out[1], 5.0 * ref[1], atol=1e-5)


def test_res_slot_scales_only_its_face():
    ref = _jacobians(_fanout, (_X3,), (0,))

    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        return {keys[0]: (None, None, _scale(2.0))}

    _, ij, _ = _run(_fanout, (_X3,), (0,), _ft, vertex=1)
    out = _eval(ij, (_X3,))

    assert np.allclose(out[0], 2.0 * ref[0], atol=1e-5)
    assert np.allclose(out[1], ref[1], atol=1e-5)


def test_res_slot_runs_after_the_per_vertex_transforms():
    """The per-vertex rule (uniform over faces) composes FIRST, then the face's
    own ``res`` choice: the recorded approx blocks appear in that order."""
    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        return {keys[0]: (None, None, Quant("bfloat16"))}

    jaxpr, ij = _build(_fanout, (_X3,), (0,), track_faces=True)
    ft = _ft(ij, jaxpr)
    ij.eliminate(1, (Quant("float16"),), ft)

    approx = [fr.approx for fr in ij.step_faces(0)]
    assert [t for t, _, _, _ in approx[0]] == ["QUANT", "QUANT"]
    assert [p["dtype"] for _, p, _, _ in approx[0]] == ["float16", "bfloat16"]
    # the untargeted face keeps ONLY the per-vertex transform
    assert [p["dtype"] for _, p, _, _ in approx[1]] == ["float16"]


def test_every_slot_is_recorded_on_the_open_face():
    """All three slots record an ``approx`` block on the face they run on, so
    the FaceSink trace stays a truthful description of what was applied."""
    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        return {keys[0]: (Quant("bfloat16"), Quant("float16"),
                          Quant("bfloat16"))}

    _, ij, step = _run(_fanout, (_X3,), (0,), _ft, vertex=1)
    approx = [fr.approx for fr in ij.step_faces(step)]

    assert [p["dtype"] for _, p, _, _ in approx[0]] == [
        "bfloat16", "float16", "bfloat16"]
    assert approx[1] == []


# ---------------------------------------------------------------------------
# typed micro-actions (Diag / Compress) in a face slot
# ---------------------------------------------------------------------------
# Unlike Quant these ARM the approx-edge normalization (``_is_approx_cfg``), so
# they exercise the path where a per-face choice has to reconcile the edge
# layout the same way a per-vertex one does.
@pytest.mark.parametrize("index,slot", [(0, "lhs"), (2, "res")])
def test_diag_in_a_face_slot_approximates_only_that_face(index, slot):
    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        slots = [None, None, None]
        slots[index] = Diag(0, 1, 2)
        return {keys[0]: tuple(slots)}

    ref = _jacobians(_square, (_X4,), (0,))
    _, ij, step = _run(_square, (_X4,), (0,), _ft, vertex=1)
    out = _eval(ij, (_X4,))
    approx = [fr.approx for fr in ij.step_faces(step)]

    assert [t for t, _, _, _ in approx[0]] == ["DIAG"], (
        f"the {slot} slot's Diag must be recorded on the targeted face")
    assert approx[1] == []
    assert not np.array_equal(out[0], ref[0]), "the targeted face is masked"
    assert np.array_equal(out[1], ref[1]), "the other face stays exact"


def test_compress_in_a_face_rhs_slot_approximates_only_that_face():
    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        return {keys[1]: (None, Compress((1,)), None)}

    ref = _jacobians(_square, (_X4,), (0,))
    _, ij, step = _run(_square, (_X4,), (0,), _ft, vertex=1)
    out = _eval(ij, (_X4,))
    approx = [fr.approx for fr in ij.step_faces(step)]

    assert approx[0] == []
    assert [t for t, _, _, _ in approx[1]] == ["COMPRESS"]
    assert np.array_equal(out[0], ref[0]), "the other face stays exact"
    assert not np.array_equal(out[1], ref[1]), "the targeted face is compressed"


# ---------------------------------------------------------------------------
# (d) regression: face_transforms=None changes nothing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fn,args,argnums",
    [(_fanout, (_X3,), (0,)), (_grid, (_X4, _Y4), (0, 1))],
)
def test_none_face_transforms_is_identical(fn, args, argnums):
    """Passing ``face_transforms=None`` emits the same jaxpr, bit for bit, as
    not passing it at all -- and both match ``jax.jacrev``."""
    jaxpr_a, ij_a = _build(fn, args, argnums, track_faces=False)
    for v in range(1, len(jaxpr_a.eqns) + 1):
        ij_a.eliminate(v)

    jaxpr_b, ij_b = _build(fn, args, argnums, track_faces=False)
    for v in range(1, len(jaxpr_b.eqns) + 1):
        ij_b.eliminate(v, (), None)

    assert str(ij_a.current_jaxpr()[0]) == str(ij_b.current_jaxpr()[0])

    out_a, out_b = _eval(ij_a, args), _eval(ij_b, args)
    ref = _jacobians(fn, args, argnums)
    assert len(out_a) == len(ref)
    for a, b, r in zip(out_a, out_b, ref):
        assert np.array_equal(a, b)
        assert np.allclose(a, r, atol=1e-5)


def test_empty_face_transforms_map_is_a_noop():
    """An empty mapping still stabilizes iteration order but must not perturb
    the result."""
    ref = _jacobians(_fanout, (_X3,), (0,))
    _, ij, _ = _run(_fanout, (_X3,), (0,), {}, vertex=1, track_faces=False)
    for out, r in zip(_eval(ij, (_X3,)), ref):
        assert np.allclose(out, r, atol=1e-5)


def test_unknown_face_key_is_ignored():
    """A key that matches no face of the vertex is silently unused -- which is
    what makes the optimistic enumeration in ``faces_of`` safe."""
    ref = _jacobians(_fanout, (_X3,), (0,))
    _, ij, _ = _run(_fanout, (_X3,), (0,),
                    {(9991, 9992): (_scale(9.0), _scale(9.0), _scale(9.0))},
                    vertex=1)
    for out, r in zip(_eval(ij, (_X3,)), ref):
        assert np.allclose(out, r, atol=1e-5)


# ---------------------------------------------------------------------------
# error semantics (mirrors the per-vertex `transforms` contract)
# ---------------------------------------------------------------------------
def test_value_error_skips_the_slot_and_records_nothing():
    """A ValueError means "this transform does not fit this operand" -- the
    documented best-effort miss: skip the slot, keep the operand, record
    nothing."""
    def _boom(st):
        raise ValueError("does not fit")

    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        return {keys[0]: (_boom, _boom, _boom)}

    ref = _jacobians(_fanout, (_X3,), (0,))
    _, ij, step = _run(_fanout, (_X3,), (0,), _ft, vertex=1)

    for out, r in zip(_eval(ij, (_X3,)), ref):
        assert np.allclose(out, r, atol=1e-5)
    assert all(fr.approx == [] for fr in ij.step_faces(step))


def test_diag_that_does_not_fit_is_skipped():
    """The same best-effort skip for a typed micro-action whose geometry misses
    (``Diag.i`` past the operand's logical rank)."""
    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        return {keys[0]: (Diag(0, 9, 2), None, None)}

    ref = _jacobians(_fanout, (_X3,), (0,))
    _, ij, step = _run(_fanout, (_X3,), (0,), _ft, vertex=1)

    for out, r in zip(_eval(ij, (_X3,)), ref):
        assert np.allclose(out, r, atol=1e-5)
    assert ij.step_faces(step)[0].approx == [], (
        "a skipped transform is not recorded")


def test_unknown_slot_type_raises_type_error():
    """A structural programming error, deliberately NOT swallowed."""
    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        return {keys[0]: (17, None, None)}

    with pytest.raises(TypeError, match="Unknown per-face transform"):
        _run(_fanout, (_X3,), (0,), _ft, vertex=1)


def test_malformed_entry_raises_type_error():
    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        return {keys[0]: (None, None)}

    with pytest.raises(TypeError, match="must be a 3-tuple"):
        _run(_fanout, (_X3,), (0,), _ft, vertex=1)


# ---------------------------------------------------------------------------
# replay bookkeeping
# ---------------------------------------------------------------------------
def test_steps_record_the_face_transforms():
    """``IncrementalJacobian.steps`` carries the per-face map so a replay of the
    step list reproduces the trace; the recorded map is a snapshot, not an alias
    of the caller's dict."""
    caller_map = {}

    def _ft(ij, jaxpr):
        keys = faces_of(ij.graph, ij.tgraph, 1, jaxpr)
        caller_map[keys[0]] = (None, None, Quant("bfloat16"))
        return caller_map

    _, ij, step = _run(_fanout, (_X3,), (0,), _ft, vertex=1)

    recorded = ij.step_face_transforms(step)
    assert recorded == caller_map
    assert recorded is not caller_map
    assert ij.step_face_transforms(step + 1) is None

    caller_map.clear()
    assert ij.step_face_transforms(step) != caller_map, "snapshot must not alias"


def test_step_accessors_still_work_with_the_wider_record():
    """The widened ``steps`` tuple must not break the existing accessors."""
    jaxpr, ij = _build(_fanout, (_X3,), (0,), track_faces=True)
    ij.eliminate(1)
    ij.eliminate(2, (), None)

    assert len(ij.step_eqns(0)) > 0
    assert len(ij.step_faces(0)) == 2
    assert len(ij.base_eqns()) == ij.n_base
