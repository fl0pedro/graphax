"""Feature pin: the per-face TWO-OP form ``((lhs, rhs, new), (jl, jr, jres))``.

The join semantics requested for the ``new`` slot are
``new = approx(new_existing) + approx(contract)`` -- BOTH addends of a merge
are hooked, not just the fresh contraction (and not the sum). The env emits
this as a pair of 3-tuples; `_unpack_face_slots` dispatches on the outer
length. Regression context: the first emission of this form sentineled every
approximated terminal measurement of v53 job 59278 because the face path only
accepted the flat 3-tuple.

Model: ``u = sin(x); w = u + x; y = W @ w`` -- eliminating vertex 1 (sin)
contracts the face ``x -> u -> w`` and MERGES the result into the existing
direct ``x -> w`` edge, so the join is exercised with an in-edge that is a
graph input (the alphagrad case).

The second model (``_join_mat``) gives BOTH addends of the merge a real,
materialized ``val`` -- the identity edge of ``_join`` has ``val is None``, so
a value-level approximation (a bf16 cast, a Quant) is a NO-OP on it and could
not distinguish "both addends approximated" from "only the contraction was".

Coverage:
  * every one of the five hooks lands on the tensor it names (one test, with
    six mutually-prime scales, so any misplacement changes the number);
  * ``old = approx(old) + approx(new)`` holds numerically for a bf16 cast, and
    approximating only ONE addend gives a DIFFERENT answer;
  * the merge-free degeneration (``jl``/``jr`` skipped, nothing recorded) --
    both on the normal path and under the deferred-output fast path;
  * the slot tags the two sinks see (FaceSink: all four join sites report
    ``"res"``; TransformLog: the fine-grained ``"res:new"`` / ... tags).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import jacve
from graphax.core import _stable_var_index, DEFERRED_OUTPUT_STATS
from graphax.incremental import IncrementalJacobian
from graphax.sparse.micro_actions import Quant

_W = jnp.asarray(np.arange(20, dtype=np.float32).reshape(5, 4) / 19.0)
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))
# Deliberately NOT bf16-representable (7 significant decimal digits), so a
# bf16 round trip is observable in float32.
_B = jnp.asarray(
    (np.arange(16, dtype=np.float64).reshape(4, 4) * 0.1234567 + 0.7654321)
    .astype(np.float32)
)


def _join(x):
    u = jnp.sin(x)                   # vertex 1, central ``u``
    w = u + x                        # vertex 2 -- ALSO consumes x directly
    return _W @ w                    # vertex 3 (output)


def _join_mat(x):
    u = jnp.sin(x)                   # vertex 1, central ``u``
    v = _B @ x                       # vertex 2 -- the OTHER path into ``w``
    w = u + v                        # vertex 3
    return _W @ w                    # vertex 4 (output)


def _build(fn, args, argnums):
    closed = jax.make_jaxpr(fn)(*args)
    ij = IncrementalJacobian(closed.jaxpr, argnums, list(closed.literals),
                             list(args), track_faces=True)
    return closed.jaxpr, ij


def _eval(ij, args):
    jaxpr, consts, _ = ij.current_jaxpr()
    return [np.asarray(v) for v in jax.core.eval_jaxpr(jaxpr, consts, *args)]


def _scale(k, log=None, tag=None):
    def _apply(st):
        if log is not None:
            log.append((tag, tuple(st.shape)))
        return st.copy(
            scalar_mult=st.scalar_mult * jnp.asarray(k, st.scalar_mult.dtype))
    return _apply


def _bf16(log=None, tag=None):
    """A bf16 CAST as a per-face approximation, round-tripped back to the
    operand's own dtype so the approximation is visible in the VALUES while
    every downstream op keeps computing in float32 (a narrow stored dtype
    would re-round the products and blur what is being measured)."""
    def _apply(st):
        if log is not None:
            log.append((tag, tuple(st.shape)))
        if st.val is None:
            return st
        return st.copy(
            val=st.val.astype(jnp.bfloat16).astype(st.val.dtype))
    return _apply


def _bf16_np(a):
    return np.asarray(jnp.asarray(a).astype(jnp.bfloat16).astype(jnp.float32))


def _face_key_v1(ij, jaxpr):
    vidx = _stable_var_index(jaxpr)
    x_var = jaxpr.invars[0]
    w_var = jaxpr.eqns[1].outvars[0]
    return (vidx[x_var], vidx[w_var])


def _run_v1(two_op):
    jaxpr, ij = _build(_join, (_X4,), (0,))
    ft = {_face_key_v1(ij, jaxpr): two_op}
    ij.eliminate(1, (), ft)
    ij.eliminate(2)
    return _eval(ij, (_X4,))


# ---------------------------------------------------------------------------
# ``_join_mat``: the merge whose BOTH addends carry a materialized ``val``.
#
# Vertices 1 (``sin``) and 2 (``B @ x``) share the SAME face key ``x -> w``:
# whichever is eliminated second MERGES into the edge the first one created.
# So ``_run_mat(1, ..., first=2)`` exercises the join between two real
# matrices (``B`` and ``diag(cos x)``), while ``_run_mat(2, ...)`` with no
# ``first`` exercises the MERGE-FREE face, where jl/jr have nothing to do.
# ---------------------------------------------------------------------------

def _mat_face_key(jaxpr):
    """The face key of ``x -> <central> -> w`` in ``_join_mat``.

    The SAME key for vertex 1 and vertex 2: both faces run from the input
    ``x`` to the add ``w``, which is exactly why one of them merges into the
    edge the other created."""
    vidx = _stable_var_index(jaxpr)
    return (vidx[jaxpr.invars[0]], vidx[jaxpr.eqns[2].outvars[0]])


def _run_mat(vertex, slots, first=None):
    """Eliminate ``_join_mat`` fully, hooking ONLY ``vertex``'s face.

    ``first`` is the vertex eliminated before it (2 to make ``vertex=1``'s face
    a MERGE, 1 to make ``vertex=2``'s face a merge). Returns
    ``(dy_dx, ij, step_index_of_the_hooked_step)``.
    """
    jaxpr, ij = _build(_join_mat, (_X4,), (0,))
    ft = {_mat_face_key(jaxpr): slots}
    if first is not None:
        ij.eliminate(first)
    hooked = len(ij.steps)
    ij.eliminate(vertex, (), ft)
    for v in (1, 2, 3):
        if v != vertex and v != first:
            ij.eliminate(v)
    return _eval(ij, (_X4,))[0], ij, hooked


def _exact_mat():
    return np.asarray(jax.jacrev(_join_mat)(_X4))


def test_two_op_identity_matches_reference():
    """Identity hooks in both triples -> the exact Jacobian, and BOTH the
    fresh contraction and the EXISTING edge are actually visited."""
    log = []
    two_op = ((None, None, _scale(1.0, log, "contract")),
              (None, _scale(1.0, log, "existing"), None))
    out = _run_v1(two_op)
    ref = jax.jacrev(_join)(_X4)
    np.testing.assert_allclose(out[0], np.asarray(ref), rtol=1e-6)
    tags = [t for t, _ in log]
    assert tags.count("contract") == 1, log
    assert tags.count("existing") == 1, log


def test_two_op_scales_both_addends():
    """new-hook x2 on the contraction, join-rhs x3 on the existing edge:
    d(w)/dx = 3*I + 2*diag(cos x)  =>  dy/dx = W @ (3*I + 2*diag(cos x)).
    Pins that the hooks land on the ADDENDS, not on the sum."""
    two_op = ((None, None, _scale(2.0)), (None, _scale(3.0), None))
    out = _run_v1(two_op)
    inner = 3.0 * np.eye(4, dtype=np.float32) \
        + 2.0 * np.diag(np.cos(np.asarray(_X4)))
    expect = np.asarray(_W) @ inner
    np.testing.assert_allclose(out[0], expect, rtol=1e-6)


def test_legacy_three_tuple_unchanged():
    """The flat 3-tuple still applies ``res`` at the post-join site: the sum
    (I + diag(cos x)) is scaled as a whole -> W @ 2*(I + diag(cos x))."""
    three = (None, None, _scale(2.0))
    out = _run_v1(three)
    inner = 2.0 * (np.eye(4, dtype=np.float32)
                   + np.diag(np.cos(np.asarray(_X4))))
    expect = np.asarray(_W) @ inner
    np.testing.assert_allclose(out[0], expect, rtol=1e-6)


# ---------------------------------------------------------------------------
# (1) ALL FIVE HOOKS, each pinned to its own tensor.
# ---------------------------------------------------------------------------

def test_every_hook_lands_on_the_tensor_it_names():
    """Six mutually-prime scales, one per hook, spell out the whole pipeline::

        contract = rhs(dw/du) @ lhs(du/dx) = 3 * 2 * diag(cos x)
        fresh    = new(contract)           = 5 * that
        merged   = jr(B) + jl(fresh)       = 11*B + 7 * that
        edge     = jres(merged)            = 13 * that
        dy/dx    = W @ edge

    Any hook landing on the wrong operand -- the sum instead of an addend, the
    existing edge instead of the fresh one, before the contraction instead of
    after -- changes the number, because no product of a subset of
    {2,3,5,7,11,13} equals another.
    """
    order = []
    two_op = (
        (_scale(2.0, order, "lhs"),
         _scale(3.0, order, "rhs"),
         _scale(5.0, order, "new")),
        (_scale(7.0, order, "jl"),
         _scale(11.0, order, "jr"),
         _scale(13.0, order, "jres")),
    )
    out, _ij, _s = _run_mat(1, two_op, first=2)

    cos = np.diag(np.cos(np.asarray(_X4)))
    inner = 11.0 * np.asarray(_B) + (7.0 * 5.0 * 3.0 * 2.0) * cos
    np.testing.assert_allclose(out, np.asarray(_W) @ (13.0 * inner), rtol=1e-5)

    # ... and they run in pipeline order, each exactly once.
    assert [t for t, _ in order] == ["lhs", "rhs", "new", "jl", "jr", "jres"]


def test_join_hooks_see_the_two_distinct_addends():
    """``jl`` is handed the FRESH contraction and ``jr`` the EXISTING edge --
    two different tensors at the same site. Pinned by giving the fresh
    contribution a scale the existing edge does not have: whichever operand
    ``jr`` gets, only ``B`` may carry ``11`` and only ``diag(cos x)`` the
    ``2*7``."""
    two_op = ((None, None, _scale(2.0)), (_scale(7.0), _scale(11.0), None))
    out, _ij, _s = _run_mat(1, two_op, first=2)
    cos = np.diag(np.cos(np.asarray(_X4)))
    inner = 11.0 * np.asarray(_B) + 14.0 * cos
    np.testing.assert_allclose(out, np.asarray(_W) @ inner, rtol=1e-5)


# ---------------------------------------------------------------------------
# (2) ``old = approx(old) + approx(new)`` for a REAL approximation.
# ---------------------------------------------------------------------------

def test_bf16_cast_approximates_both_addends():
    """The whole point of the two-op form: a bf16 cast reaches BOTH addends.

    The merged edge must equal ``bf16(B) + bf16(diag cos x)`` -- not
    ``B + bf16(diag cos x)`` (only the contraction hooked, what a flat triple's
    ``new`` gives) and not ``bf16(B + diag cos x)`` (the sum hooked, what a
    flat triple's ``res`` gives). All three references are compared against the
    SAME machinery, so the assertion isolates the join semantics.
    """
    cos = np.diag(np.cos(np.asarray(_X4)))
    Bn = np.asarray(_B)
    Wn = np.asarray(_W)

    both = _run_mat(1, ((None, None, _bf16()), (None, _bf16(), None)),
                    first=2)[0]
    only_new = _run_mat(1, ((None, None, _bf16()), (None, None, None)),
                        first=2)[0]
    on_the_sum = _run_mat(1, (None, None, _bf16()), first=2)[0]
    exact = _exact_mat()

    np.testing.assert_allclose(
        both, Wn @ (_bf16_np(Bn) + _bf16_np(cos)), rtol=1e-5)
    np.testing.assert_allclose(
        only_new, Wn @ (Bn + _bf16_np(cos)), rtol=1e-5)
    np.testing.assert_allclose(
        on_the_sum, Wn @ _bf16_np(Bn + cos), rtol=1e-5)

    # The three are genuinely different objects -- i.e. hooking the EXISTING
    # edge really did approximate a second tensor, and did not merely repeat
    # what the contraction hook already did.
    assert not np.allclose(both, only_new, rtol=1e-6, atol=0)
    assert not np.allclose(both, on_the_sum, rtol=1e-6, atol=0)
    assert not np.allclose(both, exact, rtol=1e-6, atol=0)
    # ... and the approximation is a SMALL one (a cast, not a corruption), so
    # a wrong-tensor hook cannot hide inside the tolerance above.
    np.testing.assert_allclose(both, exact, rtol=2e-2)


# ---------------------------------------------------------------------------
# (2a) GAP DECISION: a MERGE-FREE face degenerates to ``jres(new(contract))``.
# ---------------------------------------------------------------------------

def test_merge_free_face_skips_the_join_hooks():
    """``jl``/``jr`` are applied ONLY inside the "an edge already exists"
    branch, so a face that CREATES the edge silently drops them.

    Pinned, not fixed -- see ``_unpack_face_slots``: ``jr`` has no operand at
    all without an existing edge, and ``jl``'s site is then bit-identical to
    ``new``'s (nothing happens between them), so the form loses no expressive
    power. Here vertex 2 is eliminated FIRST, so its face ``x -> v -> w``
    creates the ``x -> w`` edge: only ``new`` (x2) and ``jres`` (x3) may act.
    """
    two_op = ((None, None, _scale(2.0)),
              (_scale(7.0), _scale(11.0), _scale(3.0)))
    out, ij, step = _run_mat(2, two_op, first=None)
    cos = np.diag(np.cos(np.asarray(_X4)))
    # 3 * (2 * B), then vertex 1's UNHOOKED contraction merges in exactly.
    inner = 6.0 * np.asarray(_B) + cos
    np.testing.assert_allclose(out, np.asarray(_W) @ inner, rtol=1e-5)


def test_merge_free_face_records_nothing_for_the_join_hooks():
    """The skip is SILENT and that silence is truthful: a hook that never ran
    records nothing, so no sink can claim an approximation that did not
    happen. (Contrast the merged case below, which records all four.)"""
    q = Quant("bfloat16")
    two_op = ((None, None, q), (q, q, q))
    _out, ij, step = _run_mat(2, two_op, first=None)
    slots = [r.slot for r in ij.step_transform_records(step)]
    assert slots == ["res:new", "res:jres"], slots


# ---------------------------------------------------------------------------
# (2b) GAP DECISION: the four post-contraction sites share the FaceSink slot
#      ``"res"`` and are distinguishable only on the transform log.
# ---------------------------------------------------------------------------

_Q = Quant("bfloat16")
_JOIN_SITES = [
    ("new", ((None, None, _Q), (None, None, None)), "res:new"),
    ("jl", ((None, None, None), (_Q, None, None)), "res:jl"),
    ("jr", ((None, None, None), (None, _Q, None)), "res:jr"),
    ("jres", ((None, None, None), (None, None, _Q)), "res:jres"),
]


@pytest.mark.parametrize("name,slots,log_tag", _JOIN_SITES,
                         ids=[s[0] for s in _JOIN_SITES])
def test_each_join_site_reports_res_to_the_sink_and_its_own_tag_to_the_log(
        name, slots, log_tag):
    """``FACE_SLOT_INDEX`` is a closed map and the tokenizer emits a FIXED
    number of equation blocks per face at fixed positions, so ``new`` / ``jl``
    / ``jr`` / ``jres`` all report the ``"res"`` operand -- which IS the
    correct slot for every one of them (they all act on the contraction result
    and its merge), but leaves the FaceSink unable to tell them apart. The
    always-on transform log, which has no positional consumer, gets the
    fine-grained tag instead. One hook per run, so the assertion is about the
    TAGS and not about how the four compose."""
    _out, ij, step = _run_mat(1, slots, first=2)
    recs = [r for fr in ij.step_faces(step) for r in fr.approx]
    assert [r.slot for r in recs] == ["res"], recs
    assert [r.slot for r in ij.step_transform_records(step)] == [log_tag]


def test_the_transform_log_distinguishes_the_four_join_sites():
    """The always-on transform log has no positional consumer, so it carries
    the FINE-GRAINED tag instead -- which is what makes a join debuggable."""
    q = Quant("bfloat16")
    two_op = ((None, None, q), (q, q, q))
    _out, ij, step = _run_mat(1, two_op, first=2)
    slots = [r.slot for r in ij.step_transform_records(step)]
    assert slots == ["res:new", "res:jl", "res:jr", "res:jres"], slots


def test_the_flat_form_keeps_the_bare_res_tag():
    """Unchanged for the legacy triple: ``res`` on both sinks."""
    _out, ij, step = _run_mat(1, (None, None, Quant("bfloat16")), first=2)
    assert [r.slot for r in ij.step_transform_records(step)] == ["res"]
    recs = [r for fr in ij.step_faces(step) for r in fr.approx]
    assert [r.slot for r in recs] == ["res"]


# ---------------------------------------------------------------------------
# (3) The deferred-output fast path does not test ``_face_join_t`` -- safe,
#     because it already requires the very absence that kills jl/jr.
# ---------------------------------------------------------------------------

_N = 64


def _chain(x):
    u = jnp.sin(x)                   # vertex 1, diagonal du/dx
    return jnp.exp(u)                # vertex 2 -- a PURE OUTPUT, diagonal


def _run_chain(slots, factored, monkeypatch, n):
    """``jacve`` over ``_chain``, hooking its ONE face, with the deferral flag
    on or off. Driven through ``jacve`` (not ``IncrementalJaxpr``) because that
    is the entry point the ``#46`` deferral supports end to end -- its drain
    spills the stored factor pair; the incremental builder's output path does
    not know about a ``DeferredOutputProduct``.

    ``n`` differs per test so the eliminator's prefix cache cannot alias one
    test's flag-on state into another's (same reason the ``#46`` tests use
    their own hidden sizes).
    """
    monkeypatch.setenv("GRAPHAX_FACTORED_OUTPUTS", "1" if factored else "0")
    xs = jnp.asarray(np.linspace(0.1, 0.9, n, dtype=np.float32))
    closed = jax.make_jaxpr(_chain)(xs)
    vidx = _stable_var_index(closed.jaxpr)
    # The face key is an index into the STABLE var order, so it survives the
    # retrace jacve does internally.
    key = (vidx[closed.jaxpr.invars[0]], vidx[closed.jaxpr.eqns[1].outvars[0]])
    before = DEFERRED_OUTPUT_STATS.get("defer", 0)
    out = jacve(_chain, order=[1], argnums=(0,),
                face_transforms={1: {key: slots}})(xs)
    fired = DEFERRED_OUTPUT_STATS.get("defer", 0) - before
    exact = np.diag(np.asarray(jnp.exp(jnp.sin(xs)) * jnp.cos(xs)))
    return np.asarray(out[0]), fired, exact


def test_deferred_output_fast_path_is_safe_for_the_join_hooks(monkeypatch):
    """The ``#46`` fast path stores the contraction FACTORS and ``continue``s
    past the merge site, and its gate tests ``_face_res_t``/``_face_new_t`` but
    NOT ``_face_join_t``. That is safe by construction: the gate also requires
    ``graph[in_edge][out_edge] is None``, which is exactly the condition under
    which ``jl``/``jr`` would not have run on the slow path either. Pinned by
    running the SAME join-hooked face with the fast path on and off."""
    two_op = ((None, None, None), (_scale(7.0), _scale(11.0), None))
    on, fired, exact = _run_chain(two_op, True, monkeypatch, n=64)
    off, not_fired, _ = _run_chain(two_op, False, monkeypatch, n=64)

    assert fired == 1, "the deferred fast path did not fire -- test is vacuous"
    assert not_fired == 0
    # Both agree, and both agree with the EXACT Jacobian: the join hooks are
    # dead on a merge-free face either way.
    np.testing.assert_allclose(on, off, rtol=1e-6)
    np.testing.assert_allclose(on, exact, rtol=1e-5)


def test_deferred_output_fast_path_yields_to_the_new_slot(monkeypatch):
    """The complement: a ``new`` hook DOES disable the deferral (it acts on
    the contraction result, which the fast path never materializes), so the
    approximation is not silently dropped."""
    two_op = ((None, None, _scale(2.0)), (None, None, None))
    out, fired, exact = _run_chain(two_op, True, monkeypatch, n=80)
    assert fired == 0
    np.testing.assert_allclose(out, 2.0 * exact, rtol=1e-5)


# ---------------------------------------------------------------------------
# Structural validation.
# ---------------------------------------------------------------------------

def test_a_malformed_entry_still_raises():
    """The two-op dispatch must not swallow a genuinely broken entry."""
    with pytest.raises(TypeError):
        _run_mat(1, ((None, None), (None, None)), first=2)
