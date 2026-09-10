"""How the TWO ADDENDS of a face merge are made to meet.

A vertex elimination that creates an edge which ALREADY exists adds the fresh
contraction result onto the pre-existing edge (``core._eliminate_vertex``, the
``graph[in_edge][out_edge] is not None`` branch)::

    fresh = new(contract)
    old   = the existing predecessor -> successor edge
    edge  = fresh + old

The two addends carry the SAME logical dims -- same ids, same
``logical_size`` -- because they are two contributions to one Jacobian block.
What they do NOT share is STORAGE: one may hold a dim pair as two independent
physical axes while the other holds it as a single coupled
:func:`~graphax.sparse.indexes.DiagonalIndex` axis, and one may store a dim
implicitly (``axis is None``, one copy broadcast on read) where the other
stores every slice. Measured on a transformer LM with only the contraction
result approximated: **0 of 6 merge faces had structurally identical
addends**.

That asymmetry is not cosmetic. A policy that picks one approximation rule per
face wire and has it applied at both addend sites gets a rule that is a legal
block subdivision on one tensor and an idempotent no-op on the other, so one
legality mask cannot describe both sites (finding 72, ticket dsnn-3qm.59
fault 1). This module removes the asymmetry instead of masking around it: it
returns BOTH addends in ONE common container, so there is one structure to
reason about and one site to mask.

TWO CONTAINERS, one per policy:

``UnionJoin`` (``--approx-add lossless``)
    The common container is the UNION of the two supports. Nothing either
    addend holds is dropped, so the sum is exact to floating point. This is
    what the sparse ``+`` already builds on its own -- :func:`_pair_metric` in
    :mod:`~graphax.sparse.ops.elementwise` sets the unified meta count to
    ``gcd`` of the two and the unified block to ``lcm`` of the two -- so the
    policy is implemented BY that machinery rather than beside it.

``MatchFreshJoin`` (``--approx-add lossy``)
    The common container is the one the FRESH contraction landed on, i.e. the
    structure the approximation head actually chose. The old edge is projected
    onto that support first, which is where the information is lost and the
    compute is saved: the sum then costs what the approximated contraction
    costs instead of what the union of the two costs.

BOTH guarantee structural identity of the returned addends, by construction:
the last step of each is :func:`unify_containers`, which lifts each addend
into the union of the two containers. Adding a STRUCTURAL ZERO is exact, and
the union of two identical containers is that container, so when the
projection did reach the target support the common container IS the target's.
When it did not, the two addends are still identical to each other -- only
wider than the target. :func:`reconcile_addends` reports which of the two
happened rather than quietly picking one; see ``JoinOutcome``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax.numpy as jnp

__all__ = [
    "container_of",
    "structural_ones",
    "structural_zero",
    "unify_containers",
    "support_projection_rules",
    "project_onto_support",
    "reconcile_addends",
    "JoinOutcome",
    "FaceJoinPolicy",
    "UnionJoin",
    "MatchFreshJoin",
]


# --- the structure of a tensor, as a comparable value -----------------------

def _dim_container(d) -> tuple:
    """ONE ``Index``'s STORAGE, as a hashable tuple.

    Every field that decides where a value physically lives: the meta count
    (``size``), the block extent, which physical axes hold them (``None`` = the
    extent is implicit, one copy broadcast on read), and the partner id that
    makes a pair coupled. ``logical_size`` is deliberately NOT the whole story
    -- it is equal on both addends of every merge, which is exactly why it
    cannot be used to tell their layouts apart (finding 72).
    """
    return (int(d.id), int(d.size), d.axis,
            d.other_id if d.other_id is None else int(d.other_id),
            d.block_size if d.block_size is None else int(d.block_size),
            d.block_axis, type(d).__name__)


def container_of(st) -> tuple:
    """The full STRUCTURE of a :class:`~graphax.sparse.tensor.SparseTensor`:
    ``val``'s shape, every dim's storage, and whether the implicit fill is
    statically zero. Two tensors with equal ``container_of`` store their values
    in the same places, so one legality mask describes both.

    Values are NOT part of it, and neither is ``scalar_mult`` -- a deferred
    scalar does not move a value.
    """
    return (tuple(getattr(st.val, "shape", ()) or ()),
            tuple(_dim_container(d) for d in st.out_dims),
            tuple(_dim_container(d) for d in st.primal_dims),
            st.fill_value is None)


def _dims_by_id(st) -> dict:
    return {int(d.id): d for d in st.dims}


def _logical_pos(st) -> dict:
    """dim id -> its index in ``out_dims ++ primal_dims``, the numbering
    :class:`~graphax.sparse.micro_actions.Diag` uses."""
    return {int(d.id): i for i, d in enumerate(st.dims)}


# --- structural operands ----------------------------------------------------

def structural_ones(st, dtype=None):
    """``st``'s STRUCTURE carrying 1 wherever it stores a value.

    The indicator of ``st``'s support: multiplying by it keeps the other
    operand's values exactly where ``st`` has support and sends the rest to the
    fill. ``scalar_mult`` is reset to 1 (a deferred scale moves no value) and
    the fill follows ``st``'s: statically-zero fill means the off-support cells
    really are outside the support, a non-zero fill means ``st`` has no
    off-support cells at all and the indicator is 1 everywhere.
    """
    from graphax.sparse.tensor import SparseTensor

    dt = st.dtype if dtype is None else dtype
    one = jnp.ones((), dt)
    return SparseTensor(
        st.out_dims, st.primal_dims,
        None if st.val is None else jnp.ones(st.val.shape, dt),
        scalar_mult=one,
        fill_value=None if st.fill_value is None else one,
        check_consistency=False)


def structural_zero(st, dtype=None):
    """``st``'s STRUCTURE carrying 0 everywhere.

    Adding it to a tensor is numerically the identity and structurally a LIFT:
    the sparse ``+`` reconciles the two layouts into their union container, so
    ``x + structural_zero(y)`` is ``x``'s values in the union of ``x``'s and
    ``y``'s containers. That is the whole of :func:`unify_containers`.
    """
    from graphax.sparse.tensor import SparseTensor

    dt = st.dtype if dtype is None else dtype
    return SparseTensor(
        st.out_dims, st.primal_dims,
        None if st.val is None else jnp.zeros(st.val.shape, dt),
        scalar_mult=jnp.zeros((), dt),
        fill_value=None,
        check_consistency=False)


def unify_containers(a, b):
    """``(a', b')`` holding ``a``'s and ``b``'s values in ONE container.

    Implemented as ``a + structural_zero(b)`` and ``b + structural_zero(a)``,
    so the container is whatever :mod:`~graphax.sparse.ops.elementwise` builds
    for a union op -- meta ``gcd``, block ``lcm`` (:func:`_pair_metric`) --
    and this module does not carry a second copy of that algebra. Both calls
    pair the same two dim sets and rebuild the result's axes canonically from
    the left operand's dim ORDER, which is identical for two addends of one
    merge, so the two results agree; that is ASSERTED, not assumed.

    Exact: adding a statically-zero-fill zero tensor changes no value.

    Raises ``RuntimeError`` if the two sides do not land on the same container
    -- that would mean the union algebra is not symmetric for this pair, which
    is a bug in the algebra and must not be papered over here.
    """
    a2 = a + structural_zero(b, a.dtype)
    b2 = b + structural_zero(a, b.dtype)
    ca, cb = container_of(a2), container_of(b2)
    if ca != cb:
        raise RuntimeError(
            "unify_containers: the union container is not symmetric for this "
            f"pair.\n  a + zero(b) -> {ca}\n  b + zero(a) -> {cb}\n"
            "Both sides pair the same dims, so a disagreement is a defect in "
            "elementwise's container algebra, not something to fall back from.")
    return a2, b2


# --- projecting one addend onto the other's support -------------------------

def support_projection_rules(src, target) -> tuple:
    """The micro-actions that make ``src``'s support fit inside ``target``'s.

    Returns a tuple of :class:`~graphax.sparse.micro_actions.Diag` /
    :class:`~graphax.sparse.micro_actions.Compress` to apply to ``src``, in
    order. Only the differences a micro-action can CLOSE are emitted:

    * ``target`` holds a dim pair as a meta-block-diagonal with ``size``
      (meta count) FINER than ``src``'s -> ``Diag(i, j, factor=target.size)``.
      Going finer drops the off-block values, which is the lossy step.
    * ``target`` stores a dim IMPLICITLY (one copy broadcast on read) where
      ``src`` stores every slice -> ``Compress`` of the physical axis holding
      those slices. ``kind="mean"`` is the reduction the approximation search
      already uses for a collapsed axis.

    Differences in the OTHER direction -- ``src`` already finer, or already
    implicit -- need no rule: ``src``'s support is then already inside
    ``target``'s along that dim, and :func:`unify_containers` lifts it the rest
    of the way without losing anything.

    This is a SUPPORT question, so it is deliberately silent about an
    implicit dim on the ``src`` side: implicitness is a uniform-VALUE
    restriction, not a smaller support.
    """
    from graphax.sparse.micro_actions import Compress, Diag

    s_by, t_by = _dims_by_id(src), _dims_by_id(target)
    pos = _logical_pos(src)
    rules: list = []
    used: set[int] = set()

    # 1. pair subdivision, once per coupled pair of `target`
    done_pairs: set[tuple[int, int]] = set()
    for tid, td in t_by.items():
        if not td.is_sparse:
            continue
        oid = int(td.other_id)
        key = (min(tid, oid), max(tid, oid))
        if key in done_pairs or oid not in t_by:
            continue
        done_pairs.add(key)
        sd, so = s_by.get(tid), s_by.get(oid)
        if sd is None or so is None:
            continue
        src_meta = int(sd.size) if (sd.is_sparse
                                    and int(sd.other_id or -1) == oid) else 1
        tgt_meta = int(td.size)
        if tgt_meta <= src_meta:
            continue                      # src already at least as fine
        i, j = pos.get(tid), pos.get(oid)
        if i is None or j is None:
            continue
        n_i, n_j = int(sd.logical_size), int(so.logical_size)
        if n_i % tgt_meta or n_j % tgt_meta:
            continue                      # not a divisor: no legal Diag
        if src_meta > 1 and tgt_meta % src_meta:
            continue                      # not nestable in src's blocks
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        rules.append(Diag(i=i, j=j, factor=tgt_meta))

    # 2. dims `target` stores implicitly that `src` spells out
    for tid, td in t_by.items():
        sd = s_by.get(tid)
        if sd is None:
            continue
        p = pos.get(tid)
        if p is None or p in used:
            continue
        if td.axis is None and sd.axis is not None and int(sd.size) > 1:
            rules.append(Compress(axes=(int(sd.axis),), kind="mean"))
            used.add(p)
    return tuple(rules)


def project_onto_support(src, target):
    """``src`` restricted to ``target``'s support, values otherwise untouched.

    Two steps, both existing machinery:

    1. the :func:`support_projection_rules` micro-actions, which are the only
       way to reach ``target``'s IMPLICIT dims and its finer meta blocks from
       ``src``'s layout;
    2. ``elementwise(src, structural_ones(target), multiply,
       is_intersection=True)`` -- the INTERSECTION container rule, which
       zeroes everything outside ``target``'s support and demotes the result to
       the narrower of the two containers.

    Step 2 does the support arithmetic; step 1 exists because a multiply cannot
    make a spelled-out dim uniform (an implicit dim has FULL support -- it is
    a value restriction, not a support one) nor subdivide a block.
    """
    from graphax.sparse.micro_actions import apply_micro_actions
    from graphax.sparse.ops.elementwise import elementwise

    src = apply_micro_actions(src, support_projection_rules(src, target))
    ind = structural_ones(target, src.dtype)
    out = elementwise(src, ind, jnp.multiply, is_intersection=True)
    return src if out is None else out


# --- the policies -----------------------------------------------------------

class JoinOutcome(NamedTuple):
    """What :func:`reconcile_addends` did, so a caller can MEASURE it.

    ``matched_target`` -- the common container is exactly the one the policy
    aimed at. For ``UnionJoin`` that is the union and always true; for
    ``MatchFreshJoin`` it is the fresh contraction's container, and false means
    the projection could not reach it, so the add is wider (and more
    expensive) than the head asked for. The addends are structurally identical
    either way.

    ``rules`` -- the micro-actions :func:`support_projection_rules` emitted,
    i.e. where the information was actually lost. Empty under ``lossless``.
    """
    mode: str
    container: tuple
    target: tuple
    matched_target: bool
    rules: tuple


def reconcile_addends(fresh, old, mode: str):
    """``(fresh', old', JoinOutcome)`` -- both addends in ONE container.

    ``mode``:

    ``"lossless"``
        the union of the two supports. Exact.
    ``"lossy"``
        the fresh contraction's container, with ``old`` projected onto it
        first.

    Anything else raises ``ValueError``: an unknown join mode means the caller
    believes it chose a semantics that does not exist, and picking one for it
    would make the measured object silently wrong.
    """
    if mode == "lossless":
        f2, o2 = unify_containers(fresh, old)
        c = container_of(f2)
        return f2, o2, JoinOutcome(mode, c, c, True, ())
    if mode != "lossy":
        raise ValueError(
            f"reconcile_addends: unknown join mode {mode!r}; expected "
            "'lossless' or 'lossy'.")
    want = container_of(fresh)
    rules = support_projection_rules(old, fresh)
    old_p = project_onto_support(old, fresh)
    f2, o2 = unify_containers(fresh, old_p)
    got = container_of(f2)
    return f2, o2, JoinOutcome(mode, got, want, got == want, rules)


@dataclass(frozen=True)
class FaceJoinPolicy:
    """A ``face_transforms`` two-op entry's ``jr`` position, as a POLICY.

    The ``jl`` / ``jr`` / ``jres`` hook positions each see ONE tensor, so none
    of them can express "make these two addends match" -- that is inherently a
    binary operation on the pair. A ``FaceJoinPolicy`` in the ``jr`` position
    is handed BOTH addends by :func:`~graphax.core._eliminate_vertex` and
    returns both (see ``_unpack_face_slots``). ``jr`` is the right position for
    it because ``jr`` is the old edge's slot and the old edge is what moves.

    ``pre`` -- an ordinary single-tensor hook applied to the OLD EDGE before
    the reconciliation. That is the slot a learned approximation of the old
    edge occupies; it runs FIRST so the reconciliation still has the last word
    on the structure and the two addends still come out identical.

    ``on_outcome(JoinOutcome)`` -- optional telemetry sink, called once per
    merge. The engine's counters cannot carry this: a reconciliation is not a
    micro-action and must not be counted as one.
    """

    mode: str = "lossless"
    pre: Callable | None = None
    on_outcome: Callable | None = None

    def reconcile(self, fresh, old):
        if self.pre is not None:
            old = self.pre(old)
        f2, o2, outcome = reconcile_addends(fresh, old, self.mode)
        if self.on_outcome is not None:
            self.on_outcome(outcome)
        return f2, o2


def UnionJoin(pre=None, on_outcome=None) -> FaceJoinPolicy:
    """``--approx-add lossless``: the union of the two supports."""
    return FaceJoinPolicy("lossless", pre, on_outcome)


def MatchFreshJoin(pre=None, on_outcome=None) -> FaceJoinPolicy:
    """``--approx-add lossy``: the fresh contraction's container."""
    return FaceJoinPolicy("lossy", pre, on_outcome)
