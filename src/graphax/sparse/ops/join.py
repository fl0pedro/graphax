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
    "pairing_of",
    "decouple_trivial_pairs",
    "loosen_to_pairing",
    "next_projection_rule",
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

def pairing_of(st) -> dict:
    """dim id -> the partner id it is stored DIAGONALLY with, or ``None``.

    A dim can be diagonal with AT MOST ONE partner -- that is what
    :func:`~graphax.sparse.micro_actions.apply_diag` enforces, and the reason a
    coercion cannot simply request the target's pairing: re-pairing an already
    paired dim raises ``Diag pair conflict``.
    """
    return {int(d.id): (None if d.other_id is None else int(d.other_id))
            for d in st.dims}


def loosen_to_pairing(st, target):
    """``st`` re-laid-out so no dim is paired with a partner ``target`` does not
    pair it with. LOSSLESS -- it only ever WIDENS the container.

    THE STEP THAT WAS MISSING, and the measurement that found it: on TLM the
    fresh contraction and the old edge routinely pair DIFFERENT dims of the same
    Jacobian block -- the fresh one couples ``(out1, primal3)`` while the old
    one couples ``(out0, primal2)``. Asking ``apply_diag`` for the target's pair
    then raises ``Diag pair conflict: logical index 2 is already paired``
    (3 of 13 merge faces, measured). A dim must be FREED before it can be
    re-paired, and freeing is a lossless widening with no micro-action of its
    own.

    Implemented as one union add against a structural zero whose dims carry
    ``target``'s pairing and nothing else: the union of a diagonal pair with an
    unpaired dim is the unpaired dim (meta ``gcd(K, 1) = 1``), so exactly the
    pairs ``target`` does not want are broken, and the pairs it does want are
    left for the ``Diag`` rules below to create. Every extent is IMPLICIT, so
    the zero costs one scalar and the add does no real work where nothing
    widens.
    """
    from graphax.sparse.indexes import DenseIndex, DiagonalIndex
    from graphax.sparse.tensor import SparseTensor

    t_pair = pairing_of(target)
    s_pair = pairing_of(st)
    if all(s_pair.get(i) in (None, t_pair.get(i)) for i in s_pair):
        return st                      # nothing is mis-paired: no work
    t_by = {int(d.id): d for d in target.dims}

    def _mk(d):
        i = int(d.id)
        o = t_pair.get(i)
        n = int(d.logical_size)
        if o is None or o not in t_by:
            return DenseIndex(i, n, None)
        td = t_by[i]
        blk = td.block_size
        # keep the target's meta/block split so the zero does not ALSO widen a
        # pair the target wants finer than `st` has it.
        return DiagonalIndex(i, int(td.size), None, o,
                             None if blk is None else int(blk), None)

    dt = st.dtype
    z = SparseTensor(tuple(_mk(d) for d in st.out_dims),
                     tuple(_mk(d) for d in st.primal_dims),
                     None, scalar_mult=jnp.zeros((), dt), fill_value=None,
                     check_consistency=False)
    # The union widens the unwanted pair to meta 1 -- one full block, no
    # sparsity -- but keeps its ``other_id``, which apply_diag still treats as
    # a conflict. Demote it to two dense dims so the pair is genuinely free.
    return decouple_trivial_pairs(st + z)


def decouple_trivial_pairs(st):
    """Rewrite every META-1 block pair as two DENSE dims. Pure metadata.

    A pair with ``size == 1`` and ``block_size == N`` is ONE full N x N block:
    it stores every cell, so it carries no sparsity at all. But ``other_id`` is
    still set, and :func:`~graphax.sparse.micro_actions.apply_diag` refuses to
    re-pair ANY dim whose ``other_id`` names a different partner -- so a
    structurally dense "pair" blocks the very re-pairing it no longer
    constrains. Measured: :func:`loosen_to_pairing` correctly widened a
    mismatched pair to meta 1 and the following ``Diag`` still raised
    ``Diag pair conflict``.

    The union add that does the widening cannot drop ``other_id`` itself --
    ``elementwise._reconstruct_dim_pair`` rebuilds a sparse pair from the left
    operand's dim, partner included -- so the demotion is done here, after it:
    the shared meta axis is size 1 and is squeezed out, and each dim moves onto
    its own former ``block_axis``. No value is read or written.
    """
    from graphax.sparse.indexes import DenseIndex
    from graphax.sparse.tensor import SparseTensor
    from dataclasses import replace as _replace

    triv = {int(d.id) for d in st.dims if d.is_sparse and int(d.size) == 1}
    if not triv:
        return st
    drop = sorted({int(d.axis) for d in st.dims
                   if int(d.id) in triv and d.axis is not None})
    val = st.val
    if val is not None and drop:
        for a in drop:
            if int(val.shape[a]) != 1:
                raise RuntimeError(
                    f"decouple_trivial_pairs: meta axis {a} of a size-1 pair "
                    f"has extent {val.shape[a]}, not 1 -- the tensor's dims "
                    f"and val disagree, which is a consistency failure, not "
                    f"something to squeeze through.")
        val = jnp.squeeze(val, axis=tuple(drop))
    remap = {}
    n = 0
    for a in range(0 if val is None else int((st.val).ndim)):
        if a in drop:
            continue
        remap[a] = n
        n += 1

    def _mv(a):
        return None if a is None else remap.get(int(a))

    def _mk(d):
        if int(d.id) in triv:
            return DenseIndex(int(d.id), int(d.logical_size),
                              _mv(d.block_axis))
        return _replace(d, axis=_mv(d.axis), block_axis=_mv(d.block_axis))

    return SparseTensor(tuple(_mk(d) for d in st.out_dims),
                        tuple(_mk(d) for d in st.primal_dims),
                        val, scalar_mult=st.scalar_mult,
                        fill_value=st.fill_value, check_consistency=False)


# A coercion cannot need more rules than there are dims (each rule consumes at
# least one dim of the target's layout), so this bound can only be hit by a
# derivation that fails to make progress -- which is a defect, not an input.
_MAX_PROJECTION_STEPS = 16


def _slot_of_dim(st) -> dict:
    """dim id -> its META slot in :func:`canonical_axis_order`.

    A coupled pair contributes its meta slot at the FIRST of the two dims and
    (when ``block_size`` is set) a block slot after it; the partner shares that
    storage and gets no slot of its own. Both ids map to the pair's meta slot,
    which is the one a Compress of "this dim's extent" must name.
    """
    out: dict = {}
    slot = 0
    seen: set = set()
    for d in st.dims:
        oid = getattr(d, "other_id", None)
        if oid is not None:
            key = frozenset((int(d.id), int(oid)))
            if key in seen:
                continue
            seen.add(key)
            out[int(d.id)] = slot
            out[int(oid)] = slot
            slot += 1
            if d.block_size is not None:
                slot += 1
            continue
        out[int(d.id)] = slot
        slot += 1
    return out


def next_projection_rule(src, target):
    """The NEXT micro-action that brings ``src``'s support inside ``target``'s,
    or ``None`` when it is already inside.

    ONE rule at a time, deliberately. Both rule kinds RENUMBER what the next
    one has to name: ``Diag`` takes LOGICAL dim positions and ``Compress``
    takes PHYSICAL axes, and applying either shifts the physical axis numbering
    of everything after it. Deriving a whole sequence from one snapshot
    therefore produces rules that name the wrong axis by the time they run --
    measured, before this was split: ``Compress.axes entry 3 out of range:
    tensor has 3 logical component slots``, plus two ``Compress(axes=(0,))``
    emitted for two dims that SHARE physical axis 0 (a coupled pair), which
    would have compressed one axis twice.

    The two differences a micro-action can close, in this order:

    * ``target`` holds a dim pair as a meta-block-diagonal FINER than ``src``'s
      -> ``Diag(i, j, factor=target.size)``. Going finer drops the off-block
      values: this is the lossy step.
    * ``target`` stores a dim IMPLICITLY (one copy, broadcast on read) where
      ``src`` spells every slice out -> ``Compress`` of the physical axis
      holding them, ``kind="mean"`` (the reduction the approximation search
      already uses for a collapsed axis).

    Differences the other way -- ``src`` already finer, or already implicit --
    need no rule: ``src``'s support is then already inside ``target``'s there,
    and :func:`unify_containers` lifts it the rest of the way losslessly.

    A dim already paired with a DIFFERENT partner is SKIPPED rather than
    requested: ``apply_diag`` raises on a re-pair, and
    :func:`loosen_to_pairing` is what removes that case beforehand.
    """
    from graphax.sparse.micro_actions import (
        Compress, Diag, canonical_axis_order)

    s_by, t_by = _dims_by_id(src), _dims_by_id(target)
    pos = _logical_pos(src)
    s_pair, t_pair = pairing_of(src), pairing_of(target)
    n_slots = len(src.dims)
    n_ax = 0 if src.val is None else int(src.val.ndim)

    # 1. a pair `target` holds FINER than `src` does
    for tid, td in sorted(t_by.items()):
        if not td.is_sparse:
            continue
        oid = int(td.other_id)
        if oid not in t_by or oid < tid:
            continue                      # each pair once, lower id first
        sd, so = s_by.get(tid), s_by.get(oid)
        if sd is None or so is None:
            continue
        if s_pair.get(tid) not in (None, oid) or \
                s_pair.get(oid) not in (None, tid):
            continue                      # mis-paired: loosen_to_pairing's job
        src_meta = int(sd.size) if s_pair.get(tid) == oid else 1
        tgt_meta = int(td.size)
        if tgt_meta <= src_meta:
            continue
        i, j = pos.get(tid), pos.get(oid)
        if i is None or j is None or i == j:
            continue
        # apply_diag requires one OUT and one PRIMAL index.
        n_out = len(src.out_dims)
        if (i < n_out) == (j < n_out):
            continue
        n_i, n_j = int(sd.logical_size), int(so.logical_size)
        if n_i % tgt_meta or n_j % tgt_meta:
            continue                      # not a divisor: no legal Diag
        if src_meta > 1 and tgt_meta % src_meta:
            continue                      # not nestable in src's blocks
        return Diag(i=i, j=j, factor=tgt_meta)

    # 2. a dim `target` stores implicitly that `src` spells out.
    #
    # ``Compress.axes`` are CANONICAL SLOTS (``canonical_axis_order``), NOT raw
    # physical ``val`` axes -- they coincide only for a fully dense canonical
    # layout. Naming the physical axis instead is silently a NO-OP whenever the
    # slot it lands on is implicit (``apply_compress`` computes
    # ``drops = {_canon[a] ...}``, gets the empty set, and returns ``st``
    # unchanged), so the derivation asks for the same rule again and never
    # terminates: measured, 16 identical ``Compress(axes=(1,))`` on TLM before
    # the loop bound stopped it (job 64658).
    canon = canonical_axis_order(src)
    slot_of = _slot_of_dim(src)
    for tid, td in sorted(t_by.items()):
        sd = s_by.get(tid)
        if sd is None or td.axis is not None or sd.axis is None:
            continue
        if int(sd.size) <= 1:
            continue                      # already one slice: nothing to fold
        k = slot_of.get(tid)
        if k is None or not (0 <= k < len(canon)) or canon[k] is None:
            continue                      # the component is already implicit
        return Compress(axes=(k,), kind="mean")
    return None


def support_projection_rules(src, target) -> tuple:
    """The whole micro-action sequence :func:`project_onto_support` will apply.

    Derived by actually APPLYING each rule to a working copy, because that is
    the only way to see the renumbering the next rule has to name -- see
    :func:`next_projection_rule`. Callers that want the projected tensor as
    well should use :func:`project_onto_support`, which does the same walk once.
    """
    return project_onto_support(src, target)[1]


def project_onto_support(src, target):
    """``(projected, rules)`` -- ``src`` restricted to ``target``'s support,
    values otherwise untouched.

    Three stages, all existing machinery:

    1. :func:`loosen_to_pairing` -- free any dim ``src`` pairs with a partner
       ``target`` does not. Lossless, and required: ``apply_diag`` refuses to
       re-pair an already paired dim.
    2. :func:`next_projection_rule`, applied one at a time until none is left.
       Each application renumbers what the next rule names, so the sequence is
       derived as it is applied, never from one snapshot.
    3. ``elementwise(src, structural_ones(target), multiply,
       is_intersection=True)`` -- the INTERSECTION container rule, which zeroes
       everything outside ``target``'s support and demotes the result to the
       narrower of the two containers.

    Stage 3 does the support arithmetic; stage 2 exists because a multiply
    cannot make a spelled-out dim uniform (an implicit dim has FULL support --
    it is a value restriction, not a support one) nor subdivide a block.

    ``rules`` is where the information was actually lost.
    """
    from graphax.sparse.micro_actions import apply_micro_actions
    from graphax.sparse.ops.elementwise import elementwise

    src = loosen_to_pairing(src, target)
    rules: list = []
    for _ in range(_MAX_PROJECTION_STEPS):
        rule = next_projection_rule(src, target)
        if rule is None:
            break
        before = container_of(src)
        src = apply_micro_actions(src, (rule,))
        rules.append(rule)
        if container_of(src) == before:
            # A rule that moves NOTHING would be asked for again for ever.
            # Stop and say so rather than spin: the projection is then
            # incomplete, `unify_containers` still returns identical addends,
            # and JoinOutcome.matched_target reports the container as wider
            # than the target -- which is the honest answer.
            raise RuntimeError(
                f"project_onto_support: {rule!r} left the container "
                f"unchanged, so the derivation cannot make progress. "
                f"src={before} target={container_of(target)}")
    else:
        raise RuntimeError(
            f"project_onto_support did not converge in "
            f"{_MAX_PROJECTION_STEPS} rules: {rules}. Each rule must remove a "
            f"difference, so a loop means the derivation is not making "
            f"progress -- a defect in next_projection_rule, not an input to "
            f"tolerate.")
    ind = structural_ones(target, src.dtype)
    out = elementwise(src, ind, jnp.multiply, is_intersection=True)
    return (src if out is None else out), tuple(rules)


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
    # THE TARGET IS ``fresh``'s CANONICAL container, not ``fresh``'s literal
    # one. Every path out of this function ends in `unify_containers`, which
    # rebuilds the result's physical axes canonically from the left operand's
    # dim order, so a `fresh` that happens to be stored non-canonically (an
    # out dim on a later axis than a primal dim, say) can never be returned
    # byte-for-byte. Comparing against the literal container would report
    # "wider than target" for a reconciliation that in fact landed exactly on
    # the head's chosen structure -- measured: 0 of 10 "matched" before this
    # was fixed, with the addends nonetheless identical and the error at
    # float noise.
    want = container_of(unify_containers(fresh, fresh)[0])
    old_p, rules = project_onto_support(old, fresh)
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
