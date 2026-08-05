"""Face sink for the incremental append-only tokenizer.

The Jacobian is built on ONE persistent jax trace (``graphax.incremental``); as
each vertex is eliminated its equations are appended to that trace's growing
jaxpr. A :class:`FaceSink`, installed for the duration of an ``eliminate`` call,
records -- per FACE (one ``pred -> vertex -> succ`` contraction) -- the edge
identities and the equation-index RANGES of the contraction+join and of each
approximation. It records ranges only (no ops, no sub-trace), so it is
deterministic and leaves the exact-AD path untouched (active only when a face
tokenizer installs it, gated in ``core._eliminate_vertex``).
"""
from __future__ import annotations

import threading
from collections import namedtuple

_FACE = threading.local()
_XFORM = threading.local()


def get_face_sink():
    return getattr(_FACE, "sink", None)


def get_transform_log():
    return getattr(_XFORM, "log", None)


FaceRecord = namedtuple(
    "FaceRecord", ["vertex", "in_edge", "central", "out_edge",
                   "start", "end", "approx"])
FaceRecord.__doc__ = """One FACE (one ``in_edge -> central -> out_edge``
contraction) of one vertex elimination.

``start`` / ``end`` bound the equations the whole face appended to the
persistent frame; ``approx`` is the list of :class:`ApproxRecord` for the
approximations applied INSIDE that range, in application order."""


# ---------------------------------------------------------------------------
# per-face approximation SLOTS
# ---------------------------------------------------------------------------
#
# A face is ``res = contract(lhs, rhs)``: two contraction operands and one
# result, so an approximation applied on a face lands in exactly ONE of three
# operand slots. alphagrad names them ``(pre, post, new)``, graphax names them
# ``(lhs, rhs, res)`` -- same order, same operands:
#
#     slot 0   lhs / pre    the in_edge Jacobian    (before the contraction)
#     slot 1   rhs / post   the out_edge Jacobian   (before the contraction)
#     slot 2   res / new    the contraction result  (after it)
#
# The slot is RECORDED, never inferred from the order the records arrive in: a
# slot that declined records NOTHING, so a lone record's position says nothing
# about which operand it hit. That is the same defect class the ``SKIP`` marker
# fixed for whole faces -- identity recoverable only by position, and position
# destroyed by absence.
FACE_SLOT_NAMES = ("lhs", "rhs", "res")
N_FACE_SLOTS = len(FACE_SLOT_NAMES)

# ``"vertex"`` is a per-vertex ``transforms`` entry (uniform over every face of
# the vertex, not a per-face choice). It is applied to the contraction RESULT,
# at the same site as ``res`` and immediately before it, so it hits the SAME
# operand and shares its slot.
FACE_SLOT_INDEX = {"lhs": 0, "rhs": 1, "res": 2, "vertex": 2}


def face_slot_index(slot):
    """Operand-slot index (``0`` / ``1`` / ``2``) of a recorded approximation,
    or ``None`` when it has NO operand slot -- i.e. ``SKIP``, which drops the
    whole contraction, so there is no pre/post/new operand to approximate."""
    return FACE_SLOT_INDEX.get(slot)


ApproxRecord = namedtuple(
    "ApproxRecord", ["atype", "params", "start", "end", "slot"])
ApproxRecord.__doc__ = """One approximation applied inside an open face.

Fields:
    atype (str): ``"DIAG"`` / ``"COMPRESS"`` / ``"QUANT"`` / ``"SKIP"``.
    params (dict): the micro-action's parameters (``core._approx_meta``).
    start / end (int): the equation-index range it appended to the persistent
        frame. ``start == end`` => it emitted no jax equations (``SKIP``).
    slot (str | None): the operand it was applied to -- ``"lhs"`` / ``"rhs"`` /
        ``"res"`` for a per-face slot, ``"vertex"`` for a per-vertex transform
        (same operand as ``res``), ``None`` for ``SKIP``. Map it to an index
        with :func:`face_slot_index`.
"""


class FaceSink:
    """``n_eqns`` is a callable returning the current frame equation count."""

    def __init__(self, n_eqns):
        self.n_eqns = n_eqns
        self.faces = []          # FaceRecord, in emission order
        self._open = None        # [vertex, in_edge, central, out_edge, start, approx]
        self.vidx = None         # cached _stable_var_index(jaxpr) for the run

    def __enter__(self):
        self._prev = getattr(_FACE, "sink", None)
        _FACE.sink = self
        return self

    def __exit__(self, *exc):
        _FACE.sink = self._prev
        return False

    def open_face(self, vertex, in_edge, central, out_edge):
        self._open = [int(vertex), in_edge, central, out_edge,
                      self.n_eqns(), []]

    def approx(self, atype, params, start, end, slot=None):
        """Record ONE approximation on the currently open face.

        ``slot`` names the OPERAND it was applied to (see
        :data:`FACE_SLOT_INDEX`); ``None`` for ``SKIP``, which has no operand.
        It must be carried explicitly -- a declining slot records nothing, so
        the emitter cannot recover it from the record order."""
        if self._open is not None:
            self._open[5].append(
                ApproxRecord(atype, params, start, end, slot))

    def close_face(self):
        if self._open is not None:
            v, ie, cv, oe, start, approx = self._open
            self.faces.append(FaceRecord(v, ie, cv, oe, start,
                                         self.n_eqns(), approx))
            self._open = None


# ---------------------------------------------------------------------------
# ALWAYS-ON transform log
# ---------------------------------------------------------------------------
#
# The :class:`FaceSink` above is opt-in (``track_faces=True``) and records ONLY
# what happened between an ``open_face`` / ``close_face`` pair. An RL policy that
# picks per-path approximations needs a record of what the append-only jacobian
# ACTUALLY did that is (a) always present, and (b) truthful — a micro-action that
# was dispatched but left the tensor untouched (``apply_quant`` returns its input
# unchanged when ``val is None``) must not read as "approximation applied".
#
# :class:`TransformLog` is that record. It is installed by
# ``graphax.incremental.IncrementalJaxpr`` for every ``eliminate`` call
# regardless of ``track_faces``, and it also receives the OUTPUT-EDGE DRAINS the
# builder performs when it materializes the Jacobian.

TransformRecord = namedtuple(
    "TransformRecord",
    ["kind", "vertex", "slot", "atype", "params", "in_edge", "out_edge",
     "start", "end", "applied"],
)
TransformRecord.__doc__ = """One thing the AOJ did to an edge Jacobian.

Fields:
    kind (str): ``"transform"`` for a micro-action (Diag / Compress / Quant),
        ``"drain"`` for an output-edge transform drain.
    vertex (int | None): the vertex being eliminated; ``None`` for a drain
        (draining happens at output-materialization time, not per vertex).
    slot (str): where the action was applied — ``"vertex"`` for a per-vertex
        ``transforms`` entry, ``"lhs"`` / ``"rhs"`` / ``"res"`` for a per-face
        slot, ``"output"`` for a drain.
    atype (str): ``"DIAG"`` / ``"COMPRESS"`` / ``"QUANT"``, or ``"DRAIN"``.
    params (dict): the micro-action's parameters (``_approx_meta``), or — for a
        drain — ``{"pre": (<transform class names>,), "post": (...),
        "post_first": bool}`` describing WHAT was drained.
    in_edge / out_edge (core.Var): the edge the action was applied to. For a
        drain these are the jaxpr ``invar`` and ``outvar`` of the output edge.
    start / end (int): equation-index range this action appended to the
        persistent trace. ``start == end`` ⇒ it emitted no jax equations.
    applied (bool): whether the action CHANGED the tensor. ``False`` marks a
        structural no-op — e.g. ``Quant`` on a ``val is None`` structural edge,
        which returns its input unchanged.

        NOTE on the meaning of ``applied``: it is a TRACE-TIME fact ("the
        micro-action produced a different tensor / emitted work"), not a
        numerical claim. A value-EXACT approximation still records
        ``applied=True``: e.g. symmetric int8 quantization of an edge whose
        entries all share one magnitude (a ``reduce_sum``/``neg`` structural
        ±1 Jacobian) round-trips bit-exactly, so the Jacobian is unchanged even
        though the cast genuinely happened. Numerical equality of two traced
        tensors is not decidable while tracing, so it cannot be recorded here.
"""


class TransformLog:
    """Append-only log of the micro-actions and drains applied to edge
    Jacobians. ``n_eqns`` is a callable returning the current frame equation
    count (same contract as :class:`FaceSink`)."""

    def __init__(self, n_eqns):
        self.n_eqns = n_eqns
        self.records = []        # TransformRecord, in application order

    def __enter__(self):
        self._prev = getattr(_XFORM, "log", None)
        _XFORM.log = self
        return self

    def __exit__(self, *exc):
        _XFORM.log = self._prev
        return False

    def record(self, kind, vertex, slot, atype, params, in_edge, out_edge,
               start, end, applied):
        self.records.append(TransformRecord(
            kind, None if vertex is None else int(vertex), slot, atype,
            dict(params), in_edge, out_edge, int(start), int(end),
            bool(applied)))

    def transforms(self, applied_only=True):
        """The micro-action records; ``applied_only`` drops the no-ops."""
        return [r for r in self.records
                if r.kind == "transform" and (r.applied or not applied_only)]

    def drains(self):
        """The output-edge drain records."""
        return [r for r in self.records if r.kind == "drain"]
