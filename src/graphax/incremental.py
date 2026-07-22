"""Incremental JAXPR builder (the "append-only jacobian", AOJ): ONE persistent
jax trace whose jaxpr GROWS as vertices are eliminated, so the Jacobian function
is preserved and extended step by step instead of re-traced every time.

The same persistent trace serves both purposes the append-only tokenizer needs:

* **the function** — ``current_jaxpr()`` materializes the jaxpr built so far
  (correct at every prefix; verified against ``jax.jacrev``), optionally
  emitting the PRIMAL outputs alongside the Jacobian;
* **the tokens** — each ``eliminate`` returns the NEW equations that step added
  (``frame`` deltas), which the tokenizer renders as one append-only block with
  var names that stay consistent across steps (they are real jaxpr Vars in one
  namespace, so an edge produced early keeps its name when used later).

Because there is one trace and one growing equation list, naming is
deterministic (first-appearance over real Vars) — no per-step re-trace, no
id-hashed sub-traces.

The builder is named :class:`IncrementalJaxpr` (it builds an incremental JAXPR
and returns more than a Jacobian); ``IncrementalJacobian`` remains as a
backward-compatible alias.
"""
from __future__ import annotations

import contextlib

from jax._src.interpreters import partial_eval as pe
from jax._src import core as jcore, source_info_util

# jax renamed core.get_aval -> core.typeof in 0.10; support both so the
# AOJ works across the versions this project pins.
_get_aval = getattr(jcore, "get_aval", None) or jcore.typeof

from .core import (
    _build_graph, _prune_graph, _eliminate_vertex, _drain_transforms, _force,
    faces_of,
)
from .sparse.tracer import TransformLog


def _transform_names(ts):
    """Class names of a tensor's queued transform list — the ``params`` payload
    of a ``kind="drain"`` :class:`TransformRecord` ("what was drained")."""
    return tuple(type(t).__name__ for t in ts)


class IncrementalJaxpr:
    """Append-only Jacobian builder over ONE persistent :mod:`jax` trace.

    Outputs are SPARSE by default: intermediates live in the graph as
    ``SparseTensor``s and :meth:`jacobian_outputs` hands those straight back
    (``dense=True`` densifies). :meth:`value_and_jacobian` additionally
    materializes the PRIMAL outputs, and :meth:`current_jaxpr` can emit a jaxpr
    whose outputs are ``(value..., jacobian...)``.

    Every ``eliminate`` call runs under an always-on
    :class:`~graphax.sparse.tracer.TransformLog` (:attr:`xlog`), so the record of
    which per-face / per-vertex approximations were applied — and which were
    dispatched but were no-ops — exists regardless of ``track_faces``. The
    output-edge drains :meth:`jacobian_outputs` performs are logged there too.
    """

    def __init__(self, jaxpr, argnums, consts, args, track_faces=False):
        self.jaxpr = jaxpr
        self.argnums = tuple(argnums)
        self.consts = list(consts)
        self._args = list(args)
        # optional per-FACE bookkeeping (edge identities + equation ranges) so a
        # tokenizer can label which path / which approximation each block is.
        self._track_faces = track_faces
        self.face_sink = None

        self.dbg = jcore.DebugInfo(
            "incr_jac", "incr_jac",
            tuple(f"a{i}" for i in range(len(args))), None)
        self.si = source_info_util.current()
        # the PERSISTENT trace — kept alive for the whole search, extended in
        # place by every eliminate() call.
        self.trace = pe.DynamicJaxprTrace(self.dbg, parent_trace=None)

        with jcore.set_current_trace(self.trace):
            self.in_tracers = [self.trace.new_arg(_get_aval(a), self.si)
                               for a in args]
            # base = primal forward + elemental edge partials, traced into the
            # persistent jaxpr once.
            self.env, self.graph, self.tgraph, self.vo = _build_graph(
                jaxpr, self.in_tracers, self.consts, self.argnums)
            _prune_graph(self.graph, self.tgraph, jaxpr, self.argnums)

        self.n_base = self._n_eqns()
        # per eliminated vertex, in elimination order:
        #   (vertex, rules, eqn_start, eqn_end, face_start, face_end,
        #    face_transforms, xlog_start, xlog_end)
        # Indexed positionally (not unpacked) by the accessors below so the
        # record can grow without breaking them.
        self.steps = []
        if track_faces:
            from .sparse.tracer import FaceSink
            self.face_sink = FaceSink(self._n_eqns)
        # ALWAYS-ON record of what each elimination actually did to the edge
        # Jacobians (micro-actions with a truthful ``applied`` flag) plus every
        # output-edge drain. Independent of ``track_faces``.
        self.xlog = TransformLog(self._n_eqns)

    # ---- frame helpers ------------------------------------------------
    def _n_eqns(self):
        # O(1): the raw ``tracing_eqns`` list length == the equation count.
        # (``get_eqns()`` REBUILDS every JaxprEqn from weakrefs -- O(E) -- so it
        # must not be used just to count; the face sink calls this O(faces) times.)
        return len(self.trace.frame.tracing_eqns)

    def base_eqns(self):
        return list(self.trace.frame.get_eqns()[:self.n_base])

    def step_eqns(self, i):
        s, e = self.steps[i][2:4]
        return list(self.trace.frame.get_eqns()[s:e])

    # ---- incremental elimination -------------------------------------
    def faces(self, vertex):
        """This vertex's FACE KEYS, in the order ``eliminate`` will visit them.

        Thin binding of :func:`graphax.faces_of` to this builder's live graph —
        call it BEFORE ``eliminate`` to enumerate the local paths a policy may
        approximate, then pass the chosen ``{key: (lhs, rhs, res)}`` back in as
        ``face_transforms``.
        """
        return faces_of(self.graph, self.tgraph, int(vertex), self.jaxpr)

    def eliminate(self, vertex, rules=(), face_transforms=None):
        """Eliminate one vertex, APPENDING its equations to the persistent
        jaxpr. Returns the new equations (this step's delta).

        ``rules`` are the PER-VERTEX transforms (applied uniformly to every
        face). ``face_transforms`` is the PER-FACE (per local path) mapping
        ``(vidx[in_edge], vidx[out_edge]) -> (lhs, rhs, res)`` — enumerate the
        keys up front with :func:`graphax.faces_of`. Both are recorded in
        ``self.steps`` so a replay of the step list reproduces this trace
        exactly; ``face_transforms`` is shallow-copied so a caller mutating its
        dict afterwards cannot rewrite history.
        """
        s = self._n_eqns()
        f0 = len(self.face_sink.faces) if self.face_sink is not None else 0
        x0 = len(self.xlog.records)
        sink_cm = (self.face_sink if self.face_sink is not None
                   else contextlib.nullcontext())
        with jcore.set_current_trace(self.trace), sink_cm, self.xlog:
            _eliminate_vertex(int(vertex), self.jaxpr, self.graph,
                              self.tgraph, self.vo, False,
                              transforms=tuple(rules),
                              face_transforms=face_transforms)
        e = self._n_eqns()
        f1 = len(self.face_sink.faces) if self.face_sink is not None else 0
        x1 = len(self.xlog.records)
        self.steps.append((
            int(vertex), tuple(rules), s, e, f0, f1,
            None if face_transforms is None else dict(face_transforms),
            x0, x1,
        ))
        return self.trace.frame.get_eqns()[s:e]

    def step_faces(self, i):
        """FaceRecords for step ``i`` (empty if faces aren't tracked)."""
        if self.face_sink is None:
            return []
        f0, f1 = self.steps[i][4:6]
        return self.face_sink.faces[f0:f1]

    def step_face_transforms(self, i):
        """The per-face transform map recorded for step ``i`` (``None`` if the
        step ran without one)."""
        return self.steps[i][6]

    def step_transform_records(self, i):
        """The :class:`~graphax.sparse.tracer.TransformRecord`s step ``i``
        produced — ALWAYS populated (no ``track_faces`` needed), each carrying
        its own ``applied`` flag."""
        x0, x1 = self.steps[i][7:9]
        return self.xlog.records[x0:x1]

    def transform_records(self):
        """Every :class:`~graphax.sparse.tracer.TransformRecord` so far —
        micro-actions (``kind="transform"``) in elimination order followed by
        any output-edge drains (``kind="drain"``) logged by
        :meth:`jacobian_outputs`."""
        return list(self.xlog.records)

    def drains_recorded(self):
        """The output-edge DRAIN records logged so far. Empty until
        :meth:`jacobian_outputs` (or anything built on it) has run."""
        return self.xlog.drains()

    def all_eqns(self):
        return self.trace.frame.get_eqns()

    def all_faces(self):
        """All FaceRecords across every step, in elimination order."""
        return iter(self.face_sink.faces) if self.face_sink is not None else iter(())

    def eliminate_order(self, order, transforms=None):
        tmap = {int(v): tuple(ts) for v, ts in (transforms or ())}
        for v in order:
            self.eliminate(int(v), tmap.get(int(v), ()))

    # ---- materialize the function built so far -----------------------
    def jacobian_outputs(self, dense=False):
        """The current input->output Jacobian edge tensors, with their
        ``(outvar, input-argnum)`` labels.

        SPARSE BY DEFAULT. The edges live in the graph as ``SparseTensor``s and
        are handed back as such, so an approximation's structure (block-diagonal
        / implicit / structural ``val is None``) survives to the caller instead
        of being flattened into an N x N buffer. Pass ``dense=True`` for the
        legacy behaviour (``jnp`` arrays via ``SparseTensor.dense()``), which is
        what :meth:`current_jaxpr` needs since a jaxpr output must be an array.

        Each edge is DRAINED first — its queued pre/post Jacobian transforms
        (slice / concat / reshape / transpose relabels awaiting embed) are folded
        into its data with ``_drain_transforms(t.copy(), post_first=False)``,
        the exact call ``graphax.core.vertex_elimination_jaxpr`` makes before it
        collects outputs. Without the drain an output edge carrying a queued
        transform yields a DIFFERENT (silently wrong) Jacobian. The drain works
        on a ``copy()`` and is NOT written back into the graph: this builder is
        live and may keep eliminating, so the graph must keep the un-drained
        edge that later eliminations expect. Every drain is logged on
        :attr:`xlog` as a ``kind="drain"`` record naming the transforms folded
        in and the ``(invar, outvar)`` edge they were on.

        Returns:
            (outs, labels): ``outs`` are ``SparseTensor``s (or arrays when
            ``dense``); ``labels[k]`` is ``(outvar, argnum)`` for ``outs[k]``.
        """
        outs, labels = [], []
        with jcore.set_current_trace(self.trace):
            for ov in self.jaxpr.outvars:
                for ii in self.argnums:
                    iv = self.jaxpr.invars[ii]
                    inner = self.graph.get(iv)
                    edge = inner.get(ov) if inner is not None else None
                    t = _force(edge) if edge is not None else None
                    if t is None:
                        continue
                    # Mirror core.vertex_elimination_jaxpr's final drain
                    # EXACTLY (``post_first=False``: pre then post).
                    _pre, _post = t.pre_transforms, t.post_transforms
                    _s = self._n_eqns()
                    t = _drain_transforms(t.copy(), post_first=False)
                    if _pre or _post:
                        self.xlog.record(
                            "drain", None, "output", "DRAIN",
                            {"pre": _transform_names(_pre),
                             "post": _transform_names(_post),
                             "post_first": False},
                            iv, ov, _s, self._n_eqns(), True)
                    outs.append(t.dense() if dense else t)
                    labels.append((ov, ii))
        return outs, labels

    def value_outputs(self):
        """The PRIMAL outputs of the traced function, in ``jaxpr.outvars``
        order, as values in the persistent trace.

        These have always been available — ``_build_graph`` evaluates the primal
        forward pass into ``self.env`` while it builds the edge partials — they
        were simply never extracted. Returns ``(vals, labels)`` with
        ``labels[k] == (outvar, None)``; the ``None`` argnum marks a VALUE
        output, which is what makes the combined labelling of
        :meth:`current_jaxpr` unambiguous.
        """
        vals, labels = [], []
        for ov in self.jaxpr.outvars:
            if isinstance(ov, jcore.Literal):
                vals.append(ov.val)
            else:
                vals.append(self.env[ov])
            labels.append((ov, None))
        return vals, labels

    def value_and_jacobian(self, dense=False):
        """``(values, jacobians, labels)`` for the function built so far.

        ``values`` are the primal outputs (:meth:`value_outputs`) and
        ``jacobians`` the Jacobian edges (:meth:`jacobian_outputs`, SPARSE
        unless ``dense=True``). ``labels`` covers BOTH lists concatenated —
        ``labels[:len(values)]`` are the ``(outvar, None)`` value labels and the
        rest the ``(outvar, argnum)`` Jacobian labels — so a caller that
        flattens ``values + jacobians`` keeps a 1:1 label alignment.
        """
        vals, vlabels = self.value_outputs()
        jacs, jlabels = self.jacobian_outputs(dense=dense)
        return vals, jacs, vlabels + jlabels

    def current_jaxpr(self, include_value=False):
        """The jaxpr built so far, as a ``(jaxpr, consts, labels)`` TRIPLE.

        ``jaxpr`` is closed over ``consts``; ``labels[k]`` describes output
        ``k``:

        * ``(outvar, argnum)`` — the Jacobian of ``outvar`` w.r.t.
          ``jaxpr.invars[argnum]``;
        * ``(outvar, None)`` — the PRIMAL VALUE of ``outvar``. Only present with
          ``include_value=True``, in which case every value output precedes
          every Jacobian output (``value..., jacobian...``). The ``None``
          argnum is the value/Jacobian discriminator.

        Jaxpr outputs must be arrays, so the Jacobian edges are densified HERE
        even though :meth:`jacobian_outputs` is sparse by default — use that
        accessor (or :meth:`value_and_jacobian`) when the sparse structure is
        what you want.
        """
        outs, labels = self.jacobian_outputs(dense=True)
        if include_value:
            vals, vlabels = self.value_outputs()
            outs, labels = list(vals) + list(outs), vlabels + labels
        res = self.trace.to_jaxpr(outs, self.dbg, self.si)
        jaxpr, consts = res[0], res[1]
        return jaxpr, consts, labels


# Backward-compatible alias: this class used to be called ``IncrementalJacobian``
# (alphagrad and the tokenizer still import that name). It now returns more than
# a Jacobian, hence the rename.
IncrementalJacobian = IncrementalJaxpr
