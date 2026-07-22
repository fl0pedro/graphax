"""Incremental Jacobian builder: ONE persistent jax trace whose jaxpr GROWS as
vertices are eliminated, so the Jacobian function is preserved and extended
step by step instead of re-traced every time.

The same persistent trace serves both purposes the append-only tokenizer needs:

* **the function** — ``current_jaxpr()`` materializes the Jacobian jaxpr built
  so far (correct at every prefix; verified against ``jax.jacrev``);
* **the tokens** — each ``eliminate`` returns the NEW equations that step added
  (``frame`` deltas), which the tokenizer renders as one append-only block with
  var names that stay consistent across steps (they are real jaxpr Vars in one
  namespace, so an edge produced early keeps its name when used later).

Because there is one trace and one growing equation list, naming is
deterministic (first-appearance over real Vars) — no per-step re-trace, no
id-hashed sub-traces.
"""
from __future__ import annotations

import contextlib

from jax._src.interpreters import partial_eval as pe
from jax._src import core as jcore, source_info_util

from .core import _build_graph, _prune_graph, _eliminate_vertex, _force


class IncrementalJacobian:
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
            self.in_tracers = [self.trace.new_arg(jcore.get_aval(a), self.si)
                               for a in args]
            # base = primal forward + elemental edge partials, traced into the
            # persistent jaxpr once.
            self.env, self.graph, self.tgraph, self.vo = _build_graph(
                jaxpr, self.in_tracers, self.consts, self.argnums)
            _prune_graph(self.graph, self.tgraph, jaxpr, self.argnums)

        self.n_base = self._n_eqns()
        # per eliminated vertex: (vertex, rules, eqn_start, eqn_end, face_start, face_end)
        self.steps = []
        if track_faces:
            from .sparse.tracer import FaceSink
            self.face_sink = FaceSink(self._n_eqns)

    # ---- frame helpers ------------------------------------------------
    def _n_eqns(self):
        # O(1): the raw ``tracing_eqns`` list length == the equation count.
        # (``get_eqns()`` REBUILDS every JaxprEqn from weakrefs -- O(E) -- so it
        # must not be used just to count; the face sink calls this O(faces) times.)
        return len(self.trace.frame.tracing_eqns)

    def base_eqns(self):
        return list(self.trace.frame.get_eqns()[:self.n_base])

    def step_eqns(self, i):
        _, _, s, e, _, _ = self.steps[i]
        return list(self.trace.frame.get_eqns()[s:e])

    # ---- incremental elimination -------------------------------------
    def eliminate(self, vertex, rules=()):
        """Eliminate one vertex, APPENDING its equations to the persistent
        jaxpr. Returns the new equations (this step's delta)."""
        s = self._n_eqns()
        f0 = len(self.face_sink.faces) if self.face_sink is not None else 0
        sink_cm = (self.face_sink if self.face_sink is not None
                   else contextlib.nullcontext())
        with jcore.set_current_trace(self.trace), sink_cm:
            _eliminate_vertex(int(vertex), self.jaxpr, self.graph,
                              self.tgraph, self.vo, False,
                              transforms=tuple(rules))
        e = self._n_eqns()
        f1 = len(self.face_sink.faces) if self.face_sink is not None else 0
        self.steps.append((int(vertex), tuple(rules), s, e, f0, f1))
        return self.trace.frame.get_eqns()[s:e]

    def step_faces(self, i):
        """FaceRecords for step ``i`` (empty if faces aren't tracked)."""
        if self.face_sink is None:
            return []
        _, _, _, _, f0, f1 = self.steps[i]
        return self.face_sink.faces[f0:f1]

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
    def jacobian_outputs(self):
        """The current input->output Jacobian edge tensors (dense), with their
        (outvar, input-argnum) labels."""
        outs, labels = [], []
        with jcore.set_current_trace(self.trace):
            for ov in self.jaxpr.outvars:
                for ii in self.argnums:
                    iv = self.jaxpr.invars[ii]
                    inner = self.graph.get(iv)
                    edge = inner.get(ov) if inner is not None else None
                    t = _force(edge) if edge is not None else None
                    if t is not None:
                        outs.append(t.dense())
                        labels.append((ov, ii))
        return outs, labels

    def current_jaxpr(self):
        """The Jacobian jaxpr built so far (a closed (jaxpr, consts) pair)."""
        outs, labels = self.jacobian_outputs()
        res = self.trace.to_jaxpr(outs, self.dbg, self.si)
        jaxpr, consts = res[0], res[1]
        return jaxpr, consts, labels
