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


def get_face_sink():
    return getattr(_FACE, "sink", None)


FaceRecord = namedtuple(
    "FaceRecord", ["vertex", "in_edge", "central", "out_edge",
                   "start", "end", "approx"])


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

    def approx(self, atype, params, start, end):
        if self._open is not None:
            self._open[5].append((atype, params, start, end))

    def close_face(self):
        if self._open is not None:
            v, ie, cv, oe, start, approx = self._open
            self.faces.append(FaceRecord(v, ie, cv, oe, start,
                                         self.n_eqns(), approx))
            self._open = None
