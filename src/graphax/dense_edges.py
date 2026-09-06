"""The DENSE-CONTRACTION mode: every edge a plain array (ticket dsnn-3qm.69).

WHY THIS EXISTS. ``jacve(..., sparse_representation=False)`` runs the SAME
``SparseTensor`` contractions as ``sparse_representation=True``; that flag
changes the RETURN form only (``core.py:453``, ``core.py:3199``). So an
approximated plan had no value oracle: ``jax.grad`` is an oracle for the exact
plan alone, and comparing the sparse engine against its own dense output packing
proves the packing, not the values (finding 61 verdict 4).

This module is a SECOND engine. Every edge is a plain ``jnp`` array of shape
``out_var.aval.shape + in_var.aval.shape``. Every contraction is a
``jnp.tensordot`` over the eliminated variable's axes. The three approximations
are defined on arrays:

* **Quant** — a dtype cast of the edge.
* **Reduce** — the configured kind (default the mean) over the axis, broadcast
  back to the full extent.
* **Diag(i, j, f)** — zero everything outside the block-diagonal blocks of the
  ``(i, j)`` pair.

Sparse against dense on the SAME plan is then a real value check.

WHAT IS SHARED AND WHAT IS NOT (owner rulings D5-D10, 2026-09-06). The
accumulation is INDEPENDENT: not one line of ``_eliminate_vertex`` runs here, so
a disagreement is evidence. The graph, the face keys and the order are shared
(``_build_graph``, ``_prune_graph``, ``_checkify_order``, ``_stable_var_index``),
so both engines run the same plan on the same graph. The DEFINITION of each
approximation is shared through the adapter below, so a rule legal on one side
stays legal on the other wherever the geometry allows it.

THE ADAPTER (D7). A face hook is a callable taking a ``SparseTensor``. A dense
edge is wrapped as a FULLY DENSE ``SparseTensor`` (one ``DenseIndex`` per axis,
``axis == position``), handed to the hook, and unwrapped with
``dense(keep_quantization=True)``. That unwrap already IS the dense semantics
listed above: ``apply_compress`` marks the axis implicit and ``dense()``
broadcasts it back; ``apply_diag`` builds the pair and ``dense()`` re-embeds it
with zeros off the blocks; ``keep_quantization`` keeps the narrow dtype instead
of promoting it away. Dense-native hooks are a later ticket.

AN ILLEGAL ACTION RAISES (D8). There is no best-effort skip here and no
``on_illegal`` flag. The sparse path keeps its own documented silent skip
(``core._apply_face_transform``), so the two engines can still apply DIFFERENT
action sets for the same masked hook — measured on the 2-layer MLP: Reduce
applied 8 times sparse against 10 times dense, Diag 2 against 6. That is a
different plan, not a defect, and a value comparison of it is meaningless. Hence
:class:`ActionCensus`: oracle B wraps the plan once with :func:`census_plan`,
runs both engines, and calls :func:`compare_censuses` BEFORE it compares values.

NOT A MEASUREMENT PATH. A dense edge is ``out_size * primal_size`` numbers.
``max_bytes`` (default 2 GiB) raises instead of letting a node OOM.
``count_ops`` is refused: the counts would describe the oracle, not the engine.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Sequence, Tuple

import jax.numpy as jnp
import jax._src.core as core

from .sparse.micro_actions import (
    Compress, Diag, Quant, apply_compress, apply_diag, apply_quant,
)
from .sparse.tensor import DenseIndex, SparseTensor
from .sparse.utils import zeros_like

__all__ = [
    "ActionCensus",
    "CensusMismatch",
    "DenseBudgetExceeded",
    "census_plan",
    "compare_censuses",
    "dense_vertex_elimination",
]

# The default ceiling on the total bytes of all live dense edges. A dense edge
# is out_size * primal_size numbers, so this mode is a small-shape oracle; the
# ceiling turns "too big" into a message instead of a killed job.
DEFAULT_MAX_BYTES = 2 * 1024 ** 3


class DenseBudgetExceeded(RuntimeError):
    """The dense edges of this graph do not fit in ``max_bytes``."""


class CensusMismatch(AssertionError):
    """Two eliminations did not apply the same actions, so their gradients are
    not comparable."""


# ---------------------------------------------------------------------------
# the adapter: a plain array <-> a fully dense SparseTensor
# ---------------------------------------------------------------------------
def wrap(arr, out_ndim: int) -> SparseTensor:
    """A plain array as a fully dense ``SparseTensor``.

    One ``DenseIndex`` per axis with ``axis == position`` (parameter layout),
    the out/primal split at ``out_ndim``. A rank-0 edge keeps rank 0 — unlike
    ``ops.utils._arr2st``, which expands it to ``(1,)``.
    """
    dims = tuple(DenseIndex(i, int(s), i) for i, s in enumerate(arr.shape))
    return SparseTensor(dims[:out_ndim], dims[out_ndim:], arr,
                        check_consistency=False)


def unwrap(st: SparseTensor):
    """Back to a plain array.

    ``keep_quantization=True`` so a Quant survives as a DTYPE. The default
    promotes to the full mul-capable dtype, which would keep the rounded values
    but drop the narrow storage — and then a contraction of two Quant'd operands
    would run wide, against the ruling of 2026-09-06 that it stays narrow.
    """
    return st.dense(keep_quantization=True)


def apply_slot(arr, out_ndim: int, hook, *, where: str):
    """Apply ONE slot hook to ONE dense edge, through the adapter.

    A ``Diag`` / ``Compress`` / ``Quant`` is dispatched to its own helper; any
    other callable is handed the wrapped tensor and may return either a tensor
    or a chosen micro-action (the chooser form ``core._apply_face_transform``
    also supports). An action that does not fit RAISES (D8): the sparse path's
    best-effort skip is not reproduced here.
    """
    if hook is None:
        return arr
    st = wrap(arr, out_ndim)
    if isinstance(hook, Diag):
        st = apply_diag(st, hook)
    elif isinstance(hook, Compress):
        st = apply_compress(st, hook)
    elif isinstance(hook, Quant):
        st = apply_quant(st, hook)
    elif callable(hook):
        chosen = hook(st)
        if chosen is None:
            return arr
        if isinstance(chosen, Diag):
            st = apply_diag(st, chosen)
        elif isinstance(chosen, Compress):
            st = apply_compress(st, chosen)
        elif isinstance(chosen, Quant):
            st = apply_quant(st, chosen)
        else:
            st = chosen
    else:
        raise TypeError(
            f"Unknown transform of type {type(hook).__name__} in {where}; "
            "expected None, Diag, Compress, Quant, or a callable taking a "
            "SparseTensor.")
    if not isinstance(st, SparseTensor):
        # a dense-native hook may hand back a plain array already
        out = st
    else:
        out = unwrap(st)
    if tuple(out.shape) != tuple(arr.shape):
        raise ValueError(
            f"a dense-mode transform changed the edge shape in {where}: "
            f"{tuple(arr.shape)} -> {tuple(out.shape)}. Every approximation "
            "keeps the logical shape (Quant casts, Reduce broadcasts back, "
            "Diag zeroes off the blocks), so this is a bug in the hook.")
    return out


# ---------------------------------------------------------------------------
# the applied-action census
# ---------------------------------------------------------------------------
class ActionCensus:
    """What one elimination actually APPLIED, comparable across engines.

    One entry per slot hook that ran, in application order:
    ``(vertex, face_key, slot, atype, params, applied)``.

    ``atype`` is ``"DIAG"`` / ``"COMPRESS"`` / ``"QUANT"`` for a typed action
    and ``"CALLABLE"`` for a hook that returned a tensor of its own.

    ``applied`` means WHAT THE PLAN DID, not what the buffer did, and the two
    hook kinds read it differently on purpose:

    * a TYPED action records ``applied=True`` whenever it was dispatched and
      did not raise. Whether the buffer changed is not comparable across
      engines: ``apply_quant`` is a structural no-op on a ``val is None``
      sparse edge and a real cast on its dense wrap, and ``apply_compress``
      returns its input unchanged on an already-implicit axis while the dense
      wrap always builds a new array. Both are the SAME approximation.
    * a CALLABLE records ``applied=True`` only when it returned a DIFFERENT
      object. A chooser is opaque, so "did it decline?" is the only comparable
      fact — and it is exactly the fact the masked-hook divergence turns on: a
      hook that is legal on a dense edge and illegal on the sparse one returns
      its input there and is caught here.
    """

    def __init__(self):
        self.records: list = []

    def add(self, vertex, key, slot, atype, params, applied):
        self.records.append((None if vertex is None else int(vertex),
                             tuple(key) if key is not None else None,
                             str(slot), str(atype),
                             tuple(sorted(params.items())), bool(applied)))

    def applied(self) -> tuple:
        """The records whose hook changed the tensor, in application order."""
        return tuple(r[:-1] for r in self.records if r[-1])

    def counts(self) -> Dict[str, int]:
        """``atype -> number of APPLIED actions``."""
        out: Dict[str, int] = {}
        for r in self.records:
            if r[-1]:
                out[r[3]] = out.get(r[3], 0) + 1
        return out

    def __len__(self):
        return len(self.records)

    def __repr__(self):
        return f"ActionCensus({len(self.records)} records, {self.counts()})"


def _atype_and_params(hook) -> Tuple[str, dict]:
    if isinstance(hook, Diag):
        return "DIAG", {"i": int(hook.i), "j": int(hook.j),
                        "factor": int(hook.factor)}
    if isinstance(hook, Compress):
        return "COMPRESS", {"kind": hook.kind,
                            "axes": tuple(int(a) for a in hook.axes)}
    if isinstance(hook, Quant):
        return "QUANT", {"dtype": hook.dtype}
    return "CALLABLE", {}


def _dispatch(st, action):
    """Apply ONE typed micro-action to a ``SparseTensor``."""
    if isinstance(action, Diag):
        return apply_diag(st, action)
    if isinstance(action, Compress):
        return apply_compress(st, action)
    return apply_quant(st, action)


def _changed(before, after) -> bool:
    """Did the hook change the tensor? The ``core._micro_applied`` test, for
    both a ``SparseTensor`` (either engine's operand) and a plain array."""
    if after is before:
        return False
    if isinstance(before, SparseTensor) and isinstance(after, SparseTensor):
        return not (
            after.val is before.val
            and after.scalar_mult is before.scalar_mult
            and after.fill_value is before.fill_value
            and after.out_dims == before.out_dims
            and after.primal_dims == before.primal_dims
            and after.pre_transforms == before.pre_transforms
            and after.post_transforms == before.post_transforms)
    return True


_SLOT_NAMES = ("lhs", "rhs", "res")


def _census_hook(hook, census: ActionCensus, vertex, key, slot):
    """Wrap ONE slot hook so it records into ``census``.

    The wrapper is a plain ``(SparseTensor) -> SparseTensor`` callable, so BOTH
    engines take their callable branch and the recorded facts are produced by
    the same code on both sides. A typed action is applied by the wrapper
    itself, which is why a wrapped plan bypasses the engines' own
    ``_record_micro``; the census replaces it for the duration of an oracle run.
    """
    if hook is None:
        return None
    atype, params = _atype_and_params(hook)

    def _wrapped(st):
        if isinstance(hook, (Diag, Compress, Quant)):
            # A typed action is part of the plan. It was dispatched and it did
            # not raise, so it counts as applied on both engines.
            out = _dispatch(st, hook)
            census.add(vertex, key, slot, atype, params, True)
            return out
        out = hook(st)
        if out is None:
            out = st
        if isinstance(out, (Diag, Compress, Quant)):
            # the CHOOSER form: the hook picked an action instead of a tensor
            _a, _p = _atype_and_params(out)
            inner = _dispatch(st, out)
            census.add(vertex, key, slot, _a, _p, True)
            return inner
        census.add(vertex, key, slot, atype, params, _changed(st, out))
        return out

    return _wrapped


def census_plan(face_transforms: dict, census: ActionCensus) -> dict:
    """A copy of ``face_transforms`` whose every slot hook records into
    ``census``. Hand the SAME wrapped plan to both engines, run them, then call
    :func:`compare_censuses` before comparing any value.

    Handles the flat ``(lhs, rhs, res)`` slots and the ``SKIP_FACE`` sentinel.
    The two-op form ``((lhs, rhs, new), (jl, jr, jres))`` is wrapped slot by
    slot too.
    """
    from .core import SKIP_FACE
    if not face_transforms:
        return face_transforms
    out: Dict[int, dict] = {}
    for vertex, faces in face_transforms.items():
        if not isinstance(faces, dict):
            raise TypeError(
                "census_plan expects the nested {vertex: {face_key: slots}} "
                f"form of face_transforms; vertex {vertex} holds {faces!r}.")
        inner: Dict[Any, Any] = {}
        for key, slots in faces.items():
            if slots is SKIP_FACE:
                inner[key] = slots
                continue
            if (isinstance(slots, (tuple, list)) and len(slots) == 2
                    and all(isinstance(s, (tuple, list)) and len(s) == 3
                            for s in slots)):
                inner[key] = tuple(
                    tuple(_census_hook(h, census, vertex, key, n)
                          for h, n in zip(triple, names))
                    for triple, names in zip(
                        slots, (("lhs", "rhs", "res:new"),
                                ("res:jl", "res:jr", "res:jres"))))
                continue
            inner[key] = tuple(
                _census_hook(h, census, vertex, key, n)
                for h, n in zip(slots, _SLOT_NAMES))
        out[int(vertex)] = inner
    return out


def compare_censuses(sparse: ActionCensus, dense: ActionCensus, *,
                     site: str = "oracle B") -> None:
    """Raise :class:`CensusMismatch` unless the two eliminations applied the
    same actions, in the same order, on the same faces.

    Call this BEFORE comparing values. Two runs that applied different actions
    ran DIFFERENT PLANS, so a value difference between them says nothing about
    either engine.
    """
    a, b = sparse.applied(), dense.applied()
    if a == b:
        return
    ca, cb = sparse.counts(), dense.counts()
    diff = []
    for atype in sorted(set(ca) | set(cb)):
        if ca.get(atype, 0) != cb.get(atype, 0):
            diff.append(f"{atype}: sparse applied {ca.get(atype, 0)}, "
                        f"dense applied {cb.get(atype, 0)}")
    if not diff:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                diff.append(f"first difference at applied action {i}: "
                            f"sparse {x}, dense {y}")
                break
        if not diff:
            diff.append(f"sparse applied {len(a)} actions, dense {len(b)}")
    raise CensusMismatch(
        f"[{site}] the two engines did not apply the same actions, so their "
        f"gradients are NOT comparable: " + "; ".join(diff) +
        ". A masked hook can be legal on a dense edge and illegal on the "
        "sparse one (a sparse edge already carries diagonal pairs and implicit "
        "axes, a dense edge carries none). Pass the plan's literal decoded "
        "micro-actions instead of a chooser, or drop this plan from the oracle."
    )


# ---------------------------------------------------------------------------
# the dense elimination
# ---------------------------------------------------------------------------
def _nominal(out_var, in_var) -> tuple:
    return tuple(out_var.aval.shape) + tuple(in_var.aval.shape)


def _bytes_of(arr) -> int:
    return int(math.prod(arr.shape)) * int(jnp.dtype(arr.dtype).itemsize)


def _check_budget(total: int, max_bytes: int, where: str) -> None:
    if max_bytes is not None and total > max_bytes:
        raise DenseBudgetExceeded(
            f"the dense edges need {total / 1024 ** 3:.2f} GiB at {where}, "
            f"over the {max_bytes / 1024 ** 3:.2f} GiB ceiling. A dense edge "
            "is out_size * primal_size numbers, so this mode is a small-shape "
            "value oracle: run it on a reduced shape, or raise max_bytes "
            "deliberately.")


def dense_vertex_elimination(
    jaxpr: core.Jaxpr,
    order,
    consts: Sequence[Any],
    *args,
    has_aux: bool = False,
    argnums: Sequence[int] = (0,),
    count_ops: bool = False,
    transforms=None,
    face_transforms: dict = None,
    max_bytes: int = None,
):
    """Vertex elimination with every edge a plain array.

    The signature mirrors :func:`graphax.core.vertex_elimination_jaxpr` and the
    return value obeys the same contract, so :func:`graphax.core.jacve` routes
    here on ``dense_edges=True`` and repackages the result unchanged.
    """
    from .core import (
        SKIP_FACE, _build_graph, _checkify_order, _drain_transforms, _force,
        _prune_graph, _stable_var_index, _unpack_face_slots, prune_enabled,
    )

    if count_ops:
        raise NotImplementedError(
            "count_ops is not supported with dense_edges=True: the counts "
            "would describe the value oracle, not the engine under test. Run "
            "the counts on the sparse engine.")
    if max_bytes is None:
        max_bytes = DEFAULT_MAX_BYTES

    argnums = tuple(argnums)
    jaxpr_invars = [v for i, v in enumerate(jaxpr.invars) if i in argnums]

    env, graph, tgraph, vo_vertices = _build_graph(
        jaxpr, list(args), list(consts), argnums)
    if prune_enabled():
        _prune_graph(graph, tgraph, jaxpr, argnums)
    order = _checkify_order(order, jaxpr, vo_vertices)
    vidx = _stable_var_index(jaxpr)

    # Per-vertex legacy transforms: [(vertex, (t, ...)), ...], the same shape
    # VertexEliminator.eliminate accepts. A per-vertex dict is the per-PATH
    # form, which this mode does not implement (face_transforms is the
    # supported per-face API).
    t_dict: Dict[int, tuple] = {}
    for v, ts in (transforms or ()):
        if isinstance(ts, dict):
            raise NotImplementedError(
                "the per-PATH transforms dict is not supported with "
                "dense_edges=True; use face_transforms.")
        t_dict[int(v)] = tuple(ts)

    # --- build: densify every elemental edge exactly once ------------------
    # `_drain_transforms` folds the primitives' queued relabels (reshape /
    # transpose / slice / concatenate / broadcast) into the data; `.dense()`
    # materializes. Draining order is irrelevant for a single edge: a
    # pre_transform relabels the primal side and a post_transform the out side,
    # so the two act on disjoint axes. `post_first=False` is the same order the
    # final-output drain uses (core.py:3191).
    D: Dict[Any, Dict[Any, Any]] = defaultdict(dict)
    DT: Dict[Any, Dict[Any, Any]] = defaultdict(dict)
    total = 0
    for u in list(graph.keys()):
        for v in list(graph[u].keys()):
            tensor = _force(graph[u][v])
            if tensor is None:
                continue  # null edge (e.g. stop_gradient); treat as zero
            arr = _drain_transforms(tensor.copy(), post_first=False).dense()
            nominal = _nominal(v, u)
            if tuple(arr.shape) != nominal:
                raise ValueError(
                    f"the elemental Jacobian d{v}/d{u} densified to "
                    f"{tuple(arr.shape)}, not the nominal {nominal}. A dense "
                    "edge is an array of shape out_shape + primal_shape; a "
                    "drained elemental that is not is a bug in the partial "
                    "rule or in the drain.")
            total += _bytes_of(arr)
            _check_budget(total, max_bytes, "graph build")
            D[u][v] = arr
            DT[v][u] = arr

    def _ordered(keys):
        return sorted(keys, key=lambda k: vidx.get(k, 1 << 30))

    # --- eliminate ---------------------------------------------------------
    for vertex in order:
        eqn = jaxpr.eqns[int(vertex) - 1]
        v_transforms = t_dict.get(int(vertex), ())
        v_faces = (face_transforms or {}).get(int(vertex)) or {}
        for central in eqn.outvars:
            if central not in D:
                continue  # dead or already-eliminated vertex
            central_ndim = central.aval.ndim
            for out_edge in _ordered(D[central].keys()):
                post_raw = D[central][out_edge]
                out_ndim = out_edge.aval.ndim
                for in_edge in _ordered(DT.get(central, {}).keys()):
                    pre = DT[central][in_edge]
                    post = post_raw
                    key = (vidx.get(in_edge), vidx.get(out_edge))
                    where = (f"vertex {vertex}, face "
                             f"(in={in_edge}, out={out_edge})")

                    lhs_t = rhs_t = res_t = new_t = join_t = None
                    slots = v_faces.get(key)
                    if slots is SKIP_FACE:
                        # SKIP: no contraction, no join, no store. Same
                        # semantics as core.py's SKIP_FACE branch.
                        continue
                    if slots is not None:
                        (lhs_t, rhs_t, res_t, new_t,
                         join_t) = _unpack_face_slots(slots, vertex)

                    pre = apply_slot(pre, central_ndim, lhs_t,
                                     where=f"slot lhs of {where}")
                    post = apply_slot(post, out_ndim, rhs_t,
                                      where=f"slot rhs of {where}")

                    # THE CONTRACTION. `post` is out_edge <- central, `pre` is
                    # central <- in_edge, so the shared axes are the eliminated
                    # variable's. tensordot is the general form: it reduces to
                    # jnp.matmul for a 1-axis vertex and, for a rank-0 vertex,
                    # to the outer product that is the `X @ scalar == scalar *
                    # X` scale of dsnn-3qm.68.
                    edge = jnp.tensordot(post, pre, axes=central_ndim)
                    edge = apply_slot(edge, out_ndim, new_t,
                                      where=f"slot new of {where}")

                    old = D[in_edge].get(out_edge)
                    if old is not None:
                        # THE JOIN: old = jr(old) + jl(fresh). Both addends are
                        # hooked, the same two-op semantics core.py documents;
                        # a merge-free face runs neither (see
                        # `_unpack_face_slots`, "MERGE-FREE FACES").
                        if join_t is not None:
                            jl_t, jr_t = join_t
                            edge = apply_slot(edge, out_ndim, jl_t,
                                              where=f"slot jl of {where}")
                            old = apply_slot(old, out_ndim, jr_t,
                                             where=f"slot jr of {where}")
                        if tuple(old.shape) != tuple(edge.shape):
                            raise ValueError(
                                f"join shape mismatch at {where}: the existing "
                                f"edge is {tuple(old.shape)}, the new "
                                f"contribution {tuple(edge.shape)}.")
                        edge = edge + old

                    # Per-vertex transforms, then this face's `res` slot: the
                    # same site and the same order as core.py.
                    for _t in v_transforms:
                        edge = apply_slot(edge, out_ndim, _t,
                                          where=f"per-vertex transform of {where}")
                    edge = apply_slot(edge, out_ndim, res_t,
                                      where=f"slot res of {where}")

                    nominal = _nominal(out_edge, in_edge)
                    if tuple(edge.shape) != nominal:
                        raise ValueError(
                            f"the accumulated edge at {where} has shape "
                            f"{tuple(edge.shape)}, not the nominal {nominal}.")
                    if old is None:
                        total += _bytes_of(edge)
                        _check_budget(total, max_bytes, where)
                    D[in_edge][out_edge] = edge
                    DT[out_edge][in_edge] = edge

            # Cleanup of input and output edges for this output variable.
            if central not in vo_vertices:
                for in_vertex in list(DT.get(central, {}).keys()):
                    D.get(in_vertex, {}).pop(central, None)
            for out_vertex in list(D[central].keys()):
                DT.get(out_vertex, {}).pop(central, None)
            D.pop(central, None)
            if central not in vo_vertices:
                DT.pop(central, None)

    # --- collect -----------------------------------------------------------
    # No `.dense()` and no output-layout contract: a dense edge already IS an
    # array in parameter layout. Same outvar-major / invar-minor order as
    # core.py's collection.
    jac_vals = []
    for outvar in jaxpr.outvars:
        for invar in jaxpr_invars:
            inner = D.get(invar)
            arr = inner.get(outvar) if inner is not None else None
            jac_vals.append(arr if arr is not None
                            else zeros_like(outvar, invar))

    n = len(jaxpr_invars)
    if n > 1:
        ratio = len(jac_vals) // n
        jac_vals = [tuple(jac_vals[i * n:i * n + n]) for i in range(0, ratio)]

    if has_aux:
        return ([env[var] for var in jaxpr.outvars], jac_vals)
    return jac_vals
