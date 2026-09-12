"""Ticket dsnn-3qm.28.5 — the isolated per-case experiment.

One hand-built contraction per implicit-axis case of finding 63 deliverable (f).
Every case runs under five modes in one process:

  legacy    GRAPHAX_TILED_LEGACY=1                 the incumbent executor
  off       GRAPHAX_TILED_LAZY=off                 the lazy frame, all rules off
  nodemote  GRAPHAX_TILED_LAZY=nodemote            the landed default
  full      GRAPHAX_TILED_LAZY=full                keep the axis on the storing operand
  planner   GRAPHAX_EINSUM_GENERAL=1 PLANNER_EXACT=1

Recorded per (case, mode): the pairing census read out of ``_lazy_frame``, the
growing operand broadcasts attributed per axis, the jaxpr's growing
``broadcast_in_dim`` equations, the CPU HLO kernel census, the stored element
count of the result and which output dims are implicit.
"""
import math
import os
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor
import importlib

M = importlib.import_module("graphax.sparse.ops.matmul")
try:
    ML = importlib.import_module("graphax.sparse.ops.matmul_legacy_tiled")
except Exception:  # pragma: no cover
    ML = None


def _n(shape, k):
    return jr.normal(jr.PRNGKey(k), shape).astype(jnp.float32)


# --------------------------------------------------------------------------
# The cases. Each returns (lhs, rhs, note).
# --------------------------------------------------------------------------
# Shapes are deliberately distinct primes/small numbers so every axis of a
# growing broadcast is identifiable by its length alone.
M_META, P_BLK, K_CON, Q_OUT = 4, 3, 2, 5


def case_sparse_implicit():
    """single implicit sparse axis: the meta axis of a diagonal pair, stored by
    the lhs only. lhs (M*P, M*K) block-diagonal, rhs (M*K, M*Q) block-diagonal
    with the SAME diagonal but no physical meta axis."""
    lhs = SparseTensor(
        (DiagonalIndex(0, M_META, 0, 1, P_BLK, 1),),
        (DiagonalIndex(1, M_META, 0, 0, K_CON, 2),),
        _n((M_META, P_BLK, K_CON), 1),
    )
    rhs = SparseTensor(
        (DiagonalIndex(0, M_META, None, 1, K_CON, 0),),
        (DiagonalIndex(1, M_META, None, 0, Q_OUT, 1),),
        _n((K_CON, Q_OUT), 2),
    )
    return lhs, rhs, "meta axis stored by lhs only"


def case_sparse_implicit_rhs():
    """The mirror of the above: the rhs stores the meta axis, the lhs does not."""
    lhs = SparseTensor(
        (DiagonalIndex(0, M_META, None, 1, P_BLK, 0),),
        (DiagonalIndex(1, M_META, None, 0, K_CON, 1),),
        _n((P_BLK, K_CON), 1),
    )
    rhs = SparseTensor(
        (DiagonalIndex(0, M_META, 0, 1, K_CON, 1),),
        (DiagonalIndex(1, M_META, 0, 0, Q_OUT, 2),),
        _n((M_META, K_CON, Q_OUT), 2),
    )
    return lhs, rhs, "meta axis stored by rhs only"


def case_dense_implicit_block():
    """single implicit dense axis (block): the lhs out-side BLOCK extent of a
    diagonal pair has no physical axis."""
    lhs = SparseTensor(
        (DiagonalIndex(0, M_META, 0, 1, P_BLK, None),),
        (DiagonalIndex(1, M_META, 0, 0, K_CON, 1),),
        _n((M_META, K_CON), 1),
    )
    rhs = SparseTensor(
        (DiagonalIndex(0, M_META, 0, 1, K_CON, 1),),
        (DiagonalIndex(1, M_META, 0, 0, Q_OUT, 2),),
        _n((M_META, K_CON, Q_OUT), 2),
    )
    return lhs, rhs, "lhs block extent implicit"


def case_dense_implicit_contracted():
    """single implicit dense axis (contracted): (a, b, [c]) @ (a, c, d).
    The contracted extent c is stored by the rhs only."""
    a, b, c, d = 2, 3, 4, 5
    lhs = SparseTensor(
        (DenseIndex(0, a, 0), DenseIndex(1, b, 1)),
        (DenseIndex(2, c, None),),
        _n((a, b), 1),
    )
    rhs = SparseTensor(
        (DenseIndex(0, a, 0), DenseIndex(1, c, 1)),
        (DenseIndex(2, d, 2),),
        _n((a, c, d), 2),
    )
    return lhs, rhs, "contracted extent stored by rhs only"


def case_dense_implicit_carried():
    """single implicit dense axis (carried): ([a], b, c) @ (a, c, d).
    The batch axis a is stored by the rhs only and rides to the output."""
    a, b, c, d = 2, 3, 4, 5
    lhs = SparseTensor(
        (DenseIndex(0, a, None), DenseIndex(1, b, 0)),
        (DenseIndex(2, c, 1),),
        _n((b, c), 1),
    )
    rhs = SparseTensor(
        (DenseIndex(0, a, 0), DenseIndex(1, c, 1)),
        (DenseIndex(2, d, 2),),
        _n((a, c, d), 2),
    )
    return lhs, rhs, "carried batch axis stored by rhs only"


def case_double_implicit_contracted():
    """double implicit contracted axis: (a, b, [c]) @ (a, [c], d).
    Neither side stores c, so it is an analytic scale."""
    a, b, c, d = 2, 3, 4, 5
    lhs = SparseTensor(
        (DenseIndex(0, a, 0), DenseIndex(1, b, 1)),
        (DenseIndex(2, c, None),),
        _n((a, b), 1),
    )
    rhs = SparseTensor(
        (DenseIndex(0, a, 0), DenseIndex(1, c, None)),
        (DenseIndex(2, d, 1),),
        _n((a, d), 2),
    )
    return lhs, rhs, "contracted extent stored by neither"


def case_double_implicit_batch():
    """double implicit batch axis: ([a], b, c) @ ([a], c, d).
    Neither side stores a; it rides to the output as an implicit dim."""
    a, b, c, d = 2, 3, 4, 5
    lhs = SparseTensor(
        (DenseIndex(0, a, None), DenseIndex(1, b, 0)),
        (DenseIndex(2, c, 1),),
        _n((b, c), 1),
    )
    rhs = SparseTensor(
        (DenseIndex(0, a, None), DenseIndex(1, c, 0)),
        (DenseIndex(2, d, 1),),
        _n((c, d), 2),
    )
    return lhs, rhs, "batch axis stored by neither"


def case_uniform_operand():
    """uniform operand: the lhs has val=None on every dim."""
    a, b, c, d = 2, 3, 4, 5
    lhs = SparseTensor(
        (DenseIndex(0, a, None), DenseIndex(1, b, None)),
        (DenseIndex(2, c, None),),
        None,
    )
    rhs = SparseTensor(
        (DenseIndex(0, a, 0), DenseIndex(1, c, 1)),
        (DenseIndex(2, d, 2),),
        _n((a, c, d), 2),
    )
    return lhs, rhs, "lhs val is None on every dim"


def case_lcm_grid():
    """genuine LCM grid: the two sides factor the contracted logical extent
    into meta 4 against meta 6 (gcd 2, lcm 12). explicit_matmul_test's
    test_block_block_gcd shapes."""
    a, b, c, d, e, f = 4, 6, 2, 5, 3, 7
    lhs = SparseTensor(
        (DiagonalIndex(0, a, 0, 1, d, 1),),
        (DiagonalIndex(1, a, 0, 0, e, 2),),
        _n((a, d, e), 1),
    )
    rhs = SparseTensor(
        (DiagonalIndex(0, b, 0, 1, c, 1),),
        (DiagonalIndex(1, b, 0, 0, f, 2),),
        _n((b, c, f), 2),
    )
    return lhs, rhs, "meta 4 against meta 6, gcd 2 lcm 12"


def case_spatial_sparse():
    """spatial_sparse pairing: a diagonal pair that lives on the lhs alone and
    is not contracted. The lhs has two primal dims, the rhs one out dim, so the
    leftmost lhs primal dim (the partner of an lhs out dim) is never contracted
    and ``_unmatched_pair`` classifies it as ``spatial_sparse_lhs``."""
    s_, b, c = 3, 4, 5
    lhs = SparseTensor(
        (DenseIndex(0, b, 0), DiagonalIndex(1, s_, 1, 2, None, None)),
        (DiagonalIndex(2, s_, 1, 1, None, None), DenseIndex(3, c, 2)),
        _n((b, s_, c), 1),
    )
    rhs = SparseTensor(
        (DenseIndex(0, c, 0),),
        (DenseIndex(1, b, 1),),
        _n((c, b), 2),
    )
    return lhs, rhs, "an lhs-only diagonal pair riding through"


def case_partially_stored():
    """partially stored extent: an aligned pair whose merged meta extent T is
    contributed to by both sides, with neither side contributing all of it.

    The frame algebra makes this hard to reach: ``m_l`` and ``m_r`` only take
    the values 1 or T on an aligned pair (see the finding). This construction
    is the closest reachable neighbour — the lhs carries the meta on its OUTER
    slot, the rhs on its BLOCK slot, so both sides contribute T and the frame
    falls through every lazy rule with nothing to shrink."""
    lhs = SparseTensor(
        (DiagonalIndex(0, M_META, 0, 1, P_BLK, 1),),
        (DiagonalIndex(1, M_META, 0, 0, K_CON, 2),),
        _n((M_META, P_BLK, K_CON), 1),
    )
    rhs = SparseTensor(
        (DiagonalIndex(0, M_META, 0, 1, K_CON, 1),),
        (DiagonalIndex(1, M_META, 0, 0, Q_OUT, 2),),
        _n((M_META, K_CON, Q_OUT), 2),
    )
    return lhs, rhs, "both sides store the merged meta extent (no implicit axis)"


CASES = {
    "single_implicit_sparse": case_sparse_implicit,
    "single_implicit_sparse_rhs": case_sparse_implicit_rhs,
    "single_implicit_dense_block": case_dense_implicit_block,
    "single_implicit_dense_contracted": case_dense_implicit_contracted,
    "single_implicit_dense_carried": case_dense_implicit_carried,
    "double_implicit_contracted": case_double_implicit_contracted,
    "double_implicit_batch": case_double_implicit_batch,
    "uniform_operand": case_uniform_operand,
    "lcm_grid": case_lcm_grid,
    "spatial_sparse": case_spatial_sparse,
    "no_implicit_control": case_partially_stored,
}


def _bmm(a, b):
    return jnp.einsum("abc,acd->abd", a, b)


def _mm(a, b):
    return a @ b


def _spatial(a, b):
    return jnp.einsum("abcd,de->abce", a, b)


# The dense oracle per case, applied to ``lhs.dense()`` and ``rhs.dense()``.
ORACLES = {
    "single_implicit_sparse": _mm,
    "single_implicit_sparse_rhs": _mm,
    "single_implicit_dense_block": _mm,
    "single_implicit_dense_contracted": _bmm,
    "single_implicit_dense_carried": _bmm,
    "double_implicit_contracted": _bmm,
    "double_implicit_batch": _bmm,
    "uniform_operand": _bmm,
    "lcm_grid": _mm,
    "spatial_sparse": _spatial,
    "no_implicit_control": _mm,
}


# The optimum on paper, per case: the stored element count the emission must
# ask for, and the multiply-adds it must perform. Derived by hand in the
# finding; the probe checks the engine against these.
OPTIMUM = {
    #                        stored, macs, note
    "single_implicit_sparse": (4 * 3 * 5, 4 * 3 * 2 * 5,
                               "M*P*Q buffer, M*P*K*Q products"),
    "single_implicit_sparse_rhs": (4 * 3 * 5, 4 * 3 * 2 * 5,
                                   "M*P*Q buffer, M*P*K*Q products"),
    "single_implicit_dense_block": (4 * 5, 4 * 2 * 5,
                                    "block stays implicit: M*Q buffer, M*K*Q products"),
    "single_implicit_dense_contracted": (2 * 3 * 5, 2 * 4 * 5 + 2 * 3 * 5,
                                         "sum rhs over c, then one product per output"),
    "single_implicit_dense_carried": (2 * 3 * 5, 2 * 3 * 4 * 5,
                                      "a*b*d buffer, a*b*c*d products"),
    "double_implicit_contracted": (2 * 3 * 5, 2 * 3 * 5,
                                   "c folds into scalar_mult, one product per output"),
    "double_implicit_batch": (3 * 5, 3 * 4 * 5,
                              "a stays implicit: b*d buffer, b*c*d products"),
    "uniform_operand": (2 * 5, 2 * 4 * 5,
                        "b implicit, lhs is ones: a*d buffer, a*c*d products"),
    "lcm_grid": (280, 12 * 5 * 3 * 7,
                 "the reconciled band buffer, LCM grid products"),
    "spatial_sparse": (3 * 4 * 4, 3 * 4 * 5 * 4,
                       "the lhs diagonal rides through: s*b*b_r buffer"),
    "no_implicit_control": (4 * 3 * 5, 4 * 3 * 2 * 5,
                            "both sides store the meta: M*P*Q buffer"),
}


MODES = {
    "legacy": {"GRAPHAX_TILED_LEGACY": "1", "GRAPHAX_EINSUM_GENERAL": "0"},
    "lazy_off": {"GRAPHAX_TILED_LAZY": "off", "GRAPHAX_EINSUM_GENERAL": "0"},
    "lazy_nodemote": {"GRAPHAX_TILED_LAZY": "nodemote", "GRAPHAX_EINSUM_GENERAL": "0"},
    "lazy_full": {"GRAPHAX_TILED_LAZY": "full", "GRAPHAX_EINSUM_GENERAL": "0"},
    "planner": {"GRAPHAX_EINSUM_GENERAL": "1", "GRAPHAX_PLANNER_EXACT": "1"},
}

_ENV_KEYS = (
    "GRAPHAX_TILED_LEGACY",
    "GRAPHAX_TILED_LAZY",
    "GRAPHAX_EINSUM_GENERAL",
    "GRAPHAX_PLANNER_EXACT",
)


class mode_env:
    def __init__(self, mode):
        self.mode = mode

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(MODES[self.mode])
        return self

    def __exit__(self, *a):
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        for k, v in self.saved.items():
            if v is not None:
                os.environ[k] = v
        return False


# --------------------------------------------------------------------------
# Instrumentation
# --------------------------------------------------------------------------
class Census:
    """Wraps ``_as_shape`` (in both the live module and the incumbent's verbatim
    copy) and ``_prepare_contraction_views`` / ``_lazy_frame``. Every broadcast
    that GROWS the buffer is recorded, and attributed to the axis that grew."""

    def __init__(self):
        self.growing = []      # dicts: side, slot, pair, factor, before, after
        self.frames = []       # per _lazy_frame call, one dict per pair
        self._layout = None

    # ---- attribution ------------------------------------------------------
    def _axis_labels(self, side, n_pairs, n_lead):
        """Axis label per position of the lhs/rhs "unmerged" broadcast target.

        lhs: [meta_outer_i, meta_rest_i] * N, then block_i * N, then split_i * N
        rhs: [meta_outer_i, meta_rest_i] * N, then split_i * N, then shared_i * N
        followed by the physical leftovers.
        """
        lab = []
        for i in range(n_pairs):
            lab += [("meta", i), ("meta", i)]
        if side == "lhs":
            lab += [("block", i) for i in range(n_pairs)]
            lab += [("split", i) for i in range(n_pairs)]
        else:
            lab += [("split", i) for i in range(n_pairs)]
            lab += [("shared", i) for i in range(n_pairs)]
        lab += [("leftover", k) for k in range(n_lead)]
        return lab

    def install(self):
        self._mods = [m for m in (M, ML) if m is not None]
        self._saved = []
        cen = self

        for mod in self._mods:
            orig_as_shape = mod._as_shape

            def make(orig):
                def wrapped(view, target_shape, *, mode):
                    out = orig(view, target_shape, mode=mode)
                    if mode == "broadcast":
                        before = int(math.prod(view.shape)) if view.shape else 1
                        after = int(math.prod(tuple(target_shape))) or 1
                        if after > before:
                            cen._record(view.shape, tuple(target_shape), before, after)
                    return out

                return wrapped

            self._saved.append((mod, "_as_shape", orig_as_shape))
            mod._as_shape = make(orig_as_shape)

        # _prepare_contraction_views tells us the pair count, so the axis
        # layout of the two broadcast calls above is known exactly. Each module
        # keeps its OWN body; only the layout hook is shared.
        for mod in self._mods:
            if not hasattr(mod, "_prepare_contraction_views"):
                continue
            orig_pcv = mod._prepare_contraction_views

            def make_pcv(orig):
                def pcv(lhs_val, rhs_val, pairs, *a, **kw):
                    cen._layout = (
                        len(pairs),
                        [p.pairing_type for p in pairs],
                        len(lhs_val.shape) - 3 * len(pairs),
                        len(rhs_val.shape) - 3 * len(pairs),
                    )
                    try:
                        return orig(lhs_val, rhs_val, pairs, *a, **kw)
                    finally:
                        cen._layout = None

                return pcv

            self._saved.append((mod, "_prepare_contraction_views", orig_pcv))
            mod._prepare_contraction_views = make_pcv(orig_pcv)

        orig_lazy = M._lazy_frame

        def lazy(lhs_val, rhs_val, pairs):
            eff, lz, dem = orig_lazy(lhs_val, rhs_val, pairs)
            rec = []
            for i, p in enumerate(pairs):
                lo, lb, ls = M._slot_phys(lhs_val, i)
                ro, rb, rs = M._slot_phys(rhs_val, i)
                ol, orr = int(p.lhs.outer_len), int(p.rhs.outer_len)
                T, G = math.lcm(ol, orr), math.gcd(ol, orr)
                m_l = lo * ((T // ol) if ls != 1 else 1)
                m_r = ro * ((T // orr) if rb != 1 else 1)
                rec.append(
                    dict(
                        pair=i,
                        pairing_type=p.pairing_type,
                        ol=ol,
                        orr=orr,
                        T=T,
                        G=G,
                        aligned=bool(T == G or ol == 1 or orr == 1),
                        can=p.pairing_type in M._LAZY_PAIRINGS,
                        lhs_slots=(lo, lb, ls),
                        rhs_slots=(ro, rb, rs),
                        m_l=m_l,
                        m_r=m_r,
                        meta_lazy=bool(lz[i].meta if hasattr(lz[i], "meta") else lz[i][0]),
                        lhs_lazy=bool(lz[i][1]),
                        rhs_lazy=bool(lz[i][2]),
                        demote=dem[i],
                        lhs_block=int(p.lhs.block_len),
                        rhs_shared=int(p.rhs.shared_block_len),
                        logical=int(p.logical_element_count),
                    )
                )
            self_frames = cen.frames
            self_frames.append(rec)
            return eff, lz, dem

        self._saved.append((M, "_lazy_frame", orig_lazy))
        M._lazy_frame = lazy
        return self

    def _record(self, before_shape, target, before, after):
        entry = dict(before=before, after=after, grew=after - before,
                     before_shape=list(before_shape), target=list(target),
                     axes=[])
        if self._layout is not None:
            n_pairs, ptypes, n_ll, n_rl = self._layout
            side = "lhs" if len(target) == 4 * n_pairs + n_ll else None
            if side is None and len(target) == 4 * n_pairs + n_rl:
                side = "rhs"
            if side is not None:
                lab = self._axis_labels(side, n_pairs, n_ll if side == "lhs" else n_rl)
                if len(lab) == len(target):
                    for k, (slot, pi) in enumerate(lab):
                        b = before_shape[k] if k < len(before_shape) else 1
                        t = target[k]
                        if t > b:
                            entry["axes"].append(
                                dict(side=side, slot=slot, pair=pi,
                                     pairing_type=ptypes[pi] if pi < len(ptypes) else "?",
                                     factor=int(t) // max(int(b), 1))
                            )
        self.growing.append(entry)

    def uninstall(self):
        for mod, name, orig in reversed(self._saved):
            setattr(mod, name, orig)
        self._saved = []

    def __enter__(self):
        return self.install()

    def __exit__(self, *a):
        self.uninstall()
        return False


# --------------------------------------------------------------------------
# jaxpr / HLO readers
# --------------------------------------------------------------------------
def jaxpr_growing_broadcasts(closed):
    """Count ``broadcast_in_dim`` equations whose output holds more elements
    than their input. Returns (count, grown_elements, shapes)."""
    n, grown, shapes = 0, 0, []

    def walk(jaxpr):
        nonlocal n, grown
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "broadcast_in_dim":
                inv = eqn.invars[0]
                ish = getattr(getattr(inv, "aval", None), "shape", ())
                osh = eqn.outvars[0].aval.shape
                isz = int(math.prod(ish)) if ish is not None else 1
                osz = int(math.prod(osh))
                if osz > isz:
                    n += 1
                    grown += osz - isz
                    shapes.append((tuple(ish), tuple(osh)))
            for v in eqn.params.values():
                jx = getattr(v, "jaxpr", None)
                if jx is not None:
                    walk(jx.jaxpr if hasattr(jx, "jaxpr") else jx)

    walk(closed.jaxpr if hasattr(closed, "jaxpr") else closed)
    return n, grown, shapes


_DOT_RE = re.compile(r"=\s*(\S+)\s+dot\(")
_HLO_OP = re.compile(r"^\s*(?:ROOT\s+)?%?[\w.\-]+ = (\S+) ([a-z\-_]+)\(")


def hlo_census(text):
    """Kernel census of the ENTRY computation plus the fusion bodies."""
    out = dict(dot=[], reduce=0, copy=0, top_broadcast=[], fusion=0,
               fusion_kinds={}, transpose=0, bitcast=0, total_lines=0)
    in_entry = False
    for line in text.splitlines():
        out["total_lines"] += 1
        s = line.strip()
        if s.startswith("ENTRY"):
            in_entry = True
            continue
        if in_entry and s == "}":
            in_entry = False
        m = _HLO_OP.match(line)
        if not m:
            continue
        shape, op = m.group(1), m.group(2)
        if op == "dot":
            out["dot"].append(shape)
        elif op == "reduce":
            out["reduce"] += 1
        elif op == "copy":
            out["copy"] += 1
        elif op == "transpose":
            out["transpose"] += 1
        elif op == "bitcast":
            out["bitcast"] += 1
        elif op == "fusion":
            out["fusion"] += 1
            k = re.search(r"kind=(\w+)", line)
            if k:
                out["fusion_kinds"][k.group(1)] = out["fusion_kinds"].get(k.group(1), 0) + 1
        elif op == "broadcast" and in_entry:
            out["top_broadcast"].append(shape)
    return out


def dims_report(st):
    return [
        dict(id=int(d.id), logical=int(d.logical_size), size=int(d.size),
             axis=(None if d.axis is None else int(d.axis)),
             sparse=bool(d.is_sparse),
             block=(None if getattr(d, "block_size", None) is None
                    else int(d.block_size)),
             block_axis=(None if getattr(d, "block_axis", None) is None
                         else int(d.block_axis)),
             kind=type(d).__name__)
        for d in st.dims
    ]


def logical_size(st):
    p = 1
    for d in st.dims:
        p *= int(d.logical_size)
    return p
