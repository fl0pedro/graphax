#!/usr/bin/env python
"""t287: what sparsity is actually there?

Walks a real target. For every elemental partial (the derivative of one
primitive) and for every accumulated Jacobian the elimination stores, it
DENSIFIES the tensor and measures the occupancy of the non-zero values
directly. No declared class is trusted; the pattern is read off the numbers.

Per tensor it reports:
  shape, elements, non-zeros, density
  replicated axes    axes along which every slice is equal (an implicit axis
                     would store one copy)
  zero-only axes     axes with at least one all-zero slice
  pair structure     for each candidate axis pair, whether the occupancy is
                     the identity (diagonal), a contiguous band (with width),
                     a general set, or full
  block period       the finest uniform block size that explains the pattern
"""
from __future__ import annotations
import os, sys, json, math, collections
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp

TOL = 1e-12


def occupancy_report(a: np.ndarray) -> dict:
    """Classify the non-zero pattern of one dense array."""
    a = np.asarray(a)
    nz = np.abs(a) > TOL
    rep = {"shape": list(a.shape), "elements": int(a.size),
           "nonzeros": int(nz.sum())}
    rep["density"] = round(rep["nonzeros"] / max(a.size, 1), 6)
    # replicated axes: every slice along the axis equal to the first
    reps = []
    for ax in range(a.ndim):
        # rtol=0: numpy's default 1e-5 declared slices differing by 2.5e-6
        # relative to be replicated. An all-zero tensor is not replicated.
        if (a.shape[ax] > 1 and nz.any()
                and np.allclose(a, np.take(a, [0], axis=ax), rtol=0.0, atol=1e-9)):
            reps.append(ax)
    rep["replicated_axes"] = reps
    # axes carrying an all-zero slice
    empt = []
    for ax in range(a.ndim):
        other = tuple(k for k in range(a.ndim) if k != ax)
        cnt = int((~nz.any(axis=other)).sum()) if other else int((~nz).sum())
        if cnt:
            empt.append({"axis": ax, "empty_slices": cnt, "of": a.shape[ax]})
    rep["axes_with_empty_slices"] = empt
    # pair structure over every axis pair, at unit granularity
    pairs = []
    for i in range(a.ndim):
        for j in range(i + 1, a.ndim):
            other = tuple(k for k in range(a.ndim) if k not in (i, j))
            # The `any` projection over the other axes is a UNION, so it read
            # `full` for tensors whose every slice was sparse: it disagreed
            # with the slices in 85% of entries (grill review 2026-09-07).
            # Two sound tests are combined instead. A DIAGONAL projection is a
            # sound superset (every slice is inside the identity). FULL is
            # sound only when every slice is full. Otherwise the pair has no
            # one structure and is "mixed"; the projection is kept beside it,
            # named as an over-estimate.
            moved = np.moveaxis(nz, (i, j), (0, 1))
            flat = moved.reshape(a.shape[i], a.shape[j], -1)
            kinds = [_pair_kind(flat[:, :, s]) for s in range(flat.shape[2])]
            names = {k["kind"] for k in kinds}
            proj = _pair_kind(nz.any(axis=other) if other else nz)["kind"]
            rec = {"axes": [i, j], "slices": flat.shape[2],
                   "slice_kinds": sorted(names),
                   "max_slice_hits": max(k["hits"] for k in kinds),
                   "projection": proj}
            if proj == "diagonal":
                rec["kind"] = "diagonal"
                rec["hits"] = max(k["hits"] for k in kinds)
            elif len(names) == 1:
                rec.update(kinds[0])
            else:
                rec["kind"] = "mixed"
                rec["hits"] = max(k["hits"] for k in kinds)
            pairs.append(rec)
    rep["pairs"] = pairs
    # ``block_period`` is DELETED. It searched upward for the SMALLEST block
    # size making the pattern constant, and 1 satisfies that for every array,
    # so the field was identically 1 (grill review 2026-09-07).
    return rep


def _pair_kind(o: np.ndarray) -> dict:
    n, m = o.shape
    hits = int(o.sum())
    # An EMPTY pattern and a PERMUTATION are both trivially contiguous runs.
    # Calling them bands inflated the band count by 32 of 45 (grill review).
    if hits == 0:
        return {"kind": "empty", "hits": 0}
    if hits == n * m:
        return {"kind": "full", "hits": hits}
    if n == m and bool((o == np.eye(n, dtype=bool)).all()):
        return {"kind": "diagonal", "hits": hits}
    # contiguous run per row?
    widths, contiguous = [], True
    for r in range(n):
        idx = np.flatnonzero(o[r])
        if idx.size == 0:
            widths.append(0)
            continue
        if idx.size != idx[-1] - idx[0] + 1:
            contiguous = False
        widths.append(int(idx.size))
    if contiguous and max(widths) == 1 and min(widths) == 1:
        return {"kind": "permutation", "hits": hits}
    if contiguous and max(widths) > 1:
        return {"kind": "row_band", "hits": hits,
                "width_max": max(widths), "width_min": min(widths),
                "uniform_width": len(set(widths)) == 1}
    widthsc, contig_c = [], True
    for c in range(m):
        idx = np.flatnonzero(o[:, c])
        if idx.size == 0:
            widthsc.append(0)
            continue
        if idx.size != idx[-1] - idx[0] + 1:
            contig_c = False
        widthsc.append(int(idx.size))
    if contig_c and max(widthsc) > 1:
        return {"kind": "col_band", "hits": hits,
                "width_max": max(widthsc), "width_min": min(widthsc),
                "uniform_width": len(set(widthsc)) == 1}
    return {"kind": "set", "hits": hits}




# --------------------------------------------------------------------------
# Walking a target
# --------------------------------------------------------------------------
def _force(t):
    from graphax.core import LazyEdge
    return t.value if isinstance(t, LazyEdge) else t


def _dense(t):
    from graphax.sparse.tensor import SparseTensor
    t = _force(t)
    if t is None:
        return None
    if isinstance(t, SparseTensor):
        return np.asarray(t.dense(), np.float64)
    return np.asarray(t, np.float64)


def _declared(t):
    from graphax.sparse.tensor import SparseTensor
    t = _force(t)
    if not isinstance(t, SparseTensor):
        return {"declared": "array"}
    return {"declared": [type(d).__name__ for d in t.dims],
            "axis": [d.axis for d in t.dims],
            "block_size": [d.block_size for d in t.dims],
            "block_axis": [d.block_axis for d in t.dims],
            "stored": 0 if t.val is None else int(t.val.size),
            "logical": [int(d.logical_size) for d in t.dims]}


def census(fn, args, argnums, order_name, order, limit_elems=4_000_000):
    """Elemental partials, then one record per accumulated Jacobian the
    elimination stores, in order."""
    from graphax.incremental import IncrementalJaxpr
    cj = jax.make_jaxpr(fn)(*args)
    jaxpr, consts = cj.jaxpr, cj.literals
    ij = IncrementalJaxpr(jaxpr, tuple(argnums), list(consts), list(args),
                          track_faces=False)
    recs = []

    def snap(stage, step, u, v, t):
        d = _dense(t)
        if d is None or d.size > limit_elems:
            return
        r = {"target": fn.__name__, "order": order_name, "stage": stage,
             "step": step, "edge": f"{u}->{v}",
             "primitive": PRIM.get(str(v), "")}
        r.update(_declared(t))
        r.update(occupancy_report(d))
        recs.append(r)

    PRIM = {}
    for i, e in enumerate(jaxpr.eqns):
        for ov in e.outvars:
            PRIM[str(ov)] = str(e.primitive.name)

    for u in list(ij.graph):
        for v, t in list(ij.graph[u].items()):
            snap("elemental", -1, u, v, t)

    for step, vtx in enumerate(order):
        try:
            ij.eliminate(vtx, (), None)
        except Exception as exc:
            recs.append({"target": fn.__name__, "order": order_name,
                         "stage": "error", "step": step, "edge": str(vtx),
                         "error": str(exc)[:200]})
            break
        for u in list(ij.graph):
            for v, t in list(ij.graph[u].items()):
                snap("accumulated", step, u, v, t)
    return recs


# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------
def mlp_toy():
    B, DIN, H, V = 4, 8, 8, 16
    ks = jax.random.split(jax.random.PRNGKey(0), 6)
    args = (jax.random.normal(ks[0], (B, DIN)),
            jax.random.normal(ks[1], (B, V)),
            jax.random.normal(ks[2], (DIN, H)) * 0.3,
            jax.random.normal(ks[3], (H,)) * 0.1,
            jax.random.normal(ks[4], (H, V)) * 0.3)

    def mlp_toy(x, y, w1, b1, wout):
        h = jnp.tanh(x @ w1 + b1)
        return jnp.mean((h @ wout - y) ** 2)
    return mlp_toy, args, (2, 3, 4)


def elementwise_chain():
    x = jax.random.normal(jax.random.PRNGKey(1), (12,))

    def elementwise_chain(x):
        return jnp.sum(jnp.tanh(x) * jnp.exp(x) + jnp.sin(x))
    return elementwise_chain, (x,), (0,)


def conv_like():
    ks = jax.random.split(jax.random.PRNGKey(2), 3)
    args = (jax.random.normal(ks[0], (6, 10)),
            jax.random.normal(ks[1], (10, 7)),
            jax.random.normal(ks[2], (7, 5)))

    def conv_like(x, w1, w2):
        return jnp.sum(jnp.tanh(jnp.tanh(x @ w1) @ w2))
    return conv_like, args, (1, 2)


TARGETS = {"mlp_toy": mlp_toy, "elementwise_chain": elementwise_chain,
           "conv_like": conv_like}


def orders(fn, args, argnums):
    from graphax.incremental import IncrementalJaxpr
    cj = jax.make_jaxpr(fn)(*args)
    jaxpr = cj.jaxpr
    outv = set(map(id, jaxpr.outvars))
    valid = [i + 1 for i, e in enumerate(jaxpr.eqns)
             if not any(id(o) in outv for o in e.outvars)]
    ij = IncrementalJaxpr(jaxpr, tuple(argnums), list(cj.literals), list(args),
                          track_faces=False)
    left, mk = set(valid), []
    while left:
        deg = {}
        for v in left:
            var = jaxpr.eqns[v - 1].outvars[0]
            preds = [u for u in ij.graph if var in ij.graph[u]]
            succs = list(ij.graph.get(var, {}).keys())
            deg[v] = len(preds) * len(succs)
        b = min(left, key=lambda v: (deg[v], v))
        mk.append(b)
        left.remove(b)
        ij.eliminate(b, (), None)
    return {"reverse": sorted(valid, reverse=True), "markowitz": mk}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "t287_census.jsonl"
    modes = {"tiled_nodemote": {"GRAPHAX_TILED_LAZY": "nodemote",
                                "GRAPHAX_EINSUM_GENERAL": "0"}}
    recs = []
    for name, build in TARGETS.items():
        fn, args, argnums = build()
        ords = orders(fn, args, argnums)
        for mode, env in modes.items():
            for k, v in env.items():
                os.environ[k] = v
            for oname, order in ords.items():
                got = census(fn, args, argnums, oname, order)
                for r in got:
                    r["mode"] = mode
                recs += got
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"{len(recs)} records -> {out}")
