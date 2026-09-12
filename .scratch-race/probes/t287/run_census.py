#!/usr/bin/env python
"""t287: the empirical shape census. Runs the dense oracle and classifies the
non-zero pattern of every elemental partial and every accumulated Jacobian."""
import os, sys, json, collections
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax, jax.numpy as jnp, numpy as np
from dense_elim import trace, eliminate
from t287_shape_census import occupancy_report, TARGETS
from targets2 import TARGETS2
TARGETS = {**TARGETS, **TARGETS2}


def valid_vertices(jaxpr):
    outv = set(map(id, jaxpr.outvars))
    return [e.outvars[0] for e in jaxpr.eqns
            if len(e.outvars) == 1 and id(e.outvars[0]) not in outv]


def markowitz(graph0, verts, ranks):
    """Ties break on the vertex's POSITION in the equation list. ``str(var)``
    prints a heap address, so a str tie-break made the census irreproducible
    (grill review 2026-09-07)."""
    pos = {id(v): i for i, v in enumerate(verts)}
    g = {u: dict(d) for u, d in graph0.items()}
    left, order = list(verts), []
    while left:
        deg = {}
        for v in left:
            preds = [u for u in g if v in g[u]]
            succs = list(g.get(v, {}).keys())
            deg[id(v)] = (len(preds) * len(succs), pos[id(v)])
        b = min(left, key=lambda v: deg[id(v)])
        order.append(b)
        left.remove(b)
        eliminate(g, b, ranks[b])
    return order


def stable_names(jaxpr):
    """A name per var that does not depend on the heap, so records join
    across runs."""
    nm = {}
    for i, v in enumerate(jaxpr.invars):
        nm[id(v)] = f"in{i}"
    for i, v in enumerate(jaxpr.constvars):
        nm[id(v)] = f"const{i}"
    for i, e in enumerate(jaxpr.eqns):
        for j, v in enumerate(e.outvars):
            nm[id(v)] = f"e{i}" + (f".{j}" if len(e.outvars) > 1 else "")
    return nm


def run(name, build, out):
    f, args, argnums = build()
    jaxpr, val, graph0, prim = trace(f, args)
    NM = stable_names(jaxpr)
    def nm(v):
        return NM.get(id(v), "?")
    ranks = {v: jnp.asarray(val[v]).ndim for v in val}
    verts = valid_vertices(jaxpr)
    orders = {"reverse": list(reversed(verts)),
              "markowitz": markowitz(graph0, verts, ranks)}
    recs = []
    for oname, order in orders.items():
        g = {u: dict(d) for u, d in graph0.items()}
        if oname == "reverse":                    # elemental partials, once
            for u, d in graph0.items():
                for v, t in d.items():
                    r = {"target": name, "order": "-", "stage": "elemental",
                         "step": -1, "edge": f"{nm(u)}->{nm(v)}",
                         "primitive": prim.get(v, "?"), "kind": "-",
                         "shape_u": list(jnp.shape(val[u])),
                         "shape_v": list(jnp.shape(val[v]))}
                    r["into_scalar_output"] = (list(jnp.shape(val[v])) == [])
                    r.update(occupancy_report(np.asarray(t, np.float64)))
                    recs.append(r)

        def hook(u, w, t, kind, step, _o=oname, _n=name):
            r = {"target": _n, "order": _o, "stage": "accumulated",
                 "step": step, "edge": f"{nm(u)}->{nm(w)}", "primitive": prim.get(w, "?"),
                 "kind": kind, "shape_u": list(jnp.shape(val[u])),
                 "shape_v": list(jnp.shape(val[w])),
                 "into_scalar_output": list(jnp.shape(val[w])) == []}
            r.update(occupancy_report(np.asarray(t, np.float64)))
            recs.append(r)

        for step, v in enumerate(order):
            eliminate(g, v, ranks[v], on_edge=hook, step=step)
    with open(out, "a") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return len(recs)


if __name__ == "__main__":
    out = sys.argv[1]
    open(out, "w").close()
    for name, build in TARGETS.items():
        n = run(name, build, out)
        print(f"{name}: {n} records")
