"""The dense oracle must reproduce jax.grad. Without this the census is noise."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np
from dense_elim import trace, eliminate
from t287_shape_census import TARGETS
from targets2 import TARGETS2
TARGETS = {**TARGETS, **TARGETS2}
from run_census import valid_vertices, markowitz

ok = True
for name, build in TARGETS.items():
    f, args, argnums = build()
    jaxpr, val, graph0, prim = trace(f, args)
    ranks = {v: jnp.asarray(val[v]).ndim for v in val}
    verts = valid_vertices(jaxpr)
    ref = jax.grad(f, argnums=argnums)(*args)
    for oname, order in {"reverse": list(reversed(verts)),
                         "markowitz": markowitz(graph0, verts, ranks)}.items():
        g = {u: dict(d) for u, d in graph0.items()}
        for v in order:
            eliminate(g, v, ranks[v])
        out = jaxpr.outvars[0]
        for k, an in enumerate(argnums):
            iv = jaxpr.invars[an]
            got = np.asarray(g[iv][out], np.float64)
            want = np.asarray(ref[k], np.float64)
            got = got.reshape(want.shape)
            d = float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-30))
            flag = "OK " if d < 1e-5 else "FAIL"
            if d >= 1e-5:
                ok = False
            print(f"{flag} {name:18} {oname:10} arg{an} rel {d:.3e}")
print("ALL OK" if ok else "SOME FAILED")
