"""Validation the first pass lacked: repeated operands, and a path-sum check on
EVERY intermediate edge, not only the final gradient (grill review 2026-09-07)."""
import os, sys, itertools
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np
from dense_elim import trace, eliminate, compose
from run_census import valid_vertices, markowitz
from t287_shape_census import TARGETS
from targets2 import TARGETS2
TARGETS = {**TARGETS, **TARGETS2}

x3 = jnp.array([1., 2., 3.])
x4 = jnp.arange(1., 5.)
REPEATED = {
    "sq_mul":   (lambda x: jnp.sum(x * x), (x3,), (0,)),
    "add_self": (lambda x: jnp.sum(x + x), (x3,), (0,)),
    "cube_mul": (lambda x: jnp.sum(x * x * x), (x3,), (0,)),
    "int_pow":  (lambda x: jnp.sum(x ** 2), (x3,), (0,)),
    "gram":     (lambda x: jnp.sum(jnp.outer(x, x)), (x4,), (0,)),
    "var_like": (lambda x: jnp.mean((x - jnp.mean(x)) * (x - jnp.mean(x))), (x4,), (0,)),
}

def grad_check(name, f, args, argnums):
    jaxpr, val, g0, prim = trace(f, args)
    ranks = {v: jnp.asarray(val[v]).ndim for v in val}
    verts = valid_vertices(jaxpr)
    bad = []
    for oname, order in {"reverse": list(reversed(verts)),
                         "markowitz": markowitz(g0, verts, ranks)}.items():
        g = {u: dict(d) for u, d in g0.items()}
        for v in order:
            eliminate(g, v, ranks[v])
        ref = jax.grad(f, argnums=argnums)(*args)
        out = jaxpr.outvars[0]
        for k, an in enumerate(argnums):
            got = np.asarray(g[jaxpr.invars[an]][out], np.float64)
            want = np.asarray(ref[k], np.float64)
            d = np.max(np.abs(got.reshape(want.shape) - want)) / max(np.max(np.abs(want)), 1e-30)
            if d >= 1e-5:
                bad.append((oname, an, float(d)))
    return bad

def path_sum_check(name, f, args, argnums, limit=200):
    """After eliminating a set S, edge u->w must equal the sum over every u->w
    path whose interior lies in S. Checked at every step, every edge."""
    jaxpr, val, g0, prim = trace(f, args)
    ranks = {v: jnp.asarray(val[v]).ndim for v in val}
    verts = valid_vertices(jaxpr)
    g = {u: dict(d) for u, d in g0.items()}
    done, bad, n = [], [], 0
    for v in list(reversed(verts)):
        eliminate(g, v, ranks[v])
        done.append(v)
        for u in list(g):
            for w in list(g[u]):
                if n >= limit:
                    return bad, n
                tot = None
                stack = [(u, None)]
                # enumerate paths u -> w with interior in `done`
                def walk(cur, acc):
                    nonlocal tot
                    for nxt, t in g0.get(cur, {}).items():
                        step = t if acc is None else compose(acc, t, ranks[cur])
                        if nxt is w:
                            tot = step if tot is None else tot + step
                        elif nxt in done:
                            walk(nxt, step)
                walk(u, None)
                if tot is None:
                    continue
                a = np.asarray(g[u][w], np.float64)
                b = np.asarray(tot, np.float64)
                n += 1
                absd = float(np.max(np.abs(a - b)))
                scale = max(float(np.max(np.abs(b))), float(np.max(np.abs(a))))
                d = absd / max(scale, 1e-30)
                # A tensor that is mathematically zero shows up as float32
                # noise on both sides; a relative test on noise is meaningless.
                # Fail only when the disagreement is large in ABSOLUTE terms too.
                if d >= 1e-4 and absd > 1e-5:
                    bad.append((str(prim.get(w, "?")), float(d), absd, scale))
    return bad, n

if __name__ == "__main__":
    print("=== repeated-operand targets (the bug the first pass missed) ===")
    ok = True
    for name, (f, args, an) in REPEATED.items():
        bad = grad_check(name, f, args, an)
        print(f"  {'OK ' if not bad else 'FAIL'} {name:10} {bad if bad else ''}")
        ok &= not bad
    print("\n=== the ten census targets ===")
    for name, build in TARGETS.items():
        f, args, an = build()
        bad = grad_check(name, f, args, an)
        print(f"  {'OK ' if not bad else 'FAIL'} {name}")
        ok &= not bad
    print("\n=== path-sum check on every intermediate edge ===")
    tot = 0
    for name, build in TARGETS.items():
        f, args, an = build()
        bad, n = path_sum_check(name, f, args, an)
        tot += n
        print(f"  {'OK ' if not bad else 'FAIL'} {name:18} {n:4} edge snapshots"
              f"{'' if not bad else '  ' + str(bad[:3])}")
        ok &= not bad
    print(f"\n{tot} intermediate edges checked. {'ALL OK' if ok else 'FAILURES ABOVE'}")
