"""Independent oracle: after eliminating set S, edge u->w must equal the sum
over ALL paths u->...->w whose interior vertices lie in S. Checks EVERY edge at
EVERY step, which validate.py never does."""
import os,sys,itertools; os.environ.setdefault("JAX_PLATFORMS","cpu")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np
from dense_elim import trace, eliminate, compose
from run_census import valid_vertices, markowitz
from t287_shape_census import TARGETS
from targets2 import TARGETS2
T={**TARGETS,**TARGETS2}

def pathsum(g0, ranks, u, w, S, cap=200000):
    """sum over paths u->w with interior in S"""
    acc = None
    def rec(cur, tens):
        nonlocal acc
        for nxt, e in g0.get(cur,{}).items():
            t = e if tens is None else compose(tens, e, ranks[cur])
            if nxt is w:
                acc = t if acc is None else acc + t
            if nxt in S:
                rec(nxt, t)
    rec(u, None)
    return acc

bad=0; tot=0
for name,build in T.items():
    f,args,an=build()
    jaxpr,val,g0,prim=trace(f,args)
    ranks={v:jnp.asarray(val[v]).ndim for v in val}
    verts=valid_vertices(jaxpr)
    for oname,order in {"reverse":list(reversed(verts)),
                        "markowitz":markowitz(g0,verts,ranks)}.items():
        g={u:dict(d) for u,d in g0.items()}
        S=set()
        for step,v in enumerate(order):
            eliminate(g,v,ranks[v]); S.add(v)
            for u in list(g):
                for w,t in list(g[u].items()):
                    ref=pathsum(g0,ranks,u,w,S)
                    tot+=1
                    got=np.asarray(t,np.float64)
                    if ref is None:
                        print("  MISSING REF",name,oname,step); bad+=1; continue
                    ref=np.asarray(ref,np.float64)
                    if got.shape!=ref.shape:
                        print("  SHAPE",name,oname,step,got.shape,ref.shape); bad+=1; continue
                    d=np.max(np.abs(got-ref))/max(np.max(np.abs(ref)),1e-30)
                    if d>1e-4:
                        print(f"  VALUE {name} {oname} step{step} {prim.get(w,'?')} rel={d:.2e}"); bad+=1
print(f"checked {tot} edge snapshots, {bad} mismatches")
