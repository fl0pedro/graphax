import os,sys,json; os.environ.setdefault("JAX_PLATFORMS","cpu")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
X64 = os.environ.get("X64")=="1"
if X64:
    import jax; jax.config.update("jax_enable_x64",True)
import jax, jax.numpy as jnp, numpy as np
import t287_shape_census as C
if os.environ.get("RELTOL")=="1":
    _o=C.occupancy_report
    def occ(a):
        a=np.asarray(a); s=np.max(np.abs(a)) if a.size else 0.0
        C.TOL = max(s*1e-7, 1e-30)
        return _o(a)
    C.occupancy_report=occ
from dense_elim import trace, eliminate
from run_census import valid_vertices, markowitz
from t287_shape_census import TARGETS
from targets2 import TARGETS2
T={**TARGETS,**TARGETS2}
out=[]
for name,build in T.items():
    f,args,an=build()
    if X64: args=tuple(jnp.asarray(a,jnp.float64) for a in args)
    jaxpr,val,g0,prim=trace(f,args)
    ranks={v:jnp.asarray(val[v]).ndim for v in val}
    verts=valid_vertices(jaxpr)
    for u,d in g0.items():
        for v,t in d.items():
            r=C.occupancy_report(np.asarray(t,np.float64))
            out.append({"t":name,"stage":"elemental","prim":prim.get(v,"?"),
                        "shape":r["shape"],"nnz":r["nonzeros"],
                        "rep":r["replicated_axes"],
                        "kinds":[p["kind"] for p in r["pairs"]]})
    g={u:dict(d) for u,d in g0.items()}
    def hook(u,w,t,kind,step,_n=name):
        r=C.occupancy_report(np.asarray(t,np.float64))
        out.append({"t":_n,"stage":"acc","prim":prim.get(w,"?"),"step":step,
                    "shape":r["shape"],"nnz":r["nonzeros"],
                    "rep":r["replicated_axes"],"kinds":[p["kind"] for p in r["pairs"]]})
    for s,v in enumerate(reversed(verts)): eliminate(g,v,ranks[v],on_edge=hook,step=s)
json.dump(out,open(sys.argv[1],"w"))
print(len(out),"->",sys.argv[1])
