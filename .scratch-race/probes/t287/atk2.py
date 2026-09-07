import os,sys; os.environ.setdefault("JAX_PLATFORMS","cpu")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np
from dense_elim import trace, eliminate
from run_census import valid_vertices

def check(name, f, args, argnums):
    try:
        jaxpr,val,graph0,prim = trace(f,args)
    except Exception as e:
        print(f"{name}: TRACE EXC {type(e).__name__}: {str(e)[:160]}"); return
    print(f"{name} eqns:", [(str(e.primitive), len(e.outvars)) for e in jaxpr.eqns])
    ranks = {v: jnp.asarray(val[v]).ndim for v in val}
    verts = valid_vertices(jaxpr)
    g = {u:dict(d) for u,d in graph0.items()}
    try:
        for v in reversed(verts): eliminate(g,v,ranks[v])
        out = jaxpr.outvars[0]
        for k,an in enumerate(argnums):
            iv = jaxpr.invars[an]
            got = np.asarray(g[iv][out],np.float64)
            want = np.asarray(jax.grad(f,argnums=argnums)(*args)[k],np.float64)
            r = np.max(np.abs(got.reshape(want.shape)-want))/max(np.max(np.abs(want)),1e-30)
            print(f"  {name} arg{an}: rel={r:.3e}  {'OK' if r<1e-5 else '*** WRONG ***'}")
    except Exception as e:
        print(f"  {name}: ELIM EXC {type(e).__name__}: {str(e)[:200]}")

x = jnp.array([1.,-2.,3.,0.5])
# multi-output primitive: split
check("split", lambda x: jnp.sum(jnp.tanh(jnp.split(x,2)[0])*jnp.split(x,2)[1]), (x,), (0,))
# relu -> comparison producing bool
check("relu", lambda x: jnp.sum(jax.nn.relu(x)*x), (x,), (0,))
check("where", lambda x: jnp.sum(jnp.where(x>0, x*x*1.0, jnp.exp(x))), (x,), (0,))
# dead code branch (vertex with no successors)
def dead(x):
    d = jnp.sin(x)*3.0   # computed but unused? jax DCEs this in make_jaxpr
    return jnp.sum(jnp.tanh(x))
check("dead", dead, (x,), (0,))
# top-k / sort: multiple results
check("sort", lambda x: jnp.sum(jnp.sort(x)*x), (x,), (0,))
check("cumsum", lambda x: jnp.sum(jnp.cumsum(x)*x), (x,), (0,))
# scan / while
check("scan", lambda x: jax.lax.scan(lambda c,a:(c*jnp.tanh(a),c),1.0,x)[0], (x,), (0,))
