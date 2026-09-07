import os,sys; os.environ.setdefault("JAX_PLATFORMS","cpu")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np
from dense_elim import trace, eliminate, compose
from run_census import valid_vertices

def check(name, f, args, argnums):
    jaxpr,val,graph0,prim = trace(f,args)
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
            print(f"{name} arg{an}: got={got.ravel()[:4]} want={want.ravel()[:4]} ratio={(got/want).ravel()[:4]}")
    except Exception as e:
        print(f"{name}: EXC {type(e).__name__}: {e}")
    print("   jaxpr eqns:", [(str(e.primitive), len(e.outvars)) for e in jaxpr.eqns])

x = jnp.array([1.,2.,3.])
check("x*x",         lambda x: jnp.sum(x*x), (x,), (0,))
check("x*x*x",       lambda x: jnp.sum(x*x*x), (x,), (0,))
check("dot(x,x)",    lambda x: x@x, (x,), (0,))
check("x+x",         lambda x: jnp.sum(x+x), (x,), (0,))
check("x*y sanity",  lambda x,y: jnp.sum(x*y), (x,x+1), (0,1))
