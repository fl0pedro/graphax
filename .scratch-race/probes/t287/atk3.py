import os,sys,traceback; os.environ.setdefault("JAX_PLATFORMS","cpu")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp, numpy as np
from dense_elim import trace
x = jnp.array([1.,-2.,3.,0.5])
print(jax.make_jaxpr(lambda x: jnp.sum(jax.nn.relu(x)*x))(x))
try: trace(lambda x: jnp.sum(jax.nn.relu(x)*x), (x,))
except Exception: traceback.print_exc(limit=6)
print("="*60)
from t287_shape_census import TARGETS
from targets2 import TARGETS2
T={**TARGETS,**TARGETS2}
for n,b in T.items():
    f,a,an=b()
    j=jax.make_jaxpr(f)(*a).jaxpr
    prims=[str(e.primitive) for e in j.eqns]
    multi=[(str(e.primitive),len(e.outvars)) for e in j.eqns if len(e.outvars)!=1]
    comp=[p for p in prims if p in ("jit","pjit","closed_call","custom_jvp_call","custom_vjp_call","custom_jvp_call_jaxpr")]
    print(f"{n:16} neqn={len(j.eqns)} multi={multi} composite={comp}")
    print("      ", prims)
