import os,sys,json,collections; os.environ.setdefault("JAX_PLATFORMS","cpu")
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t287_shape_census as C

# 1) does the reshape fallback in occupancy_report ever fire?
FIRE=[0]
orig=C._pair_kind
import numpy as _np
src_fired=[]
# monkeypatch by re-implementing occupancy_report's pair loop with a check
def probe(a):
    a=np.asarray(a); nz=np.abs(a)>C.TOL
    for i in range(a.ndim):
        for j in range(i+1,a.ndim):
            other=tuple(k for k in range(a.ndim) if k not in (i,j))
            o=nz.any(axis=other) if other else nz
            if o.shape!=(a.shape[i],a.shape[j]): FIRE[0]+=1

# 2) permutation vs diagonal: what does _pair_kind say?
P=np.zeros((6,6),bool); perm=[3,4,5,0,1,2]
for r,c in enumerate(perm): P[r,c]=True
print("permutation matrix ->", C._pair_kind(P))
print("identity          ->", C._pair_kind(np.eye(6,dtype=bool)))
print("anti-diagonal     ->", C._pair_kind(np.eye(6,dtype=bool)[::-1]))
D=np.zeros((6,6),bool)
for r in range(6):
    for c in range(6):
        if abs(r-c)<=1: D[r,c]=True
print("true tridiag band ->", C._pair_kind(D))
# 3) 'full' hiding structure via any-projection
a=np.zeros((4,4,4,4))
for i in range(4):
    for j in range(4): a[i,j,i,j]=1.0   # double diagonal, density 1/16
r=C.occupancy_report(a)
print("\ndouble-diagonal 4^4 density",r["density"],"pairs:",[(p["axes"],p["kind"],p.get("width_max")) for p in r["pairs"]])
# 4) replicated_axes false positive from allclose rtol=1e-5
b=np.zeros((3,4)); b[0]=1e6; b[1]=1e6+1.0; b[2]=1e6-1.0
print("\nlarge-offset slices differ by 1.0:", C.occupancy_report(b)["replicated_axes"], "(should be [])")
c=np.array([[1.0,2.0],[1.0,2.0],[1.0,2.000005]])
print("differ by 2.5e-6 rel:", C.occupancy_report(c)["replicated_axes"])
# 5) all-zero tensor
print("all-zero 3x3:", {k:v for k,v in C.occupancy_report(np.zeros((3,3))).items() if k in ("density","replicated_axes","pairs","block_period")})
