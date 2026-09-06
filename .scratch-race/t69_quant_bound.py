"""Measure the oracle-B tolerance for the Quant class on the toy (ruling D10).

Every literal Quant bf16 plan on the 2-layer MLP: each slot on every face, each
slot on one face at a time, and the combinations that make BOTH contraction
operands narrow. The census must match on all of them (literal actions), so the
value comparison is meaningful everywhere. Reported: the largest relative L2
between the sparse engine and the dense mode, and the same against jax.grad.
"""
import itertools
import jax
import numpy as np
import t69_dense_toy as T
from graphax import ActionCensus, census_plan, compare_censuses, jacve
from graphax.sparse.micro_actions import Quant

ORDER = T.markowitz_order()
CAT = T.face_catalog(ORDER)
REF = [np.asarray(g, np.float64)
       for g in jax.grad(T.loss_fn, argnums=T.ARGNUMS)(*T.ARGS)]
Q = Quant("bfloat16")


def plan(slots, faces=None):
    ft = {}
    for i, (v, k, *_r) in enumerate(CAT):
        if faces is not None and i not in faces:
            continue
        ft.setdefault(v, {})[k] = tuple(Q if s in slots else None
                                        for s in range(3))
    return ft


def run(ft, dense):
    census = ActionCensus()
    kw = dict(dense_edges=True) if dense else dict(sparse_representation=True)
    fn = jacve(T.loss_fn, list(ORDER), argnums=T.ARGNUMS, transforms=[],
               face_transforms=census_plan(ft, census), **kw)
    return T.to_np(jax.jit(fn)(*T.ARGS)), census


def main():
    cases = []
    for r in range(1, 4):
        for slots in itertools.combinations(range(3), r):
            cases.append((f"slots {slots}, every face", plan(slots)))
    for i in range(len(CAT)):
        for s in range(3):
            cases.append((f"slot {s}, face {i}", plan((s,), faces={i})))
    worst = (0.0, None)
    worst_ref = (0.0, None)
    n = 0
    for name, ft in cases:
        sp, cs = run(ft, False)
        dn, cd = run(ft, True)
        compare_censuses(cs, cd, site=name)
        d = T.rel(dn, sp)
        n += 1
        if d > worst[0]:
            worst = (d, name)
        r = T.rel(dn, REF)
        if r > worst_ref[0]:
            worst_ref = (r, name)
        print(f"{name:28s} census {cs.counts()} dense-vs-sparse {d:.3e} "
              f"dense-vs-jax.grad {r:.3e}")
    print(f"\n{n} Quant plans, every census matched.")
    print(f"WORST dense-vs-sparse : {worst[0]:.3e}  ({worst[1]})")
    print(f"WORST dense-vs-jax.grad: {worst_ref[0]:.3e}  ({worst_ref[1]})")


if __name__ == "__main__":
    main()
