"""Ticket dsnn-3qm.69, deliverable (b): the toy that shows the dense mode is needed.

The 2-layer MLP of tests/core/sparse_tensor/output_layout_test.py, eliminated on
the static minimum Markowitz degree order, with three approximated plans:
Quant bf16 on one face, Reduce mean on one face, Diag on one face where legal.

Three results per plan:

  * sparse   -- jacve(sparse_representation=True), the real engine.
  * packing  -- jacve(sparse_representation=False). That flag changes the RETURN
                form only (core.py:453, :3199), so it must be bit-identical to
                `sparse` (finding 61 verdict 4). It is NOT an oracle.
  * dense    -- the prototype of the mode this ticket asks for: every edge a
                plain jnp array, every contraction a jnp.tensordot over the
                eliminated vertex's axes. Independent of the SparseTensor
                contraction engine, so a disagreement with `sparse` is evidence.

For the exact plan all three equal jax.grad. For an approximated plan the first
two stay bit-identical (they are the same contractions packed two ways) while
`dense` differs by the approximation.

This file is the RECORD of deliverable (b): the standalone prototype that showed
the mode was needed BEFORE the mode existed. The landed implementation is
``graphax.dense_edges`` (``jacve(..., dense_edges=True)``); this prototype is
kept unchanged as the evidence, not as a second implementation to maintain.

Run: JAX_PLATFORMS=cpu PYTHONPATH=$LANE/graphax/src python t69_dense_toy.py
"""
from __future__ import annotations

import math
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from graphax import jacve
from graphax.core import (
    _build_graph, _checkify_order, _drain_transforms, _force, _prune_graph,
    _vidx_for, prune_enabled,
)
from graphax.sparse.micro_actions import (
    Compress, Diag, Quant, apply_compress, apply_diag, apply_quant,
)
from graphax.sparse.tensor import DenseIndex, SparseTensor

# --------------------------------------------------------------------------
# the target
# --------------------------------------------------------------------------
B, DIN, H, V = 4, 8, 8, 16
KEY = jax.random.PRNGKey(0)
KS = jax.random.split(KEY, 6)
X = jax.random.normal(KS[0], (B, DIN))
Y = jax.random.normal(KS[1], (B, V))
W1 = jax.random.normal(KS[2], (DIN, H)) * 0.3
B1 = jax.random.normal(KS[3], (H,)) * 0.1
WOUT = jax.random.normal(KS[4], (H, V)) * 0.3
ARGS = (X, Y, W1, B1, WOUT)
ARGNUMS = (2, 3, 4)


def loss_fn(x, y, w1, b1, wout):
    h = jnp.tanh(x @ w1 + b1)
    return jnp.mean((h @ wout - y) ** 2)


# --------------------------------------------------------------------------
# the dense-edge adapter: a plain array <-> a fully dense SparseTensor
# --------------------------------------------------------------------------
def wrap(arr, out_ndim: int) -> SparseTensor:
    """A plain array as a fully dense SparseTensor: one DenseIndex per axis,
    axis == position, the out/primal split at ``out_ndim``."""
    dims = tuple(DenseIndex(i, int(s), i) for i, s in enumerate(arr.shape))
    return SparseTensor(dims[:out_ndim], dims[out_ndim:], arr,
                        check_consistency=False)


def unwrap(st: SparseTensor):
    """Back to a plain array. ``keep_quantization=True`` so a Quant survives
    the round trip as a dtype, not merely as rounded values in f32."""
    arr = st.dense(keep_quantization=True)
    return arr, len(st.out_dims)


def apply_hook(arr, out_ndim, hook):
    """One face-slot hook on a dense edge, through the adapter."""
    if hook is None:
        return arr
    st = wrap(arr, out_ndim)
    if isinstance(hook, Diag):
        st = apply_diag(st, hook)
    elif isinstance(hook, Compress):
        st = apply_compress(st, hook)
    elif isinstance(hook, Quant):
        st = apply_quant(st, hook)
    elif callable(hook):
        st = hook(st)
    else:
        raise TypeError(f"unknown hook {hook!r}")
    out, _ = unwrap(st)
    assert out.shape == arr.shape, (out.shape, arr.shape, hook)
    return out


# --------------------------------------------------------------------------
# the dense elimination
# --------------------------------------------------------------------------
def dense_jacobian(fun, order, argnums, args, face_transforms=None,
                   trace=None):
    """Vertex elimination with every edge a plain jnp array.

    An edge (u -> v) is an array of shape ``v.aval.shape + u.aval.shape``. The
    contraction of ``post`` (central -> out_edge) with ``pre`` (in_edge ->
    central) is a tensordot over the central variable's axes. The join is ``+``.
    Every approximation preserves the shape, so the invariant holds throughout.
    """
    cj = jax.make_jaxpr(fun)(*args)
    jaxpr, consts = cj.jaxpr, cj.literals
    _, graph, tgraph, vo = _build_graph(jaxpr, list(args), list(consts),
                                        tuple(argnums))
    if prune_enabled():
        _prune_graph(graph, tgraph, jaxpr, tuple(argnums))
    order = _checkify_order(order, jaxpr, vo)
    vidx = _vidx_for(jaxpr)
    ft = face_transforms or {}

    D: dict = defaultdict(dict)
    DT: dict = defaultdict(dict)
    for u in list(graph.keys()):
        for v in list(graph[u].keys()):
            t = _force(graph[u][v])
            if t is None:
                continue
            arr = _drain_transforms(t.copy(), post_first=False).dense()
            nominal = tuple(v.aval.shape) + tuple(u.aval.shape)
            assert tuple(arr.shape) == nominal, (arr.shape, nominal)
            D[u][v] = arr
            DT[v][u] = arr

    def _ordered(keys):
        return sorted(keys, key=lambda k: vidx.get(k, 1 << 30))

    for vert in order:
        eqn = jaxpr.eqns[int(vert) - 1]
        slots_of_vertex = ft.get(int(vert)) or {}
        for cv in eqn.outvars:
            if cv not in D:
                continue
            k = int(np.prod(cv.aval.shape, dtype=np.int64)) if cv.aval.shape else 1
            del k
            for oe in _ordered(D[cv].keys()):
                post_raw = D[cv][oe]
                for ie in _ordered(DT.get(cv, {}).keys()):
                    pre = DT[cv][ie]
                    post = post_raw
                    slots = slots_of_vertex.get((vidx.get(ie), vidx.get(oe)))
                    lhs_t = rhs_t = new_t = res_t = None
                    if slots is not None:
                        lhs_t, rhs_t, res_t = slots
                    on = cv.aval.ndim
                    pre = apply_hook(pre, on, lhs_t)
                    post = apply_hook(post, oe.aval.ndim, rhs_t)
                    contract = jnp.tensordot(post, pre, axes=on)
                    contract = apply_hook(contract, oe.aval.ndim, new_t)
                    if trace is not None:
                        trace.append((int(vert), vidx.get(ie), vidx.get(oe),
                                      tuple(contract.shape),
                                      str(contract.dtype)))
                    old = D[ie].get(oe)
                    if old is not None:
                        assert old.shape == contract.shape, (old.shape,
                                                             contract.shape)
                        contract = contract + old
                    contract = apply_hook(contract, oe.aval.ndim, res_t)
                    D[ie][oe] = contract
                    DT[oe][ie] = contract
            if cv not in vo:
                for iv in list(DT.get(cv, {}).keys()):
                    D[iv].pop(cv, None)
            for ov in list(D[cv].keys()):
                DT[ov].pop(cv, None)
            D.pop(cv, None)
            if cv not in vo:
                DT.pop(cv, None)

    out = []
    for ov in jaxpr.outvars:
        for i in argnums:
            iv = jaxpr.invars[i]
            arr = D.get(iv, {}).get(ov)
            if arr is None:
                arr = jnp.zeros(tuple(ov.aval.shape) + tuple(iv.aval.shape),
                                dtype=jnp.float32)
            out.append(arr)
    return out


# --------------------------------------------------------------------------
# plans
# --------------------------------------------------------------------------
def markowitz_order():
    from graphax import faces_of  # noqa: F401
    from graphax.incremental import IncrementalJaxpr
    cj = jax.make_jaxpr(loss_fn)(*ARGS)
    jaxpr, consts = cj.jaxpr, cj.literals
    outvars = set(map(id, jaxpr.outvars))
    valid = [i + 1 for i, e in enumerate(jaxpr.eqns)
             if not any(id(o) in outvars for o in e.outvars)]
    ij = IncrementalJaxpr(jaxpr, ARGNUMS, list(consts), list(ARGS),
                          track_faces=False)
    left, order = set(valid), []
    while left:
        deg = {}
        for v in left:
            var = jaxpr.eqns[v - 1].outvars[0]
            preds = [u for u in ij.graph if var in ij.graph[u]]
            succs = list(ij.graph.get(var, {}).keys())
            deg[v] = len(preds) * len(succs)
        best = min(left, key=lambda v: (deg[v], v))
        order.append(best)
        left.remove(best)
        ij.eliminate(best, (), None)
    return order


def face_catalog(order):
    """Every (vertex, face key, pre shape, post shape) the elimination visits,
    collected by running the dense elimination with a trace."""
    cj = jax.make_jaxpr(loss_fn)(*ARGS)
    jaxpr, consts = cj.jaxpr, cj.literals
    _, graph, tgraph, vo = _build_graph(jaxpr, list(ARGS), list(consts),
                                        tuple(ARGNUMS))
    if prune_enabled():
        _prune_graph(graph, tgraph, jaxpr, tuple(ARGNUMS))
    o = _checkify_order(order, jaxpr, vo)
    vidx = _vidx_for(jaxpr)
    shapes = {}
    for u in list(graph.keys()):
        for v in list(graph[u].keys()):
            t = _force(graph[u][v])
            if t is not None:
                shapes[(u, v)] = (tuple(v.aval.shape) + tuple(u.aval.shape),
                                  v.aval.ndim)
    cat = []
    G = {u: dict(graph[u]) for u in graph}
    T = {v: dict(tgraph[v]) for v in tgraph}
    for vert in o:
        eqn = jaxpr.eqns[int(vert) - 1]
        for cv in eqn.outvars:
            if cv not in G:
                continue
            for oe in sorted(G[cv], key=lambda k: vidx.get(k, 1 << 30)):
                for ie in sorted(T.get(cv, {}), key=lambda k: vidx.get(k, 1 << 30)):
                    cat.append((int(vert), (vidx.get(ie), vidx.get(oe)),
                                shapes.get((ie, cv)), shapes.get((cv, oe)),
                                tuple(oe.aval.shape) + tuple(ie.aval.shape),
                                oe.aval.ndim))
                    G.setdefault(ie, {})[oe] = True
                    T.setdefault(oe, {})[ie] = True
            if cv not in vo:
                for iv in list(T.get(cv, {})):
                    G.get(iv, {}).pop(cv, None)
            for ov in list(G[cv]):
                T.get(ov, {}).pop(cv, None)
            G.pop(cv, None)
            if cv not in vo:
                T.pop(cv, None)
    return cat


def run_sparse(order, ft, sparse):
    fn = jacve(loss_fn, list(order), argnums=ARGNUMS,
               sparse_representation=sparse, transforms=[], face_transforms=ft)
    return jax.jit(fn)(*ARGS)


def to_np(out):
    return [np.asarray(t.dense() if isinstance(t, SparseTensor) else t,
                       np.float64) for t in out]


def rel(a, b):
    num = math.sqrt(sum(float(np.sum((x - y) ** 2)) for x, y in zip(a, b)))
    den = math.sqrt(sum(float(np.sum(y ** 2)) for y in b))
    return num / max(den, 1e-30)


def bit_identical(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def main():
    order = markowitz_order()
    print(f"markowitz order: {order}")
    cat = face_catalog(order)
    print(f"{len(cat)} faces")

    ref = [np.asarray(g, np.float64)
           for g in jax.grad(loss_fn, argnums=ARGNUMS)(*ARGS)]

    plans = {"exact": None}

    # QUANT: bf16 on the lhs slot of the first face.
    v0, k0 = cat[0][0], cat[0][1]
    plans["quant_lhs"] = {v0: {k0: (Quant("bfloat16"), None, None)}}

    # REDUCE: mean over slot 0 of the lhs edge of the first face whose lhs
    # edge has at least one axis.
    for vert, key, pre, post, _oshape, _ondim in cat:
        if pre is not None and len(pre[0]) >= 1:
            plans["reduce_lhs"] = {vert: {key: (Compress((0,), "mean"),
                                                None, None)}}
            break

    # DIAG: on the contraction result (slot new/res), the first face whose
    # result has an out axis and a primal axis with gcd > 1.
    for vert, key, pre, post, oshape, ondim in cat:
        if ondim >= 1 and len(oshape) > ondim:
            f = math.gcd(int(oshape[0]), int(oshape[ondim]))
            if f > 1:
                plans["diag_res"] = {vert: {key: (None, None,
                                                  Diag(0, ondim, f))}}
                break

    # ALL THREE AT ONCE, on three different faces.
    combined: dict = {}
    for src in ("quant_lhs", "reduce_lhs", "diag_res"):
        for vert, faces in plans[src].items():
            for key, slots in faces.items():
                cur = combined.setdefault(vert, {}).get(key, (None, None, None))
                combined[vert][key] = tuple(
                    b if b is not None else a for a, b in zip(cur, slots))
    plans["combined"] = combined

    # QUANT bf16 on the lhs slot of EVERY face.
    every: dict = {}
    for vert, key, _pre, _post, _oshape, _ondim in cat:
        every.setdefault(vert, {})[key] = (Quant("bfloat16"), None, None)
    plans["quant_every_face"] = every

    for name, ft in plans.items():
        print(f"\n--- plan {name}: {ft}")
        sp = to_np(run_sparse(order, ft, True))
        pk = to_np(run_sparse(order, ft, False))
        dn = to_np(dense_jacobian(loss_fn, order, ARGNUMS, ARGS, ft))
        print(f"  sparse vs packing (sparse_representation=False): "
              f"bit-identical={bit_identical(sp, pk)} rel_l2={rel(sp, pk):.3e}")
        print(f"  sparse  vs jax.grad: rel_l2={rel(sp, ref):.3e}")
        print(f"  dense   vs jax.grad: rel_l2={rel(dn, ref):.3e}")
        print(f"  dense   vs sparse  : rel_l2={rel(dn, sp):.3e}")


if __name__ == "__main__":
    main()
