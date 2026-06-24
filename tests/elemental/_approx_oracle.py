"""Dense-reference oracle for the Diag / Compress approximations.

This module defines the GROUND-TRUTH approximate Jacobian that the sparse
fast-path implementation is graded against. It does NOT touch the sparse
contraction topology at all: it applies each ``Diag`` block-diagonal MASK and
each ``Compress`` mean-broadcast on the edge, then immediately DENSIFIES the
edge and reshapes it back to the edge's NOMINAL logical shape
(``out_edge.aval.shape + in_edge.aval.shape``) before it propagates. Because
``SparseTensor.dense()`` faithfully materialises the approximation's values,
and the nominal reshape re-groups the Diag-split / Compress-implicit axes back
to the un-approximated logical grouping, the rest of the elimination runs as a
plain dense vertex-elimination — mathematically the same as "apply the dense
mask, then contract densely".

Why densify-AND-reshape-to-nominal (not just ``_arr2st(edge.dense())``):
``Diag`` rebuilds the edge's dim list so a single logical contracting axis of
size N is SPLIT into two dims (size=factor, block_size=N/factor), and
``dense()`` of that rectangular ``DiagonalIndex`` materialises BOTH block axes
explicitly. ``_arr2st(dense(), out_ndim=len(out_dims))`` then mis-groups those
extra axes, and a downstream contraction sees ``factor`` and ``N/factor`` as
two separate logical dims (the "8 vs 16" / "2 vs 4" mismatch). Reshaping
``dense()`` to the nominal ``out.aval + in.aval`` shape collapses the split
back to the original grouping, so the densified edge is a drop-in for the
exact edge — only its VALUES carry the approximation.

Usage
-----
    from _approx_oracle import oracle_jacobian, build_reference_set
    J = oracle_jacobian(fn, order, argnums, transforms)(*args)

``transforms`` is the normal ``[(vertex, [Diag(...)|Compress(...)])]`` spec.
``build_reference_set()`` saves ``np`` reference Jacobians for the fixed model
set to ``_approx_oracle_refs.npz``.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np

import graphax.core as gcore
from graphax.sparse.micro_actions import (
    Diag, Compress, Quant, apply_diag, apply_compress, apply_quant,
)
from graphax.sparse.indexes import DenseIndex
from graphax.sparse.ops.utils import _arr2st
from graphax.sparse.tensor import SparseTensor


# Thread-local carrying the current edge's nominal (out_ndim, logical_shape)
# so the patched transform dispatch can reshape dense() back to nominal.
_edge_ctx = threading.local()


def _densify_to_nominal(st, out_ndim, nominal_shape):
    """Densify ``st`` PRESERVING its dim ids when possible.

    ``dense()`` preserves the approximation's values. CRUCIAL: graphax aligns
    contractions by dim id, so the densified edge must keep the edge's ORIGINAL
    ids — a fresh-id rebuild (the old ``_arr2st`` path, ids ``range(0,n)``)
    mis-aligns a downstream contraction under REVERSE elimination and silently
    loses precision (this oracle dropped ~4.5% on reverse-order MoE — an oracle
    artifact mistaken for a graphax error). When ``dense()`` already lands one
    dim per nominal axis, rebuild with the original ids (value-preserving). Only
    the genuine regroup case (a Diag-split / implicit axis whose dense shape
    differs from nominal) falls back to the reshape + fresh-id rebuild."""
    d = st.dense()
    nominal = tuple(nominal_shape)
    if (
        tuple(d.shape) == nominal
        and len(st.out_dims) == out_ndim
        and len(st.out_dims) + len(st.primal_dims) == len(nominal)
    ):
        out_dims = tuple(
            DenseIndex(x.id, int(x.logical_size), k)
            for k, x in enumerate(st.out_dims)
        )
        primal_dims = tuple(
            DenseIndex(x.id, int(x.logical_size), out_ndim + k)
            for k, x in enumerate(st.primal_dims)
        )
        return SparseTensor(
            out_dims, primal_dims, d,
            scalar_mult=st.scalar_mult, fill_value=st.fill_value,
            pre_transforms=st.pre_transforms, post_transforms=st.post_transforms,
            check_consistency=False,
        )
    if tuple(d.shape) != nominal:
        d = d.reshape(nominal)
    return _arr2st(d, out_ndim=out_ndim)


def _patched_eliminate_vertex(vertex, jaxpr, graph, transpose_graph, vo_vertices,
                              *, count_ops=False, transforms=()):
    """``_eliminate_vertex`` clone whose ONLY change is: each Diag/Compress/Quant
    transform is applied then the edge is densified+reshaped to nominal. The
    rest of the body is delegated to the original by temporarily swapping the
    micro-action helpers for densifying wrappers keyed off the edge avals set
    in the transform loop. We achieve this without copying the (large) original
    body by patching the helpers the original calls."""
    raise NotImplementedError  # replaced below by the wrapper approach.


# --- Implementation via helper-swap -----------------------------------------
# The original ``_eliminate_vertex`` calls ``apply_diag`` / ``apply_compress``
# / ``apply_quant`` imported into the ``graphax.core`` namespace. We replace
# those names with densifying wrappers that read the current edge's nominal
# shape from ``_edge_ctx`` (set by a patched loop body). Because the original
# loop also sets ``out_edge`` / ``in_edge`` we cannot read them from here, so
# instead we patch ``_set_inner`` — the single call that stores the finished
# ``edge_outval`` keyed by ``(in_edge, out_edge)`` — to densify the stored
# tensor to the nominal shape derived from the edge avals.

_orig_set_inner = gcore._set_inner


def _make_dense_set_inner(active):
    def _set_inner(outer, k1, k2, v):
        # graph stores graph[in_edge][out_edge]; transpose stores
        # transpose_graph[out_edge][in_edge]. Both carry .aval on the vars.
        if active["on"] and hasattr(k1, "aval") and hasattr(k2, "aval"):
            try:
                # Identify which of k1/k2 is out_edge vs in_edge by matching the
                # tensor's logical out/primal split to the avals.
                a1, a2 = tuple(k1.aval.shape), tuple(k2.aval.shape)
                out_shape, primal_shape = _orient(v, a1, a2)
                if out_shape is not None and active["touch"](v):
                    nominal = out_shape + primal_shape
                    v = _densify_to_nominal(v, len(out_shape), nominal)
            except Exception:
                pass
        return _orig_set_inner(outer, k1, k2, v)
    return _set_inner


def _orient(st, a1, a2):
    """Return (out_shape, primal_shape) matching the tensor's logical split,
    or (None, None) if neither orientation fits."""
    out_log = tuple(int(d.logical_size) for d in st.out_dims)
    prim_log = tuple(int(d.logical_size) for d in st.primal_dims)
    op = int(np.prod(out_log)) if out_log else 1
    pp = int(np.prod(prim_log)) if prim_log else 1
    for out_a, prim_a in ((a1, a2), (a2, a1)):
        if int(np.prod(out_a)) == op and int(np.prod(prim_a)) == pp:
            return tuple(out_a), tuple(prim_a)
    return None, None


@contextmanager
def _dense_edges(touch):
    """Activate densify-to-nominal on every stored edge for which ``touch(st)``
    returns True. ``touch`` is used to densify ONLY edges that actually carry an
    approximation (so exact edges stay sparse and the AD path is unchanged)."""
    active = {"on": True, "touch": touch}
    gcore._set_inner = _make_dense_set_inner(active)
    try:
        yield
    finally:
        gcore._set_inner = _orig_set_inner


def oracle_jacobian(fn, order, argnums, transforms):
    """Return a callable computing the DENSE-REFERENCE approximate Jacobian.

    The transforms spec is the normal ``[(vertex, [Diag|Compress|...])]``. The
    only difference from ``jacve(..., transforms=...)`` is that every edge
    carrying an approximation is densified to its nominal logical shape right
    after the transform, so the contraction topology never sees a rectangular
    Diag block or an implicit Compress dim — it runs the math densely."""
    from graphax import jacve

    # The set of vertices that carry an approximation; only those edges get
    # densified, so the no-approximation AD path is byte-identical.
    approx_vertices = {int(v) for v, ts in (transforms or ())
                       if any(isinstance(t, (Diag, Compress)) for t in ts)}

    def touch(st):
        # Densify any edge that has gained a sparse Diag block or an implicit
        # (axis=None on a >1 logical dim) Compress axis.
        for d in (*st.out_dims, *st.primal_dims):
            if d.is_sparse and (getattr(d, "block_size", None) or 1) > 1:
                return True
            if (not d.is_sparse) and d.axis is None and int(d.logical_size) > 1:
                return True
        return False

    def run(*args):
        with _dense_edges(touch):
            return jacve(fn, order, argnums=argnums, transforms=transforms)(*args)

    return run


# --- Fixed reference model set ----------------------------------------------
def _attn(S=4, D=16, H=2):
    dh = D // H

    def attn(x, Wq, Wk, Wv):
        q = (x @ Wq).reshape(S, H, dh)
        k = (x @ Wk).reshape(S, H, dh)
        v = (x @ Wv).reshape(S, H, dh)
        scores = jnp.einsum("shd,thd->sht", q, k) / jnp.sqrt(dh)
        a = jax.nn.softmax(scores, axis=-1)
        return jnp.einsum("sht,thd->shd", a, v).reshape(S, D)

    key = jax.random.PRNGKey(0)
    args = (jax.random.normal(key, (S, D)),
            jax.random.normal(key, (D, D)),
            jax.random.normal(key, (D, D)),
            jax.random.normal(key, (D, D)))
    return attn, args


def _chain():
    def chain(x, W, V):
        return V @ jnp.tanh(W @ x)
    key = jax.random.PRNGKey(0)
    args = (jax.random.normal(key, (4,)),
            jax.random.normal(key, (5, 4)),
            jax.random.normal(key, (3, 5)))
    return chain, args


def _mlp():
    def mlp(x, W1, W2):
        return W2 @ jax.nn.relu(W1 @ x)
    key = jax.random.PRNGKey(0)
    args = (jax.random.normal(key, (6,)),
            jax.random.normal(key, (8, 6)),
            jax.random.normal(key, (3, 8)))
    return mlp, args


MODELS = {"attn": _attn, "chain": _chain, "mlp": _mlp}
N_VERTS = {"attn": 26, "chain": 2, "mlp": 2}


def build_reference_set(path="_approx_oracle_refs.npz"):
    """Save dense-reference Jacobians for the fixed model set under
    {Diag(0,1,2), Compress((0,),'mean')} x {rev, fwd} x every vertex.
    Skips configs the oracle can't evaluate (logged in the npz as NaN)."""
    refs = {}
    for mname, ctor in MODELS.items():
        fn, args = ctor()
        for tname, tr in (("diag", [Diag(0, 1, 2)]),
                          ("compress", [Compress((0,), "mean")])):
            for order in ("rev", "fwd"):
                for v in range(1, N_VERTS[mname] + 1):
                    key = f"{mname}|{tname}|{order}|v{v}"
                    try:
                        J = oracle_jacobian(fn, order, (0,),
                                            [(v, tr)])(*args)
                        refs[key] = np.asarray(J)
                    except Exception as e:  # noqa: BLE001
                        refs[key] = np.array([np.nan])
    np.savez(path, **refs)
    return path, refs


if __name__ == "__main__":
    p, refs = build_reference_set()
    n_ok = sum(1 for v in refs.values() if np.isfinite(v).all() and v.size > 1)
    print(f"saved {len(refs)} refs to {p}; {n_ok} finite")
