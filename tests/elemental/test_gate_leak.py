"""Regression: the approximation-active gate must survive a NESTED
``vertex_elimination_jaxpr`` (lax.cond / lax.switch macro-vertices recurse).

The elemental dispatch is a hard no-op unless ``approx_active()`` is set
(``core.vertex_elimination_jaxpr``). That call RECURSES for cond/switch
macro-vertices; if the nested call hard-reset the flag to ``False`` on exit
(instead of restoring the parent's value), the dispatch would be silently
disabled for the rest of an OUTER approx elimination, routing its remaining
Diag/Compress edges onto the existing path. This pins the save/restore.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import jax.tree_util as tu

from graphax import jacve
from graphax.sparse.elemental import dispatch as DSP
from graphax.sparse.indexes import DenseIndex
from graphax.sparse.micro_actions import Diag
from graphax.sparse.tensor import SparseTensor


def _flat(g):
    return np.concatenate([np.ravel(np.asarray(l)) for l in tu.tree_leaves(g)])


def _dense_diag(i, j, f):
    """Id-preserving DENSE block-diagonal mask of the (i, j) logical pair — a
    leak-immune reference for ``Diag(i, j, f)`` (it densifies + masks itself,
    never touching the elemental dispatch)."""

    def _t(st):
        d = np.asarray(st.dense())
        no = len(st.out_dims)
        if not (i == j or i >= d.ndim or j >= d.ndim):
            Ni, Nj = d.shape[i], d.shape[j]
            if not (Ni % f or Nj % f or f == 1):
                bi, bj = Ni // f, Nj // f
                ns = []
                for ax, s in enumerate(d.shape):
                    ns += [f, bi] if ax == i else [f, bj] if ax == j else [s]
                dr = d.reshape(ns)
                pos = 0
                fi = fj = None
                for ax, _ in enumerate(d.shape):
                    if ax == i:
                        fi = pos; pos += 2
                    elif ax == j:
                        fj = pos; pos += 2
                    else:
                        pos += 1
                fa = np.arange(dr.shape[fi]).reshape(
                    [dr.shape[fi] if a == fi else 1 for a in range(dr.ndim)])
                fb = np.arange(dr.shape[fj]).reshape(
                    [dr.shape[fj] if a == fj else 1 for a in range(dr.ndim)])
                d = (dr * (fa == fb)).reshape(d.shape)
        out_dims = tuple(
            DenseIndex(x.id, int(x.logical_size), k)
            for k, x in enumerate(st.out_dims))
        primal_dims = tuple(
            DenseIndex(x.id, int(x.logical_size), no + k)
            for k, x in enumerate(st.primal_dims))
        return SparseTensor(
            out_dims, primal_dims, jnp.asarray(d),
            scalar_mult=st.scalar_mult, fill_value=st.fill_value,
            pre_transforms=st.pre_transforms, post_transforms=st.post_transforms,
            check_consistency=False)

    return _t


def _cond_model():
    def f(x, W1, W2):
        y = jax.lax.cond(
            x.sum() > 0, lambda a: jnp.tanh(a @ W1), lambda a: a @ W1, x)
        return y @ W2

    k = jax.random.split(jax.random.PRNGKey(0), 3)
    args = (jax.random.normal(k[0], (4, 6)),
            jax.random.normal(k[1], (6, 6)),
            jax.random.normal(k[2], (6, 5)))
    return f, args


def test_exact_cond_unchanged():
    f, args = _cond_model()
    je = _flat(jacve(f, "rev", argnums=(1, 2))(*args))
    jj = _flat(jax.jacrev(f, argnums=(1, 2))(*args))
    assert abs(1 - float(je @ jj / (np.linalg.norm(je) * np.linalg.norm(jj)))) < 1e-5


def test_approx_gate_survives_nested_cond():
    f, args = _cond_model()
    spec_n = [(v, [Diag(0, 1, 2)]) for v in range(1, 40)]
    spec_r = [(v, [_dense_diag(0, 1, 2)]) for v in range(1, 40)]
    nat = _flat(jacve(f, "rev", argnums=(1, 2), transforms=spec_n)(*args))
    ref = _flat(jacve(f, "rev", argnums=(1, 2), transforms=spec_r)(*args))
    assert np.isfinite(nat).all()
    cos = float(nat @ ref / (np.linalg.norm(nat) * np.linalg.norm(ref)))
    assert abs(1 - cos) < 1e-4, f"native vs dense-Diag oracle cos={cos}"
    # the thread-local must be restored after the (recursing) elimination
    assert DSP.approx_active() is False
