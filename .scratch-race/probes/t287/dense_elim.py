"""A dense vertex-elimination oracle. No graphax. Concrete numbers only.

Edge u->v holds the full Jacobian d(v)/d(u), shape ``shape(v) + shape(u)``.
Eliminating v composes every (u->v, v->w) pair with a tensordot over v's rank
and adds the product onto u->w. That is the definition of vertex elimination,
written directly, so nothing about graphax's storage can bias the census.
"""
from __future__ import annotations
import collections
import jax
from jax.extend.core import Var, Literal
import jax.numpy as jnp
import numpy as np


def trace(f, args):
    """(jaxpr, value per var, elemental partial per edge)."""
    cj = jax.make_jaxpr(f)(*args)
    jaxpr, consts = cj.jaxpr, cj.literals
    val = {}
    for v, a in zip(jaxpr.invars, args):
        val[v] = jnp.asarray(a)
    for v, c in zip(jaxpr.constvars, consts):
        val[v] = jnp.asarray(c)

    def read(x):
        return val[x] if isinstance(x, Var) else jnp.asarray(x.val)

    graph = collections.defaultdict(dict)
    prim = {}
    for eqn in jaxpr.eqns:
        invals = [read(x) for x in eqn.invars]
        outs = eqn.primitive.bind(*invals, **eqn.params)
        outs = outs if eqn.primitive.multiple_results else [outs]
        for ov, o in zip(eqn.outvars, outs):
            val[ov] = o
            prim[ov] = eqn.primitive.name
        if len(eqn.outvars) != 1:
            continue
        ov = eqn.outvars[0]
        for pos, iv in enumerate(eqn.invars):
            if not isinstance(iv, Var):
                continue
            if not jnp.issubdtype(jnp.asarray(invals[pos]).dtype, jnp.floating):
                continue          # integer operands (gather indices) carry no
                                  # derivative; they are not graph edges

            def one(a, _pos=pos):
                xs = list(invals)
                xs[_pos] = a
                r = eqn.primitive.bind(*xs, **eqn.params)
                return r[0] if eqn.primitive.multiple_results else r

            part = jnp.asarray(jax.jacfwd(one)(invals[pos]))
            # SUM over every position the variable occupies. Assigning here
            # halved the gradient of ``x * x`` and ``x + x`` while leaving
            # ``x ** 2`` (one operand) correct, so the defect hid from every
            # validation target (grill review 2026-09-07).
            graph[iv][ov] = graph[iv].get(ov, 0) + part
    return jaxpr, val, graph, prim


def compose(pre, post, rank_v):
    """post (S_w + S_v) composed with pre (S_v + S_u) over v's rank."""
    return jnp.tensordot(post, pre, axes=rank_v)


def eliminate(graph, v, rank_v, on_edge=None, step=-1):
    """Remove v, composing through it. ``on_edge(u, w, tensor, kind)`` sees
    every edge the step writes."""
    preds = [u for u in graph if v in graph[u]]
    succs = list(graph.get(v, {}).keys())
    for u in preds:
        pre = graph[u][v]
        for w in succs:
            new = compose(pre, graph[v][w], rank_v)
            if w in graph[u]:
                graph[u][w] = graph[u][w] + new
                kind = "merge"
            else:
                graph[u][w] = new
                kind = "create"
            if on_edge is not None:
                on_edge(u, w, graph[u][w], kind, step)
    for u in preds:
        graph[u].pop(v, None)
    graph.pop(v, None)
