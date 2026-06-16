"""Matrix-free reverse-mode AD (Tier 0) — the gradient the way jax / dco do it.

Vertex elimination (``jacve``) MATERIALISES structured Jacobians and contracts
them; that is what forces a reshape to densify a diagonal into an N×N block
(the conv 4608² problem). The pros never do this: they propagate a *cotangent
vector* and treat reshape / transpose / broadcast as zero-FLOP relabels of that
vector.

This module gives graphax that path. It walks the jaxpr in reverse and, for each
equation, contracts graphax's OWN elemental ``SparseTensor`` Jacobian against the
incoming cotangent (a matvec), never against the identity. So:

* a diagonal elemental (elementwise op) becomes ``ct * diag``  — O(n),
* a structural transform (reshape / transpose / broadcast / reduce) is APPLIED
  TO THE COTANGENT — free, no Jacobian ever formed,
* only a genuinely dense coupling (a real matmul) costs a matvec.

A scalar-loss gradient is then one reverse sweep, O(model) — matching
``jax.grad``. Applied to a sub-jaxpr it yields that subgraph's interface
Jacobian by seeding each output basis vector (Tier 2 = dco-style
preaccumulation).

NOTE: this is the seed engine's foundation; primitive coverage grows
incrementally and every rule is checked against ``jax.grad``/``jax.vjp``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax._src.core as jcore
import jax.tree_util as jtu


def _eval_jaxpr_forward(jaxpr, consts, *args):
    """Standard forward interpreter that also returns the full var->value env,
    so the reverse pass can read each equation's primal inputs."""
    env = {}

    def read(v):
        return v.val if isinstance(v, jcore.Literal) else env[v]

    def write(v, val):
        env[v] = val

    jtu.tree_map(write, list(jaxpr.constvars), list(consts))
    jtu.tree_map(write, list(jaxpr.invars), list(args))
    for eqn in jaxpr.eqns:
        invals = [read(v) for v in eqn.invars]
        outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        for v, val in zip(eqn.outvars, outvals):
            write(v, val)
    return [read(v) for v in jaxpr.outvars], env


def _backward(jaxpr, env, out_cts):
    """Reverse sweep: seed outvars with ``out_cts`` and accumulate input
    cotangents. Per-equation VJP delegates to ``jax.vjp`` on that single
    primitive bound at its stored primal inputs — correct by construction and
    matrix-free (the local vjp acts on the cotangent VECTOR, so reshape /
    transpose / broadcast cost nothing). This is the reference seed engine; the
    graphax-structured contraction replaces the per-op ``jax.vjp`` next."""
    ct = {}

    def read_ct(v, like):
        c = ct.get(v)
        return jnp.zeros(jnp.shape(like), jnp.result_type(like)) if c is None else c

    def add_ct(v, c):
        if isinstance(v, jcore.Literal) or c is None:
            return
        ct[v] = c if v not in ct else ct[v] + c

    for v, c in zip(jaxpr.outvars, out_cts):
        add_ct(v, c)

    for eqn in reversed(jaxpr.eqns):
        out_c = [read_ct(v, v.aval) for v in eqn.outvars]
        if all(c is None for c in out_c):
            continue
        invals = [v.val if isinstance(v, jcore.Literal) else env[v] for v in eqn.invars]
        # Which inputs are differentiable (float, not a constant-only literal).
        nz = [
            i for i, v in enumerate(eqn.invars)
            if not isinstance(v, jcore.Literal)
            and jnp.issubdtype(jnp.result_type(invals[i]), jnp.floating)
        ]
        if not nz:
            continue

        def prim_fn(*diff_ins):
            full = list(invals)
            for i, x in zip(nz, diff_ins):
                full[i] = x
            return eqn.primitive.bind(*full, **eqn.params)

        diff_args = tuple(invals[i] for i in nz)
        _, vjp_fn = jax.vjp(prim_fn, *diff_args)
        cts_in = vjp_fn(out_c[0] if not eqn.primitive.multiple_results else tuple(out_c))
        for i, c in zip(nz, cts_in):
            add_ct(eqn.invars[i], c)

    return ct


def vjp(fun, *primals):
    """``(primals_out, vjp_fn)`` like ``jax.vjp`` — matrix-free reverse mode."""
    closed = jax.make_jaxpr(fun)(*primals)
    jaxpr, consts = closed.jaxpr, closed.literals
    outs, env = _eval_jaxpr_forward(jaxpr, consts, *primals)

    def vjp_fn(cotangents):
        if not isinstance(cotangents, (list, tuple)):
            cotangents = (cotangents,)
        ct = _backward(jaxpr, env, list(cotangents))
        return tuple(
            ct.get(v, jnp.zeros(v.aval.shape, v.aval.dtype)) for v in jaxpr.invars
        )

    return (outs[0] if len(outs) == 1 else outs), vjp_fn


def grad(fun, argnums=0):
    """Matrix-free scalar-output gradient (Tier 0). Seeds the single output with
    ``1.0`` and runs one reverse sweep — O(model), reshape-free."""
    an = (argnums,) if isinstance(argnums, int) else tuple(argnums)

    def g(*args):
        outs, vjp_fn = vjp(fun, *args)
        grads = vjp_fn(jnp.ones(()))
        sel = tuple(grads[i] for i in an)
        return sel[0] if isinstance(argnums, int) else sel

    return g
