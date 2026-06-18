"""Tangent / adjoint SEED VERTICES for vertex elimination.

Adds the tangent seed (ẋ) and the adjoint seed (ȳ) as ORDINARY vertices of the
linearized graph, so the elimination *order* — and thus a learned alphagrad
policy rewarded by real latency / peak-memory — decides HOW to propagate them:
eliminate the seed vertex early (push a forward tangent / pull a reverse adjoint,
matrix-free) or late (materialise a partial Jacobian, then contract). Forward /
reverse / cross-country / seed-timing all become one elimination-order choice.

Why it matters (measured on the dilated-conv MNIST gradient):
    reverse 66 Mflops  >  forward 9.9 Mflops  >  cross-country 3.1 Mflops
the cross-country order is 21x cheaper than reverse — exactly what an
order-optimiser (AlphaZero/PPO in alphagrad/approx, rewarded by latency_ns and
peak_memory) is meant to discover. The seed vertices give it the freedom to do
so for non-scalar outputs / JVPs / VJPs / mixed mode, not just a scalar loss.

Implementation: the seeds are introduced as plain jaxpr equations (a tangent
injection `p + t·ẋ`, or an adjoint contraction `Σⱼ ȳⱼ·yⱼ`), so they show up in
``_build_graph`` as ordinary eliminable vertices — no special-casing in the
elimination core. ``jacve`` (any ``order``) then computes the seeded derivative.

NOTE: getting the *matrix-free* optimum (the last 3.1→0.8 Mflops on conv) also
needs "seed-aware draining" so a reshape/transpose applies to a contracted seed
*vector* rather than densifying a partner diagonal — a separate follow-up. The
seed VERTICES (this module) are what hand the order-optimiser the action space.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.tree_util as jtu

from .core import jacve


def with_adjoint_seed(fun, adjoint):
    """Wrap ``fun`` as ``g(*args) = <adjoint, fun(*args)>`` (a scalar).

    The ``adjoint`` (a cotangent matching ``fun``'s output pytree) enters as a
    ``mul`` + ``sum`` — the **adjoint seed vertex**. ``jacve(g, order, argnums)``
    then computes ``adjointᵀ · J`` (the VJP / gradient), and the elimination
    order is free to eliminate that vertex first (reverse) or later
    (cross-country)."""
    a_leaves = jtu.tree_leaves(adjoint)

    def g(*args):
        o_leaves = jtu.tree_leaves(fun(*args))
        if len(a_leaves) != len(o_leaves):
            raise ValueError(
                f"adjoint seed has {len(a_leaves)} leaves but fun output has "
                f"{len(o_leaves)}"
            )
        return sum(jnp.sum(a * o) for a, o in zip(a_leaves, o_leaves))

    return g


def with_tangent_seed(fun, tangent):
    """Wrap ``fun`` as ``h(t, *primals) = fun(*(p + t·ẋ for p, ẋ in ...))``.

    The scalar ``t`` is the **tangent seed vertex**; ``jacve(h, order,
    argnums=0)`` evaluated at ``t=0`` computes ``J · tangent`` (the JVP). The
    ``tangent`` pytree must match the differentiated ``primals``."""
    x_leaves = jtu.tree_leaves(tangent)

    def h(t, *primals):
        p_leaves = jtu.tree_leaves(primals)
        if len(x_leaves) != len(p_leaves):
            raise ValueError(
                f"tangent seed has {len(x_leaves)} leaves but got "
                f"{len(p_leaves)} primals"
            )
        seeded = tuple(p + t * xd for p, xd in zip(p_leaves, x_leaves))
        return fun(*seeded)

    return h


def seed_vjp(fun, adjoint, order="rev", argnums=0):
    """VJP ``adjointᵀ J`` via an adjoint seed vertex, with a chosen elimination
    ``order`` (a learned alphagrad order is just passed here)."""
    # jacve keys differentiation off ``i in argnums``, so a bare int (the
    # default) must be wrapped — otherwise the default call path raises
    # ``TypeError: argument of type 'int' is not iterable``.
    if isinstance(argnums, int):
        argnums = (argnums,)
    return jacve(with_adjoint_seed(fun, adjoint), order, argnums=argnums)


def seed_jvp(fun, tangent, primals, order="fwd"):
    """JVP ``J · tangent`` via a tangent seed vertex. Returns the JVP evaluated
    at the given ``primals`` (seed ``t = 0``)."""
    h = with_tangent_seed(fun, tangent)
    jvp = jacve(h, order, argnums=(0,))(jnp.zeros(()), *primals)
    # jacve returns a per-argnum tuple; argnums=(0,) -> single entry, the
    # d(output)/d(t) Jacobian which (t scalar) IS J·tangent.
    return jvp[0] if isinstance(jvp, (list, tuple)) else jvp
