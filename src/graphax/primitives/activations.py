"""Named-jit activation Jacobians.

Several ``jax.nn`` activations are ``@jit``-wrapped composites with **no**
``custom_jvp`` — they decompose into ``select_n`` / ``max`` / ``min`` whose
subgradients at the kink may not match the activation's intended one. Each jit
equation carries the function name in ``params['name']`` though, so instead of
inlining + differentiating the decomposition we look the name up in
``ACTIVATION_DERIVS`` and emit graphax's own (diagonal) Jacobian directly.

The set of handled names is registered in ``jit_name_rules`` so the eliminator
keeps them as vertices (``core._jit_kept_as_vertex``) instead of inlining them.
``core``'s jit_p rule (``_make_jit_elemental_rule``) calls
``jit_named_elemental_only`` for a registered name, and falls back to
differentiating the jit body when this rule raises (non-default static args).

The per-element derivatives assume the activations' **default** static arguments
(elu/celu ``alpha=1``, leaky_relu ``negative_slope=0.01``). A call with
non-default static args bakes a different constant into the jit body that the
name alone can't reveal, so the rule recomputes the primal and fails loudly on a
mismatch rather than returning a wrong Jacobian.
"""

import jax
import jax.numpy as jnp

from .base import make_parallel_jacobian, jit_name_rules

# Standard SELU constants (lambda, alpha).
_SELU_SCALE = 1.0507009873554804934193349852946
_SELU_ALPHA = 1.6732632423543772848170429916717

# name -> (x -> d/dx) for the DEFAULT parameterization. Matches jax.nn's gradient
# at every point of differentiability; clean one-sided subgradient at the kinks.
ACTIVATION_DERIVS = {
    "elu": lambda x: jnp.where(x > 0, 1.0, jnp.exp(x)),
    "selu": lambda x: _SELU_SCALE * jnp.where(x > 0, 1.0, _SELU_ALPHA * jnp.exp(x)),
    "celu": lambda x: jnp.where(x > 0, 1.0, jnp.exp(x)),
    "leaky_relu": lambda x: jnp.where(x >= 0, 1.0, 1e-2),
    "hard_tanh": lambda x: jnp.where((x > -1) & (x < 1), 1.0, 0.0),
    "sparse_plus": lambda x: jnp.where(
        x <= -1, 0.0, jnp.where(x >= 1, 1.0, 0.5 * (x + 1.0))
    ),
    "sparse_sigmoid": lambda x: jnp.where((x > -1) & (x < 1), 0.5, 0.0),
}


def jit_named_elemental_only(primal_outs, primals, **params):
    """Dispatch a named jit to graphax's own Jacobian (``multiple_results``).

    Returns ``elementals[output_idx][invar_idx]`` — these are single-output,
    single-input elementwise activations, so the result is ``[[diag_jacobian]]``.
    """
    name = params.get("name")
    deriv_fn = ACTIVATION_DERIVS.get(name)
    if deriv_fn is None:  # only reachable if jit_name_rules and this dict drift
        raise NotImplementedError(
            f"no graphax Jacobian registered for jit function '{name}'"
        )
    x = primals[0]
    out = primal_outs[0]
    # Confirm DEFAULT static args: recompute the primal and compare. A non-default
    # alpha / negative_slope changes the value, so a mismatch means our default
    # derivative would be wrong -> fail loudly.
    ref = getattr(jax.nn, name)(x)
    if not bool(jnp.all(jnp.abs(ref - out) <= 1e-5 + 1e-4 * jnp.abs(out))):
        raise NotImplementedError(
            f"jit function '{name}' was called with non-default static arguments; "
            f"graphax's named Jacobian only covers the defaults. Inline it or "
            f"supply a custom rule."
        )
    deriv = deriv_fn(x)
    return [[make_parallel_jacobian(0, primals, out, deriv)]]


jit_name_rules.update(ACTIVATION_DERIVS.keys())
