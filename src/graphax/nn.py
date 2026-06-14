"""Activation functions with explicit ``custom_jvp`` rules.

Several ``jax.nn`` activations have **no** ``custom_jvp``: they decompose into
``select_n`` / ``max`` / ``min`` and rely on those primitives' subgradients.
graphax now *honors* ``custom_jvp`` rules (see ``custom_jvp_elemental_only`` in
``primitives/auto.py``), so wrapping these activations with their analytic jvp
makes graphax differentiate them through a single clean rule — the exact
subgradient — instead of the decomposition.

Each function's **value** delegates to ``jax.nn`` (identical numerics); only the
derivative is supplied explicitly. The jvp matches ``jax.nn``'s gradient at every
point of differentiability; at the measure-zero kinks we take the analytic
one-sided subgradient (documented per function).

These are drop-in replacements for the ``jax.nn`` versions, intended for use
inside ``graphax.jacve`` / ``graphax.grad``.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import nn as _nn

# Standard SELU constants (lambda, alpha).
_SELU_SCALE = 1.0507009873554804934193349852946
_SELU_ALPHA = 1.6732632423543772848170429916717


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def elu(x, alpha=1.0):
    """Exponential linear unit. d/dx = 1 (x>0) else alpha*exp(x); =1 at x=0."""
    return _nn.elu(x, alpha)


@elu.defjvp
def _elu_jvp(alpha, primals, tangents):
    (x,), (dx,) = primals, tangents
    deriv = jnp.where(x > 0, 1.0, alpha * jnp.exp(x))
    return elu(x, alpha), deriv * dx


@jax.custom_jvp
def selu(x):
    """Scaled ELU. d/dx = scale * (1 if x>0 else alpha*exp(x))."""
    return _nn.selu(x)


@selu.defjvp
def _selu_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    deriv = _SELU_SCALE * jnp.where(x > 0, 1.0, _SELU_ALPHA * jnp.exp(x))
    return selu(x), deriv * dx


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def celu(x, alpha=1.0):
    """Continuously-differentiable ELU. d/dx = 1 (x>0) else exp(x/alpha)."""
    return _nn.celu(x, alpha)


@celu.defjvp
def _celu_jvp(alpha, primals, tangents):
    (x,), (dx,) = primals, tangents
    deriv = jnp.where(x > 0, 1.0, jnp.exp(x / alpha))
    return celu(x, alpha), deriv * dx


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def leaky_relu(x, negative_slope=1e-2):
    """Leaky ReLU. d/dx = 1 (x>=0) else negative_slope."""
    return _nn.leaky_relu(x, negative_slope)


@leaky_relu.defjvp
def _leaky_relu_jvp(negative_slope, primals, tangents):
    (x,), (dx,) = primals, tangents
    deriv = jnp.where(x >= 0, 1.0, negative_slope)
    return leaky_relu(x, negative_slope), deriv * dx


@jax.custom_jvp
def hard_tanh(x):
    """Hard tanh = clip(x, -1, 1). d/dx = 1 on (-1, 1) else 0 (0 at +/-1)."""
    return _nn.hard_tanh(x)


@hard_tanh.defjvp
def _hard_tanh_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    deriv = jnp.where((x > -1) & (x < 1), 1.0, 0.0)
    return hard_tanh(x), deriv * dx


@jax.custom_jvp
def sparse_plus(x):
    """Sparse plus. d/dx = 0 (x<=-1), (x+1)/2 (-1<x<1), 1 (x>=1)."""
    return _nn.sparse_plus(x)


@sparse_plus.defjvp
def _sparse_plus_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    deriv = jnp.where(x <= -1, 0.0, jnp.where(x >= 1, 1.0, 0.5 * (x + 1.0)))
    return sparse_plus(x), deriv * dx


@jax.custom_jvp
def sparse_sigmoid(x):
    """Sparse sigmoid. d/dx = 1/2 on (-1, 1) else 0 (0 at +/-1)."""
    return _nn.sparse_sigmoid(x)


@sparse_sigmoid.defjvp
def _sparse_sigmoid_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    deriv = jnp.where((x > -1) & (x < 1), 0.5, 0.0)
    return sparse_sigmoid(x), deriv * dx


__all__ = [
    "elu", "selu", "celu", "leaky_relu", "hard_tanh",
    "sparse_plus", "sparse_sigmoid",
]
