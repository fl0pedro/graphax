from typing import Callable

import jax
import jax.lax as lax
import jax.numpy as jnp

from .base import defelemental, defelemental2

Array = jax.Array


defelemental(lax.neg_p, lambda x: -1.0)
# d|x|/dx = sign(x); jnp.sign avoids the 0/0 = NaN of primal/out at x==0, and
# matches jax's subgradient convention there (jax.grad(abs)(0.) == 1.0).
defelemental(lax.abs_p, lambda x: jnp.where(x == 0, 1.0, jnp.sign(x)))
defelemental(lax.integer_pow_p, lambda x, y: y * lax.integer_pow(x, y - 1))

defelemental2(lax.exp_p, lambda out, primal: out)
defelemental(lax.log_p, lambda x, accuracy: 1.0 / x)
defelemental2(lax.sqrt_p, lambda out, primal, accuracy: 0.5 / out)
defelemental(lax.square_p, lambda x: 2.0 * x)
defelemental2(lax.logistic_p, lambda out, primal, accuracy: out * (1.0 - out))
defelemental(lax.log1p_p, lambda x, accuracy: 1.0 / (1.0 + x))

defelemental(lax.sin_p, lambda x, accuracy: lax.cos(x, accuracy=accuracy))
defelemental(lax.asin_p, lambda x: 1.0 / lax.sqrt(1.0 - x**2))
defelemental(lax.cos_p, lambda x, accuracy: -lax.sin(x, accuracy=accuracy))
defelemental(lax.acos_p, lambda x: -1.0 / lax.sqrt(1.0 - x**2))
defelemental2(lax.tan_p, lambda out, primal, accuracy: 1.0 + out**2)
defelemental(lax.atan_p, lambda x: 1.0 / (1.0 + x**2))

defelemental(lax.sinh_p, lax.cosh)
defelemental(lax.asinh_p, lambda x: 1.0 / lax.sqrt(1.0 + x**2))
defelemental(lax.cosh_p, lax.sinh)
defelemental(lax.acosh_p, lambda x: 1.0 / lax.sqrt(x**2 - 1.0))
defelemental2(lax.tanh_p, lambda out, primal, accuracy: 1.0 - out**2)
defelemental(lax.atanh_p, lambda x: 1.0 / (1.0 - x**2))

defelemental(lax.erf_p, lambda x: 2.0 * lax.exp(-(x**2)) / lax.sqrt(jnp.pi))


def with_type_promotion(fn: Callable) -> Callable:
    def promoted_fn(*operands, **params) -> tuple[Array, ...]:
        operands = tuple(jnp.asarray(o) for o in operands)
        res = fn(*operands, **params)
        type = jnp.result_type(*(op.dtype for op in operands))
        return tuple(lax.convert_element_type(el, type) for el in res)

    return promoted_fn


def add_elemental_rule(x, y):
    return (1.0, 1.0)


defelemental(lax.add_p, with_type_promotion(add_elemental_rule))


def sub_elemental_rule(x, y):
    return (1.0, -1.0)


defelemental(lax.sub_p, with_type_promotion(sub_elemental_rule))


@with_type_promotion
def mul_elemental_rule(x, y, **params):
    return (y, x)


defelemental(lax.mul_p, mul_elemental_rule)


@with_type_promotion
def div_elemental_rule(x, y):
    return (1.0 / y, -x / y**2)


defelemental(lax.div_p, div_elemental_rule)


@with_type_promotion
def atan2_elemental_rule(x, y):
    abs2 = x**2 + y**2
    return (y / abs2, -x / abs2)


defelemental(lax.atan2_p, atan2_elemental_rule)


@with_type_promotion
def max_elemental_rule(x, y):
    # Balanced subgradient at a tie (x == y): 0.5/0.5, matching JAX. The old
    # (x>=y, x<y) sent the whole gradient to x at ties.
    eq = (x == y).astype(x.dtype) * 0.5
    return ((x > y).astype(x.dtype) + eq, (y > x).astype(x.dtype) + eq)


defelemental(lax.max_p, max_elemental_rule)


@with_type_promotion
def min_elemental_rule(x, y):
    eq = (x == y).astype(x.dtype) * 0.5
    return ((x < y).astype(x.dtype) + eq, (y < x).astype(x.dtype) + eq)


defelemental(lax.min_p, min_elemental_rule)


@with_type_promotion
def eq_elemental_rule(x, y):
    return (jnp.zeros_like(y), jnp.zeros_like(x))


defelemental(lax.eq_p, eq_elemental_rule)
defelemental(lax.gt_p, eq_elemental_rule)
defelemental(lax.lt_p, eq_elemental_rule)


@with_type_promotion
def pow_elemental_rule(out, x, y):
    # d/dy = log(x)*x^y; at x==0 the true derivative is 0 but log(0)*out is NaN.
    safe_x = jnp.where(x == 0, jnp.ones_like(x), x)
    dy = jnp.where(x == 0, jnp.zeros_like(out), jnp.log(safe_x) * out)
    return (y * x ** (y - 1), dy)


defelemental2(lax.pow_p, pow_elemental_rule)


# --------------------------------------------------------------------------- #
# Migrated from auto.py: unary/binary math rules math.py did not already cover.
# --------------------------------------------------------------------------- #

# rsqrt: d/dx(x^-1/2) = -1/2 x^-3/2 = -0.5 * out^3
defelemental2(lax.rsqrt_p, lambda out, primal, accuracy=None: -0.5 * out**3)
# cbrt: d/dx(x^1/3) = 1/(3 * out^2)
defelemental2(lax.cbrt_p, lambda out, primal, accuracy=None: 1.0 / (3.0 * out**2))
# expm1: d/dx(e^x - 1) = e^x = out + 1
defelemental2(lax.expm1_p, lambda out, primal, accuracy=None: out + 1.0)
# exp2: d/dx(2^x) = 2^x * ln(2) = out * ln(2)
defelemental2(lax.exp2_p, lambda out, x, accuracy=None: out * jnp.log(2.0))

# Piecewise-constant / boolean predicates -> zero gradient.
defelemental(lax.sign_p, lambda x: jnp.zeros_like(x))
defelemental(lax.floor_p, lambda x: jnp.zeros_like(x))
defelemental(lax.ceil_p, lambda x: jnp.zeros_like(x))
defelemental(lax.round_p, lambda x: jnp.zeros_like(x))
defelemental(lax.is_finite_p, lambda x: jnp.zeros_like(x))

# Identity under AD.
defelemental(lax.copy_p, lambda x: jnp.ones_like(x))
defelemental(lax.reduce_precision_p, lambda x, **kw: jnp.ones_like(x))

# erfc: d/dx(erfc(x)) = -2/sqrt(pi) e^-x^2 ; erf_inv: sqrt(pi)/2 e^{out^2}
defelemental(lax.erfc_p, lambda x: -2.0 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2)))
defelemental2(lax.erf_inv_p, lambda out, x: jnp.sqrt(jnp.pi) / 2.0 * jnp.exp(out**2))

# lgamma' = digamma ; digamma' = polygamma(1, .)
defelemental(lax.lgamma_p, lambda x: lax.digamma(x))
defelemental(lax.digamma_p, lambda x: lax.polygamma(jnp.float32(1), x))

# Exponentially-scaled modified Bessel functions.
defelemental(
    lax.bessel_i0e_p, lambda x: lax.bessel_i1e(x) - jnp.sign(x) * lax.bessel_i0e(x)
)
defelemental(
    lax.bessel_i1e_p,
    lambda x: lax.bessel_i0e(x) - lax.bessel_i1e(x) * (jnp.sign(x) + 1.0 / x),
)

# Remaining comparisons (eq/gt/lt above) -> zero gradient.
defelemental(lax.ne_p, eq_elemental_rule)
defelemental(lax.le_p, eq_elemental_rule)
defelemental(lax.ge_p, eq_elemental_rule)


# clamp(lo, x, hi): gradient flows to whichever bound is active. asarray guards
# scalar/Literal bounds (else the comparisons are Python bools without .astype).
@with_type_promotion
def clamp_elemental_rule(lo, x, hi):
    x = jnp.asarray(x); lo = jnp.asarray(lo); hi = jnp.asarray(hi)
    dt = x.dtype
    in_range = ((x >= lo) & (x <= hi)).astype(dt)
    return ((x < lo).astype(dt), in_range, (x > hi).astype(dt))


defelemental(lax.clamp_p, clamp_elemental_rule)


# rem(x, y) = x - y * trunc(x/y): d/dx = 1, d/dy = -trunc(x/y)
@with_type_promotion
def rem_elemental_rule(x, y):
    return (jnp.ones_like(y), -jnp.trunc(x / y))


defelemental(lax.rem_p, rem_elemental_rule)


# igamma(a, x): d/da via lax.igamma_grad_a; d/dx = x^(a-1) e^-x / Gamma(a).
@with_type_promotion
def igamma_elemental_rule(a, x):
    da = lax.igamma_grad_a(a, x)
    dx = jnp.exp((a - 1.0) * jnp.log(x) - x - lax.lgamma(a))
    return (da, dx)


defelemental(lax.igamma_p, igamma_elemental_rule)


# igammac(a, x) = 1 - igamma(a, x): negate both igamma derivatives.
@with_type_promotion
def igammac_elemental_rule(a, x):
    da = -lax.igamma_grad_a(a, x)
    dx = -jnp.exp((a - 1.0) * jnp.log(x) - x - lax.lgamma(a))
    return (da, dx)


defelemental(lax.igammac_p, igammac_elemental_rule)


# polygamma(n, x): d/dn = 0 (integer order); d/dx = polygamma(n+1, x)
@with_type_promotion
def polygamma_elemental_rule(n, x):
    return (jnp.zeros_like(n), lax.polygamma(n + 1, x))


defelemental(lax.polygamma_p, polygamma_elemental_rule)
