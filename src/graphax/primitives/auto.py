import copy
from dataclasses import replace
from functools import partial, reduce
from typing import Callable

import jax._src.core as core
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..sparse.tensor import (
    DenseIndex,
    DiagonalIndex,
    SparseTensor,
    _materialize_indexes,
    _swap_back_axes,
)
from ..sparse.indexes import (
    ToeplitzIndex,
    TOEPLITZ_OUT,
    TOEPLITZ_IN,
    TOEPLITZ_TAP,
)
from .base import (
    elemental_rules,
    elemental_only_rules,
    multi_output_elemental_only_rules,
)


def get_ndim(val: ArrayLike) -> int:
    """
    Safely get the ndim of an Array.

    Args:
        val (ArrayLike): Array with or without abstract value attribute, or scalar.

    Returns:
        int: The shape of the input value.
    """
    if isinstance(val, Array):
        return get_ndim(val.aval)
    elif not isinstance(val, (float, int, complex)):
        return val.ndim
    else:
        return 0


def get_shape(val: ArrayLike) -> tuple[int, ...]:
    """
    Safely get the shape of an Array.

    Args:
        val (ArrayLike): Array with or without abstract value attribute, or scalar.

    Returns:
        int: The shape of the input value.
    """
    if isinstance(val, Array):
        return get_shape(val.aval)
    elif not isinstance(val, (float, int, complex)):
        return val.shape
    else:
        return ()


# TODO simplify this!
def make_parallel_jacobian(i, primals, val_out, elemental):
    primal = primals[i]
    primal_size = get_ndim(primal)
    out_size = get_ndim(val_out)
    out_shape = get_shape(val_out)

    if len(primals) == 1:
        if primal_size == 0 and out_size == 0:
            # Singletons
            out_dims = []
            primal_dims = []
        elif primal_size == 0:
            # Handling broadcast of singletons
            out_dims = [
                DenseIndex(i, e, i) for i, e in enumerate(get_shape(val_out))
            ]
            primal_dims = []
        else:
            out_dims = [
                DiagonalIndex(i, e, i, out_size + i) for i, e in enumerate(out_shape)
            ]
            primal_dims = [
                DiagonalIndex(out_size + i, e, i, i) for i, e in enumerate(out_shape)
            ]
    else:
        # >= 2 inputs (mul/add/.../clamp's 3). The per-input logic below only
        # ever reads primals[i] / the i-th elemental, so it is N-ary; the prior
        # ``len == 2`` guard needlessly raised for 3-input elementwise primitives
        # (e.g. clamp) whose partials are each a plain diagonal Jacobian.
        if primal_size == 0 and out_size == 0:
            # Singletons
            out_dims = []
            primal_dims = []
        elif primal_size == 0:
            # Handling broadcast of singletons
            out_dims = [
                DenseIndex(i, e, i) for i, e in enumerate(get_shape(val_out))
            ]
            primal_dims = []
        elif get_shape(primals[i]) != get_shape(val_out):
            # Broadcasting case
            out_dims, primal_dims = [], []

            for i, (os, ps) in enumerate(zip(get_shape(val_out), get_shape(primal))):
                out_size = len(out_dims)
                primal_size = len(primal_dims)
                if ps != os:
                    axis = sum([1 for d in out_dims if d.axis is not None])
                    out_dims.append(DenseIndex(i, os, axis))
                    primal_dims.append(
                        DenseIndex(out_size + primal_size + 1, ps, None)
                    )
                else:
                    axis = sum([1 for d in out_dims if d.size is not None])
                    out_dims.append(
                        DiagonalIndex(i, os, axis, out_size + primal_size + 1)
                    )
                    primal_dims.append(
                        DiagonalIndex(out_size + primal_size + 1, os, axis, i)
                    )
                for j, d in enumerate(primal_dims[:-1]):
                    primal_dims[j] = replace(d, id=d.id + 1)
                    if d.is_sparse:
                        _d = out_dims[d.other_id]
                        out_dims[d.other_id] = replace(_d, other_id=_d.other_id + 1)
            return SparseTensor(out_dims, primal_dims, elemental)

        elif isinstance(elemental, float) or elemental.size == 1:
            if not isinstance(elemental, float):
                elemental = jnp.squeeze(
                    elemental
                )  # TODO dirty quick fix that needs to be properly addressed
            out_dims = [
                DiagonalIndex(i, e, None, out_size + i)
                for i, e in enumerate(get_shape(primal))
            ]
            primal_dims = [
                DiagonalIndex(out_size + i, e, None, i)
                for i, e in enumerate(get_shape(primal))
            ]
        else:
            elemental = jnp.broadcast_to(elemental, get_shape(primal))
            out_dims = [
                DiagonalIndex(i, e, i, out_size + i)
                for i, e in enumerate(get_shape(primal))
            ]
            primal_dims = [
                DiagonalIndex(out_size + i, e, i, i)
                for i, e in enumerate(get_shape(primal))
            ]

    return SparseTensor(out_dims, primal_dims, elemental)


# elemental_rules is imported from .base — registrations below populate the
# shared registry that core.py reads.


def defelemental(primitive, elementalrule):
    assert isinstance(primitive, core.Primitive)
    assert not primitive.multiple_results
    elemental_rules[primitive] = partial(standard_elemental, elementalrule, primitive)


def standard_elemental(elementalrule, primitive, primals, **params):
    assert elementalrule is not None
    val_out = primitive.bind(*primals, **params)
    elementals = elementalrule(*primals, **params)
    elementals = elementals if isinstance(elementals, tuple) else (elementals,)

    elementals_out = [
        make_parallel_jacobian(i, primals, val_out, elemental)
        for i, elemental in enumerate(elementals)
    ]
    return val_out, elementals_out


# NOTE: Useful for stuff such as exp_p
def defelemental2(primitive, elementalrule, **params):
    assert isinstance(primitive, core.Primitive)
    assert not primitive.multiple_results
    elemental_rules[primitive] = partial(
        standard_elemental2, elementalrule, primitive, **params
    )


def standard_elemental2(elementalrule, primitive, primals, **params):
    assert elementalrule is not None
    val_out = primitive.bind(*primals, **params)
    elementals = elementalrule(val_out, *primals, **params)
    elementals = elementals if isinstance(elementals, tuple) else (elementals,)
    elementals_out = [
        make_parallel_jacobian(i, primals, val_out, elemental)
        for i, elemental in enumerate(elementals)
    ]
    return val_out, elementals_out


# Define elemental partials
defelemental(lax.neg_p, lambda x: -jnp.ones_like(x))
# abs subgradient via sign (sign(0)=0 -> use 1 there); the old primal/out was a
# 0/0 NaN at x=0. (math.py registers the live elemental_only version; kept in
# sync here so the two registrations never diverge — see max/min/pow.)
defelemental(lax.abs_p, lambda x: jnp.where(x == 0, 1.0, jnp.sign(x)))
defelemental(lax.integer_pow_p, lambda x, y: y * x ** (y - 1))

defelemental2(lax.exp_p, lambda out, primal, accuracy=None: out)
defelemental(lax.log_p, lambda x, accuracy=None: 1.0 / x)
defelemental2(lax.sqrt_p, lambda out, primal, accuracy=None: 0.5 / out)
defelemental(lax.square_p, lambda x, accuracy=None: 2.0 * x)
defelemental2(lax.logistic_p, lambda out, primal, accuracy=None: out * (1.0 - out))
defelemental(lax.log1p_p, lambda x, accuracy=None: 1.0 / (1.0 + x))

defelemental(lax.sin_p, lax.cos)
defelemental(lax.asin_p, lambda x, accuracy=None: 1.0 / lax.sqrt(1.0 - x**2, accuracy))
defelemental(lax.cos_p, lambda x, accuracy=None: -lax.sin(x, accuracy))
defelemental(lax.acos_p, lambda x, accuracy=None: -1.0 / lax.sqrt(1.0 - x**2, accuracy))
defelemental2(lax.tan_p, lambda out, primal, accuracy=None: 1.0 + out**2)
defelemental(lax.atan_p, lambda x, accuracy=None: 1.0 / (1.0 + x**2))

defelemental(lax.sinh_p, lax.cosh)
defelemental(lax.asinh_p, lambda x, accuracy=None: 1.0 / lax.sqrt(1.0 + x**2, accuracy))
defelemental(lax.cosh_p, lax.sinh)
defelemental(lax.acosh_p, lambda x, accuracy=None: 1.0 / lax.sqrt(x**2 - 1.0, accuracy))
defelemental2(lax.tanh_p, lambda out, primal, accuracy=None: 1.0 - out**2)
defelemental(lax.atanh_p, lambda x, accuracy=None: 1.0 / (1.0 - x**2))

defelemental(
    lax.erf_p,
    lambda x, accuracy=None: 2.0
    * lax.exp(-(x**2), accuracy)
    / lax.sqrt(jnp.pi, accuracy),
)

# rsqrt: d/dx(1/sqrt(x)) = -1/(2*x*sqrt(x)) = -0.5 * x^(-3/2)
defelemental2(lax.rsqrt_p, lambda out, primal, accuracy=None: -0.5 * out**3)

# cbrt: d/dx(x^(1/3)) = 1/(3*x^(2/3)) = (1/3) * out / x  (where out = x^(1/3))
defelemental2(lax.cbrt_p, lambda out, primal, accuracy=None: 1.0 / (3.0 * out**2))

# expm1: d/dx(e^x - 1) = e^x = out + 1
defelemental2(lax.expm1_p, lambda out, primal, accuracy=None: out + 1.0)

# sign: d/dx(sign(x)) = 0 (non-differentiable but 0 almost everywhere)
defelemental(lax.sign_p, lambda x: jnp.zeros_like(x))

# floor/ceil/round: d/dx = 0 (piecewise constant)
defelemental(lax.floor_p, lambda x: jnp.zeros_like(x))
defelemental(lax.ceil_p, lambda x: jnp.zeros_like(x))
defelemental(lax.round_p, lambda x: jnp.zeros_like(x))

# is_finite: d/dx = 0 (boolean predicate)
defelemental(lax.is_finite_p, lambda x: jnp.zeros_like(x))

# copy: d/dx = 1 (identity)
defelemental(lax.copy_p, lambda x: jnp.ones_like(x))

# reduce_precision (mixed-precision / quantization-aware sim): identity under AD.
defelemental(lax.reduce_precision_p, lambda x, **kw: jnp.ones_like(x))

# exp2: d/dx(2^x) = 2^x * ln(2) = out * ln(2). accuracy=None: current JAX puts an
# `accuracy` param on exp2_p's eqn; without it the lambda crashes under jacve.
defelemental2(lax.exp2_p, lambda out, x, accuracy=None: out * jnp.log(2.0))

# erfc: d/dx(erfc(x)) = -2/sqrt(pi) * exp(-x^2)
defelemental(lax.erfc_p, lambda x: -2.0 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2)))

# erf_inv: d/dx(erf_inv(y)) = sqrt(pi)/2 * exp(out^2)
defelemental2(lax.erf_inv_p, lambda out, x: jnp.sqrt(jnp.pi) / 2.0 * jnp.exp(out**2))

# lgamma: d/dx(lgamma(x)) = digamma(x)
defelemental(lax.lgamma_p, lambda x: lax.digamma(x))

# digamma: d/dx(digamma(x)) = polygamma(1, x)
defelemental(lax.digamma_p, lambda x: lax.polygamma(jnp.float32(1), x))

# bessel_i0e: d/dx = bessel_i1e(x) - sign(x) * bessel_i0e(x)
defelemental(
    lax.bessel_i0e_p, lambda x: lax.bessel_i1e(x) - jnp.sign(x) * lax.bessel_i0e(x)
)

# bessel_i1e: d/dx = bessel_i0e(x) - bessel_i1e(x) * (sign(x) + 1/x)
defelemental(
    lax.bessel_i1e_p,
    lambda x: lax.bessel_i0e(x) - lax.bessel_i1e(x) * (jnp.sign(x) + 1.0 / x),
)


def with_type_promotion(fn: Callable) -> Callable:
    def promoted_fn(*operands, **params) -> tuple[Array, ...]:
        res = fn(*operands, **params)

        def safe_type(op):
            return getattr(op, "dtype", type(op))

        typ = reduce(jnp.promote_types, (safe_type(op) for op in operands))

        return tuple(lax.convert_element_type(el, typ) for el in res)

    return promoted_fn


# TODO this can be significantly optimized
# Currently we are creating a new array of ones everytime. Not smart!
@with_type_promotion
def add_elemental_rule(x, y, **kwargs):
    return (jnp.ones_like(y), jnp.ones_like(x))


defelemental(lax.add_p, add_elemental_rule)


# TODO this can also be optimized significantly
@with_type_promotion
def sub_elemental_rule(x, y, **kwargs):
    return (jnp.ones_like(y), -jnp.ones_like(x))


defelemental(lax.sub_p, sub_elemental_rule)


@with_type_promotion
def mul_elemental_rule(x, y, **kwargs):
    return (y, x)


defelemental(lax.mul_p, mul_elemental_rule)


@with_type_promotion
def div_elemental_rule(x, y, **kwargs):
    return (1.0 / y, ((-1.0) * x) / y**2)


defelemental(lax.div_p, div_elemental_rule)


@with_type_promotion
def atan2_elemental_rule(x, y, **kwargs):
    abs2 = x**2 + y**2
    return (y / abs2, ((-1.0) * x) / abs2)


defelemental(lax.atan2_p, atan2_elemental_rule)


@with_type_promotion
def max_elemental_rule(x, y, **kwargs):
    # Balanced subgradient at a tie (x == y): split 0.5/0.5, matching JAX's
    # _balanced_eq. The old (x<y, x>=y) sent the whole gradient to y at ties.
    eq = (x == y).astype(x.dtype) * 0.5
    return ((x > y).astype(x.dtype) + eq, (y > x).astype(x.dtype) + eq)


defelemental(lax.max_p, max_elemental_rule)


@with_type_promotion
def min_elemental_rule(x, y, **kwargs):
    eq = (x == y).astype(x.dtype) * 0.5
    return ((x < y).astype(x.dtype) + eq, (y < x).astype(x.dtype) + eq)


defelemental(lax.min_p, min_elemental_rule)


@with_type_promotion
def eq_elemental_rule(x, y, **kwargs):
    return (jnp.zeros_like(y), jnp.zeros_like(x))


defelemental(lax.eq_p, eq_elemental_rule)
defelemental(lax.gt_p, eq_elemental_rule)
defelemental(lax.lt_p, eq_elemental_rule)
defelemental(lax.ne_p, eq_elemental_rule)
defelemental(lax.le_p, eq_elemental_rule)
defelemental(lax.ge_p, eq_elemental_rule)


# clamp(lo, x, hi): d/dx = indicator(lo <= x <= hi), d/dlo = 0, d/dhi = 0
@with_type_promotion
def clamp_elemental_rule(lo, x, hi):
    # clamp = max(lo, min(x, hi)): gradient flows to whichever bound is active —
    # lo where x<lo, hi where x>hi, else x. (Was hard-zero for lo/hi.)
    # asarray guards scalar/Literal bounds: ``lax.clamp(0.0, x, 1.0)`` makes the
    # comparisons Python bools otherwise, which lack ``.astype``.
    x = jnp.asarray(x); lo = jnp.asarray(lo); hi = jnp.asarray(hi)
    dt = x.dtype
    in_range = ((x >= lo) & (x <= hi)).astype(dt)
    return ((x < lo).astype(dt), in_range, (x > hi).astype(dt))


defelemental(lax.clamp_p, clamp_elemental_rule)


# rem(x, y) = x - y * trunc(x/y): d/dx = 1, d/dy = -trunc(x/y)
@with_type_promotion
def rem_elemental_rule(x, y, **kwargs):
    return (jnp.ones_like(y), -jnp.trunc(x / y))


defelemental(lax.rem_p, rem_elemental_rule)


# igamma(a, x): regularized lower incomplete gamma P(a, x)
# d/da = the exact regularized-incomplete-gamma shape derivative (jax exposes it
# as lax.igamma_grad_a; was a zeros approximation). d/dx = x^(a-1) e^-x / Gamma(a).
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


# polygamma(n, x): d/dn = 0 (n is integer order), d/dx = polygamma(n+1, x)
@with_type_promotion
def polygamma_elemental_rule(n, x):
    return (jnp.zeros_like(n), lax.polygamma(n + 1, x))


defelemental(lax.polygamma_p, polygamma_elemental_rule)


@with_type_promotion
def pow_elemental_rule(out, x, y):
    # d/dy = log(x)*x^y. At x==0 the true derivative is 0, but log(0)*out is
    # 0*(-inf)=NaN; guard with a safe log. (x<0 stays NaN, matching JAX.)
    safe_x = jnp.where(x == 0, jnp.ones_like(x), x)
    dy = jnp.where(x == 0, jnp.zeros_like(out), jnp.log(safe_x) * out)
    return (y * x ** (y - 1), dy)


defelemental2(lax.pow_p, pow_elemental_rule)


def reduce_prod_elemental_rule(primals, **params):
    """d prod / d x_i = product of the OTHER reduced elements. Same dim layout as
    reduce_max (kept axes DiagonalIndex, reduced axes DenseIndex); the val is the
    per-element product-of-others, robust to zeros."""
    val_out = lax.reduce_prod_p.bind(*primals, **params)
    primal = primals[0]
    axes = params["axes"]
    shape = list(get_shape(val_out))

    new_out_dims, new_primal_dims = [], []
    reduce_all = axes is None
    if reduce_all:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseIndex(0, 1, 0))
    elif isinstance(axes, int):
        axes = (axes,)

    l = get_ndim(val_out)
    base = 1 if reduce_all else l
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            shape.insert(i, 1)
            new_primal_dims.append(DenseIndex(base + i, size, i))
        else:
            ll = len(new_out_dims)
            new_out_dims.append(DiagonalIndex(ll, size, i, l + i))
            new_primal_dims.append(DiagonalIndex(l + i, size, i, ll))

    # prod_others, robust to zeros: tot/x_i for non-zero x_i; for x_i == 0,
    # the product of the rest when it is the UNIQUE zero, else 0.
    tot = val_out.reshape(shape)                       # full product (0 if any 0)
    n_zeros = jnp.sum(primal == 0, axis=axes, keepdims=True)
    prod_nonzero = jnp.prod(jnp.where(primal == 0, 1.0, primal), axis=axes, keepdims=True)
    safe = jnp.where(primal != 0, primal, 1.0)
    new_val = jnp.where(
        primal != 0, tot / safe,
        jnp.where(n_zeros == 1, prod_nonzero, 0.0),
    ).astype(jnp.float32)
    return val_out, [
        _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))
    ]


elemental_rules[lax.reduce_prod_p] = reduce_prod_elemental_rule


def cumsum_elemental_rule(primals, **params):
    """Cumulative sum: ``out[i] = sum_{j<=i} x[j]`` along ``axis`` (``j>=i`` when
    ``reverse``). The Jacobian on the scan axis is a triangular ones matrix; all
    other axes are independent (DiagonalIndex). The triangular block is
    position-invariant, so it lives in a small ``(n, n)`` val and the other axes
    broadcast (axis=None)."""
    val_out = lax.cumsum_p.bind(*primals, **params)
    x = primals[0]
    axis = params["axis"]
    reverse = params.get("reverse", False)
    shape = get_shape(x)
    N = len(shape)
    n = shape[axis]

    new_out_dims, new_primal_dims = [], []
    for i, size in enumerate(shape):
        if i == axis:                                  # scan axis: dense (out, in) pair
            new_out_dims.append(DenseIndex(i, size, 0))
            new_primal_dims.append(DenseIndex(N + i, size, 1))
        else:                                          # independent axis: diagonal
            new_out_dims.append(DiagonalIndex(i, size, None, N + i))
            new_primal_dims.append(DiagonalIndex(N + i, size, None, i))

    idx = jnp.arange(n)
    # val[i, j] = d out[i] / d x[j]; axis0=out i, axis1=in j.
    L = (idx[None, :] >= idx[:, None]) if reverse else (idx[None, :] <= idx[:, None])
    return val_out, [
        _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, L.astype(jnp.float32)))
    ]


elemental_rules[lax.cumsum_p] = cumsum_elemental_rule


def sort_elemental_rule(primals, **params):
    """Single-operand ``sort``: ``out = x[argsort(x)]`` along ``dimension``. The
    Jacobian is the selection (permutation) matrix from the argsort indices on
    the sort axis; other axes are independent (DiagonalIndex). ``sort_p`` is
    multiple_results, hence the ``[tensor]`` (one per output) return shape."""
    operand = primals[0]
    dim = params["dimension"]
    num_keys = params.get("num_keys", 1)
    if len(primals) != 1 or num_keys != 1:
        raise NotImplementedError(
            "sort_elemental_rule supports the single-operand case only "
            f"(got {len(primals)} operands, num_keys={num_keys})."
        )
    val_out_list = lax.sort_p.bind(*primals, **params)
    shape = get_shape(operand)
    n = len(shape)
    S = shape[dim]

    perm = jnp.argsort(operand, axis=dim)
    perm_e = jnp.expand_dims(perm, dim + 1)
    j_idx = jnp.arange(S).reshape([S if k == dim + 1 else 1 for k in range(n + 1)])
    indicator = (perm_e == j_idx).astype(jnp.float32)  # out[i] picks in[perm[i]]

    new_out_dims, new_primal_dims = [], []
    for i, size in enumerate(shape):
        if i == dim:
            new_out_dims.append(DenseIndex(i, size, dim))
            new_primal_dims.append(DenseIndex(n + i, size, dim + 1))
        else:
            vd = i if i < dim else i + 1   # val axis (shifted past the extra in-axis)
            new_out_dims.append(DiagonalIndex(i, size, vd, n + i))
            new_primal_dims.append(DiagonalIndex(n + i, size, vd, i))
    st = _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, indicator))
    return val_out_list, [st]


elemental_rules[lax.sort_p] = sort_elemental_rule


def rev_elemental_rule(primals, **params):
    """``rev`` (``jnp.flip``): each flipped axis is a reverse permutation
    (``out[i] = x[n-1-i]``, Jacobian ``R[i,j] = (j == n-1-i)``); unflipped axes
    are independent (DiagonalIndex). The reverse indicator is value-independent,
    so flipped axes get a small dense (out,in) pair and the others broadcast."""
    val_out = lax.rev_p.bind(*primals, **params)
    x = primals[0]
    dims = params["dimensions"]
    shape = get_shape(x)
    N = len(shape)

    new_out_dims, new_primal_dims, Rs, vax = [], [], [], 0
    for i, size in enumerate(shape):
        if i in dims:
            new_out_dims.append(DenseIndex(i, size, vax))
            new_primal_dims.append(DenseIndex(N + i, size, vax + 1))
            idx = jnp.arange(size)
            Rs.append((idx[None, :] == (size - 1 - idx[:, None])).astype(jnp.float32))
            vax += 2
        else:
            new_out_dims.append(DiagonalIndex(i, size, None, N + i))
            new_primal_dims.append(DiagonalIndex(N + i, size, None, i))

    if not Rs:
        val = jnp.array(1.0, dtype=jnp.float32)
    else:
        val = Rs[0]
        for R in Rs[1:]:                               # outer product over flipped axes
            val = val[..., None, None] * R[None, None, ...]
    return val_out, [
        _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, val))
    ]


elemental_rules[lax.rev_p] = rev_elemental_rule


def _cumulative_dims(shape, axis, N):
    """Dim lists for a value-DEPENDENT cumulative op (cumprod/cummax/cummin):
    the scan axis is a dense (out i, in j) pair (val axes ``axis``/``axis+1``);
    every other axis is a DiagonalIndex reading its own (shifted) val axis — the
    val carries the full per-position matrix, unlike cumsum's broadcast block."""
    out_dims, primal_dims = [], []
    for i, size in enumerate(shape):
        if i == axis:
            out_dims.append(DenseIndex(i, size, axis))
            primal_dims.append(DenseIndex(N + i, size, axis + 1))
        else:
            vd = i if i < axis else i + 1
            out_dims.append(DiagonalIndex(i, size, vd, N + i))
            primal_dims.append(DiagonalIndex(N + i, size, vd, i))
    return out_dims, primal_dims


def _cumulative_mask(n, axis, N, reverse):
    """Triangular ``(j<=i)`` mask (``j>=i`` if reverse), broadcastable to the
    full val with the scan-out axis at ``axis`` and scan-in axis at ``axis+1``."""
    idx = jnp.arange(n)
    m = (idx[None, :] >= idx[:, None]) if reverse else (idx[None, :] <= idx[:, None])
    mshape = [1] * (N + 1)
    mshape[axis] = n; mshape[axis + 1] = n
    return m.astype(jnp.float32).reshape(mshape)


def cumprod_elemental_rule(primals, **params):
    """``out[i] = prod_{j<=i} x[j]``; d out[i]/d x[j] (for j<=i) = product of the
    OTHER factors up to ``i``. For ``x[j] != 0`` that is ``out[i]/x[j]``; for
    ``x[j] == 0`` it is the cumulative product of the non-zeros up to ``i`` when
    ``x[j]`` is the UNIQUE zero in that window, else 0 (a second zero ≤ i makes
    every partial-product-of-others vanish). The naive ``out[i]/x[j]`` was a 0/0
    at zeros and dropped that contribution — wrong for any input with a zero
    (e.g. a masked / post-ReLU activation)."""
    val_out = lax.cumprod_p.bind(*primals, **params)
    x = primals[0]; axis = params["axis"]; reverse = params.get("reverse", False)
    shape = get_shape(x); N = len(shape)
    out_dims, primal_dims = _cumulative_dims(shape, axis, N)
    mask = _cumulative_mask(shape[axis], axis, N, reverse)
    out_e = jnp.expand_dims(val_out, axis + 1)         # out[i] at scan-out axis
    x_e = jnp.expand_dims(x, axis)                      # x[j] at scan-in axis
    # Cumulative (same direction as the op) zero-count and non-zero product up to i.
    is_zero = (x == 0).astype(jnp.float32)
    nz_count = lax.cumsum(is_zero, axis=axis, reverse=reverse)
    prod_nz = lax.cumprod(jnp.where(x == 0, 1.0, x).astype(jnp.float32),
                          axis=axis, reverse=reverse)
    nz_count_i = jnp.expand_dims(nz_count, axis + 1)    # zeros up to i
    prod_nz_i = jnp.expand_dims(prod_nz, axis + 1)      # prod of non-zeros up to i
    safe = jnp.where(x_e != 0, x_e, 1.0)
    deriv = jnp.where(
        x_e != 0,
        out_e / safe,                                  # 0 if another zero <= i
        jnp.where(nz_count_i == 1, prod_nz_i, 0.0),    # x[j] is the unique zero
    )
    V = (mask * deriv).astype(jnp.float32)
    return val_out, [_swap_back_axes(SparseTensor(out_dims, primal_dims, V))]


elemental_rules[lax.cumprod_p] = cumprod_elemental_rule


def cumlogsumexp_elemental_rule(primals, **params):
    """``out[i] = log(sum_{j<=i} exp(x[j]))``; d out[i]/d x[j] (for j<=i) is the
    softmax weight ``exp(x[j] - out[i])`` over the prefix window. Same triangular
    value-dependent layout as cumprod. (CTC loss / HMM-CRF forward passes.)"""
    val_out = lax.cumlogsumexp_p.bind(*primals, **params)
    x = primals[0]; axis = params["axis"]; reverse = params.get("reverse", False)
    shape = get_shape(x); N = len(shape)
    out_dims, primal_dims = _cumulative_dims(shape, axis, N)
    mask = _cumulative_mask(shape[axis], axis, N, reverse)
    out_e = jnp.expand_dims(val_out, axis + 1)         # out[i] at scan-out axis
    x_e = jnp.expand_dims(x, axis)                      # x[j] at scan-in axis
    V = (mask * jnp.exp(x_e - out_e)).astype(jnp.float32)
    return val_out, [_swap_back_axes(SparseTensor(out_dims, primal_dims, V))]


elemental_rules[lax.cumlogsumexp_p] = cumlogsumexp_elemental_rule


def _cum_extremum_rule(prim, primals, params):
    """Shared cummax/cummin: d out[i]/d x[j] = (j<=i)·[x[j]==out[i]] normalised by
    the number of tie positions <= i (matches jax's tie convention)."""
    val_out = prim.bind(*primals, **params)
    x = primals[0]; axis = params["axis"]; reverse = params.get("reverse", False)
    shape = get_shape(x); N = len(shape)
    out_dims, primal_dims = _cumulative_dims(shape, axis, N)
    mask = _cumulative_mask(shape[axis], axis, N, reverse)
    out_e = jnp.expand_dims(val_out, axis + 1)
    x_e = jnp.expand_dims(x, axis)
    hit = mask * (x_e == out_e).astype(jnp.float32)
    norm = jnp.sum(hit, axis=axis + 1, keepdims=True)  # #ties up to i
    V = (hit / norm).astype(jnp.float32)
    return val_out, [_swap_back_axes(SparseTensor(out_dims, primal_dims, V))]


def cummax_elemental_rule(primals, **params):
    return _cum_extremum_rule(lax.cummax_p, primals, params)


def cummin_elemental_rule(primals, **params):
    return _cum_extremum_rule(lax.cummin_p, primals, params)


elemental_rules[lax.cummax_p] = cummax_elemental_rule
elemental_rules[lax.cummin_p] = cummin_elemental_rule


def dot_general_elemental_rule(primals, **params):
    val_out = lax.dot_general_p.bind(*primals, **params)
    lhs, rhs = primals

    dimension_numbers = params["dimension_numbers"][0]
    batch_dims = params["dimension_numbers"][1]

    lhs_contracting_dims = dimension_numbers[0]
    rhs_contracting_dims = dimension_numbers[1]

    lhs_batch_dims = batch_dims[0]
    rhs_batch_dims = batch_dims[1]

    lhs_shape = list(get_shape(lhs))
    rhs_shape = list(get_shape(rhs))
    out_shape = list(get_shape(val_out))

    lhs_out_dims, rhs_out_dims = [], []
    lhs_primal_dims, rhs_primal_dims = [], []

    num_out_dims = len(out_shape)
    num_batch = len(lhs_batch_dims)

    # The dot_general output lays batch axes out first, in batch-tuple order
    # (``lhs_batch_dims[p]`` pairs with ``rhs_batch_dims[p]`` at output position
    # ``p``). Emit the batch DiagonalIndex pairs FIRST in canonical ``p`` order so
    # output positions stay correct under permuted batch/contracting axes; primal
    # lists are indexed by axis position (id == num_out_dims + axis).
    lhs_primal_dims = [None] * len(lhs_shape)
    rhs_primal_dims = [None] * len(rhs_shape)
    for p in range(num_batch):
        l_ax, r_ax = lhs_batch_dims[p], rhs_batch_dims[p]
        size = lhs_shape[l_ax]
        lhs_out_dims.append(DiagonalIndex(p, size, r_ax, num_out_dims + l_ax))
        rhs_out_dims.append(DiagonalIndex(p, size, l_ax, num_out_dims + r_ax))
        lhs_primal_dims[l_ax] = DiagonalIndex(num_out_dims + l_ax, size, r_ax, p)
        rhs_primal_dims[r_ax] = DiagonalIndex(num_out_dims + r_ax, size, l_ax, p)

    for lid, ld in enumerate(lhs_shape):
        other_lid = lid + num_out_dims
        if lid in lhs_contracting_dims:
            # Pair this contracting lhs axis with its rhs partner *positionally*:
            # ``lhs_contracting_dims[p]`` contracts with ``rhs_contracting_dims[p]``.
            # Look ``lid`` up in ``lhs_contracting_dims`` rather than relying on
            # encounter order, which only coincides when the contracting dims are
            # listed in ascending axis order (breaks for permuted contractions).
            # The DenseIndex size is the lhs axis's own size ``ld`` (== rhs partner
            # size, since contracting axes match), and its ``axis`` (val_dim into the
            # ``rhs`` val) is the partner rhs axis.
            dim = rhs_contracting_dims[lhs_contracting_dims.index(lid)]
            lhs_primal_dims[lid] = DenseIndex(other_lid, ld, dim)
        elif lid not in lhs_batch_dims:
            _lid = len(lhs_out_dims)
            lhs_out_dims.append(DiagonalIndex(_lid, ld, None, other_lid))
            lhs_primal_dims[lid] = DiagonalIndex(other_lid, ld, None, _lid)
            rhs_out_dims.append(DenseIndex(len(rhs_out_dims), ld, lid))

    for rid, rd in enumerate(rhs_shape):
        other_rid = rid + num_out_dims
        if rid in rhs_contracting_dims:
            # Symmetric to the lhs loop: pair ``rid`` with its lhs partner
            # positionally via its index in ``rhs_contracting_dims``. Size is the
            # rhs axis's own size ``rd``; ``axis`` is the partner lhs axis (val_dim
            # into the ``lhs`` val).
            dim = lhs_contracting_dims[rhs_contracting_dims.index(rid)]
            rhs_primal_dims[rid] = DenseIndex(other_rid, rd, dim)
        elif rid not in rhs_batch_dims:
            _rid = len(rhs_out_dims)
            rhs_out_dims.append(DiagonalIndex(_rid, rd, None, other_rid))
            rhs_primal_dims[rid] = DiagonalIndex(other_rid, rd, None, _rid)
            lhs_out_dims.append(DenseIndex(len(lhs_out_dims), rd, rid))

    lhs_tensor = SparseTensor(lhs_out_dims, lhs_primal_dims, rhs)
    rhs_tensor = SparseTensor(rhs_out_dims, rhs_primal_dims, lhs)

    return val_out, [lhs_tensor, rhs_tensor]


elemental_rules[lax.dot_general_p] = dot_general_elemental_rule


def _build_windowed_jacobian_1d(out_s, in_s, wd, ws, pad_lo, bd=1, wid=1):
    eff_wd = (wd - 1) * wid + 1
    rows = jnp.repeat(jnp.arange(out_s), wd)
    offsets = jnp.tile(jnp.arange(wd) * wid, out_s)
    dilated_pos = rows * ws + offsets - pad_lo
    if bd > 1:
        valid_dilation = dilated_pos % bd == 0
        cols = dilated_pos // bd
    else:
        valid_dilation = jnp.ones_like(dilated_pos, dtype=bool)
        cols = dilated_pos
    valid = (cols >= 0) & (cols < in_s) & valid_dilation
    safe_cols = jnp.where(valid, cols, 0)
    J = jnp.zeros((out_s, in_s), dtype=jnp.float32)
    J = J.at[rows, safe_cols].add(jnp.where(valid, 1.0, 0.0))
    return J


def conv_general_dilated_elemental_rule(primals, **params):
    """Exact sparse Jacobian of ``conv_general_dilated`` w.r.t. both operands.

    For each output spatial axis the (out-pos, in-pos, kernel-tap) incidence is
    the windowed indicator ``M_s[p, q, k]`` (:meth:`ToeplitzIndex.indicator`).
    Writing the conv as ``out[n,o,P] = sum_{i,K} lhs[n,i,Q(P,K)] * rhs[o,i,K]``:

      * d out / d lhs  has value ``W[o,P,i,Q] = sum_K (prod_s M_s) rhs[o,i,K]``
        with the batch axis a DiagonalIndex pair (n == n').
      * d out / d rhs  has value ``V[n,P,i,K] = sum_Q (prod_s M_s) lhs[n,i,Q]``
        with the out-feature axis a DiagonalIndex pair (o == o').

    Rather than materialize those dense ``W`` / ``V`` blocks, each windowed
    spatial axis is emitted as a :class:`ToeplitzIndex` PAIR and ``val`` stores
    only the COMPRESSED operand — the kernel ``rhs`` for ``d out/d lhs``, the
    activations ``lhs`` for ``d out/d rhs`` (up to ``K×`` / huge savings). The
    dense band is rebuilt scatter-free by ``_densify_toeplitz`` at the op
    boundary. Batch (lhs) and out-feature (rhs) stay DiagonalIndex pairs; a
    pass-through (1x1 / pointwise) spatial axis is a DiagonalIndex pair too.
    Validated to 0 error vs ``jax.jacfwd``/``jax.jacrev``.
    """
    val_out = lax.conv_general_dilated_p.bind(*primals, **params)
    lhs, rhs = primals

    dn = params["dimension_numbers"]
    lhs_spec, rhs_spec, out_spec = dn.lhs_spec, dn.rhs_spec, dn.out_spec

    lhs_shape = list(get_shape(lhs))
    rhs_shape = list(get_shape(rhs))
    out_shape = list(get_shape(val_out))
    out_ndim = len(out_shape)
    nsp = out_ndim - 2

    if params.get("feature_group_count", 1) != 1 or params.get("batch_group_count", 1) != 1:
        raise NotImplementedError(
            "conv elemental rule supports feature_group_count == "
            "batch_group_count == 1 only (grouped / depthwise conv: TODO)"
        )

    strides = params["window_strides"]
    padding = params["padding"]
    lhs_dil = params["lhs_dilation"] or (1,) * nsp       # base / input dilation
    rhs_dil = params["rhs_dilation"] or (1,) * nsp       # window / kernel dilation

    N = lhs_shape[lhs_spec[0]]
    I = lhs_shape[lhs_spec[1]]
    O = rhs_shape[rhs_spec[0]]

    # Per-spatial geometry + pass-through flag: a stride-1, kernel-1, undilated,
    # unpadded axis whose out/in sizes match is an IDENTITY block in the
    # activation Jacobian, so it stays a DiagonalIndex pair (the 1x1 /
    # pointwise-conv win). Genuinely windowed axes become ToeplitzIndex pairs
    # whose val stores the operand (kernel / activations) COMPRESSED; the dense
    # P x Q band is rebuilt only at the op boundary, scatter-free, by
    # _densify_toeplitz.
    geom, passthrough = [], []
    for s in range(nsp):
        P_s = out_shape[out_spec[2 + s]]
        X_s = lhs_shape[lhs_spec[2 + s]]
        K_s = rhs_shape[rhs_spec[2 + s]]
        pad_lo, pad_hi = padding[s]
        geom.append(dict(out_size=P_s, in_size=X_s, kernel_size=K_s,
                         stride=strides[s], win_dilation=rhs_dil[s],
                         base_dilation=lhs_dil[s], pad_lo=pad_lo))
        passthrough.append(
            K_s == 1 and strides[s] == 1 and lhs_dil[s] == 1 and rhs_dil[s] == 1
            and pad_lo == 0 and pad_hi == 0 and P_s == X_s)

    # Canonicalize operands to (feat-first, spatial-last), independent of the
    # dimension_numbers permutation. These ARE the compressed vals.
    rhs_c = jnp.transpose(rhs, [rhs_spec[0], rhs_spec[1], *(rhs_spec[2 + s] for s in range(nsp))])
    lhs_c = jnp.transpose(lhs, [lhs_spec[0], lhs_spec[1], *(lhs_spec[2 + s] for s in range(nsp))])

    def _toe(out_id, prim_id, out_role, prim_role, out_size, prim_size, s, axis):
        """A ToeplitzIndex pair: primary (out side) owns the contracted ``axis``
        of val; the partner carries ``axis=None``. Both hold the full geometry."""
        g = geom[s]
        return (ToeplitzIndex(out_id, out_size, axis, prim_id, role=out_role,
                              primary=True, **g),
                ToeplitzIndex(prim_id, prim_size, None, out_id, role=prim_role,
                              primary=False, **g))

    # --- d out / d rhs : out-feature DiagonalIndex pair; each spatial axis is a
    #     ToeplitzIndex (OUT x TAP) contracting the input axis; val = lhs_c ---
    rhs_out = [None] * out_ndim
    rhs_primal = [None] * len(rhs_shape)
    f_out, f_prim = out_spec[1], out_ndim + rhs_spec[0]
    rhs_out[out_spec[0]] = DenseIndex(out_spec[0], N, 0)               # batch N (from lhs)
    rhs_out[out_spec[1]] = DiagonalIndex(out_spec[1], O, None, f_prim)
    rhs_primal[rhs_spec[0]] = DiagonalIndex(f_prim, O, None, f_out)
    rhs_primal[rhs_spec[1]] = DenseIndex(out_ndim + rhs_spec[1], I, 1)  # in-feature I
    for s in range(nsp):
        oa, ra = out_spec[2 + s], rhs_spec[2 + s]
        ot, pt = _toe(oa, out_ndim + ra, TOEPLITZ_OUT, TOEPLITZ_TAP,
                      out_shape[oa], rhs_shape[ra], s, 2 + s)
        rhs_out[oa], rhs_primal[ra] = ot, pt
    # check_consistency=False: a ToeplitzIndex pair links axes of DIFFERENT
    # sizes (P vs X/K), like BandedIndex — the size-equality invariant does not
    # apply to compressed pairs (they are densified before any consumer).
    rhs_tensor = SparseTensor(rhs_out, rhs_primal, lhs_c, check_consistency=False)

    # --- d out / d lhs : batch DiagonalIndex pair; pass-through spatial axes are
    #     DiagonalIndex (identity), windowed axes are ToeplitzIndex (OUT x IN)
    #     contracting the kernel-tap axis; val = the (reduced) kernel rhs_c ---
    win = [s for s in range(nsp) if not passthrough[s]]
    widx = {s: j for j, s in enumerate(win)}
    # Drop pass-through kernel axes (size 1) from the stored kernel.
    rhs_red = rhs_c[(slice(None), slice(None))
                    + tuple(0 if passthrough[s] else slice(None) for s in range(nsp))]
    lhs_out = [None] * out_ndim
    lhs_primal = [None] * len(lhs_shape)
    b_out, b_prim = out_spec[0], out_ndim + lhs_spec[0]
    lhs_out[out_spec[0]] = DiagonalIndex(out_spec[0], N, None, b_prim)
    lhs_out[out_spec[1]] = DenseIndex(out_spec[1], O, 0)
    lhs_primal[lhs_spec[0]] = DiagonalIndex(b_prim, N, None, b_out)
    lhs_primal[lhs_spec[1]] = DenseIndex(out_ndim + lhs_spec[1], I, 1)
    for s in range(nsp):
        oa, la = out_spec[2 + s], lhs_spec[2 + s]
        if passthrough[s]:
            lhs_out[oa] = DiagonalIndex(oa, out_shape[oa], None, out_ndim + la)
            lhs_primal[la] = DiagonalIndex(out_ndim + la, lhs_shape[la], None, oa)
        else:
            ot, pt = _toe(oa, out_ndim + la, TOEPLITZ_OUT, TOEPLITZ_IN,
                          out_shape[oa], lhs_shape[la], s, 2 + widx[s])
            lhs_out[oa], lhs_primal[la] = ot, pt
    lhs_tensor = SparseTensor(lhs_out, lhs_primal, rhs_red, check_consistency=False)

    return val_out, [lhs_tensor, rhs_tensor]


elemental_rules[lax.conv_general_dilated_p] = conv_general_dilated_elemental_rule


def pad_elemental_rule(primals, **params):
    val_out = lax.pad_p.bind(*primals, **params)
    x, padding_value = primals
    padding_config = params["padding_config"]

    x_shape = get_shape(x)
    out_shape = get_shape(val_out)
    x_ndim = get_ndim(x)
    out_ndim = get_ndim(val_out)

    new_out_dims = []
    new_primal_dims = []
    axis_count = 0
    padded_axes = []

    for i, (lo, hi, interior) in enumerate(padding_config):
        is_identity = lo == 0 and hi == 0 and interior == 0
        if is_identity:
            ll = len(new_out_dims)
            new_out_dims.append(DiagonalIndex(ll, x_shape[i], None, out_ndim + i))
            new_primal_dims.append(DiagonalIndex(out_ndim + i, x_shape[i], None, ll))
        else:
            out_vd = axis_count
            new_out_dims.append(DenseIndex(len(new_out_dims), out_shape[i], out_vd))
            axis_count += 1
            primal_vd = axis_count
            new_primal_dims.append(DenseIndex(out_ndim + i, x_shape[i], primal_vd))
            axis_count += 1
            padded_axes.append((i, out_shape[i], x_shape[i], lo, hi, interior))

    if len(padded_axes) == 0:
        val = jnp.array(1.0, dtype=jnp.float32)
    else:
        jac_matrices = []
        for _, out_s, in_s, lo, hi, interior in padded_axes:
            stride = interior + 1
            J = jnp.zeros((out_s, in_s), dtype=jnp.float32)
            cols = jnp.arange(in_s)
            rows = lo + cols * stride
            valid = (rows >= 0) & (rows < out_s)
            safe_rows = jnp.where(valid, rows, 0)
            J = J.at[safe_rows, cols].add(jnp.where(valid, 1.0, 0.0))
            jac_matrices.append(J)
        val = jac_matrices[0]
        for j in range(1, len(jac_matrices)):
            val = val[..., None, None] * jac_matrices[j][None, None, ...]

    x_tensor = _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, val))
    tensors_out = [x_tensor]

    if not isinstance(padding_value, (float, int, complex)):
        pad_mask = jnp.ones_like(val_out)
        slices_obj = tuple(
            slice(lo, lo + (x_shape[i] - 1) * (interior + 1) + 1, interior + 1)
            for i, (lo, hi, interior) in enumerate(padding_config)
        )
        pad_mask = pad_mask.at[slices_obj].set(0.0)
        p_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
        tensors_out.append(SparseTensor(p_out_dims, [], pad_mask))

    return val_out, tensors_out


elemental_rules[lax.pad_p] = pad_elemental_rule


def reduce_window_sum_elemental_rule(primals, **params):
    val_out = lax.reduce_window_sum_p.bind(*primals, **params)
    x = primals[0]

    window_dimensions = params["window_dimensions"]
    window_strides = params.get("window_strides", (1,) * x.ndim)
    padding_pairs = params.get("padding", tuple((0, 0) for _ in range(x.ndim)))
    base_dilation = params.get("base_dilation", (1,) * x.ndim)
    window_dilation = params.get("window_dilation", (1,) * x.ndim)

    x_shape = get_shape(x)
    out_shape = get_shape(val_out)
    x_ndim = get_ndim(x)
    out_ndim = get_ndim(val_out)

    new_out_dims = []
    new_primal_dims = []
    axis_count = 0
    windowed_axes = []

    for i in range(x_ndim):
        wd = window_dimensions[i]
        ws = window_strides[i] if window_strides else 1
        bd = base_dilation[i] if base_dilation else 1
        wid = window_dilation[i] if window_dilation else 1
        pad_lo, pad_hi = padding_pairs[i] if padding_pairs else (0, 0)

        is_passthrough = (
            wd == 1
            and ws == 1
            and bd == 1
            and wid == 1
            and pad_lo == 0
            and pad_hi == 0
            and x_shape[i] == out_shape[i]
        )

        if is_passthrough:
            ll = len(new_out_dims)
            new_out_dims.append(DiagonalIndex(ll, x_shape[i], None, out_ndim + i))
            new_primal_dims.append(DiagonalIndex(out_ndim + i, x_shape[i], None, ll))
        else:
            out_vd = axis_count
            new_out_dims.append(DenseIndex(len(new_out_dims), out_shape[i], out_vd))
            axis_count += 1
            primal_vd = axis_count
            new_primal_dims.append(DenseIndex(out_ndim + i, x_shape[i], primal_vd))
            axis_count += 1
            windowed_axes.append((i, out_shape[i], x_shape[i], wd, ws, pad_lo, bd, wid))

    if len(windowed_axes) == 0:
        val = jnp.array(1.0, dtype=jnp.float32)
    else:
        jac_matrices = []
        for _, out_s, in_s, wd, ws, pad_lo, bd, wid in windowed_axes:
            J = _build_windowed_jacobian_1d(out_s, in_s, wd, ws, pad_lo, bd, wid)
            jac_matrices.append(J)
        val = jac_matrices[0]
        for j in range(1, len(jac_matrices)):
            val = val[..., None, None] * jac_matrices[j][None, None, ...]

    return val_out, [_swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, val))]


elemental_rules[lax.reduce_window_sum_p] = reduce_window_sum_elemental_rule


def reduce_window_elemental_rule(primals, **params):
    val_out = lax.reduce_window_p.bind(*primals, **params)
    x, init_value = primals

    window_dimensions = params["window_dimensions"]
    window_strides = params.get("window_strides", (1,) * x.ndim)
    padding_pairs = params.get("padding", tuple((0, 0) for _ in range(x.ndim)))
    base_dilation = params.get("base_dilation", (1,) * x.ndim)
    window_dilation = params.get("window_dilation", (1,) * x.ndim)

    x_shape = get_shape(x)
    out_shape = get_shape(val_out)
    x_ndim = get_ndim(x)
    out_ndim = get_ndim(val_out)

    new_out_dims = []
    new_primal_dims = []
    axis_count = 0
    passthrough_axes = []
    windowed_axes = []

    for i in range(x_ndim):
        wd = window_dimensions[i]
        ws = window_strides[i] if window_strides else 1
        bd = base_dilation[i] if base_dilation else 1
        wid = window_dilation[i] if window_dilation else 1
        pad_lo, pad_hi = padding_pairs[i] if padding_pairs else (0, 0)

        is_passthrough = (
            wd == 1
            and ws == 1
            and bd == 1
            and wid == 1
            and pad_lo == 0
            and pad_hi == 0
            and x_shape[i] == out_shape[i]
        )

        if is_passthrough:
            ll = len(new_out_dims)
            new_out_dims.append(
                DiagonalIndex(ll, x_shape[i], axis_count, out_ndim + i)
            )
            new_primal_dims.append(
                DiagonalIndex(out_ndim + i, x_shape[i], axis_count, ll)
            )
            passthrough_axes.append(i)
        else:
            out_vd = axis_count
            new_out_dims.append(DenseIndex(len(new_out_dims), out_shape[i], out_vd))
            axis_count += 1
            primal_vd = axis_count
            new_primal_dims.append(DenseIndex(out_ndim + i, x_shape[i], primal_vd))
            axis_count += 1
            windowed_axes.append((i, out_shape[i], x_shape[i], wd, ws, pad_lo, bd, wid))

    # Build the indicator value: x == broadcast(val_out)
    # Reshape val_out to broadcast against x (insert size-1 dims for windowed axes)
    broadcast_shape = list(x_shape)
    for _, out_s, in_s, wd, ws, pad_lo, bd, wid in windowed_axes:
        # We need to expand val_out spatially to compare against x.
        pass

    # For simplicity, compute the full indicator and slice out the windowed axes.
    # This is analogous to the reduce_max rule (reductions.py).
    _out_shape = list(out_shape)
    for i in range(x_ndim):
        if x_shape[i] != out_shape[i]:
            _out_shape[i] = 1
    _val_out = val_out.reshape(_out_shape)
    indicator = (x == _val_out).astype(val_out.dtype)
    # Normalize for non-unique extrema
    norm_shape = list(x_shape)
    for i in range(x_ndim):
        if x_shape[i] == out_shape[i]:
            norm_shape[i] = 1
    norm = indicator
    reduce_axes = tuple(i for i in range(x_ndim) if x_shape[i] != out_shape[i])
    if reduce_axes:
        norm_sum = jnp.sum(indicator, axis=reduce_axes, keepdims=True)
        indicator = indicator / jnp.maximum(norm_sum, 1.0)

    x_tensor = _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, indicator))

    tensors = [x_tensor]
    if not isinstance(init_value, (float, int, complex)):
        init_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
        init_primal_dims = [
            DenseIndex(i + len(out_shape), s, i + len(out_shape))
            for i, s in enumerate(init_value.shape)
        ]
        init_tensor = SparseTensor(
            init_out_dims,
            init_primal_dims,
            jnp.zeros(out_shape + init_value.shape, dtype=val_out.dtype),
        )
        tensors.append(init_tensor)

    return val_out, tensors


elemental_rules[lax.reduce_window_p] = reduce_window_elemental_rule


# Max/min pooling (reduce_window with a max/min reducer) is not yet supported —
# fail loudly rather than crash with a generic "no elemental partial" message.
# Only reduce_window with the SUM reducer (average pooling) is implemented.
def _reduce_window_extremum_unsupported(primals, **params):
    raise NotImplementedError(
        "graphax does not yet support max/min pooling (reduce_window with a "
        "max/min reducer); only the sum reducer (average pooling) is implemented."
    )


elemental_rules[lax.reduce_window_max_p] = _reduce_window_extremum_unsupported
elemental_rules[lax.reduce_window_min_p] = _reduce_window_extremum_unsupported

### Transforms

Transform = Callable[[SparseTensor, SparseTensor, jnp.ndarray], SparseTensor]


class JacobianTransform:
    transform: Transform
    inverse_transform: Transform

    def __init__(
        self, transform: Transform, inverse_transform: Transform = None
    ) -> None:
        self.transform = transform
        self.inverse_transform = inverse_transform

    def __repr__(self) -> str:
        return (
            f"JacobianTransform(transform={self.transform}, "
            f"inverse_transform={self.inverse_transform})"
        )

    def apply(self, tensor: SparseTensor) -> SparseTensor:
        if self.transform is None:
            raise NotImplementedError("Transform not implemented!")
        return self.transform(tensor)

    def apply_inverse(self, tensor: SparseTensor) -> SparseTensor:
        if self.inverse_transform is None:
            raise NotImplementedError("Inverse transform not implemented!")
        return self.inverse_transform(tensor)


def make_slice_transform(start_indices, limit_indices, out_shape):
    def slice_transform(pre):
        s_idx = list(start_indices)
        l_idx = list(limit_indices)
        full_val = jnp.array(pre)
        new_out_dims = []
        new_primal_dims = []
        counter = 0

        for s in out_shape:
            new_out_dims.append(DenseIndex(counter, s, counter))
            counter += 1

        for d in pre.primal_dims:
            new_primal_dims.append(DenseIndex(counter, d.size, counter))
            s_idx.append(0)
            l_idx.append(d.size)
            counter += 1

        new_val = lax.slice(full_val, s_idx, l_idx)
        return SparseTensor(new_out_dims, new_primal_dims, new_val)

    return slice_transform


def make_inverse_slice_transform(start_indices, limit_indices, primal0_shape):
    def inverse_slice_transform(post):
        full_val = jnp.array(post)
        new_out_dims = []
        new_primal_dims = []
        counter = 0

        for d in post.out_dims:
            new_out_dims.append(DenseIndex(counter, d.size, counter))
            counter += 1

        for s in primal0_shape:
            new_primal_dims.append(DenseIndex(counter, s, counter))
            counter += 1

        new_shape = [d.size for d in new_out_dims] + [d.size for d in new_primal_dims]
        zeros = jnp.zeros(new_shape, dtype=full_val.dtype)
        start_indices_full = [0] * len(post.out_dims) + list(start_indices)
        new_val = lax.dynamic_update_slice(zeros, full_val, start_indices_full)

        return SparseTensor(new_out_dims, new_primal_dims, new_val)

    return inverse_slice_transform


def dynamic_slice_elemental_rule(primals, **params):
    val_out = lax.dynamic_slice_p.bind(*primals, **params)
    operand = primals[0]
    start_indices = primals[1:]
    slice_sizes = params["slice_sizes"]

    start_list = [int(s) for s in start_indices]
    limit_list = [s + sz for s, sz in zip(start_list, slice_sizes)]

    transform = JacobianTransform(
        make_slice_transform(start_list, limit_list, val_out.shape),
        make_inverse_slice_transform(start_list, limit_list, operand.shape),
    )
    return val_out, [SparseTensor([], [], None, pre_transforms=[transform])]


elemental_rules[lax.dynamic_slice_p] = dynamic_slice_elemental_rule


def _build_dus_update_jac(start_list, out_shape, up_shape):
    """``d out/d update`` for dynamic_update_slice: the update is embedded into
    the output window, so ``J[out_idx, up_idx] = 1`` iff
    ``out_idx == start + up_idx`` (within the window). Built as the outer product
    of per-axis shifted-identity indicators ``E_d[i, j] = (i == start_d + j)`` via
    one einsum (vectorized, scatter-free). Shape: ``out_shape + up_shape``."""
    n = len(up_shape)
    if n == 0:
        return jnp.array(1.0, dtype=jnp.float32)
    letters = "abcdefghijklmnopqrstuvwxyz"
    Es, subs, o_letters, u_letters = [], [], [], []
    for d in range(n):
        i = jnp.arange(out_shape[d])[:, None]
        j = jnp.arange(up_shape[d])[None, :]
        Es.append((i == start_list[d] + j).astype(jnp.float32))
        o, u = letters[2 * d], letters[2 * d + 1]
        subs.append(o + u); o_letters.append(o); u_letters.append(u)
    eq = ",".join(subs) + "->" + "".join(o_letters) + "".join(u_letters)
    return jnp.einsum(eq, *Es)


def dynamic_update_slice_elemental_rule(primals, **params):
    val_out = lax.dynamic_update_slice_p.bind(*primals, **params)
    operand = primals[0]
    update = primals[1]
    start_indices = primals[2:]

    op_shape = get_shape(operand)
    up_shape = get_shape(update)
    out_shape = get_shape(val_out)
    ndim = get_ndim(val_out)

    start_list = [int(s) for s in start_indices]

    # Jacobian for operand: identity except in updated region (where it's 0)
    mask = jnp.ones(op_shape, dtype=jnp.float32)
    slices_obj = tuple(slice(s, s + sz) for s, sz in zip(start_list, up_shape))
    mask = mask.at[slices_obj].set(0.0)
    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, mask)

    # Jacobian for update: the update is embedded into the output window, so
    # d out/d update is an (out_shape x up_shape) embedding — NOT an operand-
    # shaped identity. The old slice-transform built it in the operand direction
    # (wrong shape/values for the update arg).
    jac = _build_dus_update_jac(start_list, out_shape, up_shape)
    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    return val_out, [op_tensor, up_tensor]


elemental_rules[lax.dynamic_update_slice_p] = dynamic_update_slice_elemental_rule


def gather_elemental_rule(primals, **params):
    val_out = lax.gather_p.bind(*primals, **params)
    operand = primals[0]
    indices = primals[1]

    out_shape = get_shape(val_out)
    op_shape = get_shape(operand)
    out_ndim = len(out_shape)
    op_ndim = len(op_shape)

    dn = params["dimension_numbers"]
    slice_sizes = params["slice_sizes"]
    offset_dims = dn.offset_dims
    collapsed_slice_dims = dn.collapsed_slice_dims
    start_index_map = dn.start_index_map

    batch_dims = [d for d in range(out_ndim) if d not in offset_dims]
    non_collapsed = [d for d in range(op_ndim) if d not in collapsed_slice_dims]

    for nc_dim in non_collapsed:
        if slice_sizes[nc_dim] != op_shape[nc_dim]:
            raise NotImplementedError(
                f"gather: partial slice along non-collapsed dim {nc_dim} not supported"
            )

    out_dims = []
    primal_dims = []
    axis_count = 0

    batch_axiss = {}
    for bd in batch_dims:
        batch_axiss[bd] = axis_count
        axis_count += 1

    collapsed_axiss = {}
    for cd in collapsed_slice_dims:
        collapsed_axiss[cd] = axis_count
        axis_count += 1

    for i in range(out_ndim):
        if i in offset_dims:
            offset_pos = list(offset_dims).index(i)
            paired_op = non_collapsed[offset_pos]
            out_dims.append(
                DiagonalIndex(i, out_shape[i], None, out_ndim + paired_op)
            )
        else:
            out_dims.append(DenseIndex(i, out_shape[i], batch_axiss[i]))

    for j in range(op_ndim):
        if j in collapsed_slice_dims:
            primal_dims.append(
                DenseIndex(out_ndim + j, op_shape[j], collapsed_axiss[j])
            )
        else:
            nc_pos = non_collapsed.index(j)
            paired_out = offset_dims[nc_pos]
            primal_dims.append(
                DiagonalIndex(out_ndim + j, op_shape[j], None, paired_out)
            )

    val_shape = [out_shape[bd] for bd in batch_dims] + [
        op_shape[cd] for cd in collapsed_slice_dims
    ]

    if len(val_shape) == 0:
        val = jnp.array(1.0, dtype=jnp.float32)
    else:
        val = jnp.zeros(val_shape, dtype=jnp.float32)
        import itertools

        batch_sizes = [out_shape[bd] for bd in batch_dims]
        for batch_idx in itertools.product(*(range(s) for s in batch_sizes)):
            idx_val = indices[batch_idx]
            if hasattr(idx_val, "ndim") and idx_val.ndim == 0:
                idx_val = idx_val.reshape(1)

            op_pos = []
            for cd in collapsed_slice_dims:
                if cd in start_index_map:
                    k = list(start_index_map).index(cd)
                    pos = (
                        int(idx_val[k])
                        if hasattr(idx_val, "__getitem__")
                        else int(idx_val)
                    )
                else:
                    pos = 0
                op_pos.append(pos)

            full_idx = tuple(batch_idx) + tuple(op_pos)
            val = val.at[full_idx].set(1.0)

    tensor = _swap_back_axes(SparseTensor(out_dims, primal_dims, val))
    return val_out, [tensor]


elemental_rules[lax.gather_p] = gather_elemental_rule


def _update_to_output_index(up_idx, indices, dn, operand_ndim, up_ndim):
    """Map an update-array multi-index to the operand position it writes to.

    XLA scatter places each window at ``start + window_offset``: the start comes
    from the scatter index vector (selected by the update's non-window dims) and
    the offset from the update's window dims. The previous version set
    out_idx = start OR offset (never both) and skipped the start entirely when
    every update dim was a window dim (a slice scatter, e.g. ``x.at[1:2].max(u)``),
    so the whole Jacobian was mis-placed at offset 0 instead of the slice start."""
    update_window_dims = dn.update_window_dims
    inserted_window_dims = dn.inserted_window_dims
    scatter_dims_to_operand_dims = dn.scatter_dims_to_operand_dims

    # The update's non-window dims select WHICH window (its index vector).
    scatter_dims = [d for d in range(up_ndim) if d not in update_window_dims]
    scatter_idx = tuple(up_idx[d] for d in scatter_dims)
    idx_vec = jnp.reshape(indices[scatter_idx], (-1,))

    out_idx = [0] * operand_ndim
    # Base: scatter start for each operand dim named by scatter_dims_to_operand_dims.
    for k, op_dim in enumerate(scatter_dims_to_operand_dims):
        out_idx[op_dim] = int(idx_vec[k])
    # Offset: add each window dim's position within the window.
    non_inserted = [d for d in range(operand_ndim) if d not in inserted_window_dims]
    for w_i, w_dim in enumerate(update_window_dims):
        out_idx[non_inserted[w_i]] += int(up_idx[w_dim])

    return tuple(out_idx)


def _build_scatter_update_jac(indices, out_shape, up_shape, params, coeff=None):
    ndim_out = len(out_shape)
    ndim_up = len(up_shape)
    dn = params["dimension_numbers"]
    jac_shape = list(out_shape) + list(up_shape)
    jac = jnp.zeros(jac_shape, dtype=jnp.float32)

    import itertools

    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(up_idx, indices, dn, ndim_out, ndim_up)
        full_idx = tuple(out_idx) + tuple(up_idx)
        val = (
            1.0
            if coeff is None
            else float(coeff[up_idx])
            if hasattr(coeff, "__getitem__")
            else float(coeff)
        )
        jac = jac.at[full_idx].set(val)

    return jac


def _build_scatter_mask(indices, out_shape, up_shape, params):
    ndim_out = len(out_shape)
    ndim_up = len(up_shape)
    dn = params["dimension_numbers"]
    mask = jnp.zeros(out_shape, dtype=jnp.float32)

    import itertools

    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(up_idx, indices, dn, ndim_out, ndim_up)
        mask = mask.at[out_idx].set(1.0)

    return mask


def scatter_add_elemental_rule(primals, **params):
    val_out = lax.scatter_add_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]
    out_shape, op_shape, up_shape = (
        get_shape(val_out),
        get_shape(operand),
        get_shape(updates),
    )
    ndim = get_ndim(val_out)

    # d/d(operand) = identity
    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(
        op_out_dims, op_primal_dims, jnp.ones(out_shape, dtype=jnp.float32)
    )

    # d/d(updates) = one-hot embedding (coeff = 1)
    jac = _build_scatter_update_jac(indices, out_shape, up_shape, params)
    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    # invars are (operand=0, indices=1, updates=2); core indexes elementals by
    # eqn.invars position, so the updates Jacobian must sit at slot 2 with a None
    # for the non-differentiable integer indices (else it is silently dropped).
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_add_p] = scatter_add_elemental_rule


def scatter_sub_elemental_rule(primals, **params):
    val_out = lax.scatter_sub_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]
    out_shape, op_shape, up_shape = (
        get_shape(val_out),
        get_shape(operand),
        get_shape(updates),
    )
    ndim = get_ndim(val_out)

    # d/d(operand) = identity
    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(
        op_out_dims, op_primal_dims, jnp.ones(out_shape, dtype=jnp.float32)
    )

    # d/d(updates) = -1 × one-hot embedding
    jac = _build_scatter_update_jac(indices, out_shape, up_shape, params, coeff=-1.0)
    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    # invars are (operand=0, indices=1, updates=2); core indexes elementals by
    # eqn.invars position, so the updates Jacobian must sit at slot 2 with a None
    # for the non-differentiable integer indices (else it is silently dropped).
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_sub_p] = scatter_sub_elemental_rule


def scatter_set_elemental_rule(primals, **params):
    val_out = lax.scatter_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]
    out_shape, op_shape, up_shape = (
        get_shape(val_out),
        get_shape(operand),
        get_shape(updates),
    )
    ndim = get_ndim(val_out)

    # d/d(operand) = identity * (1 - mask) — zeroed at overwritten positions
    mask = _build_scatter_mask(indices, out_shape, up_shape, params)
    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, 1.0 - mask)

    # d/d(updates) = one-hot embedding (coeff = 1)
    jac = _build_scatter_update_jac(indices, out_shape, up_shape, params)
    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    # invars are (operand=0, indices=1, updates=2); core indexes elementals by
    # eqn.invars position, so the updates Jacobian must sit at slot 2 with a None
    # for the non-differentiable integer indices (else it is silently dropped).
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_p] = scatter_set_elemental_rule


def scatter_mul_elemental_rule(primals, **params):
    val_out = lax.scatter_mul_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]
    out_shape, op_shape, up_shape = (
        get_shape(val_out),
        get_shape(operand),
        get_shape(updates),
    )
    ndim = get_ndim(val_out)
    dn = params["dimension_numbers"]

    # d/d(operand): identity at non-scattered; updates value at scattered
    # Build per-element operand coefficient
    op_coeff = jnp.ones(out_shape, dtype=jnp.float32)
    import itertools

    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        op_coeff = op_coeff.at[out_idx].set(float(updates[up_idx]))

    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, op_coeff)

    # d/d(updates) = operand value at the scattered output positions (product
    # rule: out = operand * updates there, so d out/d updates = operand).
    jac_shape = list(out_shape) + list(up_shape)
    jac2 = jnp.zeros(jac_shape, dtype=jnp.float32)
    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        full_idx = tuple(out_idx) + tuple(up_idx)
        jac2 = jac2.at[full_idx].set(float(operand[out_idx]))

    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac2)

    # See scatter_add: updates Jacobian at slot 2, None for the integer indices.
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_mul_p] = scatter_mul_elemental_rule


def _min_tie_split(a, b):
    """Subgradient weight for ``a`` in ``min(a, b)``: 1 if a<b, 0.5 if a==b
    (balanced, matching JAX), 0 if a>b."""
    return 1.0 if a < b else (0.5 if a == b else 0.0)


def _max_tie_split(a, b):
    """Subgradient weight for ``a`` in ``max(a, b)``: 1 if a>b, 0.5 if a==b, 0."""
    return 1.0 if a > b else (0.5 if a == b else 0.0)


def scatter_min_elemental_rule(primals, **params):
    val_out = lax.scatter_min_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]
    out_shape, op_shape, up_shape = (
        get_shape(val_out),
        get_shape(operand),
        get_shape(updates),
    )
    ndim = get_ndim(val_out)
    dn = params["dimension_numbers"]

    # d/d(operand): 1 where operand <= updates at scattered pos, 1 elsewhere
    op_coeff = jnp.ones(out_shape, dtype=jnp.float32)
    import itertools

    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        # Balanced subgradient at a tie (operand == update): 0.5 to each side,
        # matching JAX. Was operand<=update -> 1.0 (operand took all the credit).
        op_coeff = op_coeff.at[out_idx].set(
            _min_tie_split(operand[out_idx], updates[up_idx])
        )

    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, op_coeff)

    # d/d(updates): 1 where updates < operand, 0.5 at a tie
    jac_shape = list(out_shape) + list(up_shape)
    jac = jnp.zeros(jac_shape, dtype=jnp.float32)
    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        indicator = _min_tie_split(updates[up_idx], operand[out_idx])
        full_idx = tuple(out_idx) + tuple(up_idx)
        jac = jac.at[full_idx].set(indicator)

    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    # invars are (operand=0, indices=1, updates=2); core indexes elementals by
    # eqn.invars position, so the updates Jacobian must sit at slot 2 with a None
    # for the non-differentiable integer indices (else it is silently dropped).
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_min_p] = scatter_min_elemental_rule


def scatter_max_elemental_rule(primals, **params):
    val_out = lax.scatter_max_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]
    out_shape, op_shape, up_shape = (
        get_shape(val_out),
        get_shape(operand),
        get_shape(updates),
    )
    ndim = get_ndim(val_out)
    dn = params["dimension_numbers"]

    # d/d(operand): 1 where operand >= updates at scattered pos, 1 elsewhere
    op_coeff = jnp.ones(out_shape, dtype=jnp.float32)
    import itertools

    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        # Balanced subgradient at a tie (operand == update): 0.5 each (was 1.0
        # operand / 0.0 update).
        op_coeff = op_coeff.at[out_idx].set(
            _max_tie_split(operand[out_idx], updates[up_idx])
        )

    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, op_coeff)

    # d/d(updates): 1 where updates > operand, 0.5 at a tie
    jac_shape = list(out_shape) + list(up_shape)
    jac = jnp.zeros(jac_shape, dtype=jnp.float32)
    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        indicator = _max_tie_split(updates[up_idx], operand[out_idx])
        full_idx = tuple(out_idx) + tuple(up_idx)
        jac = jac.at[full_idx].set(indicator)

    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    # invars are (operand=0, indices=1, updates=2); core indexes elementals by
    # eqn.invars position, so the updates Jacobian must sit at slot 2 with a None
    # for the non-differentiable integer indices (else it is silently dropped).
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_max_p] = scatter_max_elemental_rule


def linear_solve_elemental_rule(primals, **params):
    val_out = lax.linear_solve_p.bind(*primals, **params)
    A, b = primals
    x = val_out

    A_shape = list(get_shape(A))
    b_shape = list(get_shape(b))
    out_shape = list(get_shape(val_out))

    N = A_shape[-1]
    batch_dims = A_shape[:-2]
    num_batch = len(batch_dims)
    num_out_dims = len(out_shape)

    # 1. Jacobian w.r.t b: J_b = A^{-1}
    I = jnp.eye(N, dtype=val_out.dtype)
    I_broadcast = jnp.broadcast_to(I, batch_dims + [N, N])
    # Compute batched A^{-1}
    A_inv = lax.linear_solve_p.bind(A, I_broadcast, **params)

    b_out_dims = []
    b_primal_dims = []

    # Batch dims map strictly 1-to-1 (SparseIndexes)
    for i, s in enumerate(batch_dims):
        b_out_dims.append(DiagonalIndex(i, s, None, num_out_dims + i))
        b_primal_dims.append(DiagonalIndex(num_out_dims + i, s, None, i))

    # Matrix dims are dense within the block
    b_out_dims.append(DenseIndex(num_batch, N, 0))
    b_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 1))

    b_tensor = _swap_back_axes(SparseTensor(b_out_dims, b_primal_dims, A_inv))

    # 2. Jacobian w.r.t A: J_A = - A^{-1} \otimes x
    # We reshape to (*batch, N, N, N) to broadcast the outer product cleanly
    A_inv_exp = jnp.expand_dims(A_inv, -1)
    x_reshaped = jnp.reshape(x, batch_dims + [1, 1, N])
    J_A_val = -(A_inv_exp * x_reshaped)

    A_out_dims = []
    A_primal_dims = []

    # Batch dims map strictly 1-to-1 (SparseIndexes)
    for i, s in enumerate(batch_dims):
        A_out_dims.append(DiagonalIndex(i, s, None, num_out_dims + i))
        A_primal_dims.append(DiagonalIndex(num_out_dims + i, s, None, i))

    # The resulting tensor has one dense output dim and two dense primal dims
    A_out_dims.append(DenseIndex(num_batch, N, 0))
    A_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 1))
    A_primal_dims.append(DenseIndex(num_out_dims + num_batch + 1, N, 2))

    A_tensor = _swap_back_axes(SparseTensor(A_out_dims, A_primal_dims, J_A_val))

    return val_out, [A_tensor, b_tensor]


elemental_rules[lax.linear_solve_p] = linear_solve_elemental_rule


# top_k: returns (values, indices); values' Jacobian wrt x is a one-hot selection
# matrix, indices are integer (no gradient). multiple_results -> handled by the
# multi-output rule below.
def top_k_elemental_only(primal_outs, primals, **params):
    """Multi-output elemental rule for ``top_k`` (returns values + indices).

    ``top_k`` is ``multiple_results`` so it must go through
    ``multi_output_elemental_only_rules``: the return is
    ``elementals[outvar_idx][invar_idx]``. The single input ``x`` (invar 0) flows
    to the VALUES output via the one-hot selection Jacobian; the integer INDICES
    output carries no gradient (None). (The cache-key fix in core.py makes this
    value-dependent indicator safe across repeated jacve calls.)"""
    x = primals[0]
    values, indices = primal_outs
    k = params["k"]
    x_shape = get_shape(x)
    val_ndim = len(get_shape(values))
    x_ndim = len(x_shape)
    n = x_shape[-1]

    out_dims, primal_dims = [], []
    axis_count = 0
    for i, s in enumerate(x_shape[:-1]):  # batch dims: diagonal
        out_dims.append(DiagonalIndex(i, s, axis_count, val_ndim + i))
        primal_dims.append(DiagonalIndex(val_ndim + i, s, axis_count, i))
        axis_count += 1
    out_dims.append(DenseIndex(val_ndim - 1, k, axis_count)); axis_count += 1
    primal_dims.append(DenseIndex(val_ndim + x_ndim - 1, n, axis_count))
    indicator = (indices[..., :, None] == jnp.arange(n)[None, :]).astype(x.dtype)
    tensor = _swap_back_axes(SparseTensor(out_dims, primal_dims, indicator))
    # outputs = [values, indices]; one invar (x). indices -> no edge (None).
    return [[tensor], [None]]


multi_output_elemental_only_rules[lax.top_k_p] = top_k_elemental_only


# argmax: returns integer indices, derivative is zero everywhere
def argmax_elemental_rule(primals, **params):
    val_out = lax.argmax_p.bind(*primals, **params)
    return val_out, []


elemental_rules[lax.argmax_p] = argmax_elemental_rule


# cond_p / switch and jit_p are handled by core (they need vertex_elimination_
# jaxpr): cond differentiates the taken branch (core.cond_elemental_rule), jit is
# inlined or dispatched by name (core._make_jit_elemental_rule). Not registered
# here.


# Collective operations: psum, all_gather, reduce_scatter, all_to_all
# These operate across devices. Locally, the Jacobian is identity-like.
from jax._src.lax.parallel import all_gather_p, all_to_all_p, psum_p, reduce_scatter_p


# psum: sum across devices. Locally, d(psum(x))/dx = I (identity).
def psum_elemental_rule(primals, **params):
    val_out = psum_p.bind(*primals, **params)
    x = primals[0]
    x_shape = get_shape(x)
    x_ndim = len(x_shape)
    out_shape = get_shape(val_out)
    out_ndim = len(out_shape)

    out_dims = [
        DiagonalIndex(i, s, None, out_ndim + i) for i, s in enumerate(out_shape)
    ]
    primal_dims = [
        DiagonalIndex(out_ndim + i, s, None, i) for i, s in enumerate(x_shape)
    ]

    tensor = SparseTensor(out_dims, primal_dims, jnp.array(1.0, dtype=jnp.float32))
    return val_out, [tensor]


elemental_rules[psum_p] = psum_elemental_rule


# all_gather: gathers shards from all devices along an axis.
# Locally, the shard maps to a slice of the output => identity transform.
def all_gather_elemental_rule(primals, **params):
    val_out = all_gather_p.bind(*primals, **params)
    x = primals[0]
    x_shape = get_shape(x)
    x_ndim = len(x_shape)
    out_shape = get_shape(val_out)
    out_ndim = len(out_shape)

    out_dims = []
    primal_dims = []
    for i in range(out_ndim):
        if i < x_ndim and x_shape[i] == out_shape[i]:
            out_dims.append(DiagonalIndex(i, out_shape[i], None, out_ndim + i))
            primal_dims.append(DiagonalIndex(out_ndim + i, x_shape[i], None, i))
        elif i < x_ndim:
            out_dims.append(
                DenseIndex(
                    i,
                    out_shape[i],
                    len([d for d in out_dims if not d.is_sparse]),
                )
            )
            primal_dims.append(
                DenseIndex(
                    out_ndim + i,
                    x_shape[i],
                    len(
                        [
                            d
                            for d in out_dims + primal_dims
                            if not d.is_sparse
                        ]
                    ),
                )
            )
        else:
            out_dims.append(
                DenseIndex(
                    i,
                    out_shape[i],
                    len([d for d in out_dims if not d.is_sparse]),
                )
            )

    tensor = SparseTensor(out_dims, primal_dims, jnp.array(1.0, dtype=jnp.float32))
    return val_out, [tensor]


elemental_rules[all_gather_p] = all_gather_elemental_rule


# reduce_scatter: reduce (sum) then scatter across devices.
# Locally, d(reduce_scatter(x))/dx = I (identity).
def reduce_scatter_elemental_rule(primals, **params):
    val_out = reduce_scatter_p.bind(*primals, **params)
    x = primals[0]
    x_shape = get_shape(x)
    x_ndim = len(x_shape)
    out_shape = get_shape(val_out)
    out_ndim = len(out_shape)

    out_dims = [
        DiagonalIndex(i, s, None, out_ndim + i) for i, s in enumerate(out_shape)
    ]
    primal_dims = [
        DiagonalIndex(out_ndim + i, s, None, i) for i, s in enumerate(x_shape)
    ]

    tensor = SparseTensor(out_dims, primal_dims, jnp.array(1.0, dtype=jnp.float32))
    return val_out, [tensor]


elemental_rules[reduce_scatter_p] = reduce_scatter_elemental_rule


# all_to_all: transposes data across devices (each device sends a chunk to each other device).
# Locally, this is a reshape/permutation => identity transform.
def all_to_all_elemental_rule(primals, **params):
    val_out = all_to_all_p.bind(*primals, **params)
    x = primals[0]
    x_shape = get_shape(x)
    x_ndim = len(x_shape)
    out_shape = get_shape(val_out)
    out_ndim = len(out_shape)

    out_dims = [
        DiagonalIndex(i, s, None, out_ndim + i) for i, s in enumerate(out_shape)
    ]
    primal_dims = [
        DiagonalIndex(out_ndim + i, s, None, i) for i, s in enumerate(x_shape)
    ]

    tensor = SparseTensor(out_dims, primal_dims, jnp.array(1.0, dtype=jnp.float32))
    return val_out, [tensor]


elemental_rules[all_to_all_p] = all_to_all_elemental_rule


# ============================================================================
# Matrix decompositions and linear solvers
# ============================================================================
import jax._src.lax.linalg as lax_linalg


# triangular_solve: (A, b) -> x where Ax = b (with A triangular)
def triangular_solve_elemental_rule(primals, **params):
    val_out = lax_linalg.triangular_solve_p.bind(*primals, **params)
    A, b = primals
    x = val_out

    A_shape = list(get_shape(A))
    out_shape = list(get_shape(val_out))
    N = A_shape[-1]
    batch_dims = A_shape[:-2]
    num_batch = len(batch_dims)
    num_out_dims = len(out_shape)
    nrhs = out_shape[-1] if len(out_shape) > num_batch + 1 else 1

    # J_b = A^{-1}
    I = jnp.eye(N, dtype=val_out.dtype)
    I_broadcast = jnp.broadcast_to(I, batch_dims + [N, N])
    A_inv = lax_linalg.triangular_solve_p.bind(A, I_broadcast, **params)

    b_out_dims = []
    b_primal_dims = []
    for i, s in enumerate(batch_dims):
        b_out_dims.append(DiagonalIndex(i, s, None, num_out_dims + i))
        b_primal_dims.append(DiagonalIndex(num_out_dims + i, s, None, i))

    if len(out_shape) == num_batch + 2:
        b_out_dims.append(DenseIndex(num_batch, N, 0))
        b_out_dims.append(
            DiagonalIndex(num_batch + 1, nrhs, None, num_out_dims + num_batch + 1)
        )
        b_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 1))
        b_primal_dims.append(
            DiagonalIndex(num_out_dims + num_batch + 1, nrhs, None, num_batch + 1)
        )
    else:
        b_out_dims.append(DenseIndex(num_batch, N, 0))
        b_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 1))

    b_tensor = _swap_back_axes(SparseTensor(b_out_dims, b_primal_dims, A_inv))

    # J_A = -A^{-1} ⊗ x
    A_inv_exp = jnp.expand_dims(A_inv, -1)
    x_for_outer = x if x.ndim > num_batch + 1 else x[..., None]
    x_reshaped = jnp.reshape(x_for_outer, batch_dims + [1, 1, N, nrhs])
    # J_A_val shape: (N, nrhs, N, N) for d(X)/dA
    # A_inv_exp is (N, N, 1), x_reshaped is (1, 1, N, nrhs)
    # Product is (N, N, N, nrhs). We need (N, nrhs, N, N).
    J_A_val = -(A_inv_exp[..., None] * x_reshaped)
    J_A_val = jnp.transpose(
        J_A_val,
        (*range(num_batch), num_batch, num_batch + 3, num_batch + 1, num_batch + 2),
    )

    # Masking for triangular_solve: zero out derivatives w.r.t. unused elements of A
    lower = params.get("lower", False)
    mask = jnp.tri(N, k=0, dtype=bool) if lower else jnp.tri(N, k=0, dtype=bool).T
    mask = jnp.reshape(mask, [1] * num_batch + [1, 1, N, N])
    J_A_val = jnp.where(mask, J_A_val, 0)

    if nrhs == 1 and len(out_shape) == num_batch + 1:
        J_A_val = jnp.squeeze(J_A_val, axis=num_batch + 1)

    A_out_dims = []
    A_primal_dims = []
    for i, s in enumerate(batch_dims):
        A_out_dims.append(DiagonalIndex(i, s, None, num_out_dims + i))
        A_primal_dims.append(DiagonalIndex(num_out_dims + i, s, None, i))

    if len(out_shape) == num_batch + 2:
        A_out_dims.append(DenseIndex(num_batch, N, 0))
        A_out_dims.append(DenseIndex(num_batch + 1, nrhs, 1))
        A_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 2))
        A_primal_dims.append(DenseIndex(num_out_dims + num_batch + 1, N, 3))
    else:
        A_out_dims.append(DenseIndex(num_batch, N, 0))
        A_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 1))
        A_primal_dims.append(DenseIndex(num_out_dims + num_batch + 1, N, 2))

    A_tensor = _swap_back_axes(SparseTensor(A_out_dims, A_primal_dims, J_A_val))
    return val_out, [A_tensor, b_tensor]


elemental_rules[lax_linalg.triangular_solve_p] = triangular_solve_elemental_rule


# cholesky: A -> L
# Pure analytical: J_{ijkl} = sum_m L_{im} M_{mj} B_{mk} B_{jl}
def cholesky_elemental_rule(primals, **params):
    val_out = lax_linalg.cholesky_p.bind(*primals, **params)
    A = primals[0]
    L = val_out

    A_shape = list(get_shape(A))
    out_shape = list(get_shape(L))
    N = A_shape[-1]
    batch_dims = A_shape[:-2]
    num_batch = len(batch_dims)
    num_out_dims = len(out_shape)

    # Use actual masked lower triangle for the derivative calculation
    L_lower = jnp.tril(L)
    I = jnp.eye(N, dtype=A.dtype)
    I_bcast = jnp.broadcast_to(I, batch_dims + [N, N])

    # B = L^{-1}
    B = lax_linalg.triangular_solve_p.bind(
        L_lower,
        I_bcast,
        left_side=True,
        lower=True,
        transpose_a=False,
        conjugate_a=False,
        unit_diagonal=False,
    )

    # M_{mj} = 1/(1 + d_{mj}) for m >= j
    M = jnp.tril(jnp.ones((N, N), dtype=A.dtype)) / (1.0 + I)

    # J = L @ M @ (B x B)
    J = jnp.einsum("...im,mj,...mk,...jl->...ijkl", L_lower, M, B, B)

    out_dims = []
    primal_dims = []
    vd = 0
    for i, s in enumerate(batch_dims):
        out_dims.append(DiagonalIndex(i, s, vd, num_out_dims + i))
        primal_dims.append(DiagonalIndex(num_out_dims + i, s, vd, i))
        vd += 1
    out_dims.append(DenseIndex(num_batch, N, vd))
    vd += 1
    out_dims.append(DenseIndex(num_batch + 1, N, vd))
    vd += 1
    primal_dims.append(DenseIndex(num_out_dims + num_batch, N, vd))
    vd += 1
    primal_dims.append(DenseIndex(num_out_dims + num_batch + 1, N, vd))
    vd += 1

    tensor = _swap_back_axes(SparseTensor(out_dims, primal_dims, J))
    return val_out, [tensor]


elemental_rules[lax_linalg.cholesky_p] = cholesky_elemental_rule


# eigh: A -> (V, w) (Note: standard eigh_p returns (V, w) internally)
def eigh_elemental_rule(primals, **params):
    val_out = lax_linalg.eigh_p.bind(*primals, **params)
    A = primals[0]
    V, w = val_out  # Actual internal eigh_p returns (V, w)

    A_shape = list(get_shape(A))
    N = A_shape[-1]
    batch_dims = A_shape[:-2]
    num_batch = len(batch_dims)

    w_shape = list(get_shape(w))
    V_shape = list(get_shape(V))
    w_ndim = len(w_shape)
    V_ndim = len(V_shape)

    # dw = diag(V^T dA V) -> J_w = einsum('ik,jk->kij', V, V)
    J_w = jnp.einsum("...ik,...jk->...kij", V, V)
    # The input is symmetric, so derivative is symmetric in inputs
    J_w = 0.5 * (J_w + jnp.swapaxes(J_w, -2, -1))

    w_out_dims = []
    w_primal_dims = []
    vd = 0
    for i, s in enumerate(batch_dims):
        w_out_dims.append(DiagonalIndex(i, s, vd, w_ndim + i))
        w_primal_dims.append(DiagonalIndex(w_ndim + i, s, vd, i))
        vd += 1
    w_out_dims.append(DenseIndex(num_batch, N, vd))
    vd += 1
    w_primal_dims.append(DenseIndex(w_ndim + num_batch, N, vd))
    vd += 1
    w_primal_dims.append(DenseIndex(w_ndim + num_batch + 1, N, vd))
    vd += 1
    w_tensor = _swap_back_axes(SparseTensor(w_out_dims, w_primal_dims, J_w))

    # dV = V (F ⊙ (V^T dA V)) -> J_V = einsum('kq,qp,ip,jq->kpij', V, F, V, V)
    eye_n = jnp.eye(N, dtype=A.dtype)
    w_diff = w[..., None, :] - w[..., :, None]
    F = jnp.where(eye_n == 1, 0.0, 1.0 / jnp.where(w_diff == 0, 1.0, w_diff))

    J_V = jnp.einsum("...kq,...qp,...ip,...jq->...kpij", V, F, V, V)
    # Symmetric in inputs
    J_V = 0.5 * (J_V + jnp.swapaxes(J_V, -2, -1))

    V_out_dims = []
    V_primal_dims = []
    vd = 0
    for i, s in enumerate(batch_dims):
        V_out_dims.append(DiagonalIndex(i, s, vd, V_ndim + i))
        V_primal_dims.append(DiagonalIndex(V_ndim + i, s, vd, i))
        vd += 1
    V_out_dims.append(DenseIndex(num_batch, N, vd))
    vd += 1
    V_out_dims.append(DenseIndex(num_batch + 1, N, vd))
    vd += 1
    V_primal_dims.append(DenseIndex(V_ndim + num_batch, N, vd))
    vd += 1
    V_primal_dims.append(DenseIndex(V_ndim + num_batch + 1, N, vd))
    vd += 1
    V_tensor = _swap_back_axes(SparseTensor(V_out_dims, V_primal_dims, J_V))

    # Actual eigh_p returns (V, w), not (w, V)
    return val_out, [[V_tensor], [w_tensor]]


def _multi_output_only(rule):
    """Adapt a ``(primals, **params) -> (val_out, elementals[out][invar])`` rule
    (the matrix-decomposition rules below) to the
    ``multi_output_elemental_only_rules`` contract. These primitives are
    ``multiple_results`` (eigh -> (V, w), svd -> (s, U, Vt), ...), so they MUST
    dispatch through the multi-output path; registering them in single-output
    ``elemental_rules`` made the dispatcher try to treat the per-output lists as
    SparseTensors and crash ('list' object has no attribute 'dims')."""
    def _only(primal_outs, primals, **params):
        return rule(primals, **params)[1]
    return _only


def _unsupported_decomposition(name):
    """Loud guard for a decomposition whose elemental rule is not yet correct.

    Better to fail clearly than to (a) crash with a cryptic 'list has no attribute
    dims' (the old single-output mis-registration) or (b) silently return a wrong
    Jacobian. eigh and svd (singular values) ARE verified against jax; qr/lu/eig
    are not — their analytical rules are incorrect (and eig's eigenvector
    derivatives are unsupported by jax itself)."""
    def _only(primal_outs, primals, **params):
        raise NotImplementedError(
            f"graphax does not yet have a correct elemental rule for {name}. "
            f"eigh and svd are supported; differentiate through those, or supply "
            f"a custom rule for {name}."
        )
    return _only


multi_output_elemental_only_rules[lax_linalg.eigh_p] = _multi_output_only(
    eigh_elemental_rule
)


# svd: A -> (s, U, Vt)
def svd_elemental_rule(primals, **params):
    val_out = lax_linalg.svd_p.bind(*primals, **params)
    A = primals[0]
    compute_uv = params.get("compute_uv", True)

    # Only the SINGULAR-VALUE gradient is correct. The singular-VECTOR (U / Vt)
    # Jacobians here disagree with jax (gauge ambiguity / wrong formula), so the
    # full-uv path fails loudly rather than returning a wrong gradient. Use
    # ``jnp.linalg.svd(a, compute_uv=False)`` for singular-value gradients.
    if compute_uv:
        raise NotImplementedError(
            "graphax supports gradients of svd singular VALUES only. Differentiate "
            "jnp.linalg.svd(a, compute_uv=False); singular-vector (U/Vt) gradients "
            "are not yet correct."
        )

    A_shape = list(get_shape(A))
    M, N = A_shape[-2], A_shape[-1]
    K = min(M, N)
    batch_dims = A_shape[:-2]
    num_batch = len(batch_dims)

    if compute_uv:
        s, U, Vt = val_out
    else:
        s = val_out[0]
        # In non-uv compute we don't have U and Vt to compute analytically without recomputing
        # For pure analytical, we just stop gradient if we can't form it,
        # but in practice svd is rarely used without UV if gradients are needed.
        # Let's compute thin SVD just to get the gradients:
        s, U, Vt = lax_linalg.svd_p.bind(
            A,
            full_matrices=False,
            compute_uv=True,
            subset_by_index=params.get("subset_by_index"),
            algorithm=params.get("algorithm"),
        )

    s_shape = list(get_shape(s))   # ``s`` is already unpacked (works for compute_uv=False)
    s_ndim = len(s_shape)

    # s Jacobian: ds_q = sum_{i,j} U[i,q] * dA[i,j] * V[j,q]
    # Here Vt is V.T. So V[j,q] = Vt[q,j]
    J_s = jnp.einsum("...iq,...qj->...qij", U, Vt)

    s_out_dims = []
    s_primal_dims = []
    vd = 0
    for i, sz in enumerate(batch_dims):
        s_out_dims.append(DiagonalIndex(i, sz, vd, s_ndim + i))
        s_primal_dims.append(DiagonalIndex(s_ndim + i, sz, vd, i))
        vd += 1
    s_out_dims.append(DenseIndex(num_batch, K, vd))
    vd += 1
    s_primal_dims.append(DenseIndex(s_ndim + num_batch, M, vd))
    vd += 1
    s_primal_dims.append(DenseIndex(s_ndim + num_batch + 1, N, vd))
    vd += 1
    s_tensor = _swap_back_axes(SparseTensor(s_out_dims, s_primal_dims, J_s))

    if not compute_uv:
        return val_out, [[s_tensor]]

    s_dim = s[..., None, :]
    s_diffs = (s_dim + jnp.swapaxes(s_dim, -2, -1)) * (
        s_dim - jnp.swapaxes(s_dim, -2, -1)
    )
    s_diffs_zeros = jnp.eye(K, dtype=A.dtype)
    F = 1.0 / (s_diffs + s_diffs_zeros) - s_diffs_zeros

    s_zeros = (s == 0).astype(s.dtype)
    s_inv = 1.0 / (s + s_zeros) - s_zeros
    # Diagonalize s_inv:
    s_inv_mat = jnp.eye(K, dtype=A.dtype) * s_inv[..., None]

    # U and Vt Jacobians built analytically using pure tensor contraction
    # dS_pq = u_p^T dA v_q = sum_ij U[i, p] dA[i, j] Vt[q, j]
    # => dS_tensor[p, q, i, j] = U[i, p] * Vt[q, j]
    dS_tensor = jnp.einsum("...ip,...qj->...pqij", U, Vt)

    # dU = U @ (F * (S_dim * dS + S_dim.T * dS.T) + 0.5*(dS - dS.T)*s_inv_mat)
    T1 = jnp.einsum("...q,...pqij->...pqij", s, dS_tensor)
    T2 = jnp.einsum("...p,...qpij->...pqij", s, jnp.swapaxes(dS_tensor, -4, -3))  # dS.T
    T_sym = T1 + T2

    T_skew = (
        0.5 * (dS_tensor - jnp.swapaxes(dS_tensor, -4, -3)) * s_inv_mat[..., None, None]
    )

    inner_U = F[..., None, None] * T_sym + T_skew
    J_U = jnp.einsum("...kp,...pqij->...kqij", U, inner_U)

    # V Jacobian
    # dV = V @ (F * (S_dim.T * dS + S_dim * dS.T) )
    T3 = jnp.einsum("...p,...pqij->...pqij", s, dS_tensor)
    T4 = jnp.einsum("...q,...qpij->...pqij", s, jnp.swapaxes(dS_tensor, -4, -3))
    inner_V = F[..., None, None] * (T3 + T4)
    # V = Vt.T => V[j, q] = Vt[q, j]
    V_mat = jnp.swapaxes(Vt, -2, -1)
    J_V = jnp.einsum("...lp,...pqij->...lqij", V_mat, inner_V)

    if M > N:
        # dA @ V (size M, K)
        dAV = jnp.einsum("...lq->...ql", V_mat)  # This represents derivative mask
        # We need dAV_tensor[m, q, i, j] = d(dA @ V)_{m, q} / dA[i, j]
        # = delta_{mi} V[j, q]
        I_M = jnp.eye(M, dtype=A.dtype)
        dAV_tensor = jnp.einsum("mi,...lq->...mqli", I_M, V_mat)  # [m, q, i, j=l]
        dAV_tensor = jnp.swapaxes(dAV_tensor, -2, -1)  # m, q, j, i -> m, q, i, j

        # dU += (dAV - U @ U.T @ dAV) / s
        UUt = jnp.einsum("...mp,...kp->...mk", U, U)
        proj = dAV_tensor - jnp.einsum("...mk,...kqli->...mqli", UUt, dAV_tensor)
        J_U = J_U + proj * s_inv[..., None, :, None, None]

    if N > M:
        I_N = jnp.eye(N, dtype=A.dtype)
        # dAH U = dA.T @ U
        # dAH_U_tensor[n, q, i, j] = delta_{nj} * U[i, q]
        dAH_U_tensor = jnp.einsum("nj,...iq->...nqij", I_N, U)
        VVt = jnp.einsum("...mp,...kp->...mk", V_mat, V_mat)
        proj_V = dAH_U_tensor - jnp.einsum("...mk,...kqij->...mqij", VVt, dAH_U_tensor)
        J_V = J_V + proj_V * s_inv[..., None, :, None, None]

    # J_Vt = jnp.swapaxes(J_V, -4, -3) => [k, q, i, j] -> [q, k, i, j]
    J_Vt = jnp.swapaxes(J_V, -4, -3)

    U_shape = list(get_shape(U))
    U_ndim = len(U_shape)
    U_out_dims = []
    U_primal_dims = []
    vd = 0
    for i, sz in enumerate(batch_dims):
        U_out_dims.append(DiagonalIndex(i, sz, vd, U_ndim + i))
        U_primal_dims.append(DiagonalIndex(U_ndim + i, sz, vd, i))
        vd += 1
    U_out_dims.append(DenseIndex(num_batch, M, vd))
    vd += 1
    U_out_dims.append(DenseIndex(num_batch + 1, K, vd))
    vd += 1
    U_primal_dims.append(DenseIndex(U_ndim + num_batch, M, vd))
    vd += 1
    U_primal_dims.append(DenseIndex(U_ndim + num_batch + 1, N, vd))
    vd += 1
    U_tensor = _swap_back_axes(SparseTensor(U_out_dims, U_primal_dims, J_U))

    Vt_shape = list(get_shape(Vt))
    Vt_ndim = len(Vt_shape)
    Vt_out_dims = []
    Vt_primal_dims = []
    vd = 0
    for i, sz in enumerate(batch_dims):
        Vt_out_dims.append(DiagonalIndex(i, sz, vd, Vt_ndim + i))
        Vt_primal_dims.append(DiagonalIndex(Vt_ndim + i, sz, vd, i))
        vd += 1
    Vt_out_dims.append(DenseIndex(num_batch, K, vd))
    vd += 1
    Vt_out_dims.append(DenseIndex(num_batch + 1, N, vd))
    vd += 1
    Vt_primal_dims.append(DenseIndex(Vt_ndim + num_batch, M, vd))
    vd += 1
    Vt_primal_dims.append(DenseIndex(Vt_ndim + num_batch + 1, N, vd))
    vd += 1
    Vt_tensor = _swap_back_axes(SparseTensor(Vt_out_dims, Vt_primal_dims, J_Vt))

    return val_out, [[s_tensor], [U_tensor], [Vt_tensor]]


multi_output_elemental_only_rules[lax_linalg.svd_p] = _multi_output_only(
    svd_elemental_rule
)


# qr: A -> (Q, R)
def qr_elemental_rule(primals, **params):
    val_out = lax_linalg.qr_p.bind(*primals, **params)
    A = primals[0]

    A_shape = list(get_shape(A))
    M, N = A_shape[-2], A_shape[-1]
    K = min(M, N)
    batch_dims = A_shape[:-2]
    num_batch = len(batch_dims)
    pivoting = params.get("pivoting", False)

    if pivoting:
        Q, R, P = val_out
    else:
        Q, R = val_out

    # Recompute thin QR
    q, r, *p = lax_linalg.qr_p.bind(
        A,
        pivoting=pivoting,
        full_matrices=False,
        use_magma=params.get("use_magma", False),
    )

    # We need pure analytical J_Q, J_R via dx_rinv = dx @ R^{-1} and skew-symmetric projection
    I_bcast = jnp.broadcast_to(jnp.eye(K, dtype=A.dtype), batch_dims + [K, K])
    R_inv = lax_linalg.triangular_solve_p.bind(
        r,
        I_bcast,
        left_side=True,
        lower=False,
        transpose_a=False,
        conjugate_a=False,
        unit_diagonal=False,
    )

    # dx @ R^{-1} where dx[i, j] = 1.
    # dx_tensor[m, l, i, j] = delta_{mi} * delta_{lj}
    I_M = jnp.eye(M, dtype=A.dtype)
    I_N = jnp.eye(N, dtype=A.dtype)
    dx_tensor = jnp.einsum("mi,lj->mlij", I_M, I_N)

    # dx_rinv[m, k, i, j] = sum_l dx[m, l, i, j] R_inv[l, k]
    # = sum_l (delta_mi delta_lj) R_inv[l, k] = delta_mi R_inv[j, k]
    dx_rinv = jnp.einsum("mi,...jk->...mkij", I_M, R_inv)

    qt_dx_rinv = jnp.einsum("...pq,...mkij->...pkij", jnp.swapaxes(q, -2, -1), dx_rinv)

    mask_tril = jnp.tril(jnp.ones((K, K))) - jnp.eye(K)
    qt_dx_rinv_lower = mask_tril[..., None, None] * qt_dx_rinv
    do = qt_dx_rinv_lower - jnp.swapaxes(qt_dx_rinv_lower, -4, -3)  # skew-symmetric

    dq = jnp.einsum("...pm,...mkij->...pkij", q, do - qt_dx_rinv) + dx_rinv
    dr = jnp.einsum("...mkij,...kl->...mlij", qt_dx_rinv - do, r)

    # Optional pivot re-indexing
    if pivoting:
        # Reverse permutation logic not implemented strictly in gradients for pivot anyway
        P_inv = jnp.argsort(p[0], axis=-1)
        dr = jnp.take_along_axis(dr, P_inv[..., None, None, :], axis=-3)

    Q_shape = list(get_shape(Q))
    Q_ndim = len(Q_shape)
    Q_out_dims = []
    Q_primal_dims = []
    vd = 0
    for i, s in enumerate(batch_dims):
        Q_out_dims.append(DiagonalIndex(i, s, vd, Q_ndim + i))
        Q_primal_dims.append(DiagonalIndex(Q_ndim + i, s, vd, i))
        vd += 1
    Q_out_dims.append(DenseIndex(num_batch, M, vd))
    vd += 1
    Q_out_dims.append(DenseIndex(num_batch + 1, K, vd))
    vd += 1
    Q_primal_dims.append(DenseIndex(Q_ndim + num_batch, M, vd))
    vd += 1
    Q_primal_dims.append(DenseIndex(Q_ndim + num_batch + 1, N, vd))
    vd += 1
    Q_tensor = _swap_back_axes(SparseTensor(Q_out_dims, Q_primal_dims, dq))

    R_shape = list(get_shape(R))
    R_ndim = len(R_shape)
    R_out_dims = []
    R_primal_dims = []
    vd = 0
    for i, s in enumerate(batch_dims):
        R_out_dims.append(DiagonalIndex(i, s, vd, R_ndim + i))
        R_primal_dims.append(DiagonalIndex(R_ndim + i, s, vd, i))
        vd += 1
    R_out_dims.append(DenseIndex(num_batch, K, vd))
    vd += 1
    R_out_dims.append(DenseIndex(num_batch + 1, N, vd))
    vd += 1
    R_primal_dims.append(DenseIndex(R_ndim + num_batch, M, vd))
    vd += 1
    R_primal_dims.append(DenseIndex(R_ndim + num_batch + 1, N, vd))
    vd += 1
    R_tensor = _swap_back_axes(SparseTensor(R_out_dims, R_primal_dims, dr))

    if pivoting:
        return val_out, [[Q_tensor], [R_tensor], []]
    return val_out, [[Q_tensor], [R_tensor]]


multi_output_elemental_only_rules[lax_linalg.qr_p] = _unsupported_decomposition("qr")


# tridiagonal_solve: (dl, d, du, b) -> x
# Analytical via explicit triangular solves loop logic replaced by inverse matrix
def tridiagonal_solve_elemental_rule(primals, **params):
    val_out = lax_linalg.tridiagonal_solve_p.bind(*primals, **params)
    dl, d, du, b = primals
    x = val_out

    x_shape = list(get_shape(x))
    x_ndim = len(x_shape)
    N = x_shape[-2]
    nrhs = x_shape[-1]

    # We want J_b = T^{-1}, J_param = -T^{-1} dT/dparam @ x
    # Without vmap/jacfwd, we can construct the dense inverse using the solver
    I_N = jnp.eye(N, dtype=x.dtype)

    # Solve T @ A_inv = I. tridiagonal_solve expects right hand size of shape (N, B)
    # A_inv shape will be (N, N)
    A_inv = lax_linalg.tridiagonal_solve_p.bind(dl, d, du, I_N)

    # b_tensor is identical to A_inv with specific tensor axes
    b_out_dims = []
    b_primal_dims = []
    b_out_dims.append(DenseIndex(0, N, 0))
    b_out_dims.append(DiagonalIndex(1, nrhs, None, x_ndim + 1))
    b_primal_dims.append(DenseIndex(x_ndim, N, 1))
    b_primal_dims.append(DiagonalIndex(x_ndim + 1, nrhs, None, 1))
    b_tensor = _swap_back_axes(SparseTensor(b_out_dims, b_primal_dims, A_inv))

    # Analytical J_d = -A_inv[i, k] * x[k, j]
    J_d = -jnp.einsum("ik,kj->kij", A_inv, x)

    d_out_dims = [DenseIndex(0, N, 0), DenseIndex(1, nrhs, 1)]
    d_primal_dims = [DenseIndex(x_ndim, N, 2)]
    d_tensor = _swap_back_axes(SparseTensor(d_out_dims, d_primal_dims, J_d))

    # Analytical J_dl = -A_inv[:, k] * x[k-1, :]
    shifted_x_dl = jnp.concatenate([jnp.zeros_like(x[:1]), x[:-1]], axis=0)
    J_dl = -jnp.einsum("ik,kj->kij", A_inv, shifted_x_dl)

    dl_out_dims = [DenseIndex(0, N, 0), DenseIndex(1, nrhs, 1)]
    dl_primal_dims = [DenseIndex(x_ndim, N, 2)]
    dl_tensor = _swap_back_axes(SparseTensor(dl_out_dims, dl_primal_dims, J_dl))

    # Analytical J_du = -A_inv[:, k-1] * x[k, :] (shift A_inv instead of x for correct index matching)
    shifted_x_du = jnp.concatenate([x[1:], jnp.zeros_like(x[:1])], axis=0)
    J_du = -jnp.einsum("ik,kj->kij", A_inv, shifted_x_du)

    du_out_dims = [DenseIndex(0, N, 0), DenseIndex(1, nrhs, 1)]
    du_primal_dims = [DenseIndex(x_ndim, N, 2)]
    du_tensor = _swap_back_axes(SparseTensor(du_out_dims, du_primal_dims, J_du))

    return val_out, [dl_tensor, d_tensor, du_tensor, b_tensor]


elemental_rules[lax_linalg.tridiagonal_solve_p] = tridiagonal_solve_elemental_rule


def lu_elemental_rule(primals, **params):
    val_out = lax_linalg.lu_p.bind(*primals, **params)
    A = primals[0]
    lu, pivots, permutation = val_out

    A_shape = list(get_shape(A))
    N = A_shape[-1]
    batch_dims = A_shape[:-2]
    num_batch = len(batch_dims)
    lu_ndim = len(get_shape(lu))

    # Resolve packed components
    L = jnp.tril(lu, -1) + jnp.eye(N, dtype=lu.dtype)
    U = jnp.triu(lu)

    I_bcast = jnp.broadcast_to(jnp.eye(N, dtype=A.dtype), batch_dims + [N, N])
    L_inv = lax_linalg.triangular_solve_p.bind(
        L,
        I_bcast,
        left_side=True,
        lower=True,
        transpose_a=False,
        conjugate_a=False,
        unit_diagonal=True,
    )
    U_inv = lax_linalg.triangular_solve_p.bind(
        U,
        I_bcast,
        left_side=True,
        lower=False,
        transpose_a=False,
        conjugate_a=False,
        unit_diagonal=False,
    )

    # M = L^{-1} P dA U^{-1}
    M_tensor = jnp.einsum("...pi,...jq->...pqij", L_inv, U_inv)

    mask_lower = jnp.tril(jnp.ones((N, N)), -1)
    mask_upper = jnp.triu(jnp.ones((N, N)))

    dL_tensor = jnp.einsum(
        "...pm,...mqij->...pqij", L, M_tensor * mask_lower[..., None, None]
    )
    dU_tensor = jnp.einsum(
        "...pm,...mqij->...pqij", M_tensor * mask_upper[..., None, None], U
    )

    J_lu = (
        dL_tensor * mask_lower[..., None, None]
        + dU_tensor * mask_upper[..., None, None]
    )

    # Map the contraction dimension explicitly if the pivot permutation matrix is active
    if permutation is not None:
        P_inv = jnp.argsort(permutation, axis=-1)
        J_lu = jnp.take_along_axis(J_lu, P_inv[..., None, :, None], axis=-2)

    lu_out_dims, lu_primal_dims = [], []
    vd = 0
    for i, s in enumerate(batch_dims):
        lu_out_dims.append(DiagonalIndex(i, s, vd, lu_ndim + i))
        lu_primal_dims.append(DiagonalIndex(lu_ndim + i, s, vd, i))
        vd += 1

    lu_out_dims.append(DenseIndex(num_batch, N, vd))
    vd += 1
    lu_out_dims.append(DenseIndex(num_batch + 1, N, vd))
    vd += 1
    lu_primal_dims.append(DenseIndex(lu_ndim + num_batch, N, vd))
    vd += 1
    lu_primal_dims.append(DenseIndex(lu_ndim + num_batch + 1, N, vd))
    vd += 1

    lu_tensor = _swap_back_axes(SparseTensor(lu_out_dims, lu_primal_dims, J_lu))

    return val_out, [[lu_tensor], [], []]


multi_output_elemental_only_rules[lax_linalg.lu_p] = _unsupported_decomposition("lu")


def eig_elemental_rule(primals, **params):
    val_out = lax_linalg.eig_p.bind(*primals, **params)
    A = primals[0]
    compute_left = params.get("compute_left_eigenvectors", True)
    compute_right = params.get("compute_right_eigenvectors", True)

    A_shape = list(get_shape(A))
    N = A_shape[-1]
    batch_dims = A_shape[:-2]
    num_batch = len(batch_dims)

    idx = 0
    w = val_out[idx]
    idx += 1
    vl = val_out[idx] if compute_left else None
    idx += 1 if compute_left else 0
    vr = val_out[idx] if compute_right else None

    # Force compute eigenvectors internally if not extracted
    if vl is None or vr is None:
        _, _vl, _vr = lax_linalg.eig_p.bind(
            A, compute_left_eigenvectors=True, compute_right_eigenvectors=True
        )
        vl = vl if vl is not None else _vl
        vr = vr if vr is not None else _vr

    # Normalize explicitly to guarantee vl^H vr = I
    vl_H = jnp.conj(jnp.swapaxes(vl, -1, -2))
    norm_factor = jnp.einsum("...ik,...ki->...i", vl_H, vr)
    U_H = vl_H / norm_factor[..., :, None]
    V = vr

    # Eigenvalues Jacobian: J_w[..., k, i, j] = U_H[..., k, i] * V[..., j, k]
    J_w = jnp.einsum("...ki,...jk->...kij", U_H, V)

    w_ndim = len(get_shape(w))
    w_out_dims, w_primal_dims = [], []
    vd = 0
    for i, s in enumerate(batch_dims):
        w_out_dims.append(DiagonalIndex(i, s, vd, w_ndim + i))
        w_primal_dims.append(DiagonalIndex(w_ndim + i, s, vd, i))
        vd += 1

    w_out_dims.append(DenseIndex(num_batch, N, vd))
    vd += 1
    w_primal_dims.append(DenseIndex(w_ndim + num_batch, N, vd))
    vd += 1
    w_primal_dims.append(DenseIndex(w_ndim + num_batch + 1, N, vd))
    vd += 1
    w_tensor = _swap_back_axes(SparseTensor(w_out_dims, w_primal_dims, J_w))

    tensors_out = [[w_tensor]]

    if compute_right:
        w_diff = w[..., None, :] - w[..., :, None]
        eye_n = jnp.eye(N, dtype=w.dtype)
        F = jnp.where(eye_n == 1, 0.0, 1.0 / jnp.where(w_diff == 0, 1.0, w_diff))

        T_tensor = jnp.einsum("...qi,...jk->...qkij", U_H, V)
        inner_V = F[..., None, None] * T_tensor
        J_V = jnp.einsum("...pq,...qkij->...pkij", V, inner_V)

        V_ndim = len(get_shape(vr))
        V_out_dims, V_primal_dims = [], []
        vd = 0
        for i, s in enumerate(batch_dims):
            V_out_dims.append(DiagonalIndex(i, s, vd, V_ndim + i))
            V_primal_dims.append(DiagonalIndex(V_ndim + i, s, vd, i))
            vd += 1

        V_out_dims.append(DenseIndex(num_batch, N, vd))
        vd += 1
        V_out_dims.append(DenseIndex(num_batch + 1, N, vd))
        vd += 1
        V_primal_dims.append(DenseIndex(V_ndim + num_batch, N, vd))
        vd += 1
        V_primal_dims.append(DenseIndex(V_ndim + num_batch + 1, N, vd))
        vd += 1
        V_tensor = _swap_back_axes(SparseTensor(V_out_dims, V_primal_dims, J_V))

        if compute_left:
            tensors_out.append([])
        tensors_out.append([V_tensor])

    if compute_left:
        inner_U_H = -F[..., None, None] * T_tensor
        J_UH = jnp.einsum("...qkij,...km->...qmij", inner_U_H, U_H)
        J_vl = jnp.conj(jnp.swapaxes(J_UH, -4, -3))

        vl_ndim = len(get_shape(vl))
        vl_out_dims, vl_primal_dims = [], []
        vd = 0
        for i, s in enumerate(batch_dims):
            vl_out_dims.append(DiagonalIndex(i, s, vd, vl_ndim + i))
            vl_primal_dims.append(DiagonalIndex(vl_ndim + i, s, vd, i))
            vd += 1

        vl_out_dims.append(DenseIndex(num_batch, N, vd))
        vd += 1
        vl_out_dims.append(DenseIndex(num_batch + 1, N, vd))
        vd += 1
        vl_primal_dims.append(DenseIndex(vl_ndim + num_batch, N, vd))
        vd += 1
        vl_primal_dims.append(DenseIndex(vl_ndim + num_batch + 1, N, vd))
        vd += 1
        vl_tensor = _swap_back_axes(SparseTensor(vl_out_dims, vl_primal_dims, J_vl))

        if len(tensors_out) == 1:
            tensors_out.append([vl_tensor])
        else:
            tensors_out[1] = [vl_tensor]

    return val_out, tensors_out


multi_output_elemental_only_rules[lax_linalg.eig_p] = _unsupported_decomposition("eig")

from jax._src.lax.lax import ragged_dot_general_p


def ragged_dot_general_elemental_rule(primals, **params):
    val_out = ragged_dot_general_p.bind(*primals, **params)
    lhs, rhs, group_sizes = primals

    # Strip group_sizes to route standard bilinear relaxation mapping
    # structurally through your existing dot_general framework
    _, (lhs_jac, rhs_jac) = dot_general_elemental_rule([lhs, rhs], **params)

    return val_out, [lhs_jac, rhs_jac, []]


elemental_rules[ragged_dot_general_p] = ragged_dot_general_elemental_rule


# ---------- custom_vjp_call: honor the user's reverse rule ----------

from jax.custom_derivatives import custom_vjp_call_p as _custom_vjp_call_p


def _custom_vjp_dense_jacobians(primals, **params):
    """Build the exact Jacobian of a ``custom_vjp`` call by HONORING the user's
    ``bwd`` rule instead of structurally differentiating the primal (which would
    silently discard straight-through / surrogate / clipped gradients, and crash
    when the primal is non-differentiable). We probe ``bwd`` with one-hot output
    cotangents to read off each row of the (dense) Jacobian, exactly as reverse
    mode would.

    Returns ``elementals[output_idx][invar_idx]`` per the
    ``multi_output_elemental_only_rules`` contract; the first ``num_consts`` invars
    (closed-over constants) and any input ``bwd`` reports no cotangent for get
    ``None`` (no edge)."""
    fwd_jaxpr_thunk = params["fwd_jaxpr_thunk"]
    bwd = params["bwd"]
    out_trees = params["out_trees"]
    num_consts = params["num_consts"]

    # Reconstruct fwd (residuals + outputs) — its store must fill BEFORE out_trees.
    # The thunk takes one symbolic-zero flag per NON-const input (jax splits the
    # num_consts closed-over constants off first; see custom_derivatives.py).
    n_args = len(primals) - num_consts
    fwd_closed = core.ClosedJaxpr(
        *fwd_jaxpr_thunk.call_wrapped(*([False] * n_args))
    )
    out_tree, res_tree, input_fwds = out_trees()
    # The fwd jaxpr closes over the num_consts constants, so it takes only the
    # non-const inputs (jax: `eval_jaxpr(fwd_jaxpr, fwd_consts, *primals)`).
    args_only = primals[num_consts:]
    fwd_out = core.eval_jaxpr(fwd_closed.jaxpr, fwd_closed.consts, *args_only)

    # fwd output layout is [non-forwarded residuals ..., outputs ...]; some
    # residuals are forwarded inputs (input_fwds[i] = index into the FULL input
    # list, consts included).
    n_out = out_tree.num_leaves
    num_fwd = sum(f is not None for f in input_fwds)
    num_res_out = res_tree.num_leaves - num_fwd
    res_nonfwd = iter(fwd_out[:num_res_out])
    out_leaves = fwd_out[num_res_out:num_res_out + n_out]
    res_leaves = [
        primals[f] if f is not None else next(res_nonfwd) for f in input_fwds
    ]

    # bwd returns one cotangent per non-const arg (n_args, computed above).
    out_shapes = [get_shape(o) for o in out_leaves]
    out_sizes = [int(np.prod(s)) if s else 1 for s in out_shapes]

    elementals = []
    for li in range(n_out):
        out_shape = out_shapes[li]
        out_size = len(out_shape)
        # Probe bwd once per output element with a one-hot cotangent: the returned
        # input cotangent IS that row of the Jacobian (reverse mode is vjp).
        rows = []
        for j in range(out_sizes[li]):
            cts = [
                jnp.zeros(s, dtype=getattr(o, "dtype", jnp.float32))
                for s, o in zip(out_shapes, out_leaves)
            ]
            cts[li] = cts[li].reshape(-1).at[j].set(1.0).reshape(out_shape)
            rows.append(list(bwd.call_wrapped(*res_leaves, *cts)))

        per_invar = [None] * len(primals)
        for ai in range(n_args):
            ct_col = [rows[j][ai] for j in range(out_sizes[li])]
            if any(c is None for c in ct_col):
                continue  # bwd reports no dependency on this input
            inval = primals[num_consts + ai]
            in_shape = get_shape(inval)
            J = jnp.stack(
                [jnp.asarray(c).reshape(-1) for c in ct_col], axis=0
            ).reshape(tuple(out_shape) + tuple(in_shape))
            out_dims = [DenseIndex(k, s, k) for k, s in enumerate(out_shape)]
            primal_dims = [
                DenseIndex(out_size + k, s, out_size + k)
                for k, s in enumerate(in_shape)
            ]
            per_invar[num_consts + ai] = SparseTensor(out_dims, primal_dims, J)
        elementals.append(per_invar)
    return elementals


def custom_vjp_elemental_only(primal_outs, primals, **params):
    """Multi-output elemental rule for ``custom_vjp_call`` honoring ``bwd``.

    ``custom_vjp_call_p`` is always ``multiple_results``, so it dispatches through
    ``multi_output_elemental_only_rules``. Any reconstruction failure (e.g.
    ``symbolic_zeros=True`` or an exotic residual structure we don't model yet) is
    raised loudly rather than silently falling back to the wrong primal
    derivative."""
    try:
        return _custom_vjp_dense_jacobians(primals, **params)
    except NotImplementedError:
        raise
    except Exception as e:
        raise NotImplementedError(
            "graphax could not honor this custom_vjp rule "
            f"({type(e).__name__}: {e}). custom_vjp with symbolic_zeros or an "
            "unusual residual structure is not yet supported; differentiate the "
            "underlying primal explicitly if that is the intended gradient."
        ) from e


multi_output_elemental_only_rules[_custom_vjp_call_p] = custom_vjp_elemental_only


# ---------- custom_jvp_call: honor the user's forward (jvp) rule ----------

from jax.custom_derivatives import custom_jvp_call_p as _custom_jvp_call_p


def _custom_jvp_dense_jacobians(primals, **params):
    """Build the exact Jacobian of a ``custom_jvp`` call by HONORING the user's
    jvp rule instead of structurally differentiating the primal (which would
    disagree with jax at kinks — e.g. relu'(0) is 0 by jax's custom_jvp but 0.5
    via ``max(x, 0)``). We probe the jvp with one-hot input tangents to read off
    each COLUMN of the (dense) Jacobian (forward mode is jvp).

    Returns ``elementals[output_idx][invar_idx]`` per the
    ``multi_output_elemental_only_rules`` contract."""
    jvp_jaxpr_fun = params["jvp_jaxpr_fun"]
    num_consts = params["num_consts"]
    n_args = len(primals) - num_consts
    args_only = primals[num_consts:]

    # Build the jvp jaxpr assuming every input tangent is present (no symbolic
    # zeros). It maps (primals..., tangents...) -> (out_primals..., out_tangents).
    jvp_jaxpr, jvp_consts, out_zeros = jvp_jaxpr_fun.call_wrapped(*([False] * n_args))
    n_out = len(out_zeros)

    in_shapes = [get_shape(a) for a in args_only]
    in_sizes = [int(np.prod(s)) if s else 1 for s in in_shapes]

    def _jvp_columns(ai):
        """For input ``ai``, one out-tangent list (per output) per input element:
        the response to a one-hot tangent IS that column of the Jacobian."""
        cols = []
        for j in range(in_sizes[ai]):
            tangents = [
                jnp.zeros(s, dtype=getattr(a, "dtype", jnp.float32))
                for s, a in zip(in_shapes, args_only)
            ]
            tangents[ai] = tangents[ai].reshape(-1).at[j].set(1.0).reshape(in_shapes[ai])
            out = core.eval_jaxpr(jvp_jaxpr, jvp_consts, *args_only, *tangents)
            out_primals, nz = out[:n_out], iter(out[n_out:])
            out_tangents = [
                jnp.zeros(get_shape(out_primals[li]),
                          dtype=getattr(out_primals[li], "dtype", jnp.float32))
                if out_zeros[li] else next(nz)
                for li in range(n_out)
            ]
            cols.append(out_tangents)
        return cols

    per_input_cols = [_jvp_columns(ai) for ai in range(n_args)]
    if n_args == 0:                                  # all inputs are constants
        return [[None] * len(primals) for _ in range(n_out)]
    # Output shapes = shapes of the probed out-tangents (one per output).
    out_shapes = [get_shape(per_input_cols[0][0][li]) for li in range(n_out)]

    elementals = []
    for li in range(n_out):
        out_shape = out_shapes[li]
        out_size = len(out_shape)
        per_invar = [None] * len(primals)
        for ai in range(n_args):
            cols = [per_input_cols[ai][j][li] for j in range(in_sizes[ai])]
            # Each column has out_shape; stack along a trailing input axis.
            J = jnp.stack(
                [jnp.asarray(c).reshape(-1) for c in cols], axis=-1
            ).reshape(tuple(out_shape) + tuple(in_shapes[ai]))
            out_dims = [DenseIndex(k, s, k) for k, s in enumerate(out_shape)]
            primal_dims = [
                DenseIndex(out_size + k, s, out_size + k)
                for k, s in enumerate(in_shapes[ai])
            ]
            per_invar[num_consts + ai] = SparseTensor(out_dims, primal_dims, J)
        elementals.append(per_invar)
    return elementals


def custom_jvp_elemental_only(primal_outs, primals, **params):
    """Multi-output elemental rule for ``custom_jvp_call`` honoring the jvp rule.

    ``custom_jvp_call_p`` is always ``multiple_results``. Any reconstruction
    failure (e.g. ``symbolic_zeros=True`` or an exotic structure) is raised
    loudly rather than silently differentiating the primal decomposition."""
    try:
        return _custom_jvp_dense_jacobians(primals, **params)
    except NotImplementedError:
        raise
    except Exception as e:
        raise NotImplementedError(
            "graphax could not honor this custom_jvp rule "
            f"({type(e).__name__}: {e}). custom_jvp with symbolic_zeros or an "
            "unusual structure is not yet supported."
        ) from e


multi_output_elemental_only_rules[_custom_jvp_call_p] = custom_jvp_elemental_only
