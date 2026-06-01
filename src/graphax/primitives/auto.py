import copy
from dataclasses import replace
from functools import partial, reduce
from typing import Callable

import jax._src.core as core
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax._src.pjit import jit_p
from jax.typing import ArrayLike

from ..sparse.tensor import (
    DenseIndex,
    SparseIndex,
    SparseTensor,
    _materialize_indexes,
    _swap_back_axes,
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
                SparseIndex(i, e, i, out_size + i) for i, e in enumerate(out_shape)
            ]
            primal_dims = [
                SparseIndex(out_size + i, e, i, i) for i, e in enumerate(out_shape)
            ]
    elif len(primals) == 2:
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
                        SparseIndex(i, os, axis, out_size + primal_size + 1)
                    )
                    primal_dims.append(
                        SparseIndex(out_size + primal_size + 1, os, axis, i)
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
                SparseIndex(i, e, None, out_size + i)
                for i, e in enumerate(get_shape(primal))
            ]
            primal_dims = [
                SparseIndex(out_size + i, e, None, i)
                for i, e in enumerate(get_shape(primal))
            ]
        else:
            elemental = jnp.broadcast_to(elemental, get_shape(primal))
            out_dims = [
                SparseIndex(i, e, i, out_size + i)
                for i, e in enumerate(get_shape(primal))
            ]
            primal_dims = [
                SparseIndex(out_size + i, e, i, i)
                for i, e in enumerate(get_shape(primal))
            ]
    else:
        raise NotImplementedError(
            f"Parallel Jacobians with {len(primals)} inputs not yet supported!"
        )

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
defelemental2(
    lax.abs_p, lambda out, primal: primal / out
)  # NOTE: not differentiable here!
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

# exp2: d/dx(2^x) = 2^x * ln(2) = out * ln(2)
defelemental2(lax.exp2_p, lambda out, x: out * jnp.log(2.0))

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
    return (x < y, x >= y)


defelemental(lax.max_p, max_elemental_rule)


@with_type_promotion
def min_elemental_rule(x, y, **kwargs):
    return (x < y, x <= y)


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
    in_range = ((x >= lo) & (x <= hi)).astype(x.dtype)
    return (jnp.zeros_like(lo), in_range, jnp.zeros_like(hi))


defelemental(lax.clamp_p, clamp_elemental_rule)


# rem(x, y) = x - y * trunc(x/y): d/dx = 1, d/dy = -trunc(x/y)
@with_type_promotion
def rem_elemental_rule(x, y, **kwargs):
    return (jnp.ones_like(y), -jnp.trunc(x / y))


defelemental(lax.rem_p, rem_elemental_rule)


# igamma(a, x): regularized lower incomplete gamma P(a, x)
# d/da is complex (involves log terms), we use zeros as approximation
# d/dx = x^(a-1) * exp(-x) / Gamma(a)
@with_type_promotion
def igamma_elemental_rule(a, x):
    dx = jnp.exp((a - 1.0) * jnp.log(x) - x - lax.lgamma(a))
    return (jnp.zeros_like(a), dx)


defelemental(lax.igamma_p, igamma_elemental_rule)


# igammac(a, x) = 1 - igamma(a, x): negate the igamma derivatives
@with_type_promotion
def igammac_elemental_rule(a, x):
    dx = -jnp.exp((a - 1.0) * jnp.log(x) - x - lax.lgamma(a))
    return (jnp.zeros_like(a), dx)


defelemental(lax.igammac_p, igammac_elemental_rule)


# polygamma(n, x): d/dn = 0 (n is integer order), d/dx = polygamma(n+1, x)
@with_type_promotion
def polygamma_elemental_rule(n, x):
    return (jnp.zeros_like(n), lax.polygamma(n + 1, x))


defelemental(lax.polygamma_p, polygamma_elemental_rule)


def select_elemental_rule(primals, **params):
    val_out = lax.select_n_p.bind(*primals, **params)
    pred = primals[0]
    cases = primals[1:]
    num_cases = len(cases)
    out_shape = get_shape(val_out)
    out_size = get_ndim(val_out)
    out_dtype = getattr(val_out, "dtype", jnp.float32)

    from ..sparse.tensor import DenseIndex, SparseIndex, SparseTensor

    elementals_out = []
    # Predicate Jacobian is zero
    pred_shape = get_shape(pred)
    pred_st = SparseTensor.zeros(
        [DenseIndex(i, s, i) for i, s in enumerate(out_shape)],
        [
            DenseIndex(out_size + i, s, i + out_size)
            for i, s in enumerate(pred_shape)
        ],
        out_dtype,
    )
    elementals_out.append(pred_st)

    for k in range(num_cases):
        case_k = cases[k]
        indicator = (pred == k).astype(out_dtype)
        case_shape = get_shape(case_k)
        case_size = get_ndim(case_k)

        if case_size == 0:
            out_dims = []
            primal_dims = []
        elif case_shape == out_shape:
            out_dims = [
                SparseIndex(i, s, i, out_size + i) for i, s in enumerate(out_shape)
            ]
            primal_dims = [
                SparseIndex(out_size + i, s, i, i) for i, s in enumerate(case_shape)
            ]
        else:
            out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
            primal_dims = [
                DenseIndex(i + out_size, s, i + out_size)
                for i, s in enumerate(case_shape)
            ]
        elementals_out.append(SparseTensor(out_dims, primal_dims, indicator))

    return val_out, elementals_out


elemental_rules[lax.select_n_p] = select_elemental_rule


@with_type_promotion
def pow_elemental_rule(out, x, y):
    return (y * x ** (y - 1), jnp.log(x) * out)


defelemental2(lax.pow_p, pow_elemental_rule)


# TODO Create a general reduce rule with a custom derivative!
def reduce_sum_elemental_rule(primals, **params):
    val_out = lax.reduce_sum_p.bind(*primals, **params)

    primal = primals[0]
    axes = params["axes"]
    new_out_dims, new_primal_dims, shape = [], [], []
    if axes is None:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseIndex(0, 1, 0))
    elif isinstance(axes, int):
        axes = (axes,)
    l = get_ndim(val_out)  # TODO rename l, bad name...
    count = 0
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            # idx = len(new_out_dims) + len(new_primal_dims)
            # idx = max(idx, 1) if val_out.ndim > 0 else idx
            new_primal_dims.append(DenseIndex(l + i, size, count))
            shape.append(size)
            count += 1
        else:
            ll = len(new_out_dims)
            new_out_dims.append(SparseIndex(ll, size, None, l + i))
            new_primal_dims.append(SparseIndex(l + i, size, None, ll))

    val = jnp.ones(shape, dtype=jnp.float32)
    return val_out, [SparseTensor(new_out_dims, new_primal_dims, val)]


elemental_rules[lax.reduce_sum_p] = reduce_sum_elemental_rule


def reduce_max_elemental_rule(primals, **params):
    val_out = lax.reduce_max_p.bind(*primals, **params)

    primal = primals[0]
    axes = params["axes"]
    shape = list(get_shape(val_out))

    new_out_dims, new_primal_dims, _shape = [], [], []
    if axes is None:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseIndex(0, 1, 0))
    elif isinstance(axes, int):
        axes = (axes,)

    l = get_ndim(val_out)  # TODO rename l, bad name ...
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            shape.insert(i, 1)
            idx = len(new_out_dims) + len(new_primal_dims)
            idx = max(idx, 1) if val_out.ndim > 0 else idx
            new_primal_dims.append(DenseIndex(idx, size, i))
            _shape.append(size)
        else:
            ll = len(new_out_dims)
            new_out_dims.append(SparseIndex(ll, size, i, l + i))
            new_primal_dims.append(SparseIndex(l + i, size, i, ll))

    _val_out = val_out.reshape(shape)
    new_val = primal == _val_out
    # NOTE: Normalization is important if the maximum is not unique
    norm = jnp.sum(new_val, axis=axes, keepdims=True)
    new_val = new_val / norm

    return val_out, [
        _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))
    ]


elemental_rules[lax.reduce_max_p] = reduce_max_elemental_rule


def reduce_min_elemental_rule(primals, **params):
    val_out = lax.reduce_min_p.bind(*primals, **params)

    primal = primals[0]
    axes = params["axes"]

    new_out_dims, new_primal_dims, _shape = [], [], []
    if axes is None:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseIndex(0, 1, 0))
    elif isinstance(axes, int):
        axes = (axes,)

    l = get_ndim(val_out)
    count = 0
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            idx = len(new_out_dims) + len(new_primal_dims)
            idx = max(idx, 1) if val_out.ndim > 0 else idx
            new_primal_dims.append(DenseIndex(idx, size, i))
            _shape.append(size)
            count += 1
        else:
            ll = len(new_out_dims)
            new_out_dims.append(SparseIndex(ll, size, i, l + i))
            new_primal_dims.append(SparseIndex(l + i, size, i, ll))

    new_val = primal == val_out
    # NOTE: Normalization is important if the minimum is not unique
    norm = jnp.sum(new_val, axis=axes, keepdims=True)
    new_val = new_val / norm
    return val_out, [
        _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))
    ]


elemental_rules[lax.reduce_min_p] = reduce_min_elemental_rule


# first draft unified reduce, TODO: test!
def reduce_elemental_rule(primals, agg, **params):
    assert agg in {"sum", "min", "max"}, (
        f"{agg} is not one of the valid aggregate functions `sum`, `min`, `max`"
    )
    val_out = getattr(lax, f"reduce_{agg}_p").bind(*primals, **params)

    shape = list(get_shape(val_out))
    primal = primals[0]
    axes = params["axes"]

    new_out_dims, new_primal_dims, _shape = [], [], []
    if axes is None:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseIndex(0, 1, 0))
    elif isinstance(axes, int):
        axes = (axes,)

    l = get_ndim(val_out)
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            if agg == "sum":
                idx = l + i
            else:
                shape.insert(i, 1)
                idx = len(new_out_dims) + len(new_primal_dims)
                idx = max(idx, 1) if val_out.ndim > 0 else idx

            new_primal_dims.append(DenseIndex(idx, size, i))
            _shape.append(size)
        else:
            ll = len(new_out_dims)
            val = None if "sum" else i
            new_out_dims.append(SparseIndex(ll, size, val, l + i))
            new_primal_dims.append(SparseIndex(l + i, size, val, ll))

    if agg == "sum":
        new_val = jnp.ones(_shape, dtype=jnp.float32)
    else:
        _val_out = val_out.reshape(shape)
        new_val = primal == _val_out
        norm = jnp.sum(new_val, axis=axes, keepdims=True)
        new_val /= norm

    return val_out, [
        _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))
    ]


# elemental_rules[lax.reduce_sum_p] = partial(reduce_elemental_rule, agg="sum")
# elemental_rules[lax.reduce_min_p] = partial(reduce_elemental_rule, agg="min")
# elemental_rules[lax.reduce_max_p] = partial(reduce_elemental_rule, agg="max")


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

    i, ii = 0, 0
    batch_dim_counter = 0
    for lid, ld in enumerate(lhs_shape):
        other_lid = lid + len(out_shape)
        if lid in lhs_contracting_dims:
            dim = rhs_contracting_dims[i]
            lhs_primal_dims.append(DenseIndex(other_lid, rhs_shape[dim], dim))
            i += 1
        else:
            if lid in lhs_batch_dims:
                dim = rhs_batch_dims[ii]
                ii += 1

                lhs_out_dims.insert(
                    batch_dim_counter,
                    SparseIndex(batch_dim_counter, ld, dim, other_lid),
                )
                lhs_primal_dims.append(
                    SparseIndex(other_lid, ld, dim, batch_dim_counter)
                )
                batch_dim_counter += 1
                for _idx in range(batch_dim_counter, len(lhs_out_dims)):
                    d = lhs_out_dims[_idx]
                    lhs_out_dims[_idx] = replace(d, id=d.id + 1)
                    if d.is_sparse:
                        _d_idx = d.other_id - num_out_dims
                        _d = lhs_primal_dims[_d_idx]
                        lhs_primal_dims[_d_idx] = replace(_d, other_id=_d.other_id + 1)
            else:
                _lid = len(lhs_out_dims)
                lhs_out_dims.append(SparseIndex(_lid, ld, None, other_lid))
                lhs_primal_dims.append(SparseIndex(other_lid, ld, None, _lid))
                rhs_out_dims.append(DenseIndex(len(rhs_out_dims), ld, lid))

    j, jj = 0, 0
    batch_dim_counter = 0
    for rid, rd in enumerate(rhs_shape):
        other_rid = rid + len(out_shape)
        if rid in rhs_contracting_dims:
            dim = lhs_contracting_dims[j]
            rhs_primal_dims.append(DenseIndex(other_rid, lhs_shape[dim], dim))
            j += 1
        else:
            if rid in rhs_batch_dims:
                dim = lhs_batch_dims[jj]
                jj += 1
                rhs_out_dims.insert(
                    batch_dim_counter,
                    SparseIndex(batch_dim_counter, rd, dim, other_rid),
                )
                rhs_primal_dims.append(
                    SparseIndex(other_rid, rd, dim, batch_dim_counter)
                )
                batch_dim_counter += 1
                for _idx in range(batch_dim_counter, len(rhs_out_dims)):
                    d = rhs_out_dims[_idx]
                    rhs_out_dims[_idx] = replace(d, id=d.id + 1)
                    if d.is_sparse:
                        _d_idx = d.other_id - num_out_dims
                        _d = rhs_primal_dims[_d_idx]
                        rhs_primal_dims[_d_idx] = replace(_d, other_id=_d.other_id + 1)
            else:
                _rid = len(rhs_out_dims)
                rhs_out_dims.append(SparseIndex(_rid, rd, None, other_rid))
                rhs_primal_dims.append(SparseIndex(other_rid, rd, None, _rid))
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
    val_out = lax.conv_general_dilated_p.bind(*primals, **params)
    lhs, rhs = primals

    dimension_numbers = params["dimension_numbers"]
    lhs_spec, rhs_spec, out_spec = dimension_numbers

    lhs_shape = list(get_shape(lhs))
    rhs_shape = list(get_shape(rhs))
    out_shape = list(get_shape(val_out))
    num_out_dims = len(out_shape)

    # --- Jacobian w.r.t. lhs (activations) ---
    # Batch dim: SparseIndex pair
    # Output feature dim (from rhs): DenseIndex in out
    # Input feature dim (contracted): DenseIndex in primal
    # Spatial dims: DenseIndex (windowed relationship)
    # Value: rhs (weights)
    lhs_out_dims, lhs_primal_dims = [], []

    lhs_batch_dim = lhs_spec[0]
    out_batch_dim = out_spec[0]
    batch_size = lhs_shape[lhs_batch_dim]

    # Batch: SparseIndex pair
    lhs_out_dims.append(SparseIndex(0, batch_size, None, num_out_dims))
    # Non-batch out dims: DenseIndex
    out_val_idx = 0
    for i, s in enumerate(out_shape):
        if i == out_batch_dim:
            continue
        lhs_out_dims.append(DenseIndex(len(lhs_out_dims), s, out_val_idx))
        out_val_idx += 1
    # Primal batch dim
    lhs_primal_dims.append(SparseIndex(num_out_dims, batch_size, None, 0))
    # Non-batch primal dims: DenseIndex
    for i, s in enumerate(lhs_shape):
        if i == lhs_batch_dim:
            continue
        lhs_primal_dims.append(
            DenseIndex(num_out_dims + len(lhs_primal_dims), s, out_val_idx)
        )
        out_val_idx += 1

    lhs_tensor = _swap_back_axes(SparseTensor(lhs_out_dims, lhs_primal_dims, rhs))

    # --- Jacobian w.r.t. rhs (weights) ---
    # No batch dim in rhs typically, but output has batch.
    # Output batch dim: DenseIndex (from lhs)
    # Output feature dim: SparseIndex pair with rhs output feature
    # Spatial/channel dims: DenseIndex
    # Value: lhs (activations)
    rhs_out_dims, rhs_primal_dims = [], []

    rhs_out_feature_dim = rhs_spec[0]
    out_feature_dim = out_spec[1]
    feature_size = rhs_shape[rhs_out_feature_dim]

    # Output feature: SparseIndex pair
    rhs_out_dims.append(SparseIndex(0, feature_size, None, num_out_dims))
    # Non-feature out dims: DenseIndex
    rhs_out_val_idx = 0
    for i, s in enumerate(out_shape):
        if i == out_feature_dim:
            continue
        rhs_out_dims.append(DenseIndex(len(rhs_out_dims), s, rhs_out_val_idx))
        rhs_out_val_idx += 1
    # Primal feature dim
    rhs_primal_dims.append(SparseIndex(num_out_dims, feature_size, None, 0))
    # Non-feature primal dims: DenseIndex
    for i, s in enumerate(rhs_shape):
        if i == rhs_out_feature_dim:
            continue
        rhs_primal_dims.append(
            DenseIndex(num_out_dims + len(rhs_primal_dims), s, rhs_out_val_idx)
        )
        rhs_out_val_idx += 1

    rhs_tensor = _swap_back_axes(SparseTensor(rhs_out_dims, rhs_primal_dims, lhs))

    return val_out, [lhs_tensor, rhs_tensor]


elemental_rules[lax.conv_general_dilated_p] = conv_general_dilated_elemental_rule


def iota_elemental_rule(primals, **params):
    val_out = lax.iota_p.bind(*primals, **params)
    return val_out, []


elemental_rules[lax.iota_p] = iota_elemental_rule


def device_put_elemental_rule(primals, **params):
    val_out = lax.device_put_p.bind(*primals, **params)
    return val_out, []


elemental_rules[lax.device_put_p] = device_put_elemental_rule


def stop_gradient_elemental_rule(primals, **params):
    val_out = lax.stop_gradient_p.bind(*primals, **params)
    return val_out, []


elemental_rules[lax.stop_gradient_p] = stop_gradient_elemental_rule


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
            new_out_dims.append(SparseIndex(ll, x_shape[i], None, out_ndim + i))
            new_primal_dims.append(SparseIndex(out_ndim + i, x_shape[i], None, ll))
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
            new_out_dims.append(SparseIndex(ll, x_shape[i], None, out_ndim + i))
            new_primal_dims.append(SparseIndex(out_ndim + i, x_shape[i], None, ll))
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
                SparseIndex(ll, x_shape[i], axis_count, out_ndim + i)
            )
            new_primal_dims.append(
                SparseIndex(out_ndim + i, x_shape[i], axis_count, ll)
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
    # This is analogous to reduce_max_elemental_rule.
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

# select_n_elemental_rule is now handled by the improved select_elemental_rule above
# which uses analytical indicator masks instead of jax.jacfwd

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


def _inverse_permutation(permutation):
    inverse = [0] * len(permutation)
    for i, p in enumerate(permutation):
        inverse[p] = i
    return inverse


from collections import defaultdict

from jax._src.util import safe_map

# Proper pjit and custom grad implementation only possible with a proper tracing system


def _trace_subjaxpr(jaxpr, args, consts):
    env = {}  # env stores the primal value associated with the core.Var object

    graph = defaultdict(lambda: defaultdict())  # Input connectivity
    transpose_graph = defaultdict(lambda: defaultdict())  # Output connectivity

    vo_vertices = set()  # contains all intermediate and output vertices
    counter = 1  # vertex id counter
    var_id = {}  # associates every application of a JaxprEqn with a unique integer
    # identifier that is later used when using the vertex elimination order.
    # NOTE: This only works well if the output is a single value.
    # It is ill-defined when having functions with more than one output!.

    # Reads variable and corresponding traced shaped array
    def read(var):
        if isinstance(var, core.Literal):
            return var.val
        return env[var]

    # Adds new variable and corresponding traced shaped array
    def write(var, val):
        env[var] = val

    # Writes a new elemental partial to the graph and transpose_graph
    def write_elemental(outvar, invar, val):
        # _checkify_tensor(val)
        if isinstance(invar, core.Var):
            graph[invar][outvar] = val
            transpose_graph[outvar][invar] = val

    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    # NOTE: this is essentially the tracing part. Probably should write a proper
    # tracing system with lift etc. for better compatibility with JAX
    # Loop though elemental partials and create an abstract representation of
    # the computational graph
    for eqn in jaxpr.eqns:
        # Treatment of intermediate variables that are also output variables
        for outvar in eqn.outvars:
            if isinstance(outvar, core.Var) and outvar not in var_id.keys():
                var_id[outvar] = counter
                counter += 1

        for invar in eqn.invars:
            if invar in jaxpr._outvars:
                vertex = var_id[invar]
                vo_vertices.add(vertex)

        # print("eqn:", eqn)
        # print("invars", eqn.invars)
        # print("outvars", eqn.outvars)
        invals = safe_map(read, eqn.invars)

        if eqn.primitive not in elemental_rules:
            raise NotImplementedError(
                f"{eqn.primitive} does not have registered elemental partial."
            )
        cce = elemental_rules.get(eqn.primitive)
        primal_outvals, elemental_outvals = cce(invals, **eqn.params)
        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, primal_outvals)
        else:
            safe_map(write, eqn.outvars, [primal_outvals])
        invars = [invar for invar in eqn.invars if isinstance(invar, core.Var)]
        # NOTE: Currently only able to treat one output variable

        if len(invars) == len(elemental_outvals):
            for i, invar in enumerate(invars):
                write_elemental(eqn.outvars[0], invar, elemental_outvals[i])

    return eqn.outvars, graph, transpose_graph, vo_vertices


def _subjaxpr_dfs_traverse(jaxpr):
    stack = [jaxpr]
    res = []
    while stack:
        jaxpr = stack.pop()
        res.append(jaxpr)
        for eqn in jaxpr.eqns:
            if "jaxpr" in eqn.params:
                jaxpr = eqn.params["jaxpr"]
                stack.append(jaxpr)
    return res


# use this in reverse [::-1] if you want to traverse w/o dependency issues


# TODO: this is a very ugly hack that treats pjit as a normal primitive with a stop_grad
def pjit_elemental_rule(
    primals,
    jaxpr,
    in_shardings,
    out_shardings,
    in_layouts,
    out_layouts,
    resource_env,
    donated_invars,
    name,
    keep_unused,
    inline,
):
    # TODO Jamie: How do we handle the gradients here?
    # jaxpr_cce = cce_core.cce_jaxpr(jaxpr)
    # print("pjit primals", primals)
    # print("pjit zero", zero_elementals)
    # print("pjit jaxpr", jaxpr)
    # outs, elementals, subgraph, transpose_subgraph, vo_vertices = _trace_subjaxpr(jaxpr.jaxpr, primals, ())
    # print("### pjit outs", outs)
    # print("### pjit elementals", elementals)
    # print("### pjit jaxpr", jaxpr)
    outputs = jit_p.bind(
        *primals,
        jaxpr=jaxpr,
        in_shardings=(*in_shardings,),
        out_shardings=(*out_shardings,),
        in_layouts=(*in_layouts,),
        out_layouts=(*out_layouts,),
        resource_env=resource_env,
        donated_invars=(*donated_invars,),
        name=name,
        keep_unused=keep_unused,
        inline=inline,
    )
    # print("pjit val_out:", outputs)
    out_primals = outputs
    return out_primals, []


elemental_rules[jit_p] = pjit_elemental_rule


# Should work for high-dimensional stuff
def transpose_elemental_rule(primals, **params):
    val_out = lax.transpose_p.bind(*primals, **params)
    permutation = params["permutation"]

    def transpose_transform(pre):
        new_out_dims = []
        new_primal_dims = list(pre.primal_dims)
        counter = 0
        l = len(pre.out_dims)

        for p in permutation:
            d = pre.out_dims[p]
            new_out_dims.append(replace(d, id=counter))
            if new_out_dims[-1].is_sparse:
                other_id = d.other_id
                new_primal_dims[other_id - l] = replace(
                    new_primal_dims[other_id - l], other_id=counter
                )
            counter += 1

        return _swap_back_axes(
            SparseTensor(
                tuple(new_out_dims),
                tuple(new_primal_dims),
                pre.val,
                scalar_mult=pre.scalar_mult,
                fill_value=pre.fill_value,
            )
        )

    def inverse_transpose_transform(post):
        new_out_dims = list(post.out_dims)
        new_primal_dims = []
        counter = len(post.out_dims)

        inv_permutation = _inverse_permutation(permutation)
        for p in inv_permutation:
            d = post.primal_dims[p]
            new_primal_dims.append(replace(d, id=counter))
            if new_primal_dims[-1].is_sparse:
                other_id = d.other_id
                new_out_dims[other_id] = replace(
                    new_out_dims[other_id], other_id=counter
                )
            counter += 1

        return _swap_back_axes(
            SparseTensor(
                tuple(new_out_dims),
                tuple(new_primal_dims),
                post.val,
                scalar_mult=post.scalar_mult,
                fill_value=post.fill_value,
            )
        )

    transform = JacobianTransform(transpose_transform, inverse_transpose_transform)
    return val_out, [SparseTensor([], [], None, pre_transforms=[transform])]


elemental_rules[lax.transpose_p] = transpose_elemental_rule


def reshape_elemental_rule(primals, **params):
    val_out = lax.reshape_p.bind(*primals, **params)

    def reshape_transform(pre):
        full_val = pre.dense()
        new_shape = []
        new_out_dims = []
        new_primal_dims = []
        counter = 0

        for s in val_out.shape:
            new_out_dims.append(DenseIndex(counter, s, counter))
            new_shape.append(s)
            counter += 1

        for d in pre.primal_dims:
            new_primal_dims.append(DenseIndex(counter, d.size, counter))
            new_shape.append(d.size)
            counter += 1

        full_val = full_val.reshape(new_shape)
        return SparseTensor(new_out_dims, new_primal_dims, full_val)

    def inverse_reshape_transform(post):
        full_val = jnp.array(post)
        new_shape = []
        new_out_dims = []
        new_primal_dims = []
        counter = 0
        for d in post.out_dims:
            new_out_dims.append(DenseIndex(counter, d.size, counter))
            new_shape.append(d.size)
            counter += 1
        for s in primals[0].shape:
            new_primal_dims.append(DenseIndex(counter, s, counter))
            new_shape.append(s)
            counter += 1
        full_val = full_val.reshape(new_shape)
        return SparseTensor(new_out_dims, new_primal_dims, full_val)

    transform = JacobianTransform(reshape_transform, inverse_reshape_transform)
    return val_out, [SparseTensor([], [], None, pre_transforms=[transform])]


elemental_rules[lax.reshape_p] = reshape_elemental_rule


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


def slice_elemental_rule(primals, **params):
    val_out = lax.slice_p.bind(*primals, **params)
    start_indices = params["start_indices"]
    limit_indices = params["limit_indices"]
    transform = JacobianTransform(
        make_slice_transform(start_indices, limit_indices, val_out.shape),
        make_inverse_slice_transform(start_indices, limit_indices, primals[0].shape),
    )
    return val_out, [SparseTensor([], [], None, pre_transforms=[transform])]


elemental_rules[lax.slice_p] = slice_elemental_rule


def split_elemental_rule(primals, **params):
    val_outs = lax.split_p.bind(*primals, **params)
    x = primals[0]
    axis = params["axis"]
    sizes = params["sizes"]

    elemental_outs = []
    current_offset = 0
    for i, out in enumerate(val_outs):
        start_indices = [0] * x.ndim
        start_indices[axis] = current_offset
        limit_indices = list(x.shape)
        limit_indices[axis] = current_offset + out.shape[axis]

        transform = JacobianTransform(
            make_slice_transform(start_indices, limit_indices, out.shape),
            make_inverse_slice_transform(start_indices, limit_indices, x.shape),
        )
        elemental_outs.append([SparseTensor([], [], None, pre_transforms=[transform])])
        current_offset += out.shape[axis]

    return val_outs, elemental_outs


elemental_rules[lax.split_p] = split_elemental_rule


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
    op_out_dims = [SparseIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        SparseIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, mask)

    # Jacobian for update: identity in updated region (embed into larger array)
    up_start = start_list
    up_limit = [s + sz for s, sz in zip(start_list, up_shape)]
    up_transform = JacobianTransform(
        make_slice_transform(up_start, up_limit, up_shape),
        make_inverse_slice_transform(up_start, up_limit, op_shape),
    )
    up_tensor = SparseTensor([], [], None, pre_transforms=[up_transform])

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
                SparseIndex(i, out_shape[i], None, out_ndim + paired_op)
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
                SparseIndex(out_ndim + j, op_shape[j], None, paired_out)
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
    update_window_dims = dn.update_window_dims
    inserted_window_dims = dn.inserted_window_dims
    scatter_dims_to_operand_dims = dn.scatter_dims_to_operand_dims

    scatter_dims = [d for d in range(up_ndim) if d not in update_window_dims]
    out_idx = [0] * operand_ndim

    if len(scatter_dims) > 0:
        scatter_idx = tuple(up_idx[d] for d in scatter_dims)
        idx_val = indices[scatter_idx]
        if hasattr(idx_val, "ndim") and idx_val.ndim == 0:
            idx_val = idx_val.reshape(1)
        for k, op_dim in enumerate(scatter_dims_to_operand_dims):
            out_idx[op_dim] = (
                int(idx_val[k]) if hasattr(idx_val, "__getitem__") else int(idx_val)
            )

    non_inserted = [d for d in range(operand_ndim) if d not in inserted_window_dims]
    for w_i, w_dim in enumerate(update_window_dims):
        out_idx[non_inserted[w_i]] = up_idx[w_dim]

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
    op_out_dims = [SparseIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        SparseIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
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

    return val_out, [op_tensor, up_tensor]


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
    op_out_dims = [SparseIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        SparseIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
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

    return val_out, [op_tensor, up_tensor]


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
    op_out_dims = [SparseIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        SparseIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, 1.0 - mask)

    # d/d(updates) = one-hot embedding (coeff = 1)
    jac = _build_scatter_update_jac(indices, out_shape, up_shape, params)
    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    return val_out, [op_tensor, up_tensor]


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

    op_out_dims = [SparseIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        SparseIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, op_coeff)

    # d/d(updates): operand[out_idx] at scattered positions
    up_coeff = jnp.array(operand)
    jac = _build_scatter_update_jac(
        indices, out_shape, up_shape, params, coeff=up_coeff
    )
    # Need to use operand values at the output positions as coefficients
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

    return val_out, [op_tensor, up_tensor]


elemental_rules[lax.scatter_mul_p] = scatter_mul_elemental_rule


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
        indicator = float(operand[out_idx] <= updates[up_idx])
        op_coeff = op_coeff.at[out_idx].set(indicator)

    op_out_dims = [SparseIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        SparseIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, op_coeff)

    # d/d(updates): 1 where updates < operand at scattered positions
    jac_shape = list(out_shape) + list(up_shape)
    jac = jnp.zeros(jac_shape, dtype=jnp.float32)
    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        indicator = float(updates[up_idx] < operand[out_idx])
        full_idx = tuple(out_idx) + tuple(up_idx)
        jac = jac.at[full_idx].set(indicator)

    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    return val_out, [op_tensor, up_tensor]


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
        indicator = float(operand[out_idx] >= updates[up_idx])
        op_coeff = op_coeff.at[out_idx].set(indicator)

    op_out_dims = [SparseIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        SparseIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, op_coeff)

    # d/d(updates): 1 where updates > operand at scattered positions
    jac_shape = list(out_shape) + list(up_shape)
    jac = jnp.zeros(jac_shape, dtype=jnp.float32)
    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        indicator = float(updates[up_idx] > operand[out_idx])
        full_idx = tuple(out_idx) + tuple(up_idx)
        jac = jac.at[full_idx].set(indicator)

    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    return val_out, [op_tensor, up_tensor]


elemental_rules[lax.scatter_max_p] = scatter_max_elemental_rule


def broadcast_elemental_rule(primals, **params):
    val_out = lax.broadcast_in_dim_p.bind(*primals, **params)
    dims = sorted(params["broadcast_dimensions"])
    shape = params["shape"]

    def broadcast_transform(pre):
        # We need a robust way to shift IDs that preserves partner relationships.
        l_old = len(pre.out_dims)
        p_old = len(pre.primal_dims)
        
        insert_dims = sorted([i for i, s in enumerate(shape) if i not in dims])
        
        # 1. Create the new out_dims list with DenseIndexes inserted
        # We'll use temporary IDs to keep track of logical partners
        new_out_dims = []
        old_to_new_id = {}
        
        # Track which axiss already exist to allocate expansion axes safely
        existing_axiss = [d.axis for d in (pre.out_dims + pre.primal_dims) if d.axis is not None]
        next_axis = (max(existing_axiss) + 1) if existing_axiss else 0

        curr_old_idx = 0
        for i in range(len(shape)):
            if i in insert_dims:
                # New dimension
                new_out_dims.append(DenseIndex(i, shape[i], next_axis))
                next_axis += 1
            else:
                # Existing dimension
                d_old = pre.out_dims[curr_old_idx]
                old_to_new_id[d_old.id] = i
                new_out_dims.append(replace(d_old, id=i))
                curr_old_idx += 1
        
        # 2. Update primal_dims IDs
        new_primal_dims = []
        l_new = len(new_out_dims)
        for i, d_old in enumerate(pre.primal_dims):
            new_id = l_new + i
            old_to_new_id[d_old.id] = new_id
            new_primal_dims.append(replace(d_old, id=new_id))
            
        # 3. Fix all other_ids using the mapping table
        def fix_partners(ds):
            return [replace(d, other_id=old_to_new_id[d.other_id]) if d.is_sparse else d for d in ds]
            
        new_out_dims = fix_partners(new_out_dims)
        new_primal_dims = fix_partners(new_primal_dims)

        # 4. Update sizes of existing dimensions that are being broadcasted
        for i, pos in enumerate(dims):
            d = new_out_dims[pos]
            if d.size != shape[pos]:
                new_out_dims[pos] = replace(d, size=shape[pos])

        # 5. Compute new physical array shape and mapping
        new_phys_to_size = {}
        for d in list(new_out_dims) + list(new_primal_dims):
            if d.axis is not None:
                new_phys_to_size[d.axis] = d.size
        
        sorted_new_phys = sorted(new_phys_to_size.keys())
        broadcast_shape = [new_phys_to_size[ax] for ax in sorted_new_phys]
        new_phys_to_idx = {ax: i for i, ax in enumerate(sorted_new_phys)}
        # Map source physical axes to target physical indices
        broadcast_dimensions = [None] * pre.val.ndim
        
        # Match old out_dims to new ones
        for i, pos in enumerate(dims):
            d_old = pre.out_dims[i]
            d_new = new_out_dims[pos]
            if d_old.axis is not None:
                broadcast_dimensions[d_old.axis] = new_phys_to_idx[d_new.axis]
        
        # Match old primal_dims to new ones
        for i, d_old in enumerate(pre.primal_dims):
            d_new = new_primal_dims[i]
            if d_old.axis is not None:
                broadcast_dimensions[d_old.axis] = new_phys_to_idx[d_new.axis]
        
        # Robust Rank Alignment:
        # Identify axes that are unmapped in pre.val. If they are size 1, squeeze them.
        unmapped_axes = [i for i, b in enumerate(broadcast_dimensions) if b is None]
        squeezable = [ax for ax in unmapped_axes if pre.val.shape[ax] == 1]
        
        val_to_broadcast = pre.val
        final_broadcast_dims = [b for b in broadcast_dimensions if b is not None]
        
        if squeezable:
            val_to_broadcast = jnp.squeeze(pre.val, axis=tuple(squeezable))
            # Recalculate which original axes are now where in the squeezed val
            remaining_axes = [i for i in range(pre.val.ndim) if i not in squeezable]
            final_broadcast_dims = [broadcast_dimensions[old_ax] for old_ax in remaining_axes]
            # Ensure none are None (if some unmapped were size > 1)
            final_broadcast_dims = [b if b is not None else 0 for b in final_broadcast_dims]

        try:
            if len(final_broadcast_dims) > 0 or val_to_broadcast.shape == ():
                new_val = lax.broadcast_in_dim(
                    val_to_broadcast, shape=broadcast_shape,
                    broadcast_dimensions=tuple(final_broadcast_dims)
                )
            else:
                new_val = val_to_broadcast
        except (TypeError, ValueError):
            # Elementwise scaling fallback if manual alignment failed
            if val_to_broadcast.size == 1:
                new_val = jnp.full(broadcast_shape, val_to_broadcast.reshape(()))
            else:
                # Last resort identity
                new_val = val_to_broadcast

        return SparseTensor(
            new_out_dims,
            new_primal_dims,
            new_val,
            scalar_mult=pre.scalar_mult,
            fill_value=pre.fill_value,
        )

    def inverse_broadcast_transform(post):
        rm_dims = [d for d in range(val_out.ndim) if d not in dims]
        new_out_dims = list(copy.deepcopy(post.out_dims))
        new_primal_dims = list(copy.deepcopy(post.primal_dims))
        _rm_dims = []
        counter = 0
        for dim in rm_dims:
            if new_primal_dims[dim - counter].axis is not None:
                _rm_dims.append(new_primal_dims[dim - counter].axis)
            if not new_primal_dims[dim - counter].is_sparse:
                has_smaller_dims = (
                    sum(
                        [1 for d in new_primal_dims[: dim + 1] if d.axis is not None]
                    )
                    > 0
                )
                old_axis = new_primal_dims[dim - counter].axis
                new_primal_dims[dim - counter] = None
                for i in range(dim - counter + 1, len(new_primal_dims)):
                    pd = new_primal_dims[i]
                    if pd is None:
                        continue
                    new_primal_dims[i] = replace(
                        pd,
                        id=pd.id - 1,
                        axis=pd.axis - 1
                        if pd.axis is not None and old_axis is not None
                        else pd.axis,
                    )
                    if pd.is_sparse:
                        _d = new_out_dims[pd.other_id]
                        new_out_dims[pd.other_id] = replace(
                            _d, other_id=_d.other_id - 1
                        )
                new_primal_dims.pop(dim - counter)

            else:
                d = new_primal_dims[dim - counter]
                id = d.id
                other_id = d.other_id
                old_dim = new_out_dims[other_id]
                new_out_dims[other_id] = DenseIndex(old_dim.id, old_dim.size, None)
                has_smaller_dims = (
                    sum(
                        [1 for d in new_primal_dims[: dim + 1] if d.axis is not None]
                    )
                    > 0
                )
                new_primal_dims.pop(dim - counter)
                for i, d in enumerate(new_out_dims):
                    if d.id > id:
                        new_out_dims[i] = replace(d, id=d.id - 1)
                        if new_out_dims[i].is_sparse:
                            _d = new_primal_dims[
                                new_out_dims[i].other_id - len(pre.out_dims)
                            ]
                            new_primal_dims[
                                new_out_dims[i].other_id - len(pre.out_dims)
                            ] = replace(_d, other_id=_d.other_id - 1)
                for i, d in enumerate(new_primal_dims):
                    if d.id > id:
                        new_primal_dims[i] = replace(d, id=d.id - 1)
                        if new_primal_dims[i].is_sparse:
                            _d = new_out_dims[new_primal_dims[i].other_id]
                            new_out_dims[new_primal_dims[i].other_id] = replace(
                                _d, other_id=_d.other_id - 1
                            )
                        if d.axis is not None and has_smaller_dims:
                            new_primal_dims[i] = replace(
                                new_primal_dims[i], axis=d.axis - 1
                            )
            counter += 1

        new_out_dims = tuple(new_out_dims)
        new_primal_dims = tuple(new_primal_dims)
        if len(_rm_dims) > 0:
            if all([post.val.shape[d] == 1 for d in _rm_dims]):
                new_val = jnp.squeeze(post.val, axis=tuple(_rm_dims))
            else:
                new_val = jnp.sum(post.val, axis=tuple(_rm_dims))
        else:
            new_val = post.val
        return SparseTensor(
            new_out_dims,
            new_primal_dims,
            new_val,
            scalar_mult=post.scalar_mult,
            fill_value=post.fill_value,
        )

    transform = JacobianTransform(broadcast_transform, inverse_broadcast_transform)
    return val_out, [SparseTensor([], [], None, pre_transforms=[transform])]


elemental_rules[lax.broadcast_in_dim_p] = broadcast_elemental_rule


def squeeze_elemental_rule(primals, **params):
    val_out = lax.squeeze_p.bind(*primals, **params)

    def squeeze_transform(pre):
        dims = sorted(params["dimensions"])
        new_out_dims = list(copy.deepcopy(pre.out_dims))
        new_primal_dims = list(copy.deepcopy(pre.primal_dims))
        squeeze_dims = []
        counter = 0

        for id in dims:
            idx = [j for j, d in enumerate(new_out_dims) if d.id == id][0]
            axis = new_out_dims[idx].axis
            squeeze_dims.append(axis)

            if new_out_dims[idx].is_sparse:

                def _check(d, id):
                    if d.is_sparse:
                        return d.other_id == id
                    else:
                        return False

                other_idx = [j for j, d in enumerate(new_primal_dims) if _check(d, id)][
                    0
                ]
                other_dim = new_primal_dims[other_idx]
                new_primal_dims[other_idx] = DenseIndex(
                    other_dim.id, other_dim.size, None
                )

            del new_out_dims[idx]
            counter += 1

        out_ids = [d.id for d in new_out_dims]
        primal_ids = [d.id for d in new_primal_dims]
        new_axiss = [d.axis for d in new_out_dims if d.axis is not None]
        new_axiss += [
            d.axis
            for d in new_primal_dims
            if not d.is_sparse and d.axis is not None
        ]

        for i, d in enumerate(new_out_dims):
            new_out_dims[i] = replace(
                d,
                id=out_ids.index(d.id),
                axis=new_axiss.index(d.axis)
                if d.axis is not None
                else None,
            )
            if new_out_dims[i].is_sparse:
                new_out_dims[i] = replace(
                    new_out_dims[i],
                    other_id=len(new_out_dims)
                    + primal_ids.index(new_out_dims[i].other_id),
                )

        for i, d in enumerate(new_primal_dims):
            new_primal_dims[i] = replace(
                d,
                id=len(new_out_dims) + primal_ids.index(d.id),
                axis=new_axiss.index(d.axis)
                if d.axis is not None
                else None,
            )
            if new_primal_dims[i].is_sparse:
                new_primal_dims[i] = replace(
                    new_primal_dims[i],
                    other_id=out_ids.index(new_primal_dims[i].other_id),
                )

        squeeze_dims = [d for d in squeeze_dims if d is not None]
        if len(squeeze_dims) > 0:
            new_val = jnp.squeeze(pre.val, axis=tuple(squeeze_dims))
        else:
            new_val = pre.val
        return SparseTensor(
            new_out_dims,
            new_primal_dims,
            new_val,
            scalar_mult=pre.scalar_mult,
            fill_value=pre.fill_value,
        )

    def inverse_squeeze_transform(post):
        new_dims = params["dimensions"]
        new_out_dims = list(copy.deepcopy(post.out_dims))
        new_primal_dims = list(copy.deepcopy(post.primal_dims))
        for dim in new_dims:
            axis = sum([1 for d in new_out_dims if d.axis is not None])
            axis += sum(
                [
                    1
                    for d in new_primal_dims[:dim]
                    if d.axis is not None and not d.is_sparse
                ]
            )
            new_primal_dims.insert(dim, DenseIndex(dim, 1, axis))
            for i in range(dim + 1, len(new_primal_dims)):
                d = new_primal_dims[i]
                new_primal_dims[i] = replace(
                    d,
                    id=d.id + 1,
                    axis=d.axis + 1 if d.axis is not None else None,
                )
                if d.is_sparse:
                    _d = new_out_dims[d.other_id]
                    new_out_dims[d.other_id] = replace(
                        _d,
                        other_id=_d.other_id + 1,
                        axis=_d.axis + 1 if _d.axis is not None else None,
                    )

        new_val = jnp.expand_dims(post.val, axis=new_dims)
        return SparseTensor(
            new_out_dims,
            new_primal_dims,
            new_val,
            scalar_mult=post.scalar_mult,
            fill_value=post.fill_value,
        )

    transform = JacobianTransform(squeeze_transform, inverse_squeeze_transform)
    return val_out, [SparseTensor([], [], None, pre_transforms=[transform])]


elemental_rules[lax.squeeze_p] = squeeze_elemental_rule


def concatenate_elemental_rule(primals, **params):
    val_out = lax.concatenate_p.bind(*primals, **params)
    dim = params["dimension"]

    count = _count = primals[0].shape[dim]
    slices = {0: [0, primals[0].shape[dim]]}

    for idx, primal in enumerate(primals[1:], start=1):
        count += primal.shape[dim]
        slices[idx] = [_count, count]
        _count = count

    def concatenate_transform(primal, pre):
        new_out_dims = list(copy.deepcopy(pre.out_dims))
        new_primal_dims = list(copy.deepcopy(pre.primal_dims))
        l = len(pre.out_dims)

        d = new_out_dims[dim]
        id = d.id
        primal_idx = [idx for idx, p in enumerate(primals) if p is primal][0]
        idx, _idx = slices[primal_idx]

        if not d.is_sparse:
            if d.axis is not None:
                _size = val_out.shape[dim]
                lshape = list(pre.val.shape)
                rshape = list(pre.val.shape)
                lshape[d.axis] = idx
                rshape[d.axis] = val_out.shape[dim] - _idx
                lcat_zeros = jnp.zeros(lshape)
                rcat_zeros = jnp.zeros(rshape)

                new_val = jnp.concatenate(
                    [lcat_zeros, pre.val, rcat_zeros], axis=d.axis
                )

                new_out_dims[dim] = replace(new_out_dims[dim], size=new_val.shape[dim])
            else:
                raise NotImplementedError(
                    "DenseIndex without `axis` not yet supported!"
                )
        else:
            other_id = d.other_id
            if d.axis is not None:
                _d = new_primal_dims[d.other_id - l]

                axis = sum([1 for d_i in new_out_dims if d_i.axis is not None])
                axis += sum(
                    [
                        1
                        for d_i in new_primal_dims[: other_id - l]
                        if d_i.axis is not None and not d_i.is_sparse
                    ]
                )

                for _idx_d in range(dim + 1, len(new_primal_dims)):
                    pd = new_primal_dims[_idx_d]
                    if not pd.is_sparse and pd.axis is not None:
                        new_primal_dims[_idx_d] = replace(pd, axis=pd.axis + 1)

                new_val = _materialize_indexes(pre, [d.id])

                sub_iota = jnp.eye(d.size, dtype=jnp.float32)

                shape = [1 for _ in range(pre.val.ndim)]
                shape[_d.axis] = _d.size
                shape.insert(axis, d.size)
                sub_iota = sub_iota.reshape(shape)

                new_val = new_val * sub_iota

                _size = val_out.shape[dim]
                _shape = list(new_val.shape)
                _shape[d.axis] = _size
                _shape[axis] = d.size
                zeros = jnp.zeros(_shape, dtype=jnp.float32)

                scatter_indices = [0 for _ in _shape]
                scatter_indices[d.axis] = idx
                scatter_indices[axis] = 0

                update_window_dims = tuple(n for n in range(len(_shape)))

                scatter_dims_to_operand_dims = tuple(n for n in range(len(_shape)))

                scatter_dims = lax.ScatterDimensionNumbers(
                    update_window_dims, (), scatter_dims_to_operand_dims
                )
                new_val = lax.scatter(
                    zeros,
                    jnp.array(scatter_indices),
                    new_val,
                    scatter_dims,
                    indices_are_sorted=True,
                    unique_indices=True,
                )

                new_out_dims[dim] = DenseIndex(id, val_out.shape[dim], d.axis)
                new_primal_dims[other_id - l] = DenseIndex(
                    other_id, d.size, axis
                )
            else:
                _d = new_primal_dims[d.other_id - l]
                _size = val_out.shape[dim]

                out_axis = sum(
                    [1 for d_i in new_out_dims[:dim] if d_i.axis is not None]
                )

                primal_axis = sum(
                    [1 for d_i in new_out_dims if d_i.axis is not None]
                )
                primal_axis += sum(
                    [
                        1
                        for d_i in new_primal_dims[: other_id - l]
                        if d_i.axis is not None and not d_i.is_sparse
                    ]
                )
                primal_axis = max(1, primal_axis)

                for _idx_d in range(dim + 1, len(new_primal_dims)):
                    pd = new_primal_dims[_idx_d]
                    if not pd.is_sparse and pd.axis is not None:
                        new_primal_dims[_idx_d] = replace(pd, axis=pd.axis + 1)

                if pre.val.shape != ():
                    new_val = _materialize_indexes(pre, [d.id, d.other_id])
                else:
                    new_val = pre.val

                sub_iota = jnp.eye(d.size, dtype=jnp.float32)

                shape = [1 for _ in range(pre.val.ndim)]
                shape.insert(out_axis, _d.size)
                shape.insert(primal_axis, d.size)

                new_val = new_val * sub_iota

                _shape = list(pre.val.shape)
                _shape.insert(out_axis, _size)
                _shape.insert(primal_axis, _d.size)
                zeros = jnp.zeros(_shape, dtype=jnp.float32)

                scatter_dims = lax.ScatterDimensionNumbers(
                    [out_axis, primal_axis], [], [out_axis, primal_axis]
                )
                new_val = lax.scatter(
                    zeros,
                    jnp.array([idx, 0]),
                    new_val,
                    scatter_dims,
                    indices_are_sorted=True,
                    unique_indices=True,
                )

                new_out_dims[dim] = DenseIndex(id, val_out.shape[dim], out_axis)
                new_primal_dims[other_id - l] = DenseIndex(
                    other_id, d.size, primal_axis
                )

        return SparseTensor(
            new_out_dims,
            new_primal_dims,
            new_val,
            scalar_mult=pre.scalar_mult,
            fill_value=pre.fill_value,
        )

    def inverse_concatenate_transform(primal, post):
        new_out_dims = list(copy.deepcopy(post.out_dims))
        new_primal_dims = list(copy.deepcopy(post.primal_dims))

        primal_idx = [idx for idx, p in enumerate(primals) if p is primal][0]

        d = None
        if len(new_primal_dims) > 0:
            d = new_primal_dims[dim]
        if not d.is_sparse:
            if d.axis is not None:
                new_val = lax.slice_in_dim(
                    post.val, *slices[primal_idx], axis=d.axis
                )
                new_primal_dims[dim] = replace(d, size=new_val.shape[d.axis])
            else:
                raise NotImplementedError(
                    "DenseIndex without `axis` not yet supported!"
                )
        elif d.is_sparse:
            _d = new_out_dims[d.other_id]
            if d.axis is not None:
                new_out_dims[d.other_id] = DenseIndex(_d.id, _d.size, _d.axis)
                size = slices[primal_idx][1] - slices[primal_idx][0]

                axis = sum([1 for d_i in new_out_dims if d_i.axis is not None])
                axis += sum(
                    [
                        1
                        for d_i in new_primal_dims[:dim]
                        if d_i.axis is not None and not d_i.is_sparse
                    ]
                )
                new_primal_dims[dim] = DenseIndex(_d.other_id, size, axis)

                for _idx_d in range(dim + 1, len(new_primal_dims)):
                    pd = new_primal_dims[_idx_d]
                    if not pd.is_sparse and pd.axis is not None:
                        new_primal_dims[_idx_d] = replace(pd, axis=pd.axis + 1)

                new_val = _materialize_indexes(post, [d.id])

                sub_iota = jnp.eye(d.size, dtype=jnp.float32)

                shape = [1 for _ in range(post.val.ndim)]
                shape[_d.axis] = _d.size
                shape.insert(axis, d.size)
                sub_iota = sub_iota.reshape(shape)

                new_val = new_val * sub_iota

                new_val = lax.slice_in_dim(new_val, *slices[primal_idx], axis=axis)
                new_primal_dims[dim] = replace(
                    new_primal_dims[dim],
                    size=new_val.shape[new_primal_dims[dim].axis],
                )
                new_out_dims[d.other_id] = replace(
                    new_out_dims[d.other_id],
                    size=new_val.shape[new_out_dims[d.other_id].axis],
                )
            else:
                raise NotImplementedError("Finish the implementation!")
                _d = new_out_dims[d.other_id]
                if d.axis is not None:
                    size = slices[primal_idx][1] - slices[primal_idx][0]

                    out_axis = sum(
                        [
                            1
                            for d_i in new_out_dims[: d.other_id]
                            if d_i.axis is not None
                        ]
                    )
                    primal_axis = sum(
                        [1 for d_i in new_out_dims if d_i.axis is not None]
                    )
                    primal_axis += sum(
                        [
                            1
                            for d_i in new_primal_dims[:dim]
                            if d_i.axis is not None and not d_i.is_sparse
                        ]
                    )

                    new_out_dims[d.other_id] = DenseIndex(
                        _d.id, _d.size, out_axis
                    )
                    new_primal_dims[dim] = DenseIndex(
                        _d.other_id, size, primal_axis
                    )

                    for _idx_d in range(d.other_id, len(new_out_dims)):
                        od = new_out_dims[_idx_d]
                        if not od.is_sparse and od.axis is not None:
                            new_out_dims[_idx_d] = replace(od, axis=od.axis + 1)

                    for _idx_d in range(dim + 1, len(new_primal_dims)):
                        pd = new_primal_dims[_idx_d]
                        if not pd.is_sparse and pd.axis is not None:
                            new_primal_dims[_idx_d] = replace(
                                pd, axis=pd.axis + 1
                            )

                    new_val = _materialize_indexes(post, [d.id, d.other_id])

                    sub_iota = jnp.eye(d.size, dtype=jnp.float32)

                    shape = [1 for _ in range(post.val.ndim)]
                    shape.insert(out_axis, _d.size)
                    shape.insert(primal_axis, size)
                    sub_iota = sub_iota.reshape(shape)

                    new_val = new_val * sub_iota

                    new_val = lax.slice_in_dim(
                        new_val, *slices[primal_idx], axis=primal_axis
                    )
                    new_primal_dims[dim] = replace(
                        new_primal_dims[dim],
                        size=new_val.shape[new_primal_dims[dim].axis],
                    )
                    new_out_dims[d.other_id] = replace(
                        new_out_dims[d.other_id],
                        size=new_val.shape[new_out_dims[d.other_id].axis],
                    )
        return SparseTensor(
            new_out_dims,
            new_primal_dims,
            new_val,
            scalar_mult=post.scalar_mult,
            fill_value=post.fill_value,
        )

    return val_out, [
        SparseTensor(
            [],
            [],
            None,
            pre_transforms=[
                JacobianTransform(
                    partial(concatenate_transform, p),
                    partial(inverse_concatenate_transform, p),
                )
            ],
        )
        for p in primals
    ]


elemental_rules[lax.concatenate_p] = concatenate_elemental_rule


def convert_element_type_rule(primals, **params):
    val_out = lax.convert_element_type_p.bind(*primals, **params)
    new_dtype = params["new_dtype"]

    def convert_element_type_transform(pre):
        new_pre_val = (
            None if pre.val is None else lax.convert_element_type(pre.val, new_dtype)
        )
        new_out_dims = copy.deepcopy(pre.out_dims)
        new_primal_dims = copy.deepcopy(pre.primal_dims)
        return SparseTensor(
            new_out_dims,
            new_primal_dims,
            new_pre_val,
            scalar_mult=lax.convert_element_type(pre.scalar_mult, new_dtype),
            fill_value=lax.convert_element_type(pre.fill_value, new_dtype),
        )

    def inverse_convert_element_type_transform(post):
        new_post_val = (
            None if post.val is None else lax.convert_element_type(post.val, new_dtype)
        )
        new_out_dims = copy.deepcopy(post.out_dims)
        new_primal_dims = copy.deepcopy(post.primal_dims)
        return SparseTensor(
            new_out_dims,
            new_primal_dims,
            new_post_val,
            scalar_mult=lax.convert_element_type(post.scalar_mult, new_dtype),
            fill_value=lax.convert_element_type(post.fill_value, new_dtype),
        )

    transform = JacobianTransform(
        convert_element_type_transform, inverse_convert_element_type_transform
    )
    return val_out, [SparseTensor([], [], None, pre_transforms=[transform])]


elemental_rules[lax.convert_element_type_p] = convert_element_type_rule


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
        b_out_dims.append(SparseIndex(i, s, None, num_out_dims + i))
        b_primal_dims.append(SparseIndex(num_out_dims + i, s, None, i))

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
        A_out_dims.append(SparseIndex(i, s, None, num_out_dims + i))
        A_primal_dims.append(SparseIndex(num_out_dims + i, s, None, i))

    # The resulting tensor has one dense output dim and two dense primal dims
    A_out_dims.append(DenseIndex(num_batch, N, 0))
    A_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 1))
    A_primal_dims.append(DenseIndex(num_out_dims + num_batch + 1, N, 2))

    A_tensor = _swap_back_axes(SparseTensor(A_out_dims, A_primal_dims, J_A_val))

    return val_out, [A_tensor, b_tensor]


elemental_rules[lax.linear_solve_p] = linear_solve_elemental_rule


# top_k: returns (values, indices) where values are the k largest elements.
# Jacobian of values w.r.t. x is a selection/indicator matrix.
# indices are integer-valued so no Jacobian is needed for them.
def top_k_elemental_rule(primals, **params):
    val_out = lax.top_k_p.bind(*primals, **params)
    x = primals[0]
    k = params["k"]

    values, indices = val_out

    x_shape = get_shape(x)
    val_shape = get_shape(values)
    x_ndim = len(x_shape)
    val_ndim = len(val_shape)
    n = x_shape[-1]

    # Batch dims are diagonal (SparseIndex) with axis referencing
    # the batch axes of the indicator tensor.
    # The indicator tensor has shape (*batch, k, n).
    out_dims = []
    primal_dims = []
    batch_shape = x_shape[:-1]
    axis_count = 0

    for i, s in enumerate(batch_shape):
        out_dims.append(SparseIndex(i, s, axis_count, val_ndim + i))
        primal_dims.append(SparseIndex(val_ndim + i, s, axis_count, i))
        axis_count += 1

    # Last out dim (k) and last primal dim (n) are dense
    out_dims.append(DenseIndex(val_ndim - 1, k, axis_count))
    axis_count += 1
    primal_dims.append(DenseIndex(val_ndim + x_ndim - 1, n, axis_count))
    axis_count += 1

    # Build indicator: val[..., i, j] = 1 if indices[..., i] == j
    j_range = jnp.arange(n)
    indicator = (indices[..., :, None] == j_range[None, :]).astype(x.dtype)
    val = indicator

    tensor = _swap_back_axes(SparseTensor(out_dims, primal_dims, val))

    # Return as multiple_results: (values_jacs, indices_jacs)
    # values depend on x, indices don't (integer)
    return val_out, [[tensor], []]


elemental_rules[lax.top_k_p] = top_k_elemental_rule


# argmax: returns integer indices, derivative is zero everywhere
def argmax_elemental_rule(primals, **params):
    val_out = lax.argmax_p.bind(*primals, **params)
    return val_out, []


elemental_rules[lax.argmax_p] = argmax_elemental_rule


# cond: selects one of multiple branches based on a predicate.
# Sub-jaxpr tracing not yet supported, so we treat this as stop-gradient.
from jax._src.lax.control_flow.conditionals import cond_p


def pjit_elemental_rule(primals, **params):
    from .core import vertex_elimination_jaxpr

    jaxpr = params.get("jaxpr")
    if jaxpr is None:
        # Fallback for unexpected cases
        val_out = jit_p.bind(*primals, **params)
        return val_out, [[]] * (
            len(primals) if not getattr(jit_p, "multiple_results", False) else 1
        )

    # argnums should be indices of invars that are Variables
    argnums = [i for i, v in enumerate(jaxpr.jaxpr.invars) if isinstance(v, core.Var)]

    # Recursively compute the Jacobian of the internal jaxpr
    res = vertex_elimination_jaxpr(
        jaxpr.jaxpr,
        "fwd",
        jaxpr.consts,
        *primals,
        argnums=argnums,
        sparse_representation=True,
    )
    primal_outvals, jac_vals_flat = res

    # vertex_elimination_jaxpr returns (primal_outs, jac_vals)
    # jac_vals is [ (J_out0_in0, ...), (J_out1_in0, ...), ... ] if n > 1
    # or [ J_out0_in0, J_out1_in0, ... ] if n == 1
    num_invars = len(argnums)
    jaxpr_outvars = jaxpr.jaxpr.outvars
    jaxpr_invars = [jaxpr.jaxpr.invars[i] for i in argnums]

    from ..sparse.tensor import DenseIndex, SparseTensor

    def ensure_st(j, out_var, in_var):
        if j is not None:
            return j
        out_shape = out_var.aval.shape
        in_shape = in_var.aval.shape
        out_size = len(out_shape)
        out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
        primal_dims = [
            DenseIndex(out_size + i, s, i + out_size)
            for i, s in enumerate(in_shape)
        ]
        return SparseTensor.zeros(out_dims, primal_dims, out_var.aval.dtype)

    elemental_outvals = []
    for i, out_var in enumerate(jaxpr_outvars):
        out_jacs = []
        for j, in_var in enumerate(jaxpr_invars):
            if num_invars > 1:
                jac = jac_vals_flat[i][j]
            else:
                jac = jac_vals_flat[i]
            out_jacs.append(ensure_st(jac, out_var, in_var))
        elemental_outvals.append(out_jacs)

    # jit_p.bind returns a single value if not multiple_results
    is_multiple = getattr(jit_p, "multiple_results", False)
    print(
        f"DEBUG: jit_p.multiple_results={is_multiple}, len(outvars)={len(primal_outvals) if isinstance(primal_outvals, list) else 1}"
    )
    if not is_multiple:
        print(
            f"DEBUG: Returning single. primal type: {type(primal_outvals[0])}, elemental type: {type(elemental_outvals[0])}"
        )
        return primal_outvals[0], elemental_outvals[0]
    print(
        f"DEBUG: Returning multiple. primal type: {type(primal_outvals)}, elemental type: {type(elemental_outvals)}"
    )
    return primal_outvals, elemental_outvals


def cond_elemental_rule(primals, **params):
    val_out = cond_p.bind(*primals, **params)
    return val_out, []


elemental_rules[cond_p] = cond_elemental_rule
elemental_rules[jit_p] = pjit_elemental_rule


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
        SparseIndex(i, s, None, out_ndim + i) for i, s in enumerate(out_shape)
    ]
    primal_dims = [
        SparseIndex(out_ndim + i, s, None, i) for i, s in enumerate(x_shape)
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
            out_dims.append(SparseIndex(i, out_shape[i], None, out_ndim + i))
            primal_dims.append(SparseIndex(out_ndim + i, x_shape[i], None, i))
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
        SparseIndex(i, s, None, out_ndim + i) for i, s in enumerate(out_shape)
    ]
    primal_dims = [
        SparseIndex(out_ndim + i, s, None, i) for i, s in enumerate(x_shape)
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
        SparseIndex(i, s, None, out_ndim + i) for i, s in enumerate(out_shape)
    ]
    primal_dims = [
        SparseIndex(out_ndim + i, s, None, i) for i, s in enumerate(x_shape)
    ]

    tensor = SparseTensor(out_dims, primal_dims, jnp.array(1.0, dtype=jnp.float32))
    return val_out, [tensor]


elemental_rules[all_to_all_p] = all_to_all_elemental_rule


# ============================================================================
# Matrix decompositions and linear solvers
# ============================================================================
import jax._src.lax.linalg as lax_linalg


def _batch_diag(x):
    return jnp.diagonal(x, axis1=-2, axis2=-1)


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
        b_out_dims.append(SparseIndex(i, s, None, num_out_dims + i))
        b_primal_dims.append(SparseIndex(num_out_dims + i, s, None, i))

    if len(out_shape) == num_batch + 2:
        b_out_dims.append(DenseIndex(num_batch, N, 0))
        b_out_dims.append(
            SparseIndex(num_batch + 1, nrhs, None, num_out_dims + num_batch + 1)
        )
        b_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 1))
        b_primal_dims.append(
            SparseIndex(num_out_dims + num_batch + 1, nrhs, None, num_batch + 1)
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
        A_out_dims.append(SparseIndex(i, s, None, num_out_dims + i))
        A_primal_dims.append(SparseIndex(num_out_dims + i, s, None, i))

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
        out_dims.append(SparseIndex(i, s, vd, num_out_dims + i))
        primal_dims.append(SparseIndex(num_out_dims + i, s, vd, i))
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
        w_out_dims.append(SparseIndex(i, s, vd, w_ndim + i))
        w_primal_dims.append(SparseIndex(w_ndim + i, s, vd, i))
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
        V_out_dims.append(SparseIndex(i, s, vd, V_ndim + i))
        V_primal_dims.append(SparseIndex(V_ndim + i, s, vd, i))
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


elemental_rules[lax_linalg.eigh_p] = eigh_elemental_rule


# svd: A -> (s, U, Vt)
def svd_elemental_rule(primals, **params):
    val_out = lax_linalg.svd_p.bind(*primals, **params)
    A = primals[0]
    compute_uv = params.get("compute_uv", True)

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

    s_shape = list(get_shape(val_out[0] if compute_uv else val_out))
    s_ndim = len(s_shape)

    # s Jacobian: ds_q = sum_{i,j} U[i,q] * dA[i,j] * V[j,q]
    # Here Vt is V.T. So V[j,q] = Vt[q,j]
    J_s = jnp.einsum("...iq,...qj->...qij", U, Vt)

    s_out_dims = []
    s_primal_dims = []
    vd = 0
    for i, sz in enumerate(batch_dims):
        s_out_dims.append(SparseIndex(i, sz, vd, s_ndim + i))
        s_primal_dims.append(SparseIndex(s_ndim + i, sz, vd, i))
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
        U_out_dims.append(SparseIndex(i, sz, vd, U_ndim + i))
        U_primal_dims.append(SparseIndex(U_ndim + i, sz, vd, i))
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
        Vt_out_dims.append(SparseIndex(i, sz, vd, Vt_ndim + i))
        Vt_primal_dims.append(SparseIndex(Vt_ndim + i, sz, vd, i))
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


elemental_rules[lax_linalg.svd_p] = svd_elemental_rule


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
        Q_out_dims.append(SparseIndex(i, s, vd, Q_ndim + i))
        Q_primal_dims.append(SparseIndex(Q_ndim + i, s, vd, i))
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
        R_out_dims.append(SparseIndex(i, s, vd, R_ndim + i))
        R_primal_dims.append(SparseIndex(R_ndim + i, s, vd, i))
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


elemental_rules[lax_linalg.qr_p] = qr_elemental_rule


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
    b_out_dims.append(SparseIndex(1, nrhs, None, x_ndim + 1))
    b_primal_dims.append(DenseIndex(x_ndim, N, 1))
    b_primal_dims.append(SparseIndex(x_ndim + 1, nrhs, None, 1))
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
        lu_out_dims.append(SparseIndex(i, s, vd, lu_ndim + i))
        lu_primal_dims.append(SparseIndex(lu_ndim + i, s, vd, i))
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


elemental_rules[lax_linalg.lu_p] = lu_elemental_rule


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
        w_out_dims.append(SparseIndex(i, s, vd, w_ndim + i))
        w_primal_dims.append(SparseIndex(w_ndim + i, s, vd, i))
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
            V_out_dims.append(SparseIndex(i, s, vd, V_ndim + i))
            V_primal_dims.append(SparseIndex(V_ndim + i, s, vd, i))
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
            vl_out_dims.append(SparseIndex(i, s, vd, vl_ndim + i))
            vl_primal_dims.append(SparseIndex(vl_ndim + i, s, vd, i))
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


elemental_rules[lax_linalg.eig_p] = eig_elemental_rule

from jax._src.lax.lax import ragged_dot_general_p


def ragged_dot_general_elemental_rule(primals, **params):
    val_out = ragged_dot_general_p.bind(*primals, **params)
    lhs, rhs, group_sizes = primals

    # Strip group_sizes to route standard bilinear relaxation mapping
    # structurally through your existing dot_general framework
    _, (lhs_jac, rhs_jac) = dot_general_elemental_rule([lhs, rhs], **params)

    return val_out, [lhs_jac, rhs_jac, []]


elemental_rules[ragged_dot_general_p] = ragged_dot_general_elemental_rule
