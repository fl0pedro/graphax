
import numpy as np
import jax.lax as lax
import jax.numpy as jnp

from .base import (
    NO_EDGE,
    elemental_rules,
    elemental_only_rules,
    get_ndim,
    get_shape,
)
from ..sparse.tensor import (
    DenseIndex,
    DiagonalIndex,
    SparseTensor,
    _swap_back_axes,
)


# ---------- select_n ----------

def _select_elementals(primals, val_out, **params):
    # select_n(which, *cases) picks element-wise from `cases` indexed by `which`.
    # Jacobian wrt case_k is the identity masked by (which == k). `which` is
    # integer-valued and non-differentiable: when it comes from a Var we emit
    # NO_EDGE so the dispatcher keeps alignment but adds no edge; when it comes
    # from a Literal the dispatcher already drops its slot, so we omit it here.
    which, *cases = primals
    out_shape = get_shape(val_out)
    out_ndim = len(out_shape)
    out_dtype = val_out.dtype

    def _masked_identity(mask):
        if out_ndim == 0:
            return SparseTensor([], [], mask)
        # Diagonal Jacobian whose diagonal carries the per-element mask values, so
        # each DiagonalIndex must read its own axis of ``mask`` (axis=i). axis=None
        # marked the dim as a pure (value-less) identity, which DISCARDED the mask
        # -> select_n/where Jacobians were silently wrong (and mis-shaped when the
        # same input fed multiple branches).
        out_dims = [
            DiagonalIndex(i, s, i, out_ndim + i) for i, s in enumerate(out_shape)
        ]
        primal_dims = [
            DiagonalIndex(out_ndim + i, s, i, i) for i, s in enumerate(out_shape)
        ]
        return SparseTensor(out_dims, primal_dims, mask)

    elementals = [NO_EDGE]
    for k, _ in enumerate(cases):
        mask = (which == k).astype(out_dtype)
        elementals.append(_masked_identity(mask))
    return elementals


def select_elemental_rule(primals, **params):
    val_out = lax.select_n_p.bind(*primals, **params)
    return val_out, _select_elementals(primals, val_out, **params)


def select_elemental_only(primal_out, primals, **params):
    return _select_elementals(primals, primal_out, **params)


elemental_rules[lax.select_n_p] = select_elemental_rule
elemental_only_rules[lax.select_n_p] = select_elemental_only


# ---------- reduce_sum ----------

# TODO Create a general reduce rule with a custom derivative!
def _reduce_sum_elementals(primals, val_out_ndim, **params):
    primal = primals[0]
    axes = params["axes"]

    new_out_dims, new_primal_dims, shape = [], [], []
    reduce_all = axes is None
    if reduce_all:
        # Full reduction -> scalar output (one size-1 out axis). The append used
        # to run before this list existed -> UnboundLocalError (latent: jnp.sum
        # always lowers explicit axes, never None).
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseIndex(0, 1, 0))
    elif isinstance(axes, int):
        axes = (axes,)

    l = val_out_ndim
    base = 1 if reduce_all else l  # contiguous ids; see _reduce_max_elementals
    count = 0
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            new_primal_dims.append(DenseIndex(base + i, size, count))
            shape.append(size)
            count += 1
        else:
            ll = len(new_out_dims)
            new_out_dims.append(DiagonalIndex(ll, size, None, l + i))
            new_primal_dims.append(DiagonalIndex(l + i, size, None, ll))

    val = jnp.ones(shape, dtype=jnp.float32)
    return [SparseTensor(new_out_dims, new_primal_dims, val)]


def reduce_sum_elemental_rule(primals, **params):
    val_out = lax.reduce_sum_p.bind(*primals, **params)
    return val_out, _reduce_sum_elementals(primals, get_ndim(val_out), **params)


def reduce_sum_elemental_only(primal_out, primals, **params):
    return _reduce_sum_elementals(primals, get_ndim(primal_out), **params)


elemental_rules[lax.reduce_sum_p] = reduce_sum_elemental_rule
elemental_only_rules[lax.reduce_sum_p] = reduce_sum_elemental_only


# ---------- reduce_max ----------

def _reduce_max_elementals(primals, val_out, **params):
    primal = primals[0]
    axes = params["axes"]
    shape = list(get_shape(val_out))

    new_out_dims, new_primal_dims, _shape = [], [], []
    reduce_all = axes is None
    if reduce_all:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseIndex(0, 1, 0, True))
    elif isinstance(axes, int):
        axes = (axes,)

    l = get_ndim(val_out)  # number of kept (out) axes
    # Contiguous ids: every primal axis gets ``base + i`` (base = #out dims),
    # matching the kept DiagonalIndex pairs. The old ``len(out)+len(primal)``
    # counter collided with a kept axis following a reduced one on >=3D
    # (ids like [0,1,2,2,4] -> Topology Error).
    base = 1 if reduce_all else l
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            shape.insert(i, 1)
            new_primal_dims.append(DenseIndex(base + i, size, i))
            _shape.append(size)
        else:
            ll = len(new_out_dims)
            new_out_dims.append(DiagonalIndex(ll, size, i, l + i))
            new_primal_dims.append(DiagonalIndex(l + i, size, i, ll))

    _val_out = val_out.reshape(shape)
    new_val = jnp.where(primal == _val_out, 1, 0)
    # NOTE: Normalization is important if the maximum is not unique
    norm = jnp.sum(new_val, axis=axes, keepdims=True)
    new_val = new_val / norm

    return [_swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))]


def reduce_max_elemental_rule(primals, **params):
    val_out = lax.reduce_max_p.bind(*primals, **params)
    return val_out, _reduce_max_elementals(primals, val_out, **params)


def reduce_max_elemental_only(primal_out, primals, **params):
    return _reduce_max_elementals(primals, primal_out, **params)


elemental_rules[lax.reduce_max_p] = reduce_max_elemental_rule
elemental_only_rules[lax.reduce_max_p] = reduce_max_elemental_only


# ---------- reduce_min ----------

def _reduce_min_elementals(primals, val_out, **params):
    primal = primals[0]
    axes = params["axes"]
    shape = list(get_shape(val_out))

    new_out_dims, new_primal_dims, _shape = [], [], []
    reduce_all = axes is None
    if reduce_all:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseIndex(0, 1, 0, True))
    elif isinstance(axes, int):
        axes = (axes,)

    l = get_ndim(val_out)
    base = 1 if reduce_all else l   # contiguous ids; see _reduce_max_elementals
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            shape.insert(i, 1)
            new_primal_dims.append(DenseIndex(base + i, size, i))
            _shape.append(size)
        else:
            ll = len(new_out_dims)
            new_out_dims.append(DiagonalIndex(ll, size, i, l + i))
            new_primal_dims.append(DiagonalIndex(l + i, size, i, ll))

    # Reshape val_out with size-1 at reduced axes so the equality broadcasts
    # against the full-shape primal (reduce_max had this; reduce_min lacked it).
    _val_out = val_out.reshape(shape)
    new_val = jnp.where(primal == _val_out, 1, 0)
    # NOTE: Normalization is important if the minimum is not unique
    norm = jnp.sum(new_val, axis=axes, keepdims=True)
    new_val = new_val / norm
    return [_swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))]


def reduce_min_elemental_rule(primals, **params):
    val_out = lax.reduce_min_p.bind(*primals, **params)
    return val_out, _reduce_min_elementals(primals, val_out, **params)


def reduce_min_elemental_only(primal_out, primals, **params):
    return _reduce_min_elementals(primals, primal_out, **params)


elemental_rules[lax.reduce_min_p] = reduce_min_elemental_rule
elemental_only_rules[lax.reduce_min_p] = reduce_min_elemental_only
