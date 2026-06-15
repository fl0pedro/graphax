import jax._src.core as core
import jax.lax as lax
import jax.numpy as jnp

from ..sparse.tensor import (
    DenseIndex,
    DiagonalIndex,
    SparseTensor,
    _swap_back_axes,
)
from .base import (
    elemental_rules,
    multi_output_elemental_only_rules,
    get_shape,
)


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
