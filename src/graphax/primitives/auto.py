import jax._src.core as core
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

from ..sparse.tensor import (
    DenseIndex,
    DiagonalIndex,
    SparseTensor,
    _swap_back_axes,
)
from .base import (
    elemental_rules,
    elemental_only_rules,
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
