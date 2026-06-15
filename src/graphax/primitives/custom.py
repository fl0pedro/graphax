"""Honor user-supplied derivatives.

Three primitives carry a derivative the user wrote by hand, which graphax must
HONOR rather than re-derive by structurally differentiating the primal:

* ``custom_vjp_call`` — probe the user's ``bwd`` with one-hot output cotangents
  to read off the (dense) Jacobian rows, exactly as reverse mode would.
* ``custom_jvp_call`` — probe the user's ``jvp`` with one-hot input tangents to
  read off the Jacobian columns, exactly as forward mode would.
* named ``jit`` ``jax.nn`` activations — ``@jit``-wrapped composites with **no**
  ``custom_jvp`` whose ``select_n``/``max``/``min`` decomposition has the wrong
  subgradient at the kink. Each jit carries its function name in
  ``params['name']``; we look it up in ``ACTIVATION_DERIVS`` and emit graphax's
  own diagonal Jacobian instead of inlining the body.

In every case the alternative — differentiating the decomposition — would
silently drop straight-through / surrogate / clipped gradients or pick a wrong
kink subgradient.
"""

import jax
import jax._src.core as core
import jax.numpy as jnp
import numpy as np

from ..sparse.tensor import DenseIndex, SparseTensor
from .base import (
    multi_output_elemental_only_rules,
    make_parallel_jacobian,
    get_shape,
    jit_name_rules,
)


# ---------- named jit activations: honor jax.nn's intended kink gradient ----------

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
