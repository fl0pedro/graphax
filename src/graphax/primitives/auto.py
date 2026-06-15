import jax._src.core as core
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

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
    get_ndim,
    get_shape,
)


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
