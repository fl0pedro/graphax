import jax.lax as lax
import jax.numpy as jnp

from ..sparse.tensor import (
    DenseIndex,
    DiagonalIndex,
    SparseTensor,
    _swap_back_axes,
)
from ..sparse.indexes import (
    ToeplitzIndex,
    TOEPLITZ_OUT,
    TOEPLITZ_IN,
    TOEPLITZ_TAP,
)
from .base import elemental_rules, elemental_only_rules, get_ndim, get_shape


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


def _pad_operand_transform(primals, val_out, **params):
    """``pad`` w.r.t. the OPERAND as a SEED-DRAINABLE deferred transform. pad is a
    linear EMBEDDING (``out[lo + j*(interior+1)] = x[j]``); its adjoint is the
    CROP (a strided slice of the unpadded positions), so ``post @ pad(pre) ==
    crop(post) @ pre``: in reverse the elimination drains it onto the cotangent
    VECTOR (one strided slice, O(n)) instead of materialising the n_out×n_in
    embedding. The forward / non-drainable path densifies."""
    from .transforms import (
        JacobianTransform, _dense_grid,
        _is_scalar_identity_post, _identity_post_over,
    )
    padding_config = params["padding_config"]
    x_shape = get_shape(primals[0])

    def pad_transform(pre):                      # embed: pad the OUT axes
        full = pre.dense()
        cfg = list(padding_config) + [(0, 0, 0)] * (full.ndim - len(padding_config))
        full = lax.pad(full, jnp.array(0.0, dtype=full.dtype), cfg)
        new_out, new_primal = _dense_grid(get_shape(val_out), pre.primal_shape)
        # pre.dense() already folded scalar_mult/fill; do NOT re-carry (double-apply).
        return SparseTensor(new_out, new_primal, full)

    def inverse_pad_transform(post):             # adjoint: gather the embedded pos
        if _is_scalar_identity_post(post):
            post = _identity_post_over(post, val_out.shape, val_out.dtype)
        full = post.dense()
        n_out = len(post.out_dims)
        out_shape = get_shape(val_out)
        # The adjoint of pad is a bounds-masked gather: input position j came from
        # output position ``lo + j*(interior+1)``; positions out of [0, out_s)
        # (negative-padding CROP, or interior overhang) contribute zero. This is
        # O(in) and handles pad / crop / dilation uniformly (the slice form broke
        # on negative padding).
        for i, (lo, hi, interior) in enumerate(padding_config):
            in_s, out_s, stride, ax = x_shape[i], out_shape[i], interior + 1, n_out + i
            if lo >= 0 and lo + (in_s - 1) * stride + 1 <= out_s:
                # pure embed on this axis (no crop / overhang): a cheap strided
                # slice extracts the in_s embedded positions.
                full = lax.slice_in_dim(
                    full, lo, lo + (in_s - 1) * stride + 1, stride=stride, axis=ax)
            else:
                # crop (negative pad) or overhang: bounds-masked gather.
                pos = lo + jnp.arange(in_s) * stride
                valid = (pos >= 0) & (pos < out_s)
                full = jnp.take(full, jnp.where(valid, pos, 0), axis=ax)
                full = full * valid.astype(full.dtype).reshape(
                    [in_s if a == ax else 1 for a in range(full.ndim)])
        new_out, new_primal = _dense_grid(post.out_shape, x_shape)
        # post.dense() already folded scalar_mult/fill; do NOT re-carry (double-apply).
        return SparseTensor(new_out, new_primal, full)

    transform = JacobianTransform(pad_transform, inverse_pad_transform,
                                  seed_drainable=True)
    return SparseTensor([], [], None, pre_transforms=[transform])


def _pad_elementals(primals, val_out, **params):
    """Both pad elementals: the operand deferred transform, plus — for a
    non-constant ``padding_value`` — the value-independent pad-cell mask (pad
    cells +1, embedded operand cells 0). Shared by the eager and elemental_only
    paths so BOTH differentiate ``padding_value`` (the elemental_only path used
    to silently drop it; it is the preferred dispatch, so the gradient vanished)."""
    x, padding_value = primals
    padding_config = params["padding_config"]
    tensors_out = [_pad_operand_transform(primals, val_out, **params)]

    if not isinstance(padding_value, (float, int, complex)):
        x_shape = get_shape(x)
        out_shape = get_shape(val_out)
        pad_mask = jnp.ones_like(val_out)
        slices_obj = tuple(
            slice(lo, lo + (x_shape[i] - 1) * (interior + 1) + 1, interior + 1)
            for i, (lo, hi, interior) in enumerate(padding_config)
        )
        pad_mask = pad_mask.at[slices_obj].set(0.0)
        p_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
        tensors_out.append(SparseTensor(p_out_dims, [], pad_mask))

    return tensors_out


def pad_elemental_rule(primals, **params):
    val_out = lax.pad_p.bind(*primals, **params)
    return val_out, _pad_elementals(primals, val_out, **params)


def pad_elemental_only(primal_out, primals, **params):
    return _pad_elementals(primals, primal_out, **params)


elemental_rules[lax.pad_p] = pad_elemental_rule
elemental_only_rules[lax.pad_p] = pad_elemental_only


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
