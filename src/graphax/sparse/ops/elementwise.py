"""Tiled block-sparse elementwise op.

Pipeline:
    1. Classify dim pairs (sparse-pair / dense-pair, promoting Dense↔Sparse to a
       1-block synthetic sparse pair).
    2. Align values via the shared ``_prepare_physical_array`` helper.
    3. Promote both sides onto the per-pair LCM-block grid.
    4. Apply ``op`` element-wise; optionally demote (sum-reduce) for intersection.
    5. Rebuild ``SparseTensor`` ``out_dims`` / ``primal_dims`` from the grid axes.
"""
from __future__ import annotations
import math
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .utils import _arr2st, _is_sparse, _val_or_one, _prepare_physical_array, _materialize_compressed, _is_zero_fill
from .layout import generate_block_permutation
from graphax.sparse.indexes import SparseIndex, DenseIndex

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


# Ops where ``op(0, 0) == 0`` — used to propagate the static ``_zero_fill``
# flag through elementwise-of-zero-fills. Hardcoded because jit-time numerical
# probing always promotes operands to tracers, even concrete numpy zeros, so
# we can't introspect ``op`` numerically inside the trace.
_ZERO_PRESERVING_OPS = frozenset((
    jnp.add, jnp.subtract, jnp.multiply, jnp.maximum, jnp.minimum,
    jnp.logical_or, jnp.logical_and, jnp.logical_xor,
    jnp.bitwise_or, jnp.bitwise_and, jnp.bitwise_xor,
    jax.lax.add, jax.lax.sub, jax.lax.mul, jax.lax.max, jax.lax.min,
))

# Stricter subset: ``op(0, x) == op(x, 0) == x`` — i.e., 0 is the *additive
# identity*. The divisor fast path (and any other "leave the big buffer alone
# at off-support positions" optimization) only works when missing-side values
# leave the existing buffer unchanged. ``mul``, ``min``, ``max``, ``and``
# violate this (mul/and zero out, min/max with negatives can flip), so they're
# excluded — those ops are still routed through the general expansion path.
_ADDITIVE_IDENTITY_OPS = frozenset((
    jnp.add, jax.lax.add,
    jnp.logical_or, jnp.logical_xor,
    jnp.bitwise_or, jnp.bitwise_xor,
))


def _scaled_fill(tensor) -> Array:
    """Post-scaled fill_value: ``fill * scalar_mult`` (or ``& mask`` for bool).
    Canonical form used to compose output fills consistently — every fast path
    must produce a fill that matches the post-scaled meaning of the input
    operands so downstream consumers see one definition."""
    if tensor.dtype == jnp.bool_:
        return tensor.fill_value & tensor.scalar_mult.astype(jnp.bool_)
    return tensor.fill_value * tensor.scalar_mult


def _normalize_inputs(lhs, rhs):
    target_dtype = (lhs.dtype if _is_sparse(lhs) and not _is_sparse(rhs)
                    else rhs.dtype if _is_sparse(rhs) and not _is_sparse(lhs)
                    else None)
    inputs = [lhs, rhs]
    for i in range(2):
        if _is_sparse(inputs[i]):
            continue
        other = inputs[1 - i]
        obj = inputs[i].dense() if hasattr(inputs[i], "dense") else inputs[i]
        if hasattr(other, "shape") and getattr(obj, "size", 0) == 1:
            obj = jnp.broadcast_to(obj, other.shape)
        inputs[i] = _arr2st(obj, out_ndim=len(other.out_dims) if _is_sparse(other) else None,
                            dtype=target_dtype)
    lhs, rhs = inputs
    # Materialize compressed storage on input. ``_materialize_compressed`` picks
    # ``to_meta_blocks()`` over ``to_dense()`` when the host's dim structure is
    # already meta-block-diagonal — keeps storage at ``M·H·W`` rather than the
    # ``M²·H·W`` of the full dense form, and keeps every downstream op on the
    # block-diagonal fast path. XLA fuses either expression into the consumer.
    from .utils import _copy
    if getattr(lhs, "compressed_val", None) is not None:
        lhs = _copy(lhs, val=_materialize_compressed(lhs))
    if getattr(rhs, "compressed_val", None) is not None:
        rhs = _copy(rhs, val=_materialize_compressed(rhs))
    # Static shape comparison: ``SparseTensor.shape`` returns Python ints
    # derived from the dim metadata, so this is a trace-time check (no runtime
    # branching on traced shapes).
    if lhs.shape != rhs.shape:
        raise ValueError(f"Shape mismatch: {lhs.shape} != {rhs.shape}")
    return lhs, rhs


def _promote_dense(d, partner_id):
    """Wrap a DenseIndex into a synthetic 1-block SparseIndex paired with `partner_id`."""
    return SparseIndex(d.id, 1, axis=None, other_id=partner_id,
                           block_size=d.size, block_axis=d.axis)


def _resolve_dim_pairing(i, ldims, rdims, processed):
    """Pair dim i across (lhs, rhs); promote Dense↔Sparse to a synthetic 1-block sparse pair."""
    ld, rd = ldims[i], rdims[i]
    l_sp, r_sp = ld.is_sparse, rd.is_sparse
    if not l_sp and not r_sp:
        processed.add(i); return "dense", (ld, rd)
    if l_sp and r_sp:
        j = next(k for k, d in enumerate(ldims) if d.id == ld.other_id)
        lp, rp = ldims[j], rdims[j]
        if not rp.is_sparse or rd.other_id != rp.id:
            raise ValueError("Topology mismatch: sparse pairs do not align.")
        processed.update([i, j]); return "sparse", (ld, lp, rd, rp)
    if l_sp:
        j = next(k for k, d in enumerate(ldims) if d.id == ld.other_id)
        lp, rp = ldims[j], rdims[j]
        if not not rp.is_sparse:
            raise ValueError("Topology mismatch: expected DenseIndex partner.")
        processed.update([i, j])
        return "sparse", (ld, lp, _promote_dense(rd, rp.id), _promote_dense(rp, rd.id))
    j = next(k for k, d in enumerate(rdims) if d.id == rd.other_id)
    rp, lp = rdims[j], ldims[j]
    if not not lp.is_sparse:
        raise ValueError("Topology mismatch: expected DenseIndex partner.")
    processed.update([i, j])
    return "sparse", (_promote_dense(ld, lp.id), _promote_dense(lp, ld.id), rd, rp)


def _map_topology(lhs, rhs):
    sp, dp, processed = [], [], set()
    for i in range(len(lhs.dims)):
        if i in processed:
            continue
        kind, pair = _resolve_dim_pairing(i, lhs.dims, rhs.dims, processed)
        (sp if kind == "sparse" else dp).append(pair)
    return sp, dp


def _pair_metric(p):
    ld1, ld2, rd1, rd2 = p
    lb1, lb2 = ld1.block_size or 1, ld2.block_size or 1
    rb1, rb2 = rd1.block_size or 1, rd2.block_size or 1
    cb1, cb2 = math.lcm(lb1, rb1), math.lcm(lb2, rb2)
    exp_l1 = cb1 // lb1
    if ld1.size % exp_l1:
        raise ValueError(
            f"Pair size {ld1.size} not divisible by promotion factor {exp_l1} "
            f"(common block {cb1}, lhs block {lb1}); cannot tile onto LCM grid."
        )
    return {"unified_size": ld1.size // exp_l1,
            "common_b1": cb1, "common_b2": cb2,
            "left_b1": lb1, "left_b2": lb2,
            "right_b1": rb1, "right_b2": rb2,
            "left_size": ld1.size, "right_size": rd1.size}


def _ew_pair_axes(sparse_pairs, dense_pairs, is_left):
    """For one side, yield (source_axis_or_None, target_size) for every per-pair axis.
    Sparse pairs contribute 3 axes (outer + 2 block dims); dense pairs contribute 1 axis."""
    for p in sparse_pairs:
        d1, d2 = (p[0], p[1]) if is_left else (p[2], p[3])
        yield d1.axis, d1.size
        yield getattr(d1, "block_axis", None), d1.block_size or 1
        yield getattr(d2, "block_axis", None), d2.block_size or 1
    for p in dense_pairs:
        d = p[0] if is_left else p[1]
        yield d.axis, d.size


def _value_axes_info(tensor, sp, dp, is_left):
    value = _val_or_one(tensor)
    axes = [a for a, _ in _ew_pair_axes(sp, dp, is_left)]
    used = [a for a in axes if a is not None]
    return value, axes, [value.shape[i] for i in range(value.ndim) if i not in used]


def _align_value(value, tensor, sp, dp, axes, broadcast_unused, is_left):
    if tensor.dtype == jnp.bool_:
        value = value & tensor.scalar_mult.astype(jnp.bool_)
    else:
        value = value * tensor.scalar_mult
    target_shape = [s for _, s in _ew_pair_axes(sp, dp, is_left)]
    value = _prepare_physical_array(value, axes)
    pad = len(broadcast_unused) - (value.ndim - len(axes))
    if pad > 0:
        n = len(axes)
        value = value.reshape(value.shape[:n] + (1,) * pad + value.shape[n:])
    return jnp.broadcast_to(value, tuple(target_shape) + tuple(broadcast_unused))


def _promote_to_unified(value: Array, metrics, is_left: bool) -> Array:
    in_shape, exp_shape, out_shape = [], [], []
    needs_expansion = False
    for m in metrics:
        b1, b2 = (m["left_b1"], m["left_b2"]) if is_left else (m["right_b1"], m["right_b2"])
        cb1, cb2 = m["common_b1"], m["common_b2"]
        exp_h, exp_w = cb1 // b1, cb2 // b2
        # Per-pair invariant: source value has shape ``(M, b1, b2)``; promoting
        # to the LCM grid scatters ``exp_h`` sub-blocks onto the diagonal of an
        # ``(exp_h, exp_w)`` block-axis grid. With ``exp_h != exp_w`` the
        # current eye-mask path can't represent which positions to fill, and
        # downstream reshape to ``(unified, cb1, cb2)`` would also misalign.
        # In well-formed sparse pairs the row/col extent constraint forces
        # ``exp_h == exp_w``; reject the malformed-input case loudly.
        if exp_h != exp_w:
            raise ValueError(
                f"_promote_to_unified requires square block expansion "
                f"(exp_h={exp_h}, exp_w={exp_w}); pair metric {m} is malformed."
            )
        in_shape.extend([m["unified_size"], exp_h, b1, b2])
        exp_shape.extend([m["unified_size"], exp_h, 1, b1, b2])
        out_shape.extend([m["unified_size"], cb1, cb2])
        if exp_h > 1:
            needs_expansion = True
    rem = list(value.shape[3 * len(metrics):])
    in_shape += rem; exp_shape += rem; out_shape += rem
    if value.shape != tuple(in_shape):
        value = value.reshape(in_shape)

    # Single-pair expansion fast path: scatter the ``exp`` source sub-blocks
    # directly into a zero-initialized ``(M, LCM_h, LCM_w)`` buffer instead of
    # the broadcast+where dance below. The broadcast+where allocates an
    # intermediate ``(M, exp, exp, B_h, B_w)`` (size ``exp²·M·B_h·B_w``) full
    # of zeros except on the diagonal; the masked-where stays at peak
    # ``M·LCM_h·LCM_w`` (the same as the final output buffer). That's the
    # *divisor case*: one side already lives at LCM granularity, so only the
    # smaller side goes through this expansion. ``test_03``'s peak-mem
    # regression (16.69 → 4.17 MB targetted) sits on this path.
    #
    # Off-diagonal values are 0 (the same value the broadcast+where path
    # produces); this is correct for all zero-fill ops we support — every op
    # in ``_ZERO_PRESERVING_OPS`` returns the partner's value when one operand
    # is 0 (sub/min on negatives is the only edge case worth flagging, but the
    # existing path has the same semantics).
    if needs_expansion and len(metrics) == 1:
        m = metrics[0]
        b1, b2 = (m["left_b1"], m["left_b2"]) if is_left else (m["right_b1"], m["right_b2"])
        cb1, cb2 = m["common_b1"], m["common_b2"]
        exp = cb1 // b1
        if exp > 1:
            M = m["unified_size"]
            # Reshape source ``(M, exp, b1, b2)`` to a per-block-axis layout
            # ``(M, exp, 1, b1, b2)`` so a single ``where(eye_mask, value, 0)``
            # produces the diagonal-scattered ``(M, exp, exp, b1, b2)`` form,
            # then reshape to ``(M, cb1, cb2)``. Avoids the per-``i`` Python
            # loop over ``out.at[...].set(...)`` (one HLO op per slice).
            v = value.reshape(M, exp, 1, b1, b2, *rem)
            mask_shape = [1, exp, exp, 1, 1] + [1] * len(rem)
            eye = jnp.eye(exp, dtype=jnp.bool_).reshape(mask_shape)
            v = jnp.where(eye, v, jnp.array(0, dtype=value.dtype))
            return v.transpose([0, 1, 3, 2, 4] + list(range(5, 5 + len(rem)))) \
                    .reshape(M, cb1, cb2, *rem)

    if value.shape != tuple(exp_shape):
        value = value.reshape(exp_shape)
    if needs_expansion:
        mask = None
        for i, m in enumerate(metrics):
            b1 = m["left_b1"] if is_left else m["right_b1"]
            b2 = m["left_b2"] if is_left else m["right_b2"]
            exp_h = m["common_b1"] // b1
            exp_w = m["common_b2"] // b2
            if exp_h > 1:
                # Build the diagonal selector from independent per-axis eyes
                # (``logical_and`` of the two), so the cross-axis structure is
                # explicit even though the well-formed invariant pins
                # ``exp_h == exp_w``.
                ms_h = [1] * len(exp_shape); ms_h[5 * i + 1] = exp_h; ms_h[5 * i + 2] = exp_h
                ms_w = [1] * len(exp_shape); ms_w[5 * i + 1] = exp_w; ms_w[5 * i + 2] = exp_w
                eye_h = jnp.eye(exp_h, dtype=jnp.bool_).reshape(ms_h)
                eye_w = jnp.eye(exp_w, dtype=jnp.bool_).reshape(ms_w)
                em = jnp.logical_and(eye_h, eye_w)
                mask = em if mask is None else mask & em
        if mask is not None:
            value = jnp.where(mask, value, jnp.array(0, dtype=value.dtype))
    perm = generate_block_permutation(len(metrics), 5, [0, 1, 3, 2, 4])
    perm.extend(range(5 * len(metrics), len(exp_shape)))
    if perm != list(range(len(perm))):
        value = value.transpose(perm)
    if value.shape != tuple(out_shape):
        value = value.reshape(out_shape)
    return value


def _demote_intersection(value, metrics, is_intersection):
    if not is_intersection:
        return value, [[m["unified_size"], m["common_b1"], m["common_b2"]] for m in metrics]
    in_shape, out_shape, sum_axes, meta = [], [], [], []
    off = 0
    for m in metrics:
        mb1, mb2 = min(m["left_b1"], m["right_b1"]), min(m["left_b2"], m["right_b2"])
        dex = m["common_b1"] // mb1
        in_shape.extend([m["unified_size"], dex, mb1, dex, mb2])
        out_shape.extend([m["unified_size"] * dex, mb1, mb2])
        if dex > 1:
            sum_axes.append(off + 3)
        off += 5
        meta.append([m["unified_size"] * dex, mb1, mb2])
    rem = list(value.shape[3 * len(metrics):])
    in_shape += rem; out_shape += rem
    if sum_axes:
        value = value.reshape(in_shape).sum(axis=tuple(sum_axes)).reshape(out_shape)
    return value, meta


def _reconstruct_dim_pair(i, pair, meta, info):
    sz, b1, b2 = meta[i]
    def alloc(dim_size, squeeze_pos):
        if dim_size > 1:
            ax = info["axis"]; info["axis"] += 1; return ax
        info["squeeze"].append(squeeze_pos); return None
    v_ax, b1_ax, b2_ax = alloc(sz, 3 * i), alloc(b1, 3 * i + 1), alloc(b2, 3 * i + 2)
    return ((pair[0].id, replace(pair[0], size=sz, block_size=b1 if b1 > 1 else None,
                                 axis=v_ax, block_axis=b1_ax)),
            (pair[1].id, replace(pair[1], size=sz, block_size=b2 if b2 > 1 else None,
                                 axis=v_ax, block_axis=b2_ax)))


def _reconstruct_result(value, lhs, sp, dp, output_meta, op, rhs):
    from graphax.sparse.tensor import SparseTensor
    rec, info = {}, {"axis": 0, "squeeze": []}
    for i, pair in enumerate(sp):
        p1, p2 = _reconstruct_dim_pair(i, pair, output_meta, info)
        rec[p1[0]], rec[p2[0]] = p1[1], p2[1]
    for pair in dp:
        rec[pair[0].id] = replace(pair[0], axis=info["axis"]); info["axis"] += 1
    if info["squeeze"]:
        idx = tuple(0 if ax in info["squeeze"] else slice(None) for ax in range(value.ndim))
        value = value[idx]
    s_mult = jnp.array(True) if value.dtype == jnp.bool_ else jnp.array(1.0, dtype=value.dtype)
    new_fill = op(_scaled_fill(lhs), _scaled_fill(rhs))
    # Propagate the static zero-fill flag when both inputs have zero fill and
    # ``op(0, 0) == 0`` — otherwise downstream matmuls drop to the densify
    # fallback. We can't probe ``op`` numerically inside jit (any jax-side
    # call produces a tracer regardless of operand concreteness), so we lean
    # on a hard-coded set of zero-preserving ops covering the canonical
    # elementwise primitives. Anything else: leave ``zf=None`` (constructor
    # auto-detects, conservatively reporting ``False`` for tracer fills).
    zf = None
    lhs_zf = getattr(lhs, "_zero_fill", False)
    rhs_zf = getattr(rhs, "_zero_fill", False)
    if lhs_zf and rhs_zf and op in _ZERO_PRESERVING_OPS:
        zf = True
    return SparseTensor(
        tuple(rec[d.id] for d in lhs.out_dims),
        tuple(rec[d.id] for d in lhs.primal_dims),
        value, scalar_mult=s_mult,
        fill_value=new_fill,
        check_consistency=False,
        zero_fill=zf,
    )


def _try_divisor_fast_path(lhs, rhs, op, is_intersection):
    """Single-buffer fast path for the *divisor* misaligned-block case: one
    side's ``block_size`` equals the LCM along both axes (so its blocks are
    whole meta-blocks), and the other side's blocks are sub-blocks that fit
    on the meta-block-diagonal. The reference hand-coded path is
    ``manual_03``: pre-allocate one buffer at the big side's granularity,
    scatter the smaller side's diagonal sub-blocks into it, then ``op`` with
    the big side. Total HBM = one buffer the size of the output.

    The default LCM-promotion path here would instead materialize *both*
    sides at ``(M, LCM_h, LCM_w)`` *and* the per-sub-block intermediate the
    broadcast+where dance produces, doubling-or-worse the peak HBM
    footprint. This path matches the reference: peak HBM = output size,
    no intermediates beyond per-slice scratch (which XLA fuses).

    Conditions that fire it:
      - 2-D inputs, both single-sparse-pair, ``op`` zero-preserving.
      - Block sizes such that one side has ``block_size == LCM`` along both
        axes; the other has both axes evenly divisible by LCM (the *divisor*
        case — coprime / shared-factor cases skip).
      - ``exp_h == exp_w`` (block-diagonal sub-cell layout, the only one
        ``_promote_to_unified``'s eye mask actually handles correctly).

    Returns the result ``SparseTensor`` or ``None`` if conditions don't apply.
    """
    # Need ``op(0, x) == x`` (additive identity) — the path leaves big_v
    # untouched at off-small-support positions, which is correct only when
    # the missing side multiplies/adds to identity. ``mul`` zeros out, so
    # it's excluded; same for non-additive ``min``/``max``/``and``.
    if is_intersection or op not in _ADDITIVE_IDENTITY_OPS:
        return None
    if len(lhs.dims) != 2 or len(rhs.dims) != 2:
        return None
    if lhs.val is None or rhs.val is None:
        return None
    # The fast path assumes one out_dim + one primal_dim per side. Reject
    # 2-D tensors that put both dims on the same side (e.g. ``out=(),
    # primal=(d, d)``) — those won't satisfy the sparse-pair structural
    # checks below anyway, but indexing them here would IndexError.
    if not (lhs.out_dims and lhs.primal_dims and rhs.out_dims and rhs.primal_dims):
        return None
    ao, ai = lhs.out_dims[0], lhs.primal_dims[0]
    bo, bi = rhs.out_dims[0], rhs.primal_dims[0]
    if not all(d.is_sparse for d in (ao, ai, bo, bi)):
        return None
    if ao.other_id != ai.id or bo.other_id != bi.id:
        return None
    # All val axes must be materialized — otherwise the (M, B_h, B_w) reshape
    # below misalignes implicit / broadcast axes. Defer to the general path.
    if any(d.axis is None for d in (ao, ai, bo, bi)):
        return None
    if any(d.block_axis is None for d in (ao, ai, bo, bi)
           if d.block_size is not None and d.block_size > 1):
        return None
    # Physical val-axis order must be (outer, block_h, block_w) — i.e. axes
    # ``(0, 1, 2)`` — for the ``reshape(M, exp, b_h, b_w)`` step below to be
    # semantically correct. A different physical order would silently scramble
    # rows vs cols. Defer to the general path on a non-canonical layout rather
    # than transposing here (the general path handles arbitrary axis orders).
    canonical = lambda o, p: (o.axis, o.block_axis, p.block_axis) == (0, 1, 2)
    if not (canonical(ao, ai) and canonical(bo, bi)):
        return None
    a_b_h, a_b_w = ao.block_size or 1, ai.block_size or 1
    b_b_h, b_b_w = bo.block_size or 1, bi.block_size or 1
    a_n, b_n = ao.size, bo.size
    if a_n * a_b_h != b_n * b_b_h or a_n * a_b_w != b_n * b_b_w:
        return None
    lcm_h = math.lcm(a_b_h, b_b_h)
    lcm_w = math.lcm(a_b_w, b_b_w)
    a_is_big = (a_b_h == lcm_h and a_b_w == lcm_w)
    b_is_big = (b_b_h == lcm_h and b_b_w == lcm_w)
    if a_is_big == b_is_big:
        return None  # Either both equal-block (no expansion needed) or both need expansion.

    # Pick the big side as base; sub-block grid lives on the small side.
    if a_is_big:
        big, small = lhs, rhs
        s_b_h, s_b_w = b_b_h, b_b_w
    else:
        big, small = rhs, lhs
        s_b_h, s_b_w = a_b_h, a_b_w
    # Off-diagonal cells of each meta-block stay at the literal ``big_v`` value
    # because we represent the small side's missing entries as ``small_fill``
    # and use the additive-identity property ``op(0, x) == x``. That argument
    # only holds when ``small_fill`` is statically known to be 0; with a
    # non-zero fill the off-diagonal cell would correctly be
    # ``op(small_fill, big_v)`` and we'd lose the single-buffer save. Defer.
    if not _is_zero_fill(small):
        return None
    exp_h = lcm_h // s_b_h
    exp_w = lcm_w // s_b_w
    if exp_h != exp_w:
        return None  # Only block-diagonal layout supported (matches existing eye mask).
    exp = exp_h
    M = big.out_dims[0].size

    # Absorb scalar_mults eagerly so the output's scalar_mult is canonical.
    bool_op = lhs.dtype == jnp.bool_
    big_v = big.val & big.scalar_mult.astype(jnp.bool_) if bool_op \
        else big.val * big.scalar_mult
    small_v = small.val & small.scalar_mult.astype(jnp.bool_) if bool_op \
        else small.val * small.scalar_mult

    # Single-buffer build via ``where(eye, op(big_diag, small), big)``:
    #   * View ``big_v`` as a per-meta-block grid ``(M, exp_r, exp_c, b_h, b_w)``.
    #   * View ``small_v`` as one block per meta-block-diagonal position
    #     ``(M, exp, 1, b_h, b_w)`` (broadcasts trivially across the col-meta axis).
    #   * Apply ``op`` between the two — XLA's broadcast yields a full
    #     ``(M, exp, exp, b_h, b_w)`` op-applied tensor at trace time.
    #   * Use ``eye(exp)`` to pick the op'd value on the diagonal cells and
    #     the bare ``big`` value elsewhere — off-diagonal cells = ``big_v``
    #     (correct because ``small_fill == 0`` and ``op(big, 0) == big``).
    # Replaces the per-``i`` Python ``at[...].set(...)`` loop with one fused HLO.
    big_grid = big_v.reshape(M, exp, s_b_h, exp, s_b_w).transpose(0, 1, 3, 2, 4)
    small_diag = small_v.reshape(M, exp, 1, s_b_h, s_b_w)
    if big is lhs:
        op_applied = op(big_grid, small_diag)
    else:
        op_applied = op(small_diag, big_grid)
    eye = jnp.eye(exp, dtype=jnp.bool_).reshape(1, exp, exp, 1, 1)
    out = jnp.where(eye, op_applied, big_grid)
    out = out.transpose(0, 1, 3, 2, 4).reshape(M, exp * s_b_h, exp * s_b_w)

    new_fill = op(_scaled_fill(big), _scaled_fill(small)) if big is lhs \
        else op(_scaled_fill(small), _scaled_fill(big))
    s_mult = jnp.array(True) if bool_op else jnp.array(1.0, dtype=lhs.val.dtype)
    zf = (getattr(lhs, "_zero_fill", False) and getattr(rhs, "_zero_fill", False)) or None

    from graphax.sparse.tensor import SparseTensor
    return SparseTensor(
        big.out_dims, big.primal_dims, out,
        scalar_mult=s_mult, fill_value=new_fill,
        check_consistency=False, zero_fill=zf,
    )


def _try_compressed_union(lhs, rhs, op, is_intersection):
    """Fast path for misaligned 2-D block-diagonal union ops: output the meta-
    block-diagonal sum *lazily* as ``compressed_val=UnionBlocks(...)`` rather
    than eagerly building the ``(M, LCM_h, LCM_w)`` materialized form. Saves
    storage from ``M·LCM_h·LCM_w`` to ``M·(n_lhs·B_lhs² + n_rhs·B_rhs²)`` —
    a meaningful win whenever the source blocks are sufficiently misaligned
    (``B_a + B_b < lcm(B_a, B_b)``, e.g. user's coprime 5/11 ⇒ 3.4× tighter).

    The expression that goes into the JAXPR is identical to the eager path
    (``UnionBlocks.to_meta_blocks`` is the same broadcast+select+sum chain
    that the eager promote-to-unified used), so XLA fuses both into the same
    consumer kernel — same latency, M× less HBM footprint between ops.

    Returns the lazy ``SparseTensor`` or ``None`` if conditions don't apply.
    """
    if is_intersection or op not in _ZERO_PRESERVING_OPS:
        return None
    if len(lhs.dims) != 2 or len(rhs.dims) != 2:
        return None
    if lhs.val is None or rhs.val is None:
        return None
    # The fast path assumes one out_dim + one primal_dim per side (the sparse
    # pair straddles the split). 2-D tensors with both dims on the same side
    # don't satisfy the structural checks below; reject them up-front so we
    # don't IndexError on the indexing.
    if not (lhs.out_dims and lhs.primal_dims and rhs.out_dims and rhs.primal_dims):
        return None
    ao, ai = lhs.out_dims[0], lhs.primal_dims[0]
    bo, bi = rhs.out_dims[0], rhs.primal_dims[0]
    if not all(d.is_sparse for d in (ao, ai, bo, bi)):
        return None
    # Sparse pair siblings must match on each side.
    if ao.other_id != ai.id or ai.other_id != ao.id: return None
    if bo.other_id != bi.id or bi.other_id != bo.id: return None
    # All val axes must be materialized. ``axis=None`` means the sib outer
    # is an implicit broadcast (val is missing that axis); the reshape below
    # assumes ``val.shape == (N, B_h, B_w)`` so we'd fail on a smaller-rank val.
    # ``block_axis=None`` similarly means the block axis is broadcast — the
    # ``UnionBlocks.to_meta_blocks`` path expects fully materialized blocks.
    if any(d.axis is None for d in (ao, ai, bo, bi)):
        return None
    if any(d.block_axis is None for d in (ao, ai, bo, bi)
           if d.block_size is not None and d.block_size > 1):
        return None
    a_b_h, a_b_w = ao.block_size or 1, ai.block_size or 1
    b_b_h, b_b_w = bo.block_size or 1, bi.block_size or 1
    a_n, b_n = ao.size, bo.size
    if a_n * a_b_h != b_n * b_b_h or a_n * a_b_w != b_n * b_b_w:
        return None
    lcm_h = math.lcm(a_b_h, b_b_h)
    lcm_w = math.lcm(a_b_w, b_b_w)
    if (a_n * a_b_h) % lcm_h or (a_n * a_b_w) % lcm_w:
        return None
    M = (a_n * a_b_h) // lcm_h
    n_lhs, n_rhs = a_n // M, b_n // M
    # Storage comparison: only commit to compressed_val if it's strictly tighter
    # than the meta-block-diagonal val we'd otherwise emit. ``UnionBlocks`` is
    # tighter exactly when ``B_a_h·B_a_w·n_lhs + B_b_h·B_b_w·n_rhs < LCM_h·LCM_w``.
    union_size = n_lhs * a_b_h * a_b_w + n_rhs * b_b_h * b_b_w
    meta_size = lcm_h * lcm_w
    if union_size >= meta_size:
        return None

    # Absorb scalar_mults into the compressed buffers so the output's scalar_mult
    # is the canonical 1.0 / True (matches the eager path's ``s_mult`` choice).
    bool_op = lhs.dtype == jnp.bool_
    if bool_op:
        lhs_v = lhs.val & lhs.scalar_mult.astype(jnp.bool_)
        rhs_v = rhs.val & rhs.scalar_mult.astype(jnp.bool_)
    else:
        lhs_v = lhs.val * lhs.scalar_mult
        rhs_v = rhs.val * rhs.scalar_mult
    fill_l = _scaled_fill(lhs)
    fill_r = _scaled_fill(rhs)

    lhs_blocks = lhs_v.reshape(M, n_lhs, a_b_h, a_b_w)
    rhs_blocks = rhs_v.reshape(M, n_rhs, b_b_h, b_b_w)

    from .block_storage import UnionBlocks
    ub = UnionBlocks(lhs=lhs_blocks, rhs=rhs_blocks,
                     fill_lhs=fill_l, fill_rhs=fill_r, op=op)
    new_fill = op(fill_l, fill_r)
    s_mult = jnp.array(True) if bool_op else jnp.array(1.0, dtype=lhs.val.dtype)
    zf = (getattr(lhs, "_zero_fill", False) and getattr(rhs, "_zero_fill", False)) or None

    # Preserve source IDs from lhs's sparse pair — hardcoding ``0, 1`` produces
    # silent dim-id collisions when lhs sits inside a larger tensor whose IDs
    # already use 0/1 (the divisor fast path correctly reuses ``big.out_dims``
    # / ``big.primal_dims`` and so keeps the IDs; mirror that here).
    out_id, primal_id = lhs.out_dims[0].id, lhs.primal_dims[0].id

    from graphax.sparse.tensor import SparseTensor
    return SparseTensor(
        (SparseIndex(out_id, M, axis=0, other_id=primal_id,
                         block_size=lcm_h, block_axis=1),),
        (SparseIndex(primal_id, M, axis=0, other_id=out_id,
                         block_size=lcm_w, block_axis=2),),
        val=None, compressed_val=ub,
        scalar_mult=s_mult, fill_value=new_fill,
        check_consistency=False, zero_fill=zf,
    )


# --- Path tracing (test-only) ---------------------------------------------
# Re-exports from ``_path_tracking``. See that module for the full design;
# tests opt in via the ``track_paths()`` context manager or ``TRACK_PATHS=1``
# env var. Production runs pay nothing.
from ._path_tracking import record_path as _record_path, last_path  # noqa: E402, F401


def elementwise(
    lhs,
    rhs,
    op: Callable,
    is_intersection: bool = False,
    count: bool = False,
):
    """Sparse elementwise op dispatcher. Tries fast paths in priority order,
    falls back to the general expansion path.

    Dispatch order (first applicable wins):
      1. ``divisor_fast``     — one side's block_size == LCM (single-buffer
                                scatter+op on the big side; matches the
                                hand-coded reference for the divisor case).
      2. ``compressed_union`` — misaligned 2-D union with a zero-preserving
                                op: emits ``compressed_val=UnionBlocks``
                                (M× tighter HBM than eager LCM-grid form).
      3. ``general``          — full broadcast / promote-to-unified / op /
                                demote pipeline. Handles every other case
                                including aligned blocks, non-zero fills, and
                                operations that aren't zero-preserving.

    With ``count=True`` returns ``(result, n_ops)`` — the number of element
    positions where ``op`` actually fires, computed from the *static*
    broadcast shape of the inputs:

    * union ops (``add`` & co): every position of the broadcast shape
      contributes, ``n_ops = prod(broadcast(lhs.shape, rhs.shape))``.
    * intersection ops (``mul``): only positions where both sides have data
      contribute, ``n_ops = prod(min(lhs.shape, rhs.shape))`` (broadcast-
      paired axes; ``size 1`` collapses to the partner's extent for union
      semantics, but here it's the elementwise min of the aligned shape).

    ``add_w_counts`` / ``mul_w_counts`` package ``(out, n_ops)`` into the
    ``(adds, muls, fmas)`` triple the cost model expects.
    """
    _record_path(None)
    lhs, rhs = _normalize_inputs(lhs, rhs)
    if count:
        n = _ew_op_count(lhs, rhs, is_intersection)

    divisor = _try_divisor_fast_path(lhs, rhs, op, is_intersection)
    if divisor is not None:
        _record_path("divisor_fast")
        if count:
            return divisor, n
        return divisor
    compressed = _try_compressed_union(lhs, rhs, op, is_intersection)
    if compressed is not None:
        _record_path("compressed_union")
        if count:
            return compressed, n
        return compressed
    _record_path("general")
    sp, dp = _map_topology(lhs, rhs)
    metrics = [_pair_metric(p) for p in sp]
    vl, al, ul = _value_axes_info(lhs, sp, dp, True)
    vr, ar, ur = _value_axes_info(rhs, sp, dp, False)
    bus = list(jnp.broadcast_shapes(tuple(ul), tuple(ur)))
    vl = _align_value(vl, lhs, sp, dp, al, bus, True)
    vr = _align_value(vr, rhs, sp, dp, ar, bus, False)
    res = op(_promote_to_unified(vl, metrics, True), _promote_to_unified(vr, metrics, False))
    res, out_meta = _demote_intersection(res, metrics, is_intersection)
    out = _reconstruct_result(res, lhs, sp, dp, out_meta, op, rhs)
    if count:
        return out, n
    return out


def _ew_op_count(lhs, rhs, is_intersection: bool) -> int:
    """Element-wise op invocations, computed from logical input shapes.

    * union (``add`` etc.): output broadcast shape size — every element of
      the broadcast result is one ``op`` call.
    * intersection (``mul`` with sparse semantics): the elementwise min of
      the right-aligned shapes — positions where both sides carry data.
      For non-broadcast cases (``lhs.shape == rhs.shape``) this matches the
      output size; for broadcast (``size==1`` on one side) it stays at the
      smaller shape, reflecting that ``mul`` outside the intersection
      collapses to the fill_value and need not fire.
    """
    s_l = tuple(int(x) for x in getattr(lhs, "shape", ()) or ())
    s_r = tuple(int(x) for x in getattr(rhs, "shape", ()) or ())
    if not s_l and not s_r:
        return 1
    nd = max(len(s_l), len(s_r))
    a = (1,) * (nd - len(s_l)) + s_l
    b = (1,) * (nd - len(s_r)) + s_r
    reducer = min if is_intersection else max
    n = 1
    for x, y in zip(a, b):
        n *= int(reducer(int(x), int(y)))
    return n


def add_w_counts(a, b):
    """``elementwise(a, b, lax.add, count=True)`` packaged for the
    vertex-elimination cost model: returns ``(out, (adds, muls, fmas))``."""
    out, n = elementwise(a, b, jax.lax.add, count=True)
    return out, (n, 0, 0)


def mul_w_counts(a, b):
    """``elementwise(a, b, lax.mul, intersection, count=True)`` packaged for the
    vertex-elimination cost model: returns ``(out, (adds, muls, fmas))``."""
    out, n = elementwise(a, b, jax.lax.mul, is_intersection=True, count=True)
    return out, (0, n, 0)
