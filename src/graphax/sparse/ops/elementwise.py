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

from .utils import _arr2st, _is_sparse, _val_or_one, _prepare_physical_array, _is_zero_fill
from .layout import generate_block_permutation
from graphax.sparse.indexes import DiagonalIndex, DenseIndex

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


# Ops where ``op(0, 0) == 0`` — used to keep the statically-zero ``fill_value``
# marker (``None``) on an elementwise-of-zero-fills result. Hardcoded because
# jit-time numerical probing always promotes operands to tracers, even concrete
# numpy zeros, so we can't introspect ``op`` numerically inside the trace.
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
    """Post-scaled fill as a concrete array: ``fill * scalar_mult`` (or ``& mask``
    for bool), with a ``None`` (statically-zero) fill read as 0 via ``_eff_fill``.
    Canonical form used to compose output fills consistently — every fast path
    must produce a fill that matches the post-scaled meaning of the input
    operands so downstream consumers see one definition."""
    if tensor.dtype == jnp.bool_:
        return tensor._eff_fill & tensor.scalar_mult.astype(jnp.bool_)
    return tensor._eff_fill * tensor.scalar_mult


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
    # Phase 8: pre-densify any compressed Index dims (BandedIndex / SetIndex)
    # to DiagonalIndex / DenseIndex — elementwise consumes only those. XLA
    # fuses the densify into the consumer (SMEM, not HBM). No-op when the
    # operand carries no compressed dims.
    from .utils import _materialize_for_op
    lhs = _materialize_for_op(lhs)
    rhs = _materialize_for_op(rhs)
    # Static shape comparison: ``SparseTensor.shape`` returns Python ints
    # derived from the dim metadata, so this is a trace-time check (no runtime
    # branching on traced shapes).
    if lhs.shape != rhs.shape:
        raise ValueError(f"Shape mismatch: {lhs.shape} != {rhs.shape}")
    return lhs, rhs


def _promote_dense(d, partner_id):
    """Wrap a DenseIndex into a synthetic 1-block DiagonalIndex paired with `partner_id`."""
    return DiagonalIndex(d.id, 1, axis=None, other_id=partner_id,
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


def _promote_to_unified(value: Array, metrics, is_left: bool, fill: Array) -> Array:
    # ``fill`` is the operand's POST-scaled fill (``_scaled_fill``): ``value``
    # arrives already scaled by ``scalar_mult`` (``_align_value``), so the
    # off-diagonal LCM-grid cells — positions where this operand has no block,
    # i.e. its implicit fill — must hold the scaled fill, not a literal 0 (B3).
    # For the common zero-fill operand this is 0 (unchanged); for a non-zero
    # fill it makes ``op(promote(lhs), promote(rhs))`` produce the correct
    # ``op(fill_lhs, fill_rhs)`` off the diagonal. The cast is deferred to the
    # ``needs_expansion`` branches below — it is dead work when no axis expands.
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
    if needs_expansion:  # only the expansion branches read ``fill`` (B3)
        fill = jnp.asarray(fill, value.dtype)

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
            v = jnp.where(eye, v, fill)
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
            value = jnp.where(mask, value, fill)
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
    # Result fill: ``None`` (statically zero) when both inputs are statically
    # zero AND ``op(0, 0) == 0`` — so the output keeps fast-path eligibility.
    # We can't probe ``op`` numerically inside jit (any jax call yields a tracer
    # regardless of operand concreteness), so we lean on a hard-coded set of
    # zero-preserving ops covering the canonical elementwise primitives.
    # Otherwise compute the concrete combined fill (densify path downstream).
    if lhs.fill_value is None and rhs.fill_value is None and op in _ZERO_PRESERVING_OPS:
        new_fill = None
    else:
        new_fill = op(_scaled_fill(lhs), _scaled_fill(rhs))
    return SparseTensor(
        tuple(rec[d.id] for d in lhs.out_dims),
        tuple(rec[d.id] for d in lhs.primal_dims),
        value, scalar_mult=s_mult,
        fill_value=new_fill,
        check_consistency=False,
    )




# --- Phase 6b: DivisorRemainder emission (probe + helper) -----------------
# Inhabits the *general* dispatcher branch (path string stays "general").
# Mirrors the gating of ``_try_compressed_union`` so the new emission
# subsumes the dispatcher fast path; the ``include_remainder`` static flag
# is set conservatively (always True) — structural-identity detection
# (e.g., dropping a zero side) lands in a later refinement.
def _should_emit_divisor_remainder(lhs, rhs, op, is_intersection):
    """Static probe: should this elementwise op emit a compressed
    ``DivisorRemainder`` rather than eagerly building the meta-block-
    diagonal val? Returns a dict of geometry / IDs on success, ``None``
    otherwise.

    Gates mirror the legacy ``_try_compressed_union`` (deleted in 6b.3) and
    additionally cover ``is_intersection=True`` so multiplicative ops on
    misaligned 2-D block-diagonals get the same compression. The
    ``include_remainder`` field is always set to ``True`` for now —
    structural-identity detection (drop a provably-zero side) is a future
    refinement.
    """
    if op not in _ZERO_PRESERVING_OPS:
        return None
    # SetIndex carries no per-side fill (hashable aux_data), so densify uses
    # the output fill_value for both sides — exact only for zero fills
    # (op(0,0)=0). Non-zero-fill operands fall through to the general path.
    if not (_is_zero_fill(lhs) and _is_zero_fill(rhs)):
        return None
    if len(lhs.dims) != 2 or len(rhs.dims) != 2:
        return None
    if lhs.val is None or rhs.val is None:
        return None
    if not (lhs.out_dims and lhs.primal_dims and rhs.out_dims and rhs.primal_dims):
        return None
    ao, ai = lhs.out_dims[0], lhs.primal_dims[0]
    bo, bi = rhs.out_dims[0], rhs.primal_dims[0]
    if not all(d.is_sparse for d in (ao, ai, bo, bi)):
        return None
    if ao.other_id != ai.id or ai.other_id != ao.id:
        return None
    if bo.other_id != bi.id or bi.other_id != bo.id:
        return None
    if any(d.axis is None for d in (ao, ai, bo, bi)):
        return None
    if any(
        d.block_axis is None
        for d in (ao, ai, bo, bi)
        if d.block_size is not None and d.block_size > 1
    ):
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
    union_size = n_lhs * a_b_h * a_b_w + n_rhs * b_b_h * b_b_w
    meta_size = lcm_h * lcm_w
    if union_size >= meta_size:
        return None
    return {
        "M": M,
        "n_lhs": n_lhs,
        "n_rhs": n_rhs,
        "a_b_h": a_b_h,
        "a_b_w": a_b_w,
        "b_b_h": b_b_h,
        "b_b_w": b_b_w,
        "lcm_h": lcm_h,
        "lcm_w": lcm_w,
        "semantic": "intersection" if is_intersection else "union",
        "include_remainder": True,
        "out_id": lhs.out_dims[0].id,
        "primal_id": lhs.primal_dims[0].id,
    }


def _emit_divisor_remainder(lhs, rhs, op, geom):
    """Construct a ``SparseTensor`` whose dims are a ``SetIndex`` pair and whose
    ``val`` is the two per-side block buffers concatenated into a single 1-D
    Array (so ``val`` stays a plain Array — no constructor / unary-op surgery).
    Geometry comes from :func:`_should_emit_divisor_remainder`."""
    from graphax.sparse.indexes import SetIndex
    from graphax.sparse.tensor import SparseTensor

    M = geom["M"]
    n_lhs, n_rhs = geom["n_lhs"], geom["n_rhs"]
    a_b_h, a_b_w = geom["a_b_h"], geom["a_b_w"]
    b_b_h, b_b_w = geom["b_b_h"], geom["b_b_w"]
    lcm_h, lcm_w = geom["lcm_h"], geom["lcm_w"]

    bool_op = lhs.dtype == jnp.bool_
    if bool_op:
        lhs_v = lhs.val & lhs.scalar_mult.astype(jnp.bool_)
        rhs_v = rhs.val & rhs.scalar_mult.astype(jnp.bool_)
    else:
        lhs_v = lhs.val * lhs.scalar_mult
        rhs_v = rhs.val * rhs.scalar_mult

    lhs_shape = (M, n_lhs, a_b_h, a_b_w)
    rhs_shape = (M, n_rhs, b_b_h, b_b_w)
    combined = jnp.concatenate([lhs_v.reshape(-1), rhs_v.reshape(-1)])

    s_mult = jnp.array(True) if bool_op else jnp.array(1.0, dtype=lhs.val.dtype)
    # Emitted only when both operands are statically zero-fill (gated upstream),
    # so the output is statically zero-fill too: fill_value=None.
    out_id, primal_id = geom["out_id"], geom["primal_id"]

    out_ix = SetIndex(
        id=out_id, size=M, axis=0, other_id=primal_id, block_size=lcm_h, block_axis=1,
        semantic=geom["semantic"], lhs_shape=lhs_shape, rhs_shape=rhs_shape,
        include_remainder=geom["include_remainder"], n_meta=1, op=op,
    )
    primal_ix = SetIndex(
        id=primal_id, size=M, axis=0, other_id=out_id, block_size=lcm_w, block_axis=2,
        semantic=geom["semantic"], lhs_shape=lhs_shape, rhs_shape=rhs_shape,
        include_remainder=geom["include_remainder"], n_meta=1, op=op,
    )
    return SparseTensor(
        (out_ix,), (primal_ix,), combined,
        scalar_mult=s_mult, fill_value=None,
        check_consistency=False,
    )


# --- Phase 9: K≥2 multi-axis SetIndex emission ----------------------------
# A misaligned elementwise op on two operands that are each block-diagonal
# along K≥2 sparse pairs compresses to a SetIndex pair *per axis* (2K SetIndex
# dims) with the two operands' compact block buffers stored as W=1 multi-banded
# buffers. Densify reuses ``_densify_multi_banded`` (band_width=1 ⇒ block-
# diagonal) per side then applies the op — see ``utils._densify_compressed_dims``.
def _should_emit_multi_set(lhs, rhs, op, is_intersection):
    """Static probe for the K≥2 generalization of
    :func:`_should_emit_divisor_remainder`. Returns per-axis geometry on
    success (storing the compact dual buffers beats the dense LCM grid), else
    ``None``. Same zero-fill / zero-preserving gating as the K=1 probe."""
    if op not in _ZERO_PRESERVING_OPS:
        return None
    if not (_is_zero_fill(lhs) and _is_zero_fill(rhs)):
        return None
    if lhs.val is None or rhs.val is None:
        return None
    K = len(lhs.out_dims)
    if K < 2:
        return None
    if (len(lhs.primal_dims) != K or len(rhs.out_dims) != K
            or len(rhs.primal_dims) != K
            or len(lhs.dims) != 2 * K or len(rhs.dims) != 2 * K):
        return None

    pairs = []
    lhs_buf_size = rhs_buf_size = 1
    meta_size = 1  # compact meta-block-diagonal size = prod(M_i·lcm_h_i·lcm_w_i)
    for i in range(K):
        ao, ai = lhs.out_dims[i], lhs.primal_dims[i]
        bo, bi = rhs.out_dims[i], rhs.primal_dims[i]
        if not all(d.is_sparse for d in (ao, ai, bo, bi)):
            return None
        if ao.other_id != ai.id or ai.other_id != ao.id:
            return None
        if bo.other_id != bi.id or bi.other_id != bo.id:
            return None
        if any(d.axis is None for d in (ao, ai, bo, bi)):
            return None
        a_b_h, a_b_w = ao.block_size or 1, ai.block_size or 1
        b_b_h, b_b_w = bo.block_size or 1, bi.block_size or 1
        if any(d.block_axis is None for d in (ao, ai, bo, bi)
               if (d.block_size or 1) > 1):
            return None
        a_n, b_n = ao.size, bo.size
        if a_n * a_b_h != b_n * b_b_h or a_n * a_b_w != b_n * b_b_w:
            return None
        lcm_h, lcm_w = math.lcm(a_b_h, b_b_h), math.lcm(a_b_w, b_b_w)
        if (a_n * a_b_h) % lcm_h or (a_n * a_b_w) % lcm_w:
            return None
        M = (a_n * a_b_h) // lcm_h
        # Symmetric per-side geometry: pair["a"]/pair["b"] each carry the
        # operand's M axis, block axes (None when block_size==1), and sizes,
        # so the packer indexes p[side][...] uniformly.
        pairs.append({
            "a": {"axis": ao.axis, "bh_axis": ao.block_axis, "bw_axis": ai.block_axis,
                  "n": a_n, "b_h": a_b_h, "b_w": a_b_w},
            "b": {"axis": bo.axis, "bh_axis": bo.block_axis, "bw_axis": bi.block_axis,
                  "n": b_n, "b_h": b_b_h, "b_w": b_b_w},
            "lcm_h": lcm_h, "lcm_w": lcm_w, "M": M,
            "out_id": ao.id, "primal_id": ai.id,
        })
        lhs_buf_size *= a_n * a_b_h * a_b_w
        rhs_buf_size *= b_n * b_b_h * b_b_w
        meta_size *= M * lcm_h * lcm_w
    # The packer reshapes the operand val purely from its M / block axes, so the
    # val must have no leftover (L) axes the perm wouldn't cover.
    def _phys(side):
        return K + sum(p[side]["bh_axis"] is not None for p in pairs) \
            + sum(p[side]["bw_axis"] is not None for p in pairs)
    if lhs.val.ndim != _phys("a") or rhs.val.ndim != _phys("b"):
        return None
    # Restrict to a single meta-block per axis (M_i == 1). For M_i > 1 the
    # general path already emits a *compact* meta-block-diagonal that ops consume
    # directly; a SetIndex there would have to fully materialize (prod(M_i)×) at
    # every op boundary — the K≥2 densify has no compact meta form yet — so it
    # would be a boundary pessimization. The pure-win case (M_i == 1, where
    # compact ≡ full) is what we compress.
    if any(p["M"] != 1 for p in pairs):
        return None
    # Only compress when the dual buffer beats the *compact* meta-block-diagonal
    # the general path would otherwise emit (NOT the prod(M_i)×-larger full dense).
    if lhs_buf_size + rhs_buf_size >= meta_size:
        return None
    return {
        "K": K, "pairs": pairs,
        "semantic": "intersection" if is_intersection else "union",
    }


def _emit_multi_set(lhs, rhs, op, geom):
    """Build the K≥2 multi-axis ``SetIndex`` output: each operand's block
    structure is packed into a W=1 multi-banded buffer; the two buffers are
    concatenated into a single 1-D ``val`` and described by 2K ``SetIndex``
    dims. Densify reuses ``_densify_multi_banded`` per side then applies op."""
    from graphax.sparse.indexes import SetIndex
    from graphax.sparse.tensor import SparseTensor

    K = geom["K"]
    pairs = geom["pairs"]
    bool_op = lhs.dtype == jnp.bool_
    if bool_op:
        lhs_v = lhs.val & lhs.scalar_mult.astype(jnp.bool_)
        rhs_v = rhs.val & rhs.scalar_mult.astype(jnp.bool_)
    else:
        lhs_v = lhs.val * lhs.scalar_mult
        rhs_v = rhs.val * rhs.scalar_mult

    def _pack(v, side):
        # Permute operand val to (M_0..M_{K-1}, [existing Bh], [existing Bw]),
        # then reshape to the W=1 multi-banded layout (M_0,1,M_1,1,...,Bh*,Bw*).
        # A trivial (block_size==1) pair has no physical block axis, so it is
        # skipped in the perm and re-inserted as a size-1 dim by the reshape.
        g = [p[side] for p in pairs]
        perm = ([s["axis"] for s in g]
                + [s["bh_axis"] for s in g if s["bh_axis"] is not None]
                + [s["bw_axis"] for s in g if s["bw_axis"] is not None])
        t = v.transpose(perm)
        band_shape = []
        for s in g:
            band_shape += [s["n"], 1]
        band_shape += [s["b_h"] for s in g] + [s["b_w"] for s in g]
        return t.reshape(band_shape)

    lhs_band = _pack(lhs_v, "a")
    rhs_band = _pack(rhs_v, "b")
    combined = jnp.concatenate([lhs_band.reshape(-1), rhs_band.reshape(-1)])

    s_mult = jnp.array(True) if bool_op else jnp.array(1.0, dtype=lhs.val.dtype)
    # Gated on both operands statically zero-fill → output is too (fill_value=None).
    lhs_shape, rhs_shape = lhs_band.shape, rhs_band.shape
    sem = geom["semantic"]

    out_dims, primal_dims = [], []
    for i, p in enumerate(pairs):
        out_dims.append(SetIndex(
            id=p["out_id"], size=p["M"], axis=i, other_id=p["primal_id"],
            block_size=p["lcm_h"], block_axis=2 * K + i, semantic=sem,
            lhs_shape=lhs_shape, rhs_shape=rhs_shape, include_remainder=True,
            n_meta=1, op=op,
        ))
        primal_dims.append(SetIndex(
            id=p["primal_id"], size=p["M"], axis=K + i, other_id=p["out_id"],
            block_size=p["lcm_w"], block_axis=3 * K + i, semantic=sem,
            lhs_shape=lhs_shape, rhs_shape=rhs_shape, include_remainder=True,
            n_meta=1, op=op,
        ))
    return SparseTensor(
        tuple(out_dims), tuple(primal_dims), combined,
        scalar_mult=s_mult, fill_value=None,
        check_consistency=False,
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
    """Sparse elementwise op dispatcher.

    Single ``general`` path that handles every case: misaligned 2-D
    block-diagonal union AND intersection emissions land as a ``SetIndex``
    pair (the combined per-side block buffer in ``val``) early in the path;
    aligned blocks, non-zero fills, broadcast cases, and non-zero-preserving
    ops fall through the full promote-to-unified pipeline. The
    ``compressed_union`` dispatcher branch (Phase 6b.3) was folded into this
    general path; its path label is gone. The ``divisor_fast`` dispatcher
    branch (Phase 6a) was deleted as HLO-redundant.

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

    _record_path("general")
    # Phase 6b: DivisorRemainder emission for both union and intersection ops
    # on misaligned 2-D block-diagonals. Subsumes the deleted dispatcher
    # ``_try_compressed_union`` branch and additionally compresses
    # intersection (mul/etc.) outputs that the dispatcher never handled.
    _dr_geom = _should_emit_divisor_remainder(lhs, rhs, op, is_intersection)
    if _dr_geom is not None:
        out = _emit_divisor_remainder(lhs, rhs, op, _dr_geom)
        if count:
            return out, n
        return out
    # Phase 9: K≥2 generalization — misaligned elementwise on operands with
    # multiple sparse pairs compresses to a multi-axis SetIndex (dual block
    # buffers), densified by reusing the W=1 multi-banded kernel per side.
    _ms_geom = _should_emit_multi_set(lhs, rhs, op, is_intersection)
    if _ms_geom is not None:
        out = _emit_multi_set(lhs, rhs, op, _ms_geom)
        if count:
            return out, n
        return out
    try:
        sp, dp = _map_topology(lhs, rhs)
    except ValueError as e:
        if "Topology mismatch" not in str(e):
            raise
        # The two operands are logically the same shape but carry incompatible
        # sparse encodings — e.g. two logically-equal broadcast Jacobians where
        # one path produced an extra diagonal pair (the duplicate
        # broadcast_in_dim pattern). There is no aligned sparse form to combine
        # them in, so densify both to a common dense representation and combine
        # there (the architectural boundary fallback). Correct, just not
        # compressed for this op.
        from .dense import dense as _dense
        out = elementwise(
            _dense(lhs, hard=True), _dense(rhs, hard=True), op,
            is_intersection=is_intersection,
        )
        if count:
            return out, n
        return out
    metrics = [_pair_metric(p) for p in sp]
    vl, al, ul = _value_axes_info(lhs, sp, dp, True)
    vr, ar, ur = _value_axes_info(rhs, sp, dp, False)
    bus = list(jnp.broadcast_shapes(tuple(ul), tuple(ur)))
    vl = _align_value(vl, lhs, sp, dp, al, bus, True)
    vr = _align_value(vr, rhs, sp, dp, ar, bus, False)
    res = op(_promote_to_unified(vl, metrics, True, _scaled_fill(lhs)),
             _promote_to_unified(vr, metrics, False, _scaled_fill(rhs)))
    # The intersection demote SUMS over the LCM-expansion axis, which is only
    # valid when the off-intersection sub-blocks are zero — i.e. zero fill (for
    # a multiplicative op, fill·data vanishes only when a fill is 0). With a
    # non-zero fill the B3 off-diagonal fill written by _promote_to_unified
    # would be summed in spuriously, so fall back to the union-style
    # reconstruction (no sum), which yields the correct full elementwise result.
    eff_intersection = (
        is_intersection and _is_zero_fill(lhs) and _is_zero_fill(rhs)
    )
    res, out_meta = _demote_intersection(res, metrics, eff_intersection)
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
