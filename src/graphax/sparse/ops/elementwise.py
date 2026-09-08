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
import collections
import math
import os
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .utils import (
    _arr2st, _is_sparse, _val_or_one, _prepare_physical_array, _is_zero_fill,
    _apply_scalar_mult, _scaled_fill, _copy,
)
from .layout import generate_block_permutation
from graphax.sparse.dtype_compute import _unify_operand_dtypes, _compute_dtype
from graphax.sparse.indexes import DiagonalIndex, DenseIndex, static_eye

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


def _identity_scalar_mult(dtype) -> Array:
    """Identity ``scalar_mult`` for a freshly-built result buffer: ``True`` for
    bool, else ``1.0`` in the result dtype (the values already carry the scale)."""
    return jnp.array(True) if dtype == jnp.bool_ else jnp.array(1.0, dtype=dtype)


def _reconcile_broadcast_dims(lhs, rhs):
    """Reconcile a compact broadcast dim (``logical_size == 1``, i.e. the
    "extent N stored once" that an approximation collapsed on ONE fan-in
    contribution) against its same-``id`` MATERIALIZED partner (``logical_size
    N > 1``) BEFORE the physical-shape equality check, so accumulating two
    contributions to one vertex no longer raises ``Shape mismatch``.

    Fan-in accumulation (``cf[t] + prod`` in the face engine) adds two partial
    Jacobians of the SAME edge — dims are paired by ``id`` (elementwise operands
    share ONE id space). When a Compress/Diag approximation collapses a dense
    free dim on one contribution to its compact size-1 form while the sibling
    kept it materialized at extent N, the two operands have equal-``id`` dims of
    logical extent 1 vs N. The add of extent-1 and extent-N is exactly the
    extent-N broadcast (numpy/`op` semantics; verified byte-exact against the
    dense oracle), so promote the size-1 side to N here.

    INVARIANT (guarded): broadcast a size-1 dim ONLY toward a same-``id`` partner
    whose ``logical_size > 1``. A genuine ``logical_size == 1`` on BOTH sides has
    no such partner and is left untouched (``1`` stays ``1``) — we never fabricate
    an extent. Only the DENSE size-1 side is ever broadcast (``not d.is_sparse``);
    we never broadcast the sparse side of a pair.

    The same-``id`` PARTNER may itself be SPARSE (a diagonal side). ViT fan-in
    accumulates a Compress/Diag-collapsed dense free dim (logical 1) against a
    sibling where that dim is the DIAGONAL partner of extent N. Broadcasting the
    dense-1 side up to a materialized dense N is still exactly the ``op`` add: the
    diagonal contribution (nonzero on i==j) plus the constant-broadcast dense
    contribution sums to a DENSE result in that dim (diagonal + full = full,
    losing the diagonal structure — correct and unavoidable). The downstream
    ``_map_topology`` / ``_promote_to_unified`` path already materializes the
    diagonal onto the LCM grid (0 off-diagonal) and adds the promoted full block,
    so no new math is needed here — only the reconcile of the shrunk dense dim.

    Two SPARSE dims of equal logical size but different block granularity are a
    real LCM-grid job (both sides sparse) and stay with the downstream path.
    """
    # Reconcile can only broadcast a DENSE logical-1 dim toward a materialized
    # N-partner; when neither operand has such a dim it is a strict no-op, so
    # skip the dict build + per-dim scan entirely (the common case in the loop).
    def _has_dense_unit(t):
        return any((not d.is_sparse) and int(d.logical_size) == 1 for d in t.dims)
    if not _has_dense_unit(lhs) and not _has_dense_unit(rhs):
        return lhs, rhs

    l_map = {d.id: d for d in lhs.dims}
    r_map = {d.id: d for d in rhs.dims}

    def _fix(t, other_map):
        val = t.val
        changed = False
        new_by_id = {}
        for d in t.dims:
            od = other_map.get(d.id)
            # Broadcast the DENSE size-1 side (``not d.is_sparse``) toward its
            # same-``id`` partner of logical N>1. The partner ``od`` may be DENSE
            # (the original fan-in case) OR SPARSE (a diagonal side — ViT); in the
            # sparse-partner case the downstream promote path densifies the
            # diagonal against this now-materialized dense dim (diagonal+full=full).
            if (od is not None and not d.is_sparse
                    and int(d.logical_size) == 1 and int(od.logical_size) > 1):
                N = int(od.logical_size)
                if d.axis is not None and val is not None and val.shape[d.axis] == 1:
                    # Materialized compact axis: physically broadcast it to N.
                    val = jnp.broadcast_to(
                        val, val.shape[:d.axis] + (N,) + val.shape[d.axis + 1:]
                    )
                    new_by_id[d.id] = replace(d, size=N)
                else:
                    # No physical size-1 axis to grow (implicit / val is None):
                    # carry the true logical extent as an implicit (axis=None)
                    # broadcast dim; the downstream align/promote path expands it.
                    new_by_id[d.id] = replace(d, size=N, axis=None)
                changed = True
            else:
                new_by_id[d.id] = d
        if not changed:
            return t
        # _copy carries scalar_mult / fill_value AND pre/post_transforms (which a
        # hand-rolled SparseTensor(...) would silently drop -- the transform-drop
        # class the campaign fixed elsewhere), and keeps check_consistency=False.
        return _copy(
            t, val=val,
            out_dims=tuple(new_by_id[d.id] for d in t.out_dims),
            primal_dims=tuple(new_by_id[d.id] for d in t.primal_dims),
        )

    return _fix(lhs, r_map), _fix(rhs, l_map)


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
    # Reconcile a compact broadcast dim (logical 1) against its same-id
    # materialized partner (logical N>1) so a fan-in accumulation of two
    # contributions to one vertex broadcasts instead of raising below.
    lhs, rhs = _reconcile_broadcast_dims(lhs, rhs)
    # Static shape comparison: ``SparseTensor.shape`` returns Python ints
    # derived from the dim metadata, so this is a trace-time check (no runtime
    # branching on traced shapes).
    if lhs.shape != rhs.shape:
        raise ValueError(f"Shape mismatch: {lhs.shape} != {rhs.shape}")
    # Mixed-precision upcast: combine a narrow (Quant) operand with a
    # float one at their highest common compute dtype so the downstream
    # op() never hits JAX no-promotion guard. No-op when dtypes match.
    lhs, rhs = _unify_operand_dtypes(lhs, rhs)
    return lhs, rhs


def _promote_dense(d, partner_id):
    """Wrap a DenseIndex into a synthetic 1-block DiagonalIndex paired with `partner_id`."""
    return DiagonalIndex(d.id, 1, axis=None, other_id=partner_id,
                           block_size=d.size, block_axis=d.axis)


def _partner(dims, oid):
    """Index of the dim in `dims` whose id is `oid` (a sparse dim's other_id);
    raises Topology mismatch (caught -> densify fallback) if it dangles."""
    j = next((k for k, d in enumerate(dims) if d.id == oid), None)
    if j is None:
        raise ValueError(f"Topology mismatch: dangling other_id {oid}.")
    return j


def _resolve_dim_pairing(i, ldims, rdims, processed):
    """Pair dim i across (lhs, rhs); promote Dense↔Sparse to a synthetic 1-block sparse pair."""
    ld, rd = ldims[i], rdims[i]
    l_sp, r_sp = ld.is_sparse, rd.is_sparse
    if not l_sp and not r_sp:
        # Pair the DENSE branch by dim id, NOT by position. Elementwise operands
        # share ONE id space (no matmul-style offset; verified: both sides arrive
        # canonical+equal in 98/98 real jacve calls, so this returns rdims[i]
        # unchanged on the exact-AD edge => byte-identical). But once a normalize
        # bypass stops forcing canonical ids via _arr2st, two edges with permuted
        # free dims of coinciding extent (the ViT (8,8) case) have equal .shape
        # and would be added in the WRONG axis pairing with NO assert firing.
        # Mechanism borrowed from matmul._resolve_broadcast_topos.find_match
        # (id equality). Fail LOUDLY if no dense rhs partner carries this id.
        rj = next((k for k, d in enumerate(rdims) if d.id == ld.id), None)
        if rj is None or rdims[rj].is_sparse:
            raise ValueError(
                f"Topology mismatch: dense lhs dim id {ld.id} has no dense rhs "
                f"partner (rhs ids {[(int(d.id), d.is_sparse) for d in rdims]})."
            )
        processed.add(i); return "dense", (ld, rdims[rj])
    # Pair the SPARSE primary dims by id too (mirror the dense branch): two
    # different-id sparse blocks of equal extent at the same position would
    # otherwise be combined silently. Fail loudly -> densify fallback.
    if ld.id != rd.id:
        raise ValueError(
            f"Topology mismatch: sparse dim id {ld.id} (lhs) vs {rd.id} (rhs) "
            f"at position {i} — permuted sparse pairing."
        )
    if l_sp and r_sp:
        j = _partner(ldims, ld.other_id)
        lp, rp = ldims[j], rdims[j]
        if not rp.is_sparse or rd.other_id != rp.id:
            raise ValueError("Topology mismatch: sparse pairs do not align.")
        processed.update([i, j]); return "sparse", (ld, lp, rd, rp)
    if l_sp:
        j = _partner(ldims, ld.other_id)
        lp, rp = ldims[j], rdims[j]
        if not not rp.is_sparse:
            raise ValueError("Topology mismatch: expected DenseIndex partner.")
        processed.update([i, j])
        return "sparse", (ld, lp, _promote_dense(rd, rp.id), _promote_dense(rp, rd.id))
    j = _partner(rdims, rd.other_id)
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
    value = _apply_scalar_mult(value, tensor)
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
            eye = static_eye(exp, bool).reshape(mask_shape)
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
                eye_h = static_eye(exp_h, bool).reshape(ms_h)
                eye_w = static_eye(exp_w, bool).reshape(ms_w)
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
    s_mult = _identity_scalar_mult(value.dtype)
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




# --- WHAT THIS OP MATERIALIZES (measured 2026-09-08) ----------------------
# The contraction engine stopped broadcasting on the same day (ticket
# dsnn-3qm.72). This op did NOT. The census below counts every
# ``broadcast_in_dim`` in the jaxpr whose output holds more elements than its
# input, plus the compiled temp, on a 32-meta diagonal pair of 32x32 blocks:
#
#   aligned diagonal + diagonal            0 growing            temp 0 B
#   diagonal + meta-implicit diagonal      1 growing, 1024 -> 32768 (32x)
#   misaligned diagonal + diagonal         8 growing, 7 281 elements, 33 eqns
#     (meta 4 blocks of 12 against meta 6 blocks of 8, for 1 152 stored out)
#   uniform (val is None) + dense          1 growing, 1 -> 4096
#
# Row 2 is the elementwise MIRROR of the case D1 fixed in the contraction: a
# role implicit on one side and physical on the other, at the same id-matched
# extent. ``_align_value`` broadcasts the implicit side to the full extent
# instead of letting the op broadcast one size-1 axis. Row 3 is
# ``_promote_to_unified``: both operands go to the least-common-multiple meta
# grid, with iota / eq / select machinery to build the mask. Row 4 is a union
# op against a scalar and is unavoidable.
#
# Rows 2 and 3 are what ``lower_add`` existed to remove.


# --- The lazy general path -------------------------------------------------
# Combine two operands in the structure they ALREADY have. Dims are paired BY
# ID -- elementwise operands share ONE id space, so they are never paired
# positionally -- and ONE physical ``op`` runs over reconciled layouts while
# the output structure is built symbolically. No ``_align_value`` broadcast of
# a whole tensor, no ``_promote_to_unified`` densify.
#
# Four rules, from the deleted ``sparse/lower/add.py`` prototype (ticket
# dsnn-3qm.72, owner instruction 2026-09-08 to replace the general case):
#
#   eq      every id-matched dim pair is structurally equal (same sparse
#           pairing, same block grid, same implicitness; physical layouts
#           reconciled by transpose). ``out.val = op(lhs.val * sm,
#           rhs.val_permuted * sm)``, ``scalar_mult = 1``, metadata verbatim.
#           Covers implicit+implicit -> implicit and same-grid block+block.
#   ibroad  as ``eq``, but some axis ROLE is implicit on one side and physical
#           on the other at the SAME id-matched extent. Only that role's size-1
#           axis broadcasts, under ``op``'s own numpy semantics. A role
#           implicit on BOTH sides stays implicit in the output.
#   uu      both operands ``val is None``: the scalars combine and no buffer is
#           built. Gated on statically-zero fills and a zero-preserving ``op``.
#           OFF -- correct, but its ``val=None`` result breaks a consumer
#           downstream. See ``_lazy_uu`` for the measurement.
#   u_x     one operand ``val is None`` with matching structure: it contributes
#           ONE scalar and the other's metadata rides through verbatim.
#
# Anything else -- a sparse-to-dense promotion pair, a MISALIGNED block grid
# (genuine least-common-multiple tiling), leftover physical axes, a fill the
# rule cannot compose -- returns None, and ``_materializing_general`` runs
# unchanged. None is always the safe answer.
#
# The output-fill algebra is IDENTICAL to ``_reconstruct_result``: ``None``
# (statically zero) when both inputs are statically zero AND ``op`` is
# zero-preserving, else the concrete post-scaled combined fill.
#
# HISTORY, because this path once shipped wrong. Armed behind the deleted
# ``GRAPHAX_EINSUM_EW`` flag, a float64 model diff caught it diverging by up to
# 0.8 on ViT COMPRESS variants while every matmul in the same run stayed
# oracle-exact, and the divergence was never localized to a rule. It is armed
# now because ``elementwise_lazy_test.py`` settles that question directly: it
# compares this path against ``_materializing_general`` on the DENSE form over
# every structured signature, for a union op and an intersection op, with zero
# and non-zero fills. A rule that cannot pass that does not ship.
#
# That test alone was not enough. It passed on all four rules, and the full
# suite then failed ``RoeFlux_3d``: ``uu`` is right in isolation but returns
# ``val=None`` where the incumbent materializes, and a consumer downstream
# collapses an extent on it. ``uu`` is OFF for that reason, measured, in
# ``_lazy_uu``. The lesson is that a differential test on ONE op cannot clear a
# structural change; the whole-graph suite is the check that matters.
#
# ``LAZY_STATS`` is the totality ledger: every rule hit and every named
# fallthrough. Coverage is measured, never assumed.
LAZY_STATS: collections.Counter = collections.Counter()


def reset_lazy_stats() -> None:
    LAZY_STATS.clear()


def _skip(reason: str):
    LAZY_STATS[f"skip:{reason}"] += 1
    return None


def _finish(out, rule: str):
    LAZY_STATS[f"rule:{rule}"] += 1
    return out


def _combined_fill(lhs, rhs, op):
    """Same algebra as ``_reconstruct_result``: keep the static-zero ``None``
    marker when sound, else the concrete post-scaled combined fill."""
    if lhs.fill_value is None and rhs.fill_value is None and op in _ZERO_PRESERVING_OPS:
        return None
    return op(_scaled_fill(lhs), _scaled_fill(rhs))


def _match_structure(l_by_id, r_by_id):
    """``None`` when every id-matched dim pair is structurally compatible
    (equal logical extent; sparse pairs share partner id / meta size / block
    grid), else the skip reason. Physical-layout (implicit vs physical) mixes
    are NOT checked here — they are legal for every axis role and handled
    per-slot by the ``ibroad`` machinery in ``_lazy_pair``."""
    for i, ld in l_by_id.items():
        rd = r_by_id[i]
        if ld.is_sparse != rd.is_sparse:
            return "sparse_dense_mix"
        if ld.logical_size != rd.logical_size:
            return "extent_mismatch"
        if ld.is_sparse:
            if (ld.other_id != rd.other_id or ld.size != rd.size
                    or (ld.block_size or 1) != (rd.block_size or 1)):
                return "sparse_meta_mismatch"
    return None


def _side_axes_cover(t) -> bool:
    """True iff ``t.val``'s physical axes are exactly the axes described by
    ``t.dims`` (sparse pair meta axis counted once and REQUIRED equal on both
    members). Leftover / duplicated / dangling axes ⇒ no rule."""
    by_id = {d.id: d for d in t.dims}
    seen: set[int] = set()
    axes: list[int] = []
    for d in t.dims:
        if d.is_sparse:
            if d.id in seen:
                continue
            partner = by_id.get(d.other_id)
            if partner is None:
                return False
            seen.update((d.id, d.other_id))
            if (d.axis is None) != (partner.axis is None) or (
                    d.axis is not None and d.axis != partner.axis):
                return False
            if d.axis is not None:
                axes.append(d.axis)
            for m in (d, partner):
                if getattr(m, "block_axis", None) is not None:
                    axes.append(m.block_axis)
        elif d.axis is not None:
            axes.append(d.axis)
    return sorted(axes) == list(range(t.val.ndim))


def _place_axes(val, srcs):
    """Transpose/reshape ``val`` so that source axis ``srcs[k]`` lands at
    target position ``k`` (``None`` ⇒ a fresh size-1 axis there). Requires the
    non-None entries to be a permutation of ``range(val.ndim)`` — guaranteed by
    ``_side_axes_cover`` + slot construction. Pure layout: no broadcast, no
    copy beyond the transpose."""
    order = [a for a in srcs if a is not None]
    if order != list(range(val.ndim)):
        val = val.transpose(order)
    if len(srcs) != val.ndim:
        shape, it = [], iter(val.shape)
        for a in srcs:
            shape.append(1 if a is None else next(it))
        val = val.reshape(shape)
    return val


def _lazy_general(lhs, rhs, op: Callable, is_intersection: bool = False):
    """Try to lower ``op(lhs, rhs)``; ``None`` ⇒ no rule (caller falls through).

    ``is_intersection`` needs no special handling here: every rule operates on
    ALIGNED structure (no LCM promotion), where the general path's
    intersection demote is a no-op by construction.
    """
    l_by_id = {d.id: d for d in lhs.dims}
    r_by_id = {d.id: d for d in rhs.dims}
    if set(l_by_id) != set(r_by_id) or len(l_by_id) != len(lhs.dims) \
            or len(r_by_id) != len(rhs.dims):
        return _skip("id_mismatch")

    if lhs.val is None and rhs.val is None:
        return _lazy_uu(lhs, rhs, op, l_by_id, r_by_id)
    if lhs.val is None or rhs.val is None:
        return _lazy_u_x(lhs, rhs, op, l_by_id, r_by_id)
    return _lazy_pair(lhs, rhs, op, l_by_id, r_by_id)


def _all_implicit(t) -> bool:
    return all(d.axis is None and getattr(d, "block_axis", None) is None
               for d in t.dims)


def _lazy_uu(lhs, rhs, op, l_by_id, r_by_id):
    """U+U -> U: two pure-structure operands combine entirely in scalar_mult.

    OFF. The rule is correct in isolation and it is the leanest of the four --
    no buffer is built at all -- but its result is the one signature the rest
    of the engine cannot yet read. It returns ``val=None`` where
    ``_reconstruct_result`` materializes an array, and a consumer downstream
    then collapses a logical extent.

    MEASURED 2026-09-08 on ``RoeFlux_3d`` (order fwd), by disabling one rule at
    a time:

        lazy off entirely   PASS
        all four rules      FAIL  edge shape (1, 1), expected (3, 1)
        without uu          PASS  (eq 174, ibroad 46, u_x 66)
        without u_x         FAIL
        without eq+ibroad   FAIL
        without ibroad      FAIL
        without eq          FAIL

    Every configuration that keeps ``uu`` fails and the one that drops it
    passes, so the fault is ``uu``'s alone. Neither ``matmul`` nor
    ``elementwise`` loses the extent -- both were instrumented and reported
    zero -- so the consumer that mis-reads a ``val=None`` union result is a
    third site and has not been found yet. Until it is, this signature takes
    the materializing path, exactly as it did before, and the engine is
    unchanged for it.

    Do not re-enable this without finding that consumer.
    ``elementwise_lazy_test.py::test_uu_declines_until_its_consumer_is_fixed``
    pins the decline.
    """
    return _skip("uu_downstream_collapses_an_extent")
    reason = _match_structure(l_by_id, r_by_id)
    if reason:
        return _skip(reason)
    if not (_all_implicit(lhs) and _all_implicit(rhs)):
        return _skip("u_axes")
    if not (lhs.fill_value is None and rhs.fill_value is None
            and op in _ZERO_PRESERVING_OPS):
        return _skip("uu_fill")
    from graphax.sparse.tensor import SparseTensor

    s = op(_apply_scalar_mult(jnp.ones((), lhs.dtype), lhs),
           _apply_scalar_mult(jnp.ones((), rhs.dtype), rhs))
    out = SparseTensor(lhs.out_dims, lhs.primal_dims, None, scalar_mult=s,
                       fill_value=None, check_consistency=False)
    return _finish(out, "uu")


def _lazy_u_x(lhs, rhs, op, l_by_id, r_by_id):
    """U+x with matching structure: the U side is one scalar; x's metadata
    (and val layout) are preserved verbatim."""
    reason = _match_structure(l_by_id, r_by_id)
    if reason:
        return _skip(reason)
    u, x = (lhs, rhs) if lhs.val is None else (rhs, lhs)
    if not _all_implicit(u):
        return _skip("u_axes")
    # Support equality: x may not be "wider" than u along a dense dim — a dim
    # that is implicit on u must be implicit-or-physical on x with the SAME
    # logical extent (checked above), which makes the supports identical.
    from graphax.sparse.tensor import SparseTensor

    x_by_id = {d.id: d for d in x.dims}
    s = _apply_scalar_mult(jnp.ones((), u.dtype), u)
    xv = _apply_scalar_mult(x.val, x)
    if s.dtype != xv.dtype:
        cdt = _compute_dtype(s.dtype, xv.dtype)
        s, xv = s.astype(cdt), xv.astype(cdt)
    val = op(s, xv) if u is lhs else op(xv, s)
    out = SparseTensor(
        tuple(x_by_id[d.id] for d in lhs.out_dims),
        tuple(x_by_id[d.id] for d in lhs.primal_dims),
        val, scalar_mult=_identity_scalar_mult(val.dtype),
        fill_value=_combined_fill(lhs, rhs, op), check_consistency=False,
    )
    return _finish(out, "u_x")


def _lazy_pair(lhs, rhs, op, l_by_id, r_by_id):
    """Both sides carry a val: the eq / ibroad fast path."""
    reason = _match_structure(l_by_id, r_by_id)
    if reason:
        return _skip(reason)
    if not (_side_axes_cover(lhs) and _side_axes_cover(rhs)):
        return _skip("leftover_axes")

    # --- Pair up dims in lhs encounter order (mirrors _map_topology). ---
    sp, dp, processed = [], [], set()
    for d in lhs.dims:
        if d.id in processed:
            continue
        rd = r_by_id[d.id]
        if d.is_sparse:
            processed.update((d.id, d.other_id))
            sp.append((d, l_by_id[d.other_id], rd, r_by_id[d.other_id]))
        else:
            processed.add(d.id)
            dp.append((d, rd))

    # --- Canonical output layout: one target axis per axis role that is
    # PHYSICAL on at least one side; a role implicit on BOTH sides stays
    # implicit (retention by construction: I+I → I, implicit meta diagonals,
    # implicit-within-block). A role physical on ONE side broadcasts that
    # single size-1 axis under ``op`` (the ``ibroad`` rule) — extents are
    # id-matched equal, so this is never a genuine logical-1 stretch.
    # slots[k] = (lhs_src_axis|None, rhs_src_axis|None, target_extent).
    slots: list[tuple[int | None, int | None, int]] = []
    rec: dict[int, object] = {}
    rule = "eq"

    def _slot(l_ax, r_ax, extent):
        nonlocal rule
        if l_ax is None and r_ax is None:
            return None  # implicit on both sides — stays implicit
        if l_ax is None or r_ax is None:
            rule = "ibroad"
        slots.append((l_ax, r_ax, extent))
        return len(slots) - 1

    def _bs(d, nb):
        # A dim that gained a (size-1) physical block axis from the partner
        # side must carry an explicit block_size — block_axis without
        # block_size is inconsistent metadata.
        return 1 if (nb is not None and d.block_size is None) else d.block_size

    for ld1, ld2, rd1, rd2 in sp:
        new_axis = _slot(ld1.axis, rd1.axis, ld1.size)
        nb1 = _slot(ld1.block_axis, rd1.block_axis, ld1.block_size or 1)
        nb2 = _slot(ld2.block_axis, rd2.block_axis, ld2.block_size or 1)
        rec[ld1.id] = replace(ld1, axis=new_axis, block_axis=nb1, block_size=_bs(ld1, nb1))
        rec[ld2.id] = replace(ld2, axis=new_axis, block_axis=nb2, block_size=_bs(ld2, nb2))
    for ld, rd in dp:
        new_axis = _slot(ld.axis, rd.axis, ld.logical_size)
        rec[ld.id] = ld if new_axis is None else replace(ld, axis=new_axis)

    # Physical extents may legally be the full logical extent OR 1 (a
    # stored-once axis, which _align_value stretches with broadcast_to on the
    # general path). Anything else is malformed for these rules — fall through.
    for l_ax, r_ax, ext in slots:
        if l_ax is not None and lhs.val.shape[l_ax] not in (ext, 1):
            return _skip("phys_extent")
        if r_ax is not None and rhs.val.shape[r_ax] not in (ext, 1):
            return _skip("phys_extent")

    # --- One physical op over the reconciled layouts. ---
    la = _place_axes(_apply_scalar_mult(lhs.val, lhs), [s[0] for s in slots])
    ra = _place_axes(_apply_scalar_mult(rhs.val, rhs), [s[1] for s in slots])
    if la.dtype != ra.dtype:
        cdt = _compute_dtype(la.dtype, ra.dtype)
        la, ra = la.astype(cdt), ra.astype(cdt)
    res = op(la, ra)
    target = tuple(s[2] for s in slots)
    if res.shape != target:
        # An axis stored once (physical extent 1, logical extent N) on BOTH
        # sides: the output metadata is full-extent, so materialize it exactly
        # as the general path's broadcast_to would. Counted — this is the one
        # spot where the lowering still expands storage.
        LAZY_STATS["note:bcast_materialize"] += 1
        res = jnp.broadcast_to(res, target)

    from graphax.sparse.tensor import SparseTensor

    out = SparseTensor(
        tuple(rec[d.id] for d in lhs.out_dims),
        tuple(rec[d.id] for d in lhs.primal_dims),
        res, scalar_mult=_identity_scalar_mult(res.dtype),
        fill_value=_combined_fill(lhs, rhs, op), check_consistency=False,
    )
    return _finish(out, rule)


# --- Path tracing (test-only) ---------------------------------------------
# Re-exports from ``_path_tracking``. See that module for the full design;
# tests opt in via the ``track_paths()`` context manager or ``TRACK_PATHS=1``
# env var. Production runs pay nothing.
from ._path_tracking import record_path as _record_path  # noqa: E402, F401


def elementwise(
    lhs,
    rhs,
    op: Callable,
    is_intersection: bool = False,
    count: bool = False,
):
    """Sparse elementwise op dispatcher.

    Single ``general`` path that handles every case. Misaligned 2-D
    block-diagonal union and intersection outputs go to the least-common-
    multiple meta grid: the ``SetIndex`` compression that used to catch them
    is gone (ruling 2026-09-07 — no measured target ever built a set).
    Aligned blocks, non-zero fills, broadcast cases, and non-zero-preserving
    ops fall through the full promote-to-unified pipeline.

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
    # Edge-level LowRank (L3): unwrap FIRST (before normalization — the
    # wrapper is not a SparseTensor).
    if getattr(lhs, "_is_lowrank", False) or getattr(rhs, "_is_lowrank", False):
        from graphax.sparse.lowrank import lowrank_elementwise

        return lowrank_elementwise(
            lhs, rhs, op, is_intersection=is_intersection, count=count
        )
    _record_path(None)
    lhs, rhs = _normalize_inputs(lhs, rhs)
    # Elemental fast path (Phase: bridge-cse): route a STRUCTURED elementwise op
    # (block-diagonal / implicit dims) through the elemental kernels. Returns
    # None for a pure-dense op so the existing general path stays byte-identical
    # on the EXACT-AD edge.
    from graphax.sparse.elemental.dispatch import try_elemental_elementwise

    _elem = try_elemental_elementwise(
        lhs, rhs, op, is_intersection=is_intersection, count=count
    )
    if _elem is not None:
        _record_path("elemental")
        return _elem
    if count:
        n = _ew_op_count(lhs, rhs, is_intersection)

    # LAZY GENERAL PATH: combine the two operands in the structure they already
    # have. Returns None when no rule covers the signature, and the
    # materializing path below then runs UNCHANGED. See ``_lazy_general``.
    _lz = _lazy_general(lhs, rhs, op, is_intersection)
    if _lz is not None:
        _record_path("lazy")
        if count:
            return _lz, n
        return _lz

    _record_path("general")
    return _materializing_general(lhs, rhs, op, is_intersection, n if count else None)


def _materializing_general(lhs, rhs, op, is_intersection, n):
    """The incumbent general path: align, promote to the least-common-multiple
    meta grid, apply ``op``, demote, rebuild. It materializes -- an implicit
    role is broadcast to its partner's full extent by ``_align_value`` and a
    misaligned block grid goes to the LCM grid in ``_promote_to_unified``.

    ``n`` is the op count, or ``None`` for no count. Kept as its own function
    so the differential test can call it directly as the oracle, with no flag
    to set and nothing to monkeypatch.
    """
    count = n is not None
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
    _pl = _promote_to_unified(vl, metrics, True, _scaled_fill(lhs))
    _pr = _promote_to_unified(vr, metrics, False, _scaled_fill(rhs))
    _pldt = getattr(_pl, "dtype", None)
    _prdt = getattr(_pr, "dtype", None)
    if _pldt is not None and _prdt is not None and _pldt != _prdt:
        _cdt = _compute_dtype(_pldt, _prdt)
        _pl = _pl.astype(_cdt)
        _pr = _pr.astype(_cdt)
    res = op(_pl, _pr)
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
