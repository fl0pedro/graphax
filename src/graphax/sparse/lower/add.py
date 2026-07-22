"""``lower_add`` — the ADD-side structure-lowering engine (``GRAPHAX_STRUCT_LOWER``).

Given ``op(lhs, rhs)`` on two shape-checked, dtype-unified ``SparseTensor``
operands (exactly as produced by ``elementwise._normalize_inputs``), try to
COMPILE the minimal physical computation and construct the output structure
SYMBOLICALLY. Returns ``None`` when no rule applies — the caller falls through
to the existing general path unchanged. Every hit AND every skip is counted in
``STATS`` (coverage is measured, never assumed).

Rules (dims are matched BY ID — elementwise operands share ONE dim-id space):

* ``eq`` fast path: every id-matched dim pair is structurally EQUAL (same
  sparse pairing / block grid / implicitness; physical layouts reconciled by
  transpose). ``out.val = op(lhs.val * lhs_sm, rhs.val_permuted * rhs_sm)``,
  ``sm = 1``, metadata preserved verbatim. NO broadcast, NO LCM-grid dance.
  Covers I+I → I (both implicit: the representative values combine, the dim
  stays implicit) and same-grid B+B → B.
* ``ibroad``: as ``eq`` but some axis role (a dense dim, a sparse pair's meta
  diagonal, or a within-block axis) is implicit on one side and physical on
  the other (equal id-matched extent — this is the sound case for
  broadcasting, never a genuine ``logical_size==1`` stretch). ONLY that
  role's size-1 axis broadcasts under ``op``'s numpy semantics; the
  whole-tensor ``_align_value`` broadcast is never invoked. A role implicit
  on BOTH sides stays implicit in the output (I+I → I, implicit meta
  diagonals, implicit-within-block — retention by construction).
* ``uu``: both ``val is None`` (pure structure × scalar_mult). The scalars
  combine — no buffer is ever built. Gated on statically-zero fills and a
  zero-preserving ``op`` so the fill/support algebra stays exact.
* ``u_x``: one side ``val is None`` with matching structure — that side
  contributes ONE scalar; ``out.val = op(s, x.val * x_sm)`` with x's
  metadata verbatim.

Anything else (sparse↔dense promotion, misaligned block grids, leftover
physical axes) falls through, loudly counted, to the existing path —
including the Phase-6b/9 SetIndex emissions which already compress the
misaligned zero-fill cases.

Output-fill algebra is IDENTICAL to ``_reconstruct_result``: ``None`` (static
zero) iff both inputs are statically zero AND ``op`` is zero-preserving, else
the concrete ``op(scaled_fill(lhs), scaled_fill(rhs))``.
"""
from __future__ import annotations

import collections
import os
from dataclasses import replace
from typing import Callable

import jax.numpy as jnp

from graphax.sparse.dtype_compute import _compute_dtype
from graphax.sparse.ops.elementwise import _ZERO_PRESERVING_OPS, _identity_scalar_mult
from graphax.sparse.ops.utils import _apply_scalar_mult, _scaled_fill

# rule:* = lowered here; skip:* = fell through to the general path (and why).
STATS: collections.Counter = collections.Counter()


def reset_stats() -> None:
    STATS.clear()


def enabled() -> bool:
    """Read the gate dynamically so tests / harnesses can toggle in-process."""
    return os.environ.get("GRAPHAX_STRUCT_LOWER", "0") not in ("", "0", "false", "False")


def _skip(reason: str):
    STATS[f"skip:{reason}"] += 1
    return None


def _finish(out, rule: str):
    STATS[f"rule:{rule}"] += 1
    if os.environ.get("GRAPHAX_STRUCT_LOWER_SABOTAGE"):  # negative-control hook
        # Deliberately corrupt the result so the differential harness can prove
        # it would catch a wrong rule. NEVER set outside tests.
        from graphax.sparse.ops.utils import _copy

        if out.val is not None:
            bad = ~out.val if out.val.dtype == jnp.bool_ else out.val + jnp.asarray(1, out.val.dtype)
            return _copy(out, val=bad)
        bad_sm = (~out.scalar_mult.astype(jnp.bool_) if out.scalar_mult.dtype == jnp.bool_
                  else out.scalar_mult + jnp.asarray(1, out.scalar_mult.dtype))
        return _copy(out, scalar_mult=bad_sm)
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
    per-slot by the ``ibroad`` machinery in ``_lower_pair``."""
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


def lower_add(lhs, rhs, op: Callable, is_intersection: bool = False):
    """Try to lower ``op(lhs, rhs)``; ``None`` ⇒ no rule (caller falls through).

    ``is_intersection`` needs no special handling here: every rule operates on
    ALIGNED structure (no LCM promotion), where the general path's
    intersection demote is a no-op by construction.
    """
    for d in (*lhs.dims, *rhs.dims):
        if getattr(d, "is_compressed", False):
            return _skip("compressed")

    l_by_id = {d.id: d for d in lhs.dims}
    r_by_id = {d.id: d for d in rhs.dims}
    if set(l_by_id) != set(r_by_id) or len(l_by_id) != len(lhs.dims) \
            or len(r_by_id) != len(rhs.dims):
        return _skip("id_mismatch")

    if lhs.val is None and rhs.val is None:
        return _lower_uu(lhs, rhs, op, l_by_id, r_by_id)
    if lhs.val is None or rhs.val is None:
        return _lower_u_x(lhs, rhs, op, l_by_id, r_by_id)
    return _lower_pair(lhs, rhs, op, l_by_id, r_by_id)


def _all_implicit(t) -> bool:
    return all(d.axis is None and getattr(d, "block_axis", None) is None
               for d in t.dims)


def _lower_uu(lhs, rhs, op, l_by_id, r_by_id):
    """U+U → U: two pure-structure operands combine entirely in scalar_mult."""
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


def _lower_u_x(lhs, rhs, op, l_by_id, r_by_id):
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


def _lower_pair(lhs, rhs, op, l_by_id, r_by_id):
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
        STATS["note:bcast_materialize"] += 1
        res = jnp.broadcast_to(res, target)

    from graphax.sparse.tensor import SparseTensor

    out = SparseTensor(
        tuple(rec[d.id] for d in lhs.out_dims),
        tuple(rec[d.id] for d in lhs.primal_dims),
        res, scalar_mult=_identity_scalar_mult(res.dtype),
        fill_value=_combined_fill(lhs, rhs, op), check_consistency=False,
    )
    return _finish(out, rule)
