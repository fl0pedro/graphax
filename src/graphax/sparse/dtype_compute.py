"""Highest-common-dtype promotion for mixed-precision SparseTensor arithmetic.

A :class:`~graphax.sparse.micro_actions.Quant` stores ``SparseTensor.val`` in a
chosen (possibly narrow) dtype while ``scalar_mult`` / ``fill_value`` keep their
native dtype. JAX deliberately gives the narrow dtypes — every ``float8_*``,
``float4_e2m1fn``, and the sub-byte ints ``int2``/``int4``/``uint2``/``uint4`` —
NO promotion-lattice entry: not just the implicit binary-op promotion but the
explicit ``jnp.promote_types`` / ``jnp.result_type`` queries themselves raise
``TypePromotionError`` for them (verified against the pinned JAX). So a bare
``val * scalar_mult`` fails whenever ``val`` was quantized to one of these, and
the codebase's :func:`with_type_promotion` decorators (which call
``jnp.result_type`` and convert the op's *output*) can't help — the op raises
*before* any output conversion, so the *inputs* must be upcast first.

The fix is mixed precision: keep ``val`` stored narrow (the at-rest memory /
precision-loss signal a Quant models), but combine operands at their HIGHEST
COMMON dtype. We use JAX's native ``result_type`` directly; only when that
raises (a narrow operand is present) do we map each narrow dtype to the
smallest STANDARD dtype that contains it and retry — so the answer respects
float64 / complex and stays at float16 when that already suffices, never a
blanket upcast to float32.

Imports only ``jax.numpy`` so it is usable from ``tensor.py``, ``ops/utils.py``,
``ops/dense.py`` etc. without import cycles.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as _np


# Each narrow dtype JAX won't promote -> the smallest STANDARD dtype that
# contains its range/precision. float8 (≤5 exponent bits) and float4 fit in
# float16; ``float8_e8m0fnu`` carries 8 exponent bits (float32-range, no
# mantissa) so it needs float32; sub-byte ints fit in int8/uint8. Used only to
# resolve the promotion target when a narrow operand makes ``result_type`` raise.
_NARROW_PROMOTION_REP: dict[str, Any] = {
    "float8_e3m4": jnp.float16,
    "float8_e4m3": jnp.float16,
    "float8_e4m3b11fnuz": jnp.float16,
    "float8_e4m3fn": jnp.float16,
    "float8_e4m3fnuz": jnp.float16,
    "float8_e5m2": jnp.float16,
    "float8_e5m2fnuz": jnp.float16,
    "float8_e8m0fnu": jnp.float32,
    "float4_e2m1fn": jnp.float16,
    "int2": jnp.int8,
    "int4": jnp.int8,
    "uint2": jnp.uint8,
    "uint4": jnp.uint8,
}


def _full_float_dtype() -> Any:
    """The full-precision scalar dtype: float64 when ``jax_enable_x64`` is
    enabled, float32 otherwise. ``jnp.result_type(float)`` reports exactly
    JAX's configured default, so this needs no ``jax.config`` import (which
    would create a cycle from ``tensor.py``)."""
    return jnp.dtype(jnp.result_type(float))


def _is_narrow(dt) -> bool:
    """True for the dtypes JAX refuses to promote (every float8/float4 and the
    sub-byte ints) -- i.e. the ones only reachable via a ``Quant``."""
    try:
        return jnp.dtype(dt).name in _NARROW_PROMOTION_REP
    except Exception:
        return False


def _scalar_store_dtype(val_dtype) -> Any:
    """At-rest dtype for ``scalar_mult`` / ``fill_value``.

    ONLY ``val`` is allowed to be narrow. For float or quantized tensors the
    scalars are kept at full precision so the quantization is a property of the
    stored buffer alone and densification can choose its output precision.
    Genuine integer / bool tensors keep their own dtype -- forcing those to
    float would change integer arithmetic.
    """
    dt = jnp.dtype(val_dtype)
    if _is_narrow(dt) or jnp.issubdtype(dt, jnp.floating):
        return _full_float_dtype()
    return dt


def _check_downcast_safe(value, target_dtype, *, what="value") -> None:
    """Raise ``OverflowError`` if CONCRETE ``value`` is not representable in
    ``target_dtype``.

    Checked on magnitude, not on dtype ranges: a static range comparison would
    reject every float32->bfloat16 downcast (same exponent range, far less
    mantissa) while waving through genuinely lossy ones. Under jit the operand
    is a tracer, no static check is possible, and IEEE saturation applies -- so
    this is a best-effort guard on the eager path, which is where Quant scales
    are actually built.
    """
    tdt = jnp.dtype(target_dtype)
    if not (jnp.issubdtype(tdt, jnp.floating) or _is_narrow(tdt)):
        return
    try:
        arr = _np.asarray(value, dtype=_np.float64)
    except Exception:
        return  # tracer / unconvertible: nothing static to check
    if arr.size == 0:
        return
    finite = arr[_np.isfinite(arr)]
    if finite.size == 0:
        return
    try:
        lim = float(jnp.finfo(tdt).max)
    except Exception:
        return
    peak = float(_np.max(_np.abs(finite)))
    if peak > lim:
        raise OverflowError(
            f"{what}: magnitude {peak:.6g} exceeds {tdt.name} max {lim:.6g}. "
            f"Refusing to silently overflow to inf. Densify with "
            f"keep_quantization=False to read this tensor in full precision."
        )


def _compute_dtype(*dtypes) -> Any:
    """Highest common arithmetic dtype of ``dtypes``.

    Native-first: try JAX's own ``jnp.result_type``; only if it raises (a
    narrow, non-promotable operand is present) map each narrow dtype to the
    smallest standard dtype that contains it and retry. Standard-dtype
    combinations therefore go straight through JAX with no shim.
    """
    dts = [jnp.dtype(d) for d in dtypes]
    try:
        return jnp.result_type(*dts)
    except Exception:
        reps = [_NARROW_PROMOTION_REP.get(d.name, d) for d in dts]
        return jnp.result_type(*reps)


def _scaled_mul(value, scalar_mult, *, keep_narrow: bool = False):
    """``value * scalar_mult``.

    ``keep_narrow=False`` (default) combines at the highest common dtype, which
    -- now that ``scalar_mult`` is stored at full precision -- yields a
    full-precision result: the safe, lossless read.

    ``keep_narrow=True`` instead broadcasts ``scalar_mult`` DOWN into ``value``'s
    (narrow) dtype and multiplies there, so a Quant survives densification.
    Guarded by :func:`_check_downcast_safe`.

    Original contract below.

    Upcasts both operands to ``_compute_dtype(value, scalar_mult)`` first (via
    the native ``astype``), so a narrow (Quant'd) ``value`` never trips JAX's
    implicit-promotion guard. The result carries the common dtype.

    A python-scalar operand (e.g. a ``SparseTensor`` built with a raw
    ``fill_value=1`` / ``scalar_mult=2.0`` — the constructor wraps only
    ``val`` in ``jnp.asarray``) has no ``.dtype`` and can never be a narrow
    JAX dtype, so it can't hit the promotion guard: fall back to the plain
    multiply, which also preserves JAX weak-typing for that operand.
    """
    vdt = getattr(value, "dtype", None)
    sdt = getattr(scalar_mult, "dtype", None)
    if vdt is None or sdt is None:
        return value * scalar_mult
    # bf16 narrow-compute mode implies keep_narrow for a bf16 value: the
    # densify/materialize read (val * f32 scalar_mult) was re-promoting the
    # Quant'd array to f32 right BEFORE the contraction consumed it, so the
    # heavyweight dots never ran bf16 (see _quant_narrow_bf16_pair).
    if not keep_narrow and _quant_narrow_bf16_pair(jnp.dtype(vdt),
                                                   jnp.dtype(sdt))             and jnp.dtype(vdt) == jnp.dtype(jnp.bfloat16):
        keep_narrow = True
    if keep_narrow and jnp.dtype(vdt) != jnp.dtype(sdt):
        _check_downcast_safe(scalar_mult, vdt, what="scalar_mult")
        return value * jnp.asarray(scalar_mult).astype(vdt)
    cdt = _compute_dtype(vdt, sdt)
    # No zero-point unshift: unsigned Quant targets store the sign-flip
    # half-range magnitudes with the arm's polarity folded into ``scalar_mult``
    # (see :func:`graphax.sparse.micro_actions.apply_quant`), so ``val *
    # scalar_mult`` already dequantizes to the correct signed value.
    return value.astype(cdt) * scalar_mult.astype(cdt)


def _unify_operand_dtypes(lhs, rhs):
    """Upcast two ``SparseTensor`` operands to their highest common compute
    dtype so downstream arithmetic (``op(promote_l, promote_r)`` in
    elementwise, ``dot_general`` in matmul) never combines a narrow Quant'd
    ``val`` with a float ``val`` -- JAX gives those pairs no implicit promotion
    path and raises ``TypePromotionError`` / ``lax.add same-dtype`` BEFORE any
    output conversion can help.

    Mixed-precision design: ``val`` is stored narrow at rest (the Quant
    precision-loss signal) but operands are combined at the HIGHEST COMMON
    dtype. We cast each operand's ``val`` / ``scalar_mult`` / ``fill_value`` to
    ``_compute_dtype(lhs.dtype, rhs.dtype)`` (which maps each narrow dtype to
    the smallest standard dtype that contains it only when JAX itself cannot
    promote, so float64 / complex / float16 are respected -- never a blanket
    upcast to float32).

    No-op fast path: when both operands already share the common dtype (the
    overwhelmingly common case -- no Quant in the chain) the originals are
    returned unchanged, so jit-tracing / bool tensors / the statically-zero
    ``fill_value=None`` marker are untouched.
    """
    ldt = jnp.dtype(lhs.dtype)
    rdt = jnp.dtype(rhs.dtype)
    if ldt == rdt:
        # Same at-rest dtype on both sides -- no cross-operand promotion needed.
        # (Two same-narrow operands also take the native path; the op runs in
        # that narrow dtype, matching the stored precision.)
        return lhs, rhs
    # DELIBERATE (2026-08-04): a {bf16, f32} MIXED pair UPCASTS to f32 --
    # quantizing one edge must never silently approximate its exact partner
    # (user decision; an earlier draft downcast the pair). The bf16 fast
    # path exists only when BOTH edges were made bf16: same-dtype pairs
    # return above unchanged, and the tiled dot sites then run the bf16
    # GEMM with f32 accumulation (matmul._gx_dot_general). The companion
    # keep-narrow read in _scaled_mul stops a lone scalar drain from
    # re-promoting an already-bf16 edge before that both-bf16 meeting.
    cdt = _compute_dtype(ldt, rdt)
    return _cast_operand(lhs, cdt), _cast_operand(rhs, cdt)


def _quant_narrow_bf16_pair(ldt, rdt) -> bool:
    """True iff bf16 narrow-compute handling applies to this dtype pair
    (consumed by _scaled_mul's keep-narrow read; the operand unify above
    deliberately does NOT use it -- mixed pairs upcast). Env read per call:
    GRAPHAX_QUANT_NARROW_GEMM default ON, and the einsum_general planner
    must be OFF -- the planner emits its own contractions with no
    f32-accumulate plumbing, so narrow vals must not leak into it.

    2026-08 MEASURED (G3): the stated rationale above is FALSE. The planner
    already receives both-bf16 operand pairs -- whenever BOTH faces of a
    contraction were quantized (the per-face ``lhs``/``rhs`` slots) -- and it
    lowers them to genuine bf16 dots (6 on mlp2, 16 on mlp4, 26 on attn).
    Narrow vals do not need this guard to stay out of the planner, because
    they were never kept out of it.

    Relaxing the exclusion was tried and measured. On the CONTRACTION side it
    buys nothing: with the toggle isolated on mlp2 the lowered dot census and
    convert count are unchanged ({f32:4, bf16:6}, 22 converts either way),
    and the per-vertex plan's error against exact f32 gets slightly WORSE
    (relerr 2.93e-3 -> 3.17e-3). It is NOT a no-op, though: it changes the
    elementwise JOIN. With the exclusion removed, ``bf16 + bf16`` keeps its
    bf16 storage (the bit-exact bf16 add, ``scalar_mult`` stored bf16);
    with it in place the f32 ``scalar_mult`` drain re-promotes both addends
    and the sum comes back f32. Neither introduces a rescale. That is a real
    precision/storage trade with no measured contraction benefit, so the
    exclusion stays until someone wants the trade deliberately."""
    import os as _os
    if _os.environ.get("GRAPHAX_QUANT_NARROW_GEMM", "1") == "0":
        return False
    if _os.environ.get("GRAPHAX_EINSUM_GENERAL", "1") != "0":
        return False
    return {ldt, rdt} == {jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)}


def _cast_operand(t, cdt):
    """Return ``t`` with ``val`` / ``scalar_mult`` / ``fill_value`` cast to
    ``cdt``. ``val=None`` (pure structure) and ``fill_value=None`` (statically
    zero) markers are preserved; ``scalar_mult`` always exists. Avoids a rebuild
    when nothing changes."""
    new_val = (
        t.val.astype(cdt)
        if t.val is not None and jnp.dtype(t.val.dtype) != cdt
        else t.val
    )
    new_sm = (
        t.scalar_mult.astype(cdt)
        if getattr(t.scalar_mult, "dtype", None) is not None
        and jnp.dtype(t.scalar_mult.dtype) != cdt
        else t.scalar_mult
    )
    new_fill = (
        t.fill_value.astype(cdt)
        if t.fill_value is not None
        and getattr(t.fill_value, "dtype", None) is not None
        and jnp.dtype(t.fill_value.dtype) != cdt
        else t.fill_value
    )
    if new_val is t.val and new_sm is t.scalar_mult and new_fill is t.fill_value:
        return t
    return t.copy(val=new_val, scalar_mult=new_sm, fill_value=new_fill)
