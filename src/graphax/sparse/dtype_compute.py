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


# Each narrow dtype JAX won't promote -> the STANDARD dtype arithmetic runs in
# when a narrow operand makes ``result_type`` raise. Narrow FLOATS go to
# bfloat16, not float16: the quantized value is a scaled code, and the
# contraction it enters has the exponent range of the f32 Jacobians it meets.
# Measured 2026-09-13 (TLM, Markowitz order, every face slot float8_e4m3fn or
# float8_e5m2): with float16 as the rep the gradient came back NaN, and so did
# a float16 Quant itself. bfloat16 keeps float32's 8 exponent bits (the owner
# ruling: f32 and bf16 are the only compute dtypes). Sub-byte ints fit in
# int8 / uint8.
_NARROW_PROMOTION_REP: dict[str, Any] = {
    "float8_e3m4": jnp.bfloat16,
    "float8_e4m3": jnp.bfloat16,
    "float8_e4m3b11fnuz": jnp.bfloat16,
    "float8_e4m3fn": jnp.bfloat16,
    "float8_e4m3fnuz": jnp.bfloat16,
    "float8_e5m2": jnp.bfloat16,
    "float8_e5m2fnuz": jnp.bfloat16,
    "float8_e8m0fnu": jnp.float32,
    "float4_e2m1fn": jnp.bfloat16,
    # float16 is a STANDARD dtype for JAX, so result_type never raises on it;
    # it is listed here so that `_compute_dtype` replaces a float16 RESULT by
    # float32. float16 storage is fine (one quantized operand against f32
    # partners: q = 1.000 on three measured edges), float16 ARITHMETIC is not
    # (two float16 operands met, overflowed at 65504 and the TLM gradient was
    # NaN). f32 rather than bf16: float16 carries 10 mantissa bits.
    "float16": jnp.float32,
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
        out = jnp.result_type(*dts)
    except Exception:
        reps = [_NARROW_PROMOTION_REP.get(d.name, d) for d in dts]
        out = jnp.result_type(*reps)
    # An integer Quant stores CODES: ``val`` is ``round(x / s)`` with ``s``
    # folded into ``scalar_mult``. Two integer operands meeting (both slots of a
    # face quantized) must not multiply as integers: the product of two codes
    # overflows the code type and the reduction sums wrapped values. Measured
    # 2026-09-13 (TLM, Markowitz order, every face slot int8 / int16): the
    # gradient came back with cosine 0.000 against exact, and an int4 grid
    # could not even be summed. Codes are storage; arithmetic on them is
    # float32, which holds every int16 code and every int8 x int8 product
    # exactly. Owner ruling: f32 and bf16 are the compute dtypes.
    if out == jnp.dtype(bool) or jnp.issubdtype(out, jnp.integer):
        return jnp.dtype(jnp.float32)
    # Same rule for a NARROW FLOAT result: ``result_type(f8, f8)`` is f8 and
    # does not raise, so two float8 operands would otherwise compute -- and
    # write their result -- in float8 (measured 2026-09-13: standalone
    # float8 x float8 contraction, result.val float8, NaN).
    if out.name in _NARROW_PROMOTION_REP:
        return jnp.dtype(_NARROW_PROMOTION_REP[out.name])
    return out


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
    # A mixed {bf16, f32} value / scalar_mult pair is UPCAST here, so a Quant'd
    # array is re-promoted to f32 by the densify read before the contraction
    # consumes it. Keeping it narrow instead was measured (2026-08, G3) and it
    # bought nothing on the contraction: the lowered dot census and the convert
    # count were unchanged, and the per-vertex error against exact f32 got
    # slightly worse (relerr 2.93e-3 -> 3.17e-3). It is NOT a no-op, though. It
    # changes the elementwise JOIN: kept narrow, ``bf16 + bf16`` keeps bf16
    # storage; upcast, the f32 scalar_mult drain re-promotes both addends and
    # the sum comes back f32. That is a real precision and storage trade with
    # no measured contraction benefit, so the upcast stays. Take the trade
    # deliberately or not at all.
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
    cdt = _compute_dtype(ldt, rdt)
    if ldt == rdt == cdt:
        # Same at-rest dtype on both sides AND it is a compute dtype: nothing
        # to unify. Two same-NARROW operands (float8 x float8, int8 x int8) do
        # NOT take this path any more: the contraction ran natively on the
        # codes and wrote its result back in the code dtype, which saturated
        # float8 to NaN and wrapped integer sums (measured 2026-09-13,
        # standalone 16x64 @ 64x32: float8 result.val float8 -> NaN, int8 /
        # int16 cosine 0.00 against exact). Codes are storage; arithmetic
        # runs in the compute dtype, bf16 for narrow floats, f32 for ints.
        return lhs, rhs
    # DELIBERATE (2026-08-04): a {bf16, f32} MIXED pair UPCASTS to f32 --
    # quantizing one edge must never silently approximate its exact partner
    # (user decision; an earlier draft downcast the pair). The bf16 fast
    # path exists only when BOTH edges were made bf16: same-dtype pairs
    # return above unchanged, and the contraction then asks for f32
    # accumulation (matmul._emit_einsum).
    return _cast_operand(lhs, cdt), _cast_operand(rhs, cdt)


def _cast_val(val, cdt):
    """``val.astype(cdt)``, behind an optimization barrier when ``val`` is a
    float8_e4m3 variant. XLA's GPU GEMM rewriter pattern-matches
    ``convert(f8e4m3) -> dot`` into a cuBLAS FP8 GEMM and, on Blackwell with
    jax 0.7 / XLA of 2026-09, builds a cyclic graph from it: "A cycle is
    detected while visiting instruction get-tuple-element(cublas-gemm.clone)"
    (TLM, every face slot float8_e4m3fn; float8_e5m2 compiles). The barrier
    keeps the convert out of the pattern. No arithmetic changes.
    """
    out = val.astype(cdt)
    if str(val.dtype).startswith("float8_e4m3"):
        import jax
        out = jax.lax.optimization_barrier(out)
    return out


def _cast_operand(t, cdt):
    """Return ``t`` with ``val`` / ``scalar_mult`` / ``fill_value`` cast to
    ``cdt``. ``val=None`` (pure structure) and ``fill_value=None`` (statically
    zero) markers are preserved; ``scalar_mult`` always exists. Avoids a rebuild
    when nothing changes."""
    new_val = (
        _cast_val(t.val, cdt)
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
