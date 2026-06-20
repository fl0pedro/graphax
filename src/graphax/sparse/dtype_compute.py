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


def _scaled_mul(value, scalar_mult):
    """``value * scalar_mult`` at their highest common dtype.

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
    cdt = _compute_dtype(vdt, sdt)
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
    cdt = _compute_dtype(ldt, rdt)
    return _cast_operand(lhs, cdt), _cast_operand(rhs, cdt)


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
