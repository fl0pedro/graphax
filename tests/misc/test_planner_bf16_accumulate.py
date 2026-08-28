"""G3: bf16 accumulation in the einsum_general PLANNER.

WHAT WAS EXPECTED: the planner emitted a bare ``jnp.einsum`` with no
``preferred_element_type``, unlike the tiled / densify dot sites
(``ops.matmul._gx_dot_general``), so a quantize-every-vertex bf16 plan was
believed to lower to ZERO bf16 dots. Adding f32 accumulation was supposed to
make bf16 dots appear.

WHAT WAS MEASURED (2026-08, mlp2 / mlp4_multimatmul / attn, planner +
demand-emit):

  1. A quantize-every-VERTEX plan really does lower to zero bf16 dots -- but
     NOT because of the missing kwarg. The per-vertex hook fires on the
     contraction RESULT, so every contraction is (quantized accumulated edge)
     x (exact elemental partial): a MIXED {bf16, f32} pair, which
     ``_unify_operand_dtypes`` upcasts to f32 by deliberate design (see the
     2026-08-04 note there). No planner change can alter that.

  2. When BOTH operands are quantized (the per-face ``lhs``/``rhs`` slots),
     the planner ALREADY lowers genuine bf16 dots -- 6 on mlp2, 16 on mlp4,
     26 on attn. The premise that narrow vals never reach the planner is
     false.

  3. Passing ``preferred_element_type=float32`` to the planner einsum makes
     things WORSE: for the batched, size-1-carrying einsum forms the planner
     actually emits, ``jnp.einsum`` implements the request by converting both
     operands to f32 first, deleting every bf16 dot and adding ~50% more
     converts, with no change in error against the exact f32 Jacobian.

So the accumulation request ships behind GRAPHAX_PLANNER_F32_ACCUM, default
OFF and byte-identical. These tests pin all three findings.
"""
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import jacve
from graphax.sparse.dtype_compute import (
    _quant_narrow_bf16_pair, _unify_operand_dtypes)
from graphax.sparse.indexes import DiagonalIndex
from graphax.sparse.lower.matmul import _einsum_accum_dtype
from graphax.sparse.micro_actions import Quant, apply_quant
from graphax.sparse.tensor import SparseTensor

BF16 = jnp.dtype(jnp.bfloat16)
F32 = jnp.dtype(jnp.float32)

_DOT = re.compile(r"stablehlo\.dot_general|stablehlo\.dot\b")
_TY = re.compile(r"tensor<[^>]*x(f32|bf16|f16|f64)>")


def _dot_census(hlo):
    out = {}
    for line in hlo.splitlines():
        if _DOT.search(line):
            k = ",".join(sorted(set(_TY.findall(line)))) or "?"
            out[k] = out.get(k, 0) + 1
    return out


def _n_bf16_dots(hlo):
    return sum(n for k, n in _dot_census(hlo).items() if "bf16" in k)


# --------------------------------------------------------------------------#
# the accumulation-dtype decision
# --------------------------------------------------------------------------#
def test_accum_dtype_is_off_by_default(monkeypatch):
    """Default is byte-identical: no preferred_element_type is ever asked
    for, whatever the operand dtypes."""
    monkeypatch.delenv("GRAPHAX_PLANNER_F32_ACCUM", raising=False)
    monkeypatch.delenv("GRAPHAX_QUANT_NARROW_GEMM", raising=False)
    for dts in ([BF16, BF16], [BF16], [F32, F32], [BF16, F32], []):
        assert _einsum_accum_dtype(dts) is None


def test_accum_dtype_opt_in_only_for_all_bf16(monkeypatch):
    monkeypatch.setenv("GRAPHAX_PLANNER_F32_ACCUM", "1")
    monkeypatch.delenv("GRAPHAX_QUANT_NARROW_GEMM", raising=False)
    assert _einsum_accum_dtype([BF16, BF16]) is jnp.float32
    assert _einsum_accum_dtype([BF16]) is jnp.float32
    # anything else keeps jnp.einsum's own default -> exact AD is untouched
    assert _einsum_accum_dtype([F32, F32]) is None
    assert _einsum_accum_dtype([BF16, F32]) is None
    assert _einsum_accum_dtype([jnp.dtype(jnp.float16)] * 2) is None
    assert _einsum_accum_dtype([jnp.dtype(jnp.int32)] * 2) is None
    assert _einsum_accum_dtype([]) is None


def test_accum_dtype_respects_the_narrow_gemm_knob(monkeypatch):
    monkeypatch.setenv("GRAPHAX_PLANNER_F32_ACCUM", "1")
    monkeypatch.setenv("GRAPHAX_QUANT_NARROW_GEMM", "0")
    assert _einsum_accum_dtype([BF16, BF16]) is None


def test_planner_exclusion_in_the_narrow_pair_guard_is_intact(monkeypatch):
    """Finding 2. Relaxing this changes no contraction (same dot census and
    convert count on mlp2 under an isolated toggle) but DOES change the
    elementwise join dtype -- see test_policy_quant_dtypes for the measured
    table. No contraction benefit, so the exclusion stays."""
    monkeypatch.delenv("GRAPHAX_QUANT_NARROW_GEMM", raising=False)
    monkeypatch.setenv("GRAPHAX_EINSUM_GENERAL", "1")
    assert _quant_narrow_bf16_pair(BF16, F32) is False
    monkeypatch.setenv("GRAPHAX_EINSUM_GENERAL", "0")
    assert _quant_narrow_bf16_pair(BF16, F32) is True
    assert _quant_narrow_bf16_pair(F32, BF16) is True
    assert _quant_narrow_bf16_pair(F32, F32) is False
    assert _quant_narrow_bf16_pair(BF16, BF16) is False
    monkeypatch.setenv("GRAPHAX_QUANT_NARROW_GEMM", "0")
    assert _quant_narrow_bf16_pair(BF16, F32) is False


# --------------------------------------------------------------------------#
# finding 1: a MIXED pair is upcast, so quantizing one edge is never enough
# --------------------------------------------------------------------------#
def _bd(M, B, key, dt=jnp.float32):
    v = jax.random.normal(jax.random.PRNGKey(key), (M, B, B)).astype(dt)
    return SparseTensor((DiagonalIndex(0, M, 0, 1, B, 1),),
                        (DiagonalIndex(1, M, 0, 0, B, 2),), v)


def test_mixed_bf16_f32_pair_upcasts_to_f32():
    """This is WHY a quantize-every-vertex plan yields zero bf16 dots: the
    contraction is always (quantized result) x (exact elemental partial)."""
    a = apply_quant(_bd(4, 3, 1), Quant("bfloat16"))
    b = _bd(4, 3, 2)
    la, rb = _unify_operand_dtypes(a, b)
    assert jnp.dtype(la.dtype) == F32 and jnp.dtype(rb.dtype) == F32


def test_both_bf16_pair_is_left_alone():
    """... and WHY the per-face lhs/rhs slots do produce bf16 dots."""
    a = apply_quant(_bd(4, 3, 1), Quant("bfloat16"))
    b = apply_quant(_bd(4, 3, 2), Quant("bfloat16"))
    la, rb = _unify_operand_dtypes(a, b)
    assert la is a and rb is b
    assert jnp.dtype(la.dtype) == BF16 and jnp.dtype(rb.dtype) == BF16


def test_planner_lowers_a_both_bf16_contraction_to_a_bf16_dot():
    """Finding 2, end to end at the SparseTensor level: narrow vals DO reach
    the planner's einsum and lower to a real bf16 dot."""
    from graphax.sparse.ops.matmul import matmul

    def f(x, y):
        return matmul(x, y).val

    a32, b32 = _bd(8, 6, 1), _bd(8, 6, 2)
    a16, b16 = _bd(8, 6, 1, jnp.bfloat16), _bd(8, 6, 2, jnp.bfloat16)
    assert _n_bf16_dots(jax.jit(f).lower(a32, b32).as_text()) == 0
    assert _n_bf16_dots(jax.jit(f).lower(a16, b16).as_text()) > 0


# --------------------------------------------------------------------------#
# end-to-end sanity: the default lowering is unchanged and still correct
# --------------------------------------------------------------------------#
def _mlp2():
    rng = np.random.RandomState(0)
    n = 8
    W1 = jnp.asarray(rng.randn(n, n).astype(np.float32) / np.sqrt(n))
    W2 = jnp.asarray(rng.randn(n, n).astype(np.float32) / np.sqrt(n))
    X = jnp.asarray(rng.randn(4, n).astype(np.float32))

    def loss(X, W1, W2):
        h = jnp.tanh(X @ W1)
        y = jnp.tanh(h @ W2)
        return jnp.sum(y * y)

    return loss, [X, W1, W2], (1, 2)


def _lower(fn, args, argnums, order, transforms):
    kw = {"transforms": transforms} if transforms else {}
    j = jax.jit(jacve(fn, order, argnums=argnums, has_aux=False,
                      sparse_representation=False, **kw), keep_unused=True)
    return j, j.lower(*args).as_text()


def test_exact_ad_lowering_has_no_bf16_dots():
    fn, args, argnums = _mlp2()
    n = len(jax.make_jaxpr(fn)(*args).jaxpr.eqns)
    _, hlo = _lower(fn, args, argnums, list(range(1, n + 1)), None)
    assert _n_bf16_dots(hlo) == 0, "exact AD must never carry narrow vals"


def test_per_vertex_bf16_plan_lowers_to_zero_bf16_dots():
    """Finding 1, end to end. Documented, not aspirational: it is the mixed
    -pair upcast, and it is the same before and after this change."""
    fn, args, argnums = _mlp2()
    n = len(jax.make_jaxpr(fn)(*args).jaxpr.eqns)
    order = list(range(1, n + 1))
    _, hlo = _lower(fn, args, argnums, order,
                    [(v, (Quant("bfloat16"),)) for v in order])
    assert _n_bf16_dots(hlo) == 0


def test_per_vertex_bf16_plan_still_computes_the_right_jacobian():
    fn, args, argnums = _mlp2()
    n = len(jax.make_jaxpr(fn)(*args).jaxpr.eqns)
    order = list(range(1, n + 1))
    j_exact, _ = _lower(fn, args, argnums, order, None)
    j_quant, _ = _lower(fn, args, argnums, order,
                        [(v, (Quant("bfloat16"),)) for v in order])

    def flat(o):
        return np.concatenate(
            [np.asarray(x, np.float64).ravel() for x in jax.tree.leaves(o)])

    ve, vq = flat(j_exact(*args)), flat(j_quant(*args))
    cos = float(ve @ vq / (np.linalg.norm(ve) * np.linalg.norm(vq)))
    assert cos > 0.99, f"bf16 plan cosine vs exact f32 = {cos}"
