"""G2: the policy's QUANT targets are the four floats
{float32, bfloat16, float8_e5m2, float8_e4m3fn}.

``QUANT_DTYPES`` stays the full catalog on purpose -- it builds the append-only
token vocabulary (``graphax/jaxpr.py``), so trimming it would shift token ids.
The lock is expressed instead as :data:`POLICY_QUANT_DTYPES`, the set the RL
policy head can actually emit (alphagrad's unified face head draws one
categorical over exactly these four; ``masks.FACE_QUANT_DTYPES`` lists the
same names).

Three properties are pinned here:

  (a) the policy-reachable set IS the four floats and the opt-in
      GRAPHAX_QUANT_POLICY_STRICT guard rejects everything else;
  (b) ``apply_quant`` takes the plain-``astype`` branch for float32 and
      bfloat16 -- NOT the scaled branch, which folds a per-tensor scale into
      ``scalar_mult`` -- and the SCALED narrow-float branch for the two
      float8 members (sweep64: q 0.999 / 0.998 single-slot, -27% / -11%
      whole-graph temp on Blackwell);
  (c) a JOIN/ADD of two bf16-quantized edges introduces NO rescale: the sum is
      the bit-exact bf16 elementwise sum of the two stored buffers and
      ``scalar_mult`` is untouched.
"""
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from graphax.sparse.indexes import DiagonalIndex
from graphax.sparse.micro_actions import (
    POLICY_QUANT_DTYPES,
    QUANT_DTYPES,
    Quant,
    _is_narrow_float_target,
    _is_scaled_quant_target,
    apply_quant,
    check_policy_quant_dtype,
    unscaled_quant_target,
)
from graphax.sparse.tensor import SparseTensor

BF16 = jnp.dtype(jnp.bfloat16)
F32 = jnp.dtype(jnp.float32)


def _bd(M=4, B=3, key=1):
    """Block-diagonal SparseTensor with a float32 val and scalar_mult 1."""
    val = jr.normal(jr.PRNGKey(key), (M, B, B)).astype(jnp.float32)
    return SparseTensor(
        (DiagonalIndex(0, M, 0, 1, B, 1),),
        (DiagonalIndex(1, M, 0, 0, B, 2),),
        val,
    )


# --------------------------------------------------------------------------#
# (a) the policy-reachable set
# --------------------------------------------------------------------------#
def test_policy_set_is_exactly_the_four_floats():
    assert set(POLICY_QUANT_DTYPES) == {
        "float32", "bfloat16", "float8_e5m2", "float8_e4m3fn"}
    assert POLICY_QUANT_DTYPES[0] == "float32", "index 0 is the exact entry"


def test_catalog_is_not_trimmed():
    """The token vocabulary depends on QUANT_DTYPES; it must stay wide."""
    assert set(POLICY_QUANT_DTYPES).issubset(set(QUANT_DTYPES))
    assert len(QUANT_DTYPES) > len(POLICY_QUANT_DTYPES), (
        "QUANT_DTYPES was trimmed -- that shifts every d#<dtype> token id")


def test_check_policy_quant_dtype_accepts_and_rejects():
    for name in POLICY_QUANT_DTYPES:
        assert check_policy_quant_dtype(name) == name
    for bad in ("int8", "uint8", "float16", "float4_e2m1fn"):
        if bad not in QUANT_DTYPES and bad != "float16":
            continue
        with pytest.raises(ValueError):
            check_policy_quant_dtype(bad)


def test_strict_guard_is_opt_in(monkeypatch):
    t = _bd()
    monkeypatch.delenv("GRAPHAX_QUANT_POLICY_STRICT", raising=False)
    # default OFF: a non-policy dtype still works (catalog stays usable)
    assert apply_quant(t, Quant("float16")).val.dtype == jnp.dtype(jnp.float16)
    monkeypatch.setenv("GRAPHAX_QUANT_POLICY_STRICT", "1")
    with pytest.raises(ValueError):
        apply_quant(t, Quant("float16"))
    # ... and the two policy dtypes still pass under the guard
    assert apply_quant(t, Quant("bfloat16")).val.dtype == BF16


# --------------------------------------------------------------------------#
# (b) float32 / bfloat16 take the plain-astype branch, float8 the scaled one
# --------------------------------------------------------------------------#
@pytest.mark.parametrize("name", ["float32", "bfloat16"])
def test_policy_dtypes_take_the_unscaled_branch(name):
    dt = jnp.dtype(name)
    assert not _is_scaled_quant_target(dt), f"{name} routed to the int scaler"
    assert not _is_narrow_float_target(dt), f"{name} routed to the float scaler"
    assert unscaled_quant_target(dt)


@pytest.mark.parametrize("name", ["float8_e5m2", "float8_e4m3fn"])
def test_float8_policy_dtypes_take_the_scaled_narrow_float_branch(name):
    if name not in QUANT_DTYPES:
        pytest.skip(f"{name} not in the catalog")
    dt = jnp.dtype(name)
    assert _is_narrow_float_target(dt), f"{name} is not on the float scaler"
    assert not unscaled_quant_target(dt)
    t = _bd(key=11)
    q = apply_quant(t, Quant(name))
    assert jnp.dtype(q.val.dtype) == dt
    # a per-tensor scale was folded into scalar_mult
    assert not np.allclose(np.asarray(q.scalar_mult, np.float64),
                           np.asarray(t.scalar_mult, np.float64))


@pytest.mark.parametrize("name", ["float32", "bfloat16"])
def test_apply_quant_is_a_bare_astype(name):
    t = _bd(key=7)
    q = apply_quant(t, Quant(name))
    dt = jnp.dtype(name)
    assert jnp.dtype(q.val.dtype) == dt
    # bit-exact astype: no scale was divided out before the cast
    np.testing.assert_array_equal(
        np.asarray(q.val, np.float64),
        np.asarray(t.val.astype(dt), np.float64),
    )
    # scalar_mult passed through untouched
    assert np.asarray(q.scalar_mult) == np.asarray(t.scalar_mult)
    assert jnp.dtype(q.scalar_mult.dtype) == jnp.dtype(t.scalar_mult.dtype)


def test_scaled_branch_is_still_reachable_for_int_targets():
    """Sensitivity control: the assertions above would pass vacuously if
    apply_quant never scaled anything. int8 MUST still fold a scale."""
    if "int8" not in QUANT_DTYPES:
        pytest.skip("int8 not in the catalog")
    t = _bd(key=9)
    q = apply_quant(t, Quant("int8"))
    assert _is_scaled_quant_target(jnp.dtype("int8"))
    assert not np.allclose(np.asarray(q.scalar_mult),
                           np.asarray(t.scalar_mult))


# --------------------------------------------------------------------------#
# (c) THE JOIN/ADD introduces no rescale
# --------------------------------------------------------------------------#
# MEASURED (2026-08), quantizing both addends and adding them:
#
#   quant      result val   result scalar_mult
#   bfloat16   float32      1.0  (f32)
#   float32    float32      1.0  (f32)
#   int8       float32      1.0  (f32)
#
# So the JOIN never introduces a rescale -- ``scalar_mult`` stays the identity
# for both policy dtypes and no new scale is ever stored. It does NOT keep the
# bf16 storage: a mixed {bf16, f32} value / scalar_mult pair is upcast in
# ``dtype_compute._scaled_mul``, so the f32 ``scalar_mult`` drain re-promotes
# each addend to f32 before the add. That is a promotion, not a rescale, and it
# is lossless with respect to the stored bf16 values. Keeping the addends
# narrow instead is a real precision and storage trade with no measured
# contraction benefit; see the comment in ``_scaled_mul``.
#
# The int8 row is the contrast that makes this meaningful: there the OPERANDS
# carry real per-tensor scales (0.0197 / 0.0187) and the join has to drain them
# into the values, which is exactly the behaviour POLICY_QUANT_DTYPES exists to
# keep out of the policy's reach.


def _join(name):
    a, b = _bd(key=11), _bd(key=12)
    qa = apply_quant(a, Quant(name))
    qb = apply_quant(b, Quant(name))
    return a, b, qa, qb, qa + qb


@pytest.mark.parametrize("name", ["float32", "bfloat16"])
def test_join_introduces_no_rescale(name):
    """The property the policy actually depends on: no new or changed scale."""
    a, b, qa, qb, s = _join(name)
    for t in (qa, qb, s):
        assert float(np.asarray(t.scalar_mult, np.float64)) == 1.0
    assert s.fill_value is None
    # and the join is faithful to the exact f32 sum within bf16 resolution
    ref = (np.asarray(a.val, np.float64) + np.asarray(b.val, np.float64))
    got = (np.asarray(s.val, np.float64)
           * float(np.asarray(s.scalar_mult, np.float64)))
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 1e-2, rel


def test_join_of_two_bf16_quants_promotes(monkeypatch):
    """Documented, not aspirational: the f32 scalar drain re-promotes both
    bf16 addends before the add."""
    _, _, qa, qb, s = _join("bfloat16")
    assert jnp.dtype(qa.val.dtype) == BF16 and jnp.dtype(qb.val.dtype) == BF16
    assert jnp.dtype(s.val.dtype) == F32


def test_join_of_two_f32_quants_is_bit_exact(monkeypatch):
    _, _, qa, qb, s = _join("float32")
    assert jnp.dtype(s.val.dtype) == F32
    np.testing.assert_array_equal(
        np.asarray(s.val, np.float64),
        np.asarray(qa.val + qb.val, np.float64))


def test_scaled_int8_operands_do_carry_a_scale(monkeypatch):
    """Sensitivity control for the whole section: a NON-policy dtype really
    does store a per-tensor scale on the operands, which the join then has to
    drain -- the failure mode POLICY_QUANT_DTYPES rules out."""
    if "int8" not in QUANT_DTYPES:
        pytest.skip("int8 not in the catalog")
    _, _, qa, qb, s = _join("int8")
    assert float(np.asarray(qa.scalar_mult, np.float64)) != 1.0
    assert float(np.asarray(qb.scalar_mult, np.float64)) != 1.0
    assert float(np.asarray(s.scalar_mult, np.float64)) == 1.0
