"""G1: ``GRAPHAX_QUANT_PULLDOWN`` is gone, and removing it changed nothing.

The pulldown branch made a mixed ``{half, full}`` float pair compute at the
HALF dtype instead of being promoted. It was gated on an env var that defaults
to "0", so with the flag unset ``_compute_dtype`` was already exactly the
historical highest-common policy: native ``jnp.result_type``, falling back to
the narrow->standard representative map only when JAX itself refuses to
promote. These tests pin that the collapsed function reproduces that policy on
a full matrix of dtype pairs / triples, and that the env var no longer has any
effect at all.

2026-09-13 (owner ruling: f32 and bf16 are the compute dtypes) the policy
gained two post-rules on top of the historical one, both measured on TLM with
every face slot quantized (Blackwell):
  * an integer or bool RESULT computes in float32 -- integer Quant codes
    multiplied as integers overflowed and the gradient cosine was 0.000;
  * a narrow-float RESULT (float8 / float4, and float16) computes in its
    representative -- two float8 operands contracted natively wrote a float8
    result (NaN), and float16 arithmetic overflowed at 65504 (NaN).
The reference below is the historical policy WITH those two rules, stated
separately so a change to either is a visible diff here.
"""
import itertools

import jax.numpy as jnp
import pytest

import graphax.sparse.dtype_compute as DC
from graphax.sparse.dtype_compute import _NARROW_PROMOTION_REP, _compute_dtype


def _historical_compute_dtype(*dtypes):
    """The pre-2026-09-13 function's FLAG-OFF path, transcribed verbatim."""
    dts = [jnp.dtype(d) for d in dtypes]
    try:
        return jnp.result_type(*dts)
    except Exception:
        reps = [_NARROW_PROMOTION_REP.get(d.name, d) for d in dts]
        return jnp.result_type(*reps)


def _reference_compute_dtype(*dtypes):
    """Historical policy plus the two 2026-09-13 rules (module docstring)."""
    out = _historical_compute_dtype(*dtypes)
    if out == jnp.dtype(bool) or jnp.issubdtype(out, jnp.integer):
        return jnp.dtype(jnp.float32)          # codes are storage, not arithmetic
    if out.name in _NARROW_PROMOTION_REP:
        return jnp.dtype(_NARROW_PROMOTION_REP[out.name])
    return out


_CANDIDATES = [
    "bool", "int8", "int16", "int32", "uint8", "uint16", "uint32",
    "float16", "bfloat16", "float32", "complex64",
] + sorted(_NARROW_PROMOTION_REP)


def _usable(name):
    try:
        jnp.dtype(name)
        return True
    except Exception:
        return False


ALL_DTYPES = [d for d in _CANDIDATES if _usable(d)]
NARROW = [d for d in sorted(_NARROW_PROMOTION_REP) if _usable(d)]


def test_pulldown_symbol_is_gone():
    assert not hasattr(DC, "_quant_pulldown")


@pytest.mark.parametrize("a", ALL_DTYPES)
def test_pairs_match_historical_policy(a):
    for b in ALL_DTYPES:
        assert _compute_dtype(a, b) == _reference_compute_dtype(a, b), (a, b)


def test_singletons_match_historical_policy():
    for a in ALL_DTYPES:
        assert _compute_dtype(a) == _reference_compute_dtype(a), a


def test_triples_match_historical_policy():
    for combo in itertools.combinations_with_replacement(ALL_DTYPES, 3):
        assert _compute_dtype(*combo) == _reference_compute_dtype(*combo), combo


def test_env_var_no_longer_changes_anything(monkeypatch):
    before = {(a, b): _compute_dtype(a, b)
              for a in ALL_DTYPES for b in ALL_DTYPES}
    for value in ("1", "0", "true"):
        monkeypatch.setenv("GRAPHAX_QUANT_PULLDOWN", value)
        after = {(a, b): _compute_dtype(a, b)
                 for a in ALL_DTYPES for b in ALL_DTYPES}
        assert after == before, f"GRAPHAX_QUANT_PULLDOWN={value} still bites"


def test_mixed_half_full_pair_promotes_up():
    """The specific case the pulldown branch used to invert."""
    assert _compute_dtype(jnp.bfloat16, jnp.float32) == jnp.dtype(jnp.float32)
    assert _compute_dtype(jnp.float16, jnp.float32) == jnp.dtype(jnp.float32)
    assert _compute_dtype(jnp.bfloat16, jnp.bfloat16) == jnp.dtype(jnp.bfloat16)


def test_narrow_operands_still_map_to_their_representative():
    for name in NARROW:
        rep = jnp.dtype(_NARROW_PROMOTION_REP[name])
        got = _compute_dtype(name, name)
        # same-narrow pairs: result_type may or may not accept them natively
        # (it accepts float16 x float16 and float8 x float8), but either way
        # the arithmetic dtype is the representative -- or float32 when the
        # representative is itself an integer container (int4 -> int8 -> f32).
        want = jnp.dtype(jnp.float32) if jnp.issubdtype(rep, jnp.integer) else rep
        assert got == want, (name, got, want)
        assert _compute_dtype(name, jnp.float32) == _reference_compute_dtype(
            name, jnp.float32)


def test_the_2026_09_13_rules_by_example():
    """The measured cases, spelled out (TLM, every face slot quantized)."""
    f32, bf16 = jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16)
    assert _compute_dtype("int8", "int8") == f32          # cosine 0.000 before
    assert _compute_dtype("int16", "int16") == f32
    assert _compute_dtype("int8", "float32") == f32
    assert _compute_dtype("bool", "bool") == f32
    assert _compute_dtype("float16", "float16") == f32    # NaN at 65504 before
    assert _compute_dtype("float16", "float32") == f32
    assert _compute_dtype("bfloat16", "bfloat16") == bf16 # the compute dtype itself
    for n in ("float8_e4m3fn", "float8_e5m2", "float4_e2m1fn"):
        if n in NARROW:
            assert _compute_dtype(n, n) == bf16, n        # float8 result, NaN before
            assert _compute_dtype(n, "float32") == f32, n
    assert _compute_dtype("float32", "float32") == f32
