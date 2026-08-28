"""G1: ``GRAPHAX_QUANT_PULLDOWN`` is gone, and removing it changed nothing.

The pulldown branch made a mixed ``{half, full}`` float pair compute at the
HALF dtype instead of being promoted. It was gated on an env var that defaults
to "0", so with the flag unset ``_compute_dtype`` was already exactly the
historical highest-common policy: native ``jnp.result_type``, falling back to
the narrow->standard representative map only when JAX itself refuses to
promote. These tests pin that the collapsed function reproduces that policy on
a full matrix of dtype pairs / triples, and that the env var no longer has any
effect at all.
"""
import itertools

import jax.numpy as jnp
import pytest

import graphax.sparse.dtype_compute as DC
from graphax.sparse.dtype_compute import _NARROW_PROMOTION_REP, _compute_dtype


def _reference_compute_dtype(*dtypes):
    """The pre-change function's FLAG-OFF path, transcribed verbatim."""
    dts = [jnp.dtype(d) for d in dtypes]
    try:
        return jnp.result_type(*dts)
    except Exception:
        reps = [_NARROW_PROMOTION_REP.get(d.name, d) for d in dts]
        return jnp.result_type(*reps)


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
        # same-narrow pairs: result_type may or may not accept them natively,
        # but either way the answer must equal the historical policy.
        assert got == _reference_compute_dtype(name, name)
        assert _compute_dtype(name, jnp.float32) == _reference_compute_dtype(
            name, jnp.float32)
        assert jnp.dtype(rep) is not None
