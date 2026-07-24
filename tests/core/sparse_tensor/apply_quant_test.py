"""Unit tests for the ``Quant`` micro-action.

Pins the contract that ``apply_quant`` casts ``val`` only, that the
sequential semantics in ``apply_micro_actions`` are last-wins, and that
``QUANT_DTYPES`` resolves on the running JAX.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex
from graphax.sparse.micro_actions import (
    Compress,
    Diag,
    QUANT_DTYPES,
    QUANT_DTYPE_INDEX,
    Quant,
    apply_compress,
    apply_micro_actions,
    apply_quant,
    quant,
)
from graphax.sparse.tensor import SparseTensor


def _make_dense_pair_st(
    n1: int, n2: int, dtype=jnp.float32, fill_value=None,
) -> tuple[SparseTensor, jnp.ndarray]:
    """Build a SparseTensor backed by a fully-dense (n1, n2) val.

    ``fill_value`` defaults to ``None`` (the post-Phase-8 statically-zero
    marker); pass an explicit array to exercise a materialized fill.
    """
    val = jnp.arange(n1 * n2, dtype=dtype).reshape(n1, n2)
    d1 = DenseIndex(id=0, size=n1, axis=0)
    d2 = DenseIndex(id=1, size=n2, axis=1)
    return SparseTensor((d1,), (d2,), val, fill_value=fill_value), val


def test_quant_basic_cast_val_only():
    """float32 → float16 casts val; scalar_mult and fill_value stay native."""
    # Explicit float32 fill so we can assert it is preserved unquantized
    # (the default fill_value=None stays None — checked separately below).
    st, _ = _make_dense_pair_st(4, 4, fill_value=jnp.array(0.0, dtype=jnp.float32))
    assert st.val.dtype == jnp.float32
    assert st.scalar_mult.dtype == jnp.float32
    assert st.fill_value.dtype == jnp.float32

    new_st = apply_quant(st, Quant("float16"))

    assert new_st.val.dtype == jnp.float16
    # The user's "cast val only" decision: scalar_mult / fill_value are
    # intentionally preserved in their native dtype.
    assert new_st.scalar_mult.dtype == jnp.float32
    assert new_st.fill_value.dtype == jnp.float32

    # The statically-zero (fill_value=None) default is likewise preserved.
    st_none, _ = _make_dense_pair_st(4, 4)
    assert st_none.fill_value is None
    assert apply_quant(st_none, Quant("float16")).fill_value is None


def test_quant_noop_same_dtype_returns_same_instance():
    """A Quant whose target matches val.dtype is a true no-op."""
    st, _ = _make_dense_pair_st(4, 4)
    out = apply_quant(st, Quant("float32"))
    # Implementation early-returns `st` unchanged when the dtype matches.
    assert out is st


def test_quant_val_none_returns_unchanged():
    """val=None ⇒ Quant is a no-op (the ST has no array to cast)."""
    d1 = DenseIndex(id=0, size=4, axis=None)
    d2 = DenseIndex(id=1, size=4, axis=None)
    # sort_val=False bypasses the val=None path inside _sort_val.
    st = SparseTensor((d1,), (d2,), None, sort_val=False)
    assert st.val is None
    out = apply_quant(st, Quant("float16"))
    assert out is st


def test_quant_sequential_last_wins():
    """[Quant('float16'), Quant('int8')] ⇒ final val.dtype == int8."""
    st, _ = _make_dense_pair_st(4, 4)
    out = apply_micro_actions(st, [Quant("float16"), Quant("int8")])
    assert out.val.dtype == jnp.int8
    # Last cast wins; since the target is an integer, it undergoes symmetric scaling.
    # The maximum value is 15, and int8 max is 127. So the scale is 15/127.
    # We verify that the values match the expected scaled output.
    s = 15.0 / 127.0
    expected = np.round(np.arange(16, dtype=np.float32) / s).astype(np.int8).reshape(4, 4)
    np.testing.assert_array_equal(
        np.asarray(out.val), expected,
    )


def test_quant_mixed_with_diag_and_compress():
    """[Diag, Quant, Compress] runs end-to-end; final val dtype is float16."""
    st, _ = _make_dense_pair_st(4, 4)
    out = apply_micro_actions(
        st,
        [Diag(i=0, j=1, factor=2), Quant("float16"), Compress(axes=(0,), kind="mean")],
    )
    # jnp.mean over a float16 array stays float16 (XLA does not auto-promote).
    assert out.val.dtype == jnp.float16


def test_quant_invalid_dtype_raises():
    """An unknown dtype name raises ValueError naming the legal set."""
    with pytest.raises(ValueError, match="Quant.dtype must be one of"):
        Quant("not_a_dtype")


def test_quant_factory_helper_matches_apply_quant():
    """``quant(name)(st)`` matches ``apply_quant(st, Quant(name))``."""
    st, _ = _make_dense_pair_st(4, 4)
    fn = quant("bfloat16")
    via_factory = fn(st)
    via_direct = apply_quant(st, Quant("bfloat16"))
    assert via_factory.val.dtype == via_direct.val.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(via_factory.val.astype(jnp.float32)),
        np.asarray(via_direct.val.astype(jnp.float32)),
    )
    assert fn.__name__ == "quant('bfloat16')"


def test_quant_dtype_catalog_resolves_and_roundtrips():
    """Every QUANT_DTYPES name builds a Quant and resolves via jnp.dtype()."""
    assert len(QUANT_DTYPES) > 0
    assert len(QUANT_DTYPES) == len(QUANT_DTYPE_INDEX)
    for name in QUANT_DTYPES:
        # The index map must agree with positional order.
        assert QUANT_DTYPE_INDEX[name] == QUANT_DTYPES.index(name)
        # The catalog name must construct a Quant without error.
        action = Quant(name)
        assert action.dtype == name
        # And the name must be resolvable by jnp.dtype() on the running JAX.
        try:
            jnp.dtype(name)
        except (TypeError, ValueError) as e:
            pytest.skip(
                f"jnp.dtype({name!r}) not resolvable on this JAX/platform: {e}"
            )


# ---------------------------------------------------------------------------
# Unsigned sign-flip half-range (no zero-point)
# ---------------------------------------------------------------------------


def _signed_st() -> SparseTensor:
    """A (2, 2) edge whose val straddles zero, max magnitude 4."""
    return SparseTensor(
        (DenseIndex(id=0, size=2, axis=0),),
        (DenseIndex(id=1, size=2, axis=1),),
        jnp.array([[-4.0, -2.0], [2.0, 4.0]], dtype=jnp.float32),
    )


def _dequant(st: SparseTensor) -> np.ndarray:
    """The logical value a consumer sees: ``val * scalar_mult``."""
    return np.asarray(
        st.val.astype(np.float32) * np.float32(st.scalar_mult)).reshape(-1)


def test_quant_uint_sign_flip_keeps_the_chosen_arm():
    """``scale_sign`` picks which arm of a signed val fills the unsigned range;
    the other arm clips to 0. No zero-point offset is stored."""
    st = _signed_st()
    pos = apply_quant(st, Quant("uint8", scale_sign=1))
    neg = apply_quant(st, Quant("uint8", scale_sign=-1))
    assert pos.val.dtype == jnp.uint8 and neg.val.dtype == jnp.uint8
    # +1 -> positive arm survives ([-4,-2] clip to 0); -1 -> negative arm.
    np.testing.assert_allclose(_dequant(pos), [0.0, 0.0, 2.0, 4.0], atol=0.05)
    np.testing.assert_allclose(_dequant(neg), [-4.0, -2.0, 0.0, 0.0], atol=0.05)


def test_quant_scales_to_the_target_max_not_the_value_max():
    """The kept arm's largest magnitude maps onto the TARGET dtype's max code
    (uint8 -> 255), i.e. the full range of the new type is used."""
    q = apply_quant(_signed_st(), Quant("uint8", scale_sign=1))
    assert int(np.asarray(q.val).max()) == 255            # 4 -> 255, scale 4/255


def test_quant_scale_sign_must_be_plus_or_minus_one():
    with pytest.raises(ValueError, match="scale_sign"):
        Quant("uint8", scale_sign=0)


def test_quant_signed_target_is_unchanged_by_sign_plus_one():
    """A signed target with the default sign reduces to the old symmetric
    quantizer (guards the no-regression claim for the int path)."""
    st = _signed_st()
    out = apply_quant(st, Quant("int8"))          # default scale_sign=1
    # scale = 4/127; codes are round(val / scale), symmetric about 0.
    s = 4.0 / 127.0
    expected = np.round(np.array([-4, -2, 2, 4], np.float32) / s).astype(np.int8)
    np.testing.assert_array_equal(np.asarray(out.val).reshape(-1), expected)
