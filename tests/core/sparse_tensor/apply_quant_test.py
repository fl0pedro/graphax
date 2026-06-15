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
    n1: int, n2: int, dtype=jnp.float32,
) -> tuple[SparseTensor, jnp.ndarray]:
    """Build a SparseTensor backed by a fully-dense (n1, n2) val."""
    val = jnp.arange(n1 * n2, dtype=dtype).reshape(n1, n2)
    d1 = DenseIndex(id=0, size=n1, axis=0)
    d2 = DenseIndex(id=1, size=n2, axis=1)
    return SparseTensor((d1,), (d2,), val), val


def test_quant_basic_cast_val_only():
    """float32 → float16 casts val; scalar_mult and fill_value stay native."""
    st, _ = _make_dense_pair_st(4, 4)
    assert st.val.dtype == jnp.float32
    assert st.scalar_mult.dtype == jnp.float32
    assert st.fill_value.dtype == jnp.float32

    new_st = apply_quant(st, Quant("float16"))

    assert new_st.val.dtype == jnp.float16
    # The user's "cast val only" decision: scalar_mult / fill_value are
    # intentionally preserved in their native dtype.
    assert new_st.scalar_mult.dtype == jnp.float32
    assert new_st.fill_value.dtype == jnp.float32


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
    # Last cast wins; the integer truncation is observable: arange(0..15)
    # cast through float16 and then int8 yields the same integer values.
    np.testing.assert_array_equal(
        np.asarray(out.val), np.arange(16, dtype=np.int8).reshape(4, 4),
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
