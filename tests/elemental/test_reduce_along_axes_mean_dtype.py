"""Regression tests for the fp8/bf16 mean-accumulation defect (dsnn-dfw.92).

``_reduce_along_axes`` used to run ``jnp.mean`` in the operand dtype. A 700
entry mean in a float8 format with no infinity overflows to NaN, and in a
format with a small max it can also read a wrong finite value. The fix
accumulates the mean in float32 for every dtype narrower than 32 bits and
casts the result back to the operand dtype.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from graphax.sparse.micro_actions import _reduce_along_axes

jax.config.update("jax_enable_x64", False)


NARROW_FLOAT_DTYPES = [jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.bfloat16]


@pytest.mark.parametrize("dtype", NARROW_FLOAT_DTYPES)
def test_mean_of_ones_is_finite_and_one(dtype):
    val = jnp.ones((128, 700), dtype=dtype)
    result = _reduce_along_axes(val, (1,), "mean")
    assert result.dtype == dtype
    assert bool(jnp.all(jnp.isfinite(result.astype(jnp.float32))))
    assert bool(jnp.all(result.astype(jnp.float32) == 1.0))


def test_mean_float32_is_bit_identical_to_plain_mean():
    key = jax.random.PRNGKey(0)
    val = jax.random.normal(key, (128, 700), dtype=jnp.float32)
    result = _reduce_along_axes(val, (1,), "mean")
    expected = jnp.mean(val, axis=1)
    assert result.dtype == jnp.float32
    assert bool(jnp.all(result == expected))


def test_mean_of_zeros_float8_e4m3fn_is_zero_not_nan():
    dtype = jnp.float8_e4m3fn
    val = jnp.zeros((128, 700), dtype=dtype)
    result = _reduce_along_axes(val, (1,), "mean")
    assert result.dtype == dtype
    assert bool(jnp.all(jnp.isfinite(result.astype(jnp.float32))))
    assert bool(jnp.all(result.astype(jnp.float32) == 0.0))
