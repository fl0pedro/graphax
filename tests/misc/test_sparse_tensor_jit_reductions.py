"""``SparseTensor.{all,any,max,min,sum,prod}`` used to mix Python ``and``/``or``
and the ``max``/``min`` builtins with JAX arrays. Inside ``jit`` that path
raises ``ConcretizationTypeError`` (the truth-value of a tracer can't be
materialized at trace time); outside jit it fires ``__bool__`` on a 0-d
traced array. After the fix, all six reductions use ``jnp.logical_and`` /
``jnp.logical_or`` / ``jnp.maximum`` / ``jnp.minimum`` and stay traceable.
"""
import jax
import jax.numpy as jnp

from graphax.sparse.indexes import DenseIndex
from graphax.sparse.tensor import SparseTensor


def _make_st():
    val = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    return SparseTensor(
        (DenseIndex(0, 2, axis=0),),
        (DenseIndex(1, 3, axis=1),),
        val,
    )


def test_jit_all():
    fn = jax.jit(lambda st: st.all())
    fn(_make_st())


def test_jit_any():
    fn = jax.jit(lambda st: st.any())
    fn(_make_st())


def test_jit_max():
    fn = jax.jit(lambda st: st.max())
    fn(_make_st())


def test_jit_min():
    fn = jax.jit(lambda st: st.min())
    fn(_make_st())


def test_jit_sum():
    fn = jax.jit(lambda st: st.sum())
    fn(_make_st())


def test_jit_prod():
    fn = jax.jit(lambda st: st.prod())
    fn(_make_st())


def test_jit_sum_with_val_none():
    """Used to crash because ``val.size`` was evaluated before the
    ``val is None`` ternary."""
    st = SparseTensor(
        (DenseIndex(0, 4, axis=None),),
        (DenseIndex(1, 4, axis=None),),
        val=None,
        fill_value=jnp.array(2.0),
    )
    out = jax.jit(lambda s: s.sum())(st)
    assert float(out) == 2.0 * 16  # fill_value * size


def test_jit_max_with_val_none():
    """``max()`` used to bottom out at literal ``1`` instead of
    ``fill_value * scalar_mult``."""
    st = SparseTensor(
        (DenseIndex(0, 4, axis=None),),
        (DenseIndex(1, 4, axis=None),),
        val=None,
        fill_value=jnp.array(7.0),
    )
    out = jax.jit(lambda s: s.max())(st)
    assert float(out) == 7.0
