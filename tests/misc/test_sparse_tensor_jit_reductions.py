"""``SparseTensor.{all,any,max,min,sum,prod}`` used to mix Python ``and``/``or``
and the ``max``/``min`` builtins with JAX arrays. Inside ``jit`` that path
raises ``ConcretizationTypeError`` (the truth-value of a tracer can't be
materialized at trace time); outside jit it fires ``__bool__`` on a 0-d
traced array. After the fix, all six reductions use ``jnp.logical_and`` /
``jnp.logical_or`` / ``jnp.maximum`` / ``jnp.minimum`` and stay traceable.
"""
import jax
import jax.numpy as jnp

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
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
    """``val is None`` means the structure is all-ones (same reading as
    ``dense()``); ``fill_value`` only paints OFF-structure cells. A FULLY-DENSE
    tensor has none, so every cell is ``1`` and ``sum == size`` — ``fill_value``
    is irrelevant. Reductions must equal ``reduction(dense())``."""
    st = SparseTensor(
        (DenseIndex(0, 4, axis=None),),
        (DenseIndex(1, 4, axis=None),),
        val=None,
        fill_value=jnp.array(2.0),  # off-structure fill — no such cells here
    )
    out = jax.jit(lambda s: s.sum())(st)
    assert float(out) == 16.0  # ones * size (NOT fill * size)
    assert float(out) == float(st.dense().sum())


def test_jit_max_with_val_none():
    """A fully-dense ``val is None`` tensor is all-ones, so ``max == 1`` (×
    scalar_mult), NOT the fill — matching ``dense().max()``."""
    st = SparseTensor(
        (DenseIndex(0, 4, axis=None),),
        (DenseIndex(1, 4, axis=None),),
        val=None,
        fill_value=jnp.array(7.0),
    )
    out = jax.jit(lambda s: s.max())(st)
    assert float(out) == 1.0
    assert float(out) == float(st.dense().max())


def test_jit_reductions_sparse_val_none_match_dense():
    """For a SPARSE ``val is None`` tensor (a block-diagonal identity), the
    structure cells are ones and the off-block-diagonal cells are ``fill`` —
    every reduction must equal ``reduction(dense())`` (structure-aware, NOT the
    old 'every cell is fill' reading). Here: a 2×2 block-diagonal of 3×3 blocks,
    so 18 structure-ones and 18 off-block fill=5 cells, scalar_mult=2."""
    st = SparseTensor(
        (DiagonalIndex(0, 2, axis=None, other_id=1, block_size=3),),
        (DiagonalIndex(1, 2, axis=None, other_id=0, block_size=3),),
        val=None,
        scalar_mult=jnp.array(2.0),
        fill_value=jnp.array(5.0),
    )
    dense = st.dense()
    for op in ("sum", "max", "min", "all", "any"):
        out = jax.jit(lambda s, op=op: getattr(s, op)())(st)
        assert float(out) == float(getattr(dense, op)()), op
    # sum = 18 ones*2  +  18 fill(5)*2  = 36 + 180 = 216
    assert float(jax.jit(lambda s: s.sum())(st)) == 216.0
