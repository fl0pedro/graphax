"""Tests for the elemental composition + dispatch layer
(``graphax.sparse.elemental.dispatch``).

The dispatch is wired as the first fast path in ``matmul`` / ``elementwise``;
these tests assert (a) it returns ``None`` for pure-dense / plain-diagonal
contractions (so EXACT-AD stays on the existing path), (b) it routes a single
structured pair through the nnz-optimal pairwise kernel, (c) it composes a
multi-structured contraction into a correct, canonically-id'd result, and
(d) every routed result equals the plain dense contraction to ~1e-4.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.elemental import dispatch as DSP
from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.ops.elementwise import elementwise
from graphax.sparse.ops.matmul import matmul
from graphax.sparse.tensor import SparseTensor


@pytest.fixture(autouse=True)
def _approx_active():
    """The dispatch is a hard no-op unless an approximation is active (so plain
    exact AD stays byte-identical — see dispatch.set_approx_active). These tests
    exercise the routing itself, i.e. the approximation-active condition, so we
    flag it on for the duration of each test and reset afterwards."""
    DSP.set_approx_active(True)
    try:
        yield
    finally:
        DSP.set_approx_active(False)


def _dense_st(seed, out_sizes, primal_sizes):
    rng = np.random.default_rng(seed)
    shape = tuple(out_sizes) + tuple(primal_sizes)
    arr = jnp.asarray(rng.standard_normal(shape), dtype=jnp.float32)
    out = tuple(DenseIndex(i, s, i) for i, s in enumerate(out_sizes))
    n = len(out_sizes)
    primal = tuple(
        DenseIndex(n + i, s, n + i) for i, s in enumerate(primal_sizes)
    )
    return SparseTensor(out, primal, arr, check_consistency=False)


def _rect_B(seed, N, Brow, Bcol):
    """A rectangular block-diagonal (out block Brow, primal block Bcol)."""
    rng = np.random.default_rng(seed)
    val = jnp.asarray(rng.standard_normal((N, Brow, Bcol)), dtype=jnp.float32)
    out = DiagonalIndex(0, N, axis=0, other_id=1, block_size=Brow, block_axis=1)
    primal = DiagonalIndex(1, N, axis=0, other_id=0, block_size=Bcol, block_axis=2)
    return SparseTensor((out,), (primal,), val, check_consistency=False)


def _close(out_st, ref):
    return np.allclose(np.asarray(out_st.dense()), np.asarray(ref), atol=1e-4)


# --------------------------------------------------------------------------- #
# (a) pure-dense / plain-diagonal → dispatch returns None (existing path owns)
# --------------------------------------------------------------------------- #
def test_pure_dense_returns_none():
    lhs = _dense_st(0, (4,), (5,))
    rhs = _dense_st(1, (5,), (3,))
    assert DSP.try_elemental_matmul(lhs, rhs) is None


def test_plain_diagonal_returns_none():
    # A plain (block_size 1) diagonal contraction is handled byte-identically by
    # the existing path; dispatch must NOT intercept it.
    lhs = _dense_st(0, (6,), (4,))
    d_out = DiagonalIndex(0, 4, axis=0, other_id=1)
    d_pri = DiagonalIndex(1, 4, axis=0, other_id=0)
    rhs = SparseTensor((d_out,), (d_pri,),
                       jnp.asarray(np.random.default_rng(2).standard_normal(4),
                                   dtype=jnp.float32),
                       check_consistency=False)
    assert DSP.try_elemental_matmul(lhs, rhs) is None


# --------------------------------------------------------------------------- #
# (b) single structured pair → nnz-optimal kernel, correct vs dense
# --------------------------------------------------------------------------- #
def test_single_rect_D_B_routes_to_kernel():
    DSP.reset_stats()
    lhs = _dense_st(0, (6,), (4,))           # dense (6,4)
    rhs = _rect_B(1, 2, 2, 3)                # block-diag 4 -> 6
    out = matmul(lhs, rhs)
    ref = np.asarray(lhs.dense()) @ np.asarray(rhs.dense())
    assert out.shape == (6, 6)
    assert _close(out, ref)
    assert DSP.DISPATCH_STATS["matmul_kernel_D_B"] >= 1


def test_single_B_B_routes_to_kernel():
    DSP.reset_stats()
    lhs = _rect_B(0, 2, 3, 2)   # out 6, contract 4
    rhs = _rect_B(1, 2, 2, 3)   # contract 4, primal 6
    out = matmul(lhs, rhs)
    ref = np.asarray(lhs.dense()) @ np.asarray(rhs.dense())
    assert _close(out, ref)
    assert DSP.DISPATCH_STATS["matmul_kernel_B_B"] >= 1


# --------------------------------------------------------------------------- #
# (c) multiple structured pairs → composed dense, correct + canonical ids
# --------------------------------------------------------------------------- #
def test_multi_structured_pair_composes_correctly():
    DSP.reset_stats()
    # lhs: two diagonal pairs on the contracted side + a dense out dim.
    rng = np.random.default_rng(3)
    lo = [DiagonalIndex(0, 4, axis=0, other_id=2),
          DiagonalIndex(1, 2, axis=1, other_id=3)]
    lp = [DiagonalIndex(2, 4, axis=0, other_id=0),
          DiagonalIndex(3, 2, axis=1, other_id=1)]
    lval = jnp.asarray(rng.standard_normal((4, 2)), dtype=jnp.float32)
    lhs = SparseTensor(tuple(lo), tuple(lp), lval, check_consistency=False)
    # rhs: two diagonal pairs (matching contracted sizes) + dense primal.
    ro = [DiagonalIndex(0, 4, axis=0, other_id=2),
          DiagonalIndex(1, 2, axis=1, other_id=3)]
    rp = [DiagonalIndex(2, 4, axis=0, other_id=0),
          DiagonalIndex(3, 2, axis=1, other_id=1)]
    rval = jnp.asarray(rng.standard_normal((4, 2)), dtype=jnp.float32)
    rhs = SparseTensor(tuple(ro), tuple(rp), rval, check_consistency=False)

    out = matmul(lhs, rhs)
    # reference: contract lhs.primal (4,2) against rhs.out (4,2).
    Ld = np.asarray(lhs.dense())
    Rd = np.asarray(rhs.dense())
    ref = np.tensordot(Ld, Rd, axes=([2, 3], [0, 1]))
    assert out.shape == ref.shape
    assert _close(out, ref)
    # canonical output ids: out 0.., primal n_out..
    n_out = len(out.out_dims)
    assert [d.id for d in out.out_dims] == list(range(n_out))
    assert [d.id for d in out.primal_dims] == list(
        range(n_out, n_out + len(out.primal_dims))
    )
    assert DSP.DISPATCH_STATS["matmul_composed_dense"] >= 1


# --------------------------------------------------------------------------- #
# (d) elementwise routing
# --------------------------------------------------------------------------- #
def test_elementwise_D_plus_B_routes_to_kernel():
    DSP.reset_stats()
    D = _dense_st(0, (4,), (6,))
    B = _rect_B(1, 2, 2, 3)
    out = elementwise(D, B, jnp.add)
    ref = np.asarray(D.dense()) + np.asarray(B.dense())
    assert _close(out, ref)
    assert DSP.DISPATCH_STATS["elementwise_kernel_D_B"] >= 1


def test_elementwise_pure_dense_returns_none():
    D1 = _dense_st(0, (4,), (6,))
    D2 = _dense_st(1, (4,), (6,))
    assert DSP.try_elemental_elementwise(D1, D2, jnp.add) is None


def test_count_path_returns_triple():
    lhs = _dense_st(0, (6,), (4,))
    rhs = _rect_B(1, 2, 2, 3)
    res = DSP.try_elemental_matmul(lhs, rhs, count=True)
    assert res is not None
    out, counts = res
    assert len(counts) == 3
    assert all(isinstance(c, int) for c in counts)
