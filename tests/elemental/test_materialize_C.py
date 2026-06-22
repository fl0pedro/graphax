"""Dense-oracle tests for the ``materialize_C`` elemental kernel.

The correctness gate (per the design): ``materialize_compressed(C)`` MUST equal
``C.dense()`` cell for cell (~1e-5), and the result must be CLOSED in ``{D, B}``
(no ``CompressedIndex`` dim survives).  Coverage:

  * BandedIndex:
      - W>1 band (and rectangular meta) -> Dense (D).
      - W=1 identity band -> Block-diagonal (B); ``to_meta_blocks`` checked.
      - n_meta batching; non-zero fill.
  * ToeplitzIndex (conv incidence) -> Dense (D), for ``d out/d lhs`` (val=TAP)
    and ``d out/d rhs`` (val=IN) shapes.
  * SetIndex (union / intersection) -> Block-diagonal (B); always reduces.
  * COMPOSITION: a materialized operand then contracts correctly with a Dense
    partner (vs a dense matmul oracle), the whole point of the boundary adapter.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.elemental import expansion_kind, materialize_compressed
from graphax.sparse.elemental.contract_D_B import contract_dense_block_diagonal
from graphax.sparse.indexes import (
    BandedIndex,
    DenseIndex,
    SetIndex,
    ToeplitzIndex,
    TOEPLITZ_IN,
    TOEPLITZ_OUT,
    TOEPLITZ_TAP,
)
from graphax.sparse.tensor import SparseTensor

ATOL = 1e-5


def _rand(shape, seed):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))


# --------------------------------------------------------------------------- #
# Builders for the three compressed pair types
# --------------------------------------------------------------------------- #
def _banded(val, *, W, B_row, B_col, n_secondary, offset=(), n_meta=1, fill=None):
    """A pure BandedIndex pair carrying canonical val
    ``(n_meta*M_primary, W, B_row, B_col)``.  ``size`` is the meta count."""
    M_primary = val.shape[0] // n_meta
    out = BandedIndex(
        id=0, size=n_meta * M_primary, axis=0, other_id=1,
        block_size=B_row, block_axis=1,
        band_width=W, offset=offset, primary=True,
        n_secondary=n_secondary, n_meta=n_meta,
    )
    primal = BandedIndex(
        id=1, size=n_meta * n_secondary, axis=0, other_id=0,
        block_size=B_col, block_axis=1,
        band_width=W, offset=offset, primary=False,
        n_secondary=n_secondary, n_meta=n_meta,
    )
    return SparseTensor((out,), (primal,), val, fill_value=fill)


def _setpair(lhs_buf, rhs_buf, *, semantic, fill=None):
    """A pure SetIndex pair: concatenated 1-D val + per-side block-buffer shapes.
    Buffers are ``(n_meta*M, n, Bh, Bw)``."""
    combined = jnp.concatenate([lhs_buf.reshape(-1), rhs_buf.reshape(-1)])
    M = lhs_buf.shape[0]
    LCM_h = lhs_buf.shape[1] * lhs_buf.shape[2]
    LCM_w = lhs_buf.shape[1] * lhs_buf.shape[3]
    out = SetIndex(
        id=0, size=M, axis=0, other_id=1, block_size=LCM_h, block_axis=1,
        semantic=semantic, lhs_shape=lhs_buf.shape, rhs_shape=rhs_buf.shape,
        include_remainder=True, n_meta=1,
    )
    primal = SetIndex(
        id=1, size=M, axis=0, other_id=0, block_size=LCM_w, block_axis=2,
        semantic=semantic, lhs_shape=lhs_buf.shape, rhs_shape=rhs_buf.shape,
        include_remainder=True, n_meta=1,
    )
    return SparseTensor((out,), (primal,), combined, fill_value=fill)


def _toeplitz(val, *, P, X, K, val_role, partner_role, stride=1, win_dilation=1,
              base_dilation=1, pad_lo=0):
    """A ToeplitzIndex pair occupying the two roles NOT stored in ``val``.  The
    primary side owns the contracted ``val`` axis (``axis=0``)."""
    out_role = ({TOEPLITZ_OUT, TOEPLITZ_IN, TOEPLITZ_TAP} - {val_role, partner_role}).pop()
    sizes = {TOEPLITZ_OUT: P, TOEPLITZ_IN: X, TOEPLITZ_TAP: K}
    common = dict(out_size=P, in_size=X, kernel_size=K, stride=stride,
                  win_dilation=win_dilation, base_dilation=base_dilation, pad_lo=pad_lo)
    out = ToeplitzIndex(id=0, size=sizes[out_role], axis=0, other_id=1,
                        role=out_role, primary=True, **common)
    primal = ToeplitzIndex(id=1, size=sizes[partner_role], axis=None, other_id=0,
                           role=partner_role, primary=False, **common)
    return SparseTensor((out,), (primal,), val)


# --------------------------------------------------------------------------- #
# Core gate: materialize == dense, and closure {D,B}
# --------------------------------------------------------------------------- #
def _assert_materialize(C, expect_kind):
    """materialize(C) == C.dense() (~1e-5) and the result is closed {D,B}."""
    mat = materialize_compressed(C)
    got = np.asarray(mat.dense())
    oracle = np.asarray(C.dense())
    assert got.shape == oracle.shape, (got.shape, oracle.shape)
    np.testing.assert_allclose(got, oracle, atol=ATOL)
    # Closure: no compressed dim survives.
    for d in mat.dims:
        assert not d.is_compressed, mat.dims
    # Kind: B -> every dim sparse-Diagonal; D -> every dim Dense.
    if expect_kind == "B":
        assert all(d.is_sparse and not d.is_compressed for d in mat.dims), mat.dims
    elif expect_kind == "D":
        assert all(not d.is_sparse for d in mat.dims), mat.dims
    assert expansion_kind(C) == expect_kind
    return mat


# --------------------------------------------------------------------------- #
# BandedIndex  ->  D  (W > 1) and rectangular / offset
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("M,W,B_row,B_col", [
    (5, 3, 4, 4),   # square centered band
    (4, 3, 2, 3),   # rectangular blocks
    (6, 3, 2, 2),
])
def test_banded_wide_to_dense(M, W, B_row, B_col):
    val = _rand((M, W, B_row, B_col), seed=hash((M, W, B_row, B_col)) % 2**31)
    C = _banded(val, W=W, B_row=B_row, B_col=B_col, n_secondary=M)
    _assert_materialize(C, "D")


def test_banded_rectangular_staircase_to_dense():
    M_r, W, Br, Bc = 11, 2, 5, 5
    off = (0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4)
    val = _rand((M_r, W, Br, Bc), seed=1)
    C = _banded(val, W=W, B_row=Br, B_col=Bc, n_secondary=5, offset=off)
    _assert_materialize(C, "D")


def test_banded_nonzero_fill_to_dense():
    M, W, B = 4, 3, 2
    val = _rand((M, W, B, B), seed=3)
    C = _banded(val, W=W, B_row=B, B_col=B, n_secondary=M, fill=jnp.array(-9.0))
    _assert_materialize(C, "D")


def test_banded_n_meta_to_dense():
    M_per, W, B = 3, 3, 2
    val = _rand((2 * M_per, W, B, B), seed=5)
    C = _banded(val, W=W, B_row=B, B_col=B, n_secondary=M_per, n_meta=2)
    _assert_materialize(C, "D")


# --------------------------------------------------------------------------- #
# BandedIndex  ->  B  (W == 1 identity band reduces to block-diagonal)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("M,B_row,B_col", [
    (3, 2, 2),
    (4, 3, 2),   # rectangular block
    (5, 2, 4),
])
def test_banded_identity_to_block_diagonal(M, B_row, B_col):
    val = _rand((M, 1, B_row, B_col), seed=hash((M, B_row, B_col)) % 2**31)
    C = _banded(val, W=1, B_row=B_row, B_col=B_col, n_secondary=M)
    mat = _assert_materialize(C, "B")
    # B form: a DiagonalIndex pair with meta count M.
    assert mat.out_dims[0].size == M
    assert (mat.out_dims[0].block_size or 1) == B_row
    assert (mat.primal_dims[0].block_size or 1) == B_col


# --------------------------------------------------------------------------- #
# ToeplitzIndex  ->  D  (windowed conv incidence)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("P,X,K,stride,pad_lo", [
    (4, 6, 3, 1, 0),
    (4, 4, 3, 1, 1),   # padded -> square P==X
    (3, 8, 3, 2, 0),   # strided
])
def test_toeplitz_dout_dlhs_to_dense(P, X, K, stride, pad_lo):
    # d out/d lhs: pair = OUT x IN, val carries the TAP role (kernel weights).
    val = _rand((K,), seed=hash((P, X, K, stride, pad_lo)) % 2**31)
    C = _toeplitz(val, P=P, X=X, K=K, val_role=TOEPLITZ_TAP, partner_role=TOEPLITZ_IN,
                  stride=stride, pad_lo=pad_lo)
    mat = _assert_materialize(C, "mixed")
    assert mat.shape == (P, X)


@pytest.mark.parametrize("P,X,K", [(4, 6, 3), (5, 5, 3)])
def test_toeplitz_dout_drhs_to_dense(P, X, K):
    # d out/d rhs: pair = OUT x TAP, val carries the IN role (activations).
    val = _rand((X,), seed=hash((P, X, K, "rhs")) % 2**31)
    C = _toeplitz(val, P=P, X=X, K=K, val_role=TOEPLITZ_IN, partner_role=TOEPLITZ_TAP)
    mat = _assert_materialize(C, "mixed")
    assert mat.shape == (P, K)


# --------------------------------------------------------------------------- #
# SetIndex  ->  B  (always reduces to meta-block-diagonal)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("semantic", ["union", "intersection"])
def test_set_to_block_diagonal(semantic):
    lhs = _rand((1, 3, 2, 2), seed=8)
    rhs = _rand((1, 2, 3, 3), seed=9)
    C = _setpair(lhs, rhs, semantic=semantic)
    mat = _assert_materialize(C, "B")
    assert mat.out_dims[0].size == 1  # M meta count


@pytest.mark.parametrize("semantic", ["union", "intersection"])
def test_set_multi_meta_to_block_diagonal(semantic):
    # M=2 meta blocks, each a per-meta block-diagonal LCM grid.
    lhs = _rand((2, 2, 2, 2), seed=10)
    rhs = _rand((2, 2, 2, 2), seed=11)
    C = _setpair(lhs, rhs, semantic=semantic)
    mat = _assert_materialize(C, "B")
    assert mat.out_dims[0].size == 2


# --------------------------------------------------------------------------- #
# expansion_kind on a no-compressed tensor is a no-op
# --------------------------------------------------------------------------- #
def test_materialize_noop_on_plain_dense():
    arr = _rand((3, 4), seed=12)
    st = SparseTensor((DenseIndex(0, 3, 0),), (DenseIndex(1, 4, 1),), arr)
    assert expansion_kind(st) == "none"
    assert materialize_compressed(st) is st


# --------------------------------------------------------------------------- #
# COMPOSITION: a materialized operand contracts with a Dense partner correctly
# --------------------------------------------------------------------------- #
def _dense_operand(out_size, primal_size, val):
    return SparseTensor(
        (DenseIndex(0, out_size, 0),), (DenseIndex(1, primal_size, 1),), val,
    )


def test_compose_banded_B_contract_with_dense():
    # W=1 banded -> B; then D @ B vs the dense-matmul oracle.
    M, B_c, B_f = 3, 2, 3
    band_val = _rand((M, 1, B_c, B_f), seed=20)
    C = _banded(band_val, W=1, B_row=B_c, B_col=B_f, n_secondary=M)
    rhs_B = materialize_compressed(C)  # B: out-side B_c, primal-side B_f
    assert expansion_kind(C) == "B"

    P, K = 4, M * B_c
    lhs = _dense_operand(P, K, _rand((P, K), seed=21))
    res = contract_dense_block_diagonal(lhs, rhs_B)

    oracle = np.asarray(lhs.dense()) @ np.asarray(C.dense())
    np.testing.assert_allclose(np.asarray(res.dense()), oracle, atol=ATOL)


def test_compose_set_B_contract_with_dense():
    # SetIndex -> B; then D @ B vs the dense-matmul oracle.
    lhs_buf = _rand((1, 2, 2, 3), seed=30)
    rhs_buf = _rand((1, 2, 2, 3), seed=31)
    C = _setpair(lhs_buf, rhs_buf, semantic="union")
    rhs_B = materialize_compressed(C)
    assert expansion_kind(C) == "B"

    R, Ccols = C.shape
    P = 5
    lhs = _dense_operand(P, R, _rand((P, R), seed=32))
    res = contract_dense_block_diagonal(lhs, rhs_B)

    oracle = np.asarray(lhs.dense()) @ np.asarray(C.dense())
    np.testing.assert_allclose(np.asarray(res.dense()), oracle, atol=ATOL)


def test_compose_toeplitz_D_matmul_with_dense():
    # Toeplitz -> D; compose via a plain dense matmul vs the dense oracle.
    P, X, K = 4, 6, 3
    val = _rand((K,), seed=40)
    C = _toeplitz(val, P=P, X=X, K=K, val_role=TOEPLITZ_TAP, partner_role=TOEPLITZ_IN)
    mat = materialize_compressed(C)
    assert expansion_kind(C) == "mixed"

    # (B, P) dense @ (P, X) materialized -> (B, X), vs dense oracle.
    Bn = 7
    A = _rand((Bn, P), seed=41)
    got = np.asarray(A) @ np.asarray(mat.dense())
    oracle = np.asarray(A) @ np.asarray(C.dense())
    np.testing.assert_allclose(got, oracle, atol=ATOL)
