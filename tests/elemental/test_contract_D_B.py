"""Dense-oracle tests for the Dense<->BlockDiagonal contraction kernel.

Every test asserts the kernel ``contract_dense_block_diagonal`` equals the DENSE
ORACLE: materialize both operands via ``.dense()`` and run a plain
``jnp.matmul``. Covers ``D @ B`` and ``B @ D``, square and RECTANGULAR blocks,
batch/leftover dense axes, ``val=None`` operands, scalar_mult folding, and
search-shaped (attention / mlp) instances with arbitrary block combinations.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor
from graphax.sparse.elemental.contract_D_B import contract_dense_block_diagonal

jax.config.update("jax_enable_x64", False)

ATOL = 1e-5


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _rand(shape, key):
    return jax.random.normal(key, shape, dtype=jnp.float32)


def _diag(out_id, primal_id, N, B_out, B_primal, val, axis=0):
    """A 2-D meta-block-diagonal SparseTensor (out-side block B_out, primal-side
    block B_primal). ``val`` has shape ``(N, B_out, B_primal)``."""
    out = DiagonalIndex(
        out_id, N, axis=axis, other_id=primal_id,
        block_size=B_out if B_out > 1 else None,
        block_axis=axis + 1 if B_out > 1 else None,
    )
    primal = DiagonalIndex(
        primal_id, N, axis=axis, other_id=out_id,
        block_size=B_primal if B_primal > 1 else None,
        block_axis=axis + 2 if (B_out > 1 and B_primal > 1) else (axis + 1 if B_primal > 1 else None),
    )
    # Keep val layout canonical (meta, B_out, B_primal); squeeze size-1 block axes.
    v = val
    if B_out == 1:
        v = v[:, 0, :]
        primal = DiagonalIndex(
            primal_id, N, axis=axis, other_id=out_id,
            block_size=B_primal if B_primal > 1 else None,
            block_axis=axis + 1 if B_primal > 1 else None,
        )
    if B_primal == 1 and B_out > 1:
        v = val[:, :, 0]
    if B_out == 1 and B_primal == 1:
        v = val[:, 0, 0]
    return SparseTensor((out,), (primal,), v)


def _dense(out_ids, out_sizes, primal_ids, primal_sizes, val):
    out = [DenseIndex(i, s, ax) for ax, (i, s) in enumerate(zip(out_ids, out_sizes))]
    off = len(out_sizes)
    primal = [
        DenseIndex(i, s, off + ax)
        for ax, (i, s) in enumerate(zip(primal_ids, primal_sizes))
    ]
    return SparseTensor(tuple(out), tuple(primal), val)


def _dense_oracle(lhs, rhs):
    """Plain dense vertex-elim contraction: materialize both operands, then
    ``tensordot`` over the single contracted pair (lhs's last primal axis — its
    logical-K axis — against rhs's first out axis). ``lhs.dense()`` axis order is
    ``(out..., primal...)``, so the contracted axis is its LAST; ``rhs.dense()``
    is ``(out..., primal...)`` so the contracted axis is its FIRST.
    """
    ld = lhs.dense()
    rd = rhs.dense()
    n_rhs_out = len(rhs.out_dims)
    # The kernels handle a single contracted pair; the contracted axes are
    # lhs's trailing primal axis and rhs's leading out axis.
    return jnp.tensordot(ld, rd, axes=([ld.ndim - 1], [0])) if n_rhs_out == 1 else (
        jnp.tensordot(ld, rd, axes=(list(range(ld.ndim - n_rhs_out, ld.ndim)),
                                    list(range(n_rhs_out))))
    )


def _assert_oracle(lhs, rhs):
    res = contract_dense_block_diagonal(lhs, rhs)
    oracle = _dense_oracle(lhs, rhs)
    assert res.dense().shape == oracle.shape, (res.dense().shape, oracle.shape)
    np.testing.assert_allclose(np.asarray(res.dense()), np.asarray(oracle), atol=ATOL)
    return res


# --------------------------------------------------------------------------- #
# D @ B  (block-diagonal on the right / contracted out-side)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "P,N,B_c,B_f",
    [
        (5, 3, 2, 4),   # rectangular B_c < B_f
        (4, 3, 4, 2),   # rectangular B_c > B_f
        (3, 4, 2, 2),   # square
        (6, 2, 3, 3),   # square, bigger
        (3, 5, 1, 3),   # B_c == 1 (contracted side has no block)
        (3, 4, 2, 1),   # B_f == 1 (free side has no block)
        (1, 3, 2, 4),   # P == 1 (degenerate dense row)
    ],
)
def test_D_at_B(P, N, B_c, B_f):
    k = jax.random.PRNGKey(hash((P, N, B_c, B_f)) % (2**31))
    k1, k2 = jax.random.split(k)
    K = N * B_c
    lhs = _dense([0], [P], [1], [K], _rand((P, K), k1))
    rhs = _diag(0, 1, N, B_c, B_f, _rand((N, B_c, B_f), k2))
    _assert_oracle(lhs, rhs)


def test_nonzero_fill_fallback():
    # A non-zero fill breaks the zero-fill fast path, so the kernel falls back to
    # the dense oracle. Regression guard: that fallback used to pass the primal
    # ndim as _arr2st's `dtype` argument -> "Cannot interpret '1' as a data type".
    k1, k2 = jax.random.split(jax.random.PRNGKey(7))
    P, N, B_c, B_f = 4, 3, 2, 2
    K = N * B_c
    lhs = SparseTensor(
        (DenseIndex(0, P, 0),),
        (DenseIndex(1, K, 1),),
        _rand((P, K), k1),
        fill_value=jnp.asarray(1.5, dtype=jnp.float32),
    )
    rhs = _diag(0, 1, N, B_c, B_f, _rand((N, B_c, B_f), k2))
    _assert_oracle(lhs, rhs)


# --------------------------------------------------------------------------- #
# B @ D  (block-diagonal on the left / contracted primal-side)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "N,B_f,B_c,Q",
    [
        (3, 4, 2, 5),   # rectangular B_f > B_c
        (3, 2, 4, 4),   # rectangular B_f < B_c
        (4, 2, 2, 3),   # square
        (2, 3, 3, 6),   # square bigger
        (5, 3, 1, 3),   # B_c == 1
        (4, 1, 2, 3),   # B_f == 1
        (3, 2, 4, 1),   # Q == 1
    ],
)
def test_B_at_D(N, B_f, B_c, Q):
    k = jax.random.PRNGKey(hash((N, B_f, B_c, Q)) % (2**31))
    k1, k2 = jax.random.split(k)
    K = N * B_c
    lhs = _diag(0, 1, N, B_f, B_c, _rand((N, B_f, B_c), k1))
    rhs = _dense([0], [K], [1], [Q], _rand((K, Q), k2))
    _assert_oracle(lhs, rhs)


# --------------------------------------------------------------------------- #
# Batch / leftover dense axes carried through
# --------------------------------------------------------------------------- #
def test_D_at_B_multi_out():
    k = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(k)
    A, P, N, B_c, B_f = 2, 3, 2, 2, 3
    K = N * B_c
    lhs = _dense([0, 1], [A, P], [2], [K], _rand((A, P, K), k1))
    rhs = _diag(0, 1, N, B_c, B_f, _rand((N, B_c, B_f), k2))
    _assert_oracle(lhs, rhs)


def test_B_at_D_multi_primal():
    k = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(k)
    N, B_f, B_c, Q1, Q2 = 2, 3, 2, 2, 4
    K = N * B_c
    lhs = _diag(0, 1, N, B_f, B_c, _rand((N, B_f, B_c), k1))
    rhs = _dense([0], [K], [1, 2], [Q1, Q2], _rand((K, Q1, Q2), k2))
    _assert_oracle(lhs, rhs)


# --------------------------------------------------------------------------- #
# scalar_mult folding (both operands)
# --------------------------------------------------------------------------- #
def test_scalar_mult_fold():
    k = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(k)
    P, N, B_c, B_f = 4, 3, 2, 3
    K = N * B_c
    lhs = _dense([0], [P], [1], [K], _rand((P, K), k1)).copy(scalar_mult=jnp.array(2.5))
    rhs = _diag(0, 1, N, B_c, B_f, _rand((N, B_c, B_f), k2)).copy(
        scalar_mult=jnp.array(-1.5)
    )
    _assert_oracle(lhs, rhs)


# --------------------------------------------------------------------------- #
# val=None operands
# --------------------------------------------------------------------------- #
def test_val_none_dense_lhs():
    # val=None dense lhs ≡ all-ones; dense() oracle handles it.
    k = jax.random.PRNGKey(4)
    P, N, B_c, B_f = 3, 2, 2, 3
    K = N * B_c
    lhs = SparseTensor((DenseIndex(0, P, None),), (DenseIndex(1, K, None),), None)
    rhs = _diag(0, 1, N, B_c, B_f, _rand((N, B_c, B_f), k))
    _assert_oracle(lhs, rhs)


def test_val_none_block_diagonal_rhs():
    # val=None block-diagonal ≡ all-ones blocks. The generic dense() oracle has
    # a pre-existing crash densifying a val=None DiagonalIndex with explicit
    # block axes, so compare the kernel's val=None result against the explicit
    # all-ones-block equivalent oracle instead.
    k = jax.random.PRNGKey(5)
    P, N, B_c, B_f = 3, 2, 2, 3
    K = N * B_c
    lhs = _dense([0], [P], [1], [K], _rand((P, K), k))
    rhs_none = _diag(0, 1, N, B_c, B_f, jnp.ones((N, B_c, B_f)))
    rhs_none = SparseTensor(rhs_none.out_dims, rhs_none.primal_dims, None)
    rhs_ones = _diag(0, 1, N, B_c, B_f, jnp.ones((N, B_c, B_f)))
    res = contract_dense_block_diagonal(lhs, rhs_none)
    oracle = jnp.matmul(lhs.dense(), rhs_ones.dense())
    np.testing.assert_allclose(np.asarray(res.dense()), np.asarray(oracle), atol=ATOL)


# --------------------------------------------------------------------------- #
# Result is fully Dense (closure: D@B -> D, B@D -> D)
# --------------------------------------------------------------------------- #
def test_result_is_dense():
    k = jax.random.PRNGKey(6)
    k1, k2 = jax.random.split(k)
    P, N, B_c, B_f = 4, 3, 2, 3
    K = N * B_c
    lhs = _dense([0], [P], [1], [K], _rand((P, K), k1))
    rhs = _diag(0, 1, N, B_c, B_f, _rand((N, B_c, B_f), k2))
    res = contract_dense_block_diagonal(lhs, rhs)
    assert all(not d.is_sparse for d in res.dims), res.dims
    # Free dim of the diagonal survives as a single dense axis of size N*B_f.
    assert res.primal_shape[-1] == N * B_f


# --------------------------------------------------------------------------- #
# Mis-dispatch guards
# --------------------------------------------------------------------------- #
def test_rejects_D_at_D():
    k = jax.random.PRNGKey(7)
    lhs = _dense([0], [3], [1], [4], _rand((3, 4), k))
    rhs = _dense([0], [4], [1], [5], _rand((4, 5), k))
    with pytest.raises(ValueError):
        contract_dense_block_diagonal(lhs, rhs)


def test_rejects_B_at_B():
    k = jax.random.PRNGKey(8)
    k1, k2 = jax.random.split(k)
    N, B = 3, 2
    lhs = _diag(0, 1, N, B, B, _rand((N, B, B), k1))
    rhs = _diag(0, 1, N, B, B, _rand((N, B, B), k2))
    with pytest.raises(ValueError):
        contract_dense_block_diagonal(lhs, rhs)


# --------------------------------------------------------------------------- #
# Search-shaped instances (attention + mlp), arbitrary block combinations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("S,H,dk", [(4, 2, 4), (8, 2, 4), (8, 2, 8)])
def test_attention_shaped_D_at_B(S, H, dk):
    # Attention-flavoured: a per-head block-diagonal Jacobian factor (H heads as
    # meta blocks, each head a dk x dk square block on the contracted axis)
    # contracted against a dense (S, H*dk) activation slab.
    k = jax.random.PRNGKey(hash((S, H, dk)) % (2**31))
    k1, k2 = jax.random.split(k)
    K = H * dk
    lhs = _dense([0], [S], [1], [K], _rand((S, K), k1))
    rhs = _diag(0, 1, H, dk, dk, _rand((H, dk, dk), k2))
    _assert_oracle(lhs, rhs)


@pytest.mark.parametrize("S,H,dk", [(4, 2, 4), (8, 2, 4), (8, 2, 8)])
def test_attention_shaped_B_at_D(S, H, dk):
    k = jax.random.PRNGKey(hash((S, H, dk, "bd")) % (2**31))
    k1, k2 = jax.random.split(k)
    K = H * dk
    lhs = _diag(0, 1, H, dk, dk, _rand((H, dk, dk), k1))
    rhs = _dense([0], [K], [1], [S], _rand((K, S), k2))
    _assert_oracle(lhs, rhs)


@pytest.mark.parametrize(
    "B_in,N,B_c,B_f",
    list(itertools.product([1, 8], [2, 4], [2, 3], [2, 3])),
)
def test_mlp_shaped_arbitrary_blocks(B_in, N, B_c, B_f):
    # MLP-flavoured: a batch of B_in dense rows contracted against a
    # block-diagonal weight Jacobian with arbitrary (possibly rectangular)
    # block sizes — exercises arbitrary {B_c, B_f} combinations.
    k = jax.random.PRNGKey(hash((B_in, N, B_c, B_f)) % (2**31))
    k1, k2 = jax.random.split(k)
    K = N * B_c
    lhs = _dense([0], [B_in], [1], [K], _rand((B_in, K), k1))
    rhs = _diag(0, 1, N, B_c, B_f, _rand((N, B_c, B_f), k2))
    _assert_oracle(lhs, rhs)


# --------------------------------------------------------------------------- #
# Randomized fuzz over arbitrary component instances
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(12))
def test_fuzz_D_at_B(seed):
    rng = np.random.default_rng(seed)
    P = int(rng.integers(1, 7))
    N = int(rng.integers(1, 5))
    B_c = int(rng.integers(1, 5))
    B_f = int(rng.integers(1, 5))
    k = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(k)
    K = N * B_c
    lhs = _dense([0], [P], [1], [K], _rand((P, K), k1))
    rhs = _diag(0, 1, N, B_c, B_f, _rand((N, B_c, B_f), k2))
    _assert_oracle(lhs, rhs)


@pytest.mark.parametrize("seed", range(12))
def test_fuzz_B_at_D(seed):
    rng = np.random.default_rng(1000 + seed)
    N = int(rng.integers(1, 5))
    B_f = int(rng.integers(1, 5))
    B_c = int(rng.integers(1, 5))
    Q = int(rng.integers(1, 7))
    k = jax.random.PRNGKey(1000 + seed)
    k1, k2 = jax.random.split(k)
    K = N * B_c
    lhs = _diag(0, 1, N, B_f, B_c, _rand((N, B_f, B_c), k1))
    rhs = _dense([0], [K], [1], [Q], _rand((K, Q), k2))
    _assert_oracle(lhs, rhs)
