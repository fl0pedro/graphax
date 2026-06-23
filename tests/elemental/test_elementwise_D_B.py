"""Dense-oracle tests for the Dense<->BlockDiagonal ELEMENTWISE kernel.

Every test asserts the kernel ``elementwise_dense_block_diagonal`` equals the
DENSE ORACLE: materialize both operands via ``.dense()`` and run the plain dense
``op`` (``jnp.add`` / ``jnp.multiply`` / ``jnp.subtract`` / ...).  Covers
``D op B``, ``B op D``, ``B op B`` (matched grid), square and RECTANGULAR blocks,
leftover/batch axes, ``val=None`` operands, scalar_mult folding, closure of the
result component (union -> D, intersection / B-op-B -> B), and search-shaped
(attention / mlp) instances with arbitrary block combinations.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor
from graphax.sparse.elemental.elementwise_D_B import (
    elementwise_dense_block_diagonal,
)

jax.config.update("jax_enable_x64", False)

ATOL = 1e-5


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _rand(shape, key):
    return jax.random.normal(key, shape, dtype=jnp.float32)


def _diag(out_id, primal_id, N, B_out, B_primal, val, axis=0):
    """A 2-D meta-block-diagonal SparseTensor (out-side block B_out, primal-side
    block B_primal). ``val`` has shape ``(N, B_out, B_primal)``; size-1 block
    axes are squeezed to the canonical compact layout."""
    has_o = B_out > 1
    has_i = B_primal > 1
    v = val
    bo_axis = None
    bi_axis = None
    next_ax = axis + 1
    if has_o:
        bo_axis = next_ax
        next_ax += 1
    else:
        v = v[:, 0]
    if has_i:
        bi_axis = next_ax
        next_ax += 1
    else:
        drop = 2 if has_o else 1
        v = jnp.take(v, 0, axis=drop)
    out = DiagonalIndex(
        out_id, N, axis=axis, other_id=primal_id,
        block_size=B_out if has_o else None, block_axis=bo_axis,
    )
    primal = DiagonalIndex(
        primal_id, N, axis=axis, other_id=out_id,
        block_size=B_primal if has_i else None, block_axis=bi_axis,
    )
    return SparseTensor((out,), (primal,), v)


def _dense(R, C, val):
    out = (DenseIndex(0, R, 0),)
    primal = (DenseIndex(1, C, 1),)
    return SparseTensor(out, primal, val)


# --------------------------------------------------------------------------- #
# Oracle
# --------------------------------------------------------------------------- #
def _assert_oracle(lhs, rhs, op, is_intersection=False):
    res = elementwise_dense_block_diagonal(lhs, rhs, op, is_intersection)
    oracle = op(lhs.dense(), rhs.dense())
    assert res.dense().shape == oracle.shape, (res.dense().shape, oracle.shape)
    np.testing.assert_allclose(
        np.asarray(res.dense()), np.asarray(oracle), atol=ATOL
    )
    return res


# --------------------------------------------------------------------------- #
# D op B  /  B op D  : union (add) -> Dense
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "N,B_o,B_i",
    [
        (3, 2, 4),   # rectangular B_o < B_i
        (3, 4, 2),   # rectangular B_o > B_i
        (4, 2, 2),   # square
        (2, 3, 3),   # square bigger
        (5, 1, 3),   # B_o == 1
        (4, 2, 1),   # B_i == 1
        (3, 1, 1),   # pure diagonal
    ],
)
def test_D_add_B(N, B_o, B_i):
    k = jax.random.PRNGKey(hash((N, B_o, B_i)) % (2**31))
    k1, k2 = jax.random.split(k)
    R, C = N * B_o, N * B_i
    lhs = _dense(R, C, _rand((R, C), k1))
    rhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k2))
    _assert_oracle(lhs, rhs, jnp.add)


@pytest.mark.parametrize(
    "N,B_o,B_i",
    [(3, 2, 4), (3, 4, 2), (4, 2, 2), (5, 1, 3), (4, 2, 1)],
)
def test_B_add_D(N, B_o, B_i):
    k = jax.random.PRNGKey(hash((N, B_o, B_i, "bd")) % (2**31))
    k1, k2 = jax.random.split(k)
    R, C = N * B_o, N * B_i
    lhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k1))
    rhs = _dense(R, C, _rand((R, C), k2))
    _assert_oracle(lhs, rhs, jnp.add)


# Non-commutative union-shaped op: subtract. op(x, 0) == x (lhs side); op(0, x)
# = -x (rhs side) — both are union (op(D, 0) leaves D's sign). The oracle pins
# operand order.
@pytest.mark.parametrize("N,B_o,B_i", [(3, 2, 3), (4, 2, 2), (3, 3, 1)])
def test_D_sub_B(N, B_o, B_i):
    k = jax.random.PRNGKey(hash((N, B_o, B_i, "sub")) % (2**31))
    k1, k2 = jax.random.split(k)
    R, C = N * B_o, N * B_i
    lhs = _dense(R, C, _rand((R, C), k1))
    rhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k2))
    _assert_oracle(lhs, rhs, jnp.subtract)


# B as the LHS of a NON-COMMUTATIVE union op: off the block-diagonal B is zero,
# so the result is op(0, D) — subtract -> -D, divide -> 0/D == 0 — NOT D. (A
# regression guard: the kernel previously scattered D's raw grid, giving op(D, 0)
# regardless of operand order, so B - D wrongly yielded +D and B / D yielded D.)
@pytest.mark.parametrize("N,B_o,B_i", [(3, 2, 3), (4, 2, 2), (3, 3, 1), (3, 1, 1)])
def test_B_sub_D(N, B_o, B_i):
    k = jax.random.PRNGKey(hash((N, B_o, B_i, "bsubd")) % (2**31))
    k1, k2 = jax.random.split(k)
    R, C = N * B_o, N * B_i
    lhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k1))
    rhs = _dense(R, C, _rand((R, C), k2))
    _assert_oracle(lhs, rhs, jnp.subtract)


@pytest.mark.parametrize("N,B_o,B_i", [(3, 2, 3), (4, 2, 2), (3, 1, 1)])
def test_B_div_D(N, B_o, B_i):
    # divisor D is dense (no zeros after the +1 offset below), so off-diagonal
    # 0 / D == 0 stays finite.
    k = jax.random.PRNGKey(hash((N, B_o, B_i, "bdivd")) % (2**31))
    k1, k2 = jax.random.split(k)
    R, C = N * B_o, N * B_i
    lhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k1))
    rhs = _dense(R, C, _rand((R, C), k2) + 2.0)  # bias away from 0 divisors
    _assert_oracle(lhs, rhs, jnp.divide)


# --------------------------------------------------------------------------- #
# D op B  /  B op D  : intersection (multiply) -> Block-diagonal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "N,B_o,B_i",
    [(3, 2, 4), (3, 4, 2), (4, 2, 2), (2, 3, 3), (5, 1, 3), (4, 2, 1), (3, 1, 1)],
)
def test_D_mul_B(N, B_o, B_i):
    k = jax.random.PRNGKey(hash((N, B_o, B_i, "mul")) % (2**31))
    k1, k2 = jax.random.split(k)
    R, C = N * B_o, N * B_i
    lhs = _dense(R, C, _rand((R, C), k1))
    rhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k2))
    res = _assert_oracle(lhs, rhs, jnp.multiply, is_intersection=True)
    # Closure: D * B -> B (stays meta-block-diagonal, NOT densified).
    assert all(d.is_sparse for d in res.dims), res.dims


@pytest.mark.parametrize("N,B_o,B_i", [(3, 2, 4), (4, 2, 2), (5, 1, 3)])
def test_B_mul_D(N, B_o, B_i):
    k = jax.random.PRNGKey(hash((N, B_o, B_i, "muld")) % (2**31))
    k1, k2 = jax.random.split(k)
    R, C = N * B_o, N * B_i
    lhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k1))
    rhs = _dense(R, C, _rand((R, C), k2))
    res = _assert_oracle(lhs, rhs, jnp.multiply, is_intersection=True)
    assert all(d.is_sparse for d in res.dims), res.dims


# --------------------------------------------------------------------------- #
# B op B  (matched grid)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "N,B_o,B_i,op,is_int",
    [
        (3, 2, 4, jnp.add, False),
        (3, 4, 2, jnp.add, False),
        (4, 2, 2, jnp.multiply, True),
        (2, 3, 3, jnp.multiply, True),
        (5, 1, 3, jnp.add, False),
        (4, 2, 1, jnp.multiply, True),
        (3, 3, 2, jnp.subtract, False),
    ],
)
def test_B_op_B(N, B_o, B_i, op, is_int):
    k = jax.random.PRNGKey(hash((N, B_o, B_i, op.__name__)) % (2**31))
    k1, k2 = jax.random.split(k)
    lhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k1))
    rhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k2))
    res = _assert_oracle(lhs, rhs, op, is_intersection=is_int)
    # Closure: B op B -> B (same grid).
    assert all(d.is_sparse for d in res.dims), res.dims


# --------------------------------------------------------------------------- #
# val=None operands (uniform-ones structure)
# --------------------------------------------------------------------------- #
def test_val_none_block_diagonal():
    k = jax.random.PRNGKey(11)
    N, B_o, B_i = 2, 2, 3
    R, C = N * B_o, N * B_i
    lhs = _dense(R, C, _rand((R, C), k))
    rhs_ones = _diag(0, 1, N, B_o, B_i, jnp.ones((N, B_o, B_i)))
    rhs_none = SparseTensor(rhs_ones.out_dims, rhs_ones.primal_dims, None)
    # add: D + ones-blocks
    res = elementwise_dense_block_diagonal(lhs, rhs_none, jnp.add)
    oracle = jnp.add(lhs.dense(), rhs_ones.dense())
    np.testing.assert_allclose(np.asarray(res.dense()), np.asarray(oracle), atol=ATOL)


# --------------------------------------------------------------------------- #
# scalar_mult folding (both operands)
# --------------------------------------------------------------------------- #
def test_scalar_mult_fold_add():
    k = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(k)
    N, B_o, B_i = 3, 2, 3
    R, C = N * B_o, N * B_i
    lhs = _dense(R, C, _rand((R, C), k1)).copy(scalar_mult=jnp.array(2.5))
    rhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k2)).copy(
        scalar_mult=jnp.array(-1.5)
    )
    _assert_oracle(lhs, rhs, jnp.add)


def test_scalar_mult_fold_mul():
    k = jax.random.PRNGKey(4)
    k1, k2 = jax.random.split(k)
    N, B_o, B_i = 3, 3, 2
    R, C = N * B_o, N * B_i
    lhs = _dense(R, C, _rand((R, C), k1)).copy(scalar_mult=jnp.array(0.5))
    rhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k2)).copy(
        scalar_mult=jnp.array(3.0)
    )
    _assert_oracle(lhs, rhs, jnp.multiply, is_intersection=True)


# --------------------------------------------------------------------------- #
# Non-zero fill -> dense oracle fallback (still correct)
# --------------------------------------------------------------------------- #
def test_nonzero_fill_fallback():
    k = jax.random.PRNGKey(5)
    k1, k2 = jax.random.split(k)
    N, B_o, B_i = 2, 2, 2
    R, C = N * B_o, N * B_i
    lhs = _dense(R, C, _rand((R, C), k1))
    rhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k2)).copy(
        fill_value=jnp.array(0.7)
    )
    _assert_oracle(lhs, rhs, jnp.add)
    _assert_oracle(lhs, rhs, jnp.multiply, is_intersection=True)


# --------------------------------------------------------------------------- #
# Mismatched B/B grid -> dense oracle fallback
# --------------------------------------------------------------------------- #
def test_mismatched_BB_grid_fallback():
    k = jax.random.PRNGKey(6)
    k1, k2 = jax.random.split(k)
    # lhs: N=2 blocks of (3,3) -> 6x6 ; rhs: N=3 blocks of (2,2) -> 6x6
    lhs = _diag(0, 1, 2, 3, 3, _rand((2, 3, 3), k1))
    rhs = _diag(0, 1, 3, 2, 2, _rand((3, 2, 2), k2))
    assert lhs.shape == rhs.shape == (6, 6)
    _assert_oracle(lhs, rhs, jnp.add)


# --------------------------------------------------------------------------- #
# Search-shaped (attention + mlp) instances
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("S,H,dk", [(4, 2, 4), (8, 2, 4), (8, 2, 8)])
def test_attention_shaped_D_add_B(S, H, dk):
    # A per-head block-diagonal Jacobian (H heads as meta blocks, dk x dk square
    # blocks) merged (add) onto a dense (H*dk, H*dk) slab.
    k = jax.random.PRNGKey(hash((S, H, dk)) % (2**31))
    k1, k2 = jax.random.split(k)
    R = C = H * dk
    lhs = _dense(R, C, _rand((R, C), k1))
    rhs = _diag(0, 1, H, dk, dk, _rand((H, dk, dk), k2))
    _assert_oracle(lhs, rhs, jnp.add)


@pytest.mark.parametrize("S,H,dk", [(4, 2, 4), (8, 2, 4), (8, 2, 8)])
def test_attention_shaped_D_mul_B(S, H, dk):
    k = jax.random.PRNGKey(hash((S, H, dk, "m")) % (2**31))
    k1, k2 = jax.random.split(k)
    R = C = H * dk
    lhs = _dense(R, C, _rand((R, C), k1))
    rhs = _diag(0, 1, H, dk, dk, _rand((H, dk, dk), k2))
    res = _assert_oracle(lhs, rhs, jnp.multiply, is_intersection=True)
    assert all(d.is_sparse for d in res.dims)


@pytest.mark.parametrize(
    "N,B_o,B_i",
    list(itertools.product([2, 4], [2, 3], [2, 3])),
)
def test_mlp_shaped_arbitrary_blocks(N, B_o, B_i):
    k = jax.random.PRNGKey(hash((N, B_o, B_i, "mlp")) % (2**31))
    k1, k2 = jax.random.split(k)
    R, C = N * B_o, N * B_i
    lhs = _dense(R, C, _rand((R, C), k1))
    rhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k2))
    _assert_oracle(lhs, rhs, jnp.add)
    _assert_oracle(lhs, rhs, jnp.multiply, is_intersection=True)


# --------------------------------------------------------------------------- #
# Randomized fuzz over arbitrary component instances
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(16))
def test_fuzz_D_op_B(seed):
    rng = np.random.default_rng(seed)
    N = int(rng.integers(1, 5))
    B_o = int(rng.integers(1, 5))
    B_i = int(rng.integers(1, 5))
    k = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(k)
    R, C = N * B_o, N * B_i
    lhs = _dense(R, C, _rand((R, C), k1))
    rhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k2))
    # D op B both orders, both union (add) and intersection (mul).
    _assert_oracle(lhs, rhs, jnp.add)
    _assert_oracle(rhs, lhs, jnp.add)
    _assert_oracle(lhs, rhs, jnp.multiply, is_intersection=True)
    _assert_oracle(rhs, lhs, jnp.multiply, is_intersection=True)


@pytest.mark.parametrize("seed", range(16))
def test_fuzz_B_op_B(seed):
    rng = np.random.default_rng(2000 + seed)
    N = int(rng.integers(1, 5))
    B_o = int(rng.integers(1, 5))
    B_i = int(rng.integers(1, 5))
    k = jax.random.PRNGKey(2000 + seed)
    k1, k2 = jax.random.split(k)
    lhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k1))
    rhs = _diag(0, 1, N, B_o, B_i, _rand((N, B_o, B_i), k2))
    _assert_oracle(lhs, rhs, jnp.add)
    _assert_oracle(lhs, rhs, jnp.multiply, is_intersection=True)
    _assert_oracle(lhs, rhs, jnp.subtract)
