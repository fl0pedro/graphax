"""Dense-oracle tests for the Dense <-> MULTI-block-diagonal contraction kernel
(``contract_dense_multi_block_diagonal``).

Every test asserts the kernel == the DENSE ORACLE: materialize both operands via
``.dense()`` and run a plain dense matmul over the contracted axes. Covers:

  * 2 and 3 block-diagonal CONTRACTED pairs, SQUARE and RECTANGULAR blocks,
  * mixes of contracted B-pairs + dense CONTRACTED + dense FREE dims,
  * attention-shaped instances (the structured contraction a ViT/attn jacve
    produces — several mixed-type dims contracted at once, one operand fully
    dense, the other carrying multiple coupled DiagonalIndex pairs).

The kernel stays nnz-sparse (single batched einsum over all meta axes, no full
``prod_p N_p`` dense intermediate); these tests grade VALUES only — the sparsity
gate is checked at the dispatch level (DISPATCH_STATS).
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor
from graphax.sparse.elemental.contract_D_B import (
    contract_dense_multi_block_diagonal,
)

jax.config.update("jax_enable_x64", False)

ATOL = 1e-5


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _rand(shape, key):
    return jax.random.normal(key, shape, dtype=jnp.float32)


def _build_multi_struct(pairs, dense_contract, dense_free, key):
    """Build a structured SparseTensor with several contracted block-diagonal
    pairs coexisting with a dense-contracted dim and a dense-free dim.

    ``pairs`` is a list of ``(N, Bc, Bf)`` — each a meta-block-diagonal pair
    whose CONTRACTED side (block ``Bc``) lives on the OUT side and whose FREE
    partner (block ``Bf``) lives on the PRIMAL side.  ``dense_contract`` is the
    size ``Kd`` of an extra dense OUT (contracted) dim (0 = none); ``dense_free``
    is the size ``Q`` of an extra dense PRIMAL (free) dim (0 = none).

    The val is packed in canonical order:
        (N1, Bc1, N2, Bc2, ..., [Kd], Bf1, Bf2, ..., [Q])
    with every meta axis shared between the pair's two sides (block-diagonal).
    """
    out_dims = []
    primal_dims = []
    val_shape = []
    # contracted (out-side) packed axes: per pair (meta, Bc), then dense Kd.
    axis = 0
    meta_axis = {}  # pair index -> physical meta axis
    bc_axis = {}
    for p, (N, Bc, Bf) in enumerate(pairs):
        meta_axis[p] = axis
        val_shape.append(N)
        axis += 1
        bc_axis[p] = axis
        val_shape.append(Bc)
        axis += 1
    kd_axis = None
    if dense_contract:
        kd_axis = axis
        val_shape.append(dense_contract)
        axis += 1
    # free (primal-side) packed axes: per pair Bf, then dense Q.
    bf_axis = {}
    for p, (N, Bc, Bf) in enumerate(pairs):
        bf_axis[p] = axis
        val_shape.append(Bf)
        axis += 1
    q_axis = None
    if dense_free:
        q_axis = axis
        val_shape.append(dense_free)
        axis += 1

    val = _rand(tuple(val_shape), key)

    # ids: out diag-contracted (0..np-1), dense contracted (np), primal free
    # partners (np+? ...). Keep partner other_id linkage.
    n_pairs = len(pairs)
    next_id = 0
    out_ids = []
    for p in range(n_pairs):
        out_ids.append(next_id)
        next_id += 1
    kd_id = None
    if dense_contract:
        kd_id = next_id
        next_id += 1
    partner_ids = []
    for p in range(n_pairs):
        partner_ids.append(next_id)
        next_id += 1
    q_id = None
    if dense_free:
        q_id = next_id
        next_id += 1

    for p, (N, Bc, Bf) in enumerate(pairs):
        out_dims.append(
            DiagonalIndex(
                out_ids[p], N, axis=meta_axis[p], other_id=partner_ids[p],
                block_size=Bc if Bc > 1 else None,
                block_axis=bc_axis[p] if Bc > 1 else None,
            )
        )
    if dense_contract:
        out_dims.append(DenseIndex(kd_id, dense_contract, axis=kd_axis))

    for p, (N, Bc, Bf) in enumerate(pairs):
        primal_dims.append(
            DiagonalIndex(
                partner_ids[p], N, axis=meta_axis[p], other_id=out_ids[p],
                block_size=Bf if Bf > 1 else None,
                block_axis=bf_axis[p] if Bf > 1 else None,
            )
        )
    if dense_free:
        primal_dims.append(DenseIndex(q_id, dense_free, axis=q_axis))

    return SparseTensor(tuple(out_dims), tuple(primal_dims), val)


def _dense_lhs_for(rhs, out_sizes, key):
    """A fully-dense LHS whose out dims are ``out_sizes`` and whose primal
    (contracted) side MIRRORS ``rhs``'s out dims — one dense primal dim per rhs
    out dim, SAME id and logical size — so the matmul topology aligns them
    pairwise by id (the real jacve layout: lhs.dims=[D,D,...] with one contracted
    D per structured rhs out dim)."""
    contract_sizes = [int(d.logical_size) for d in rhs.out_dims]
    val = _rand(tuple(out_sizes) + tuple(contract_sizes), key)
    n_out = len(out_sizes)
    out = [DenseIndex(i, s, i) for i, s in enumerate(out_sizes)]
    primal = [
        DenseIndex(n_out + k, cs, n_out + k)
        for k, cs in enumerate(contract_sizes)
    ]
    return SparseTensor(tuple(out), tuple(primal), val)


def _dense_oracle(lhs, rhs):
    """Plain dense vertex-elim contraction over the aligned contracted pairs."""
    from graphax.sparse.ops.matmul import matmul as _matmul
    from graphax.sparse.elemental.dispatch import _to_dense_st

    return np.asarray(_matmul(_to_dense_st(lhs), _to_dense_st(rhs)).dense())


def _assert_oracle(lhs, rhs):
    res = contract_dense_multi_block_diagonal(lhs, rhs)
    assert res is not None, "kernel declined (returned None) — out of scope"
    oracle = _dense_oracle(lhs, rhs)
    got = np.asarray(res.dense())
    assert got.shape == oracle.shape, (got.shape, oracle.shape)
    np.testing.assert_allclose(got, oracle, atol=ATOL)
    # closure: result is fully dense
    assert all(not d.is_sparse for d in res.dims), res.dims
    return res


# --------------------------------------------------------------------------- #
# 2 contracted pairs — square + rectangular
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "p1,p2",
    [
        ((2, 2, 2), (3, 2, 2)),   # square blocks
        ((2, 3, 4), (3, 2, 3)),   # rectangular both
        ((2, 1, 3), (3, 2, 1)),   # mixed block-1 sides
        ((4, 2, 2), (2, 3, 2)),   # bigger meta
    ],
)
def test_two_pairs(p1, p2):
    pairs = [p1, p2]
    P = 3
    k1, k2 = jax.random.split(jax.random.PRNGKey(hash((p1, p2)) % (2**31)))
    rhs = _build_multi_struct(pairs, dense_contract=0, dense_free=0, key=k2)
    lhs = _dense_lhs_for(rhs, [P], k1)
    _assert_oracle(lhs, rhs)


# --------------------------------------------------------------------------- #
# 3 contracted pairs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "pairs",
    [
        [(2, 2, 2), (2, 2, 2), (2, 2, 2)],       # all square
        [(2, 2, 3), (3, 1, 2), (2, 3, 1)],       # rectangular mix
    ],
)
def test_three_pairs(pairs):
    P = 2
    k1, k2 = jax.random.split(jax.random.PRNGKey(hash(tuple(pairs)) % (2**31)))
    rhs = _build_multi_struct(pairs, dense_contract=0, dense_free=0, key=k2)
    lhs = _dense_lhs_for(rhs, [P], k1)
    _assert_oracle(lhs, rhs)


# --------------------------------------------------------------------------- #
# Mix: contracted B-pairs + dense CONTRACTED + dense FREE
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "pairs,Kd,Q",
    [
        ([(2, 2, 2), (3, 2, 2)], 3, 2),
        ([(2, 3, 2), (2, 2, 3)], 2, 4),
        # 3 pairs + dense contracted + dense free: kept SMALL so the dense
        # oracle (which densifies the multi-pair RHS to its full matrix) stays
        # tractable — the kernel itself is nnz-cheap.
        ([(2, 1, 2), (2, 2, 1), (2, 1, 1)], 2, 2),
    ],
)
def test_mixed_dense_contracted_and_free(pairs, Kd, Q):
    P = 3
    k1, k2 = jax.random.split(jax.random.PRNGKey(hash((tuple(pairs), Kd, Q)) % (2**31)))
    rhs = _build_multi_struct(pairs, dense_contract=Kd, dense_free=Q, key=k2)
    lhs = _dense_lhs_for(rhs, [P], k1)
    _assert_oracle(lhs, rhs)


# --------------------------------------------------------------------------- #
# Multiple LHS free out dims (batch axes ride through)
# --------------------------------------------------------------------------- #
def test_multi_lhs_out_dims():
    pairs = [(2, 2, 2), (2, 2, 2)]
    k1, k2 = jax.random.split(jax.random.PRNGKey(11))
    rhs = _build_multi_struct(pairs, dense_contract=0, dense_free=0, key=k2)
    lhs = _dense_lhs_for(rhs, [2, 3], k1)
    _assert_oracle(lhs, rhs)


# --------------------------------------------------------------------------- #
# val=None structured operand (all-ones blocks)
# --------------------------------------------------------------------------- #
def test_val_none_struct():
    # val=None structured ≡ all-ones blocks. The generic dense() oracle has a
    # pre-existing crash densifying a val=None DiagonalIndex with explicit block
    # axes (documented in test_contract_D_B::test_val_none_block_diagonal_rhs),
    # so compare the kernel's val=None result against the explicit all-ones-block
    # equivalent operand instead.
    pairs = [(2, 2, 2), (3, 2, 2)]
    k1, k2 = jax.random.split(jax.random.PRNGKey(13))
    rhs_ones = _build_multi_struct(pairs, dense_contract=0, dense_free=0, key=k2)
    rhs_ones = SparseTensor(
        rhs_ones.out_dims, rhs_ones.primal_dims, jnp.ones_like(rhs_ones.val)
    )
    rhs_none = SparseTensor(rhs_ones.out_dims, rhs_ones.primal_dims, None)
    lhs = _dense_lhs_for(rhs_ones, [3], k1)
    res = contract_dense_multi_block_diagonal(lhs, rhs_none)
    assert res is not None
    oracle = _dense_oracle(lhs, rhs_ones)
    np.testing.assert_allclose(np.asarray(res.dense()), oracle, atol=ATOL)


# --------------------------------------------------------------------------- #
# scalar_mult folding
# --------------------------------------------------------------------------- #
def test_scalar_mult_fold():
    pairs = [(2, 2, 3), (2, 3, 2)]
    k1, k2 = jax.random.split(jax.random.PRNGKey(17))
    rhs = _build_multi_struct(pairs, 0, 0, k2).copy(scalar_mult=jnp.array(-1.5))
    lhs = _dense_lhs_for(rhs, [3], k1).copy(scalar_mult=jnp.array(2.5))
    _assert_oracle(lhs, rhs)


# --------------------------------------------------------------------------- #
# Attention-shaped: plain diagonals (block 1) + dense contracted (the real
# jacve case: lhs.dims=[D,D,D,D,D] @ rhs.dims=[B,B,D,B,B,D])
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("H,Hk,Kd", [(4, 2, 4), (2, 2, 4), (4, 4, 2)])
def test_attention_shaped_plain_diags(H, Hk, Kd):
    # Two plain-diagonal pairs (block 1) + one dense contracted dim — exactly the
    # structured contraction the attention jacve densified.
    pairs = [(H, 1, 1), (Hk, 1, 1)]
    k1, k2 = jax.random.split(jax.random.PRNGKey(hash((H, Hk, Kd)) % (2**31)))
    rhs = _build_multi_struct(pairs, dense_contract=Kd, dense_free=1, key=k2)
    lhs = _dense_lhs_for(rhs, [H, Kd], k1)  # 2 free out dims
    _assert_oracle(lhs, rhs)


# --------------------------------------------------------------------------- #
# Fuzz over random multi-pair instances
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(16))
def test_fuzz(seed):
    rng = np.random.default_rng(seed)
    n_pairs = int(rng.integers(2, 4))
    pairs = [
        (int(rng.integers(1, 4)), int(rng.integers(1, 4)), int(rng.integers(1, 4)))
        for _ in range(n_pairs)
    ]
    Kd = int(rng.integers(0, 3))
    Q = int(rng.integers(0, 3))
    P = int(rng.integers(1, 4))
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    rhs = _build_multi_struct(pairs, dense_contract=Kd, dense_free=Q, key=k2)
    lhs = _dense_lhs_for(rhs, [P], k1)
    _assert_oracle(lhs, rhs)
