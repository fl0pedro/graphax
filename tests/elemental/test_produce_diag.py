"""Dense-oracle tests for the ``Diag`` producer kernel (``D -> B``).

``produce_diag(st, i, j, factor)`` block-diagonalises the logical Dense pair
``(i, j)`` of a ``SparseTensor`` into a coupled meta-block-diagonal
``DiagonalIndex`` pair (``factor`` meta-diagonal blocks; off-block-diagonal mass
zeroed). Two oracles are asserted, both to ``~1e-5``:

  (A) MASK CORRECTNESS — ``produce_diag(st).dense()`` equals the original dense
      slab masked by the hand-built meta-block-diagonal indicator
      ``[gi == gj]`` (SQUARE and RECTANGULAR ``Ni != Nj`` pairs, several
      factors, ``block_size == 1`` edges, ``val=None``, untouched extra axes).

  (B) DOWNSTREAM CONTRACTION — the produced ``B`` then routes correctly through
      :func:`contract_dense_block_diagonal` against a Dense partner, equalling
      the ``(block-masked-then-matmul)`` dense oracle for BOTH ``B @ D`` (the
      produced pair's free side contracted) and ``D @ B`` (its block side
      contracted). This guards the produced dim ids / ``other_id`` alignment
      that earlier caused permutation bugs.

Plus search-shaped (attention ``S in {4,8}, H=2, dk=4/8``) instances.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex
from graphax.sparse.tensor import SparseTensor
from graphax.sparse.elemental.produce_diag import produce_diag
from graphax.sparse.elemental.contract_D_B import contract_dense_block_diagonal

jax.config.update("jax_enable_x64", False)

ATOL = 1e-5


# --------------------------------------------------------------------------- #
# Builders / oracle
# --------------------------------------------------------------------------- #
def _rand(shape, key):
    return jax.random.normal(key, shape, dtype=jnp.float32)


def _block_mask(Ni, Nj, f):
    """The ``(Ni, Nj)`` meta-block-diagonal indicator ``[gi == gj]`` where
    ``gi = i // (Ni / f)`` and ``gj = j // (Nj / f)`` (the dense mask the
    producer applies)."""
    bi, bj = Ni // f, Nj // f
    gi = np.arange(Ni) // bi
    gj = np.arange(Nj) // bj
    return (gi[:, None] == gj[None, :]).astype(np.float32)


def _dense2d(Ni, Nj, val):
    """A 2-D plain-Dense SparseTensor: out dim ``i`` (axis 0), primal dim ``j``
    (axis 1)."""
    return SparseTensor(
        (DenseIndex(0, Ni, 0),), (DenseIndex(1, Nj, 1),), val
    )


def _assert_mask(Ni, Nj, f, val):
    """(A): produce_diag(D).dense() == original dense masked by the block mask."""
    st = _dense2d(Ni, Nj, val)
    res = produce_diag(st, 0, 1, f)
    mask = _block_mask(Ni, Nj, f)
    base = np.ones((Ni, Nj), np.float32) if val is None else np.asarray(val)
    oracle = base * mask
    d = np.asarray(res.dense())
    assert d.shape == oracle.shape, (d.shape, oracle.shape)
    np.testing.assert_allclose(d, oracle, atol=ATOL)
    return res


# --------------------------------------------------------------------------- #
# (A) MASK CORRECTNESS — square / rectangular / several factors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "Ni,Nj,f",
    [
        (6, 6, 3),   # square, bi=bj=2
        (4, 4, 2),   # square, bi=bj=2
        (8, 8, 4),   # square, bi=bj=2
        (6, 6, 2),   # square, bi=bj=3
        (6, 9, 3),   # rectangular bi=2 < bj=3
        (8, 4, 2),   # rectangular bi=4 > bj=2
        (4, 6, 2),   # rectangular bi=2 < bj=3
        (9, 6, 3),   # rectangular bi=3 > bj=2
        (12, 8, 4),  # rectangular bigger
    ],
)
def test_mask_square_and_rectangular(Ni, Nj, f):
    val = _rand((Ni, Nj), jax.random.PRNGKey(hash((Ni, Nj, f)) % (2**31)))
    _assert_mask(Ni, Nj, f, val)


# --------------------------------------------------------------------------- #
# (A) block_size == 1 edges (a side whose block collapses to None)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "Ni,Nj,f",
    [
        (3, 6, 3),   # bi == 1  -> i side has no block axis
        (6, 3, 3),   # bj == 1  -> j side has no block axis
        (3, 3, 3),   # both bi == bj == 1 (pure permutation diagonal)
    ],
)
def test_mask_block_size_one(Ni, Nj, f):
    res = _assert_mask(
        Ni, Nj, f, _rand((Ni, Nj), jax.random.PRNGKey(Ni * 31 + Nj))
    )
    bi, bj = Ni // f, Nj // f
    di, dj = res.dims
    assert di.block_size == (bi if bi > 1 else None)
    assert dj.block_size == (bj if bj > 1 else None)
    # The produced pair is a coupled DiagonalIndex (B) with matching ids.
    assert di.is_sparse and dj.is_sparse
    assert di.other_id == dj.id and dj.other_id == di.id
    assert di.size == f and dj.size == f


# --------------------------------------------------------------------------- #
# (A) factor == 1 is identity (no degenerate meta-count-1 pair fabricated)
# --------------------------------------------------------------------------- #
def test_factor_one_is_identity():
    val = _rand((6, 6), jax.random.PRNGKey(1))
    st = _dense2d(6, 6, val)
    res = produce_diag(st, 0, 1, 1)
    assert res is st
    assert all(not d.is_sparse for d in res.dims)


# --------------------------------------------------------------------------- #
# (A) val=None (uniform all-ones) stays val=None, masks to ones-on-diagonal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Ni,Nj,f", [(6, 6, 3), (6, 9, 3), (8, 4, 2)])
def test_mask_val_none(Ni, Nj, f):
    res = _assert_mask(Ni, Nj, f, None)
    assert res.val is None


# --------------------------------------------------------------------------- #
# (A) untouched extra dims ride through; their axes are remapped correctly
# --------------------------------------------------------------------------- #
def test_untouched_leading_dim():
    A, Ni, Nj, f = 3, 6, 9, 3
    val = _rand((A, Ni, Nj), jax.random.PRNGKey(21))
    st = SparseTensor(
        (DenseIndex(0, A, 0), DenseIndex(1, Ni, 1)),
        (DenseIndex(2, Nj, 2),),
        val,
    )
    res = produce_diag(st, 1, 2, f)
    oracle = np.asarray(val) * _block_mask(Ni, Nj, f)[None, :, :]
    np.testing.assert_allclose(np.asarray(res.dense()), oracle, atol=ATOL)


def test_untouched_trailing_dim():
    Ni, Nj, C, f = 6, 9, 4, 3
    val = _rand((Ni, Nj, C), jax.random.PRNGKey(22))
    st = SparseTensor(
        (DenseIndex(0, Ni, 0),),
        (DenseIndex(1, Nj, 1), DenseIndex(2, C, 2)),
        val,
    )
    res = produce_diag(st, 0, 1, f)
    oracle = np.asarray(val) * _block_mask(Ni, Nj, f)[:, :, None]
    np.testing.assert_allclose(np.asarray(res.dense()), oracle, atol=ATOL)


def test_reversed_arg_order():
    # i / j given out-of-order (i is the primal dim, j is the out dim).
    Ni, Nj, f = 6, 9, 3
    val = _rand((Ni, Nj), jax.random.PRNGKey(23))
    st = _dense2d(Ni, Nj, val)
    res = produce_diag(st, 1, 0, f)
    oracle = np.asarray(val) * _block_mask(Ni, Nj, f)
    np.testing.assert_allclose(np.asarray(res.dense()), oracle, atol=ATOL)


# --------------------------------------------------------------------------- #
# Mis-dispatch guards
# --------------------------------------------------------------------------- #
def test_rejects_non_divisor_factor():
    st = _dense2d(6, 6, _rand((6, 6), jax.random.PRNGKey(2)))
    with pytest.raises(ValueError):
        produce_diag(st, 0, 1, 4)  # 4 does not divide 6


def test_rejects_same_index():
    st = _dense2d(6, 6, _rand((6, 6), jax.random.PRNGKey(3)))
    with pytest.raises(ValueError):
        produce_diag(st, 0, 0, 3)


def test_rejects_out_of_range():
    st = _dense2d(6, 6, _rand((6, 6), jax.random.PRNGKey(4)))
    with pytest.raises(ValueError):
        produce_diag(st, 0, 5, 3)


def test_rejects_already_sparse_dim():
    # Run produce_diag once to get a B pair, then feed it back.
    # With GRAPHAX_KEEP_BLOCKDIAG (default ON) re-masking the SAME coupled pair by
    # the SAME factor is an idempotent no-op (returns the block-diagonal unchanged)
    # so the keep-sparse per-vertex re-mask is sound. With it forced OFF the legacy
    # "reject already-sparse dim" contract holds. A DIFFERENT factor still raises
    # either way (not the matching-coupled short-circuit).
    st = _dense2d(6, 6, _rand((6, 6), jax.random.PRNGKey(5)))
    B = produce_diag(st, 0, 1, 3)
    import graphax.sparse.elemental.produce_diag as _pd

    if _pd._KEEP_BLOCKDIAG:
        again = produce_diag(B, 0, 1, 3)
        assert again is B  # idempotent no-op on the matching coupled pair
    else:
        with pytest.raises(ValueError):
            produce_diag(B, 0, 1, 3)
    # A non-matching factor is rejected regardless of the flag.
    with pytest.raises(ValueError):
        produce_diag(B, 0, 1, 2)


# --------------------------------------------------------------------------- #
# (B) DOWNSTREAM CONTRACTION via contract_D_B
# --------------------------------------------------------------------------- #
def _downstream_B_at_D(Ni, Nj, f, Q, key):
    """produce_diag -> B as lhs; contract its FREE primal side (Nj) against a
    dense rhs out-side. Oracle: (masked-dense) @ rhs.dense()."""
    k1, k2 = jax.random.split(key)
    val = _rand((Ni, Nj), k1)
    B = produce_diag(_dense2d(Ni, Nj, val), 0, 1, f)
    rhs = SparseTensor(
        (DenseIndex(0, Nj, 0),), (DenseIndex(1, Q, 1),), _rand((Nj, Q), k2)
    )
    res = contract_dense_block_diagonal(B, rhs)
    oracle = (np.asarray(val) * _block_mask(Ni, Nj, f)) @ np.asarray(rhs.dense())
    assert res.dense().shape == oracle.shape, (res.dense().shape, oracle.shape)
    np.testing.assert_allclose(np.asarray(res.dense()), oracle, atol=ATOL)


def _downstream_D_at_B(P, Ni, Nj, f, key):
    """produce_diag -> B as rhs; contract its BLOCK out-side (Ni) against a dense
    lhs primal-side. Oracle: lhs.dense() @ (masked-dense)."""
    k1, k2 = jax.random.split(key)
    val = _rand((Ni, Nj), k1)
    B = produce_diag(_dense2d(Ni, Nj, val), 0, 1, f)
    lhs = SparseTensor(
        (DenseIndex(0, P, 0),), (DenseIndex(1, Ni, 1),), _rand((P, Ni), k2)
    )
    res = contract_dense_block_diagonal(lhs, B)
    oracle = np.asarray(lhs.dense()) @ (np.asarray(val) * _block_mask(Ni, Nj, f))
    assert res.dense().shape == oracle.shape, (res.dense().shape, oracle.shape)
    np.testing.assert_allclose(np.asarray(res.dense()), oracle, atol=ATOL)


@pytest.mark.parametrize(
    "Ni,Nj,f,Q",
    [
        (6, 6, 3, 5),   # square
        (6, 9, 3, 4),   # rectangular bi<bj
        (8, 4, 2, 3),   # rectangular bi>bj
        (6, 3, 3, 5),   # bj == 1
        (3, 6, 3, 4),   # bi == 1
    ],
)
def test_downstream_B_at_D(Ni, Nj, f, Q):
    _downstream_B_at_D(Ni, Nj, f, Q, jax.random.PRNGKey(hash((Ni, Nj, f, Q)) % (2**31)))


@pytest.mark.parametrize(
    "P,Ni,Nj,f",
    [
        (4, 6, 6, 3),   # square
        (4, 6, 9, 3),   # rectangular bi<bj
        (3, 8, 4, 2),   # rectangular bi>bj
        (5, 3, 6, 3),   # bi == 1
        (5, 6, 3, 3),   # bj == 1
    ],
)
def test_downstream_D_at_B(P, Ni, Nj, f):
    _downstream_D_at_B(P, Ni, Nj, f, jax.random.PRNGKey(hash((P, Ni, Nj, f)) % (2**31)))


# --------------------------------------------------------------------------- #
# Search-shaped (attention): block-diagonalise a per-head Jacobian factor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("S,H,dk", [(4, 2, 4), (8, 2, 4), (8, 2, 8)])
def test_attention_shaped(S, H, dk):
    # An (H*dk, H*dk) Jacobian factor block-diagonalised over the H heads
    # (factor = H, each head a dk x dk square block), then contracted against a
    # dense (S, H*dk) activation slab on the LEFT (D @ B).
    N = H * dk
    key = jax.random.PRNGKey(hash((S, H, dk)) % (2**31))
    k1, k2 = jax.random.split(key)
    val = _rand((N, N), k1)
    B = produce_diag(_dense2d(N, N, val), 0, 1, H)
    mask = _block_mask(N, N, H)
    # (A) mask
    np.testing.assert_allclose(
        np.asarray(B.dense()), np.asarray(val) * mask, atol=ATOL
    )
    # (B) downstream D @ B
    lhs = SparseTensor(
        (DenseIndex(0, S, 0),), (DenseIndex(1, N, 1),), _rand((S, N), k2)
    )
    res = contract_dense_block_diagonal(lhs, B)
    oracle = np.asarray(lhs.dense()) @ (np.asarray(val) * mask)
    np.testing.assert_allclose(np.asarray(res.dense()), oracle, atol=ATOL)


# --------------------------------------------------------------------------- #
# Randomized fuzz over arbitrary (Ni, Nj, factor)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(12))
def test_fuzz_mask(seed):
    rng = np.random.default_rng(seed)
    f = int(rng.integers(2, 5))
    bi = int(rng.integers(1, 4))
    bj = int(rng.integers(1, 4))
    Ni, Nj = f * bi, f * bj
    val = _rand((Ni, Nj), jax.random.PRNGKey(seed))
    _assert_mask(Ni, Nj, f, val)


@pytest.mark.parametrize("seed", range(12))
def test_fuzz_downstream_B_at_D(seed):
    rng = np.random.default_rng(2000 + seed)
    f = int(rng.integers(2, 5))
    bi = int(rng.integers(1, 4))
    bj = int(rng.integers(1, 4))
    Q = int(rng.integers(1, 6))
    Ni, Nj = f * bi, f * bj
    _downstream_B_at_D(Ni, Nj, f, Q, jax.random.PRNGKey(2000 + seed))
