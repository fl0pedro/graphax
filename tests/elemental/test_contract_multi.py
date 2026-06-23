"""Dense-oracle tests for ``contract_multi_structured`` — the B@B-multi kernel
(BOTH operands carry block-diagonal structure on their contracted sides).

Every test asserts the kernel == the DENSE ORACLE: materialize both operands via
``.dense()`` and contract over the aligned contracted axes with ``jnp``. Covers:

  * two B@B contracted pairs (square + rectangular blocks),
  * mixed plain-Dense contracted pair + B@B pair,
  * ``val is None`` (all-ones blocks),
  * attention-shaped (per-head block-diagonal Jacobians),
  * scalar_mult folding, plain-diagonal (block-1) pairs, free dense ride-through.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor
from graphax.sparse.elemental.contract_multi import contract_multi_structured

jax.config.update("jax_enable_x64", False)

ATOL = 1e-5


def _rand(shape, key):
    return jax.random.normal(key, shape, dtype=jnp.float32)


def _build_two_pair_operand(p1, p2, key, *, con_on_out):
    """Build a SparseTensor with TWO block-diagonal pairs.

    Each pair is ``(N, B_first, B_second)``.  If ``con_on_out`` the FIRST block
    side lives on the OUT side and the SECOND on the PRIMAL side (the rhs
    orientation: out=contracted, primal=free); otherwise the FIRST is on out and
    SECOND on primal too but interpreted as (free_out, contracted_primal) (the
    lhs orientation).  The two interpretations are identical in layout — the
    caller picks which side contracts by aligning ids.

    Layout chosen so the contracting sides are: lhs.primal pairs contract
    rhs.out pairs.  Here we just build a generic 2-pair operand:
      out  = [B(id a0, N1, blk Po1), B(id a1, N2, blk Po2)]
      primal = [B(id b0, N1, blk Qf1), B(id b1, N2, blk Qf2)]
    val packed (N1,N2, Po1,Po2, Qf1,Qf2).
    """
    (N1, Po1, Qf1) = p1
    (N2, Po2, Qf2) = p2
    val = _rand((N1, N2, Po1, Po2, Qf1, Qf2), key)
    o0 = DiagonalIndex(0, N1, axis=0, other_id=2,
                       block_size=Po1 if Po1 > 1 else None,
                       block_axis=2 if Po1 > 1 else None)
    o1 = DiagonalIndex(1, N2, axis=1, other_id=3,
                       block_size=Po2 if Po2 > 1 else None,
                       block_axis=3 if Po2 > 1 else None)
    p2d = DiagonalIndex(2, N1, axis=0, other_id=0,
                        block_size=Qf1 if Qf1 > 1 else None,
                        block_axis=4 if Qf1 > 1 else None)
    p3d = DiagonalIndex(3, N2, axis=1, other_id=1,
                        block_size=Qf2 if Qf2 > 1 else None,
                        block_axis=5 if Qf2 > 1 else None)
    return SparseTensor((o0, o1), (p2d, p3d), val, check_consistency=False)


def _oracle(lhs, rhs):
    """Plain dense vertex-elim contraction over the aligned contracted pairs."""
    from graphax.sparse.ops.matmul import matmul as _matmul
    from graphax.sparse.elemental.dispatch import _to_dense_st

    return np.asarray(_matmul(_to_dense_st(lhs), _to_dense_st(rhs)).dense())


def _assert(lhs, rhs):
    from graphax.sparse.ops.matmul import _align_contract_dims
    from graphax.sparse.elemental.dispatch import _classify_pair

    pairs = _align_contract_dims(lhs.primal_dims, rhs.out_dims, embed=True)
    kinds = [_classify_pair(ld, rd) for ld, rd in pairs]
    res = contract_multi_structured(lhs, rhs, pairs, kinds)
    assert res is not None, "kernel declined (returned None) — out of scope"
    oracle = _oracle(lhs, rhs)
    got = np.asarray(res.dense())
    assert got.shape == oracle.shape, (got.shape, oracle.shape)
    np.testing.assert_allclose(got, oracle, atol=ATOL)
    for d in res.dims:
        assert not d.is_compressed
    return res


# --------------------------------------------------------------------------- #
# two B@B pairs — the core ('B','B','B','B') @ ('B','B','B','B') case
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "lp1,lp2,rp1,rp2",
    [
        # lhs pair p: (N, Po_out, Kc_primal); rhs pair p: (N, Kc_out, Qf_primal).
        # contracted block Kc must match between lhs.primal and rhs.out.
        ((2, 1, 1), (2, 1, 1), (2, 1, 1), (2, 1, 1)),     # plain diagonals
        ((2, 3, 2), (2, 2, 4), (2, 2, 5), (2, 4, 3)),     # rectangular blocks
        ((3, 2, 2), (2, 2, 2), (3, 2, 3), (2, 2, 2)),     # mixed meta counts
        ((2, 4, 1), (3, 1, 2), (2, 1, 3), (3, 2, 1)),     # block-1 mix
    ],
)
def test_two_bb_pairs(lp1, lp2, rp1, rp2):
    k = jax.random.PRNGKey(hash((lp1, lp2, rp1, rp2)) % (2**31))
    k1, k2 = jax.random.split(k)
    # lhs pairs: (N, Po, Kc)  -> out block Po, primal(contracted) block Kc
    lhs = _build_two_pair_operand(lp1, lp2, k1, con_on_out=False)
    # rhs pairs: (N, Kc, Qf)  -> out(contracted) block Kc, primal block Qf
    rhs = _build_two_pair_operand(rp1, rp2, k2, con_on_out=True)
    # contracted sizes must match: lhs.primal logical == rhs.out logical
    assert lhs.primal_dims[0].logical_size == rhs.out_dims[0].logical_size
    assert lhs.primal_dims[1].logical_size == rhs.out_dims[1].logical_size
    _assert(lhs, rhs)


# --------------------------------------------------------------------------- #
# single B@B pair via the multi kernel (degenerate; the multi path still works)
# --------------------------------------------------------------------------- #
def test_single_bb_pair():
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    # lhs: out B(0,N2,blk3) - primal B(1,N2,blk4)
    lval = _rand((2, 3, 4), k1)
    lo = DiagonalIndex(0, 2, axis=0, other_id=1, block_size=3, block_axis=1)
    lp = DiagonalIndex(1, 2, axis=0, other_id=0, block_size=4, block_axis=2)
    lhs = SparseTensor((lo,), (lp,), lval, check_consistency=False)
    # rhs: out B(0,N2,blk4) - primal B(1,N2,blk5)
    rval = _rand((2, 4, 5), k2)
    ro = DiagonalIndex(0, 2, axis=0, other_id=1, block_size=4, block_axis=1)
    rp = DiagonalIndex(1, 2, axis=0, other_id=0, block_size=5, block_axis=2)
    rhs = SparseTensor((ro,), (rp,), rval, check_consistency=False)
    _assert(lhs, rhs)


# --------------------------------------------------------------------------- #
# mixed: one B@B pair + one plain-Dense CONTRACTED pair (the dense pair fully
# contracts away; the B@B pair survives meta-block-diagonal — verified the
# off-meta result cells are exactly zero).  Out=[B-survivor, lhs free D], etc.
# --------------------------------------------------------------------------- #
def test_mixed_bb_and_dense_contracted():
    k1, k2 = jax.random.split(jax.random.PRNGKey(2))
    N = 2
    Po2, Kc2, Qf2 = 2, 2, 3
    Po_d, Kd = 3, 4
    # lhs: out=[B(0,N,Po2), D(1,Po_d free)], primal=[B(2,N,Kc2)contr, D(3,Kd)contr]
    lval = _rand((N, Po2, Kc2, Po_d, Kd), k1)
    lo_b = DiagonalIndex(0, N, axis=0, other_id=2, block_size=Po2, block_axis=1)
    lo_d = DenseIndex(1, Po_d, axis=3)
    lp_b = DiagonalIndex(2, N, axis=0, other_id=0, block_size=Kc2, block_axis=2)
    lp_d = DenseIndex(3, Kd, axis=4)
    lhs = SparseTensor((lo_b, lo_d), (lp_b, lp_d), lval, check_consistency=False)
    # rhs: out=[B(0,N,Kc2)contr, D(1,Kd)contr], primal=[B(2,N,Qf2), D(3,Qprim free)]
    Qprim = 5
    rval = _rand((N, Kc2, Kd, Qf2, Qprim), k2)
    ro_b = DiagonalIndex(0, N, axis=0, other_id=2, block_size=Kc2, block_axis=1)
    ro_d = DenseIndex(1, Kd, axis=2)
    rp_b = DiagonalIndex(2, N, axis=0, other_id=0, block_size=Qf2, block_axis=3)
    rp_d = DenseIndex(3, Qprim, axis=4)
    rhs = SparseTensor((ro_b, ro_d), (rp_b, rp_d), rval, check_consistency=False)
    _assert(lhs, rhs)


# --------------------------------------------------------------------------- #
# mixed: one B@B pair + one D_B (dense<->block) contracted pair.  The D_B pair
# densifies its surviving partner; the B@B pair survives block-diagonal.
# --------------------------------------------------------------------------- #
def test_mixed_bb_and_db_contracted():
    k1, k2 = jax.random.split(jax.random.PRNGKey(8))
    Na, Nb = 2, 2                  # distinct meta axes per pair
    Po, KcB, Qf = 2, 2, 2          # B@B pair blocks
    Kd_blk, Bf_db = 2, 3           # D_B pair: block contracted Kd_blk, free Bf
    # lhs: out=[B(0,Na,Po)], primal=[B(1,Na,KcB)contr (B@B), D(2,Nb*Kd_blk)contr]
    KdD = Nb * Kd_blk
    # lhs.val axes: metaA(0), Po(1), KcB(2), KdD(3)
    lval = _rand((Na, Po, KcB, KdD), k1)
    lo_b = DiagonalIndex(0, Na, axis=0, other_id=1, block_size=Po, block_axis=1)
    lp_b = DiagonalIndex(1, Na, axis=0, other_id=0, block_size=KcB, block_axis=2)
    lp_d = DenseIndex(2, KdD, axis=3)
    lhs = SparseTensor((lo_b,), (lp_b, lp_d), lval, check_consistency=False)
    # rhs: out=[B(0,Na,KcB)contr (B@B), B(1,Nb,Kd_blk)contr (D_B block side)],
    #      primal=[B(2,Na,Qf), B(3,Nb,Bf_db)free (D_B survivor)]
    # rhs.val axes: metaA(0), KcB(1), metaB(2), Kd_blk(3), Qf(4), Bf_db(5)
    rval = _rand((Na, KcB, Nb, Kd_blk, Qf, Bf_db), k2)
    ro_b0 = DiagonalIndex(0, Na, axis=0, other_id=2, block_size=KcB, block_axis=1)
    ro_b1 = DiagonalIndex(1, Nb, axis=2, other_id=3, block_size=Kd_blk, block_axis=3)
    rp_b2 = DiagonalIndex(2, Na, axis=0, other_id=0, block_size=Qf, block_axis=4)
    rp_b3 = DiagonalIndex(3, Nb, axis=2, other_id=1, block_size=Bf_db, block_axis=5)
    rhs = SparseTensor((ro_b0, ro_b1), (rp_b2, rp_b3), rval, check_consistency=False)
    _assert(lhs, rhs)


# --------------------------------------------------------------------------- #
# val=None (all-ones blocks) on both operands
# --------------------------------------------------------------------------- #
def test_val_none():
    # Build concrete all-ones operands and a val=None equivalent; compare kernel
    # outputs (the generic dense() oracle on a val=None diagonal-with-blocks has a
    # pre-existing crash, documented in test_contract_D_multiB::test_val_none).
    lp1, lp2 = (2, 2, 2), (2, 2, 2)
    rp1, rp2 = (2, 2, 2), (2, 2, 2)
    k1, k2 = jax.random.split(jax.random.PRNGKey(3))
    lhs = _build_two_pair_operand(lp1, lp2, k1, con_on_out=False)
    rhs = _build_two_pair_operand(rp1, rp2, k2, con_on_out=True)
    lhs_ones = SparseTensor(lhs.out_dims, lhs.primal_dims, jnp.ones_like(lhs.val))
    rhs_ones = SparseTensor(rhs.out_dims, rhs.primal_dims, jnp.ones_like(rhs.val))
    lhs_none = SparseTensor(lhs.out_dims, lhs.primal_dims, None)
    rhs_none = SparseTensor(rhs.out_dims, rhs.primal_dims, None)

    from graphax.sparse.ops.matmul import _align_contract_dims
    from graphax.sparse.elemental.dispatch import _classify_pair

    pairs = _align_contract_dims(lhs_none.primal_dims, rhs_none.out_dims, embed=True)
    kinds = [_classify_pair(ld, rd) for ld, rd in pairs]
    res = contract_multi_structured(lhs_none, rhs_none, pairs, kinds)
    assert res is not None
    oracle = _oracle(lhs_ones, rhs_ones)
    np.testing.assert_allclose(np.asarray(res.dense()), oracle, atol=ATOL)


# --------------------------------------------------------------------------- #
# scalar_mult folding
# --------------------------------------------------------------------------- #
def test_scalar_mult():
    k1, k2 = jax.random.split(jax.random.PRNGKey(4))
    lhs = _build_two_pair_operand((2, 2, 3), (2, 3, 2), k1, con_on_out=False)
    rhs = _build_two_pair_operand((2, 3, 2), (2, 2, 3), k2, con_on_out=True)
    lhs = lhs.copy(scalar_mult=jnp.array(2.5, dtype=jnp.float32))
    rhs = rhs.copy(scalar_mult=jnp.array(-1.5, dtype=jnp.float32))
    _assert(lhs, rhs)


# --------------------------------------------------------------------------- #
# attention-shaped: per-head block-diagonal Jacobians, two coupled pairs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("H,dk", [(2, 4), (4, 2), (2, 2)])
def test_attention_shaped(H, dk):
    k1, k2 = jax.random.split(jax.random.PRNGKey(100 + H * 10 + dk))
    # two per-head block-diagonal pairs (e.g. dScores/dV and dV/dx couplings)
    lhs = _build_two_pair_operand((H, dk, dk), (H, dk, dk), k1, con_on_out=False)
    rhs = _build_two_pair_operand((H, dk, dk), (H, dk, dk), k2, con_on_out=True)
    _assert(lhs, rhs)


# --------------------------------------------------------------------------- #
# out-of-scope geometry returns None (mismatched meta -> fallback)
# --------------------------------------------------------------------------- #
def test_mismatched_meta_returns_none():
    k1, k2 = jax.random.split(jax.random.PRNGKey(5))
    # lhs pair0 meta 4 vs rhs pair0 meta 2 with matched contracted logical size.
    # lhs.primal[0] logical = 4*Kc ; rhs.out[0] logical = 2*Kc' ; pick to match.
    lhs = _build_two_pair_operand((4, 2, 2), (2, 2, 2), k1, con_on_out=False)
    # rhs out[0] contracted: need logical == lhs.primal[0].logical = 4*2 = 8.
    # rhs pair0 meta 2 -> Kc'=4. partner Qf arbitrary.
    rhs = _build_two_pair_operand((2, 4, 3), (2, 2, 2), k2, con_on_out=True)
    from graphax.sparse.ops.matmul import _align_contract_dims
    from graphax.sparse.elemental.dispatch import _classify_pair

    pairs = _align_contract_dims(lhs.primal_dims, rhs.out_dims, embed=True)
    kinds = [_classify_pair(ld, rd) for ld, rd in pairs]
    res = contract_multi_structured(lhs, rhs, pairs, kinds)
    assert res is None  # cross-meta refinement left to the fallback


# --------------------------------------------------------------------------- #
# fuzz over random aligned-meta multi-pair instances
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(12))
def test_fuzz(seed):
    rng = np.random.default_rng(seed)
    N1 = int(rng.integers(1, 4))
    N2 = int(rng.integers(1, 4))
    Po1, Kc1, Qf1 = (int(rng.integers(1, 4)) for _ in range(3))
    Po2, Kc2, Qf2 = (int(rng.integers(1, 4)) for _ in range(3))
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    lhs = _build_two_pair_operand((N1, Po1, Kc1), (N2, Po2, Kc2), k1, con_on_out=False)
    rhs = _build_two_pair_operand((N1, Kc1, Qf1), (N2, Kc2, Qf2), k2, con_on_out=True)
    _assert(lhs, rhs)
