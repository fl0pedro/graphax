"""Dense-oracle tests for the ``contract_B_B`` elemental kernel.

The correctness gate (per the design): the kernel result, densified, MUST equal
contracting the two operands DENSELY — materialize each via ``.dense()``, run a
plain ``jnp`` matmul, compare to ~1e-5.

Covered:
  * aligned geometry (``Na == Nb``) — randomized small + rectangular blocks.
  * misaligned geometry (``Na != Nb``, shared logical contracted size).
  * ``G == 1`` collapse to a DenseIndex pair.
  * scalar_mult folding and ``val is None`` (all-ones) operands.
  * implementation-defined val axis order (indirection via .axis/.block_axis).
  * search-shaped instances: attention (S, H, dk) and an MLP/conv block shape.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.elemental import contract_B_B
from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor


def _make_B(seed, N, Brow, Bcol, *, val=None, scalar_mult=1.0,
            meta_axis=0, row_axis=1, col_axis=2):
    """Build a block-diagonal SparseTensor with meta count N, out-block Brow,
    primal-block Bcol. ``val`` laid out per the requested axis order so we test
    .axis/.block_axis indirection. Returns (tensor, dense_array)."""
    rng = np.random.default_rng(seed)
    if val is None:
        buf = rng.standard_normal((N, Brow, Bcol)).astype(np.float32)
        # place into the requested physical axis order
        order = {meta_axis: N, row_axis: Brow, col_axis: Bcol}
        phys_shape = [order[a] for a in sorted(order)]
        # inverse perm: canonical (meta,row,col) -> physical
        canon_to_phys = [meta_axis, row_axis, col_axis]
        phys = np.transpose(buf, np.argsort(canon_to_phys))
        valarr = jnp.asarray(phys)
    else:
        # val=None: the structure is all-ones; the dims carry NO physical axes
        # (there is no buffer to index into) — the kernel densifies to ones.
        valarr = None
        out = DiagonalIndex(0, N, axis=None, other_id=1,
                            block_size=Brow, block_axis=None)
        primal = DiagonalIndex(1, N, axis=None, other_id=0,
                               block_size=Bcol, block_axis=None)
        st = SparseTensor((out,), (primal,), valarr,
                          scalar_mult=jnp.asarray(scalar_mult, dtype=jnp.float32))
        return st, st.dense()

    out = DiagonalIndex(0, N, axis=meta_axis, other_id=1,
                        block_size=Brow, block_axis=row_axis)
    primal = DiagonalIndex(1, N, axis=meta_axis, other_id=0,
                           block_size=Bcol, block_axis=col_axis)
    st = SparseTensor((out,), (primal,), valarr,
                      scalar_mult=jnp.asarray(scalar_mult, dtype=jnp.float32))
    return st, st.dense()


def _check(lhs, rhs):
    """Run kernel vs dense oracle; assert match and that result is closed {D,B}."""
    lc = lhs.primal_dims[0]
    rc = rhs.out_dims[0]
    res = contract_B_B(lhs, rhs, lc, rc)
    got = np.asarray(res.dense())
    oracle = np.asarray(lhs.dense() @ rhs.dense())
    assert got.shape == oracle.shape, (got.shape, oracle.shape)
    assert np.allclose(got, oracle, atol=1e-5), np.abs(got - oracle).max()
    # closure: every result dim is Dense or (non-compressed) Diagonal
    for d in res.dims:
        assert not d.is_compressed
    return res


# --- Aligned geometry (Na == Nb) ------------------------------------------
@pytest.mark.parametrize("N,P,K,Q", [
    (1, 3, 4, 5),
    (2, 3, 4, 5),
    (3, 2, 2, 2),
    (4, 1, 6, 1),   # vector-ish blocks
    (2, 5, 3, 7),   # rectangular
    (5, 4, 4, 4),
])
def test_aligned(N, P, K, Q):
    lhs, _ = _make_B(1, N, P, K)
    rhs, _ = _make_B(2, N, K, Q)
    res = _check(lhs, rhs)
    # aligned, N>1 stays Diagonal; N==1 collapses to Dense
    if N == 1:
        assert all(not d.is_sparse for d in res.dims)
    else:
        assert all(d.is_sparse for d in res.dims)


# --- Misaligned geometry (Na != Nb, shared logical K) ---------------------
@pytest.mark.parametrize("Na,P,Ka, Nb,Q", [
    (2, 3, 4, 4, 5),   # K=8, G=2, ra=1, rb=2
    (4, 2, 2, 2, 3),   # K=8, G=2, ra=2, rb=1
    (6, 1, 2, 4, 5),   # K=12, G=2, ra=3, rb=2
    (3, 2, 4, 6, 1),   # K=12, G=3, ra=1, rb=2
    (2, 4, 6, 3, 2),   # K=12, G=1 -> dense collapse
])
def test_misaligned(Na, P, Ka, Nb, Q):
    K = Na * Ka
    assert K % Nb == 0, "test setup: shared logical size must divide"
    Kb = K // Nb
    lhs, _ = _make_B(3, Na, P, Ka)
    rhs, _ = _make_B(4, Nb, Kb, Q)
    res = _check(lhs, rhs)
    G = np.gcd(Na, Nb)
    if G == 1:
        assert all(not d.is_sparse for d in res.dims)
    else:
        assert all(d.is_sparse for d in res.dims)
        assert res.out_dims[0].size == G


# --- scalar_mult folding --------------------------------------------------
def test_scalar_mult():
    lhs, _ = _make_B(5, 3, 2, 4, scalar_mult=2.5)
    rhs, _ = _make_B(6, 3, 4, 2, scalar_mult=-1.5)
    _check(lhs, rhs)


# --- val=None (all-ones structure) ----------------------------------------
def test_val_none():
    lhs, _ = _make_B(0, 2, 3, 4, val="ones")
    rhs, _ = _make_B(0, 2, 4, 5, val="ones")
    _check(lhs, rhs)
    # mixed: one None, one concrete
    lhs2, _ = _make_B(7, 2, 3, 4)
    rhs2, _ = _make_B(0, 2, 4, 5, val="ones")
    _check(lhs2, rhs2)


# --- implementation-defined val axis order --------------------------------
@pytest.mark.parametrize("perm", list(itertools.permutations([0, 1, 2])))
def test_val_axis_order(perm):
    m, r, c = perm
    lhs, _ = _make_B(8, 3, 2, 4, meta_axis=m, row_axis=r, col_axis=c)
    m2, r2, c2 = list(itertools.permutations([0, 1, 2]))[5]
    rhs, _ = _make_B(9, 3, 4, 5, meta_axis=m2, row_axis=r2, col_axis=c2)
    _check(lhs, rhs)


# --- search-shaped: attention (per-head block-diagonal Jacobians) ---------
@pytest.mark.parametrize("S,H,dk", [(4, 2, 4), (8, 2, 4), (8, 4, 8), (16, 2, 4)])
def test_attention_shape(S, H, dk):
    # H heads = meta count; per-head block dk x dk (e.g. dV/dscores style factor).
    lhs, _ = _make_B(100 + S, H, dk, dk)
    rhs, _ = _make_B(200 + S, H, dk, dk)
    _check(lhs, rhs)


# --- search-shaped: mlp/conv block (channel-group block-diagonal) ---------
@pytest.mark.parametrize("groups,cin,cout", [(4, 8, 8), (2, 6, 10), (8, 4, 4)])
def test_mlp_grouped_shape(groups, cin, cout):
    # grouped linear: groups meta blocks, each (cout x cin) @ (cin x cout).
    lhs, _ = _make_B(300, groups, cout, cin)
    rhs, _ = _make_B(400, groups, cin, cout)
    _check(lhs, rhs)


# --- generalize: arbitrary combinations -----------------------------------
@pytest.mark.parametrize("Na,Nb", [(2, 2), (2, 4), (4, 2), (3, 6), (6, 3), (3, 5)])
@pytest.mark.parametrize("P,Q", [(1, 1), (2, 3), (3, 2)])
def test_arbitrary_combinations(Na, Nb, P, Q):
    # pick block sizes so Na*Ka == Nb*Kb with a clean shared K
    base = np.lcm(Na, Nb)
    Ka = (base // Na)
    Kb = (base // Nb)
    lhs, _ = _make_B(11 + Na * 31 + Nb, Na, P, Ka)
    rhs, _ = _make_B(13 + Na + Nb * 17, Nb, Kb, Q)
    _check(lhs, rhs)
