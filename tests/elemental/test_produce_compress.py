"""Dense-oracle tests for the IMPLICIT (compressed) dim algebra.

Every test asserts the kernels in ``produce_compress`` (``contract_implicit`` /
``elementwise_implicit``) equal the DENSE ORACLE: apply ``Compress`` *densely*
(reduce-then-broadcast on ``.dense()``) and run a plain ``jnp.matmul`` / ``op``.

The implicit operands are built the production way — through
:func:`graphax.sparse.micro_actions.apply_compress` — so the tests pin the real
``axis=None`` / kept-``logical_size`` layout the producer emits.  Covers all six
Compress kinds, single & multi-axis, implicit-contract-with-Dense,
implicit-contract-with-implicit, implicit under add (merge / union) and multiply
(intersection), plus attention- and mlp/conv-shaped instances and a generalize
fuzz over arbitrary combinations.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor
from graphax.sparse.micro_actions import apply_compress, Compress, COMPRESS_KINDS
from graphax.sparse.elemental.produce_compress import (
    contract_implicit,
    elementwise_implicit,
    _is_implicit,
)

jax.config.update("jax_enable_x64", False)

ATOL = 1e-5


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _rand(shape, key):
    return jax.random.normal(key, shape, dtype=jnp.float32)


def _dense(out_ids, out_sizes, primal_ids, primal_sizes, val):
    out = [DenseIndex(i, s, ax) for ax, (i, s) in enumerate(zip(out_ids, out_sizes))]
    off = len(out_sizes)
    primal = [
        DenseIndex(i, s, off + ax)
        for ax, (i, s) in enumerate(zip(primal_ids, primal_sizes))
    ]
    return SparseTensor(tuple(out), tuple(primal), val)


def _diag(out_id, primal_id, N, B_out, B_primal, val, axis=0):
    """A 2-D meta-block-diagonal SparseTensor (out block B_out, primal block
    B_primal); ``val`` shape ``(N, B_out, B_primal)``. Mirrors the sibling
    test builder (canonical val layout, size-1 block axes squeezed)."""
    out = DiagonalIndex(
        out_id, N, axis=axis, other_id=primal_id,
        block_size=B_out if B_out > 1 else None,
        block_axis=axis + 1 if B_out > 1 else None,
    )
    v = val
    if B_out > 1 and B_primal > 1:
        primal = DiagonalIndex(
            primal_id, N, axis=axis, other_id=out_id,
            block_size=B_primal, block_axis=axis + 2,
        )
    elif B_out == 1 and B_primal > 1:
        primal = DiagonalIndex(
            primal_id, N, axis=axis, other_id=out_id,
            block_size=B_primal, block_axis=axis + 1,
        )
        v = val[:, 0, :]
    elif B_out > 1 and B_primal == 1:
        primal = DiagonalIndex(primal_id, N, axis=axis, other_id=out_id)
        v = val[:, :, 0]
    else:
        primal = DiagonalIndex(primal_id, N, axis=axis, other_id=out_id)
        v = val[:, 0, 0]
    return SparseTensor((out,), (primal,), v)


# --------------------------------------------------------------------------- #
# Dense oracles
# --------------------------------------------------------------------------- #
def _contract_oracle(lhs, rhs):
    """Plain dense vertex-elim contraction over the single contracted pair:
    lhs's trailing primal axis against rhs's leading out axis."""
    ld = lhs.dense()
    rd = rhs.dense()
    n_rhs_out = len(rhs.out_dims)
    if n_rhs_out == 1:
        return jnp.tensordot(ld, rd, axes=([ld.ndim - 1], [0]))
    return jnp.tensordot(
        ld, rd,
        axes=(list(range(ld.ndim - n_rhs_out, ld.ndim)), list(range(n_rhs_out))),
    )


def _assert_contract(lhs, rhs):
    # Sanity: the kernel really is being exercised on an implicit pair.
    lc = lhs.primal_dims[-1]
    rc = rhs.out_dims[0]
    assert _is_implicit(lc) or _is_implicit(rc), "test did not build an implicit pair"
    res = contract_implicit(lhs, rhs)
    oracle = _contract_oracle(lhs, rhs)
    assert res.dense().shape == oracle.shape, (res.dense().shape, oracle.shape)
    np.testing.assert_allclose(np.asarray(res.dense()), np.asarray(oracle), atol=ATOL)
    return res


def _assert_elementwise(lhs, rhs, op, is_intersection=False):
    res = elementwise_implicit(lhs, rhs, op, is_intersection=is_intersection)
    oracle = op(lhs.dense(), rhs.dense())
    assert res.dense().shape == oracle.shape, (res.dense().shape, oracle.shape)
    np.testing.assert_allclose(np.asarray(res.dense()), np.asarray(oracle), atol=ATOL)
    return res


# --------------------------------------------------------------------------- #
# Implicit producers (built through apply_compress — the production path)
# --------------------------------------------------------------------------- #
def _compress_primal(P, N, key, kind):
    """A dense (P, N) tensor whose PRIMAL (N) axis is compressed to implicit."""
    st = _dense([0], [P], [1], [N], _rand((P, N), key))
    c = apply_compress(st, Compress(axes=(1,), kind=kind))
    assert _is_implicit(c.primal_dims[-1])
    return c


def _compress_out(N, Q, key, kind):
    """A dense (N, Q) tensor whose OUT (N) axis is compressed to implicit."""
    st = _dense([0], [N], [1], [Q], _rand((N, Q), key))
    c = apply_compress(st, Compress(axes=(0,), kind=kind))
    assert _is_implicit(c.out_dims[0])
    return c


# --------------------------------------------------------------------------- #
# 1. Each Compress kind — implicit @ Dense
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", COMPRESS_KINDS)
def test_kind_implicit_at_dense(kind):
    k = jax.random.PRNGKey(hash(("k", kind)) % (2**31))
    k1, k2 = jax.random.split(k)
    P, N, Q = 3, 5, 4
    lhs = _compress_primal(P, N, k1, kind)               # (P | N-implicit)
    rhs = _dense([0], [N], [1], [Q], _rand((N, Q), k2))  # (N | Q)
    _assert_contract(lhs, rhs)


@pytest.mark.parametrize("kind", COMPRESS_KINDS)
def test_kind_dense_at_implicit(kind):
    k = jax.random.PRNGKey(hash(("kd", kind)) % (2**31))
    k1, k2 = jax.random.split(k)
    P, N, Q = 3, 5, 4
    lhs = _dense([0], [P], [1], [N], _rand((P, N), k1))  # (P | N)
    rhs = _compress_out(N, Q, k2, kind)                  # (N-implicit | Q)
    _assert_contract(lhs, rhs)


# --------------------------------------------------------------------------- #
# 2. Multi-axis Compress (several free axes dropped at once)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["mean", "abs_max", "median"])
def test_multi_axis_compress_contract(kind):
    # A (A, P, N) tensor: out=(A,P), primal=N. Compress out-axis A AND primal N.
    # Contract the implicit primal N against a dense (N, Q).
    k = jax.random.PRNGKey(hash(("ma", kind)) % (2**31))
    k1, k2 = jax.random.split(k)
    A, P, N, Q = 2, 3, 4, 5
    st = SparseTensor(
        (DenseIndex(0, A, 0), DenseIndex(1, P, 1)),
        (DenseIndex(2, N, 2),),
        _rand((A, P, N), k1),
    )
    lhs = apply_compress(st, Compress(axes=(0, 2), kind=kind))
    # A is now implicit (out), N is implicit (primal). Contract N.
    assert _is_implicit(lhs.primal_dims[-1])
    assert _is_implicit(lhs.out_dims[0])
    rhs = _dense([0], [N], [1], [Q], _rand((N, Q), k2))
    _assert_contract(lhs, rhs)


# --------------------------------------------------------------------------- #
# 3. Implicit contracts with another implicit  (c_l * c_r * N)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["mean", "max", "abs_min"])
def test_implicit_at_implicit(kind):
    k = jax.random.PRNGKey(hash(("ii", kind)) % (2**31))
    k1, k2 = jax.random.split(k)
    P, N, Q = 3, 6, 4
    lhs = _compress_primal(P, N, k1, kind)   # (P | N-implicit)
    rhs = _compress_out(N, Q, k2, kind)      # (N-implicit | Q)
    _assert_contract(lhs, rhs)


# --------------------------------------------------------------------------- #
# 4. Implicit under elementwise add (union/merge) and multiply (intersection)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", COMPRESS_KINDS)
def test_implicit_add_dense(kind):
    # Both operands share logical shape (P, N); lhs has N implicit, rhs full.
    k = jax.random.PRNGKey(hash(("add", kind)) % (2**31))
    k1, k2 = jax.random.split(k)
    P, N = 4, 5
    lhs = _compress_primal(P, N, k1, kind)               # (P | N-implicit)
    rhs = _dense([0], [P], [1], [N], _rand((P, N), k2))  # (P | N)
    _assert_elementwise(lhs, rhs, jnp.add, is_intersection=False)


@pytest.mark.parametrize("kind", ["mean", "abs_max"])
def test_implicit_mul_dense(kind):
    k = jax.random.PRNGKey(hash(("mul", kind)) % (2**31))
    k1, k2 = jax.random.split(k)
    P, N = 4, 5
    lhs = _compress_primal(P, N, k1, kind)
    rhs = _dense([0], [P], [1], [N], _rand((P, N), k2))
    _assert_elementwise(lhs, rhs, jnp.multiply, is_intersection=True)


def test_implicit_add_implicit():
    # Both operands have the SAME axis implicit — both constants broadcast.
    k = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(k)
    P, N = 4, 5
    lhs = _compress_primal(P, N, k1, "mean")
    rhs = _compress_primal(P, N, k2, "max")
    _assert_elementwise(lhs, rhs, jnp.add, is_intersection=False)


# --------------------------------------------------------------------------- #
# 5. Implicit contracted against a BLOCK-DIAGONAL partner (impl @ B / B @ impl)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("N,B_c,B_f", [(3, 2, 2), (4, 2, 3), (2, 3, 2)])
def test_implicit_at_block_diagonal(N, B_c, B_f):
    # lhs implicit primal of logical size K=N*B_c; rhs block-diagonal with
    # contracted out-side (N, B_c) coupled to free primal (N, B_f).
    k = jax.random.PRNGKey(hash(("ib", N, B_c, B_f)) % (2**31))
    k1, k2 = jax.random.split(k)
    K = N * B_c
    P = 3
    lhs = _compress_primal(P, K, k1, "mean")          # (P | K-implicit)
    rhs = _diag(0, 1, N, B_c, B_f, _rand((N, B_c, B_f), k2))
    _assert_contract(lhs, rhs)


@pytest.mark.parametrize("N,B_f,B_c", [(3, 2, 2), (4, 3, 2), (2, 2, 3)])
def test_block_diagonal_at_implicit(N, B_f, B_c):
    # lhs block-diagonal (free out N*B_f, contracted primal N*B_c); rhs implicit
    # out of logical size K=N*B_c.
    k = jax.random.PRNGKey(hash(("bi", N, B_f, B_c)) % (2**31))
    k1, k2 = jax.random.split(k)
    K = N * B_c
    Q = 4
    lhs = _diag(0, 1, N, B_f, B_c, _rand((N, B_f, B_c), k1))
    rhs = _compress_out(K, Q, k2, "mean")             # (K-implicit | Q)
    _assert_contract(lhs, rhs)


# --------------------------------------------------------------------------- #
# 6. scalar_mult folding
# --------------------------------------------------------------------------- #
def test_scalar_mult_fold_contract():
    k = jax.random.PRNGKey(11)
    k1, k2 = jax.random.split(k)
    P, N, Q = 3, 5, 4
    lhs = _compress_primal(P, N, k1, "mean").copy(scalar_mult=jnp.array(2.5))
    rhs = _dense([0], [N], [1], [Q], _rand((N, Q), k2)).copy(
        scalar_mult=jnp.array(-1.5)
    )
    _assert_contract(lhs, rhs)


# --------------------------------------------------------------------------- #
# 7. Result closure: implicit @ Dense -> Dense
# --------------------------------------------------------------------------- #
def test_result_is_dense():
    k = jax.random.PRNGKey(12)
    k1, k2 = jax.random.split(k)
    P, N, Q = 4, 5, 3
    lhs = _compress_primal(P, N, k1, "mean")
    rhs = _dense([0], [N], [1], [Q], _rand((N, Q), k2))
    res = contract_implicit(lhs, rhs)
    assert all(not d.is_sparse for d in res.dims), res.dims
    assert res.out_shape == (P,)
    assert res.primal_shape == (Q,)


# --------------------------------------------------------------------------- #
# 8. Mis-dispatch guards
# --------------------------------------------------------------------------- #
def test_rejects_no_implicit():
    k = jax.random.PRNGKey(13)
    k1, k2 = jax.random.split(k)
    lhs = _dense([0], [3], [1], [4], _rand((3, 4), k1))
    rhs = _dense([0], [4], [1], [5], _rand((4, 5), k2))
    with pytest.raises(ValueError):
        contract_implicit(lhs, rhs)


def test_elementwise_rejects_no_implicit():
    k = jax.random.PRNGKey(14)
    k1, k2 = jax.random.split(k)
    lhs = _dense([0], [3], [1], [4], _rand((3, 4), k1))
    rhs = _dense([0], [3], [1], [4], _rand((3, 4), k2))
    with pytest.raises(ValueError):
        elementwise_implicit(lhs, rhs, jnp.add)


# --------------------------------------------------------------------------- #
# 9. Attention-shaped (S, H, dk) — per-head, with a compressed axis
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("S,H,dk", [(4, 2, 4), (8, 2, 4), (8, 2, 8)])
def test_attention_shaped_implicit_contract(S, H, dk):
    # Attention-flavoured: a dense (S, H*dk) activation slab whose feature axis
    # (H*dk) is compressed to an implicit constant per row, contracted against a
    # dense (H*dk, S) factor.
    k = jax.random.PRNGKey(hash((S, H, dk)) % (2**31))
    k1, k2 = jax.random.split(k)
    K = H * dk
    lhs = _compress_primal(S, K, k1, "mean")               # (S | K-implicit)
    rhs = _dense([0], [K], [1], [S], _rand((K, S), k2))     # (K | S)
    _assert_contract(lhs, rhs)


@pytest.mark.parametrize("S,H,dk", [(4, 2, 4), (8, 2, 4)])
def test_attention_shaped_implicit_add(S, H, dk):
    k = jax.random.PRNGKey(hash((S, H, dk, "add")) % (2**31))
    k1, k2 = jax.random.split(k)
    K = H * dk
    lhs = _compress_primal(S, K, k1, "abs_max")            # (S | K-implicit)
    rhs = _dense([0], [S], [1], [K], _rand((S, K), k2))    # (S | K)
    _assert_elementwise(lhs, rhs, jnp.add)


# --------------------------------------------------------------------------- #
# 10. Conv / MLP shape (batch B_in dense rows, compressed feature)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "B_in,N,Q", list(itertools.product([1, 8], [4, 6], [3, 5]))
)
def test_mlp_shaped_implicit_contract(B_in, N, Q):
    k = jax.random.PRNGKey(hash((B_in, N, Q)) % (2**31))
    k1, k2 = jax.random.split(k)
    lhs = _compress_primal(B_in, N, k1, "mean")
    rhs = _dense([0], [N], [1], [Q], _rand((N, Q), k2))
    _assert_contract(lhs, rhs)


# --------------------------------------------------------------------------- #
# 11. Generalize: fuzz over arbitrary combos (kind, shapes, side)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(16))
def test_fuzz_implicit_contract(seed):
    rng = np.random.default_rng(seed)
    kind = COMPRESS_KINDS[rng.integers(0, len(COMPRESS_KINDS))]
    P = int(rng.integers(1, 6))
    N = int(rng.integers(2, 8))
    Q = int(rng.integers(1, 6))
    left = bool(rng.integers(0, 2))
    k = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(k)
    if left:
        lhs = _compress_primal(P, N, k1, kind)
        rhs = _dense([0], [N], [1], [Q], _rand((N, Q), k2))
    else:
        lhs = _dense([0], [P], [1], [N], _rand((P, N), k1))
        rhs = _compress_out(N, Q, k2, kind)
    _assert_contract(lhs, rhs)


@pytest.mark.parametrize("seed", range(12))
def test_fuzz_implicit_elementwise(seed):
    rng = np.random.default_rng(1000 + seed)
    kind = COMPRESS_KINDS[rng.integers(0, len(COMPRESS_KINDS))]
    P = int(rng.integers(1, 6))
    N = int(rng.integers(2, 8))
    op, is_int = (jnp.add, False) if rng.integers(0, 2) else (jnp.multiply, True)
    k = jax.random.PRNGKey(1000 + seed)
    k1, k2 = jax.random.split(k)
    lhs = _compress_primal(P, N, k1, kind)
    rhs = _dense([0], [P], [1], [N], _rand((P, N), k2))
    _assert_elementwise(lhs, rhs, op, is_intersection=is_int)
