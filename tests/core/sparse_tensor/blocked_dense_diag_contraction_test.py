"""A BLOCKED DENSE contracted dim against a DIAGONAL partner (ticket dsnn-tsl).

The reproduction is the two operand pairs of the a9a fuzzer's seeds 4 and 27 on
TransformerLM (job 66611, graphax 4701d950), read off the failing
``matmul._execute_tiled`` frame, plus the two minimal pairs that carry the same
blocked dim. Before the fix each of them raised ``NominalOrderViolation``: a
surviving dim came out at the contracted partner's block COUNT (32 of 128, 8 of
1024) because ``shared_factors`` took the gcd of the two metas and the meta's
grid axis was then squeezed away, so the VALUES were wrong too.

The randomized half contracts the same two forms against the dense contraction
of the two operands' dense forms, in float64, over 64 draws of shape, block
size, orientation and storage -- including the plain dense partner, which the
fix must leave alone.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex, DiagonalIndex, Index
from graphax.sparse.tensor import SparseTensor


# x64 is process-global, so it is set and restored around each test here rather
# than written at import (the reason is in tests/conftest.py).
@pytest.fixture(autouse=True)
def _x64():
    old = bool(getattr(jax.config, "jax_enable_x64", False))
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _rand(shape, k):
    rng = np.random.default_rng(k)
    return jnp.asarray(rng.standard_normal(shape))


def _oracle(lhs, rhs):
    n = len(lhs.primal_dims)
    return jnp.tensordot(
        lhs.dense(), rhs.dense(),
        axes=(list(range(len(lhs.out_dims), len(lhs.out_dims) + n)),
              list(range(n))),
    )


def _assert_exact(lhs, rhs, atol=1e-12):
    res = lhs @ rhs
    assert tuple(int(d.logical_size) for d in res.out_dims) == \
        tuple(int(d.logical_size) for d in lhs.out_dims)
    assert tuple(int(d.logical_size) for d in res.primal_dims) == \
        tuple(int(d.logical_size) for d in rhs.primal_dims)
    got, want = res.dense(), _oracle(lhs, rhs)
    assert tuple(got.shape) == tuple(want.shape)
    err = float(jnp.max(jnp.abs(got - want)))
    assert err <= atol, f"value mismatch: max diff {err}"
    return res


# --- the two measured operand pairs ---------------------------------------
def test_seed4_operands_of_the_a9a_fuzzer():
    # lhs primal dim id=2 is the blocked dense dim a Compress left (4 x 32);
    # rhs out dim id=1 is the diagonal partner whose meta is 128.
    lhs = SparseTensor(
        (DiagonalIndex(0, 32, axis=None, other_id=1),),
        (DiagonalIndex(1, 32, axis=None, other_id=0),
         Index(2, 4, 0, None, 32, None)),
        _rand((4,), 40),
    )
    rhs = SparseTensor(
        (DenseIndex(0, 32, 0), DiagonalIndex(1, 128, axis=None, other_id=3)),
        (DenseIndex(2, 128, 1), DiagonalIndex(3, 128, axis=None, other_id=1)),
        _rand((32, 128), 41),
    )
    assert tuple(lhs.shape) == (32, 32, 128)
    res = _assert_exact(lhs, rhs)
    assert tuple(int(d.logical_size) for d in res.primal_dims) == (128, 128)


def test_seed27_operands_of_the_a9a_fuzzer():
    # rhs out dim id=1 is the blocked dense dim (128 x 8); lhs out dim id=1 is
    # the diagonal partner whose meta is 1024.
    lhs = SparseTensor(
        (DiagonalIndex(0, 32, axis=0, other_id=2),
         DiagonalIndex(1, 1024, axis=1, other_id=3)),
        (DiagonalIndex(2, 32, axis=0, other_id=0),
         DiagonalIndex(3, 1024, axis=1, other_id=1)),
        _rand((32, 1), 42),
    )
    rhs = SparseTensor(
        (DenseIndex(0, 32, None), Index(1, 128, None, None, 8, None)),
        (DenseIndex(2, 128, None),),
        None,
    )
    res = _assert_exact(lhs, rhs)
    assert tuple(int(d.logical_size) for d in res.out_dims) == (32, 1024)


# --- the two minimal pairs ------------------------------------------------
def test_minimal_blocked_rhs_out_against_diagonal_lhs():
    s, b, m = 4, 8, 5
    n = s * b
    lhs = SparseTensor(
        (DiagonalIndex(0, n, axis=0, other_id=1),),
        (DiagonalIndex(1, n, axis=0, other_id=0),),
        _rand((n,), 1),
    )
    rhs = SparseTensor(
        (Index(0, s, 0, None, b, None),),
        (DenseIndex(1, m, 1),),
        _rand((s, m), 2),
    )
    _assert_exact(lhs, rhs)


def test_minimal_blocked_lhs_primal_against_diagonal_rhs():
    s, b, k = 4, 8, 3
    n = s * b
    lhs = SparseTensor(
        (DenseIndex(0, k, 0),),
        (Index(1, s, 1, None, b, None),),
        _rand((k, s), 5),
    )
    rhs = SparseTensor(
        (DiagonalIndex(0, n, axis=0, other_id=1),),
        (DiagonalIndex(1, n, axis=0, other_id=0),),
        _rand((n,), 6),
    )
    _assert_exact(lhs, rhs)


# --- randomized exactness against the dense contraction -------------------
def _draw(rng, orientation, partner, stored):
    s = int(rng.integers(2, 7))
    b = int(rng.integers(2, 7))
    n = s * b
    p = int(rng.integers(1, 5))
    k = int(rng.integers(1, 5))
    seed = int(rng.integers(0, 1 << 30))
    if orientation == "rhs":
        lhs = (
            SparseTensor(
                (DiagonalIndex(0, n, axis=0, other_id=1),),
                (DiagonalIndex(1, n, axis=0, other_id=0),),
                _rand((n,), seed),
            )
            if partner == "diag"
            else SparseTensor(
                (DenseIndex(0, k, 0),),
                (DenseIndex(1, n, 1),),
                _rand((k, n), seed),
            )
        )
        if stored:
            rhs = SparseTensor(
                (Index(0, s, 0, None, b, None),),
                (DenseIndex(1, p, 1),),
                _rand((s, p), seed + 1),
            )
        else:
            rhs = SparseTensor(
                (Index(0, s, None, None, b, None),),
                (DenseIndex(1, p, 0),),
                _rand((p,), seed + 1),
            )
        return lhs, rhs
    if stored:
        lhs = SparseTensor(
            (DenseIndex(0, k, 0),),
            (Index(1, s, 1, None, b, None),),
            _rand((k, s), seed),
        )
    else:
        lhs = SparseTensor(
            (DenseIndex(0, k, 0),),
            (Index(1, s, None, None, b, None),),
            _rand((k,), seed),
        )
    rhs = (
        SparseTensor(
            (DiagonalIndex(0, n, axis=0, other_id=1),),
            (DiagonalIndex(1, n, axis=0, other_id=0),),
            _rand((n,), seed + 1),
        )
        if partner == "diag"
        else SparseTensor(
            (DenseIndex(0, n, 0),),
            (DenseIndex(1, p, 1),),
            _rand((n, p), seed + 1),
        )
    )
    return lhs, rhs


_FAMILIES = [
    (o, q, st)
    for o in ("lhs", "rhs")
    for q in ("diag", "dense")
    for st in (True, False)
]


@pytest.mark.parametrize("draw", range(64))
def test_blocked_dim_contraction_is_exact(draw):
    rng = np.random.default_rng(9000 + draw)
    orientation, partner, stored = _FAMILIES[draw % len(_FAMILIES)]
    lhs, rhs = _draw(rng, orientation, partner, stored)
    _assert_exact(lhs, rhs)
