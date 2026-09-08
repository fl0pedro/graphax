"""The sublist einsum is a drop-in for ``dot_general`` (ticket dsnn-3qm.72).

Owner ruling 2026-09-08: the tiled path keeps its dimension numbers and emits
an einsum over INTEGER sublists instead of a dot, so XLA keeps the choice of
lowering. These tests pin that the two forms agree exactly -- same values, same
output axis order -- because the sublists are DERIVED from the very dimension
numbers the dot would have taken.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.ops.matmul import (_dims_to_sublists, _gx_dot_general,
                                       _gx_einsum)

KEY = jax.random.PRNGKey(5)


def _n(shape, i):
    return jax.random.normal(jax.random.split(KEY, 12)[i], shape)


CASES = [
    # name, lhs shape, rhs shape, ((lhs_contract, rhs_contract), (lhs_b, rhs_b))
    ("plain_matmul", (4, 3), (3, 5), (((1,), (0,)), ((), ()))),
    ("batched", (2, 4, 3), (2, 3, 5), (((2,), (1,)), ((0,), (0,)))),
    ("two_batch", (2, 6, 4, 3), (2, 6, 3, 5), (((3,), (2,)), ((0, 1), (0, 1)))),
    ("two_contract", (4, 3, 2), (3, 2, 5), (((1, 2), (0, 1)), ((), ()))),
    ("outer_product", (4,), (5,), (((), ()), ((), ()))),
    ("batch_and_two_contract", (2, 4, 3, 6), (2, 3, 6, 5),
     (((2, 3), (1, 2)), ((0,), (0,)))),
    ("contract_all_of_lhs", (3, 2), (3, 2, 5), (((0, 1), (0, 1)), ((), ()))),
    ("rank1_by_matrix", (3,), (3, 5), (((0,), (0,)), ((), ()))),
]


@pytest.mark.parametrize("name,ls,rs,dims", CASES, ids=[c[0] for c in CASES])
def test_einsum_equals_dot_general(name, ls, rs, dims):
    a, b = _n(ls, 0), _n(rs, 1)
    want = np.asarray(_gx_dot_general(a, b, dims), np.float64)
    got = np.asarray(_gx_einsum(a, b, dims), np.float64)
    assert got.shape == want.shape, f"{name}: {got.shape} vs {want.shape}"
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name,ls,rs,dims", CASES, ids=[c[0] for c in CASES])
def test_the_same_under_jit(name, ls, rs, dims):
    a, b = _n(ls, 2), _n(rs, 3)
    f = jax.jit(lambda x, y: _gx_einsum(x, y, dims))
    g = jax.jit(lambda x, y: _gx_dot_general(x, y, dims))
    np.testing.assert_allclose(np.asarray(f(a, b), np.float64),
                               np.asarray(g(a, b), np.float64),
                               rtol=1e-6, atol=1e-6)


def test_the_output_axis_order_is_dot_generals():
    """batch, then the lhs's kept axes in order, then the rhs's kept axes.
    Downstream index arithmetic on the result depends on this."""
    dims = (((2,), (1,)), ((0,), (0,)))
    lhs_sub, rhs_sub, out_sub = _dims_to_sublists(3, 3, dims)
    assert lhs_sub == [0, 2, 1]
    assert rhs_sub == [0, 1, 3]
    assert out_sub == [0, 2, 3]


def test_labels_are_integers_so_there_is_no_alphabet_cap():
    """The letter form caps at 52 indices. The sublist form does not, and a
    deep Jacobian chain does reach past a handful of axes."""
    n = 30
    dims = (((n - 1,), (0,)), (tuple(range(n - 1)), tuple(range(1, n))))
    lhs_sub, rhs_sub, out_sub = _dims_to_sublists(n, n, dims)
    assert all(isinstance(x, int) for x in lhs_sub + rhs_sub + out_sub)
    assert max(out_sub) >= 26


def test_a_bf16_pair_accumulates_like_the_dot():
    """Only Quant makes bf16 operands. The emission must not change the
    accumulation, or a switch of emission would move a value."""
    a = _n((8, 8), 4).astype(jnp.bfloat16)
    b = _n((8, 8), 5).astype(jnp.bfloat16)
    dims = (((1,), (0,)), ((), ()))
    got, want = _gx_einsum(a, b, dims), _gx_dot_general(a, b, dims)
    assert jnp.dtype(got.dtype) == jnp.dtype(want.dtype)
    np.testing.assert_allclose(np.asarray(got, np.float64),
                               np.asarray(want, np.float64), rtol=2e-2, atol=2e-2)
