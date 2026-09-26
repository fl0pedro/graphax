"""The sublist einsum is a drop-in for ``dot_general`` (ticket dsnn-3qm.72).

Owner ruling 2026-09-08: the contraction engine keeps its dimension numbers and
emits an einsum over INTEGER sublists instead of a dot, so XLA keeps the choice
of lowering. These tests pin that the two forms agree exactly -- same values,
same output axis order -- because the sublists are DERIVED from the very
dimension numbers the dot would have taken.

The dot is no longer emitted anywhere in graphax, so ``_reference_dot`` below
is the reference, written out here. It restates ``_emit_einsum``'s dtype rule:
a bf16 x bf16 contraction is a plain bf16 dot (owner ruling 2026-09-26).
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.ops.matmul import (_dims_to_sublists, _frame_sublists,
                                       _gx_einsum)


def _reference_dot(a, b, dims):
    """``lax.dot_general``, the form ``_emit_einsum`` states as an einsum."""
    return jax.lax.dot_general(a, b, dims)

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
    want = np.asarray(_reference_dot(a, b, dims), np.float64)
    got = np.asarray(_gx_einsum(a, b, dims), np.float64)
    assert got.shape == want.shape, f"{name}: {got.shape} vs {want.shape}"
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name,ls,rs,dims", CASES, ids=[c[0] for c in CASES])
def test_the_same_under_jit(name, ls, rs, dims):
    a, b = _n(ls, 2), _n(rs, 3)
    f = jax.jit(lambda x, y: _gx_einsum(x, y, dims))
    g = jax.jit(lambda x, y: _reference_dot(x, y, dims))
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


def test_a_bf16_pair_is_a_plain_bf16_contraction():
    """Only Quant makes bf16 operands. The pair is a plain bf16 operation: bf16
    in, bf16 out, and XLA's own kernel forms the sums (owner ruling 2026-09-26,
    dsnn-dfw.273). No f32 result is cast back by hand."""
    a = _n((8, 8), 4).astype(jnp.bfloat16)
    b = _n((8, 8), 5).astype(jnp.bfloat16)
    dims = (((1,), (0,)), ((), ()))
    got, want = _gx_einsum(a, b, dims), _reference_dot(a, b, dims)
    assert jnp.dtype(got.dtype) == jnp.dtype(jnp.bfloat16)
    jaxpr = jax.make_jaxpr(lambda x, y: _gx_einsum(x, y, dims))(a, b).jaxpr
    for e in jaxpr.eqns:
        assert all(jnp.dtype(v.aval.dtype) == jnp.dtype(jnp.bfloat16) for v in e.outvars), e
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))


# --- The frame's own sublists (ticket dsnn-3qm.72) -------------------------
# ``_frame_sublists`` builds the labels from the prepared operands' SLOT
# layout, not from dimension numbers. The slot layout is three slots per pair:
#   lhs  [meta_i] + [lhs block_i] + [split_i] + lhs leftover
#   rhs  [meta_i] + [split_i] + [rhs shared block_i] + rhs leftover


class _P:
    """The one field ``_frame_sublists`` reads off a Pair."""

    def __init__(self, pairing_type):
        self.pairing_type = pairing_type


def test_one_contracting_pair_is_a_plain_matmul():
    pairs = [_P("contract")]
    lhs_sub, rhs_sub, out_sub = _frame_sublists(
        1, pairs, (2, 3, 4), (2, 4, 5), 0, 0)
    # meta shared, lhs block free, split contracted (absent from the output),
    # rhs shared block free.
    assert lhs_sub == [0, 2, 1]
    assert rhs_sub == [0, 1, 3]
    assert out_sub == [0, 2, 3]


def test_a_riding_through_pair_keeps_its_split_in_the_output():
    pairs = [_P("batch_out")]
    _, _, out_sub = _frame_sublists(1, pairs, (2, 3, 4), (2, 4, 5), 0, 0)
    assert out_sub == [0, 1, 2, 3]


def test_an_axis_only_one_side_stores_gets_a_private_label():
    """The lhs stores nothing along the meta axis. Its label must not be the
    rhs's, or the two extents would have to be lined up by a broadcast."""
    pairs = [_P("contract")]
    lhs_sub, rhs_sub, out_sub = _frame_sublists(
        1, pairs, (1, 3, 4), (6, 4, 5), 0, 0)
    assert lhs_sub[0] not in rhs_sub          # private
    assert lhs_sub[0] not in out_sub          # therefore summed, free at 1
    assert rhs_sub[0] == 0 and out_sub[0] == 0  # the rhs carries the meta


def test_a_contracted_axis_only_one_side_stores_becomes_a_sum():
    """The rhs stores nothing along the contracted axis: the lhs's label is
    absent from the output, so einsum sums the lhs over it."""
    pairs = [_P("contract")]
    lhs_sub, rhs_sub, out_sub = _frame_sublists(
        1, pairs, (2, 3, 4), (2, 1, 5), 0, 0)
    assert lhs_sub[2] == 1 and lhs_sub[2] not in out_sub
    assert rhs_sub[1] not in lhs_sub and rhs_sub[1] not in out_sub


def test_a_private_label_gives_the_same_number_as_the_broadcast():
    """The whole point: dropping a size-1 axis and broadcasting it up to its
    partner are the same product. Checked on both roles."""
    for lhs_shape, rhs_shape, grown in (
        ((1, 3, 4), (6, 4, 5), "lhs"),   # meta stored only by the rhs
        ((2, 3, 4), (2, 1, 5), "rhs"),   # contracted axis stored only by the lhs
    ):
        pairs = [_P("contract")]
        a, b = _n(lhs_shape, 6), _n(rhs_shape, 7)
        l_sub, r_sub, o_sub = _frame_sublists(
            1, pairs, lhs_shape, rhs_shape, 0, 0)
        got = np.asarray(jnp.einsum(a, l_sub, b, r_sub, o_sub), np.float64)
        # the broadcast form: line the two up, then contract with equal labels
        n_meta = max(lhs_shape[0], rhs_shape[0])
        n_split = max(lhs_shape[2], rhs_shape[1])
        ab = np.broadcast_to(np.asarray(a, np.float64),
                             (n_meta, lhs_shape[1], n_split))
        bb = np.broadcast_to(np.asarray(b, np.float64),
                             (n_meta, n_split, rhs_shape[2]))
        want = np.einsum("mbs,msf->mbf", ab, bb)
        assert got.shape == want.shape, grown
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
