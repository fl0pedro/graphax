"""The frame classifies every implicit-axis case and counts the multiply grid
(ticket dsnn-3qm.72).

Phases 1 to 3 touch no array, so these tests build operands and assert on the
FRAME, never on a result. That is the point of the split: a storage claim is
checkable without running a contraction.

The eleven cases are CONTEXT.md's vocabulary, and the grid numbers are the ones
finding 65 derived in closed form.
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from graphax.sparse.contract import (AxisCase, EmissionKind, build_frame,
                                     decide_emission)
from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor

M, P, K, Q = 4, 3, 2, 5


def _cases(frame):
    return [a.case for a in frame.axes]


def test_plain_dense_contraction():
    """(a, b) @ (b, c): one dot axis, two carried. Grid is a*b*c."""
    lhs = SparseTensor((DenseIndex(0, 4, 0),), (DenseIndex(1, 3, 1),), jnp.ones((4, 3)))
    rhs = SparseTensor((DenseIndex(0, 3, 0),), (DenseIndex(1, 5, 1),), jnp.ones((3, 5)))
    f = build_frame(lhs, rhs)
    assert _cases(f) == [AxisCase.KEEP_ON_STORER, AxisCase.DOT,
                         AxisCase.KEEP_ON_STORER]
    assert f.multiply_grid == 60
    assert f.out_arity == 1


def test_a_diagonal_pair_counts_its_meta_once():
    """A block-diagonal matmul with meta M and blocks P, K, Q multiplies
    M*P*K*Q times. The LOGICAL extents multiply to (M*P)*(M*K)*(M*Q), which
    over-counts the meta twice, and the budget rule reads this number."""
    lhs = SparseTensor((DiagonalIndex(0, M, 0, 1, P, 1),),
                       (DiagonalIndex(1, M, 0, 0, K, 2),), jnp.ones((M, P, K)))
    rhs = SparseTensor((DiagonalIndex(0, M, None, 1, K, 0),),
                       (DiagonalIndex(1, M, None, 0, Q, 1),), jnp.ones((K, Q)))
    f = build_frame(lhs, rhs)
    assert f.multiply_grid == M * P * K * Q == 120
    assert f.multiply_grid != 12 * 8 * 20
    assert len({a.group for a in f.axes}) == 1, "one diagonal threads all three"


def test_single_implicit_sparse_axis_keeps_the_pair_on_the_storer():
    """The meta of a diagonal pair is stored by the lhs only."""
    lhs = SparseTensor((DiagonalIndex(0, M, 0, 1, P, 1),),
                       (DiagonalIndex(1, M, 0, 0, K, 2),), jnp.ones((M, P, K)))
    rhs = SparseTensor((DiagonalIndex(0, M, None, 1, K, 0),),
                       (DiagonalIndex(1, M, None, 0, Q, 1),), jnp.ones((K, Q)))
    f = build_frame(lhs, rhs)
    assert f.axes[0].case is AxisCase.KEEP_PAIR
    assert f.axes[2].case is AxisCase.IMPLICIT_PAIR
    assert f.axes[0].implicit_count == 0     # lhs stores it, rhs has no such axis
    assert f.axes[1].implicit_count == 1     # contracted, stored by the lhs only


@pytest.mark.xfail(strict=True, reason=(
    "phase 1 expresses the matmul convention only: lhs.primal meets rhs.out, "
    "positionally. A BATCH axis shared by both operands' out_dims has no way "
    "to say so, and the positional rule pairs it with the contracted axis "
    "instead. build_frame needs an explicit batch spec. Ticket dsnn-3qm.72."))
def test_double_implicit_batch_axis_rides_out_implicit():
    """([a], b, c) @ ([a], c, d): neither operand stores a."""
    lhs = SparseTensor((DenseIndex(0, 2, None), DenseIndex(1, 3, 0)),
                       (DenseIndex(2, 4, 1),), jnp.ones((3, 4)))
    rhs = SparseTensor((DenseIndex(0, 2, None), DenseIndex(1, 4, 0)),
                       (DenseIndex(2, 5, 1),), jnp.ones((4, 5)))
    f = build_frame(lhs, rhs, n_contract=1)
    assert f.axes[0].case is AxisCase.IMPLICIT_OUT
    assert f.axes[0].implicit_count == 1


def test_a_contracted_axis_stored_by_neither_folds_into_a_scale():
    """(a, b, [c]) @ ([c], d): the extent c is stored by neither, so the
    contraction over it is a scale by c, not a dot."""
    lhs = SparseTensor((DenseIndex(0, 2, 0), DenseIndex(1, 3, 1)),
                       (DenseIndex(2, 4, None),), jnp.ones((2, 3)))
    rhs = SparseTensor((DenseIndex(0, 4, None),), (DenseIndex(1, 5, 0),),
                       jnp.ones((5,)))
    f = build_frame(lhs, rhs)
    contracted = [a for a in f.axes if a.contracted]
    assert [a.case for a in contracted] == [AxisCase.FOLD_SCALE]
    assert decide_emission(f) is EmissionKind.ANALYTIC, (
        "no dot axis survives, so no dot is emitted")


def test_a_contracted_axis_stored_by_one_sums_the_storer():
    """(a, b, [c]) @ (c, d): sum the storing side over c, then scale. No dot."""
    lhs = SparseTensor((DenseIndex(0, 2, 0), DenseIndex(1, 3, 1)),
                       (DenseIndex(2, 4, None),), jnp.ones((2, 3)))
    rhs = SparseTensor((DenseIndex(0, 4, 0),), (DenseIndex(1, 5, 1),),
                       jnp.ones((4, 5)))
    f = build_frame(lhs, rhs)
    contracted = [a for a in f.axes if a.contracted]
    assert [a.case for a in contracted] == [AxisCase.SUM_STORER]
    assert decide_emission(f) is EmissionKind.ANALYTIC


def test_the_budget_is_what_switches_the_emission():
    """The rule reads the grid and nothing else. No device branch exists."""
    lhs = SparseTensor((DenseIndex(0, 64, 0),), (DenseIndex(1, 64, 1),),
                       jnp.ones((64, 64)))
    rhs = SparseTensor((DenseIndex(0, 64, 0),), (DenseIndex(1, 64, 1),),
                       jnp.ones((64, 64)))
    f = build_frame(lhs, rhs)
    assert f.multiply_grid == 64 ** 3
    big = decide_emission(f, itemsize=4, budget_bytes=64 ** 3 * 4)
    small = decide_emission(f, itemsize=4, budget_bytes=64 ** 3 * 4 - 1)
    assert big is EmissionKind.MULTIPLY_REDUCE
    assert small is EmissionKind.DOT_GENERAL


def test_a_narrower_dtype_fits_a_larger_grid():
    """``itemsize`` is the only thing the dtype changes in phase 3."""
    lhs = SparseTensor((DenseIndex(0, 64, 0),), (DenseIndex(1, 64, 1),),
                       jnp.ones((64, 64)))
    rhs = SparseTensor((DenseIndex(0, 64, 0),), (DenseIndex(1, 64, 1),),
                       jnp.ones((64, 64)))
    f = build_frame(lhs, rhs)
    budget = 64 ** 3 * 2
    assert decide_emission(f, itemsize=2, budget_bytes=budget) is EmissionKind.MULTIPLY_REDUCE
    assert decide_emission(f, itemsize=4, budget_bytes=budget) is EmissionKind.DOT_GENERAL


def test_broadcasting_is_not_an_emission():
    """Owner ruling D1: an implicit axis is never broadcast into the physical
    grid. Only three emissions exist, and none of them is that."""
    assert {e.value for e in EmissionKind} == {
        "analytic", "multiply_reduce", "dot_general"}


def test_a_mismatched_contracted_extent_raises():
    lhs = SparseTensor((DenseIndex(0, 4, 0),), (DenseIndex(1, 3, 1),), jnp.ones((4, 3)))
    rhs = SparseTensor((DenseIndex(0, 7, 0),), (DenseIndex(1, 5, 1),), jnp.ones((7, 5)))
    with pytest.raises(ValueError, match="logical extent"):
        build_frame(lhs, rhs)


def test_the_frame_touches_no_value():
    """``val`` may be None on both operands and the frame still builds. That is
    the guarantee the four-phase split exists for."""
    lhs = SparseTensor((DenseIndex(0, 4, None),), (DenseIndex(1, 3, None),), None)
    rhs = SparseTensor((DenseIndex(0, 3, None),), (DenseIndex(1, 5, None),), None)
    f = build_frame(lhs, rhs)
    assert f.multiply_grid == 60
    assert all(a.implicit_count >= 1 for a in f.axes)
