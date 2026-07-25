"""Minimal, RAM-light reproducer for the forced-densify that ``Diag`` hits when an
index is already tied to a DIFFERENT index (Q2/Q3 "prove the sparser equivalent").

This is the unit-scale core of the slice/concat ViT ~24 GB densify that
``tests/elemental/test_approx_regression.py`` can only exercise RAM-guarded: keeping
an approximation edge block-diagonal makes a later ``Diag``'s pair CONFLICT. A
Jacobian diagonal ties one OUT axis to one PRIMAL axis, so ``out_i ~ primal_a`` AND
``out_i ~ primal_b`` (a != b) implies ``primal_a == primal_b`` — a 3-way tie the
pairwise :class:`DiagonalIndex` cannot represent. The sparse path therefore has only
two honest options:

  * DEFAULT (``GRAPHAX_BEST_EFFORT_TRANSFORMS`` unset/0): raise loudly — never
    silently produce a wrong (under-masked) edge (:func:`test_..._raises_by_default`).
  * best-effort: skip the second ``Diag`` → keep only the first mask → diverge from
    the dense oracle. That divergence is real (not a no-op), which is exactly why the
    production path DENSIFIES here to stay correct
    (:func:`test_..._two_diag_answer_is_strictly_sparser`).

The second test also quantifies the prize a correct n-way-sparse representation would
win: the true intersection is 25% dense here, versus 50% for the (wrong) mask-1-only
edge and 100% for the densified edge.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex
from graphax.sparse.micro_actions import Diag, apply_diag
from graphax.sparse.tensor import SparseTensor


def _one_out_two_primal(n=4):
    """A dense Jacobian ``(n, n, n)`` = ``[out_0, primal_0, primal_1]`` with distinct
    nonzero entries, so block masking (and its absence) is observable."""
    val = jnp.arange(n ** 3, dtype=jnp.float32).reshape(n, n, n) + 1.0
    out = (DenseIndex(id=0, size=n, axis=0),)
    primal = (DenseIndex(id=1, size=n, axis=1), DenseIndex(id=2, size=n, axis=2))
    return SparseTensor(out, primal, val), np.asarray(val)


def _block_mask(n, factor):
    """``(n, n)`` block-diagonal mask: True where ``row // bs == col // bs``,
    ``bs = n // factor`` (the exact structure ``Diag(_, _, factor)`` imposes)."""
    bs = n // factor
    return (np.arange(n)[:, None] // bs) == (np.arange(n)[None, :] // bs)


def test_diag_conflict_on_repaired_index_raises_by_default():
    """out_0 ~ primal_0, then out_0 ~ primal_1: the pairwise form can't hold both,
    so the second ``Diag`` raises rather than silently under-masking."""
    st, _ = _one_out_two_primal(4)
    st1 = apply_diag(st, Diag(0, 1, 2))  # out_0 ~ primal_0 (index 1), factor 2
    with pytest.raises(ValueError, match="conflict|already paired"):
        apply_diag(st1, Diag(0, 2, 2))  # out_0 ~ primal_1 (index 2) -> CONFLICT


def test_first_diag_is_mask1_and_the_two_diag_answer_is_strictly_sparser():
    """graphax's first ``Diag`` == mask-1 exactly; the CORRECT two-``Diag`` answer
    (the dense oracle's intersection) is strictly sparser and non-empty — so skipping
    the conflicting second ``Diag`` is genuinely wrong, and densifying is the correct
    (if memory-heavy) fallback."""
    n = 4
    st, val = _one_out_two_primal(n)
    st1 = apply_diag(st, Diag(0, 1, 2))
    got1 = np.asarray(st1.dense())

    m1 = _block_mask(n, 2)  # out_0 ~ primal_0
    m2 = _block_mask(n, 2)  # out_0 ~ primal_1
    mask1_only = val * m1[:, :, None]  # what best-effort skip yields
    oracle = val * m1[:, :, None] * m2[:, None, :]  # the correct intersection

    np.testing.assert_array_equal(got1, mask1_only)  # first Diag == mask-1
    assert not np.array_equal(oracle, mask1_only)  # 2nd Diag removes real entries
    assert oracle.sum() < mask1_only.sum()  # strictly sparser than the under-mask
    assert oracle.any()  # ...but a real, non-empty answer
    # the intersection is 25% dense; under-mask 50%; densified 100%.
    assert (oracle != 0).mean() == pytest.approx(0.25)
