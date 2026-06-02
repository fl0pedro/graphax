"""Bug: ``_assert_sparse_tensor_consistency`` was using bare ``assert``.

Under ``python -O`` (or ``PYTHONOPTIMIZE=1``) the bytecode compiler strips
``assert`` statements entirely, so the contiguous-IDs invariant and the
DiagonalIndex pairing invariant — which every downstream op assumes — would
silently disappear. A mismatched-id ``SparseTensor`` would then be
constructed without protest and propagate misaligned dims into matmul /
elementwise, producing wildly wrong shapes downstream.

Fix: use ``raise ValueError`` so the check survives optimisation.

This test is also run under ``-O`` from the unit's verification step:

    PYTHONHASHSEED=0 python -O -m pytest tests/misc/test_consistency_under_optimize.py
"""

import jax.numpy as jnp
import pytest

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor


def test_mismatched_dim_ids_raise_not_assert():
    """Construct a SparseTensor with deliberately non-contiguous dim ids.

    The check must raise ``ValueError`` — not silently accept (which is what
    happens when the original ``assert`` is stripped by ``-O``).
    """
    # ids are {0, 5} — not the contiguous range {0, 1} the invariant requires.
    bad_out = (DenseIndex(0, 3, 0),)
    bad_primal = (DenseIndex(5, 3, 1),)
    val = jnp.zeros((3, 3))
    with pytest.raises(ValueError, match="contiguous"):
        SparseTensor(bad_out, bad_primal, val)


def test_unpaired_sparse_dim_raises_not_assert():
    """A DiagonalIndex whose ``other_id`` doesn't point to a matching sibling
    must be rejected by the consistency check — under ``-O`` too."""
    # DiagonalIndex.other_id=99 is bogus; no sibling in the tensor.
    bad_out = (DiagonalIndex(0, 3, 0, other_id=99),)
    bad_primal = (DiagonalIndex(1, 3, 1, other_id=0),)
    val = jnp.zeros((3, 3))
    with pytest.raises(ValueError, match="sparse dimension pair"):
        SparseTensor(bad_out, bad_primal, val)
