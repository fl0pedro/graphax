"""The implicit-block expansion is the only rewrite that dropped a queued
Jacobian transform (ticket dsnn-dfw.79).

A BLOCKED DENSE dim (``other_id is None`` with a ``block_size``) stores one
cell per block. ``_apply_block_diagonal`` pays for that block up front through
``_expand_implicit_blocks``, which rewrites the dim to its full extent. The
rewrite is rank-preserving and logical-shape-preserving, so a queued relabel
still addresses the same dims -- but the rebuild dropped the queue, and a
relabel is the only thing that puts a stored edge back into nominal dim ORDER.

The edge then merges in its pre-drain order and ``_eliminate_vertex`` raises
``Computed edge shape (10, 784, 256) does not match expected shape
(10, 256, 784)``, measured on job 67006 (arm C, nn256/mnist, free order,
vertex 22, face in=2 out=28, the face's ``new`` slot ``Diag(i=0, j=2,
factor=2)``). ``apply_quant``, ``apply_compress`` and ``_apply_block_diagonal``
all state the same postcondition; this one was missed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from graphax.core import (
    _drain_transforms,
    contract_face_operands,
    prepare_face_operands,
)
from graphax.primitives.transforms import _transpose_elementals
from graphax.sparse.indexes import DenseIndex, Index
from graphax.sparse.micro_actions import Diag, apply_diag
from graphax.sparse.ops.dense import _expand_implicit_blocks
from graphax.sparse.tensor import SparseTensor


def _relabel(rows: int, cols: int):
    """The queued transpose relabel ``lax.transpose_p``'s elemental rule emits
    for ``W.T`` -- a permutation carried as metadata, drained later."""
    seed = _transpose_elementals(
        None, jnp.zeros((rows, cols), jnp.float32), permutation=(1, 0))[0]
    return seed.pre_transforms[0]


def _blocked_dense(dim_id: int, size: int, block: int, axis=None) -> Index:
    """A blocked DENSE dim: ``size`` stored cells, ``block`` positions each."""
    return Index(dim_id, size, axis, None, block, None)


# ---------------------------------------------------------------------------
# 1. the rewrite itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_val", [False, True])
def test_expand_implicit_blocks_keeps_the_queues(with_val):
    t = _relabel(4, 6)
    val = jnp.arange(3, dtype=jnp.float32) if with_val else None
    st = SparseTensor(
        (DenseIndex(0, 5, None),),
        (_blocked_dense(1, 3, 2, 0 if with_val else None),),
        val,
        pre_transforms=(t,),
        post_transforms=(),
    )
    out = _expand_implicit_blocks(st, only_ids=frozenset((1,)))

    assert out.pre_transforms == (t,), (
        "the expansion dropped the queued relabel; nothing else can put the "
        "edge back into nominal dim order")
    assert out.shape == st.shape, "the rewrite must preserve the logical shape"


def test_expand_implicit_blocks_keeps_post_transforms_too():
    t = _relabel(4, 6)
    st = SparseTensor(
        (DenseIndex(0, 5, None),),
        (_blocked_dense(1, 3, 2),),
        None,
        post_transforms=(t,),
    )
    out = _expand_implicit_blocks(st, only_ids=frozenset((1,)))
    assert out.post_transforms == (t,)


def test_expand_implicit_blocks_is_value_exact():
    """The queue survives AND the data does: the expanded tensor densifies to
    exactly what the original densifies to."""
    st = SparseTensor(
        (DenseIndex(0, 2, 0),),
        (_blocked_dense(1, 3, 2, 1),),
        jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
    )
    out = _expand_implicit_blocks(st, only_ids=frozenset((1,)))
    np.testing.assert_allclose(np.asarray(out.dense()), np.asarray(st.dense()))


def test_expand_implicit_blocks_returns_the_same_object_when_idle():
    """Nothing to expand: no rebuild, so no queue to lose."""
    st = SparseTensor((DenseIndex(0, 4, None),), (DenseIndex(1, 6, None),),
                      None, pre_transforms=(_relabel(4, 6),))
    assert _expand_implicit_blocks(st) is st


# ---------------------------------------------------------------------------
# 2. the route that reaches it
# ---------------------------------------------------------------------------
def test_apply_diag_on_a_blocked_dense_operand_keeps_the_queue():
    t = _relabel(4, 6)
    st = SparseTensor(
        (DenseIndex(0, 4, None),),
        (DenseIndex(1, 7, None), _blocked_dense(2, 3, 2)),
        None,
        scalar_mult=jnp.float32(1.0),
        pre_transforms=(t,),
    )
    out = apply_diag(st, Diag(i=0, j=2, factor=2))
    assert out.pre_transforms == (t,)
    assert out.shape == st.shape


# ---------------------------------------------------------------------------
# 3. the ticket's face, end to end through the two contraction halves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_out, n_in, n_mid, factor",
    [
        (10, 784, 256, 2),   # the measured case, job 67006
        (4, 7, 6, 2),
        (6, 5, 9, 3),
        (6, 6, 6, 2),        # square: the defect is SILENT here
    ],
)
def test_the_face_drains_to_nominal(n_out, n_in, n_mid, factor):
    """``out.aval.shape + in.aval.shape`` is an invariant of the DRAINED edge.

    The face is the ticket's: a rank-0-ish post operand, a pre operand whose
    trailing dims are stored transposed behind a queued relabel, and a ``Diag``
    in the face's ``new`` slot. Without the fix the drained edge keeps the
    pre-relabel order and the next merge raises.
    """
    block = n_mid // factor
    post = SparseTensor((DenseIndex(0, n_out, None),),
                        (DenseIndex(1, 1, None),), None,
                        scalar_mult=jnp.float32(1.0))
    pre = SparseTensor(
        (DenseIndex(0, 1, None),),
        (DenseIndex(1, n_in, None), _blocked_dense(2, block, factor)),
        None, scalar_mult=jnp.float32(1.0),
        pre_transforms=(_relabel(n_in, n_mid),))

    assert tuple(pre.shape) == (1, n_in, n_mid)
    assert tuple(_drain_transforms(pre.copy()).shape) == (1, n_mid, n_in)

    ops = prepare_face_operands(post, pre, approx=False)
    edge = contract_face_operands(ops).val
    assert tuple(edge.shape) == (n_out, n_in, n_mid)

    control = _drain_transforms(edge.copy())
    assert tuple(control.shape) == (n_out, n_mid, n_in), (
        "the re-attach alone already fails; this fixture is wrong")

    after = apply_diag(edge, Diag(i=0, j=2, factor=factor))
    drained = _drain_transforms(after)
    assert tuple(drained.shape) == (n_out, n_mid, n_in), (
        f"stored edge drains to {tuple(drained.shape)}, nominal "
        f"{(n_out, n_mid, n_in)}")


def test_the_face_keeps_the_relabel_across_the_new_slot():
    """The queue the re-attach put on the edge is still there after the slot
    hook -- the statement the shape assert is downstream of."""
    post = SparseTensor((DenseIndex(0, 10, None),),
                        (DenseIndex(1, 1, None),), None,
                        scalar_mult=jnp.float32(1.0))
    t = _relabel(784, 256)
    pre = SparseTensor(
        (DenseIndex(0, 1, None),),
        (DenseIndex(1, 784, None), _blocked_dense(2, 128, 2)),
        None, scalar_mult=jnp.float32(1.0), pre_transforms=(t,))

    edge = contract_face_operands(
        prepare_face_operands(post, pre, approx=False)).val
    assert edge.pre_transforms == (t,)
    assert apply_diag(edge, Diag(i=0, j=2, factor=2)).pre_transforms == (t,)
