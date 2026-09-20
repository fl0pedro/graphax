"""A contraction branch that COPIES its operand re-queues the operand's
transform (tickets dsnn-dfw.10 and dsnn-dfw.70).

WHAT THE CONTRACT IS.  ``prepare_face_operands`` hands the operands' queued
Jacobian transforms to ``contract_face_operands`` as ``pre_reattach`` /
``post_reattach``, and that re-attach is the ONLY carrier of them: it holds
because ``sparse_matmul`` and ``unload_*`` rebuild the tensor and DROP its
queue, so the re-attach restores it exactly once.

WHAT WENT WRONG.  Two branches do not rebuild anything.  A rank-0 partner
routes the face through ``scale_by_scalar``, which is a ``copy()`` with a
folded ``scalar_mult``; the identity pass-through is a ``copy()`` too.  The
copy KEEPS the queue, the re-attach adds the same transform again, and the
edge is stored with ``pre_transforms = (transpose, transpose)``.  Draining a
transpose relabel twice is the identity, so the stored edge is TRANSPOSED
against its nominal ``out_edge.aval.shape + in_edge.aval.shape``.  The
pass-through half was fixed inside ``_identity_passthrough``; the scale half
was not, and it is the one the campaign hit.

THE THREE FACES OF THE SAME DEFECT, all reproduced below on
``sin(sum(x.T))`` and its siblings at ``x`` of shape ``(3, 5)``:

  * the next merge onto that edge raises ``Existing edge shape (5, 3) does
    not match expected shape (3, 5)!`` -- dsnn-dfw.10, the TLM order;
  * a fresh contraction off it raises ``Computed edge shape (5, 3) does not
    match expected shape (3, 5)!`` -- dsnn-dfw.70, the NN256 C rows;
  * with no merge at all nothing raises and the JACOBIAN COMES OUT
    TRANSPOSED, which is the grad_cosine structure mismatch of dsnn-dfw.70.

THE INVARIANT.  The operands ``prepare_face_operands`` returns carry no queue
on the side it re-attaches.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import jacve
from graphax.core import (
    _drain_transforms,
    contract_face_operands,
    prepare_face_operands,
)
from graphax.primitives.transforms import transpose_elemental_only
from graphax.sparse.indexes import DenseIndex
from graphax.sparse.tensor import SparseTensor


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _transpose_relabel(m, n):
    """The queued relabel ``lax.transpose_p``'s elemental rule produces for
    ``x`` of shape ``(m, n)`` -- the real transform, not a stand-in."""
    seed = transpose_elemental_only(
        jnp.zeros((n, m)), (jnp.zeros((m, n)),), permutation=(1, 0)
    )[0]
    return seed.pre_transforms[0]


def _pre_edge(m, n, seed=0):
    """The edge the engine stores after the transpose vertex is eliminated:
    ``d(scalar)/dx`` for ``x`` of shape ``(m, n)``, held in the TRANSPOSED
    dim order ``(n, m)`` with the relabel still queued.  Draining it gives the
    nominal ``(m, n)``."""
    rng = np.random.default_rng(seed)
    val = jnp.asarray(rng.standard_normal((n, m)).astype(np.float32))
    return SparseTensor(
        out_dims=[],
        primal_dims=[DenseIndex(0, n, 0), DenseIndex(1, m, 1)],
        val=val,
        pre_transforms=[_transpose_relabel(m, n)],
    )


def _scalar_edge(value=2.0):
    """A rank-0 edge: the Jacobian of one scalar op against another.  It is
    what routes the face through ``scale_by_scalar``."""
    return SparseTensor([], [], val=jnp.asarray(value, dtype=jnp.float32))


# ---------------------------------------------------------------------------
# 1. the stored operand really is a transposed edge with one queued relabel
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("m,n", [(3, 5), (32, 128), (5, 3), (4, 4)])
def test_the_operand_drains_to_the_nominal_shape(m, n):
    pre = _pre_edge(m, n)
    assert tuple(pre.shape) == (n, m)
    assert len(pre.pre_transforms) == 1
    assert tuple(_drain_transforms(pre.copy()).shape) == (m, n)


# ---------------------------------------------------------------------------
# 2. the face: the relabel is queued ONCE, and the result drains to nominal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("approx", [False, True])
@pytest.mark.parametrize("m,n", [(3, 5), (32, 128), (5, 3), (4, 4)])
def test_a_rank0_partner_does_not_requeue_the_relabel(approx, m, n):
    """The ticket, as one statement."""
    ops = prepare_face_operands(_scalar_edge(), _pre_edge(m, n), approx=approx)
    res = contract_face_operands(ops).val
    assert len(res.pre_transforms) == 1
    assert tuple(_drain_transforms(res.copy()).shape) == (m, n)


@pytest.mark.parametrize("approx", [False, True])
@pytest.mark.parametrize("m,n", [(3, 5), (32, 128), (4, 4)])
def test_the_scaled_face_equals_the_dense_computation(approx, m, n):
    """EXACTNESS.  The square row is the silent half: its shape is a fixed
    point of the permutation, so only the VALUE separates the two engines."""
    pre = _pre_edge(m, n, seed=7)
    ops = prepare_face_operands(_scalar_edge(3.0), pre, approx=approx)
    got = np.asarray(_drain_transforms(contract_face_operands(ops).val).dense())
    want = 3.0 * np.asarray(_drain_transforms(pre.copy()).dense())
    assert got.shape == want.shape == (m, n)
    assert np.allclose(got, want, rtol=1e-6, atol=1e-6), \
        f"max |diff| = {np.max(np.abs(got - want))}"


@pytest.mark.parametrize("approx", [False, True])
def test_the_operands_carry_no_reattached_queue(approx):
    """The invariant itself, so a new contraction branch inherits it."""
    ops = prepare_face_operands(_scalar_edge(), _pre_edge(3, 5), approx=approx)
    assert ops.pre.pre_transforms == ()
    assert ops.post.post_transforms == ()
    assert len(ops.pre_reattach) == 1


# ---------------------------------------------------------------------------
# 3. end to end: the engine, against jax.jacrev
# ---------------------------------------------------------------------------
_X = jnp.asarray(np.arange(15, dtype=np.float32).reshape(3, 5) / 14.0 - 0.3)


def _silent(x):
    """No merge onto the edge: nothing raises and the Jacobian comes out
    transposed."""
    return jnp.sin(jnp.sum(jnp.transpose(x)))


def _merged(x):
    """A second contribution to the same edge: the merge raises ``Existing
    edge shape ...`` -- dsnn-dfw.10."""
    return jnp.sin(jnp.sum(jnp.transpose(x))) + jnp.sum(x * 2.0)


def _contracted(x):
    """A further contraction off the stored edge: ``Computed edge shape ...``
    -- dsnn-dfw.70."""
    s = jnp.sum(jnp.exp(jnp.transpose(x)))
    return jnp.sin(s) * jnp.cos(s)


_MODELS = {"silent": _silent, "merged": _merged, "contracted": _contracted}
_ORDERS = {"silent": [[1, 2, 3], [1, 3, 2], [3, 1, 2]],
           "merged": [[1, 2, 3, 4, 5, 6], [1, 2, 4, 3, 5, 6]],
           "contracted": [[1, 2, 3, 4, 5, 6], [1, 2, 3, 5, 4, 6]]}


@pytest.mark.parametrize("name", sorted(_MODELS))
@pytest.mark.parametrize("order", ["fwd", "rev"])
def test_engine_matches_jax_in_the_canonical_orders(name, order):
    f = _MODELS[name]
    got = np.asarray(jacve(f, order, argnums=(0,))(_X)[0])
    want = np.asarray(jax.jacrev(f)(_X))
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-5, atol=1e-5), \
        f"max |diff| = {np.max(np.abs(got - want))}"


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_engine_matches_jax_in_the_free_orders_that_hit_the_face(name):
    f = _MODELS[name]
    want = np.asarray(jax.jacrev(f)(_X))
    for order in _ORDERS[name]:
        got = np.asarray(jacve(f, list(order), argnums=(0,))(_X)[0])
        assert got.shape == want.shape, f"order={order} shape={got.shape}"
        assert np.allclose(got, want, rtol=1e-5, atol=1e-5), \
            f"order={order} max |diff| = {np.max(np.abs(got - want))}"
