"""A CROSSED diagonal pair is a PERMUTATION, not the identity (ticket dsnn-a9a).

WHAT THE STORED TENSOR IS.  ``lax.transpose_p``'s elemental rule returns a
dim-less seed with the permutation queued as a lazy ``JacobianTransform``
(``primitives/transforms.py::_transpose_elementals``).  When that queue is
DRAINED -- which ``_eliminate_vertex`` does at two sites, both armed only under
an approximation config -- the relabel is written into the tensor itself.  The
result is a ``val is None`` diagonal whose out/primal pairing is CROSSED: out
dim 0 is tied to primal dim 1 and out dim 1 to primal dim 0.  That tensor is
legal, and its dense form is the transpose Jacobian.

WHAT WENT WRONG.  ``core._acts_as_identity`` asked only whether every out dim
HAS a diagonal partner of the same logical size.  It did not ask whether the
partner sits at the same POSITION.  A crossed pair therefore read as the
identity, ``prepare_face_operands`` set ``need_contract=False``, and
``contract_face_operands`` passed the OTHER operand through unchanged -- the
permutation was silently dropped.

THE TWO FACES OF THE SAME DEFECT:

  * on a RECTANGULAR transpose the dropped permutation also changes the dim
    list, so ``core._set_inner`` raised
    ``StoredEdgeShapeMismatch: (32, 128, 32, 128) vs (128, 32, 32, 128)``
    -- that is the ticket, reproduced at the campaign's own TransformerLM
    shapes below;
  * on a SQUARE transpose nothing raises.  The shape is a fixed point of the
    permutation, so the edge passes every check and the JACOBIAN IS SIMPLY
    WRONG.  That case is the reason the fix is at ``_acts_as_identity`` and not
    at the store.

THE INVARIANT.  ``val is None`` plus a diagonal pairing is a pass-through ONLY
when the pairing is POSITION-ALIGNED: ``out_dims[k]`` must be tied to
``primal_dims[k]`` for every ``k``.  Any other pairing is a permutation and
must go through the real contraction, which is the one piece of code that
already composes a permutation with anything.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.core import (
    _acts_as_identity,
    contract_face_operands,
    prepare_face_operands,
)
from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _crossed_identity(n, m, dtype=jnp.float32):
    """``d(v.T)/dv`` for ``v`` of shape ``(m, n)``: dense form ``(n, m, m, n)``
    with a one at ``[i, j, j, i]``.  Stored as a ``val is None`` diagonal whose
    pairing is CROSSED, which is exactly what draining the transpose seed's
    queued relabel produces."""
    return SparseTensor(
        out_dims=[DiagonalIndex(0, n, None, 3), DiagonalIndex(1, m, None, 2)],
        primal_dims=[DiagonalIndex(2, m, None, 1),
                     DiagonalIndex(3, n, None, 0)],
        val=None,
        dtype=dtype,
    )


def _aligned_identity(a, b, dtype=jnp.float32):
    """The real identity on a rank-2 variable: out k tied to primal k."""
    return SparseTensor(
        out_dims=[DiagonalIndex(0, a, None, 2), DiagonalIndex(1, b, None, 3)],
        primal_dims=[DiagonalIndex(2, a, None, 0),
                     DiagonalIndex(3, b, None, 1)],
        val=None,
        dtype=dtype,
    )


def _dense_edge(out_shape, primal_shape, seed=0):
    """A fully explicit edge in nominal dim order."""
    n_out = len(out_shape)
    shape = tuple(out_shape) + tuple(primal_shape)
    rng = np.random.default_rng(seed)
    val = jnp.asarray(rng.standard_normal(shape).astype(np.float32))
    out_dims = [DenseIndex(i, int(s), i) for i, s in enumerate(out_shape)]
    primal_dims = [DenseIndex(n_out + i, int(s), n_out + i)
                   for i, s in enumerate(primal_shape)]
    return SparseTensor(out_dims=out_dims, primal_dims=primal_dims, val=val)


# ---------------------------------------------------------------------------
# 1. the predicate itself
# ---------------------------------------------------------------------------
def test_aligned_pairing_is_the_identity():
    assert _acts_as_identity(_aligned_identity(32, 128)) is True


def test_crossed_pairing_is_not_the_identity():
    """The ticket, as one statement."""
    assert _acts_as_identity(_crossed_identity(128, 32)) is False


def test_square_crossed_pairing_is_not_the_identity():
    """The SILENT half: a square transpose has the nominal shape, so nothing
    downstream can catch it.  The predicate has to."""
    assert _acts_as_identity(_crossed_identity(64, 64)) is False


# ---------------------------------------------------------------------------
# 2. the tensor really is the transpose Jacobian
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,m", [(128, 32), (64, 64), (5, 3), (7, 2)])
def test_crossed_identity_densifies_to_the_transpose_jacobian(n, m):
    got = np.asarray(_crossed_identity(n, m).dense())
    want = np.zeros((n, m, m, n), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            want[i, j, j, i] = 1.0
    assert got.shape == want.shape
    assert np.array_equal(got, want)


# ---------------------------------------------------------------------------
# 3. the face contraction: nominal dims and the dense answer
# ---------------------------------------------------------------------------
def _face(n, m, primal_shape, approx, seed=0):
    """``post = d(v.T)/dv``, ``pre = dv/du``; the face's result is
    ``d(v.T)/du``."""
    post = _crossed_identity(n, m)
    pre = _dense_edge((m, n), primal_shape, seed=seed)
    ops = prepare_face_operands(post, pre, approx=approx)
    return post, pre, ops, contract_face_operands(ops).val


@pytest.mark.parametrize("approx", [False, True])
def test_the_ticket_shape_is_nominal(approx):
    """``StoredEdgeShapeMismatch: (32, 128, 32, 128)`` vs the nominal
    ``(128, 32, 32, 128)`` -- the campaign's own TransformerLM face, at its own
    shapes.  ``v`` is ``(32, 128)``, ``v.T`` is ``(128, 32)`` and ``u`` is
    ``(32, 128)``, so the nominal edge is ``(128, 32) + (32, 128)``."""
    _, _, _, res = _face(128, 32, (32, 128), approx)
    assert tuple(res.shape) == (128, 32, 32, 128)


@pytest.mark.parametrize("approx", [False, True])
@pytest.mark.parametrize(
    "n,m,primal",
    [(128, 32, (32, 128)), (64, 64, (64, 64)), (5, 3, (4,)),
     (7, 2, (3, 6)), (3, 5, (5, 3)), (2, 2, (2, 2))],
)
def test_face_equals_the_dense_computation(approx, n, m, primal):
    """EXACTNESS.  The face result is the dense product of the two operands'
    dense forms, contracted over ``v``'s two axes.  The square rows are the
    silent half: they pass the shape check either way, so only the VALUE
    separates the fixed engine from the broken one."""
    post, pre, ops, res = _face(n, m, primal, approx)
    want = np.einsum(
        "ijkl,klmn->ijmn" if len(primal) == 2 else "ijkl,klm->ijm",
        np.asarray(post.dense()), np.asarray(pre.dense()))
    got = np.asarray(res.dense())
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=1e-6, atol=1e-6), \
        f"max |diff| = {np.max(np.abs(got - want))}"


@pytest.mark.parametrize("approx", [False, True])
def test_aligned_identity_still_passes_through(approx):
    """The fast path must survive: a genuinely aligned identity operand still
    yields the OTHER operand, bit for bit."""
    post = _aligned_identity(4, 6)
    pre = _dense_edge((4, 6), (5,), seed=3)
    ops = prepare_face_operands(post, pre, approx=approx)
    assert ops.need_contract is False
    res = contract_face_operands(ops).val
    assert np.array_equal(np.asarray(res.dense()), np.asarray(pre.dense()))


# ---------------------------------------------------------------------------
# 4. randomized sweep around the failing shape
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", list(range(12)))
def test_randomised_shapes_around_the_failure(seed):
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(2, 9))
    m = int(rng.integers(2, 9))
    primal = tuple(int(rng.integers(2, 6))
                   for _ in range(int(rng.integers(1, 3))))
    for approx in (False, True):
        post, pre, _ops, res = _face(n, m, primal, approx, seed=seed)
        sub = "klm" if len(primal) == 1 else "klmn"
        want = np.einsum(f"ijkl,{sub}->ij{sub[2:]}",
                         np.asarray(post.dense()), np.asarray(pre.dense()))
        got = np.asarray(res.dense())
        assert tuple(res.shape) == (n, m, *primal)
        assert np.allclose(got, want, rtol=1e-6, atol=1e-6), \
            f"n={n} m={m} primal={primal} approx={approx} " \
            f"max|diff|={np.max(np.abs(got - want))}"


# ---------------------------------------------------------------------------
# 5. end to end: the engine, against jax.jacrev
# ---------------------------------------------------------------------------
_W = jnp.asarray(np.arange(12, dtype=np.float32).reshape(4, 3) / 11.0 + 0.1)
_X = jnp.asarray(np.arange(15, dtype=np.float32).reshape(3, 5) / 14.0 - 0.3)
_Y = jnp.asarray(np.arange(24, dtype=np.float32).reshape(4, 6) / 23.0 + 0.2)


def _transposed_model(W, X):
    """A rectangular transpose sitting between two contractions, so the
    transpose edge is a real face operand rather than an output seed."""
    v = jnp.tanh(W @ X)          # (4, 5)
    return jnp.sum((v.T @ _Y) ** 2)   # v.T is (5, 4)


@pytest.mark.parametrize("order", ["fwd", "rev"])
def test_engine_matches_jax_on_a_transposed_model(order):
    from graphax import jacve
    got = jacve(_transposed_model, order, argnums=(0, 1))(_W, _X)
    want = jax.jacrev(_transposed_model, argnums=(0, 1))(_W, _X)
    for g, w in zip(got, want):
        assert np.allclose(np.asarray(g), np.asarray(w),
                           rtol=1e-5, atol=1e-5), \
            f"max|diff| = {np.max(np.abs(np.asarray(g) - np.asarray(w)))}"


# ---------------------------------------------------------------------------
# 6. the ARMED engine on a transposed model, against jax.jacrev
# ---------------------------------------------------------------------------
# An identity CALLABLE arms the approximation path (``face_config_is_approx``
# and ``_is_approx_cfg``) without changing one number, so the expected answer
# stays the exact Jacobian.  That isolates the defect from every real
# approximation: the armed run drains the transpose seed's queued relabel into
# the crossed diagonal, which is exactly the state this ticket is about.
def _identity_rule(t):
    return t


def _order_for(jaxpr, vo, seed):
    from graphax.core import _checkify_order
    import random as _random
    order = list(_checkify_order(list(range(1, len(jaxpr.eqns) + 1)),
                                 jaxpr, vo))
    _random.Random(seed).shuffle(order)
    return order


@pytest.mark.parametrize("seed", list(range(8)))
def test_armed_free_order_matches_jax_on_a_transposed_model(seed):
    from graphax import inline_call_primitives
    from graphax.incremental import IncrementalJaxpr

    args = (_W, _X)
    argnums = (0, 1)
    closed = jax.make_jaxpr(_transposed_model)(*args)
    jaxpr, consts = inline_call_primitives(closed.jaxpr,
                                           list(closed.literals))
    ij = IncrementalJaxpr(jaxpr, argnums, consts, list(args))
    for v in _order_for(jaxpr, ij.vo, seed):
        ij.eliminate(v, (_identity_rule,))
    outs, _labels = ij.jacobian_outputs(dense=True)
    want = jax.jacrev(_transposed_model, argnums=argnums)(*args)
    assert len(outs) == len(want)
    for g, w in zip(outs, want):
        g = np.asarray(g).reshape(np.asarray(w).shape)
        assert np.allclose(g, np.asarray(w), rtol=1e-5, atol=1e-5), \
            f"seed={seed} max|diff| = {np.max(np.abs(g - np.asarray(w)))}"
