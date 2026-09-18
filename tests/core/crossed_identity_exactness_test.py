"""EXACTNESS of the crossed diagonal pair against the dense contraction
(ticket dsnn-a9a, the proof run of 2026-09-18).

``tests/core/crossed_identity_passthrough_test.py`` states the invariant and
proves the RECTANGULAR half, which used to raise ``StoredEdgeShapeMismatch``.
This file adds the part the ticket needs before it can be closed:

  1. the same face contraction run through the REAL path
     (``prepare_face_operands`` then ``contract_face_operands``) against the
     DENSE contraction of the two operands' dense forms, over many shapes AND
     over many STORAGE MIXES of the ``pre`` operand -- explicit dense, an
     implicit (broadcast) axis on either side, a pure diagonal pair, a
     block-diagonal pair, a blocked-dense dim;
  2. the SQUARE half, in numbers. A square transpose is a fixed point of the
     permutation, so the dropped permutation changed no shape and nothing
     raised. Before the fix the face returned the OTHER operand VERBATIM, so
     the Jacobian was the un-permuted one. The test below measures both the
     error against the dense answer and the distance to that verbatim operand,
     so the same file prints the proof at the broken revision and at the fixed
     one.

Every case prints one ``A9A-EXACT`` line with its numbers, so a run at
``a70bd8a`` (before) and a run at ``3df48f2`` (after) can be compared by
grepping the two logs rather than by trusting a pass or a fail.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from graphax.core import (
    _acts_as_identity,
    contract_face_operands,
    prepare_face_operands,
)
from graphax.sparse.indexes import DenseIndex, DiagonalIndex, Index
from graphax.sparse.tensor import SparseTensor


# ---------------------------------------------------------------------------
# the post operand: d(v.T)/dv for v of shape (m, n), stored CROSSED
# ---------------------------------------------------------------------------
def _crossed_identity(n, m, dtype=jnp.float32):
    """Dense form ``(n, m, m, n)`` with a one at ``[i, j, j, i]``. ``val is
    None``; out dim 0 is tied to primal dim 1 and out dim 1 to primal dim 0.
    That is what draining the transpose seed's queued relabel produces."""
    return SparseTensor(
        out_dims=[DiagonalIndex(0, n, None, 3), DiagonalIndex(1, m, None, 2)],
        primal_dims=[DiagonalIndex(2, m, None, 1),
                     DiagonalIndex(3, n, None, 0)],
        val=None,
        dtype=dtype,
    )


def _rand(shape, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape).astype(np.float32))


# ---------------------------------------------------------------------------
# the pre operand: dv/du, out dims (m, n), in SEVEN storage mixes
#
# Every builder returns (tensor, primal_logical_shape). ``m`` and ``n`` are
# v's two logical extents, so the pre operand's out dims always read (m, n)
# and the face is always well posed.
# ---------------------------------------------------------------------------
def _pre_dense(m, n, seed):
    """All four dims explicit and dense. The plain case."""
    p, q = 3, 2
    val = _rand((m, n, p, q), seed)
    return SparseTensor(
        out_dims=[DenseIndex(0, m, 0), DenseIndex(1, n, 1)],
        primal_dims=[DenseIndex(2, p, 2), DenseIndex(3, q, 3)],
        val=val,
    ), (p, q)


def _pre_implicit_out(m, n, seed):
    """The SECOND out axis is IMPLICIT: one copy stored, every position along
    ``n`` reads it."""
    p = 4
    val = _rand((m, p), seed)
    return SparseTensor(
        out_dims=[DenseIndex(0, m, 0), DenseIndex(1, n, None)],
        primal_dims=[DenseIndex(2, p, 1)],
        val=val,
    ), (p,)


def _pre_implicit_primal(m, n, seed):
    """The PRIMAL axis is IMPLICIT: the Jacobian does not vary along ``u``."""
    p = 3
    val = _rand((m, n), seed)
    return SparseTensor(
        out_dims=[DenseIndex(0, m, 0), DenseIndex(1, n, 1)],
        primal_dims=[DenseIndex(2, p, None)],
        val=val,
    ), (p,)


def _pre_diag_second(m, n, seed):
    """A PURE DIAGONAL pair on v's SECOND axis: ``u`` has extent ``n`` and the
    Jacobian is diagonal in it."""
    val = _rand((m, n), seed)
    return SparseTensor(
        out_dims=[DenseIndex(0, m, 0), DiagonalIndex(1, n, 1, 2)],
        primal_dims=[DiagonalIndex(2, n, 1, 1)],
        val=val,
    ), (n,)


def _pre_diag_first(m, n, seed):
    """A PURE DIAGONAL pair on v's FIRST axis. The diagonal now sits on the
    axis the permutation moves to the OTHER side, which is the arrangement the
    dropped permutation used to hide."""
    val = _rand((m, n), seed)
    return SparseTensor(
        out_dims=[DiagonalIndex(0, m, 0, 2), DenseIndex(1, n, 1)],
        primal_dims=[DiagonalIndex(2, m, 0, 0)],
        val=val,
    ), (m,)


def _pre_blockdiag(m, n, seed):
    """A BLOCK-DIAGONAL pair on v's second axis: ``n = N * B_o`` and the primal
    side carries ``B_i`` per block."""
    b_o = 2 if n % 2 == 0 else 1
    n_meta = n // b_o
    b_i = 3
    val = _rand((m, n_meta, b_o, b_i), seed)
    return SparseTensor(
        out_dims=[DenseIndex(0, m, 0),
                  DiagonalIndex(1, n_meta, 1, 2, block_size=b_o,
                                block_axis=2)],
        primal_dims=[DiagonalIndex(2, n_meta, 1, 1, block_size=b_i,
                                   block_axis=3)],
        val=val,
    ), (n_meta * b_i,)


def _pre_blocked_dense(m, n, seed):
    """A BLOCKED DENSE primal dim: one stored entry per block, the positions
    inside a block are implicit (ticket dsnn-3qm.62)."""
    n_blk, b = 2, 3
    val = _rand((m, n, n_blk), seed)
    return SparseTensor(
        out_dims=[DenseIndex(0, m, 0), DenseIndex(1, n, 1)],
        primal_dims=[Index(2, n_blk, 2, None, b, None)],
        val=val,
    ), (n_blk * b,)


MIXES = {
    "dense": _pre_dense,
    "implicit_out": _pre_implicit_out,
    "implicit_primal": _pre_implicit_primal,
    "diag_second": _pre_diag_second,
    "diag_first": _pre_diag_first,
    "blockdiag": _pre_blockdiag,
    "blocked_dense": _pre_blocked_dense,
}


# ---------------------------------------------------------------------------
# the real path, and the dense oracle
# ---------------------------------------------------------------------------
def _run_face(n, m, mix, approx, seed):
    """Contract ``d(v.T)/dv`` against ``dv/du`` through the code the
    elimination uses, and return the measurement of the result."""
    post = _crossed_identity(n, m)
    pre, primal_shape = MIXES[mix](m, n, seed)
    ops = prepare_face_operands(post, pre, approx=approx)
    res = contract_face_operands(ops).val

    post_d = np.asarray(post.dense())
    pre_d = np.asarray(pre.dense())
    # The face contracts the post operand's PRIMAL axes (v) against the pre
    # operand's OUT axes (v). That is the dense chain rule, nothing else.
    want = np.einsum("ijkl,kl...->ij...", post_d, pre_d)
    got = np.asarray(res.dense())
    return got, want, pre_d, primal_shape, ops


def _report(tag, got, want, pre_d, extra=""):
    if got.shape == want.shape:
        diff = float(np.max(np.abs(got - want)))
        scale = float(np.max(np.abs(want))) or 1.0
    else:
        diff = float("inf")
        scale = 1.0
    print(f"A9A-EXACT {tag} got_shape={tuple(got.shape)} "
          f"want_shape={tuple(want.shape)} max_abs_diff={diff:.6e} "
          f"rel={diff / scale:.6e} {extra}", flush=True)
    return diff


# ---------------------------------------------------------------------------
# 1. the predicate, on both halves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n,m", [(128, 32), (64, 64), (5, 3), (4, 4)])
def test_crossed_pair_is_never_the_identity(n, m):
    assert _acts_as_identity(_crossed_identity(n, m)) is False


# ---------------------------------------------------------------------------
# 2. exactness over every storage mix, rectangular AND square
# ---------------------------------------------------------------------------
_SHAPES = [(5, 3), (3, 5), (7, 2), (2, 7),      # rectangular
           (4, 4), (6, 6), (3, 3)]              # square


@pytest.mark.parametrize("mix", sorted(MIXES))
@pytest.mark.parametrize("n,m", _SHAPES)
@pytest.mark.parametrize("approx", [False, True])
def test_face_equals_the_dense_contraction(mix, n, m, approx):
    got, want, pre_d, primal_shape, ops = _run_face(n, m, mix, approx, seed=7)
    square = "SQUARE" if n == m else "RECT"
    diff = _report(f"mix={mix} n={n} m={m} approx={approx} {square}",
                   got, want, pre_d,
                   extra=f"need_contract={ops.need_contract}")
    assert got.shape == want.shape, (
        f"mix={mix} n={n} m={m} approx={approx}: the face returned "
        f"{tuple(got.shape)} where the dense contraction is {tuple(want.shape)}"
    )
    assert diff <= 1e-5, (
        f"mix={mix} n={n} m={m} approx={approx}: max|diff| = {diff}"
    )


# ---------------------------------------------------------------------------
# 3. the SQUARE half, in numbers: it was WRONG before and it is right after
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mix", sorted(MIXES))
@pytest.mark.parametrize("k", [3, 4, 6])
@pytest.mark.parametrize("approx", [False, True])
def test_square_transpose_is_not_the_pass_through(mix, k, approx):
    """A square transpose has the nominal shape either way, so no shape check
    can catch the dropped permutation. Two numbers separate the two engines:

      * ``max|got - dense|`` -- zero-ish when the permutation is applied;
      * ``max|got - pre|``   -- ZERO when the face returned the other operand
        verbatim, which is exactly what the pass-through did.
    """
    got, want, pre_d, _primal, ops = _run_face(k, k, mix, approx, seed=11)
    verbatim = (got.shape == pre_d.shape
                and float(np.max(np.abs(got - pre_d))) == 0.0)
    diff = _report(f"square mix={mix} k={k} approx={approx}",
                   got, want, pre_d,
                   extra=f"need_contract={ops.need_contract} "
                         f"verbatim_pre={verbatim}")
    assert not verbatim, (
        f"mix={mix} k={k} approx={approx}: the face returned the pre operand "
        f"verbatim, so the transpose was dropped"
    )
    assert diff <= 1e-5, (
        f"mix={mix} k={k} approx={approx}: max|diff| = {diff}"
    )


# ---------------------------------------------------------------------------
# 4. randomized shapes and mixes, rectangular and square
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", list(range(24)))
def test_randomised_shapes_and_storage_mixes(seed):
    rng = np.random.default_rng(20260918 + seed)
    mixes = sorted(MIXES)
    mix = mixes[int(rng.integers(0, len(mixes)))]
    if seed % 2 == 0:                      # half the draws are SQUARE
        n = m = int(rng.integers(2, 9))
    else:
        n = int(rng.integers(2, 9))
        m = int(rng.integers(2, 9))
        while m == n:
            m = int(rng.integers(2, 9))
    for approx in (False, True):
        got, want, pre_d, _primal, ops = _run_face(n, m, mix, approx,
                                                   seed=seed)
        diff = _report(f"rand seed={seed} mix={mix} n={n} m={m} "
                       f"approx={approx}", got, want, pre_d,
                       extra=f"need_contract={ops.need_contract}")
        assert got.shape == want.shape, (
            f"seed={seed} mix={mix} n={n} m={m} approx={approx}: "
            f"{tuple(got.shape)} vs {tuple(want.shape)}")
        assert diff <= 1e-5, (
            f"seed={seed} mix={mix} n={n} m={m} approx={approx}: "
            f"max|diff| = {diff}")


# ---------------------------------------------------------------------------
# 5. the ticket's own shape, at the campaign's TransformerLM face
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("approx", [False, True])
def test_transformerlm_face_shape_and_value(approx):
    """``v`` is ``(32, 128)``, ``v.T`` is ``(128, 32)``, ``u`` is ``(3, 2)``.
    The nominal edge is ``(128, 32) + (3, 2)``; the broken engine stored
    ``(32, 128, ...)``."""
    got, want, pre_d, primal_shape, ops = _run_face(128, 32, "dense", approx,
                                                    seed=5)
    diff = _report(f"tlm approx={approx}", got, want, pre_d,
                   extra=f"need_contract={ops.need_contract}")
    assert tuple(got.shape) == (128, 32) + tuple(primal_shape)
    assert diff <= 1e-5


# ---------------------------------------------------------------------------
# 6. an ALIGNED identity still takes the shortcut, in every mix
# ---------------------------------------------------------------------------
def _aligned_identity(a, b, dtype=jnp.float32):
    return SparseTensor(
        out_dims=[DiagonalIndex(0, a, None, 2), DiagonalIndex(1, b, None, 3)],
        primal_dims=[DiagonalIndex(2, a, None, 0),
                     DiagonalIndex(3, b, None, 1)],
        val=None,
        dtype=dtype,
    )


@pytest.mark.parametrize("mix", sorted(MIXES))
@pytest.mark.parametrize("approx", [False, True])
def test_aligned_identity_still_passes_through(mix, approx):
    """No case that worked before may change: an aligned identity operand is
    still handed the other operand, bit for bit."""
    m, n = 5, 4
    post = _aligned_identity(m, n)
    pre, _primal = MIXES[mix](m, n, seed=13)
    ops = prepare_face_operands(post, pre, approx=approx)
    assert ops.need_contract is False
    res = contract_face_operands(ops).val
    got = np.asarray(res.dense())
    want = np.asarray(pre.dense())
    assert got.shape == want.shape
    assert np.array_equal(got, want)


# ---------------------------------------------------------------------------
# 7. THE ENGINE, on a SQUARE transpose, under the armed free order
#
# Sections 2 to 4 measure the face contraction directly. This section asks the
# same question of the whole builder. A SQUARE transpose is the silent half:
# the dropped permutation changes no shape, so the elimination runs to the end
# and the only evidence is the number. The approximation path is armed with an
# IDENTITY callable, which turns on the drain that materialises the crossed
# diagonal without changing one number, so the expected answer stays the exact
# Jacobian.
# ---------------------------------------------------------------------------
import jax

_SW = jnp.asarray(np.arange(12, dtype=np.float32).reshape(4, 3) / 11.0 + 0.1)
_SX = jnp.asarray(np.arange(12, dtype=np.float32).reshape(3, 4) / 13.0 - 0.3)
_SY = jnp.asarray(np.arange(24, dtype=np.float32).reshape(4, 6) / 23.0 + 0.2)


def _square_transposed_model(W, X):
    """``v`` is ``(4, 4)``, so ``v.T`` is ``(4, 4)`` -- a fixed point of the
    permutation. Nothing downstream can catch a dropped transpose here."""
    v = jnp.tanh(W @ X)
    return jnp.sum((v.T @ _SY) ** 2)


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
def test_armed_free_order_on_a_SQUARE_transpose(seed):
    from graphax import inline_call_primitives
    from graphax.incremental import IncrementalJaxpr

    args = (_SW, _SX)
    argnums = (0, 1)
    closed = jax.make_jaxpr(_square_transposed_model)(*args)
    jaxpr, consts = inline_call_primitives(closed.jaxpr, list(closed.literals))
    ij = IncrementalJaxpr(jaxpr, argnums, consts, list(args))
    for v in _order_for(jaxpr, ij.vo, seed):
        ij.eliminate(v, (_identity_rule,))
    outs, _labels = ij.jacobian_outputs(dense=True)
    res = ij.trace.to_jaxpr(list(outs), ij.dbg, ij.si)
    got = [np.asarray(o) for o in jax.core.eval_jaxpr(res[0], res[1], *args)]
    want = jax.jacrev(_square_transposed_model, argnums=argnums)(*args)
    worst = 0.0
    for g, w in zip(got, want):
        w = np.asarray(w)
        g = g.reshape(w.shape)
        worst = max(worst, float(np.max(np.abs(g - w))))
    print(f"A9A-ENGINE square seed={seed} max_abs_diff={worst:.6e}",
          flush=True)
    assert worst <= 1e-5, f"seed={seed} max|diff| = {worst}"
