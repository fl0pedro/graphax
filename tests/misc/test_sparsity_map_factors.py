"""Regression tests for per-vertex ``Diag`` transforms under ``jacve``.

These supersede the old ``sparsity_map`` factor tests: the
``apply_dynamic_sparsity`` machinery (with its ``factor`` sentinels ``-1`` =
gcd-collapse and ``0`` = drop-axes) was removed in favour of the explicit
``Diag(i, j, factor)`` micro-action, threaded per vertex via
``jacve(..., transforms=[(vertex_id, [Diag(i, j, factor), ...]), ...])``.

``Diag.factor`` is now an explicit positive block count (so a gcd-collapse is
just ``factor = gcd(N_i, N_j)`` passed explicitly):

* ``factor == 1``  — one block ⇒ dense ⇒ recognised as a no-op (``apply_diag``
  returns the tensor unchanged), so the call succeeds and stays finite.
* ``factor == K``  with ``K > 1`` and ``K | N_i`` and ``K | N_j`` — a real
  block-diagonal (``DiagonalIndex(size=K, block_size=N/K)``) with dedicated
  ``axis`` / ``block_axis``.
* ``factor == K`` that does *not* divide both dims — ``apply_diag`` raises and
  ``_eliminate_vertex`` skips that transform (best-effort), so the call still
  produces a finite Jacobian. (The old code "fell back to gcd"; the new code
  simply drops the non-fitting rule — same observable outcome: no crash.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax import jacve
from graphax.sparse.micro_actions import Diag


# A minimal real-world jaxpr that triggers all the matmul edge cases
# (vmapped neural network used by alphagrad's `VmappedNeuralNetwork` example).
def _vmapped_nn(x, y, W1, b1, W2, b2):
    a1 = jnp.tanh(x @ W1.T + b1)
    return 0.5 * (jnp.tanh(a1 @ W2.T + b2) - y) ** 2


def _make_args(seed=0, hidden=8, batch=16, in_dim=4, out_dim=4):
    """Tiny shapes — keep test runtime well under a second when graphax is
    in JAX cache; full mnist-style shapes are not needed since the matmul edge
    cases trigger on shape mismatches that scale-down preserves."""
    keys = jrand.split(jrand.PRNGKey(seed), 6)
    shapes = [
        (batch, in_dim),
        (batch, out_dim),
        (hidden, in_dim),
        (hidden,),
        (out_dim, hidden),
        (out_dim,),
    ]
    return [jrand.normal(k, s) for k, s in zip(keys, shapes)]


def _vmapped_fn():
    return jax.vmap(_vmapped_nn, in_axes=(0, 0, None, None, None, None))


def _flat_finite(out):
    leaves = jax.tree_util.tree_leaves(out)
    flat = jnp.concatenate([jnp.ravel(l) for l in leaves])
    return bool(jnp.all(jnp.isfinite(flat)))


# ---------------------------------------------------------------------------
# factor == 1: no-op semantics, no crash
# ---------------------------------------------------------------------------


def test_diag_factor_one_is_treated_as_noop():
    """``factor=1`` (one block ⇒ dense, no real sparsification) is recognised
    by ``apply_diag`` and returns the tensor unchanged — the call must succeed
    and produce a finite result."""
    args = _make_args()
    fn = _vmapped_fn()
    order = [11, 9, 10, 7, 12, 5]
    transforms = [(12, [Diag(1, 3, 1)])]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), transforms=transforms)(*args)
    assert _flat_finite(out)


# ---------------------------------------------------------------------------
# factor == K (K > 1, K | N): block-diagonal extraction, no crash
# ---------------------------------------------------------------------------


def test_diag_factor_two_block_diagonal_is_finite():
    """``Diag(1, 3, 2)`` on a size-4 dim produces ``DiagonalIndex(size=2,
    block=2)`` with a real block axis in the val."""
    args = _make_args(out_dim=4)  # 4 % 2 == 0
    fn = _vmapped_fn()
    order = [9, 7]
    transforms = [(9, [Diag(1, 3, 2)])]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), transforms=transforms)(*args)
    assert _flat_finite(out)


def test_diag_block_diagonal_then_more_transforms():
    """Block-diagonal extraction + downstream transforms — the post-extraction
    val has an extra physical axis; subsequent eliminations must still work."""
    args = _make_args(out_dim=4)
    fn = _vmapped_fn()
    order = [8, 4, 11, 9, 6, 3, 1, 10, 7, 2, 12, 5]
    transforms = [
        (9, [Diag(1, 3, 2)]),   # block-diagonal extraction
        (12, [Diag(0, 2, 2)]),  # subsequent block-diagonalise
        (5, [Diag(0, 2, 2)]),
    ]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), transforms=transforms)(*args)
    assert _flat_finite(out)


# ---------------------------------------------------------------------------
# factor == K that doesn't divide N: must NOT crash; the rule is skipped
# ---------------------------------------------------------------------------


def test_diag_non_divisor_factor_is_skipped():
    """``Diag(1, 3, 4)`` where 4 doesn't divide 10: ``apply_diag`` raises and
    ``_eliminate_vertex`` skips the transform (best-effort), so the call still
    produces a finite Jacobian — defence in depth for direct API callers."""
    args = _make_args(out_dim=10)
    fn = _vmapped_fn()
    order = [9, 7]
    transforms = [(9, [Diag(1, 3, 4)])]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), transforms=transforms)(*args)
    assert _flat_finite(out)


# ---------------------------------------------------------------------------
# multi-rule per vertex / multi-vertex: the alphagrad `_callback` shape
# ---------------------------------------------------------------------------


def test_diag_multi_rule_per_vertex_is_finite():
    args = _make_args()
    fn = _vmapped_fn()
    order = [8, 4, 11, 9, 6, 3, 1, 10, 7, 2, 12, 5]
    transforms = [
        (4, [Diag(0, 2, 2), Diag(1, 3, 2)]),
        (9, [Diag(0, 2, 2), Diag(1, 3, 2)]),
        (7, [Diag(0, 2, 2), Diag(1, 3, 2)]),
        (12, [Diag(0, 2, 2)]),
        (5, [Diag(0, 2, 2)]),
    ]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), transforms=transforms)(*args)
    assert _flat_finite(out)


def test_diag_transforms_list_form_accepted():
    """Public-API guard: jacve accepts the ``(vertex, [Diag(...)])`` list form
    across many vertices and stays finite."""
    args = _make_args()
    fn = _vmapped_fn()
    order = [8, 4, 11, 9, 6, 3, 1, 10, 7, 2, 12, 5]
    transforms = [(v, [Diag(0, 2, 2)]) for v in order if v not in (8, 10)]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), transforms=transforms)(*args)
    leaves = jax.tree_util.tree_leaves(out)
    assert leaves and _flat_finite(out)
