"""Regression tests for `sparsity_map` factor handling under `jacve`.

The companion test in `test_sparsity_map.py` only pins the no-op cases. This
file covers the *interesting* paths that the alphagrad `_callback` actually
relies on after the autoreg policy started picking arbitrary factors.

Status today (after the `apply_dynamic_sparsity` rewrite):

* ``factor == -1``  — gcd-collapse. Single- and multi-rule per vertex both
  produce finite Jacobians; this was already true and is pinned to catch
  regressions.
* ``factor == 0``   — drop axes (legacy behaviour preserved).
* ``factor == 1``   — recognised as a no-op (one block ⇒ dense). The rule
  is silently dropped instead of producing a degenerate
  ``DiagonalIndex(size=1, block_size=N)`` that the matmul cannot consume.
* ``factor == K``   with ``K > 1`` and ``K | N1`` and ``K | N2`` — produces
  a real block-diagonal (``DiagonalIndex(size=K, block_size=N/K)``) with both
  ``axis`` and ``block_axis`` pointing to dedicated physical axes; the val
  is reshaped to expose the block axis.
* ``factor == K`` that does *not* divide both dims — falls back to the
  ``factor == -1`` (gcd) extraction so the call still succeeds. The
  alphagrad env masks these out at policy time, but the fallback keeps
  direct API callers safe too.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax import jacve


# A minimal real-world jaxpr that triggers all the matmul edge cases
# (vmapped neural network used by alphagrad's `VmappedNeuralNetwork` example).
def _vmapped_nn(x, y, W1, b1, W2, b2):
    a1 = jnp.tanh(x @ W1.T + b1)
    return 0.5 * (jnp.tanh(a1 @ W2.T + b2) - y) ** 2


def _make_args(seed=0, hidden=8, batch=16, in_dim=4, out_dim=4):
    """Tiny shapes — keep test runtime well under a second when graphax is
    in JAX cache; full mnist-style shapes (784/128/10) are not needed since
    the matmul bug triggers on shape mismatches that scale-down preserves."""
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
# factor=-1 paths (multi-rule per vertex) must keep working
# ---------------------------------------------------------------------------


def test_factor_minus_one_single_rule_is_finite():
    args = _make_args()
    fn = _vmapped_fn()
    order = [8, 4, 11, 9, 6, 3, 1, 10, 7, 2, 12, 5]
    sparsity_map = [
        (4, ((1, 2, -1),)),
        (11, ((0, 2, -1),)),
        (9, ((0, 2, -1),)),
        (6, ((0, 2, -1),)),
        (3, ((0, 2, -1),)),
        (1, ((0, 3, -1),)),
        (7, ((0, 3, -1),)),
        (2, ((1, 2, -1),)),
        (12, ((0, 2, -1),)),
        (5, ((0, 2, -1),)),
    ]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), sparsity_map=sparsity_map)(*args)
    assert _flat_finite(out)


def test_factor_minus_one_multi_rule_is_finite():
    args = _make_args()
    fn = _vmapped_fn()
    order = [8, 4, 11, 9, 6, 3, 1, 10, 7, 2, 12, 5]
    sparsity_map = [
        (4, ((0, 2, -1), (1, 3, -1))),
        (11, ((0, 2, -1),)),
        (9, ((0, 2, -1), (1, 3, -1))),
        (1, ((0, 3, -1),)),
        (7, ((0, 2, -1), (1, 3, -1))),
        (12, ((0, 2, -1), (1, 3, -1))),
        (5, ((0, 2, -1),)),
    ]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), sparsity_map=sparsity_map)(*args)
    assert _flat_finite(out)


# ---------------------------------------------------------------------------
# factor=1: no-op semantics, no crash
# ---------------------------------------------------------------------------


def test_factor_one_is_treated_as_noop():
    """Factor=1 used to crash with a duplicate-source transpose perm in
    `_prepare_physical_array`. The fix recognises factor=1 (one block ⇒
    dense, no real sparsification) and silently drops the rule — so the
    call must succeed *and* produce a finite result."""
    args = _make_args()
    fn = _vmapped_fn()
    order = [11, 9, 10, 7, 12, 5]
    sm = [(12, ((1, 3, 1),))]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), sparsity_map=sm)(*args)
    assert _flat_finite(out)


# ---------------------------------------------------------------------------
# factor=K (K>1, K | N): block-diagonal extraction, no crash
# ---------------------------------------------------------------------------


def test_factor_two_block_diagonal_is_finite():
    """Factor=2 on a size-4 dim produces ``DiagonalIndex(size=2, block=2)``
    with a real block axis in the val. Used to crash with an "Incompatible
    shapes for broadcasting" error in `_prepare_contraction_views`."""
    args = _make_args(out_dim=4)  # 4 % 2 == 0
    fn = _vmapped_fn()
    order = [9, 7]
    sm = [(9, ((1, 3, 2),))]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), sparsity_map=sm)(*args)
    assert _flat_finite(out)


def test_factor_two_block_diagonal_then_more_rules():
    """Factor=2 + downstream rules — the post-block-diagonal val has an
    extra physical axis. Subsequent eliminations must still work."""
    args = _make_args(out_dim=4)
    fn = _vmapped_fn()
    order = [8, 4, 11, 9, 6, 3, 1, 10, 7, 2, 12, 5]
    sm = [
        (9, ((1, 3, 2),)),         # block-diagonal extraction
        (12, ((0, 2, -1),)),       # subsequent gcd-collapse
        (5, ((0, 2, -1),)),
    ]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), sparsity_map=sm)(*args)
    assert _flat_finite(out)


# ---------------------------------------------------------------------------
# factor=K that doesn't divide N: must NOT crash; falls back to gcd
# ---------------------------------------------------------------------------


def test_factor_four_on_size_ten_falls_back():
    """Factor=4 doesn't divide 10. Used to crash with a 'logical_size'
    mismatch in matmul. Now `apply_dynamic_sparsity` recognises the bad
    factor and falls back to factor=-1 (gcd) so the call still produces a
    finite Jacobian — defence in depth even if the env mask glitches."""
    args = _make_args(out_dim=10)
    fn = _vmapped_fn()
    order = [9, 7]
    sm = [(9, ((1, 3, 4),))]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), sparsity_map=sm)(*args)
    assert _flat_finite(out)


# ---------------------------------------------------------------------------
# factor=0: drop axes, legacy behaviour preserved
# ---------------------------------------------------------------------------


def test_factor_zero_is_finite():
    """Factor=0 zeros the pair out (drops the axes). Should never crash
    and should produce a finite (likely zero-heavy) Jacobian."""
    args = _make_args()
    fn = _vmapped_fn()
    order = [11, 9, 10, 7, 12, 5]
    sm = [(12, ((1, 3, 0),))]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), sparsity_map=sm)(*args)
    assert _flat_finite(out)


# ---------------------------------------------------------------------------
# Public-API guard: jacve must accept the 3-tuple form
# ---------------------------------------------------------------------------


def test_sparsity_map_three_tuple_form_accepted():
    args = _make_args()
    fn = _vmapped_fn()
    order = [8, 4, 11, 9, 6, 3, 1, 10, 7, 2, 12, 5]
    sparsity_map = [(v, ((0, 2, -1),)) for v in order if v != 8 and v != 10]
    out = jacve(fn, order, argnums=(2, 3, 4, 5), sparsity_map=sparsity_map)(*args)
    leaves = jax.tree_util.tree_leaves(out)
    assert leaves
