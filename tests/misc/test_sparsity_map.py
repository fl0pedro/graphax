"""Feature pin: per-vertex Jacobian ``transforms`` plumb through ``jacve``.

``transforms`` is the successor to the old ``sparsity_map`` (which fed the
removed ``apply_dynamic_sparsity``). It is
``Sequence[(vertex_id, Sequence[Diag | Compress | Callable])]``: for each
elimination step on ``vertex_id`` the listed transforms are applied IN ORDER to
the composed edge Jacobian. A ``Diag(i, j, factor)`` block-diagonalises the
logical-index pair ``(i, j)`` with block count ``factor``; a transform that
doesn't fit the edge geometry is skipped (best-effort). This file pins the
plumbing:

1. ``transforms=None`` (default) is a no-op — same Jacobian as without.
2. ``transforms=()`` (empty) is also a no-op.
3. The parameter is accepted on ``jacve``.

Verifying that the transforms produce *correct* sparser Jacobians is the
responsibility of the sparse_tensor / micro_action tests; here we only pin the
plumbing.
"""

import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose


def _f(x, y):
    return jnp.sin(x * y).sum()


def test_transforms_none_is_noop():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    plain = jax.jit(jacve(_f, order="rev", argnums=(0, 1)))(x, y)
    none = jax.jit(jacve(_f, order="rev", argnums=(0, 1), transforms=None))(x, y)
    assert bool(tree_allclose(plain, none))


def test_transforms_empty_is_noop():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    plain = jax.jit(jacve(_f, order="rev", argnums=(0, 1)))(x, y)
    empty = jax.jit(jacve(_f, order="rev", argnums=(0, 1), transforms=()))(x, y)
    assert bool(tree_allclose(plain, empty))


def test_transforms_accepted_in_jacve_signature():
    """If someone reverts the signature change, this test fails immediately."""
    import inspect

    sig = inspect.signature(jacve)
    assert "transforms" in sig.parameters
