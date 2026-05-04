"""Feature pin: `sparsity_map` plumbs through `jacve` and `apply_dynamic_sparsity`.

`sparsity_map` is `Sequence[(vertex_id, ((idx1, idx2[, factor]), ...))]`. For
each elimination step on `vertex_id`, the listed dimension pairs of the
composed edge are forced sparse via `apply_dynamic_sparsity`. The pin checks:

1. `sparsity_map=None` (default) is a no-op — same Jacobian as without.
2. `sparsity_map=()` (empty tuple) is also a no-op.
3. The parameter is accepted on `jacve` and threaded through without crashing.

Verifying that `apply_dynamic_sparsity` produces *correct* sparser Jacobians
is the responsibility of the sparse_tensor tests; here we're just pinning the
plumbing.
"""

import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose


def _f(x, y):
    return jnp.sin(x * y).sum()


def test_sparsity_map_none_is_noop():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    plain = jax.jit(jacve(_f, order="rev", argnums=(0, 1)))(x, y)
    none = jax.jit(jacve(_f, order="rev", argnums=(0, 1), sparsity_map=None))(x, y)
    assert bool(tree_allclose(plain, none))


def test_sparsity_map_empty_is_noop():
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    plain = jax.jit(jacve(_f, order="rev", argnums=(0, 1)))(x, y)
    empty = jax.jit(jacve(_f, order="rev", argnums=(0, 1), sparsity_map=()))(x, y)
    assert bool(tree_allclose(plain, empty))


def test_sparsity_map_accepted_in_jacve_signature():
    """If someone reverts the signature change, this test fails immediately."""
    import inspect

    sig = inspect.signature(jacve)
    assert "sparsity_map" in sig.parameters
