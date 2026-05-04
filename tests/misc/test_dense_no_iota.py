"""Bug: SparseTensor.dense() no longer takes `iota`; transforms still passed it.

The local refactor changed `dense(iota)` -> `dense()` (the SparseTensor now
constructs its own identity on demand). Several places in `core.py` and
`primitives/transforms.py` were inherited from upstream and still called
`pre.dense(iota)` / `post.dense(iota)`, raising

    TypeError: SparseTensor.dense() takes 1 positional argument but 2 were given

This test pins the new signature and exercises a transform that goes through
`dense()` to make sure no caller has crept back to the old form.
"""

import inspect

import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose
from graphax.sparse.tensor import SparseTensor


def test_dense_signature_takes_no_extra_args():
    sig = inspect.signature(SparseTensor.dense)
    # Only `self` — no iota / hard / etc. — so calling sites can't regress to
    # passing positional iota again without the test catching it.
    assert list(sig.parameters.keys()) == ["self"]


def test_reshape_jacobian_via_jacve_does_not_crash():
    """reshape's transform internally calls pre.dense() — exercise it end-to-end."""
    def f(x):
        return jnp.reshape(x, (6,)).sum()

    x = jnp.ones((2, 3))
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
    refres = jax.jit(jax.jacrev(f, argnums=(0,)))(x)
    assert bool(tree_allclose(veres, refres))


def test_slice_jacobian_via_jacve_does_not_crash():
    """slice's transform internally calls pre.dense() — exercise it end-to-end."""
    def f(x):
        return jax.lax.slice(x, (0,), (3,)).sum()

    x = jnp.ones((5,))
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
    refres = jax.jit(jax.jacrev(f, argnums=(0,)))(x)
    assert bool(tree_allclose(veres, refres))
