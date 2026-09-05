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

import pytest

import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose
from graphax.sparse.tensor import SparseTensor


def test_dense_signature_takes_no_extra_args():
    sig = inspect.signature(SparseTensor.dense)
    # No positional iota -- callers can't regress to `dense(iota)`. The only
    # extra parameter is the deliberate `keep_quantization` keyword (quant
    # dtype restructure); anything else creeping in still fails here.
    assert list(sig.parameters.keys()) == ["self", "keep_quantization"]


# The strict xfail here is GONE (ticket dsnn-3qm.67): the scalar-output
# micro-case is the "X @ scalar" contraction whose one-sided primal dims
# the tiled path used to collapse to size 1 (finding 61, verdict 6). With
# _build_pair_dims reading spatial_primal_lhs from the rhs half of the
# grid, the Jacobian is right and this test passes.
def test_reshape_jacobian_via_jacve_does_not_crash():
    """reshape's transform internally calls pre.dense() — exercise it end-to-end."""
    def f(x):
        return jnp.reshape(x, (6,)).sum()

    x = jnp.ones((2, 3))
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
    refres = jax.jit(jax.jacrev(f, argnums=(0,)))(x)
    assert bool(tree_allclose(veres, refres))


# The strict xfail here is GONE (ticket dsnn-3qm.67): the scalar-output
# micro-case is the "X @ scalar" contraction whose one-sided primal dims
# the tiled path used to collapse to size 1 (finding 61, verdict 6). With
# _build_pair_dims reading spatial_primal_lhs from the rhs half of the
# grid, the Jacobian is right and this test passes.
def test_slice_jacobian_via_jacve_does_not_crash():
    """slice's transform internally calls pre.dense() — exercise it end-to-end."""
    def f(x):
        return jax.lax.slice(x, (0,), (3,)).sum()

    x = jnp.ones((5,))
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
    refres = jax.jit(jax.jacrev(f, argnums=(0,)))(x)
    assert bool(tree_allclose(veres, refres))
