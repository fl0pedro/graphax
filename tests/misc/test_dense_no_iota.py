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
    # No positional iota -- callers can't regress to `dense(iota)`. The only
    # extra parameter is the deliberate `keep_quantization` keyword (quant
    # dtype restructure); anything else creeping in still fails here.
    assert list(sig.parameters.keys()) == ["self", "keep_quantization"]


def test_reshape_jacobian_via_jacve_does_not_crash():
    """reshape's transform internally calls pre.dense() — exercise it end-to-end.

    Used to xfail: the scalar-output (``.sum()``) case is ``X @ scalar``,
    which the tiled engine's ``_build_pair_dims`` collapsed to a size-1 slice
    of X instead of the scaled tensor (dsnn-3qm.68, finding 61 verdict 6).
    Fixed in ``_build_pair_dims`` and, independently, by ``matmul`` routing
    any single-0-rank-operand contraction to an elementwise scale."""
    def f(x):
        return jnp.reshape(x, (6,)).sum()

    x = jnp.ones((2, 3))
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
    refres = jax.jit(jax.jacrev(f, argnums=(0,)))(x)
    assert bool(tree_allclose(veres, refres))


def test_slice_jacobian_via_jacve_does_not_crash():
    """slice's transform internally calls pre.dense() — exercise it end-to-end.

    Used to xfail for the same rank-0-operand reason as the reshape test
    above (dsnn-3qm.68)."""
    def f(x):
        return jax.lax.slice(x, (0,), (3,)).sum()

    x = jnp.ones((5,))
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
    refres = jax.jit(jax.jacrev(f, argnums=(0,)))(x)
    assert bool(tree_allclose(veres, refres))
