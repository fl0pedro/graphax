"""Bug: `concat([pl, pl, pl])` (same primal in multiple slots) lost contributions.

When the same primal feeds multiple slots of one concatenate (e.g.
`concat([pl, pl, pl])` where `pl` has shape (1,) and the output has shape
(3,)), the graph stores a single edge per (invar, outvar) pair. The
elemental for that invar must therefore represent the *sum* of all slot
contributions.

The bug: pre-port, the SparseTensor wrapping the concatenate transform was
constructed positionally, so the transform never landed in pre_transforms.
The resulting Jacobian shape was then determined by whatever the surrounding
code did, not by the actual concat structure.

This test exercises the same-primal-multiple-slots path end-to-end.
"""

import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose


def test_two_concatenates_same_scalar_primal():
    """The exact reproduction from concatenate_test.py — pinned here too."""

    def fn(v):
        pl = jnp.sum(v, keepdims=True)             # [1]
        _p = jnp.concatenate([pl, jnp.zeros(2, dtype=jnp.float32)])  # [3]
        dF = jnp.concatenate([pl, pl, pl])         # [3] — same primal in 3 slots
        return _p - dF                             # [3]

    v = jnp.array([0.1, 0.2, 0.3])
    veres = jax.jit(jacve(fn, order="rev", argnums=(0,)))(v)
    refres = jax.jit(jax.jacrev(fn, argnums=(0,)))(v)
    assert bool(tree_allclose(veres, refres))


def test_concat_same_vector_three_times():
    """concat([x, x, x]) — same vector primal in 3 slots."""

    def fn(x):
        return jnp.concatenate([x, x, x]).sum()

    x = jnp.array([1.0, 2.0, 3.0])
    veres = jax.jit(jacve(fn, order="rev", argnums=(0,)))(x)
    refres = jax.jit(jax.jacrev(fn, argnums=(0,)))(x)
    assert bool(tree_allclose(veres, refres))
