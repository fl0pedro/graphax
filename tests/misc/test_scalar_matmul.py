"""Bug: scalar @ scalar in `SparseTensor.__matmul__` dropped `val` -> None.

`matmul(lhs, rhs)` where both lhs and rhs have no out_dims/primal_dims (pure
scalar SparseTensors) was hitting `_build_matmul_topology` with no pairs,
producing a result with `val=None`. Vertex elimination uses `_post_val @
_pre_val` to compose two scalar edges (e.g. tan'(x) and (-1)·sin'(x)), so the
returned tensor silently lost its value and downstream Jacobians collapsed to
the structural identity.

Caught by the test_Simple example which silently produced
((8., 6.), (-2.34, -2.34)) instead of the correct
((0.674, 0.482), (14.77, 10.55)).
"""

import jax.numpy as jnp

from graphax.sparse.tensor import SparseTensor


def test_scalar_at_scalar_preserves_val():
    """`SparseTensor((), (), x) @ SparseTensor((), (), y)` must equal x*y."""
    a = SparseTensor((), (), jnp.array(2.0))
    b = SparseTensor((), (), jnp.array(3.0))
    res = a @ b
    assert res.val is not None, "scalar @ scalar dropped val"
    assert float(res.val) == 6.0


def test_scalar_at_scalar_with_none_val_uses_one():
    """Identity-shaped scalar (val=None) should multiply as 1."""
    a = SparseTensor((), (), None)
    b = SparseTensor((), (), jnp.array(7.0))
    res = a @ b
    assert res.val is not None
    assert float(res.val) == 7.0


def test_scalar_at_scalar_via_jacve():
    """End-to-end: a function that triggers scalar @ scalar during elimination."""
    import jax
    from graphax import jacve, tree_allclose

    def f(x, y):
        z = x * y                # scalar
        w = jnp.sin(z)           # scalar elemental cos(z)
        return w + z, jnp.log(w)  # log(w) elemental 1/w (also scalar)

    x = jnp.array(5.0)
    y = jnp.array(7.0)
    veres = jax.jit(jacve(f, order="rev", argnums=(0, 1)))(x, y)
    refres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)
    assert bool(tree_allclose(veres, refres))


def test_scalar_at_scalar_both_val_none_composes_scalar_mult():
    """Both operands have ``val=None`` and non-1 ``scalar_mult`` — the
    composed multiplier must be ``lhs.scalar_mult * rhs.scalar_mult``,
    not silently reset to 1. Vertex elimination chains structural-identity
    Jacobians (e.g. ``-1 * sin'(x)`` ∘ another scalar Jacobian) this way,
    and dropping the multipliers makes the chained Jacobian wrong by a
    factor of the missing scalars."""
    a = SparseTensor((), (), None, scalar_mult=jnp.array(2.0))
    b = SparseTensor((), (), None, scalar_mult=jnp.array(3.0))
    res = a @ b
    # Composed effective scalar = 2.0 * 3.0 = 6.0; with val=None the value
    # rides entirely on ``scalar_mult``.
    assert res.val is None
    assert float(res.scalar_mult) == 6.0
