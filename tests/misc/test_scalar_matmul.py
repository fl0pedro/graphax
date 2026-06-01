"""Scalar `SparseTensor` composition under ``*`` (elementwise).

`@` (matmul) of two 0-rank SparseTensors is rejected — callers must use
``*``. Vertex elimination in ``core.py`` routes scalar Jacobian edges
through ``*`` (e.g. tan'(x) ∘ (-1)·sin'(x)), so the elementwise path must
preserve ``val`` and compose ``scalar_mult`` correctly.

These tests pin the elementwise-scalar invariants so vertex elimination's
chain rule stays correct across refactors.
"""

import jax.numpy as jnp
import pytest

from graphax.sparse.tensor import SparseTensor


def test_scalar_times_scalar_preserves_val():
    """`SparseTensor((), (), x) * SparseTensor((), (), y)` must equal x*y."""
    a = SparseTensor((), (), jnp.array(2.0))
    b = SparseTensor((), (), jnp.array(3.0))
    res = a * b
    assert res.val is not None, "scalar * scalar dropped val"
    assert float(res.val) == 6.0


def test_scalar_times_scalar_with_none_val_uses_one():
    """Identity-shaped scalar (val=None) should multiply as 1."""
    a = SparseTensor((), (), None)
    b = SparseTensor((), (), jnp.array(7.0))
    res = a * b
    assert res.val is not None
    assert float(res.val) == 7.0


def test_scalar_times_scalar_via_jacve():
    """End-to-end: a function that triggers scalar composition during
    vertex elimination. Tests the ``core.py`` scalar guard in
    ``_eliminate_vertex`` that routes scalar Jacobian edges through
    ``*`` instead of ``@``."""
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


def test_scalar_times_scalar_both_val_none_composes_correctly():
    """Both operands have ``val=None`` and non-1 ``scalar_mult`` — the
    composed effective value must be ``lhs.scalar_mult * rhs.scalar_mult``.

    Elementwise ``*`` materializes the result (``val`` becomes a concrete
    array, ``scalar_mult`` resets to 1), unlike the old ``_scalar_matmul``
    which kept ``val=None`` and rode the multiplier on ``scalar_mult``.
    The effective value (``val * scalar_mult``, treating ``val=None`` as 1)
    is what matters here; the val=None Kronecker-identity preservation is
    OPTIMIZATION_PLAN issue 2 (a separate perf optimization)."""
    a = SparseTensor((), (), None, scalar_mult=jnp.array(2.0))
    b = SparseTensor((), (), None, scalar_mult=jnp.array(3.0))
    res = a * b
    effective = (1.0 if res.val is None else float(res.val)) * float(res.scalar_mult)
    assert effective == 6.0


def test_scalar_at_scalar_rejected():
    """Scalar @ scalar is no longer supported on ``matmul``; callers must
    use ``*``. Locks in the new contract so a future regression that
    silently re-enables a scalar matmul path fails loudly."""
    a = SparseTensor((), (), jnp.array(2.0))
    b = SparseTensor((), (), jnp.array(3.0))
    with pytest.raises(ValueError, match="0-rank SparseTensors"):
        a @ b
