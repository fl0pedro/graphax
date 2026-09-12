"""A rank-0 operand in ``matmul`` RAISES (ticket dsnn-3qm.68).

History. The tiled engine used to return a size-1 SLICE of the tensor operand
for ``X @ scalar``: ``_build_pair_dims`` built the ``spatial_primal_lhs``
pairing's one-sided primal dim from the LHS half of the internal grid
(``final_l`` / ``la``), which is always 1 for that pairing. That was a wrong
Jacobian in exact AD. The first fix rerouted ``X @ scalar`` into
``scalar_mult`` silently, which produced the right number.

Owner ruling 2026-09-07 reverses that: a scalar has no axes, so it cannot be
contracted, and every module with a real matmul rejects it. Silently accepting
``X @ scalar`` hides the call site that is wrong. ``matmul`` raises
``ScalarMatmul``; ``scale_by_scalar`` is the operation the caller means.
"""

import jax.numpy as jnp
import pytest

from graphax.sparse.ops.matmul import ScalarMatmul, matmul, scale_by_scalar
from graphax.sparse.ops.utils import _arr2st
from graphax.sparse.tensor import SparseTensor


def _x():
    return _arr2st(jnp.arange(6.0).reshape(2, 3), out_ndim=1)


def _scalar(v=4.0):
    return SparseTensor((), (), jnp.array(v))


def test_tensor_at_scalar_raises():
    with pytest.raises(ScalarMatmul) as exc:
        matmul(_x(), _scalar())
    msg = str(exc.value)
    assert "scale, not a matmul" in msg
    assert "scale_by_scalar" in msg


def test_scalar_at_tensor_raises():
    with pytest.raises(ScalarMatmul):
        matmul(_scalar(), _x())


def test_scalar_at_scalar_raises():
    """Two rank-0 operands have no axes between them either. This used to be
    routed through ``*`` under GRAPHAX_SEED_VERTICES_SCALAR_MM."""
    with pytest.raises(ScalarMatmul):
        matmul(_scalar(2.0), _scalar(3.0))


def test_scale_by_scalar_is_the_operation_the_caller_means():
    """The fold is public and gives the mathematically correct answer, with
    the tensor's own shape preserved -- the original bug was a size-1 slice."""
    X, s = _x(), _scalar()
    out = scale_by_scalar(X, s)
    expected = 4.0 * X.dense()
    assert out.dense().shape == expected.shape
    assert jnp.allclose(out.dense(), expected)


def test_scale_by_scalar_touches_no_values():
    """The scale is deferred into ``scalar_mult``: ``val``, ``fill_value`` and
    dims come through untouched, so no per-element work happens now."""
    X, s = _x(), _scalar()
    out = scale_by_scalar(X, s)
    assert out.dims == X.dims
    assert jnp.array_equal(out.val, X.val)
    assert out.fill_value == X.fill_value


def test_a_second_scale_does_not_grow_scalar_mult():
    """Regression: reading the scalar through ``_stored_val()`` returned a flat
    ``(N,)`` array, so each fold added a size-1 axis and two folds in one chain
    compounded to ``(1, 1)``, breaking a downstream broadcast."""
    out = scale_by_scalar(scale_by_scalar(_x(), _scalar(2.0)), _scalar(3.0))
    assert jnp.ndim(out.scalar_mult) == 0
    assert jnp.allclose(out.dense(), 6.0 * _x().dense())
