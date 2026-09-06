"""The third emission of the frame contraction (ticket dsnn-3qm.28.4).

``GRAPHAX_TILED_MULREDUCE`` replaces the frame's ``dot_general`` with
``sum(lhs * rhs, axis=contracted)``. It is a race-only knob and it is OFF by
default. Two things are pinned here:

1. ``_gx_mul_reduce`` returns exactly what ``lax.dot_general`` returns, for
   every dimension-number shape the frame emits (batch axes, contracted axes,
   free axes on either side, and the no-contraction case).
2. With the knob off, the emitted jaxpr of a real ``jacve`` is unchanged. That
   is the flag-off identity bit.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.ops.matmul import _gx_mul_reduce

DIMS = [
    # (lhs shape, rhs shape, dimension_numbers)
    ((4, 3, 5), (4, 5, 7), (((2,), (1,)), ((0,), (0,)))),      # batch + contract
    ((6, 5), (5, 7), (((1,), (0,)), ((), ()))),                # plain 2-D dot
    ((2, 3, 4, 5), (2, 4, 5, 6), (((2, 3), (1, 2)), ((0,), (0,)))),  # two contracted
    ((3, 4), (3, 4), (((1,), (1,)), ((0,), (0,)))),            # no free axis
    ((2, 3, 4), (2, 4), (((2,), (1,)), ((0,), (0,)))),         # rhs has no free axis
    ((2, 3), (2, 4), (((), ()), ((0,), (0,)))),                # nothing contracted
]


@pytest.mark.parametrize("lhs_shape,rhs_shape,dims", DIMS)
def test_mul_reduce_equals_dot_general(lhs_shape, rhs_shape, dims):
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal(lhs_shape), dtype=jnp.float32)
    b = jnp.asarray(rng.standard_normal(rhs_shape), dtype=jnp.float32)
    want = jax.lax.dot_general(a, b, dims)
    got = _gx_mul_reduce(a, b, dims)
    assert want.shape == got.shape
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-5)


def test_knob_is_off_by_default():
    from graphax.sparse.ops.matmul import _mul_reduce_enabled
    saved = os.environ.pop("GRAPHAX_TILED_MULREDUCE", None)
    try:
        assert _mul_reduce_enabled() is False
    finally:
        if saved is not None:
            os.environ["GRAPHAX_TILED_MULREDUCE"] = saved


def test_flag_off_leaves_the_frame_unchanged():
    """With the knob absent and with it set to 0, the same jaxpr comes out."""
    from graphax import jacve

    def f(x, y):
        return jnp.sum(jnp.sin(x @ y))

    x = jnp.asarray(np.random.default_rng(1).standard_normal((4, 6)), jnp.float32)
    y = jnp.asarray(np.random.default_rng(2).standard_normal((6, 3)), jnp.float32)
    order = "rev"
    saved = os.environ.pop("GRAPHAX_TILED_MULREDUCE", None)
    try:
        base = str(jax.make_jaxpr(jacve(f, order, argnums=(0, 1)))(x, y))
        os.environ["GRAPHAX_TILED_MULREDUCE"] = "0"
        off = str(jax.make_jaxpr(jacve(f, order, argnums=(0, 1)))(x, y))
    finally:
        os.environ.pop("GRAPHAX_TILED_MULREDUCE", None)
        if saved is not None:
            os.environ["GRAPHAX_TILED_MULREDUCE"] = saved
    assert base == off


def test_knob_on_gives_the_same_gradient():
    from graphax import jacve

    def f(x, y):
        return jnp.sum(jnp.sin(x @ y))

    x = jnp.asarray(np.random.default_rng(3).standard_normal((4, 6)), jnp.float32)
    y = jnp.asarray(np.random.default_rng(4).standard_normal((6, 3)), jnp.float32)
    saved = os.environ.pop("GRAPHAX_TILED_MULREDUCE", None)
    try:
        want = jacve(f, "rev", argnums=(0, 1))(x, y)
        os.environ["GRAPHAX_TILED_MULREDUCE"] = "1"
        got = jacve(f, "rev", argnums=(0, 1))(x, y)
    finally:
        os.environ.pop("GRAPHAX_TILED_MULREDUCE", None)
        if saved is not None:
            os.environ["GRAPHAX_TILED_MULREDUCE"] = saved
    for a, b in zip(jax.tree_util.tree_leaves(want), jax.tree_util.tree_leaves(got)):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), atol=1e-5)
