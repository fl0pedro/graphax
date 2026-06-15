"""Named-jit activation dispatch.

``jax.nn`` activations like elu/selu/celu/leaky_relu/hard_tanh/sparse_plus/
sparse_sigmoid are ``@jit``-wrapped with no custom_jvp. graphax dispatches them
by their ``jit_p`` name (``primitives/custom.py``) to its own diagonal
Jacobian — the exact analytic subgradient — instead of differentiating the
select_n/max/min decomposition. Plain ``jax.nn.*`` is used directly (no wrapper).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jax.nn as jnn

import graphax.primitives.custom as act
from graphax import jacve, tree_allclose

_NAMED = ["elu", "selu", "celu", "leaky_relu", "hard_tanh",
          "sparse_plus", "sparse_sigmoid"]


@pytest.mark.parametrize("name", _NAMED)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_named_matches_jax(name, seed):
    fn = getattr(jnn, name)
    x = jax.random.normal(jax.random.PRNGKey(seed), (16,)) * 2.0
    ref = jax.jacfwd(fn, (0,))(x)
    for mode in ("fwd", "rev"):
        assert tree_allclose(jacve(fn, mode, (0,))(x), ref, rtol=1e-4, atol=1e-5)


def test_dispatch_actually_fires():
    # Poison a deriv: if graphax's named rule is used, the gradient reflects it;
    # if it silently fell back to differentiating the decomposition, it wouldn't.
    saved = act.ACTIVATION_DERIVS["elu"]
    act.ACTIVATION_DERIVS["elu"] = lambda x: jnp.full_like(x, 99.0)
    try:
        g = np.asarray(jacve(jnn.elu, "rev", (0,))(jnp.array([-1.0, 0.5, 2.0]))[0])
        assert np.allclose(np.diag(g), 99.0)
    finally:
        act.ACTIVATION_DERIVS["elu"] = saved


@pytest.mark.parametrize("fn", [
    lambda z: jnn.leaky_relu(z, 0.2),   # non-default negative_slope
    lambda z: jnn.elu(z, 2.0),          # non-default alpha
], ids=["leaky0.2", "elu_a2"])
def test_non_default_params_fall_back(fn):
    # The named rule covers defaults; a non-default static arg must fall back to
    # differentiating the jit body (which bakes the actual arg) -> still correct.
    x = jnp.array([-2.0, -0.5, 1.0, 3.0])
    assert tree_allclose(jacve(fn, "rev", (0,))(x), jax.jacfwd(fn, (0,))(x),
                         rtol=1e-4, atol=1e-5)


def test_anonymous_jit_still_inlines():
    # A jit whose name isn't in the database is inlined as before.
    f = lambda z: jax.jit(lambda y: jnp.sin(y) * 2.0)(z)
    x = jax.random.normal(jax.random.PRNGKey(0), (6,))
    assert tree_allclose(jacve(f, "rev", (0,))(x), jax.jacfwd(f, (0,))(x),
                         rtol=1e-4, atol=1e-5)
