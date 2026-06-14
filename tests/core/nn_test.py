"""graphax.nn activations carry explicit custom_jvp rules (the jax.nn versions
have none). graphax honors custom_jvp, so it differentiates them via the exact
analytic subgradient. Values must equal jax.nn; gradients must equal
jax.jacfwd(jax.nn.*) at points of differentiability, in BOTH elimination orders.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jax.nn as jnn

import graphax.nn as gnn
from graphax import jacve, tree_allclose

_PAIRS = [
    ("elu", gnn.elu, jnn.elu),
    ("selu", gnn.selu, jnn.selu),
    ("celu", gnn.celu, jnn.celu),
    ("leaky_relu", gnn.leaky_relu, jnn.leaky_relu),
    ("hard_tanh", gnn.hard_tanh, jnn.hard_tanh),
    ("sparse_plus", gnn.sparse_plus, jnn.sparse_plus),
    ("sparse_sigmoid", gnn.sparse_sigmoid, jnn.sparse_sigmoid),
]


@pytest.mark.parametrize("name,gfn,jfn", _PAIRS, ids=[p[0] for p in _PAIRS])
def test_value_matches_jax_nn(name, gfn, jfn):
    x = jax.random.normal(jax.random.PRNGKey(0), (16,))
    assert np.allclose(np.asarray(gfn(x)), np.asarray(jfn(x)), atol=1e-6)


@pytest.mark.parametrize("name,gfn,jfn", _PAIRS, ids=[p[0] for p in _PAIRS])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grad_matches_jax_nn(name, gfn, jfn, seed):
    # random draws avoid the measure-zero kinks (where the analytic subgradient
    # may differ from jax.nn's decomposition by definition).
    x = jax.random.normal(jax.random.PRNGKey(seed), (16,)) * 2.0
    ref = jax.jacfwd(jfn, (0,))(x)
    for mode in ("fwd", "rev"):
        got = jacve(gfn, mode, (0,))(x)
        assert tree_allclose(got, ref, rtol=1e-4, atol=1e-5), f"{name} {mode}"


def test_composed_network():
    # custom_jvp activations composed (and reused) through a small graph.
    x = jax.random.normal(jax.random.PRNGKey(3), (8,))

    def net(z):
        return jnp.sum(gnn.elu(gnn.leaky_relu(2 * z)) + gnn.selu(z) + gnn.celu(z))

    assert tree_allclose(jacve(net, "rev", (0,))(x), jax.jacrev(net, (0,))(x),
                         rtol=1e-4, atol=1e-5)
    assert tree_allclose(jacve(net, "fwd", (0,))(x), jacve(net, "rev", (0,))(x),
                         rtol=1e-4, atol=1e-5)
