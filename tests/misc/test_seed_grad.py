"""Matrix-free reverse-mode gradient (graphax.seed, Tier 0) vs jax.grad.

The seed engine propagates a cotangent VECTOR, so reshape/transpose/broadcast
are free relabels of the cotangent — no Jacobian is ever materialised. This is
what lets the ConvNet gradient avoid the 4608x4608 densification that the
vertex-elimination (jacve) reverse path hits. Each case checks exact agreement
with jax.grad; the vision-model cases also exercise gelu/softmax/conv/vmap.
"""
import jax
import jax.numpy as jnp
import jax.random as jrand
import pytest

import graphax.seed as gs


def _close(a, b, atol=1e-4):
    return all(bool(jnp.allclose(x, y, atol=atol)) for x, y in zip(a, b))


def test_reshape_matmul_elementwise():
    """The conv-head pattern in miniature: elementwise -> flatten -> matmul."""
    def f(W, x):
        return jnp.sum(W @ jnp.tanh(x).reshape(-1, 1))
    key = jrand.PRNGKey(0)
    x = jrand.normal(key, (2, 3)); W = jrand.normal(key, (4, 6))
    g_seed = gs.grad(f, argnums=(0, 1))(W, x)
    g_jax = jax.grad(f, argnums=(0, 1))(W, x)
    assert _close(g_seed, g_jax)


def test_scalar_argnum_returns_bare_array():
    f = lambda W, x: jnp.sum(jnp.sin(W @ x))
    key = jrand.PRNGKey(1)
    W = jrand.normal(key, (3, 4)); x = jrand.normal(key, (4,))
    g = gs.grad(f, argnums=0)(W, x)
    assert g.shape == (3, 4)
    assert bool(jnp.allclose(g, jax.grad(f, argnums=0)(W, x), atol=1e-5))


def test_jit_matches():
    f = lambda W, x: jnp.sum(jnp.tanh(W @ x))
    key = jrand.PRNGKey(2)
    W = jrand.normal(key, (5, 6)); x = jrand.normal(key, (6,))
    gj = jax.jit(gs.grad(f, argnums=(0, 1)))(W, x)
    assert _close(gj, jax.grad(f, argnums=(0, 1))(W, x))


@pytest.mark.parametrize("name", ["ConvNet", "MoE", "ViT"])
def test_vision_models_match_jax_grad(name):
    from graphax.examples import vision
    key = jrand.PRNGKey(0)
    x = jrand.normal(key, (784,)); y = jrand.normal(key, (10,))
    fn = getattr(vision, name)
    w = {"ConvNet": vision.conv_weights, "MoE": vision.moe_weights,
         "ViT": vision.vit_weights}[name](key)
    args = (x, y, *w)
    an = tuple(range(2, len(args)))
    loss = lambda *a: jnp.sum(fn(*a))
    g_seed = gs.grad(loss, argnums=an)(*args)
    g_jax = jax.grad(loss, argnums=an)(*args)
    assert _close(g_seed, g_jax, atol=1e-3)
