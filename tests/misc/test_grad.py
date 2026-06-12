"""``graphax.grad`` / ``graphax.value_and_grad`` — jax.grad analogues via
vertex elimination.

Pins: (1) numerical agreement with ``jax.grad`` / ``jax.value_and_grad`` for
"rev" and "fwd" elimination (and under jit); (2) jax.grad argnums conventions
(bare array for int argnums, tuple for a sequence); (3) cross-country (custom
vertex order) gradients agree with jax.grad; (4) per-vertex ``transforms``
apply during the accumulation (a doubling transform on the only intermediate
vertex exactly doubles the gradient — the structured-approximation hook
jax.grad cannot express); (5) the scalar-output / inexact-dtype guards; (6)
``count_ops`` returns the accounting aux.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import graphax


def _f2(x, y):
    return jnp.sum(jnp.sin(x) * y)


_x = jnp.array([0.1, 0.7, -0.4, 1.3])
_y = jnp.array([1.0, -2.0, 0.5, 3.0])


@pytest.mark.parametrize("order", ["rev", "fwd"])
def test_grad_matches_jax_grad(order):
    g = graphax.grad(_f2, order, argnums=(0, 1))(_x, _y)
    ref = jax.grad(_f2, argnums=(0, 1))(_x, _y)
    assert graphax.tree_allclose(g, ref)


def test_grad_under_jit():
    g = jax.jit(graphax.grad(_f2, "rev", argnums=(0, 1)))(_x, _y)
    ref = jax.grad(_f2, argnums=(0, 1))(_x, _y)
    assert graphax.tree_allclose(g, ref)


def test_grad_argnums_conventions():
    # int argnums -> bare array (jax.grad convention)
    g0 = graphax.grad(_f2, "rev", argnums=0)(_x, _y)
    assert isinstance(g0, jax.Array) and g0.shape == _x.shape
    assert np.allclose(g0, jax.grad(_f2, argnums=0)(_x, _y))
    # sequence argnums -> tuple
    gt = graphax.grad(_f2, "rev", argnums=(1,))(_x, _y)
    assert isinstance(gt, tuple) and len(gt) == 1
    assert np.allclose(gt[0], jax.grad(_f2, argnums=1)(_x, _y))


def test_value_and_grad_matches_jax():
    (v, g) = graphax.value_and_grad(_f2, "rev", argnums=(0, 1))(_x, _y)
    v_ref, g_ref = jax.value_and_grad(_f2, argnums=(0, 1))(_x, _y)
    assert np.allclose(v, v_ref)
    assert graphax.tree_allclose(g, g_ref)
    # int argnums: bare gradient
    v2, g2 = graphax.value_and_grad(_f2, "rev", argnums=0)(_x, _y)
    assert np.allclose(v2, v_ref) and np.allclose(g2, g_ref[0])


def test_grad_cross_country_order():
    """An explicit non-fwd/rev (cross-country) vertex order gives the same
    gradient — order changes the accumulation, not the result."""

    def f(x, y):
        z = x * y
        return jnp.sum(jnp.sin(z) * jnp.cos(z))

    jaxpr = jax.make_jaxpr(f)(_x, _y).jaxpr
    elim = [
        i for i, eqn in enumerate(jaxpr.eqns, start=1)
        if all(ov not in jaxpr.outvars for ov in eqn.outvars)
    ]
    assert len(elim) >= 3, "need a non-trivial graph for a cross-country order"
    order = elim[1:] + elim[:1]  # deterministic rotation: neither fwd nor rev
    assert order != elim and order != elim[::-1], (
        "rotation coincides with fwd/rev order — test would not actually "
        "exercise cross-country accumulation"
    )
    g = graphax.grad(f, order, argnums=(0, 1))(_x, _y)
    ref = jax.grad(f, argnums=(0, 1))(_x, _y)
    assert graphax.tree_allclose(g, ref)


def test_grad_transform_scales_gradient():
    """A per-vertex callable transform applies DURING elimination: doubling
    the only intermediate vertex's edge Jacobian exactly doubles the
    gradient (the approximation hook)."""

    def f(x):
        return jnp.sum(jnp.sin(x))

    jaxpr = jax.make_jaxpr(f)(_x).jaxpr
    (sin_vertex,) = [
        i for i, eqn in enumerate(jaxpr.eqns, start=1)
        if eqn.primitive.name == "sin"
    ]
    double = lambda st: st.copy(scalar_mult=st.scalar_mult * 2.0)
    g = graphax.grad(f, "rev", argnums=0, transforms=[(sin_vertex, [double])])(_x)
    assert np.allclose(g, 2.0 * jax.grad(f)(_x))


def test_grad_rejects_nonscalar_output():
    with pytest.raises(TypeError, match="scalar-output"):
        graphax.grad(lambda x: jnp.sin(x), "rev")(_x)


def test_grad_rejects_integer_output():
    with pytest.raises(TypeError, match="floating"):
        graphax.grad(lambda x: jnp.sum(x).astype(jnp.int32), "rev")(_x)


def test_grad_rejects_complex_output():
    # complex IS inexact but graphax has no holomorphic handling -> reject loudly
    # (jax.grad also raises without holomorphic=True) instead of a wrong result.
    with pytest.raises(TypeError, match="floating"):
        graphax.grad(lambda x: jnp.sum(x.astype(jnp.complex64) ** 2), "rev")(_x)


def test_grad_rejects_integer_input():
    n = jnp.array([1, 2, 3], dtype=jnp.int32)
    with pytest.raises(TypeError, match="floating"):
        graphax.grad(lambda a, b: jnp.sum(a.astype(jnp.float32) * b), "rev",
                     argnums=0)(n, jnp.ones(3))


def test_grad_rejects_kwargs():
    # jacve threads only positional args; a kwarg used to crash cryptically deep
    # in elimination — now a clear error at the boundary.
    with pytest.raises(TypeError, match="keyword argument"):
        graphax.grad(lambda x, scale: jnp.sum(jnp.sin(x) * scale), "rev")(_x, scale=2.0)


def test_grad_rejects_pytree_input():
    # dict-of-params (canonical jax.grad use) used to crash with a confusing
    # 'takes 1 positional argument but 2 were given'; now a clear NotImplemented.
    d = {"a": _x, "b": _y}
    with pytest.raises(NotImplementedError, match="pytree"):
        graphax.grad(lambda p: jnp.sum(jnp.sin(p["a"]) * p["b"]), "rev")(d)


def test_grad_count_ops_aux():
    g, aux = graphax.grad(_f2, "rev", argnums=(0, 1), count_ops=True)(_x, _y)
    assert graphax.tree_allclose(g, jax.grad(_f2, argnums=(0, 1))(_x, _y))
    assert {"adds", "muls", "fmas", "mem"} <= set(aux.keys())
