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


def test_grad_random_order_is_exact():
    """Vertex elimination is order-INVARIANT: a RANDOM elimination-order
    permutation gives the same grad AND value_and_grad as jax and as graphax's
    default 'rev' order. tree_allclose uses RELATIVE tolerance — different
    accumulation orders reassociate float32 ops, so large-magnitude gradients
    differ only in the last ~ULP (an absolute tolerance would false-fail)."""
    def f(a, b, c):  # well-conditioned multi-input scalar (no nan to confound)
        return jnp.sum(jnp.tanh(a * b) + jnp.sin(b + c) * jnp.cos(a))

    a = jnp.array([0.3, 1.1, -0.7]); b = jnp.array([0.5, -0.2, 0.9])
    c = jnp.array([1.2, 0.4, -0.3]); args, an = (a, b, c), (0, 1, 2)
    N = len(jax.make_jaxpr(f)(*args).jaxpr.eqns)
    assert N >= 5

    jg = jax.grad(f, an)(*args)
    jv, jvg = jax.value_and_grad(f, an)(*args)
    rev_g = graphax.grad(f, "rev", an)(*args)

    rng = np.random.default_rng(0)
    seen_noncrev = False
    for _ in range(8):
        order = [int(o) for o in rng.permutation(np.arange(1, N + 1))]
        seen_noncrev |= order != list(range(N, 0, -1))
        g = graphax.grad(f, order, an)(*args)
        v, vg = graphax.value_and_grad(f, order, an)(*args)
        assert graphax.tree_allclose(g, jg), f"grad != jax.grad for order {order}"
        assert graphax.tree_allclose(g, rev_g), f"grad != graphax-rev for order {order}"
        assert graphax.tree_allclose(vg, jvg), f"value_and_grad != jax for order {order}"
        assert np.allclose(np.asarray(v), np.asarray(jv), rtol=1e-4, atol=1e-6)
    assert seen_noncrev, "random orders coincided with reverse — not a real test"


def test_grad_count_ops_aux():
    g, aux = graphax.grad(_f2, "rev", argnums=(0, 1), count_ops=True)(_x, _y)
    assert graphax.tree_allclose(g, jax.grad(_f2, argnums=(0, 1))(_x, _y))
    assert {"adds", "muls", "fmas", "mem"} <= set(aux.keys())


def test_grad_count_ops_scalar_contraction():
    """count_ops must handle scalar x scalar contractions — an elementwise
    multiply, NOT a matmul (sparse_matmul rejects 0-rank operands). A
    scalar-output gradient whose accumulation hits such an edge (RoeFlux does)
    used to crash with 'matmul of two 0-rank SparseTensors'. The count path must
    produce the SAME gradient as the non-count path (and as jax)."""
    import graphax.examples as ex
    args = tuple(jnp.array(v) for v in (.1, .2, .3, .15, .25, .35))
    an = tuple(range(6))
    sf = lambda *a: sum(jnp.sum(o) for o in jax.tree_util.tree_leaves(ex.RoeFlux_1d(*a)))
    g_nc = graphax.grad(sf, "rev", an)(*args)
    # a non-rev order that genuinely hits scalar x scalar contractions (used to
    # raise); equal_nan=True because RoeFlux is ill-conditioned at these inputs.
    N = len(jax.make_jaxpr(sf)(*args).jaxpr.eqns)
    order = [int(o) for o in np.random.default_rng(0).permutation(np.arange(1, N + 1))]
    g_c, aux = graphax.grad(sf, order, an, count_ops=True)(*args)
    assert graphax.tree_allclose(g_c, g_nc, equal_nan=True)
    assert {"adds", "muls", "fmas", "mem"} <= set(aux) and aux["muls"] > 0
    (v, vg), _ = graphax.value_and_grad(sf, order, an, count_ops=True)(*args)
    assert graphax.tree_allclose(vg, g_nc, equal_nan=True)


def test_grad_count_ops_distinguishes_orders():
    """count_ops gives DIFFERENT (muls, fmas) for different elimination orders of
    the same scalar gradient, while the gradient stays exact — i.e. the order
    genuinely changes the accumulation, not the result."""
    def f(a, b, c):
        return jnp.sum(jnp.tanh(a * b) + jnp.sin(b + c) * jnp.cos(a))
    a, b, c = jnp.array([0.3, 1.1, -0.7]), jnp.array([0.5, -0.2, 0.9]), jnp.array([1.2, 0.4, -0.3])
    args, an = (a, b, c), (0, 1, 2)
    N = len(jax.make_jaxpr(f)(*args).jaxpr.eqns)
    ref = graphax.grad(f, "rev", an)(*args)
    rng = np.random.default_rng(0)
    costs = set()
    for _ in range(8):
        order = [int(o) for o in rng.permutation(np.arange(1, N + 1))]
        g, aux = graphax.grad(f, order, an, count_ops=True)(*args)
        assert graphax.tree_allclose(g, ref)
        costs.add((aux["muls"], aux["fmas"]))
    assert len(costs) > 1, "different orders should yield different op counts"
