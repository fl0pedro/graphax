# dsnn-dfw.192: under the reverse order the RTRL step keeps the vmapped batch axis a diagonal pair.
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrand
import pytest

from graphax import jacve, tree_allclose
from graphax.examples.neuromorphic import RSNN_SHD, rsnn_zero_carry

ARGNUMS = (7, 8, 9)
N_IN, H, N_OUT = 7, 6, 5


def vmapped_step():
    f = jax.vmap(RSNN_SHD, in_axes=(0, 0) + (0,) * 5 + (None,) * 9 + (0,) * 5)

    def step(*a):
        out = f(*a)
        return (jnp.mean(out[0]),) + tuple(out[1:])
    return step


def make_args(key, B):
    ks = jrand.split(key, 11)
    x = (jrand.uniform(ks[0], (B, N_IN)) < 0.3).astype(jnp.float32)
    y = jax.nn.one_hot(jrand.randint(ks[1], (B,), 0, N_OUT), N_OUT)
    S = (jrand.uniform(ks[2], (B, H)) < 0.3).astype(jnp.float32)
    I = jrand.normal(ks[3], (B, H)) * 0.1
    U = jrand.normal(ks[4], (B, H)) * 0.1
    a = jrand.uniform(ks[5], (B, H)) * 0.1
    Uo = jrand.normal(ks[6], (B, N_OUT)) * 0.1
    W = jrand.normal(ks[7], (H, N_IN)) * 0.5
    V = jrand.normal(ks[8], (H, H)) * 0.5
    Wo = jrand.normal(ks[9], (N_OUT, H)) * 0.5
    consts = [jnp.float32(c) for c in (0.8, 0.9, 0.9, 0.95, 0.07, 0.1)]
    zero = rsnn_zero_carry("exact", (W, V, Wo), (B,))
    carry = [jrand.normal(k, z.shape) * 0.1
             for k, z in zip(jrand.split(ks[10], len(zero)), zero)]
    return [x, y, S, I, U, a, Uo, W, V, Wo, *consts, *carry]


def reverse_step():
    return jacve(vmapped_step(), "rev", argnums=ARGNUMS, has_aux=True)


def _sub_jaxprs(eqn):
    for p in eqn.params.values():
        for x in (p if isinstance(p, (list, tuple)) else (p,)):
            if hasattr(x, "eqns"):
                yield x
            elif hasattr(getattr(x, "jaxpr", None), "eqns"):
                yield x.jaxpr


def _batch_by_batch(jaxpr, B, found):
    for e in jaxpr.eqns:
        for v in list(e.invars) + list(e.outvars):
            shape = tuple(getattr(v.aval, "shape", ()))
            if sum(int(s) == B for s in shape) >= 2:
                found.append((e.primitive.name, shape))
        for sub in _sub_jaxprs(e):
            _batch_by_batch(sub, B, found)
    return found


def test_reverse_step_holds_no_batch_by_batch_block():
    B = 8
    xs = make_args(jrand.PRNGKey(0), B)
    closed = jax.make_jaxpr(reverse_step())(*xs)
    found = _batch_by_batch(closed.jaxpr, B, [])
    assert not found, found[:10]


def _cost(B):
    xs = make_args(jrand.PRNGKey(0), B)
    exe = jax.jit(reverse_step()).lower(*xs).compile()
    ca = exe.cost_analysis()
    ca = ca[0] if isinstance(ca, (list, tuple)) else ca
    return float(ca["flops"]), int(exe.memory_analysis().temp_size_in_bytes)


def test_reverse_step_cost_per_sample_does_not_grow_with_batch():
    (f2, t2), (f8, t8) = _cost(2), _cost(8)
    assert f8 / 8 <= 1.25 * f2 / 2, (f2, f8)
    assert t8 / 8 <= 1.25 * t2 / 2, (t2, t8)


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_reverse_step_equals_jacrev(B, seed):
    xs = make_args(jrand.PRNGKey(seed), B)
    step = vmapped_step()
    primal, jac = jax.jit(reverse_step())(*xs)
    ref = jax.jit(jax.jacrev(step, argnums=ARGNUMS))(*xs)
    grad = jax.jit(jax.grad(lambda *a: step(*a)[0], argnums=ARGNUMS))(*xs)
    assert tree_allclose(primal, jax.jit(step)(*xs), atol=1e-6, rtol=1e-6)
    assert tree_allclose(jac, ref, atol=1e-5, rtol=1e-4)
    assert tree_allclose(jac[0], grad, atol=1e-5, rtol=1e-4)
