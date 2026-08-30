"""Fusability parity pins: jacve('rev') vs jax.grad on the SAME scalar loss.

Measured 2026-08-08 on the TLM/wikitext grad target (Blackwell, campaign
compile options): jacve_rev 132.0us vs jax.grad 133.8us (ratio 0.987), 31 vs
30 fusions, ZERO unfused top-level ops on either side -- graphax's exact
path is at gold-standard fusability even though its trace carries 3.6x the
equations (bookkeeping reshapes/broadcasts lower to bitcasts; redundant
contractions CSE away).

These tests pin that property on small models so a future emitter change
that breaks fusion shows up in CI, without depending on wall time or GPU:
(1) numerical parity, (2) compiled fusion count within a bounded factor of
jax.grad's ON THE SAME BACKEND, (3) no top-level op left outside a fusion
beyond what jax.grad itself leaves.
"""
import re
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import jacve

_A = jnp.asarray(np.random.RandomState(0).randn(64, 32).astype(np.float32))
_B = jnp.asarray(np.random.RandomState(1).randn(10, 64).astype(np.float32))
_x = jnp.asarray(np.random.RandomState(2).randn(32).astype(np.float32))
_t = jnp.asarray(np.random.RandomState(3).randn(10).astype(np.float32))


def _mlp_loss(x, t, W1, W2):
    h = jnp.tanh(W1 @ x)
    y = W2 @ h
    d = y - t
    return jnp.sum(d * d)


def _attn_loss(x, t, Wq, Wk, Wv):
    q, k, v = Wq @ x, Wk @ x, Wv @ x
    a = jax.nn.softmax(jnp.outer(q, k) / jnp.sqrt(q.shape[0]))
    y = a @ v
    return jnp.sum((y - t) * (y - t))


_S = jnp.asarray(np.random.RandomState(4).randn(16, 16).astype(np.float32))
_x16 = jnp.asarray(np.random.RandomState(5).randn(16).astype(np.float32))
_t16 = jnp.asarray(np.random.RandomState(6).randn(16).astype(np.float32))

CASES = [
    ("mlp", _mlp_loss, (_x, _t, _A, _B), (2, 3)),
    ("attn", _attn_loss, (_x16, _t16, _S, _S + 0.1, _S - 0.1), (2, 3, 4)),
]


def _hlo_stats(f, args):
    ex = jax.jit(f).lower(*args).compile()
    txt = ex.as_text()
    fus = txt.count(" fusion(")
    unfused = Counter()
    in_entry = False
    for line in txt.splitlines():
        if line.startswith("ENTRY "):
            in_entry = True
            continue
        if re.match(r"^%?[\w.-]+ \(", line):
            in_entry = False
            continue
        if not in_entry:
            continue
        m = re.match(r"\s+(?:ROOT )?%?[\w.-]+ = \S+ (\w[\w-]*)\(", line)
        if m and m.group(1) not in (
                "fusion", "custom-call", "parameter", "tuple",
                "get-tuple-element", "constant", "bitcast", "copy"):
            unfused[m.group(1)] += 1
    return fus, unfused, ex


@pytest.mark.parametrize("name,fn,args,argnums", CASES)
def test_numerical_parity(name, fn, args, argnums):
    gold = jax.grad(fn, argnums=argnums)(*args)
    ours = jacve(fn, "rev", argnums=argnums)(*args)
    gl = jax.tree_util.tree_leaves(gold)
    xl = jax.tree_util.tree_leaves(ours)
    assert len(gl) == len(xl)
    for a, b in zip(gl, xl):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("name,fn,args,argnums", CASES)
def test_fusion_count_parity(name, fn, args, argnums):
    """jacve's compiled module must not fragment: fusion count within 1.5x of
    jax.grad's on the same backend (measured today: 31 vs 30 on TLM, and
    equal on these small cases). A regression here is an emitter that XLA
    can no longer fuse across."""
    f_gold, _, _ = _hlo_stats(jax.grad(fn, argnums=argnums), args)
    f_ours, _, _ = _hlo_stats(jacve(fn, "rev", argnums=argnums), args)
    assert f_ours <= max(f_gold, 1) * 1.5 + 2, (f_gold, f_ours)


@pytest.mark.parametrize("name,fn,args,argnums", CASES)
def test_no_extra_unfused_ops(name, fn, args, argnums):
    """Every top-level op class jacve leaves outside a fusion must also be
    left outside by jax.grad -- zero NEW unfused op kinds. (Measured today:
    both sides empty on the TLM target.)"""
    _, u_gold, _ = _hlo_stats(jax.grad(fn, argnums=argnums), args)
    _, u_ours, _ = _hlo_stats(jacve(fn, "rev", argnums=argnums), args)
    extra = set(u_ours) - set(u_gold)
    assert not extra, f"new unfused op kinds vs jax.grad: {extra}"
