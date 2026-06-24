"""Regression: an approximated Jacobian under a NON-rev elimination order must
match the dense-Diag/Compress oracle.

This pins the exact path that two memory "optimizations" silently broke this
cycle: keeping approximation edges block-diagonal (to dodge the per-vertex
densify) makes a later ``Diag``'s pair conflict and SKIP, which under-masks the
edge — a different, lighter approximation (cos collapsed to ~0.05 vs this oracle).
The local suites never exercised it, so both bad attempts reached the cluster
before being caught. This test is that missing guard.

The under-masking only surfaces on the richer slice/concat multi-head ViT at
``S >= 16`` (the small ``attn`` model stays cos 1.0 even when broken), and that
size needs ~24 GB — so the ViT case is RAM-guarded (it runs on CI / the cluster,
skips on a laptop), matching the other memory-heavy elemental suites. The small
``attn`` smoke runs everywhere and guards against gross random-order breakage
(crashes, id-misalignment) cheaply.
"""

import os

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as tu
import numpy as np
import pytest

from graphax import jacve
from graphax.sparse.micro_actions import Compress, Diag

import _approx_oracle as O


def _flat(j):
    return np.concatenate([np.ravel(np.asarray(l)) for l in tu.tree_leaves(j)])


def _cos(a, b):
    a = _flat(a).astype(np.float64)
    b = _flat(b).astype(np.float64)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else 1.0


def _random_order(n_verts, seed):
    return [
        int(v)
        for v in np.random.default_rng(seed).permutation(np.arange(1, n_verts + 1))
    ]


def _total_ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


_TRANSFORMS = {"diag": [Diag(0, 1, 2)], "compress": [Compress((0,), "mean")]}


# --------------------------------------------------------------------------- #
# Small smoke — runs everywhere. attn is too small to trigger the under-masking
# conflict (it stays cos 1.0 even when broken), but it does catch a random-order
# crash / id-misalignment regression at near-zero cost.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tname", ["diag", "compress"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_attn_random_order_matches_oracle(seed, tname):
    fn, args = O.MODELS["attn"]()
    nv = O.N_VERTS["attn"]
    order = _random_order(nv, seed)
    spec = [(v, _TRANSFORMS[tname]) for v in range(1, nv + 1)]
    got = jacve(fn, order, argnums=(0,), transforms=spec)(*args)
    ref = O.oracle_jacobian(fn, order, (0,), spec)(*args)
    assert _cos(got, ref) > 0.999, f"attn {tname} seed{seed}: cos={_cos(got, ref)}"


# --------------------------------------------------------------------------- #
# The actual under-masking guard — slice/concat ViT, random order. RAM-guarded:
# the conflict needs S>=16, which densifies to ~24 GB on a bad order.
# --------------------------------------------------------------------------- #
def _vit(S, Dm, H):
    dh = Dm // H

    def vit(x, Wq, Wk, Wv, Wo, W1, W2):
        q = x @ Wq
        k = x @ Wk
        v = x @ Wv

        def head(h):
            qh = q[:, h * dh : (h + 1) * dh]
            kh = k[:, h * dh : (h + 1) * dh]
            vh = v[:, h * dh : (h + 1) * dh]
            return jax.nn.softmax((qh @ kh.T) / jnp.sqrt(dh), axis=-1) @ vh

        x = x + jnp.concatenate([head(h) for h in range(H)], axis=-1) @ Wo
        return x + jnp.tanh(x @ W1) @ W2

    ks = jr.split(jr.PRNGKey(7), 12)
    args = (
        (jr.normal(ks[6], (S, Dm)) * 0.5,)
        + tuple(jr.normal(ks[i], (Dm, Dm)) * 0.3 for i in range(4))
        + (jr.normal(ks[4], (Dm, 4 * Dm)) * 0.3, jr.normal(ks[5], (4 * Dm, Dm)) * 0.3)
    )
    return vit, args


@pytest.mark.skipif(
    _total_ram_gb() < 24.0,
    reason="slice/concat ViT S=16 random-order densifies to ~24 GB",
)
@pytest.mark.parametrize("seed", [101, 202, 303])
def test_vit_random_order_matches_oracle(seed):
    vit, args = _vit(16, 64, 4)
    argnums = tuple(range(1, 7))
    order = _random_order(199, seed)
    spec = [(v, [Diag(0, 1, 2)]) for v in range(1, 200)]
    got = jacve(vit, order, argnums=argnums, transforms=spec)(*args)
    ref = O.oracle_jacobian(vit, order, argnums, spec)(*args)
    # The previous (broken) keep-sparse versions gave cos 0.05-0.80 here.
    assert _cos(got, ref) > 0.999, f"ViT seed{seed}: cos={_cos(got, ref)}"
