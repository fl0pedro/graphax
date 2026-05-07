"""Hutchinson-style probe estimators for off-diagonal Jacobian energy.

This is the universal fallback for primitives without an analytical
:mod:`graphax.instrumentation.structure` rule. It estimates ``||J||_F^2`` and
the diagonal energy under the natural axis pairing using ``jax.jvp`` calls
with random-sign Rademacher and one-hot probes — the same identity that
underlies stochastic-trace estimators in numerical linear algebra:

    E_ξ[ ξ^T (J^T J) ξ ] = ||J||_F^2     for Rademacher ξ ∈ {±1}^d.

The cost is ``n_probes`` JVPs per primitive call. Each JVP is a small JAX
computation that gets traced and compiled on first call and cached thereafter.
None of this enters the path of the *training* computation that the env's
reward harness compiles — those are entirely separate XLA programs. So
turning instrumentation on adds only the bookkeeping cost of the probes
themselves; turning it off (the default) means none of this code runs at all
and the existing graphax XLA compilation is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np


@dataclass
class HutchinsonResult:
    frob_sq: float
    diagonal_energy: float
    off_diag_ratio: float
    n_probes: int


def _rademacher(key, shape, dtype=jnp.float32):
    return 2.0 * jrand.bernoulli(key, p=0.5, shape=shape).astype(dtype) - 1.0


def hutchinson_frob_sq(f, x: jax.Array, n_probes: int, key) -> float:
    """Unbiased estimate of ``||J||_F^2`` where ``J = ∂f(x)/∂x``.

    ``f`` must accept a single-array input matching ``x``'s shape; for
    primitives with multiple inputs, partial application gives you the
    per-input Jacobian (which is exactly the "elemental Jacobian"
    decomposition graphax already uses).
    """
    keys = jrand.split(key, n_probes)
    total = 0.0
    for k in keys:
        xi = _rademacher(k, x.shape, x.dtype)
        _, jvp_val = jax.jvp(f, (x,), (xi,))
        total += float(jnp.sum(jvp_val ** 2))
    return total / n_probes


def hutchinson_diagonal_energy(
    f, x: jax.Array, n_probes: int, key
) -> float:
    """Estimate of ``sum_i J_ii^2`` under the natural axis pairing.

    Probes with one-hot input vectors and reads out only the matching
    flat-output index. For elementwise ops this returns ``||J||_F^2`` exactly
    (so off-diagonal ratio = 0); for shape-changing or contraction primitives
    it returns the strict-diagonal energy and the ratio gives an honest
    fraction of energy *not* captured by a flat-index diagonalization.
    """
    flat_size = int(np.prod(x.shape))
    out_shape = jax.eval_shape(f, x).shape
    out_flat_size = int(np.prod(out_shape))
    n_match = min(flat_size, out_flat_size)
    keys = jrand.split(key, n_probes)
    total = 0.0
    for k in keys:
        idx = int(jrand.randint(k, (), 0, n_match))
        xi_flat = jnp.zeros(flat_size, dtype=x.dtype).at[idx].set(1.0)
        xi = xi_flat.reshape(x.shape)
        _, jvp_val = jax.jvp(f, (x,), (xi,))
        diag_entry = jvp_val.reshape(-1)[idx]
        total += float(diag_entry ** 2) * n_match  # MC correction
    return total / n_probes


def hutchinson_off_diag_ratio(
    f, x: jax.Array, n_probes: int, key
) -> HutchinsonResult:
    """Combined estimator: returns frob_sq, diagonal_energy, off-diag ratio.

    Splits the probe budget half-and-half between the two sub-estimators so
    they don't reuse the same random draws (mixing would double-count noise
    and bias the ratio).
    """
    n_frob = max(1, n_probes // 2)
    n_diag = max(1, n_probes - n_frob)
    frob_key, diag_key = jrand.split(key)
    frob_sq = hutchinson_frob_sq(f, x, n_frob, frob_key)
    diag = hutchinson_diagonal_energy(f, x, n_diag, diag_key)
    return HutchinsonResult(
        frob_sq=frob_sq,
        diagonal_energy=diag,
        off_diag_ratio=float(max(0.0, frob_sq - diag) / max(frob_sq, 1e-12)),
        n_probes=n_probes,
    )


# ---------------------------------------------------------------------------
# Walk a jaxpr and Hutchinson-probe each eqn (against its first invar).
# ---------------------------------------------------------------------------


def compute_per_vertex_off_diag_hutchinson(
    jaxpr, consts: tuple, args: tuple, n_probes: int = 16, key=None,
):
    """For each eqn in ``jaxpr``, build a single-input partial of the
    primitive around the live activations and probe it. Multi-input primitives
    are partially evaluated against the *first* invar — this matches graphax's
    "per-invar elemental Jacobian" decomposition.

    Returns a list of ``HutchinsonResult | None`` aligned with ``jaxpr.eqns``.
    """
    from jax._src import core

    if key is None:
        key = jrand.PRNGKey(0)

    env: dict = {}
    for var, val in zip(jaxpr.constvars, consts):
        env[var] = val
    for var, val in zip(jaxpr.invars, args):
        env[var] = val

    def read(v):
        return v.val if isinstance(v, core.Literal) else env[v]

    keys = jrand.split(key, len(jaxpr.eqns) or 1)
    results = []
    for i, eqn in enumerate(jaxpr.eqns):
        # Always evaluate the eqn first so subsequent eqns can read its
        # output, even if we ultimately skip the probe (e.g. when the first
        # invar is a literal there's no Jacobian to probe against, but the
        # forward value is still needed downstream).
        try:
            in_vals = [read(v) for v in eqn.invars]
            primal_out = eqn.primitive.bind(*in_vals, **eqn.params)
            outs = primal_out if eqn.primitive.multiple_results else (primal_out,)
            for var, val in zip(eqn.outvars, outs):
                if not isinstance(var, core.DropVar):
                    env[var] = val
        except Exception:
            results.append(None)
            continue

        if not eqn.invars or isinstance(eqn.invars[0], core.Literal):
            results.append(None)
            continue

        try:
            x = in_vals[0]
            other = in_vals[1:]
            params = eqn.params
            prim = eqn.primitive

            def f(x, _other=other, _params=params, _prim=prim):
                if _other:
                    out = _prim.bind(x, *_other, **_params)
                else:
                    out = _prim.bind(x, **_params)
                if _prim.multiple_results:
                    out = out[0]
                return out

            results.append(hutchinson_off_diag_ratio(f, x, n_probes, keys[i]))
        except Exception:
            results.append(None)
    return results
