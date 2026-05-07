"""Analytical Jacobian-structure rules — additive companion to ``elemental_rules``.

The ``elemental_rules`` registry in :mod:`graphax.primitives` answers "what is
the elemental Jacobian factor for this primitive?", which is what the vertex
elimination needs in order to *propagate* gradients. This module answers a
strictly different question used only for *instrumentation*: "for this
primitive call, given its inputs and output, what is the (Frobenius²,
diagonal-energy, off-diagonal-energy-ratio) summary of its local Jacobian?".

Those are exactly the features the architecture spec wants the encoder to
condition on — the off-diagonal energy ratio in particular, since it is the
quantity the Frobenius reward will actually measure.

Crucially, this module never participates in the gradient computation itself.
It is a separate pass that happens after the forward executions inside the
reward harness, lives entirely in Python (no JIT), and can be skipped wholesale
via :func:`graphax.instrumentation.toggle.is_enabled`. When the toggle is off
nothing here is invoked — the XLA compilation of ``jacve`` is byte-identical
to a graphax build that has this subpackage removed.

The registry is mutable on purpose so a downstream user (alphagrad and friends)
can swap in their own rule for any primitive at any time — same idiom as
graphax's :func:`defelemental`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np


@dataclass
class JacobianStructure:
    """Compact per-primitive Jacobian summary.

    The four scalars are what the architecture's factor head will condition on;
    ``params`` is a free-form bag for primitive-specific extras (saturation
    rate, dominant axis, rank-one components for layernorm-style ops, …) that
    a richer agent can tap into.
    """

    kind: str  # "diagonal", "rank_one_plus_diagonal", "generic", "permutation", ...
    frob_sq: float  # ||J_v||_F^2
    diagonal_energy: float  # sum_i J_ii^2 under the natural axis pairing
    off_diag_ratio: float  # (frob_sq - diagonal_energy) / max(frob_sq, eps), in [0, 1]
    params: dict = field(default_factory=dict)


# Public registry. Keyed on `jax.core.Primitive`; the rule signature mirrors
# graphax's elemental_only_rules so it composes naturally with their walk.
StructureRule = Callable  # (primal_out, *primals, **params) -> JacobianStructure
STRUCTURE_RULES: dict = {}


def defstructure(primitive, rule: StructureRule) -> None:
    """Register an analytical structure rule for ``primitive``.

    Mirrors :func:`graphax.primitives.base.defelemental`: idempotent overwrite
    is intentional so a downstream package can install custom rules without
    fighting graphax's load order.
    """
    STRUCTURE_RULES[primitive] = rule


def lookup_structure_rule(primitive) -> StructureRule | None:
    return STRUCTURE_RULES.get(primitive)


# ---------------------------------------------------------------------------
# Built-in rules: elementwise primitives where there's a single canonical
# answer. Shape-changing primitives (dot_general, reduce_sum, broadcast_in_dim)
# have axis-pairing-dependent off-diag energy and intentionally fall through
# to the Hutchinson estimator instead.
# ---------------------------------------------------------------------------


_EPS = 1e-12


def _ratio(frob_sq: float, diag: float) -> float:
    return float(max(0.0, frob_sq - diag) / max(frob_sq, _EPS))


def _elementwise_diagonal(primal_out, primal, deriv: np.ndarray) -> JacobianStructure:
    d2 = float(np.sum(np.asarray(deriv) ** 2))
    return JacobianStructure(
        kind="diagonal",
        frob_sq=d2,
        diagonal_energy=d2,
        off_diag_ratio=0.0,
        params={"deriv_l1": float(np.sum(np.abs(np.asarray(deriv))))},
    )


def _tanh_rule(primal_out, primal, **params):
    return _elementwise_diagonal(primal_out, primal, 1.0 - np.asarray(primal_out) ** 2)


def _exp_rule(primal_out, primal, **params):
    return _elementwise_diagonal(primal_out, primal, np.asarray(primal_out))


def _log_rule(primal_out, primal, **params):
    return _elementwise_diagonal(primal_out, primal, 1.0 / np.asarray(primal))


def _logistic_rule(primal_out, primal, **params):
    out = np.asarray(primal_out)
    return _elementwise_diagonal(primal_out, primal, out * (1.0 - out))


def _neg_rule(primal_out, primal, **params):
    return _elementwise_diagonal(
        primal_out, primal, -np.ones_like(np.asarray(primal_out))
    )


def _relu_via_max_rule(primal_out, x, y, **params):
    """Rule for ``lax.max_p(x, 0)`` — the relu-as-elementwise-max idiom."""
    deriv = (np.asarray(x) > np.asarray(y)).astype(np.float32)
    return _elementwise_diagonal(primal_out, x, deriv)


def _sin_rule(primal_out, primal, **params):
    return _elementwise_diagonal(primal_out, primal, np.cos(np.asarray(primal)))


def _cos_rule(primal_out, primal, **params):
    return _elementwise_diagonal(primal_out, primal, -np.sin(np.asarray(primal)))


def _abs_rule(primal_out, primal, **params):
    return _elementwise_diagonal(
        primal_out, primal, np.sign(np.asarray(primal))
    )


def _sqrt_rule(primal_out, primal, **params):
    return _elementwise_diagonal(
        primal_out, primal, 0.5 / np.asarray(primal_out)
    )


def _square_rule(primal_out, primal, **params):
    return _elementwise_diagonal(primal_out, primal, 2.0 * np.asarray(primal))


# --- Register the built-ins. Importing this module is sufficient. ---
defstructure(lax.tanh_p, _tanh_rule)
defstructure(lax.exp_p, _exp_rule)
defstructure(lax.log_p, _log_rule)
defstructure(lax.logistic_p, _logistic_rule)
defstructure(lax.neg_p, _neg_rule)
defstructure(lax.max_p, _relu_via_max_rule)
defstructure(lax.sin_p, _sin_rule)
defstructure(lax.cos_p, _cos_rule)
defstructure(lax.abs_p, _abs_rule)
defstructure(lax.sqrt_p, _sqrt_rule)
defstructure(lax.square_p, _square_rule)


# ---------------------------------------------------------------------------
# Walk a jaxpr and return per-eqn analytical structure (or None if no rule).
# ---------------------------------------------------------------------------


def compute_per_vertex_structure(jaxpr, consts: tuple, args: tuple):
    """Walk ``jaxpr`` eqn-by-eqn, applying registered structure rules.

    Returns a list (length = ``len(jaxpr.eqns)``) of ``JacobianStructure | None``;
    ``None`` slots are exactly the eqns the Hutchinson estimator would handle.

    No JIT — runs as plain Python. Failures on individual eqns (unsupported
    control flow, etc.) record ``None`` and continue rather than blowing up
    the whole feature pipeline.
    """
    from jax._src import core

    env: dict = {}
    for var, val in zip(jaxpr.constvars, consts):
        env[var] = val
    for var, val in zip(jaxpr.invars, args):
        env[var] = val

    def read(v):
        return v.val if isinstance(v, core.Literal) else env[v]

    out = []
    for eqn in jaxpr.eqns:
        try:
            in_vals = [read(v) for v in eqn.invars]
            primal_out = eqn.primitive.bind(*in_vals, **eqn.params)
            outs = primal_out if eqn.primitive.multiple_results else (primal_out,)
            for var, val in zip(eqn.outvars, outs):
                if not isinstance(var, core.DropVar):
                    env[var] = val
            primal_first = outs[0]
            rule = lookup_structure_rule(eqn.primitive)
            if rule is None:
                out.append(None)
                continue
            structure = rule(primal_first, *in_vals, **eqn.params)
            out.append(structure)
        except Exception:
            out.append(None)
    return out
