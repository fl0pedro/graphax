"""``graphax.instrumentation`` — Jacobian-structure features for downstream agents.

This subpackage is *purely additive* to graphax's core. It provides a parallel
registry of analytical Jacobian-structure rules keyed on primitive (alongside
graphax's ``elemental_rules``), plus a Hutchinson-probe fallback for primitives
without a registered rule. Together they yield, for every vertex of a jaxpr,
a compact ``(frob_sq, diagonal_energy, off_diag_ratio)`` summary the agent can
condition on.

**Toggle.** All instrumentation is gated by the ``GRAPHAX_JACOBIAN_INSTRUMENTATION``
environment variable. When unset (the default) the high-level entry point
:func:`extract_jacobian_features` returns ``None`` immediately and no
instrumentation work happens. When set, the function walks the jaxpr and
returns features. The toggle is read at call time so tests can flip it without
reimporting the package.

**Compilation isolation.** None of this code participates in the gradient
computation that ``jacve`` traces and XLA compiles. Hutchinson probes do
trigger their own JIT compilations of small ``jax.jvp`` calls, but those are
separate XLA programs from anything graphax compiles for the actual training
or evaluation pass — so flipping the toggle on or off has *no* effect on
graphax's existing compiled artefacts.

**Usage.**

    >>> import graphax
    >>> # Default off — returns None, near-zero overhead.
    >>> graphax.instrumentation.extract_jacobian_features(jaxpr, consts, args)
    None
    >>> # Flip on (typically via env var; here we just read the variable).
    >>> import os; os.environ["GRAPHAX_JACOBIAN_INSTRUMENTATION"] = "1"
    >>> features = graphax.instrumentation.extract_jacobian_features(jaxpr, consts, args)
    >>> features.frob_sq.shape, features.off_diag_ratio.shape
    ((6,), (6,))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hutchinson import (
    HutchinsonResult,
    compute_per_vertex_off_diag_hutchinson,
    hutchinson_diagonal_energy,
    hutchinson_frob_sq,
    hutchinson_off_diag_ratio,
)
from .structure import (
    STRUCTURE_RULES,
    JacobianStructure,
    compute_per_vertex_structure,
    defstructure,
    lookup_structure_rule,
)
from .toggle import ENV_VAR_NAME, is_enabled

__all__ = [
    "ENV_VAR_NAME",
    "HutchinsonResult",
    "JacobianFeatures",
    "JacobianStructure",
    "STRUCTURE_RULES",
    "compute_per_vertex_off_diag_hutchinson",
    "compute_per_vertex_structure",
    "defstructure",
    "extract_jacobian_features",
    "hutchinson_diagonal_energy",
    "hutchinson_frob_sq",
    "hutchinson_off_diag_ratio",
    "is_enabled",
    "lookup_structure_rule",
]


@dataclass
class JacobianFeatures:
    """Stacked per-vertex feature output of :func:`extract_jacobian_features`.

    Each field is a ``(num_eqns,)`` numpy array. ``source[i]`` records which
    estimator produced the entry — ``"analytical"`` (analytical rule applied),
    ``"hutchinson"`` (Hutchinson fallback), or ``"missing"`` (both failed for
    that eqn).
    """

    frob_sq: np.ndarray            # (num_eqns,) float32
    diagonal_energy: np.ndarray    # (num_eqns,) float32
    off_diag_ratio: np.ndarray     # (num_eqns,) float32
    source: list[str]              # length = num_eqns


def extract_jacobian_features(
    jaxpr,
    consts: tuple,
    args: tuple,
    *,
    n_probes: int = 16,
    key=None,
) -> JacobianFeatures | None:
    """Return per-vertex ``(frob_sq, diagonal_energy, off_diag_ratio)`` features.

    For each equation in ``jaxpr``:

    * If a structure rule is registered for the primitive, the analytical
      computation runs in pure NumPy (no JIT, no XLA, microsecond cost).
    * Otherwise, fall back to a Hutchinson probe with ``n_probes`` JVPs.

    Returns ``None`` immediately if the ``GRAPHAX_JACOBIAN_INSTRUMENTATION``
    env var is not set, ensuring the off-path is a true no-op for callers
    that always invoke this function.
    """
    if not is_enabled():
        return None

    structures = compute_per_vertex_structure(jaxpr, consts, args)
    needs_hutchinson = any(s is None for s in structures)
    hutch_results: list = [None] * len(structures)
    if needs_hutchinson:
        hutch_results = compute_per_vertex_off_diag_hutchinson(
            jaxpr, consts, args, n_probes=n_probes, key=key,
        )

    n = len(structures)
    frob = np.zeros(n, dtype=np.float32)
    diag = np.zeros(n, dtype=np.float32)
    ratio = np.zeros(n, dtype=np.float32)
    source: list[str] = []
    for i in range(n):
        s = structures[i]
        if s is not None:
            frob[i] = s.frob_sq
            diag[i] = s.diagonal_energy
            ratio[i] = s.off_diag_ratio
            source.append("analytical")
            continue
        h = hutch_results[i]
        if h is not None:
            frob[i] = h.frob_sq
            diag[i] = h.diagonal_energy
            ratio[i] = h.off_diag_ratio
            source.append("hutchinson")
            continue
        source.append("missing")
    return JacobianFeatures(
        frob_sq=frob,
        diagonal_energy=diag,
        off_diag_ratio=ratio,
        source=source,
    )
