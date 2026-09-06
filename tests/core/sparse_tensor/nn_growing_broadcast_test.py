"""Growing-broadcast census on the NeuralNetwork(mnist) exact plan (ticket
dsnn-3qm.28.1, deliverable b). Turns lane B's ad hoc probe
(.scratch/trustworthy-approx-search/probes/t28b/broadcast_census.py) into an
asserted test, on the second target the ticket names (the first is the MLP
toy of ``output_layout_test.py``).

Needs alphagrad importable (``landscape_map.build_env`` builds the target the
same way the campaign does) — skipped cleanly if it is not on the path, the
same pattern ``analyze_and_smoke_test.py`` uses for ``jax_memory_monitor``.
"""
from __future__ import annotations

import importlib
import math
import os

import jax
import pytest

alphagrad_lm = pytest.importorskip("alphagrad.approx.tools.landscape_map")

from graphax import jacve  # noqa: E402  (after importorskip, deliberately)

_mm = importlib.import_module("graphax.sparse.ops.matmul")
_mm_legacy = importlib.import_module("graphax.sparse.ops.matmul_legacy_tiled")

_CLI = ["--example", "NeuralNetwork", "--dataset", "mnist", "--seed", "250197",
        "--latency-inner-reps", "1", "--num-data-points", "1", "--reps-per-point", "1",
        "--quality-metric", "grad_cosine", "--approx-old", "same",
        "--out-dir", "/tmp/t28b-repro", "--dry-run",
        "--quant-slots", "0,1,2", "--diag-slots", "2", "--compress-slots", "2"]


def _build():
    args = alphagrad_lm.make_argparser().parse_args(_CLI)
    alphagrad_lm.ARGS = args
    env, _s, _ = alphagrad_lm.build_env(args)
    order = [int(v) for v in alphagrad_lm.rev_order(env)]
    return env, order


def _as_shape_growth_census(env, order, *, tiled_legacy, lazy_rules="nodemote"):
    saved = {k: os.environ.get(k) for k in
             ("GRAPHAX_TILED_LEGACY", "GRAPHAX_TILED_LAZY",
              "GRAPHAX_EINSUM_GENERAL", "GRAPHAX_PLANNER_EXACT",
              "ALPHAGRAD_SKIP_COUNT_OPS", "ALPHAGRAD_SKIP_COST_ANALYSIS")}
    os.environ["GRAPHAX_TILED_LEGACY"] = "1" if tiled_legacy else "0"
    os.environ["GRAPHAX_TILED_LAZY"] = lazy_rules
    os.environ["GRAPHAX_EINSUM_GENERAL"] = "0"
    os.environ["GRAPHAX_PLANNER_EXACT"] = "0"
    os.environ["ALPHAGRAD_SKIP_COUNT_OPS"] = "1"
    os.environ["ALPHAGRAD_SKIP_COST_ANALYSIS"] = "1"
    orig_as_shape = _mm._as_shape
    grew = {"calls": 0, "elems": 0}

    def _wrapped(view, target_shape, *, mode):
        out = orig_as_shape(view, target_shape, mode=mode)
        if mode == "broadcast":
            in_n = math.prod(view.shape) if view.shape else 1
            target = tuple(target_shape)
            out_n = math.prod(target) if target else 1
            if out_n > in_n:
                grew["calls"] += 1
                grew["elems"] += out_n - in_n
        return out

    # Same import-time-binding caveat as output_layout_test.py: patch both
    # module-level names, since matmul_legacy_tiled imports _as_shape by name.
    _mm._as_shape = _wrapped
    _mm_legacy._as_shape = _wrapped
    try:
        cfg = env.config
        fn = jacve(cfg.target_fun, list(order), argnums=cfg.argnums, has_aux=cfg.has_aux,
                   sparse_representation=True, transforms=[], face_transforms=None)
        jax.eval_shape(fn, *env.args)
    finally:
        _mm._as_shape = orig_as_shape
        _mm_legacy._as_shape = orig_as_shape
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return grew["calls"], grew["elems"]


def test_neuralnetwork_exact_plan_full_rules_has_zero_growing_broadcasts():
    """``GRAPHAX_TILED_LAZY=full`` (demote ON) reaches ZERO growing
    ``_as_shape(mode="broadcast")`` calls on the NeuralNetwork(mnist) exact
    plan — matching T28B-RESULT.md section 3's "15 calls / 50258 elements to
    ZERO" claim exactly. That claim was measured under ``full``, not under
    the landed default (see the next test)."""
    env, order = _build()
    calls, elems = _as_shape_growth_census(env, order, tiled_legacy=False, lazy_rules="full")
    assert calls == 0, (
        f"NeuralNetwork exact, GRAPHAX_TILED_LAZY=full: {calls} growing "
        f"_as_shape(mode='broadcast') call(s) ({elems} elements grown), expected zero"
    )


def test_neuralnetwork_exact_plan_default_rules_grow_fewer_than_the_incumbent():
    """The LANDED DEFAULT (nodemote, demote OFF — the GPU-favoring choice of
    T28B-RESULT.md section 4) does NOT reach zero on this target either: 10
    calls / 49 353 elements remain (all from the one-sided meta axis the
    demote rule would otherwise take out of the dot_general batch list), down
    from 15 calls / 49 398 elements on the incumbent. Still a strict
    improvement, never a regression; not zero."""
    env, order = _build()
    lazy_calls, lazy_elems = _as_shape_growth_census(
        env, order, tiled_legacy=False, lazy_rules="nodemote")
    legacy_calls, legacy_elems = _as_shape_growth_census(env, order, tiled_legacy=True)
    assert lazy_calls < legacy_calls, (
        f"NeuralNetwork exact, nodemote (default): {lazy_calls} growing "
        f"_as_shape(mode='broadcast') call(s) ({lazy_elems} elements grown); "
        f"incumbent has {legacy_calls} ({legacy_elems} elements grown) — "
        "expected the default lazy frame to grow strictly fewer, even though not zero"
    )
