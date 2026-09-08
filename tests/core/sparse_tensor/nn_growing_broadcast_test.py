"""Growing-broadcast census on the NeuralNetwork(mnist) exact plan (ticket
dsnn-3qm.28.1, deliverable b). Turns lane B's ad hoc probe
(.scratch/trustworthy-approx-search/probes/t28b/broadcast_census.py) into an
asserted test, on the second target the ticket names (the first is the MLP
toy of ``output_layout_test.py``).

The property under test: the contraction engine never grows a buffer to line
two operands up. An axis one operand does not store is stated in the einsum
and summed away at extent 1, never broadcast to its partner's extent. The
census counts every ``_as_shape(mode="broadcast")`` call that makes the array
bigger. It must be zero.

That number used to depend on an environment variable, because a
``dot_general`` forced a choice: keeping the axis in the batch list needed the
broadcast, and taking it out lost the fusion on GPU. The einsum emission does
not force the choice, so the rule is unconditional and the target is zero
(ticket dsnn-3qm.72). Under the old ``full`` rules this target already
reached zero, down from 15 calls and 50 258 grown elements on the incumbent
frame (T28B-RESULT.md section 3).

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


def _as_shape_growth_census(env, order):
    saved = {k: os.environ.get(k) for k in
             ("ALPHAGRAD_SKIP_COUNT_OPS", "ALPHAGRAD_SKIP_COST_ANALYSIS")}
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

    _mm._as_shape = _wrapped
    try:
        cfg = env.config
        fn = jacve(cfg.target_fun, list(order), argnums=cfg.argnums, has_aux=cfg.has_aux,
                   sparse_representation=True, transforms=[], face_transforms=None)
        jax.eval_shape(fn, *env.args)
    finally:
        _mm._as_shape = orig_as_shape
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return grew["calls"], grew["elems"]


def test_neuralnetwork_exact_plan_has_zero_growing_broadcasts():
    env, order = _build()
    calls, elems = _as_shape_growth_census(env, order)
    assert calls == 0, (
        f"NeuralNetwork exact: {calls} growing _as_shape(mode='broadcast') "
        f"call(s) ({elems} elements grown), expected zero"
    )
