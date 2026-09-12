#!/usr/bin/env python
"""Confirm the analytic covers against the engine, case by case."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.expanduser(
    "/tmp/claude-1001/-home-claude-dsnn--claude-worktrees-alpha-pprox/"
    "22ed5f10-dee2-44f0-b385-b953fa5bc6a1/scratchpad/race/G/graphax/src"))
sys.path.insert(0, os.path.expanduser(
    "/tmp/claude-1001/-home-claude-dsnn--claude-worktrees-alpha-pprox/"
    "22ed5f10-dee2-44f0-b385-b953fa5bc6a1/scratchpad/race/G/graphax/tests/core/sparse_tensor"))
sys.path.insert(0, os.path.expanduser(
    "/tmp/claude-1001/-home-claude-dsnn--claude-worktrees-alpha-pprox/"
    "22ed5f10-dee2-44f0-b385-b953fa5bc6a1/scratchpad/race/G/graphax/.scratch-race/probes/t285"))

import jax.numpy as jnp
from graphax.sparse.ops.matmul import matmul
from utils import (class_covers, replication_factor, stored_elements,
                   assert_constant_along_implicit)
import t285_lib as L

rows = []
for name, build in L.CASES.items():
    lhs, rhs, note = build()
    with L.mode_env("lazy_nodemote"):
        got = matmul(lhs, rhs)
    ref = L.ORACLES[name](lhs.dense(), rhs.dense())
    try:
        cov = class_covers(ref, got)
        r = replication_factor(got)
        assert_constant_along_implicit(ref, got)
        const = "yes"
    except AssertionError as e:
        rows.append((name, "-", "-", "-", "-", stored_elements(got),
                     L.OPTIMUM[name][0], str(e)[:70]))
        continue
    rows.append((name, cov["support"], cov["declared"], cov["best"], r,
                 stored_elements(got), L.OPTIMUM[name][0], const))

W = max(len(r[0]) for r in rows)
print(f"{'case'.ljust(W)} | supp | cover | best | repl | stored | hand | honest | verdict")
print("-" * (W + 66))
for name, supp, dec, best, r, stored, hand, note in rows:
    if dec == "-":
        print(f"{name.ljust(W)} | {'':4} | {'':5} | {'':4} | {'':4} | {stored:6} | {hand:4} |    ?   | {note}")
        continue
    honest = dec // r
    v = "OK" if stored == honest else f"MISMATCH ({stored} != {honest})"
    if honest != best // r:
        v += f"; class costs {honest * r / best:.2f}x over best class"
    print(f"{name.ljust(W)} | {supp:4} | {dec:5} | {best:4} | {r:4} | "
          f"{stored:6} | {hand:4} | {honest:6} | {v}")
