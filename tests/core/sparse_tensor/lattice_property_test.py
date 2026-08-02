"""Property suite for the structure lattice (graphax.sparse.lattice).

Draws random structured operand pairs per lattice family and checks
``matmul`` / ``elementwise`` against the dense oracle. Seeded (no wall-clock
randomness); every case name encodes its seed so a failure reproduces with
``LATTICE_SEED=<n> pytest -k <family>``.

Layout convention generated here (matches the codebase): a coupled pair
shares ONE meta axis (both members' ``axis``); each side's block axis is its
own val axis when block > 1. Contraction pairs lhs.primal_dims positionally
with rhs.out_dims (equal logical sizes by construction).
"""
import os
import random
import unittest

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.lattice import L, classify
from graphax.sparse.ops.elementwise import elementwise
from graphax.sparse.ops.matmul import matmul
from graphax.sparse.tensor import SparseTensor
from graphax.sparse.elemental.dispatch import set_approx_active

N_CASES = int(os.environ.get("LATTICE_CASES", "60"))
BASE_SEED = int(os.environ.get("LATTICE_SEED", "20260802"))
TOL = 1e-9  # float64


def _mk(rng, out_specs, primal_specs):
    """Build a SparseTensor from per-dim specs.

    spec = ("dense", logical) | ("implicit", logical)
         | ("pair", logical_out, logical_primal, meta)  -- couples out slot i
           with primal slot j (consumed pairwise in order of appearance).
    Returns the tensor; val axes are allocated in spec order:
    per pair -> meta axis then out-block then primal-block (blocks > 1 only),
    dense dims -> one axis each, implicit -> none.
    """
    dims_out, dims_primal = [], []
    shape = []
    next_id = [0]

    def nid():
        next_id[0] += 1
        return next_id[0] - 1

    # collect pair specs first to couple out/primal slots
    pair_queue = [s for s in out_specs if s[0] == "pair"]
    axes_used = [0]

    def ax():
        axes_used[0] += 1
        return axes_used[0] - 1

    pair_axes = {}
    for k, s in enumerate(pair_queue):
        _, lo, lp, meta = s
        b1, b2 = lo // meta, lp // meta
        meta_ax = ax(); shape.append(meta)
        b1_ax = None
        if b1 > 1:
            b1_ax = ax(); shape.append(b1)
        b2_ax = None
        if b2 > 1:
            b2_ax = ax(); shape.append(b2)
        pair_axes[k] = (meta_ax, b1_ax, b2_ax, meta, b1, b2)

    pk = [0]
    for s in out_specs:
        if s[0] == "pair":
            meta_ax, b1_ax, b2_ax, meta, b1, b2 = pair_axes[pk[0]]
            i1, i2 = nid(), nid()
            dims_out.append(DiagonalIndex(
                i1, meta, meta_ax, i2, b1 if b1 > 1 else None, b1_ax))
            dims_primal.append(DiagonalIndex(
                i2, meta, meta_ax, i1, b2 if b2 > 1 else None, b2_ax))
            pk[0] += 1
        elif s[0] == "dense":
            a = ax(); shape.append(s[1])
            dims_out.append(DenseIndex(nid(), s[1], a))
        else:
            dims_out.append(DenseIndex(nid(), s[1], None))
    for s in primal_specs:
        if s[0] == "pair":
            continue  # already placed by its out partner
        if s[0] == "dense":
            a = ax(); shape.append(s[1])
            dims_primal.append(DenseIndex(nid(), s[1], a))
        else:
            dims_primal.append(DenseIndex(nid(), s[1], None))

    val = None
    if shape:
        val = jnp.asarray(rng.standard_normal(shape))
    return SparseTensor(tuple(dims_out), tuple(dims_primal), val,
                        check_consistency=False)


def _dense_mat(t):
    """Oracle form: (prod out logical, prod primal logical) matrix."""
    d = np.asarray(t.dense())
    po = int(np.prod([x.logical_size for x in t.out_dims])) or 1
    pp = int(np.prod([x.logical_size for x in t.primal_dims])) or 1
    return d.reshape(po, pp)


def _case_pair(rng):
    """One random (lhs, rhs) with contractable middle. Families:
    contracted middle is dense/implicit/coupled-with-a-free-dim on either
    side; metas equal, commensurable, coprime, or degenerate (1)."""
    Lc = rng.choice([4, 6, 12])          # contracted logical size
    Lo = rng.choice([3, 4, 8])           # lhs free out logical
    Lp = rng.choice([2, 5, 6])           # rhs free primal logical

    def side(free_logical, con_logical, is_lhs):
        style = rng.choice(["dense", "implicit", "pair"])
        if style == "pair":
            divs = [m for m in (1, 2, 3, 4, 6, 12)
                    if free_logical % m == 0 and con_logical % m == 0]
            meta = int(rng.choice(divs))
            if is_lhs:
                return [("pair", free_logical, con_logical, meta)], ["pair"]
            return [("pair", con_logical, free_logical, meta)], ["pair"]
        if is_lhs:
            return ([("dense", free_logical)],
                    [(style if style == "implicit" else "dense", con_logical)])
        return ([(style if style == "implicit" else "dense", con_logical)],
                [("dense", free_logical)])

    lo, lp = side(Lo, Lc, True)
    lhs = _mk(rng,
              lo,
              lp if lp != ["pair"] else [("pair",)])
    ro, rp = side(Lp, Lc, False)
    rhs = _mk(rng,
              ro if ro != ["pair"] else [("pair",)],
              rp if rp != ["pair"] else [("pair",)])
    # pair placeholder handling: _mk consumes pair specs from OUT list only
    return lhs, rhs


class TestContractOracle(unittest.TestCase):
    def test_random_contractions(self):
        fails = []
        for i in range(N_CASES):
            rng = np.random.default_rng(BASE_SEED + i)

            Lc = int(rng.choice([4, 6, 12]))
            Lo = int(rng.choice([3, 4, 8]))
            Lp = int(rng.choice([2, 5, 6]))

            def specs(free_l, con_l, lhs_side, rng=rng):
                style = str(rng.choice(["dense", "implicit", "pair"]))
                if style == "pair":
                    divs = [m for m in (1, 2, 3, 4, 6, 12)
                            if free_l % m == 0 and con_l % m == 0]
                    meta = int(rng.choice(divs))
                    if lhs_side:
                        return [("pair", free_l, con_l, meta)], []
                    return [("pair", con_l, free_l, meta)], []
                con = ("implicit", con_l) if style == "implicit" \
                    else ("dense", con_l)
                if lhs_side:
                    return [("dense", free_l)], [con]
                return [con], [("dense", free_l)]

            lo, lp = specs(Lo, Lc, True)
            rο, rp = specs(Lp, Lc, False)
            lhs = _mk(rng, lo, lp)
            rhs = _mk(rng, rο, rp)
            want = _dense_mat(lhs) @ _dense_mat(rhs)
            for eng, approx in (("incumbent", False), ("planner", True)):
                set_approx_active(approx)
                try:
                    got = matmul(lhs, rhs)
                    gm = _dense_mat(got)
                except Exception as e:
                    fails.append((i, f"{eng} RAISE {type(e).__name__}: "
                                     f"{str(e)[:90]}",
                                  [classify(d).value for d in lhs.dims],
                                  [classify(d).value for d in rhs.dims]))
                    continue
                finally:
                    set_approx_active(False)
                if gm.shape != want.shape:
                    fails.append((i, f"{eng} SHAPE {gm.shape} vs {want.shape}",
                                  [classify(d).value for d in lhs.dims],
                                  [classify(d).value for d in rhs.dims]))
                elif not np.allclose(gm, want, atol=TOL, rtol=TOL):
                    err = float(np.abs(gm - want).max())
                    fails.append((i, f"{eng} VALUE max|d|={err:.2e}",
                                  [classify(d).value for d in lhs.dims],
                                  [classify(d).value for d in rhs.dims]))
        msg = "\n".join(f"  case {i}: {m}  lhs={a} rhs={b}"
                        for i, m, a, b in fails[:15])
        self.assertEqual(
            len(fails), 0,
            f"{len(fails)}/{N_CASES} contraction cases disagree with the "
            f"dense oracle:\n{msg}")


class TestAddOracle(unittest.TestCase):
    def test_random_adds(self):
        fails = []
        for i in range(N_CASES):
            rng = np.random.default_rng(BASE_SEED + 10_000 + i)
            Lo = int(rng.choice([4, 6, 12]))
            Lp = int(rng.choice([4, 6, 8]))

            def one(rng=rng, Lo=Lo, Lp=Lp):
                style = str(rng.choice(["dense", "pair", "pair", "dense"]))
                if style == "pair":
                    divs = [m for m in (1, 2, 3, 4, 6)
                            if Lo % m == 0 and Lp % m == 0]
                    meta = int(rng.choice(divs))
                    return _mk(rng, [("pair", Lo, Lp, meta)], [])
                return _mk(rng, [("dense", Lo)], [("dense", Lp)])

            a, b = one(), one()
            want = _dense_mat(a) + _dense_mat(b)
            for eng, approx in (("incumbent", False), ("planner", True)):
              set_approx_active(approx)
              try:
                got = elementwise(a, b, jnp.add)
                gm = _dense_mat(got)
              except Exception as e:
                fails.append((i, f"{eng} RAISE {type(e).__name__}: {str(e)[:90]}",
                              [classify(d).value for d in a.dims],
                              [classify(d).value for d in b.dims]))
                continue
              finally:
                set_approx_active(False)
              if gm.shape != want.shape:
                fails.append((i, f"{eng} SHAPE {gm.shape} vs {want.shape}",
                              [classify(d).value for d in a.dims],
                              [classify(d).value for d in b.dims]))
              elif not np.allclose(gm, want, atol=TOL, rtol=TOL):
                fails.append((i, f"{eng} VALUE "
                                 f"max|d|={float(np.abs(gm-want).max()):.2e}",
                              [classify(d).value for d in a.dims],
                              [classify(d).value for d in b.dims]))
        msg = "\n".join(f"  case {i}: {m}  a={x} b={y}"
                        for i, m, x, y in fails[:15])
        self.assertEqual(
            len(fails), 0,
            f"{len(fails)}/{N_CASES} add cases disagree with the dense "
            f"oracle:\n{msg}")


if __name__ == "__main__":
    unittest.main()
