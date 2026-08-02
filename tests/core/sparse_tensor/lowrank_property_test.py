"""Property suite for the edge-level LowRank lattice class (L3).

Random factored pairs vs the float64 dense oracle, through the REAL op
entries (``matmul`` / ``elementwise``), under BOTH engines
(``set_approx_active`` toggles incumbent / planner for the factor-side ops).
Covers: dense round-trip, factor_left / factor_right / middle_fold
contraction, rank-concat add (+ subtract), the max-rank spill, dense-absorb
spill, multiply spill, and count=True plumbing.
"""
import os
import unittest

import numpy as np

import jax
import jax.numpy as jnp

from graphax.sparse.indexes import DenseIndex
from graphax.sparse.lowrank import LOWRANK_STATS, LowRankTensor
from graphax.sparse.ops.elementwise import elementwise
from graphax.sparse.ops.matmul import matmul
from graphax.sparse.tensor import SparseTensor
from graphax.sparse.elemental.dispatch import set_approx_active

N_CASES = int(os.environ.get("LOWRANK_CASES", "20"))
BASE_SEED = int(os.environ.get("LOWRANK_SEED", "20260803"))
TOL = 1e-9

_X64_PREV = None


def setUpModule():
    global _X64_PREV
    _X64_PREV = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)


def tearDownModule():
    jax.config.update("jax_enable_x64", _X64_PREV)


def _dense_st(rng, sizes, n_out):
    # ids are PER-TENSOR contiguous from 0 (graphax consistency assert);
    # cross-operand alignment is positional.
    val = jnp.asarray(rng.standard_normal(sizes))
    dims = tuple(DenseIndex(i, s, i) for i, s in enumerate(sizes))
    return SparseTensor(dims[:n_out], dims[n_out:], val)


def _lr(rng, m, r, n):
    """Random LowRank (m x n) of rank r."""
    u = _dense_st(rng, [m, r], 1)
    v = _dense_st(rng, [r, n], 1)
    return LowRankTensor(u, v)


def _oracle(t):
    if getattr(t, "_is_lowrank", False):
        return (np.asarray(t.u.dense(), dtype=np.float64)
                @ np.asarray(t.v.dense(), dtype=np.float64))
    d = np.asarray(t.dense(), dtype=np.float64)
    po = int(np.prod([x.logical_size for x in t.out_dims])) or 1
    return d.reshape(po, -1)


def _engines():
    for label, approx in (("incumbent", False), ("planner", True)):
        set_approx_active(approx)
        try:
            yield label
        finally:
            set_approx_active(False)


class LowRankPropertyTest(unittest.TestCase):
    def _rng(self, case, tag):
        # str hash() is per-process randomized — use a stable digest so every
        # case reproduces from LOWRANK_SEED alone.
        stable = int.from_bytes(tag.encode(), "little") % 100000
        return np.random.default_rng(BASE_SEED + 1000 * stable + case)

    def test_dense_roundtrip(self):
        for case in range(N_CASES):
            rng = self._rng(case, "rt")
            m, r, n = rng.integers(2, 9, size=3)
            lr = _lr(rng, int(m), int(r), int(n))
            np.testing.assert_allclose(
                np.asarray(lr.dense(), dtype=np.float64).reshape(int(m), int(n)),
                _oracle(lr), atol=TOL)

    def test_contract_factor_right(self):
        for case in range(N_CASES):
            rng = self._rng(case, "fr")
            m, r, n, q = (int(x) for x in rng.integers(2, 9, size=4))
            lr = _lr(rng, m, r, n)
            b = _dense_st(rng, [n, q], 1)
            for eng in _engines():
                with self.subTest(case=case, engine=eng):
                    out = matmul(lr, b)
                    self.assertTrue(getattr(out, "_is_lowrank", False))
                    self.assertEqual(out.rank, r)
                    np.testing.assert_allclose(
                        _oracle(out), _oracle(lr) @ _oracle(b), atol=TOL)

    def test_contract_factor_left(self):
        for case in range(N_CASES):
            rng = self._rng(case, "fl")
            p, m, r, n = (int(x) for x in rng.integers(2, 9, size=4))
            a = _dense_st(rng, [p, m], 1)
            lr = _lr(rng, m, r, n)
            for eng in _engines():
                with self.subTest(case=case, engine=eng):
                    out = matmul(a, lr)
                    self.assertTrue(getattr(out, "_is_lowrank", False))
                    np.testing.assert_allclose(
                        _oracle(out), _oracle(a) @ _oracle(lr), atol=TOL)

    def test_contract_middle_fold(self):
        for case in range(N_CASES):
            rng = self._rng(case, "mf")
            m, r1, k, r2, n = (int(x) for x in rng.integers(2, 7, size=5))
            lr1 = _lr(rng, m, r1, k)
            lr2 = _lr(rng, k, r2, n)
            for eng in _engines():
                with self.subTest(case=case, engine=eng):
                    out = matmul(lr1, lr2)
                    self.assertTrue(getattr(out, "_is_lowrank", False))
                    np.testing.assert_allclose(
                        _oracle(out), _oracle(lr1) @ _oracle(lr2), atol=TOL)

    def test_add_rank_concat_and_subtract(self):
        for case in range(N_CASES):
            rng = self._rng(case, "rc")
            m, r1, r2, n = (int(x) for x in rng.integers(2, 7, size=4))
            lr1 = _lr(rng, m, r1, n)
            lr2 = _lr(rng, m, r2, n)
            for name, op in (("add", jnp.add), ("subtract", jnp.subtract)):
                with self.subTest(case=case, op=name):
                    out = elementwise(lr1, lr2, op)
                    self.assertTrue(getattr(out, "_is_lowrank", False))
                    self.assertEqual(out.rank, r1 + r2)
                    want = (_oracle(lr1) + _oracle(lr2) if name == "add"
                            else _oracle(lr1) - _oracle(lr2))
                    np.testing.assert_allclose(_oracle(out), want, atol=TOL)

    def test_add_max_rank_spills_dense(self):
        rng = self._rng(0, "cap")
        lr1 = _lr(rng, 5, 3, 4)
        lr2 = _lr(rng, 5, 3, 4)
        old = os.environ.get("GRAPHAX_LOWRANK_MAX_RANK")
        os.environ["GRAPHAX_LOWRANK_MAX_RANK"] = "4"
        try:
            out = elementwise(lr1, lr2, jnp.add)
        finally:
            if old is None:
                os.environ.pop("GRAPHAX_LOWRANK_MAX_RANK", None)
            else:
                os.environ["GRAPHAX_LOWRANK_MAX_RANK"] = old
        self.assertFalse(getattr(out, "_is_lowrank", False))
        np.testing.assert_allclose(
            _oracle(out), _oracle(lr1) + _oracle(lr2), atol=TOL)

    def test_add_dense_absorb_spills(self):
        for case in range(N_CASES):
            rng = self._rng(case, "da")
            m, r, n = (int(x) for x in rng.integers(2, 7, size=3))
            lr = _lr(rng, m, r, n)
            d = _dense_st(rng, [m, n], 1)
            out = elementwise(lr, d, jnp.add)
            self.assertFalse(getattr(out, "_is_lowrank", False))
            np.testing.assert_allclose(
                _oracle(out), _oracle(lr) + _oracle(d), atol=TOL)

    def test_multiply_spills(self):
        rng = self._rng(0, "mul")
        lr1 = _lr(rng, 4, 2, 5)
        lr2 = _lr(rng, 4, 2, 5)
        out = elementwise(lr1, lr2, jnp.multiply, is_intersection=True)
        self.assertFalse(getattr(out, "_is_lowrank", False))
        np.testing.assert_allclose(
            _oracle(out), _oracle(lr1) * _oracle(lr2), atol=TOL)

    def test_count_plumbing(self):
        rng = self._rng(0, "cnt")
        lr = _lr(rng, 4, 2, 5)
        b = _dense_st(rng, [5, 6], 1)
        out, counts = matmul(lr, b, count=True)
        self.assertTrue(getattr(out, "_is_lowrank", False))
        self.assertEqual(len(counts), 3)
        lr2 = _lr(rng, 4, 3, 5)
        out2, n = elementwise(lr, lr2, jnp.add, count=True)
        self.assertTrue(getattr(out2, "_is_lowrank", False))
        self.assertEqual(int(n), 0)

    def test_stats_populated(self):
        self.assertTrue(any(k.startswith("contract:")
                            for k in LOWRANK_STATS))
        self.assertTrue(any(k.startswith("add:") for k in LOWRANK_STATS))


if __name__ == "__main__":
    unittest.main()
