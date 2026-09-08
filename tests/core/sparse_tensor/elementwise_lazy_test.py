"""The lazy elementwise path against the materializing one (dsnn-3qm.72).

``elementwise`` combines two operands in the structure they already have when
a rule covers the signature, and falls through to ``_materializing_general``
when none does. This file is what lets the lazy path ship: for every
structured signature it compares the two on the DENSE form, which is the only
comparison that cannot be fooled by a different-but-equivalent encoding.

The oracle is ``_materializing_general`` itself -- the code that shipped before
and that the rest of the suite already checks against ``jax.grad``. Calling it
directly needs no flag and no monkeypatch.

This exists because the path once shipped wrong. Armed behind the deleted
``GRAPHAX_EINSUM_EW`` flag, a float64 model diff caught it diverging by up to
0.8 on ViT COMPRESS variants and the divergence was never localized to a rule.
Every rule is exercised here for a UNION op and an INTERSECTION op, with zero
and non-zero fills, and the storage is asserted too: a rule that returns the
right numbers by materializing has not done its job.
"""
from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor

# ``graphax.sparse.ops.__init__`` re-exports the elementwise FUNCTION under the
# module's own name, so a plain ``from ... import elementwise`` binds the
# function. Take the module itself.
ew = importlib.import_module("graphax.sparse.ops.elementwise")

M, P, K, Q = 4, 3, 2, 5


def _n(shape, k):
    return jr.normal(jr.PRNGKey(k), shape).astype(jnp.float32)


def _st(out_dims, primal_dims, val, **kw):
    return SparseTensor(out_dims, primal_dims, val, check_consistency=False, **kw)


def _pair(size, axis, block, block_axis, i=0, j=1, **kw):
    """A DiagonalIndex pair with ids ``i`` and ``j``."""
    return (DiagonalIndex(i, size, axis, j, block, block_axis),
            DiagonalIndex(j, size, axis, i, block, block_axis))


# --- the signatures ---------------------------------------------------------
# (name, lhs, rhs, expected rule or None when the lazy path must decline)
def _cases():
    out = []

    # eq: same block grid, both physical
    a1, a2 = _pair(M, 0, P, 1)
    b1, b2 = _pair(M, 0, P, 1)
    out.append(("eq_block_pair",
                _st((a1,), (a2,), _n((M, P, P), 1)),
                _st((b1,), (b2,), _n((M, P, P), 2)), "eq"))

    # eq: plain dense dims, matching layout
    out.append(("eq_dense",
                _st((DenseIndex(0, M, 0),), (DenseIndex(1, Q, 1),), _n((M, Q), 3)),
                _st((DenseIndex(0, M, 0),), (DenseIndex(1, Q, 1),), _n((M, Q), 4)),
                "eq"))

    # eq: transposed physical layout, same structure
    out.append(("eq_transposed",
                _st((DenseIndex(0, M, 0),), (DenseIndex(1, Q, 1),), _n((M, Q), 5)),
                _st((DenseIndex(0, M, 1),), (DenseIndex(1, Q, 0),), _n((Q, M), 6)),
                "eq"))

    # eq: a role implicit on BOTH sides stays implicit
    out.append(("eq_both_implicit",
                _st((DenseIndex(0, M, None),), (DenseIndex(1, Q, 0),), _n((Q,), 7)),
                _st((DenseIndex(0, M, None),), (DenseIndex(1, Q, 0),), _n((Q,), 8)),
                "eq"))

    # ibroad: a dense role implicit on one side, physical on the other
    out.append(("ibroad_dense_role",
                _st((DenseIndex(0, M, 0),), (DenseIndex(1, Q, 1),), _n((M, Q), 9)),
                _st((DenseIndex(0, M, None),), (DenseIndex(1, Q, 0),), _n((Q,), 10)),
                "ibroad"))

    # ibroad: the META diagonal of a pair implicit on one side
    c1, c2 = _pair(M, None, P, 0)
    out.append(("ibroad_meta_diagonal",
                _st((a1,), (a2,), _n((M, P, P), 11)),
                _st((c1,), (c2,), _n((P, P), 12)), "ibroad"))

    # ibroad: a within-block axis implicit on one side
    d1 = DiagonalIndex(0, M, 0, 1, P, None)
    d2 = DiagonalIndex(1, M, 0, 0, P, 1)
    out.append(("ibroad_within_block",
                _st((a1,), (a2,), _n((M, P, P), 13)),
                _st((d1,), (d2,), _n((M, P), 14)), "ibroad"))

    # uu: both pure structure
    out.append(("uu_both_uniform",
                _st((DenseIndex(0, M, None),), (DenseIndex(1, Q, None),), None,
                    scalar_mult=jnp.asarray(2.0)),
                _st((DenseIndex(0, M, None),), (DenseIndex(1, Q, None),), None,
                    scalar_mult=jnp.asarray(3.0)), "uu"))

    # u_x: one pure structure against a physical partner
    out.append(("u_x_uniform_lhs",
                _st((DenseIndex(0, M, None),), (DenseIndex(1, Q, None),), None,
                    scalar_mult=jnp.asarray(2.0)),
                _st((DenseIndex(0, M, 0),), (DenseIndex(1, Q, 1),), _n((M, Q), 15)),
                "u_x"))
    out.append(("u_x_uniform_rhs",
                _st((DenseIndex(0, M, 0),), (DenseIndex(1, Q, 1),), _n((M, Q), 16)),
                _st((DenseIndex(0, M, None),), (DenseIndex(1, Q, None),), None,
                    scalar_mult=jnp.asarray(5.0)), "u_x"))

    # --- signatures the lazy path MUST decline -----------------------------
    # a MISALIGNED block grid: genuine least-common-multiple tiling
    e1, e2 = _pair(4, 0, 3, 1)
    f1, f2 = _pair(6, 0, 2, 1)
    out.append(("misaligned_grid",
                _st((e1,), (e2,), _n((4, 3, 3), 17)),
                _st((f1,), (f2,), _n((6, 2, 2), 18)), None))

    # a sparse role against a dense role at the same id
    out.append(("sparse_dense_mix",
                _st((a1,), (a2,), _n((M, P, P), 19)),
                _st((DenseIndex(0, M * P, 0),), (DenseIndex(1, M * P, 1),),
                    _n((M * P, M * P), 20)), None))

    return out


CASES = _cases()
IDS = [c[0] for c in CASES]
OPS = {"union": (jnp.add, False), "intersection": (jnp.multiply, True)}


def _dense(t):
    return np.asarray(t.dense(), np.float64)


def _stored(t):
    return 0 if t.val is None else int(t.val.size)


def _oracle(lhs, rhs, op, is_intersection):
    return ew._materializing_general(lhs, rhs, op, is_intersection, None)


@pytest.mark.parametrize("name,lhs,rhs,rule", CASES, ids=IDS)
@pytest.mark.parametrize("opname", sorted(OPS))
def test_lazy_equals_the_materializing_path(name, lhs, rhs, rule, opname):
    op, is_inter = OPS[opname]
    ew.reset_lazy_stats()
    got = ew.elementwise(lhs, rhs, op, is_intersection=is_inter)
    want = _oracle(lhs, rhs, op, is_intersection=is_inter)
    g, w = _dense(got), _dense(want)
    assert g.shape == w.shape, f"{name}/{opname}: {g.shape} vs {w.shape}"
    np.testing.assert_allclose(g, w, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name,lhs,rhs,rule", CASES, ids=IDS)
@pytest.mark.parametrize("opname", sorted(OPS))
def test_the_expected_rule_fires(name, lhs, rhs, rule, opname):
    """A rule that returns the right numbers by materializing has not done its
    job, so the rule that fires is pinned too -- and so is the decline."""
    op, is_inter = OPS[opname]
    ew.reset_lazy_stats()
    ew.elementwise(lhs, rhs, op, is_intersection=is_inter)
    hits = [k[len("rule:"):] for k in ew.LAZY_STATS if k.startswith("rule:")]
    if rule is None:
        assert not hits, f"{name}/{opname}: expected a decline, got {hits}"
        assert any(k.startswith("skip:") for k in ew.LAZY_STATS), ew.LAZY_STATS
    else:
        assert hits == [rule], f"{name}/{opname}: expected {rule}, got {dict(ew.LAZY_STATS)}"


@pytest.mark.parametrize("name,lhs,rhs,rule", CASES, ids=IDS)
def test_the_lazy_path_stores_no_more_than_the_materializing_one(name, lhs, rhs, rule):
    """The point of the path. Storage may only go DOWN."""
    if rule is None:
        pytest.skip("declines; storage is the materializing path's by definition")
    got = ew.elementwise(lhs, rhs, jnp.add)
    want = _oracle(lhs, rhs, jnp.add, False)
    assert _stored(got) <= _stored(want), (
        f"{name}: lazy stores {_stored(got)}, materializing stores {_stored(want)}")


@pytest.mark.parametrize("name,lhs,rhs,rule", CASES, ids=IDS)
def test_a_non_zero_fill_still_matches(name, lhs, rhs, rule):
    """The fill algebra is the part most easily got wrong: the lazy path must
    reproduce ``_reconstruct_result``'s, not its own."""
    lf = lhs.copy(fill_value=jnp.asarray(0.25, lhs.dtype))
    rf = rhs.copy(fill_value=jnp.asarray(-0.5, rhs.dtype))
    got = ew.elementwise(lf, rf, jnp.add)
    want = _oracle(lf, rf, jnp.add, False)
    np.testing.assert_allclose(_dense(got), _dense(want), rtol=1e-6, atol=1e-6)


def test_the_zero_fill_marker_survives():
    """Both fills statically zero and a zero-preserving op keeps ``None``, so
    downstream code can still read the result as statically zero."""
    lhs, rhs = CASES[0][1], CASES[0][2]
    out = ew.elementwise(lhs, rhs, jnp.add)
    assert lhs.fill_value is None and rhs.fill_value is None
    assert out.fill_value is None


def test_every_rule_is_exercised():
    """Totality: the ledger must show all four rules across the battery, or
    this file is not testing what it claims to."""
    ew.reset_lazy_stats()
    for _n_, lhs, rhs, _r in CASES:
        for op, is_inter in OPS.values():
            ew.elementwise(lhs, rhs, op, is_intersection=is_inter)
    fired = {k[len("rule:"):] for k in ew.LAZY_STATS if k.startswith("rule:")}
    assert {"eq", "ibroad", "uu", "u_x"} <= fired, fired
