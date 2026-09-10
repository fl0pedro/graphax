"""``sparse/ops/join.py``: the two addends of a face merge get ONE container.

The two contributions to one Jacobian block share their logical dims -- same
ids, same ``logical_size`` -- and routinely disagree on STORAGE. Measured on a
transformer LM with only the contraction result approximated: 0 of 6 merge
faces had structurally identical addends, and the pairs they couple are often
DIFFERENT pairs of the same four dims.

Pinned here, on hand-built tensors whose layouts are the ones that measurement
found:

1. ``unify_containers`` returns two tensors with EQUAL ``container_of`` and
   changes no value.
2. ``lossless`` drops no non-zero -- checked densely, to the bit.
3. ``lossy`` returns structurally identical addends and lands on the fresh
   addend's own canonical container.
4. The case that used to RAISE: the two addends pair DIFFERENT dims, so the
   target's ``Diag`` would hit an already-paired dim. ``loosen_to_pairing``
   frees it first, losslessly.
5. An unknown mode RAISES rather than picking one.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.ops.join import (
    MatchFreshJoin, UnionJoin, container_of, loosen_to_pairing, pairing_of,
    reconcile_addends, structural_zero, unify_containers)
from graphax.sparse.tensor import SparseTensor

N = 4


def _dense_pair(vals):
    """out0 x primal1 as two INDEPENDENT physical axes."""
    return SparseTensor((DenseIndex(0, N, 0),), (DenseIndex(1, N, 1),),
                        jnp.asarray(vals, jnp.float32), fill_value=None)


def _diag_01(vals):
    """out0 coupled with primal1: a pure diagonal on ONE physical axis."""
    return SparseTensor((DiagonalIndex(0, N, 0, 1),),
                        (DiagonalIndex(1, N, 0, 0),),
                        jnp.asarray(vals, jnp.float32), fill_value=None)


def _d(st):
    return np.asarray(st.dense(), np.float64)


# --------------------------------------------------------------------------
# 1. unify_containers: one container, no value moved
# --------------------------------------------------------------------------

def test_unify_containers_equalises_the_structure_and_keeps_every_value():
    a = _diag_01([1.0, 2.0, 3.0, 4.0])
    b = _dense_pair(np.arange(16, dtype=np.float32).reshape(N, N) + 10.0)
    assert container_of(a) != container_of(b)
    da, db = _d(a), _d(b)
    a2, b2 = unify_containers(a, b)
    assert container_of(a2) == container_of(b2), (
        container_of(a2), container_of(b2))
    assert np.max(np.abs(_d(a2) - da)) == 0.0
    assert np.max(np.abs(_d(b2) - db)) == 0.0


def test_structural_zero_is_numerically_the_identity():
    a = _dense_pair(np.arange(16, dtype=np.float32).reshape(N, N) - 7.5)
    got = a + structural_zero(a)
    assert np.max(np.abs(_d(got) - _d(a))) == 0.0


# --------------------------------------------------------------------------
# 2. lossless drops no non-zero
# --------------------------------------------------------------------------

def test_lossless_drops_no_non_zero():
    a = _diag_01([1.0, -2.0, 3.0, -4.0])
    b = _dense_pair(np.arange(16, dtype=np.float32).reshape(N, N) + 100.0)
    ref = _d(a) + _d(b)
    f2, o2, out = reconcile_addends(a, b, "lossless")
    assert out.mode == "lossless" and out.matched_target and out.rules == ()
    assert container_of(f2) == container_of(o2)
    assert np.max(np.abs(_d(f2 + o2) - ref)) == 0.0
    # every non-zero of the DIAGONAL addend survives: it is the part a
    # container narrower than the union would have thrown away.
    assert np.count_nonzero(_d(f2)) == np.count_nonzero(_d(a))


def test_the_plain_add_is_what_lossless_emits_and_agrees_with_it():
    """``--approx-add lossless`` installs NO policy: graphax's sparse ``+``
    already builds the union container. The two must therefore agree."""
    a = _diag_01([5.0, 6.0, 7.0, 8.0])
    b = _dense_pair(np.arange(16, dtype=np.float32).reshape(N, N) * -0.5)
    f2, o2, _ = reconcile_addends(a, b, "lossless")
    assert np.max(np.abs(_d(a + b) - _d(f2 + o2))) == 0.0


# --------------------------------------------------------------------------
# 3. lossy: identical addends, on the FRESH addend's container
# --------------------------------------------------------------------------

def test_lossy_returns_identical_addends_on_the_fresh_container():
    fresh = _diag_01([1.0, 2.0, 3.0, 4.0])       # the approximated one
    old = _dense_pair(np.arange(16, dtype=np.float32).reshape(N, N) + 10.0)
    f2, o2, out = reconcile_addends(fresh, old, "lossy")
    assert out.mode == "lossy"
    assert container_of(f2) == container_of(o2), (
        container_of(f2), container_of(o2))
    assert out.matched_target, (out.container, out.target)
    # the fresh addend's values are untouched; the old edge lost its
    # off-diagonal, which is the whole point of `lossy`.
    assert np.max(np.abs(_d(f2) - _d(fresh))) == 0.0
    off = ~np.eye(N, dtype=bool)
    assert np.max(np.abs(_d(o2)[off])) == 0.0
    assert np.max(np.abs(np.diag(_d(o2)) - np.diag(_d(old)))) == 0.0


def test_lossy_is_cheaper_than_lossless_on_the_same_pair():
    fresh = _diag_01([1.0, 2.0, 3.0, 4.0])
    old = _dense_pair(np.arange(16, dtype=np.float32).reshape(N, N) + 10.0)
    fl, ol, _ = reconcile_addends(fresh, old, "lossless")
    fy, oy, _ = reconcile_addends(fresh, old, "lossy")
    def nbytes(st):
        return 0 if st.val is None else st.val.size * st.val.dtype.itemsize
    assert nbytes(fy + oy) < nbytes(fl + ol), (
        nbytes(fy + oy), nbytes(fl + ol))


# --------------------------------------------------------------------------
# 4. the case that used to raise: the two addends pair DIFFERENT dims
# --------------------------------------------------------------------------

def _four_dims_pairing(pair_a, pair_b, vals, shape):
    """A 4-dim Jacobian block (out0, out1 | primal2, primal3) where ONE
    out/primal pair is stored diagonally and the other two dims are free."""
    i, j = pair_a, pair_b          # i in {0,1}, j in {2,3}
    ax = {0: None, 1: None, 2: None, 3: None}
    ax[i] = 0
    ax[j] = 0                      # the coupled pair shares one physical axis
    free = [k for k in (0, 1, 2, 3) if k not in (i, j)]
    for n, k in enumerate(free, start=1):
        ax[k] = n

    def mk(k):
        if k == i:
            return DiagonalIndex(k, N, ax[k], j)
        if k == j:
            return DiagonalIndex(k, N, ax[k], i)
        return DenseIndex(k, N, ax[k])
    return SparseTensor((mk(0), mk(1)), (mk(2), mk(3)),
                        jnp.asarray(vals, jnp.float32).reshape(shape),
                        fill_value=None)


def test_a_mis_paired_dim_is_FREED_before_it_is_re_paired():
    """THE DEFECT THE FIRST TLM MEASUREMENT FOUND.

    ``apply_diag`` refuses to re-pair a dim already paired with a different
    partner -- and the two addends of a merge routinely pair different dims of
    the same block (the fresh one couples ``(out1, primal3)``, the old one
    ``(out0, primal2)``). Asking for the target's pair raised
    ``Diag pair conflict`` on 2 of 10 and 3 of 13 TLM merge faces.
    """
    rng = np.random.default_rng(0)
    fresh = _four_dims_pairing(1, 3, rng.normal(size=N * N * N),
                               (N, N, N))          # couples (1, 3)
    old = _four_dims_pairing(0, 2, rng.normal(size=N * N * N),
                             (N, N, N))            # couples (0, 2)
    assert pairing_of(fresh)[1] == 3 and pairing_of(old)[0] == 2
    # the loosening frees exactly the pair the target does not want
    loose = loosen_to_pairing(old, fresh)
    assert pairing_of(loose).get(0) is None, pairing_of(loose)
    # ... and it is LOSSLESS
    assert np.max(np.abs(_d(loose) - _d(old))) == 0.0
    # so the reconciliation now succeeds instead of raising
    f2, o2, out = reconcile_addends(fresh, old, "lossy")
    assert container_of(f2) == container_of(o2)


def test_loosening_is_a_no_op_when_the_pairings_already_agree():
    a = _diag_01([1.0, 2.0, 3.0, 4.0])
    assert loosen_to_pairing(a, a) is a


# --------------------------------------------------------------------------
# 5. an unknown mode raises
# --------------------------------------------------------------------------

def test_an_unknown_mode_raises_rather_than_picking_one():
    a = _diag_01([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="unknown join mode"):
        reconcile_addends(a, a, "same")


def test_the_policy_constructors_name_their_mode():
    assert UnionJoin().mode == "lossless"
    assert MatchFreshJoin().mode == "lossy"
    # the policy is NOT callable: it takes two addends, so graphax dispatches
    # on its TYPE and a single-argument wrapper around it would be wrong.
    assert not callable(UnionJoin())


def test_the_policy_pre_hook_runs_on_the_old_edge_before_reconciling():
    seen = []

    def pre(st):
        seen.append(container_of(st))
        return st

    fresh = _diag_01([1.0, 2.0, 3.0, 4.0])
    old = _dense_pair(np.arange(16, dtype=np.float32).reshape(N, N))
    f2, o2 = MatchFreshJoin(pre=pre).reconcile(fresh, old)
    assert seen == [container_of(old)], seen
    assert container_of(f2) == container_of(o2)
