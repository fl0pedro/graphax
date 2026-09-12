"""``ops/bilinear.py``: the lazy accumulators equal the dense ones, and never
build the dense form (ticket dsnn-3qm.62, owner ruling (c)).

Every case is checked against ``dense()`` -- the oracle is graphax's own
densifier, so a disagreement is a bug in one of the two and the test says which
structure it was. The no-materialize claim is checked by COUNTING ``dense()``
calls, not by reading the code.
"""
from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.ops.bilinear import (
    LazyContractionUnsupported, bilinear_accumulators, squared_norm)
from graphax.sparse.tensor import SparseTensor

F32 = jnp.float32


def _rng(n, seed):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(n), np.float32)


def _dense_acc(e, a):
    """The four accumulators from the two DENSE forms -- the oracle."""
    ef = np.asarray(e, np.float64).ravel()
    af = np.asarray(a, np.float64).ravel()
    return (float(ef @ af), float(ef @ ef), float(af @ af),
            float(((ef - af) ** 2).sum()))


def _check(e, a, *, atol=2e-5):
    """``bilinear_accumulators(e, a)`` against the dense oracle."""
    dot, e2, a2 = bilinear_accumulators(e, a, dtype=F32)
    rr = float(e2) - 2.0 * float(dot) + float(a2)
    want = _dense_acc(_oracle(e), _oracle(a))
    got = (float(dot), float(e2), float(a2), rr)
    scale = max(1.0, max(abs(x) for x in want))
    for name, g, w in zip(("dot", "||e||^2", "||a||^2", "||e-a||^2"), got, want):
        assert abs(g - w) <= atol * scale, (
            f"{name}: lazy {g!r} vs dense {w!r}  (e={e!r} a={a!r})")
    return got


# --------------------------------------------------------------------------
# fixtures: one constructor per structural kind
# --------------------------------------------------------------------------
def st_dense(val, sizes, implicit=(), sm=1.0, fill=None):
    """Materialized dense dims, ``implicit`` positions storing one
    representative (``axis is None``)."""
    dims, ax = [], 0
    for pos, n in enumerate(sizes):
        if pos in implicit:
            dims.append(DenseIndex(pos, n, axis=None))
        else:
            dims.append(DenseIndex(pos, n, axis=ax))
            ax += 1
    return SparseTensor((), tuple(dims), val, scalar_mult=jnp.asarray(sm, F32),
                        fill_value=fill, check_consistency=False)


def st_pair(N, Bo, Bi, *, val=None, sm=1.0, fill=None, seed=0):
    """One meta-block-diagonal pair: logical ``(N*Bo, N*Bi)``."""
    d0 = DiagonalIndex(0, N, axis=0, other_id=1, block_size=Bo, block_axis=1)
    d1 = DiagonalIndex(1, N, axis=0, other_id=0, block_size=Bi, block_axis=2)
    if val is _UNIFORM:
        v = None
    elif val is None:
        v = _rng(N * Bo * Bi, seed).reshape(N, Bo, Bi)
    else:
        v = val
    return SparseTensor((d0,), (d1,), v, scalar_mult=jnp.asarray(sm, F32),
                        fill_value=fill, check_consistency=False)


class _Uniform:
    pass


_UNIFORM = _Uniform()


# --------------------------------------------------------------------------
# 1. a plain array on both sides: the accumulators are the textbook ones
# --------------------------------------------------------------------------
def test_two_arrays():
    e = _rng(12, 1).reshape(4, 3)
    a = _rng(12, 2).reshape(4, 3)
    _check(e, a)


def test_identical_arrays_give_a_zero_residual():
    e = _rng(12, 3).reshape(4, 3)
    dot, e2, a2 = bilinear_accumulators(e, e, dtype=F32)
    assert float(dot) == pytest.approx(float(e2), rel=1e-6)
    assert float(e2) - 2 * float(dot) + float(a2) == pytest.approx(0.0, abs=1e-4)


# --------------------------------------------------------------------------
# 2. a materialized SparseTensor is the array case plus a scalar_mult
# --------------------------------------------------------------------------
@pytest.mark.parametrize("sm", [1.0, 0.5, -2.0, 0.0])
def test_materialized_sparse_against_array(sm):
    e = _rng(12, 4).reshape(4, 3)
    a = st_dense(_rng(12, 5).reshape(4, 3), (4, 3), sm=sm)
    _check(e, a)


def test_scalar_mult_is_not_baked_into_a_copy_of_val():
    """The scale is folded into the scalar result, so ``val`` is untouched."""
    v = _rng(12, 6).reshape(4, 3)
    a = st_dense(v, (4, 3), sm=3.0)
    bilinear_accumulators(_rng(12, 7).reshape(4, 3), a, dtype=F32)
    assert a.val is v


# --------------------------------------------------------------------------
# 3. IMPLICIT dims -- a broadcast, not a fill
# --------------------------------------------------------------------------
@pytest.mark.parametrize("implicit", [(0,), (1,), (0, 1), (0, 2), (1, 2)])
def test_implicit_dims_against_an_array(implicit):
    sizes = (4, 3, 2)
    rep_shape = tuple(n for pos, n in enumerate(sizes) if pos not in implicit)
    rep = _rng(int(np.prod(rep_shape)) if rep_shape else 1, 8).reshape(rep_shape)
    a = st_dense(rep, sizes, implicit=implicit, sm=0.75)
    e = _rng(int(np.prod(sizes)), 9).reshape(sizes)
    _check(e, a)


def test_both_sides_implicit_in_the_same_frame():
    rep_e = _rng(3, 10)
    rep_a = _rng(3, 11)
    e = st_dense(rep_e, (4, 3), implicit=(0,), sm=2.0)
    a = st_dense(rep_a, (4, 3), implicit=(0,), sm=-0.5)
    _check(e, a)


def test_uniform_leaf_with_only_implicit_dims_and_a_zero_scale():
    """The owner's degenerate case: ``val is None`` + every dim implicit +
    ``scalar_mult == 0`` IS a dense zero tensor, and costs nothing."""
    z = SparseTensor((), (DenseIndex(0, 4, axis=None), DenseIndex(1, 3, axis=None)),
                     None, scalar_mult=jnp.asarray(0.0, F32),
                     check_consistency=False)
    e = _rng(12, 12).reshape(4, 3)
    dot, e2, a2 = bilinear_accumulators(e, z, dtype=F32)
    assert float(dot) == 0.0 and float(a2) == 0.0
    assert float(e2) == pytest.approx(float(jnp.sum(e ** 2)), rel=1e-6)
    # and the residual IS ||e||^2
    assert float(e2) - 2 * float(dot) + float(a2) == pytest.approx(float(e2), rel=1e-6)


# --------------------------------------------------------------------------
# 4. DIAGONAL PAIRS
# --------------------------------------------------------------------------
@pytest.mark.parametrize("N,Bo,Bi", [(4, 1, 1), (2, 2, 2), (2, 3, 2), (3, 2, 1),
                                     (1, 3, 4), (5, 1, 2)])
@pytest.mark.parametrize("sm", [1.0, -0.25])
def test_one_diagonal_pair_against_an_array(N, Bo, Bi, sm):
    a = st_pair(N, Bo, Bi, sm=sm, seed=13)
    e = _rng(N * Bo * N * Bi, 14).reshape(N * Bo, N * Bi)
    _check(e, a)


@pytest.mark.parametrize("fill", [0.0, 1.5, -3.0])
def test_a_diagonal_pair_with_a_NON_ZERO_fill(fill):
    a = st_pair(3, 2, 2, sm=0.5, fill=jnp.asarray(fill, F32), seed=15)
    e = _rng(6 * 6, 16).reshape(6, 6)
    _check(e, a)


def test_a_uniform_diagonal_pair():
    a = st_pair(3, 2, 2, val=_UNIFORM, sm=1.25, seed=0)
    e = _rng(36, 17).reshape(6, 6)
    _check(e, a)


def test_two_pairs_in_the_same_frame():
    e = st_pair(3, 2, 2, sm=1.0, seed=18)
    a = st_pair(3, 2, 2, sm=-0.5, seed=19)
    _check(e, a)


def test_the_exact_side_may_be_the_structured_one():
    e = st_pair(4, 1, 1, sm=1.0, seed=20)
    a = _rng(16, 21).reshape(4, 4)
    _check(e, a)


# --------------------------------------------------------------------------
# 4b. THE DEGENERATE PAIR FORMS -- the ones Helmholtz actually produces
#
# Refusing these was a REGRESSION: tests/landscape_map_sweep_test.py::
# test_measure_singleton_and_stacks went red on all four plan classes (skip,
# quant, compress, diag) because alphagrad's grad-cosine on Helmholtz compares
# exactly this pair of forms. The dims below are transcribed verbatim from job
# 65010's census, so these two tests are the regression pinned.
# --------------------------------------------------------------------------
def st_one_block_pair(val):
    """``size=1, axis=None, block_size=B, block_axis=0/1``: ONE meta block, so
    the dense form is just ``val``. Helmholtz's EXACT Jacobian leaf."""
    bo, bi = val.shape
    d0 = DiagonalIndex(0, 1, axis=None, other_id=1, block_size=bo, block_axis=0)
    d1 = DiagonalIndex(1, 1, axis=None, other_id=0, block_size=bi, block_axis=1)
    return SparseTensor((d0,), (d1,), val, check_consistency=False)


def st_pure_diagonal(diag):
    """``size=N, axis=0, block_size=None, block_axis=None``: ``val (N,)`` is the
    diagonal of a logical ``(N, N)``. Helmholtz's APPROXIMATED leaf after a
    SKIP / QUANT / COMPRESS / DIAG."""
    n = int(diag.shape[0])
    d0 = DiagonalIndex(0, n, axis=0, other_id=1)
    d1 = DiagonalIndex(1, n, axis=0, other_id=0)
    return SparseTensor((d0,), (d1,), diag, check_consistency=False)


def test_a_one_block_pair_is_a_dense_matrix():
    t = st_one_block_pair(_rng(16, 60).reshape(4, 4))
    assert t.shape == (4, 4)
    got = float(squared_norm(t, F32))
    want = float((_oracle(t) ** 2).sum())
    assert got == pytest.approx(want, rel=2e-5)
    _check(_rng(16, 61).reshape(4, 4), t)


def test_a_pure_diagonal_pair():
    t = st_pure_diagonal(_rng(4, 62))
    assert t.shape == (4, 4)
    _check(_rng(16, 63).reshape(4, 4), t)


def test_THE_HELMHOLTZ_PAIR_one_block_exact_against_a_pure_diagonal_approx():
    """The exact combination that made landscape_map_sweep_test red."""
    e = st_one_block_pair(_rng(16, 64).reshape(4, 4))
    a = st_pure_diagonal(_rng(4, 65))
    _check(e, a)
    # and the other order, since _gradient_similarity sees both
    _check(a, e)


def test_two_one_block_pairs_against_each_other():
    """Also in the Helmholtz census: both sides the one-block form."""
    _check(st_one_block_pair(_rng(16, 66).reshape(4, 4)),
           st_one_block_pair(_rng(16, 67).reshape(4, 4)))


def test_a_broadcast_meta_pair_with_N_gt_1():
    """``size=N>1, axis=None``: the SAME block on every diagonal position -- a
    broadcast along the meta, not a missing axis."""
    d0 = DiagonalIndex(0, 3, axis=None, other_id=1, block_size=2, block_axis=0)
    d1 = DiagonalIndex(1, 3, axis=None, other_id=0, block_size=2, block_axis=1)
    t = SparseTensor((d0,), (d1,), _rng(4, 68).reshape(2, 2),
                     scalar_mult=jnp.asarray(0.5, F32), check_consistency=False)
    assert t.shape == (6, 6)
    _check(_rng(36, 69).reshape(6, 6), t)
    assert float(squared_norm(t, F32)) == pytest.approx(
        float((_oracle(t) ** 2).sum()), rel=2e-5)


def test_a_broadcast_block_axis():
    """``block_size=2, block_axis=None``: the block content is constant."""
    d0 = DiagonalIndex(0, 3, axis=0, other_id=1, block_size=2, block_axis=None)
    d1 = DiagonalIndex(1, 3, axis=0, other_id=0, block_size=2, block_axis=1)
    t = SparseTensor((d0,), (d1,), _rng(6, 70).reshape(3, 2),
                     check_consistency=False)
    assert t.shape == (6, 6)
    _check(_rng(36, 71).reshape(6, 6), t)


# --------------------------------------------------------------------------
# 5. THE CLAIM: nothing materializes
# --------------------------------------------------------------------------
def test_no_operand_is_ever_densified(monkeypatch):
    """Count ``SparseTensor.dense`` calls across every structural kind above."""
    calls = []
    real = SparseTensor.dense

    def spy(self, **kw):
        calls.append(tuple(int(d.logical_size) for d in self.dims))
        return real(self, **kw)

    monkeypatch.setattr(SparseTensor, "dense", spy)
    e_arr = _rng(36, 22).reshape(6, 6)
    for a in (st_pair(3, 2, 2, seed=23),
              st_pair(3, 2, 2, val=_UNIFORM),
              st_pair(3, 2, 2, fill=jnp.asarray(2.0, F32), seed=24),
              st_dense(_rng(6, 25), (6, 6), implicit=(0,)),
              st_dense(_rng(36, 26).reshape(6, 6), (6, 6))):
        bilinear_accumulators(e_arr, a, dtype=F32)
        squared_norm(a, F32)
    assert calls == [], f"densified: {calls}"


def test_the_gather_is_compact_not_logical():
    """A 1024x1024 logical pair storing 1024 values must not allocate 1 Mi
    cells anywhere: the peak intermediate is the size of the gather."""
    N = 1024
    a = st_pair(N, 1, 1, seed=27)
    e = _rng(N * N, 28).reshape(N, N)
    dot, e2, a2 = bilinear_accumulators(e, a, dtype=F32)
    diag = jnp.diagonal(e)
    assert float(dot) == pytest.approx(float(jnp.sum(diag * a.val.reshape(-1))),
                                       rel=1e-4)
    assert float(a2) == pytest.approx(float(jnp.sum(a.val ** 2)), rel=1e-4)


# --------------------------------------------------------------------------
# 6. WHAT IT REFUSES -- loudly, never by densifying
# --------------------------------------------------------------------------
def test_mismatched_logical_shapes_raise():
    with pytest.raises(LazyContractionUnsupported, match="logical shapes differ"):
        bilinear_accumulators(_rng(12, 29).reshape(4, 3),
                              _rng(12, 30).reshape(3, 4), dtype=F32)


def test_two_DIFFERENT_structures_raise_instead_of_densifying():
    """Two operands that BOTH carry a broadcast or a fill, with different
    frames: a pure diagonal (fill over 12 of its 16 cells) against an
    implicit-dim tensor (a broadcast over its rows), same logical shape. Neither
    can serve as a gather source and the frames differ, so there is no honest
    lazy contraction -- this must RAISE, not fall back to .dense()."""
    e = st_pair(4, 1, 1, seed=31)                       # logical (4, 4), n_fill 12
    a = st_dense(_rng(4, 32), (4, 4), implicit=(0,))    # logical (4, 4), n_bcast 4
    with pytest.raises(LazyContractionUnsupported, match="share no frame"):
        bilinear_accumulators(e, a, dtype=F32)
    with pytest.raises(LazyContractionUnsupported, match="share no frame"):
        bilinear_accumulators(a, e, dtype=F32)


def test_stale_layout_metadata_raises():
    """A dim claiming a val axis whose extent disagrees with it."""
    bad = SparseTensor((), (DenseIndex(0, 4, axis=0), DenseIndex(1, 3, axis=1)),
                       _rng(12, 33).reshape(3, 4), check_consistency=False)
    with pytest.raises(LazyContractionUnsupported, match="stale layout metadata"):
        bilinear_accumulators(_rng(12, 34).reshape(4, 3), bad, dtype=F32)


def test_an_orphan_val_axis_raises():
    orphan = SparseTensor((), (DenseIndex(0, 4, axis=0),),
                          _rng(12, 35).reshape(4, 3), check_consistency=False)
    with pytest.raises(LazyContractionUnsupported, match="named by no dim"):
        bilinear_accumulators(_rng(4, 36), orphan, dtype=F32)


# --------------------------------------------------------------------------
# 7. squared_norm agrees with dense() on everything above
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# the oracle, and the ONE structure graphax's own densifier refuses
# --------------------------------------------------------------------------
def _ref_dense(st: SparseTensor) -> np.ndarray:
    """A plain-numpy dense form, built from the structural model by index
    arithmetic -- the SECOND oracle.

    ``dense()`` is the primary oracle everywhere it works, and every test below
    checks against it. It does NOT work for a ``val is None`` DIAGONAL PAIR:
    ``ops/dense.py:129`` seeds ``values = jnp.array(1.0)`` (rank 0) and then
    ``_broadcast_and_append_dimensions`` only grows axes for dims whose ``axis``
    is None -- a uniform pair's dims all HAVE axes, so the rank-0 buffer reaches
    the scatter and jax raises ``ValueError: axis 0 is out of bounds for array
    of dimension 0`` (measured 2026-09-12, job 65007). That is a defect in
    ``dense()``, reported separately and deliberately not fixed here. For that
    one structure this reference is the oracle instead, and
    :func:`test_the_reference_densifier_agrees_with_graphax_dense` pins the two
    against each other everywhere both exist so the weaker oracle is not
    trusted blind.
    """
    dims = st.dims
    logical = tuple(int(d.logical_size) for d in dims)
    sm = float(np.asarray(st.scalar_mult))
    fill = 0.0 if st.fill_value is None else float(np.asarray(st.fill_value))
    pairs, seen = [], set()
    for pos, d in enumerate(dims):
        if d.is_sparse:
            key = frozenset((int(d.id), int(d.other_id)))
            if key in seen:
                continue
            seen.add(key)
            partner = next(q for q, x in enumerate(dims)
                           if q != pos and int(x.id) == int(d.other_id))
            pairs.append((pos, partner))
    out = np.full(logical, fill, np.float64)
    val = None if st.val is None else np.asarray(st.val, np.float64)
    for idx in np.ndindex(*logical):
        on = True
        for (po, pi) in pairs:
            Bo = int(dims[po].block_size or 1)
            Bi = int(dims[pi].block_size or 1)
            if idx[po] // Bo != idx[pi] // Bi:
                on = False
                break
        if not on:
            continue
        if val is None:
            out[idx] = 1.0
            continue
        phys = [0] * val.ndim
        for (po, pi) in pairs:
            Bo = int(dims[po].block_size or 1)
            Bi = int(dims[pi].block_size or 1)
            meta_ax = dims[po].axis if dims[po].axis is not None else dims[pi].axis
            if meta_ax is not None:                 # else a broadcast meta
                phys[int(meta_ax)] = idx[po] // Bo
            if dims[po].block_axis is not None:     # else a broadcast block
                phys[int(dims[po].block_axis)] = idx[po] % Bo
            if dims[pi].block_axis is not None:
                phys[int(dims[pi].block_axis)] = idx[pi] % Bi
        for pos, d in enumerate(dims):
            if d.is_sparse or d.axis is None:
                continue
            phys[int(d.axis)] = idx[pos]
        out[idx] = val[tuple(phys)]
    return out * sm


def _oracle(x):
    """The dense form of one operand: ``dense()`` when it works, the numpy
    reference when ``dense()`` raises -- and the reference is cross-checked
    against ``dense()`` by its own test below."""
    if not isinstance(x, SparseTensor):
        return np.asarray(x, np.float64)
    try:
        return np.asarray(x.dense(), np.float64)
    except Exception:
        return _ref_dense(x)


def test_the_reference_densifier_agrees_with_graphax_dense():
    """Everywhere ``dense()`` works, the two oracles must agree -- otherwise the
    weaker one cannot be trusted for the case where ``dense()`` raises. Also
    RECORDS which fixtures ``dense()`` refuses, so the defect stays visible."""
    refused = []
    checked = 0
    fixtures = [
        st_dense(_rng(12, 90).reshape(4, 3), (4, 3), sm=0.5),
        st_dense(_rng(3, 91), (4, 3), implicit=(0,), sm=-2.0),
        st_dense(_rng(6, 92).reshape(3, 2), (4, 3, 2), implicit=(0,)),
        st_pair(4, 1, 1, seed=93),
        st_pair(3, 2, 2, sm=0.5, seed=94),
        st_pair(2, 3, 2, fill=jnp.asarray(1.5, F32), seed=95),
        st_pair(3, 2, 2, val=_UNIFORM, sm=1.25),
        SparseTensor((), (DenseIndex(0, 4, axis=None), DenseIndex(1, 3, axis=None)),
                     None, scalar_mult=jnp.asarray(0.0, F32), check_consistency=False),
        st_one_block_pair(_rng(16, 96).reshape(4, 4)),
        st_pure_diagonal(_rng(4, 97)),
    ]
    for t in fixtures:
        try:
            got = np.asarray(t.dense(), np.float64)
        except Exception as exc:
            refused.append((t.dims, f"{type(exc).__name__}: {exc}"))
            continue
        want = _ref_dense(t)
        assert got.shape == want.shape, (t.dims, got.shape, want.shape)
        assert np.allclose(got, want, atol=1e-5), (t.dims, got, want)
        checked += 1
    assert checked >= 6, "the cross-check covered too little"
    # Every refusal must be one graphax itself routes through dense(hard=True):
    # a pair whose meta or block axis is absent, or a uniform (val=None) pair.
    # Anything else is a NEW refusal and this test is where it surfaces.
    for dims, err in refused:
        routed = any(
            d.is_sparse and (d.axis is None or d.block_axis is None) for d in dims)
        assert routed, (
            f"dense() refused a structure for a NEW reason, so the second "
            f"oracle's coverage is no longer understood: {dims} -> {err}")
        print("dense() refuses (known, routed via dense(hard=True)): %s -> %s"
              % (dims, err))


@pytest.mark.parametrize("maker", [
    lambda: st_pair(3, 2, 2, seed=37),
    lambda: st_pair(3, 2, 2, val=_UNIFORM, sm=0.5),
    lambda: st_pair(3, 2, 2, fill=jnp.asarray(-1.5, F32), sm=2.0, seed=38),
    lambda: st_dense(_rng(6, 39), (6, 6), implicit=(0,), sm=-1.0),
    lambda: st_dense(_rng(36, 40).reshape(6, 6), (6, 6), sm=3.0),
])
def test_squared_norm_equals_the_dense_one(maker):
    t = maker()
    got = float(squared_norm(t, F32))
    want = float((_oracle(t) ** 2).sum())
    assert got == pytest.approx(want, rel=2e-5, abs=1e-5)
