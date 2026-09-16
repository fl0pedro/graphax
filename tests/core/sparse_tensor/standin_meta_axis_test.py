"""A SIZE-1 PHYSICAL AXIS IS A BROADCAST STAND-IN (ticket dsnn-lvm).

A physical axis of a ``SparseTensor`` may carry extent 1 while its dim declares
``size`` (or ``block_size``) above 1. It then says exactly what an IMPLICIT
axis says: one copy is stored and every position along the axis reads it. The
form is legal and stored, not a defect:

* ``_assert_sparse_tensor_consistency`` admits it in so many words -- "a size-1
  physical axis is allowed as a broadcast/implicit stand-in" -- so it survives
  every construction check;
* ``dense()`` implements it, in ``_calculate_target_shape``, by growing such an
  axis to ``dim.size`` before it materialises anything.

So a reader broadcasts; the builder is not asked to pre-expand. The two coupled
block-diagonal re-factorings did not: they read the STORED extent of a physical
axis instead of the DECLARED one, and reshaped the one-element buffer into the
full block grid.

The failure this reproduces came out of the thesis smoke of 2026-09-16 (job
66054, arm C on TransformerLM, free spatial order, all approximation classes):

    graphax/sparse/ops/matmul.py matmul -> _reframe_misaligned_contraction
      -> _coarsen_pair_to -> tensor.py _coarsen_coupled_blockdiag
    TypeError: cannot reshape array of shape (1,) (size 1) into shape [128, 1, 1]

A constant diagonal of logical length 128 stored as a single value on a
PHYSICAL meta axis. ``_coarsen_coupled_blockdiag`` took the axis for a
full-length one because it was not ``None``.
"""
import itertools
import unittest

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.tensor import (
    SparseTensor, _coarsen_coupled_blockdiag, _subdivide_coupled_blockdiag)
from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.ops.matmul import matmul


def _n(shape, key=0):
    return jr.normal(jr.PRNGKey(key), shape).astype(jnp.float32)


def _coupled(meta, b1, b2, *, meta_ext, b1_ext, b2_ext, key=0, trailing=()):
    """A coupled block-diagonal pair, each side PHYSICAL, with the STORED
    extents given separately from the declared ones.

    ``meta_ext=1`` with ``meta`` above 1 is the stand-in form. ``None`` for an
    extent makes that side IMPLICIT (no physical axis at all), which is the
    form the code already handled.
    """
    shape, ax = [], {}
    for name, ext in (("m", meta_ext), ("b1", b1_ext), ("b2", b2_ext)):
        if ext is None:
            ax[name] = None
        else:
            ax[name] = len(shape)
            shape.append(ext)
    rest_ids = []
    for t in trailing:
        rest_ids.append((len(shape), t))
        shape.append(t)
    val = _n(tuple(shape), key) if shape else _n((), key)
    od = [DiagonalIndex(0, meta, ax["m"], 1,
                        b1 if b1 > 1 else None, ax["b1"] if b1 > 1 else None)]
    pd = [DiagonalIndex(1, meta, ax["m"], 0,
                        b2 if b2 > 1 else None, ax["b2"] if b2 > 1 else None)]
    for i, (a, t) in enumerate(rest_ids):
        (od if i % 2 == 0 else pd).append(DenseIndex(2 + i, t, a))
    return SparseTensor(tuple(od), tuple(pd), val)


def _coarsen(t, g):
    return _coarsen_coupled_blockdiag(
        t, True, 0, t.out_dims[0], False, 0, t.primal_dims[0], g)


def _subdivide(t, f):
    return _subdivide_coupled_blockdiag(
        t, True, 0, t.out_dims[0], False, 0, t.primal_dims[0], f)


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


class TestTicketReproduction(unittest.TestCase):
    """The exact shape job 66054 died on."""

    def _constant_diagonal_128(self):
        # A 128 x 128 diagonal whose 128 diagonal entries are all the same
        # number, stored ONCE on a PHYSICAL meta axis of extent 1.
        od = (DiagonalIndex(0, 128, 0, 1, None, None),)
        pd = (DiagonalIndex(1, 128, 0, 0, None, None),)
        return SparseTensor(od, pd, _n((1,), 3))

    def test_the_stand_in_form_is_accepted_at_construction(self):
        # It has to be, or the fix belongs at the builder instead. The
        # consistency check runs by default here (no check_consistency=False).
        t = self._constant_diagonal_128()
        self.assertEqual(t.val.shape, (1,))
        self.assertEqual(t.out_dims[0].size, 128)
        self.assertEqual(t.out_dims[0].axis, 0)

    def test_dense_reads_the_stand_in_as_a_broadcast(self):
        t = self._constant_diagonal_128()
        d = np.asarray(t.dense())
        self.assertEqual(d.shape, (128, 128))
        v = float(np.asarray(t.val)[0]) * float(np.asarray(t.scalar_mult))
        self.assertTrue(np.array_equal(np.diag(d), np.full(128, v)))
        self.assertEqual(float(np.abs(d - np.eye(128, dtype=d.dtype) * v).max()),
                         0.0)

    def test_coarsen_128_to_64_keeps_the_dense_form(self):
        t = self._constant_diagonal_128()
        before = np.asarray(t.dense())
        after = np.asarray(_coarsen(t, 64).dense())
        self.assertEqual(float(np.abs(before - after).max()), 0.0)

    def test_coarsen_every_divisor_keeps_the_dense_form(self):
        t = self._constant_diagonal_128()
        before = np.asarray(t.dense())
        for g in _divisors(128):
            with self.subTest(g=g):
                after = np.asarray(_coarsen(t, g).dense())
                self.assertEqual(float(np.abs(before - after).max()), 0.0)

    def test_matmul_through_the_reframe_path_equals_the_dense_einsum(self):
        """The reported call chain, end to end.

        ``lhs`` is the constant diagonal at meta 128. ``rhs`` cuts the same
        logical extent 128 at meta 64 with blocks of 2. The metas disagree, the
        gcd is 64 and ``a * b = 8192`` is above the logical extent, so
        ``_reframe_misaligned_contraction`` coarsens ``lhs`` from 128 to 64 --
        which is where job 66054 raised.
        """
        lhs = self._constant_diagonal_128()
        rhs = SparseTensor(
            (DiagonalIndex(0, 64, 0, 1, 2, 1),),
            (DiagonalIndex(1, 64, 0, 0, 2, 2),),
            _n((64, 2, 2), 4))
        got = np.asarray(matmul(lhs, rhs).dense())
        want = np.einsum("ij,jk->ik",
                         np.asarray(lhs.dense()), np.asarray(rhs.dense()))
        self.assertEqual(got.shape, want.shape)
        self.assertLessEqual(float(np.abs(got - want).max()), 1e-5)


class TestStandInCoarsenExactness(unittest.TestCase):
    """Every mix of implicit / stand-in / full-length on the three coupled
    axes, over a set of coupled block-diagonal shapes. Coarsening is lossless,
    so the dense form must not move by one bit."""

    SHAPES = [
        (128, 1, 1),
        (16, 1, 1),
        (8, 4, 4),
        (8, 4, 1),
        (8, 1, 4),
        (12, 3, 2),
        (6, 2, 5),
    ]

    def test_coarsen_matches_dense_on_every_storage_mix(self):
        seen_standin = 0
        for k, (meta, b1, b2) in enumerate(self.SHAPES):
            # None = implicit, 1 = stand-in, full = ordinary physical axis.
            m_opts = [None, 1, meta] if meta > 1 else [None, meta]
            b1_opts = ([None, 1, b1] if b1 > 1 else [None])
            b2_opts = ([None, 1, b2] if b2 > 1 else [None])
            for me, e1, e2 in itertools.product(m_opts, b1_opts, b2_opts):
                if me is None and e1 is None and e2 is None:
                    # Nothing physical at all: that is the ``val is None``
                    # uniform form, covered by ``coarsen_blockdiag_test``.
                    continue
                if me == 1 and meta > 1:
                    seen_standin += 1
                if e1 == 1 and b1 > 1:
                    seen_standin += 1
                if e2 == 1 and b2 > 1:
                    seen_standin += 1
                t = _coupled(meta, b1, b2, meta_ext=me, b1_ext=e1, b2_ext=e2,
                             key=k + 1)
                before = np.asarray(t.dense())
                for g in _divisors(meta):
                    with self.subTest(shape=(meta, b1, b2),
                                      stored=(me, e1, e2), g=g):
                        after = np.asarray(_coarsen(t, g).dense())
                        self.assertEqual(
                            float(np.abs(before - after).max()), 0.0)
        # Guard the guard: the loop above must actually build stand-ins.
        self.assertGreater(seen_standin, 0)

    def test_coarsen_with_trailing_dense_axes(self):
        # The stand-in sits in front of ordinary dense axes, so the ``rest``
        # bookkeeping is exercised too.
        t = _coupled(16, 2, 2, meta_ext=1, b1_ext=2, b2_ext=2, key=9,
                     trailing=(3, 4))
        before = np.asarray(t.dense())
        for g in _divisors(16):
            with self.subTest(g=g):
                after = np.asarray(_coarsen(t, g).dense())
                self.assertEqual(float(np.abs(before - after).max()), 0.0)

    def test_subdivide_matches_dense_on_a_stand_in_meta(self):
        # The dual re-factoring makes the same assumption and needs the same
        # reading of the invariant.
        for meta, b1, b2 in [(4, 4, 4), (2, 8, 8), (4, 2, 6)]:
            t = _coupled(meta, b1, b2, meta_ext=1, b1_ext=b1, b2_ext=b2, key=11)
            before = np.asarray(t.dense())
            for k in (2, 4):
                if b1 % k or b2 % k:
                    continue
                with self.subTest(shape=(meta, b1, b2), k=k):
                    after = np.asarray(_subdivide(t, meta * k).dense())
                    self.assertEqual(
                        float(np.abs(before - after).max()), 0.0)

    def test_subdivide_matches_dense_on_a_stand_in_block_side(self):
        for meta, b1, b2 in [(4, 4, 4), (3, 6, 2)]:
            t = _coupled(meta, b1, b2, meta_ext=meta, b1_ext=1, b2_ext=b2,
                         key=12)
            before = np.asarray(t.dense())
            for k in (2,):
                if b1 % k or b2 % k:
                    continue
                with self.subTest(shape=(meta, b1, b2), k=k):
                    after = np.asarray(_subdivide(t, meta * k).dense())
                    self.assertEqual(
                        float(np.abs(before - after).max()), 0.0)


class TestStandInMatmulExactness(unittest.TestCase):
    """The contraction itself, over misaligned meta grids with stand-ins on
    either side, judged two ways.

    Against ``jnp.einsum`` on the densified operands, to a tolerance: both
    routes sum the same products, but the sparse route sums only the live ones
    and in its own order, and float addition is not associative. The repo's own
    misaligned-block tests use ``atol=1e-5`` for the same reason. MEASURED
    here: the largest disagreement over the whole case set is about 5e-7 on
    values of order 10, which is fp32 rounding.

    Against the SAME logical operands stored WITHOUT the stand-in, bit for
    bit. This is the exactness claim that belongs to this ticket: a size-1
    physical axis says one copy is stored and every position reads it, so
    reading it must give exactly what writing the copies out gives -- not
    nearly.
    """

    # (logical L, lhs meta, rhs meta). Both metas divide L, the gcd is above 1
    # and lhs meta * rhs meta > L, which is what makes ``matmul`` take the
    # reframe-to-gcd route rather than the lcm one.
    CASES = [
        (128, 128, 64),
        (128, 64, 32),
        (64, 64, 32),
        (64, 32, 16),
        (36, 36, 12),
        (48, 24, 16),
    ]

    def _side(self, L, meta, stand_in, key):
        b = L // meta
        ext_m = 1 if stand_in else meta
        if b > 1:
            val = _n((ext_m, b, b), key)
            od = (DiagonalIndex(0, meta, 0, 1, b, 1),)
            pd = (DiagonalIndex(1, meta, 0, 0, b, 2),)
        else:
            val = _n((ext_m,), key)
            od = (DiagonalIndex(0, meta, 0, 1, None, None),)
            pd = (DiagonalIndex(1, meta, 0, 0, None, None),)
        return SparseTensor(od, pd, val)

    def _pair(self, L, meta, key):
        """One logical operand stored two ways: with a size-1 stand-in meta
        axis, and with that axis written out. Same numbers either way."""
        b = L // meta
        if b > 1:
            core = _n((1, b, b), key)
            full = jnp.broadcast_to(core, (meta, b, b))
            od = (DiagonalIndex(0, meta, 0, 1, b, 1),)
            pd = (DiagonalIndex(1, meta, 0, 0, b, 2),)
        else:
            core = _n((1,), key)
            full = jnp.broadcast_to(core, (meta,))
            od = (DiagonalIndex(0, meta, 0, 1, None, None),)
            pd = (DiagonalIndex(1, meta, 0, 0, None, None),)
        return SparseTensor(od, pd, core), SparseTensor(od, pd, full)

    def test_a_stand_in_reads_exactly_as_the_written_out_copies(self):
        for i, (L, a, b) in enumerate(self.CASES):
            l_in, l_out = self._pair(L, a, 60 + i)
            r_in, r_out = self._pair(L, b, 80 + i)
            ref = np.asarray(matmul(l_out, r_out).dense())
            for use_l, use_r in itertools.product((False, True), repeat=2):
                with self.subTest(L=L, lhs_meta=a, rhs_meta=b,
                                  lhs_standin=use_l, rhs_standin=use_r):
                    got = np.asarray(matmul(l_in if use_l else l_out,
                                            r_in if use_r else r_out).dense())
                    self.assertEqual(got.shape, ref.shape)
                    self.assertEqual(float(np.abs(got - ref).max()), 0.0)

    def test_dense_of_a_stand_in_equals_dense_of_the_written_out_copies(self):
        for i, (L, a, _b) in enumerate(self.CASES):
            t_in, t_out = self._pair(L, a, 100 + i)
            with self.subTest(L=L, meta=a):
                got = np.asarray(t_in.dense())
                ref = np.asarray(t_out.dense())
                self.assertEqual(got.shape, ref.shape)
                self.assertEqual(float(np.abs(got - ref).max()), 0.0)

    def test_matmul_equals_the_dense_einsum(self):
        for i, (L, a, b) in enumerate(self.CASES):
            for si_l, si_r in itertools.product((False, True), repeat=2):
                with self.subTest(L=L, lhs_meta=a, rhs_meta=b,
                                  lhs_standin=si_l, rhs_standin=si_r):
                    lhs = self._side(L, a, si_l, 20 + i)
                    rhs = self._side(L, b, si_r, 40 + i)
                    got = np.asarray(matmul(lhs, rhs).dense())
                    want = np.einsum(
                        "ij,jk->ik",
                        np.asarray(lhs.dense()), np.asarray(rhs.dense()))
                    self.assertEqual(got.shape, want.shape)
                    self.assertEqual(float(np.abs(got - want).max()), 0.0)


if __name__ == "__main__":
    unittest.main()
