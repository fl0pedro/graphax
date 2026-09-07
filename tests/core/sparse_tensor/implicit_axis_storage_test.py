"""The storage contract of every implicit-axis case (ticket dsnn-3qm.28.5).

One hand-built contraction per case of finding 63 deliverable (f). For each the
test pins three things that no other test in this directory pins:

* the STORED element count of the result, hand-written from the algebra and
  never read off the result itself,
* which output dims stayed implicit (the axis-is-None pattern),
* the number of growing ``broadcast_in_dim`` equations in the jaxpr, which is
  zero where the algebra says the contraction needs no broadcast at all.

The growing-broadcast count is asserted EXACTLY, including where it is not
zero today. A change that removes one of those broadcasts (ticket dsnn-3qm.28.2
does exactly that for the single implicit sparse axis and for the
spatial-sparse pairing) must therefore edit this file on purpose, not silently.

Everything here runs under the DEFAULT engine. The per-mode sweep lives in
``.scratch-race/probes/t285/`` and is a probe, not a test.
"""
import math
import unittest

import jax
import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.indexes import BandedIndex, DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor

from utils import assert_axis_pattern, stored_elements, support_size

M_META, P_BLK, K_CON, Q_OUT = 4, 3, 2, 5
A_, B_, C_, D_ = 2, 3, 4, 5


def _n(shape, k):
    return jr.normal(jr.PRNGKey(k), shape).astype(jnp.float32)


def growing_broadcasts(lhs, rhs):
    """(count, grown elements) over the jaxpr of ``lhs @ rhs``.

    A ``broadcast_in_dim`` whose output holds more elements than its input is
    a materialisation: the engine wrote a bigger operand than it was given.
    """
    closed = jax.make_jaxpr(lambda a, b: (a @ b).val)(lhs, rhs)
    n, grown = 0, 0

    def walk(jaxpr):
        nonlocal n, grown
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "broadcast_in_dim":
                ish = getattr(getattr(eqn.invars[0], "aval", None), "shape", ())
                osz = int(math.prod(eqn.outvars[0].aval.shape))
                isz = int(math.prod(ish)) if ish is not None else 1
                if osz > isz:
                    n += 1
                    grown += osz - isz
            for v in eqn.params.values():
                jx = getattr(v, "jaxpr", None)
                if jx is not None:
                    walk(jx.jaxpr if hasattr(jx, "jaxpr") else jx)

    walk(closed.jaxpr)
    return n, grown


class TestImplicitAxisStorage(unittest.TestCase):
    def _check(self, lhs, rhs, oracle, *, stored, pattern, growing,
               support=None, banded=None):
        res = lhs @ rhs
        self.assertTrue(
            jnp.allclose(res.dense(), oracle, atol=1e-5),
            f"value mismatch: max diff "
            f"{float(jnp.max(jnp.abs(res.dense() - oracle)))}",
        )
        self.assertEqual(
            stored_elements(res), stored,
            f"stored {stored_elements(res)} != expected {stored}")
        assert_axis_pattern(res, pattern)
        self.assertEqual(
            growing_broadcasts(lhs, rhs), growing,
            "the growing-broadcast count changed; if that was on purpose, "
            "update the expectation here and say so in the commit")
        if support is not None:
            self.assertEqual(support_size(oracle), support,
                             "the hand-written support is wrong")
            self.assertLessEqual(
                stored, support,
                "the emission stores more than the structural support")
        if banded is not None:
            self.assertEqual(
                any(isinstance(d, BandedIndex) for d in res.dims), banded)

    # --- single implicit sparse axis: the meta axis of a diagonal pair ----
    # Optimum: M*P*Q = 60 stored, M*P*K*Q = 120 products. The default engine
    # reaches both and pays a 30-element operand broadcast for the fusion it
    # buys on GPU (finding 63 deliverable c).
    def test_single_implicit_sparse_lhs_stores_meta(self):
        lhs = SparseTensor(
            (DiagonalIndex(0, M_META, 0, 1, P_BLK, 1),),
            (DiagonalIndex(1, M_META, 0, 0, K_CON, 2),),
            _n((M_META, P_BLK, K_CON), 1),
        )
        rhs = SparseTensor(
            (DiagonalIndex(0, M_META, None, 1, K_CON, 0),),
            (DiagonalIndex(1, M_META, None, 0, Q_OUT, 1),),
            _n((K_CON, Q_OUT), 2),
        )
        self._check(lhs, rhs, lhs.dense() @ rhs.dense(),
                    stored=60, pattern="SS", growing=(1, 30), support=60)

    def test_single_implicit_sparse_rhs_stores_meta(self):
        lhs = SparseTensor(
            (DiagonalIndex(0, M_META, None, 1, P_BLK, 0),),
            (DiagonalIndex(1, M_META, None, 0, K_CON, 1),),
            _n((P_BLK, K_CON), 1),
        )
        rhs = SparseTensor(
            (DiagonalIndex(0, M_META, 0, 1, K_CON, 1),),
            (DiagonalIndex(1, M_META, 0, 0, Q_OUT, 2),),
            _n((M_META, K_CON, Q_OUT), 2),
        )
        self._check(lhs, rhs, lhs.dense() @ rhs.dense(),
                    stored=60, pattern="SS", growing=(1, 18), support=60)

    # --- single implicit dense axis, block kind ---------------------------
    # Optimum: M*Q = 20 stored, M*K*Q = 40 products, because the out-side
    # block is a replication. The incumbent executor and the planner both
    # store 60 and spend 120. The lazy frame is the better emission here.
    def test_single_implicit_dense_block(self):
        lhs = SparseTensor(
            (DiagonalIndex(0, M_META, 0, 1, P_BLK, None),),
            (DiagonalIndex(1, M_META, 0, 0, K_CON, 1),),
            _n((M_META, K_CON), 1),
        )
        rhs = SparseTensor(
            (DiagonalIndex(0, M_META, 0, 1, K_CON, 1),),
            (DiagonalIndex(1, M_META, 0, 0, Q_OUT, 2),),
            _n((M_META, K_CON, Q_OUT), 2),
        )
        self._check(lhs, rhs, lhs.dense() @ rhs.dense(),
                    stored=20, pattern="SS", growing=(0, 0), support=60)

    # --- single implicit dense axis, contracted kind ----------------------
    # Optimum: sum the storing side over c, then one product per output.
    # 30 stored, 40 adds plus 30 products, and no dot at all.
    def test_single_implicit_dense_contracted(self):
        lhs = SparseTensor(
            (DenseIndex(0, A_, 0), DenseIndex(1, B_, 1)),
            (DenseIndex(2, C_, None),),
            _n((A_, B_), 1),
        )
        rhs = SparseTensor(
            (DenseIndex(0, A_, 0), DenseIndex(1, C_, 1)),
            (DenseIndex(2, D_, 2),),
            _n((A_, C_, D_), 2),
        )
        self._check(lhs, rhs, jnp.einsum("abc,acd->abd", lhs.dense(), rhs.dense()),
                    stored=30, pattern="DDD", growing=(0, 0))

    # --- single implicit dense axis, carried kind -------------------------
    # The batch axis rides to the output, so nothing is saved in the buffer.
    # 30 stored, 120 products. The default pays a 12-element broadcast.
    # NOTE: finding 63 (f) lists the carried case as broadcast-free under the
    # default. That row covers the BLOCK-slot variant. This is the META-slot
    # variant of a batch_out pairing, which the demote rule governs, and it
    # still grows under the default.
    def test_single_implicit_dense_carried(self):
        lhs = SparseTensor(
            (DenseIndex(0, A_, None), DenseIndex(1, B_, 0)),
            (DenseIndex(2, C_, 1),),
            _n((B_, C_), 1),
        )
        rhs = SparseTensor(
            (DenseIndex(0, A_, 0), DenseIndex(1, C_, 1)),
            (DenseIndex(2, D_, 2),),
            _n((A_, C_, D_), 2),
        )
        self._check(lhs, rhs, jnp.einsum("abc,acd->abd", lhs.dense(), rhs.dense()),
                    stored=30, pattern="DDD", growing=(1, 12))

    # --- double implicit contracted axis ----------------------------------
    # c folds into scalar_mult. 30 stored, 30 products, no dot.
    def test_double_implicit_contracted(self):
        lhs = SparseTensor(
            (DenseIndex(0, A_, 0), DenseIndex(1, B_, 1)),
            (DenseIndex(2, C_, None),),
            _n((A_, B_), 1),
        )
        rhs = SparseTensor(
            (DenseIndex(0, A_, 0), DenseIndex(1, C_, None)),
            (DenseIndex(2, D_, 1),),
            _n((A_, D_), 2),
        )
        self._check(lhs, rhs, jnp.einsum("abc,acd->abd", lhs.dense(), rhs.dense()),
                    stored=30, pattern="DDD", growing=(0, 0))

    # --- double implicit batch axis ---------------------------------------
    # The batch axis stays implicit in the result. 15 stored, 60 products.
    def test_double_implicit_batch(self):
        lhs = SparseTensor(
            (DenseIndex(0, A_, None), DenseIndex(1, B_, 0)),
            (DenseIndex(2, C_, 1),),
            _n((B_, C_), 1),
        )
        rhs = SparseTensor(
            (DenseIndex(0, A_, None), DenseIndex(1, C_, 0)),
            (DenseIndex(2, D_, 1),),
            _n((C_, D_), 2),
        )
        self._check(lhs, rhs, jnp.einsum("abc,acd->abd", lhs.dense(), rhs.dense()),
                    stored=15, pattern="IDD", growing=(0, 0))

    # --- uniform operand: val is None on every dim ------------------------
    # 10 stored, 40 adds. The one residual growing element is the carried
    # batch axis above, not the uniform operand (finding 63 corrects
    # finding 62 on this point).
    def test_uniform_operand(self):
        lhs = SparseTensor(
            (DenseIndex(0, A_, None), DenseIndex(1, B_, None)),
            (DenseIndex(2, C_, None),),
            None,
        )
        rhs = SparseTensor(
            (DenseIndex(0, A_, 0), DenseIndex(1, C_, 1)),
            (DenseIndex(2, D_, 2),),
            _n((A_, C_, D_), 2),
        )
        self._check(lhs, rhs, jnp.einsum("abc,acd->abd", lhs.dense(), rhs.dense()),
                    stored=10, pattern="DID", growing=(1, 1))

    # --- genuine LCM grid --------------------------------------------------
    # meta 4 against meta 6, gcd 2, lcm 12. The band form stores 280, which is
    # exactly the structural support of the dense product, so it is provably
    # the smallest honest buffer. The planner stores 840.
    def test_lcm_grid(self):
        a, b, c, d, e, f = 4, 6, 2, 5, 3, 7
        lhs = SparseTensor(
            (DiagonalIndex(0, a, 0, 1, d, 1),),
            (DiagonalIndex(1, a, 0, 0, e, 2),),
            _n((a, d, e), 1),
        )
        rhs = SparseTensor(
            (DiagonalIndex(0, b, 0, 1, c, 1),),
            (DiagonalIndex(1, b, 0, 0, f, 2),),
            _n((b, c, f), 2),
        )
        self._check(lhs, rhs, lhs.dense() @ rhs.dense(),
                    stored=280, pattern="SS", growing=(3, 2087),
                    support=280, banded=True)

    # --- spatial_sparse pairing -------------------------------------------
    # An lhs-only diagonal pair riding through an uncontracted primal dim.
    # Optimum 48 stored, 240 products, which the engine reaches. It also
    # writes 40 extra operand elements, a 3.00 blow-up of a 20-element
    # operand, because "spatial_sparse_lhs" is not in _LAZY_PAIRINGS so no
    # lazy rule fires. Its frame numbers are m_l/m_r = 3/1 with T = 3, which
    # is the shape the demote rule already handles for a contract pairing.
    def test_spatial_sparse_pairing(self):
        s_, b, c = 3, 4, 5
        lhs = SparseTensor(
            (DenseIndex(0, b, 0), DiagonalIndex(1, s_, 1, 2, None, None)),
            (DiagonalIndex(2, s_, 1, 1, None, None), DenseIndex(3, c, 2)),
            _n((b, s_, c), 1),
        )
        rhs = SparseTensor(
            (DenseIndex(0, c, 0),),
            (DenseIndex(1, b, 1),),
            _n((c, b), 2),
        )
        self._check(lhs, rhs,
                    jnp.einsum("abcd,de->abce", lhs.dense(), rhs.dense()),
                    stored=48, pattern="DSSD", growing=(1, 40), support=48)

    # --- no implicit axis (the control) ------------------------------------
    # Both sides store the merged meta extent. Nothing to shrink and nothing
    # to broadcast, in every mode.
    def test_no_implicit_axis_control(self):
        lhs = SparseTensor(
            (DiagonalIndex(0, M_META, 0, 1, P_BLK, 1),),
            (DiagonalIndex(1, M_META, 0, 0, K_CON, 2),),
            _n((M_META, P_BLK, K_CON), 1),
        )
        rhs = SparseTensor(
            (DiagonalIndex(0, M_META, 0, 1, K_CON, 1),),
            (DiagonalIndex(1, M_META, 0, 0, Q_OUT, 2),),
            _n((M_META, K_CON, Q_OUT), 2),
        )
        self._check(lhs, rhs, lhs.dense() @ rhs.dense(),
                    stored=60, pattern="SS", growing=(0, 0), support=60)


if __name__ == "__main__":
    unittest.main()
