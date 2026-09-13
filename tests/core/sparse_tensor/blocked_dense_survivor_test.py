"""The survivor of a COMPRESS'd Diag block keeps its own storage description.

Ticket dsnn-3qm.62, owner ruling 2026-09-12. Measured operands (job 65090,
``VmappedNeuralNetwork`` on mnist, seed 0): a face applied DIAG(factor 2) to the
pair ``(10 | 784)`` -- blocks ``(5 | 392)`` -- and then COMPRESS'd the val axis
of the PRIMAL block, so that block is implicit (``block_axis=None``) while the
dims still say the dim spans ``2 * 392 == 784`` positions. Contracting that
Jacobian kills the pair on the out side and leaves the primal side alone, and
the engine used to describe the survivor as ``DenseIndex(size=784)`` on a val
axis holding 2 -- metadata that contradicts its own buffer, which
``SparseTensor.__init__`` rejects ("declares size=784 but val.shape[1]=2").

The survivor is a BLOCKED DENSE dim instead: ``size=2`` (the stored block
count), ``block_size=392`` (implicit), ``block_axis=None``, ``other_id=None``.
``logical_size`` is still 784, and NOTHING is materialized -- the block is
expanded by ``dense()`` like any other implicit extent, which is why the
growing-broadcast count of the contraction is asserted here too.

Three cases, because the form has three roles: produced on the PRIMAL side
(the measured one), produced on the OUT side (the mirror), and consumed as an
operand of the NEXT contraction.
"""
import unittest

import pytest

import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.indexes import DenseIndex, DiagonalIndex, Index
from graphax.sparse.tensor import SparseTensor

from implicit_axis_storage_test import growing_broadcasts

B, N_IN, META, BLK_OUT, BLK_PRI = 16, 10, 2, 5, 392
LOGICAL = META * BLK_PRI          # 784


def _n(shape, k):
    return jr.normal(jr.PRNGKey(k), shape).astype(jnp.float32)


def _compressed_jacobian(key=1):
    """The measured rhs: shape ``(16, 10 | 16, 784)``, val ``(16, 2, 5)``.

    Two diagonal pairs. ``(0, 2)`` is a plain 16-wide diagonal on val axis 0.
    ``(1, 3)`` is the DIAG(2): the out side keeps its block of 5 on val axis 2,
    the primal side's block of 392 is IMPLICIT -- that is the COMPRESS.
    """
    return SparseTensor(
        (
            DiagonalIndex(0, B, axis=0, other_id=2),
            DiagonalIndex(1, META, axis=1, other_id=3,
                          block_size=BLK_OUT, block_axis=2),
        ),
        (
            DiagonalIndex(2, B, axis=0, other_id=0),
            DiagonalIndex(3, META, axis=1, other_id=1,
                          block_size=BLK_PRI, block_axis=None),
        ),
        _n((B, META, BLK_OUT), key),
    )


class TestBlockedDenseSurvivor(unittest.TestCase):
    def assertBlocked(self, d, *, size, block, axis):
        self.assertIsNone(d.other_id, f"{d} must not be sparse")
        self.assertIsNone(d.block_axis, f"{d} block must stay implicit")
        self.assertEqual((int(d.size), int(d.block_size), d.axis),
                         (size, block, axis), f"unexpected survivor {d}")
        self.assertEqual(int(d.logical_size), size * block)

    # -- produced on the primal side: the measured case (job 65090) ---------
    def test_primal_side_survivor(self):
        lhs = SparseTensor(
            (),
            (DenseIndex(0, B, axis=0), DenseIndex(1, N_IN, axis=1)),
            _n((B, N_IN), 0),
        )
        rhs = _compressed_jacobian()
        self.assertEqual(tuple(rhs.shape), (B, N_IN, B, LOGICAL))

        res = lhs @ rhs

        self.assertEqual(tuple(res.shape), (B, LOGICAL))
        self.assertEqual(res.out_dims, ())
        self.assertEqual(len(res.primal_dims), 2)
        self.assertIsNotNone(res.val)
        self.assertEqual(tuple(res.val.shape), (B, META))
        self.assertEqual(res.primal_dims[0], DenseIndex(0, B, axis=0))
        self.assertBlocked(res.primal_dims[1], size=META, block=BLK_PRI, axis=1)

        oracle = jnp.tensordot(lhs.dense(), rhs.dense(), axes=([0, 1], [0, 1]))
        self.assertEqual(tuple(oracle.shape), (B, LOGICAL))
        self.assertTrue(
            jnp.allclose(res.dense(), oracle, atol=1e-5),
            f"value mismatch: max diff "
            f"{float(jnp.max(jnp.abs(res.dense() - oracle)))}",
        )
        # The whole point of the blocked form: the 392 is never written out.
        self.assertEqual(
            growing_broadcasts(lhs, rhs), (0, 0),
            "the contraction materialized the implicit block; if that was on "
            "purpose, update the expectation here and say so in the commit")

    # -- produced on the out side: the mirror ------------------------------
    def test_out_side_survivor(self):
        lhs = SparseTensor(
            (DiagonalIndex(0, META, axis=0, other_id=1,
                           block_size=BLK_PRI, block_axis=None),),
            (DiagonalIndex(1, META, axis=0, other_id=0,
                           block_size=BLK_OUT, block_axis=1),),
            _n((META, BLK_OUT), 2),
        )
        rhs = SparseTensor(
            (DenseIndex(0, N_IN, axis=0),),
            (DenseIndex(1, B, axis=1),),
            _n((N_IN, B), 3),
        )
        self.assertEqual(tuple(lhs.shape), (LOGICAL, N_IN))

        res = lhs @ rhs

        self.assertEqual(tuple(res.shape), (LOGICAL, B))
        self.assertEqual(len(res.out_dims), 1)
        self.assertBlocked(res.out_dims[0], size=META, block=BLK_PRI, axis=0)
        self.assertEqual(tuple(res.val.shape), (META, B))

        oracle = jnp.tensordot(lhs.dense(), rhs.dense(), axes=([1], [0]))
        self.assertTrue(
            jnp.allclose(res.dense(), oracle, atol=1e-5),
            f"value mismatch: max diff "
            f"{float(jnp.max(jnp.abs(res.dense() - oracle)))}",
        )
        self.assertEqual(growing_broadcasts(lhs, rhs), (0, 0))

    # -- consumed: the blocked dim is an operand of the NEXT face -----------
    def test_blocked_survivor_contracts_again(self):
        lhs = SparseTensor(
            (),
            (DenseIndex(0, B, axis=0), DenseIndex(1, N_IN, axis=1)),
            _n((B, N_IN), 0),
        )
        mid = lhs @ _compressed_jacobian()
        self.assertBlocked(mid.primal_dims[1], size=META, block=BLK_PRI, axis=1)

        K = 3
        nxt = SparseTensor(
            (DenseIndex(0, B, axis=0), DenseIndex(1, LOGICAL, axis=1)),
            (DenseIndex(2, K, axis=2),),
            _n((B, LOGICAL, K), 4),
        )

        res = mid @ nxt

        self.assertEqual(tuple(res.shape), (K,))
        self.assertEqual(res.out_dims, ())
        oracle = jnp.tensordot(mid.dense(), nxt.dense(), axes=([0, 1], [0, 1]))
        self.assertTrue(
            jnp.allclose(res.dense(), oracle, atol=1e-4),
            f"value mismatch: max diff "
            f"{float(jnp.max(jnp.abs(res.dense() - oracle)))}",
        )
        # Summing the 784 against an operand that stores one cell per block must
        # not write the block out either: ``mid``'s 2 cells are each used for a
        # 392-wide group of ``nxt``, which the einsum does by summing ``nxt``.
        self.assertEqual(growing_broadcasts(mid, nxt), (0, 0))

    # -- the forms the readers must refuse rather than mis-read -------------
    def test_explicit_block_axis_without_a_partner_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "block_axis"):
            Index(0, META, 0, None, BLK_PRI, 1)

    def test_blocked_dim_reduces_over_its_implicit_block(self):
        t = SparseTensor(
            (),
            (Index(0, META, 0, None, BLK_PRI, None),),
            jnp.array([1.0, 2.0]),
        )
        self.assertEqual(int(t.shape[0]), LOGICAL)
        self.assertEqual(t._n_fill_cells, 0,
                         "an implicit block is structure, not fill")
        self.assertAlmostEqual(float(t.sum()), 3.0 * BLK_PRI, places=2)
        self.assertAlmostEqual(float(t.max()), 2.0, places=5)

    @pytest.mark.xfail(
        strict=True, raises=ValueError,
        reason="a blocked dense dim as a BATCH dim of the next contraction is "
               "not implemented: matmul raises on batch pairings (race lane B, "
               "2026-09-13). Lane A carried it through; this is its test, kept "
               "so the raise is a recorded gap and not a silent one.")
    def test_blocked_batch_dim_rides_through_both_operands(self):
        """A blocked dense dim both operands carry (the same id on both out
        sides, the same (size, block) factoring) is a ``batch_out`` pair. It
        should ride through: the frame batches over the 2 stored entries and
        the survivor keeps the 3 implicit, so neither side is materialized."""
        N, BLK, K, Q = 2, 3, 5, 7
        lhs = SparseTensor(
            (Index(0, N, 0, None, BLK, None),),
            (DenseIndex(1, K, 1),),
            _n((N, K), 10),
        )
        rhs = SparseTensor(
            (Index(0, N, 0, None, BLK, None), DenseIndex(1, K, 1)),
            (DenseIndex(2, Q, 2),),
            _n((N, K, Q), 11),
        )
        res = lhs @ rhs

        self.assertEqual(res.shape, (N * BLK, Q))
        self.assertEqual(tuple(res.val.shape), (N, Q))
        d = res.out_dims[0]
        self.assertTrue(d.is_blocked_dense)
        self.assertEqual((d.size, d.block_size, d.block_axis, d.other_id),
                         (N, BLK, None, None))

        oracle = jnp.einsum("bk,bkq->bq", lhs.dense(), rhs.dense())
        self.assertTrue(
            jnp.allclose(res.dense(), oracle, atol=1e-5),
            f"max diff {float(jnp.max(jnp.abs(res.dense() - oracle)))}",
        )
        self.assertEqual(growing_broadcasts(lhs, rhs), (0, 0))



if __name__ == "__main__":
    unittest.main()
