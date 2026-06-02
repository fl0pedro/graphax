"""Phase 8.B — unit tests for the new compressed `Index` types
(`BandedIndex`, `SetIndex`) and their `densify_axis` kernels, including the
code-review carry-forward regressions (CR-1 n_meta fallback, CR-3 gather-free
centered band, CR-4 static-zero-fill skip).

These exercise the Index `densify_axis` methods directly against the legacy
`BlockBanded` / `DivisorRemainder` primitives (which still exist this commit)
to lock in equivalence before the producers (matmul/elementwise emission) are
migrated in 8.E/8.F.
"""
import unittest

import jax
import jax.numpy as jnp
import jax.random as jr

import graphax.sparse.ops.block_storage as bs
from graphax.sparse.ops.block_storage import BlockBanded, DivisorRemainder
from graphax.sparse.indexes import (
    Index,
    DenseIndex,
    DiagonalIndex,
    BandedIndex,
    SetIndex,
)


def _n(shape, key=0):
    return jr.normal(jr.PRNGKey(key), shape).astype(jnp.float32)


class TestBandedIndexDensify(unittest.TestCase):
    """`BandedIndex.densify_axis` must match `BlockBanded.to_dense` on the
    canonical band-val layout `(n_meta*M_primary, W, B_row, B_col, *L)`."""

    def _banded(self, M, W, B_row, B_col, n_secondary, offset, primary, n_meta):
        size = (n_meta * (M if primary else n_secondary)) * B_row
        return BandedIndex(
            id=0, size=size, axis=0, other_id=1,
            block_size=B_row, block_axis=1,
            band_width=W, offset=offset, primary=primary,
            n_secondary=n_secondary, n_meta=n_meta,
        )

    def test_centered_square_matches_blockbanded(self):
        M, W, B = 5, 3, 4
        data = _n((M, W, B, B), 0)
        ref = BlockBanded(data=data, fill_value=jnp.array(0.0)).to_dense()
        bx = self._banded(M, W, B, B, M, (), True, 1)
        self.assertTrue(jnp.allclose(bx.densify_axis(data, jnp.array(0.0)), ref, atol=1e-5))

    def test_rectangular_staircase_matches_blockbanded(self):
        M_r, W, Br, Bc = 11, 2, 5, 5
        off = (0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4)
        data = _n((M_r, W, Br, Bc), 1)
        ref = BlockBanded(
            data=data, fill_value=jnp.array(0.0), n_secondary=5, offset=off
        ).to_dense()
        bx = self._banded(M_r, W, Br, Bc, 5, off, True, 1)
        self.assertTrue(jnp.allclose(bx.densify_axis(data, jnp.array(0.0)), ref, atol=1e-5))

    def test_col_primary_matches_blockbanded(self):
        M, W, B = 5, 3, 4
        data = _n((M, W, B, B), 2)
        ref = BlockBanded(data=data, fill_value=jnp.array(0.0), primary_axis=1).to_dense()
        bx = self._banded(M, W, B, B, M, (), False, 1)
        self.assertTrue(jnp.allclose(bx.densify_axis(data, jnp.array(0.0)), ref, atol=1e-5))

    def test_nonzero_fill_masks_out_of_band(self):
        M, W, B = 4, 3, 2
        data = _n((M, W, B, B), 3)
        ref = BlockBanded(data=data, fill_value=jnp.array(-9.0)).to_dense()
        bx = self._banded(M, W, B, B, M, (), True, 1)
        self.assertTrue(jnp.allclose(bx.densify_axis(data, jnp.array(-9.0)), ref, atol=1e-5))

    def test_jit_roundtrip(self):
        M, W, B = 5, 3, 4
        data = _n((M, W, B, B), 4)
        bx = self._banded(M, W, B, B, M, (), True, 1)
        ref = bx.densify_axis(data, jnp.array(0.0))
        got = jax.jit(lambda d: bx.densify_axis(d, jnp.array(0.0)))(data)
        self.assertTrue(jnp.allclose(got, ref, atol=1e-5))


class TestBandedIndexNMetaRegression(unittest.TestCase):
    """CR-1: the large-band streaming fallback must NOT drop n_meta batches."""

    def test_n_meta_fast_path(self):
        M_per, W, B = 3, 3, 2
        data = _n((2 * M_per, W, B, B), 5)
        ref = BlockBanded(data=data, fill_value=jnp.array(0.0), n_meta=2).to_dense()
        bx = BandedIndex(
            id=0, size=2 * M_per * B, axis=0, other_id=1, block_size=B, block_axis=1,
            band_width=W, offset=(), primary=True, n_secondary=M_per, n_meta=2,
        )
        got = bx.densify_axis(data, jnp.array(0.0))
        self.assertEqual(got.shape, (2 * M_per * B, 2 * M_per * B))
        self.assertTrue(jnp.allclose(got, ref, atol=1e-5))

    def test_n_meta_streaming_fallback_keeps_all_batches(self):
        # Force the broadcast limit tiny → streaming fallback path. With the
        # CR-1 bug this returned only batch 0 (half the shape). After the fix
        # it must match the full fast-path output.
        M_per, W, B = 3, 2, 2
        data = _n((2 * M_per, W, B, B), 6)
        bx = BandedIndex(
            id=0, size=2 * M_per * B, axis=0, other_id=1, block_size=B, block_axis=1,
            band_width=W, offset=(), primary=True, n_secondary=M_per, n_meta=2,
        )
        fast = bx.densify_axis(data, jnp.array(0.0))
        orig = bs._BLOCK_BANDED_BROADCAST_LIMIT
        try:
            bs._BLOCK_BANDED_BROADCAST_LIMIT = 1
            streamed = bx.densify_axis(data, jnp.array(0.0))
        finally:
            bs._BLOCK_BANDED_BROADCAST_LIMIT = orig
        self.assertEqual(streamed.shape, fast.shape)  # (6*B, 6*B) — no dropped batch
        self.assertTrue(jnp.allclose(streamed, fast, atol=1e-5))


class TestBandedIndexGatherFree(unittest.TestCase):
    """CR-3: a centered band densifies gather-free; only explicit offsets gather."""

    def _bx(self, M, W, B, offset):
        return BandedIndex(
            id=0, size=M * B, axis=0, other_id=1, block_size=B, block_axis=1,
            band_width=W, offset=offset, primary=True, n_secondary=M, n_meta=1,
        )

    def test_centered_band_no_gather_no_scatter(self):
        M, W, B = 6, 3, 4
        data = _n((M, W, B, B), 7)
        bx = self._bx(M, W, B, ())  # centered sentinel
        text = jax.jit(lambda d: bx.densify_axis(d, jnp.array(0.0))).lower(data).compile().as_text().lower()
        self.assertEqual(text.count("gather("), 0)
        self.assertEqual(text.count("scatter("), 0)


class TestSetIndexDensify(unittest.TestCase):
    """`SetIndex.densify_axis` (combined 1-D val + lhs/rhs shapes) must match
    `DivisorRemainder.to_dense`."""

    def _bufs(self):
        lhs = _n((1, 3, 2, 2), 8)
        rhs = _n((1, 2, 3, 3), 9)
        return lhs, rhs

    def _set_and_val(self, lhs, rhs, semantic):
        combined = jnp.concatenate([lhs.reshape(-1), rhs.reshape(-1)])
        sx = SetIndex(
            id=0, size=1 * lhs.shape[1] * lhs.shape[2], axis=0, other_id=1,
            block_size=lhs.shape[1] * lhs.shape[2], block_axis=1,
            semantic=semantic, lhs_shape=lhs.shape, rhs_shape=rhs.shape,
            include_remainder=True, n_meta=1,
        )
        return sx, combined

    def test_union_matches_divisor_remainder(self):
        lhs, rhs = self._bufs()
        fl, fr = jnp.array(0.0), jnp.array(0.0)
        ref = DivisorRemainder(
            divisor=lhs, remainder=rhs, fill_divisor=fl, fill_remainder=fr,
            semantic="union", include_remainder=True, op=jnp.add,
        ).to_dense()
        sx, val = self._set_and_val(lhs, rhs, "union")
        self.assertTrue(jnp.allclose(sx.densify_axis(val, (fl, fr)), ref, atol=1e-5))

    def test_intersection_matches_divisor_remainder(self):
        lhs, rhs = self._bufs()
        fl, fr = jnp.array(0.0), jnp.array(0.0)
        ref = DivisorRemainder(
            divisor=lhs, remainder=rhs, fill_divisor=fl, fill_remainder=fr,
            semantic="intersection", include_remainder=True, op=jnp.multiply,
        ).to_dense()
        sx, val = self._set_and_val(lhs, rhs, "intersection")
        self.assertTrue(jnp.allclose(sx.densify_axis(val, (fl, fr)), ref, atol=1e-5))


class TestCompressedIndexInterface(unittest.TestCase):
    """The new types must satisfy the duck-typed Index interface (110+ sites
    read `.is_sparse` etc.) and stay hashable (dims live in static aux_data)."""

    def test_is_compressed_flag(self):
        bx = BandedIndex(id=0, size=8, axis=0, other_id=1, block_size=2, block_axis=1,
                         band_width=3, offset=(), n_secondary=4, n_meta=1)
        sx = SetIndex(id=0, size=6, axis=0, other_id=1, block_size=6, block_axis=1)
        self.assertTrue(bx.is_compressed and sx.is_compressed)
        self.assertFalse(DenseIndex(0, 5, 0).is_compressed)
        self.assertFalse(DiagonalIndex(0, 5, 0, 1, 2, 1).is_compressed)

    def test_is_instance_index_and_sparse(self):
        bx = BandedIndex(id=0, size=8, axis=0, other_id=1, block_size=2, block_axis=1,
                         band_width=3, offset=(), n_secondary=4, n_meta=1)
        self.assertIsInstance(bx, Index)
        self.assertTrue(bx.is_sparse)  # banded pair has other_id

    def test_hashable(self):
        bx = BandedIndex(id=0, size=8, axis=0, other_id=1, block_size=2, block_axis=1,
                         band_width=3, offset=(0, 0, 1, 1), n_secondary=4, n_meta=1)
        sx = SetIndex(id=0, size=6, axis=0, other_id=1, block_size=6, block_axis=1)
        self.assertIsInstance(hash(bx), int)
        self.assertIsInstance(hash(sx), int)


if __name__ == "__main__":
    unittest.main()
