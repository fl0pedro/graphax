"""Phase 8.B — unit tests for the new compressed `Index` types
(`BandedIndex`, `SetIndex`) and their `densify_axis` kernels, including the
code-review carry-forward regressions (CR-1 n_meta fallback, CR-3 gather-free
centered band, CR-4 static-zero-fill skip).

These exercise the Index `densify_axis` methods against an independent numpy
reference (the legacy `BlockBanded` / `DivisorRemainder` pytree oracles were
deleted in Phase 8.G; the numpy oracles below are a stronger, self-contained
check that doesn't reduce to "matches the kernel it wraps").
"""
import unittest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import graphax.sparse.ops.block_storage as bs
from graphax.sparse.indexes import (
    Index,
    DenseIndex,
    DiagonalIndex,
    BandedIndex,
    SetIndex,
)


def _n(shape, key=0):
    return jr.normal(jr.PRNGKey(key), shape).astype(jnp.float32)


def _dense_band_ref(data, *, n_secondary=None, offset=(), primary_axis=0,
                    n_meta=1, fill=0.0):
    """Independent numpy oracle: densify a band buffer ``(n_meta*M_p, W,
    B_row, B_col)`` to its dense ``(n_meta*M_p*B_row, n_meta*M_s*B_col)`` form.

    For col-primary (``primary_axis=1``) the (M_p, W) prefix describes columns:
    build the row-primary dense with the sub-block axes swapped, then transpose.
    """
    d = np.asarray(data)
    MW, W, B_row, B_col = d.shape
    M_p = MW // n_meta
    M_s = n_secondary if (n_secondary is not None and n_secondary >= 0) else M_p
    if primary_axis == 1:
        # Col-primary: swap (B_row, B_col), densify row-primary, transpose.
        ref_T = _dense_band_ref(
            d.swapaxes(2, 3), n_secondary=n_secondary, offset=offset,
            primary_axis=0, n_meta=n_meta, fill=fill,
        )
        return jnp.asarray(np.asarray(ref_T).swapaxes(0, 1))
    if offset:
        off = list(offset)
    else:
        w0 = (W - 1) // 2
        off = [i - w0 for i in range(M_p)]
    R = n_meta * M_p * B_row
    C = n_meta * M_s * B_col
    out = np.full((R, C), fill, dtype=d.dtype)
    for g in range(n_meta):
        for i in range(M_p):
            for w in range(W):
                j = off[i] + w
                if 0 <= j < M_s:
                    r0 = (g * M_p + i) * B_row
                    c0 = (g * M_s + j) * B_col
                    out[r0:r0 + B_row, c0:c0 + B_col] = d[g * M_p + i, w]
    return jnp.asarray(out)


def _block_diag_meta_ref(buf, fill):
    """numpy oracle for one set-side: ``(M, n, Bh, Bw)`` → ``(M, n*Bh, n*Bw)``
    per-meta block-diagonal grid (fill off the per-meta diagonal)."""
    b = np.asarray(buf)
    M, n, Bh, Bw = b.shape
    out = np.full((M, n * Bh, n * Bw), fill, dtype=b.dtype)
    for m in range(M):
        for k in range(n):
            out[m, k * Bh:(k + 1) * Bh, k * Bw:(k + 1) * Bw] = b[m, k]
    return out


def _dense_set_ref(lhs, rhs, *, op, fill_lhs=0.0, fill_rhs=0.0):
    """Independent numpy oracle for the set-theoretic densify: place each side's
    per-meta blocks on its diagonal, combine the two meta grids via ``op``, then
    stitch onto the M-meta diagonal of the full dense form (fill outside)."""
    lhs_meta = _block_diag_meta_ref(lhs, fill_lhs)  # (M, LCM_h, LCM_w)
    rhs_meta = _block_diag_meta_ref(rhs, fill_rhs)
    op_np = np.add if op is jnp.add else np.multiply
    meta = op_np(lhs_meta, rhs_meta)               # (M, LCM_h, LCM_w)
    M, H, Wc = meta.shape
    outside = float(op_np(np.asarray(fill_lhs), np.asarray(fill_rhs)))
    out = np.full((M * H, M * Wc), outside, dtype=meta.dtype)
    for m in range(M):
        out[m * H:(m + 1) * H, m * Wc:(m + 1) * Wc] = meta[m]
    return jnp.asarray(out)


class TestBandedIndexDensify(unittest.TestCase):
    """`BandedIndex.densify_axis` must match the independent numpy band oracle
    on the canonical band-val layout `(n_meta*M_primary, W, B_row, B_col, *L)`."""

    def _banded(self, M, W, B_row, B_col, n_secondary, offset, primary, n_meta):
        # ``size`` is the META count (n_meta * M_primary), matching the matmul
        # producer; ``logical_size = size * block_size`` is the dense dim.
        size = n_meta * (M if primary else n_secondary)
        return BandedIndex(
            id=0, size=size, axis=0, other_id=1,
            block_size=B_row, block_axis=1,
            band_width=W, offset=offset, primary=primary,
            n_secondary=n_secondary, n_meta=n_meta,
        )

    def test_centered_square_matches_blockbanded(self):
        M, W, B = 5, 3, 4
        data = _n((M, W, B, B), 0)
        ref = _dense_band_ref(data)
        bx = self._banded(M, W, B, B, M, (), True, 1)
        self.assertTrue(jnp.allclose(bx.densify_axis(data, jnp.array(0.0)), ref, atol=1e-5))

    def test_rectangular_staircase_matches_blockbanded(self):
        M_r, W, Br, Bc = 11, 2, 5, 5
        off = (0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4)
        data = _n((M_r, W, Br, Bc), 1)
        ref = _dense_band_ref(data, n_secondary=5, offset=off)
        bx = self._banded(M_r, W, Br, Bc, 5, off, True, 1)
        self.assertTrue(jnp.allclose(bx.densify_axis(data, jnp.array(0.0)), ref, atol=1e-5))

    def test_col_primary_matches_blockbanded(self):
        M, W, B = 5, 3, 4
        data = _n((M, W, B, B), 2)
        ref = _dense_band_ref(data, primary_axis=1)
        bx = self._banded(M, W, B, B, M, (), False, 1)
        self.assertTrue(jnp.allclose(bx.densify_axis(data, jnp.array(0.0)), ref, atol=1e-5))

    def test_nonzero_fill_masks_out_of_band(self):
        M, W, B = 4, 3, 2
        data = _n((M, W, B, B), 3)
        ref = _dense_band_ref(data, fill=-9.0)
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
        ref = _dense_band_ref(data, n_secondary=M_per, n_meta=2)
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

    def test_large_m_single_batch_fallback_matches_reference(self):
        # Migrated from test_block_storage_jit (BlockBanded large-M fallback):
        # when the (M, M, W, B, B) broadcast intermediate exceeds the limit,
        # the per-band streaming fallback must still match the numpy oracle.
        M, W, B = 5, 3, 4
        data = _n((M, W, B, B), 11)
        bx = BandedIndex(
            id=0, size=M * B, axis=0, other_id=1, block_size=B, block_axis=1,
            band_width=W, offset=(), primary=True, n_secondary=M, n_meta=1,
        )
        ref = _dense_band_ref(data)
        orig = bs._BLOCK_BANDED_BROADCAST_LIMIT
        try:
            bs._BLOCK_BANDED_BROADCAST_LIMIT = 1  # force streaming fallback
            streamed = bx.densify_axis(data, jnp.array(0.0))
        finally:
            bs._BLOCK_BANDED_BROADCAST_LIMIT = orig
        self.assertEqual(streamed.shape, (M * B, M * B))
        self.assertTrue(jnp.allclose(streamed, ref, atol=1e-5))


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

    def test_set_densify_no_gather_no_scatter(self):
        # The set-theoretic densify (block-diag + stitch + op) is pure
        # broadcast/select — no scatter, no gather (migrated from the deleted
        # block_storage_test HLO-fusion checks for UnionBlocks/Intersection).
        lhs = _n((1, 3, 2, 2), 8)
        rhs = _n((1, 2, 3, 3), 9)
        combined = jnp.concatenate([lhs.reshape(-1), rhs.reshape(-1)])
        sx = SetIndex(
            id=0, size=1 * 3 * 2, axis=0, other_id=1, block_size=3 * 2, block_axis=1,
            semantic="union", lhs_shape=lhs.shape, rhs_shape=rhs.shape,
            include_remainder=True, n_meta=1,
        )
        fz = jnp.array(0.0)
        text = jax.jit(
            lambda v: sx.densify_axis(v, (fz, fz))
        ).lower(combined).compile().as_text().lower()
        self.assertEqual(text.count("gather("), 0)
        self.assertEqual(text.count("scatter("), 0)


class TestSetIndexDensify(unittest.TestCase):
    """`SetIndex.densify_axis` (combined 1-D val + lhs/rhs shapes) must match
    the independent numpy set-densify oracle."""

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
        ref = _dense_set_ref(lhs, rhs, op=jnp.add, fill_lhs=fl, fill_rhs=fr)
        sx, val = self._set_and_val(lhs, rhs, "union")
        self.assertTrue(jnp.allclose(sx.densify_axis(val, (fl, fr)), ref, atol=1e-5))

    def test_intersection_matches_divisor_remainder(self):
        lhs, rhs = self._bufs()
        fl, fr = jnp.array(0.0), jnp.array(0.0)
        ref = _dense_set_ref(lhs, rhs, op=jnp.multiply, fill_lhs=fl, fill_rhs=fr)
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
