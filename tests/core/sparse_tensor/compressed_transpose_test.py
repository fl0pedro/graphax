"""Phase 8.H — transpose recognition for compressed Index dims +
gather-free K-axis densify.

Two properties:

  1. ``transpose()`` of a tensor carrying compressed ``BandedIndex`` / ``SetIndex``
     dims must equal the transpose of the dense form, AND keep the tensor
     compressed (the swap is a view on the compressed buffer — leaf-block
     transpose + band-direction flip / set h↔w swap, see
     ``_try_compressed_transpose``). This holds for the 2-D pair and for K≥2
     all-banded tensors under any band-preserving permutation; other
     permutations fall back to a compact densify.

  2. The K-axis ``_densify_multi_banded`` kernel is gather-free for centered
     bands (CR-3 parity with the single-axis path).
"""
import unittest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from graphax.sparse.tensor import SparseTensor
from graphax.sparse.indexes import DiagonalIndex, BandedIndex, SetIndex
from graphax.sparse.ops.block_storage import _densify_multi_banded, BandAxisSpec


def _n(shape, key=0):
    return jr.normal(jr.PRNGKey(key), shape).astype(jnp.float32)


def _banded_tensor(M, W, B, n_meta=1, n_secondary=None, key=7):
    """A 2-D ``SparseTensor`` whose dims are a ``BandedIndex`` pair over a
    centered band buffer ``(n_meta*M, W, B, B)``."""
    n_sec = n_secondary if n_secondary is not None else M
    data = _n((n_meta * M, W, B, B), key)
    # Both sides share the band orientation flag (producer convention: a band is
    # row- or col-primary as a whole), out axis=0/block_axis=1, primal
    # axis=1/block_axis=3 — matching matmul's BandedIndex emission.
    out = (BandedIndex(id=0, size=n_meta * M, axis=0, other_id=1,
                       block_size=B, block_axis=1, band_width=W, offset=(),
                       primary=True, n_secondary=n_sec, n_meta=n_meta),)
    primal = (BandedIndex(id=1, size=n_meta * n_sec, axis=1, other_id=0,
                          block_size=B, block_axis=3, band_width=W, offset=(),
                          primary=True, n_secondary=n_sec, n_meta=n_meta),)
    return SparseTensor(out, primal, data, check_consistency=False)


def _set_tensor(M, B_a, B_b, semantic="union", op=None, key=8):
    """A 2-D ``SparseTensor`` whose dims are a ``SetIndex`` pair (combined 1-D
    val) for the misaligned coprime ``B_a``/``B_b`` case, ``M`` meta-blocks."""
    import math
    lcm = math.lcm(B_a, B_b)
    n_a, n_b = lcm // B_a, lcm // B_b
    lhs = _n((M, n_a, B_a, B_a), key)
    rhs = _n((M, n_b, B_b, B_b), key + 1)
    combined = jnp.concatenate([lhs.reshape(-1), rhs.reshape(-1)])
    op = op if op is not None else (jnp.multiply if semantic == "intersection" else jnp.add)

    def mk(i, o, ax, bax):
        return SetIndex(id=i, size=M * lcm, axis=ax, other_id=o,
                        block_size=lcm, block_axis=bax, semantic=semantic,
                        lhs_shape=lhs.shape, rhs_shape=rhs.shape,
                        include_remainder=True, n_meta=1, op=op)

    return SparseTensor((mk(0, 1, 0, 1),), (mk(1, 0, 0, 2),), combined,
                        check_consistency=False)


class TestBandedIndexTranspose(unittest.TestCase):
    def test_transpose_matches_dense_transpose(self):
        for M, W, B in [(3, 3, 4), (4, 1, 3), (5, 3, 2)]:
            with self.subTest(M=M, W=W, B=B):
                t = _banded_tensor(M, W, B)
                self.assertTrue(any(d.is_compressed for d in t.dims))
                self.assertTrue(jnp.allclose(t.T.dense(), t.dense().T, atol=1e-4))

    def test_n_meta_band_transpose(self):
        t = _banded_tensor(M=3, W=3, B=2, n_meta=2, n_secondary=3)
        self.assertTrue(jnp.allclose(t.T.dense(), t.dense().T, atol=1e-4))

    def test_double_transpose_is_identity(self):
        t = _banded_tensor(M=3, W=3, B=4)
        self.assertTrue(jnp.allclose(t.T.T.dense(), t.dense(), atol=1e-4))

    def test_transpose_preserves_banded_compression(self):
        # The 2-D out↔primal swap is a view on the band buffer: the transposed
        # tensor stays a BandedIndex pair (not densified) — same compact storage.
        for W in (1, 3):
            with self.subTest(W=W):
                t = _banded_tensor(M=4, W=W, B=3)
                tt = t.T
                self.assertTrue(jnp.allclose(tt.dense(), t.dense().T, atol=1e-4))
                self.assertTrue(all(isinstance(d, BandedIndex) for d in tt.dims))
                self.assertEqual(int(tt.val.size), int(t.val.size))  # view, no growth
                self.assertLess(int(tt.val.size), tt.dense().size)

    def test_nonsquare_band_transpose(self):
        # B_row != B_col and M_row != M_col: the swap must still match dense.T.
        data = _n((5, 3, 3, 2), 4)  # M=5, W=3, B_row=3, B_col=2
        out = (BandedIndex(id=0, size=5, axis=0, other_id=1, block_size=3,
                           block_axis=1, band_width=3, offset=(), primary=True,
                           n_secondary=4, n_meta=1),)
        primal = (BandedIndex(id=1, size=4, axis=1, other_id=0, block_size=2,
                              block_axis=3, band_width=3, offset=(), primary=True,
                              n_secondary=4, n_meta=1),)
        t = SparseTensor(out, primal, data, check_consistency=False)
        self.assertTrue(jnp.allclose(t.T.dense(), t.dense().T, atol=1e-4))


class TestSetIndexTranspose(unittest.TestCase):
    def test_union_transpose_matches_dense(self):
        t = _set_tensor(M=2, B_a=5, B_b=11, semantic="union")
        self.assertTrue(any(d.is_compressed for d in t.dims))
        self.assertTrue(jnp.allclose(t.T.dense(), t.dense().T, atol=1e-4))

    def test_intersection_transpose_matches_dense(self):
        t = _set_tensor(M=2, B_a=3, B_b=5, semantic="intersection")
        self.assertTrue(jnp.allclose(t.T.dense(), t.dense().T, atol=1e-4))

    def test_transpose_preserves_setindex_compression(self):
        # The out↔primal swap transposes each per-side block buffer in place:
        # the transposed tensor stays a SetIndex pair (not densified), same val.
        t = _set_tensor(M=4, B_a=2, B_b=3, semantic="union")
        tt = t.T
        self.assertTrue(jnp.allclose(tt.dense(), t.dense().T, atol=1e-4))
        self.assertTrue(all(isinstance(d, SetIndex) for d in tt.dims))
        self.assertEqual(int(tt.val.size), int(t.val.size))  # view, no growth
        self.assertLess(int(tt.val.size), tt.dense().size)

    def test_transpose_is_scatter_free_under_jit(self):
        t = _set_tensor(M=2, B_a=5, B_b=11, semantic="union")

        @jax.jit
        def f(v):
            st = SparseTensor(t.out_dims, t.primal_dims, v, check_consistency=False)
            return st.T.dense()

        text = f.lower(t.val).compile().as_text().lower()
        self.assertEqual(text.count("scatter("), 0)


class TestMultiBandedGatherFree(unittest.TestCase):
    """CR-3 parity: the K-axis densify is gather-free for centered bands."""

    def _data_and_specs(self, K=2, M=4, W=3, B_row=2, B_col=1, offset=()):
        # layout: (M_p_0, W_0, ..., M_p_{K-1}, W_{K-1}, B_row*, B_col*)
        shape = []
        for _ in range(K):
            shape += [M, W]
        shape += [B_row] * K + [B_col] * K
        data = _n(tuple(shape), 1)
        specs = tuple(
            BandAxisSpec(band_width=W, block_row=B_row, block_col=B_col, offset=offset)
            for _ in range(K)
        )
        return data, specs

    def test_centered_k2_no_gather_no_scatter(self):
        data, specs = self._data_and_specs(K=2, offset=())
        text = jax.jit(
            lambda d: _densify_multi_banded(d, specs, jnp.float32(0))
        ).lower(data).compile().as_text().lower()
        self.assertEqual(text.count("gather("), 0)
        self.assertEqual(text.count("scatter("), 0)

    def test_explicit_centered_offset_also_gather_free(self):
        # An explicit but geometrically-centered offset is detected as centered.
        M, W = 4, 3
        off = tuple(a - (W - 1) // 2 for a in range(M))
        data, specs = self._data_and_specs(K=2, M=M, W=W, offset=off)
        text = jax.jit(
            lambda d: _densify_multi_banded(d, specs, jnp.float32(0))
        ).lower(data).compile().as_text().lower()
        self.assertEqual(text.count("gather("), 0)

    def test_centered_matches_explicit_offset_values(self):
        M, W = 4, 3
        off = tuple(a - (W - 1) // 2 for a in range(M))
        data, specs_c = self._data_and_specs(K=2, M=M, W=W, offset=())
        _, specs_e = self._data_and_specs(K=2, M=M, W=W, offset=off)
        got_c = _densify_multi_banded(data, specs_c, jnp.float32(0))
        got_e = _densify_multi_banded(data, specs_e, jnp.float32(0))
        self.assertTrue(jnp.allclose(got_c, got_e, atol=1e-5))


class TestMultiBandedTranspose(unittest.TestCase):
    """K≥2 multi-axis banded transpose is a compression-preserving view."""

    def _k2_banded(self):
        from graphax.sparse.ops.matmul import matmul
        a = SparseTensor(
            (DiagonalIndex(0, 11, 0, 2, 5, 2), DiagonalIndex(1, 3, 1, 3, 4, 4)),
            (DiagonalIndex(2, 11, 0, 0, 5, 3), DiagonalIndex(3, 3, 1, 1, 2, 5)),
            _n((11, 3, 5, 5, 4, 2), 1),
        )
        b = SparseTensor(
            (DiagonalIndex(0, 5, 0, 2, 11, 2), DiagonalIndex(1, 2, 1, 3, 3, 4)),
            (DiagonalIndex(2, 5, 0, 0, 7, 3), DiagonalIndex(3, 2, 1, 1, 9, 5)),
            _n((5, 2, 11, 7, 3, 9), 2),
        )
        res = matmul(a, b)
        assert all(isinstance(d, BandedIndex) for d in res.dims)
        return res

    def test_full_reverse_T_preserves_compression(self):
        from graphax.sparse.ops.transpose import transpose
        res = self._k2_banded()
        D = res.dense()
        t = res.T
        self.assertTrue(jnp.allclose(t.dense(), jnp.transpose(D), atol=1e-4))
        self.assertTrue(all(isinstance(d, BandedIndex) for d in t.dims))
        self.assertEqual(int(t.val.size), int(res.val.size))  # view, no growth
        self.assertLess(int(t.val.size), D.size)

    def test_double_transpose_is_identity(self):
        res = self._k2_banded()
        self.assertTrue(jnp.allclose(res.T.T.dense(), res.dense(), atol=1e-4))

    def test_band_preserving_permutations(self):
        from graphax.sparse.ops.transpose import transpose
        res = self._k2_banded()
        D = res.dense()
        # (out↔primal keep order), (swap pair order both groups), (full reverse)
        for oa, pa, perm in [([2, 3], [0, 1], (2, 3, 0, 1)),
                             ([1, 0], [3, 2], (1, 0, 3, 2)),
                             ([3, 2], [1, 0], (3, 2, 1, 0))]:
            with self.subTest(perm=perm):
                t = transpose(res, oa, pa)
                self.assertTrue(jnp.allclose(t.dense(), jnp.transpose(D, perm), atol=1e-4))
                self.assertTrue(all(isinstance(d, BandedIndex) for d in t.dims))

    def test_transpose_scatter_free_under_jit(self):
        res = self._k2_banded()

        @jax.jit
        def f(v):
            st = SparseTensor(res.out_dims, res.primal_dims, v, check_consistency=False)
            return st.T.dense()

        text = f.lower(res.val).compile().as_text().lower()
        self.assertEqual(text.count("scatter("), 0)


if __name__ == "__main__":
    unittest.main()
