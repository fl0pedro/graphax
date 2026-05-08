"""Correctness + HLO-fusion checks for ``ops.block_storage``.

These cover the dual-buffer ``UnionBlocks`` / ``IntersectionBlocks`` and the
single-buffer skewed ``BlockBanded`` — three pytrees that pack the dense
outputs of elementwise / matmul ops between two block-diagonal sources whose
block sizes can disagree (so the storage is *not* a single rectangular
tensor — it's a pytree of differently-shaped buffers).

For Union/Intersection, ``op`` is a Python callable so it can't be a JAX
pytree leaf; we therefore JIT a small closure that constructs the namedtuple
internally with ``op`` baked in (matching the production usage pattern in
``_demo_block_storage.py`` and ``_probe_orderings.py``).
"""

import unittest

import jax
import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.ops.block_storage import BlockBanded, IntersectionBlocks, UnionBlocks


def _n(shape, key_idx, dtype=jnp.float32):
    return jr.normal(jr.PRNGKey(key_idx), shape).astype(dtype)


def _hlo_op_counts(fn, *args, **kwargs):
    """Return (fusions, scatters, gathers) in the compiled HLO."""
    hlo = jax.jit(fn).lower(*args, **kwargs).compile().as_text()
    fusions = sum(1 for l in hlo.split("\n") if " fusion(" in l)
    scatters = sum(1 for l in hlo.split("\n") if " scatter(" in l.lower())
    gathers = sum(1 for l in hlo.split("\n") if " gather(" in l.lower())
    return fusions, scatters, gathers


def _expected_dual(lhs, rhs, fill_lhs, fill_rhs, op):
    """Reference dense form for the dual-buffer types.

    Builds the full dense ``lhs`` and ``rhs`` each on their own meta-block-
    diagonal of block-diagonals (filled with ``fill_lhs`` / ``fill_rhs``
    everywhere else), then applies ``op`` cell-wise. This matches the
    semantics of ``UnionBlocks.to_dense`` / ``IntersectionBlocks.to_dense``
    by construction:

      - on positions where both sources have stored data:  ``op(l, r)``
      - on positions where only one has data:              ``op(value, other_fill)``
      - on positions where neither has data:               ``op(fill_lhs, fill_rhs)``
    """
    M, n_lhs, B_lhs_h, B_lhs_w, *L = lhs.shape
    _, n_rhs, B_rhs_h, B_rhs_w, *_ = rhs.shape
    LCM_h, LCM_w = n_lhs * B_lhs_h, n_lhs * B_lhs_w
    assert n_rhs * B_rhs_h == LCM_h and n_rhs * B_rhs_w == LCM_w, \
        f"mismatched LCM: lhs=({LCM_h},{LCM_w}), rhs=({n_rhs*B_rhs_h},{n_rhs*B_rhs_w})"

    dense_lhs = jnp.full((M * LCM_h, M * LCM_w, *L), fill_lhs, dtype=lhs.dtype)
    for k in range(M):
        for i in range(n_lhs):
            r0, c0 = k * LCM_h + i * B_lhs_h, k * LCM_w + i * B_lhs_w
            dense_lhs = dense_lhs.at[r0:r0 + B_lhs_h, c0:c0 + B_lhs_w].set(lhs[k, i])

    dense_rhs = jnp.full((M * LCM_h, M * LCM_w, *L), fill_rhs, dtype=rhs.dtype)
    for k in range(M):
        for i in range(n_rhs):
            r0, c0 = k * LCM_h + i * B_rhs_h, k * LCM_w + i * B_rhs_w
            dense_rhs = dense_rhs.at[r0:r0 + B_rhs_h, c0:c0 + B_rhs_w].set(rhs[k, i])

    return op(dense_lhs, dense_rhs)


def _dual_check(test, cls, default_op, M, lhs_shape, rhs_shape, *,
                fill_lhs=0.0, fill_rhs=0.0, op=None, leftover=()):
    """Shared eager+JIT correctness driver for Union/Intersection."""
    op = op if op is not None else default_op
    n_lhs, B_lhs_h, B_lhs_w = lhs_shape
    n_rhs, B_rhs_h, B_rhs_w = rhs_shape
    lhs = _n((M, n_lhs, B_lhs_h, B_lhs_w, *leftover), 1)
    rhs = _n((M, n_rhs, B_rhs_h, B_rhs_w, *leftover), 2)
    fl = jnp.array(fill_lhs, dtype=jnp.float32)
    fr = jnp.array(fill_rhs, dtype=jnp.float32)

    inst = cls(lhs=lhs, rhs=rhs, fill_lhs=fl, fill_rhs=fr, op=op)
    LCM_h, LCM_w = n_lhs * B_lhs_h, n_lhs * B_lhs_w
    test.assertEqual(inst.shape, (M * LCM_h, M * LCM_w, *leftover))

    expected = _expected_dual(lhs, rhs, fl, fr, op)

    # Eager
    with test.subTest(jit=False):
        got = inst.to_dense()
        test.assertEqual(got.shape, expected.shape)
        test.assertTrue(
            jnp.allclose(got, expected, atol=1e-5),
            f"eager max diff {float(jnp.max(jnp.abs(got - expected))):.3e}",
        )

    # JIT — bake ``op`` into a closure since Callable can't be a JAX leaf.
    with test.subTest(jit=True):
        @jax.jit
        def _to_dense(lhs_, rhs_, fl_, fr_):
            return cls(lhs_, rhs_, fl_, fr_, op=op).to_dense()

        got = _to_dense(lhs, rhs, fl, fr)
        test.assertTrue(
            jnp.allclose(got, expected, atol=1e-5),
            f"jit max diff {float(jnp.max(jnp.abs(got - expected))):.3e}",
        )


# ============================================================================
# UnionBlocks  (dual-buffer; default op = jnp.add)
# ============================================================================
class TestUnionBlocks(unittest.TestCase):
    """``UnionBlocks(lhs, rhs, fill_lhs, fill_rhs, op=jnp.add)`` →
    ``(M*LCM_h, M*LCM_w, *L)`` where ``LCM_h = n_lhs*B_lhs_h = n_rhs*B_rhs_h``
    and ``LCM_w = n_lhs*B_lhs_w = n_rhs*B_rhs_w``."""

    def _check(self, **kw):
        _dual_check(self, UnionBlocks, jnp.add, **kw)

    def test_equal_blocks(self):
        """lhs and rhs have identical block layout — degenerate but valid."""
        self._check(M=3, lhs_shape=(2, 2, 2), rhs_shape=(2, 2, 2))

    def test_coprime_blocks(self):
        """Coprime block counts (gcd(n_lhs, n_rhs) = 1) — the canonical
        misaligned case where the two diagonals interleave with finest
        granularity."""
        self._check(M=2, lhs_shape=(3, 5, 5), rhs_shape=(5, 3, 3))

    def test_divisor_blocks(self):
        """One source's block count divides the other's."""
        self._check(M=2, lhs_shape=(2, 2, 2), rhs_shape=(4, 1, 1))

    def test_asymmetric_blocks(self):
        """Rectangular blocks within each source: ``B_h != B_w``."""
        self._check(M=2, lhs_shape=(2, 2, 4), rhs_shape=(4, 1, 2))

    def test_single_block_each(self):
        """``n_lhs = n_rhs = 1`` — one big block per meta-block."""
        self._check(M=3, lhs_shape=(1, 4, 4), rhs_shape=(1, 4, 4))

    def test_with_leftover_dim(self):
        self._check(M=2, lhs_shape=(2, 2, 2), rhs_shape=(2, 2, 2), leftover=(3,))

    def test_with_two_leftover_dims(self):
        self._check(M=2, lhs_shape=(2, 2, 2), rhs_shape=(2, 2, 2), leftover=(3, 5))

    def test_nonzero_fills(self):
        """Non-zero ``fill_lhs`` / ``fill_rhs`` — exercises off-block-diagonal
        and off-meta-block placement (where ``op(fill_lhs, fill_rhs)`` lands)."""
        self._check(M=2, lhs_shape=(2, 2, 2), rhs_shape=(2, 2, 2),
                    fill_lhs=0.5, fill_rhs=0.25)

    def test_custom_op_maximum(self):
        """``op = jnp.maximum`` instead of the default ``jnp.add``."""
        self._check(M=2, lhs_shape=(2, 2, 2), rhs_shape=(2, 2, 2), op=jnp.maximum)

    def test_combined_round_trip(self):
        """``UnionBlocks.combined`` ↔ ``UnionBlocks.from_combined`` round-trip:
        flatten both buffers into one 1-D array, then reconstruct."""
        lhs = _n((2, 3, 5, 5), 1)
        rhs = _n((2, 5, 3, 3), 2)
        fl = jnp.array(0.0, dtype=jnp.float32)
        fr = jnp.array(0.0, dtype=jnp.float32)
        ub = UnionBlocks(lhs, rhs, fl, fr, op=jnp.add)
        flat, meta = ub.combined()
        # Flat buffer is the concatenation of both flattened sources.
        self.assertEqual(int(flat.size), int(lhs.size + rhs.size))
        ub2 = UnionBlocks.from_combined(flat, meta, fl, fr, op=jnp.add)
        self.assertTrue(jnp.allclose(ub.lhs, ub2.lhs))
        self.assertTrue(jnp.allclose(ub.rhs, ub2.rhs))
        self.assertTrue(jnp.allclose(ub.to_dense(), ub2.to_dense(), atol=1e-5))

    def test_lcm_properties(self):
        """``lcm_h`` / ``lcm_w`` properties report the per-meta-block size."""
        ub = UnionBlocks(
            lhs=_n((2, 3, 5, 5), 1), rhs=_n((2, 5, 3, 3), 2),
            fill_lhs=jnp.array(0.0), fill_rhs=jnp.array(0.0),
        )
        self.assertEqual(ub.lcm_h, 15)
        self.assertEqual(ub.lcm_w, 15)
        self.assertEqual(ub.shape, (2 * 15, 2 * 15))

    def test_hlo_no_scatter_no_gather(self):
        """``UnionBlocks.to_dense`` is a broadcast+select+reshape chain on each
        source plus a stitch — no scatter, no gather."""
        lhs = _n((2, 3, 5, 5), 1)
        rhs = _n((2, 5, 3, 3), 2)
        fl = jnp.array(0.0, dtype=jnp.float32)
        fr = jnp.array(0.0, dtype=jnp.float32)

        def _to_dense(lhs_, rhs_, fl_, fr_):
            return UnionBlocks(lhs_, rhs_, fl_, fr_, op=jnp.add).to_dense()

        _, scatters, gathers = _hlo_op_counts(_to_dense, lhs, rhs, fl, fr)
        self.assertEqual(scatters, 0, "expected zero scatter ops")
        self.assertEqual(gathers, 0, "expected zero gather ops")


# ============================================================================
# IntersectionBlocks  (dual-buffer; default op = jnp.multiply)
# ============================================================================
class TestIntersectionBlocks(unittest.TestCase):
    """``IntersectionBlocks(lhs, rhs, fill_lhs, fill_rhs, op=jnp.multiply)`` —
    same dual-buffer layout as ``UnionBlocks``, but combined via ``op`` whose
    natural choice (``multiply`` with zero fills) yields non-fill output only
    where *both* sources have stored data."""

    def _check(self, **kw):
        _dual_check(self, IntersectionBlocks, jnp.multiply, **kw)

    def test_equal_blocks(self):
        self._check(M=3, lhs_shape=(2, 2, 2), rhs_shape=(2, 2, 2))

    def test_coprime_blocks(self):
        self._check(M=2, lhs_shape=(3, 5, 5), rhs_shape=(5, 3, 3))

    def test_divisor_blocks(self):
        self._check(M=2, lhs_shape=(2, 2, 2), rhs_shape=(4, 1, 1))

    def test_asymmetric_blocks(self):
        """``B_h != B_w`` — the whole point of the intersection mode is to
        survive different rectangular block shapes per source."""
        self._check(M=2, lhs_shape=(2, 2, 4), rhs_shape=(4, 1, 2))

    def test_with_leftover_dim(self):
        self._check(M=2, lhs_shape=(2, 2, 2), rhs_shape=(2, 2, 2), leftover=(3,))

    def test_with_two_leftover_dims(self):
        self._check(M=2, lhs_shape=(2, 2, 2), rhs_shape=(2, 2, 2), leftover=(3, 5))

    def test_nonzero_fills(self):
        """``op = multiply`` with non-zero fills — verifies the asymmetric fill
        semantics (``op(value, other_fill)`` at single-source positions)."""
        self._check(M=2, lhs_shape=(2, 2, 2), rhs_shape=(2, 2, 2),
                    fill_lhs=0.5, fill_rhs=0.25)

    def test_custom_op_minimum(self):
        """``op = jnp.minimum`` — useful for elementwise-min variants."""
        self._check(M=2, lhs_shape=(2, 2, 2), rhs_shape=(2, 2, 2), op=jnp.minimum)

    def test_hlo_no_scatter_no_gather(self):
        lhs = _n((2, 3, 5, 5), 1)
        rhs = _n((2, 5, 3, 3), 2)
        fl = jnp.array(0.0, dtype=jnp.float32)
        fr = jnp.array(0.0, dtype=jnp.float32)

        def _to_dense(lhs_, rhs_, fl_, fr_):
            return IntersectionBlocks(lhs_, rhs_, fl_, fr_, op=jnp.multiply).to_dense()

        _, scatters, gathers = _hlo_op_counts(_to_dense, lhs, rhs, fl, fr)
        self.assertEqual(scatters, 0, "expected zero scatter ops")
        self.assertEqual(gathers, 0, "expected zero gather ops")


# ============================================================================
# BlockBanded  (single-buffer skewed; broadcast+select+sum after the rewrite)
# ============================================================================
class TestBlockBanded(unittest.TestCase):
    """``BlockBanded(data: (M, 2w+1, B, B, *L), fill_value) → (M*B, M*B, *L)``."""

    def _expected(self, data, fill):
        M, W, B, _, *L = data.shape
        w = (W - 1) // 2
        out = jnp.full((M * B, M * B, *L), fill, dtype=data.dtype)
        for k in range(M):
            for b in range(W):
                col = k + b - w
                if 0 <= col < M:
                    out = out.at[k * B : (k + 1) * B, col * B : (col + 1) * B].set(
                        data[k, b]
                    )
        return out

    def _check(self, M, w, B, leftover=()):
        W = 2 * w + 1
        data = _n((M, W, B, B, *leftover), 1)
        fill = jnp.array(0.0, dtype=jnp.float32)
        bb = BlockBanded(data=data, fill_value=fill)
        self.assertEqual(bb.shape, (M * B, M * B, *leftover))
        self.assertEqual(bb.half_bandwidth, w)
        for use_jit in (False, True):
            with self.subTest(jit=use_jit):
                fn = jax.jit(BlockBanded.to_dense) if use_jit else BlockBanded.to_dense
                got = fn(bb)
                expected = self._expected(data, fill)
                self.assertEqual(got.shape, expected.shape)
                self.assertTrue(jnp.allclose(got, expected, atol=1e-5))

    def test_diagonal_only(self):
        # bandwidth=0 ⇒ pure block-diagonal
        self._check(M=4, w=0, B=2)

    def test_tridiagonal_blocks(self):
        # bandwidth=1 ⇒ block-tridiagonal
        self._check(M=4, w=1, B=3)

    def test_pentadiagonal_blocks(self):
        self._check(M=5, w=2, B=2)

    def test_with_leftover(self):
        self._check(M=3, w=1, B=2, leftover=(4,))

    def test_hlo_no_scatter_no_gather(self):
        """``BlockBanded.to_dense`` runs entirely on broadcast+select+sum:
        no scatter and (since the gather→broadcast+select+sum rewrite) no
        gather either, so the densify is forward-fusable into a downstream
        kernel without a materialization barrier."""
        bb = BlockBanded(
            data=_n((4, 3, 2, 2), 1), fill_value=jnp.array(0.0, dtype=jnp.float32)
        )
        _, scatters, gathers = _hlo_op_counts(BlockBanded.to_dense, bb)
        self.assertEqual(scatters, 0, "expected zero scatter ops")
        self.assertEqual(gathers, 0, "expected zero gather ops")


# ============================================================================
# PyTree compatibility
# ============================================================================
class TestPyTreeRoundTrip(unittest.TestCase):
    """All three are NamedTuples so JAX auto-registers them as pytrees. For
    ``Union`` / ``Intersection`` the ``op`` field is a Python callable rather
    than a JAX leaf, so the round-trip is tested by *constructing* the pytree
    inside a jitted closure and checking the densified output matches the
    eager result. ``BlockBanded`` has no callable field and round-trips a
    direct ``jit(identity)``."""

    def test_union_blocks_jit_construct_and_dense(self):
        lhs = _n((2, 3, 5, 5), 1)
        rhs = _n((2, 5, 3, 3), 2)
        fl = jnp.array(0.0, dtype=jnp.float32)
        fr = jnp.array(0.0, dtype=jnp.float32)

        @jax.jit
        def _build_and_dense(l, r):
            return UnionBlocks(l, r, fl, fr, op=jnp.add).to_dense()

        out = _build_and_dense(lhs, rhs)
        ref = UnionBlocks(lhs, rhs, fl, fr, op=jnp.add).to_dense()
        self.assertTrue(jnp.allclose(out, ref, atol=1e-5))

    def test_intersection_blocks_jit_construct_and_dense(self):
        lhs = _n((2, 3, 5, 5), 1)
        rhs = _n((2, 5, 3, 3), 2)
        fl = jnp.array(0.0, dtype=jnp.float32)
        fr = jnp.array(0.0, dtype=jnp.float32)

        @jax.jit
        def _build_and_dense(l, r):
            return IntersectionBlocks(l, r, fl, fr, op=jnp.multiply).to_dense()

        out = _build_and_dense(lhs, rhs)
        ref = IntersectionBlocks(lhs, rhs, fl, fr, op=jnp.multiply).to_dense()
        self.assertTrue(jnp.allclose(out, ref, atol=1e-5))

    def test_block_banded_jit_roundtrip(self):
        """``BlockBanded`` has only ``Array`` leaves, so a direct ``jit``
        identity preserves the pytree structure."""
        bb = BlockBanded(
            data=_n((3, 3, 2, 2), 1), fill_value=jnp.array(0.0, dtype=jnp.float32)
        )
        out = jax.jit(lambda x: x)(bb)
        self.assertIsInstance(out, BlockBanded)
        self.assertTrue(jnp.allclose(out.data, bb.data))
        self.assertTrue(jnp.allclose(out.fill_value, bb.fill_value))


if __name__ == "__main__":
    unittest.main()
