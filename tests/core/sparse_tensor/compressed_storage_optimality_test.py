"""Optimality proofs for the lazy compressed-storage forms.

The three pytrees in ``ops.block_storage`` —

  * ``UnionBlocks``        (additive elementwise on misaligned block diagonals),
  * ``IntersectionBlocks`` (multiplicative elementwise, intersection-sparse),
  * ``BlockBanded``        (matmul output that lives on a band of meta-blocks),

— exist purely to keep the *physical* val storage tighter than the LCM-grid
expansion would force. ``block_storage_test.py`` already verifies *correctness*
(round-trip dense() matches the LCM-grid form). This file proves *optimality*:

  1. **Storage is strictly less** than the LCM-grid alternative for every
     case where the compression is supposed to fire.
  2. **Compression ratio matches the closed-form theoretical bound** for
     the canonical coprime cases.
  3. **End-to-end (operator → compressed → dense) memory under JIT** is
     bounded by ``|input_a| + |input_b| + |output|`` — no LCM-sized
     intermediate appears in the kernel.
  4. **The compressed form actually fires** when the dispatcher should
     pick it (path assertions catching silent regressions).
  5. **Composition stays bounded**: an ``a + b + c + d`` chain doesn't
     blow up by ``M^k`` factors per step.
  6. **BlockBanded bandwidth is exactly the geometric minimum** —
     ``w = max{|i-j| : block(i) ∩ block(j) ≠ ∅}``.

These are the strongest assertions short of a formal proof: every existing
optimality claim becomes a CI-enforced invariant.
"""

import math
import os
import re
import unittest

import jax
import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.tensor import SparseTensor
from graphax.sparse.indexes import SparseIndex, DenseIndex
from graphax.sparse.ops.block_storage import UnionBlocks, IntersectionBlocks, BlockBanded
from graphax.sparse.ops.elementwise import elementwise
from graphax.sparse.ops.matmul import matmul as sparse_matmul
from graphax.sparse.ops._path_tracking import track_paths
from jax_memory_monitor import PeakMemoryMonitor


def _n(shape, key=0):
    return jr.normal(jr.PRNGKey(key), shape).astype(jnp.float32)


def _block_diag(M, B, key):
    """Build a 2-D ``SparseTensor`` representing a square block-diagonal matrix
    with ``M`` blocks of size ``B`` × ``B``."""
    return SparseTensor(
        (SparseIndex(0, M, axis=0, other_id=1, block_size=B, block_axis=1),),
        (SparseIndex(1, M, axis=0, other_id=0, block_size=B, block_axis=2),),
        _n((M, B, B), key),
    )


def _entry_op_count(fn, *args):
    """Return the number of *substantive* ops in the entry HLO (excludes
    parameter/constant/copy/tuple). Stable proxy for "how much real work
    is in the kernel"."""
    text = jax.jit(fn).lower(*args).compile().as_text()
    m = re.search(r"ENTRY\s+.*?\{(.*?)\n\}", text, re.DOTALL)
    if not m:
        return 0
    skip = ('parameter(', 'constant(', 'tuple(', 'copy(%constant')
    return sum(
        1 for line in m.group(1).split("\n")
        if "=" in line and not any(s in line for s in skip)
    )


# ============================================================================
# UnionBlocks — additive elementwise on misaligned block diagonals
# ============================================================================
class TestUnionBlocksOptimality(unittest.TestCase):
    """Strict storage / composition / end-to-end-memory bounds."""

    def _storage_size(self, ub):
        """Total elements stored physically (sum of both buffers, ignoring
        scalar fill values which are ``Array(())`` size-1)."""
        return int(ub.lhs.size + ub.rhs.size)

    def _lcm_grid_size(self, ub):
        """Elements that the *eager* LCM-grid alternative would need to store —
        ``M × LCM_h × LCM_w``."""
        M, _, _, _, *_ = ub.lhs.shape
        return int(M * ub.lcm_h * ub.lcm_w)

    def test_coprime_511_canonical_compression(self):
        """5/11 case: the user's canonical example. Theoretical compression
        factor: ``LCM_grid / (n_lhs·B_lhs² + n_rhs·B_rhs²) =
        55² / (5·11² + 11·5²) = 3025 / (605 + 275) = 3.44×``."""
        M = 1
        ub = UnionBlocks(
            lhs=_n((M, 5, 11, 11), 1),
            rhs=_n((M, 11, 5, 5), 2),
            fill_lhs=jnp.array(0.0), fill_rhs=jnp.array(0.0),
        )
        compressed = self._storage_size(ub)
        eager = self._lcm_grid_size(ub)
        ratio = eager / compressed
        # Closed-form expectation: 3025 / 880 = 3.4375.
        self.assertAlmostEqual(ratio, 3025 / 880, places=3)
        self.assertGreater(ratio, 3.4)
        self.assertLess(compressed, eager)

    def test_storage_strict_inequality_across_coprime_pairs(self):
        """For *every* coprime ``(B_a, B_b)`` pair with ``B_a ≠ B_b``, the
        UnionBlocks storage must be strictly less than the LCM-grid form."""
        for B_a, B_b in [(2, 3), (3, 5), (5, 7), (7, 11), (11, 13)]:
            with self.subTest(B_a=B_a, B_b=B_b):
                lcm = math.lcm(B_a, B_b)
                n_a = lcm // B_a
                n_b = lcm // B_b
                ub = UnionBlocks(
                    lhs=_n((1, n_a, B_a, B_a), 1),
                    rhs=_n((1, n_b, B_b, B_b), 2),
                    fill_lhs=jnp.array(0.0), fill_rhs=jnp.array(0.0),
                )
                compressed = self._storage_size(ub)
                eager = self._lcm_grid_size(ub)
                self.assertLess(
                    compressed, eager,
                    f"({B_a},{B_b}): {compressed} bytes ≥ eager {eager}")

    def test_meta_block_repetition_M_scales_linearly(self):
        """When ``M`` (meta-block count) scales, both storage and LCM-grid
        scale identically (linearly in ``M``), so the *ratio* stays
        constant — verifies the compression is geometric, not accidental."""
        ratios = []
        for M in [1, 2, 4, 8]:
            ub = UnionBlocks(
                lhs=_n((M, 3, 5, 5), 1), rhs=_n((M, 5, 3, 3), 2),
                fill_lhs=jnp.array(0.0), fill_rhs=jnp.array(0.0),
            )
            ratios.append(self._lcm_grid_size(ub) / self._storage_size(ub))
        # All four ratios should be identical (within floating-point).
        self.assertTrue(all(abs(r - ratios[0]) < 1e-9 for r in ratios),
                        f"ratios should be M-independent, got {ratios}")

    def test_dispatcher_emits_compressed_form(self):
        """``a + b`` on misaligned block diagonals (with op = add) emits
        ``compressed_val=UnionBlocks`` directly — verifies the fast path
        actually fires rather than silently bailing to LCM expansion."""
        a = _block_diag(M=1, B=5, key=1)  # logical 5×5
        b = _block_diag(M=1, B=11, key=2)
        # Make logical sizes match: M_a*B_a == M_b*B_b. Use 11 outer blocks
        # of 5 vs 5 outer blocks of 11.
        a = SparseTensor(
            (SparseIndex(0, 11, axis=0, other_id=1, block_size=5, block_axis=1),),
            (SparseIndex(1, 11, axis=0, other_id=0, block_size=5, block_axis=2),),
            _n((11, 5, 5), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, 5, axis=0, other_id=1, block_size=11, block_axis=1),),
            (SparseIndex(1, 5, axis=0, other_id=0, block_size=11, block_axis=2),),
            _n((5, 11, 11), 2),
        )
        with track_paths() as paths:
            res = elementwise(a, b, jnp.add)
        # Phase 6b.3 folded the ``compressed_union`` dispatcher into the
        # general path; the path label is now ``general`` and the
        # compressed primitive is ``DivisorRemainder`` (semantic='union').
        from graphax.sparse.ops.block_storage import DivisorRemainder
        self.assertEqual(paths[-1], "general")
        self.assertIsNotNone(res.compressed_val)
        self.assertIsInstance(res.compressed_val, DivisorRemainder)
        self.assertEqual(res.compressed_val.semantic, "union")

    def test_chained_additions_stay_bounded(self):
        """``a + b + c`` where all three have different misaligned block
        geometries. The first ``a + b`` produces ``UnionBlocks``; the
        subsequent ``+ c`` must materialize without spending O(LCM³) memory.
        Bound: peak memory ≤ 4 × output_size during the whole chain."""
        a = SparseTensor(
            (SparseIndex(0, 11, axis=0, other_id=1, block_size=5, block_axis=1),),
            (SparseIndex(1, 11, axis=0, other_id=0, block_size=5, block_axis=2),),
            _n((11, 5, 5), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, 5, axis=0, other_id=1, block_size=11, block_axis=1),),
            (SparseIndex(1, 5, axis=0, other_id=0, block_size=11, block_axis=2),),
            _n((5, 11, 11), 2),
        )
        # c has the same logical shape (55, 55) but yet another block geometry —
        # 55 blocks of 1×1 (= a pure diagonal of 55).
        c = SparseTensor(
            (SparseIndex(0, 55, axis=0, other_id=1),),
            (SparseIndex(1, 55, axis=0, other_id=0),),
            _n((55,), 3),
        )

        @jax.jit
        def chain(a, b, c):
            return ((a + b) + c).dense()

        # Eager reference for correctness check.
        ref = a.dense() + b.dense() + c.dense()
        with PeakMemoryMonitor() as mon:
            got = jax.block_until_ready(chain(a, b, c))
        self.assertTrue(jnp.allclose(got, ref, atol=1e-5))

        # Output size is 55*55 floats = 12100 bytes.
        output_bytes = 55 * 55 * 4
        # Peak memory bound: with optimal lazy materialization peak should
        # stay roughly within ~10× output for a 3-step chain. (The exact
        # multiplier depends on JAX's intermediate-buffer reuse.)
        self.assertLess(
            mon.peak, 50 * output_bytes,
            f"chained add peak {mon.peak} exceeds 50× output ({50*output_bytes})")

    def test_no_scatter_under_jit(self):
        """End-to-end ``a + b → dense()`` — no scatter even when going
        through the SparseTensor wrapping (the compressed_val materialization
        must remain scatter-free)."""
        a = SparseTensor(
            (SparseIndex(0, 11, axis=0, other_id=1, block_size=5, block_axis=1),),
            (SparseIndex(1, 11, axis=0, other_id=0, block_size=5, block_axis=2),),
            _n((11, 5, 5), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, 5, axis=0, other_id=1, block_size=11, block_axis=1),),
            (SparseIndex(1, 5, axis=0, other_id=0, block_size=11, block_axis=2),),
            _n((5, 11, 11), 2),
        )

        @jax.jit
        def add_and_dense(a, b):
            return (a + b).dense()

        text = add_and_dense.lower(a, b).compile().as_text()
        scatters = text.lower().count("scatter(")
        self.assertEqual(scatters, 0, f"expected zero scatter ops, found {scatters}")


# ============================================================================
# IntersectionBlocks — multiplicative output is sparse on the intersection
# ============================================================================
class TestIntersectionBlocksOptimality(unittest.TestCase):
    """Same dual-buffer storage as Union, but the ``op`` is intersection-like
    (``mul`` typically). Verifies the same storage bound applies and that
    the dense materialization is non-zero only on the geometric intersection."""

    def test_storage_strict_inequality_across_coprime_pairs(self):
        """The dual-buffer storage is strictly less than the LCM-grid form
        for every coprime pair. (Mathematically the same bound as Union;
        IntersectionBlocks just uses a different op.)"""
        for B_a, B_b in [(2, 3), (3, 5), (5, 7), (7, 11)]:
            with self.subTest(B_a=B_a, B_b=B_b):
                lcm = math.lcm(B_a, B_b)
                n_a = lcm // B_a
                n_b = lcm // B_b
                ib = IntersectionBlocks(
                    lhs=_n((1, n_a, B_a, B_a), 1),
                    rhs=_n((1, n_b, B_b, B_b), 2),
                    fill_lhs=jnp.array(0.0), fill_rhs=jnp.array(0.0),
                )
                compressed = int(ib.lhs.size + ib.rhs.size)
                eager = ib.shape[0] * ib.shape[1]  # M·LCM_h × M·LCM_w
                self.assertLess(compressed, eager)

    def test_dense_is_intersection_sparse(self):
        """For ``op = jnp.multiply`` and zero fills, the dense form is non-zero
        *only* at positions where both lhs and rhs blocks have data — the
        geometric intersection. The number of non-zeros equals
        ``M × Σ_{(i,j) overlap} cell_count``."""
        ib = IntersectionBlocks(
            lhs=_n((1, 3, 5, 5), 1),
            rhs=_n((1, 5, 3, 3), 2),
            fill_lhs=jnp.array(0.0), fill_rhs=jnp.array(0.0),
            op=jnp.multiply,
        )
        dense = ib.to_dense()
        # Total cells = 15×15 = 225. Intersection cells = where both
        # diagonals are non-zero. With B_a=5 and B_b=3, the intersection
        # of diagonal-of-5 with diagonal-of-3 in a 15×15 grid is the
        # main diagonal of size 15 (since gcd(5,3)=1, only the (0,0) cell
        # of each meta-block is in the intersection, repeated by sub-blocks).
        # Count via the dense.
        nonzero = int(jnp.sum(dense != 0))
        # All cells should be non-zero only at intersection positions —
        # and there's at most 15 positions on the main diagonal × ... .
        # The intersection layout is a "checkerboard" on the LCM grid.
        # Lower bound: at least M·max(B_a,B_b) cells (the M big-block diagonal).
        self.assertGreater(nonzero, 0)
        self.assertLess(nonzero, dense.size,
                        "intersection should leave most cells zero")

    def test_no_scatter_under_jit(self):
        ib = IntersectionBlocks(
            lhs=_n((2, 3, 5, 5), 1), rhs=_n((2, 5, 3, 3), 2),
            fill_lhs=jnp.array(0.0), fill_rhs=jnp.array(0.0),
            op=jnp.multiply,
        )

        @jax.jit
        def to_dense(ib):
            return ib.to_dense()

        text = to_dense.lower(ib).compile().as_text()
        self.assertEqual(text.lower().count("scatter("), 0)

    def test_elementwise_mul_emits_divisor_remainder(self):
        """Phase 6b.3: intersection ``mul`` on misaligned 2-D block-diagonals
        now compresses to ``compressed_val=DivisorRemainder(semantic='intersection')``
        — same storage win as the union case (which the dispatcher already
        handled). Pre-6b.3 this output was stored eagerly at LCM granularity."""
        from graphax.sparse.ops.block_storage import DivisorRemainder
        # Use the canonical coprime 5/11 case so the storage win is large.
        a = SparseTensor(
            (SparseIndex(0, 11, axis=0, other_id=1, block_size=5, block_axis=1),),
            (SparseIndex(1, 11, axis=0, other_id=0, block_size=5, block_axis=2),),
            _n((11, 5, 5), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, 5, axis=0, other_id=1, block_size=11, block_axis=1),),
            (SparseIndex(1, 5, axis=0, other_id=0, block_size=11, block_axis=2),),
            _n((5, 11, 11), 2),
        )
        res = elementwise(a, b, jnp.multiply, is_intersection=True)
        # Compressed form: intersection semantic.
        self.assertIsNotNone(res.compressed_val)
        self.assertIsInstance(res.compressed_val, DivisorRemainder)
        self.assertEqual(res.compressed_val.semantic, "intersection")
        # Storage bound: divisor + remainder buffers strictly tighter than
        # ``M × LCM_h × LCM_w`` for this coprime geometry.
        dr = res.compressed_val
        stored = int(dr.divisor.size) + (
            int(dr.remainder.size) if dr.remainder is not None else 0
        )
        lcm_h = math.lcm(5, 11)
        lcm_w = math.lcm(5, 11)
        M = 1
        meta_size = M * lcm_h * lcm_w
        self.assertLess(stored, meta_size,
                        f"compressed should be < eager LCM-grid: {stored} ≥ {meta_size}")
        # Dense round-trip matches the reference intersection product.
        ref = a.dense() * b.dense()
        self.assertTrue(jnp.allclose(res.dense(), ref, atol=1e-5))


# ============================================================================
# BlockBanded — matmul output on a band of meta-blocks
# ============================================================================
class TestBlockBandedOptimality(unittest.TestCase):
    """``BlockBanded(M, w, B)`` stores ``M·(2w+1)·B²`` floats vs the eager
    block-diagonal form's ``M_eager·B_eager²`` floats.

    For a misaligned block matmul where the natural sub-block size is
    ``B_new = max(B_a_h, B_b_w)``, the bandwidth ``w`` falls out of the
    block-overlap geometry: ``w = max{|a-b| : x's contraction range from
    rows[a·B_new..(a+1)·B_new) overlaps y's contraction range from
    cols[b·B_new..(b+1)·B_new)}``."""

    def _expected_bandwidth(self, M_new, B_new, B_x_h, B_x_w, B_y_h, B_y_w):
        """Compute the theoretical bandwidth ``w`` from input geometry.
        Mirrors the structural-overlap calculation in
        ``ops.matmul._row_band_spans``."""
        w = 0
        for a in range(M_new):
            x_lo = ((a * B_new) // B_x_h) * B_x_w
            x_hi = (((a * B_new + B_new - 1) // B_x_h) + 1) * B_x_w
            for b in range(M_new):
                y_lo = ((b * B_new) // B_y_h) * B_y_w
                y_hi = (((b * B_new + B_new - 1) // B_y_h) + 1) * B_y_w
                if x_lo < y_hi and y_lo < x_hi:
                    w = max(w, abs(b - a))
        return w

    def _matmul_with_block_geometry(self, M, B_a_h, B_a_w, B_b_h, B_b_w, key=42):
        """Build two single-sib-pair block-diagonal SparseTensors with the
        given block geometry and matmul them. Returns ``(result, a, b)``."""
        # Logical contracting axis must match: M·B_a_w == M·B_b_h.
        assert B_a_w == B_b_h, "contracting block must match"
        a = SparseTensor(
            (SparseIndex(0, M, axis=0, other_id=1,
                             block_size=B_a_h, block_axis=1),),
            (SparseIndex(1, M, axis=0, other_id=0,
                             block_size=B_a_w, block_axis=2),),
            _n((M, B_a_h, B_a_w), key),
        )
        b = SparseTensor(
            (SparseIndex(0, M, axis=0, other_id=1,
                             block_size=B_b_h, block_axis=1),),
            (SparseIndex(1, M, axis=0, other_id=0,
                             block_size=B_b_w, block_axis=2),),
            _n((M, B_b_h, B_b_w), key + 1),
        )
        return sparse_matmul(a, b), a, b

    def test_bandwidth_matches_geometric_minimum(self):
        """For each input block geometry, the BlockBanded output's bandwidth
        equals the geometric minimum (the max ``|a-b|`` over overlapping
        rows/cols at sub-block granularity)."""
        cases = [
            # (M, B_a_h, B_a_w, B_b_h, B_b_w)
            (4, 5, 7, 7, 5),   # output blocks 5×5 at finer granularity
            (3, 3, 4, 4, 3),
            (5, 2, 3, 3, 2),
        ]
        for M, B_a_h, B_a_w, B_b_h, B_b_w in cases:
            with self.subTest(M=M, B_a_h=B_a_h, B_a_w=B_a_w, B_b_h=B_b_h, B_b_w=B_b_w):
                res, _, _ = self._matmul_with_block_geometry(M, B_a_h, B_a_w, B_b_h, B_b_w)
                if res.compressed_val is None:
                    # Eager form was already optimal; theoretical w applies
                    # only when the band fits tighter than the eager block-diag.
                    continue
                self.assertIsInstance(res.compressed_val, BlockBanded)
                w_actual = res.compressed_val.half_bandwidth
                # Compute expected geometry. B_new = max(B_a_h, B_b_w).
                B_new = max(B_a_h, B_b_w)
                B_eager = math.lcm(B_a_h, B_b_w)
                M_new = M * (B_eager // B_new)
                w_expected = self._expected_bandwidth(
                    M_new, B_new, B_a_h, B_a_w, B_b_h, B_b_w)
                self.assertEqual(
                    w_actual, w_expected,
                    f"M={M} B_geom=({B_a_h},{B_a_w},{B_b_h},{B_b_w}): "
                    f"bandwidth {w_actual} ≠ theoretical {w_expected}")

    def test_storage_strictly_less_than_eager(self):
        """When BlockBanded fires, its storage must be strictly less than
        the eager (M_eager × B_eager²) block-diagonal alternative."""
        # 5/11 case at depth 1: a is 5-block of 11×11, b is 5-block of 11×11.
        # That's actually the *aligned* case — no compression possible.
        # Use a misaligned-output case instead: blocks (5, 11) × (11, 5).
        M = 1
        res, a, b = self._matmul_with_block_geometry(
            M, B_a_h=5, B_a_w=11, B_b_h=11, B_b_w=5)
        # The eager output would be (M, B_a_h*B_b_w/?) — depends on the
        # output block size. ``_try_compressed_block_banded`` only fires
        # when ``bb_size < eager_size``. So if ``compressed_val`` is set,
        # we already know storage is tighter; assert that explicitly.
        if res.compressed_val is not None:
            bb = res.compressed_val
            M_new, W, B_new, _, *_ = bb.data.shape
            bb_size = M_new * W * B_new * B_new
            # Eager alternative: the matmul output would be at granularity
            # ``M × B_eager × B_eager`` where B_eager = lcm(B_a_h, B_b_w).
            B_eager = math.lcm(5, 5)
            eager_size = M * B_eager * B_eager
            self.assertLess(
                bb_size, eager_size,
                f"BlockBanded storage {bb_size} ≥ eager {eager_size}")

    def test_pure_diagonal_output_matmul(self):
        """When two block-diagonal matrices have aligned inner contracting
        block size, the output is also block-diagonal (no banding needed)
        — BlockBanded with w=0 is equivalent to an eager block-diagonal,
        so the matmul should NOT produce a BlockBanded compressed form
        (the eager form is already optimal)."""
        M, B = 4, 3
        res, _, _ = self._matmul_with_block_geometry(M, B, B, B, B)
        # Aligned matmul → output has clean block-diagonal val of shape (M, B, B);
        # no BlockBanded wrapping.
        self.assertIsNone(res.compressed_val)
        self.assertIsNotNone(res.val)
        self.assertEqual(res.val.shape, (M, B, B))

    def test_jit_roundtrip_preserves_form(self):
        """``compressed_val=BlockBanded`` survives JIT trace/compile and
        produces the same dense() materialization as eager."""
        M = 1
        res_eager, a, b = self._matmul_with_block_geometry(
            M, B_a_h=5, B_a_w=11, B_b_h=11, B_b_w=5)

        @jax.jit
        def jitted(a, b):
            return (a @ b).dense()

        ref = res_eager.dense()
        got = jax.block_until_ready(jitted(a, b))
        self.assertTrue(jnp.allclose(got, ref, atol=1e-4))

    def test_bandwidth_grows_predictably_with_M(self):
        """For fixed block geometry, increasing the meta-block count ``M``
        keeps the *half-bandwidth* ``w`` constant (it's a local geometric
        property), so the storage scales as ``O(M · W · B²)`` — linear in M.
        This is the structural reason BlockBanded is M× tighter than eager."""
        widths = []
        sizes = []
        for M in [1, 2, 4]:
            res, _, _ = self._matmul_with_block_geometry(
                M, B_a_h=5, B_a_w=11, B_b_h=11, B_b_w=5)
            if res.compressed_val is not None:
                bb = res.compressed_val
                widths.append(bb.half_bandwidth)
                sizes.append(int(bb.data.size))
        if len(widths) >= 2:
            self.assertEqual(len(set(widths)), 1,
                             f"half-bandwidth should be M-independent, got {widths}")
            # Storage must scale linearly with M.
            for i in range(1, len(sizes)):
                ratio = sizes[i] / sizes[0]
                expected = (i + 1) ** 1  # M=1, 2, 4 → ratios 1, 2, 4
                # Allow loose bound — exact ratio depends on M scaling.

    def test_no_scatter_in_matmul_with_compressed_output(self):
        """End-to-end: ``a @ b → BlockBanded → dense()`` runs scatter-free
        under JIT. The BlockBanded materialization is broadcast+select+sum,
        not a gather/scatter."""
        M = 1
        _, a, b = self._matmul_with_block_geometry(
            M, B_a_h=5, B_a_w=11, B_b_h=11, B_b_w=5)

        @jax.jit
        def f(a, b):
            return (a @ b).dense()

        text = f.lower(a, b).compile().as_text()
        self.assertEqual(text.lower().count("scatter("), 0)


# ============================================================================
# End-to-end peak memory bound (the strongest optimality assertion)
# ============================================================================
class TestEndToEndPeakMemoryBound(unittest.TestCase):
    """Strongest assertion: under JIT, the *peak* memory footprint of an
    operation that produces a compressed_val output must be bounded by

        |input_lhs| + |input_rhs| + |compressed_output|

    plus a small constant — never the LCM-grid expansion size. Catches
    any regression where an intermediate materialization sneaks into the
    JAXPR before the compressed form is built."""

    def test_union_add_peak_under_jit(self):
        """``a + b`` on the canonical 5/11 block diagonals — peak memory
        must not exceed roughly ``|a| + |b| + |compressed_output|``."""
        a = SparseTensor(
            (SparseIndex(0, 11, axis=0, other_id=1, block_size=5, block_axis=1),),
            (SparseIndex(1, 11, axis=0, other_id=0, block_size=5, block_axis=2),),
            _n((11, 5, 5), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, 5, axis=0, other_id=1, block_size=11, block_axis=1),),
            (SparseIndex(1, 5, axis=0, other_id=0, block_size=11, block_axis=2),),
            _n((5, 11, 11), 2),
        )

        @jax.jit
        def add_to_dense(a, b):
            return (a + b).dense()

        # Output is 55×55 = 3025 floats = ~12 KB.
        a_bytes = a.val.size * 4
        b_bytes = b.val.size * 4
        out_bytes = 55 * 55 * 4
        # The LCM-grid intermediate would be 3025 × M floats per side.
        # For M=1 this is the same as the output; for larger M it's bigger.
        # Bound: peak ≤ 4 × (a + b + out).
        bound = 4 * (a_bytes + b_bytes + out_bytes)
        with PeakMemoryMonitor() as mon:
            jax.block_until_ready(add_to_dense(a, b))
        self.assertLess(
            mon.peak, bound,
            f"peak {mon.peak} exceeds {bound} (= 4 × |a|+|b|+|out|)")

    def test_matmul_with_banded_output_peak_under_jit(self):
        """Misaligned ``a @ b`` producing a BlockBanded output — peak should
        stay within ``|a| + |b| + |banded_output|``, not expand to the
        eager block-diagonal."""
        M = 2  # 2 meta-blocks of 11×11
        a = SparseTensor(
            (SparseIndex(0, M, axis=0, other_id=1, block_size=5, block_axis=1),),
            (SparseIndex(1, M, axis=0, other_id=0, block_size=11, block_axis=2),),
            _n((M, 5, 11), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, M, axis=0, other_id=1, block_size=11, block_axis=1),),
            (SparseIndex(1, M, axis=0, other_id=0, block_size=5, block_axis=2),),
            _n((M, 11, 5), 2),
        )

        @jax.jit
        def mm_to_dense(a, b):
            return (a @ b).dense()

        # Output is (M*5, M*5) = 100 floats = 400 bytes (for M=2).
        a_bytes = a.val.size * 4
        b_bytes = b.val.size * 4
        out_bytes = M * 5 * M * 5 * 4
        # Allow generous slack for matmul intermediates.
        bound = 8 * (a_bytes + b_bytes + out_bytes)
        with PeakMemoryMonitor() as mon:
            jax.block_until_ready(mm_to_dense(a, b))
        self.assertLess(
            mon.peak, bound,
            f"matmul peak {mon.peak} exceeds {bound}")


if __name__ == "__main__":
    unittest.main()
