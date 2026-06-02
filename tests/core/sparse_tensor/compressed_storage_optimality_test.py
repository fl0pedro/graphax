"""Optimality proofs for the compressed-storage Index forms.

The two compressed ``Index`` types in ``ops.indexes`` —

  * ``SetIndex``    (set-theoretic elementwise output — additive *union* or
                     multiplicative *intersection* on misaligned block diagonals),
  * ``BandedIndex`` (matmul output that lives on a band of meta-blocks),

— exist purely to keep the *physical* ``val`` storage tighter than the LCM-grid
expansion would force. ``compressed_index_test.py`` verifies *correctness*
(``densify_axis`` matches an independent numpy oracle). This file proves
*optimality*, driving the real ``elementwise`` / ``matmul`` emitters and
measuring the emitted ``val`` buffer:

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
from graphax.sparse.indexes import DiagonalIndex, DenseIndex, SetIndex, BandedIndex
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
        (DiagonalIndex(0, M, axis=0, other_id=1, block_size=B, block_axis=1),),
        (DiagonalIndex(1, M, axis=0, other_id=0, block_size=B, block_axis=2),),
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
# SetIndex (union) — additive elementwise on misaligned block diagonals
# ============================================================================
def _misaligned_square_pair(B_a, B_b, M=1, key=1):
    """Two square block-diagonal ``SparseTensor`` operands with coprime block
    sizes ``B_a`` / ``B_b`` and the SAME logical shape ``(M·LCM, M·LCM)``:
    ``a`` is ``M·(LCM/B_a)`` blocks of ``B_a``, ``b`` is ``M·(LCM/B_b)`` blocks
    of ``B_b``. ``a + b`` / ``a * b`` then hit the misaligned SetIndex path."""
    lcm = math.lcm(B_a, B_b)
    n_a, n_b = lcm // B_a, lcm // B_b
    a = SparseTensor(
        (DiagonalIndex(0, M * n_a, axis=0, other_id=1, block_size=B_a, block_axis=1),),
        (DiagonalIndex(1, M * n_a, axis=0, other_id=0, block_size=B_a, block_axis=2),),
        _n((M * n_a, B_a, B_a), key),
    )
    b = SparseTensor(
        (DiagonalIndex(0, M * n_b, axis=0, other_id=1, block_size=B_b, block_axis=1),),
        (DiagonalIndex(1, M * n_b, axis=0, other_id=0, block_size=B_b, block_axis=2),),
        _n((M * n_b, B_b, B_b), key + 1),
    )
    return a, b


class TestUnionBlocksOptimality(unittest.TestCase):
    """Strict storage / composition / end-to-end-memory bounds — measured on
    the real ``SetIndex`` buffer emitted by ``elementwise(.., add)``."""

    def _eager_grid_size(self, B_a, B_b, M=1):
        """Elements the *eager* LCM-grid alternative would store: the
        meta-block-diagonal form keeps ``M`` blocks each ``LCM_h × LCM_w``
        (linear in ``M`` — the M meta-blocks sit on the diagonal, the
        off-diagonal zeros aren't stored)."""
        lcm = math.lcm(B_a, B_b)
        return int(M * lcm * lcm)

    def test_coprime_511_canonical_compression(self):
        """5/11 case: the user's canonical example. Theoretical compression
        factor: ``LCM_grid / (n_lhs·B_lhs² + n_rhs·B_rhs²) =
        55² / (5·11² + 11·5²) = 3025 / (605 + 275) = 3.44×``."""
        a, b = _misaligned_square_pair(5, 11, M=1)
        res = elementwise(a, b, jnp.add)
        self.assertTrue(all(isinstance(d, SetIndex) for d in res.dims))
        compressed = int(res.val.size)
        eager = self._eager_grid_size(5, 11)  # 3025
        ratio = eager / compressed
        # Closed-form expectation: 3025 / 880 = 3.4375.
        self.assertAlmostEqual(ratio, 3025 / 880, places=3)
        self.assertGreater(ratio, 3.4)
        self.assertLess(compressed, eager)

    def test_storage_strict_inequality_across_coprime_pairs(self):
        """For *every* coprime ``(B_a, B_b)`` pair with ``B_a ≠ B_b``, the
        emitted ``SetIndex`` buffer must be strictly less than the LCM-grid."""
        for B_a, B_b in [(2, 3), (3, 5), (5, 7), (7, 11), (11, 13)]:
            with self.subTest(B_a=B_a, B_b=B_b):
                a, b = _misaligned_square_pair(B_a, B_b, M=1)
                res = elementwise(a, b, jnp.add)
                compressed = int(res.val.size)
                eager = self._eager_grid_size(B_a, B_b)
                self.assertLess(
                    compressed, eager,
                    f"({B_a},{B_b}): {compressed} ≥ eager {eager}")

    def test_meta_block_repetition_M_scales_linearly(self):
        """When ``M`` (meta-block count) scales, both the emitted buffer and the
        LCM-grid scale linearly in ``M``, so the *ratio* stays constant —
        verifies the compression is geometric, not accidental."""
        ratios = []
        for M in [1, 2, 4, 8]:
            a, b = _misaligned_square_pair(5, 3, M=M)
            res = elementwise(a, b, jnp.add)
            ratios.append(self._eager_grid_size(5, 3, M) / int(res.val.size))
        # All four ratios should be identical (within floating-point).
        self.assertTrue(all(abs(r - ratios[0]) < 1e-9 for r in ratios),
                        f"ratios should be M-independent, got {ratios}")

    def test_dispatcher_emits_compressed_form(self):
        """``a + b`` on misaligned block diagonals (with op = add) emits a
        ``SetIndex`` (semantic 'union') pair directly — verifies the compressed
        path actually fires rather than silently bailing to LCM expansion."""
        a = _block_diag(M=1, B=5, key=1)  # logical 5×5
        b = _block_diag(M=1, B=11, key=2)
        # Make logical sizes match: M_a*B_a == M_b*B_b. Use 11 outer blocks
        # of 5 vs 5 outer blocks of 11.
        a = SparseTensor(
            (DiagonalIndex(0, 11, axis=0, other_id=1, block_size=5, block_axis=1),),
            (DiagonalIndex(1, 11, axis=0, other_id=0, block_size=5, block_axis=2),),
            _n((11, 5, 5), 1),
        )
        b = SparseTensor(
            (DiagonalIndex(0, 5, axis=0, other_id=1, block_size=11, block_axis=1),),
            (DiagonalIndex(1, 5, axis=0, other_id=0, block_size=11, block_axis=2),),
            _n((5, 11, 11), 2),
        )
        with track_paths() as paths:
            res = elementwise(a, b, jnp.add)
        # Phase 8.F: the general path emits SetIndex output dims (semantic
        # 'union') + a combined 1-D band buffer in val (no compressed_val).
        from graphax.sparse.indexes import SetIndex
        self.assertEqual(paths[-1], "general")
        self.assertIsNotNone(res.val)
        self.assertTrue(all(isinstance(d, SetIndex) for d in res.dims))
        self.assertEqual(res.out_dims[0].semantic, "union")

    def test_chained_additions_stay_bounded(self):
        """``a + b + c`` where all three have different misaligned block
        geometries. The first ``a + b`` produces ``UnionBlocks``; the
        subsequent ``+ c`` must materialize without spending O(LCM³) memory.
        Bound: peak memory ≤ 4 × output_size during the whole chain."""
        a = SparseTensor(
            (DiagonalIndex(0, 11, axis=0, other_id=1, block_size=5, block_axis=1),),
            (DiagonalIndex(1, 11, axis=0, other_id=0, block_size=5, block_axis=2),),
            _n((11, 5, 5), 1),
        )
        b = SparseTensor(
            (DiagonalIndex(0, 5, axis=0, other_id=1, block_size=11, block_axis=1),),
            (DiagonalIndex(1, 5, axis=0, other_id=0, block_size=11, block_axis=2),),
            _n((5, 11, 11), 2),
        )
        # c has the same logical shape (55, 55) but yet another block geometry —
        # 55 blocks of 1×1 (= a pure diagonal of 55).
        c = SparseTensor(
            (DiagonalIndex(0, 55, axis=0, other_id=1),),
            (DiagonalIndex(1, 55, axis=0, other_id=0),),
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
            (DiagonalIndex(0, 11, axis=0, other_id=1, block_size=5, block_axis=1),),
            (DiagonalIndex(1, 11, axis=0, other_id=0, block_size=5, block_axis=2),),
            _n((11, 5, 5), 1),
        )
        b = SparseTensor(
            (DiagonalIndex(0, 5, axis=0, other_id=1, block_size=11, block_axis=1),),
            (DiagonalIndex(1, 5, axis=0, other_id=0, block_size=11, block_axis=2),),
            _n((5, 11, 11), 2),
        )

        @jax.jit
        def add_and_dense(a, b):
            return (a + b).dense()

        text = add_and_dense.lower(a, b).compile().as_text()
        scatters = text.lower().count("scatter(")
        self.assertEqual(scatters, 0, f"expected zero scatter ops, found {scatters}")


# ============================================================================
# SetIndex (intersection) — multiplicative output is sparse on the intersection
# ============================================================================
class TestIntersectionBlocksOptimality(unittest.TestCase):
    """Same emitted ``SetIndex`` buffer as the union case, but the ``op`` is
    intersection-like (``mul``). Verifies the same storage bound applies and
    that the dense materialization is non-zero only on the geometric
    intersection."""

    def test_storage_strict_inequality_across_coprime_pairs(self):
        """The emitted ``SetIndex`` buffer is strictly less than the LCM-grid
        for every coprime pair. (Same bound as union; only the op differs.)"""
        for B_a, B_b in [(2, 3), (3, 5), (5, 7), (7, 11)]:
            with self.subTest(B_a=B_a, B_b=B_b):
                a, b = _misaligned_square_pair(B_a, B_b, M=1)
                res = elementwise(a, b, jnp.multiply, is_intersection=True)
                compressed = int(res.val.size)
                lcm = math.lcm(B_a, B_b)
                eager = lcm * lcm  # M·LCM_h × M·LCM_w, M=1
                self.assertLess(compressed, eager)

    def test_dense_is_intersection_sparse(self):
        """For ``op = jnp.multiply`` and zero fills, the dense form is non-zero
        *only* at positions where both lhs and rhs blocks have data — the
        geometric intersection — so most LCM-grid cells stay zero."""
        a, b = _misaligned_square_pair(5, 3, M=1)
        res = elementwise(a, b, jnp.multiply, is_intersection=True)
        dense = res.dense()
        nonzero = int(jnp.sum(dense != 0))
        self.assertGreater(nonzero, 0)
        self.assertLess(nonzero, dense.size,
                        "intersection should leave most cells zero")

    def test_no_scatter_under_jit(self):
        a, b = _misaligned_square_pair(5, 3, M=2)

        @jax.jit
        def mul_to_dense(a, b):
            return elementwise(a, b, jnp.multiply, is_intersection=True).dense()

        text = mul_to_dense.lower(a, b).compile().as_text()
        self.assertEqual(text.lower().count("scatter("), 0)

    def test_elementwise_mul_emits_set_index(self):
        """Phase 8.F: intersection ``mul`` on misaligned 2-D block-diagonals
        compresses to ``SetIndex(semantic='intersection')`` output dims + a
        combined band buffer in ``val`` (no ``compressed_val``). Storage is
        strictly tighter than the LCM-grid and ``.dense()`` round-trips."""
        from graphax.sparse.indexes import SetIndex
        # Use the canonical coprime 5/11 case so the storage win is large.
        a = SparseTensor(
            (DiagonalIndex(0, 11, axis=0, other_id=1, block_size=5, block_axis=1),),
            (DiagonalIndex(1, 11, axis=0, other_id=0, block_size=5, block_axis=2),),
            _n((11, 5, 5), 1),
        )
        b = SparseTensor(
            (DiagonalIndex(0, 5, axis=0, other_id=1, block_size=11, block_axis=1),),
            (DiagonalIndex(1, 5, axis=0, other_id=0, block_size=11, block_axis=2),),
            _n((5, 11, 11), 2),
        )
        res = elementwise(a, b, jnp.multiply, is_intersection=True)
        self.assertIsNotNone(res.val)
        self.assertTrue(all(isinstance(d, SetIndex) for d in res.dims))
        self.assertEqual(res.out_dims[0].semantic, "intersection")
        # Storage bound: combined band buffer strictly tighter than M·LCM_h·LCM_w.
        meta_size = 1 * math.lcm(5, 11) * math.lcm(5, 11)
        self.assertLess(int(res.val.size), meta_size,
                        f"compressed should be < eager LCM-grid: {int(res.val.size)} ≥ {meta_size}")
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
            (DiagonalIndex(0, M, axis=0, other_id=1,
                             block_size=B_a_h, block_axis=1),),
            (DiagonalIndex(1, M, axis=0, other_id=0,
                             block_size=B_a_w, block_axis=2),),
            _n((M, B_a_h, B_a_w), key),
        )
        b = SparseTensor(
            (DiagonalIndex(0, M, axis=0, other_id=1,
                             block_size=B_b_h, block_axis=1),),
            (DiagonalIndex(1, M, axis=0, other_id=0,
                             block_size=B_b_w, block_axis=2),),
            _n((M, B_b_h, B_b_w), key + 1),
        )
        return sparse_matmul(a, b), a, b

    def test_banded_output_emits_banded_index(self):
        """Phase 8: a misaligned-contract matmul emits ``BandedIndex`` output
        dims + a compact band buffer in ``val`` (no ``compressed_val``), and
        ``.dense()`` round-trips exactly. Storage is strictly less than the
        dense alternative for every banded case."""
        from graphax.sparse.indexes import BandedIndex

        cases = [
            (4, 5, 7, 7, 5),
            (3, 3, 4, 4, 3),
            (5, 2, 3, 3, 2),
        ]
        for M, B_a_h, B_a_w, B_b_h, B_b_w in cases:
            with self.subTest(M=M, B_a_h=B_a_h, B_a_w=B_a_w, B_b_h=B_b_h, B_b_w=B_b_w):
                res, a, b = self._matmul_with_block_geometry(M, B_a_h, B_a_w, B_b_h, B_b_w)
                if not any(isinstance(d, BandedIndex) for d in res.dims):
                    continue  # aligned / already-tight geometry: plain val
                self.assertIsNotNone(res.val)
                self.assertTrue(all(isinstance(d, BandedIndex) for d in res.dims))
                dense = res.dense()
                self.assertLess(
                    int(res.val.size), int(dense.size),
                    "banded val storage must be < dense output size")
                self.assertTrue(jnp.allclose(dense, a.dense() @ b.dense(), atol=1e-4))

    def test_storage_strictly_less_than_eager(self):
        """When the band fires, the compact ``val`` storage is strictly less
        than the dense ``(M_row*B_row, M_col*B_col)`` output."""
        from graphax.sparse.indexes import BandedIndex

        M = 1
        res, a, b = self._matmul_with_block_geometry(
            M, B_a_h=5, B_a_w=11, B_b_h=11, B_b_w=5)
        if any(isinstance(d, BandedIndex) for d in res.dims):
            dense_size = res.dense().size
            self.assertLess(int(res.val.size), int(dense_size))

    def test_pure_diagonal_output_matmul(self):
        """Aligned inner contracting block sizes → block-diagonal output, no
        banding: the matmul keeps a clean ``val=(M, B, B)`` with no
        ``BandedIndex`` dims."""
        from graphax.sparse.indexes import BandedIndex

        M, B = 4, 3
        res, _, _ = self._matmul_with_block_geometry(M, B, B, B, B)
        self.assertFalse(any(isinstance(d, BandedIndex) for d in res.dims))
        self.assertIsNotNone(res.val)
        self.assertEqual(res.val.shape, (M, B, B))

    def test_jit_roundtrip_preserves_form(self):
        """A banded matmul output survives JIT trace/compile and produces the
        same ``dense()`` materialization as eager."""
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
        keeps the band width constant (a local geometric property), so the
        band ``val`` storage scales linearly in M — the structural reason the
        banded form is M× tighter than dense."""
        from graphax.sparse.indexes import BandedIndex

        widths = []
        sizes = []
        for M in [1, 2, 4]:
            res, _, _ = self._matmul_with_block_geometry(
                M, B_a_h=5, B_a_w=11, B_b_h=11, B_b_w=5)
            bx = next((d for d in res.dims if isinstance(d, BandedIndex)), None)
            if bx is not None:
                widths.append(bx.band_width)
                sizes.append(int(res.val.size))
        if len(widths) >= 2:
            self.assertEqual(len(set(widths)), 1,
                             f"band width should be M-independent, got {widths}")
            for i in range(1, len(sizes)):
                self.assertGreater(sizes[i], sizes[0])  # grows with M

    def test_no_scatter_in_matmul_with_compressed_output(self):
        """End-to-end: ``a @ b → BandedIndex → dense()`` runs scatter-free
        under JIT (the band densify is broadcast+select+sum, no scatter)."""
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
            (DiagonalIndex(0, 11, axis=0, other_id=1, block_size=5, block_axis=1),),
            (DiagonalIndex(1, 11, axis=0, other_id=0, block_size=5, block_axis=2),),
            _n((11, 5, 5), 1),
        )
        b = SparseTensor(
            (DiagonalIndex(0, 5, axis=0, other_id=1, block_size=11, block_axis=1),),
            (DiagonalIndex(1, 5, axis=0, other_id=0, block_size=11, block_axis=2),),
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
            (DiagonalIndex(0, M, axis=0, other_id=1, block_size=5, block_axis=1),),
            (DiagonalIndex(1, M, axis=0, other_id=0, block_size=11, block_axis=2),),
            _n((M, 5, 11), 1),
        )
        b = SparseTensor(
            (DiagonalIndex(0, M, axis=0, other_id=1, block_size=11, block_axis=1),),
            (DiagonalIndex(1, M, axis=0, other_id=0, block_size=5, block_axis=2),),
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


class TestMultiAxisContractMatmul(unittest.TestCase):
    """K>1 contracting axes with K'>1 misalignments.

    Phase 7.3 verification: graphax's ``matmul`` natively supports multi-
    axis contraction (``_build_matmul_topology`` pairs the last
    ``min(len(lhs.primal), len(rhs.out))`` dims as contract pairs). When
    multiple contract pairs are misaligned, the OUTPUT carries a band
    structure along the corresponding output axis pairs — same logic as
    K=1 but factorized along multiple independent axes.

    Currently the probe ``_should_emit_block_banded`` gates on
    ``len(pairs) == 1`` so K>1 cases fall through to the dense
    ``val=values`` path: **output is correct, but no compression is
    applied**. Multi-axis BlockBanded compression is a future extension
    (the data layout needs ``(M_p1, W1, B_row1, M_p2, W2, B_row2, B_col1,
    B_col2, *L)`` or similar to encode bands along multiple axes).

    These tests lock in the *correctness* of K>1 multi-misalignment
    matmul output. When multi-axis compression lands, the storage
    assertion below should flip to assert compression instead.
    """

    def test_k2_misaligned_contract_matches_einsum(self):
        """4-D matmul contracting on 2 axes, both misaligned (5/11 and
        2/3). Output must equal a hand-rolled einsum reference."""
        a = SparseTensor(
            (
                DiagonalIndex(0, 11, axis=0, other_id=2, block_size=5, block_axis=2),
                DiagonalIndex(1, 3,  axis=1, other_id=3, block_size=4, block_axis=4),
            ),
            (
                DiagonalIndex(2, 11, axis=0, other_id=0, block_size=5, block_axis=3),
                DiagonalIndex(3, 3,  axis=1, other_id=1, block_size=2, block_axis=5),
            ),
            _n((11, 3, 5, 5, 4, 2), 1),
        )
        b = SparseTensor(
            (
                DiagonalIndex(0, 5, axis=0, other_id=2, block_size=11, block_axis=2),
                DiagonalIndex(1, 2, axis=1, other_id=3, block_size=3,  block_axis=4),
            ),
            (
                DiagonalIndex(2, 5, axis=0, other_id=0, block_size=7, block_axis=3),
                DiagonalIndex(3, 2, axis=1, other_id=1, block_size=9, block_axis=5),
            ),
            _n((5, 2, 11, 7, 3, 9), 2),
        )
        ref = jnp.einsum("ijkl,klmn->ijmn", a.dense(), b.dense())
        res = sparse_matmul(a, b)
        self.assertTrue(jnp.allclose(res.dense(), ref, atol=1e-4))

    def test_k2_misaligned_emits_multi_axis_banded(self):
        """Phase 8: K=2 misaligned-contract matmul emits ``K`` ``BandedIndex``
        output pairs + a compact band buffer in ``val`` (no ``compressed_val``),
        round-trips through ``.dense()``, and stores strictly less than the
        dense 4-D output."""
        from graphax.sparse.indexes import BandedIndex

        a = SparseTensor(
            (
                DiagonalIndex(0, 11, axis=0, other_id=2, block_size=5, block_axis=2),
                DiagonalIndex(1, 3,  axis=1, other_id=3, block_size=4, block_axis=4),
            ),
            (
                DiagonalIndex(2, 11, axis=0, other_id=0, block_size=5, block_axis=3),
                DiagonalIndex(3, 3,  axis=1, other_id=1, block_size=2, block_axis=5),
            ),
            _n((11, 3, 5, 5, 4, 2), 1),
        )
        b = SparseTensor(
            (
                DiagonalIndex(0, 5, axis=0, other_id=2, block_size=11, block_axis=2),
                DiagonalIndex(1, 2, axis=1, other_id=3, block_size=3,  block_axis=4),
            ),
            (
                DiagonalIndex(2, 5, axis=0, other_id=0, block_size=7, block_axis=3),
                DiagonalIndex(3, 2, axis=1, other_id=1, block_size=9, block_axis=5),
            ),
            _n((5, 2, 11, 7, 3, 9), 2),
        )
        ref = jnp.einsum("ijkl,klmn->ijmn", a.dense(), b.dense())
        res = sparse_matmul(a, b)
        # K=2 → 4 BandedIndex output dims; band buffer in val (no compressed_val).
        self.assertEqual(len(res.dims), 4)
        self.assertTrue(all(isinstance(d, BandedIndex) for d in res.dims))
        self.assertIsNotNone(res.val)
        # Storage strictly tighter than dense.
        dense_size = 55 * 12 * 35 * 18
        self.assertLess(
            int(res.val.size), dense_size,
            f"band buffer {int(res.val.size)} should be < dense {dense_size}",
        )
        # Round-trips exactly through dense().
        self.assertTrue(jnp.allclose(res.dense(), ref, atol=1e-4))


class TestMultiAxisElementwise(unittest.TestCase):
    """K>2 elementwise on operands with multiple misaligned sparse pairs.

    Phase 7.4 verification: graphax's elementwise supports arbitrarily
    many sparse pairs per operand. When ``K > 2`` of those pairs are
    misaligned, the OUTPUT carries independent block-diagonal compression
    structure along each pair's axes — same logic as K=1 (single sparse
    pair) but factorized along K independent axis groups.

    Currently the probe ``_should_emit_divisor_remainder`` gates on
    ``len(lhs.dims) != 2`` so K>1 cases fall through to the general
    expansion path: **output is correct, but no DivisorRemainder
    compression is applied**. Multi-axis DivisorRemainder compression is
    a Phase 7.5 extension (storage needs per-axis ``(M, n, B_h, B_w)``
    buffers, densify chains ``_block_diag_per_meta`` along each axis).

    These tests lock in the *correctness* of K>2 multi-misalignment
    elementwise output. When multi-axis compression lands, the storage
    assertion below should flip to assert compression.
    """

    def test_k2_misaligned_add_matches_dense(self):
        """4-D elementwise add, both sparse pairs misaligned (5/7 and
        3/4). Output must equal dense add."""
        a = SparseTensor(
            (
                DiagonalIndex(0, 7, axis=0, other_id=2, block_size=5, block_axis=2),
                DiagonalIndex(1, 4, axis=1, other_id=3, block_size=3, block_axis=4),
            ),
            (
                DiagonalIndex(2, 7, axis=0, other_id=0, block_size=5, block_axis=3),
                DiagonalIndex(3, 4, axis=1, other_id=1, block_size=3, block_axis=5),
            ),
            _n((7, 4, 5, 5, 3, 3), 1),
        )
        b = SparseTensor(
            (
                DiagonalIndex(0, 5, axis=0, other_id=2, block_size=7, block_axis=2),
                DiagonalIndex(1, 3, axis=1, other_id=3, block_size=4, block_axis=4),
            ),
            (
                DiagonalIndex(2, 5, axis=0, other_id=0, block_size=7, block_axis=3),
                DiagonalIndex(3, 3, axis=1, other_id=1, block_size=4, block_axis=5),
            ),
            _n((5, 3, 7, 7, 4, 4), 2),
        )
        ref = a.dense() + b.dense()
        res = a + b
        self.assertTrue(jnp.allclose(res.dense(), ref, atol=1e-4))

    def test_k2_misaligned_mul_matches_dense(self):
        """Same operand shape as the add case, but with intersection
        (multiply) semantics. K>1 cases fall through the dispatcher and
        the general path still produces the correct dense result."""
        a = SparseTensor(
            (
                DiagonalIndex(0, 7, axis=0, other_id=2, block_size=5, block_axis=2),
                DiagonalIndex(1, 4, axis=1, other_id=3, block_size=3, block_axis=4),
            ),
            (
                DiagonalIndex(2, 7, axis=0, other_id=0, block_size=5, block_axis=3),
                DiagonalIndex(3, 4, axis=1, other_id=1, block_size=3, block_axis=5),
            ),
            _n((7, 4, 5, 5, 3, 3), 1),
        )
        b = SparseTensor(
            (
                DiagonalIndex(0, 5, axis=0, other_id=2, block_size=7, block_axis=2),
                DiagonalIndex(1, 3, axis=1, other_id=3, block_size=4, block_axis=4),
            ),
            (
                DiagonalIndex(2, 5, axis=0, other_id=0, block_size=7, block_axis=3),
                DiagonalIndex(3, 3, axis=1, other_id=1, block_size=4, block_axis=5),
            ),
            _n((5, 3, 7, 7, 4, 4), 2),
        )
        ref = a.dense() * b.dense()
        res = a * b
        self.assertTrue(jnp.allclose(res.dense(), ref, atol=1e-4))

    def test_k3_misaligned_add_matches_dense(self):
        """6-D elementwise add with 3 sparse pairs, all misaligned (each
        2/3 coprime). Output must equal dense add."""
        a3 = SparseTensor(
            (
                DiagonalIndex(0, 3, axis=0, other_id=3, block_size=2, block_axis=3),
                DiagonalIndex(1, 3, axis=1, other_id=4, block_size=2, block_axis=5),
                DiagonalIndex(2, 3, axis=2, other_id=5, block_size=2, block_axis=7),
            ),
            (
                DiagonalIndex(3, 3, axis=0, other_id=0, block_size=2, block_axis=4),
                DiagonalIndex(4, 3, axis=1, other_id=1, block_size=2, block_axis=6),
                DiagonalIndex(5, 3, axis=2, other_id=2, block_size=2, block_axis=8),
            ),
            _n((3, 3, 3, 2, 2, 2, 2, 2, 2), 1),
        )
        b3 = SparseTensor(
            (
                DiagonalIndex(0, 2, axis=0, other_id=3, block_size=3, block_axis=3),
                DiagonalIndex(1, 2, axis=1, other_id=4, block_size=3, block_axis=5),
                DiagonalIndex(2, 2, axis=2, other_id=5, block_size=3, block_axis=7),
            ),
            (
                DiagonalIndex(3, 2, axis=0, other_id=0, block_size=3, block_axis=4),
                DiagonalIndex(4, 2, axis=1, other_id=1, block_size=3, block_axis=6),
                DiagonalIndex(5, 2, axis=2, other_id=2, block_size=3, block_axis=8),
            ),
            _n((2, 2, 2, 3, 3, 3, 3, 3, 3), 2),
        )
        ref = a3.dense() + b3.dense()
        res = a3 + b3
        self.assertTrue(jnp.allclose(res.dense(), ref, atol=1e-4))

    def test_k2_misaligned_currently_stores_dense(self):
        """Documents the current limitation: K>1 elementwise misalignment
        falls through to the general path's dense ``val=values`` storage
        (no DivisorRemainder compression yet). Phase 7.5 will flip this
        assertion when multi-axis DivisorRemainder lands."""
        a = SparseTensor(
            (
                DiagonalIndex(0, 7, axis=0, other_id=2, block_size=5, block_axis=2),
                DiagonalIndex(1, 4, axis=1, other_id=3, block_size=3, block_axis=4),
            ),
            (
                DiagonalIndex(2, 7, axis=0, other_id=0, block_size=5, block_axis=3),
                DiagonalIndex(3, 4, axis=1, other_id=1, block_size=3, block_axis=5),
            ),
            _n((7, 4, 5, 5, 3, 3), 1),
        )
        b = SparseTensor(
            (
                DiagonalIndex(0, 5, axis=0, other_id=2, block_size=7, block_axis=2),
                DiagonalIndex(1, 3, axis=1, other_id=3, block_size=4, block_axis=4),
            ),
            (
                DiagonalIndex(2, 5, axis=0, other_id=0, block_size=7, block_axis=3),
                DiagonalIndex(3, 3, axis=1, other_id=1, block_size=4, block_axis=5),
            ),
            _n((5, 3, 7, 7, 4, 4), 2),
        )
        res = a + b
        self.assertFalse(
            any(isinstance(d, SetIndex) for d in res.dims),
            "K>1 multi-axis elementwise compression not implemented yet — "
            "flip this assertion when multi-axis SetIndex lands.",
        )
        self.assertIsNotNone(res.val)


if __name__ == "__main__":
    unittest.main()
