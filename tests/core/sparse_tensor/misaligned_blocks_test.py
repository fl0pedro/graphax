"""Misaligned-block tests for elementwise (union & intersection) and matmul.

Three regimes per pair of block sizes ``(B_lhs, B_rhs)``:

  * **coprime**  (e.g. 2/3, 3/5)
        ``LCM(B_lhs, B_rhs) = B_lhs * B_rhs`` — much bigger than either input.
        Output has FEW LARGE blocks (each of size ``B_lhs * B_rhs``).
  * **shared factor**  (e.g. 4/6, 6/9)
        ``LCM`` strictly exceeds ``max(B_lhs, B_rhs)`` but is smaller than the product.
        Output has a moderate number of mid-sized blocks.
  * **divisor**  (e.g. 2/4, 3/9)
        ``LCM(B_lhs, B_rhs) = max(B_lhs, B_rhs)`` — the smaller block fits cleanly into
        the larger. Output has MANY blocks at the larger of the two sizes.

For matmul the same regimes apply along the *contracting* dimension; we exercise 2D,
3D, and 4D cases with one and two misalignments on the contraction axes.

Sparsity assertions
-------------------
Beyond correctness, every union / intersection test also pins down the *structure*
of the output: the elementwise output of two block-diagonal sources with LCM-block
size ``L`` over an ``M``-meta-block grid must be a meta-block-diagonal
``SparseTensor`` (one ``SparseIndex`` pair of size ``M`` with ``block_size = L``,
val of shape ``(M, L_h, L_w, *L)``) — *not* a fully dense ``(M*L_h, M*L_w)`` buffer.
That's the structure ``SparseTensor.from_compressed(UnionBlocks(...))`` produces, and
the structure that lets every downstream sparse op stay on the block-diagonal fast
path instead of touching M²-many zero meta-blocks.

Reference for every test: densify both operands and compute the dense expected value via
``jnp.matmul`` / the elementwise op. We assert agreement in both no-JIT and JIT modes.
"""
import math
import unittest

import jax
import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.tensor import SparseTensor
from graphax.sparse.indexes import DenseIndex, SparseIndex
from graphax.sparse.ops.matmul import matmul
from graphax.sparse.ops.elementwise import elementwise
from graphax.sparse.ops.block_storage import UnionBlocks, IntersectionBlocks


def _n(shape, key_idx, dtype=jnp.float32):
    return jr.normal(jr.PRNGKey(key_idx), shape).astype(dtype)


def _sparse_pair_2d(N, B_o, B_i, key_idx):
    """A 2-D SparseTensor with one sparse pair of size ``N`` and block_sizes ``(B_o, B_i)``."""
    return SparseTensor(
        (SparseIndex(0, N, axis=0, other_id=1, block_size=B_o, block_axis=1),),
        (SparseIndex(1, N, axis=0, other_id=0, block_size=B_i, block_axis=2),),
        _n((N, B_o, B_i), key_idx),
    )


# ============================================================================
# Elementwise — union mode (add)
# ============================================================================
class TestElementwiseUnionMisalignedBlocks(unittest.TestCase):
    """``add`` over two block-sparse operands. Both sides must agree on logical shape;
    block sizes are allowed to differ — the algorithm reconciles them via LCM blocks."""

    def _check(self, a, b, atol=1e-5):
        for use_jit in (False, True):
            with self.subTest(jit=use_jit):
                fn = jax.jit(lambda x, y: elementwise(x, y, jax.lax.add)) if use_jit \
                     else (lambda x, y: elementwise(x, y, jax.lax.add))
                got = fn(a, b)
                expected = a.dense() + b.dense()
                self.assertEqual(got.shape, a.shape)
                self.assertTrue(
                    jnp.allclose(got.dense(), expected, atol=atol),
                    f"add mismatch (jit={use_jit}): max diff "
                    f"{float(jnp.max(jnp.abs(got.dense() - expected)))}",
                )
                self._assert_meta_block_diagonal_output(got, a, b)

    @staticmethod
    def _expected_meta(a: SparseTensor, b: SparseTensor) -> tuple[int, int, int]:
        """``(M, LCM_h, LCM_w)`` for the union output of two 2-D sparse-pair operands."""
        ao, ai = a.out_dims[0], a.primal_dims[0]
        bo, bi = b.out_dims[0], b.primal_dims[0]
        lcm_h = math.lcm(ao.block_size, bo.block_size)
        lcm_w = math.lcm(ai.block_size, bi.block_size)
        # Meta-axis count is total_h / lcm_h (== total_w / lcm_w by construction).
        M = (ao.size * ao.block_size) // lcm_h
        return M, lcm_h, lcm_w

    def _assert_meta_block_diagonal_output(self, got, a, b):
        """The output must be a meta-block-diagonal ``SparseTensor`` over M LCM-blocks
        — *not* a fully dense buffer. Storage stays at ``M·LCM_h·LCM_w`` rather than
        the M× larger ``M²·LCM_h·LCM_w`` you'd get if we materialized the
        off-meta-block fill positions, and downstream ops keep hitting the
        block-diagonal fast path.

        When ``M == 1`` (the entire tensor is one LCM-block), the M axis is
        squeezed away and val carries shape ``(LCM_h, LCM_w)`` directly — that's
        an additional optimization, not a regression."""
        M, lcm_h, lcm_w = self._expected_meta(a, b)
        out, primal = got.out_dims[0], got.primal_dims[0]
        self.assertIsInstance(out, SparseIndex)
        self.assertIsInstance(primal, SparseIndex)
        self.assertEqual(out.size, M)
        self.assertEqual(primal.size, M)
        self.assertEqual(out.block_size, lcm_h,
                         f"out_dim block_size {out.block_size} ≠ lcm_h {lcm_h}")
        self.assertEqual(primal.block_size, lcm_w,
                         f"primal_dim block_size {primal.block_size} ≠ lcm_w {lcm_w}")
        meta_size = M * lcm_h * lcm_w
        full_dense_size = (M * lcm_h) * (M * lcm_w)
        if got.compressed_val is not None:
            # Lazy form: ``DivisorRemainder`` (Phase 6b.3 — replaces the legacy
            # ``UnionBlocks``) with matching meta-block-diag shape, and storage
            # strictly tighter than the eager ``(M, LCM_h, LCM_w)`` form.
            from graphax.sparse.ops.block_storage import DivisorRemainder
            self.assertIsInstance(got.compressed_val, DivisorRemainder)
            self.assertEqual(got.compressed_val.meta_block_shape, (M, lcm_h, lcm_w))
            stored = got.compressed_val.divisor.size + (
                got.compressed_val.remainder.size
                if got.compressed_val.remainder is not None
                else 0
            )
            self.assertLess(stored, meta_size,
                            f"lazy form should be < eager: {stored} ≥ {meta_size}")
            return
        # Eager form: val carries the M meta-blocks directly.
        expected_val_shape = (M, lcm_h, lcm_w) if M > 1 else (lcm_h, lcm_w)
        self.assertEqual(got.val.shape, expected_val_shape,
                         f"val shape {got.val.shape} ≠ {expected_val_shape} — "
                         f"output isn't using meta-block-diagonal storage")
        # Storage budget: meta-block-diagonal = M·LCM_h·LCM_w, fully dense would
        # be (M·LCM_h)·(M·LCM_w) = M²·LCM_h·LCM_w. Lock in the M× compression.
        self.assertEqual(got.val.size, M * lcm_h * lcm_w)
        full_dense_size = (M * lcm_h) * (M * lcm_w)
        self.assertEqual(got.val.size * M, full_dense_size,
                         "meta-block-diagonal must give M× compression vs full dense")

    # --- coprime block sizes → few large LCM blocks ----------------------
    def test_coprime_2x3(self):
        # logical 12x12: lhs as 6 outer blocks of 2x2, rhs as 4 outer blocks of 3x3
        # LCM(2,3)=6 → 2 unified blocks of 6x6 — "few large blocks"
        a = _sparse_pair_2d(N=6, B_o=2, B_i=2, key_idx=1)
        b = _sparse_pair_2d(N=4, B_o=3, B_i=3, key_idx=2)
        self.assertEqual(a.shape, (12, 12))
        self.assertEqual(b.shape, (12, 12))
        self._check(a, b)

    def test_coprime_3x5(self):
        # logical 30x30: LCM(3,5)=15 → 2 unified blocks of 15x15 (very few, very large)
        a = _sparse_pair_2d(N=10, B_o=3, B_i=3, key_idx=1)
        b = _sparse_pair_2d(N=6,  B_o=5, B_i=5, key_idx=2)
        self.assertEqual(a.shape, (30, 30))
        self.assertEqual(b.shape, (30, 30))
        self._check(a, b)

    # --- divisor → many small blocks at the larger size -----------------
    def test_divisor_2x4(self):
        # logical 16x16: LCM(2,4)=4 → 4 unified blocks of 4x4 (many small-ish)
        a = _sparse_pair_2d(N=8, B_o=2, B_i=2, key_idx=1)
        b = _sparse_pair_2d(N=4, B_o=4, B_i=4, key_idx=2)
        self.assertEqual(a.shape, (16, 16))
        self.assertEqual(b.shape, (16, 16))
        self._check(a, b)

    def test_divisor_3x9(self):
        # logical 18x18: LCM(3,9)=9 → 2 unified blocks of 9x9
        a = _sparse_pair_2d(N=6, B_o=3, B_i=3, key_idx=1)
        b = _sparse_pair_2d(N=2, B_o=9, B_i=9, key_idx=2)
        self.assertEqual(a.shape, (18, 18))
        self.assertEqual(b.shape, (18, 18))
        self._check(a, b)

    # --- shared factor (neither coprime nor divisor) --------------------
    def test_shared_factor_4x6(self):
        # logical 24x24: LCM(4,6)=12 → 2 unified blocks of 12x12
        a = _sparse_pair_2d(N=6, B_o=4, B_i=4, key_idx=1)
        b = _sparse_pair_2d(N=4, B_o=6, B_i=6, key_idx=2)
        self.assertEqual(a.shape, (24, 24))
        self.assertEqual(b.shape, (24, 24))
        self._check(a, b)

    def test_shared_factor_6x9(self):
        # logical 18x18: LCM(6,9)=18 → 1 unified block of 18x18 (the entire grid)
        a = _sparse_pair_2d(N=3, B_o=6, B_i=6, key_idx=1)
        b = _sparse_pair_2d(N=2, B_o=9, B_i=9, key_idx=2)
        self.assertEqual(a.shape, (18, 18))
        self.assertEqual(b.shape, (18, 18))
        self._check(a, b)

    # --- asymmetric: outer block ≠ inner block -------------------------
    def test_asymmetric_blocks_coprime(self):
        # lhs blocks 2x3, rhs blocks 4x5 → LCM(2,4)=4 outer, LCM(3,5)=15 inner
        # logical 24x60: lhs N=12 (12*2=24, 12*5=60? no...) Need shapes to match.
        # lhs: outer=12*B_o, inner=12*B_i must match rhs: outer=N_r*B_o', inner=N_r*B_i'
        # Choose N_l=12, B_lo=2, B_li=3 → 24x36; N_r=6, B_ro=4, B_ri=6 → 24x36 ✓
        a = SparseTensor(
            (SparseIndex(0, 12, axis=0, other_id=1, block_size=2, block_axis=1),),
            (SparseIndex(1, 12, axis=0, other_id=0, block_size=3, block_axis=2),),
            _n((12, 2, 3), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, 6, axis=0, other_id=1, block_size=4, block_axis=1),),
            (SparseIndex(1, 6, axis=0, other_id=0, block_size=6, block_axis=2),),
            _n((6, 4, 6), 2),
        )
        self.assertEqual(a.shape, (24, 36))
        self.assertEqual(b.shape, (24, 36))
        self._check(a, b)


# ============================================================================
# Elementwise — intersection mode (mul with is_intersection=True)
# ============================================================================
class TestElementwiseIntersectionMisalignedBlocks(unittest.TestCase):
    """``mul`` with ``is_intersection=True``: the result is collapsed to the *minimum*
    block size on each side, so the output has more (smaller) blocks than the union path."""

    def _check(self, a, b, atol=1e-5):
        for use_jit in (False, True):
            with self.subTest(jit=use_jit):
                fn = jax.jit(lambda x, y: elementwise(x, y, jax.lax.mul, is_intersection=True)) \
                    if use_jit else (lambda x, y: elementwise(x, y, jax.lax.mul, is_intersection=True))
                got = fn(a, b)
                expected = a.dense() * b.dense()
                self.assertTrue(
                    jnp.allclose(got.dense(), expected, atol=atol),
                    f"mul-intersection mismatch (jit={use_jit}): max diff "
                    f"{float(jnp.max(jnp.abs(got.dense() - expected)))}",
                )
                self._assert_intersection_block_diag_output(got, a, b)

    def _assert_intersection_block_diag_output(self, got, a, b):
        """Intersection mode collapses to the *minimum* block size on each side
        (since multiplying by zero kills any cell where one source has fill).
        Output: a block-diagonal ``SparseTensor`` over ``M = total / min(B_a, B_b)``
        blocks of size ``(min_h, min_w)`` — strictly more (and smaller) blocks
        than the union path's LCM-grouped output, but still meta-block-diagonal."""
        ao, ai = a.out_dims[0], a.primal_dims[0]
        bo, bi = b.out_dims[0], b.primal_dims[0]
        min_h = min(ao.block_size, bo.block_size)
        min_w = min(ai.block_size, bi.block_size)
        # Total = M·min_h ⇒ M = ao.size·ao.block_size / min_h
        M = (ao.size * ao.block_size) // min_h
        out, primal = got.out_dims[0], got.primal_dims[0]
        self.assertIsInstance(out, SparseIndex)
        self.assertIsInstance(primal, SparseIndex)
        self.assertEqual(out.size, M)
        self.assertEqual(primal.size, M)
        self.assertEqual(out.block_size, min_h,
                         f"out_dim block_size {out.block_size} ≠ min_h {min_h}")
        self.assertEqual(primal.block_size, min_w,
                         f"primal_dim block_size {primal.block_size} ≠ min_w {min_w}")
        # M× compression vs the (M·min_h)·(M·min_w) fully-dense form.
        self.assertEqual(got.val.size, M * min_h * min_w)

    def test_intersection_coprime_2x3(self):
        a = _sparse_pair_2d(N=6, B_o=2, B_i=2, key_idx=1)
        b = _sparse_pair_2d(N=4, B_o=3, B_i=3, key_idx=2)
        self._check(a, b)

    def test_intersection_coprime_3x5(self):
        a = _sparse_pair_2d(N=10, B_o=3, B_i=3, key_idx=1)
        b = _sparse_pair_2d(N=6,  B_o=5, B_i=5, key_idx=2)
        self._check(a, b)

    def test_intersection_divisor_2x4(self):
        a = _sparse_pair_2d(N=8, B_o=2, B_i=2, key_idx=1)
        b = _sparse_pair_2d(N=4, B_o=4, B_i=4, key_idx=2)
        self._check(a, b)

    def test_intersection_shared_factor_4x6(self):
        a = _sparse_pair_2d(N=6, B_o=4, B_i=4, key_idx=1)
        b = _sparse_pair_2d(N=4, B_o=6, B_i=6, key_idx=2)
        self._check(a, b)


# ============================================================================
# Matmul — 2D / 3D / 4D with misaligned contraction blocks
# ============================================================================
class TestMatmulMisalignedBlocks(unittest.TestCase):
    """For each test, ``a @ b`` where ``a``'s primal-side block(s) and ``b``'s out-side
    block(s) deliberately disagree. The contraction's logical size matches; the
    algorithm reconciles via per-axis LCM/GCD tiling."""

    def _check(self, a, b, atol=1e-4):
        for use_jit in (False, True):
            with self.subTest(jit=use_jit):
                fn = jax.jit(matmul) if use_jit else matmul
                got = fn(a, b)
                expected = jnp.matmul(a.dense(), b.dense())
                self.assertTrue(
                    jnp.allclose(got.dense(), expected, atol=atol),
                    f"matmul mismatch (jit={use_jit}): max diff "
                    f"{float(jnp.max(jnp.abs(got.dense() - expected)))}",
                )

    # --- 2D matmul: sparse-pair LHS, dense RHS, coprime contracting blocks ---
    def test_2d_coprime_2x3_contract(self):
        # lhs is 12x12 with sparse pair (N=6, B=2/2 — outer/inner both 2)
        # rhs is 12x6 dense
        a = _sparse_pair_2d(N=6, B_o=2, B_i=2, key_idx=1)
        b = SparseTensor(
            (DenseIndex(0, 12, 0),), (DenseIndex(1, 6, 1),),
            _n((12, 6), 2),
        )
        self._check(a, b)

    def test_2d_coprime_3x5_contract(self):
        # Bigger coprime: LCM(3,5)=15 along the contracting axis
        a = SparseTensor(
            (SparseIndex(0, 10, axis=0, other_id=1, block_size=3, block_axis=1),),
            (SparseIndex(1, 10, axis=0, other_id=0, block_size=3, block_axis=2),),
            _n((10, 3, 3), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, 6, axis=0, other_id=1, block_size=5, block_axis=1),),
            (SparseIndex(1, 6, axis=0, other_id=0, block_size=5, block_axis=2),),
            _n((6, 5, 5), 2),
        )
        # logical: a is 30x30, b is 30x30 → result 30x30
        self._check(a, b)

    def test_2d_divisor_2x4_contract(self):
        # LCM(2,4)=4 — many small blocks
        a = _sparse_pair_2d(N=8, B_o=2, B_i=2, key_idx=1)
        b = _sparse_pair_2d(N=4, B_o=4, B_i=4, key_idx=2)
        # both 16x16 → 16x16 result
        self._check(a, b)

    def test_2d_shared_factor_4x6_contract(self):
        a = _sparse_pair_2d(N=6, B_o=4, B_i=4, key_idx=1)
        b = _sparse_pair_2d(N=4, B_o=6, B_i=6, key_idx=2)
        # both 24x24 → 24x24
        self._check(a, b)

    # --- 3D matmul (one batch axis + sparse-pair contraction) ----------
    def test_3d_one_misalignment(self):
        # 3D batched matmul: shape (s1, K, K_out) @ (s1, K, K_out), contracting the K axis.
        # The contracting axis is sparse with misaligned block sizes (LCM(2,3)=6).
        s1 = 3   # batch
        a = SparseTensor(
            (
                DenseIndex(0, s1, 0),
                SparseIndex(1, 6, axis=1, other_id=2, block_size=2, block_axis=2),
            ),
            (SparseIndex(2, 6, axis=1, other_id=1, block_size=2, block_axis=3),),
            _n((s1, 6, 2, 2), 1),  # (s1, N, B_o, B_i)
        )
        b = SparseTensor(
            (
                DenseIndex(0, s1, 0),
                SparseIndex(1, 4, axis=1, other_id=2, block_size=3, block_axis=2),
            ),
            (SparseIndex(2, 4, axis=1, other_id=1, block_size=3, block_axis=3),),
            _n((s1, 4, 3, 3), 2),
        )
        # logical a: (s1, 12, 12), b: (s1, 12, 12)
        self._check(a, b)

    # --- 4D matmul with one misalignment on the contraction axis -------
    def test_4d_one_misalignment(self):
        # Batched 4D matmul: (B1, B2, M, K) @ (B1, B2, K, N), contracting K.
        s1, s2 = 2, 3  # outer batch dims
        a = SparseTensor(
            (
                DenseIndex(0, s1, 0),
                DenseIndex(1, s2, 1),
                SparseIndex(2, 6, axis=2, other_id=3, block_size=2, block_axis=3),
            ),
            (SparseIndex(3, 6, axis=2, other_id=2, block_size=2, block_axis=4),),
            _n((s1, s2, 6, 2, 2), 1),  # (s1, s2, N, B_o, B_i)
        )
        b = SparseTensor(
            (
                DenseIndex(0, s1, 0),
                DenseIndex(1, s2, 1),
                SparseIndex(2, 4, axis=2, other_id=3, block_size=3, block_axis=3),
            ),
            (SparseIndex(3, 4, axis=2, other_id=2, block_size=3, block_axis=4),),
            _n((s1, s2, 4, 3, 3), 2),
        )
        # logical a: (s1, s2, 12, 12), b: (s1, s2, 12, 12)
        self._check(a, b)

    # --- 4D matmul with TWO misalignments ------------------------------
    def test_4d_two_misalignments(self):
        # Two sparse pairs straddling out/primal in both operands; both pairs misaligned.
        # First pair: lhs B=2 vs rhs B=3 (coprime, LCM=6)
        # Second pair: lhs B=2 vs rhs B=4 (divisor, LCM=4)
        # Use dot_general for the reference (jnp.matmul wouldn't contract two axes).
        a = SparseTensor(
            (
                SparseIndex(0, 6, axis=0, other_id=2, block_size=2, block_axis=1),
                SparseIndex(1, 8, axis=2, other_id=3, block_size=2, block_axis=3),
            ),
            (
                SparseIndex(2, 6, axis=0, other_id=0, block_size=2, block_axis=4),
                SparseIndex(3, 8, axis=2, other_id=1, block_size=2, block_axis=5),
            ),
            _n((6, 2, 8, 2, 2, 2), 1),  # (N1, B_o1, N2, B_o2, B_i1, B_i2)
        )
        b = SparseTensor(
            (
                SparseIndex(0, 4, axis=0, other_id=2, block_size=3, block_axis=1),
                SparseIndex(1, 4, axis=2, other_id=3, block_size=4, block_axis=3),
            ),
            (
                SparseIndex(2, 4, axis=0, other_id=0, block_size=3, block_axis=4),
                SparseIndex(3, 4, axis=2, other_id=1, block_size=4, block_axis=5),
            ),
            _n((4, 3, 4, 4, 3, 4), 2),
        )
        # logical a: (12, 16, 12, 16), b: (12, 16, 12, 16); contract (a.primal=2,3 ↔ b.out=0,1).
        for use_jit in (False, True):
            with self.subTest(jit=use_jit):
                fn = jax.jit(matmul) if use_jit else matmul
                got = fn(a, b)
                expected = jax.lax.dot_general(
                    a.dense(), b.dense(),
                    (((2, 3), (0, 1)), ((), ())),
                )
                self.assertTrue(
                    jnp.allclose(got.dense(), expected, atol=1e-4),
                    f"matmul mismatch (jit={use_jit}): max diff "
                    f"{float(jnp.max(jnp.abs(got.dense() - expected)))}",
                )


# ============================================================================
# Round-trip via the structured pytrees: ``UnionBlocks`` / ``IntersectionBlocks``
# ============================================================================
class TestStructuredRoundTrip(unittest.TestCase):
    """A misaligned ``elementwise(a, b, op)`` should produce *the same* dense form
    as wrapping its inputs in the matching structured pytree and going through
    ``SparseTensor.from_compressed``. Two paths to the same tensor:

      1. Sparse: ``SparseTensor`` with sparse-pair dims + ``elementwise(...)``.
      2. Compressed: build the structured pytree (``UnionBlocks`` /
         ``IntersectionBlocks``), wrap with ``from_compressed`` — this yields a
         meta-block-diagonal ``SparseTensor`` with ``val`` of shape
         ``(M, LCM_h, LCM_w)`` and ``M·LCM_h·LCM_w`` floats of storage,
         instead of materializing the M× larger fully-dense form.

    The two paths must agree element-wise *and* on the storage footprint."""

    def _build_blocks(self, N, B_o, B_i, M_lcm, key_idx):
        """Reshape a flat ``(N, B_o, B_i)`` block list into the per-meta-block
        ``(M, n, B_o, B_i)`` layout that ``UnionBlocks`` / ``IntersectionBlocks``
        expect: ``M = total / LCM`` meta-blocks, each carrying ``n = N / M``
        of the original blocks on its own block-diagonal."""
        blocks = jr.normal(jr.PRNGKey(key_idx), (N, B_o, B_i), dtype=jnp.float32)
        n_per_meta = N // M_lcm
        return blocks.reshape(M_lcm, n_per_meta, B_o, B_i)

    def test_union_coprime_2x3_roundtrip(self):
        # logical 12x12: lhs N=6, B=2 ; rhs N=4, B=3 ; LCM=6 ; M=2
        N_a, B_a, N_b, B_b = 6, 2, 4, 3
        M, lcm = 2, 6

        lhs_blocks = self._build_blocks(N_a, B_a, B_a, M, 1)   # (2, 3, 2, 2)
        rhs_blocks = self._build_blocks(N_b, B_b, B_b, M, 2)   # (2, 2, 3, 3)
        ub = UnionBlocks(lhs=lhs_blocks, rhs=rhs_blocks,
                         fill_lhs=jnp.array(0.0, jnp.float32),
                         fill_rhs=jnp.array(0.0, jnp.float32),
                         op=jnp.add)
        # The compressed form's meta-block-diagonal storage:
        st_compressed = SparseTensor.from_compressed(ub)
        self.assertEqual(st_compressed.shape, (12, 12))
        self.assertEqual(st_compressed.val.shape, (M, lcm, lcm))
        self.assertEqual(st_compressed.val.size, M * lcm * lcm)
        self.assertIsInstance(st_compressed.out_dims[0], SparseIndex)
        self.assertEqual(st_compressed.out_dims[0].block_size, lcm)
        # Bit-exact dense form against the union pytree's own to_dense.
        self.assertTrue(jnp.allclose(st_compressed.dense(), ub.to_dense(), atol=1e-5))

    def test_intersection_divisor_2x4_roundtrip(self):
        # logical 16x16: lhs N=8, B=2 ; rhs N=4, B=4 ; LCM=4 ; M=4
        N_a, B_a, N_b, B_b = 8, 2, 4, 4
        M, lcm = 4, 4

        lhs_blocks = self._build_blocks(N_a, B_a, B_a, M, 1)   # (4, 2, 2, 2)
        rhs_blocks = self._build_blocks(N_b, B_b, B_b, M, 2)   # (4, 1, 4, 4)
        ib = IntersectionBlocks(lhs=lhs_blocks, rhs=rhs_blocks,
                                fill_lhs=jnp.array(0.0, jnp.float32),
                                fill_rhs=jnp.array(0.0, jnp.float32),
                                op=jnp.multiply)
        st_compressed = SparseTensor.from_compressed(ib)
        self.assertEqual(st_compressed.shape, (16, 16))
        self.assertEqual(st_compressed.val.shape, (M, lcm, lcm))
        self.assertTrue(jnp.allclose(st_compressed.dense(), ib.to_dense(), atol=1e-5))

    def test_compression_factor(self):
        """For ``M`` meta-blocks, ``from_compressed(meta_block_diagonal)`` saves M× of
        storage vs ``compressed_val.to_dense()``: ``M·H·W`` floats vs ``(M·H)·(M·W)
        = M²·H·W``. This is exactly the compression that lets misaligned outputs
        scale to large meta-counts."""
        for M, n_lhs, B_lhs, n_rhs, B_rhs in [
            (2, 11, 5, 5, 11),  # the user's canonical mismatched-block example
            (4, 3, 2, 2, 3),    # tighter coprime 2/3 with M=4
            (8, 6, 4, 4, 6),    # shared-factor 4/6 with M=8
        ]:
            with self.subTest(M=M, n_lhs=n_lhs, B_lhs=B_lhs, n_rhs=n_rhs, B_rhs=B_rhs):
                lhs = jr.normal(jr.PRNGKey(M),
                                (M, n_lhs, B_lhs, B_lhs), jnp.float32)
                rhs = jr.normal(jr.PRNGKey(M + 1),
                                (M, n_rhs, B_rhs, B_rhs), jnp.float32)
                ub = UnionBlocks(lhs=lhs, rhs=rhs,
                                 fill_lhs=jnp.array(0.0, jnp.float32),
                                 fill_rhs=jnp.array(0.0, jnp.float32),
                                 op=jnp.add)
                st = SparseTensor.from_compressed(ub)
                stored = st.val.size
                fully_dense = (M * n_lhs * B_lhs) * (M * n_lhs * B_lhs)
                self.assertEqual(stored * M, fully_dense,
                                 f"M={M}: meta-block-diag must give M× compression")


if __name__ == "__main__":
    unittest.main()
