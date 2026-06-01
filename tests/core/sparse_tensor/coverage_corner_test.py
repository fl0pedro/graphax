"""Corner-case tests targeting algorithmic branches that the rest of the suite misses.

Specifically:

* ``batch_sparse`` arm of ``ops.matmul._matched_pair`` — fires when both operands carry
  the same sparse pair (same dim ids on both sides) at non-contracting positions, so
  the pair survives the contract step and is then resolved by ``_resolve_broadcast_topos``.
  Standard tests don't reach this because the contract step takes the *last* ``n_contract``
  dims of ``lhs.primal`` / ``rhs.out``, which on most natural inputs swallows any sparse
  pair that's positioned to be id-matched. We construct an explicit non-straddle layout
  (both pair members in ``out_dims`` on each side) to keep the pair out of the contract.

* ``batch_primal`` is its primal-side mirror — same structural idea, both pair members in
  ``primal_dims`` on each side.
"""
import unittest

import jax
import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.tensor import SparseTensor
from graphax.sparse.indexes import DenseIndex, SparseIndex
from graphax.sparse.ops.matmul import matmul


def _n(shape, key_idx, dtype=jnp.float32):
    return jr.normal(jr.PRNGKey(key_idx), shape).astype(dtype)


class TestMatmulBatchPrimalKnownLimitation(unittest.TestCase):
    """``batch_primal`` matched-pair arm (matmul.py:129-133) — would fire when an
    id-matched dim survives the contract on both sides AND is in ``primal_dims`` on
    both. The arm is structurally reachable, but constructing such an input crashes
    in ``_prepare_contraction_views`` (the tiled core's val-reshape assumes that
    surviving primals come from sparse-pair siblings, not from id-matched batches).

    Marking this as expected-fail documents the limitation alongside the test rather
    than letting line 129-133 sit silently uncovered.
    """

    @unittest.expectedFailure
    def test_batch_primal_dense(self):
        K, B = 4, 5
        a = SparseTensor(
            (DenseIndex(0, 3, axis=0),),
            (DenseIndex(1, B, axis=1), DenseIndex(2, K, axis=2)),
            _n((3, B, K), 1),
        )
        b = SparseTensor(
            (DenseIndex(0, K, axis=0),),
            (DenseIndex(1, B, axis=1),),
            _n((K, B), 2),
        )
        # If/when the tiled core gains batch_primal handling, this should compute
        # einsum('ibk,kb->ib', a.dense(), b.dense()).
        ref_dense = jnp.einsum('ibk,kb->ib', a.dense(), b.dense())
        got = matmul(a, b)
        self.assertTrue(jnp.allclose(got.dense(), ref_dense, atol=1e-4))


class TestMatmulBatchedSparsePair(unittest.TestCase):
    """``batch_sparse`` matched-pair arm — both operands carry the same sparse pair on the
    non-contracting side, and the contract eats only a separate dense axis."""

    def test_batch_sparse_pair_in_out(self):
        # lhs.out has the sparse pair (ids 0,1) PLUS a non-contracting DenseDim(2);
        # lhs.primal has the contracting DenseDim(3). rhs mirrors the structure with the
        # contracting dim on the rhs.out side. The pair (0,1) survives the contract step
        # and gets matched id-to-id by `_resolve_broadcast_topos` → fires `batch_sparse`.
        N, A, K, B = 3, 2, 4, 5
        a = SparseTensor(
            (
                SparseIndex(0, N, axis=0, other_id=1),
                SparseIndex(1, N, axis=0, other_id=0),
                DenseIndex(2, A, axis=1),
            ),
            (DenseIndex(3, K, axis=2),),
            _n((N, A, K), 1),
            check_consistency=False,  # both pair members in out_dims is unusual but valid
        )
        b = SparseTensor(
            (
                SparseIndex(0, N, axis=0, other_id=1),
                SparseIndex(1, N, axis=0, other_id=0),
                DenseIndex(2, K, axis=1),  # contracts with a.primal DenseDim(3)
            ),
            (DenseIndex(3, B, axis=2),),
            _n((N, K, B), 2),
            check_consistency=False,
        )
        # logical: a (N, N, A, K), b (N, N, K, B). jnp.matmul gives (N, N, A, B), but a
        # SparseTensor matmul output has a `batch_sparse` pair straddling out/primal,
        # so its logical layout is (N, A, N, B) instead — same data, axes 1↔2 swapped.
        # Only the no-JIT (tiled) path exercises `batch_sparse`; under JIT the densify
        # fallback fires and produces a different (also-correct) all-dense topology.
        ref_dense = jnp.transpose(jnp.matmul(a.dense(), b.dense()), (0, 2, 1, 3))
        got = matmul(a, b)
        self.assertTrue(
            jnp.allclose(got.dense(), ref_dense, atol=1e-4),
            f"batch_sparse mismatch: max diff "
            f"{float(jnp.max(jnp.abs(got.dense() - ref_dense)))}",
        )
        # Sanity-check that the result actually carries a SparseIndex pair (the proof
        # that the `batch_sparse` arm built the output and not, say, a `spatial_*` arm).
        sparse_dims_out = [d for d in got.dims if d.is_sparse]
        self.assertEqual(len(sparse_dims_out), 2, "expected one sparse pair in output")


class TestTiledIndexNoPurePermutation(unittest.TestCase):
    """Verifies that ``_tiled_index`` is either identity (for divisor cases) or has
    collisions (for misaligned cases). It NEVER produces a "pure permutation but not
    identity" output — so the corresponding optimization branch in ``_reduce_grid``
    that previously sat alongside the segment_sum fallback was dead code (now removed).

    If someone reworks ``_tiled_index`` such that this assumption no longer holds,
    this test will fail and tell them: re-introduce the pure-permutation fast path
    in ``_reduce_grid``."""

    def test_no_pure_permutation_across_configurations(self):
        import math
        import numpy as np
        from graphax.sparse.ops.matmul import _tiled_index, Pair, PairData

        configs = [
            (2, 3), (3, 5), (2, 4), (4, 6), (6, 9), (6, 4), (8, 12),
            (3, 4), (4, 9), (5, 7), (9, 12), (8, 6), (15, 10),
            (10, 14), (12, 18), (16, 24), (20, 30),
        ]
        for a, b in configs:
            with self.subTest(a=a, b=b):
                gcd, lcm = math.gcd(a, b), math.lcm(a, b)
                p = Pair("contract", 1,
                         PairData(a, 1, 1, 0, None, None),
                         PairData(b, 1, 1, 0, None, None))
                idx, _ = _tiled_index(p, gcd, lcm)
                is_identity = np.array_equal(idx, np.arange(len(idx)))
                is_unique = len(np.unique(idx)) == len(idx)
                # Either identity (divisor case) or has collisions — never a pure permutation.
                self.assertFalse(
                    is_unique and not is_identity,
                    f"_tiled_index({a}, {b}) is a pure permutation (not identity, no "
                    f"collisions) — re-introduce the fast path in _reduce_grid."
                )


class TestZeroFillFlagSurvivesJit(unittest.TestCase):
    """Regression guard for a perf bug that caused 1000–1,000,000× slowdowns in
    every jit'd matmul: when ``_is_zero_fill`` was forced to read the tracer
    fill_value at trace time, ``np.asarray(tracer)`` raised, the helper
    fell back to ``False``, and the matmul rerouted through the
    densify-then-dot_general fallback even for the canonical zero-fill case.

    The fix caches the flag at SparseTensor construction (when fill_value is
    still concrete) and propagates it through the pytree's static aux_data
    so it survives jit. These tests pin that behavior:

    1. The flag is statically ``True`` for default-construction zero fills.
    2. The flag round-trips through ``jit(identity)`` (i.e. ``tree_unflatten``
       restores it from aux_data).
    3. A jit'd ``matmul`` of two zero-fill operands compiles to small HLO
       (proxy for "stayed on the tiled path, didn't materialize a dense
       intermediate") — the densify fallback would balloon the HLO with a
       dense ``dot_general`` over densified operands.
    """

    def _zero_fill_tensor(self, shape, key_idx, fill_value=None):
        return SparseTensor(
            (DenseIndex(0, shape[0], axis=0),
             SparseIndex(1, shape[1], axis=1, other_id=2)),
            (SparseIndex(2, shape[1], axis=1, other_id=1),),
            _n(shape, key_idx),
            **({"fill_value": fill_value} if fill_value is not None else {}),
        )

    def test_default_fill_is_static_zero(self):
        """Default ``fill_value=None`` ⇒ ``jnp.array(0)`` ⇒ flag must be True."""
        t = self._zero_fill_tensor((4, 6), 1)
        self.assertTrue(t._zero_fill,
                        "default-constructed SparseTensor must have _zero_fill=True")

    def test_explicit_nonzero_fill_is_static_false(self):
        """Concretely non-zero fill ⇒ flag must be False (forces densify path)."""
        t = self._zero_fill_tensor((4, 6), 1,
                                   fill_value=jnp.array(0.5, dtype=jnp.float32))
        self.assertFalse(t._zero_fill)

    def test_flag_survives_jit_identity(self):
        """``tree_unflatten`` after a jit must restore ``_zero_fill``. If aux_data
        loses it, downstream ops downgrade silently to the densify path."""
        t = self._zero_fill_tensor((4, 6), 1)
        t2 = jax.jit(lambda x: x)(t)
        self.assertTrue(t2._zero_fill)

    def test_jitted_matmul_uses_tiled_path(self):
        """If the densify fallback fires, HLO carries a full ``dot`` over
        ``densify(lhs) × densify(rhs)`` and grows by an order of magnitude.
        The tiled path keeps it tight. We assert the HLO is well under the
        densify-fallback footprint as a coarse but reliable signal."""
        a = self._zero_fill_tensor((4, 6), 1)
        b = self._zero_fill_tensor((4, 6), 2)

        @jax.jit
        def f(x, y):
            return matmul(x, y).val

        hlo = f.lower(a, b).compile().as_text()
        # Densify fallback for these sizes lands at ~10–11k chars of HLO; the
        # tiled path lands well under 6k. The 8k cutoff is a safe middle.
        self.assertLess(
            len(hlo), 8000,
            f"HLO is {len(hlo)} chars — densify fallback likely fired again. "
            f"Check that ``_is_zero_fill`` reads the cached static flag.",
        )


class TestCompressedValStorage(unittest.TestCase):
    """SparseTensor can hold a structured pytree from ``ops.block_storage``
    (UnionBlocks / IntersectionBlocks / BlockBanded) in ``compressed_val``
    instead of a dense ``val``. Operations that need the dense form call
    ``compressed_val.to_dense()`` — a gather-free broadcast+select(+sum)
    chain that XLA folds into the consuming kernel (SMEM, no HBM).

    These tests pin three properties:

    1. SparseTensor accepts compressed storage and round-trips through jit
       without materializing in HBM.
    2. ``tensor.dense()`` returns the same numbers as the equivalent
       dense-val tensor.
    3. ``matmul`` and ``elementwise`` materialize compressed inputs as a
       *traced* JAX expression — XLA fuses the densify into the consumer.
    """

    def _bb_pair(self, M, B, w):
        """Build a ``(SparseTensor with BlockBanded compressed_val,
        equivalent dense SparseTensor)`` pair for testing."""
        from graphax.sparse.ops.block_storage import BlockBanded

        W = 2 * w + 1
        data = _n((M, W, B, B), 7)
        bb = BlockBanded(data=data, fill_value=jnp.array(0.0, dtype=jnp.float32))
        # User-facing API: ``SparseTensor.from_compressed`` wraps a structured
        # pytree as a SparseTensor with ``compressed_val`` storage in one call.
        compressed = SparseTensor.from_compressed(bb)
        plain = SparseTensor(
            (DenseIndex(0, M * B, axis=0),),
            (DenseIndex(1, M * B, axis=1),),
            val=bb.to_dense(),
            fill_value=jnp.array(0.0, dtype=jnp.float32),
        )
        return compressed, plain

    def test_construct_with_compressed_val_and_dense(self):
        """``compressed_val`` storage round-trips to the same dense form as
        the equivalent uncompressed ``SparseTensor``."""
        c, p = self._bb_pair(M=3, B=4, w=1)
        self.assertIsNone(c.val)
        self.assertIsNotNone(c.compressed_val)
        self.assertTrue(jnp.allclose(c.dense(), p.dense(), atol=1e-5))

    def test_compressed_val_survives_jit(self):
        """Pytree round-trip via ``jit(identity)`` preserves the compressed
        layout — no early HBM materialization."""
        c, _ = self._bb_pair(M=3, B=4, w=1)
        out = jax.jit(lambda x: x)(c)
        self.assertIsInstance(out, SparseTensor)
        self.assertIsNone(out.val)
        self.assertIsNotNone(out.compressed_val)

    def test_matmul_consumes_compressed_input(self):
        """A jit'd matmul of (compressed, dense) gives the same result as
        (dense, dense). The compressed input is materialized inline via the
        gather-free ``BlockBanded.to_dense`` expression — XLA folds it
        forward into the matmul kernel."""
        c, p = self._bb_pair(M=3, B=4, w=1)
        # Build a dense rhs to multiply against.
        rhs = SparseTensor(
            (DenseIndex(0, 3 * 4, axis=0),),
            (DenseIndex(1, 5, axis=1),),
            val=_n((3 * 4, 5), 8),
            fill_value=jnp.array(0.0, dtype=jnp.float32),
        )

        @jax.jit
        def f(a, b):
            return matmul(a, b).dense()

        got_compressed = f(c, rhs)
        got_dense = f(p, rhs)
        self.assertTrue(jnp.allclose(got_compressed, got_dense, atol=1e-4),
                        f"max diff: {float(jnp.max(jnp.abs(got_compressed - got_dense))):.3e}")

    def test_elementwise_consumes_compressed_input(self):
        """Elementwise add of (compressed, dense) matches (dense, dense)."""
        from graphax.sparse.ops.elementwise import elementwise

        c, p = self._bb_pair(M=3, B=4, w=1)
        # Same shape; build with regular val.
        other = SparseTensor(
            c.out_dims, c.primal_dims,
            val=_n((3 * 4, 3 * 4), 9),
            fill_value=jnp.array(0.0, dtype=jnp.float32),
        )

        @jax.jit
        def f(a, b):
            return elementwise(a, b, jnp.add).dense()

        got_compressed = f(c, other)
        got_dense = f(p, other)
        self.assertTrue(jnp.allclose(got_compressed, got_dense, atol=1e-4),
                        f"max diff: {float(jnp.max(jnp.abs(got_compressed - got_dense))):.3e}")

    def test_compressed_uses_no_scatter_under_jit(self):
        """The whole pipeline (compressed storage → matmul → dense output)
        compiles to scatter-free HLO — the densify is broadcast+select+sum
        and the matmul is a plain ``dot_general``."""
        c, _ = self._bb_pair(M=3, B=4, w=1)
        rhs = SparseTensor(
            (DenseIndex(0, 3 * 4, axis=0),),
            (DenseIndex(1, 5, axis=1),),
            val=_n((3 * 4, 5), 8),
            fill_value=jnp.array(0.0, dtype=jnp.float32),
        )

        @jax.jit
        def f(a, b):
            return matmul(a, b).dense()

        hlo = f.lower(c, rhs).compile().as_text()
        scatters = sum(1 for l in hlo.split("\n") if " scatter(" in l.lower())
        self.assertEqual(scatters, 0, "compressed-val pipeline must be scatter-free")


if __name__ == "__main__":
    unittest.main()
