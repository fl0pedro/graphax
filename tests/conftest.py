"""Suite-wide configuration.

WHY THIS FILE PINS MATMUL PRECISION
===================================
Almost every test in this suite is a COMPARISON OF TWO ARRANGEMENTS OF THE SAME
CONTRACTIONS: the sparse engine's kernel against ``lhs.dense() @ rhs.dense()``,
``jacve`` against ``jax.jacrev`` / ``jax.grad``, the sparse output packing
against the dense one, one factoring of a block-diagonal axis against a coarser
one. The engine's claim is STRUCTURAL -- it accumulates the same Jacobian with
fewer flops and less storage -- so the oracle only works if a dot is a function
of its inputs at the accuracy of its input dtype.

On an Ampere-or-later GPU that is NOT true by default. XLA:GPU lowers an f32
``dot_general`` onto TF32 tensor cores: the inputs are truncated to a 10-bit
mantissa, i.e. ~2^-11 == 4.9e-4 relative. Two different arrangements of the
same mathematics then disagree at 1e-4 .. 6e-3 absolute on O(1) data --
MEASURED on an RTX 3090, 151 failures across 19 files, every one of them a
value mismatch in exactly that band, with the structural assertions (storage
counts, sparsity patterns, physical shapes, action censuses) all passing. The
disagreement is a property of the device's tensor cores, not of the engine.

Pinning ``highest`` makes the dot f32-accurate, so these tests measure the
algebra they were written to measure. The alternative -- widening 19 files of
tolerances to ~6e-3 -- would have thrown away three orders of magnitude of
resolution and, with it, the ability to SEE a real 1e-3 engine defect. That is
not hypothetical: ``_subdivide_coupled_blockdiag`` selected block-diagonal
sub-blocks through a CONTRACTING einsum, which lowers to a dot, so a step that
only MOVES numbers rounded them to ~8 mantissa bits (2026-09-09). A tolerance
wide enough to absorb TF32 would have hidden that defect permanently.

This is a TEST-SUITE decision and deliberately NOT a library default. Graphax
must not impose a precision/throughput tradeoff on its callers; a user who
wants TF32 speed gets it. What the suite needs is a trustworthy oracle.

The companion guard against the defect class this pin could otherwise mask
lives in ``tests/core/sparse_tensor/no_data_movement_dot_test.py``: it asserts
that the structural (number-moving) steps emit NO dot at all, which is a
statement about the emitted jaxpr and therefore independent of precision.
"""

import jax


# Set at conftest import, before any test module builds a jaxpr, so it applies
# uniformly to collection-time and run-time tracing. Deliberately in-repo
# rather than left to a ``JAX_DEFAULT_MATMUL_PRECISION`` in whatever launcher
# happens to run the suite: an invisible external setting that silently decides
# what the suite measures is precisely the failure mode that let a module-level
# ``jax_platform_name = "cpu"`` pin run 99.9% of this suite on the CPU while it
# reported itself green on the GPU (dsnn-3qm.72).
jax.config.update("jax_default_matmul_precision", "highest")
