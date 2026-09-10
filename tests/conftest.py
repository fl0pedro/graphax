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

WHY THE PIN IS A FIXTURE AND NOT A MODULE-LEVEL ``jax.config.update``
====================================================================
It used to be a bare ``jax.config.update`` at this module's top level. That is
PROCESS-GLOBAL and irreversible, and it is exactly the shape of the bug that
made the pin necessary in the first place: ``auto_primitive_test.py`` set
``jax_platform_name = "cpu"`` the same way, so 99.9% of this suite ran on the
CPU while reporting itself green on the GPU, for an unknown number of runs
(dsnn-3qm.72). A process-global write also cannot be scoped -- not to a
backend, not away from a test that must NOT have it -- and it leaks into
anything that imports the suite.

The autouse fixture below is equivalent where the pin is wanted and scoped
everywhere else:

* ON GPU ONLY. On a CPU (and on any backend with no tensor-core dot) an f32
  ``dot_general`` is already f32-accurate, so the pin buys nothing and the
  suite should exercise the stock configuration. MEASURED at 0a95874, the same
  128x256x64 f32 dot against the f64 answer: on an RTX 3090 2.96e-4 unpinned
  and 1.53e-7 pinned; on this machine's CPU 2.86e-7 BOTH pinned and unpinned.
  The pin is a no-op on the CPU, so it should not be on there.

* NOT on a test marked ``device_matmul_precision``. Those tests MEASURE what a
  narrow dtype costs, so the arithmetic they run on has to be the device's own;
  a suite-wide precision override is the one thing that must not be able to be
  confused with the quantity under test.

``precision=HIGHEST`` reaching the jaxpr is also what produced the
``jaxpr tokenizer: uncaptured value rendered as '?': 'HIGHEST'`` warning from
``jaxpr.py``: that tokenizer feeds the alphagrad search representation and its
vocabulary has no token for a precision, so a pin that is live during a
tokenizer test changes the string the tokenizer sees. Scoping the pin off the
CPU removes the warning from every CPU run.

WHAT THE PIN DOES **NOT** DO (measured, 2026-09-10)
===================================================
It does not undo a ``Quant`` -- the suspicion that motivated this scoping.
``highest`` changes how a dot ACCUMULATES; it cannot restore mantissa bits that
a cast already threw away. MEASURED on an RTX 3090, one 128x256x64 dot with the
lhs rounded to bfloat16, against the f64 answer:

    precision    bf16 x f32 vs exact f32 math   vs exact *bf16-rounded* math
    default      1.6662e-03                     2.0992e-04   (TF32 noise on top)
    highest      1.6555e-03                     1.5093e-07   (the cast, alone)

So the pin PRESERVES the full 1.66e-3 quantization effect and removes only the
2.1e-4 of tensor-core noise that was riding on it. A bf16 x bf16 dot keeps its
bfloat16 output dtype under every setting.

It is also NOT what makes the four bfloat16 Quant cases in
``tests/core/dense_edges_test.py`` fail. Those fail because XLA:GPU's optimizer
DELETES the dense engine's bf16 casts: for the reverse order with Quant on slot
lhs, ``jax.make_jaxpr`` shows all 10 ``convert_element_type[bfloat16]`` and the
pre-optimisation StableHLO shows 60 bf16 mentions, but the OPTIMIZED HLO shows
ZERO bf16 and a single fused ``dot``. The same program on XLA:CPU keeps 7 bf16
converts and 11 dots. So the dense value oracle does not execute the
approximation it reports, under ``jax.jit`` on a GPU -- run EAGER, the same plan
moves the gradient by 2.86e-3 and the two engines agree to 6e-8. Independent of
the precision setting: byte-identical numbers pinned and unpinned, and at the
pre-pin commit dc1aabc too.
"""

import jax
import pytest

# The backends whose f32 ``dot_general`` is not f32-accurate by default, i.e.
# the ones with tensor cores. ``jax.default_backend()`` reports "gpu" on some
# builds and the platform name on others, so both spellings are listed.
_TENSOR_CORE_BACKENDS = frozenset({"gpu", "cuda", "rocm"})

#: A test carrying this marker runs at the DEVICE's default matmul precision.
PRECISION_OPT_OUT = "device_matmul_precision"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{PRECISION_OPT_OUT}: run at the device's default matmul precision "
        "instead of the suite-wide 'highest' pin. For a test whose SUBJECT is "
        "a numeric approximation (a bfloat16 Quant), where a precision "
        "override must not be confusable with the quantity being measured.",
    )


def _backend_needs_the_pin() -> bool:
    try:
        return jax.default_backend() in _TENSOR_CORE_BACKENDS
    except Exception:                                           # noqa: BLE001
        # No backend yet / a backend that failed to initialise. Either way there
        # is nothing to pin, and a conftest must not be the thing that raises.
        return False


@pytest.fixture(autouse=True)
def matmul_precision(request):
    """Pin ``highest`` for the duration of ONE test, on GPU, unless opted out.

    ``jax.default_matmul_precision`` is a context manager, so the setting is
    unwound when the test ends -- nothing leaks into the next test, into a
    later session, or into a process that merely imports this suite.

    The jit cache IS keyed on the precision, so a function compiled outside the
    context is recompiled inside it rather than silently reused -- VERIFIED on
    an RTX 3090: one jitted 128x256x64 f32 dot compiled at the ambient setting
    reads 2.96e-4 from the f64 answer, the SAME jitted callable called inside
    this context reads 1.53e-7, and outside again 2.96e-4. No
    ``jax.clear_caches()`` is needed.
    """
    if not _backend_needs_the_pin():
        yield
        return
    if request.node.get_closest_marker(PRECISION_OPT_OUT) is not None:
        yield
        return
    with jax.default_matmul_precision("highest"):
        yield
