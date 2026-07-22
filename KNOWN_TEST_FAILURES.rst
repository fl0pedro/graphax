Known pre-existing test failures (norm-removal line)
====================================================

These 13 tests fail on the ``norm-removal`` line and are carried into the
core-v2 merge **unchanged**. They are **not** introduced by the merge: the
failure set is byte-identical to ``norm-removal``'s, and all 103727 EXHAUSTIVE
subtests pass. They fall into two independent categories.

A. Fail-loud on a non-fitting Diag/Compress transform (12 tests)
----------------------------------------------------------------

``tests/elemental/test_approx_regression.py`` (attn/vit random-order +
partial-approx variants) and
``tests/elemental/test_gate_leak.py::test_approx_gate_survives_nested_cond``.

Root cause: commit ``52cbe44`` ("core(approx): fail LOUDLY when a transform does
not fit") makes ``_eliminate_vertex`` **raise** ``TRANSFORM DID NOT FIT``
(core.py) when a Diag/Compress action is structurally invalid for the edge it
lands on -- e.g. ``Diag pair (0,1) is not split across out/primal: both indices
are in out_dims``. The legacy behavior was to **silently skip** the action,
which desynced the two edges of a shared variable and made a no-op approximation
report ``cos=1.0``.

These tests apply oracle/random Diag/Compress actions **without pre-masking
their validity**, so they hit the raise. The intended discipline (per the raise
message) is to mask invalid actions up front (``diag_mask`` / ``compress_mask``)
rather than discover them by throwing; the tests predate that discipline.

Escape hatch (restores the legacy silent skip -- makes these pass but
reintroduces the shared-variable desync hazard the fail-loud guard was added to
prevent): ``GRAPHAX_BEST_EFFORT_TRANSFORMS=1``.

B. Quant sequential-application value mismatch (1 test)
-------------------------------------------------------

``tests/core/sparse_tensor/apply_quant_test.py::test_quant_sequential_last_wins``.

Applies ``[Quant('float16'), Quant('int8')]`` and asserts the resulting int8
values equal ``arange(16).reshape(4, 4)``. The output **dtype** is correct
(``int8``); only the ``np.testing.assert_array_equal`` on the **values** fails --
a sequential-quant (float16 -> int8) rounding/truncation discrepancy. Independent
of category A and of the merge.
