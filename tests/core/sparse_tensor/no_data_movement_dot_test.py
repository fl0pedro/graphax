"""A step that only MOVES numbers must not emit a dot.

WHY THIS EXISTS
===============
``tests/conftest.py`` pins ``jax_default_matmul_precision = "highest"`` so the
suite's dense oracles are f32-accurate and the value comparisons measure the
engine's algebra instead of the GPU's tensor cores. That pin is necessary, but
on its own it would MASK a specific defect class: a structural step -- a
selection, a placement, a mask, a relabel -- implemented as a CONTRACTING
einsum. Such an einsum lowers to ``dot_general``, and a dot on a GPU runs at
the device's matmul precision, so a step that is supposed to be exact silently
rounds its data.

This is not hypothetical. ``_subdivide_coupled_blockdiag`` selected the
meta-diagonal sub-blocks by contracting two identity matrices against the
buffer (``gi,gj,nirjc...->ngrc...``). On an RTX 3090 the selected values came
back rounded to about 8 mantissa bits -- MEASURED -0.15441894 for -0.15443718
-- on a step whose entire job is to copy them. Its dual
``_coarsen_coupled_blockdiag`` placed blocks the same way. Both now use a
static index / broadcast multiply.

The assertion here is about the EMITTED JAXPR, not about values, so it holds at
every precision and on every backend: whatever the tolerance elsewhere, these
steps must contain no dot. A value test cannot replace it -- at ``highest`` the
rounding disappears, and at TF32 it is indistinguishable from the tensor-core
noise of the genuine contractions around it.

WHAT IS *NOT* CLAIMED
=====================
Nothing here says the engine is dot-free. ``contract_B_B`` / ``contract_D_B``
and the matmul kernels SHOULD emit dots: they contract. The claim is narrower
and is the one that matters: a refactoring of a block-diagonal axis MOVES stored
numbers (and, when refining, DROPS the ones off the finer diagonal). It never
combines two numbers into one, so its cost is a copy and its arithmetic is exact
-- which is why a dot has no business appearing in it.

Note the two directions are not symmetric, and the value tests below say so:
coarsening preserves the dense form bit for bit, subdividing masks it. Getting
that backwards was this file's own first mistake.
"""
import unittest

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from graphax.sparse.tensor import (
    SparseTensor, _coarsen_coupled_blockdiag, _subdivide_coupled_blockdiag)
from graphax.sparse.indexes import DenseIndex, DiagonalIndex


def _all_primitives(jaxpr):
    """Every primitive name in ``jaxpr``, descending into nested jaxprs.

    ``pjit`` / ``closed_call`` / ``custom_vjp`` hide their body in params, so a
    flat walk over ``jaxpr.eqns`` can report 'no dot' for a computation that
    contains one.
    """
    names = []
    for eqn in jaxpr.eqns:
        names.append(eqn.primitive.name)
        for param in eqn.params.values():
            for sub in _subjaxprs(param):
                names.extend(_all_primitives(sub))
    return names


def _subjaxprs(param):
    """Jaxprs reachable from an eqn param.

    Duck-typed on purpose: the concrete classes moved from ``jax.core`` to
    ``jax.extend.core`` (they are absent from ``jax.core`` as of jax 0.10), so
    an ``isinstance`` against either location silently stops descending on the
    other and turns this guard into a no-op that always passes.
    """
    if hasattr(param, "eqns"):                 # Jaxpr
        return [param]
    if hasattr(param, "jaxpr"):                # ClosedJaxpr
        return _subjaxprs(param.jaxpr)
    if isinstance(param, (tuple, list)):
        out = []
        for p in param:
            out.extend(_subjaxprs(p))
        return out
    return []


DOTS = {"dot_general", "dot"}


def _n(shape, key=0):
    return jr.normal(jr.PRNGKey(key), shape).astype(jnp.float32)


# Both coupled block axes MATERIALISED (block_axis set on each side) is the
# branch that carried the eye-einsum, so it is the branch under test here.
# meta N=2, blocks 8 and 128: k=2 divides both, so factor=4 is legal.
def _block2(extra=6):
    od = (DiagonalIndex(0, 2, 0, 2, 8, 1), DenseIndex(1, 10, 2))
    pd = (DiagonalIndex(2, 2, 0, 0, 128, 3), DenseIndex(3, extra, 4))
    return SparseTensor(od, pd, _n((2, 8, 10, 128, extra), 2),
                        check_consistency=False)


def _diag16(extra=10):
    od = (DiagonalIndex(0, 16, 0, 2, None, None), DenseIndex(1, extra, 1))
    pd = (DiagonalIndex(2, 16, 0, 0, None, None), DenseIndex(3, extra, 2))
    return SparseTensor(od, pd, _n((16, extra, extra), 1),
                        check_consistency=False)


class TestNoDataMovementDot(unittest.TestCase):
    def _assert_dot_free(self, rebuild, val, what):
        """``rebuild(val)`` runs the refactor and returns the new buffer."""
        jaxpr = jax.make_jaxpr(rebuild)(val).jaxpr
        prims = _all_primitives(jaxpr)
        found = sorted(DOTS.intersection(prims))
        self.assertEqual(
            found, [],
            f"{what} emitted {found} -- a step that only moves numbers must "
            f"not contract. A dot runs at the device matmul precision (TF32 on "
            f"Ampere, ~10 mantissa bits), so this would silently round data it "
            f"is only supposed to copy. Use a static index or a broadcast "
            f"multiply. Full primitive list: {sorted(set(prims))}")

    def test_subdivide_emits_no_dot(self):
        t = _block2()

        def rebuild(val):
            st = SparseTensor(t.out_dims, t.primal_dims, val,
                              check_consistency=False)
            out = _subdivide_coupled_blockdiag(
                st, True, 0, st.out_dims[0], False, 0, st.primal_dims[0], 4)
            return out.val

        self._assert_dot_free(rebuild, t.val, "_subdivide_coupled_blockdiag")

    def test_coarsen_emits_no_dot(self):
        t = _diag16()

        def rebuild(val):
            st = SparseTensor(t.out_dims, t.primal_dims, val,
                              check_consistency=False)
            out = _coarsen_coupled_blockdiag(
                st, True, 0, st.out_dims[0], False, 0, st.primal_dims[0], 2)
            return out.val

        self._assert_dot_free(rebuild, t.val, "_coarsen_coupled_blockdiag")

    # The exactness claim the dot-freedom buys, asserted directly. Note the two
    # directions have DIFFERENT contracts, and conflating them is easy:
    #
    #   * COARSENING is lossless. A coarser meta block holds its finer blocks on
    #     the sub-diagonal and the new off-sub-diagonal positions are EXPLICIT
    #     stored zeros, so the dense form is unchanged, bit for bit.
    #   * SUBDIVIDING is a REFINEMENT. It imposes a FINER block-diagonal mask, so
    #     entries off the finer diagonal are dropped. The dense form is NOT
    #     preserved -- MEASURED max diff 4.55 on the tensor below, which is the
    #     mask doing its job, not an error. The contract (per the docstring) is
    #     that it equals the dense form MASKED to ``factor`` blocks, byte for
    #     byte -- no arithmetic, so no rounding.
    #
    # Asserting bit-identity of the dense form for BOTH was this file's own first
    # mistake; the subdivide oracle below is the correct statement.
    def test_subdivide_equals_the_masked_dense_form_exactly(self):
        t = _block2()
        before = np.asarray(t.dense())          # (N*b1, 10, N*b2, extra)
        factor = 4
        out = _subdivide_coupled_blockdiag(
            t, True, 0, t.out_dims[0], False, 0, t.primal_dims[0], factor)
        after = np.asarray(out.dense())
        self.assertEqual(after.shape, before.shape)

        # Keep [i, :, j, :] iff i and j land in the same one of ``factor``
        # meta-diagonal blocks. The refinement is a subset of the mask the
        # tensor already carried, so this only ever removes.
        rows, cols = before.shape[0], before.shape[2]
        i = np.arange(rows)[:, None]
        j = np.arange(cols)[None, :]
        keep = (i // (rows // factor)) == (j // (cols // factor))
        oracle = before * keep[:, None, :, None]

        self.assertEqual(np.abs(after - oracle).max(), 0.0,
                         "subdivide must equal the dense form masked to "
                         "`factor` meta-diagonal blocks, EXACTLY -- any nonzero "
                         "difference means the refinement did arithmetic")
        # And it really did drop something, so the oracle is not trivially equal.
        self.assertGreater(np.abs(before - after).max(), 0.0)

    def test_coarsen_is_bit_identical_in_the_dense_form(self):
        """Coarsening only ADDS explicit zeros, so the dense form is untouched.

        (``coarsen_blockdiag_test.py`` covers this more widely; it is repeated
        here as the contrast that makes the subdivide contract above readable.)
        """
        t = _diag16()
        before = np.asarray(t.dense())
        out = _coarsen_coupled_blockdiag(
            t, True, 0, t.out_dims[0], False, 0, t.primal_dims[0], 2)
        after = np.asarray(out.dense())
        self.assertEqual(np.abs(before - after).max(), 0.0)


if __name__ == "__main__":
    unittest.main()
