"""Bug: a rank-0 operand in ``matmul`` on the tiled (incumbent) engine
returned a size-1 slice of the tensor operand instead of the scaled tensor.

``X @ scalar`` (a SparseTensor with real dims contracted against a true
0-rank scalar SparseTensor) has no shared dimension to contract: the
mathematically-correct result is ``scalar * X`` (owner ruling, dsnn-3qm.68,
2026-09-06: X @ scalar == scalar * X, always, on every engine).

Before the fix, ``_build_pair_dims`` (src/graphax/sparse/ops/matmul.py) built
the ``spatial_primal_lhs`` pairing's one-sided primal dim by reading the
extent/axis from the LHS half of the internal pairing grid (``final_l``/
``la``), which is always 1 for this pairing -- collapsing X's primal dim to
size 1 and taking slice 0 of its values, changing the OUTPUT SHAPE (not just
the values). The real extent and values live on the RHS half of the grid
(``final_r``/``ra``). This reproduces with X as the LHS operand of ``@``
(the scalar as RHS) -- ``scalar @ X`` (scalar as LHS) already took a
different, correct code path.

Finding: .scratch/trustworthy-approx-search/grill2/F3-rank0-and-demand-emit.md
Fix ported from graphax branch wip/t28b-20260905, commit 4c98596.
"""

import jax.numpy as jnp

from graphax.sparse.tensor import SparseTensor
from graphax.sparse.ops.utils import _arr2st
from graphax.sparse.ops.matmul import matmul


def test_tensor_at_scalar_returns_scaled_tensor_not_slice():
    """``X @ scalar`` on the tiled engine must equal ``scalar * X.dense()``,
    with X's full shape preserved -- not a size-1 slice."""
    x = jnp.arange(6.0).reshape(2, 3)
    X = _arr2st(x, out_ndim=1)
    scalar = SparseTensor((), (), jnp.array(4.0))

    out = matmul(X, scalar)

    expected = 4.0 * X.dense()
    assert out.dense().shape == expected.shape, (
        f"X @ scalar changed shape: got {out.dense().shape}, "
        f"want {expected.shape} (X's own shape) -- this is the size-1-slice bug"
    )
    assert jnp.allclose(out.dense(), expected)


def test_scalar_at_tensor_already_correct():
    """``scalar @ X`` (scalar as the LHS operand) took the unaffected code
    path even before the fix -- pins it stays correct."""
    x = jnp.arange(6.0).reshape(2, 3)
    X = _arr2st(x, out_ndim=1)
    scalar = SparseTensor((), (), jnp.array(4.0))

    out = matmul(scalar, X)

    expected = 4.0 * X.dense()
    assert jnp.allclose(out.dense(), expected)
