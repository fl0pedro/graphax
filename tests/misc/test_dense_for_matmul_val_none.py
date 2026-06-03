"""Bug: ``dense_for_matmul`` fallback crashed with ``TypeError`` on ``val=None``.

For tensors that don't hit the fully-dense or single-pair fast paths,
``dense_for_matmul`` calls ``dense(tensor, hard=True)`` and multiplies the
result's ``.val`` by ``scalar_mult``. The original line ``dense(...).val *
scalar_mult`` raised ``TypeError`` when the densified tensor's ``val`` was
``None`` (pure-structure tensors).

The fix broadcasts the ``fill_value`` when the densified ``val`` is ``None``
instead of multiplying ``None`` by ``scalar_mult``.
"""
import importlib

import jax.numpy as jnp
import numpy as np

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.ops.dense import dense_for_matmul
from graphax.sparse.tensor import SparseTensor

dense_mod = importlib.import_module("graphax.sparse.ops.dense")


def test_fallback_does_not_crash_with_none_densified_val(monkeypatch):
    """When ``dense(t, hard=True).val`` is ``None`` (the compressed-only
    early-return path), the fallback must broadcast the fill_value rather
    than try to multiply ``None`` by a scalar.

    We force the None case by monkeypatching ``dense`` to return a tensor
    with ``val=None`` — the original buggy ``dense(...).val * scalar_mult``
    line would raise ``TypeError`` on this; the fixed path returns the
    fill * scalar broadcast instead.
    """
    # Build a 3-dim tensor that defeats the fully-dense and single-pair
    # fast paths so the fallback runs.
    out_a = DiagonalIndex(id=0, size=3, axis=0, other_id=2)
    out_b = DiagonalIndex(id=1, size=2, axis=1, other_id=3)
    primal_a = DiagonalIndex(id=2, size=3, axis=0, other_id=0)
    primal_b = DiagonalIndex(id=3, size=2, axis=1, other_id=1)

    val = jnp.ones((3, 2), dtype=jnp.float32)
    st = SparseTensor(
        out_dims=(out_a, out_b),
        primal_dims=(primal_a, primal_b),
        val=val,
        scalar_mult=jnp.array(3.0),
        fill_value=jnp.array(7.0),
    )

    # Patch dense() to return a tensor with val=None (simulates the
    # compressed early-return path that preserves val=None).
    real_dense = dense_mod.dense

    def fake_dense(tensor, axes=None, hard=False):
        densified = real_dense(tensor, axes=axes, hard=hard)
        # Replace val with None; everything else (dims, fill_value, shape) intact.
        from graphax.sparse.tensor import SparseTensor as _ST
        return _ST(
            densified.out_dims, densified.primal_dims, None,
            scalar_mult=densified.scalar_mult,
            fill_value=densified.fill_value,
            check_consistency=False,
        )

    monkeypatch.setattr(dense_mod, "dense", fake_dense)

    # Should not raise — the bug was a TypeError from None * Array.
    result = dense_for_matmul(st)

    # The fallback emits fill_value * scalar_mult broadcast to the densified
    # logical shape. Logical shape = (3, 2, 3, 2).
    assert result.shape == (3, 2, 3, 2)
    arr = np.asarray(result)
    # Every cell should equal fill * scalar = 7 * 3 = 21.
    assert np.allclose(arr, 21.0)


# NOTE: the former ``test_fallback_works_with_compressed_pytree_val`` was
# deleted in the Phase-8 follow-up. It exercised ``_resolve_val`` materializing
# a non-Array ``.to_dense()`` pytree left in ``val`` — but post-Phase-8 ``val``
# is always a plain Array (compressed structure lives in the dim ``Index``
# types), ``dense()`` never returns a pytree val, and ``_resolve_val`` is gone.
# The remaining ``val is None`` fallback (above) is the live behavior.
