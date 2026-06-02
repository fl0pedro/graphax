"""Bug: ``dense_for_matmul`` fallback crashed with ``TypeError`` on ``val=None``.

For tensors that don't hit the fully-dense or single-pair fast paths,
``dense_for_matmul`` calls ``dense(tensor, hard=True)`` and multiplies the
result's ``.val`` by ``scalar_mult``. The original line ``dense(...).val *
scalar_mult`` raised ``TypeError`` when the densified tensor's ``val`` was
``None`` — which can happen on the early-return path through ``dense()`` (a
compressed-only tensor whose requested axes don't touch the compressed
pair leaves ``val=None`` intact) or whenever the densified ``val`` is one of
the compressed-storage pytrees rather than a plain ``Array``.

The fix routes the densified ``.val`` through ``_resolve_val`` (which expands
compressed pytrees and returns ``None`` unchanged) and broadcasts the
``fill_value`` when the resolved ``val`` is genuinely ``None``.
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


def test_fallback_works_with_compressed_pytree_val(monkeypatch):
    """When ``dense(t, hard=True).val`` is a compressed-storage pytree (has
    a ``.to_dense()`` method but isn't a JAX array), the old fallback would
    try ``pytree * scalar_mult`` which is ill-typed. The fix routes through
    ``_resolve_val`` which materializes the pytree first.
    """
    out_a = DiagonalIndex(id=0, size=3, axis=0, other_id=2)
    out_b = DiagonalIndex(id=1, size=2, axis=1, other_id=3)
    primal_a = DiagonalIndex(id=2, size=3, axis=0, other_id=0)
    primal_b = DiagonalIndex(id=3, size=2, axis=1, other_id=1)
    val = jnp.ones((3, 2), dtype=jnp.float32)
    st = SparseTensor(
        out_dims=(out_a, out_b),
        primal_dims=(primal_a, primal_b),
        val=val,
        scalar_mult=jnp.array(2.0),
    )

    sentinel_dense = jnp.full((3, 2, 3, 2), 5.0, dtype=jnp.float32)

    class FakeCompressed:
        """Stand-in for ``UnionBlocks``-style storage: opaque, has ``to_dense``."""

        def to_dense(self):
            return sentinel_dense

    real_dense = dense_mod.dense

    def fake_dense(tensor, axes=None, hard=False):
        densified = real_dense(tensor, axes=axes, hard=hard)
        from graphax.sparse.tensor import SparseTensor as _ST
        # Stuff the FakeCompressed into the .val slot — bypassing the
        # constructor's validation by post-assigning. This mimics how a
        # future densifier could leave ``val`` as a not-yet-materialized
        # pytree.
        st_out = _ST(
            densified.out_dims, densified.primal_dims, None,
            scalar_mult=densified.scalar_mult,
            fill_value=densified.fill_value,
            check_consistency=False,
        )
        object.__setattr__(st_out, "val", FakeCompressed())
        return st_out

    monkeypatch.setattr(dense_mod, "dense", fake_dense)

    result = dense_for_matmul(st)
    # Expected: sentinel_dense * scalar_mult = 5 * 2 = 10 everywhere.
    assert result.shape == (3, 2, 3, 2)
    np.testing.assert_allclose(np.asarray(result), 10.0)
