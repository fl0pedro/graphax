"""Bug: ``dense()`` collapsed multiple sparse pairs into one combined diagonal.

``_densify_diagonal_scatter`` was called once on a tensor whose leading axes
had been linearised into a single ``B = prod(N_i)`` axis. After reshape back
to ``(N1, N2, N1, N2, *trailing)`` the math coincidentally lined up for the
no-block / no-extra-trailing case (because the combined diagonal in linearised
space picks the same positions as independent diagonals do). But the inner
broadcast still allocated a ``B^2 = (N1*N2)^2`` mask and — when block axes
were involved — the downstream reshape in ``_apply_dense_scattering`` failed
with a size mismatch because the same physical block axis cannot land at two
distinct output positions from a single linearised scatter.

The fix densifies pairs **sequentially**, one ``(N_i, N_i)`` diagonal at a
time, so each pair is independent and never gets fused into a combined
``prod(N_i)``-sized scatter.

Tests pin both:
  * the value-correctness contract for two pairs + a trailing dense dim, and
  * the structural contract that the inner densifier is called per pair (i.e.
    the eye_mask emitted is at most ``max(N_i)`` square, not ``prod(N_i)``).
"""
import jax.numpy as jnp
import numpy as np

from graphax.sparse.indexes import DenseIndex, SparseIndex
from graphax.sparse.ops.dense import dense
from graphax.sparse.tensor import SparseTensor


def test_two_pairs_with_trailing_dense_yields_independent_block_diagonals():
    """``dense()`` of a two-pair tensor with a trailing dense axis must place
    ``val[i1, i2, t]`` at output ``[i1, i2, t, j1, j2]`` iff ``i1==j1 ∧ i2==j2``."""
    N1, N2, T = 3, 4, 2
    out1 = SparseIndex(id=0, size=N1, axis=0, other_id=2)
    out2 = SparseIndex(id=1, size=N2, axis=1, other_id=3)
    primal1 = SparseIndex(id=2, size=N1, axis=0, other_id=0)
    primal2 = SparseIndex(id=3, size=N2, axis=1, other_id=1)
    extra = DenseIndex(id=4, size=T, axis=2)

    val = jnp.arange(N1 * N2 * T, dtype=jnp.float32).reshape(N1, N2, T) + 1.0
    st = SparseTensor(
        out_dims=(out1, out2, extra),
        primal_dims=(primal1, primal2),
        val=val,
    )
    result = dense(st, hard=True)

    expected = np.zeros((N1, N2, T, N1, N2), dtype=np.float32)
    for i1 in range(N1):
        for i2 in range(N2):
            for t in range(T):
                # The independent block-diagonal: nonzero only at (i1, i2, _, i1, i2).
                # Crucially NOT at any (i1, i2, _, j1, j2) where the
                # linearised-but-product-size diagonal would also fire — those
                # off-diagonal pairs are the same set here only because i1==j1
                # AND i2==j2 are both implied by i1*N2+i2 == j1*N2+j2.
                expected[i1, i2, t, i1, i2] = float(val[i1, i2, t])

    assert result.val.shape == (N1, N2, T, N1, N2)
    np.testing.assert_allclose(np.asarray(result.val), expected)


def test_two_pairs_no_trailing_yields_independent_block_diagonals():
    """Same contract for the no-trailing-dim case — pins value-equivalence with
    the independent-per-pair densifier (which is what the fix enforces)."""
    N1, N2 = 3, 4
    out1 = SparseIndex(id=0, size=N1, axis=0, other_id=2)
    out2 = SparseIndex(id=1, size=N2, axis=1, other_id=3)
    primal1 = SparseIndex(id=2, size=N1, axis=0, other_id=0)
    primal2 = SparseIndex(id=3, size=N2, axis=1, other_id=1)

    val = jnp.arange(N1 * N2, dtype=jnp.float32).reshape(N1, N2) + 1.0
    st = SparseTensor(
        out_dims=(out1, out2),
        primal_dims=(primal1, primal2),
        val=val,
    )
    result = dense(st, hard=True)

    expected = np.zeros((N1, N2, N1, N2), dtype=np.float32)
    for i1 in range(N1):
        for i2 in range(N2):
            expected[i1, i2, i1, i2] = float(val[i1, i2])

    assert result.val.shape == (N1, N2, N1, N2)
    np.testing.assert_allclose(np.asarray(result.val), expected)


def test_per_pair_densifier_eye_mask_size_is_per_pair_not_product():
    """Structural pin: the inner densifier is invoked per pair, so the largest
    eye-mask it ever allocates is ``max(N_i)``, not ``prod(N_i)``.

    If a future regression linearises pairs again, the eye-mask traced here
    would be ``(N1*N2, N1*N2)`` instead of two separate per-pair eye-masks.
    We probe by counting calls to ``_densify_diagonal_scatter``."""
    from unittest.mock import patch
    import importlib

    dense_mod = importlib.import_module("graphax.sparse.ops.dense")

    N1, N2 = 3, 4
    out1 = SparseIndex(id=0, size=N1, axis=0, other_id=2)
    out2 = SparseIndex(id=1, size=N2, axis=1, other_id=3)
    primal1 = SparseIndex(id=2, size=N1, axis=0, other_id=0)
    primal2 = SparseIndex(id=3, size=N2, axis=1, other_id=1)

    val = jnp.arange(N1 * N2, dtype=jnp.float32).reshape(N1, N2) + 1.0
    st = SparseTensor(
        out_dims=(out1, out2),
        primal_dims=(primal1, primal2),
        val=val,
    )

    leading_sizes: list[int] = []
    real_densifier = dense_mod._densify_diagonal_scatter

    def spy(v, fv):
        leading_sizes.append(v.shape[0])
        return real_densifier(v, fv)

    with patch.object(dense_mod, "_densify_diagonal_scatter", spy):
        dense(st, hard=True)

    # Two pairs → two separate calls, each with leading axis size N_i.
    assert leading_sizes == [N1, N2], (
        f"expected per-pair densifier calls of sizes [{N1}, {N2}], got {leading_sizes}"
    )
