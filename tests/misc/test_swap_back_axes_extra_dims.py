"""Bug: ``_swap_back_axes`` could silently produce a duplicated-axis transpose.

Original code in ``ops/utils.py:_swap_back_axes`` had:

    seen = set(permutation[:i])
    for j in range(st.val.ndim):
        if j not in seen and i < len(permutation):
            permutation[i] = j
            i += 1

The ``i < len(permutation)`` clamp was supposed to be a "safety" guard, but
in practice it silently swallowed cases where the first loop had already
written duplicates into ``permutation[:i]`` (e.g. shared block_axis on a
sparse pair, or any other path that lets ``i`` exceed the count of unique
seen axes). When that happened, ``permutation[i:]`` retained its initial
``0`` values from ``[0] * st.val.ndim`` — and ``val.transpose(permutation)``
silently duplicated axis 0, producing a wrong-shape result downstream.

Fix: drop the clamp and add a final assertion that the permutation is a
valid permutation of ``range(val.ndim)``. This test pins the contract:
either we get a correct permutation, or we get an error — never silent
corruption.
"""

import jax.numpy as jnp

from graphax.sparse.indexes import DenseIndex
from graphax.sparse.ops.utils import _swap_back_axes
from graphax.sparse.tensor import SparseTensor


def test_extra_batch_axes_either_succeed_or_raise():
    """Construct a SparseTensor whose ``val`` carries extra physical axes that
    no Index references. ``_swap_back_axes`` must either produce a valid
    permutation that preserves shape semantics, or raise — never silently
    duplicate an axis."""
    # 2 dims (out + primal) describe physical axes 0 and 1; val has 4 physical
    # axes total (axes 2 and 3 are extra batch dims not referenced by any Index).
    out_dims = (DenseIndex(0, 5, 0),)
    primal_dims = (DenseIndex(1, 7, 1),)
    val = jnp.arange(5 * 7 * 3 * 2, dtype=jnp.float32).reshape((5, 7, 3, 2))
    st = SparseTensor(out_dims, primal_dims, val, sort_val=False)
    original_shape = st.val.shape

    try:
        result = _swap_back_axes(st)
    except (AssertionError, ValueError):
        # Acceptable — the helper detected the irregular layout.
        return

    # Otherwise: the resulting val.shape must contain the same multiset of axis
    # sizes as the input — duplicating an axis would change the multiset.
    assert sorted(result.val.shape) == sorted(original_shape), (
        f"_swap_back_axes silently corrupted the shape: "
        f"{original_shape} -> {result.val.shape}"
    )
    # And the result must be a permutation, not a duplication: each unique
    # axis size that appears once in the input must appear once in the output.
    assert result.val.size == st.val.size


def test_axis_zero_only_dim_does_not_silently_duplicate():
    """Stress: a single dim referencing axis 0 leaves ``permutation[1:]`` as
    initial 0s in the buggy code path. The fix's sanity assert (or the fixed
    fill loop) must catch this rather than letting transpose duplicate axis 0.
    """
    # 1 dim describing axis 0; val has 3 physical axes — so axes 1 and 2 are
    # "unseen" and the second loop must fill them in.
    out_dims = (DenseIndex(0, 4, 0),)
    primal_dims: tuple = ()
    val = jnp.arange(4 * 5 * 6, dtype=jnp.float32).reshape((4, 5, 6))
    st = SparseTensor(out_dims, primal_dims, val, sort_val=False,
                      check_consistency=False)
    original_shape = st.val.shape

    try:
        result = _swap_back_axes(st)
    except (AssertionError, ValueError):
        return

    assert sorted(result.val.shape) == sorted(original_shape)
    assert result.val.size == st.val.size
