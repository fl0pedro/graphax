"""Bug: `_copy` lost pre/post_transforms because of the same positional-arg trap.

`_copy` (in `sparse/ops/utils.py`) constructed the new SparseTensor as

    SparseTensor(od, pd, v, s, f, st.pre_transforms, st.post_transforms, ...)

With the new __init__ signature (`val, scalar_mult, fill_value, dtype,
pre_transforms, ...`) the 6th positional slot is `dtype`, not
`pre_transforms` — so `st.pre_transforms` got silently dropped (or rather,
landed in the wrong slot and was discarded), and `st.post_transforms` ended up
in the `pre_transforms` slot.

This test pins the contract: copying a SparseTensor preserves both transform
lists.
"""

import jax.numpy as jnp

from graphax.sparse.tensor import SparseTensor


def test_copy_preserves_pre_transforms():
    sentinel = object()
    st = SparseTensor([], [], None, pre_transforms=[sentinel])
    cp = st.copy()
    assert cp.pre_transforms == (sentinel,)


def test_copy_preserves_post_transforms():
    sentinel = object()
    st = SparseTensor([], [], None, post_transforms=[sentinel])
    cp = st.copy()
    assert cp.post_transforms == (sentinel,)


def test_copy_preserves_both_transforms_independently():
    pre = object()
    post = object()
    st = SparseTensor([], [], None, pre_transforms=[pre], post_transforms=[post])
    cp = st.copy()
    assert cp.pre_transforms == (pre,)
    assert cp.post_transforms == (post,)


def test_copy_preserves_scalar_mult_and_fill_value():
    """Same trap could also shift these. Pin that they survive a round-trip."""
    st = SparseTensor(
        [], [], None,
        scalar_mult=jnp.array(3.0),
        fill_value=jnp.array(2.0),
    )
    cp = st.copy()
    assert float(cp.scalar_mult) == 3.0
    assert float(cp.fill_value) == 2.0
