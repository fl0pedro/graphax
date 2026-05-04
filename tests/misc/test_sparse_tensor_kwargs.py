"""Bug: SparseTensor positional args silently shifted between branches.

The `original/master` upstream had `SparseTensor.__init__(out_dims, primal_dims,
val, pre_transforms=None, post_transforms=None)` — 5 positional slots.

The local `core-v2` refactor inserted `scalar_mult`, `fill_value`, `dtype`
between `val` and `pre_transforms`, so:

    SparseTensor([], [], None, [transform])

silently put `[transform]` into the **scalar_mult** slot instead of
**pre_transforms**. The transform was never registered on the tensor, so
no-op edges replaced real Jacobian transforms — manifesting as wildly wrong
Jacobian shapes and silently-incorrect numerical results.

The fix: always pass `pre_transforms=[...]` / `post_transforms=[...]` as
keyword args. This test pins that contract on the current `__init__`.
"""

import jax.numpy as jnp

from graphax.sparse.tensor import SparseTensor


def test_pre_transforms_kwarg_lands_in_pre_transforms():
    """Passing pre_transforms as kwarg actually populates the field."""
    sentinel = object()
    st = SparseTensor([], [], None, pre_transforms=[sentinel])
    assert st.pre_transforms == (sentinel,)
    # And nothing leaked into scalar_mult / fill_value
    assert st.val is None


def test_post_transforms_kwarg_lands_in_post_transforms():
    sentinel = object()
    st = SparseTensor([], [], None, post_transforms=[sentinel])
    assert st.post_transforms == (sentinel,)


def test_positional_fourth_arg_is_scalar_mult_not_transforms():
    """Document the trap: 4th positional is scalar_mult, NOT pre_transforms.

    If someone reverts to the upstream-style positional call, they'll
    misinterpret a list of transforms as a scalar — this test verifies which
    slot the 4th positional truly maps to so the trap is explicit.
    """
    st = SparseTensor([], [], None, jnp.array(5.0))
    assert float(st.scalar_mult) == 5.0
    # And pre_transforms / post_transforms stayed empty (didn't accidentally
    # absorb anything).
    assert st.pre_transforms == ()
    assert st.post_transforms == ()
