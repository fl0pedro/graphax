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

The fix: ``__init__`` makes everything after ``val`` keyword-only with a
``*,`` barrier, so the old positional misalignment now raises ``TypeError``
at the call site. This test pins the migration so a future refactor can't
silently re-introduce the trap.
"""

import pytest

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


def test_old_master_positional_signature_raises_typeerror():
    """The old upstream signature ``SparseTensor(out, primal, val, pre, post)``
    must now raise TypeError — that's the migration guarantee."""
    sentinel = object()
    with pytest.raises(TypeError):
        SparseTensor([], [], None, [sentinel])
    with pytest.raises(TypeError):
        SparseTensor([], [], None, [sentinel], [sentinel])
