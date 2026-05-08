"""Bug: matmul of non-zero-fill operands silently fell through to a
zero-fill fast path when ``_densify_is_safe`` returned False.

The dispatcher previously routed non-zero-fill matmul through the densify
escape hatch only when contracting dim sizes lined up positionally. When the
safety check failed (e.g. graphax's AD reordering produces incompatible
logical sizes), control fell through to the tiled / aligned-pair / dot_general
fast paths, all of which assume implicit positions are zero. The result was
silently tagged ``zero_fill=True`` with the fill contributions dropped on
the floor.

The fix raises ``NotImplementedError`` instead of producing a silently-wrong
tensor.
"""

import jax.numpy as jnp
import pytest

from graphax.sparse.tensor import SparseTensor
from graphax.sparse.indexes import DenseIndex


def test_matmul_nonzero_fill_with_incompatible_sizes_raises():
    """Non-zero ``fill_value`` on either side AND incompatible logical sizes
    on the contracting dims must raise ``NotImplementedError`` rather than
    silently routing through a zero-fill fast path."""
    # 2x3 lhs with non-zero fill, 4x5 rhs with zero fill — the contracting
    # primal/out sizes (3 vs 4) don't match positionally so ``_densify_is_safe``
    # returns False. Each tensor's ids must form a contiguous range starting
    # from 0 (enforced by ``_assert_sparse_tensor_consistency``).
    lhs = SparseTensor(
        (DenseIndex(0, 2, axis=0),),
        (DenseIndex(1, 3, axis=1),),
        jnp.zeros((2, 3), dtype=jnp.float32),
        fill_value=jnp.array(0.5, dtype=jnp.float32),
    )
    rhs = SparseTensor(
        (DenseIndex(0, 4, axis=0),),
        (DenseIndex(1, 5, axis=1),),
        jnp.zeros((4, 5), dtype=jnp.float32),
    )
    with pytest.raises(NotImplementedError, match="non-zero fill_value"):
        _ = lhs @ rhs
