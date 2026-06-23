r"""Shared helpers for the elemental ``{D, B}`` kernels.

Two pieces of logic were hand-rolled across the pairwise kernels
(``contract_D_B`` / ``contract_B_B`` / ``elementwise_D_B`` / ``produce_compress``):

  1. **Block-buffer canonicalization** — transpose a ``DiagonalIndex`` pair's
     ``val`` (whose physical axis order is implementation-defined, addressed via
     each dim's ``.axis`` / ``.block_axis``) into the canonical
     ``(meta, block_row, block_col [, *L])`` order, inserting singleton axes for
     absent (size-1 / implicit) block axes and broadcasting up.

  2. **Dense-result rewrap** — build a fully-Dense ``SparseTensor`` whose out /
     primal dims are freshly-id'd ``DenseIndex`` objects on axes ``0..n``, with
     ``scalar_mult`` folded to 1 and ``fill_value=None``.

The call sites are NOT all identical (a code review flagged this): e.g.
``contract_B_B`` uses a STRICTER present-axis check than its siblings, and
``elementwise_D_B`` carries a leftover ``*L`` tail and folds ``scalar_mult`` into
the buffer.  These differences are PRESERVED here via explicit parameters — the
helpers reproduce each variant's exact behavior; nothing is silently reconciled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import jax.numpy as jnp

from graphax.sparse.dtype_compute import _scaled_mul
from graphax.sparse.indexes import DenseIndex

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


# --------------------------------------------------------------------------- #
# 1. Block-buffer canonicalization
# --------------------------------------------------------------------------- #
def canonical_block_buffer(
    val: "jnp.ndarray | None",
    slots: Sequence[tuple[int | None, int]],
    *,
    dtype,
    scalar_mult=None,
    keep_leftover_tail: bool = False,
    strict_present: bool = False,
) -> jnp.ndarray:
    """Canonicalize ``val`` into the block layout described by ``slots``.

    ``slots`` is the ordered list of ``(physical_axis_or_None, size)`` canonical
    axes (e.g. ``[(meta_axis, N), (row_axis, B_r), (col_axis, B_c)]``).  The
    present physical axes are transposed to the front in ``slots`` order; absent
    axes (``axis is None``, or — under ``strict_present`` — out-of-range / wrong
    length) become broadcast singletons.  The result is broadcast to the full
    ``tuple(size for _, size in slots)`` shape.

    Parameters reproduce each call site's exact behavior:

    * ``scalar_mult`` — when not ``None``, ``val`` is pre-scaled by it before
      canonicalization (``elementwise_D_B._diag_grid`` folds the operand's
      ``scalar_mult`` into the block buffer; the contraction kernels fold it
      later, downstream).
    * ``keep_leftover_tail`` — when ``True``, any physical axes NOT claimed by a
      slot are kept as a trailing ``*L`` (the ``elementwise_D_B`` leftover/batch
      axes ride along); the broadcast target becomes ``slot_sizes + L``.  When
      ``False`` (the contraction kernels), leftover axes are assumed size-1 and
      collapsed away.
    * ``strict_present`` — when ``True``, a slot's physical axis counts as
      "present" only if it is in range AND its length matches the expected size
      (``contract_B_B._block_buffer``'s stricter check); when ``False``, any
      non-``None`` axis counts (the looser check used by ``contract_D_B`` /
      ``elementwise_D_B``).

    ``val is None`` (uniform-ones structure) yields a ``ones`` buffer of the
    canonical slot shape (never a ``*L`` tail — matches every call site).
    """
    sizes = tuple(size for _ax, size in slots)
    if val is None:
        return jnp.ones(sizes, dtype=dtype)

    if scalar_mult is not None:
        val = _scaled_mul(val, scalar_mult)

    if strict_present:
        present = [
            ax
            for ax, length in slots
            if ax is not None and ax < val.ndim and val.shape[ax] == length
        ]
    else:
        present = [ax for ax, _length in slots if ax is not None]

    leftover = [a for a in range(val.ndim) if a not in present]
    perm = present + leftover
    v = jnp.transpose(val, perm) if perm != list(range(val.ndim)) else val

    n_present = len(present)
    tail = v.shape[n_present:] if keep_leftover_tail else ()
    if not keep_leftover_tail and v.ndim > n_present:
        # Leftover physical axes are assumed size-1 for a clean buffer; collapse.
        v = v.reshape(v.shape[:n_present])

    # Walk slots: consume present axes in order, insert a singleton for absent.
    expand_shape: list[int] = []
    ptr = 0
    present_set = set(present)
    for ax, _size in slots:
        if ax in present_set:
            expand_shape.append(v.shape[ptr])
            ptr += 1
        else:
            expand_shape.append(1)
    v = v.reshape(tuple(expand_shape) + tail)
    return jnp.broadcast_to(v, sizes + tail)


# --------------------------------------------------------------------------- #
# 2. Dense-result rewrap
# --------------------------------------------------------------------------- #
def emit_dense_result(
    values: jnp.ndarray,
    out_specs: Sequence[tuple[int, int]],
    primal_specs: Sequence[tuple[int, int]],
    *,
    dtype,
) -> "SparseTensor":
    """Wrap a contracted/combined dense array into a fully-Dense ``SparseTensor``.

    ``out_specs`` / ``primal_specs`` are ordered ``(logical_size, axis)`` pairs
    for the out- and primal-side ``DenseIndex`` dims, in their physical axis
    order.  Dims are freshly id'd ``0..n_out-1`` (out) then ``n_out..`` (primal),
    each carrying its given physical ``axis``.  ``scalar_mult`` is assumed already
    folded into ``values`` (set to 1) and ``fill_value`` is ``None`` (the closed
    ``{D, B}`` zero-fill result).
    """
    from graphax.sparse.tensor import SparseTensor

    n_out = len(out_specs)
    out_dims = tuple(
        DenseIndex(i, size, axis) for i, (size, axis) in enumerate(out_specs)
    )
    primal_dims = tuple(
        DenseIndex(n_out + i, size, axis)
        for i, (size, axis) in enumerate(primal_specs)
    )
    return SparseTensor(
        out_dims,
        primal_dims,
        values.astype(dtype),
        scalar_mult=jnp.array(1, dtype=dtype),
        fill_value=None,
        check_consistency=False,
    )


def is_block_diagonal(d) -> bool:
    """True iff ``d`` is a (non-compressed) block-diagonal ``DiagonalIndex`` dim —
    a meta-block-diagonal factor the elemental kernels operate on, as opposed to a
    plain ``DenseIndex`` or a ``CompressedIndex``. (``is_compressed`` lives on the
    ``Index`` base class, so plain attribute access is total over all dims.)"""
    return d.is_sparse and not d.is_compressed
