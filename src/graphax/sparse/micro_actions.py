"""Atomic SparseTensor micro-actions: DIAG and COMPRESS.

Two operations the RL policy can emit per sub-step:

* :class:`Diag` — block-diagonalise a pair of *logical* indices ``(i, j)`` with
  an explicit positive integer factor. Replaces the legacy
  ``apply_dynamic_sparsity`` factor-table sentinels (``-1`` for gcd, ``0`` for
  drop). gcd-collapse is *not* a sentinel here — the policy passes the actual
  integer it picked, even if that integer happens to equal ``gcd(N_i, N_j)``.
* :class:`Compress` — mean-compress one or more *physical* axes of the val
  array and mark every Index that pointed at those axes as ``axis=None``. The
  per-step semantics: ``val ← jnp.mean(val, axis=axes)``, then physical-axis
  bookkeeping shifts the surviving indices down.

The two operations are atomic and order-dependent:
``DIAG ∘ COMPRESS ≠ COMPRESS ∘ DIAG`` in general. :func:`apply_micro_actions`
applies an ordered sequence; multi-axis :class:`Compress` is the natural way
to batch several physical-axis reductions into one ``jnp.mean`` call when the
policy emits them in the same coordinate frame.

Legality
--------
The atomic helpers raise :class:`ValueError` on structural illegality —
``i == j``, axes out of range, duplicate physical axes, mismatched
``SparseIndex`` pairings, etc. Policy code is expected to either mask these
choices out before sampling or catch the ValueError at rollout time.

The legacy :func:`apply_dynamic_sparsity` (in ``tensor.py``) silently filters
invalid pairs — this module's contract is stricter so policy bugs surface
rather than being masked by upstream filtering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence, Union

import jax.numpy as jnp

from graphax.sparse.indexes import DenseIndex, Index, SparseIndex
from graphax.sparse.tensor import SparseTensor, _apply_block_diagonal


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diag:
    """A single block-diagonalisation of the logical index pair (i, j).

    ``i`` and ``j`` index into ``st.out_dims + st.primal_dims`` (the
    concatenated logical axis list). ``factor`` is the block size — the
    resulting :class:`SparseIndex` will have ``size=factor`` and
    ``block_size = N // factor`` for each side. ``factor`` must be a positive
    divisor of both logical sizes. If you want gcd-collapse, pass
    ``factor = math.gcd(N_i, N_j)`` explicitly.
    """

    i: int
    j: int
    factor: int

    def __post_init__(self):
        if self.factor <= 0:
            raise ValueError(
                f"Diag.factor must be a positive integer, got {self.factor!r}. "
                "Use math.gcd(...) for gcd-collapse and pass it explicitly; "
                "the -1 / 0 sentinels from apply_dynamic_sparsity are no "
                "longer accepted."
            )
        if self.i == self.j:
            raise ValueError(f"Diag pair must be distinct, got i = j = {self.i}.")


@dataclass(frozen=True)
class Compress:
    """Mean-compress one or more *physical* axes of the underlying val array.

    ``axes`` is a tuple of physical-axis positions (``0 <= a < val.ndim``).
    Each axis must appear at most once. The physical axes are interpreted in
    the SparseTensor's current frame — if you intend to compress axes that
    were renumbered by an earlier micro-action in the same sub-episode, use
    the *current* numbering, not the numbering at sub-episode start.

    All :class:`Index` instances whose ``axis`` or ``block_axis`` referenced a
    compressed physical axis have that pointer set to ``None``; remaining
    pointers are shifted down to account for the dropped axes.
    """

    axes: tuple[int, ...]

    def __post_init__(self):
        if len(self.axes) != len(set(self.axes)):
            raise ValueError(
                f"Compress.axes must be unique, got {self.axes!r}."
            )
        for a in self.axes:
            if a < 0:
                raise ValueError(
                    f"Compress.axes entries must be non-negative, got {a!r}."
                )


MicroAction = Union[Diag, Compress]


# ---------------------------------------------------------------------------
# Atomic DIAG
# ---------------------------------------------------------------------------


def apply_diag(st: SparseTensor, action: Diag) -> SparseTensor:
    """Apply a single block-diagonalisation rule to ``st``.

    Validates structurally before delegating to the existing
    :func:`graphax.sparse.tensor._apply_block_diagonal` primitive — the
    primitive does the val-axis reshape + ``jnp.diagonal`` and metadata
    rebuild. Returns ``st`` unchanged for the degenerate ``factor == 1`` case
    (no real diagonalisation).
    """
    out_len = len(st.out_dims)
    total_ndim = out_len + len(st.primal_dims)

    if not (0 <= action.i < total_ndim):
        raise ValueError(
            f"Diag.i = {action.i} out of range [0, {total_ndim}); "
            f"out_dims={out_len}, primal_dims={total_ndim - out_len}."
        )
    if not (0 <= action.j < total_ndim):
        raise ValueError(
            f"Diag.j = {action.j} out of range [0, {total_ndim}); "
            f"out_dims={out_len}, primal_dims={total_ndim - out_len}."
        )

    is_out1 = action.i < out_len
    rel_i = action.i if is_out1 else action.i - out_len
    is_out2 = action.j < out_len
    rel_j = action.j if is_out2 else action.j - out_len

    d1 = st.out_dims[rel_i] if is_out1 else st.primal_dims[rel_i]
    d2 = st.out_dims[rel_j] if is_out2 else st.primal_dims[rel_j]

    if isinstance(d1, SparseIndex) and d1.other_id != d2.id:
        raise ValueError(
            f"Diag pair conflict: logical index {action.i} is already paired "
            f"with another index (other_id={d1.other_id}, but d2.id={d2.id})."
        )
    if isinstance(d2, SparseIndex) and d2.other_id != d1.id:
        raise ValueError(
            f"Diag pair conflict: logical index {action.j} is already paired "
            f"with another index (other_id={d2.other_id}, but d1.id={d1.id})."
        )

    N1, N2 = d1.logical_size, d2.logical_size
    factor = action.factor
    if N1 % factor != 0 or N2 % factor != 0:
        raise ValueError(
            f"Diag.factor = {factor} does not divide both logical sizes "
            f"({N1}, {N2})."
        )
    if factor == 1:
        # Degenerate: no real diagonalisation. Keep st as-is.
        return st

    b1, b2 = N1 // factor, N2 // factor
    return _apply_block_diagonal(
        st, is_out1, rel_i, is_out2, rel_j, factor, b1, b2,
    )


# ---------------------------------------------------------------------------
# Atomic COMPRESS
# ---------------------------------------------------------------------------


def apply_compress(st: SparseTensor, action: Compress) -> SparseTensor:
    """Mean-compress the listed physical axes.

    The reductions happen as one ``jnp.mean(val, axis=sorted_axes)`` call —
    multi-axis Compress is the efficient form because all axes are dropped in
    a single XLA op rather than a sequence of one-axis means.
    """
    if st.val is None:
        if action.axes:
            raise ValueError(
                "Cannot compress a SparseTensor with val=None along "
                f"axes={action.axes!r}."
            )
        return st
    if not action.axes:
        return st

    val_ndim = st.val.ndim
    for a in action.axes:
        if a >= val_ndim:
            raise ValueError(
                f"Compress.axes entry {a} out of range for val.ndim = {val_ndim}."
            )

    drops = sorted(set(action.axes))
    new_val = jnp.mean(st.val, axis=tuple(drops))

    drop_set = set(drops)

    def _shift_after_drops(p: int | None) -> int | None:
        if p is None:
            return None
        if p in drop_set:
            return None
        return p - sum(1 for d in drops if d < p)

    def _remap(d: Index) -> Index:
        new_axis = _shift_after_drops(getattr(d, "axis", None))
        if isinstance(d, SparseIndex):
            new_block_axis = _shift_after_drops(d.block_axis)
            return replace(d, axis=new_axis, block_axis=new_block_axis)
        if isinstance(d, DenseIndex):
            return replace(d, axis=new_axis)
        return replace(d, axis=new_axis)

    new_out = tuple(_remap(d) for d in st.out_dims)
    new_primal = tuple(_remap(d) for d in st.primal_dims)

    return SparseTensor(
        new_out,
        new_primal,
        new_val,
        scalar_mult=st.scalar_mult,
        sort_val=False,
        check_consistency=False,
    )


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


def apply_micro_actions(
    st: SparseTensor,
    actions: Sequence[MicroAction],
) -> SparseTensor:
    """Apply an ordered sequence of :class:`Diag` and :class:`Compress` actions.

    Order matters: ``DIAG ∘ COMPRESS ≠ COMPRESS ∘ DIAG`` in general because
    DIAG sees logical indices (which depend on which axes have been compressed
    away) and COMPRESS sees physical axes (which depend on prior shape edits).

    No coalescing: if you want to batch several Compress actions into one
    ``jnp.mean`` call, emit a single :class:`Compress` with multiple axes —
    that's exactly what the multi-axis form is for. Coalescing across
    intervening DIAGs is not generally sound because the physical-axis
    numbering shifts.
    """
    for action in actions:
        if isinstance(action, Diag):
            st = apply_diag(st, action)
        elif isinstance(action, Compress):
            st = apply_compress(st, action)
        else:
            raise TypeError(
                f"apply_micro_actions expected Diag or Compress, "
                f"got {type(action).__name__}."
            )
    return st
