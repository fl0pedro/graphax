"""Atomic SparseTensor micro-actions: DIAG, COMPRESS, and QUANT.

Operations the RL policy can emit per sub-step:

* :class:`Diag` — block-diagonalise a pair of *logical* indices ``(i, j)`` with
  an explicit positive integer factor. gcd-collapse is *not* a sentinel here —
  the caller passes the actual integer it picked, even if that integer happens
  to equal ``gcd(N_i, N_j)``.
* :class:`Compress` — reduce one or more *physical* axes of the val array via
  one of six elementwise reductions (``mean`` / ``min`` / ``max`` / ``median``
  / ``abs_min`` / ``abs_max``), and mark every Index that pointed at those
  axes as ``axis=None``. The per-step semantics: ``val ← reduce(val,
  axis=axes)``, then physical-axis bookkeeping shifts the surviving indices
  down. ``abs_min`` / ``abs_max`` pick the entry whose absolute value is
  smallest / largest (closest to zero / furthest from zero), preserving the
  original sign.

The two operations are atomic and order-dependent:
``DIAG ∘ COMPRESS ≠ COMPRESS ∘ DIAG`` in general. :func:`apply_micro_actions`
applies an ordered sequence; multi-axis :class:`Compress` is the natural way
to batch several physical-axis reductions into one ``jnp.mean`` call when the
policy emits them in the same coordinate frame.

Legality
--------
The atomic helpers raise :class:`ValueError` on structural illegality —
``i == j``, axes out of range, duplicate physical axes, mismatched
``DiagonalIndex`` pairings, etc. Policy code is expected to either mask these
choices out before sampling or catch the ValueError at rollout time.

This module's atomic helpers raise on structural illegality rather than
silently filtering (as the now-removed ``apply_dynamic_sparsity`` did) so
policy bugs surface immediately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Sequence, Union

import jax.numpy as jnp

from graphax.sparse.indexes import DenseIndex, Index, DiagonalIndex
from graphax.sparse.tensor import SparseTensor, _apply_block_diagonal


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diag:
    """A single block-diagonalisation of the logical index pair (i, j).

    ``i`` and ``j`` index into ``st.out_dims + st.primal_dims`` (the
    concatenated logical axis list). ``factor`` is the block size — the
    resulting :class:`DiagonalIndex` will have ``size=factor`` and
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
                "Pass an explicit positive divisor (the -1 / 0 sentinels "
                "from the legacy sparsity_map API are not accepted)."
            )
        if self.i == self.j:
            raise ValueError(f"Diag pair must be distinct, got i = j = {self.i}.")


COMPRESS_KINDS: tuple[str, ...] = (
    "mean", "min", "max", "median", "abs_min", "abs_max",
)
NUM_COMPRESS_KINDS = len(COMPRESS_KINDS)
COMPRESS_KIND_INDEX: dict[str, int] = {k: i for i, k in enumerate(COMPRESS_KINDS)}


@dataclass(frozen=True)
class Compress:
    """Reduce one or more *physical* axes of the underlying val array.

    ``axes`` is a tuple of physical-axis positions (``0 <= a < val.ndim``).
    Each axis must appear at most once. The physical axes are interpreted in
    the SparseTensor's current frame — if you intend to compress axes that
    were renumbered by an earlier micro-action in the same sub-episode, use
    the *current* numbering, not the numbering at sub-episode start.

    ``kind`` picks the reduction:

    * ``"mean"`` — arithmetic mean (default; the legacy behaviour).
    * ``"min"`` / ``"max"`` — elementwise min / max.
    * ``"median"`` — elementwise median.
    * ``"abs_min"`` — the entry whose absolute value is smallest along the
      axis (closest to zero); the original sign is preserved.
    * ``"abs_max"`` — the entry whose absolute value is largest (furthest
      from zero); the original sign is preserved.

    All :class:`Index` instances whose ``axis`` or ``block_axis`` referenced a
    compressed physical axis have that pointer set to ``None``; remaining
    pointers are shifted down to account for the dropped axes.
    """

    axes: tuple[int, ...]
    kind: str = "mean"

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
        if self.kind not in COMPRESS_KIND_INDEX:
            raise ValueError(
                f"Compress.kind must be one of {COMPRESS_KINDS!r}, "
                f"got {self.kind!r}."
            )


# Canonical JAX dtype names accepted by :class:`Quant`. Strings (not
# ``jnp.dtype`` objects) keep :class:`Quant` hashable and match the
# ``Compress.kind: str`` pattern. The int index into :data:`QUANT_DTYPE_INDEX`
# is the wire-format contract between a policy head (samples an int) and the
# env-side translator (looks the int up to construct a ``Quant``) — same
# convention as :data:`COMPRESS_KINDS`.
QUANT_DTYPES: tuple[str, ...] = (
    "bool",
    "int2", "int4", "int8", "int16", "int32", "int64",
    "uint2", "uint4", "uint8", "uint16", "uint32", "uint64",
    "float4_e2m1fn",
    "float8_e3m4", "float8_e4m3", "float8_e4m3b11fnuz", "float8_e4m3fn",
    "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz", "float8_e8m0fnu",
    "bfloat16", "float16", "float32", "float64",
    "complex64", "complex128",
)
NUM_QUANT_DTYPES = len(QUANT_DTYPES)
QUANT_DTYPE_INDEX: dict[str, int] = {d: i for i, d in enumerate(QUANT_DTYPES)}


@dataclass(frozen=True)
class Quant:
    """Cast :attr:`SparseTensor.val` to a chosen JAX dtype.

    Multiple Quant actions in a sub-episode are applied sequentially in
    order — last one wins, no special rounding. Equivalent to chained
    ``val.astype(d_1).astype(d_2)...``; the chain matters when intermediate
    dtypes are lossy (e.g. ``int8`` then ``float32``). Only ``val`` is cast;
    ``scalar_mult`` and ``fill_value`` keep their native dtype.

    A ``val=None`` SparseTensor (uniform grid) is returned unchanged.

    ``dtype`` is the canonical name (e.g. ``"float16"``, ``"float8_e4m3fn"``,
    ``"bfloat16"``) and must be a member of :data:`QUANT_DTYPES`. Narrow
    dtypes (float8 / float4 / sub-byte int / complex) have no implicit JAX
    promotion path; the ops upcast operands to their highest common dtype
    before arithmetic (see :func:`graphax.sparse.ops.utils._compute_dtype`),
    so a quantized ``val`` is stored narrow but computable everywhere.
    """

    dtype: str

    def __post_init__(self):
        if self.dtype not in QUANT_DTYPE_INDEX:
            raise ValueError(
                f"Quant.dtype must be one of {QUANT_DTYPES!r}, "
                f"got {self.dtype!r}."
            )


MicroAction = Union[Diag, Compress, Quant]


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

    if d1.is_sparse and d1.other_id != d2.id:
        raise ValueError(
            f"Diag pair conflict: logical index {action.i} is already paired "
            f"with another index (other_id={d1.other_id}, but d2.id={d2.id})."
        )
    if d2.is_sparse and d2.other_id != d1.id:
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


def _reduce_along_axes(val: jnp.ndarray, axes: tuple[int, ...], kind: str):
    """Reduce ``val`` along ``axes`` using the named elementwise rule.

    ``abs_min`` / ``abs_max`` pick the entry whose absolute value is smallest
    / largest along the reduction axes, preserving the original sign. Both
    are implemented as a take_along_axis over the argmin / argmax of ``|val|``
    after merging the reduction axes into a single trailing one — this keeps
    the reduction to a single XLA op even for multi-axis Compress and avoids
    a Python-level fold over the axes.
    """
    axes = tuple(sorted(set(axes)))
    if kind == "mean":
        return jnp.mean(val, axis=axes)
    if kind == "min":
        return jnp.min(val, axis=axes)
    if kind == "max":
        return jnp.max(val, axis=axes)
    if kind == "median":
        return jnp.median(val, axis=axes)
    if kind in ("abs_min", "abs_max"):
        # Move every reduction axis to the trailing positions, flatten them
        # into a single axis, then argmin / argmax over |·|. take_along_axis
        # against the original (flattened) values keeps the sign.
        keep = [a for a in range(val.ndim) if a not in axes]
        perm = keep + list(axes)
        moved = jnp.transpose(val, perm)
        flat_shape = moved.shape[: len(keep)] + (-1,)
        flat = moved.reshape(flat_shape)
        abs_flat = jnp.abs(flat)
        if kind == "abs_min":
            idx = jnp.argmin(abs_flat, axis=-1, keepdims=True)
        else:
            idx = jnp.argmax(abs_flat, axis=-1, keepdims=True)
        picked = jnp.take_along_axis(flat, idx, axis=-1)
        return jnp.squeeze(picked, axis=-1)
    raise ValueError(f"Unknown Compress.kind {kind!r}")


def apply_compress(st: SparseTensor, action: Compress) -> SparseTensor:
    """Reduce the listed physical axes via ``action.kind`` (default ``mean``).

    The reduction happens as a single ``jnp.<reduce>(val, axis=sorted_axes)``
    call — multi-axis Compress is the efficient form because all axes are
    dropped in one XLA op rather than a sequence of single-axis reductions.
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
    new_val = _reduce_along_axes(st.val, tuple(drops), action.kind)

    drop_set = set(drops)

    def _shift_after_drops(p: int | None) -> int | None:
        if p is None:
            return None
        if p in drop_set:
            return None
        return p - sum(1 for d in drops if d < p)

    def _remap(d: Index) -> Index:
        new_axis = _shift_after_drops(getattr(d, "axis", None))
        if d.is_sparse:
            new_block_axis = _shift_after_drops(d.block_axis)
            return replace(d, axis=new_axis, block_axis=new_block_axis)
        if not d.is_sparse:
            return replace(d, axis=new_axis)
        return replace(d, axis=new_axis)

    new_out = tuple(_remap(d) for d in st.out_dims)
    new_primal = tuple(_remap(d) for d in st.primal_dims)

    return SparseTensor(
        new_out,
        new_primal,
        new_val,
        scalar_mult=st.scalar_mult,
        fill_value=st.fill_value,
        # Forward the deferred-transform queues (the bare constructor defaults
        # them to ()): dropping them silently erases pending reshape/slice/...
        # relabels — the same defect fixed in apply_quant. Compress only changes
        # the at-rest val STORAGE (drops size-1/compressed axes); the transforms
        # act on the .dense() form, whose logical shape is unchanged, so they
        # ride through correctly.
        pre_transforms=st.pre_transforms,
        post_transforms=st.post_transforms,
        check_consistency=False,
    )


# ---------------------------------------------------------------------------
# Atomic QUANT
# ---------------------------------------------------------------------------


def apply_quant(st: SparseTensor, action: Quant) -> SparseTensor:
    """Cast ``st.val`` to ``action.dtype``.

    Returns ``st`` unchanged if ``val is None`` or the target dtype already
    matches the current one. Only ``val`` is cast — ``scalar_mult`` and
    ``fill_value`` keep their native dtype. Narrow targets (float8 /
    sub-byte int / float4 / complex) have no implicit JAX promotion path, so
    downstream densify / matmul / elementwise upcast to the highest common
    dtype before arithmetic (see :func:`graphax.sparse.ops.utils._compute_dtype`);
    the narrow dtype only sets the at-rest ``val.dtype``.
    """
    if st.val is None:
        return st
    target = jnp.dtype(action.dtype)
    if st.val.dtype == target:
        return st
    return SparseTensor(
        st.out_dims,
        st.primal_dims,
        st.val.astype(target),
        scalar_mult=st.scalar_mult,
        fill_value=st.fill_value,
        # The cast is shape-preserving, so any deferred Jacobian transform queued
        # on this edge (a reshape/slice/concatenate relabel awaiting drain) must
        # ride along — dropping it leaves ``val`` at its pre-drain shape and later
        # desyncs the contraction size / shape assert (jit-latent under canonical
        # orders, surfaced by non-canonical elimination orders).
        pre_transforms=st.pre_transforms,
        post_transforms=st.post_transforms,
        check_consistency=False,
    )


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Copy-pastable factory helpers
# ---------------------------------------------------------------------------
#
# These return a callable ``(SparseTensor) -> SparseTensor`` so the policy
# logs can render a sub-episode as a list of evaluable Python expressions
# (e.g. ``[diag(0, 1, 2), compress("abs_min", 3)]``) that round-trip through
# the typed-transform API in :mod:`graphax.core`.


def diag(i: int, j: int, factor: int) -> Callable[[SparseTensor], SparseTensor]:
    """Return a function that applies :class:`Diag` with the given args.

    Equivalent to ``lambda st: apply_diag(st, Diag(i, j, factor))``; the
    closure is hashable in graphax's `transforms` cache because the inner
    ``Diag`` is a frozen dataclass.
    """
    action = Diag(i=int(i), j=int(j), factor=int(factor))

    def _apply(st: SparseTensor) -> SparseTensor:
        return apply_diag(st, action)

    _apply.__name__ = f"diag({i}, {j}, {factor})"
    return _apply


def compress(
    kind: str, *axes: int,
) -> Callable[[SparseTensor], SparseTensor]:
    """Return a function that applies :class:`Compress` with the given kind/axes.

    ``compress("mean", 3)`` is the single-axis form; multi-axis is
    ``compress("abs_max", 0, 2)``. Mirrors the API of :func:`diag`.
    """
    action = Compress(axes=tuple(int(a) for a in axes), kind=kind)

    def _apply(st: SparseTensor) -> SparseTensor:
        return apply_compress(st, action)

    _apply.__name__ = f"compress({kind!r}, {', '.join(str(a) for a in axes)})"
    return _apply


def quant(dtype: str) -> Callable[[SparseTensor], SparseTensor]:
    """Return a function that applies :class:`Quant` with the given dtype.

    Equivalent to ``lambda st: apply_quant(st, Quant(dtype))``; the closure
    is hashable in graphax's ``transforms`` cache because the inner
    ``Quant`` is a frozen dataclass. Mirrors the API of :func:`diag` /
    :func:`compress`.
    """
    action = Quant(dtype=str(dtype))

    def _apply(st: SparseTensor) -> SparseTensor:
        return apply_quant(st, action)

    _apply.__name__ = f"quant({dtype!r})"
    return _apply


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
        elif isinstance(action, Quant):
            st = apply_quant(st, action)
        else:
            raise TypeError(
                f"apply_micro_actions expected Diag, Compress, or Quant, "
                f"got {type(action).__name__}."
            )
    return st
