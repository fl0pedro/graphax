from __future__ import annotations

from dataclasses import dataclass


def _split_fill(fill):
    """Per-side ``(fill_lhs, fill_rhs)`` for a ``SetIndex`` fill, which may be a
    single shared value or an explicit ``(lhs, rhs)`` tuple."""
    return fill if isinstance(fill, tuple) else (fill, fill)


@dataclass(frozen=True)
class Index:
    """Unified dim descriptor.

    Discriminator: ``other_id is None`` ⇒ dense; otherwise sparse-paired.
    ``block_size`` / ``block_axis`` are only meaningful for sparse pairs
    (block-diagonal storage) and stay ``None`` for dense and plain sparse.

    The ``DenseIndex`` and ``DiagonalIndex`` factory functions below construct
    this class with the appropriate field set. ``DiagonalIndex`` was
    historically named ``SparseIndex``; the new name is clearer because the
    underlying structure is always a meta-block-diagonal pair (an outer-meta
    + per-meta-block-diagonal block). ``SparseIndex`` is kept as an alias
    at the bottom of this file for in-flight callers; it will be deleted
    after the Phase 8 migration completes.
    """

    id: int
    size: int
    axis: int | None
    other_id: int | None = None
    block_size: int | None = None
    block_axis: int | None = None

    def __post_init__(self):
        if self.size < 0:
            raise ValueError(f"Index size must be non-negative, got {self.size}")
        if self.block_size is not None and self.block_size <= 0:
            raise ValueError(
                f"Index block_size must be positive, got {self.block_size}"
            )

    @property
    def is_sparse(self) -> bool:
        return self.other_id is not None

    @property
    def is_compressed(self) -> bool:
        """``True`` for the banded / set compressed Index subclasses, which
        carry structure that matmul / elementwise cannot consume directly and
        must be densified at the op boundary (Phase 8). ``False`` for the
        plain dense / meta-block-diagonal ``Index``."""
        return False

    @property
    def logical_size(self) -> int:
        return self.size * (self.block_size or 1)

    @property
    def shape(self) -> tuple[int, ...]:
        if self.block_size is None:
            return (self.size,)
        return (self.size, self.block_size)


@dataclass(frozen=True)
class CompressedIndex(Index):
    """Base for the compressed Index types (:class:`BandedIndex` /
    :class:`SetIndex`) — structure that matmul / elementwise cannot consume
    directly and so is densified at the op boundary (Phase 8). The single thing
    shared here is the ``is_compressed`` discriminator; each subclass provides
    its own ``densify_axis`` / ``reduces_to_diagonal`` / ``to_meta_blocks`` (the
    band-gather vs set-op kernels differ structurally). ``isinstance(d,
    CompressedIndex)`` is the canonical structural test."""

    @property
    def is_compressed(self) -> bool:
        return True


@dataclass(frozen=True)
class BandedIndex(CompressedIndex):
    """Banded compressed dim — one side of a band axis-pair.

    A banded output (from a misaligned-contract matmul) is non-zero only
    within a finite block-band. The two ``BandedIndex`` of a pair (linked by
    ``other_id``) describe the row and col axes of that band; ``primary``
    selects which side owns the ``(M_p, W)`` prefix of the data layout.

    The data ``val`` for a single band pair is the canonical layout
    ``(n_meta * M_primary, W, B_row, B_col, *L)`` — ``densify_axis`` reads
    ``W`` / ``B_row`` / ``B_col`` straight from ``val.shape`` and the meta
    counts from ``size`` / ``n_secondary`` / ``n_meta``, so it needs no
    reference to the partner index.

    Extra fields beyond the base ``Index``:
      * ``band_width`` (``W``): in-band sub-blocks per primary slot.
      * ``offset``: per-primary secondary offsets; ``()`` = centered band.
      * ``primary``: ``True`` if this side is the band-primary (row-primary
        densify); ``False`` = col-primary.
      * ``n_secondary``: per-batch meta count on the non-primary axis
        (``-1`` ⇒ square = ``M_primary``).
      * ``n_meta``: outer-batch count (independent bands stacked on the
        meta-diagonal; the ``gcd(M_a, M_b) > 1`` case).
    """

    band_width: int = 1
    offset: tuple[int, ...] = ()
    primary: bool = True
    n_secondary: int = -1
    n_meta: int = 1

    def reduces_to_diagonal(self) -> bool:
        """``True`` iff this band is actually a pure meta-block-diagonal —
        ``band_width == 1`` with a centered (≡ identity, since W=1) offset and
        square per-batch meta counts. Such a band has no off-diagonal content,
        so it can be represented as a ``DiagonalIndex`` (the compact
        meta-block form) instead of being materialized fully dense.

        ``size`` is the meta count (``n_meta * M_primary``); the per-batch
        primary count is ``size // n_meta``, which must equal ``n_secondary``
        for the band to be square-diagonal.

        Only the PRIMARY side can self-determine this: on a col-primary
        (``primary=False``) index ``size // n_meta`` is the *secondary* meta
        count, so the ``!= n_secondary`` test would be vacuously satisfied
        (``n_secondary != n_secondary``) and wrongly report a non-square band as
        diagonal. The true row count lives on the partner, which a pure method
        can't see, so a non-primary index conservatively reports ``False``
        (densify fully) — never wrong, just less compact. In practice the only
        caller picks the primary side, so this is a latent-safety guard."""
        if self.band_width != 1:
            return False
        if not self.primary:
            return False
        m_primary = self.size // self.n_meta
        if self.n_secondary >= 0 and self.n_secondary != m_primary:
            return False
        # offset () (centered, W=1 ⇒ offset[a]=a) or explicit identity.
        if self.offset and tuple(self.offset) != tuple(range(len(self.offset))):
            return False
        return True

    def to_meta_blocks(self, val):
        """Compact ``(n_meta*M, B_row, B_col, *L)`` meta-diagonal blocks.
        Only valid when :meth:`reduces_to_diagonal` (W=1 identity band); the
        band's single in-band slot per row IS the diagonal block."""
        return val[:, 0]  # (n_meta*M, B_row, B_col, *L)

    def densify_axis(self, val, fill):
        """Expand this band pair's canonical ``val`` layout
        ``(n_meta*M_primary, W, B_row, B_col, *L)`` into the dense
        ``(n_meta*M_row*B_row, n_meta*M_col*B_col, *L)`` via the shared,
        CR-fixed ``_densify_band`` orchestrator (n_meta-aware, broadcast-
        limit guarded, gather-free for centered offsets, static-zero-fill
        skip). Lazy import avoids an indexes→ops import cycle."""
        from graphax.sparse.ops.block_storage import _densify_band

        n_meta = self.n_meta
        M_primary = val.shape[0] // n_meta
        W = val.shape[1]
        B_row = val.shape[2]
        B_col = val.shape[3]
        L = tuple(val.shape[4:])
        M_secondary = self.n_secondary if self.n_secondary >= 0 else M_primary
        return _densify_band(
            val, M_primary, M_secondary, W, B_row, B_col,
            tuple(self.offset), 0 if self.primary else 1, n_meta, fill, L,
        )


@dataclass(frozen=True)
class SetIndex(CompressedIndex):
    """Set-theoretic compressed dim for elementwise outputs — one side of a
    pair (linked by ``other_id``).

    Unifies the legacy ``UnionBlocks`` / ``IntersectionBlocks`` /
    ``DivisorRemainder`` pytrees. The two per-side block buffers are stored as
    a SINGLE concatenated 1-D ``val`` (like the legacy ``UnionBlocks.combined``)
    so ``SparseTensor.val`` stays a plain ``Array`` — no constructor / unary-op
    surgery. ``lhs_shape`` / ``rhs_shape`` reconstruct the buffers:
    ``lhs_blocks = val[:prod(lhs_shape)].reshape(lhs_shape)`` etc., where
    ``lhs_shape = (n_meta*M, n_lhs, B_lhs_h, B_lhs_w, *L)`` and similarly rhs.

    Extra fields:
      * ``semantic``: descriptive label ('union' | 'intersection').
      * ``lhs_shape`` / ``rhs_shape``: per-side block-buffer shapes.
      * ``include_remainder``: when ``False`` the rhs buffer is omitted
        (``val`` is just the lhs buffer flat).
      * ``n_meta``: outer batch.
      * ``op``: the actual binary densify op (e.g. ``jnp.add`` / ``jnp.multiply``)
        — what densify uses. ``a * b`` routes with ``is_intersection=False``
        (semantic 'union') yet ``op=multiply``, so ``op`` ≠ a function of
        ``semantic``. Callables are hashable, safe in static-aux metadata.
    """

    semantic: str = "union"
    lhs_shape: tuple[int, ...] = ()
    rhs_shape: tuple[int, ...] = ()
    include_remainder: bool = True
    n_meta: int = 1
    op: object = None

    def reduces_to_diagonal(self) -> bool:
        """A SetIndex output is always meta-block-diagonal (its content sits on
        the M-meta diagonal of ``(M*LCM_h, M*LCM_w)``), so it ALWAYS reduces to
        a compact ``DiagonalIndex`` via :meth:`to_meta_blocks`."""
        return True

    def _op(self):
        if self.op is not None:
            return self.op
        import jax.numpy as _jnp

        return _jnp.multiply if self.semantic == "intersection" else _jnp.add

    def combined_fill(self, fill):
        """The op-combined implicit-cell fill ``op(fill_lhs, fill_rhs)`` that
        densify stitches off the meta-diagonal — i.e. the ``fill_value`` of the
        materialized result tensor. The single source of truth for "what fill
        does densifying this SetIndex produce" (used by both the densify kernels
        here and the op-boundary re-wrap in ``ops/utils``).

        A ``None`` (statically-zero) fill stays ``None``: densifying a zero-fill
        SetIndex yields a zero-fill result, so the marker propagates and callers
        need no separate None guard."""
        if fill is None:
            return None
        return self._op()(*_split_fill(fill))

    def _split(self, val):
        """Split the concatenated 1-D ``val`` back into ``(lhs_blocks,
        rhs_blocks)`` (rhs ``None`` when ``include_remainder`` is False)."""
        import math as _math

        n_lhs = _math.prod(self.lhs_shape)
        lhs_blocks = val[:n_lhs].reshape(self.lhs_shape)
        if self.include_remainder and self.rhs_shape:
            rhs_blocks = val[n_lhs:].reshape(self.rhs_shape)
        else:
            rhs_blocks = None
        return lhs_blocks, rhs_blocks

    def to_meta_blocks(self, val, fill):
        """Compact ``(n_meta*M, LCM_h, LCM_w, *L)`` meta-block grid — the
        per-meta-diagonal contributions without surrounding zero padding
        (M× tighter than :meth:`densify_axis`; the op-boundary form)."""
        from graphax.sparse.ops.block_storage import _block_diag_per_meta

        lhs_blocks, rhs_blocks = self._split(val)
        fill_lhs, fill_rhs = _split_fill(fill)
        op = self._op()
        lhs_meta = _block_diag_per_meta(lhs_blocks, fill_lhs)
        if rhs_blocks is not None:
            rhs_meta = _block_diag_per_meta(rhs_blocks, fill_rhs)
            return op(lhs_meta, rhs_meta)
        # include_remainder=False ⟹ the rhs buffer is omitted because the rhs
        # has NO explicit blocks: it is ``fill_rhs`` everywhere. So the exact
        # densify is ``op(lhs_meta, fill_rhs)`` — for an intersection (multiply)
        # with ``fill_rhs == 0`` this correctly yields 0 (``x ∩ ∅ = ∅``); it is
        # not a data-zeroing bug but the right answer for an empty rhs.
        return op(lhs_meta, fill_rhs)

    def densify_axis(self, val, fill):
        """Densify to the FULL dense ``(n_meta*M*LCM_h, n_meta*M*LCM_w, *L)``
        form via :meth:`to_meta_blocks` + ``_stitch_meta`` + ``op``
        (gather-free; mirrors the legacy ``DivisorRemainder.to_dense``)."""
        from graphax.sparse.ops.block_storage import _stitch_meta

        per_meta = self.to_meta_blocks(val, fill)
        return _stitch_meta(per_meta, self.combined_fill(fill))


def DenseIndex(id: int, size: int, axis: int | None) -> Index:
    """Construct a dense ``Index`` (``other_id`` left ``None``)."""
    return Index(id, size, axis)


def DiagonalIndex(
    id: int,
    size: int,
    axis: int | None,
    other_id: int,
    block_size: int | None = None,
    block_axis: int | None = None,
) -> Index:
    """Construct a meta-block-diagonal ``Index`` (``other_id`` points at the
    partner; together the pair encodes a meta-block-diagonal storage layout).
    Historically named ``SparseIndex``; the new name better describes the
    underlying structure."""
    return Index(id, size, axis, other_id, block_size, block_axis)


# Back-compat alias for the rename in Phase 8.A. Delete after every call
# site migrates to ``DiagonalIndex``.
SparseIndex = DiagonalIndex
