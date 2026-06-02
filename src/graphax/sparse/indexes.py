from __future__ import annotations

from dataclasses import dataclass


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
class BandedIndex(Index):
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

    @property
    def is_compressed(self) -> bool:
        return True

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
class SetIndex(Index):
    """Set-theoretic compressed dim for elementwise outputs — one side of a
    pair (linked by ``other_id``).

    Unifies the legacy ``UnionBlocks`` / ``IntersectionBlocks`` /
    ``DivisorRemainder`` pytrees. The data ``val`` is a dual buffer
    ``(lhs_blocks, rhs_blocks)`` (same layout the legacy types used):
    ``lhs_blocks`` is ``(n_meta*M, n_lhs, B_lhs_h, B_lhs_w, *L)`` and
    ``rhs_blocks`` is ``(n_meta*M, n_rhs, B_rhs_h, B_rhs_w, *L)``.

    Extra fields:
      * ``semantic``: ``'union'`` (densify = ``op(lhs_grid, rhs_grid)``) or
        ``'intersection'``.
      * ``n_lhs`` / ``n_rhs``: sub-blocks per meta per side.
      * ``include_remainder``: when ``False`` the rhs buffer is omitted.
      * ``n_meta``: outer batch.
    """

    semantic: str = "union"
    n_lhs: int = 1
    n_rhs: int = 1
    include_remainder: bool = True
    n_meta: int = 1

    @property
    def is_compressed(self) -> bool:
        return True

    def densify_axis(self, val, fill):
        """Densify the dual-buffer ``val=(lhs_blocks, rhs_blocks)`` to the
        meta-block-diagonal grid via ``_block_diag_per_meta`` + ``_stitch_meta``
        + the semantic ``op``. ``fill`` is ``(fill_lhs, fill_rhs)`` or a single
        scalar applied to both. Mirrors the legacy ``DivisorRemainder.to_dense``
        densify chain (gather-free)."""
        from graphax.sparse.ops.block_storage import (
            _block_diag_per_meta, _stitch_meta,
        )
        import jax.numpy as _jnp

        lhs_blocks, rhs_blocks = (val if isinstance(val, tuple) else (val, None))
        fill_lhs, fill_rhs = (fill if isinstance(fill, tuple) else (fill, fill))
        op = _jnp.multiply if self.semantic == "intersection" else _jnp.add

        lhs_meta = _block_diag_per_meta(lhs_blocks, fill_lhs)
        if self.include_remainder and rhs_blocks is not None:
            rhs_meta = _block_diag_per_meta(rhs_blocks, fill_rhs)
            per_meta = op(lhs_meta, rhs_meta)
        else:
            per_meta = op(lhs_meta, fill_rhs)
        return _stitch_meta(per_meta, op(fill_lhs, fill_rhs))


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
