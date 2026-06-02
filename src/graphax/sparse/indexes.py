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
    def logical_size(self) -> int:
        return self.size * (self.block_size or 1)

    @property
    def shape(self) -> tuple[int, ...]:
        if self.block_size is None:
            return (self.size,)
        return (self.size, self.block_size)


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
