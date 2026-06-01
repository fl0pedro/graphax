from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Index:
    """Unified dim descriptor.

    Discriminator: ``other_id is None`` ⇒ dense; otherwise sparse-paired.
    ``block_size`` / ``block_axis`` are only meaningful for sparse pairs
    (block-diagonal storage) and stay ``None`` for dense and plain sparse.

    The ``DenseIndex`` and ``SparseIndex`` factory functions below construct
    this class with the appropriate field set, so existing callsites stay
    unchanged.
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


def SparseIndex(
    id: int,
    size: int,
    axis: int | None,
    other_id: int,
    block_size: int | None = None,
    block_axis: int | None = None,
) -> Index:
    """Construct a sparse-paired ``Index`` (``other_id`` points at the partner)."""
    return Index(id, size, axis, other_id, block_size, block_axis)
