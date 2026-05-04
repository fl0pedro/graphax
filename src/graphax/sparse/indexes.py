from __future__ import annotations
from dataclasses import dataclass

@dataclass(slots=True)
class Index:
    id: int
    size: int
    axis: int | None

    def __post_init__(self):
        if self.size < 0:
            raise ValueError(f"Index size must be non-negative, got {self.size}")

    @property
    def logical_size(self) -> int:
        return self.size

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.size,)


@dataclass(slots=True)
class DenseIndex(Index):
    pass


@dataclass(slots=True)
class SparseIndex(Index):
    other_id: int
    block_size: int | None = None
    block_axis: int | None = None

    def __post_init__(self):
        if self.size < 0:
            raise ValueError(f"Index size must be non-negative, got {self.size}")
        if self.block_size is not None and self.block_size <= 0:
            raise ValueError(
                f"SparseIndex block_size must be positive, got {self.block_size}"
            )

    @property
    def logical_size(self) -> int:
        return self.size * (self.block_size or 1)

    @property
    def shape(self) -> tuple[int, ...]:
        if self.block_size is None:
            return (self.size,)
        else:
            return (self.size, self.block_size)
