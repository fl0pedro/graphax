from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Dimension:
    id: int
    size: int
    val_dim: int | None

    def __post_init__(self):
        if self.size < 0:
            raise ValueError(f"Dimension size must be non-negative, got {self.size}")

    @property
    def logical_size(self) -> int:
        return self.size

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.size,)


@dataclass(frozen=True, slots=True)
class DenseDimension(Dimension):
    pass


@dataclass(frozen=True, slots=True)
class SparseDimension(Dimension):
    other_id: int
    block_size: int | None = None
    block_val_dim: int | None = None

    def __post_init__(self):
        if self.size < 0:
            raise ValueError(f"Dimension size must be non-negative, got {self.size}")
        if self.block_size is not None and self.block_size <= 0:
            raise ValueError(
                f"SparseDimension block_size must be positive, got {self.block_size}"
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
