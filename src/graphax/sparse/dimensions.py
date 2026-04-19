from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import override


@dataclass(frozen=True)
class Dimension(ABC):
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
    def shape(self) -> tuple[int]:
        return (self.size,)


@dataclass(frozen=True)
class DenseDimension(Dimension):
    pass


@dataclass(frozen=True)
class SparseDimension(Dimension):
    other_id: int
    block_size: int | None = None
    block_val_dim: int | None = None

    @override
    def __post_init__(self):
        super().__post_init__()
        if self.block_size is not None and self.block_size <= 0:
            raise ValueError(
                f"SparseDimension block_size must be positive, got {self.block_size}"
            )

    @property
    @override
    def logical_size(self) -> int:
        return self.size * (self.block_size or 1)

    @property
    def shape(self) -> tuple[int]:
        if self.block_size is None:
            return (self.size,)
        else:
            return (self.size, self.block_size)
