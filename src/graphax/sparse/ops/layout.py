from __future__ import annotations
from typing import Sequence


class Layout:
    def __init__(self):
        self.names_to_indices: dict[str, int] = {}

    def set(self, name: str, index: int) -> int:
        self.names_to_indices[name] = index
        return index

    def get(self, name: str) -> int:
        return self.names_to_indices[name]

    def permutation(self, *names: str | list[str]) -> list[int]:
        res = []
        for name in names:
            if isinstance(name, list):
                res.extend(self.names_to_indices[n] for n in name)
            else:
                res.append(self.names_to_indices[name])
        return res


def generate_block_permutation(
    num_pairs: int, axes_per_pair: int, offsets: Sequence[int]
) -> list[int]:
    """Generates a permutation by repeating offsets for each pair."""
    return [
        pair_idx * axes_per_pair + offset
        for pair_idx in range(num_pairs)
        for offset in offsets
    ]


def generate_grouped_permutation(
    num_pairs: int, axes_per_pair: int, group_offsets: Sequence[int]
) -> list[int]:
    """Generates a permutation by grouping specific offsets across all pairs."""
    return [
        pair_idx * axes_per_pair + offset
        for offset in group_offsets
        for pair_idx in range(num_pairs)
    ]
