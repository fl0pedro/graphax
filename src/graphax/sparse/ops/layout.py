"""Per-pair axis-permutation helpers used by the matmul / elementwise grid layout."""
from __future__ import annotations
from typing import Sequence


def generate_block_permutation(
    num_pairs: int, axes_per_pair: int, offsets: Sequence[int]
) -> list[int]:
    """Permutation built by repeating ``offsets`` (relative axis positions) for each pair:
    ``[p0+o0, p0+o1, …, p1+o0, p1+o1, …]``."""
    return [
        pair_idx * axes_per_pair + offset
        for pair_idx in range(num_pairs)
        for offset in offsets
    ]


def generate_grouped_permutation(
    num_pairs: int, axes_per_pair: int, group_offsets: Sequence[int]
) -> list[int]:
    """Permutation built by grouping each offset across all pairs first:
    ``[p0+o0, p1+o0, …, p0+o1, p1+o1, …]``."""
    return [
        pair_idx * axes_per_pair + offset
        for offset in group_offsets
        for pair_idx in range(num_pairs)
    ]
