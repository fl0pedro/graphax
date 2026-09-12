from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Index:
    """Unified dim descriptor.

    Discriminator: ``other_id is None`` ⇒ dense; otherwise sparse-paired.

    ``block_size`` splits the dim's ``logical_size`` into ``size`` blocks of
    ``block_size`` positions each, and ``block_axis`` says whether ``val``
    carries the block physically. It is meaningful in TWO forms:

    * a SPARSE PAIR (``other_id`` set): block-diagonal storage, where the two
      partners' blocks are the rows and columns of each diagonal block;
    * a BLOCKED DENSE dim (``other_id is None``, ``block_axis is None``, ticket
      dsnn-3qm.62): ``val`` stores ONE entry per block along ``axis`` and the
      block extent is IMPLICIT — the value is uniform inside each block, and
      ``dense()`` expands it like any other implicit extent. This is what a
      COMPRESS of the val axis of a DIAG'd block leaves behind, and what a
      contraction emits for the survivor of such a pair: ``logical_size`` stays
      the dim's true extent while ``val.shape[axis] == size``.

    ``block_axis`` without ``other_id`` ("implicit outer, explicit inner") is
    NOT a form: ``axis`` is the outer pointer, so there is nothing for the outer
    extent to hang off. Both stay ``None`` for a plain dense dim and for a plain
    (pure-diagonal) sparse pair.

    The ``DenseIndex`` and ``DiagonalIndex`` factory functions below construct
    this class with the appropriate field set. These two are the ONLY index
    forms. An axis is additionally explicit (it owns a physical ``val`` axis)
    or implicit (``axis is None``, one copy stored).

    ``BandedIndex``, ``SetIndex``, ``ToeplitzIndex`` and their base
    ``CompressedIndex`` are gone (ruling 2026-09-07). Over 88 runs on 17
    targets, both engines, both orders, exact and approximated, no target ever
    built a band or a set, and only an explicit convolution built a Toeplitz.
    A misaligned contraction now falls back to the least-common-multiple meta
    grid and a convolution Jacobian is stored dense.
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
        if self.other_id is None and self.block_axis is not None:
            raise ValueError(
                f"Index {self.id}: block_axis={self.block_axis} with "
                f"other_id=None. A dim with no partner can only carry an "
                f"IMPLICIT block (see the class docstring): ``axis`` is the "
                f"outer pointer, so an explicit block axis would leave the "
                f"outer extent nothing to hang off."
            )

    @property
    def is_sparse(self) -> bool:
        return self.other_id is not None

    @property
    def is_blocked_dense(self) -> bool:
        """The BLOCKED DENSE form (ticket dsnn-3qm.62): no partner, yet a block.

        ``val`` stores ONE entry per block along ``axis`` and the ``block_size``
        positions inside each block are IMPLICIT — uniform, expanded by
        ``dense()`` like any other implicit extent. So ``val.shape[axis] ==
        size`` while the dim spans ``logical_size``, and any reader that takes
        ``size`` for the whole extent (or checks ``is_sparse`` before looking at
        the block fields at all) needs this."""
        return self.other_id is None and self.block_size is not None

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


# --------------------------------------------------------------------------- #
# Static routing
# --------------------------------------------------------------------------- #
def static_eye(n: int, dtype):
    """The ``n`` x ``n`` identity as a COMPILE-TIME constant.

    Every caller uses an identity to express *routing*: which sub-block of a
    block-diagonal operand lands on which slot of a coarser or finer grid. The
    grid is known when the jaxpr is built, so the identity is known too.

    ``jnp.eye`` does not say that. It traces to ``iota``, ``iota`` and ``eq``,
    which stay in the jaxpr as live equations, and the routing then reads as a
    value XLA has to compute rather than a layout it can fold. The numpy form
    is a literal, so the routing is static. This is the same choice
    ``matmul._reduce_grid`` already makes for its one-hot fold.

    ``n`` is a block-count ratio in every caller (a meta ratio or an expansion
    factor), so the baked constant is small.
    """
    try:
        return np.eye(n, dtype=dtype)
    except TypeError:
        # Extended dtypes (bfloat16 and friends) that numpy cannot construct
        # directly still accept a cast from the default float form.
        return np.eye(n).astype(dtype)
