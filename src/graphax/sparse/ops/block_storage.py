"""Dense-storage representations for tensor structures that arise from block-sparse ops.

These are the **outputs** of elementwise / matmul ops between two block-diagonal
sources whose block sizes can disagree. They are *not* a single rectangular
tensor — they're pytrees of varying-sized buffers (one per source contribution).
The only thing that lets us pack them densely is the GCD/LCM periodicity:
when total dims are integer multiples of the LCM, the per-LCM-block pattern
repeats, giving a "depth" / "batch" axis ``M`` along which all the differently-
shaped buffers can be stacked.

Three structures, one per algorithmic case:

1. :class:`UnionBlocks` — output of an elementwise *union* op (e.g. ``add``)
   between two block-diagonal sources with different block sizes:

       lhs = block_diag(11×(5,10))      rhs = block_diag(5×(11,22))
       result = lhs + rhs               # interleaved diagonals, see demo

   Stored as two separate per-source buffers ``(M, n, B_h, B_w, *L)``. Densify
   places each source on its own meta-block-diagonal, then sums.

2. :class:`IntersectionBlocks` — output of an elementwise *intersection* op
   (e.g. ``mul``) between the same two sources:

       result = lhs * rhs               # only positions where BOTH have data

   Same dual-buffer storage as ``UnionBlocks``; densify multiplies (or applies
   any user-supplied binary op) so the result is non-fill only where both sides
   were non-fill.

3. :class:`BlockBanded` — output of a *matmul* of two such block-diagonals
   (``x @ y.T`` or ``x.T @ y`` in the user's example): the result is non-zero
   only within a finite block-band around the diagonal, since contractions
   couple block ``i`` of the lhs only to nearby blocks of the rhs.

   Stored in skewed rectangular form (``(M, 2w+1, B, B, *L)``) — the bands are
   shifted into one tensor.

XLA-fusion notes
----------------
``to_dense`` for each is a single broadcast/select/where chain — no scatter,
gather only where the band lookup makes it strictly necessary. Rendered HLO
shows one ``broadcast_select_fusion`` per source plus the final mask, all of
which XLA can fold into a downstream consumer.

NamedTuples are JAX pytrees out of the box, so all three round-trip through
``jit`` / ``vmap`` / ``grad`` without registration.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax


_BLOCK_BANDED_BROADCAST_LIMIT = 10**8


# ----------------------------------------------------------------------------
#  Shared helper: place an (M, n, Bh, Bw, *L) buffer onto its block-diagonal
# ----------------------------------------------------------------------------
def _block_diag_per_meta(blocks: Array, fill: Array) -> Array:
    """Materialize the dense form of one source's block-diagonal contribution.

    ``blocks`` has shape ``(M, n, Bh, Bw, *L)`` — within each of ``M`` meta-
    blocks, ``n`` rectangular ``(Bh, Bw)`` blocks sit on the diagonal. Returns
    shape ``(M, n*Bh, n*Bw, *L)`` per-meta-block-diagonal grid (we still keep
    the ``M`` axis unflattened — the caller stitches that into a meta-block-
    diagonal of meta-blocks).

    Implemented as ``broadcast → reshape → where``, no gather, no scatter — a
    single ``kLoop`` fusion that XLA can fold into the consuming kernel.
    """
    M, n, Bh, Bw, *L = blocks.shape
    NH, NW = n * Bh, n * Bw
    # blocks_2d[m, i, k, *l] == blocks[m, i // Bh, i % Bh, k, *l]
    blocks_2d = blocks.reshape(M, NH, Bw, *L)
    # Tile across the inner axis: gathered[m, i, j, *l] == blocks[m, i // Bh, i % Bh, j % Bw, *l]
    blocks_3d = jnp.broadcast_to(blocks_2d[:, :, None, ...], (M, NH, n, Bw, *L))
    gathered = blocks_3d.reshape(M, NH, NW, *L)
    # Diagonal mask: keep only on-block positions (i // Bh == j // Bw).
    blk_o = jnp.arange(NH) // Bh
    blk_i = jnp.arange(NW) // Bw
    mask = blk_o[:, None] == blk_i[None, :]
    if L:
        mask = mask[(..., *((None,) * len(L)))]
    return jnp.where(mask, gathered, fill)


def _to_meta_blocks(blocks_obj, op_default: Callable) -> Array:
    """Shared meta-block materialization for ``UnionBlocks`` and
    ``IntersectionBlocks``.

    Both store dual ``(M, n, Bh, Bw, *L)`` buffers and combine them via a
    binary ``op`` whose only difference is the default (``jnp.add`` vs
    ``jnp.multiply``). ``op_default`` documents that default at the call
    site; the runtime op comes from ``blocks_obj.op``.
    """
    del op_default
    lhs_meta = _block_diag_per_meta(blocks_obj.lhs, blocks_obj.fill_lhs)
    rhs_meta = _block_diag_per_meta(blocks_obj.rhs, blocks_obj.fill_rhs)
    return blocks_obj.op(lhs_meta, rhs_meta)


def _stitch_meta(per_meta: Array, fill: Array) -> Array:
    """Place an ``(M, H, W, *L)`` per-meta-block tensor onto the meta-block-
    diagonal of the full ``(M*H, M*W, *L)`` dense form. Same broadcast+select
    trick — no scatter, no gather."""
    M, H, W, *L = per_meta.shape
    NMH, NMW = M * H, M * W
    per_2d = per_meta.reshape(NMH, W, *L)
    per_3d = jnp.broadcast_to(per_2d[:, None, ...], (NMH, M, W, *L))
    gathered = per_3d.reshape(NMH, NMW, *L)
    meta_o = jnp.arange(NMH) // H
    meta_i = jnp.arange(NMW) // W
    mask = meta_o[:, None] == meta_i[None, :]
    if L:
        mask = mask[(..., *((None,) * len(L)))]
    return jnp.where(mask, gathered, fill)


# ----------------------------------------------------------------------------
#  1.  UnionBlocks  — elementwise union (e.g. add)
# ----------------------------------------------------------------------------
class UnionBlocks(NamedTuple):
    """Output of an elementwise *union* op between two block-diagonal sources
    with potentially different block sizes.

    Layout
    ------
    ``lhs`` : ``(M, n_lhs, B_lhs_h, B_lhs_w, *L)``
        ``n_lhs`` lhs blocks of shape ``(B_lhs_h, B_lhs_w)`` per meta-block,
        sitting on the meta-block's block-diagonal.
    ``rhs`` : ``(M, n_rhs, B_rhs_h, B_rhs_w, *L)``
        Same idea for the rhs source. Block sizes need not match the lhs.
    ``fill_lhs`` / ``fill_rhs`` : the original sources' fill values (the
        ``op(fill_lhs, fill_rhs)`` is the result's fill outside both diagonals).
    ``op`` : the binary union op (default ``jnp.add``).

    Constraints (per-meta-block consistency):
        ``n_lhs * B_lhs_h == n_rhs * B_rhs_h == LCM_h``
        ``n_lhs * B_lhs_w == n_rhs * B_rhs_w == LCM_w``

    Dense form: ``(M*LCM_h, M*LCM_w, *L)``.

    Combined-buffer alternative
    ---------------------------
    :meth:`combined` packs ``(lhs.flatten(), rhs.flatten())`` into a single 1-D
    buffer (with metadata to reconstruct shapes) for callers who want a single
    array; :meth:`from_combined` is the inverse.
    """

    lhs: Array
    rhs: Array
    fill_lhs: Array
    fill_rhs: Array
    op: Callable = jnp.add

    @property
    def lcm_h(self) -> int:
        _, n_lhs, B_lhs_h, *_ = self.lhs.shape
        return n_lhs * B_lhs_h

    @property
    def lcm_w(self) -> int:
        _, n_lhs, _, B_lhs_w, *_ = self.lhs.shape
        return n_lhs * B_lhs_w

    @property
    def shape(self) -> tuple[int, ...]:
        M = self.lhs.shape[0]
        L = self.lhs.shape[4:]
        return (M * self.lcm_h, M * self.lcm_w, *L)

    @property
    def meta_block_shape(self) -> tuple[int, int, int]:
        """``(M, LCM_h, LCM_w)`` — the natural block-diagonal layout of this
        compressed form: M meta-blocks each ``(LCM_h, LCM_w)``, sitting on the
        meta-block-diagonal of the dense ``(M*LCM_h, M*LCM_w)`` shape."""
        return (self.lhs.shape[0], self.lcm_h, self.lcm_w)

    def to_meta_blocks(self) -> Array:
        """Materialize the M per-meta-block contributions as a single
        ``(M, LCM_h, LCM_w, *L)`` tensor — the *meta-block-diagonal* values
        without the surrounding zero-padding of the full dense form.

        This is the storage that lets a ``SparseTensor`` represent the union
        as a meta-block-diagonal pair (``SparseIndex(M, block_size=LCM_*)``)
        — M× less storage than ``to_dense`` and lets every downstream sparse
        op (matmul / elementwise) leverage the block-diagonal fast path
        instead of touching the M²-many zero meta-blocks."""
        return _to_meta_blocks(self, jnp.add)

    def to_dense(self) -> Array:
        """Materialize the *fully* dense ``(M*LCM_h, M*LCM_w, *L)`` form.

        Builds the M per-meta-block grids via :meth:`to_meta_blocks` then
        stitches them onto the meta-block-diagonal with ``op(fill_lhs,
        fill_rhs)`` outside. Most consumers should prefer :meth:`to_meta_blocks`
        + a meta-block-diagonal ``SparseTensor`` wrapper, which is M× cheaper
        in storage and unlocks the sparse fast paths.
        """
        per_meta = self.to_meta_blocks()
        return _stitch_meta(per_meta, self.op(self.fill_lhs, self.fill_rhs))

    def combined(self) -> tuple[Array, tuple]:
        """Pack ``(lhs, rhs)`` flat into a single 1-D buffer, with shape metadata
        to reconstruct via :meth:`from_combined`."""
        flat = jnp.concatenate([self.lhs.reshape(-1), self.rhs.reshape(-1)])
        meta = (self.lhs.shape, self.rhs.shape)
        return flat, meta

    @classmethod
    def from_combined(
        cls,
        flat: Array,
        meta: tuple,
        fill_lhs: Array,
        fill_rhs: Array,
        op: Callable = jnp.add,
    ) -> "UnionBlocks":
        lhs_shape, rhs_shape = meta
        lhs_size = math.prod(lhs_shape)
        return cls(
            lhs=flat[:lhs_size].reshape(lhs_shape),
            rhs=flat[lhs_size:].reshape(rhs_shape),
            fill_lhs=fill_lhs,
            fill_rhs=fill_rhs,
            op=op,
        )


# ----------------------------------------------------------------------------
#  2.  IntersectionBlocks — elementwise intersection (e.g. mul)
# ----------------------------------------------------------------------------
class IntersectionBlocks(NamedTuple):
    """Output of an elementwise *intersection* op between two block-diagonal sources.

    Same dual-buffer layout as :class:`UnionBlocks` (shapes / constraints
    identical). The only difference is the densification semantics: positions
    where *both* sources have stored data take ``op(lhs, rhs)``, positions
    where *exactly one* has data fall back to ``op(value, other_fill)``, and
    positions where *neither* has data take ``op(fill_lhs, fill_rhs)``.

    For the canonical intersection op ``op = jnp.multiply`` and zero fills,
    only positions that lie inside *both* lhs's and rhs's block-diagonals end
    up non-zero — the small diamonds you see in the user's ``x * y`` plot.
    """

    lhs: Array
    rhs: Array
    fill_lhs: Array
    fill_rhs: Array
    op: Callable = jnp.multiply

    @property
    def lcm_h(self) -> int:
        _, n_lhs, B_lhs_h, *_ = self.lhs.shape
        return n_lhs * B_lhs_h

    @property
    def lcm_w(self) -> int:
        _, n_lhs, _, B_lhs_w, *_ = self.lhs.shape
        return n_lhs * B_lhs_w

    @property
    def shape(self) -> tuple[int, ...]:
        M = self.lhs.shape[0]
        L = self.lhs.shape[4:]
        return (M * self.lcm_h, M * self.lcm_w, *L)

    @property
    def meta_block_shape(self) -> tuple[int, int, int]:
        return (self.lhs.shape[0], self.lcm_h, self.lcm_w)

    def to_meta_blocks(self) -> Array:
        """``(M, LCM_h, LCM_w, *L)`` per-meta-block contributions; see
        :meth:`UnionBlocks.to_meta_blocks` — only the binary ``op`` differs."""
        return _to_meta_blocks(self, jnp.multiply)

    def to_dense(self) -> Array:
        """Fully-dense ``(M*LCM_h, M*LCM_w, *L)`` form. Prefer
        :meth:`to_meta_blocks` + meta-block-diagonal ``SparseTensor`` storage."""
        per_meta = self.to_meta_blocks()
        return _stitch_meta(per_meta, self.op(self.fill_lhs, self.fill_rhs))


def _to_dense_banded(
    data: Array,
    M_primary: int,
    M_secondary: int,
    W: int,
    B_p: int,
    B_s: int,
    offset_tuple: tuple[int, ...] | None,
    centered_w: int | None,
    fill_value: Array,
    L: tuple[int, ...],
    L_pad: tuple,
) -> Array:
    """Shared row-primary banded-densify kernel for :class:`BlockBanded`.

    Treats ``data`` as ``(M_primary, W, B_p, B_s, *L)`` with ``data[a, w_idx]``
    sitting at primary-meta ``a``, secondary-meta ``offset[a] + w_idx``.
    Returns ``(M_primary * B_p, M_secondary * B_s, *L)``.

    Two offset modes (mutually exclusive):
      • ``centered_w`` given (``offset_tuple is None``): arithmetic centered
        band, ``offset[a] = a - centered_w``. All masks/in-band checks are
        pure arithmetic on ``jnp.arange`` — XLA emits no gather. This is the
        legacy fast path.
      • ``offset_tuple`` given: explicit per-primary offsets. The in-band
        mask becomes ``offset_arr[blk_i]``, an unavoidable gather. Use this
        for staircase / rectangular / non-centered bands.

    For col-primary :class:`BlockBanded` the caller passes the sub-block
    axes swapped and transposes the result at the boundary (see
    ``BlockBanded.to_dense``).
    """
    # Step 1: tile data across a new secondary-axis (pure broadcast, no copy).
    data_bcast = jnp.broadcast_to(
        data[:, None, ...],  # (M_primary, 1, W, B_p, B_s, *L)
        (M_primary, M_secondary, W, B_p, B_s, *L),
    )

    # Step 2: build one-hot ``w_idx == b - offset[a]`` selector along W.
    bj_idx = jnp.arange(M_secondary, dtype=jnp.int32)  # (M_secondary,)
    if centered_w is not None:
        bi_idx = jnp.arange(M_primary, dtype=jnp.int32)[:, None]
        target_w = bj_idx[None, :] - bi_idx + centered_w  # arithmetic, no gather
    else:
        off_arr = jnp.asarray(offset_tuple, dtype=jnp.int32)  # (M_primary,)
        target_w = bj_idx[None, :] - off_arr[:, None]
    select_mask = (
        target_w[:, :, None] == jnp.arange(W, dtype=jnp.int32)[None, None, :]
    )
    select_mask = select_mask[..., None, None]  # (M_primary, M_secondary, W, 1, 1)
    if L:
        select_mask = select_mask[(..., *L_pad)]

    # Step 3: where+sum collapses W → out_meta has data on the band, 0 elsewhere.
    out_meta = jnp.where(select_mask, data_bcast, 0).sum(
        axis=2
    )  # (M_primary, M_secondary, B_p, B_s, *L)

    # Step 4: interleave (a, si, b, sj) → (a*B_p+si, b*B_s+sj).
    perm = (0, 2, 1, 3, *range(4, 4 + len(L)))
    out_meta = out_meta.transpose(perm).reshape(M_primary * B_p, M_secondary * B_s, *L)

    # Step 5: replace the zero-pad outside the band with fill_value.
    blk_i = jnp.arange(M_primary * B_p) // B_p
    blk_j = jnp.arange(M_secondary * B_s) // B_s
    if centered_w is not None:
        # offset[i] = i - centered_w. diff = j - (i - w) = j - i + w. No gather.
        diff = blk_j[None, :] - blk_i[:, None] + centered_w
    else:
        off_arr_for_band = jnp.asarray(offset_tuple, dtype=jnp.int32)
        co_per_row = off_arr_for_band[blk_i]  # gather (unavoidable for explicit)
        diff = blk_j[None, :] - co_per_row[:, None]
    in_band = (diff >= 0) & (diff < W)
    if L:
        in_band = in_band[(..., *L_pad)]
    return jnp.where(in_band, out_meta, fill_value)


# ----------------------------------------------------------------------------
#  3.  BlockBanded — matmul of two block-diagonals (x @ y.T or x.T @ y)
# ----------------------------------------------------------------------------
class BlockBanded(NamedTuple):
    """Block-banded matrix with optional rectangular sub-blocks, asymmetric
    meta-counts, and row/col-primary orientation.

    Output of contracting two block-diagonal sources where the contracting
    axes' block sizes don't divide evenly: the result is non-zero only within
    a finite block-band. The band's natural orientation (along rows or along
    cols) depends on which operand has more meta-blocks — matmul picks the
    tighter orientation at emission time.

    Layout
    ------
    ``data`` : ``(M_primary, W, B_row, B_col, *L)``
        ``data[a, w_idx]`` is the ``(B_row, B_col)`` sub-block at meta-coord
        determined by ``primary_axis``:
          • ``primary_axis=0`` (row-primary): row meta ``a``, col meta
            ``offset[a] + w_idx``.
          • ``primary_axis=1`` (col-primary): col meta ``a``, row meta
            ``offset[a] + w_idx``.
        ``W`` is the band width (max in-band sub-blocks per primary slot).
    ``fill_value`` : value at positions outside the band.
    ``primary_axis`` : ``0`` (default; row-primary) or ``1`` (col-primary).
    ``n_secondary`` : total meta-blocks along the secondary axis. ``-1``
        sentinel = ``M_primary`` (square / legacy).
    ``offset`` : per-primary integer offsets along secondary. ``()`` sentinel
        = centered band: ``offset[a] = a - (W - 1) // 2`` (legacy behavior).

    Backward-compat
    ---------------
    ``BlockBanded(data=(M, 2w+1, B, B, *L), fill_value=...)`` with the new
    fields at their defaults reproduces the legacy symmetric centered square
    band: ``primary_axis=0``, ``n_secondary=M``, ``offset[a]=a-w``,
    ``B_row=B_col=B``.

    Generalized cases (post Phase 5d)
    --------------------------------
    Rectangular sub-blocks (``B_row != B_col``), asymmetric meta-counts
    (``n_secondary != M_primary``), skewed/staircase bands (explicit
    ``offset``), and col-primary orientation are all encoded by setting the
    new fields. These shapes arise from misaligned-contract matmuls where the
    contracting block sizes have non-trivial LCM/GCD ratios.
    """

    data: Array
    fill_value: Array
    primary_axis: int = 0
    n_secondary: int = -1
    offset: tuple[int, ...] = ()

    @property
    def _M_primary(self) -> int:
        return self.data.shape[0]

    @property
    def _W(self) -> int:
        return self.data.shape[1]

    @property
    def _B_row(self) -> int:
        return self.data.shape[2]

    @property
    def _B_col(self) -> int:
        return self.data.shape[3]

    @property
    def _M_secondary(self) -> int:
        return self.n_secondary if self.n_secondary >= 0 else self._M_primary

    @property
    def _offset_arr(self) -> tuple[int, ...]:
        if self.offset:
            return self.offset
        # Centered-band sentinel: offset[a] = a - (W-1)//2.
        w = (self._W - 1) // 2
        return tuple(a - w for a in range(self._M_primary))

    @property
    def _M_row(self) -> int:
        return self._M_primary if self.primary_axis == 0 else self._M_secondary

    @property
    def _M_col(self) -> int:
        return self._M_secondary if self.primary_axis == 0 else self._M_primary

    @property
    def half_bandwidth(self) -> int:
        """Legacy property: ``(W-1)//2`` for centered symmetric bands. For
        non-centered / staircase / rectangular bands this concept doesn't
        apply uniformly — callers should consult ``data.shape[1]`` (W) and
        ``offset`` instead.
        """
        return (self._W - 1) // 2

    @property
    def shape(self) -> tuple[int, ...]:
        L = self.data.shape[4:]
        return (self._M_row * self._B_row, self._M_col * self._B_col, *L)

    @property
    def meta_block_shape(self) -> tuple[int, int, int] | None:
        """``(M, B_row, B_col)`` when this is a pure meta-block-diagonal:
        ``W=1`` + square meta-counts (``M_row == M_col``) + identity offset
        (``offset[a] = a``). Returns ``None`` for any banded / rectangular /
        skewed form (those can't be expressed as a single meta-block-diagonal
        SparseTensor pair).

        Note: square sub-blocks are NOT required — the existing block-diagonal
        SparseTensor wrapper supports rectangular block_size pairs.
        """
        if self._W != 1:
            return None
        if self._M_row != self._M_col:
            return None
        # Check identity offset: data[a, 0] sits at primary-meta = secondary-meta = a.
        off = self._offset_arr
        if tuple(off) != tuple(range(self._M_primary)):
            return None
        return (self._M_primary, self._B_row, self._B_col)

    def to_meta_blocks(self) -> Array:
        """``(M, B_row, B_col, *L)`` meta-diagonal blocks. Only defined when
        this is a pure meta-block-diagonal (see :py:meth:`meta_block_shape`);
        for any banded / rectangular / skewed form use :meth:`to_dense`."""
        if self.meta_block_shape is None:
            raise ValueError(
                "BlockBanded.to_meta_blocks requires W=1 + square meta-counts "
                f"+ identity offset; got W={self._W}, M_primary={self._M_primary}, "
                f"M_secondary={self._M_secondary}, offset={self._offset_arr}. "
                "Use to_dense() for banded forms."
            )
        return self.data[:, 0]   # (M, B_row, B_col, *L)

    def to_dense(self) -> Array:
        """Materialize the dense ``(M_row*B_row, M_col*B_col, *L)`` form via a
        single fused broadcast+select+sum chain — no scatter, **no gather**.

        Algorithm. For each output meta-position ``(a, b)`` (primary, secondary)
        the in-band slot is ``w_idx = b - offset[a]``; if ``w_idx ∈ [0, W)``
        the sub-block is ``data[a, w_idx]``, else ``fill_value``. We avoid the
        per-cell index lookup (which would be a 1-D fancy gather) by:

          1. broadcasting ``data`` across a new secondary-axis to logical
             shape ``(M_primary, M_secondary, W, B_row, B_col, *L)``
             (pure broadcast, zero-copy);
          2. masking with a one-hot ``w_idx == b - offset[a]`` selector;
          3. summing over W to collapse the band axis.

        Since exactly one ``w_idx`` matches per in-band ``(a, b)`` (and zero
        match out-of-band), the sum is equivalent to a per-cell select. XLA
        fuses the broadcast+where+sum into one ``kLoop`` pass — no gather,
        no scatter, no W× HBM materialization. That makes the densify
        forward-fusable into a downstream consumer kernel (gathers force a
        materialization barrier; this no longer does).

        For ``primary_axis=1`` (col-primary) we reuse the same kernel by
        swapping the sub-block axes inside ``data`` and transposing the
        final output — two cheap reshape/transpose ops on top of the same
        fused chain.
        """
        M_primary, W, B_row, B_col, *L = self.data.shape
        M_secondary = self._M_secondary
        L_pad = (None,) * len(L)

        # Two offset modes: centered (no gather) vs explicit (one unavoidable gather).
        if self.offset:
            offset_tuple = self.offset
            centered_w = None
            off_for_fallback = self.offset
        else:
            offset_tuple = None
            centered_w = (W - 1) // 2
            off_for_fallback = tuple(a - centered_w for a in range(M_primary))

        if (
            math.prod((M_primary, M_secondary, W, B_row, B_col, *L))
            > _BLOCK_BANDED_BROADCAST_LIMIT
        ):
            return self._to_dense_per_band(
                M_primary, M_secondary, W, B_row, B_col, off_for_fallback, L
            )

        if self.primary_axis == 0:
            # Row-primary: M_primary = M_row, M_secondary = M_col.
            return _to_dense_banded(
                self.data, M_primary, M_secondary, W, B_row, B_col,
                offset_tuple, centered_w, self.fill_value, L, L_pad,
            )

        # Col-primary: data[b, w] is at (row meta off[b]+w, col meta b). Reuse
        # the row-primary kernel on the transposed view (B_col, B_row sub-blocks
        # swapped) producing (M_primary*B_col, M_secondary*B_row, *L) which is
        # the TRANSPOSE of the desired dense; swap back at the end.
        data_t = self.data.swapaxes(2, 3)  # (M_primary, W, B_col, B_row, *L)
        dense_T = _to_dense_banded(
            data_t, M_primary, M_secondary, W, B_col, B_row,
            offset_tuple, centered_w, self.fill_value, L, L_pad,
        )
        return dense_T.swapaxes(0, 1)

    def _to_dense_per_band(
        self,
        M_primary: int,
        M_secondary: int,
        W: int,
        B_row: int,
        B_col: int,
        offset: tuple[int, ...],
        L: tuple[int, ...],
    ) -> Array:
        """Per-band ``lax.fori_loop`` fallback for the dense materialization.

        The broadcast path's logical intermediate is
        ``(M_primary, M_secondary, W, B_row, B_col, *L)`` — for large M the
        static shape exceeds practical limits even though XLA fuses the
        ``where+sum`` at runtime. This path iterates the primary slots and
        writes their W in-band sub-blocks via ``lax.dynamic_update_slice``,
        never materializing the ``(M_primary, M_secondary)`` square.

        Handles both ``primary_axis=0`` (row-primary) and ``primary_axis=1``
        (col-primary) — for col-primary the write coordinates swap.
        """
        M_row, M_col = (
            (M_primary, M_secondary)
            if self.primary_axis == 0
            else (M_secondary, M_primary)
        )
        out = jnp.full(
            (M_row * B_row, M_col * B_col, *L), self.fill_value, dtype=self.data.dtype
        )
        zero_idx = (jnp.int32(0),) * len(L)
        # ``offset[k]`` is read inside ``fori_loop`` where ``k`` is a tracer —
        # the Python tuple has to become a JAX array so indexing works.
        off_arr = jnp.asarray(offset, dtype=jnp.int32)
        row_primary = self.primary_axis == 0

        def body(k, acc):
            base = off_arr[k]
            for b in range(W):
                sec = base + b
                if row_primary:
                    in_range = jnp.logical_and(sec >= 0, sec < M_col)
                    row_start = jnp.int32(k * B_row)
                    col_start = jnp.int32(sec * B_col)
                else:
                    in_range = jnp.logical_and(sec >= 0, sec < M_row)
                    row_start = jnp.int32(sec * B_row)
                    col_start = jnp.int32(k * B_col)
                block = self.data[k, b]
                existing = lax.dynamic_slice(
                    acc, (row_start, col_start) + zero_idx, (B_row, B_col, *L)
                )
                replacement = jnp.where(in_range, block, existing)
                acc = lax.dynamic_update_slice(
                    acc, replacement, (row_start, col_start) + zero_idx
                )
            return acc

        return lax.fori_loop(0, M_primary, body, out)


# ----------------------------------------------------------------------------
#  PyTree registration: ``op`` is static (Callable, hashable), buffers are leaves
# ----------------------------------------------------------------------------
# Auto-NamedTuple flattening would put ``op`` in the children list, which
# breaks ``jax.jit`` round-trips (functions aren't valid jax types). Override
# with a custom split that pushes ``op`` into the static aux_data — same
# treatment elementwise.py uses for its op closures.
def _blocks_flatten(ub):
    return (ub.lhs, ub.rhs, ub.fill_lhs, ub.fill_rhs), (ub.op,)


def _union_unflatten(aux, children):
    return UnionBlocks(*children, op=aux[0])


def _intersection_unflatten(aux, children):
    return IntersectionBlocks(*children, op=aux[0])


jax.tree_util.register_pytree_node(UnionBlocks, _blocks_flatten, _union_unflatten)
jax.tree_util.register_pytree_node(IntersectionBlocks, _blocks_flatten, _intersection_unflatten)


def _block_banded_flatten(bb):
    """Flatten ``BlockBanded`` for pytree traversal. Two array leaves; the
    static metadata (``primary_axis``, ``n_secondary``, ``offset``) goes into
    aux_data so JIT treats it as compile-time constant."""
    return (bb.data, bb.fill_value), (bb.primary_axis, bb.n_secondary, bb.offset)


def _block_banded_unflatten(aux, children):
    return BlockBanded(
        *children,
        primary_axis=aux[0],
        n_secondary=aux[1],
        offset=aux[2],
    )


jax.tree_util.register_pytree_node(
    BlockBanded, _block_banded_flatten, _block_banded_unflatten
)
