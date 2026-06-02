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

    The leading ``M`` axis is ``blocks_obj.n_meta * M_per_batch`` —
    ``_block_diag_per_meta`` processes all batches concurrently because
    the per-batch diagonals concatenate naturally along the leading axis.
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
    n_meta: int = 1

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
        """Full dense output shape ``(M*LCM_h, M*LCM_w, *L)``. ``M`` is the
        total meta-block count (= ``n_meta * M_per_batch``); the per-batch
        diagonals concatenate seamlessly along the main meta-diagonal."""
        M = self.lhs.shape[0]
        L = self.lhs.shape[4:]
        return (M * self.lcm_h, M * self.lcm_w, *L)

    @property
    def meta_block_shape(self) -> tuple[int, int, int]:
        """``(M, LCM_h, LCM_w)`` — natural meta-block-diagonal layout. For
        ``n_meta > 1`` the ``M`` axis is ``n_meta * M_per_batch`` flat;
        downstream consumers that need per-batch indexing reshape via
        ``self.n_meta`` to recover ``(n_meta, M_per_batch, ...)``."""
        return (self.lhs.shape[0], self.lcm_h, self.lcm_w)

    def to_meta_blocks(self) -> Array:
        """Materialize the M per-meta-block contributions as a single
        ``(M, LCM_h, LCM_w, *L)`` tensor — the *meta-block-diagonal* values
        without the surrounding zero-padding of the full dense form.

        This is the storage that lets a ``SparseTensor`` represent the union
        as a meta-block-diagonal pair (``DiagonalIndex(M, block_size=LCM_*)``)
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
        n_meta: int = 1,
    ) -> "UnionBlocks":
        lhs_shape, rhs_shape = meta
        lhs_size = math.prod(lhs_shape)
        return cls(
            lhs=flat[:lhs_size].reshape(lhs_shape),
            rhs=flat[lhs_size:].reshape(rhs_shape),
            fill_lhs=fill_lhs,
            fill_rhs=fill_rhs,
            op=op,
            n_meta=n_meta,
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
    n_meta: int = 1

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
        """Full dense output shape. ``M = n_meta * M_per_batch`` is the
        flattened total meta-block count along the leading axis."""
        M = self.lhs.shape[0]
        L = self.lhs.shape[4:]
        return (M * self.lcm_h, M * self.lcm_w, *L)

    @property
    def meta_block_shape(self) -> tuple[int, int, int]:
        """``(M, LCM_h, LCM_w)`` — same as :class:`UnionBlocks`. For
        ``n_meta > 1`` the leading axis is ``n_meta * M_per_batch`` flat."""
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


# ----------------------------------------------------------------------------
#  2b.  DivisorRemainder — unified primitive (subsumes UnionBlocks /
#       IntersectionBlocks with an explicit semantic + ``include_remainder`` flag)
# ----------------------------------------------------------------------------
class DivisorRemainder(NamedTuple):
    """Unified compressed-storage primitive for elementwise op outputs.

    Subsumes :class:`UnionBlocks` (``semantic='union'``) and
    :class:`IntersectionBlocks` (``semantic='intersection'``) under one type
    with explicit semantic + a static ``include_remainder`` flag.

    Layout
    ------
    ``divisor`` : ``(M, n_d, B_d_h, B_d_w, *L)``
        Per-meta-block content for the "primary" side. Naming mirrors the
        algebraic intuition: this is the "intersection-of-supports" content
        the op consumes at every positioned cell.
    ``remainder`` : ``(M, n_r, B_r_h, B_r_w, *L) | None``
        Per-meta-block content for the "secondary" side. ``None`` when
        ``include_remainder=False`` — the densify path then folds in
        ``fill_remainder`` as a scalar overlay, no buffer materialized.
    ``fill_divisor`` / ``fill_remainder`` : scalars used outside each side's
        meta-block-diagonal.
    ``semantic`` : ``'union'`` | ``'intersection'`` | ``'custom'``
        Controls densification:
          * ``'union'``: ``out = op(divisor_grid, remainder_grid)`` (or
            ``op(divisor_grid, fill_remainder)`` when ``include_remainder=False``).
          * ``'intersection'``: at positions where both sides have stored
            data, ``op(divisor, remainder)``; at positions where only one
            side has data, ``op(value, other_fill)``; elsewhere
            ``op(fill_divisor, fill_remainder)``. ``op`` is typically
            ``jnp.multiply`` with zero fills.
          * ``'custom'``: callers operate on the two parts independently
            (rare; the SparseTensor wrapper exposes ``divisor`` /
            ``remainder`` via ``compressed_val`` for them to inspect).
    ``include_remainder`` : ``bool`` (static field, JIT-constant)
        When ``False``: ``remainder`` is ``None``, saves HBM AND trace-time
        work. Set by emission probes when one side is provably-zero or
        equal to fill.
    ``op`` : the binary op (default ``jnp.add`` matching the legacy
        ``UnionBlocks`` default).

    Backward-compat
    ---------------
    The classic dual-buffer ``UnionBlocks(lhs, rhs, fill_lhs, fill_rhs, op)``
    and ``IntersectionBlocks(...)`` classes are kept (separate NamedTuples
    at the top of this module) so existing isinstance checks and
    field accesses continue to work. ``DivisorRemainder`` is used by new
    emission paths going forward; the old classes will be deleted in a
    cleanup pass after all call sites migrate.

    Constraints (same as :class:`UnionBlocks`, mutatis mutandis):
        ``n_d * B_d_h == n_r * B_r_h == LCM_h``
        ``n_d * B_d_w == n_r * B_r_w == LCM_w``
    """

    divisor: Array
    remainder: Array | None
    fill_divisor: Array
    fill_remainder: Array
    semantic: str = "union"
    include_remainder: bool = True
    op: Callable = jnp.add
    n_meta: int = 1

    @property
    def lcm_h(self) -> int:
        _, n_d, B_d_h, *_ = self.divisor.shape
        return n_d * B_d_h

    @property
    def lcm_w(self) -> int:
        _, n_d, _, B_d_w, *_ = self.divisor.shape
        return n_d * B_d_w

    @property
    def shape(self) -> tuple[int, ...]:
        """Full dense output shape. ``M = n_meta * M_per_batch`` is the
        flattened total meta-block count along the leading axis; per-batch
        diagonals concatenate seamlessly along the main meta-diagonal."""
        M = self.divisor.shape[0]
        L = self.divisor.shape[4:]
        return (M * self.lcm_h, M * self.lcm_w, *L)

    @property
    def meta_block_shape(self) -> tuple[int, int, int]:
        """``(M, LCM_h, LCM_w)`` — natural meta-block-diagonal layout. For
        ``n_meta > 1`` the leading axis is ``n_meta * M_per_batch`` flat;
        downstream consumers reshape via ``self.n_meta`` for per-batch
        indexing."""
        return (self.divisor.shape[0], self.lcm_h, self.lcm_w)

    def to_meta_blocks(self) -> Array:
        """``(M, LCM_h, LCM_w, *L)`` per-meta-block contributions, materialized
        through one fused broadcast+select+sum chain — no scatter, no gather.

        When ``include_remainder=False`` the rhs grid is *not* allocated —
        instead the densify path folds ``fill_remainder`` in as a scalar.
        For ``op=add`` and ``fill_remainder=0`` this is the identity; for
        ``op=multiply`` and ``fill_remainder=0`` it collapses the result to
        zero (which the caller has verified is the desired structural
        identity at emission time).
        """
        div_meta = _block_diag_per_meta(self.divisor, self.fill_divisor)
        if self.include_remainder and self.remainder is not None:
            rem_meta = _block_diag_per_meta(self.remainder, self.fill_remainder)
            return self.op(div_meta, rem_meta)
        # remainder omitted: treat as the constant ``fill_remainder`` grid.
        return self.op(div_meta, self.fill_remainder)

    def to_dense(self) -> Array:
        """Materialize the fully-dense ``(M*LCM_h, M*LCM_w, *L)`` form.

        For ``semantic='union'`` / ``'intersection'`` the densify chain is
        identical to the legacy ``UnionBlocks`` / ``IntersectionBlocks``:
        both compute ``to_meta_blocks()`` + ``_stitch_meta`` with
        ``op(fill_divisor, fill_remainder)`` outside. Only the binary
        ``op`` differs (``add`` vs ``multiply`` for the canonical cases).
        """
        per_meta = self.to_meta_blocks()
        outside_fill = self.op(self.fill_divisor, self.fill_remainder)
        return _stitch_meta(per_meta, outside_fill)


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
    n_meta: int = 1

    @property
    def _M_primary(self) -> int:
        """Per-batch primary meta-block count.

        ``data.shape[0]`` is the *flattened* leading axis ``n_meta * M_per``;
        this property returns ``M_per`` so length-checks (``offset`` length,
        in-band gating) operate at per-batch granularity.
        """
        return self.data.shape[0] // self.n_meta

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
        """Per-batch secondary meta-block count. ``-1`` sentinel infers
        ``M_per_primary`` (square)."""
        return self.n_secondary if self.n_secondary >= 0 else self._M_primary

    @property
    def _offset_arr(self) -> tuple[int, ...]:
        if self.offset:
            return self.offset
        # Centered-band sentinel: offset[a] = a - (W-1)//2. Length = M_per_primary
        # (each of the ``n_meta`` batches uses the SAME offset pattern).
        w = (self._W - 1) // 2
        return tuple(a - w for a in range(self._M_primary))

    @property
    def _M_row(self) -> int:
        """Total row meta-blocks in the dense output (= n_meta * per-batch row)."""
        per_batch = (
            self._M_primary if self.primary_axis == 0 else self._M_secondary
        )
        return self.n_meta * per_batch

    @property
    def _M_col(self) -> int:
        """Total col meta-blocks in the dense output (= n_meta * per-batch col)."""
        per_batch = (
            self._M_secondary if self.primary_axis == 0 else self._M_primary
        )
        return self.n_meta * per_batch

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
        """``(M_total, B_row, B_col)`` when this is a pure meta-block-diagonal:
        ``W=1`` + square per-batch meta-counts (``M_per_row == M_per_col``) +
        identity offset (``offset[a] = a``). ``M_total = n_meta * M_per`` —
        the full meta-block count across all batches.

        Returns ``None`` for any banded / rectangular / skewed form (those
        can't be expressed as a single meta-block-diagonal SparseTensor pair).

        Note: square sub-blocks are NOT required — the existing
        block-diagonal SparseTensor wrapper supports rectangular block_size
        pairs.
        """
        if self._W != 1:
            return None
        if self._M_row != self._M_col:
            return None
        # Identity per-batch offset: data[a, 0] sits at primary-meta = secondary-meta = a.
        off = self._offset_arr
        if tuple(off) != tuple(range(self._M_primary)):
            return None
        return (self.data.shape[0], self._B_row, self._B_col)

    def to_meta_blocks(self) -> Array:
        """``(M_total, B_row, B_col, *L)`` meta-diagonal blocks. Only defined
        when this is a pure meta-block-diagonal (see
        :py:meth:`meta_block_shape`); for any banded / rectangular / skewed
        form use :meth:`to_dense`.

        ``M_total = n_meta * M_per`` — for ``n_meta > 1`` the per-batch
        diagonals concatenate naturally since each batch's diagonal continues
        the previous batch's at the same meta-pitch.
        """
        if self.meta_block_shape is None:
            raise ValueError(
                "BlockBanded.to_meta_blocks requires W=1 + square per-batch "
                f"meta-counts + identity offset; got W={self._W}, "
                f"M_per_primary={self._M_primary}, "
                f"M_per_secondary={self._M_secondary}, "
                f"offset={self._offset_arr}. Use to_dense() for banded forms."
            )
        return self.data[:, 0]   # (M_total, B_row, B_col, *L)

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

        For ``n_meta > 1`` the ``data`` leading axis encodes ``n_meta``
        independent banded blocks stacked on the output's meta-diagonal:
        reshape ``data`` to ``(n_meta, M_per, W, B_row, B_col, *L)``, build
        each batch's banded dense via the per-batch kernel, then stitch
        onto an ``n_meta × n_meta`` meta-block-diagonal via
        :func:`_stitch_meta` (same broadcast+where pattern, no gather).
        """
        _, W, B_row, B_col, *L = self.data.shape
        M_per_primary = self._M_primary  # per-batch
        M_per_secondary = self._M_secondary  # per-batch
        L_pad = (None,) * len(L)

        # Two offset modes: centered (no gather) vs explicit (one unavoidable gather).
        if self.offset:
            offset_tuple = self.offset
            centered_w = None
            off_for_fallback = self.offset
        else:
            offset_tuple = None
            centered_w = (W - 1) // 2
            off_for_fallback = tuple(a - centered_w for a in range(M_per_primary))

        if (
            math.prod(
                (M_per_primary, M_per_secondary, W, B_row, B_col, *L)
            ) * self.n_meta
            > _BLOCK_BANDED_BROADCAST_LIMIT
        ):
            return self._to_dense_per_band(
                M_per_primary, M_per_secondary, W, B_row, B_col, off_for_fallback, L
            )

        def _one_batch_dense(data_batch: Array) -> Array:
            """Per-batch banded densify: ``data_batch`` shape
            ``(M_per_primary, W, B_row, B_col, *L)`` → per-batch dense
            ``(M_per_primary * B_row, M_per_secondary * B_col, *L)`` for
            row-primary; transposed for col-primary."""
            if self.primary_axis == 0:
                return _to_dense_banded(
                    data_batch, M_per_primary, M_per_secondary, W, B_row, B_col,
                    offset_tuple, centered_w, self.fill_value, L, L_pad,
                )
            data_t = data_batch.swapaxes(2, 3)
            dense_T = _to_dense_banded(
                data_t, M_per_primary, M_per_secondary, W, B_col, B_row,
                offset_tuple, centered_w, self.fill_value, L, L_pad,
            )
            return dense_T.swapaxes(0, 1)

        if self.n_meta == 1:
            return _one_batch_dense(self.data)

        # n_meta > 1: reshape, vmap, stitch.
        data_b = self.data.reshape(
            self.n_meta, M_per_primary, W, B_row, B_col, *L
        )
        per_batch_dense = jax.vmap(_one_batch_dense)(data_b)
        # per_batch_dense shape: (n_meta, M_per_row * B_row, M_per_col * B_col, *L)
        return _stitch_meta(per_batch_dense, self.fill_value)

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
#  4.  Multi-axis primitives — K>1 independent bands / block-diagonals
# ----------------------------------------------------------------------------
class BandAxisSpec(NamedTuple):
    """Per-axis metadata for one band in :class:`MultiAxisBlockBanded`.

    Mirrors the single-axis ``BlockBanded`` fields. Each axis-pair of the
    output gets one of these — together encoding K independent bands.
    """

    primary_axis: int = 0     # 0 = row-primary, 1 = col-primary (this axis pair)
    n_secondary: int = -1     # -1 = M_per_primary (square per-batch)
    offset: tuple[int, ...] = ()  # () = centered band; else explicit per-row offsets
    n_meta: int = 1           # outer batch over independent bands within this axis
    band_width: int = 1       # W for this axis
    block_row: int = 1        # B_row of sub-blocks for this axis
    block_col: int = 1        # B_col of sub-blocks for this axis


class MultiAxisBlockBanded(NamedTuple):
    """K-axis block-banded storage for multi-contract matmul output.

    For each of K independent output axis pairs (each pair = one
    contract pair from the source matmul), this carries a band geometry
    via :class:`BandAxisSpec`. The dense form is the Cartesian product
    of K per-axis bands: cell ``(a_1, ..., a_K, b_1, ..., b_K)`` is
    non-zero iff EVERY axis pair ``(a_i, b_i)`` is in its corresponding
    band.

    Layout
    ------
    ``data`` : ``(M_p_1, W_1, M_p_2, W_2, ..., M_p_K, W_K,
                  B_row_1, B_row_2, ..., B_row_K,
                  B_col_1, B_col_2, ..., B_col_K, *L)``
        2K + 2K + leftover = 4K + len(L) axes total. The leading
        ``(M_p_i * n_meta_i)`` axes interleave with ``W_i`` axes; then
        K B_row axes, K B_col axes, then leftover.
    ``axes`` : tuple of K :class:`BandAxisSpec`, one per axis pair.

    For K=1: identical-semantics to :class:`BlockBanded` (kept separate
    for back-compat — existing 1-axis call sites use BlockBanded).

    Currently implemented for K=2 only; K>2 reduces by ``vmap`` over
    additional leading axes once the K=2 path is verified.
    """

    data: Array
    fill_value: Array
    axes: tuple[BandAxisSpec, ...]

    @property
    def K(self) -> int:
        return len(self.axes)

    @property
    def _per_axis_M_p(self) -> tuple[int, ...]:
        """Per-axis ``M_per_primary`` (= ``data axis size`` // ``n_meta``)."""
        return tuple(
            self.data.shape[2 * i] // spec.n_meta
            for i, spec in enumerate(self.axes)
        )

    @property
    def _per_axis_M_secondary(self) -> tuple[int, ...]:
        """Per-axis ``M_per_secondary`` (= ``n_secondary`` or ``M_per_primary``)."""
        return tuple(
            spec.n_secondary if spec.n_secondary >= 0 else self._per_axis_M_p[i]
            for i, spec in enumerate(self.axes)
        )

    @property
    def _per_axis_offset_arr(self) -> tuple[tuple[int, ...], ...]:
        """Per-axis offset tuple; centered-band sentinel ``()`` expands to
        ``a - (W-1)//2``. Length = ``M_per_primary`` for that axis."""
        result = []
        for i, spec in enumerate(self.axes):
            if spec.offset:
                result.append(spec.offset)
            else:
                w = (spec.band_width - 1) // 2
                M_p = self._per_axis_M_p[i]
                result.append(tuple(a - w for a in range(M_p)))
        return tuple(result)

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical dense output shape: 2K + len(L) axes.

        For K=2 row-primary: ``(M_p_1*B_row_1, M_p_2*B_row_2,
        M_col_1*B_col_1, M_col_2*B_col_2, *L)``. Total
        ``n_meta_i * M_p_i * B_row_i`` per row-axis i, etc.
        """
        K = self.K
        M_p = self._per_axis_M_p
        M_s = self._per_axis_M_secondary
        L = self.data.shape[4 * K :]
        out_axes: list[int] = []
        # Row axes (one per pair).
        for i, spec in enumerate(self.axes):
            m_row = M_p[i] if spec.primary_axis == 0 else M_s[i]
            out_axes.append(spec.n_meta * m_row * spec.block_row)
        # Col axes (one per pair).
        for i, spec in enumerate(self.axes):
            m_col = M_s[i] if spec.primary_axis == 0 else M_p[i]
            out_axes.append(spec.n_meta * m_col * spec.block_col)
        return (*out_axes, *L)

    def to_dense(self) -> Array:
        """Materialize the dense form for ANY K via per-axis
        broadcast+where+sum.

        Algorithm:
          1. Insert ``M_s_i`` secondary axes after each (M_p_i, W_i) pair
             via singleton broadcast — pure metadata, zero copy.
          2. For each axis i, build a one-hot selection mask of shape
             ``(M_p_i, M_s_i, W_i)`` and broadcast to the full expanded
             tensor shape; AND all per-axis masks together.
          3. ``where(mask, data, 0).sum(axis=W_axes)`` — collapse all K
             W axes in a single fused reduction. Each (M_p_i, M_s_i)
             cell gets exactly one matching W slot, so the sum acts as
             per-axis selection.
          4. Permute / reshape to the dense output layout
             ``(M_p_0*B_row_0, ..., M_s_0*B_col_0, ..., *L)``.
          5. Apply the K per-axis in-band masks (ANDed across axes) to
             swap ``fill_value`` in for any out-of-band cell along any
             axis.

        Constraints (until follow-up extensions):
          * ``n_meta`` per axis must be 1 (per-axis batching is a
            straightforward fori_loop wrap on top of this kernel).
          * ``primary_axis`` per axis must be 0 (col-primary per-axis
            mirrors the K=1 swap+transpose pattern; will be added when
            an actual col-primary K>1 case arises).
        """
        K = self.K
        if K == 1:
            spec = self.axes[0]
            bb = BlockBanded(
                data=self.data,
                fill_value=self.fill_value,
                primary_axis=spec.primary_axis,
                n_secondary=spec.n_secondary,
                offset=spec.offset,
                n_meta=spec.n_meta,
            )
            return bb.to_dense()

        # Per-axis bookkeeping (all Python int lists — static at trace time).
        M_p = list(self._per_axis_M_p)
        M_s = list(self._per_axis_M_secondary)
        W = [ax.band_width for ax in self.axes]
        B_row = [ax.block_row for ax in self.axes]
        B_col = [ax.block_col for ax in self.axes]
        offsets = list(self._per_axis_offset_arr)
        L = self.data.shape[4 * K :]

        for ax in self.axes:
            if ax.n_meta != 1 or ax.primary_axis != 0:
                raise NotImplementedError(
                    "MultiAxisBlockBanded K>1 supports n_meta=1 + row-primary "
                    "per axis. Multi-batch and col-primary per-axis follow the "
                    "K=1 patterns; add when needed."
                )

        # Step 1: insert M_s_i singleton axes via slicing indexer.
        # Original data axes (in order): M_p_0, W_0, M_p_1, W_1, ..., M_p_{K-1},
        # W_{K-1}, B_row_0, ..., B_row_{K-1}, B_col_0, ..., B_col_{K-1}, *L
        # We insert ``None`` after each M_p_i to make room for M_s_i:
        # M_p_0, M_s_0(None), W_0, M_p_1, M_s_1(None), W_1, ...
        indexer = []
        for _ in range(K):
            indexer.append(slice(None))  # M_p_i
            indexer.append(None)          # M_s_i (inserted)
            indexer.append(slice(None))  # W_i
        indexer += [slice(None)] * (2 * K + len(L))  # B_row, B_col, L axes
        data_e = self.data[tuple(indexer)]
        # Broadcast to fill M_s_i sizes.
        expanded = []
        for i in range(K):
            expanded += [M_p[i], M_s[i], W[i]]
        expanded += B_row
        expanded += B_col
        expanded += list(L)
        data_b = jnp.broadcast_to(data_e, tuple(expanded))

        # Step 2: build per-axis selection masks and combine via AND.
        combined_mask = None
        for i in range(K):
            off_arr = jnp.asarray(offsets[i], dtype=jnp.int32)
            bj = jnp.arange(M_s[i], dtype=jnp.int32)
            target_w = bj[None, :] - off_arr[:, None]  # (M_p_i, M_s_i)
            sel = target_w[:, :, None] == jnp.arange(W[i], dtype=jnp.int32)[None, None, :]
            # sel shape (M_p_i, M_s_i, W_i). Place at expanded axes
            # (3i, 3i+1, 3i+2); 1 on every other expanded axis.
            sel_shape = [1] * len(expanded)
            sel_shape[3 * i] = M_p[i]
            sel_shape[3 * i + 1] = M_s[i]
            sel_shape[3 * i + 2] = W[i]
            sel_r = sel.reshape(*sel_shape)
            combined_mask = sel_r if combined_mask is None else (combined_mask & sel_r)

        # Step 3: where + sum over all W_i axes (positions 2, 5, 8, ...).
        W_axes = tuple(3 * i + 2 for i in range(K))
        out = jnp.where(combined_mask, data_b, 0).sum(axis=W_axes)
        # ``out`` shape after dropping W: per-axis (M_p_i, M_s_i) at
        # positions (2i, 2i+1), then K B_row, K B_col, then *L.

        # Step 4: permute + reshape to dense layout
        # (M_p_0*B_row_0, ..., M_p_{K-1}*B_row_{K-1}, M_s_0*B_col_0,
        #  ..., M_s_{K-1}*B_col_{K-1}, *L).
        perm = []
        for i in range(K):
            perm.append(2 * i)         # M_p_i
            perm.append(2 * K + i)     # B_row_i
        for i in range(K):
            perm.append(2 * i + 1)     # M_s_i
            perm.append(3 * K + i)     # B_col_i
        perm += list(range(4 * K, 4 * K + len(L)))
        out = out.transpose(perm)
        final_shape = []
        for i in range(K):
            final_shape.append(M_p[i] * B_row[i])
        for i in range(K):
            final_shape.append(M_s[i] * B_col[i])
        final_shape += list(L)
        out = out.reshape(*final_shape)

        # Step 5: AND per-axis in-band masks to swap fill_value in for
        # out-of-band cells along any axis.
        per_axis_in_band = []
        for i in range(K):
            blk_r = jnp.arange(M_p[i] * B_row[i]) // B_row[i]
            blk_c = jnp.arange(M_s[i] * B_col[i]) // B_col[i]
            off_arr = jnp.asarray(offsets[i], dtype=jnp.int32)
            co = off_arr[blk_r]
            diff = blk_c[None, :] - co[:, None]
            per_axis_in_band.append((diff >= 0) & (diff < W[i]))
        # Combine via outer product over K axis pairs.
        full_mask = None
        for i in range(K):
            mask_shape = [1] * (2 * K)
            mask_shape[i] = M_p[i] * B_row[i]
            mask_shape[K + i] = M_s[i] * B_col[i]
            m = per_axis_in_band[i].reshape(*mask_shape)
            full_mask = m if full_mask is None else (full_mask & m)
        if L:
            L_pad = (None,) * len(L)
            full_mask = full_mask[(..., *L_pad)]
        return jnp.where(full_mask, out, self.fill_value)


class MultiAxisDivisorRemainder(NamedTuple):
    """K-axis elementwise compressed storage. K=1 reduces to
    :class:`DivisorRemainder`. Subsumes the K-axis analogue of
    :class:`UnionBlocks` / :class:`IntersectionBlocks` under one type
    via the same ``semantic`` enum.

    For each of K independent output axis pairs, the operand carries
    its own ``(M_i, n_i, B_h_i, B_w_i)`` per-meta-block structure. The
    full data is a single multi-dim buffer with all K axis groups
    interleaved.

    Layout (K=2 example, ``divisor`` shape)
    ----------------------------------------
    ``(M_0, n_d_0, M_1, n_d_1, B_d_h_0, B_d_w_0, B_d_h_1, B_d_w_1, *L)``

    Per axis i: ``M_i`` meta-blocks each containing ``n_i`` sub-blocks
    of shape ``(B_h_i, B_w_i)`` arranged on the meta-block-diagonal.
    The output dense shape is the Cartesian product:
    ``(M_0 * LCM_h_0, M_0 * LCM_w_0, M_1 * LCM_h_1, M_1 * LCM_w_1, *L)``
    where ``LCM_h_i = n_d_i * B_d_h_i`` (and matches the rhs side's
    ``n_r_i * B_r_h_i`` by the same logical-size constraint as K=1).

    Currently K=1 (back-compat alias) and K=2 are implemented; K>2
    follows the same per-axis chaining of ``_block_diag_per_meta``.
    """

    divisor: Array
    remainder: Array | None
    fill_divisor: Array
    fill_remainder: Array
    semantic: str = "union"
    include_remainder: bool = True
    op: Callable = jnp.add
    K: int = 1
    # Per-axis ``(M_i, n_d_i, n_r_i, B_h_i, B_w_i)`` metadata. Inferred
    # from ``divisor`` shape when K=1; explicit for K>1.
    axes_meta: tuple[tuple[int, int, int, int, int], ...] = ()

    def _per_axis(self) -> tuple[tuple[int, int, int, int, int], ...]:
        """Per-axis ``(M_i, n_d_i, n_r_i, B_h_i, B_w_i)`` tuple."""
        if self.axes_meta:
            return self.axes_meta
        # K=1 inference from divisor shape: (M, n_d, B_h, B_w, *L).
        if self.K != 1:
            raise ValueError(
                f"axes_meta required when K={self.K} > 1"
            )
        M, n_d, B_h, B_w, *_ = self.divisor.shape
        if self.remainder is not None:
            _, n_r, _, _, *_ = self.remainder.shape
        else:
            n_r = 0
        return ((M, n_d, n_r, B_h, B_w),)

    @property
    def shape(self) -> tuple[int, ...]:
        """Full dense output shape ``(M_0*LCM_h_0, M_0*LCM_w_0, M_1*LCM_h_1,
        M_1*LCM_w_1, ..., *L)``."""
        per_axis = self._per_axis()
        K = self.K
        # Locate the leftover L axes: divisor has 2K (M_i, n_d_i) + 2K (B_h, B_w) prefix axes.
        L = self.divisor.shape[4 * K :] if K > 1 else self.divisor.shape[4:]
        out_axes: list[int] = []
        for M_i, n_d_i, _, B_h_i, B_w_i in per_axis:
            out_axes.append(M_i * n_d_i * B_h_i)
            out_axes.append(M_i * n_d_i * B_w_i)
        return (*out_axes, *L)

    def to_dense(self) -> Array:
        """Materialize the dense form via per-axis ``_block_diag_per_meta``
        chaining + ``_stitch_meta`` on each meta-axis.

        K=1: delegates to :class:`DivisorRemainder` (identical semantics).
        K=2: per-axis densify + op combine + per-axis stitch.
        K>2: ``NotImplementedError``.
        """
        if self.K == 1:
            # Back-compat: behave like DivisorRemainder.
            dr = DivisorRemainder(
                divisor=self.divisor,
                remainder=self.remainder,
                fill_divisor=self.fill_divisor,
                fill_remainder=self.fill_remainder,
                semantic=self.semantic,
                include_remainder=self.include_remainder,
                op=self.op,
            )
            return dr.to_dense()

        # General K. Per-axis bookkeeping.
        K = self.K
        per_axis = self._per_axis()
        M = [t[0] for t in per_axis]
        n_d = [t[1] for t in per_axis]
        n_r = [t[2] for t in per_axis]
        B_h = [t[3] for t in per_axis]
        B_w = [t[4] for t in per_axis]
        L = self.divisor.shape[4 * K :]

        # Step 1: per-axis ``_block_diag_per_meta`` chain on divisor (and
        # remainder if included). Apply axis-i by bringing
        # ``(M_i, n_d_i, B_h_i, B_w_i)`` to the leading 4 axes via
        # transpose, applying the kernel, then moving the resulting
        # ``(M_i, n_d_i*B_h_i, n_d_i*B_w_i)`` to its target slot.

        def _chain_block_diag(buf: Array, n: list[int], B_h_list: list[int],
                              B_w_list: list[int], fill: Array) -> Array:
            """Apply ``_block_diag_per_meta`` along each of K axes.

            ``buf`` starts in layout ``(M_0, n_0, M_1, n_1, ..., M_{K-1},
            n_{K-1}, B_h_0, ..., B_h_{K-1}, B_w_0, ..., B_w_{K-1}, *L)``.
            After K applications: ``(M_0, H_0, M_1, H_1, ..., M_{K-1},
            H_{K-1}, W_0, ..., W_{K-1}, *L)`` where
            ``H_i = n_i * B_h_i`` and ``W_i = n_i * B_w_i``.
            """
            cur = buf
            # Process axes left-to-right. After processing axis i:
            #   leading axes: (M_0, H_0, M_1, H_1, ..., M_i, H_i, ...
            #                  remaining (M_{i+1}, n_{i+1}, ..., M_{K-1},
            #                  n_{K-1}), B_h_{i+1}, ..., B_h_{K-1},
            #                  W_0, W_1, ..., W_i, B_w_{i+1}, ..., B_w_{K-1}, *L)
            # We always bring (M_i, n_i, B_h_i, B_w_i) to the front via
            # transpose, apply, then transpose back.
            for i in range(K):
                # Current layout (after i prior applications):
                #   indices 0..(2i-1): (M_0, H_0, M_1, H_1, ..., M_{i-1}, H_{i-1})
                #   index 2i: M_i
                #   index 2i+1: n_i
                #   indices 2i+2..(2K-1): M_{i+1}, n_{i+1}, ..., M_{K-1}, n_{K-1}
                #   indices 2K..(2K+i-1): W_0, W_1, ..., W_{i-1}
                #   index 2K + i: B_h_i (preceded by B_h_{i-1} which got consumed)
                #
                # Hmm — tracking shape positions is delicate. Take a
                # different approach: use ``jnp.moveaxis`` to bring the
                # relevant axes to the front, apply
                # ``_block_diag_per_meta``, then move output axes back.

                # Find current positions of (M_i, n_i, B_h_i, B_w_i).
                # After ``i`` prior axes processed, the layout is:
                #   [M_0, H_0, M_1, H_1, ..., M_{i-1}, H_{i-1},
                #    M_i, n_i, M_{i+1}, n_{i+1}, ..., M_{K-1}, n_{K-1},
                #    B_h_i, B_h_{i+1}, ..., B_h_{K-1},
                #    W_0, ..., W_{i-1}, B_w_i, B_w_{i+1}, ..., B_w_{K-1}, *L]
                # Position of M_i:    2*i
                # Position of n_i:    2*i + 1
                # Position of B_h_i:  2*K + (i)             — first B_h slot after i-th processing
                # Position of B_w_i:  2*K + (K - i) + i     — first B_w slot after i-th processing
                #                   = 3*K
                # Hmm this depends on remaining unprocessed. Let me redo.
                #
                # The simplest robust approach: track shape positions by
                # scanning. For axis i, compute the current positions.
                ndim = cur.ndim
                # Build position map. After processing axes 0..i-1:
                #   - 2 axes per processed: (M_j, H_j) for j < i  →  positions 0..2i-1
                #   - 2 axes per unprocessed of the (M, n) pairs: positions 2i..2K-1 (M_i, n_i, M_{i+1}, n_{i+1}, ...)
                #   - 1 axis per processed: W_j for j < i → positions 2K..2K+i-1
                #   - 1 axis per unprocessed B_h: positions 2K+i..3K-1 (B_h_i, B_h_{i+1}, ...)
                #   - 1 axis per unprocessed B_w: positions 3K..3K + (K-i) - 1 (B_w_i, B_w_{i+1}, ...)
                #     Wait this should be 1 axis per ALL B_w (we haven't consumed any B_w_j with j < i).
                #     Hmm but for processed axes, the B_w_j became W_j and is now between the (M, H) pairs and the (B_h, B_w) section. So B_w axes for processed don't appear separately — they're folded into H/W via _block_diag_per_meta.
                #
                # Actually, _block_diag_per_meta produces (M, n*B_h, n*B_w, *L) = (M, H, W, *L). So after processing axis i, we have M_i, H_i, W_i in the output. The W_i ends up at a specific position.
                pos_M_i = 2 * i
                pos_n_i = 2 * i + 1
                # B_h_i sits among the unprocessed B_h block: position 2K + i (counting K unprocessed M/n pairs from 2i to 2K, then 0 processed W positions... wait).
                # After i prior applications, layout was reassembled to:
                #   (M_0, H_0, ..., M_{i-1}, H_{i-1}, M_i, n_i, M_{i+1}, n_{i+1}, ..., B_h_i, ..., W_0, ..., W_{i-1}, B_w_i, ...)
                # Hmm this is getting complicated. Let me use moveaxis explicitly.
                # M_i is at axis 2*i (the next un-processed pair's M slot).
                # n_i is at axis 2*i + 1.
                # B_h_i and B_w_i positions depend on what's left.
                #
                # Simpler: at the start of each iteration, we know the
                # layout positions. After processing axis i:
                #   * (M_i, n_i, B_h_i, B_w_i) get consumed
                #   * (M_i, H_i = n_i*B_h_i, W_i = n_i*B_w_i) get produced
                # The kernel ``_block_diag_per_meta`` returns
                # ``(M, n*B_h, n*B_w, *trailing)``. So if we put (M_i, n_i, B_h_i, B_w_i, *rest) at the LEAD, then the kernel produces (M_i, H_i, W_i, *rest).
                # Then we move (M_i, H_i) to position (2i, 2i+1) of the output, and W_i to position 2K + i of the output.

                # Find current axis indices.
                # We need to track them since the layout changes after each iteration.
                # Approach: at iteration i, the layout has structure as documented above.
                # Compute the explicit positions:
                #   M_i at 2i, n_i at 2i+1.
                #   The remaining axes after position 2K: W_0..W_{i-1}, then B_h_i, B_h_{i+1}, ..., B_h_{K-1}, then B_w_i, ..., B_w_{K-1}, *L.
                # So B_h_i is at position 2K + i.
                # B_w_i is at position 2K + i + (K - i) = 3K, since there are (K - i) remaining B_h axes after B_h_i, then B_w_i comes next... wait no, all remaining B_h's come first.
                # Number of remaining B_h: K - i (B_h_i ... B_h_{K-1}). Position of B_h_i: 2K + i (since W_0..W_{i-1} take positions 2K..2K+i-1).
                # Position of B_w_i: 2K + i + (K - i) = 3K + 0... wait that's still K - i positions for B_h. So after B_h block (positions 2K+i to 3K+i-1 ?), B_w starts at 3K+i. Hmm.
                # Let me recount.
                # At start of iteration i, the layout has these groups (and counts):
                #   * processed (M, H) pairs: 2i axes (positions 0..2i-1)
                #   * unprocessed (M, n) pairs: 2(K-i) axes (positions 2i..2K-1)
                #   * processed W axes: i axes (positions 2K..2K+i-1)
                #   * unprocessed B_h axes: K-i axes (positions 2K+i..3K-1)
                #   * unprocessed B_w axes: K-i axes (positions 3K..3K + K-i-1 = 4K-i-1)
                #   * L: len(L) axes (positions 4K-i..)
                # Total: 2i + 2(K-i) + i + (K-i) + (K-i) + len(L) = 2i + 2K - 2i + i + K - i + K - i + len(L) = 4K - i + len(L)
                # That looks wrong (should be invariant at 4K + len(L) since we're transforming, not removing axes).
                # Actually after each application, we LOSE 1 axis (n_i fuses with B_h_i into H_i, and n_i also fuses with B_w_i into W_i; n_i appears only once but contributes to both H and W). Wait the kernel consumes (M, n, B_h, B_w) — 4 axes — and produces (M, H, W) — 3 axes. So we lose 1 axis per iteration.
                # Starting axes: 4K + len(L). After K iterations: 4K - K + len(L) = 3K + len(L).
                # Final layout: M_0, H_0, ..., M_{K-1}, H_{K-1}, W_0, ..., W_{K-1}, *L = 2K + K + len(L) = 3K + len(L). ✓

                # So during iteration i:
                #   axes so far: 4K + len(L) - i
                #   - 2i (processed M, H)
                #   - 2(K-i) (unprocessed M, n)
                #   - i (processed W)
                #   - (K-i) (unprocessed B_h)
                #   - (K-i) (unprocessed B_w)
                #   - len(L)
                # Total: 2i + 2(K-i) + i + (K-i) + (K-i) + len(L) = 4K - i + len(L). Matches!
                #
                # Positions:
                #   M_i:     2i
                #   n_i:     2i + 1
                #   W_(j<i): 2K..2K+i-1
                #   B_h_i:   2K + i
                #   B_w_i:   2K + i + (K-i) = 3K (since B_h block has K-i elements)
                #   L:       3K + (K-i) = 4K - i

                pos_M = 2 * i
                pos_n = 2 * i + 1
                pos_B_h = 2 * K + i
                pos_B_w = 3 * K  # always 3K because unprocessed B_h takes positions 2K+i..3K-1; B_w_i starts at 3K (regardless of i because by the time we get there, 3K is constant since len(L) shifts)
                # Hmm wait, I said B_w_i at 3K + i + (K-i) = 3K. Let me re-verify.
                # 2i (M, H pairs) + 2(K-i) (M, n pairs) + i (W) + (K-i) (B_h) = 2i + 2K - 2i + i + K - i = 3K.
                # So B_w block starts at 3K. ✓

                # Move (M_i, n_i, B_h_i, B_w_i) to positions (0, 1, 2, 3).
                cur_t = jnp.moveaxis(
                    cur,
                    (pos_M, pos_n, pos_B_h, pos_B_w),
                    (0, 1, 2, 3),
                )
                # Apply _block_diag_per_meta: produces (M_i, H_i=n_i*B_h_i, W_i=n_i*B_w_i, *rest)
                applied = _block_diag_per_meta(cur_t, fill)
                # Now applied has axes: (M_i, H_i, W_i, *rest). Total axes = 3 + (cur.ndim - 4) = cur.ndim - 1.
                # The "rest" axes are in the ORIGINAL ORDER (minus the 4 we moved).
                # We want the new layout to have (M_i, H_i) at positions (2i, 2i+1) and W_i at position 2K + i (of the new layout, which has 1 fewer axis).
                # The "rest" axes count: original ndim - 4. Their order: M_0, H_0, ..., M_{i-1}, H_{i-1}, M_{i+1}, n_{i+1}, ..., M_{K-1}, n_{K-1}, W_0, ..., W_{i-1}, B_h_{i+1}, ..., B_h_{K-1}, B_w_{i+1}, ..., B_w_{K-1}, *L.
                # Source positions in `applied`: M_i at 0, H_i at 1, W_i at 2, then rest at 3.. .
                # Target positions in the new layout:
                #   M_i at 2i
                #   H_i at 2i + 1
                #   M_{i+1} at 2(i+1) = 2i + 2
                #   n_{i+1} at 2i + 3
                #   ...
                #   W_0 at 2K
                #   ...
                #   W_i at 2K + i
                #   B_h_{i+1} at 2K + i + 1
                #
                # The "rest" axes order is already what we need IF we move M_i, H_i to (2i, 2i+1) and W_i to 2K + i (in the new layout's indexing).
                # In the `applied` tensor:
                #   axis 0 = M_i  → target 2i
                #   axis 1 = H_i  → target 2i + 1
                #   axis 2 = W_i  → target 2K + i (in new layout numbering, which has 1 fewer axis)
                #   axes 3.. = M_0, H_0, ..., M_{i-1}, H_{i-1}, M_{i+1}, n_{i+1}, ..., B_w_{K-1}, *L → these need to fill the OTHER positions in the new layout.
                new_ndim = cur.ndim - 1
                cur = jnp.moveaxis(
                    applied,
                    (0, 1, 2),
                    (2 * i, 2 * i + 1, 2 * K + i),
                )
            return cur

        d_meta = _chain_block_diag(self.divisor, n_d, B_h, B_w, self.fill_divisor)
        # d_meta shape: (M_0, H_0, M_1, H_1, ..., M_{K-1}, H_{K-1},
        #                W_0, ..., W_{K-1}, *L). H_i = n_d_i*B_h_i,
        #                W_i = n_d_i*B_w_i.

        if not (self.include_remainder and self.remainder is not None):
            grid_meta = d_meta
        else:
            r_meta = _chain_block_diag(
                self.remainder, n_r, B_h, B_w, self.fill_remainder
            )
            grid_meta = self.op(d_meta, r_meta)

        outside_fill = self.op(self.fill_divisor, self.fill_remainder)

        # Step 2: stitch onto K-dim meta-block-diagonal.
        # ``grid_meta`` shape (3K + len(L) axes):
        #   (M_0, H_0, M_1, H_1, ..., M_{K-1}, H_{K-1},
        #    W_0, ..., W_{K-1}, *L)
        # For each axis i we need to insert an M_i_c (col-side meta) axis
        # after H_i. Build via slicing + broadcast.
        H = [n_d[i] * B_h[i] for i in range(K)]
        W = [n_d[i] * B_w[i] for i in range(K)]
        indexer = []
        for _ in range(K):
            indexer.append(slice(None))  # M_i_r
            indexer.append(slice(None))  # H_i
            indexer.append(None)          # M_i_c (inserted)
        indexer += [slice(None)] * K       # W axes
        indexer += [slice(None)] * len(L)  # L axes
        g_e = grid_meta[tuple(indexer)]
        expanded = []
        for i in range(K):
            expanded += [M[i], H[i], M[i]]
        expanded += W
        expanded += list(L)
        g_b = jnp.broadcast_to(g_e, tuple(expanded))

        # Build per-axis diagonal mask M_i_r == M_i_c.
        combined_mask = None
        for i in range(K):
            m_idx = jnp.arange(M[i], dtype=jnp.int32)
            diag = m_idx[:, None] == m_idx[None, :]  # (M_i, M_i)
            sel_shape = [1] * len(expanded)
            sel_shape[3 * i] = M[i]
            sel_shape[3 * i + 2] = M[i]
            d_r = diag.reshape(*sel_shape)
            combined_mask = d_r if combined_mask is None else (combined_mask & d_r)

        L_pad = (None,) * len(L)
        if L:
            combined_mask = combined_mask[(..., *L_pad)]
        gathered = jnp.where(combined_mask, g_b, outside_fill)

        # Permute + reshape to dense, per-axis interleaved layout:
        # ``(M_0*H_0, M_0*W_0, M_1*H_1, M_1*W_1, ..., M_{K-1}*H_{K-1},
        #    M_{K-1}*W_{K-1}, *L)``.
        # Matches the natural multi-axis operand layout (each axis-pair
        # contributes (h, w) consecutively), so chaining elementwise ops
        # preserves the per-axis grouping.
        #
        # Current ``gathered`` axes (positions):
        #   per i: M_i_r at 3i, H_i at 3i+1, M_i_c at 3i+2
        #   then W axes at positions 3K..4K-1
        #   then L axes.
        # Target permutation: for each i, [M_i_r, H_i, M_i_c, W_i]
        # consecutively, so source positions (3i, 3i+1, 3i+2, 3K+i)
        # → target positions (4i, 4i+1, 4i+2, 4i+3).
        perm = []
        for i in range(K):
            perm.append(3 * i)         # M_i_r
            perm.append(3 * i + 1)     # H_i
            perm.append(3 * i + 2)     # M_i_c
            perm.append(3 * K + i)     # W_i
        perm += list(range(4 * K, 4 * K + len(L)))
        out = gathered.transpose(perm)
        # Reshape: combine (M_i_r, H_i) → axis 2i, (M_i_c, W_i) → axis 2i+1.
        final_shape = []
        for i in range(K):
            final_shape.append(M[i] * H[i])
            final_shape.append(M[i] * W[i])
        final_shape += list(L)
        return out.reshape(*final_shape)


# ----------------------------------------------------------------------------
#  PyTree registration: ``op`` is static (Callable, hashable), buffers are leaves
# ----------------------------------------------------------------------------
# Auto-NamedTuple flattening would put ``op`` in the children list, which
# breaks ``jax.jit`` round-trips (functions aren't valid jax types). Override
# with a custom split that pushes ``op`` into the static aux_data — same
# treatment elementwise.py uses for its op closures.
def _blocks_flatten(ub):
    return (ub.lhs, ub.rhs, ub.fill_lhs, ub.fill_rhs), (ub.op, ub.n_meta)


def _union_unflatten(aux, children):
    return UnionBlocks(*children, op=aux[0], n_meta=aux[1])


def _intersection_unflatten(aux, children):
    return IntersectionBlocks(*children, op=aux[0], n_meta=aux[1])


jax.tree_util.register_pytree_node(UnionBlocks, _blocks_flatten, _union_unflatten)
jax.tree_util.register_pytree_node(IntersectionBlocks, _blocks_flatten, _intersection_unflatten)


def _block_banded_flatten(bb):
    """Flatten ``BlockBanded`` for pytree traversal. Two array leaves; the
    static metadata (``primary_axis``, ``n_secondary``, ``offset``,
    ``n_meta``) goes into aux_data so JIT treats it as compile-time
    constant."""
    return (
        (bb.data, bb.fill_value),
        (bb.primary_axis, bb.n_secondary, bb.offset, bb.n_meta),
    )


def _block_banded_unflatten(aux, children):
    return BlockBanded(
        *children,
        primary_axis=aux[0],
        n_secondary=aux[1],
        offset=aux[2],
        n_meta=aux[3],
    )


jax.tree_util.register_pytree_node(
    BlockBanded, _block_banded_flatten, _block_banded_unflatten
)


def _divisor_remainder_flatten(dr):
    """Flatten ``DivisorRemainder`` for pytree traversal. ``divisor`` and
    ``remainder`` (may be ``None``) are array children; ``fill_*`` are
    array children too. ``semantic`` / ``include_remainder`` / ``op`` /
    ``n_meta`` go into aux_data — JIT treats them as compile-time
    constants."""
    return (
        (dr.divisor, dr.remainder, dr.fill_divisor, dr.fill_remainder),
        (dr.semantic, dr.include_remainder, dr.op, dr.n_meta),
    )


def _divisor_remainder_unflatten(aux, children):
    return DivisorRemainder(
        divisor=children[0],
        remainder=children[1],
        fill_divisor=children[2],
        fill_remainder=children[3],
        semantic=aux[0],
        include_remainder=aux[1],
        op=aux[2],
        n_meta=aux[3],
    )


jax.tree_util.register_pytree_node(
    DivisorRemainder, _divisor_remainder_flatten, _divisor_remainder_unflatten
)


def _multi_axis_bb_flatten(mab):
    """Flatten ``MultiAxisBlockBanded`` for pytree traversal. Two array
    leaves; the ``axes`` tuple-of-BandAxisSpec goes into aux_data as a
    plain tuple of int/tuple fields."""
    aux_axes = tuple(
        (s.primary_axis, s.n_secondary, s.offset, s.n_meta, s.band_width,
         s.block_row, s.block_col)
        for s in mab.axes
    )
    return (mab.data, mab.fill_value), aux_axes


def _multi_axis_bb_unflatten(aux, children):
    axes = tuple(BandAxisSpec(*t) for t in aux)
    return MultiAxisBlockBanded(
        data=children[0], fill_value=children[1], axes=axes
    )


jax.tree_util.register_pytree_node(
    MultiAxisBlockBanded, _multi_axis_bb_flatten, _multi_axis_bb_unflatten
)


def _multi_axis_dr_flatten(mdr):
    """Flatten ``MultiAxisDivisorRemainder``. Array children: divisor,
    remainder (may be None), fill_*. Aux: semantic, include_remainder,
    op, K, axes_meta."""
    return (
        (mdr.divisor, mdr.remainder, mdr.fill_divisor, mdr.fill_remainder),
        (mdr.semantic, mdr.include_remainder, mdr.op, mdr.K, mdr.axes_meta),
    )


def _multi_axis_dr_unflatten(aux, children):
    return MultiAxisDivisorRemainder(
        divisor=children[0],
        remainder=children[1],
        fill_divisor=children[2],
        fill_remainder=children[3],
        semantic=aux[0],
        include_remainder=aux[1],
        op=aux[2],
        K=aux[3],
        axes_meta=aux[4],
    )


jax.tree_util.register_pytree_node(
    MultiAxisDivisorRemainder, _multi_axis_dr_flatten, _multi_axis_dr_unflatten
)
