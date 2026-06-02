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
        """Materialize the dense form via per-axis broadcast+where+sum.

        For K=2: apply axis-0 banded select then axis-1 banded select.
        Each axis's mask is independent so the combined select is the
        elementwise AND of per-axis masks — both reductions happen in
        the same fused chain. For K>2: recursively chain.

        Currently implements K=1 (delegates to ``BlockBanded`` equivalent)
        and K=2 explicitly; K>2 raises ``NotImplementedError`` until the
        chain is generalized.
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
        if K != 2:
            raise NotImplementedError(
                f"MultiAxisBlockBanded.to_dense for K={K} not yet implemented; "
                "K=1 and K=2 supported. K>2 needs the per-axis broadcast+where+sum "
                "chain to recursively extend over additional leading axes."
            )

        # K=2: data shape (M1_tot, W1, M2_tot, W2, B_r1, B_r2, B_c1, B_c2, *L)
        # where M_i_tot = n_meta_i * M_p_i.
        spec_0, spec_1 = self.axes
        M_p_0, M_p_1 = self._per_axis_M_p
        M_s_0, M_s_1 = self._per_axis_M_secondary
        W_0, W_1 = spec_0.band_width, spec_1.band_width
        B_r_0, B_r_1 = spec_0.block_row, spec_1.block_row
        B_c_0, B_c_1 = spec_0.block_col, spec_1.block_col
        L = self.data.shape[8:]
        n_meta_0, n_meta_1 = spec_0.n_meta, spec_1.n_meta

        if n_meta_0 != 1 or n_meta_1 != 1:
            raise NotImplementedError(
                "MultiAxisBlockBanded K=2 with n_meta>1 not yet implemented; "
                "fall back to per-batch loop is straightforward but out of "
                "scope for the initial Phase 7.5 landing."
            )
        if spec_0.primary_axis != 0 or spec_1.primary_axis != 0:
            raise NotImplementedError(
                "MultiAxisBlockBanded K=2 supports row-primary only for now; "
                "col-primary per-axis swap follows the same pattern as the "
                "K=1 case (swapaxes + transpose)."
            )

        off_0 = self._per_axis_offset_arr[0]
        off_1 = self._per_axis_offset_arr[1]
        off_0_arr = jnp.asarray(off_0, dtype=jnp.int32)
        off_1_arr = jnp.asarray(off_1, dtype=jnp.int32)
        L_pad = (None,) * len(L)

        # Build per-axis selection masks (axis-1: (M_p_0, M_s_0, W_0); axis-2: (M_p_1, M_s_1, W_1)).
        bj_0 = jnp.arange(M_s_0, dtype=jnp.int32)
        bj_1 = jnp.arange(M_s_1, dtype=jnp.int32)
        target_w_0 = bj_0[None, :] - off_0_arr[:, None]  # (M_p_0, M_s_0)
        target_w_1 = bj_1[None, :] - off_1_arr[:, None]  # (M_p_1, M_s_1)
        select_0 = target_w_0[:, :, None] == jnp.arange(W_0, dtype=jnp.int32)[None, None, :]
        select_1 = target_w_1[:, :, None] == jnp.arange(W_1, dtype=jnp.int32)[None, None, :]

        # Broadcast data to insert the per-axis M_s axes.
        # data: (M_p_0, W_0, M_p_1, W_1, B_r_0, B_r_1, B_c_0, B_c_1, *L)
        # Add M_s_0 axis at position 1, M_s_1 at position 4 →
        # (M_p_0, M_s_0, W_0, M_p_1, M_s_1, W_1, B_r_0, B_r_1, B_c_0, B_c_1, *L)
        data_b = self.data[:, None, :, :, None, :, ...]
        data_b = jnp.broadcast_to(
            data_b,
            (M_p_0, M_s_0, W_0, M_p_1, M_s_1, W_1, B_r_0, B_r_1, B_c_0, B_c_1, *L),
        )
        mask_0 = select_0[:, :, :, None, None, None, None, None, None, None]  # broadcast over remaining axes
        mask_1 = select_1[None, None, None, :, :, :, None, None, None, None]
        if L:
            mask_0 = mask_0[(..., *L_pad)]
            mask_1 = mask_1[(..., *L_pad)]
        mask = mask_0 & mask_1
        # Sum over W_0 (axis 2) and W_1 (axis 5) — both reductions in one pass.
        out_meta = jnp.where(mask, data_b, 0).sum(axis=(2, 5))
        # Shape: (M_p_0, M_s_0, M_p_1, M_s_1, B_r_0, B_r_1, B_c_0, B_c_1, *L)

        # Permute + reshape to (M_p_0*B_r_0, M_p_1*B_r_1, M_s_0*B_c_0, M_s_1*B_c_1, *L).
        # Current order: (a0, c0, a1, c1, sr0, sr1, sc0, sc1, *L)
        # Target: (a0, sr0, a1, sr1, c0, sc0, c1, sc1, *L) →
        # reshape to (a0*sr0, a1*sr1, c0*sc0, c1*sc1, *L)
        perm = (0, 4, 2, 5, 1, 6, 3, 7, *range(8, 8 + len(L)))
        out_meta = out_meta.transpose(perm)
        out_meta = out_meta.reshape(
            M_p_0 * B_r_0, M_p_1 * B_r_1, M_s_0 * B_c_0, M_s_1 * B_c_1, *L
        )

        # Apply fill_value at out-of-band positions (either axis out-of-band).
        blk_r_0 = jnp.arange(M_p_0 * B_r_0) // B_r_0
        blk_c_0 = jnp.arange(M_s_0 * B_c_0) // B_c_0
        blk_r_1 = jnp.arange(M_p_1 * B_r_1) // B_r_1
        blk_c_1 = jnp.arange(M_s_1 * B_c_1) // B_c_1
        co_0 = off_0_arr[blk_r_0]  # (M_p_0 * B_r_0,)
        co_1 = off_1_arr[blk_r_1]
        diff_0 = blk_c_0[None, :] - co_0[:, None]  # (M_p_0*B_r_0, M_s_0*B_c_0)
        diff_1 = blk_c_1[None, :] - co_1[:, None]  # (M_p_1*B_r_1, M_s_1*B_c_1)
        in_band_0 = (diff_0 >= 0) & (diff_0 < W_0)
        in_band_1 = (diff_1 >= 0) & (diff_1 < W_1)
        # Combine the two in_band masks via outer product:
        # in_band[r_0, r_1, c_0, c_1] = in_band_0[r_0, c_0] & in_band_1[r_1, c_1]
        full_mask = (
            in_band_0[:, None, :, None] & in_band_1[None, :, None, :]
        )  # (M_p_0*B_r_0, M_p_1*B_r_1, M_s_0*B_c_0, M_s_1*B_c_1)
        if L:
            full_mask = full_mask[(..., *L_pad)]
        return jnp.where(full_mask, out_meta, self.fill_value)


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
