"""Densify kernels for the compressed dim structures that arise from block-sparse ops.

These materialize the **outputs** of elementwise / matmul ops between two
block-diagonal sources whose block sizes can disagree. Such an output is not a
single rectangular tensor — it's a band / set-theoretic structure described by
the compressed ``Index`` types (``BandedIndex`` / ``SetIndex`` in
``graphax.sparse.indexes``). The structure lives in the dim ``Index`` metadata;
``val`` carries the packed buffer. What lets us pack densely is GCD/LCM
periodicity: when total dims are integer multiples of the LCM, the per-LCM-block
pattern repeats, giving a "depth" / "batch" axis ``M`` along which the
differently-shaped buffers stack.

This module holds the gather-free densify kernels those Index types delegate to:

* :func:`_block_diag_per_meta` / :func:`_stitch_meta` — place a per-meta buffer
  onto its (meta-)block-diagonal. Used by ``SetIndex.densify_axis`` /
  ``to_meta_blocks``.
* :func:`_to_dense_banded` / :func:`_per_band_stream` / :func:`_densify_band` —
  the single-axis band densify (n_meta-aware, broadcast-limit guarded,
  gather-free for centered offsets). Used by ``BandedIndex.densify_axis``.
* :func:`_densify_multi_banded` (+ :class:`BandAxisSpec`) — the K-axis band
  densify, one fused broadcast+where+sum across all axes. Used by
  ``utils._densify_compressed_dims`` for the K≥2 case.

XLA-fusion notes
----------------
Each densify is a single broadcast/select/where chain — no scatter, gather only
where a non-centered band lookup makes it strictly necessary — so XLA folds the
materialization into a downstream consumer's operand fetch (SMEM, not HBM).
"""

from __future__ import annotations

import math
from typing import NamedTuple

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


def _is_static_zero(x) -> bool:
    """True iff ``x`` is a compile-time-constant zero (Python scalar or a
    concrete, non-traced array equal to 0). Used to skip redundant
    ``where(..., fill)`` masks when the fill is provably zero (CR-4)."""
    try:
        import numpy as _np

        return bool(_np.asarray(x) == 0) if _np.ndim(x) == 0 else False
    except Exception:
        return False


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
    skip_fill_mask: bool = False,
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

    # Step 5: replace the zero-pad outside the band with fill_value. Out-of-band
    # cells of ``out_meta`` are already 0 (the where+sum in Step 3 left them
    # zero), so when ``fill_value`` is a static zero (CR-4 — the matmul path
    # always builds ``fill=0``) this mask is a no-op; skip it to avoid a second
    # dense-sized boolean + where over the whole output.
    if skip_fill_mask or _is_static_zero(fill_value):
        return out_meta
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


def _centered_offset(offset: tuple[int, ...], W: int) -> bool:
    """True iff ``offset`` is the centered band ``offset[a] = a - (W-1)//2``
    (CR-3). When centered, the densify can take the gather-free arithmetic
    branch instead of the explicit ``offset_arr[blk_i]`` gather."""
    w = (W - 1) // 2
    return tuple(offset) == tuple(a - w for a in range(len(offset)))


def _per_band_stream(
    data: Array,
    M_primary: int,
    M_secondary: int,
    W: int,
    B_row: int,
    B_col: int,
    offset: tuple[int, ...],
    primary_axis: int,
    fill_value: Array,
    L: tuple[int, ...],
) -> Array:
    """``lax.fori_loop`` streaming densify for a SINGLE band (no n_meta).

    ``data`` is ``(M_primary, W, B_row, B_col, *L)``; returns
    ``(M_row*B_row, M_col*B_col, *L)`` writing the W in-band sub-blocks per
    primary slot via ``dynamic_update_slice`` — never materializes the
    ``(M_primary, M_secondary)`` square. n_meta-free by construction: the
    caller (``_densify_band``) maps this over batches and stitches, so the
    CR-1 batch-drop bug cannot recur here.
    """
    M_row, M_col = (
        (M_primary, M_secondary) if primary_axis == 0 else (M_secondary, M_primary)
    )
    out = jnp.full((M_row * B_row, M_col * B_col, *L), fill_value, dtype=data.dtype)
    zero_idx = (jnp.int32(0),) * len(L)
    off_arr = jnp.asarray(offset, dtype=jnp.int32)
    row_primary = primary_axis == 0

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
            block = data[k, b]
            existing = lax.dynamic_slice(
                acc, (row_start, col_start) + zero_idx, (B_row, B_col, *L)
            )
            replacement = jnp.where(in_range, block, existing)
            acc = lax.dynamic_update_slice(
                acc, replacement, (row_start, col_start) + zero_idx
            )
        return acc

    return lax.fori_loop(0, M_primary, body, out)


def _densify_band(
    data: Array,
    M_primary: int,
    M_secondary: int,
    W: int,
    B_row: int,
    B_col: int,
    offset: tuple[int, ...],
    primary_axis: int,
    n_meta: int,
    fill_value: Array,
    L: tuple[int, ...],
) -> Array:
    """Consolidated, CR-fixed band densify. Single source of truth shared by
    ``BlockBanded.to_dense`` (legacy) and ``BandedIndex.densify_axis`` (Phase 8).

    ``data`` : ``(n_meta * M_primary, W, B_row, B_col, *L)`` — leading axis is
    the flat ``n_meta`` batch × per-batch primary meta. ``offset`` is the
    per-primary secondary offset (length ``M_primary``); ``()`` is treated as
    the centered band.

    Returns the dense ``(n_meta*M_row*B_row, n_meta*M_col*B_col, *L)``.

    Fixes folded in from the code review:
      * **CR-1**: ``n_meta>1`` is ALWAYS handled by per-batch densify +
        ``_stitch_meta`` (never the n_meta-blind fallback that dropped
        batches).
      * **CR-2**: the ``_BLOCK_BANDED_BROADCAST_LIMIT`` guard is applied
        per-batch; a single oversized batch streams via ``_per_band_stream``,
        and when the *total* (n_meta × per-batch) exceeds the limit but each
        batch fits, batches run sequentially via ``lax.map`` instead of a
        simultaneous ``vmap``.
      * **CR-3**: a centered ``offset`` takes the gather-free arithmetic
        branch in ``_to_dense_banded``.
      * **CR-4**: the out-of-band fill mask is skipped when ``fill_value`` is
        a static zero.
    """
    L_pad = (None,) * len(L)
    centered = (not offset) or _centered_offset(offset, W)
    if centered:
        offset_tuple, centered_w = None, (W - 1) // 2
        off_for_stream = tuple(a - centered_w for a in range(M_primary))
    else:
        offset_tuple, centered_w = tuple(offset), None
        off_for_stream = tuple(offset)

    per_batch_size = math.prod((M_primary, M_secondary, W, B_row, B_col, *L))

    def _one_batch(data_batch: Array) -> Array:
        if per_batch_size > _BLOCK_BANDED_BROADCAST_LIMIT:
            return _per_band_stream(
                data_batch, M_primary, M_secondary, W, B_row, B_col,
                off_for_stream, primary_axis, fill_value, L,
            )
        if primary_axis == 0:
            return _to_dense_banded(
                data_batch, M_primary, M_secondary, W, B_row, B_col,
                offset_tuple, centered_w, fill_value, L, L_pad,
            )
        # Col-primary: swap sub-block axes, densify row-primary, transpose back.
        data_t = data_batch.swapaxes(2, 3)
        dense_T = _to_dense_banded(
            data_t, M_primary, M_secondary, W, B_col, B_row,
            offset_tuple, centered_w, fill_value, L, L_pad,
        )
        return dense_T.swapaxes(0, 1)

    if n_meta == 1:
        return _one_batch(data)

    # n_meta > 1: per-batch densify, then stitch onto the n_meta meta-diagonal.
    data_b = data.reshape(n_meta, M_primary, W, B_row, B_col, *L)
    if per_batch_size * n_meta > _BLOCK_BANDED_BROADCAST_LIMIT:
        per_batch_dense = jax.lax.map(_one_batch, data_b)  # sequential, bounded HBM
    else:
        per_batch_dense = jax.vmap(_one_batch)(data_b)
    return _stitch_meta(per_batch_dense, fill_value)


def _densify_multi_banded(
    data: Array,
    axes: "tuple[BandAxisSpec, ...]",
    fill_value: Array,
) -> Array:
    """Materialize the dense form of a K-axis block-banded buffer for ANY K
    via a single fused per-axis broadcast+where+sum. Standalone successor to
    ``MultiAxisBlockBanded.to_dense`` (Phase 8) so the K≥2 densify no longer
    depends on the legacy ``MultiAxisBlockBanded`` pytree class.

    ``data`` layout: ``(M_p_0, W_0, ..., M_p_{K-1}, W_{K-1}, B_row_0, ...,
    B_row_{K-1}, B_col_0, ..., B_col_{K-1}, *L)``. ``axes`` is K
    :class:`BandAxisSpec`, one per output axis pair.

    Algorithm: insert per-axis ``M_s_i`` secondary axes (singleton broadcast),
    build a one-hot band-selection mask per axis, AND them, ``where``+sum the
    W axes in one reduction, then permute/reshape to the dense layout and apply
    the per-axis in-band fill mask. K=1 routes to :func:`_densify_band`.
    """
    K = len(axes)

    # Per-axis bookkeeping (Python ints — static at trace time).
    M_p = [data.shape[2 * i] // ax.n_meta for i, ax in enumerate(axes)]
    M_s = [
        ax.n_secondary if ax.n_secondary >= 0 else M_p[i]
        for i, ax in enumerate(axes)
    ]
    W = [ax.band_width for ax in axes]
    B_row = [ax.block_row for ax in axes]
    B_col = [ax.block_col for ax in axes]
    offsets = []
    centered = []  # per-axis: True ⇒ in-band test is pure arithmetic (no gather)
    for i, ax in enumerate(axes):
        w = (ax.band_width - 1) // 2
        if ax.offset:
            offsets.append(ax.offset)
            centered.append(_centered_offset(tuple(ax.offset), ax.band_width))
        else:
            offsets.append(tuple(a - w for a in range(M_p[i])))
            centered.append(True)  # () sentinel ⇒ centered band
    w0 = [(ax.band_width - 1) // 2 for ax in axes]
    L = data.shape[4 * K :]

    if K == 1:
        ax = axes[0]
        return _densify_band(
            data, M_p[0], M_s[0], W[0], B_row[0], B_col[0],
            tuple(ax.offset), ax.primary_axis, ax.n_meta, fill_value, tuple(L),
        )

    for ax in axes:
        if ax.n_meta != 1 or ax.primary_axis != 0:
            raise NotImplementedError(
                "multi-axis banded densify K>1 supports n_meta=1 + row-primary "
                "per axis. Multi-batch and col-primary per-axis follow the "
                "K=1 patterns; add when needed."
            )

    # Step 1: insert M_s_i singleton axes after each (M_p_i, W_i) pair.
    indexer = []
    for _ in range(K):
        indexer.append(slice(None))  # M_p_i
        indexer.append(None)          # M_s_i (inserted)
        indexer.append(slice(None))  # W_i
    indexer += [slice(None)] * (2 * K + len(L))  # B_row, B_col, L axes
    data_e = data[tuple(indexer)]
    expanded = []
    for i in range(K):
        expanded += [M_p[i], M_s[i], W[i]]
    expanded += B_row
    expanded += B_col
    expanded += list(L)
    data_b = jnp.broadcast_to(data_e, tuple(expanded))

    # Step 2: per-axis one-hot band masks, combined via AND.
    combined_mask = None
    for i in range(K):
        off_arr = jnp.asarray(offsets[i], dtype=jnp.int32)
        bj = jnp.arange(M_s[i], dtype=jnp.int32)
        target_w = bj[None, :] - off_arr[:, None]  # (M_p_i, M_s_i)
        sel = target_w[:, :, None] == jnp.arange(W[i], dtype=jnp.int32)[None, None, :]
        sel_shape = [1] * len(expanded)
        sel_shape[3 * i] = M_p[i]
        sel_shape[3 * i + 1] = M_s[i]
        sel_shape[3 * i + 2] = W[i]
        sel_r = sel.reshape(*sel_shape)
        combined_mask = sel_r if combined_mask is None else (combined_mask & sel_r)

    # Step 3: where + sum over all W_i axes in one fused reduction.
    W_axes = tuple(3 * i + 2 for i in range(K))
    out = jnp.where(combined_mask, data_b, 0).sum(axis=W_axes)

    # Step 4: permute + reshape to dense layout.
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

    # Step 5: AND per-axis in-band masks, swap fill_value for out-of-band cells.
    # For a centered band the per-primary col offset is ``blk_r - w0`` (pure
    # arithmetic) — avoids the ``off_arr[blk_r]`` gather (CR-3, mirroring the
    # single-axis ``_to_dense_banded``); only an explicit non-centered offset
    # falls back to the gather.
    per_axis_in_band = []
    for i in range(K):
        blk_r = jnp.arange(M_p[i] * B_row[i]) // B_row[i]
        blk_c = jnp.arange(M_s[i] * B_col[i]) // B_col[i]
        if centered[i]:
            co = blk_r - w0[i]
        else:
            off_arr = jnp.asarray(offsets[i], dtype=jnp.int32)
            co = off_arr[blk_r]
        diff = blk_c[None, :] - co[:, None]
        per_axis_in_band.append((diff >= 0) & (diff < W[i]))
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
    return jnp.where(full_mask, out, fill_value)


# ----------------------------------------------------------------------------
#  Per-axis band metadata for the K-axis densify helper
# ----------------------------------------------------------------------------
class BandAxisSpec(NamedTuple):
    """Per-axis band geometry for :func:`_densify_multi_banded`.

    Each output axis-pair of a K-axis banded buffer gets one of these —
    together they encode K independent bands. Built from the per-axis
    ``BandedIndex`` dims at the densify boundary (``utils._densify_compressed_dims``).
    """

    primary_axis: int = 0     # 0 = row-primary, 1 = col-primary (this axis pair)
    n_secondary: int = -1     # -1 = M_per_primary (square per-batch)
    offset: tuple[int, ...] = ()  # () = centered band; else explicit per-row offsets
    n_meta: int = 1           # outer batch over independent bands within this axis
    band_width: int = 1       # W for this axis
    block_row: int = 1        # B_row of sub-blocks for this axis
    block_col: int = 1        # B_col of sub-blocks for this axis
