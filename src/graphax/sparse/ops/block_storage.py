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


# ----------------------------------------------------------------------------
#  3.  BlockBanded — matmul of two block-diagonals (x @ y.T or x.T @ y)
# ----------------------------------------------------------------------------
class BlockBanded(NamedTuple):
    """Block-banded matrix in skewed (rectangular) storage.

    Output of contracting two block-diagonal sources where the contracting axes'
    block sizes don't divide evenly: the result is non-zero only within a finite
    block-band, since each lhs block can only couple to a few rhs blocks whose
    block-ranges along the contracted axis overlap with it.

    Layout
    ------
    ``data`` : ``(M, 2w+1, B, B, *L)``
        ``data[k, b]`` is the block at meta-row ``k``, meta-column ``k+(b-w)``.
        ``b == w`` is the main diagonal; ``b > w`` upper bands; ``b < w`` lower.
    ``fill_value`` : value at positions outside the band.

    The half-bandwidth ``w = (data.shape[1] - 1) // 2`` is determined by the
    storage. Out-of-range slots at the corners (``k+b-w < 0`` or ``≥ M``) are
    masked out by ``to_dense``; this is the standard banded-matrix-storage
    tradeoff for a rectangular buffer ("shifted into one tensor").
    """

    data: Array
    fill_value: Array

    @property
    def half_bandwidth(self) -> int:
        return (self.data.shape[1] - 1) // 2

    @property
    def shape(self) -> tuple[int, ...]:
        M, _, B, _, *L = self.data.shape
        return (M * B, M * B, *L)

    @property
    def meta_block_shape(self) -> tuple[int, int, int] | None:
        """``(M, B, B)`` when ``w == 0`` (pure block-diagonal); ``None`` for
        ``w > 0`` because banded structure can't be expressed as a single
        meta-block-diagonal SparseTensor pair."""
        if self.half_bandwidth == 0:
            M, _, B, _, *_ = self.data.shape
            return (M, B, B)
        return None

    def to_meta_blocks(self) -> Array:
        """``(M, B, B, *L)`` meta-diagonal blocks. Only defined when ``w == 0``;
        for ``w > 0`` use :meth:`to_dense` directly."""
        if self.half_bandwidth != 0:
            raise ValueError(
                f"BlockBanded.to_meta_blocks requires w=0; got w={self.half_bandwidth}. "
                f"Use to_dense() for banded forms."
            )
        return self.data[:, 0]   # (M, B, B, *L)

    def to_dense(self) -> Array:
        """Materialize the dense ``(M*B, M*B, *L)`` block-banded form via a
        single fused broadcast+select+sum chain — no scatter, **no gather**.

        Algorithm. For each output meta-position ``(bi, bj)`` the in-band band
        slot is ``b_target = bj - bi + w``; if ``b_target ∈ [0, W)`` the cell
        is ``data[bi, b_target, si, sj]``, else ``fill_value``. We can avoid the
        per-cell index lookup (which the prior implementation expressed as a
        1-D fancy gather) by:

          1. broadcasting ``data`` across a new meta-col axis to logical shape
             ``(M, M, W, B, B, *L)`` (pure broadcast, zero-copy);
          2. masking with a one-hot ``b == b_target`` selector along W;
          3. summing over W to collapse the band axis.

        Since exactly one ``b`` matches per in-band ``(bi, bj)`` (and zero
        match out-of-band), the sum is equivalent to a per-cell select. XLA
        fuses the broadcast+where+sum into one ``kLoop`` pass — no gather,
        no scatter, no W× HBM materialization. That makes the densify
        forward-fusable into a downstream consumer kernel (gathers force a
        materialization barrier; this no longer does).

        Cost. The logical intermediate is W× the output size; for typical
        small bandwidths (``W ∈ {1, 3, 5}``) this is negligible, and XLA's
        loop-fuser collapses it.
        """
        M, W, B, B_, *L = self.data.shape
        if B != B_:
            raise ValueError(f"banded blocks must be square, got ({B}, {B_})")
        if W % 2 == 0:
            raise ValueError(
                f"BlockBanded data axis 1 must be 2w+1 (odd); got data.shape[1]={W}"
            )
        w = (W - 1) // 2
        L_pad = (None,) * len(L)

        if math.prod((M, M, W, B, B, *L)) > _BLOCK_BANDED_BROADCAST_LIMIT:
            return self._to_dense_per_band(M, W, B, w, L)

        # Step 1: tile data across a new meta-col axis (pure broadcast, no copy).
        data_bcast = jnp.broadcast_to(
            self.data[:, None, ...],  # (M, 1, W, B, B, *L)
            (M, M, W, B, B, *L),
        )

        # Step 2: build the one-hot ``b == bj - bi + w`` selector along W.
        bi_idx = jnp.arange(M)[:, None]
        bj_idx = jnp.arange(M)[None, :]
        select_mask = (bj_idx - bi_idx + w)[..., None] == jnp.arange(W)[None, None, :]
        select_mask = select_mask[..., None, None]  # (M, M, W, 1, 1)
        if L:
            select_mask = select_mask[(..., *L_pad)]

        # Step 3: where+sum collapses W → out_meta has data on the band, 0 elsewhere.
        out_meta = jnp.where(select_mask, data_bcast, 0).sum(axis=2)  # (M, M, B, B, *L)

        # Step 4: interleave (bi, si, bj, sj) → (bi*B+si, bj*B+sj).
        perm = (0, 2, 1, 3, *range(4, 4 + len(L)))
        out_meta = out_meta.transpose(perm).reshape(M * B, M * B, *L)

        # Step 5: replace the zero-pad outside the band with fill_value.
        blk_i = jnp.arange(M * B) // B
        blk_j = jnp.arange(M * B) // B
        in_band = jnp.abs(blk_j[None, :] - blk_i[:, None]) <= w  # (M*B, M*B)
        if L:
            in_band = in_band[(..., *L_pad)]
        return jnp.where(in_band, out_meta, self.fill_value)

    def _to_dense_per_band(
        self, M: int, W: int, B: int, w: int, L: tuple[int, ...]
    ) -> Array:
        """Per-band ``lax.fori_loop`` fallback for the dense materialization.

        The broadcast path's logical intermediate is ``(M, M, W, B, B, *L)`` —
        for ``M`` in the thousands the static shape exceeds practical limits
        even though XLA fuses the ``where+sum`` at runtime. This path iterates
        the ``M`` rows of each band and writes them into the result via
        ``lax.dynamic_update_slice``, never materializing the ``M × M`` square.
        """
        out = jnp.full((M * B, M * B, *L), self.fill_value, dtype=self.data.dtype)
        zero_idx = (jnp.int32(0),) * len(L)

        def body(k, acc):
            for b in range(W):
                col = k + (b - w)
                in_range = jnp.logical_and(col >= 0, col < M)
                row_start = jnp.int32(k * B)
                col_start = jnp.int32(col * B)
                block = self.data[k, b]
                existing = lax.dynamic_slice(
                    acc, (row_start, col_start) + zero_idx, (B, B, *L)
                )
                replacement = jnp.where(in_range, block, existing)
                acc = lax.dynamic_update_slice(
                    acc, replacement, (row_start, col_start) + zero_idx
                )
            return acc

        return lax.fori_loop(0, M, body, out)


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
