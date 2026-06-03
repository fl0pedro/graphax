"""Transpose a ``SparseTensor`` by reordering its out and primal dimensions.

View-only: ``val`` is not permuted. Each dim's ``axis`` / ``block_axis`` already
encodes the physical position in ``val``; consumers index by ``dim.axis``, so a
logical reorder of the dims tuple needs no array work. Sparse pairs are still
required to straddle the out/primal split — if a permutation would land both
ends on the same side, that pair is densified first.

Compressed dims (``BandedIndex`` / ``SetIndex``) describe band / set-theoretic
structure whose physical layout is NOT a plain ``axis``-indexed view, but the
out↔primal swap (a 2-D ``.T``) maps cleanly onto the compressed buffer — see
:func:`_try_compressed_transpose` — so it is done as a view that *preserves
compression*. Any other permutation of a compressed tensor (a K≥2 multi-axis
band, or a partial reorder) falls back to pre-densifying the compressed dims to
their ``DiagonalIndex`` / ``DenseIndex`` equivalents (``compact=True`` keeps the
``M×`` meta-block-diagonal form wherever the structure reduces to a diagonal)
before the relabel-only transpose runs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
from dataclasses import replace

import jax.numpy as jnp

from .dense import dense
from .utils import _copy, _compressed_dims, _densify_compressed_dims

from graphax.sparse.indexes import DiagonalIndex, BandedIndex, SetIndex

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


def _multi_banded_transpose(tensor, full_perm, new_out_axes):
    """View-transpose a K≥2 all-``BandedIndex`` tensor under any band-preserving
    permutation, keeping it compressed. Returns the transposed tensor, or
    ``None`` (fall back to densify) when the permutation splits a band pair
    across slots or doesn't straddle the out/primal boundary.

    The K val pairs (each ``M_p_i, W_i, B_row_i, B_col_i`` in the interleaved
    band buffer) are permuted to their new slots; a pair whose row/col roles
    flip gets its leaf ``(B_row, B_col)`` axes swapped and ``primary`` flipped —
    the K-axis generalization of the 2-D band swap, materialized by the
    col-primary path in ``_densify_multi_banded``."""
    K = len(tensor.out_dims)
    if len(tensor.primal_dims) != K or len(new_out_axes) != K:
        return None
    # inverse perm: new logical position of each old logical dim. Pair i's row
    # is old logical i (out[i]); its col is old logical K+i (primal[i]).
    inv = [0] * (2 * K)
    for q, old in enumerate(full_perm):
        inv[old] = q
    slot_src = [None] * K  # new slot s ← (old pair i, flipped?)
    for i in range(K):
        rp, cp = inv[i], inv[K + i]
        if rp % K != cp % K or (rp < K) == (cp < K):
            return None  # pair split across slots, or both ends same side
        s = rp % K
        if slot_src[s] is not None:
            return None
        slot_src[s] = (i, rp >= K)  # flipped iff the original row landed in primal

    nd = tensor.val.ndim
    perm_val = list(range(nd))
    out_old, primal_old = tensor.out_dims, tensor.primal_dims
    new_out, new_primal = [], []
    for s in range(K):
        i, flipped = slot_src[s]
        perm_val[2 * s] = 2 * i          # M_p
        perm_val[2 * s + 1] = 2 * i + 1  # W
        perm_val[2 * K + s] = (3 * K + i) if flipped else (2 * K + i)  # B_row (swap if flipped)
        perm_val[3 * K + s] = (2 * K + i) if flipped else (3 * K + i)  # B_col
        row_src, col_src = (primal_old[i], out_old[i]) if flipped else (out_old[i], primal_old[i])
        new_out.append(replace(row_src, id=s, axis=s, other_id=K + s,
                               block_axis=2 * K + s,
                               primary=(not row_src.primary) if flipped else row_src.primary))
        new_primal.append(replace(col_src, id=K + s, axis=K + s, other_id=s,
                                  block_axis=3 * K + s,
                                  primary=(not col_src.primary) if flipped else col_src.primary))
    new_val = tensor.val.transpose(perm_val)
    return _copy(tensor, val=new_val, out_dims=tuple(new_out), primal_dims=tuple(new_primal))


def _try_compressed_transpose(tensor, full_perm, new_out_axes):
    """View-transpose a compressed tensor (``BandedIndex`` / ``SetIndex``)
    preserving compression. Returns the transposed ``SparseTensor``, or ``None``
    to tell the caller to fall back to densify (a permutation that isn't a clean
    band-preserving / out↔primal swap).

    ``BandedIndex``: ``densify(val.swapaxes(2,3), col-primary) ==
    densify(val, row-primary).swapaxes(0,1) == Dᵀ`` — so swap each leaf block's
    ``(B_row, B_col)`` axes and flip the band direction; the offset is reused
    unchanged (handled for any K by :func:`_multi_banded_transpose` + the
    col-primary path in ``_densify_multi_banded``). ``SetIndex``: the set op
    (add/mul) is elementwise so ``op(BD(lhs), BD(rhs))ᵀ == op(BD(lhsᵀ),
    BD(rhsᵀ))`` — transpose each per-side block buffer's ``(B_h, B_w)`` axes and
    swap the ``lhs``/``rhs`` shapes h↔w. In both, the out↔primal dims swap."""
    out_d, primal_d = tensor.out_dims, tensor.primal_dims

    # K≥2 all-banded: general band-preserving permutation.
    if (len(out_d) >= 2 and len(out_d) == len(primal_d)
            and all(isinstance(d, BandedIndex) for d in (*out_d, *primal_d))):
        return _multi_banded_transpose(tensor, full_perm, new_out_axes)

    # 2-D pure pair (K=1) under the out↔primal swap.
    if len(out_d) != 1 or len(primal_d) != 1:
        return None
    if tuple(full_perm) != (1, 0) or len(new_out_axes) != 1:
        return None
    o, p = tensor.out_dims[0], tensor.primal_dims[0]

    if isinstance(o, BandedIndex) and isinstance(p, BandedIndex):
        new_val = tensor.val.swapaxes(2, 3)  # swap leaf (B_row, B_col)
        new_out = (replace(p, id=0, axis=0, other_id=1, block_axis=1,
                           primary=not p.primary),)
        new_primal = (replace(o, id=1, axis=1, other_id=0, block_axis=3,
                              primary=not o.primary),)
        return _copy(tensor, val=new_val, out_dims=new_out, primal_dims=new_primal)

    if isinstance(o, SetIndex) and isinstance(p, SetIndex):
        lhs, rhs = o._split(tensor.val)
        lhs_t = lhs.swapaxes(2, 3)
        lhs_shape_t = lhs.shape[:2] + (lhs.shape[3], lhs.shape[2]) + lhs.shape[4:]
        if rhs is not None:
            rhs_t = rhs.swapaxes(2, 3)
            rhs_shape_t = rhs.shape[:2] + (rhs.shape[3], rhs.shape[2]) + rhs.shape[4:]
            new_val = jnp.concatenate([lhs_t.reshape(-1), rhs_t.reshape(-1)])
        else:
            rhs_shape_t = o.rhs_shape
            new_val = lhs_t.reshape(-1)
        new_out = (replace(p, id=0, axis=0, other_id=1, block_axis=1,
                           lhs_shape=lhs_shape_t, rhs_shape=rhs_shape_t),)
        new_primal = (replace(o, id=1, axis=1, other_id=0, block_axis=2,
                              lhs_shape=lhs_shape_t, rhs_shape=rhs_shape_t),)
        return _copy(tensor, val=new_val, out_dims=new_out, primal_dims=new_primal)

    return None


def _get_full_permutation(num_out, num_primal, out_axes=None, primal_axes=None):
    """Normalize and validate the permutation. Default is full reverse: ``out`` becomes the
    last ``num_primal`` axes (reversed) and ``primal`` becomes the first ``num_out`` (reversed)."""
    total_ndim = num_out + num_primal
    if out_axes is None and primal_axes is None:
        full_reversed = tuple(range(total_ndim - 1, -1, -1))
        out_axes = list(full_reversed[:num_primal])
        primal_axes = list(full_reversed[num_primal:])
    else:
        out_axes = list(out_axes) if out_axes is not None else list(range(num_out))
        primal_axes = list(primal_axes) if primal_axes is not None else list(range(num_out, total_ndim))
    out_axes = [i % total_ndim for i in out_axes]
    primal_axes = [i % total_ndim for i in primal_axes]
    full = tuple(out_axes) + tuple(primal_axes)
    if len(full) != total_ndim or len(set(full)) != total_ndim:
        raise ValueError(f"Invalid transpose permutation: {full} for ndim {total_ndim}")
    return full, out_axes, primal_axes


def _ensure_valid_sparsity(tensor, out_axes, primal_axes):
    """Densify sparse pairs whose two members would land on the same side post-transpose."""
    dim_id_to_index = {d.id: i for i, d in enumerate(tensor.dims)}
    out_set, primal_set = set(out_axes), set(primal_axes)
    axes_to_densify = [
        i for i, dim in enumerate(tensor.dims)
        if dim.is_sparse
        and ((i in out_set and dim_id_to_index[dim.other_id] in out_set)
             or (i in primal_set and dim_id_to_index[dim.other_id] in primal_set))
    ]
    if axes_to_densify:
        tensor = dense(tensor, axes=tuple(axes_to_densify), hard=True)
    return tensor


def transpose(tensor: SparseTensor, out_axes: Sequence[int] | None = None,
              primal_axes: Sequence[int] | None = None) -> SparseTensor:
    full_perm, new_out_axes, new_primal_axes = _get_full_permutation(
        len(tensor.out_dims), len(tensor.primal_dims), out_axes, primal_axes)

    # Identity permutation + unchanged out/primal split → no-op.
    if (full_perm == tuple(range(len(full_perm)))
            and len(new_out_axes) == len(tensor.out_dims)):
        return tensor

    # Compressed dims: the 2-D out↔primal swap maps onto the compressed buffer
    # directly (preserving compression); take that view when it applies.
    # Otherwise materialize the compressed dims to their Diagonal / Dense
    # equivalents (compact form keeps the meta-block-diagonal compression where
    # the structure reduces to a diagonal) and transpose via the relabel below.
    if _compressed_dims(tensor):
        viewed = _try_compressed_transpose(tensor, full_perm, new_out_axes)
        if viewed is not None:
            return viewed
        tensor = _densify_compressed_dims(tensor, compact=True)

    tensor = _ensure_valid_sparsity(tensor, new_out_axes, new_primal_axes)

    reordered = [tensor.dims[i] for i in full_perm]
    id_map = {d.id: i for i, d in enumerate(reordered)}
    updated = [
        replace(d, id=i, **({"other_id": id_map[d.other_id]} if d.is_sparse else {}))
        for i, d in enumerate(reordered)
    ]

    n_out = len(new_out_axes)
    return _copy(tensor, val=tensor.val,
                 out_dims=tuple(updated[:n_out]), primal_dims=tuple(updated[n_out:]))
