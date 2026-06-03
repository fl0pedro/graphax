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


def _try_compressed_transpose(tensor, full_perm, new_out_axes):
    """View-transpose a pure 2-D compressed pair (one out + one primal
    ``BandedIndex`` / ``SetIndex``) under the out↔primal swap, preserving
    compression. Returns the transposed ``SparseTensor``, or ``None`` to tell
    the caller to fall back to densify (K≥2 multi-axis bands, or any
    permutation that isn't the plain 2-D swap).

    ``BandedIndex``: ``densify(val.swapaxes(2,3), col-primary) ==
    densify(val, row-primary).swapaxes(0,1) == Dᵀ`` — so swap each leaf block's
    ``(B_row, B_col)`` axes and flip the band direction; the offset is reused
    unchanged. ``SetIndex``: the set op (add/mul) is elementwise so
    ``op(BD(lhs), BD(rhs))ᵀ == op(BD(lhsᵀ), BD(rhsᵀ))`` — transpose each per-side
    block buffer's ``(B_h, B_w)`` axes and swap the ``lhs``/``rhs`` shapes h↔w.
    In both, the out↔primal dims swap (carrying their block sizes / band params)."""
    if len(tensor.out_dims) != 1 or len(tensor.primal_dims) != 1:
        return None
    # The only meaningful 2-D transpose is the out↔primal swap: perm (1, 0) with
    # a single out axis afterwards.
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
