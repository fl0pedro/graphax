"""Transpose a ``SparseTensor`` by reordering its out and primal dimensions.

View-only: ``val`` is not permuted. Each dim's ``axis`` / ``block_axis`` already
encodes the physical position in ``val``; consumers index by ``dim.axis``, so a
logical reorder of the dims tuple needs no array work. Sparse pairs are still
required to straddle the out/primal split — if a permutation would land both
ends on the same side, that pair is densified first.

Compressed dims (``BandedIndex`` / ``SetIndex``) describe band / set-theoretic
structure whose physical layout is NOT a plain ``axis``-indexed view, so the
relabel-only transpose below would corrupt them. They are pre-densified to their
``DiagonalIndex`` / ``DenseIndex`` equivalents (``compact=True`` keeps the
``M×`` meta-block-diagonal form wherever the structure reduces to a diagonal —
every ``SetIndex`` and any width-1 band — and only fully materializes a genuine
band) before the view transpose runs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
from dataclasses import replace

from .dense import dense
from .utils import _copy, _compressed_dims, _densify_compressed_dims

from graphax.sparse.indexes import DiagonalIndex

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


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

    # Compressed dims can't be transposed as a relabel-only view — materialize
    # them to their Diagonal / Dense equivalents first (compact form preserves
    # the meta-block-diagonal compression wherever the structure reduces to a
    # diagonal). The resulting Diagonal / Dense dims transpose correctly below.
    if _compressed_dims(tensor):
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
