# pyright: reportImportCycles=false
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, TypeAlias
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .utils import _arr2st
from .layout import generate_block_permutation, generate_grouped_permutation

from ..dimensions import Dimension, SparseDimension, DenseDimension

if TYPE_CHECKING:
    from ..tensor import SparseTensor


AXES_PER_PAIR = 3  # (outer, block, shared_block)
SPLIT_AXES = 4  # (outer_split, block, outer_remain, shared_remain)
GRID_AXES_PER_PAIR = 3  # (shared, lhs, rhs)

PairingType: TypeAlias = Literal[
    "batch_sparse",
    "batch_out",
    "batch_primal",
    "spatial_sparse_lhs",
    "patial_out_lhs",
    "spatial_primal_lhs",
    "spatial_sparse_rhs",
    "spatial_out_rhs",
    "spatial_primal_rhs",
    "contract",
]


@dataclass(frozen=True)
class DimensionPairData:
    """Core length and val_dim information for a pair of dimensions."""

    outer_len: int
    block_len: int
    shared_block_len: int
    outer_val_dim: int | None = None
    block_val_dim: int | None = None
    shared_block_val_dim: int | None = None
    dim: Dimension | None = None
    shared_dim: Dimension | None = None


@dataclass(frozen=True)
class DimensionPair:
    """Consolidated metadata for a pair of interacting dimensions."""

    pairing_type: PairingType
    logical_element_count: int
    lhs: DimensionPairData
    rhs: DimensionPairData


@dataclass(frozen=True)
class MatmulContext:
    """Global state for a matmul operation to reduce argument passing."""

    lhs: SparseTensor
    rhs: SparseTensor
    pairs: list[DimensionPair]
    rhs_id_offset: int

    @property
    def num_pairs(self) -> int:
        return len(self.pairs)


@dataclass(frozen=True)
class ContractionResult:
    """Groups results from the block-sparse contraction step."""

    grid: Array
    shared_tiling_factors: list[int]
    lhs_block_lens: list[int]
    rhs_block_lens: list[int]
    scalar_multiplier: float


def _is_sparse(obj) -> bool: # maybe we can add this to dimensions
    return getattr(type(obj), "__name__", "") == "SparseTensor"

@jax.custom_vjp
def matmul(lhs: SparseTensor | Array, rhs: SparseTensor | Array, count: bool = False) -> SparseTensor | Array:
    from ..tensor import SparseTensor

    if not _is_sparse(lhs) and not _is_sparse(rhs):
        return jnp.matmul(lhs, rhs)
    if not _is_sparse(lhs):
        assert _is_sparse(rhs)
        lhs = _arr2st(lhs, out_ndim=lhs.ndim - len(rhs.out_dims))
    if not _is_sparse(rhs):
        assert _is_sparse(lhs)
        rhs = _arr2st(rhs, out_ndim=len(lhs.primal_dims))

    rhs_out_dims, rhs_primal_dims, rhs_id_offset = _align_tensor_ids(lhs, rhs)
    rhs_dims = rhs_out_dims + rhs_primal_dims

    pairs = _build_matmul_topology(lhs, rhs_out_dims, rhs_primal_dims, rhs_id_offset)

    ctx = MatmulContext(
        lhs=lhs,
        rhs=rhs,
        pairs=pairs,
        rhs_id_offset=rhs_id_offset,
    )

    left_val = lhs.val if lhs.val is not None else jnp.array(1.0, dtype=lhs.dtype)
    right_val = rhs.val if rhs.val is not None else jnp.array(1.0, dtype=rhs.dtype)

    left_val, right_val = _prepare_physical_arrays(left_val, right_val, ctx.pairs)

    (
        contracted_grid,
        shared_tiling_factors,
        final_lhs_block_lens,
        final_rhs_block_lens,
        scalar_multiplier,
        adds,
        muls,
        fmas
    ) = _execute_block_sparse_contraction(left_val, right_val, ctx.pairs, count)

    res = ContractionResult(
        grid=contracted_grid,
        shared_tiling_factors=shared_tiling_factors,
        lhs_block_lens=final_lhs_block_lens,
        rhs_block_lens=final_rhs_block_lens,
        scalar_multiplier=scalar_multiplier,
    )

    st, (_adds, _muls, _fmas) = _build_output_tensor(ctx, rhs_dims, res, count)
    if count is not False:
        return st, (_adds + adds, _muls + muls, _fmas + fmas)
    return st


def _resolve_matched_dims(
    lhs_out_dim: Dimension | None,
    lhs_primal_dim: Dimension | None,
    rhs_out_dim: Dimension | None,
    rhs_primal_dim: Dimension | None,
) -> DimensionPair | None:
    if (
        lhs_out_dim
        and rhs_out_dim
        and lhs_out_dim.logical_size != rhs_out_dim.logical_size
    ):
        raise ValueError(
            f"Implicit batch mapping disallowed: Matched output dimensions (ID {lhs_out_dim.id}) "
            f"have different sizes ({lhs_out_dim.logical_size} != {rhs_out_dim.logical_size})."
        )
    if (
        lhs_primal_dim
        and rhs_primal_dim
        and lhs_primal_dim.logical_size != rhs_primal_dim.logical_size
    ):
        raise ValueError(
            f"Implicit batch mapping disallowed: Matched primal dimensions (ID {lhs_primal_dim.id}) "
            f"have different sizes ({lhs_primal_dim.logical_size} != {rhs_primal_dim.logical_size})."
        )

    if lhs_out_dim and lhs_primal_dim and rhs_out_dim and rhs_primal_dim:
        l_out_len, _ = _get_dim_vals(lhs_out_dim, True)
        r_out_len, _ = _get_dim_vals(rhs_primal_dim, True)
        l_block_len, l_block_val_dim = _get_dim_vals(lhs_out_dim, False)
        l_shared_len, l_shared_val_dim = _get_dim_vals(lhs_primal_dim, False)
        r_block_len, r_block_val_dim = _get_dim_vals(rhs_out_dim, False)
        r_shared_len, r_shared_val_dim = _get_dim_vals(rhs_primal_dim, False)

        return DimensionPair(
            pairing_type="batch_sparse",
            logical_element_count=1,
            lhs=DimensionPairData(
                outer_len=l_out_len,
                block_len=l_block_len,
                shared_block_len=l_shared_len,
                outer_val_dim=_get_sparse_outer_v(lhs_out_dim, lhs_primal_dim),
                block_val_dim=l_block_val_dim,
                shared_block_val_dim=l_shared_val_dim,
                dim=lhs_out_dim,
                shared_dim=lhs_primal_dim,
            ),
            rhs=DimensionPairData(
                outer_len=r_out_len,
                block_len=r_block_len,
                shared_block_len=r_shared_len,
                outer_val_dim=_get_sparse_outer_v(rhs_primal_dim, rhs_out_dim),
                block_val_dim=r_block_val_dim,
                shared_block_val_dim=r_shared_val_dim,
                dim=rhs_out_dim,
                shared_dim=rhs_primal_dim,
            ),
        )
    elif lhs_out_dim and rhs_out_dim:
        l_out_len, l_out_val_dim = _get_dim_vals(lhs_out_dim, False)
        r_out_len, r_out_val_dim = _get_dim_vals(rhs_out_dim, False)
        return DimensionPair(
            pairing_type="batch_out",
            logical_element_count=1,
            lhs=DimensionPairData(
                outer_len=l_out_len,
                block_len=1,
                shared_block_len=1,
                outer_val_dim=l_out_val_dim,
                dim=lhs_out_dim,
            ),
            rhs=DimensionPairData(
                outer_len=r_out_len,
                block_len=1,
                shared_block_len=1,
                outer_val_dim=r_out_val_dim,
                dim=rhs_out_dim,
            ),
        )
    elif lhs_primal_dim and rhs_primal_dim:
        l_out_len, l_out_val_dim = _get_dim_vals(lhs_primal_dim, False)
        r_out_len, r_out_val_dim = _get_dim_vals(rhs_primal_dim, False)
        return DimensionPair(
            pairing_type="batch_primal",
            logical_element_count=1,
            lhs=DimensionPairData(
                outer_len=l_out_len,
                block_len=1,
                shared_block_len=1,
                shared_block_val_dim=l_out_val_dim,
                shared_dim=lhs_primal_dim,
            ),
            rhs=DimensionPairData(
                outer_len=r_out_len,
                block_len=1,
                shared_block_len=1,
                shared_block_val_dim=r_out_val_dim,
                shared_dim=rhs_primal_dim,
            ),
        )

    return None


def _resolve_unmatched_lhs_dims(
    lhs_out_dim: Dimension | None, lhs_primal_dim: Dimension | None
) -> DimensionPair | None:
    if lhs_out_dim and lhs_primal_dim:
        l_out_len, _ = _get_dim_vals(lhs_out_dim, True)
        l_block_len, l_block_val_dim = _get_dim_vals(lhs_out_dim, False)
        l_shared_len, l_shared_val_dim = _get_dim_vals(lhs_primal_dim, False)
        return DimensionPair(
            pairing_type="spatial_sparse_lhs",
            logical_element_count=1,
            lhs=DimensionPairData(
                outer_len=l_out_len,
                block_len=l_block_len,
                shared_block_len=l_shared_len,
                outer_val_dim=_get_sparse_outer_v(lhs_out_dim, lhs_primal_dim),
                block_val_dim=l_block_val_dim,
                shared_block_val_dim=l_shared_val_dim,
                dim=lhs_out_dim,
                shared_dim=lhs_primal_dim,
            ),
            rhs=DimensionPairData(outer_len=1, block_len=1, shared_block_len=1),
        )
    elif lhs_out_dim:
        l_block_len, l_block_val_dim = _get_dim_vals(lhs_out_dim, False)
        return DimensionPair(
            pairing_type="spatial_out_lhs",
            logical_element_count=1,
            lhs=DimensionPairData(
                outer_len=1,
                block_len=l_block_len,
                shared_block_len=1,
                block_val_dim=l_block_val_dim,
                dim=lhs_out_dim,
            ),
            rhs=DimensionPairData(outer_len=1, block_len=1, shared_block_len=1),
        )
    elif lhs_primal_dim:
        l_shared_len, l_shared_val_dim = _get_dim_vals(lhs_primal_dim, False)
        return DimensionPair(
            pairing_type="spatial_primal_lhs",
            logical_element_count=1,
            lhs=DimensionPairData(
                outer_len=1,
                block_len=1,
                shared_block_len=l_shared_len,
                shared_block_val_dim=l_shared_val_dim,
                shared_dim=lhs_primal_dim,
            ),
            rhs=DimensionPairData(outer_len=1, block_len=1, shared_block_len=1),
        )
    return None


def _resolve_unmatched_rhs_dims(
    rhs_out_dim: Dimension | None, rhs_primal_dim: Dimension | None
) -> DimensionPair | None:
    if rhs_out_dim and rhs_primal_dim:
        r_out_len, _ = _get_dim_vals(rhs_primal_dim, True)
        r_block_len, r_block_val_dim = _get_dim_vals(rhs_out_dim, False)
        r_shared_len, r_shared_val_dim = _get_dim_vals(rhs_primal_dim, False)
        return DimensionPair(
            pairing_type="spatial_sparse_rhs",
            logical_element_count=1,
            lhs=DimensionPairData(outer_len=1, block_len=1, shared_block_len=1),
            rhs=DimensionPairData(
                outer_len=r_out_len,
                block_len=r_block_len,
                shared_block_len=r_shared_len,
                outer_val_dim=_get_sparse_outer_v(rhs_primal_dim, rhs_out_dim),
                block_val_dim=r_block_val_dim,
                shared_block_val_dim=r_shared_val_dim,
                dim=rhs_out_dim,
                shared_dim=rhs_primal_dim,
            ),
        )
    elif rhs_out_dim:
        r_block_len, r_block_val_dim = _get_dim_vals(rhs_out_dim, False)
        return DimensionPair(
            pairing_type="spatial_out_rhs",
            logical_element_count=1,
            lhs=DimensionPairData(outer_len=1, block_len=1, shared_block_len=1),
            rhs=DimensionPairData(
                outer_len=1,
                block_len=r_block_len,
                shared_block_len=1,
                block_val_dim=r_block_val_dim,
                dim=rhs_out_dim,
            ),
        )
    elif rhs_primal_dim:
        r_shared_len, r_shared_val_dim = _get_dim_vals(rhs_primal_dim, False)
        return DimensionPair(
            pairing_type="spatial_primal_rhs",
            logical_element_count=1,
            lhs=DimensionPairData(outer_len=1, block_len=1, shared_block_len=1),
            rhs=DimensionPairData(
                outer_len=1,
                block_len=1,
                shared_block_len=r_shared_len,
                shared_block_val_dim=r_shared_val_dim,
                shared_dim=rhs_primal_dim,
            ),
        )
    return None


def _align_tensor_ids(
    lhs: SparseTensor, rhs: SparseTensor
) -> tuple[tuple[Dimension, ...], tuple[Dimension, ...], int]:
    rhs_id_offset = max([d.id for d in lhs.dims] + [-1]) + 1

    def offset_rhs(d: Dimension) -> Dimension:
        kwargs: dict[str, Any] = {"id": d.id + rhs_id_offset}
        if isinstance(d, SparseDimension):
            kwargs["other_id"] = d.other_id + rhs_id_offset
        return replace(d, **kwargs)

    rhs_out_dims = tuple(offset_rhs(d) for d in rhs.out_dims)
    rhs_primal_dims = tuple(offset_rhs(d) for d in rhs.primal_dims)
    return rhs_out_dims, rhs_primal_dims, rhs_id_offset


def _get_dim_vals(
    dim: Dimension | None, is_outer: bool = False
) -> tuple[int, int | None]:
    if not dim:
        return 1, None

    if isinstance(dim, DenseDimension):
        return (1, None) if is_outer else (dim.size, dim.val_dim)
    # SparseDimension
    assert isinstance(dim, SparseDimension)
    if is_outer:
        return dim.size, dim.val_dim
    bs = dim.block_size if dim.block_size is not None else 1
    return bs, dim.block_val_dim


def _get_sparse_outer_v(dim: Dimension | None, sibling: Dimension | None) -> int | None:
    if dim is None:
        return None
    if dim.val_dim is not None:
        return dim.val_dim
    if (
        sibling is not None
        and isinstance(dim, SparseDimension)
        and isinstance(sibling, SparseDimension)
    ):
        if dim.other_id == sibling.id:
            return sibling.val_dim
    return None


def _extract_topo_info(
    d: Dimension,
    dim_map: dict[int, Dimension],
    processed_set: set[int],
    target_list: Sequence[Dimension],
) -> tuple[tuple[Dimension | None, Dimension | None], int]:
    if d.id in processed_set:
        return (None, None), -1

    if not isinstance(d, SparseDimension):
        return ((d, None) if d in target_list else (None, d)), d.id

    other = dim_map.get(d.other_id)
    if not other or other.id in processed_set:
        return ((d, None) if d in target_list else (None, d)), d.id

    return ((d, other) if d in target_list else (other, d)), other.id


def _get_unprocessed_topos(
    dims_list: Sequence[Dimension],
    dim_map: dict[int, Dimension],
    processed_set: set[int],
    target_list: Sequence[Dimension],
) -> list[tuple[Dimension | None, Dimension | None]]:
    topos = []
    seen = set()
    for d in dims_list:
        if d.id in seen:
            continue

        topo, extra_id = _extract_topo_info(d, dim_map, processed_set, target_list)
        if topo != (None, None):
            topos.append(topo)
            if extra_id != -1:
                seen.add(extra_id)
        seen.add(d.id)
    return topos


def _resolve_contract_pair(
    lhs_primal_dim: Dimension,
    rhs_out_dim: Dimension,
    lhs_out_map: dict[int, Dimension],
    rhs_primal_map: dict[int, Dimension],
) -> tuple[DimensionPair, list[int]]:
    if lhs_primal_dim.logical_size != rhs_out_dim.logical_size:
        raise ValueError(
            f"Contraction dimensions must have the same logical size. "
            f"Got LHS size {lhs_primal_dim.logical_size} and RHS size {rhs_out_dim.logical_size}."
        )

    lhs_out_dim = (
        lhs_out_map.get(getattr(lhs_primal_dim, "other_id", -1))
        if isinstance(lhs_primal_dim, SparseDimension)
        else None
    )
    rhs_primal_dim = (
        rhs_primal_map.get(getattr(rhs_out_dim, "other_id", -1))
        if isinstance(rhs_out_dim, SparseDimension)
        else None
    )

    ids = [lhs_primal_dim.id, rhs_out_dim.id]
    if lhs_out_dim:
        ids.append(lhs_out_dim.id)
    if rhs_primal_dim:
        ids.append(rhs_primal_dim.id)

    l_out_len, _ = _get_dim_vals(lhs_out_dim, True)
    l_outer_val_dim = _get_sparse_outer_v(lhs_out_dim, lhs_primal_dim)
    r_out_len, _ = _get_dim_vals(rhs_primal_dim, True)
    r_outer_val_dim = _get_sparse_outer_v(rhs_primal_dim, rhs_out_dim)

    l_block_len, l_block_val_dim = _get_dim_vals(lhs_out_dim, False)
    l_shared_len, l_shared_val_dim = _get_dim_vals(lhs_primal_dim, False)
    r_block_len, r_block_val_dim = _get_dim_vals(rhs_out_dim, False)
    r_shared_len, r_shared_val_dim = _get_dim_vals(rhs_primal_dim, False)

    logical_element_count = lhs_primal_dim.logical_size

    meta = DimensionPair(
        pairing_type="contract",
        logical_element_count=logical_element_count,
        lhs=DimensionPairData(
            outer_len=l_out_len,
            block_len=l_block_len,
            shared_block_len=l_shared_len,
            outer_val_dim=l_outer_val_dim,
            block_val_dim=l_block_val_dim,
            shared_block_val_dim=l_shared_val_dim,
            dim=lhs_out_dim,
            shared_dim=lhs_primal_dim,
        ),
        rhs=DimensionPairData(
            outer_len=r_out_len,
            block_len=r_block_len,
            shared_block_len=r_shared_len,
            outer_val_dim=r_outer_val_dim,
            block_val_dim=r_block_val_dim,
            shared_block_val_dim=r_shared_val_dim,
            dim=rhs_out_dim,
            shared_dim=rhs_primal_dim,
        ),
    )
    return meta, ids


def _resolve_broadcast_topos(
    lhs_topos: list[tuple[Dimension | None, Dimension | None]],
    rhs_topos: list[tuple[Dimension | None, Dimension | None]],
    rhs_id_offset: int,
) -> list[DimensionPair]:
    pairs_meta = []
    remaining_rhs = list(rhs_topos)

    for lhs_out_dim, lhs_primal_dim in lhs_topos:
        match_idx = _find_matching_rhs_topo(
            lhs_out_dim, lhs_primal_dim, remaining_rhs, rhs_id_offset
        )

        if match_idx != -1:
            rhs_out_dim, rhs_primal_dim = remaining_rhs.pop(match_idx)
            meta = _resolve_matched_dims(
                lhs_out_dim,
                lhs_primal_dim,
                rhs_out_dim,
                rhs_primal_dim,
            )
        else:
            meta = _resolve_unmatched_lhs_dims(lhs_out_dim, lhs_primal_dim)

        if meta:
            pairs_meta.append(meta)

    for rhs_out_dim, rhs_primal_dim in remaining_rhs:
        meta = _resolve_unmatched_rhs_dims(rhs_out_dim, rhs_primal_dim)
        if meta:
            pairs_meta.append(meta)

    return pairs_meta


def _find_matching_rhs_topo(
    lhs_out_dim: Dimension | None,
    lhs_primal_dim: Dimension | None,
    rhs_topos: list[tuple[Dimension | None, Dimension | None]],
    rhs_id_offset: int,
) -> int:
    for i, (rhs_out_dim, rhs_primal_dim) in enumerate(rhs_topos):
        if (
            lhs_out_dim
            and rhs_out_dim
            and lhs_out_dim.id == rhs_out_dim.id - rhs_id_offset
        ):
            return i
        if (
            lhs_primal_dim
            and rhs_primal_dim
            and lhs_primal_dim.id == rhs_primal_dim.id - rhs_id_offset
        ):
            return i
    return -1


def _build_matmul_topology(
    lhs: SparseTensor,
    rhs_out_dims: tuple[Dimension, ...],
    rhs_primal_dims: tuple[Dimension, ...],
    rhs_id_offset: int,
) -> list[DimensionPair]:
    lhs_out_map = {d.id: d for d in lhs.out_dims}
    rhs_primal_map = {d.id: d for d in rhs_primal_dims}
    rhs_dims = rhs_out_dims + rhs_primal_dims

    num_contract = min(len(lhs.primal_dims), len(rhs_out_dims))
    lhs_contract = list(lhs.primal_dims[-num_contract:] if num_contract > 0 else [])
    rhs_contract = list(rhs_out_dims[-num_contract:] if num_contract > 0 else [])

    pairs_meta, processed_lhs_dims, processed_rhs_dims = [], set(), set()

    for lp, ro in zip(lhs_contract, rhs_contract):
        meta, ids = _resolve_contract_pair(lp, ro, lhs_out_map, rhs_primal_map)
        pairs_meta.append(meta)
        processed_lhs_dims.update(ids[: len(ids) // 2 + 1])
        processed_lhs_dims.add(lp.id)
        processed_rhs_dims.add(ro.id)
        if meta.lhs.dim:
            processed_lhs_dims.add(meta.lhs.dim.id)
        if meta.rhs.shared_dim:
            processed_rhs_dims.add(meta.rhs.shared_dim.id)

    lhs_topos = _get_unprocessed_topos(
        lhs.dims, {d.id: d for d in lhs.dims}, processed_lhs_dims, lhs.out_dims
    )
    rhs_topos = _get_unprocessed_topos(
        rhs_dims, {d.id: d for d in rhs_dims}, processed_rhs_dims, rhs_out_dims
    )

    pairs_meta.extend(_resolve_broadcast_topos(lhs_topos, rhs_topos, rhs_id_offset))
    return pairs_meta


def _extract_broadcast_info(
    metadata_value_axes: Sequence[tuple[int | None, int | None, int | None]],
    metadata_logical_lengths: Sequence[tuple[int, int, int]],
) -> tuple[list[int | None], list[int]]:
    broadcast_dims, canonical_shape = [], []
    for val_axes, logical_lengths in zip(metadata_value_axes, metadata_logical_lengths):
        for v_ax, l_len in zip(val_axes, logical_lengths):
            broadcast_dims.append(v_ax)
            canonical_shape.append(l_len)
    return broadcast_dims, canonical_shape


def _prepare_physical_array(
    val: Array,
    metadata_value_axes: Sequence[tuple[int | None, int | None, int | None]],
    metadata_logical_lengths: Sequence[tuple[int, int, int]],
) -> Array:
    """Prepare raw values without eagerly broadcasting to the canonical shape."""
    broadcast_dims, canonical_shape = _extract_broadcast_info(
        metadata_value_axes, metadata_logical_lengths
    )

    valid_indices = tuple(i for i, v in enumerate(broadcast_dims) if v is not None and v < val.ndim)
    source_dims = tuple(broadcast_dims[i] for i in valid_indices)

    leftover_axes = [v for v in range(val.ndim) if v not in source_dims]

    full_source_dims = source_dims + tuple(leftover_axes)
    full_target_map = valid_indices + tuple(
        len(broadcast_dims) + i for i in range(len(leftover_axes))
    )

    N = len(broadcast_dims) + len(leftover_axes)
    
    # 0-cost passthrough if array is already inherently aligned 
    if full_source_dims == full_target_map and N == val.ndim:
        return val

    perm = [-1] * N
    for src, tgt in zip(full_source_dims, full_target_map):
        perm[tgt] = src
        
    ones_idx = val.ndim
    for i in range(N):
        if perm[i] == -1:
            perm[i] = ones_idx
            ones_idx += 1
            
    if ones_idx > val.ndim:
        val = val.reshape(val.shape + (1,) * (ones_idx - val.ndim))
        
    if perm != list(range(N)):
        val = val.transpose(perm)
    
    return val


def _prepare_physical_arrays(
    lhs_val: Array, rhs_val: Array, pairs_meta: list[DimensionPair]
) -> tuple[Array, Array]:
    def get_axes_and_lens(
        is_lhs: bool,
    ) -> tuple[
        list[tuple[int | None, int | None, int | None]], list[tuple[int, int, int]]
    ]:
        if is_lhs:
            axes = [
                (p.lhs.outer_val_dim, p.lhs.block_val_dim, p.lhs.shared_block_val_dim)
                for p in pairs_meta
            ]
            lens = [
                (p.lhs.outer_len, p.lhs.block_len, p.lhs.shared_block_len)
                for p in pairs_meta
            ]
        else:
            axes = [
                (p.rhs.outer_val_dim, p.rhs.block_val_dim, p.rhs.shared_block_val_dim)
                for p in pairs_meta
            ]
            lens = [
                (p.rhs.outer_len, p.rhs.block_len, p.rhs.shared_block_len)
                for p in pairs_meta
            ]
        return axes, lens

    l_axes, l_lens = get_axes_and_lens(True)
    r_axes, r_lens = get_axes_and_lens(False)
    return _prepare_physical_array(lhs_val, l_axes, l_lens), _prepare_physical_array(
        rhs_val, r_axes, r_lens
    )


def _process_contraction_factor_pair(
    p: DimensionPair,
    shared_tiling_factors: list[int],
    total_tiled_lengths: list[int],
    block_split_factors: list[int],
) -> float:
    if (
        p.pairing_type == "contract"
        and p.lhs.outer_len == 1
        and p.rhs.outer_len == 1
        and p.lhs.shared_block_len == 1
        and p.rhs.block_len == 1
    ):
        mult = float(max(p.logical_element_count, 1))
    else:
        mult = 1.0

    gcd_len, lcm_len = (
        math.gcd(p.lhs.outer_len, p.rhs.outer_len),
        math.lcm(p.lhs.outer_len, p.rhs.outer_len),
    )
    shared_tiling_factors.append(gcd_len)
    total_tiled_lengths.append(lcm_len)

    split = (
        p.lhs.shared_block_len // (lcm_len // p.lhs.outer_len)
        if p.lhs.shared_block_len > 1
        else (
            p.rhs.block_len // (lcm_len // p.rhs.outer_len)
            if p.rhs.block_len > 1
            else 1
        )
    )
    block_split_factors.append(split)
    return mult


def _calculate_contraction_factors(
    pairs: list[DimensionPair],
) -> tuple[list[int], list[int], list[int], float]:
    shared_tiling_factors, total_tiled_lengths, block_split_factors = [], [], []
    scalar_multiplier = 1.0
    for p in pairs:
        scalar_multiplier *= _process_contraction_factor_pair(
            p, shared_tiling_factors, total_tiled_lengths, block_split_factors
        )

    return (
        shared_tiling_factors,
        total_tiled_lengths,
        block_split_factors,
        scalar_multiplier,
    )


def _process_bc_shape_pair(
    i: int,
    p: DimensionPair,
    total_tiled_lengths: list[int],
    block_split_factors: list[int],
    lhs_bc_shape: list[int],
    rhs_bc_shape: list[int],
) -> None:
    total_len = total_tiled_lengths[i]
    split = block_split_factors[i]
    lhs_bc_shape[AXES_PER_PAIR * i], rhs_bc_shape[AXES_PER_PAIR * i] = (
        p.lhs.outer_len,
        p.rhs.outer_len,
    )
    lhs_bc_shape[AXES_PER_PAIR * i + 2] = (total_len // p.lhs.outer_len) * split
    rhs_bc_shape[AXES_PER_PAIR * i + 1] = (total_len // p.rhs.outer_len) * split


def _calculate_contraction_splits(
    num_pairs: int,
    pairs_meta: list[DimensionPair],
    total_tiled_lengths: list[int],
    block_split_factors: list[int],
    lhs_bc_shape: list[int],
    rhs_bc_shape: list[int],
) -> tuple[list[int], list[int]]:
    lhs_split, rhs_split = [], []
    for i in range(num_pairs):
        lhs_split.extend(
            [
                lhs_bc_shape[AXES_PER_PAIR * i],
                pairs_meta[i].lhs.block_len,
                total_tiled_lengths[i] // lhs_bc_shape[AXES_PER_PAIR * i],
                block_split_factors[i],
            ]
        )
        rhs_split.extend(
            [
                rhs_bc_shape[AXES_PER_PAIR * i],
                total_tiled_lengths[i] // rhs_bc_shape[AXES_PER_PAIR * i],
                block_split_factors[i],
                pairs_meta[i].rhs.shared_block_len,
            ]
        )

    return lhs_split, rhs_split


def _calculate_contraction_perms(num_pairs: int) -> tuple[list[int], list[int]]:
    perm_lhs = (
        generate_block_permutation(num_pairs, SPLIT_AXES, [0, 2])
        + generate_grouped_permutation(num_pairs, SPLIT_AXES, [1])
        + generate_grouped_permutation(num_pairs, SPLIT_AXES, [3])
    )
    perm_rhs = (
        generate_block_permutation(num_pairs, SPLIT_AXES, [0, 1])
        + generate_grouped_permutation(num_pairs, SPLIT_AXES, [2])
        + generate_grouped_permutation(num_pairs, SPLIT_AXES, [3])
    )

    return perm_lhs, perm_rhs


def _prepare_contraction_views(
    lhs_val: Array,
    rhs_val: Array,
    pairs_meta: list[DimensionPair],
    shared_tiling_factors: list[int],
    total_tiled_lengths: list[int],
    block_split_factors: list[int],
) -> tuple[Array, Array, list[int], list[int]]:
    num_pairs = len(pairs_meta)

    lhs_phy_split = []
    rhs_phy_split = []

    for i, p in enumerate(pairs_meta):
        phy_0 = lhs_val.shape[AXES_PER_PAIR * i]
        phy_1 = lhs_val.shape[AXES_PER_PAIR * i + 1]
        phy_2 = lhs_val.shape[AXES_PER_PAIR * i + 2]

        lhs_phy_split.append(phy_0)
        lhs_phy_split.append(phy_1)
        if phy_2 == 1:
            lhs_phy_split.extend([1, 1])
        else:
            lhs_phy_split.extend([total_tiled_lengths[i] // p.lhs.outer_len, block_split_factors[i]])

        phy_0_r = rhs_val.shape[AXES_PER_PAIR * i]
        phy_1_r = rhs_val.shape[AXES_PER_PAIR * i + 1]
        phy_2_r = rhs_val.shape[AXES_PER_PAIR * i + 2]

        rhs_phy_split.append(phy_0_r)
        if phy_1_r == 1:
            rhs_phy_split.extend([1, 1])
        else:
            rhs_phy_split.extend([total_tiled_lengths[i] // p.rhs.outer_len, block_split_factors[i]])
        rhs_phy_split.append(phy_2_r)

    lhs_leftovers = list(lhs_val.shape[AXES_PER_PAIR * num_pairs :])
    rhs_leftovers = list(rhs_val.shape[AXES_PER_PAIR * num_pairs :])
    lhs_phy_split.extend(lhs_leftovers)
    rhs_phy_split.extend(rhs_leftovers)

    perm_lhs, perm_rhs = _calculate_contraction_perms(num_pairs)
    for ax in range(SPLIT_AXES * num_pairs, len(lhs_phy_split)):
        perm_lhs.append(ax)
    for ax in range(SPLIT_AXES * num_pairs, len(rhs_phy_split)):
        perm_rhs.append(ax)

    # Conditionally execute memory-heavy topology mappings
    if tuple(lhs_phy_split) != lhs_val.shape:
        lhs_view = lhs_val.reshape(lhs_phy_split)
    else:
        lhs_view = lhs_val
        
    if perm_lhs != list(range(len(lhs_phy_split))):
        lhs_view = lhs_view.transpose(perm_lhs)

    if tuple(rhs_phy_split) != rhs_val.shape:
        rhs_view = rhs_val.reshape(rhs_phy_split)
    else:
        rhs_view = rhs_val
        
    if perm_rhs != list(range(len(rhs_phy_split))):
        rhs_view = rhs_view.transpose(perm_rhs)

    # Target unmerged shape (Broadcast BEFORE merge) - Fixed loop variable leakage
    lhs_unmerged_target = []
    for i in range(num_pairs):
        p = pairs_meta[i]
        lhs_unmerged_target.extend([p.lhs.outer_len, total_tiled_lengths[i] // p.lhs.outer_len])
    for i in range(num_pairs):
        p = pairs_meta[i]
        lhs_unmerged_target.append(p.lhs.block_len)
    for i in range(num_pairs):
        lhs_unmerged_target.append(block_split_factors[i])
    lhs_unmerged_target.extend(lhs_leftovers)

    rhs_unmerged_target = []
    for i in range(num_pairs):
        p = pairs_meta[i]
        rhs_unmerged_target.extend([p.rhs.outer_len, total_tiled_lengths[i] // p.rhs.outer_len])
    for i in range(num_pairs):
        rhs_unmerged_target.append(block_split_factors[i])
    for i in range(num_pairs):
        p = pairs_meta[i]
        rhs_unmerged_target.append(p.rhs.shared_block_len)
    rhs_unmerged_target.extend(rhs_leftovers)

    # Free broadcast without materializing a dense grid yet
    if lhs_view.shape != tuple(lhs_unmerged_target):
        lhs_view = jnp.broadcast_to(lhs_view, tuple(lhs_unmerged_target))
    if rhs_view.shape != tuple(rhs_unmerged_target):
        rhs_view = jnp.broadcast_to(rhs_view, tuple(rhs_unmerged_target))

    # Now safely merge the tiles (Materializes only when tiling is required)
    lhs_merged_shape = list(total_tiled_lengths)
    for i in range(num_pairs):
        p = pairs_meta[i]
        lhs_merged_shape.append(p.lhs.block_len)
    for i in range(num_pairs):
        lhs_merged_shape.append(block_split_factors[i])
    lhs_merged_shape.extend(lhs_leftovers)

    rhs_merged_shape = list(total_tiled_lengths)
    for i in range(num_pairs):
        rhs_merged_shape.append(block_split_factors[i])
    for i in range(num_pairs):
        p = pairs_meta[i]
        rhs_merged_shape.append(p.rhs.shared_block_len)
    rhs_merged_shape.extend(rhs_leftovers)

    if lhs_view.shape != tuple(lhs_merged_shape):
        lhs_view = lhs_view.reshape(lhs_merged_shape)
    if rhs_view.shape != tuple(rhs_merged_shape):
        rhs_view = rhs_view.reshape(rhs_merged_shape)

    lhs_bc_shape = []
    rhs_bc_shape = []
    for i, p in enumerate(pairs_meta):
        lhs_bc_shape.extend([
            p.lhs.outer_len,
            p.lhs.block_len,
            (total_tiled_lengths[i] // p.lhs.outer_len) * block_split_factors[i]
        ])
        rhs_bc_shape.extend([
            p.rhs.outer_len,
            (total_tiled_lengths[i] // p.rhs.outer_len) * block_split_factors[i],
            p.rhs.shared_block_len
        ])

    return lhs_view, rhs_view, lhs_bc_shape, rhs_bc_shape


def _calculate_tiled_index(
    i: int, p: DimensionPair, gcd_len: int, lcm_len: int
) -> tuple[np.ndarray, int]:
    a, b = p.lhs.outer_len, p.rhs.outer_len
    if gcd_len == lcm_len:
        return np.arange(lcm_len), lcm_len

    r = np.arange(lcm_len)
    idx = (
        (r // (lcm_len // gcd_len)) * ((a // gcd_len) * (b // gcd_len))
        + ((r // (lcm_len // a)) % (a // gcd_len)) * (b // gcd_len)
        + ((r // (lcm_len // b)) % (b // gcd_len))
    )
    return idx, gcd_len * (a // gcd_len) * (b // gcd_len)


def _reduce_contraction_grid(
    res_view: Array,
    pairs_meta: list[DimensionPair],
    shared_tiling_factors: list[int],
    total_tiled_lengths: list[int],
    final_lhs_block_lens: list[int],
    final_rhs_block_lens: list[int],
    count: bool = False,
) -> Array:
    num_dimension_pairs = len(pairs_meta)
    per_idx, per_num = [], []
    for i, p in enumerate(pairs_meta):
        idx, num = _calculate_tiled_index(
            i, p, shared_tiling_factors[i], total_tiled_lengths[i]
        )
        per_idx.append(idx)
        per_num.append(num)

    flat_idx = np.zeros(tuple(total_tiled_lengths), dtype=np.int32)
    for i in range(num_dimension_pairs):
        shape = [1] * num_dimension_pairs
        shape[i] = total_tiled_lengths[i]
        flat_idx += per_idx[i].reshape(shape) * (
            math.prod(per_num[i + 1 :]) if i + 1 < num_dimension_pairs else 1
        )

    extra = final_lhs_block_lens + final_rhs_block_lens
    flat_idx_array = flat_idx.flatten()
    
    # OPTIMIZATION: Fast path for identity mapping (no collisions, already in order)
    if np.array_equal(flat_idx_array, np.arange(len(flat_idx_array))):
        return res_view.reshape(*per_num, *extra), (0, 0, 0)
        
    # OPTIMIZATION: Fast path for pure permutation (no collisions, needs fast reorder)
    if len(np.unique(flat_idx_array)) == len(flat_idx_array):
        inverse_idx = np.argsort(flat_idx_array)
        res_reduced = res_view.reshape(math.prod(total_tiled_lengths), math.prod(extra))[inverse_idx]
        return res_reduced.reshape(*per_num, *extra), (0, 0, 0)

    # Fallback to segment_sum for actual reductions (e.g. misaligned block collisions)
    res_reduced = jax.ops.segment_sum(
        res_view.reshape(math.prod(total_tiled_lengths), math.prod(extra)),
        jnp.array(flat_idx_array),
        num_segments=math.prod(per_num),
    )
    res = res_reduced.reshape(*per_num, *extra)
    adds = 0
    if count is not False:
        adds = res_view.size - math.prod(per_num)
    return res, (adds, 0, 0)


def _collect_dot_general_axes(
    num_dimension_pairs: int, pairs_meta: list[DimensionPair]
) -> tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]:
    contract_lhs, contract_rhs = [], []
    batch_lhs, batch_rhs = (
        list(range(num_dimension_pairs)),
        list(range(num_dimension_pairs)),
    )
    for i, p in enumerate(pairs_meta):
        if p.pairing_type == "contract":
            contract_lhs.append(2 * num_dimension_pairs + i)
            contract_rhs.append(num_dimension_pairs + i)
        else:
            batch_lhs.append(2 * num_dimension_pairs + i)
            batch_rhs.append(num_dimension_pairs + i)
    return ((contract_lhs, contract_rhs), (batch_lhs, batch_rhs))


def _build_interleave_reorder(
    num_dimension_pairs: int, pairs_meta: list[DimensionPair]
) -> list[int]:
    interleave_reorder = list(range(num_dimension_pairs))
    for i in range(num_dimension_pairs):
        interleave_reorder.append(num_dimension_pairs + i)
        if pairs_meta[i].pairing_type == "spatial_sparse_rhs":
            interleave_reorder.append(2 * num_dimension_pairs + i)

    for i in range(num_dimension_pairs):
        if pairs_meta[i].pairing_type != "spatial_sparse_rhs":
            interleave_reorder.append(2 * num_dimension_pairs + i)
        interleave_reorder.append(3 * num_dimension_pairs + i)
    return interleave_reorder


def _build_final_grid(
    num_dimension_pairs: int,
    shared_tiling_factors: list[int],
    lhs_bc_shape: list[int],
    rhs_bc_shape: list[int],
    final_lhs_block_lens: list[int],
    final_rhs_block_lens: list[int],
) -> list[int]:
    grid = []
    for i in range(num_dimension_pairs):
        grid.extend(
            [
                shared_tiling_factors[i],
                lhs_bc_shape[AXES_PER_PAIR * i] // shared_tiling_factors[i],
                rhs_bc_shape[AXES_PER_PAIR * i] // shared_tiling_factors[i],
            ]
        )
    grid.extend(final_lhs_block_lens + final_rhs_block_lens)
    return grid


def _finalize_contraction_output(
    num_dimension_pairs: int,
    res_raw: Array,
    total_tiled_lengths: list[int],
    pairs_meta: list[DimensionPair],
    block_split_factors: list[int],
    shared_tiling_factors: list[int],
    lhs_bc_shape: list[int],
    rhs_bc_shape: list[int],
    lhs_leftovers: list[int],
    rhs_leftovers: list[int],
    count: bool = False,
) -> tuple[Array, list[int], list[int], tuple[int, int, int]]:
    N = num_dimension_pairs
    non_contract_indices = [
        i for i, p in enumerate(pairs_meta) if p.pairing_type != "contract"
    ]

    d_ls = [p.lhs.block_len for p in pairs_meta]
    f_ls = [p.rhs.shared_block_len for p in pairs_meta]
    ss_out = [
        block_split_factors[i] if i in non_contract_indices else 1
        for i in range(N)
    ]

    expanded_shape = []
    expanded_shape.extend(total_tiled_lengths)
    expanded_shape.extend(ss_out)
    expanded_shape.extend(d_ls)
    expanded_shape.extend(f_ls)
    expanded_shape.extend(lhs_leftovers)
    expanded_shape.extend(rhs_leftovers)

    if res_raw.shape != tuple(expanded_shape):
        res_expanded = res_raw.reshape(expanded_shape)
    else:
        res_expanded = res_raw

    fast_perm = []
    for i in range(N):
        fast_perm.append(i)
    for i in range(N):
        fast_perm.append(2 * N + i)
        if pairs_meta[i].pairing_type == "spatial_sparse_rhs":
            fast_perm.append(N + i)
    for i in range(N):
        if pairs_meta[i].pairing_type != "spatial_sparse_rhs":
            fast_perm.append(N + i)
        fast_perm.append(3 * N + i)

    for ax in range(4 * N, len(expanded_shape)):
        fast_perm.append(ax)

    final_lhs_block_lens = [
        d_ls[i] * (ss_out[i] if pairs_meta[i].pairing_type == "spatial_sparse_rhs" else 1)
        for i in range(N)
    ]
    final_rhs_block_lens = [
        f_ls[i] * (ss_out[i] if pairs_meta[i].pairing_type != "spatial_sparse_rhs" else 1)
        for i in range(N)
    ]

    target_shape = (*total_tiled_lengths, *final_lhs_block_lens, *final_rhs_block_lens, *lhs_leftovers, *rhs_leftovers)
    
    if fast_perm != list(range(len(fast_perm))):
        res_view = res_expanded.transpose(fast_perm)
        if res_view.shape != target_shape:
            res_view = res_view.reshape(target_shape)
    else:
        if res_expanded.shape != target_shape:
            res_view = res_expanded.reshape(target_shape)
        else:
            res_view = res_expanded

    if any(shared_tiling_factors[i] != total_tiled_lengths[i] for i in range(N)):
        res_reduced, counts = _reduce_contraction_grid(
            res_view,
            pairs_meta,
            shared_tiling_factors,
            total_tiled_lengths,
            final_lhs_block_lens,
            final_rhs_block_lens + lhs_leftovers + rhs_leftovers,
            count
        )
    else:
        counts = (0, 0, 0)
        res_reduced = res_view

    grid = _build_final_grid(
        N,
        shared_tiling_factors,
        lhs_bc_shape,
        rhs_bc_shape,
        final_lhs_block_lens,
        final_rhs_block_lens,
    )
    grid.extend(lhs_leftovers)
    grid.extend(rhs_leftovers)

    perm_out = (
        generate_grouped_permutation(N, GRID_AXES_PER_PAIR, [0])
        + [ax for i in range(N) for ax in (GRID_AXES_PER_PAIR * i + 1, GRID_AXES_PER_PAIR * N + i)]
        + [ax for i in range(N) for ax in (GRID_AXES_PER_PAIR * i + 2, (GRID_AXES_PER_PAIR + 1) * N + i)]
    )
    
    # FIX: Base grid without leftovers has size 5*N (shared, lhs_outer, rhs_outer, lhs_block, rhs_block).
    for ax in range(5 * N, len(grid)):
        perm_out.append(ax)
        
    if res_reduced.shape != tuple(grid):
        res_reduced = res_reduced.reshape(grid)
        
    if perm_out != list(range(len(grid))):
        res_reduced = res_reduced.transpose(perm_out)

    return (
        res_reduced,
        final_lhs_block_lens,
        final_rhs_block_lens,
        counts
    )


def _execute_block_sparse_contraction(
    lhs_val: Array, rhs_val: Array, pairs_meta: list[DimensionPair], count: bool = False
) -> tuple[Array, list[int], list[int], list[int], float]:
    num_dimension_pairs = len(pairs_meta)
    N = num_dimension_pairs
    (
        shared_tiling_factors,
        total_tiled_lengths,
        block_split_factors,
        scalar_multiplier,
    ) = _calculate_contraction_factors(pairs_meta)

    lhs_view, rhs_view, lhs_bc_shape, rhs_bc_shape = _prepare_contraction_views(
        lhs_val,
        rhs_val,
        pairs_meta,
        shared_tiling_factors,
        total_tiled_lengths,
        block_split_factors,
    )

    lhs_leftovers = list(lhs_view.shape[3 * N :])
    rhs_leftovers = list(rhs_view.shape[3 * N :])

    dot_axes = _collect_dot_general_axes(N, pairs_meta)
    res_raw = jax.lax.dot_general(lhs_view, rhs_view, dot_axes)

    nc_len = sum(1 for p in pairs_meta if p.pairing_type != "contract")
    
    # Reorder dot_general output to pull shared_R ahead of lhs_leftovers
    dg_perm = (
        list(range(2 * N + nc_len)) +
        list(range(2 * N + nc_len + len(lhs_leftovers), 3 * N + nc_len + len(lhs_leftovers))) +
        list(range(2 * N + nc_len, 2 * N + nc_len + len(lhs_leftovers))) +
        list(range(3 * N + nc_len + len(lhs_leftovers), res_raw.ndim))
    )
    if dg_perm != list(range(len(dg_perm))):
        res_raw = jnp.transpose(res_raw, dg_perm)

    lhs_shape = lhs_view.shape
    rhs_shape = rhs_view.shape
    L, R, B, C = 1, 1, 1, 1
    r_dims = dot_axes[0][1] + dot_axes[1][1]
    for i in range(lhs_view.ndim):
        if i in dot_axes[0][0]:
            B *= lhs_shape[i]
        elif i in dot_axes[1][0]:
            C *= lhs_shape[i]
        else:
            L *= lhs_shape[i]
    for i in range(rhs_view.ndim):
        if i not in r_dims:
            R *= rhs_shape[i]
            
    muls = L * R * B
    fmas = L * R * B * (C - 1)
    adds = fmas

    contracted_grid, final_lhs_block_lens, final_rhs_block_lens, (adds_r, _, _) = (
        _finalize_contraction_output(
            num_dimension_pairs,
            res_raw,
            total_tiled_lengths,
            pairs_meta,
            block_split_factors,
            shared_tiling_factors,
            lhs_bc_shape,
            rhs_bc_shape,
            lhs_leftovers,
            rhs_leftovers,
            count
        )
    )

    return (
        contracted_grid,
        shared_tiling_factors,
        final_lhs_block_lens,
        final_rhs_block_lens,
        scalar_multiplier,
        adds + adds_r,
        muls,
        fmas,
    )


def _handle_output_dim(
    dims_list: list[Dimension],
    dim_id: int,
    size: int,
    value_axis: int | None,
    is_value_present: bool,
    squeeze_axes: list[int],
) -> None:
    from ..dimensions import DenseDimension

    if is_value_present:
        dims_list.append(DenseDimension(dim_id, size, val_dim=value_axis))
    else:
        if value_axis is not None and value_axis not in squeeze_axes:
            squeeze_axes.append(value_axis)
        dims_list.append(DenseDimension(dim_id, size, val_dim=None))


def _handle_output_sparse(
    dims_list: list[Dimension],
    dim_id: int,
    shared_factor: int,
    shared_val_ax: int | None,
    is_shared_val_pres: bool,
    other_id: int,
    block_sz: int | None,
    block_val_ax: int | None,
    is_block_val_pres: bool,
    squeeze_axes: list[int],
) -> None:
    from ..dimensions import SparseDimension

    if shared_factor == 1:
        if shared_val_ax is not None and shared_val_ax not in squeeze_axes:
            squeeze_axes.append(shared_val_ax)
        _handle_output_dim(
            dims_list,
            dim_id,
            block_sz or 1,
            block_val_ax,
            is_block_val_pres,
            squeeze_axes,
        )
    else:
        if not is_shared_val_pres:
            if shared_val_ax is not None and shared_val_ax not in squeeze_axes:
                squeeze_axes.append(shared_val_ax)
            shared_val_ax = None

        if block_sz == 1:
            if block_val_ax is not None and block_val_ax not in squeeze_axes:
                squeeze_axes.append(block_val_ax)
            block_sz = None
            block_val_ax = None
            is_block_val_pres = False

        if is_block_val_pres:
            dims_list.append(
                SparseDimension(
                    dim_id,
                    shared_factor,
                    val_dim=shared_val_ax,
                    other_id=other_id,
                    block_size=block_sz,
                    block_val_dim=block_val_ax,
                )
            )
        else:
            if block_val_ax is not None and block_val_ax not in squeeze_axes:
                squeeze_axes.append(block_val_ax)
            dims_list.append(
                SparseDimension(
                    dim_id,
                    shared_factor,
                    val_dim=shared_val_ax,
                    other_id=other_id,
                    block_size=(
                        block_sz if block_sz is not None and block_sz > 1 else None
                    ),
                    block_val_dim=None,
                )
            )


@dataclass(frozen=True)
class OutputAxisMap:
    shared: dict[int, int]
    lhs: dict[int, int]
    rhs: dict[int, int]


def _resolve_output_shape_and_axes(
    ctx: MatmulContext, res: ContractionResult
) -> tuple[list[int], OutputAxisMap, list[int]]:
    reshaped_output_shape: list[int] = []
    shared_val_axis_map: dict[int, int] = {}
    lhs_val_axis_map: dict[int, int] = {}
    rhs_val_axis_map: dict[int, int] = {}
    squeeze_axes: list[int] = []
    current_physical_axis = 0

    # 1. Shared axes
    for i, factor in enumerate(res.shared_tiling_factors):
        reshaped_output_shape.append(factor)
        shared_val_axis_map[i] = current_physical_axis
        current_physical_axis += 1

    # 2. LHS axes
    for i, p in enumerate(ctx.pairs):
        factor = p.lhs.outer_len // res.shared_tiling_factors[i]
        if p.pairing_type == "spatial_sparse_lhs":
            reshaped_output_shape.extend([factor, res.lhs_block_lens[i]])
            squeeze_axes.append(shared_val_axis_map[i])
            shared_val_axis_map[i] = current_physical_axis
            current_physical_axis += 1
            lhs_val_axis_map[i] = current_physical_axis
            current_physical_axis += 1
        else:
            reshaped_output_shape.append(factor * res.lhs_block_lens[i])
            lhs_val_axis_map[i] = current_physical_axis
            current_physical_axis += 1

    # 3. RHS axes
    for i, p in enumerate(ctx.pairs):
        factor = p.rhs.outer_len // res.shared_tiling_factors[i]
        if p.pairing_type == "spatial_sparse_rhs":
            reshaped_output_shape.extend([factor, res.rhs_block_lens[i]])
            squeeze_axes.append(shared_val_axis_map[i])
            shared_val_axis_map[i] = current_physical_axis
            current_physical_axis += 1
            rhs_val_axis_map[i] = current_physical_axis
            current_physical_axis += 1
        else:
            reshaped_output_shape.append(factor * res.rhs_block_lens[i])
            rhs_val_axis_map[i] = current_physical_axis
            current_physical_axis += 1

    # 4. Append leftover unmapped dense axes at the end
    # The grid is built out of 5*N layout dimensions (shared, lhs_bc, rhs_bc, lhs_block, rhs_block), 
    # anything beyond that are the dense leftovers perfectly preserved.
    leftover_axes = list(res.grid.shape[5 * ctx.num_pairs :])
    reshaped_output_shape.extend(leftover_axes)

    axis_map = OutputAxisMap(
        shared=shared_val_axis_map, lhs=lhs_val_axis_map, rhs=rhs_val_axis_map
    )
    return reshaped_output_shape, axis_map, squeeze_axes


@dataclass
class OutputState:
    out_dims: list[Dimension]
    primal_dims: list[Dimension]
    squeeze_axes: list[int]
    next_id: int

    def get_new_id(self) -> int:
        res = self.next_id
        self.next_id += 1
        return res


@dataclass(frozen=True)
class PresenceInfo:
    shared: bool
    lhs: bool
    rhs: bool


def _build_output_tensor(
    ctx: MatmulContext,
    rhs_dims: tuple[Dimension, ...],
    res: ContractionResult,
    count: bool = False,
) -> SparseTensor:
    from ..tensor import SparseTensor

    reshaped_output_shape, axis_map, squeeze_axes = _resolve_output_shape_and_axes(
        ctx, res
    )

    global_max_id = max([d.id for d in ctx.lhs.dims] + [d.id for d in rhs_dims] + [-1])

    state = OutputState(
        out_dims=[],
        primal_dims=[],
        squeeze_axes=squeeze_axes,
        next_id=global_max_id + 1,
    )

    for i, pm in enumerate(ctx.pairs):
        is_shared_pres = (
            pm.lhs.outer_val_dim is not None or pm.rhs.outer_val_dim is not None
        )
        is_lhs_pres = (
            pm.lhs.outer_val_dim is not None or pm.lhs.block_val_dim is not None
        )
        is_rhs_pres = (
            pm.rhs.outer_val_dim is not None or pm.rhs.shared_block_val_dim is not None
        )

        presence = PresenceInfo(shared=is_shared_pres, lhs=is_lhs_pres, rhs=is_rhs_pres)

        _resolve_output_pair(i, pm, res, axis_map, state, presence)

    out_dims, primal_dims = state.out_dims, state.primal_dims
    squeeze_axes = state.squeeze_axes

    if res.grid.shape != tuple(reshaped_output_shape):
        grid_view = res.grid.reshape(reshaped_output_shape)
    else:
        grid_view = res.grid

    if squeeze_axes:
        unique_squeeze_axes = tuple(sorted(set(squeeze_axes)))
        final_shape = [
            s for i, s in enumerate(reshaped_output_shape)
            if i not in unique_squeeze_axes
        ]
        
        # 0-cost reshape if squeezed axes are all size 1, fallback to slice if size > 1
        if grid_view.size == math.prod(final_shape):
            contracted_values = grid_view.reshape(final_shape)
        else:
            idx = tuple(
                0 if i in unique_squeeze_axes else slice(None)
                for i in range(len(reshaped_output_shape))
            )
            contracted_values = grid_view[idx]
            if contracted_values.shape != tuple(final_shape):
                contracted_values = contracted_values.reshape(final_shape)

        def shift_ax(axis_idx: int | None) -> int | None:
            return (
                axis_idx - sum(1 for ax in unique_squeeze_axes if ax < axis_idx)
                if axis_idx is not None
                else None
            )

        out_dims = [
            replace(
                d,
                val_dim=shift_ax(d.val_dim),
                **(
                    {"block_val_dim": shift_ax(d.block_val_dim)}
                    if isinstance(d, SparseDimension)
                    else {}
                ),
            )
            for d in out_dims
        ]
        primal_dims = [
            replace(
                d,
                val_dim=shift_ax(d.val_dim),
                **(
                    {"block_val_dim": shift_ax(d.block_val_dim)}
                    if isinstance(d, SparseDimension)
                    else {}
                ),
            )
            for d in primal_dims
        ]
    else:
        contracted_values = grid_view

    final_out_dims = tuple(sorted(out_dims, key=lambda d: d.id))
    final_primal_dims = tuple(sorted(primal_dims, key=lambda d: d.id))
    id_map = {d.id: i for i, d in enumerate(final_out_dims + final_primal_dims)}

    def update_ids(d: Dimension) -> Dimension:
        return replace(
            d,
            id=id_map[d.id],
            **(
                {
                    "other_id": id_map.get(
                        getattr(d, "other_id", -1), getattr(d, "other_id", -1)
                    )
                }
                if isinstance(d, SparseDimension)
                else {}
            ),
        )

    final_out_dims = tuple(update_ids(d) for d in final_out_dims)
    final_primal_dims = tuple(update_ids(d) for d in final_primal_dims)

    has_val = any(
        d.val_dim is not None for d in final_out_dims + final_primal_dims
    ) or any(
        isinstance(d, SparseDimension) and d.block_val_dim is not None
        for d in final_out_dims + final_primal_dims
    )

    final_mult = ctx.lhs.scalar_mult * ctx.rhs.scalar_mult * res.scalar_multiplier
    if not has_val and contracted_values is not None and contracted_values.size == 1:
        # Use jnp.squeeze to avoid item() call which fails on tracers
        final_mult *= jnp.squeeze(contracted_values)
        contracted_values = None

    res = SparseTensor(
        final_out_dims,
        final_primal_dims,
        contracted_values,
        scalar_mult=jnp.array(final_mult, dtype=ctx.lhs.dtype),
        sort_val=True
    )

    final_counts = (0, 2, 0) if count is not False else (0, 0, 0)
    return res, final_counts


def _resolve_output_pair_ids(pm: DimensionPair, state: OutputState) -> tuple[int, int]:
    lhs_id = pm.lhs.dim.id if pm.lhs.dim else state.get_new_id()
    rhs_id = pm.rhs.shared_dim.id if pm.rhs.shared_dim else state.get_new_id()
    return lhs_id, rhs_id


@dataclass(frozen=True)
class OutputPairInfo:
    shared_factor: int
    shared_val_axis: int
    lhs_val_axis: int
    rhs_val_axis: int
    final_lhs_block_size: int
    final_rhs_block_size: int


def _handle_contract_case(
    pm: DimensionPair,
    info: OutputPairInfo,
    presence: PresenceInfo,
    state: OutputState,
) -> None:
    lid, rid = _resolve_output_pair_ids(pm, state)
    if not pm.lhs.dim and not pm.rhs.shared_dim:
        state.squeeze_axes.extend(
            [info.shared_val_axis, info.lhs_val_axis, info.rhs_val_axis]
        )
    elif not pm.lhs.dim:
        state.squeeze_axes.extend([info.shared_val_axis, info.lhs_val_axis])
        _handle_output_dim(
            state.primal_dims,
            rid,
            info.final_rhs_block_size,
            info.rhs_val_axis,
            presence.rhs,
            state.squeeze_axes,
        )
    elif not pm.rhs.shared_dim:
        state.squeeze_axes.extend([info.shared_val_axis, info.rhs_val_axis])
        _handle_output_dim(
            state.out_dims,
            lid,
            info.final_lhs_block_size,
            info.lhs_val_axis,
            presence.lhs,
            state.squeeze_axes,
        )
    else:
        _handle_output_sparse(
            state.out_dims,
            lid,
            info.shared_factor,
            info.shared_val_axis,
            presence.shared,
            rid,
            info.final_lhs_block_size,
            info.lhs_val_axis,
            presence.lhs,
            state.squeeze_axes,
        )
        _handle_output_sparse(
            state.primal_dims,
            rid,
            info.shared_factor,
            info.shared_val_axis,
            presence.shared,
            lid,
            info.final_rhs_block_size,
            info.rhs_val_axis,
            presence.rhs,
            state.squeeze_axes,
        )


def _handle_batch_case(
    pm: DimensionPair,
    info: OutputPairInfo,
    presence: PresenceInfo,
    state: OutputState,
) -> None:
    dim_id = (
        pm.lhs.dim.id
        if pm.lhs.dim
        else (pm.lhs.shared_dim.id if pm.lhs.shared_dim else -1)
    )
    dimension = DenseDimension(
        dim_id,
        info.shared_factor,
        val_dim=info.shared_val_axis if presence.shared else None,
    )
    if not presence.shared:
        state.squeeze_axes.append(info.shared_val_axis)
    (state.out_dims if pm.pairing_type == "batch_out" else state.primal_dims).append(
        dimension
    )
    state.squeeze_axes.extend([info.lhs_val_axis, info.rhs_val_axis])


def _handle_batch_sparse_case(
    pm: DimensionPair,
    info: OutputPairInfo,
    presence: PresenceInfo,
    state: OutputState,
) -> None:
    l_id = pm.lhs.dim.id if pm.lhs.dim else -1
    r_id = pm.rhs.shared_dim.id if pm.rhs.shared_dim else -1
    if pm.lhs.outer_len == 1 and pm.rhs.outer_len == 1:
        state.squeeze_axes.append(info.shared_val_axis)
        _handle_output_dim(
            state.out_dims,
            l_id,
            info.final_lhs_block_size,
            info.lhs_val_axis,
            presence.lhs,
            state.squeeze_axes,
        )
        _handle_output_dim(
            state.primal_dims,
            r_id,
            info.final_rhs_block_size,
            info.rhs_val_axis,
            presence.rhs,
            state.squeeze_axes,
        )
    else:
        _handle_output_sparse(
            state.out_dims,
            l_id,
            info.shared_factor,
            info.shared_val_axis,
            presence.shared,
            r_id,
            info.final_lhs_block_size,
            info.lhs_val_axis,
            presence.lhs,
            state.squeeze_axes,
        )
        _handle_output_sparse(
            state.primal_dims,
            r_id,
            info.shared_factor,
            info.shared_val_axis,
            presence.shared,
            l_id,
            info.final_rhs_block_size,
            info.rhs_val_axis,
            presence.rhs,
            state.squeeze_axes,
        )


def _handle_spatial_sparse_case(
    pm: DimensionPair,
    info: OutputPairInfo,
    presence: PresenceInfo,
    state: OutputState,
) -> None:
    if pm.pairing_type == "spatial_sparse_lhs":
        l_id = pm.lhs.dim.id if pm.lhs.dim else -1
        s_id = pm.lhs.shared_dim.id if pm.lhs.shared_dim else -1
        if pm.lhs.outer_len == 1:
            state.squeeze_axes.append(info.shared_val_axis)
            _handle_output_dim(
                state.out_dims,
                l_id,
                -1,
                info.lhs_val_axis,
                presence.lhs,
                state.squeeze_axes,
            )
            _handle_output_dim(
                state.primal_dims,
                s_id,
                -1,
                info.rhs_val_axis,
                pm.lhs.shared_block_val_dim is not None,
                state.squeeze_axes,
            )
        else:
            _handle_output_sparse(
                state.out_dims,
                l_id,
                pm.lhs.outer_len,
                info.shared_val_axis,
                presence.shared,
                s_id,
                pm.lhs.block_len,
                info.lhs_val_axis,
                presence.lhs,
                state.squeeze_axes,
            )
            _handle_output_sparse(
                state.primal_dims,
                s_id,
                pm.lhs.outer_len,
                info.shared_val_axis,
                presence.shared,
                l_id,
                pm.lhs.shared_block_len,
                info.rhs_val_axis,
                presence.rhs,
                state.squeeze_axes,
            )
    else:  # spatial_sparse_rhs
        r_id = pm.rhs.dim.id if pm.rhs.dim else -1
        s_id = pm.rhs.shared_dim.id if pm.rhs.shared_dim else -1
        if pm.rhs.outer_len == 1:
            state.squeeze_axes.append(info.shared_val_axis)
            _handle_output_dim(
                state.out_dims,
                r_id,
                info.final_lhs_block_size,
                info.lhs_val_axis,
                pm.rhs.block_val_dim is not None,
                state.squeeze_axes,
            )
            _handle_output_dim(
                state.primal_dims,
                s_id,
                info.final_rhs_block_size,
                info.rhs_val_axis,
                presence.rhs,
                state.squeeze_axes,
            )
        else:
            _handle_output_sparse(
                state.out_dims,
                r_id,
                pm.rhs.outer_len,
                info.shared_val_axis,
                presence.shared,
                s_id,
                pm.rhs.block_len,
                info.lhs_val_axis,
                pm.rhs.block_val_dim is not None,
                state.squeeze_axes,
            )
            _handle_output_sparse(
                state.primal_dims,
                s_id,
                pm.rhs.outer_len,
                info.shared_val_axis,
                presence.shared,
                r_id,
                pm.rhs.shared_block_len,
                info.rhs_val_axis,
                presence.rhs,
                state.squeeze_axes,
            )


def _handle_spatial_simple_case(
    pm: DimensionPair,
    info: OutputPairInfo,
    presence: PresenceInfo,
    state: OutputState,
) -> None:
    if pm.pairing_type in ("spatial_out_lhs", "spatial_out_rhs"):
        is_A = pm.pairing_type == "spatial_out_lhs"
        target_dim = pm.lhs.dim if is_A else pm.rhs.dim
        val_ax = info.lhs_val_axis if is_A else info.rhs_val_axis
        is_pres = presence.lhs if is_A else (pm.rhs.block_val_dim is not None)
        if target_dim:
            _handle_output_dim(
                state.out_dims,
                target_dim.id,
                info.final_lhs_block_size if is_A else info.final_rhs_block_size,
                val_ax,
                is_pres,
                state.squeeze_axes,
            )
        state.squeeze_axes.extend(
            [info.shared_val_axis, info.rhs_val_axis if is_A else info.lhs_val_axis]
        )
    else:  # spatial_primal_lhs / B
        is_A = pm.pairing_type == "spatial_primal_lhs"
        target_dim = pm.lhs.shared_dim if is_A else pm.rhs.shared_dim
        val_ax = info.lhs_val_axis if is_A else info.rhs_val_axis
        is_pres = (pm.lhs.shared_block_val_dim is not None) if is_A else presence.rhs
        if target_dim:
            _handle_output_dim(
                state.primal_dims,
                target_dim.id,
                info.final_lhs_block_size if is_A else info.final_rhs_block_size,
                val_ax,
                is_pres,
                state.squeeze_axes,
            )
        state.squeeze_axes.extend(
            [info.shared_val_axis, info.rhs_val_axis if is_A else info.lhs_val_axis]
        )


def _resolve_output_pair(
    i: int,
    pm: DimensionPair,
    res: ContractionResult,
    axis_map: OutputAxisMap,
    state: OutputState,
    presence: PresenceInfo,
) -> None:
    shared_factor = res.shared_tiling_factors[i]
    final_lhs_block_size = (pm.lhs.outer_len // shared_factor) * res.lhs_block_lens[i]
    final_rhs_block_size = (pm.rhs.outer_len // shared_factor) * res.rhs_block_lens[i]

    any_val_present = (
        presence.shared
        or presence.lhs
        or presence.rhs
        or pm.lhs.shared_block_val_dim is not None
        or pm.rhs.block_val_dim is not None
    )
    if any_val_present:
        presence = PresenceInfo(
            shared=presence.shared or (shared_factor > 1),
            lhs=presence.lhs or (final_lhs_block_size > 1),
            rhs=presence.rhs or (final_rhs_block_size > 1),
        )

    info = OutputPairInfo(
        shared_factor=shared_factor,
        shared_val_axis=axis_map.shared[i],
        lhs_val_axis=axis_map.lhs[i],
        rhs_val_axis=axis_map.rhs[i],
        final_lhs_block_size=final_lhs_block_size,
        final_rhs_block_size=final_rhs_block_size,
    )

    if pm.pairing_type == "contract":
        _handle_contract_case(pm, info, presence, state)
    elif pm.pairing_type in ("batch_out", "batch_primal"):
        _handle_batch_case(pm, info, presence, state)
    elif pm.pairing_type == "batch_sparse":
        _handle_batch_sparse_case(pm, info, presence, state)
    elif pm.pairing_type in ("spatial_sparse_lhs", "spatial_sparse_rhs"):
        _handle_spatial_sparse_case(pm, info, presence, state)
    else:  # spatial_out_* / spatial_primal_*
        _handle_spatial_simple_case(pm, info, presence, state)


def _matmul_fwd_rule(lhs, rhs):
    return matmul(lhs, rhs), (lhs, rhs)


def _matmul_bwd_rule(res, g):
    return matmul(g, res[1].T), matmul(res[0].T, g)


matmul.defvjp(_matmul_fwd_rule, _matmul_bwd_rule)
