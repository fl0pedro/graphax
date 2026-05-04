from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from graphax.sparse.ops.utils import _arr2st, _is_sparse
from graphax.sparse.ops.layout import generate_block_permutation

from graphax.sparse.indexes import Index, SparseIndex, DenseIndex

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


def _normalize_inputs(
    lhs: SparseTensor | Array, rhs: SparseTensor | Array
) -> tuple[SparseTensor, SparseTensor]:
    # Determine target dtype for potential dense-to-sparse promotion
    target_dtype = None
    if _is_sparse(lhs) and not _is_sparse(rhs):
        target_dtype = lhs.dtype
    elif _is_sparse(rhs) and not _is_sparse(lhs):
        target_dtype = rhs.dtype

    inputs = [lhs, rhs]
    for i in range(2):
        if not _is_sparse(inputs[i]):
            other = inputs[1 - i]
            obj = inputs[i].dense() if hasattr(inputs[i], "dense") else inputs[i]
            if hasattr(other, "shape") and getattr(obj, "size", 0) == 1:
                obj = jnp.broadcast_to(obj, other.shape)
            inputs[i] = _arr2st(
                obj,
                out_ndim=len(other.out_dims) if _is_sparse(other) else None,
                dtype=target_dtype,
            )
        elif isinstance(inputs[i], tuple):
            inputs[i] = inputs[i][0]

    lhs, rhs = tuple(inputs)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Shape mismatch: {lhs.shape} != {rhs.shape}")
    return lhs, rhs


def _handle_sparse_sparse_pair(left_dim, right_dim, left_dims, right_dims):
    partner_idx = next(j for j, d in enumerate(left_dims) if d.id == left_dim.other_id)
    left_partner, right_partner = left_dims[partner_idx], right_dims[partner_idx]

    if (
        not isinstance(right_partner, SparseIndex)
        or right_dim.other_id != right_partner.id
    ):
        raise ValueError(
            "Topology mismatch: LHS and RHS sparse pairs do not positionally align."
        )
    return (left_dim, left_partner, right_dim, right_partner), partner_idx


def _promote_left_dense_to_sparse(left_dim, right_dim, left_dims, right_dims):
    partner_idx = next(
        j for j, d in enumerate(right_dims) if d.id == right_dim.other_id
    )
    right_partner, left_partner = right_dims[partner_idx], left_dims[partner_idx]

    if not isinstance(left_partner, DenseIndex):
        raise ValueError(
            "Topology mismatch: Expected DenseIndex to pair with SparseIndex"
        )

    synthetic_left = SparseIndex(
        left_dim.id,
        1,
        axis=None,
        other_id=left_partner.id,
        block_size=left_dim.size,
        block_axis=left_dim.axis,
    )
    synthetic_left_partner = SparseIndex(
        left_partner.id,
        1,
        axis=None,
        other_id=left_dim.id,
        block_size=left_partner.size,
        block_axis=left_partner.axis,
    )
    return (
        synthetic_left,
        synthetic_left_partner,
        right_dim,
        right_partner,
    ), partner_idx


def _promote_right_dense_to_sparse(left_dim, right_dim, left_dims, right_dims):
    partner_idx = next(j for j, d in enumerate(left_dims) if d.id == left_dim.other_id)
    left_partner, right_partner = left_dims[partner_idx], right_dims[partner_idx]

    if not isinstance(right_partner, DenseIndex):
        raise ValueError(
            "Topology mismatch: Expected DenseIndex to pair with SparseIndex"
        )

    synthetic_right = SparseIndex(
        right_dim.id,
        1,
        axis=None,
        other_id=right_partner.id,
        block_size=right_dim.size,
        block_axis=right_dim.axis,
    )
    synthetic_right_partner = SparseIndex(
        right_partner.id,
        1,
        axis=None,
        other_id=right_dim.id,
        block_size=right_partner.size,
        block_axis=right_partner.axis,
    )
    return (
        left_dim,
        left_partner,
        synthetic_right,
        synthetic_right_partner,
    ), partner_idx


def _resolve_dim_pairing(i, left_dims, right_dims, processed_indices):
    left_dim, right_dim = left_dims[i], right_dims[i]

    if isinstance(left_dim, SparseIndex) and isinstance(right_dim, SparseIndex):
        pair, partner_idx = _handle_sparse_sparse_pair(
            left_dim, right_dim, left_dims, right_dims
        )
        processed_indices.update([i, partner_idx])
        return "sparse", pair

    if isinstance(left_dim, DenseIndex) and isinstance(right_dim, SparseIndex):
        pair, partner_idx = _promote_left_dense_to_sparse(
            left_dim, right_dim, left_dims, right_dims
        )
        processed_indices.update([i, partner_idx])
        return "sparse", pair

    if isinstance(right_dim, DenseIndex) and isinstance(left_dim, SparseIndex):
        pair, partner_idx = _promote_right_dense_to_sparse(
            left_dim, right_dim, left_dims, right_dims
        )
        processed_indices.update([i, partner_idx])
        return "sparse", pair

    if not isinstance(left_dim, SparseIndex) and not isinstance(
        right_dim, SparseIndex
    ):
        processed_indices.add(i)
        return "dense", (left_dim, right_dim)

    raise ValueError("Topology mismatch: Unhandled dimension combination.")


def _map_topology(
    left_tensor: SparseTensor, right_tensor: SparseTensor
) -> tuple[list[tuple], list[tuple]]:
    sparse_pairs, dense_pairs, processed_indices = [], [], set()
    for i in range(len(left_tensor.dims)):
        if i not in processed_indices:
            type_key, pair = _resolve_dim_pairing(
                i, left_tensor.dims, right_tensor.dims, processed_indices
            )
            (sparse_pairs if type_key == "sparse" else dense_pairs).append(pair)
    return sparse_pairs, dense_pairs


def _calculate_pair_metric(pair):
    left_d1, left_d2, right_d1, right_d2 = pair
    left_b1, left_b2 = left_d1.block_size or 1, left_d2.block_size or 1
    right_b1, right_b2 = right_d1.block_size or 1, right_d2.block_size or 1
    common_b1, common_b2 = math.lcm(left_b1, right_b1), math.lcm(left_b2, right_b2)
    return {
        "unified_size": left_d1.size // (common_b1 // left_b1),
        "common_b1": common_b1,
        "common_b2": common_b2,
        "left_b1": left_b1,
        "left_b2": left_b2,
        "right_b1": right_b1,
        "right_b2": right_b2,
        "left_size": left_d1.size,
        "right_size": right_d1.size,
    }


def _compute_pair_metrics(sparse_pairs: list[tuple]) -> list[dict]:
    return [_calculate_pair_metric(p) for p in sparse_pairs]


def _get_axes_info(sparse_pairs, dense_pairs, is_left):
    axes = []
    for pair in sparse_pairs:
        d1, d2 = (pair[0], pair[1]) if is_left else (pair[2], pair[3])
        axes.extend(
            [
                d1.axis,
                getattr(d1, "block_axis", None),
                getattr(d2, "block_axis", None),
            ]
        )
    for pair in dense_pairs:
        axes.append((pair[0] if is_left else pair[1]).axis)
    return axes


def _get_value_axes_info(
    tensor: SparseTensor,
    sparse_pairs: list[tuple],
    dense_pairs: list[tuple],
    is_left: bool,
):
    value = tensor.val if tensor.val is not None else jnp.array(1.0, dtype=tensor.dtype)
    axes = _get_axes_info(sparse_pairs, dense_pairs, is_left)
    used_axes = [ax for ax in axes if ax is not None]
    unused_shape = [value.shape[i] for i in range(value.ndim) if i not in used_axes]
    return value, axes, unused_shape


def _align_tensor_values(
    value: Array,
    tensor: SparseTensor,
    sparse_pairs: list[tuple],
    dense_pairs: list[tuple],
    axes: list[int | None],
    broadcast_unused_shape: list[int],
    is_left: bool,
) -> Array:
    if tensor.dtype == jnp.bool_:
        value = value & tensor.scalar_mult.astype(jnp.bool_)
    else:
        value = value * tensor.scalar_mult
    target_shape = []
    for pair in sparse_pairs:
        d1 = pair[0] if is_left else pair[2]
        target_shape.extend(
            [
                d1.size,
                d1.block_size or 1,
                (pair[1] if is_left else pair[3]).block_size or 1,
            ]
        )
    for pair in dense_pairs:
        target_shape.append((pair[0] if is_left else pair[1]).size)

    used_axes = [ax for ax in axes if ax is not None]
    unused_indices = [i for i in range(value.ndim) if i not in used_axes]
    value = value.transpose(used_axes + unused_indices)

    broadcast_dims = [i for i, ax in enumerate(axes) if ax is not None]
    offset = len(axes) + len(broadcast_unused_shape) - len(unused_indices)
    broadcast_dims.extend([offset + i for i in range(len(unused_indices))])

    return jax.lax.broadcast_in_dim(
        value,
        tuple(target_shape) + tuple(broadcast_unused_shape),
        tuple(broadcast_dims),
    )


def _promote_to_unified_blocks(
    value: Array, pair_metrics: list[dict], is_left: bool
) -> Array:
    input_reshape, expansion_reshape, output_reshape = [], [], []
    needs_expansion = False

    for metrics in pair_metrics:
        b1, b2 = (
            (metrics["left_b1"], metrics["left_b2"])
            if is_left
            else (metrics["right_b1"], metrics["right_b2"])
        )
        common_b1, common_b2 = metrics["common_b1"], metrics["common_b2"]
        exp = common_b1 // b1

        input_reshape.extend([metrics["unified_size"], exp, b1, b2])
        expansion_reshape.extend([metrics["unified_size"], exp, 1, b1, b2])
        output_reshape.extend([metrics["unified_size"], common_b1, common_b2])

        if exp > 1:
            needs_expansion = True

    rem = list(value.shape[3 * len(pair_metrics) :])
    input_reshape.extend(rem)
    expansion_reshape.extend(rem)
    output_reshape.extend(rem)

    value = value.reshape(input_reshape).reshape(expansion_reshape)

    # FAST PATH: If block sizes match identically, avoid masking entirely
    if not needs_expansion:
        perm = generate_block_permutation(len(pair_metrics), 5, [0, 1, 3, 2, 4])
        perm.extend(range(5 * len(pair_metrics), len(expansion_reshape)))
        return value.transpose(perm).reshape(output_reshape)

    # Construct a pure boolean mask only for dimensions that need expansion
    mask = None
    for i, m in enumerate(pair_metrics):
        exp = m["common_b1"] // (m["left_b1"] if is_left else m["right_b1"])
        if exp > 1:
            m_shape = [1] * len(expansion_reshape)
            m_shape[5 * i + 1] = exp
            m_shape[5 * i + 2] = exp
            eye_mask = jnp.eye(exp, dtype=jnp.bool_).reshape(m_shape)
            mask = eye_mask if mask is None else mask & eye_mask

    # Predication (jnp.where) instead of arithmetic mapping (value * mask)
    if mask is not None:
        value = jnp.where(mask, value, jnp.array(0, dtype=value.dtype))

    perm = generate_block_permutation(len(pair_metrics), 5, [0, 1, 3, 2, 4])
    perm.extend(range(5 * len(pair_metrics), len(expansion_reshape)))
    return value.transpose(perm).reshape(output_reshape)


def _demote_intersection(
    value: Array, pair_metrics: list[dict], is_intersection: bool
) -> tuple[Array, list[list[int]]]:
    if not is_intersection:
        return value, [
            [m["unified_size"], m["common_b1"], m["common_b2"]] for m in pair_metrics
        ]

    input_reshape, output_reshape, sum_axes, meta = [], [], [], []
    off = 0
    for m in pair_metrics:
        min_b1, min_b2 = (
            min(m["left_b1"], m["right_b1"]),
            min(m["left_b2"], m["right_b2"]),
        )
        dex = m["common_b1"] // min_b1
        input_reshape.extend([m["unified_size"], dex, min_b1, dex, min_b2])
        output_reshape.extend([m["unified_size"] * dex, min_b1, min_b2])
        if dex > 1:
            sum_axes.append(off + 3)
        off += 5
        meta.append([m["unified_size"] * dex, min_b1, min_b2])

    rem = list(value.shape[3 * len(pair_metrics) :])
    input_reshape.extend(rem)
    output_reshape.extend(rem)
    if sum_axes:
        value = (
            value.reshape(input_reshape)
            .sum(axis=tuple(sum_axes))
            .reshape(output_reshape)
        )
    return value, meta


def _reconstruct_dimension_pair(i, pair, meta, next_info):
    lid1, lid2 = pair[0].id, pair[1].id
    size, b1, b2 = meta[i]

    v_ax = next_info["axis"] if size > 1 else None
    if size > 1:
        next_info["axis"] += 1
    else:
        next_info["squeeze"].append(3 * i)

    b1_ax = next_info["axis"] if b1 > 1 else None
    if b1 > 1:
        next_info["axis"] += 1
    else:
        next_info["squeeze"].append(3 * i + 1)

    b2_ax = next_info["axis"] if b2 > 1 else None
    if b2 > 1:
        next_info["axis"] += 1
    else:
        next_info["squeeze"].append(3 * i + 2)

    return (
        lid1,
        replace(
            pair[0],
            size=size,
            block_size=b1 if b1 > 1 else None,
            axis=v_ax,
            block_axis=b1_ax,
        ),
    ), (
        lid2,
        replace(
            pair[1],
            size=size,
            block_size=b2 if b2 > 1 else None,
            axis=v_ax,
            block_axis=b2_ax,
        ),
    )


def _reconstruct_result_tensor(
    value: Array,
    lhs: SparseTensor,
    sparse_pairs: list[tuple],
    dense_pairs: list[tuple],
    output_pairs_meta: list[list[int]],
    op: Callable,
    rhs: SparseTensor,
) -> SparseTensor:
    from graphax.sparse.tensor import SparseTensor

    reconstructed = {}
    info = {"axis": 0, "squeeze": []}
    for i, pair in enumerate(sparse_pairs):
        p1, p2 = _reconstruct_dimension_pair(i, pair, output_pairs_meta, info)
        reconstructed[p1[0]], reconstructed[p2[0]] = p1[1], p2[1]
    for pair in dense_pairs:
        reconstructed[pair[0].id] = replace(pair[0], axis=info["axis"])
        info["axis"] += 1

    if info["squeeze"]:
        value = value[
            tuple(
                0 if ax in info["squeeze"] else slice(None) for ax in range(value.ndim)
            )
        ]

    s_mult = (
        jnp.array(1.0, dtype=value.dtype)
        if value.dtype != jnp.bool_
        else jnp.array(True)
    )
    return SparseTensor(
        tuple(reconstructed[d.id] for d in lhs.out_dims),
        tuple(reconstructed[d.id] for d in lhs.primal_dims),
        value,
        scalar_mult=s_mult,
        fill_value=op(
            lhs.fill_value * lhs.scalar_mult, rhs.fill_value * rhs.scalar_mult
        ),
        check_consistency=False,
    )


def elementwise(
    lhs: SparseTensor | Array,
    rhs: SparseTensor | Array,
    op: Callable,
    is_intersection: bool = False,
    count: bool = False,
) -> SparseTensor | tuple[SparseTensor, Array]:
    lhs, rhs = _normalize_inputs(lhs, rhs)
    sparse_pairs, dense_pairs = _map_topology(lhs, rhs)
    metrics = _compute_pair_metrics(sparse_pairs)
    vl, al, ul = _get_value_axes_info(lhs, sparse_pairs, dense_pairs, True)
    vr, ar, ur = _get_value_axes_info(rhs, sparse_pairs, dense_pairs, False)
    bus = list(jnp.broadcast_shapes(tuple(ul), tuple(ur)))
    vl = _align_tensor_values(vl, lhs, sparse_pairs, dense_pairs, al, bus, True)
    vr = _align_tensor_values(vr, rhs, sparse_pairs, dense_pairs, ar, bus, False)
    res_val = op(
        _promote_to_unified_blocks(vl, metrics, True),
        _promote_to_unified_blocks(vr, metrics, False),
    )
    res_val, out_meta = _demote_intersection(res_val, metrics, is_intersection)
    res = _reconstruct_result_tensor(
        res_val, lhs, sparse_pairs, dense_pairs, out_meta, op, rhs
    )

    if count:
        return res, math.prod(np.minimum(vl.shape, vr.shape))
    else:
        return res


def add_w_counts(a, b):
    res = elementwise(a, b, jax.lax.add, count=True)
    return res[0], (res[1], 0, 0)


def mul_w_counts(a, b):
    res = elementwise(a, b, jax.lax.mul, is_intersection=True, count=True)
    return res[0], (0, res[1], 0)