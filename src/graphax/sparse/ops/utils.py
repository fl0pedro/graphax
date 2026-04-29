from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from typing import TYPE_CHECKING, Any, Sequence
from itertools import chain, count
from dataclasses import replace
import copy

from graphax.sparse.dimensions import Dimension, DenseDimension, SparseDimension

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


def _is_sparse(obj) -> bool:
    if getattr(type(obj), "__name__", "") == "SparseTensor":
        return True
    if isinstance(obj, tuple) and len(obj) > 0:
        return getattr(type(obj[0]), "__name__", "") == "SparseTensor"
    return False


def _check_sparse_dim_pair(d, dim_map):
    other = dim_map.get(d.other_id)
    if (
        not isinstance(other, SparseDimension)
        or other.other_id != d.id
        or d.size != other.size
    ):
        return False
    return True


def _check_block_val_dim(d, dim_map, block_val_dims):
    if d.block_val_dim in block_val_dims:
        other = dim_map.get(d.other_id)
        if not (
            isinstance(other, SparseDimension)
            and other.block_val_dim == d.block_val_dim
        ):
            raise ValueError(
                f"Topology Error: Duplicate block_val_dim {d.block_val_dim} found in SparseDimension {d.id}"
            )
    block_val_dims.add(d.block_val_dim)


def _assert_sparse_tensor_consistency(st: SparseTensor):
    from graphax.sparse.dimensions import SparseDimension

    dim_ids = [d.id for d in st.dims]
    expected_ids = list(range(len(dim_ids)))
    assert sorted(dim_ids) == expected_ids, (
        f"Topology Error: Dimension IDs must be a contiguous sequence. Got {dim_ids}"
    )

    dim_map = {d.id: d for d in st.dims}
    block_val_dims = set()
    for d in st.dims:
        if isinstance(d, SparseDimension):
            assert _check_sparse_dim_pair(d, dim_map), (
                f"Topology Error: Invalid sparse dimension pair configuration for dimension {d.id}"
            )
            if getattr(d, "block_val_dim", None) is not None:
                _check_block_val_dim(d, dim_map, block_val_dims)


def _copy(
    st: SparseTensor,
    val: Array | None = None,
    scalar_mult: Array | None = None,
    fill_value: Array | None = None,
    out_dims: Sequence[Dimension] | None = None,
    primal_dims: Sequence[Dimension] | None = None,
    deep=False,
):
    from graphax.sparse.tensor import SparseTensor

    v = val if val is not None else st.val
    s = scalar_mult if scalar_mult is not None else st.scalar_mult
    f = fill_value if fill_value is not None else st.fill_value
    od = out_dims if out_dims is not None else st.out_dims
    pd = primal_dims if primal_dims is not None else st.primal_dims

    if deep:
        v = copy.deepcopy(v) if v is not None else None
        s = copy.deepcopy(s)
        od = copy.deepcopy(od)
        pd = copy.deepcopy(pd)

    return SparseTensor(
        od,
        pd,
        v,
        s,
        f,
        st.pre_transforms,
        st.post_transforms,
        sort_val=False,
        check_consistency=False,
    )


def _map_sparse_axes(dims_list, sparse_axis_map, counter, perm):
    for d in dims_list:
        if (
            isinstance(d, SparseDimension)
            and d.val_dim is not None
            and d.val_dim not in sparse_axis_map
        ):
            sparse_axis_map[d.val_dim] = next(counter)
            perm.append(d.val_dim)


def _map_dense_axes(dims, dense_axis_map, counter, perm):
    for d in dims:
        if (
            isinstance(d, DenseDimension)
            and d.val_dim is not None
            and d.val_dim not in dense_axis_map
        ):
            dense_axis_map[d.val_dim] = next(counter)
            perm.append(d.val_dim)
        if (
            isinstance(d, SparseDimension)
            and d.block_val_dim is not None
            and d.block_val_dim not in dense_axis_map
        ):
            dense_axis_map[d.block_val_dim] = next(counter)
            perm.append(d.block_val_dim)


def _update_dim_axes(ds, s_map, d_map):
    res = []
    for d in ds:
        if isinstance(d, SparseDimension):
            nv = s_map.get(d.val_dim) if d.val_dim is not None else None
            nb = d_map.get(d.block_val_dim) if d.block_val_dim is not None else None
            res.append(replace(d, val_dim=nv, block_val_dim=nb))
        else:
            nv = d_map.get(d.val_dim) if d.val_dim is not None else None
            res.append(replace(d, val_dim=nv))
    return tuple(res)


def _sort_val(
    out_dims: Sequence[Dimension], primal_dims: Sequence[Dimension], val: Array | None
) -> tuple[tuple[Dimension, ...], tuple[Dimension, ...], Array | None]:
    if val is None:
        return tuple(out_dims), tuple(primal_dims), None

    s_map, s_perm, c_s = {}, [], count()
    _map_sparse_axes(out_dims, s_map, c_s, s_perm)
    _map_sparse_axes(primal_dims, s_map, c_s, s_perm)

    d_map, d_perm, c_d = {}, [], count(len(s_map))
    _map_dense_axes(tuple(out_dims) + tuple(primal_dims), d_map, c_d, d_perm)

    new_out, new_primal = (
        _update_dim_axes(out_dims, s_map, d_map),
        _update_dim_axes(primal_dims, s_map, d_map),
    )

    full_perm, seen = [], set()
    for p in s_perm + d_perm:
        if p not in seen:
            full_perm.append(p)
            seen.add(p)
    full_perm.extend(i for i in range(val.ndim) if i not in seen)

    return new_out, new_primal, val.transpose(full_perm)


def _arr2st(
    arr: Array, out_ndim: int | None = None, dtype: Any = None, **kwargs: Any
) -> SparseTensor:
    from graphax.sparse.tensor import SparseTensor
    from graphax.sparse.dimensions import DenseDimension
    if dtype is not None:
        arr = arr.astype(dtype)
    if out_ndim is None:
        out_ndim = arr.ndim // 2
    # Ensure rank 1 as minimum for non-empty arrays
    if arr.ndim == 0:
        arr = jnp.expand_dims(arr, 0)
        out_ndim = max(out_ndim or 0, 0)

    dims = tuple(DenseDimension(i, s, i) for i, s in enumerate(arr.shape))
    return SparseTensor(
        dims[:out_ndim],
        dims[out_ndim:],
        arr,
        sort_val=False,
        check_consistency=False,
        **kwargs,
    )


def _materialize_dimensions(st: SparseTensor, dims: Sequence[int]) -> Array:
    """
    Function that materializes the `val` property of a `SparseTensor` object
    along a given set of axes.
    """
    val = st.val if st.val is not None else jnp.array(1.0, dtype=st.dtype)
    if not dims:
        return val

    dims = sorted(dims)
    _dims, counter = [], val.ndim
    for d in dims:
        if d < counter:
            _dims.append(d)
        else:
            _dims.append(counter)
            counter += 1

    return jnp.expand_dims(val, axis=_dims)


def _swap_back_axes(st: SparseTensor) -> SparseTensor:
    """
    After operations that might have permuted dimensions, this function restores
    the canonical order where physical axes match logical order.
    """
    if st.val is None:
        return st

    l = len(st.out_dims)
    i = 0
    permutation = [0] * st.val.ndim
    for d in st.dims:
        if d.val_dim is not None:
            if isinstance(d, DenseDimension) or d.id < getattr(
                d, "other_id", float("inf")
            ):
                permutation[i] = d.val_dim
                i += 1
        if (
            isinstance(d, SparseDimension)
            and getattr(d, "block_val_dim", None) is not None
        ):
            permutation[i] = d.block_val_dim
            i += 1

    # Fill remaining axes if any
    seen = set(permutation[:i])
    for j in range(st.val.ndim):
        if j not in seen:
            if i < len(permutation):
                permutation[i] = j
                i += 1

    new_val = st.val.transpose(permutation)

    # Update dimension objects
    i = 0
    new_dims = []
    dim_map = {d.id: d for d in st.dims}

    # We need to update both out_dims and primal_dims, but they might share SparseDimensions
    processed_ids = {}

    def update_dim(d, current_i):
        if d.id in processed_ids:
            return processed_ids[d.id], current_i

        nv, nb = d.val_dim, getattr(d, "block_val_dim", None)
        if nv is not None:
            if isinstance(d, DenseDimension) or d.id < d.other_id:
                nv = current_i
                current_i += 1
            else:
                # It's a SparseDimension and we are at the second one of the pair
                other = dim_map[d.other_id]
                nv = processed_ids[other.id].val_dim

        if nb is not None:
            nb = current_i
            current_i += 1

        new_d = replace(d, val_dim=nv)
        if isinstance(new_d, SparseDimension):
            new_d = replace(new_d, block_val_dim=nb)

        processed_ids[d.id] = new_d
        return new_d, current_i

    new_out = []
    for d in st.out_dims:
        nd, i = update_dim(d, i)
        new_out.append(nd)

    new_primal = []
    for d in st.primal_dims:
        nd, i = update_dim(d, i)
        new_primal.append(nd)

    return _copy(st, val=new_val, out_dims=tuple(new_out), primal_dims=tuple(new_primal))