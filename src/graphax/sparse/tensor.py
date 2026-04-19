from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from itertools import chain, count
from math import gcd, prod
from typing import Any, Callable, Literal, Sequence
from functools import reduce, partial

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.tree_util import register_pytree_node_class


from dataclasses import dataclass, replace, field

@dataclass(frozen=True)
class Dimension(ABC):
    id: int
    size: int
    val_dim: int | None
    logical_id: int | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if self.size < 0:
            raise ValueError(f"Dimension size must be non-negative, got {self.size}")

    @property
    @abstractmethod
    def logical_size(self) -> int:
        pass


@dataclass(frozen=True)
class DenseDimension(Dimension):
    @property
    def logical_size(self) -> int:
        return self.size


@dataclass(frozen=True)
class SparseDimension(Dimension):
    other_id: int
    block_size: int | None = None
    block_val_dim: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.block_size is not None and self.block_size <= 0:
            raise ValueError(
                f"SparseDimension block_size must be positive, got {self.block_size}"
            )

    @property
    def logical_size(self) -> int:
        return self.size * (self.block_size or 1)


Transform = Callable[["SparseTensor", "SparseTensor", Array], "SparseTensor"]


@register_pytree_node_class
@dataclass(init=False, frozen=True)
class SparseTensor:
    out_dims: tuple[Dimension, ...]
    primal_dims: tuple[Dimension, ...]
    val: Array | None
    scalar_mult: Array
    pre_transforms: tuple[Transform, ...]
    post_transforms: tuple[Transform, ...]

    def tree_flatten(self):
        children = (self.val, self.scalar_mult)
        dynamic_kwargs = tuple(
            (k, getattr(self, k)) for k in getattr(self, "_dynamic_keys", ())
        )
        aux_data = (
            self.out_dims,
            self.primal_dims,
            self.pre_transforms,
            self.post_transforms,
            dynamic_kwargs,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        val, scalar_mult = children
        out_dims, primal_dims, pre_transforms, post_transforms, dynamic_kwargs = (
            aux_data
        )
        kwargs = dict(dynamic_kwargs)
        obj = cls.__new__(cls)
        _immutably_assign(
            obj,
            out_dims,
            primal_dims,
            val,
            scalar_mult,
            pre_transforms,
            post_transforms,
            _dynamic_keys=tuple(kwargs.keys()),
            **kwargs,
        )
        _assert_sparse_tensor_consistency(obj)
        return obj

    def __init__(
        self,
        out_dims: Sequence[Dimension],
        primal_dims: Sequence[Dimension],
        val: Array | None,
        scalar_mult: Array | None = None,
        dtype: DTypeLike | None = None,
        pre_transforms: Sequence[Callable] | None = None,
        post_transforms: Sequence[Callable] | None = None,
        **kwargs,
    ):
        if scalar_mult is None:
            scalar_mult = jnp.array(1.0)

        if pre_transforms is not None:
            pre_transforms = tuple(pre_transforms)
        else:
            pre_transforms = ()

        if post_transforms is not None:
            post_transforms = tuple(post_transforms)
        else:
            post_transforms = ()

        out_dims, primal_dims, val = _sort_val(out_dims, primal_dims, val)
        _immutably_assign(
            self,
            out_dims,
            primal_dims,
            val,
            scalar_mult,
            pre_transforms,
            post_transforms,
            **kwargs,
        )
        _assert_sparse_tensor_consistency(self)

    def __repr__(self) -> str:
        def _repr_tuple(t: tuple) -> str:
            res = "(\n    " + ",\n    ".join((str(x) for x in t)) + ",\n  )"
            return res if t else "()"

        return (
            f"SparseTensor(\n"
            f"  shape = {self._repr_shape()},\n"
            f"  out_dims = {_repr_tuple(self.out_dims)},\n"
            f"  primal_dims = {_repr_tuple(self.primal_dims)},\n"
            f"  val = {self._repr_val()},\n"
            f"  pre_transforms = {_repr_tuple(self.pre_transforms)},\n"
            f"  post_transforms = {_repr_tuple(self.post_transforms)}\n"
            f")"
        )

    def _repr_val(self) -> str:
        if self.val is not None:
            return f"Array(shape=({str(list(self.sparse_shape))[1:-1]}{'; ' if self.sparse_shape else ''}{str(list(self.dense_shape))[1:-1]}), dtype={self.dtype})"
        return "None"

    def _repr_shape(self) -> str:
        return f"({str(list(self.out_shape))[1:-1]} | {str(list(self.primal_shape))[1:-1]})"

    @property
    def out_shape(self) -> tuple[int, ...]:
        return tuple(d.logical_size for d in self.out_dims)

    @property
    def primal_shape(self) -> tuple[int, ...]:
        return tuple(d.logical_size for d in self.primal_dims)

    def sparse_pairs(self, key: Literal["out", "primal"] = "out") -> dict[int, int]:
        res = {}
        for d in self.out_dims:
            if isinstance(d, SparseDimension):
                if key == "out":
                    res[d.id] = d.other_id
                elif key == "primal":
                    res[d.other_id] = d.id
        return res

    @property
    def sparse_shape(self) -> tuple[int, ...]:
        return tuple(d.size for d in self.out_dims if isinstance(d, SparseDimension))

    @property
    def sparse_size(self) -> int:
        return prod(self.sparse_shape)

    @property
    def sparse_ndim(self) -> int:
        return len(self.sparse_shape)

    @property
    def dense_shape(self) -> tuple[int, ...]:
        return tuple(
            d.block_size or 1 if isinstance(d, SparseDimension) else d.size
            for d in self.dims
        )

    @property
    def dense_size(self) -> int:
        return prod(self.dense_shape)

    @property
    def dense_ndim(self) -> int:
        return len(self.dense_shape)

    def transpose(
        self,
        out_transpose: Sequence[int] | None = None,
        primal_transpose: Sequence[int] | None = None,
    ) -> SparseTensor:
        return _transpose(self, out_transpose, primal_transpose)

    def swapdims(self) -> SparseTensor:
        return self.transpose(
            [d.id for d in self.primal_dims], [d.id for d in self.out_dims]
        )

    @property
    def dims(self) -> tuple[Dimension, ...]:
        return self.out_dims + self.primal_dims

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(d.logical_size for d in self.dims)

    @property
    def batch_size(self) -> int:
        for d in self.dims:
            if isinstance(d, SparseDimension) and d.val_dim is not None:
                return d.size
        return 1

    @property
    def size(self) -> int:
        return prod(self.shape)

    def __array__(self, *args, **kwargs):
        import numpy as np

        return np.array(jnp.array(self), *args, **kwargs)

    def __jax_array__(self):
        return self.dense(hard=True).val * self.scalar_mult

    def dense(
        self, axes: tuple[int, ...] | None = None, hard: bool = False
    ) -> SparseTensor:
        # Proactively repair any topology or shape inconsistencies before densification
        st = _create_optimized(self.out_dims, self.primal_dims, self.val, self.scalar_mult)
        return _dense(st, axes, hard)

    @property
    def T(self) -> SparseTensor:
        return self.transpose()

    def block_until_ready(self) -> SparseTensor:
        if self.val is not None:
            self.val.block_until_ready()
        return self

    @property
    def dtype(self) -> DTypeLike:
        if self.val is None:
            return jnp.float32  # default, see _densify_implicit
        return self.val.dtype

    def copy(self, val: Array | None = None, scalar_mult: Array | None = None):
        return _copy(self, val, scalar_mult)

    def __matmul__(self, other: Any) -> SparseTensor | Array:
        return _matmul(self, other)

    def __rmatmul__(self, other: Any) -> SparseTensor | Array:
        return _rmatmul(self, other)

    def __copy__(self, val: Array | None = None, scalar_mult: Array | None = None):
        return _copy(self, val, scalar_mult)

    def __deepcopy__(self):
        return _copy(self, deep=True)

    def __add__(self, other: Any) -> SparseTensor:
        return _add(self, other)

    def __radd__(self, other: Any) -> SparseTensor:
        return _add(self, other)

    def __iadd__(self, other: Any) -> SparseTensor:
        return _add(self, other)

    def __mul__(self, other: Any) -> SparseTensor:
        return _mul(self, other)

    def __rmul__(self, other: Any) -> SparseTensor:
        return _mul(self, other)

    def __eq__(self, other: Any) -> SparseTensor | Array:
        if not isinstance(other, SparseTensor):
            if hasattr(other, "__jax_array__") or isinstance(
                other, (np.ndarray, jax.Array, int, float, bool)
            ):
                return jnp.array(self) == jnp.array(other)
            return NotImplemented
        if self.out_dims == other.out_dims and self.primal_dims == other.primal_dims:
            return _copy(
                self,
                self.val * self.scalar_mult == other.val * other.scalar_mult,
                scalar_mult=jnp.array(1.0),
                deep=True,
            )
        return jnp.array(False)

    def all(self) -> Array:
        return jnp.all(self.val)


def _copy(
    st: SparseTensor,
    val: Array | None = None,
    scalar_mult: Array | None = None,
    deep=False,
):
    v = val if val is not None else st.val
    s = scalar_mult if scalar_mult is not None else st.scalar_mult
    if deep:
        v = copy.deepcopy(v) if v is not None else None
        s = copy.deepcopy(s)
    return _create_optimized(
        st.out_dims, st.primal_dims, v, s, st.pre_transforms, st.post_transforms
    )


def _immutably_assign(
    st: SparseTensor,
    out_dims,
    primal_dims,
    val,
    scalar_mult,
    pre_transforms,
    post_transforms,
    _dynamic_keys=None,
    **kwargs,
):
    object.__setattr__(st, "out_dims", out_dims)
    object.__setattr__(st, "primal_dims", primal_dims)
    object.__setattr__(st, "val", val)
    object.__setattr__(st, "scalar_mult", scalar_mult)
    object.__setattr__(st, "pre_transforms", pre_transforms)
    object.__setattr__(st, "post_transforms", post_transforms)

    if len(kwargs) > 0:
        if _dynamic_keys is not None:
            object.__setattr__(st, "_dynamic_keys", _dynamic_keys)
        else:
            object.__setattr__(st, "_dynamic_keys", tuple(kwargs.keys()))

        for k, v in kwargs.items():
            object.__setattr__(st, k, v)


def _create_optimized(
    out_dims: Sequence[Dimension],
    primal_dims: Sequence[Dimension],
    val: Array | None,
    scalar_mult: Array | None = None,
    pre_transforms: Sequence[Callable] = (),
    post_transforms: Sequence[Callable] = (),
    **kwargs,
) -> SparseTensor:
    obj = SparseTensor.__new__(SparseTensor)
    if scalar_mult is None:
        scalar_mult = jnp.array(1.0)
    # Force IDs to match physical indices and repair topology/size inconsistencies
    new_dims = list(out_dims) + list(primal_dims)
    
    to_densify = set()
    new_dims_repaired = []
    
    for i, d in enumerate(new_dims):
        if isinstance(d, SparseDimension):
            # Check 1: Topology - Partner bounds
            if d.other_id < 0 or d.other_id >= len(new_dims):
                to_densify.add(i)
                continue
                
            # Check 2: Topology - Partner type
            other = new_dims[d.other_id]
            if not isinstance(other, SparseDimension):
                to_densify.add(i)
                continue
                
            # Check 3: Topology - Partner symmetry
            if other.other_id != i:
                to_densify.add(i)
                to_densify.add(d.other_id)
                continue
                
            # Check 4: Size - Pair agreement
            if d.size != other.size:
                to_densify.add(i)
                to_densify.add(d.other_id)
                continue
                
            # Check 5: Size - Value compatibility
            if val is not None:
                if d.val_dim is not None:
                    if d.val_dim >= val.ndim or (val.shape[d.val_dim] != d.size and val.shape[d.val_dim] != 1):
                        to_densify.add(i)
                        to_densify.add(d.other_id)
                if d.block_val_dim is not None:
                    if d.block_val_dim >= val.ndim or (val.shape[d.block_val_dim] != (d.block_size or 1) and val.shape[d.block_val_dim] != 1):
                        to_densify.add(i)
                        to_densify.add(d.other_id)
        else:
            # Check 6: DenseDimension - Value compatibility
            if val is not None and d.val_dim is not None:
                if d.val_dim < val.ndim and val.shape[d.val_dim] != d.size and val.shape[d.val_dim] != 1:
                    # Robust repair for DenseDimension: update size to match reality if it mismatch
                    # This prevents broadcasting errors in _dense
                    new_dims[i] = replace(d, size=val.shape[d.val_dim])

    # Iterative propagation of densification to ensure symmetry
    changed = True
    while changed:
        changed = False
        for i in list(to_densify):
            d = new_dims[i]
            if isinstance(d, SparseDimension) and d.other_id not in to_densify:
                to_densify.add(d.other_id)
                changed = True
                
    # Execute repairs
    for i in to_densify:
        d = new_dims[i]
        # Keep the logical identity (id and logical_id), but use physical size for the repaired DenseDimension
        new_dims[i] = DenseDimension(d.id, d.size, d.val_dim, logical_id=d.logical_id)
            
    new_out = tuple(new_dims[:len(out_dims)])
    new_primal = tuple(new_dims[len(out_dims):])

    _immutably_assign(
        obj,
        new_out,
        new_primal,
        val,
        scalar_mult,
        pre_transforms,
        post_transforms,
        **kwargs,
    )
    _assert_sparse_tensor_consistency(obj)
    return obj


def _dense(
    st: SparseTensor, axes: Sequence[int] | None = None, hard: bool = False
) -> SparseTensor:
    indices = set(range(st.ndim)) if axes is None else set(axes)
    id_map = {d.id: i for i, d in enumerate(st.dims)}

    implicit = set()
    for i in indices:
        d = st.dims[i]
        get_val = d.val_dim is None
        get_blk = (
            isinstance(d, SparseDimension) and d.block_size and d.block_val_dim is None
        )
        if get_val or get_blk:
            implicit.add(i)
            if isinstance(d, SparseDimension):
                implicit.add(id_map[d.other_id])

    if not hard:
        indices -= implicit

    val = jnp.array(1.0, dtype=st.dtype) if st.val is None else st.val
    if st.val is not None:
        target_shape = list(val.shape)
        # First pass: find required rank
        max_v = -1
        for d in st.dims:
            if d.val_dim is not None: max_v = max(max_v, d.val_dim)
            if isinstance(d, SparseDimension) and d.block_val_dim is not None: 
                max_v = max(max_v, d.block_val_dim)
        
        # Ensure target_shape is large enough
        if max_v >= len(target_shape):
            target_shape.extend([1] * (max_v - len(target_shape) + 1))
            val = val.reshape(val.shape + (1,) * (max_v - len(val.shape) + 1))
            
        for d in st.dims:
            if d.val_dim is not None:
                target_shape[d.val_dim] = max(target_shape[d.val_dim], d.size)
            if isinstance(d, SparseDimension) and d.block_val_dim is not None:
                target_shape[d.block_val_dim] = max(
                    target_shape[d.block_val_dim], d.block_size or 1
                )
        if tuple(target_shape) != val.shape:
            val = jnp.broadcast_to(val, tuple(target_shape))

    new_dims = list(st.dims)
    dims_to_append = []
    current_ndim = val.ndim
    sparse_pair_val_dim_map = {}

    for i in range(len(new_dims)):
        d = new_dims[i]
        if i in implicit and d.val_dim is None:
            if isinstance(d, SparseDimension):
                pair_key = tuple(sorted((d.id, d.other_id)))
                if pair_key in sparse_pair_val_dim_map:
                    new_val_idx = sparse_pair_val_dim_map[pair_key]
                else:
                    new_val_idx = current_ndim + len(dims_to_append)
                    dims_to_append.append(d.size)
                    sparse_pair_val_dim_map[pair_key] = new_val_idx
            else:
                new_val_idx = current_ndim + len(dims_to_append)
                dims_to_append.append(d.size)
            new_dims[i] = replace(new_dims[i], val_dim=new_val_idx)

    for i in range(len(new_dims)):
        d = new_dims[i]
        if i in implicit and isinstance(d, SparseDimension):
            if d.block_val_dim is None and d.block_size is not None:
                new_block_idx = current_ndim + len(dims_to_append)
                dims_to_append.append(d.block_size)
                new_dims[i] = replace(new_dims[i], block_val_dim=new_block_idx)

    if dims_to_append:
        val = lax.broadcast_in_dim(
            val, val.shape + tuple(dims_to_append), tuple(range(current_ndim))
        )

    actual_to_scatter_logical = set()
    for i in indices:
        d = new_dims[i]
        actual_to_scatter_logical.add(i)
        if isinstance(d, SparseDimension):
            actual_to_scatter_logical.add(id_map[d.other_id])

    phys_to_scatter = sorted(
        list(
            {
                new_dims[i].val_dim
                for i in actual_to_scatter_logical
                if isinstance(new_dims[i], SparseDimension)
                and new_dims[i].val_dim is not None
            }
        )
    )

    val, res_dims = _apply_dense_scattering(
        val, new_dims, actual_to_scatter_logical, phys_to_scatter
    )

    return _create_optimized(
        tuple(res_dims[: len(st.out_dims)]),
        tuple(res_dims[len(st.out_dims) :]),
        val,
        st.scalar_mult,
    )


def _apply_dense_scattering(
    val: Array,
    new_dims: list[Dimension],
    actual_to_scatter_logical: set,
    phys_to_scatter: list[int],
) -> tuple[Array, list[Dimension]]:
    if phys_to_scatter:
        other_phys = [i for i in range(val.ndim) if i not in phys_to_scatter]
        val = val.transpose(phys_to_scatter + other_phys)

        k = len(phys_to_scatter)
        sparse_shape = val.shape[:k]
        dense_part_shape = val.shape[k:]

        val = val.reshape((-1,) + dense_part_shape)
        val = _densify_diagonal_scatter(val)
        val = val.reshape(sparse_shape + sparse_shape + dense_part_shape)

        phys_map = {old: (i, k + i) for i, old in enumerate(phys_to_scatter)}
        for i, old in enumerate(other_phys):
            phys_map[old] = 2 * k + i
    else:
        phys_map = {i: i for i in range(val.ndim)}

    all_phys_axes = list(range(val.ndim))
    final_val_perm = []
    final_val_shape = []

    active_axes = set()

    curr_final_idx = 0
    visited_pairs_scattered = set()
    visited_pairs_sparse = set()
    res_dims = []
    phys_to_final = {}

    for i, d in enumerate(new_dims):
        if i in actual_to_scatter_logical and isinstance(d, SparseDimension):
            pair_key = tuple(sorted((d.id, d.other_id)))
            is_col = pair_key in visited_pairs_scattered
            visited_pairs_scattered.add(pair_key)

            row_idx, col_idx = phys_map[d.val_dim]
            v_idx = col_idx if is_col else row_idx

            final_val_perm.append(v_idx)
            active_axes.add(v_idx)
            logical_size = d.size
            if d.block_val_dim is not None:
                bv_idx = phys_map[d.block_val_dim]
                final_val_perm.append(bv_idx)
                active_axes.add(bv_idx)
                logical_size *= d.block_size

            final_val_shape.append(logical_size)
            res_dims.append(DenseDimension(len(res_dims), logical_size, curr_final_idx))
            curr_final_idx += 1
        else:
            if isinstance(d, SparseDimension):
                pair_key = tuple(sorted((d.id, d.other_id)))
                if pair_key not in visited_pairs_sparse:
                    if d.val_dim is not None:
                        v_idx = phys_map[d.val_dim]
                        final_val_perm.append(v_idx)
                        active_axes.add(v_idx)
                        final_val_shape.append(d.size)
                        phys_to_final[d.val_dim] = curr_final_idx
                        curr_final_idx += 1
                    visited_pairs_sparse.add(pair_key)

                nv = phys_to_final.get(d.val_dim) if d.val_dim is not None else None
                nb = None
                if d.block_val_dim is not None:
                    bv_idx = phys_map[d.block_val_dim]
                    final_val_perm.append(bv_idx)
                    active_axes.add(bv_idx)
                    final_val_shape.append(d.block_size)
                    nb = curr_final_idx
                    curr_final_idx += 1
                res_dims.append(replace(d, id=len(res_dims), val_dim=nv, block_val_dim=nb))
            else:
                nv = None
                if d.val_dim is not None:
                    v_idx = phys_map[d.val_dim]
                    final_val_perm.append(v_idx)
                    active_axes.add(v_idx)
                    final_val_shape.append(d.size)
                    nv = curr_final_idx
                    curr_final_idx += 1
                res_dims.append(replace(d, val_dim=nv))

    # Ensure unique axes for transposition
    seen_perm = set()
    unique_perm = []
    for p in final_val_perm:
        if p not in seen_perm and p < val.ndim:
            unique_perm.append(p)
            seen_perm.add(p)
    
    remaining_axes = [ax for ax in range(val.ndim) if ax not in seen_perm]
    
    # Use the ordered unique perm list
    val = val.transpose(tuple(unique_perm) + tuple(remaining_axes))
    
    # The tracked portion shape should only include unique physical axes
    tracked_portion_shape = []
    seen_phys = set()
    for p, s in zip(final_val_perm, final_val_shape):
        if p not in seen_phys and p < val.ndim:
            tracked_portion_shape.append(s)
            seen_phys.add(p)
    
    # The batch portion is any hidden vmap dimensions
    batch_portion_shape = tuple(val.shape[len(unique_perm):])
    
    total_new_shape = tuple(tracked_portion_shape) + batch_portion_shape
    val = val.reshape(total_new_shape)
    
    # Now transpose back to put the "tracked" dimensions in their original logical slots?
    # No, _apply_dense_scattering returns val and res_dims to _create_optimized.
    # _create_optimized expects the tracked dimensions to come first in val?
    # Actually, it depends on val_dim.

    return val, res_dims

    return _create_optimized(
        tuple(res_dims[: len(st.out_dims)]),
        tuple(res_dims[len(st.out_dims) :]),
        val,
        st.scalar_mult,
    )


def _densify_diagonal_scatter(val: Array) -> Array:
    B = val.shape[0]
    dense_part_shape = val.shape[1:]
    out_shape = (B, B) + dense_part_shape
    out = jnp.zeros(out_shape, dtype=val.dtype)
    idx = jnp.arange(B)
    out = out.at[idx, idx].set(val)
    return out


def _align_to_block(pure_st, block_st):
    pure_val = pure_st.val
    if pure_val is None:
        sparse_dims = [
            d
            for d in pure_st.dims
            if isinstance(d, SparseDimension) and d.val_dim is not None
        ]
        if sparse_dims:
            pure_val = jnp.ones(tuple(d.size for d in sparse_dims), dtype=pure_st.dtype)
        else:
            pure_val = jnp.array(1.0, dtype=pure_st.dtype)
    pure_val = pure_val * pure_st.scalar_mult

    target_shape = []
    block_info = []
    for d in block_st.dims:
        if isinstance(d, SparseDimension):
            if d.val_dim is not None:
                target_shape.append(d.size)
            if d.block_val_dim is not None:
                block_info.append((len(target_shape), d.block_size, d.val_dim))
                target_shape.append(d.block_size)
        elif isinstance(d, DenseDimension) and d.val_dim is not None:
            target_shape.append(d.size)

    if not block_info:
        return jnp.broadcast_to(pure_val.reshape(target_shape), target_shape)

    reshape_target = []
    for i, s in enumerate(target_shape):
        is_block_ax = any(bax == i for bax, _, _ in block_info)
        if not is_block_ax:
            reshape_target.append(s)

    try:
        pure_val = pure_val.reshape(reshape_target)
    except Exception:
        pure_val = jnp.broadcast_to(pure_val.ravel(), reshape_target)

    for bax, bs, vax in sorted(block_info, key=lambda x: x[0]):
        pure_val = jnp.expand_dims(pure_val, axis=bax)
        broad_shape = list(pure_val.shape)
        broad_shape[bax] = bs
        pure_val = jnp.broadcast_to(pure_val, broad_shape)

        if vax is not None and vax < pure_val.ndim and pure_val.shape[vax] == bs:
            eye = jnp.eye(bs, dtype=pure_val.dtype)
            eye_shape = [1] * pure_val.ndim
            eye_shape[vax] = bs
            eye_shape[bax] = bs
            pure_val = pure_val * eye.reshape(eye_shape)

    return pure_val


def _dims_structurally_equal(ld, rd):
    if type(ld) != type(rd):
        return False
    if ld.size != rd.size:
        return False
    if isinstance(ld, SparseDimension):
        return ld.block_size == rd.block_size
    return True


def _get_physical_val(st: SparseTensor) -> Array:
    """Expands implicit dimensions in `val` to their effective physical shape."""
    if st.val is None:
        return jnp.array(st.scalar_mult)

    target_shape = list(st.val.shape)
    # Ensure target_shape is large enough for all val_dims
    max_v = -1
    for d in st.dims:
        if d.val_dim is not None: max_v = max(max_v, d.val_dim)
        if isinstance(d, SparseDimension) and getattr(d, 'block_val_dim', None) is not None:
            max_v = max(max_v, d.block_val_dim)
            
    val = st.val
    if max_v >= len(target_shape):
        target_shape.extend([1] * (max_v - len(target_shape) + 1))
        val = val.reshape(val.shape + (1,) * (max_v - len(val.shape) + 1))

    for d in st.dims:
        if d.val_dim is not None:
            target_shape[d.val_dim] = max(target_shape[d.val_dim], d.size)
        if (
            isinstance(d, SparseDimension)
            and getattr(d, "block_val_dim", None) is not None
        ):
            target_shape[d.block_val_dim] = max(
                target_shape[d.block_val_dim], d.block_size or 1
            )

    val = (
        jnp.broadcast_to(val, target_shape)
        if tuple(target_shape) != val.shape
        else val
    )
    return val * st.scalar_mult


def _get_sparse_indices(st: SparseTensor) -> tuple:
    """Generates JAX advanced indexing arrays mapping physical layout to logical target shape."""
    if st.val is None:
        return ()

    logical_indices = []

    for d in st.dims:
        if isinstance(d, SparseDimension):
            base_idx = jnp.arange(d.size)
            if getattr(d, "block_size", None) is not None:
                base_idx = (
                    base_idx[:, None] * d.block_size + jnp.arange(d.block_size)[None, :]
                )

            target_view = [1] * st.val.ndim
            if d.val_dim is not None:
                target_view[d.val_dim] = d.size
            if getattr(d, "block_val_dim", None) is not None:
                target_view[d.block_val_dim] = d.block_size

            logical_indices.append(base_idx.reshape(target_view))

        elif isinstance(d, DenseDimension):
            idx = jnp.arange(d.size)
            target_view = [1] * st.val.ndim
            if d.val_dim is not None:
                target_view[d.val_dim] = d.size
            logical_indices.append(idx.reshape(target_view))

    return tuple(logical_indices)


def _elementwise(lhs, rhs, op):
    """Applies elementwise operations utilizing scatter/gather for sparse topological efficiency."""
    if not isinstance(lhs, SparseTensor) and not isinstance(rhs, SparseTensor):
        return _arr2st(op(jnp.array(lhs), jnp.array(rhs)))

    l_shape = lhs.shape if isinstance(lhs, SparseTensor) else jnp.shape(lhs)
    r_shape = rhs.shape if isinstance(rhs, SparseTensor) else jnp.shape(rhs)

    if l_shape != r_shape:
        try:
            out_shape = jnp.broadcast_shapes(l_shape, r_shape)
        except ValueError:
            raise ValueError(
                f"Shapes {l_shape} and {r_shape} not compatible for elementwise op"
            )
    else:
        out_shape = l_shape

    if getattr(lhs, "val", False) is None and getattr(rhs, "val", False) is None:
        return _copy(lhs, scalar_mult=op(lhs.scalar_mult, rhs.scalar_mult))

    if isinstance(lhs, SparseTensor) and isinstance(rhs, SparseTensor):
        if len(lhs.dims) == len(rhs.dims) and all(
            _dims_structurally_equal(ld, rd) for ld, rd in zip(lhs.dims, rhs.dims)
        ):
            l_val = (
                (lhs.val * lhs.scalar_mult) if lhs.val is not None else lhs.scalar_mult
            )
            r_val = (
                (rhs.val * rhs.scalar_mult) if rhs.val is not None else rhs.scalar_mult
            )
            new_val = op(l_val, r_val)
            return _copy(lhs, val=new_val, scalar_mult=jnp.array(1.0))

    dtype = jnp.result_type(
        getattr(lhs, "dtype", jnp.float32), getattr(rhs, "dtype", jnp.float32)
    )
    if op is getattr(jnp, "add", None) or op.__name__ == "add":
        keep_l_sparsity = False
        keep_r_sparsity = False
    elif op is getattr(jnp, "multiply", None) or op.__name__ == "multiply":
        keep_l_sparsity = True
        keep_r_sparsity = True
    else:
        import numpy as np

        try:
            z_n = np.zeros((), dtype=np.dtype(dtype))
            o_n = np.ones((), dtype=np.dtype(dtype))
            keep_l_sparsity = bool(np.all(np.asarray(op(z_n, o_n)) == z_n))
            keep_r_sparsity = bool(np.all(np.asarray(op(o_n, z_n)) == z_n))
        except Exception:
            keep_l_sparsity = False
            keep_r_sparsity = False

    if keep_l_sparsity and keep_r_sparsity:
        if getattr(lhs, "val", False) is None:
            return _copy(
                rhs,
                val=op(lhs.scalar_mult, _get_physical_val(rhs)),
                scalar_mult=jnp.array(1.0),
            )
        if getattr(rhs, "val", False) is None:
            return _copy(
                lhs,
                val=op(_get_physical_val(lhs), rhs.scalar_mult),
                scalar_mult=jnp.array(1.0),
            )

        template, other = (lhs, rhs) if isinstance(lhs, SparseTensor) else (rhs, lhs)
        if template.shape == out_shape:
            idx = _get_sparse_indices(template)
            other_dense = jnp.broadcast_to(jnp.array(other), template.shape)

            gathered_other = other_dense[idx] if idx else other_dense
            new_val = op(_get_physical_val(template), gathered_other)
            return _copy(template, val=new_val, scalar_mult=jnp.array(1.0))
        else:
            # If template needs to be broadcasted, we fall back to dense for now
            res_dense = op(jnp.array(lhs), jnp.array(rhs))
            # We try to preserve the split between out and primal dims if possible
            if isinstance(lhs, SparseTensor):
                out_ndim = len(lhs.out_dims)
            elif isinstance(rhs, SparseTensor):
                out_ndim = len(rhs.out_dims)
            else:
                out_ndim = res_dense.ndim // 2
            return _arr2st(res_dense, out_ndim=out_ndim)

    elif keep_l_sparsity and isinstance(lhs, SparseTensor):
        idx = _get_sparse_indices(lhs)
        r_dense = jnp.broadcast_to(jnp.array(rhs), l_shape)
        new_val = op(_get_physical_val(lhs), r_dense[idx] if idx else r_dense)
        return _copy(lhs, val=new_val, scalar_mult=jnp.array(1.0))

    elif keep_r_sparsity and isinstance(rhs, SparseTensor):
        idx = _get_sparse_indices(rhs)
        l_dense = jnp.broadcast_to(jnp.array(lhs), r_shape)
        new_val = op(l_dense[idx] if idx else l_dense, _get_physical_val(rhs))
        return _copy(rhs, val=new_val, scalar_mult=jnp.array(1.0))

    else:
        if isinstance(lhs, SparseTensor) and not isinstance(rhs, SparseTensor):
            base = jnp.broadcast_to(jnp.array(rhs), out_shape).copy()
            idx = _get_sparse_indices(lhs)
            sparse_data = _get_physical_val(lhs)

            if hasattr(base.at[idx], op.__name__):
                res = getattr(base.at[idx], op.__name__)(sparse_data)
            else:
                res = base.at[idx].set(op(sparse_data, base[idx]))
            return _arr2st(res)

        elif isinstance(rhs, SparseTensor) and not isinstance(lhs, SparseTensor):
            base = jnp.broadcast_to(jnp.array(lhs), out_shape).copy()
            idx = _get_sparse_indices(rhs)
            sparse_data = _get_physical_val(rhs)

            if hasattr(base.at[idx], op.__name__):
                res = getattr(base.at[idx], op.__name__)(sparse_data)
            else:
                res = base.at[idx].set(op(base[idx], sparse_data))
            return _arr2st(res)

        else:
            # Both are SparseTensor but structurally different or other complex cases
            # Fall back to dense
            res_dense = op(jnp.array(lhs), jnp.array(rhs))
            # We try to preserve the split between out and primal dims if possible
            if isinstance(lhs, SparseTensor):
                out_ndim = len(lhs.out_dims)
            elif isinstance(rhs, SparseTensor):
                out_ndim = len(rhs.out_dims)
            else:
                out_ndim = res_dense.ndim // 2
            return _arr2st(res_dense, out_ndim=out_ndim)


def _add(lhs, rhs):
    if isinstance(rhs, (int, float)):
        if rhs == 0:
            return lhs
        return _elementwise(lhs, rhs, jnp.add)
    return _elementwise(lhs, rhs, jnp.add)


def _mul(lhs, rhs):
    if isinstance(rhs, (int, float, jax.Array, np.ndarray)):
        s = jnp.asarray(rhs)
        if s.ndim == 0:
            return _create_optimized(
                lhs.out_dims, lhs.primal_dims, lhs.val, lhs.scalar_mult * s
            )
    if isinstance(rhs, SparseTensor):
        return _elementwise(lhs, rhs, jnp.multiply)
    s = jnp.asarray(rhs)
    if s.ndim == 0:
        return _create_optimized(
            lhs.out_dims, lhs.primal_dims, lhs.val, lhs.scalar_mult * s
        )
    return _elementwise(lhs, rhs, jnp.multiply)


def _sort_val(
    out_dims: Sequence[Dimension],
    primal_dims: Sequence[Dimension],
    val: Array | None,
) -> tuple[list[Dimension], list[Dimension], Array | None]:
    if val is None:
        return list(out_dims), list(primal_dims), None

    dims = list(chain(out_dims, primal_dims))

    sparse_axis_map = {}
    dense_axis_map = {}

    c_sparse = count()
    sparse_perm = []

    for d in out_dims:
        if isinstance(d, SparseDimension) and d.val_dim is not None:
            if d.val_dim not in sparse_axis_map:
                new_idx = next(c_sparse)
                sparse_axis_map[d.val_dim] = new_idx
                sparse_perm.append(d.val_dim)

    for d in primal_dims:
        if isinstance(d, SparseDimension) and d.val_dim is not None:
            if d.val_dim not in sparse_axis_map:
                new_idx = next(c_sparse)
                sparse_axis_map[d.val_dim] = new_idx
                sparse_perm.append(d.val_dim)

    sparse_ndim = len(sparse_axis_map)
    c_dense = count(sparse_ndim)
    dense_perm = []

    for d in dims:
        if isinstance(d, DenseDimension) and d.val_dim is not None:
            if d.val_dim not in dense_axis_map:
                new_idx = next(c_dense)
                dense_axis_map[d.val_dim] = new_idx
                dense_perm.append(d.val_dim)
        if isinstance(d, SparseDimension) and d.block_val_dim is not None:
            if d.block_val_dim not in dense_axis_map:
                new_idx = next(c_dense)
                dense_axis_map[d.block_val_dim] = new_idx
                dense_perm.append(d.block_val_dim)

    def update(ds):
        new_ds = []
        for d in ds:
            if isinstance(d, SparseDimension):
                nv = sparse_axis_map.get(d.val_dim) if d.val_dim is not None else None
                nb = (
                    dense_axis_map.get(d.block_val_dim)
                    if d.block_val_dim is not None
                    else None
                )
                new_ds.append(replace(d, val_dim=nv, block_val_dim=nb))
            else:
                nv = dense_axis_map.get(d.val_dim) if d.val_dim is not None else None
                new_ds.append(replace(d, val_dim=nv))
        return tuple(new_ds)

    new_out = update(out_dims)
    new_primal = update(primal_dims)

    # Any physical axes not mapped should be kept at the end
    all_perm = []
    seen = set()
    for ax in (sparse_perm + dense_perm):
        if ax is not None and ax < val.ndim and ax not in seen:
            all_perm.append(ax)
            seen.add(ax)
            
    unmapped_axes = [i for i in range(val.ndim) if i not in seen]
    final_perm = all_perm + unmapped_axes
    
    # Final check: JAX transpose requires exactly ndim axes
    if len(final_perm) > val.ndim:
        final_perm = final_perm[:val.ndim]
        
    val = val.transpose(final_perm)

    return new_out, new_primal, val


def _assert_sparse_tensor_consistency(st: "SparseTensor"):
    for d in st.dims:
        if isinstance(d, SparseDimension):
            other = st.dims[d.other_id]
            if not isinstance(other, SparseDimension) or other.other_id != d.id or d.size != other.size:
                partner_info = f"SparseDimension id={getattr(other, 'id', 'N/A')} other={getattr(other, 'other_id', 'N/A')} size={getattr(other, 'size', 'N/A')}" if hasattr(other, 'id') else f"Non-SparseDimension type={type(other)}"
                raise ValueError(
                    f"Topology Error: SparseDimension {d.id} index {st.dims.index(d) if d in st.dims else 'N/A'} (size {d.size}) has mismatching partner index {d.other_id}: {partner_info}"
                )


def _is_pure_sparse(dims):
    return all(isinstance(d, SparseDimension) and d.block_size is None for d in dims)


def _is_all_dense(dims):
    return all(isinstance(d, DenseDimension) for d in dims)


def _has_blocks(dims):
    return any(
        isinstance(d, SparseDimension) and d.block_size is not None for d in dims
    )


def _val_shape(dims):
    shape = []
    for d in dims:
        if d.val_dim is not None:
            shape.append(d.size)
        if isinstance(d, SparseDimension) and d.block_val_dim is not None:
            shape.append(d.block_size)
    return shape


def _find_n_contract(l_shape, r_shape, hint=None, max_n=None):
    n = hint if hint is not None else 1
    limit = min(len(l_shape), len(r_shape))
    if max_n is not None:
        limit = min(limit, max_n)
    if n <= limit and l_shape[-n:] == r_shape[:n]:
        return n
    for i in range(limit, 0, -1):
        if l_shape[-i:] == r_shape[:i]:
            return i
    return 1


def _repaired_topology(lhs, rhs):
    l_to_r = {lp.id: ro.id for lp, ro in zip(lhs.primal_dims, rhs.out_dims)}
    r_to_l = {v: k for k, v in l_to_r.items()}
    l_paired = {d.id: d.other_id for d in lhs.dims if isinstance(d, SparseDimension)}
    r_paired = {d.id: d.other_id for d in rhs.dims if isinstance(d, SparseDimension)}

    id_map, cid = {}, 0
    for d in lhs.out_dims:
        id_map[(True, d.id)] = cid
        cid += 1
    for d in rhs.primal_dims:
        id_map[(False, d.id)] = cid
        cid += 1

    def remap(d, is_lhs):
        nid = id_map[(is_lhs, d.id)]
        other = getattr(d, "other_id", None)
        if other is not None:
            contract_map = l_to_r if is_lhs else r_to_l
            paired = r_paired if is_lhs else l_paired
            mapped = contract_map.get(other)
            if mapped is not None:
                target = paired.get(mapped)
                if target is not None:
                    return replace(
                        d, id=nid, other_id=id_map.get((not is_lhs, target), target)
                    )
        return replace(d, id=nid)

    return (
        tuple(remap(d, True) for d in lhs.out_dims),
        tuple(remap(d, False) for d in rhs.primal_dims),
    )


def _resolve_implicits(lhs, rhs, n_contract):
    if (
        n_contract <= 0
        or n_contract != len(lhs.primal_dims)
        or n_contract != len(rhs.out_dims)
    ):
        return None
    if lhs.val is not None or rhs.val is not None:
        return None

    factor = 1
    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if isinstance(ld, SparseDimension) and isinstance(rd, SparseDimension):
            if ld.size != rd.size:
                return None
            factor *= ld.block_size or 1
        elif isinstance(ld, DenseDimension) and isinstance(rd, DenseDimension):
            if ld.size != rd.size:
                return None
            factor *= ld.size
        else:
            ls = ld.size * (getattr(ld, "block_size", None) or 1)
            rs = rd.size * (getattr(rd, "block_size", None) or 1)
            if ls != rs:
                return None
            factor *= (
                (getattr(ld, "block_size", None) or 1)
                if isinstance(ld, SparseDimension)
                else ld.size
            )

    new_out, new_primal = _repaired_topology(lhs, rhs)
    return _create_optimized(
        new_out,
        new_primal,
        None,
        lhs.scalar_mult * rhs.scalar_mult * factor,
        _matmul_path="resolve_implicits",
    )


def _pure_dense_mul(sparse_st, dense_st, sparse_is_lhs):
    n_sparse = len(dense_st.primal_dims) if sparse_is_lhs else len(dense_st.out_dims)
    if sparse_st.val is None and dense_st.val is None:
        new_val = None
    elif sparse_st.val is None:
        new_val = dense_st.val
    elif dense_st.val is None:
        new_val = sparse_st.val
    else:
        # Broadcasting safety check
        if sparse_is_lhs:
            # sparse_st.val has shape (s1, ..., sk)
            # dense_st.val has shape (d1, ..., dm)
            # We want (s1, ..., sk, 1, ..., 1) * dense_st.val
            # where k is sparse_ndim, m is dense_ndim
            idx = (Ellipsis,) + (None,) * n_sparse
        else:
            # sparse_st.val has shape (s1, ..., sk)
            # dense_st.val has shape (d1, ..., dm)
            # We want dense_st.val * (1, ..., 1, s1, ..., sk)
            idx = (None,) * n_sparse + (Ellipsis,)
            
        try:
            s_val_expanded = sparse_st.val[idx]
            # Verify broadcasting compatibility
            jnp.broadcast_shapes(s_val_expanded.shape, dense_st.val.shape)
            
            new_val = (
                (s_val_expanded * dense_st.val)
                if sparse_is_lhs
                else (dense_st.val * s_val_expanded)
            )
        except (ValueError, TypeError):
            # Fallback if broadcasting fails due to rank/shape mismatch
            return None

    out_dims_src = sparse_st.out_dims if sparse_is_lhs else dense_st.out_dims
    primal_dims_src = dense_st.primal_dims if sparse_is_lhs else sparse_st.primal_dims

    # Preserve original dimensions (including IDs and logical_ids) but update val_dim mapping
    new_out = tuple(
        replace(d, val_dim=(i if new_val is not None else None)) 
        for i, d in enumerate(out_dims_src)
    )
    new_primal = tuple(
        replace(d, val_dim=(i + len(new_out) if new_val is not None else None))
        for i, d in enumerate(primal_dims_src)
    )

    return _create_optimized(
        new_out,
        new_primal,
        new_val,
        sparse_st.scalar_mult * dense_st.scalar_mult,
    )


def _block_pure_scale(block_st, pure_st, block_is_lhs):
    ref_dims = block_st.primal_dims if block_is_lhs else block_st.out_dims
    pure_dims = pure_st.out_dims if block_is_lhs else pure_st.primal_dims

    for bd, pd in zip(ref_dims, pure_dims):
        if not isinstance(bd, SparseDimension):
            return None
        N, B = bd.size, getattr(bd, "block_size", 1) or 1
        if pd.size != N * B:
            return None

    if pure_st.val is None:
        new_val = block_st.val
    else:
        reshape_shape = []
        broadcast_axes = []

        # 1. Map physical axes of pure_st to their contracted logical blocks
        ax_map = {}
        for bd, pd in zip(ref_dims, pure_dims):
            if getattr(pd, "val_dim", None) is not None:
                ax_map[pd.val_dim] = (bd, pd)

        # 2. Map remaining physical axes to uncontracted/batch dimensions
        uncontracted_ref = block_st.out_dims if block_is_lhs else block_st.primal_dims
        uncontracted_pure = pure_st.primal_dims if block_is_lhs else pure_st.out_dims
        for bd, pd in zip(uncontracted_ref, uncontracted_pure):
            if getattr(pd, "val_dim", None) is not None and pd.val_dim not in ax_map:
                ax_map[pd.val_dim] = (bd, pd)

        for ax in range(pure_st.val.ndim):
            if ax in ax_map:
                bd, pd = ax_map[ax]
                N, B = bd.size, getattr(bd, "block_size", 1) or 1
                if B > 1:
                    reshape_shape.extend([N, B])
                    broadcast_axes.extend(
                        [
                            getattr(bd, "val_dim", None),
                            getattr(bd, "block_val_dim", None),
                        ]
                    )
                else:
                    reshape_shape.append(pd.size)
                    broadcast_axes.append(getattr(bd, "val_dim", None))
            else:
                reshape_shape.append(pure_st.val.shape[ax])
                broadcast_axes.append(None)

        val_p = pure_st.val.reshape(reshape_shape)

        active = [ax for ax in broadcast_axes if ax is not None]
        squeeze_axes = tuple(i for i, ax in enumerate(broadcast_axes) if ax is None)

        if squeeze_axes:
            for ax in squeeze_axes:
                if val_p.shape[ax] != 1:
                    return None
            val_p = val_p.squeeze(axis=squeeze_axes)

        if block_st.val is None:
            target_shape = _val_shape(block_st.out_dims) + _val_shape(
                block_st.primal_dims
            )
            if not target_shape:
                new_val = val_p
            else:
                new_val = jnp.ones(
                    target_shape, dtype=block_st.dtype
                ) * lax.broadcast_in_dim(val_p, tuple(target_shape), tuple(active))
        else:
            new_val = block_st.val * lax.broadcast_in_dim(
                val_p, block_st.val.shape, tuple(active)
            )

    path_name = (
        "block_pure_scale_block_lhs" if block_is_lhs else "block_pure_scale_block_rhs"
    )
    return _create_optimized(
        block_st.out_dims,
        block_st.primal_dims,
        new_val,
        block_st.scalar_mult * pure_st.scalar_mult,
        _matmul_path=path_name,
    )


def _optimized_sparse_matmul(
    lhs: SparseTensor, rhs: SparseTensor
) -> SparseTensor | None:
    import math

    can_optimize = True
    pairs_info = []
    l_val_dim, r_val_dim = None, None

    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if not (isinstance(ld, SparseDimension) and isinstance(rd, SparseDimension)):
            can_optimize = False
            break
        if (
            ld.val_dim is None
            or rd.val_dim is None
            or ld.block_size is None
            or rd.block_size is None
            or ld.block_val_dim is None
            or rd.block_val_dim is None
        ):
            can_optimize = False
            break
        if l_val_dim is None:
            l_val_dim = ld.val_dim
        elif l_val_dim != ld.val_dim:
            can_optimize = False
            break
        if r_val_dim is None:
            r_val_dim = rd.val_dim
        elif r_val_dim != rd.val_dim:
            can_optimize = False
            break
        g = math.gcd(ld.block_size, rd.block_size)
        pairs_info.append(
            {
                "ld": ld,
                "rd": rd,
                "g": g,
                "k_L": ld.block_size // g,
                "k_R": rd.block_size // g,
                "L": ld.block_size // g * rd.block_size // g * g,
            }
        )

    if can_optimize:
        K_L = math.prod(p["k_L"] for p in pairs_info)
        K_R = math.prod(p["k_R"] for p in pairs_info)
        N_LHS = lhs.val.shape[l_val_dim]
        N_RHS = rhs.val.shape[r_val_dim]
        if N_LHS % K_R != 0 or N_RHS % K_L != 0:
            can_optimize = False
        else:
            N_lcm = N_LHS // K_R
            if N_lcm != N_RHS // K_L:
                can_optimize = False

    if can_optimize:
        try:
            l_shape_e = [[s] for s in lhs.val.shape]
            l_shape_e[l_val_dim] = [N_lcm] + [p["k_R"] for p in pairs_info]
            for p in pairs_info:
                l_shape_e[p["ld"].block_val_dim] = [p["k_L"], p["g"]]

            r_shape_e = [[s] for s in rhs.val.shape]
            r_shape_e[r_val_dim] = [N_lcm] + [p["k_L"] for p in pairs_info]
            for p in pairs_info:
                r_shape_e[p["rd"].block_val_dim] = [p["k_R"], p["g"]]

            def flat(sh):
                return tuple(x for s in sh for x in (s if isinstance(s, list) else [s]))

            def ax_map(sh):
                m, off = {}, 0
                for i, s in enumerate(sh):
                    n = len(s) if isinstance(s, list) else 1
                    m[i] = list(range(off, off + n))
                    off += n
                return m

            l_val = lhs.val.reshape(flat(l_shape_e))
            r_val = rhs.val.reshape(flat(r_shape_e))
            la, ra = ax_map(l_shape_e), ax_map(r_shape_e)

            l_batch, r_batch = [la[l_val_dim][0]], [ra[r_val_dim][0]]
            l_contract, r_contract = [], []
            for i, p in enumerate(pairs_info):
                l_batch.append(la[l_val_dim][1 + i])
                r_batch.append(ra[p["rd"].block_val_dim][0])
                l_batch.append(la[p["ld"].block_val_dim][0])
                r_batch.append(ra[r_val_dim][1 + i])
                l_contract.append(la[p["ld"].block_val_dim][1])
                r_contract.append(ra[p["rd"].block_val_dim][1])

            dn = (
                (tuple(l_contract), tuple(r_contract)),
                (tuple(l_batch), tuple(r_batch)),
            )
            dg_out = lax.dot_general(l_val, r_val, dimension_numbers=dn)
            new_scalar = lhs.scalar_mult * rhs.scalar_mult

            l_paired = {
                d.id: getattr(d, "other_id", None)
                for d in lhs.dims
                if isinstance(d, SparseDimension)
            }
            r_paired = {
                d.id: getattr(d, "other_id", None)
                for d in rhs.dims
                if isinstance(d, SparseDimension)
            }

            l_out_to_pair = {}
            for i, info in enumerate(pairs_info):
                lo_id = l_paired.get(info["ld"].id)
                if lo_id is not None:
                    lo = next((d for d in lhs.out_dims if d.id == lo_id), None)
                    if lo and getattr(lo, "block_val_dim", None) is not None:
                        l_out_to_pair[lo.block_val_dim] = i

            r_in_to_pair = {}
            for i, info in enumerate(pairs_info):
                ri_id = r_paired.get(info["rd"].id)
                if ri_id is not None:
                    ri = next((d for d in rhs.primal_dims if d.id == ri_id), None)
                    if ri and getattr(ri, "block_val_dim", None) is not None:
                        r_in_to_pair[ri.block_val_dim] = i

            dg_axes = {"N": 0}
            curr = 1
            for i in range(len(pairs_info)):
                dg_axes[f"kR{i}"] = curr
                curr += 1
                dg_axes[f"kL{i}"] = curr
                curr += 1

            contracted_bvds_l = {p["ld"].block_val_dim for p in pairs_info}
            contracted_bvds_r = {p["rd"].block_val_dim for p in pairs_info}

            for ax in range(lhs.val.ndim):
                if ax != l_val_dim and ax not in contracted_bvds_l:
                    dg_axes[f"lr{ax}"] = curr
                    curr += 1
            for ax in range(rhs.val.ndim):
                if ax != r_val_dim and ax not in contracted_bvds_r:
                    dg_axes[f"rr{ax}"] = curr
                    curr += 1

            perm, shape = [dg_axes["N"]], [N_lcm]
            l_new_phys, out_pair_phys = {}, {}
            cp = 1

            for ax in range(lhs.val.ndim):
                if ax == l_val_dim or ax in contracted_bvds_l:
                    continue
                if ax in l_out_to_pair:
                    i = l_out_to_pair[ax]
                    perm.extend([dg_axes[f"kR{i}"], dg_axes[f"lr{ax}"]])
                    shape.append(pairs_info[i]["k_R"] * lhs.val.shape[ax])
                    out_pair_phys[i] = cp
                else:
                    perm.append(dg_axes[f"lr{ax}"])
                    shape.append(lhs.val.shape[ax])
                l_new_phys[ax] = cp
                cp += 1

            l_pair_new_bphys = {}
            for i, p in enumerate(pairs_info):
                lo_id = l_paired.get(p["ld"].id)
                if lo_id is not None:
                    lo = next((d for d in lhs.out_dims if d.id == lo_id), None)
                    if lo and getattr(lo, "block_val_dim", None) is None:
                        perm.append(dg_axes[f"kR{i}"])
                        shape.append(p["k_R"])
                        l_pair_new_bphys[i] = cp
                        cp += 1

            r_new_phys, in_pair_phys = {}, {}
            for ax in range(rhs.val.ndim):
                if ax == r_val_dim or ax in contracted_bvds_r:
                    continue
                if ax in r_in_to_pair:
                    i = r_in_to_pair[ax]
                    perm.extend([dg_axes[f"kL{i}"], dg_axes[f"rr{ax}"]])
                    shape.append(pairs_info[i]["k_L"] * rhs.val.shape[ax])
                    in_pair_phys[i] = cp
                else:
                    perm.append(dg_axes[f"rr{ax}"])
                    shape.append(rhs.val.shape[ax])
                r_new_phys[ax] = cp
                cp += 1

            r_pair_new_bphys = {}
            for i, p in enumerate(pairs_info):
                ri_id = r_paired.get(p["rd"].id)
                if ri_id is not None:
                    ri = next((d for d in rhs.primal_dims if d.id == ri_id), None)
                    if ri and getattr(ri, "block_val_dim", None) is None:
                        perm.append(dg_axes[f"kL{i}"])
                        shape.append(p["k_L"])
                        r_pair_new_bphys[i] = cp
                        cp += 1

            for i, p in enumerate(pairs_info):
                if i not in out_pair_phys and i not in l_pair_new_bphys:
                    perm.append(dg_axes[f"kR{i}"])
                    shape.append(p["k_R"])
                if i not in in_pair_phys and i not in r_pair_new_bphys:
                    perm.append(dg_axes[f"kL{i}"])
                    shape.append(p["k_L"])

            new_val = dg_out.transpose(perm).reshape(shape)

            l_to_r_c = {ld.id: rd.id for ld, rd in zip(lhs.primal_dims, rhs.out_dims)}
            r_to_l_c = {v: k for k, v in l_to_r_c.items()}
            nid = 0
            old2new = {}
            for d in lhs.out_dims:
                old2new[(True, d.id)] = nid
                nid += 1
            for d in rhs.primal_dims:
                old2new[(False, d.id)] = nid
                nid += 1

            def update_dim(d, is_lhs):
                did = old2new[(is_lhs, d.id)]
                if isinstance(d, SparseDimension):
                    vd = d.val_dim
                    if vd is not None:
                        nv = (
                            0
                            if vd == (l_val_dim if is_lhs else r_val_dim)
                            else (l_new_phys if is_lhs else r_new_phys).get(vd)
                        )
                    else:
                        nv = None

                    nb, nsz, nbs = None, d.size, d.block_size
                    phys_map = l_new_phys if is_lhs else r_new_phys
                    pair_to_pair = l_out_to_pair if is_lhs else r_in_to_pair
                    pair_phys = out_pair_phys if is_lhs else in_pair_phys
                    pair_new_bp = l_pair_new_bphys if is_lhs else r_pair_new_bphys
                    paired_map = l_paired if is_lhs else r_paired
                    contract_map = l_to_r_c if is_lhs else r_to_l_c
                    k_key = "k_R" if is_lhs else "k_L"

                    bvd = getattr(d, "block_val_dim", None)
                    if bvd in pair_to_pair:
                        i = pair_to_pair[bvd]
                        nb = pair_phys[i]
                        nsz, nbs = N_lcm, pairs_info[i][k_key] * (d.block_size or 1)
                    elif d.id in paired_map and paired_map[d.id] in contract_map:
                        i = next(
                            idx
                            for idx, info in enumerate(pairs_info)
                            if info[("ld" if is_lhs else "rd")].id == paired_map[d.id]
                        )
                        if i in pair_new_bp:
                            nb = pair_new_bp[i]
                            nsz, nbs = N_lcm, pairs_info[i][k_key] * (d.block_size or 1)
                    elif bvd is not None:
                        nb = phys_map.get(bvd)

                    other = getattr(d, "other_id", None)
                    if is_lhs and other in l_to_r_c:
                        nother = old2new[(False, r_paired[l_to_r_c[other]])]
                    elif not is_lhs and other in r_to_l_c:
                        nother = old2new[(True, l_paired[r_to_l_c[other]])]
                    else:
                        nother = old2new.get((is_lhs, other), other)

                    return replace(
                        d,
                        id=did,
                        size=nsz,
                        block_size=nbs,
                        val_dim=nv,
                        block_val_dim=nb,
                        other_id=nother,
                    )
                else:
                    nv = (
                        (l_new_phys if is_lhs else r_new_phys).get(d.val_dim)
                        if d.val_dim is not None
                        else None
                    )
                    return replace(d, id=did, val_dim=nv)

            new_out = tuple(update_dim(d, True) for d in lhs.out_dims)
            new_primal = tuple(update_dim(d, False) for d in rhs.primal_dims)

            pvd = {}
            for d in chain(new_out, new_primal):
                if isinstance(d, SparseDimension) and d.val_dim is not None:
                    pvd.setdefault(frozenset({d.id, d.other_id}), d.val_dim)

            def sync(d):
                if isinstance(d, SparseDimension):
                    k = frozenset({d.id, d.other_id})
                    if k in pvd:
                        return replace(d, val_dim=pvd[k])
                return d

            return _create_optimized(
                tuple(sync(d) for d in new_out),
                tuple(sync(d) for d in new_primal),
                new_val,
                new_scalar,
            )
        except (KeyError, ValueError, TypeError):
            pass
    return None


def _matmul(lhs: Any, rhs: Any) -> SparseTensor | Array:
    import math

    if not isinstance(lhs, SparseTensor) and not isinstance(rhs, SparseTensor):
        return jnp.matmul(lhs, rhs)

    l_shape = lhs.shape if isinstance(lhs, SparseTensor) else jnp.shape(lhs)
    r_shape = rhs.shape if isinstance(rhs, SparseTensor) else jnp.shape(rhs)

    if not isinstance(lhs, SparseTensor):
        n_contract = _find_n_contract(l_shape, r_shape, hint=len(rhs.out_dims))
        lhs = _arr2st(lhs, out_ndim=len(l_shape) - n_contract)
    elif not isinstance(rhs, SparseTensor):
        n_contract = _find_n_contract(
            l_shape, r_shape, hint=len(lhs.primal_dims), max_n=len(lhs.primal_dims)
        )
        rhs = _arr2st(rhs, out_ndim=n_contract)
    else:
        n_contract = _find_n_contract(l_shape, r_shape, hint=len(lhs.primal_dims))
        if n_contract != len(lhs.primal_dims):
            pass

    both_st = isinstance(lhs, SparseTensor) and isinstance(rhs, SparseTensor)
    matched = (
        both_st
        and n_contract > 0
        and n_contract == len(lhs.primal_dims)
        and n_contract == len(rhs.out_dims)
    )

    if both_st:
        result = _resolve_implicits(lhs, rhs, n_contract)
        if result is not None:
            return result

    if matched:
        if _is_pure_sparse(lhs.dims) and _is_all_dense(rhs.dims):
            result = _pure_dense_mul(lhs, rhs, sparse_is_lhs=True)
            return result
        if _is_all_dense(lhs.dims) and _is_pure_sparse(rhs.dims):
            result = _pure_dense_mul(rhs, lhs, sparse_is_lhs=False)
            return result

        if _has_blocks(lhs.dims) and _is_pure_sparse(rhs.dims):
            result = _block_pure_scale(lhs, rhs, block_is_lhs=True)
            if result is not None:
                return result
        elif _has_blocks(rhs.dims) and _is_pure_sparse(lhs.dims):
            result = _block_pure_scale(rhs, lhs, block_is_lhs=False)
            if result is not None:
                return result

    if both_st and lhs.val is not None and rhs.val is not None and matched:
        result = _optimized_sparse_matmul(lhs, rhs)
        if result is not None:
            return result

    l_dense = jnp.array(lhs)
    r_dense = jnp.array(rhs)
    
    # We use the SparseTensor metadata to find the exact axes to contract and batch
    l_dense = jnp.array(lhs)
    r_dense = jnp.array(rhs)
    
    l_ndim = l_dense.ndim
    r_ndim = r_dense.ndim
    
    # We prioritize logical IDs for perfect alignment. 
    # n_contract derived from ID intersection is more reliable than shape guessing.
    l_primal_lids = [d.logical_id for d in lhs.primal_dims]
    r_out_lids = [d.logical_id for d in rhs.out_dims]
    l_out_lids = [d.logical_id for d in lhs.out_dims]
    
    l_contract = []
    r_contract = []
    l_batch = []
    r_batch = []
    used_l = set()
    used_r = set()
    
    # 1. Matching Batch: lhs.out vs rhs.out (dimensions shared across whole op)
    # Batch axes must be identified first to prevent them being seen as contractions
    for i, lid in enumerate(l_out_lids):
        if lid is not None and lid in r_out_lids:
            j = r_out_lids.index(lid)
            l_ax = lhs.out_dims[i].val_dim
            r_ax = rhs.out_dims[j].val_dim
            if l_ax is not None and r_ax is not None:
                # Robust batching: only align if shapes match
                if l_dense.shape[l_ax] == r_dense.shape[r_ax]:
                    l_batch.append(l_ax)
                    r_batch.append(r_ax)
                    used_l.add(l_ax)
                    used_r.add(r_ax)
    
    # 2. Matching Contraction: lhs.primal vs rhs.out using logical identity
    for i, lid in enumerate(l_primal_lids):
        if lid is not None and lid in r_out_lids:
            j = r_out_lids.index(lid)
            l_ax = lhs.primal_dims[i].val_dim
            r_ax = rhs.out_dims[j].val_dim
            if l_ax is not None and r_ax is not None:
                # Ensure we don't contract over a dimension already used for batching
                if l_ax not in used_l and r_ax not in used_r:
                    l_contract.append(l_ax)
                    r_contract.append(r_ax)
                    used_l.add(l_ax)
                    used_r.add(r_ax)

    # Use found IDs if available, otherwise fall back to hints/shapes
    if len(l_contract) > 0:
        actual_n_contract = len(l_contract)
    else:
        # If ID matching failed, don't trust n_contract if it's too high for the geometry
        actual_n_contract = n_contract if n_contract is not None else 1
        if actual_n_contract > min(l_ndim, r_ndim):
            actual_n_contract = 1
            

    match_failed = (len(l_contract) == 0 and actual_n_contract > 0)
    if not match_failed:
        for la, ra in zip(l_contract, r_contract):
            if l_dense.shape[la] != r_dense.shape[ra]:
                match_failed = True; break
    
    if match_failed:
        # Fall back to size-based heuristic only if ID matching failed
        l_contract = []
        r_contract = []
        # Find exactly ONE axis that matches if we are desperate
        l_candidates = list(range(l_ndim - actual_n_contract, l_ndim))
        used_r = set()
        for l_ax in l_candidates:
            target_size = l_dense.shape[l_ax]
            found = False
            # Search RHS for a matching size
            # Start from the end of RHS to find feature dims, avoiding batches at the front
            for r_ax in range(r_ndim - 1, -1, -1):
                if r_ax not in used_r and r_dense.shape[r_ax] == target_size:
                    l_contract.append(l_ax)
                    r_contract.append(r_ax)
                    used_r.add(r_ax)
                    found = True
                    break
            if not found:
                # Last resort: just try matmul which handles batches automatically
                try:
                    res = jnp.matmul(l_dense, r_dense)
                    return _arr2st(res, out_ndim=len(lhs.out_dims))
                except (TypeError, ValueError):
                    # Final final fallback: find ANY pair of matching shapes for a 1-axis contraction
                    found_fallback = False
                    for l_i in range(l_ndim - 1, -1, -1):
                        target_size = l_dense.shape[l_i]
                        for r_j in range(r_ndim - 1, -1, -1):
                            if r_dense.shape[r_j] == target_size:
                                l_ax, r_ax = l_i, r_j
                                found_fallback = True
                                break
                        if found_fallback: break
                    
                    if found_fallback:
                        res = jnp.tensordot(l_dense, r_dense, axes=((l_ax,), (r_ax,)))
                        return _arr2st(res, out_ndim=len(lhs.out_dims))
                    else:
                        # Scaling fallback: if one is size 1, it's a scalar multiply
                        if l_dense.size == 1:
                            res = l_dense.reshape(()) * r_dense
                            return _arr2st(res, out_ndim=len(rhs.out_dims))
                        if r_dense.size == 1:
                            res = l_dense * r_dense.reshape(())
                            return _arr2st(res, out_ndim=len(lhs.out_dims))
                            
                        # Re-raise original error if even size-matching fails
                        raise ValueError(f"Geometric Mismatch: No matching dimensions found between LHS {l_dense.shape} and RHS {r_dense.shape}")

    dimension_numbers = ((tuple(l_contract), tuple(r_contract)), (tuple(l_batch), tuple(r_batch)))
    res = jax.lax.dot_general(l_dense, r_dense, dimension_numbers)
    
    # Inherit dimensions from parents to preserve IDs
    # result out_dims = lhs.out_dims + (any lhs.primal_dims not contracted)
    # result primal_dims = (any rhs.out_dims not contracted) + rhs.primal_dims
    
    uncontracted_l_primal = [d for i, d in enumerate(lhs.primal_dims) if i not in l_contract]
    uncontracted_r_out = [d for i, d in enumerate(rhs.out_dims) if i not in r_contract]
    
    new_out_dims = list(lhs.out_dims) + uncontracted_l_primal
    new_primal_dims = uncontracted_r_out + list(rhs.primal_dims)
    
    # Re-map physical val dimensions
    res_out = []
    for i, d in enumerate(new_out_dims):
        res_out.append(replace(d, val_dim=i))
    res_primal = []
    for i, d in enumerate(new_primal_dims):
        res_primal.append(replace(d, val_dim=i + len(res_out)))
        
    return _create_optimized(
        tuple(res_out),
        tuple(res_primal),
        res,
        lhs.scalar_mult * rhs.scalar_mult
    )
# Final absolute safety: Ensure batch and contract are truly disjoint
    # This must happen after ALL logic that might have updated l_contract/r_contract
    l_batch_final, r_batch_final = [], []
    for lb, rb in zip(l_batch, r_batch):
        if lb not in l_contract and rb not in r_contract:
            l_batch_final.append(lb)
            r_batch_final.append(rb)
    l_batch, r_batch = l_batch_final, r_batch_final

    try:
        res = jax.lax.dot_general(
            l_dense, r_dense, 
            dimension_numbers=((tuple(l_contract), tuple(r_contract)), (tuple(l_batch), tuple(r_batch)))
        )
    except (TypeError, ValueError) as e:
        # If dot_general fails, it's usually because axes don't match. 
        # try matmul as a definitive fallback for rank-consistency
        try:
            res = jnp.matmul(l_dense, r_dense)
        except:
            raise e

    out_ndim = len(lhs.out_dims)
    return _arr2st(res, out_ndim=out_ndim)


def _rmatmul(rhs: SparseTensor, lhs: Any) -> SparseTensor:
    return _matmul(lhs, rhs)


def _arr2st(
    arr: Array, out_ndim: int | None = None, _matmul_path: str | None = None
) -> SparseTensor:
    if arr.ndim == 0:
        arr = jnp.expand_dims(arr, 0)
    if out_ndim is None:
        out_ndim = arr.ndim // 2

    dims = tuple(DenseDimension(i, s, i) for i, s in enumerate(arr.shape))
    return _create_optimized(
        dims[:out_ndim], dims[out_ndim:], arr, _matmul_path=_matmul_path
    )


def _transpose(
    st: SparseTensor,
    out_transpose: Sequence[int] | None = None,
    primal_transpose: Sequence[int] | None = None,
) -> SparseTensor:
    n_out = len(st.out_dims)
    n_primal = len(st.primal_dims)
    total_ndim = n_out + n_primal

    if out_transpose is None and primal_transpose is None:
        full_transpose = tuple(range(total_ndim - 1, -1, -1))
        out_transpose = full_transpose[:n_primal]
        primal_transpose = full_transpose[n_primal:]
    else:
        out_transpose = (
            list(out_transpose) if out_transpose is not None else list(range(n_out))
        )
        primal_transpose = (
            list(primal_transpose)
            if primal_transpose is not None
            else list(range(n_out, total_ndim))
        )

    out_transpose = [i % total_ndim for i in out_transpose]
    primal_transpose = [i % total_ndim for i in primal_transpose]
    full_transpose = tuple(out_transpose) + tuple(primal_transpose)

    if len(full_transpose) != total_ndim or len(set(full_transpose)) != total_ndim:
        raise ValueError(
            f"Invalid transpose permutation: {full_transpose} for ndim {total_ndim}"
        )

    id_map = {d.id: i for i, d in enumerate(st.dims)}
    out_set = set(out_transpose)
    primal_set = set(primal_transpose)

    to_densify = []
    for i, d in enumerate(st.dims):
        if isinstance(d, SparseDimension):
            other_pos = id_map[d.other_id]
            if (i in out_set and other_pos in out_set) or (
                i in primal_set and other_pos in primal_set
            ):
                to_densify.append(i)

    if to_densify:
        st = st.dense(axes=tuple(to_densify), hard=True)
        n_out = len(st.out_dims)
        n_primal = len(st.primal_dims)
        total_ndim = n_out + n_primal
        id_map = {d.id: i for i, d in enumerate(st.dims)}

    dims = st.dims

    def update_logical(d, new_idx):
        if isinstance(d, SparseDimension):
            # other_id is an index. we need to know where the dimension that was at d.other_id moved to.
            # full_transpose[new_idx] is the old index.
            # We need the new index of the dimension that was at d.other_id.
            # new_other_id = old_to_new_index[d.other_id]
            return replace(d, id=new_idx, other_id=old_to_new_index[d.other_id])
        return replace(d, id=new_idx)

    old_to_new_index = {old_idx: new_idx for new_idx, old_idx in enumerate(full_transpose)}

    new_out_dims = [update_logical(dims[i], idx) for idx, i in enumerate(out_transpose)]
    new_primal_dims = [
        update_logical(dims[i], len(out_transpose) + idx)
        for idx, i in enumerate(primal_transpose)
    ]

    new_out_dims, new_primal_dims, new_val = _sort_val(
        new_out_dims, new_primal_dims, st.val
    )

    return _create_optimized(
        tuple(new_out_dims), tuple(new_primal_dims), new_val, st.scalar_mult
    )


def sparse_tensor_zeros_like(st: SparseTensor) -> SparseTensor:
    """
    Function that generates a new `SparseTensor` with only zeros with the
    same shape as the given `SparseTensor` `st`.

    Args:
        st (SparseTensor): `SparseTensor` whose shape we want to use to initialize
                            the new `SparseTensor` object.
    Returns:
        SparseTensor:  A copy of a given `SparseTensor` object with only zeros.
    """
    return st.copy(jnp.zeros_like(st.val))


def _materialize_dimensions(st: SparseTensor, dims: Sequence[int]) -> Array:
    """
    Function that materializes the `val` property of a `SparseTensor` object
    along a given set of axes. This is necessary to enable broadcasting multiplication
    of two `SparseTensor` objects where one of them has a `DenseDimension` object
    in its `out_dims` list and the other one has a `SparseDimension` object in
    the corresponding `primal_dims` list or vice versa.


    Args:
        st (SparseTensor): The `SparseTensor` object whose `val` property we want
                            to materialize along the axes given in `dims`.
        dims (Sequence[int]): The axes along which we want to materialize the `val`
                                property of `st`.

    Returns:
        Array: The `val` property of `st` materialized along the axes given in `dims`.
    """
    if len(dims) == 0:
        return st.val
    dims = sorted(dims)  # reverse=True
    # dims = [d if d <= st.val.ndim else -1 for d in dims]
    _dims, counter = [], st.val.ndim
    for d in dims:
        if d <= st.val.ndim:
            _dims.append(d)
            counter += 1
        else:
            _dims.append(counter)
            counter += 1
    return jnp.expand_dims(st.val, axis=_dims)


def _swap_back_axes(st: SparseTensor) -> SparseTensor:
    """
    After two `SparseTensor` objects have been broadcast multiplied, the
    resulting tensor usually has the `val` not reshaped so that the dimensions
    of it are sorted in ascending order according to the order in which the
    corresponding dimensions appear. This function does this.

    Example:
    We might end up with a `SparseTensor` object that looks like
    `out_dims=(SparseDimension(0, 2, 1, 3), DenseDimension(1, 3, 2))`
    `primal_dims=(DenseDimension(2, 4, 0), SparseDimension(3, 2, 1, 0))`
    `val.shape = (4, 2, 3)`
    but we want to have `val.shape = (2, 3, 4)`.
    This function computes the necessary permutation and applies it as a
    `jnp.transpose` to the `val` property.

    Args:
        st (SparseTensor): SparseTensor object whose `val` property we want to
                            swap back around after broadcasting multiplication.

    Returns:
        SparseTensor: SparseTensor object with `val` property with dimensions
                        sorted in ascending order.
    """
    l = len(st.out_dims)
    permutation = []
    seen = set()
    for d in st.out_dims + st.primal_dims:
        if d.val_dim is not None:
            if isinstance(d, DenseDimension):
                if d.val_dim < len(st.val.shape) and d.val_dim not in seen:
                    permutation.append(d.val_dim)
                    seen.add(d.val_dim)
            else:
                if d.id < d.other_id:
                    if d.val_dim < len(st.val.shape) and d.val_dim not in seen:
                        permutation.append(d.val_dim)
                        seen.add(d.val_dim)

    # Fill in any missing axes to ensure a complete permutation
    unmapped = [j for j in range(len(st.val.shape)) if j not in seen]
    final_permutation = permutation + unmapped
    
    if len(final_permutation) != len(st.val.shape):
        final_permutation = final_permutation[:len(st.val.shape)]

    new_val = jnp.transpose(st.val, final_permutation)

    val_dim_map = {}
    i = 0
    for d in st.out_dims + st.primal_dims:
        if d.val_dim is not None:
            if isinstance(d, DenseDimension):
                val_dim_map[(d in st.out_dims, d.id)] = i
                i += 1
            else:
                if d.id < d.other_id:
                    val_dim_map[(d in st.out_dims, d.id)] = i
                    other_is_out = d.other_id < l
                    val_dim_map[(other_is_out, d.other_id)] = i
                    i += 1

    def update_dim(d, is_out):
        new_vd = val_dim_map.get((is_out, d.id))
        return replace(d, val_dim=new_vd)

    new_out_dims = tuple(update_dim(d, True) for d in st.out_dims)
    new_primal_dims = tuple(update_dim(d, False) for d in st.primal_dims)

    return _create_optimized(new_out_dims, new_primal_dims, new_val, st.scalar_mult)


def matmul_fmas(lhs: "SparseTensor", rhs: "SparseTensor") -> int:
    """Analytical calculation of fused multiply-adds (FMAs) for SparseTensor @ SparseTensor."""
    import math
    from math import prod

    # Rule 3: All-implicit matrices require 0 tensor MACs (just scalar bookkeeping)
    if lhs.val is None and rhs.val is None:
        return 0

    n_contract = _find_n_contract(lhs.shape, rhs.shape, hint=len(lhs.primal_dims))

    # Fallback for unexpected shapes (e.g. partial contraction)
    if n_contract != len(lhs.primal_dims) or n_contract != len(rhs.out_dims):
        return prod(lhs.val.shape if lhs.val is not None else (1,)) * prod(
            rhs.val.shape if rhs.val is not None else (1,)
        )

    consumed_lhs_axes = set()
    consumed_rhs_axes = set()
    fmas = 1

    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        is_dense_l = isinstance(ld, DenseDimension)
        is_dense_r = isinstance(rd, DenseDimension)

        # Helper to identify pure identity matrices
        def is_pure_identity(d):
            return (
                isinstance(d, SparseDimension)
                and d.val_dim is None
                and getattr(d, "block_size", None) is None
            )

        # Rule 1: Identity Elimination (0 MACs required)
        if is_pure_identity(ld) or is_pure_identity(rd):
            return 0

        if is_dense_l and is_dense_r:
            if ld.val_dim is not None:
                consumed_lhs_axes.add(ld.val_dim)
            if rd.val_dim is not None:
                consumed_rhs_axes.add(rd.val_dim)
            fmas *= ld.size

        elif not is_dense_l and is_dense_r:
            if ld.val_dim is None:
                return 0  # Identity pass-through
            consumed_lhs_axes.add(ld.val_dim)
            if getattr(ld, "block_val_dim", None) is not None:
                consumed_lhs_axes.add(ld.block_val_dim)
            if rd.val_dim is not None:
                consumed_rhs_axes.add(rd.val_dim)
            fmas *= rd.size  # Element-wise scale

        elif is_dense_l and not is_dense_r:
            if rd.val_dim is None:
                return 0  # Identity pass-through
            if ld.val_dim is not None:
                consumed_lhs_axes.add(ld.val_dim)
            consumed_rhs_axes.add(rd.val_dim)
            if getattr(rd, "block_val_dim", None) is not None:
                consumed_rhs_axes.add(rd.block_val_dim)
            fmas *= ld.size  # Element-wise scale

        else:  # Sparse @ Sparse
            if ld.val_dim is None or rd.val_dim is None:
                return 0  # Identity elimination

            consumed_lhs_axes.add(ld.val_dim)
            consumed_rhs_axes.add(rd.val_dim)

            B_L = ld.block_size or 1
            B_R = rd.block_size or 1
            g = math.gcd(B_L, B_R)

            # Block alignment constraints
            lcm_B = (B_L * B_R) // g
            N_lcm = (ld.size * B_L) // lcm_B
            k_L = B_L // g
            k_R = B_R // g

            b_imp_L = ld.block_val_dim is None
            b_imp_R = rd.block_val_dim is None

            # Rule 2: Ones Contraction
            if b_imp_L and b_imp_R:
                fmas *= N_lcm * k_L * k_R
            elif b_imp_L:
                consumed_rhs_axes.add(rd.block_val_dim)
                fmas *= N_lcm * k_L * k_R * B_R
            elif b_imp_R:
                consumed_lhs_axes.add(ld.block_val_dim)
                fmas *= N_lcm * k_L * k_R * B_L
            else:
                consumed_lhs_axes.add(ld.block_val_dim)
                consumed_rhs_axes.add(rd.block_val_dim)
                fmas *= N_lcm * k_L * k_R * g

    # Multiply by the sizes of the unconsumed (outer/batch) physical axes
    if lhs.val is not None:
        for i, size in enumerate(lhs.val.shape):
            if i not in consumed_lhs_axes:
                fmas *= size

    if rhs.val is not None:
        for i, size in enumerate(rhs.val.shape):
            if i not in consumed_rhs_axes:
                fmas *= size

    return int(fmas)


def elementwise_fmas(
    lhs: "SparseTensor", rhs: "SparseTensor", is_mul: bool = False
) -> int:
    """
    Analytical calculation of mathematical FMAs for mul SparseTensor ops.

    Tracks the active fused multiply-adds (or adds/muls) by mapping the physical
    intersection of both tensors' non-zero topologies.
    """
    if lhs.shape != rhs.shape:
        return 0

    if getattr(lhs, "val", False) is None and getattr(rhs, "val", False) is None:
        return 1

    fmas = 1

    for ld, rd in zip(lhs.dims, rhs.dims):
        is_dense_l = isinstance(ld, DenseDimension)
        is_dense_r = isinstance(rd, DenseDimension)

        # l_implicit = getattr(ld, 'val_dim', None) is None or getattr(ld, 'block_val_dim', None) is None
        # r_implicit = getattr(rd, 'val_dim', None) is None or getattr(rd, 'block_val_dim', None) is None

        if is_dense_l and is_dense_r:
            fmas *= ld.size

        elif is_dense_l != is_dense_r:
            explicit_dim = ld if is_dense_l else rd
            implicit_dim = rd if is_dense_l else ld

            if is_mul and (getattr(implicit_dim, "val_dim", None) is None):
                return 0
            else:
                fmas *= explicit_dim.size
                if getattr(explicit_dim, "block_size", None) is not None:
                    fmas *= explicit_dim.block_size

        else:
            if ld.id != rd.id and getattr(ld, "other_id", None) != rd.id:
                return 0

            if is_mul and (ld.val_dim is None or rd.val_dim is None):
                return 0

            if ld.val_dim is not None or rd.val_dim is not None:
                fmas *= ld.size

            b_l = getattr(ld, "block_size", None)
            b_r = getattr(rd, "block_size", None)

            if b_l is not None or b_r is not None:
                b_imp_l = getattr(ld, "block_val_dim", None) is None
                b_imp_r = getattr(rd, "block_val_dim", None) is None

                if is_mul and (b_imp_l or b_imp_r):
                    return 0  # 1*x block copy

                if not b_imp_l and not b_imp_r:
                    fmas *= min(b_l or 1, b_r or 1)
                elif not b_imp_l:
                    fmas *= b_l
                elif not b_imp_r:
                    fmas *= b_r

    return int(fmas)

def factors(n):
    return tuple(reduce(list.__add__,([i, n//i] for i in range(1, int(n**0.5) + 1) if not n % i)))

@partial(jax.jit, static_argnums=(1, 2))
def apply_generalized_dynamic_sparsity(
    st: SparseTensor,
    grouping_vector: tuple[int, ...],  
    factor_vector: tuple[int, ...] 
) -> SparseTensor:
    """
    grouping_vector: Tuple of length ndim mapping dims to pairing groups (-1 for unpaired).
    factor_vector: Tuple of factors for each group (-1 for gcd, 0 for drop).
    """
    if st.val is None:
        return st

    out_len = len(st.out_dims)
    factors = factor_vector
    
    # 1. Parse groupings and identify paired logical axes
    pairs = defaultdict(list)
    for logical_idx, group_id in enumerate(grouping_vector):
        if group_id >= 0 and group_id < len(factors):
            is_out = logical_idx < out_len
            idx = logical_idx if is_out else logical_idx - out_len
            pairs[group_id].append((is_out, idx))

    valid_pairs = {
        gid: p for gid, p in pairs.items() 
        if len(p) == 2 and factors[gid] != 1
    }

    if not valid_pairs:
        return st

    # 2. Compute static shapes and dimension updates
    new_out_dims = list(st.out_dims)
    new_primal_dims = list(st.primal_dims)
    
    axes_to_front = []
    idx_arrays = []
    
    drop_all = True
    current_val_dim = 0
    val_dim_map = {}

    for gid, ((is_out1, idx1), (is_out2, idx2)) in valid_pairs.items():
        d1 = st.out_dims[idx1] if is_out1 else st.primal_dims[idx1]
        d2 = st.out_dims[idx2] if is_out2 else st.primal_dims[idx2]
        
        factor = factors[gid]
        size = gcd(d1.logical_size, d2.logical_size) if factor == -1 else factor
        
        if factor == 0:
            # Mark for dropping
            if d1.val_dim is not None: val_dim_map[d1.val_dim] = None
            if d2.val_dim is not None: val_dim_map[d2.val_dim] = None
        else:
            drop_all = False
            # Calculate block sizes and queue axes for the multi-diagonal extraction
            b1 = d1.logical_size // size
            b2 = d2.logical_size // size
            
            if d1.val_dim is not None and d2.val_dim is not None:
                axes_to_front.extend([d1.val_dim, d2.val_dim])
                idx_arrays.append(jnp.arange(size))
                
                # Update logical dimensions tracking
                val_dim_map[d1.val_dim] = current_val_dim
                val_dim_map[d2.val_dim] = current_val_dim
                current_val_dim += 1
                
                if b1 > 1 and getattr(d1, 'block_val_dim', None) is not None:
                    val_dim_map[d1.block_val_dim] = current_val_dim
                    current_val_dim += 1
                if b2 > 1 and getattr(d2, 'block_val_dim', None) is not None:
                    val_dim_map[d2.block_val_dim] = current_val_dim
                    current_val_dim += 1

        # Reconstruct SparseDimension objects
        def _update_dim(d, b_size):
            return SparseDimension(
                id=d.id, size=size, val_dim=None if factor == 0 else d.val_dim,
                other_id=d2.id if d is d1 else d1.id,
                block_size=b_size if b_size > 1 else None,
                block_val_dim=None if factor == 0 else getattr(d, 'block_val_dim', None)
            )

        if is_out1: new_out_dims[idx1] = _update_dim(d1, b1)
        else: new_primal_dims[idx1] = _update_dim(d1, b1)
        
        if is_out2: new_out_dims[idx2] = _update_dim(d2, b2)
        else: new_primal_dims[idx2] = _update_dim(d2, b2)

    if drop_all:
        # All factors were 0; tensor becomes entirely implicit
        return _create_optimized(
            tuple(new_out_dims), tuple(new_primal_dims), None, st.scalar_mult
        )

    # 3. Vectorized Multi-Diagonal Extraction
    remaining_axes = [i for i in range(st.val.ndim) if i not in axes_to_front]
    perm = axes_to_front + remaining_axes
    val_transposed = jnp.transpose(st.val, perm)
    
    # Advanced indexing to extract diagonals in one XLA op
    if idx_arrays:
        mesh_indices = jnp.ix_(*idx_arrays) 
        val_diagonal = val_transposed[tuple(idx for pair in mesh_indices for idx in (pair, pair))]
    else:
        val_diagonal = val_transposed

    # 4. Final dimension mapping
    for old_ax in remaining_axes:
        val_dim_map[old_ax] = current_val_dim
        current_val_dim += 1

    def _remap_val_dims(d):
        nv = val_dim_map.get(d.val_dim) if d.val_dim is not None else None
        if isinstance(d, SparseDimension):
            nb = val_dim_map.get(d.block_val_dim) if getattr(d, 'block_val_dim', None) is not None else None
            return replace(d, val_dim=nv, block_val_dim=nb)
        return replace(d, val_dim=nv)

    final_out = tuple(_remap_val_dims(d) for d in new_out_dims)
    final_primal = tuple(_remap_val_dims(d) for d in new_primal_dims)

    res = _create_optimized(final_out, final_primal, val_diagonal, st.scalar_mult)
    jax.debug.print("Result dims: {d}", d=[(type(d).__name__, d.id, getattr(d, 'other_id', None)) for d in res.dims])
    return res

def get_valid_pairings(
    st: SparseTensor, 
    dim_id: int, 
    grouping_vector: tuple[int, ...] | None = None
) -> list[int]:
    """
    Finds valid dimension IDs that can be paired with the given dim_id.
    If grouping_vector is provided, dimensions already paired (not -1) are ignored.
    """
    target_dim = None
    is_out_dim = False
    target_logical_idx = -1
    
    out_len = len(st.out_dims)

    # 1. Locate the dimension and logical index
    for i, d in enumerate(st.out_dims):
        if d.id == dim_id:
            target_dim = d
            is_out_dim = True
            target_logical_idx = i
            break
            
    if target_dim is None:
        for i, d in enumerate(st.primal_dims):
            if d.id == dim_id:
                target_dim = d
                target_logical_idx = out_len + i
                break
                
    if target_dim is None:
        raise ValueError(f"Dimension ID {dim_id} not found in SparseTensor.")

    # If the target itself is already paired, return empty
    if grouping_vector is not None and target_logical_idx < len(grouping_vector):
        if grouping_vector[target_logical_idx] != -1:
            return []

    # 2. Find logical index of a given ID (helper)
    def _get_logical_idx(search_id: int) -> int:
        for i, d in enumerate(st.out_dims):
            if d.id == search_id: return i
        for i, d in enumerate(st.primal_dims):
            if d.id == search_id: return out_len + i
        return -1

    # 3. Identify valid partners
    valid_ids = []

    if isinstance(target_dim, SparseDimension):
        # Sparse dimensions are locked to their other_id
        valid_ids = [target_dim.other_id]
    else:
        # Dense dimensions scan the opposite side
        opposite_dims = st.primal_dims if is_out_dim else st.out_dims
        for d in opposite_dims:
            if isinstance(d, DenseDimension) and gcd(target_dim.logical_size, d.logical_size) > 1:
                valid_ids.append(d.id)

    # 4. Filter against the grouping vector
    if grouping_vector is not None:
        filtered_ids = []
        for v_id in valid_ids:
            idx = _get_logical_idx(v_id)
            if idx != -1 and idx < len(grouping_vector) and grouping_vector[idx] == -1:
                filtered_ids.append(v_id)
        valid_ids = filtered_ids

    return valid_ids

@partial(jax.jit, static_argnums=1)
def apply_dynamic_sparsity(
    st: SparseTensor, 
    sp_rules: tuple[tuple[int, ...], ...]
) -> SparseTensor:
    """
    Applies vectorized dynamic sparsification.
    
    Args:
        st: The input SparseTensor.
        sp_rules: A static tuple of tuples. Each inner tuple represents a pair of 
                  logical dimension indices to sparsify together: (idx1, idx2). 
                  Optionally supports (idx1, idx2, factor) where factor is:
                  -1 (default) for GCD, or 0 to drop the dimension.
    """
    if not sp_rules or st.val is None:
        return st

    out_len = len(st.out_dims)
    total_ndim = out_len + len(st.primal_dims)
    
    valid_pairs = {}
    factors = {}
    
    for group_id, rule in enumerate(sp_rules):
        if len(rule) == 2:
            idx1, idx2 = rule
            factor = -1
        elif len(rule) == 3:
            idx1, idx2, factor = rule
        else:
            continue
            
        if idx1 < total_ndim and idx2 < total_ndim:
            is_out1 = idx1 < out_len
            is_out2 = idx2 < out_len
            
            rel_idx1 = idx1 if is_out1 else idx1 - out_len
            rel_idx2 = idx2 if is_out2 else idx2 - out_len
            
            valid_pairs[group_id] = ((is_out1, rel_idx1), (is_out2, rel_idx2))
            factors[group_id] = factor

    # Filter to ensure each dimension is only used in one rule
    used_axes = set()
    filtered_pairs = {}
    for gid, ((is_out1, idx1), (is_out2, idx2)) in valid_pairs.items():
        abs_idx1 = idx1 if is_out1 else idx1 + out_len
        abs_idx2 = idx2 if is_out2 else idx2 + out_len
        
        # FINAL SAFETY CHECK: Each dimension can only be part of ONE rule
        if abs_idx1 in used_axes or abs_idx2 in used_axes or abs_idx1 == abs_idx2:
            continue
        if abs_idx1 >= len(st.dims) or abs_idx2 >= len(st.dims):
            continue
            
        used_axes.add(abs_idx1)
        used_axes.add(abs_idx2)
        filtered_pairs[gid] = ((is_out1, idx1), (is_out2, idx2))
    
    valid_pairs = filtered_pairs

    if not valid_pairs:
        return st

    if any(isinstance(d, SparseDimension) for d in st.dims):
        st = st.dense()
    
    if not valid_pairs:
        return st

    new_out_dims = list(st.out_dims)
    new_primal_dims = list(st.primal_dims)
    
    axes_to_front = []
    idx_arrays = []
    
    drop_all = True
    current_val_dim = 0
    val_dim_map = {}

    for gid, ((is_out1, idx1), (is_out2, idx2)) in valid_pairs.items():
        d1 = st.out_dims[idx1] if is_out1 else st.primal_dims[idx1]
        d2 = st.out_dims[idx2] if is_out2 else st.primal_dims[idx2]
        
        abs_idx1 = idx1 if is_out1 else idx1 + out_len
        abs_idx2 = idx2 if is_out2 else idx2 + out_len
        
        factor = factors[gid]
        size = gcd(d1.logical_size, d2.logical_size) if factor == -1 else factor
        
        if factor == 0:
            if d1.val_dim is not None: val_dim_map[d1.val_dim] = None
            if d2.val_dim is not None: val_dim_map[d2.val_dim] = None
        else:
            drop_all = False
            b1 = d1.logical_size // size
            b2 = d2.logical_size // size
            
            if d1.val_dim is not None and d2.val_dim is not None:
                axes_to_front.extend([d1.val_dim, d2.val_dim])
                idx_arrays.append(jnp.arange(size))
                
                val_dim_map[d1.val_dim] = current_val_dim
                val_dim_map[d2.val_dim] = current_val_dim
                current_val_dim += 1
                
                if b1 > 1 and getattr(d1, 'block_val_dim', None) is not None:
                    val_dim_map[d1.block_val_dim] = current_val_dim
                    current_val_dim += 1
                if b2 > 1 and getattr(d2, 'block_val_dim', None) is not None:
                    val_dim_map[d2.block_val_dim] = current_val_dim
                    current_val_dim += 1

        def _update_dim(d, b_size, my_idx, other_idx):
            return SparseDimension(
                id=my_idx, size=size, val_dim=None if factor == 0 else d.val_dim,
                other_id=other_idx,
                block_size=b_size if b_size > 1 else None,
                block_val_dim=None if factor == 0 else getattr(d, 'block_val_dim', None)
            )

        if is_out1: new_out_dims[idx1] = _update_dim(d1, b1, abs_idx1, abs_idx2)
        else: new_primal_dims[idx1] = _update_dim(d1, b1, abs_idx1, abs_idx2)
        
        if is_out2: new_out_dims[idx2] = _update_dim(d2, b2, abs_idx2, abs_idx1)
        else: new_primal_dims[idx2] = _update_dim(d2, b2, abs_idx2, abs_idx1)

    # Ensure unique axes for permutation
    seen_axes = set()
    unique_front = []
    for ax in axes_to_front:
        if ax is not None and ax not in seen_axes and ax < st.val.ndim:
            unique_front.append(ax)
            seen_axes.add(ax)
    
    remaining_axes = [i for i in range(st.val.ndim) if i not in seen_axes]
    perm = tuple(unique_front) + tuple(remaining_axes)
    val_transposed = jnp.transpose(st.val, perm)
    
    if idx_arrays:
        mesh_indices = jnp.ix_(*idx_arrays) 
        val_diagonal = val_transposed[tuple(idx for pair in mesh_indices for idx in (pair, pair))]
    else:
        val_diagonal = val_transposed

    def _remap_val_dims(d):
        nv = val_dim_map.get(d.val_dim) if d.val_dim is not None else None
        if isinstance(d, SparseDimension):
            nb = val_dim_map.get(d.block_val_dim) if getattr(d, 'block_val_dim', None) is not None else None
            return replace(d, val_dim=nv, block_val_dim=nb)
        return replace(d, val_dim=nv)

    final_out = tuple(_remap_val_dims(d) for d in new_out_dims)
    final_primal = tuple(_remap_val_dims(d) for d in new_primal_dims)

    return _create_optimized(final_out, final_primal, val_diagonal, st.scalar_mult)