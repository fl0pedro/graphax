from __future__ import annotations

from dataclasses import replace
from math import prod
from typing import Callable, Literal, override
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class
from jax.typing import DTypeLike

import math
from functools import partial

from graphax.sparse.ops.utils import (
    _assert_sparse_tensor_consistency,
    _copy,
    _sort_val,
    _arr2st,
    _swap_back_axes,
    _materialize_indexes
)
from graphax.sparse.ops.dense import dense
from graphax.sparse.ops.transpose import transpose
from graphax.sparse.ops.matmul import matmul
from graphax.sparse.ops.elementwise import elementwise

from graphax.sparse.indexes import Index, DenseIndex, SparseIndex


Transform = Callable[["SparseTensor", "SparseTensor", Array], "SparseTensor"]


class SparseMathMixin:
    """Mixin for SparseTensor to handle math operations."""

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __add__(self, other):
        return elementwise(self, other, jnp.add)

    def __radd__(self, other):
        return elementwise(other, self, jnp.add)

    def __sub__(self, other):
        return elementwise(self, other, jnp.subtract)

    def __rsub__(self, other):
        return elementwise(other, self, jnp.subtract)

    def __mul__(self, other):
        return elementwise(self, other, jnp.multiply)

    def __rmul__(self, other):
        return elementwise(other, self, jnp.multiply)

    def __truediv__(self, other):
        return elementwise(self, other, jnp.divide)

    def __rtruediv__(self, other):
        return elementwise(other, self, jnp.divide)

    def __floordiv__(self, other):
        return elementwise(self, other, jnp.floor_divide)

    def __rfloordiv__(self, other):
        return elementwise(other, self, jnp.floor_divide)

    def __mod__(self, other):
        return elementwise(self, other, jax.lax.rem)

    def __rmod__(self, other):
        return elementwise(other, self, jax.lax.rem)

    def __pow__(self, other):
        return elementwise(self, other, jax.lax.pow, is_intersection=False) # seems wrong

    def __rpow__(self, other):
        return elementwise(other, self, jax.lax.pow)

    def __and__(self, other):
        return elementwise(self, other, jax.lax.bitwise_and, is_intersection=True)

    def __rand__(self, other):
        return elementwise(other, self, jax.lax.bitwise_and, is_intersection=True)

    def __or__(self, other):
        return elementwise(self, other, jax.lax.bitwise_or)

    def __ror__(self, other):
        return elementwise(other, self, jax.lax.bitwise_or)

    def __xor__(self, other):
        return elementwise(self, other, jax.lax.bitwise_xor)

    def __rxor__(self, other):
        return elementwise(other, self, jax.lax.bitwise_xor)

    def __lshift__(self, other):
        return elementwise(self, other, jax.lax.shift_left)

    def __rlshift__(self, other):
        return elementwise(other, self, jax.lax.shift_left)

    def __rshift__(self, other):
        return elementwise(self, other, jax.lax.shift_right_logical)

    def __rrshift__(self, other):
        return elementwise(other, self, jax.lax.shift_right_logical)

    def __eq__(self, other):
        return elementwise(self, other, jax.lax.eq)

    def __ne__(self, other):
        return elementwise(self, other, jax.lax.ne)

    def __lt__(self, other):
        return elementwise(self, other, jax.lax.lt)

    def __le__(self, other):
        return elementwise(self, other, jax.lax.le)

    def __gt__(self, other):
        return elementwise(self, other, jax.lax.gt)

    def __ge__(self, other):
        return elementwise(self, other, jax.lax.ge)

    def __neg__(self):
        return self.copy(scalar_mult=-self.scalar_mult)

    def __pos__(self):
        return self.copy()

    def __abs__(self):
        return self.copy(
            val=jnp.abs(self.val) if self.val is not None else None,
            scalar_mult=jnp.abs(self.scalar_mult),
        )

    def __invert__(self):
        return self.copy(
            val=jax.lax.bitwise_not(self.val) if self.val is not None else None
        )

    def __round__(self, ndigits=None):
        return self.copy(
            val=jnp.round(self.val, ndigits) if self.val is not None else None,
            scalar_mult=jnp.round(self.scalar_mult, ndigits),
        )



@register_pytree_node_class
class SparseTensor(SparseMathMixin):
    """
    The core JAX PyTree node representing a block-sparse tensor.

    A SparseTensor partitions its dimensions into two semantic groups: `out_dims` and `primal_dims`.
    This bipartite graph topology enables mathematically rigorous generalized tensor contractions
    and element-wise operations while avoiding premature dense materializations.

    Attributes:
        out_dims (tuple[Index, ...]): Indexes mapped to the output/batch subspace.
        primal_dims (tuple[Index, ...]): Indexes mapped to the contractible/inner subspace.
        val (Array | None): The underlying physical JAX array storing the compressed non-zero values.
            If None, the tensor represents a uniform grid initialized by `fill_value`.
        scalar_mult (Array): A global scalar multiplier to scale the tensor's values without reallocating `val`.
        fill_value (Array): The structural background value (typically 0) of the sparse regions.
    """
    out_dims: tuple[Index, ...]
    primal_dims: tuple[Index, ...]
    val: Array | None
    scalar_mult: Array
    fill_value: Array
    pre_transforms: tuple[Transform, ...]
    post_transforms: tuple[Transform, ...]

    def __init__(
        self,
        out_dims: Sequence[Index],
        primal_dims: Sequence[Index],
        val: Array | None,
        scalar_mult: Array | None = None,
        fill_value: Array | None = None,
        dtype: DTypeLike | None = None,
        pre_transforms: Sequence[Callable] | None = None,
        post_transforms: Sequence[Callable] | None = None,
        sort_val=True,
        check_consistency=True,
        **kwargs,
    ):
        if dtype is None:
            dtype = jnp.dtype("float32")

        if scalar_mult is None:
            scalar_mult = jnp.array(1, dtype=dtype)

        if pre_transforms is None:
            pre_transforms = ()

        if post_transforms is None:
            post_transforms = ()

        if sort_val:
            out_dims, primal_dims, val = _sort_val(out_dims, primal_dims, val)

        if val is not None:
            dtype = dtype or val.dtype
            val = val.astype(dtype)

        if fill_value is None:
            fill_value = jnp.array(0, dtype=dtype or jnp.float32)

        self.out_dims = tuple(out_dims)
        self.primal_dims = tuple(primal_dims)
        self.val = val
        self.scalar_mult = scalar_mult
        self.fill_value = fill_value
        self.pre_transforms = tuple(pre_transforms)
        self.post_transforms = tuple(post_transforms)

        self._dynamic_keys = tuple(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, v)

        if check_consistency:
            _assert_sparse_tensor_consistency(self)

    def tree_flatten(self):
        children = (self.val, self.scalar_mult, self.fill_value)
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
        val, scalar_mult, fill_value = children
        out_dims, primal_dims, pre_transforms, post_transforms, dynamic_kwargs = (
            aux_data
        )
        kwargs = dict(dynamic_kwargs)

        st = cls.__new__(cls)
        st.out_dims = out_dims
        st.primal_dims = primal_dims
        st.val = val
        st.scalar_mult = scalar_mult
        st.fill_value = fill_value
        st.pre_transforms = pre_transforms
        st.post_transforms = post_transforms
        st._dynamic_keys = tuple(kwargs.keys())

        for k, v in kwargs.items():
            setattr(st, k, v)

        return st

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
            if isinstance(d, SparseIndex):
                if key == "out":
                    res[d.id] = d.other_id
                elif key == "primal":
                    res[d.other_id] = d.id
        return res

    @property
    def sparse_shape(self) -> tuple[int, ...]:
        sparse_dims = []
        seen_axiss = set()
        for d in self.dims:
            if isinstance(d, SparseIndex) and d.axis is not None:
                if d.axis not in seen_axiss:
                    sparse_dims.append(d)
                    seen_axiss.add(d.axis)
        sparse_dims.sort(key=lambda d: d.axis)
        return tuple(d.size for d in sparse_dims)

    @property
    def sparse_size(self) -> int:
        return prod(self.sparse_shape)

    @property
    def sparse_ndim(self) -> int:
        return len(self.sparse_shape)

    @property
    def dense_shape(self) -> tuple[int, ...]:
        dense_dims_meta = []
        for d in self.dims:
            if isinstance(d, DenseIndex) and d.axis is not None:
                dense_dims_meta.append((d.axis, d.size))
            if isinstance(d, SparseIndex) and d.block_axis is not None:
                dense_dims_meta.append((d.block_axis, d.block_size))
        dense_dims_meta.sort(key=lambda x: x[0])
        return tuple(x[1] for x in dense_dims_meta)

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
        return transpose(self, out_transpose, primal_transpose)

    def swapdims(self) -> SparseTensor:
        return self.transpose(
            [d.id for d in self.primal_dims], [d.id for d in self.out_dims]
        )

    @property
    def dims(self) -> tuple[Index, ...]:
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
            if isinstance(d, SparseIndex) and d.axis is not None:
                return d.size
        return 1

    @property
    def size(self) -> int:
        return prod(self.shape)

    def dense(self) -> Array:
        return dense(self, hard=True).val * self.scalar_mult

    @property
    def T(self) -> SparseTensor:
        return self.transpose()

    @property
    def _target_arr(self) -> Array:
        return self.val if self.val is not None else self.scalar_mult

    def block_until_ready(self) -> SparseTensor:
        _ = self._target_arr.block_until_ready()
        return self

    @property
    def dtype(self) -> DTypeLike:
        if self.val is not None:
            return self.val.dtype
        # Infer from scalar_mult or fill_value if val is None
        return self.fill_value.dtype

    def astype(
        self, dtype: DTypeLike, copy: bool = True, **kwargs: Any
    ) -> SparseTensor:
        new_val = self.val.astype(dtype, **kwargs) if self.val is not None else None
        new_fill = self.fill_value.astype(dtype)
        # Use dtype for scalar_mult to ensure consistency
        new_scalar_mult = (
            self.scalar_mult.astype(dtype)
            if self.scalar_mult.dtype != jnp.bool_
            else self.scalar_mult
        )
        return self.copy(val=new_val, fill_value=new_fill, scalar_mult=new_scalar_mult)

    def copy(
        self,
        val: Array | None = None,
        scalar_mult: Array | None = None,
        fill_value: Array | None = None,
        out_dims: Sequence[Index] | None = None,
        primal_dims: Sequence[Index] | None = None,
        deep=False,
    ):
        return _copy(
            self,
            val,
            scalar_mult,
            fill_value,
            out_dims=out_dims,
            primal_dims=primal_dims,
            deep=deep,
        )

    # Low priority TODO: axis, and other args
    def all(self) -> Array:
        return jnp.array(
            (
                (self.val is not None and jnp.all(self.val * self.scalar_mult))
                or (self.val is None and self.scalar_mult)
            )
            and self.fill_value * self.scalar_mult
        )

    def any(self) -> Array:
        return jnp.array(
            (
                (self.val is not None and jnp.any(self.val * self.scalar_mult))
                or (self.val is None and self.scalar_mult)
            )
            or self.fill_value * self.scalar_mult
        )

    def sum(self) -> Array:
        return jnp.sum(self.val * self.scalar_mult) + (
            self.fill_value * self.scalar_mult
        ) * (self.size - (self.val.size if self.val is not None else 0))

    def prod(self) -> Array:
        return jnp.prod(self.val * self.scalar_mult) * (
            self.fill_value * self.scalar_mult
        ) ** (self.size - (self.val.size if self.val is not None else 0))

    def max(self) -> Array:
        return (
            max(jnp.max(self.val if self.val is not None else 1), self.fill_value)
            * self.scalar_mult
        )

    def min(self) -> Array:
        return (
            min(jnp.min(self.val if self.val is not None else 1), self.fill_value)
            * self.scalar_mult
        )

    def mean(self) -> Array: ...

    def std(self) -> Array: ...

    def dot(self, other) -> SparseTensor:
        return self @ other

    @property
    def flat(self):
        return self.dense().flat

    def flatten(self):
        return self.dense().flatten()

    def ravel(self):
        return self.dense().ravel()

    def __bool__(self):
        return bool(self.astype(bool).dense())

    def __int__(self):
        return int(self.dense())

    def __float__(self):
        return float(self.dense())

    def __complex__(self):
        return complex(self.dense())

    def __len__(self):
        return self.val.shape[0] if self.val is not None else 0

    # TODO make __iter__ return an iterable of sparse tensors along the sparse component?
    def __iter__(self):
        if self.ndim == 0:
            return iter([self.dense()])
        return iter(self.dense())

    # TODO as with at, make this sparse aware for generalization.
    def __getitem__(self, key):
        return self.dense()[key]

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo=None):
        return self.copy(deep=True)

    def astype(self, dtype: DTypeLike, **kwargs) -> SparseTensor:
        return self.copy(
            val=self.val.astype(dtype, **kwargs) if self.val is not None else None,
            scalar_mult=self.scalar_mult.astype(dtype, **kwargs),
            fill_value=self.fill_value.astype(dtype, **kwargs),
        )

    def conj(self) -> SparseTensor:
        return self.copy(
            val=jnp.conj(self.val) if self.val is not None else None,
            scalar_mult=jnp.conj(self.scalar_mult),
            fill_value=jnp.conj(self.fill_value),
        )

    def conjugate(self) -> SparseTensor:
        return self.conj()

    @property
    def real(self) -> SparseTensor:
        return self.copy(
            val=jnp.real(self.val) if self.val is not None else None,
            scalar_mult=jnp.real(self.scalar_mult),
            fill_value=jnp.real(self.fill_value),
        )

    @property
    def imag(self) -> SparseTensor:
        return self.copy(
            val=jnp.imag(self.val) if self.val is not None else None,
            scalar_mult=jnp.imag(self.scalar_mult),
            fill_value=jnp.imag(self.fill_value),
        )

    def item(self):
        if self.ndim == 0:
            return self.tolist()
        else:
            raise ValueError(
                "can only convert a SparseTensor of size 1 to a Python scalar"
            )

    def tolist(self):
        return self.dense().tolist()

    def tobytes(self, *args, **kwargs):
        return self.dense().tobytes(*args, **kwargs)

    @property
    def sharding(self):
        return self._target_arr.sharding

    def devices(self):
        if hasattr(self._target_arr, "devices"):
            return self._target_arr.devices()
        raise AttributeError(
            f"'{type(self._target_arr).__name__}' object has no attribute 'devices'"
        )

    @property
    def device(self):
        if hasattr(self._target_arr, "device"):
            return self._target_arr.device
        raise AttributeError(
            f"'{type(self._target_arr).__name__}' object has no attribute 'device'"
        )

    @property
    def platform(self):
        if hasattr(self._target_arr, "platform"):
            return self._target_arr.platform
        raise AttributeError(
            f"'{type(self._target_arr).__name__}' object has no attribute 'platform'"
        )

    @property
    def is_deleted(self):
        return self._target_arr.is_deleted()

    @property
    def is_fully_addressable(self):
        return getattr(self._target_arr, "is_fully_addressable", True)

    @property
    def is_fully_replicated(self):
        return getattr(self._target_arr, "is_fully_replicated", True)

    @property
    def is_ready(self):
        if hasattr(self._target_arr, "is_ready"):
            return self._target_arr.is_ready()
        return True

    @property
    def committed(self):
        return getattr(self._target_arr, "committed", False)

    @property
    def addressable_data(self):
        if hasattr(self._target_arr, "addressable_data"):
            return self._target_arr.addressable_data()
        return None

    @property
    def addressable_shards(self):
        if hasattr(self._target_arr, "addressable_shards"):
            return self._target_arr.addressable_shards()
        return None

    @property
    def global_shards(self):
        if hasattr(self._target_arr, "global_shards"):
            return self._target_arr.global_shards()
        return None

    @property
    def unsafe_buffer_pointer(self):
        if hasattr(self._target_arr, "unsafe_buffer_pointer"):
            return self._target_arr.unsafe_buffer_pointer()
        raise AttributeError(
            f"'{type(self._target_arr).__name__}' object has no attribute 'unsafe_buffer_pointer'"
        )

    @property
    def device_buffer(self):
        if hasattr(self._target_arr, "device_buffer"):
            return self._target_arr.device_buffer()
        raise AttributeError(
            f"'{type(self._target_arr).__name__}' object has no attribute 'device_buffer'"
        )

    @property
    def device_buffers(self):
        if hasattr(self._target_arr, "device_buffers"):
            return self._target_arr.device_buffers()
        raise AttributeError(
            f"'{type(self._target_arr).__name__}' object has no attribute 'device_buffers'"
        )

    @property
    def traceback(self):
        return getattr(self._target_arr, "traceback", None)

    def delete(self):
        if hasattr(self._target_arr, "delete"):
            self._target_arr.delete()

    def copy_to_host_async(self):
        if hasattr(self._target_arr, "copy_to_host_async"):
            self._target_arr.copy_to_host_async()

    def clone(self):
        return self.copy(deep=True)

    def to_device(self, device):
        return self.copy(
            val=jax.device_put(self.val, device) if self.val is not None else None,
            scalar_mult=jax.device_put(self.scalar_mult, device),
            fill_value=jax.device_put(self.fill_value, device),
        )

    @property
    def aval(self):
        return getattr(self._target_arr, "aval", None)

    @property
    def weak_type(self):
        return getattr(self._target_arr, "weak_type", False)

    @property
    def itemsize(self):
        return self.dtype.itemsize

    @property
    def nbytes(self):
        return self.size * self.itemsize

    def on_device_size_in_bytes(self):
        return self._target_arr.on_device_size_in_bytes()



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
    target_dim_pos = -1
    
    out_len = len(st.out_dims)

    # 1. Locate the dimension and positional index
    for i, d in enumerate(st.out_dims):
        if d.id == dim_id:
            target_dim = d
            is_out_dim = True
            target_dim_pos = i
            break
            
    if target_dim is None:
        for i, d in enumerate(st.primal_dims):
            if d.id == dim_id:
                target_dim = d
                target_dim_pos = out_len + i
                break
                
    if target_dim is None:
        raise ValueError(f"Index ID {dim_id} not found in SparseTensor.")

    # If the target itself is already paired, return empty
    if grouping_vector is not None and target_dim_pos < len(grouping_vector):
        if grouping_vector[target_dim_pos] != -1:
            return []

    # 2. Find positional index of a given ID (helper)
    def _get_dim_pos(search_id: int) -> int:
        for i, d in enumerate(st.out_dims):
            if d.id == search_id: return i
        for i, d in enumerate(st.primal_dims):
            if d.id == search_id: return out_len + i
        return -1

    # 3. Identify valid partners
    valid_ids = []

    if isinstance(target_dim, SparseIndex):
        # Sparse dimensions are locked to their other_id
        valid_ids = [target_dim.other_id]
    else:
        # Dense dimensions scan the opposite side
        opposite_dims = st.primal_dims if is_out_dim else st.out_dims
        for d in opposite_dims:
            if isinstance(d, DenseIndex) and math.gcd(target_dim.logical_size, d.logical_size) > 1:
                valid_ids.append(d.id)

    # 4. Filter against the grouping vector
    if grouping_vector is not None:
        filtered_ids = []
        for v_id in valid_ids:
            idx = _get_dim_pos(v_id)
            if idx != -1 and idx < len(grouping_vector) and grouping_vector[idx] == -1:
                filtered_ids.append(v_id)
        valid_ids = filtered_ids

    return valid_ids

@partial(jax.jit, static_argnums=1)
def apply_dynamic_sparsity(
    st: SparseTensor, 
    sp_rules: tuple[tuple[int, ...], ...]
) -> SparseTensor:
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

    filtered_pairs = {}
    used_axes = set()
    for gid, ((is_out1, idx1), (is_out2, idx2)) in valid_pairs.items():
        abs_idx1 = idx1 if is_out1 else idx1 + out_len
        abs_idx2 = idx2 if is_out2 else idx2 + out_len
        
        if abs_idx1 in used_axes or abs_idx2 in used_axes or abs_idx1 == abs_idx2:
            continue
            
        d1 = st.out_dims[idx1] if is_out1 else st.primal_dims[idx1]
        d2 = st.out_dims[idx2] if is_out2 else st.primal_dims[idx2]
        
        # Prevent breaking existing external pairs
        conflict = False
        if isinstance(d1, SparseIndex) and getattr(d1, 'other_id', None) != d2.id:
            conflict = True
        if isinstance(d2, SparseIndex) and getattr(d2, 'other_id', None) != d1.id:
            conflict = True
            
        if conflict:
            continue
            
        used_axes.add(abs_idx1)
        used_axes.add(abs_idx2)
        filtered_pairs[gid] = ((is_out1, idx1), (is_out2, idx2))
        
    if not filtered_pairs:
        return st

    # 1. Determine axes to extract and drop
    axes_to_front = []
    axes_to_drop = []
    idx_arrays = []

    for gid, ((is_out1, idx1), (is_out2, idx2)) in filtered_pairs.items():
        d1 = st.out_dims[idx1] if is_out1 else st.primal_dims[idx1]
        d2 = st.out_dims[idx2] if is_out2 else st.primal_dims[idx2]
        
        factor = factors[gid]
        size = math.gcd(d1.logical_size, d2.logical_size) if factor == -1 else factor
        
        v1 = getattr(d1, 'axis', None)
        v2 = getattr(d2, 'axis', None)
        if v1 is not None and v1 >= st.val.ndim: v1 = None
        if v2 is not None and v2 >= st.val.ndim: v2 = None
        
        if factor == 0:
            if v1 is not None: axes_to_drop.append(v1)
            if v2 is not None: axes_to_drop.append(v2)
        else:
            # FIX 1: Only extract if they are distinct physical axes!
            if v1 is not None and v2 is not None and v1 != v2:
                axes_to_front.extend([v1, v2])
                idx_arrays.append(jnp.arange(size))

    # 2. Build explicit map of physical axes
    seen_axes = set()
    unique_front = []
    for ax in axes_to_front:
        if ax not in seen_axes:
            unique_front.append(ax)
            seen_axes.add(ax)
            
    axes_to_drop = list(set(axes_to_drop))
    remaining_axes = [i for i in range(st.val.ndim) if i not in seen_axes and i not in axes_to_drop]
    
    axis_map = {}
    current_axis = 0
    
    # Paired extractions collapse into a single target axis
    for i in range(0, len(unique_front), 2):
        axis_map[unique_front[i]] = current_axis
        axis_map[unique_front[i+1]] = current_axis
        current_axis += 1
        
    for ax in remaining_axes:
        axis_map[ax] = current_axis
        current_axis += 1
        
    # 3. Vectorized Array Extraction
    perm = tuple(unique_front) + tuple(remaining_axes) + tuple(axes_to_drop)
    val_transposed = jnp.transpose(st.val, perm)
    
    if idx_arrays:
        mesh_indices = jnp.ix_(*idx_arrays) 
        val_diagonal = val_transposed[tuple(idx for pair in mesh_indices for idx in (pair, pair))]
    else:
        val_diagonal = val_transposed

    if axes_to_drop:
        for _ in range(len(axes_to_drop)):
            val_diagonal = val_diagonal[..., 0]

    # 4. Reconstruct Indexes enforcing exact shared assignments
    new_out_dims = list(st.out_dims)
    new_primal_dims = list(st.primal_dims)
    touched_out = set()
    touched_primal = set()

    for gid, ((is_out1, idx1), (is_out2, idx2)) in filtered_pairs.items():
        d1 = st.out_dims[idx1] if is_out1 else st.primal_dims[idx1]
        d2 = st.out_dims[idx2] if is_out2 else st.primal_dims[idx2]
        
        factor = factors[gid]
        size = math.gcd(d1.logical_size, d2.logical_size) if factor == -1 else factor
        b1 = d1.logical_size // size
        b2 = d2.logical_size // size
        
        v1 = getattr(d1, 'axis', None)
        v2 = getattr(d2, 'axis', None)
        if v1 is not None and v1 >= st.val.ndim: v1 = None
        if v2 is not None and v2 >= st.val.ndim: v2 = None
        
        if factor == 0:
            nv = None
        else:
            # Force both halves to adopt the exact same physical axis mappings
            old_v = v1 if v1 is not None else v2
            nv = axis_map.get(old_v) if old_v is not None else None

        def _update_dim(d, other_d, b_size):
            old_b = getattr(d, 'block_axis', None)
            if old_b is not None and old_b >= st.val.ndim: old_b = None
            nb = axis_map.get(old_b) if (factor != 0 and old_b is not None) else None
            
            # FIX: Route the primary physical axis to the block if the outer size is 1
            actual_nv = nv if size > 1 else None
            actual_nb = nb
            if size == 1 and b_size > 1 and nv is not None:
                actual_nb = nv
                
            return SparseIndex(
                id=d.id, size=size, axis=actual_nv,
                other_id=other_d.id,
                block_size=b_size if b_size > 1 else None,
                block_axis=actual_nb if b_size > 1 else None
            )

        if is_out1: 
            new_out_dims[idx1] = _update_dim(d1, d2, b1)
            touched_out.add(idx1)
        else: 
            new_primal_dims[idx1] = _update_dim(d1, d2, b1)
            touched_primal.add(idx1)
            
        if is_out2: 
            new_out_dims[idx2] = _update_dim(d2, d1, b2)
            touched_out.add(idx2)
        else: 
            new_primal_dims[idx2] = _update_dim(d2, d1, b2)
            touched_primal.add(idx2)

    def _remap_untouched(d):
        old_v = getattr(d, 'axis', None)
        old_b = getattr(d, 'block_axis', None)
        nv = axis_map.get(old_v) if old_v is not None else None
        nb = axis_map.get(old_b) if old_b is not None else None
        
        if isinstance(d, SparseIndex):
            return replace(d, axis=nv, block_axis=nb)
        return replace(d, axis=nv)

    final_out = tuple(_remap_untouched(d) if i not in touched_out else new_out_dims[i] for i, d in enumerate(st.out_dims))
    final_primal = tuple(_remap_untouched(d) if i not in touched_primal else new_primal_dims[i] for i, d in enumerate(st.primal_dims))

    # 5. Lock Canonical Physical Ordering
    final_dims = final_out + final_primal
    target_perm = []
    seen_target = set()
    
    for d in final_dims:
        if getattr(d, 'axis', None) is not None and d.axis not in seen_target:
            target_perm.append(d.axis)
            seen_target.add(d.axis)
        if getattr(d, 'block_axis', None) is not None and d.block_axis not in seen_target:
            target_perm.append(d.block_axis)
            seen_target.add(d.block_axis)
            
    for i in range(val_diagonal.ndim):
        if i not in seen_target:
            target_perm.append(i)
            seen_target.add(i)
            
    if target_perm != list(range(val_diagonal.ndim)):
        val_diagonal = jnp.transpose(val_diagonal, target_perm)
        inv_perm = {old: new for new, old in enumerate(target_perm)}
        
        def _reorder_axis(d):
            nv = inv_perm.get(getattr(d, 'axis', None)) if getattr(d, 'axis', None) is not None else None
            if isinstance(d, SparseIndex):
                nb = inv_perm.get(getattr(d, 'block_axis', None)) if getattr(d, 'block_axis', None) is not None else None
                return replace(d, axis=nv, block_axis=nb)
            return replace(d, axis=nv)
            
        final_out = tuple(_reorder_axis(d) for d in final_out)
        final_primal = tuple(_reorder_axis(d) for d in final_primal)

    return SparseTensor(final_out, final_primal, val_diagonal, st.scalar_mult, sort_val=False, check_consistency=False)