from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from math import prod
from typing import Callable, Literal, override
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class
from jax.typing import DTypeLike

from .ops.utils import (
    _assert_sparse_tensor_consistency,
    _copy,
    _sort_val,
    _arr2st,
)
from .ops.dense import dense
from .ops.transpose import transpose
from .ops.matmul import matmul
from .ops.elementwise import elementwise

from .dimensions import Dimension, DenseDimension, SparseDimension


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
    out_dims: tuple[Dimension, ...]
    primal_dims: tuple[Dimension, ...]
    val: Array | None
    scalar_mult: Array
    fill_value: Array
    pre_transforms: tuple[Transform, ...]
    post_transforms: tuple[Transform, ...]

    def __init__(
        self,
        out_dims: Sequence[Dimension],
        primal_dims: Sequence[Dimension],
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
            if isinstance(d, SparseDimension):
                if key == "out":
                    res[d.id] = d.other_id
                elif key == "primal":
                    res[d.other_id] = d.id
        return res

    @property
    def sparse_shape(self) -> tuple[int, ...]:
        sparse_dims = []
        seen_val_dims = set()
        for d in self.dims:
            if isinstance(d, SparseDimension) and d.val_dim is not None:
                if d.val_dim not in seen_val_dims:
                    sparse_dims.append(d)
                    seen_val_dims.add(d.val_dim)
        sparse_dims.sort(key=lambda d: d.val_dim)
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
            if isinstance(d, DenseDimension) and d.val_dim is not None:
                dense_dims_meta.append((d.val_dim, d.size))
            if isinstance(d, SparseDimension) and d.block_val_dim is not None:
                dense_dims_meta.append((d.block_val_dim, d.block_size))
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
        return jnp.asarray(self.fill_value).dtype

    def astype(
        self, dtype: DTypeLike, copy: bool = True, **kwargs: Any
    ) -> SparseTensor:
        new_val = self.val.astype(dtype, **kwargs) if self.val is not None else None
        new_fill = jnp.asarray(self.fill_value).astype(dtype)
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
    ):
        return _copy(self, val, scalar_mult, fill_value)

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


