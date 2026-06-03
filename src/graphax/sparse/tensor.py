from __future__ import annotations

import math
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial, wraps
from math import prod
from typing import Any, Callable, Literal, override

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.tree_util import register_pytree_node_class
from jax.typing import DTypeLike

from graphax.sparse.indexes import DenseIndex, Index, DiagonalIndex
from graphax.sparse.ops.dense import dense
from graphax.sparse.ops.elementwise import elementwise
from graphax.sparse.ops.matmul import matmul
from graphax.sparse.ops.transpose import transpose
from graphax.sparse.ops.utils import (
    _arr2st,
    _assert_sparse_tensor_consistency,
    _copy,
    _materialize_indexes,
    _swap_back_axes,
)


def _compute_zero_fill_flag(fill_value) -> bool:
    """Static probe: ``True`` iff ``fill_value`` is concretely known to be zero.

    Called once at ``SparseTensor`` construction (where ``fill_value`` is still
    a concrete jax/numpy/python scalar in the common case) so the flag can ride
    along in the pytree's static aux_data, surviving jit tracing. If the value
    is already a tracer (rare — happens only when the tensor is built inside
    a traced function), we fall back to ``False`` (the densify path is correct
    in all cases; we just lose the fast-path opportunity)."""
    try:
        return bool(np.all(np.asarray(fill_value) == 0))
    except (
        TypeError,
        ValueError,
        AttributeError,
        jax.errors.TracerArrayConversionError,
    ):
        # Tracer or non-array; conservatively report non-zero.
        return False


Transform = Callable[["SparseTensor", "SparseTensor", Array], "SparseTensor"]


def _on_materialized(method):
    """Decorator for value-semantic methods (reductions + non-linear unary ops)
    that must NOT read a raw compressed ``val``: when the tensor has compressed
    (``BandedIndex`` / ``SetIndex``) dims, run ``method`` on the materialized
    ``{Diagonal, Dense}`` equivalent instead. No-op for non-compressed tensors.
    A band buffer carries out-of-band padding and a set buffer the
    pre-combination per-side blocks, so reducing either directly is wrong."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        t = self._materialize_compressed()
        if t is not self:
            return getattr(t, method.__name__)(*args, **kwargs)
        return method(self, *args, **kwargs)
    return wrapper


class SparseMathMixin:
    """Mixin for SparseTensor to handle math operations."""

    # ``__eq__`` returns a SparseTensor (elementwise), not a bool, so the
    # default identity-based ``__hash__`` would silently make instances
    # hashable in inconsistent ways. Mark explicitly unhashable.
    __hash__ = None

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
        return elementwise(
            self, other, jax.lax.pow, is_intersection=False
        )  # seems wrong

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
        return self.copy(
            scalar_mult=-self.scalar_mult,
            fill_value=-self.fill_value,
        )

    def __pos__(self):
        return self.copy()

    # Non-linear in val → @_on_materialized densifies compressed storage first
    # (abs(a+b) != abs(a)+abs(b); a set buffer stores the pre-combination
    # per-side blocks). __neg__ / __pos__ are linear and stay lazy.
    @_on_materialized
    def __abs__(self):
        return self.copy(
            val=jnp.abs(self.val) if self.val is not None else None,
            scalar_mult=jnp.abs(self.scalar_mult),
            fill_value=jnp.abs(self.fill_value),
        )

    @_on_materialized
    def __invert__(self):
        return self.copy(
            val=jax.lax.bitwise_not(self.val) if self.val is not None else None
        )

    @_on_materialized
    def __round__(self, ndigits=None):
        return self.copy(
            val=jnp.round(self.val, ndigits) if self.val is not None else None,
            scalar_mult=jnp.round(self.scalar_mult, ndigits),
            fill_value=jnp.round(self.fill_value, ndigits),
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
        *,
        scalar_mult: Array | None = None,
        fill_value: Array | None = None,
        dtype: DTypeLike | None = None,
        pre_transforms: Sequence[Callable] | None = None,
        post_transforms: Sequence[Callable] | None = None,
        check_consistency=True,
        zero_fill: bool | None = None,  # this depends on fill_value... should just be
        **kwargs,
    ):
        if val is not None and not hasattr(val, "dtype"):
            val = jnp.asarray(val)

        if dtype is None:
            dtype = val.dtype if val is not None else jnp.dtype("float32")

        if scalar_mult is None:
            scalar_mult = jnp.array(1, dtype=dtype)

        if pre_transforms is None:
            pre_transforms = ()

        if post_transforms is None:
            post_transforms = ()

        if val is not None and val.dtype != dtype:
            val = val.astype(dtype)

        if fill_value is None:
            fill_value = jnp.array(0, dtype=dtype)

        self.out_dims = tuple(out_dims)
        self.primal_dims = tuple(primal_dims)
        self.val = val
        self.scalar_mult = scalar_mult
        self.fill_value = fill_value
        self.pre_transforms = tuple(pre_transforms)
        self.post_transforms = tuple(post_transforms)
        self._zero_fill = (
            zero_fill if zero_fill is not None else _compute_zero_fill_flag(fill_value)
        )

        self._dynamic_keys = tuple(kwargs.keys())
        for k, v in kwargs.items():
            try:
                hash(v)
            except TypeError as e:
                raise TypeError(
                    f"SparseTensor **kwargs values must be hashable (jit aux_data): "
                    f"{k}={v!r} ({e})"
                )
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
            self._zero_fill,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        val, scalar_mult, fill_value = children
        (
            out_dims,
            primal_dims,
            pre_transforms,
            post_transforms,
            dynamic_kwargs,
            zero_fill,
        ) = aux_data
        kwargs = dict(dynamic_kwargs)

        st = cls.__new__(cls)
        st.out_dims = out_dims
        st.primal_dims = primal_dims
        st.val = val
        st.scalar_mult = scalar_mult
        st.fill_value = fill_value
        st.pre_transforms = pre_transforms
        st.post_transforms = post_transforms
        st._zero_fill = zero_fill
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
            if d.is_sparse:
                if key == "out":
                    res[d.id] = d.other_id
                elif key == "primal":
                    res[d.other_id] = d.id
        return res

    @property
    def sparse_shape(self) -> tuple[int, ...]:
        sparse_dims = []
        seen_axes = set()
        for d in self.dims:
            if d.is_sparse and d.axis is not None:
                if d.axis not in seen_axes:
                    sparse_dims.append(d)
                    seen_axes.add(d.axis)
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
            if not d.is_sparse and d.axis is not None:
                dense_dims_meta.append((d.axis, d.size))
            if d.is_sparse and d.block_axis is not None:
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
            if d.is_sparse and d.axis is not None:
                return d.size
        return 1

    @property
    def size(self) -> int:
        return prod(self.shape)

    def dense(self) -> Array:
        # Compressed Index dims (BandedIndex / SetIndex) carry structure the
        # scatter-based ``dense`` flow doesn't understand; densify them to
        # plain DenseIndex / DiagonalIndex first (no-op when there are none).
        from graphax.sparse.ops.utils import _compressed_dims, _densify_compressed_dims
        t = _densify_compressed_dims(self) if _compressed_dims(self) else self
        return dense(t, hard=True).val * t.scalar_mult

    @property
    def T(self) -> SparseTensor:
        return self.transpose()

    @property
    def _target_arr(self) -> Array:
        if self.val is not None:
            return self.val
        return self.scalar_mult

    def eff_val(self) -> Array | None:  # put this somewhere else?
        return self.val

    def block_until_ready(self) -> SparseTensor:
        _ = self._target_arr.block_until_ready()
        return self

    @property
    def dtype(self) -> DTypeLike:
        if self.val is not None:
            return self.val.dtype
        return self.scalar_mult.dtype

    def copy(
        self,
        val: Array | None = None,
        scalar_mult: Array | None = None,
        fill_value: Array | None = None,
    ):
        return _copy(self, val, scalar_mult, fill_value)

    def _materialize_compressed(self) -> SparseTensor:
        """Return an equivalent tensor with no compressed (``BandedIndex`` /
        ``SetIndex``) dims — they are densified to their compact
        ``DiagonalIndex`` / ``DenseIndex`` form so ``val`` again holds *exactly*
        the non-fill values with ``size - val.size`` implicit fill cells. Any
        value-semantic reduction / non-linear unary op below must route through
        this first: a raw band / set buffer carries out-of-band padding slots
        (banded) or pre-combination per-side blocks (set) whose element multiset
        does NOT match the dense form, so reducing it directly is wrong. No-op
        (returns ``self``) when the tensor has no compressed dims."""
        from graphax.sparse.ops.utils import _compressed_dims, _materialize_for_op

        return _materialize_for_op(self) if _compressed_dims(self) else self

    # Low priority TODO: axis, and other args
    @_on_materialized
    def all(self) -> Array:
        val_part = (
            jnp.all(self.val * self.scalar_mult) if self.val is not None else True
        )
        return jnp.logical_and(val_part, self.fill_value * self.scalar_mult != 0)

    @_on_materialized
    def any(self) -> Array:
        val_part = (
            jnp.any(self.val * self.scalar_mult) if self.val is not None else False
        )
        return jnp.logical_or(val_part, self.fill_value * self.scalar_mult != 0)

    @_on_materialized
    def sum(self) -> Array:
        if self.val is None:
            return self.fill_value * self.scalar_mult * self.size
        return jnp.sum(self.val * self.scalar_mult) + (
            self.fill_value * self.scalar_mult
        ) * (self.size - self.val.size)

    @_on_materialized
    def prod(self) -> Array:
        if self.val is None:
            return (self.fill_value * self.scalar_mult) ** self.size
        return jnp.prod(self.val * self.scalar_mult) * (
            self.fill_value * self.scalar_mult
        ) ** (self.size - self.val.size)

    @_on_materialized
    def max(self) -> Array:
        if self.val is None:
            return self.fill_value * self.scalar_mult
        return jnp.maximum(jnp.max(self.val), self.fill_value) * self.scalar_mult

    @_on_materialized
    def min(self) -> Array:
        if self.val is None:
            return self.fill_value * self.scalar_mult
        return jnp.minimum(jnp.min(self.val), self.fill_value) * self.scalar_mult

    def mean(self) -> Array:
        return self.sum() / self.size

    def std(self) -> Array:
        raise NotImplementedError("std not yet implemented for SparseTensor")

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
        if self.val is not None:
            return self.val.shape[0]
        # Structurally non-empty but ``val=None``: report the first logical
        # dim's size (out_dims preferred, then primal_dims).
        if self.out_dims:
            return self.out_dims[0].logical_size
        if self.primal_dims:
            return self.primal_dims[0].logical_size
        return 0

    # TODO make __iter__ return an iterable of sparse tensors along the sparse component?
    def __iter__(self):
        return iter(self.dense())

    # TODO as with at, make this sparse aware for generalization.
    def __getitem__(self, key):
        return self.dense()[key]

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo=None):
        return self.copy()

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
        arr = self._target_arr
        if hasattr(arr, "devices"):
            return arr.devices()
        raise AttributeError(
            f"'{type(arr).__name__}' object has no attribute 'devices'"
        )

    @property
    def device(self):
        arr = self._target_arr
        if hasattr(arr, "device"):
            return arr.device
        raise AttributeError(f"'{type(arr).__name__}' object has no attribute 'device'")

    @property
    def platform(self):
        arr = self._target_arr
        if hasattr(arr, "platform"):
            return arr.platform
        raise AttributeError(
            f"'{type(arr).__name__}' object has no attribute 'platform'"
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
        arr = self._target_arr
        if hasattr(arr, "is_ready"):
            return arr.is_ready()
        return True

    @property
    def committed(self):
        return getattr(self._target_arr, "committed", False)

    @property
    def addressable_data(self):
        arr = self._target_arr
        if hasattr(arr, "addressable_data"):
            return arr.addressable_data()
        return None

    @property
    def addressable_shards(self):
        arr = self._target_arr
        if hasattr(arr, "addressable_shards"):
            return arr.addressable_shards()
        return None

    @property
    def global_shards(self):
        arr = self._target_arr
        if hasattr(arr, "global_shards"):
            return arr.global_shards()
        return None

    @property
    def unsafe_buffer_pointer(self):
        arr = self._target_arr
        if hasattr(arr, "unsafe_buffer_pointer"):
            return arr.unsafe_buffer_pointer()
        raise AttributeError(
            f"'{type(arr).__name__}' object has no attribute 'unsafe_buffer_pointer'"
        )

    @property
    def device_buffer(self):
        arr = self._target_arr
        if hasattr(arr, "device_buffer"):
            return arr.device_buffer()
        raise AttributeError(
            f"'{type(arr).__name__}' object has no attribute 'device_buffer'"
        )

    @property
    def device_buffers(self):
        arr = self._target_arr
        if hasattr(arr, "device_buffers"):
            return arr.device_buffers()
        raise AttributeError(
            f"'{type(arr).__name__}' object has no attribute 'device_buffers'"
        )

    @property
    def traceback(self):
        return getattr(self._target_arr, "traceback", None)

    def delete(self):
        # TODO this does not free buffers...
        arr = self._target_arr
        if hasattr(arr, "delete"):
            arr.delete()

    def copy_to_host_async(self):
        arr = self._target_arr
        if hasattr(arr, "copy_to_host_async"):
            arr.copy_to_host_async()

    def clone(self):
        return self.copy()

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
    grouping_vector: tuple[int, ...] | None = None,
) -> list[int]:
    """Find dimension IDs that can be paired with ``dim_id`` in ``st``.

    If ``grouping_vector`` is provided, dimensions already paired
    (entry != -1) are excluded from the result.
    """
    target_dim = None
    is_out_dim = False
    target_dim_pos = -1

    out_len = len(st.out_dims)

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

    if grouping_vector is not None and target_dim_pos < len(grouping_vector):
        if grouping_vector[target_dim_pos] != -1:
            return []

    def _get_dim_pos(search_id: int) -> int:
        for i, d in enumerate(st.out_dims):
            if d.id == search_id:
                return i
        for i, d in enumerate(st.primal_dims):
            if d.id == search_id:
                return out_len + i
        return -1

    valid_ids: list[int] = []
    if target_dim.is_sparse:
        valid_ids = [target_dim.other_id]
    else:
        opposite_dims = st.primal_dims if is_out_dim else st.out_dims
        for d in opposite_dims:
            if (
                not d.is_sparse
                and math.gcd(target_dim.logical_size, d.logical_size) > 1
            ):
                valid_ids.append(d.id)

    if grouping_vector is not None:
        filtered_ids = []
        for v_id in valid_ids:
            idx = _get_dim_pos(v_id)
            if idx != -1 and idx < len(grouping_vector) and grouping_vector[idx] == -1:
                filtered_ids.append(v_id)
        valid_ids = filtered_ids

    return valid_ids


def _apply_block_diagonal(
    st: SparseTensor,
    is_out1: bool,
    idx1: int,
    is_out2: bool,
    idx2: int,
    size: int,
    b1: int,
    b2: int,
) -> SparseTensor:
    """Apply a single ``(idx1, idx2, size, b1, b2)`` block-diagonal rule.

    The two paired physical val axes (one of size ``size*b1``, one of
    ``size*b2``) are reshaped to ``(size, b)`` then collapsed via
    ``jnp.diagonal`` so a single ``size`` axis survives. The block axes
    survive as the DiagonalIndex block axes — exactly what matmul needs to
    broadcast and reduce a true block-diagonal.

    Special cases:
    * ``size == 1``: factor=1 ⇒ no real diagonalisation; return ``st``.
    * One of v1/v2 is implicit (None or out of range): just reshape the
      explicit side to expose the block axis.
    * v1 == v2: same physical axis already encodes a diagonal — bail out
      and keep the existing pairing.
    """
    if size == 1:
        return st

    d1 = st.out_dims[idx1] if is_out1 else st.primal_dims[idx1]
    d2 = st.out_dims[idx2] if is_out2 else st.primal_dims[idx2]

    if d1.is_sparse and getattr(d1, "other_id", None) != d2.id:
        return st
    if d2.is_sparse and getattr(d2, "other_id", None) != d1.id:
        return st

    v1 = getattr(d1, "axis", None)
    v2 = getattr(d2, "axis", None)
    if v1 is not None and v1 >= st.val.ndim:
        v1 = None
    if v2 is not None and v2 >= st.val.ndim:
        v2 = None

    val = st.val
    new_K_axis: int | None = None
    new_b1_axis: int | None = None
    new_b2_axis: int | None = None
    other_shift_threshold: int | None = None

    if v1 is None and v2 is None:
        pass
    elif v1 is None or v2 is None:
        v_present = v1 if v1 is not None else v2
        b_present = b1 if v1 is not None else b2
        new_shape = list(val.shape)
        new_shape[v_present] = size
        new_shape.insert(v_present + 1, b_present)
        val = val.reshape(new_shape)
        new_K_axis = v_present
        if v1 is not None:
            new_b1_axis = v_present + 1 if b1 > 1 else None
        else:
            new_b2_axis = v_present + 1 if b2 > 1 else None
        other_shift_threshold = v_present + 1
    elif v1 == v2:
        return st
    else:
        new_shape = list(val.shape)
        if v1 < v2:
            lo, hi = v1, v2
            lo_b, hi_b = b1, b2
            lo_is_v1 = True
        else:
            lo, hi = v2, v1
            lo_b, hi_b = b2, b1
            lo_is_v1 = False
        new_shape[hi] = size
        new_shape.insert(hi + 1, hi_b)
        new_shape[lo] = size
        new_shape.insert(lo + 1, lo_b)
        val = val.reshape(new_shape)
        lo_K, lo_b_pos = lo, lo + 1
        hi_K, hi_b_pos = hi + 1, hi + 2

        val = jnp.diagonal(val, axis1=lo_K, axis2=hi_K)
        diag_pos = val.ndim - 1
        target_K = lo
        lo_b_post_diag = lo_b_pos - 1
        hi_b_post_diag = hi_b_pos - 2
        if diag_pos != target_K:
            perm = list(range(val.ndim))
            perm.pop(diag_pos)
            perm.insert(target_K, diag_pos)
            val = jnp.transpose(val, perm)

            def _shift_for_permute(p):
                if p < target_K:
                    return p
                if p < diag_pos:
                    return p + 1
                return target_K

            lo_b_final = _shift_for_permute(lo_b_post_diag)
            hi_b_final = _shift_for_permute(hi_b_post_diag)
        else:
            lo_b_final = lo_b_post_diag
            hi_b_final = hi_b_post_diag

        new_K_axis = target_K
        if lo_is_v1:
            new_b1_axis = lo_b_final if b1 > 1 else None
            new_b2_axis = hi_b_final if b2 > 1 else None
        else:
            new_b2_axis = lo_b_final if b2 > 1 else None
            new_b1_axis = hi_b_final if b1 > 1 else None
        other_shift_threshold = target_K + 1

    def _shift_other(p):
        if p is None or other_shift_threshold is None:
            return p
        return p + 1 if p >= other_shift_threshold else p

    def _build(d, other_d, axis, block_axis, b_size):
        return DiagonalIndex(
            id=d.id,
            size=size,
            axis=axis,
            other_id=other_d.id,
            block_size=b_size if b_size > 1 else None,
            block_axis=block_axis if b_size > 1 else None,
        )

    new_d1 = _build(d1, d2, new_K_axis, new_b1_axis, b1)
    new_d2 = _build(d2, d1, new_K_axis, new_b2_axis, b2)

    def _remap_other(d):
        old_v = getattr(d, "axis", None)
        old_b = getattr(d, "block_axis", None)
        nv = _shift_other(old_v)
        nb = _shift_other(old_b)
        if d.is_sparse:
            return replace(d, axis=nv, block_axis=nb)
        return replace(d, axis=nv)

    new_out = list(st.out_dims)
    new_primal = list(st.primal_dims)
    for i, d in enumerate(st.out_dims):
        if i == idx1 and is_out1:
            new_out[i] = new_d1
        elif i == idx2 and is_out2:
            new_out[i] = new_d2
        else:
            new_out[i] = _remap_other(d)
    for i, d in enumerate(st.primal_dims):
        if i == idx1 and not is_out1:
            new_primal[i] = new_d1
        elif i == idx2 and not is_out2:
            new_primal[i] = new_d2
        else:
            new_primal[i] = _remap_other(d)

    return SparseTensor(
        new_out,
        new_primal,
        val,
        scalar_mult=st.scalar_mult,
        check_consistency=False,
    )


def sparse_tensor_zeros_like(st: SparseTensor) -> SparseTensor:
    return _copy(st, jnp.zeros_like(st.val), jnp.array(1.0), jnp.array(0.0))
