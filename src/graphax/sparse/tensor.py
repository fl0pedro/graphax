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
from graphax.sparse.dtype_compute import _scaled_mul
from graphax.sparse.ops.dense import dense  # noqa: F401  (re-exported: callers do `from graphax.sparse.tensor import dense`)
from graphax.sparse.ops.elementwise import elementwise
from graphax.sparse.ops.matmul import matmul
from graphax.sparse.ops.transpose import transpose
from graphax.sparse.ops.utils import (
    _KEEP,
    _arr2st,
    _assert_sparse_tensor_consistency,
    _copy,
    _materialize_indexes,
    _swap_back_axes,
)


def _map_fill(fill, fn):
    """Apply a zero-preserving unary ``fn`` to a tensor's ``fill_value``,
    propagating the ``None`` sentinel. ``fill_value is None`` means the fill is
    statically zero (the fast-path marker); since ``fn`` preserves zero (neg /
    abs / round / astype / conj / real / imag all map 0 → 0), ``None`` stays
    ``None`` rather than materializing a concrete zero and losing the marker."""
    return None if fill is None else fn(fill)


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

    # pow is union, NOT intersection: ``x ** 0 == 1`` keeps a position present
    # even where the exponent is an implicit (fill) zero, so the output support
    # must not be narrowed to the intersection of the operands' supports. (0 is
    # absorbing only in the *base*, and only for positive exponents — a single
    # is_intersection flag can't capture that, so the safe/correct choice is the
    # broader union.) Both directions are explicit + consistent.
    def __pow__(self, other):
        return elementwise(self, other, jax.lax.pow, is_intersection=False)

    def __rpow__(self, other):
        return elementwise(other, self, jax.lax.pow, is_intersection=False)

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
        # Zero-preserving on the fill (-0 == 0): ``_map_fill`` keeps a None fill
        # None so a negation inside jit doesn't force the next matmul off the
        # fast path.
        return self.copy(
            scalar_mult=-self.scalar_mult,
            fill_value=_map_fill(self.fill_value, lambda f: -f),
        )

    def __pos__(self):
        return self.copy()

    def _map_unary(self, g):
        """Apply a zero-preserving unary ``g`` to ``val`` / ``scalar_mult`` /
        ``fill_value`` (``g(0) == 0`` for cast / conj / real / imag / abs / round),
        keeping the ``None`` fill fast-path marker via ``_map_fill``. Callers wrap
        with ``@_on_materialized`` when ``g`` is non-linear in ``val``. NOT for
        ``__neg__`` (linear: scales scalar_mult only, leaves val lazy) or
        ``__invert__`` (maps val only)."""
        return self.copy(
            val=g(self.val) if self.val is not None else None,
            scalar_mult=g(self.scalar_mult),
            fill_value=_map_fill(self.fill_value, g),
        )

    # Non-linear in val → @_on_materialized densifies compressed storage first
    # (abs(a+b) != abs(a)+abs(b); a set buffer stores the pre-combination
    # per-side blocks). __neg__ / __pos__ are linear and stay lazy.
    @_on_materialized
    def __abs__(self):
        return self._map_unary(jnp.abs)

    @_on_materialized
    def __invert__(self):
        return self.copy(
            val=jax.lax.bitwise_not(self.val) if self.val is not None else None
        )

    @_on_materialized
    def __round__(self, ndigits=None):
        # round(x) with no arg → ndigits=None; numpy rounds to 0 decimals there
        # (jnp.round rejects None), so normalise.
        nd = 0 if ndigits is None else ndigits
        return self._map_unary(lambda v: jnp.round(v, nd))


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
    fill_value: Array | None  # None ⇒ statically-zero fill (fast-path marker)
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

        # ``fill_value is None`` is the canonical "statically-zero fill" marker:
        # it lands in the pytree treedef (static aux), so it survives jit and
        # lets matmul / elementwise branch onto the tiled fast path at trace
        # time without re-probing a (possibly traced) value. A concrete array —
        # even one that happens to equal 0 at runtime — is treated as "maybe
        # non-zero" and takes the densify path. So a defaulted fill stays
        # ``None`` rather than materializing a concrete ``jnp.array(0)``.
        self.out_dims = tuple(out_dims)
        self.primal_dims = tuple(primal_dims)
        self.val = val
        self.scalar_mult = scalar_mult
        self.fill_value = fill_value
        self.pre_transforms = tuple(pre_transforms)
        self.post_transforms = tuple(post_transforms)

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
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # ``fill_value`` is a child: when it is None (statically-zero fill) the
        # None lives in the treedef, so the static fast-path distinction is
        # preserved across flatten/unflatten without a separate aux field.
        val, scalar_mult, fill_value = children
        (
            out_dims,
            primal_dims,
            pre_transforms,
            post_transforms,
            dynamic_kwargs,
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
        # ``dense_for_matmul`` is the single Array-producing densifier (fusion
        # fast paths, with a ``dense(hard=True)`` fallback for shapes they don't
        # cover) — see the densification map in ``ops/dense.py``. It consumes
        # only Dense/Diagonal, so first strip any compressed (BandedIndex /
        # SetIndex) dims (no-op when there are none).
        from graphax.sparse.ops.dense import dense_for_matmul
        from graphax.sparse.ops.utils import _compressed_dims, _densify_compressed_dims
        t = _densify_compressed_dims(self) if _compressed_dims(self) else self
        return dense_for_matmul(t)

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

    @property
    def _eff_fill(self) -> Array:
        """The fill as a concrete array: ``fill_value`` itself, or a zero scalar
        when ``fill_value is None`` (the statically-zero marker). Used wherever
        the fill is consumed as a *value* (reductions, densify); the matmul /
        elementwise fast-path dispatch instead tests ``fill_value is None``."""
        return self.fill_value if self.fill_value is not None else jnp.zeros((), self.dtype)

    def copy(
        self,
        val: Array | None = None,
        scalar_mult: Array | None = None,
        fill_value=_KEEP,
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
    @property
    def _structural_val_size(self) -> int:
        """Number of on-structure (non-fill) cells — the size ``val`` carries, or
        WOULD carry if materialized for a ``val is None`` tensor.

        For ``val`` present this is just ``val.size``. For ``val is None`` (the
        structure is all-ones — same reading as ``dense()``), it is
        ``size // ∏ N_pair``: each sparse pair's two sides both carry the shared
        meta count ``N``, so the logical ``size`` holds ``N²`` per pair while the
        stored diagonal holds only ``N`` — divide it back out once per pair. A
        fully-dense ``val is None`` tensor has no pairs ⇒ every cell is structure
        (all ones); a diagonal pair contributes its ``N·B_row·B_col`` blocks."""
        if self.val is not None:
            return self.val.size
        n = self.size
        seen = set()
        for d in self.dims:
            if d.is_sparse:
                key = frozenset((d.id, d.other_id))
                if key not in seen:
                    seen.add(key)
                    n //= d.size
        return n

    @property
    def _n_fill_cells(self) -> int:
        """Number of implicit fill cells = logical size minus on-structure cells.
        ``0`` ⇒ ``val`` covers every cell, so ``fill_value`` must NOT enter a
        reduction (a fully-dense tensor has no off-block-diagonal fill positions)."""
        return self.size - self._structural_val_size

    def _stored_val(self) -> Array:
        """The on-structure values as a concrete array: ``val`` itself, or — for a
        structural ``val is None`` tensor — ``ones`` of the would-be stored size
        (``_structural_val_size``). ``val=None`` means the structure is all-ones
        (matching ``dense()``), so reductions fold those ones in rather than
        treating every cell as fill."""
        if self.val is not None:
            return self.val
        return jnp.ones(self._structural_val_size, dtype=self.dtype)

    def _reduce(self, reduce_fn, fold_fn, *, weighted: bool = False) -> Array:
        """Shared skeleton for all/any/sum/prod (NOT max/min — see ``_extremum``,
        which can't seed an identity without introducing ``-inf``).

        Reduce the scaled on-structure values (``_stored_val()`` — ``val``, or
        ``ones`` when ``val is None``; ``_stored_val`` uses the tensor's own dtype
        so an integer tensor stays integer), then fold the scaled implicit fill
        via ``fold_fn(reduced, scaled_fill, n_fill)``:

        * ``weighted`` (sum/prod): the fold weights by the implicit-cell count
          (``+ f*n`` / ``* f**n``) and is branchless — it vanishes when ``n_fill
          == 0``.
        * otherwise (all/any): an idempotent fold applied only when fill cells
          exist; ``fold_fn`` maps the fill's truthiness itself."""
        val_part = reduce_fn(_scaled_mul(self._stored_val(), self.scalar_mult))
        scaled_fill = _scaled_mul(self._eff_fill, self.scalar_mult)
        # weighted (sum/prod) always folds (branchless, vanishes when n_fill==0);
        # all/any fold only when fill cells exist.
        if weighted or self._n_fill_cells > 0:
            return fold_fn(val_part, scaled_fill, self._n_fill_cells)
        return jnp.asarray(val_part)

    @_on_materialized
    def all(self) -> Array:
        return self._reduce(jnp.all, lambda v, f, n: jnp.logical_and(v, f != 0))

    @_on_materialized
    def any(self) -> Array:
        return self._reduce(jnp.any, lambda v, f, n: jnp.logical_or(v, f != 0))

    @_on_materialized
    def sum(self) -> Array:
        return self._reduce(jnp.sum, lambda v, f, n: v + f * n, weighted=True)

    @_on_materialized
    def prod(self) -> Array:
        return self._reduce(jnp.prod, lambda v, f, n: v * f ** n, weighted=True)

    def _extremum(self, reduce_fn, fold_fn) -> Array:
        """Shared skeleton for max()/min(): scale BEFORE the extremum (a negative
        scalar_mult reverses order) over the on-structure values (``_stored_val``
        — ``val``, or ``ones`` when ``val is None``), then fold the scaled fill
        when implicit fill cells exist. A tensor with NO on-structure cells (an
        empty / pure-fill tensor) has no extremum to take, so it IS the scaled
        fill."""
        val = self._stored_val()
        if val.size == 0:
            return _scaled_mul(self._eff_fill, self.scalar_mult)
        m = reduce_fn(_scaled_mul(val, self.scalar_mult))
        if self._n_fill_cells > 0:
            m = fold_fn(m, _scaled_mul(self._eff_fill, self.scalar_mult))
        return m

    @_on_materialized
    def max(self) -> Array:
        return self._extremum(jnp.max, jnp.maximum)

    @_on_materialized
    def min(self) -> Array:
        return self._extremum(jnp.min, jnp.minimum)

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

    # astype / conj / real / imag are zero-preserving on the fill
    # (cast(0)=conj(0)=real(0)=imag(0)=0), so ``_map_unary`` carries the source's
    # zero-fill marker through — otherwise a cast/conj inside jit would drop
    # fast-path eligibility.
    def astype(self, dtype: DTypeLike, **kwargs) -> SparseTensor:
        return self._map_unary(lambda v: v.astype(dtype, **kwargs))

    def conj(self) -> SparseTensor:
        return self._map_unary(jnp.conj)

    def conjugate(self) -> SparseTensor:
        return self.conj()

    @property
    def real(self) -> SparseTensor:
        return self._map_unary(jnp.real)

    @property
    def imag(self) -> SparseTensor:
        return self._map_unary(jnp.imag)

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
        # device_put preserves values (incl. zero), so _map_unary keeps a None
        # fill None — a statically-zero fill has no data to move, and the old
        # ``jax.device_put(None, device)`` dropped that fast-path marker.
        return self._map_unary(lambda v: jax.device_put(v, device))

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


def _subdivide_coupled_blockdiag(
    st, is_out1, rel_i, d1, is_out2, rel_j, d2, factor,
):
    """Subdivide an ALREADY coupled block-diagonal pair ``(d1, d2)`` into ``factor``
    (a MULTIPLE of the current meta count) finer meta-diagonal blocks.

    ``d1`` / ``d2`` are the coupled ``DiagonalIndex`` pair: a shared meta axis
    (``d1.axis == d2.axis``, size ``N = d1.size``) and per-side block axes
    (``d1.block_axis`` size ``b1``, ``d2.block_axis`` size ``b2``). With
    ``k = factor // N`` (``k | b1``, ``k | b2``) each ``b1 x b2`` block is further
    block-diagonalised into ``k`` meta-diagonal ``b1' x b2'`` sub-blocks. The new
    meta axis is ``N * k = factor``. Byte-identical to producing ``factor`` blocks
    straight from the dense form (verified vs the dense-mask oracle, diff 0).

    ``val is None`` (uniform-ones) stays ``val is None``. Otherwise a pure reshape
    + one ``eye``-einsum on the two block axes — no gather, no densify.
    """
    N = d1.size
    k = factor // N
    b1 = d1.block_size or 1
    b2 = d2.block_size or 1
    if b1 % k != 0 or b2 % k != 0:
        raise ValueError(
            f"Diag: cannot subdivide block sizes ({b1}, {b2}) by k={k} "
            f"(factor={factor} over current meta={N}); block must be divisible."
        )
    b1n, b2n = b1 // k, b2 // k

    meta_ax = d1.axis if d1.axis is not None else d2.axis
    b1_ax = d1.block_axis
    b2_ax = d2.block_axis

    def _rebuild(new_d1, new_d2, new_val, *, moved=None, n_lead=3):
        def _shift(p):
            if p is None or moved is None:
                return p
            # ``new_val`` carries ``n_lead`` leading axes: the grown meta axis
            # (``N*k``) plus one physical axis per block side that is BOTH present
            # in the INPUT val AND still > 1 after the ``k`` split. An IMPLICIT
            # (broadcast, block_axis=None) block never had a val axis, and a block
            # that collapses to size 1 is dropped -- neither occupies a leading
            # axis. The count is therefore passed in, not hard-coded: the old flat
            # ``3`` over-counted whenever a coupled side was implicit, landing every
            # surviving dim's axis one-or-more past its own data (the ViT reshape
            # storm / size-32-on-a-size-1-axis densify bug).
            n_before = sum(1 for a in range(p) if a not in moved)
            return n_lead + n_before

        def _map(d, slot_is_out, slot_rel):
            if slot_is_out == is_out1 and slot_rel == rel_i:
                return new_d1
            if slot_is_out == is_out2 and slot_rel == rel_j:
                return new_d2
            if moved is None:
                return d
            na = _shift(getattr(d, "axis", None))
            if d.is_sparse:
                nb = _shift(getattr(d, "block_axis", None))
                return replace(d, axis=na, block_axis=nb)
            return replace(d, axis=na)

        new_out = tuple(_map(d, True, p) for p, d in enumerate(st.out_dims))
        new_primal = tuple(_map(d, False, p) for p, d in enumerate(st.primal_dims))
        return SparseTensor(
            new_out, new_primal, new_val,
            scalar_mult=st.scalar_mult, fill_value=st.fill_value,
            check_consistency=False,
        )

    if st.val is None:
        new_d1 = DiagonalIndex(
            id=d1.id, size=factor, axis=d1.axis, other_id=d2.id,
            block_size=b1n if b1n > 1 else None,
            block_axis=d1.block_axis if b1n > 1 else None,
        )
        new_d2 = DiagonalIndex(
            id=d2.id, size=factor, axis=d2.axis, other_id=d1.id,
            block_size=b2n if b2n > 1 else None,
            block_axis=d2.block_axis if b2n > 1 else None,
        )
        return _rebuild(new_d1, new_d2, None)

    val = st.val
    # Only the PHYSICALLY-present axes live in the buffer. Either coupled side may
    # carry a purely IMPLICIT (broadcast) block (block_size > 1, block_axis=None),
    # and the shared meta axis itself may be IMPLICIT (axis=None) -- an implicit
    # meta densifies as ``N`` IDENTICAL diagonal blocks, so the buffer stores a
    # single block with no meta axis. The old code reshaped as
    # ``[N, k, b1n, k, b2n]`` unconditionally (assuming meta AND both blocks were
    # materialised), over-counting the buffer whenever any of them was implicit --
    # the ViT ``cannot reshape (4, 2) into [4, 2, 1, 2, 1]`` (implicit block) and
    # the reshape-by-factor-N mismatch (implicit meta) crashes.
    p1 = b1_ax is not None
    p2 = b2_ax is not None
    pm = meta_ax is not None
    moved = (meta_ax, b1_ax, b2_ax)
    front = [a for a in moved if a is not None]
    rest = [a for a in range(val.ndim) if a not in front]
    v = jnp.transpose(val, front + rest)  # ([N], [b1], [b2], *rest)
    rest_shape = list(v.shape[len(front):])
    if not pm:
        # IMPLICIT meta: materialise the shared meta axis by broadcasting the
        # single stored block into ``N`` identical diagonal blocks. This restores
        # the canonical (N, [b1], [b2], *rest) leading layout so the split logic
        # below is identical to the meta-present case; every ``rest`` axis keeps
        # its original index (the synthetic meta axis is not one of them), so the
        # ``_shift`` bookkeeping over ``moved`` stays correct.
        v = jnp.broadcast_to(v[None, ...], (N,) + v.shape)

    # A block side occupies a NEW leading val axis iff it is present AND still
    # bigger than 1 after the split; otherwise it is implicit / collapses away.
    keep1 = p1 and b1n > 1
    keep2 = p2 and b2n > 1
    out_lead = [N * k] + ([b1n] if keep1 else []) + ([b2n] if keep2 else [])
    n_lead = len(out_lead)

    if p1 and p2:
        # Both block axes materialised: split each into (k, bn) and keep the
        # meta-diagonal (ki == kj == g) sub-block via the eye-einsum.
        v = v.reshape([N, k, b1n, k, b2n] + rest_shape)  # (N, ki, b1n, kj, b2n, *rest)
        eye = jnp.eye(k, dtype=v.dtype)
        # sub[N, g, r, c, *rest] = sum_{ki,kj} eye[g,ki] eye[g,kj] v[N,ki,r,kj,c,*rest]
        sub = jnp.einsum("gi,gj,nirjc...->ngrc...", eye, eye, v)  # (N, k, b1n, b2n, *rest)
        new_val = sub.reshape(out_lead + rest_shape)  # (factor, [b1n], [b2n], *rest)
    elif p1 or p2:
        # Exactly one block axis is materialised; its ``k`` split IS the new meta
        # sub-index (the finer diagonal), and the implicit side stays broadcast.
        # No eye-einsum: contracting eye[g, kj] against a val that is CONSTANT over
        # the implicit ``kj`` just reselects the same (broadcast) value, so masking
        # the present side and relabelling ``k`` into the meta axis is sufficient.
        bn = b1n if p1 else b2n
        v = v.reshape([N, k, bn] + rest_shape)  # (N, k, bn, *rest)
        new_val = v.reshape(out_lead + rest_shape)  # (factor, [bn], *rest)
    else:
        # Both blocks implicit: the meta block is CONSTANT over b1 x b2, so each of
        # the ``k`` finer meta-diagonal sub-blocks carries that same constant. Grow
        # the physical meta axis N -> N*k by repeating; both sub-blocks stay implicit.
        v = jnp.broadcast_to(v.reshape([N, 1] + rest_shape), [N, k] + rest_shape)
        new_val = v.reshape(out_lead + rest_shape)  # (factor, *rest)

    new_d1 = DiagonalIndex(
        id=d1.id, size=factor, axis=0, other_id=d2.id,
        block_size=b1n if b1n > 1 else None,
        block_axis=(1 if keep1 else None),
    )
    new_d2 = DiagonalIndex(
        id=d2.id, size=factor, axis=0, other_id=d1.id,
        block_size=b2n if b2n > 1 else None,
        block_axis=((1 + (1 if keep1 else 0)) if keep2 else None),
    )
    return _rebuild(new_d1, new_d2, new_val, moved=moved, n_lead=n_lead)


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
        # Neither paired dim has a physical val axis to carve the ``factor``
        # blocks from. A PURE-DIAGONAL implicit pair (its own block-diagonal) is
        # intercepted upstream in ``apply_diag`` as a no-op; reaching here with
        # ``size > 1`` means an UNrepresentable block split (block_size>1,
        # block_axis=None ⇒ a malformed, un-densifiable dim). Signal it as a
        # per-edge geometry miss so the caller's best-effort loop skips this
        # transform instead of fabricating a corrupt edge.
        if size > 1 and (b1 > 1 or b2 > 1):
            raise ValueError(
                "Diag: cannot block-diagonalise an implicit (axis=None) pair "
                "that is not a pure diagonal — no physical axis to carve blocks."
            )
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
    # Definitionally zero-fill → fill_value=None keeps the fast-path marker.
    return _copy(st, jnp.zeros_like(st.val), jnp.array(1.0), fill_value=None)
