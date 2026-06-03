"""Shared helpers used across the ``ops`` package.

* Type / consistency: ``_is_sparse``, ``_assert_sparse_tensor_consistency``.
* Shared primitives (used by both elementwise and matmul):
    ``_val_or_one``, ``_prepare_physical_array``, ``_is_zero_fill``.
* Construction / mutation: ``_arr2st``, ``_copy``.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from graphax.sparse.indexes import Index, DenseIndex, DiagonalIndex

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


# Lazily resolved on first ``_is_sparse`` call to dodge a circular import
# (``graphax.sparse.tensor`` imports from this module). After resolution every
# subsequent ``_is_sparse`` is a single ``isinstance`` instead of a string
# compare on ``type(obj).__name__``.
_SparseTensor = None


def _get_sparse_tensor_cls():
    global _SparseTensor
    if _SparseTensor is None:
        from graphax.sparse.tensor import SparseTensor
        _SparseTensor = SparseTensor
    return _SparseTensor


def _is_sparse(obj) -> bool:
    cls = _get_sparse_tensor_cls()
    if isinstance(obj, cls):
        return True
    if isinstance(obj, tuple) and len(obj) > 0:
        return isinstance(obj[0], cls)
    return False


# --- Shared primitives (elementwise + matmul) ----------------------------


def _compressed_dims(tensor) -> list:
    """The ``out_dims`` + ``primal_dims`` that are compressed Index types
    (``BandedIndex`` / ``SetIndex``). Empty when the tensor carries no
    compressed structure — the common case, for which the densify helpers
    below are no-ops."""
    return [d for d in (*tensor.out_dims, *tensor.primal_dims)
            if getattr(d, "is_compressed", False)]


def _densify_compressed_dims(tensor, compact: bool = False):
    """Replace a tensor's compressed Index dims (``BandedIndex`` / ``SetIndex``)
    with their densified ``DenseIndex`` / ``DiagonalIndex`` equivalents, writing
    the expanded data into ``val``.

    * ``compact=False`` (for ``dense()``): full materialization — each
      compressed pair becomes a ``DenseIndex`` pair over the dense
      ``(rows, cols)`` block.
    * ``compact=True`` (for ``_materialize_for_op``): the meta-block-diagonal
      form when the pair ``reduces_to_diagonal`` (``SetIndex`` always;
      ``BandedIndex`` when W=1 identity), yielding a ``DiagonalIndex`` pair at
      ``M×`` lower storage; full dense otherwise.

    No-op (returns ``tensor`` unchanged) when there are no compressed dims.

    Currently handles the single-compressed-pair (K=1) banded case whose
    ``val`` is the canonical ``(n_meta*M_p, W, B_row, B_col, *L)`` Array.
    Multi-axis (K≥2) and the ``SetIndex`` dual-buffer ``val`` are wired with
    their producers (Phase 8.E / 8.F).
    """
    comp = _compressed_dims(tensor)
    if not comp:
        return tensor

    from graphax.sparse.tensor import SparseTensor
    from graphax.sparse.indexes import BandedIndex, SetIndex, DenseIndex, DiagonalIndex

    banded = [d for d in comp if isinstance(d, BandedIndex)]
    fill = tensor.fill_value

    # Only a *pure* compressed pair (the whole tensor is one out + one primal
    # compressed dim) is handled — the branches below index ``dims[0]`` directly.
    # A compressed pair coexisting with extra Dense/Diagonal dims would mislabel
    # those, so it falls through to the explicit ``NotImplementedError``.
    pure_pair = len(tensor.out_dims) == 1 and len(tensor.primal_dims) == 1

    # --- K=1 SetIndex pair (combined 1-D Array val) ---
    set_dims = [d for d in comp if isinstance(d, SetIndex)]
    if pure_pair and len(set_dims) == 2 and isinstance(tensor.val, jax.Array):
        sx = tensor.out_dims[0]
        o, p = tensor.out_dims[0], tensor.primal_dims[0]
        if compact:  # SetIndex always reduces to a meta-block-diagonal
            meta = sx.to_meta_blocks(tensor.val, fill)  # (M, LCM_h, LCM_w, *L)
            M, H, W = meta.shape[0], meta.shape[1], meta.shape[2]
            new_out = (DiagonalIndex(o.id, M, 0, p.id, H, 1),)
            new_primal = (DiagonalIndex(p.id, M, 0, o.id, W, 2),)
            new_val = meta
        else:
            dense = sx.densify_axis(tensor.val, fill)  # (M*LCM_h, M*LCM_w, *L)
            rows, cols = dense.shape[0], dense.shape[1]
            new_out = (DenseIndex(o.id, rows, 0),)
            new_primal = (DenseIndex(p.id, cols, 1),)
            new_val = dense
        return SparseTensor(
            new_out, new_primal, new_val,
            scalar_mult=tensor.scalar_mult, fill_value=tensor.fill_value,
            check_consistency=False,
            zero_fill=getattr(tensor, "_zero_fill", None),
        )

    # --- K=1 banded pair (Array val) ---
    if pure_pair and len(banded) == 2 and isinstance(tensor.val, jax.Array):
        primary = next((d for d in banded if d.primary), banded[0])
        o, p = tensor.out_dims[0], tensor.primal_dims[0]
        if compact and primary.reduces_to_diagonal():
            meta = primary.to_meta_blocks(tensor.val)  # (n_meta*M, B_row, B_col, *L)
            M, B_row, B_col = meta.shape[0], meta.shape[1], meta.shape[2]
            new_out = (DiagonalIndex(o.id, M, 0, p.id, B_row, 1),)
            new_primal = (DiagonalIndex(p.id, M, 0, o.id, B_col, 2),)
            new_val = meta
        else:
            dense = primary.densify_axis(tensor.val, fill)  # (rows, cols, *L)
            rows, cols = dense.shape[0], dense.shape[1]
            new_out = (DenseIndex(o.id, rows, 0),)
            new_primal = (DenseIndex(p.id, cols, 1),)
            new_val = dense
        return SparseTensor(
            new_out, new_primal, new_val,
            scalar_mult=tensor.scalar_mult, fill_value=tensor.fill_value,
            check_consistency=False,
            zero_fill=getattr(tensor, "_zero_fill", None),
        )

    # --- K≥2 SetIndex (multi-axis dual block-diagonal buffers) ---
    K = len(tensor.out_dims)
    if (set_dims and len(set_dims) == 2 * K and K >= 2
            and isinstance(tensor.val, jax.Array)
            and all(isinstance(d, SetIndex) for d in tensor.out_dims)
            and all(isinstance(d, SetIndex) for d in tensor.primal_dims)):
        from graphax.sparse.ops.block_storage import (
            _densify_multi_banded, BandAxisSpec,
        )

        sx = tensor.out_dims[0]
        lhs_blocks, rhs_blocks = sx._split(tensor.val)  # W=1 multi-banded buffers
        # Each per-side block-diagonal is a W=1 multi-banded buffer; densify it
        # via the shared band kernel, then combine the two sides with the op.
        def _specs(shape):
            return tuple(
                BandAxisSpec(band_width=1, block_row=shape[2 * K + i],
                             block_col=shape[3 * K + i], n_secondary=shape[2 * i])
                for i in range(K)
            )
        fill_lhs, fill_rhs = (fill if isinstance(fill, tuple) else (fill, fill))
        lhs_dense = _densify_multi_banded(lhs_blocks, _specs(sx.lhs_shape), fill_lhs)
        if rhs_blocks is not None:
            rhs_dense = _densify_multi_banded(rhs_blocks, _specs(sx.rhs_shape), fill_rhs)
            dense = sx._op()(lhs_dense, rhs_dense)
        else:
            dense = sx._op()(lhs_dense, fill_rhs)
        new_out = tuple(
            DenseIndex(tensor.out_dims[i].id, dense.shape[i], i) for i in range(K)
        )
        new_primal = tuple(
            DenseIndex(tensor.primal_dims[i].id, dense.shape[K + i], K + i)
            for i in range(K)
        )
        return SparseTensor(
            new_out, new_primal, dense,
            scalar_mult=tensor.scalar_mult, fill_value=tensor.fill_value,
            check_consistency=False,
            zero_fill=getattr(tensor, "_zero_fill", None),
        )

    # --- K≥2 banded (multi-axis interleaved Array val) ---
    if (banded and len(banded) == 2 * K and isinstance(tensor.val, jax.Array)
            and all(isinstance(d, BandedIndex) for d in tensor.out_dims)
            and all(isinstance(d, BandedIndex) for d in tensor.primal_dims)):
        from graphax.sparse.ops.block_storage import (
            _densify_multi_banded, BandAxisSpec,
        )

        specs = tuple(
            BandAxisSpec(
                primary_axis=0 if o.primary else 1,
                n_secondary=o.n_secondary, offset=o.offset, n_meta=o.n_meta,
                band_width=o.band_width, block_row=o.block_size,
                block_col=p.block_size,
            )
            for o, p in zip(tensor.out_dims, tensor.primal_dims)
        )
        dense = _densify_multi_banded(
            tensor.val, specs, fill,
        )  # (rows_0..rows_{K-1}, cols_0..cols_{K-1}, *L)
        new_out = tuple(
            DenseIndex(tensor.out_dims[i].id, dense.shape[i], i) for i in range(K)
        )
        new_primal = tuple(
            DenseIndex(tensor.primal_dims[i].id, dense.shape[K + i], K + i)
            for i in range(K)
        )
        return SparseTensor(
            new_out, new_primal, dense,
            scalar_mult=tensor.scalar_mult, fill_value=tensor.fill_value,
            check_consistency=False,
            zero_fill=getattr(tensor, "_zero_fill", None),
        )

    raise NotImplementedError(
        f"densify of compressed dims: unhandled shape — {len(comp)} compressed "
        f"dims across {len(tensor.out_dims)} out / {len(tensor.primal_dims)} "
        f"primal dims, val type {type(tensor.val).__name__}. Handled: a pure "
        "K=1 SetIndex / banded pair, or an all-BandedIndex K≥2 tensor. A "
        "compressed pair coexisting with Dense/Diagonal dims is not yet "
        "supported (no producer emits it today)."
    )


def _materialize_for_op(tensor):
    """Pre-densify a tensor's compressed Index dims to ``DiagonalIndex`` /
    ``DenseIndex`` so matmul / elementwise — which consume only those — never
    see a ``BandedIndex`` / ``SetIndex``. No-op when the tensor has no
    compressed dims (the case for every operand today, until the Phase 8.E/8.F
    producers land)."""
    if not _compressed_dims(tensor):
        return tensor
    return _densify_compressed_dims(tensor, compact=True)


def _val_or_one(tensor: SparseTensor) -> Array:
    """A tensor's stored ``val``, or a scalar 1 in its dtype if the tensor
    carries pure structure (``val is None``). ``val`` is always a plain
    ``Array`` post-Phase-8 (compressed structure lives in the dim ``Index``
    types, not in ``val``)."""
    val = tensor.val
    return val if val is not None else jnp.array(1.0, dtype=tensor.dtype)


def _prepare_physical_array(val: Array, axis_axes: Sequence[int | None]) -> Array:
    """Transpose ``val`` so that each entry of ``axis_axes`` (a flat list of source axiss,
    ``None`` for synthetic axes) lands at its position index in the result; trailing axes
    preserved in their original order. Used by both elementwise (per pair-axis) and matmul
    (per ``PairData`` triple)."""
    valid = tuple(i for i, v in enumerate(axis_axes) if v is not None and v < val.ndim)
    src = tuple(axis_axes[i] for i in valid)
    leftover = [v for v in range(val.ndim) if v not in src]
    full_src = src + tuple(leftover)
    full_tgt = valid + tuple(len(axis_axes) + i for i in range(len(leftover)))
    N = len(axis_axes) + len(leftover)
    if full_src == full_tgt and N == val.ndim:
        return val
    perm = [-1] * N
    for s, t in zip(full_src, full_tgt):
        perm[t] = s
    ones_idx = val.ndim
    for i in range(N):
        if perm[i] == -1:
            perm[i] = ones_idx; ones_idx += 1
    if ones_idx > val.ndim:
        val = val.reshape(val.shape + (1,) * (ones_idx - val.ndim))
    if perm != list(range(N)):
        val = val.transpose(perm)
    return val


def _is_zero_fill(tensor: SparseTensor) -> bool:
    """True iff ``tensor.fill_value`` is statically known to be zero.

    Reads the cached ``_zero_fill`` flag set at ``SparseTensor`` construction
    time and propagated through the pytree's static aux_data. This flag is
    available even inside ``jit`` (where the actual ``fill_value`` becomes a
    tracer with no concrete value), letting matmul / elementwise statically
    branch between the fast tiled path (``fill = 0``) and the densify fallback
    (``fill ≠ 0``) without paying for a runtime check.

    Falls back to inspecting ``fill_value`` directly when the flag is missing
    (e.g. a tensor produced before the static-flag mechanism was added)."""
    flag = getattr(tensor, "_zero_fill", None)
    # Honour any explicit boolean — the cache exists precisely to avoid
    # re-probing the (possibly traced) ``fill_value``. Only fall through
    # when the flag is genuinely unset (``None``).
    if flag is not None:
        return bool(flag)
    fv = tensor.fill_value
    try:
        return bool(np.all(np.asarray(fv) == 0))
    except (TypeError, ValueError, AttributeError,
            jax.errors.TracerArrayConversionError,
            jax.errors.ConcretizationTypeError):
        return False


# --- Consistency checks --------------------------------------------------
def _check_sparse_dim_pair(d, dim_map):
    other = dim_map.get(d.other_id)
    return (other.is_sparse and other.other_id == d.id and d.size == other.size)


def _check_block_axis(d, dim_map, block_axiss):
    if d.block_axis in block_axiss:
        other = dim_map.get(d.other_id)
        if not (other.is_sparse and other.block_axis == d.block_axis):
            raise ValueError(
                f"Topology Error: Duplicate block_axis {d.block_axis} in DiagonalIndex {d.id}"
            )
    block_axiss.add(d.block_axis)


def _assert_sparse_tensor_consistency(st: SparseTensor):
    # Raise (not ``assert``) so the invariant survives ``python -O`` /
    # ``PYTHONOPTIMIZE`` — every downstream op assumes contiguous IDs and
    # paired DiagonalIndex links, and silently dropping the check has caused
    # wrong-shape Jacobians in the past.
    dim_ids = [d.id for d in st.dims]
    if set(dim_ids) != set(range(len(dim_ids))):
        raise ValueError(
            f"Topology Error: Index IDs must be a contiguous sequence. Got {dim_ids}"
        )
    dim_map = {d.id: d for d in st.dims}
    block_axiss = set()
    for d in st.dims:
        if d.is_sparse:
            if not _check_sparse_dim_pair(d, dim_map):
                raise ValueError(
                    f"Topology Error: Invalid sparse dimension pair configuration for dimension {d.id}"
                )
            if getattr(d, "block_axis", None) is not None:
                _check_block_axis(d, dim_map, block_axiss)


# --- Construction / mutation --------------------------------------------
def _copy(st: SparseTensor, val: Array | None = None, scalar_mult: Array | None = None,
          fill_value: Array | None = None, out_dims: Sequence[Index] | None = None,
          primal_dims: Sequence[Index] | None = None, deep: bool = False):
    from graphax.sparse.tensor import SparseTensor
    s = scalar_mult if scalar_mult is not None else st.scalar_mult
    f = fill_value if fill_value is not None else st.fill_value
    od = out_dims if out_dims is not None else st.out_dims
    pd = primal_dims if primal_dims is not None else st.primal_dims
    v = val if val is not None else st.val
    if deep:
        v = copy.deepcopy(v) if v is not None else None
        s = copy.deepcopy(s); od = copy.deepcopy(od); pd = copy.deepcopy(pd)
    # Preserve the source's zero-fill flag whenever ``fill_value`` is unchanged
    # — otherwise inside jit a fresh tracer would force the constructor to
    # conservatively report ``False`` and we'd lose the fast-path eligibility.
    zf = getattr(st, "_zero_fill", None) if fill_value is None else None
    return SparseTensor(
        od, pd, v,
        scalar_mult=s, fill_value=f,
        pre_transforms=st.pre_transforms,
        post_transforms=st.post_transforms,
        check_consistency=False,
        zero_fill=zf,
    )


def _arr2st(arr: Array, out_ndim: int | None = None, dtype: Any = None, **kwargs: Any) -> SparseTensor:
    # Surface the bug instead of silently flipping to 0. Callers like
    # ``matmul._normalize_inputs`` compute ``out_ndim`` as
    # ``lhs.ndim - len(rhs.out_dims)`` which can go negative when the
    # operand shapes don't line up for a matmul.
    if out_ndim is not None and out_ndim < 0:
        raise ValueError(f"_arr2st out_ndim must be non-negative, got {out_ndim}")
    from graphax.sparse.tensor import SparseTensor
    if dtype is not None:
        arr = arr.astype(dtype)
    if out_ndim is None:
        out_ndim = arr.ndim // 2
    if arr.ndim == 0:
        arr = jnp.expand_dims(arr, 0)
    dims = tuple(DenseIndex(i, s, i) for i, s in enumerate(arr.shape))
    return SparseTensor(dims[:out_ndim], dims[out_ndim:], arr,
                        check_consistency=False, **kwargs)


# --- graphax-specific extensions ----------------------------------------
# These helpers are used by graphax/primitives/{transforms,reductions}. They
# are not part of the matmul project; preserved here so external graphax
# callers keep working.
def _materialize_indexes(st: "SparseTensor", dims: Sequence[int]) -> Array:
    """Materialize ``st.val`` along the given (sparse) dimensions by inserting
    fresh broadcast axes."""
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


def _swap_back_axes(st: "SparseTensor") -> "SparseTensor":
    """Restore the canonical val-axis order: dims with ``axis`` set come first
    in dim order, then any sparse-pair ``block_axis``, then any leftover
    physical axes. Updates the ``axis``/``block_axis`` fields to match."""
    if st.val is None:
        return st

    i = 0
    permutation = [0] * st.val.ndim
    for d in st.dims:
        if d.axis is not None:
            if not d.is_sparse or d.id < getattr(d, "other_id", float("inf")):
                permutation[i] = d.axis
                i += 1
        if d.is_sparse and getattr(d, "block_axis", None) is not None:
            permutation[i] = d.block_axis
            i += 1

    seen = set(permutation[:i])
    for j in range(st.val.ndim):
        if j not in seen:
            permutation[i] = j
            i += 1

    # Sanity guard — the loops above must produce a valid permutation of
    # ``range(val.ndim)``. The previous ``i < len(permutation)`` clamp would
    # silently leave initial 0s in trailing slots when val carried physical
    # axes not described by any Index, producing a transpose that duplicated
    # axis 0.
    assert sorted(permutation) == list(range(st.val.ndim)), (
        f"_swap_back_axes produced invalid permutation {permutation}"
    )

    new_val = st.val.transpose(permutation)

    i = 0
    dim_map = {d.id: d for d in st.dims}
    processed_ids: dict[int, Index] = {}

    def update_dim(d, current_i: int):
        if d.id in processed_ids:
            return processed_ids[d.id], current_i

        nv, nb = d.axis, getattr(d, "block_axis", None)
        if nv is not None:
            if not d.is_sparse or d.id < d.other_id:
                nv = current_i
                current_i += 1
            else:
                other = dim_map[d.other_id]
                nv = processed_ids[other.id].axis

        if nb is not None:
            nb = current_i
            current_i += 1

        new_d = replace(d, axis=nv)
        if new_d.is_sparse:
            new_d = replace(new_d, block_axis=nb)

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
