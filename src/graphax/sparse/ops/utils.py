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

import jax.numpy as jnp
import numpy as np
from jax import Array

from graphax.sparse.indexes import Index, DenseIndex
from graphax.sparse.dtype_compute import _compute_dtype, _scaled_mul

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


# Shared "keep the source's value" sentinel for the optional ``fill_value``
# override on ``_copy``: passed verbatim
# ⇒ reuse the source's fill; any other value — INCLUDING ``None`` (the
# statically-zero marker) — is used as the explicit override. (A plain ``None``
# default would be ambiguous now that ``None`` is a meaningful fill.)
_KEEP = object()


# --- Shared primitives (elementwise + matmul) ----------------------------
def _apply_scalar_mult(value: Array, tensor) -> Array:
    """Scale ``value`` by ``tensor.scalar_mult`` — bitwise-and (with a bool-cast
    mult) for bool tensors, multiply otherwise. The single definition of "apply
    this operand's scalar_mult to a buffer", shared by elementwise + matmul."""
    if tensor.dtype == jnp.bool_:
        return value & tensor.scalar_mult.astype(jnp.bool_)
    # Highest-common-dtype multiply: a Quant'd narrow ``value`` (float8 /
    # sub-byte int / …) has no implicit promotion path with a float32
    # scalar_mult, so upcast both to their common dtype before the op.
    return _scaled_mul(value, tensor.scalar_mult)


def _scaled_fill(tensor) -> Array:
    """Post-scaled fill as a concrete array: ``fill * scalar_mult`` (or ``& mask``
    for bool), with a ``None`` (statically-zero) fill read as 0 via ``_eff_fill``.
    Canonical form used to compose output fills consistently — every fast path
    must produce a fill that matches the post-scaled meaning of the input
    operands so downstream consumers see one definition."""
    return _apply_scalar_mult(tensor._eff_fill, tensor)





def _is_approx(st) -> bool:
    """True iff ``st`` carries an approximation structure that the downstream
    sparse contraction / drain cannot consume directly — a ``Diag`` block (a
    sparse dim with ``block_size > 1``) or a ``Compress`` implicit dim (a
    non-sparse logical dim with no physical axis and ``logical_size > 1``).

    This is the exact structure the rectangular-Diag / implicit-Compress gaps
    choke on. A pure-diagonal Diag (``block_size in {None, 1}``) and a plain
    dense edge both return False, so the cleanly-contractible fast path is
    preserved and the no-approximation EXACT-AD edge is never touched."""
    for d in (*st.out_dims, *st.primal_dims):
        if d.is_sparse and (getattr(d, "block_size", None) or 1) > 1:
            return True
        if (not d.is_sparse) and d.axis is None and int(d.logical_size) > 1:
            return True
    return False


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
    # NB: a source axis >= val.ndim is INTENTIONAL for an implicit dim whose
    # nominal axis lies beyond the compact val; it is treated as synthetic
    # (size-1) below, so the `v < val.ndim` filter is load-bearing.
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
    """True iff ``tensor.fill_value`` is STATICALLY known to be zero, i.e.
    ``fill_value is None`` — the canonical fast-path marker.

    ``None`` lives in the pytree treedef, so this is a compile-time test that
    holds inside ``jit`` (where a concrete ``fill_value`` would be an opaque
    tracer), letting matmul / elementwise branch between the tiled fast path
    (``fill = 0``) and the densify fallback without a runtime check. A concrete
    array — even one equal to 0 at runtime — is conservatively "maybe non-zero"
    and takes the densify path."""
    return tensor.fill_value is None


# --- Consistency checks --------------------------------------------------
def _check_sparse_dim_pair(d, dim_map):
    other = dim_map.get(d.other_id)
    # ``other`` is None when ``other_id`` names no dim in this tensor (an
    # unpaired sparse dim) — return False so the caller raises the intended
    # ValueError rather than an AttributeError on ``None.is_sparse``.
    return (other is not None and other.is_sparse
            and other.other_id == d.id and d.size == other.size)


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

    # Physical-buffer agreement. Every dim's ``axis`` /
    # ``block_axis`` must point at a val axis whose extent equals the dim's
    # declared ``size`` / ``block_size`` (a size-1 physical axis is allowed as a
    # broadcast/implicit stand-in). This catches metadata that disagrees with the
    # buffer -- e.g. a dense dim whose ``axis`` was renumbered onto a
    # DiagonalIndex block axis (the MoE-9 squeeze construction bug) -- LOUDLY at
    # construction, instead of as an opaque "transpose permutation isn't a
    # permutation" far downstream in a contraction.
    val = getattr(st, "val", None)
    if val is not None:
        try:
            vshape = tuple(val.shape)
        except Exception:
            vshape = None
        if vshape is not None:
            ndim = len(vshape)
            for d in st.dims:
                ax = d.axis
                if ax is not None:
                    if ax < 0 or ax >= ndim:
                        raise ValueError(
                            f"Topology Error: Index {d.id} axis={ax} out of range "
                            f"for val.ndim={ndim}"
                        )
                    if vshape[ax] != d.size and vshape[ax] != 1:
                        raise ValueError(
                            f"Topology Error: Index {d.id} axis={ax} declares "
                            f"size={d.size} but val.shape[{ax}]={vshape[ax]}"
                        )
                if d.is_sparse and getattr(d, "block_axis", None) is not None:
                    ba = d.block_axis
                    if ba < 0 or ba >= ndim:
                        raise ValueError(
                            f"Topology Error: Index {d.id} block_axis={ba} out of "
                            f"range for val.ndim={ndim}"
                        )
                    bsz = d.block_size
                    if bsz is not None and vshape[ba] != bsz and vshape[ba] != 1:
                        raise ValueError(
                            f"Topology Error: Index {d.id} block_axis={ba} declares "
                            f"block_size={bsz} but val.shape[{ba}]={vshape[ba]}"
                        )


def _squeeze_unreferenced_val_axes(st: "SparseTensor") -> "SparseTensor":
    """Drop physical ``val`` axes of size 1 that NO dim references.

    Repeated block-diagonal restructuring (``_apply_block_diagonal`` /
    ``_subdivide_coupled_blockdiag``) and ``apply_diag`` / ``apply_compress``
    INSERT a fresh physical axis per meta / block side and never fold the
    leftovers, so an approx edge accumulates a long tail of size-1 ``val`` axes
    that carry no data (measured: a ViT approx edge with logical rank 4 but
    ``val.ndim`` 11-12, the trailing axes all size 1). They are pure rank waste
    — every downstream op works off ``dim.axis`` / ``dim.block_axis`` pointers,
    not the raw ``val.ndim`` — and march the physical rank toward the numpy/XLA
    32-axis cap.

    Removing a size-1 axis that no ``dim.axis`` / ``dim.block_axis`` points at is
    a pure reshape (byte-identical: a size-1 axis contributes nothing to the
    buffer), after which every surviving dim's pointer is shifted down to its new
    position. A no-op — returns ``self`` unchanged, hence byte-identical — when
    there are no such axes, which is ALWAYS the case on the EXACT-AD path (the
    tiled ``_build_output_tensor`` already squeezes its output, and matmul /
    elementwise never emit unreferenced size-1 axes). Referenced size-1 axes (a
    genuinely size-1 dim carried explicitly) are kept."""
    val = getattr(st, "val", None)
    if val is None:
        return st
    try:
        ndim = val.ndim
    except Exception:
        return st
    referenced = set()
    for d in st.dims:
        if d.axis is not None:
            referenced.add(d.axis)
        if d.is_sparse and getattr(d, "block_axis", None) is not None:
            referenced.add(d.block_axis)
    drop = [a for a in range(ndim) if a not in referenced and int(val.shape[a]) == 1]
    if not drop:
        return st
    keep = [a for a in range(ndim) if a not in drop]
    shift = {old: new for new, old in enumerate(keep)}
    new_val = val.reshape(tuple(val.shape[a] for a in keep))

    def _remap(d):
        kw = {}
        if d.axis is not None:
            kw["axis"] = shift[d.axis]
        if d.is_sparse and getattr(d, "block_axis", None) is not None:
            kw["block_axis"] = shift[d.block_axis]
        return replace(d, **kw) if kw else d

    cls = _get_sparse_tensor_cls()
    return cls(
        tuple(_remap(d) for d in st.out_dims),
        tuple(_remap(d) for d in st.primal_dims),
        new_val,
        scalar_mult=st.scalar_mult,
        fill_value=st.fill_value,
        pre_transforms=st.pre_transforms,
        post_transforms=st.post_transforms,
        check_consistency=False,
    )


# --- Construction / mutation --------------------------------------------
def _copy(st: SparseTensor, val: Array | None = None, scalar_mult: Array | None = None,
          fill_value=_KEEP, out_dims: Sequence[Index] | None = None,
          primal_dims: Sequence[Index] | None = None, deep: bool = False):
    from graphax.sparse.tensor import SparseTensor
    s = scalar_mult if scalar_mult is not None else st.scalar_mult
    # ``fill_value`` carries the static-zero marker directly: ``_KEEP`` ⇒ reuse
    # the source's (preserving a ``None`` fast-path marker through jit); an
    # explicit value (incl. ``None``) overrides it.
    f = st.fill_value if fill_value is _KEEP else fill_value
    od = out_dims if out_dims is not None else st.out_dims
    pd = primal_dims if primal_dims is not None else st.primal_dims
    v = val if val is not None else st.val
    if deep:
        v = copy.deepcopy(v) if v is not None else None
        s = copy.deepcopy(s); od = copy.deepcopy(od); pd = copy.deepcopy(pd)
    return SparseTensor(
        od, pd, v,
        scalar_mult=s, fill_value=f,
        pre_transforms=st.pre_transforms,
        post_transforms=st.post_transforms,
        check_consistency=False,
    )


def _arr2st(arr: Array, out_ndim: int | None = None, dtype: Any = None, **kwargs: Any) -> SparseTensor:
    """Wrap a dense array as a plain ``DenseIndex`` SparseTensor with FRESH
    canonical ids ``range(0, ndim)`` (out side ``0..out_ndim-1``, primal the rest).

    ID-CONVENTION FOOTGUN: this assigns NEW ids and is NOT interchangeable with
    ``dispatch._to_dense_st``, which PRESERVES each dim's id so the id-based
    contraction resolver pairs the right axes. Use ``_arr2st`` only where the result
    STARTS a fresh nominal edge (canonical layout, e.g. a reconciled approx edge);
    use ``_to_dense_st`` inside a contraction where downstream id pairing must hold.
    Mixing them mis-aligns multi-edge merges (a real past bug — see core.py's
    removed fresh-id band-aid note).
    """
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


class NominalOrderViolation(ValueError):
    """An op returned ``out_dims`` / ``primal_dims`` whose logical extents are
    not in its operands' nominal order (ticket dsnn-3qm.71).

    The ops build their output dim lists from the operands' own dim order, so
    this is a postcondition of the op, not a property of the inputs. It RAISES
    (a ``ValueError`` subclass) rather than ``assert``-ing: ``python -O``
    deletes an ``assert``, and a contract that disappears under a flag is not
    a contract.
    """


def check_nominal_order(out, out_src, primal_src, where: str) -> None:
    """Raise :class:`NominalOrderViolation` unless ``out.out_dims`` carry the
    logical extents of ``out_src.out_dims`` in order and ``out.primal_dims``
    those of ``primal_src.primal_dims`` in order."""
    got_o = tuple(int(d.logical_size) for d in out.out_dims)
    exp_o = tuple(int(d.logical_size) for d in out_src.out_dims)
    got_p = tuple(int(d.logical_size) for d in out.primal_dims)
    exp_p = tuple(int(d.logical_size) for d in primal_src.primal_dims)
    if got_o != exp_o or got_p != exp_p:
        raise NominalOrderViolation(
            f"{where}: the result's dim order is not the operands' nominal "
            f"order -- out_dims {got_o} (operand {exp_o}), primal_dims "
            f"{got_p} (operand {exp_p})."
        )
