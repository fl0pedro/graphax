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

from graphax.sparse.indexes import Index, DenseIndex, DiagonalIndex, _split_fill
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
# override on ``_copy`` / ``_rewrap`` / ``_dense_pair_result``: passed verbatim
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





def _compressed_dims(tensor) -> list:
    """The ``out_dims`` + ``primal_dims`` that are compressed Index types
    (``BandedIndex`` / ``SetIndex``). Empty when the tensor carries no
    compressed structure — the common case, for which the densify helpers
    below are no-ops."""
    return [d for d in (*tensor.out_dims, *tensor.primal_dims)
            if getattr(d, "is_compressed", False)]


def _rewrap(tensor, out_dims, primal_dims, val, fill_value=_KEEP):
    """Re-emit ``tensor`` with new dims/val, carrying scalar_mult / fill_value /
    zero_fill (the shared tail of every densify branch).

    ``fill_value`` defaults to the source's, but SetIndex densify branches
    override it with the COMBINED ``op(fill_lhs, fill_rhs)``: the source carries
    a per-side fill (scalar or tuple), whereas the densified result's implicit
    cells hold the op-combined fill that ``densify_axis`` / ``to_meta_blocks``
    already stitch into the data (L3)."""
    from graphax.sparse.tensor import SparseTensor

    fv = tensor.fill_value if fill_value is _KEEP else fill_value
    return SparseTensor(
        out_dims, primal_dims, val,
        scalar_mult=tensor.scalar_mult, fill_value=fv,
        check_consistency=False,
    )


def _dense_pair_result(tensor, dense, K, fill_value=_KEEP):
    """Wrap a fully-materialized ``dense`` array as a K-pair ``DenseIndex``
    tensor (out axes 0..K-1, primal axes K..2K-1), reusing the source dim ids."""
    from graphax.sparse.indexes import DenseIndex

    out = tuple(DenseIndex(tensor.out_dims[i].id, dense.shape[i], i) for i in range(K))
    primal = tuple(
        DenseIndex(tensor.primal_dims[i].id, dense.shape[K + i], K + i)
        for i in range(K)
    )
    return _rewrap(tensor, out, primal, dense, fill_value=fill_value)


def _densify_toeplitz(tensor):
    """Densify a tensor whose compressed dims are ``ToeplitzIndex`` pairs (conv
    Jacobians), which may COEXIST with plain Dense/Diagonal dims — unlike the
    pure-pair banded/set path below.

    Each Toeplitz pair contracts its stored ``val`` axis (the kernel taps for
    ``d out/d lhs``, the input positions for ``d out/d rhs``) against the
    scatter-free windowed indicator ``M[p, q, k]`` and produces the pair's two
    roles as fresh dense axes. Dense dims ride through the same einsum; Diagonal
    (``axis is None``) dims are untouched (not stored in ``val``). One einsum
    per tensor — XLA fuses the iota-built ``M`` into it. The result is a plain
    Dense/Diagonal tensor, exactly what matmul / elementwise consume."""
    from graphax.sparse.indexes import ToeplitzIndex, DenseIndex

    val = tensor.val
    out_dims, primal_dims = list(tensor.out_dims), list(tensor.primal_dims)
    by_id = {d.id: d for d in (*out_dims, *primal_dims)}

    pool = iter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    val_letter = {a: next(pool) for a in range(val.ndim)}
    out_letter: dict[int, str] = {}      # dim id -> its produced einsum letter
    m_subs, m_arrays = [], []

    # Toeplitz pairs: the primary owns the contracted val axis.
    handled: set[int] = set()
    for d in (*out_dims, *primal_dims):
        if not (isinstance(d, ToeplitzIndex) and d.primary):
            continue
        partner = by_id[d.other_id]
        contracted = val_letter[d.axis]                  # the val-role axis
        roles = [None, None, None]
        val_role = ({0, 1, 2} - {d.role, partner.role}).pop()
        roles[val_role] = contracted
        roles[d.role] = lo = next(pool)
        roles[partner.role] = lp = next(pool)
        m_subs.append("".join(roles))
        m_arrays.append(d.indicator())
        out_letter[d.id], out_letter[partner.id] = lo, lp
        handled.add(d.id); handled.add(partner.id)

    # Dense (non-Toeplitz, val-backed) dims carry their letter straight through.
    for d in (*out_dims, *primal_dims):
        if d.id not in handled and d.axis is not None:
            out_letter[d.id] = val_letter[d.axis]

    out_block = [d for d in out_dims if d.id in out_letter]
    primal_block = [d for d in primal_dims if d.id in out_letter]
    eq = (",".join(["".join(val_letter[a] for a in range(val.ndim))] + m_subs)
          + "->" + "".join(out_letter[d.id] for d in out_block + primal_block))
    dense = jnp.einsum(eq, val, *m_arrays)

    no = len(out_block)
    new_out, oi = [], 0
    for d in out_dims:
        if d.id in out_letter:
            new_out.append(DenseIndex(d.id, d.size, oi)); oi += 1
        else:
            new_out.append(d)                            # diagonal pass-through
    new_primal, pi = [], no
    for d in primal_dims:
        if d.id in out_letter:
            new_primal.append(DenseIndex(d.id, d.size, pi)); pi += 1
        else:
            new_primal.append(d)
    return _rewrap(tensor, tuple(new_out), tuple(new_primal), dense)


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

    from graphax.sparse.indexes import (
        BandedIndex, SetIndex, DiagonalIndex, ToeplitzIndex,
    )

    # ToeplitzIndex (conv) pairs may coexist with Dense/Diagonal dims, so they
    # have their own mixed-tensor densify rather than the pure-pair paths below.
    if any(isinstance(d, ToeplitzIndex) for d in comp):
        return _densify_toeplitz(tensor)

    banded = [d for d in comp if isinstance(d, BandedIndex)]
    # Kernels consume the fill as a concrete array (``_eff_fill`` → 0 when the
    # fill is the statically-zero ``None`` marker); the RESULT tensor's fill,
    # however, preserves ``None`` so the materialized tensor stays fast-path
    # eligible. ``raw_fill`` carries that marker through.
    raw_fill = tensor.fill_value
    fill = tensor._eff_fill

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
        # The densified data's implicit cells hold op(fill_lhs, fill_rhs); the
        # result tensor's fill must match that combined value, not the raw
        # per-side fill (L3). combined_fill keeps a None (statically-zero) None.
        set_fill = sx.combined_fill(raw_fill)
        if compact:  # SetIndex always reduces to a meta-block-diagonal
            meta = sx.to_meta_blocks(tensor.val, fill)  # (M, LCM_h, LCM_w, *L)
            M, H, W = meta.shape[0], meta.shape[1], meta.shape[2]
            return _rewrap(tensor, (DiagonalIndex(o.id, M, 0, p.id, H, 1),),
                           (DiagonalIndex(p.id, M, 0, o.id, W, 2),), meta,
                           fill_value=set_fill)
        dense = sx.densify_axis(tensor.val, fill)  # (M*LCM_h, M*LCM_w, *L)
        return _dense_pair_result(tensor, dense, 1, fill_value=set_fill)

    # --- K=1 banded pair (Array val) ---
    if pure_pair and len(banded) == 2 and isinstance(tensor.val, jax.Array):
        primary = next((d for d in banded if d.primary), banded[0])
        o, p = tensor.out_dims[0], tensor.primal_dims[0]
        if compact and primary.reduces_to_diagonal():
            meta = primary.to_meta_blocks(tensor.val)  # (n_meta*M, B_row, B_col, *L)
            M, B_row, B_col = meta.shape[0], meta.shape[1], meta.shape[2]
            return _rewrap(tensor, (DiagonalIndex(o.id, M, 0, p.id, B_row, 1),),
                           (DiagonalIndex(p.id, M, 0, o.id, B_col, 2),), meta)
        dense = primary.densify_axis(tensor.val, fill)  # (rows, cols, *L)
        return _dense_pair_result(tensor, dense, 1)

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
        fill_lhs, fill_rhs = _split_fill(fill)
        lhs_dense = _densify_multi_banded(lhs_blocks, _specs(sx.lhs_shape), fill_lhs)
        if rhs_blocks is not None:
            rhs_dense = _densify_multi_banded(rhs_blocks, _specs(sx.rhs_shape), fill_rhs)
            dense = sx._op()(lhs_dense, rhs_dense)
        else:
            dense = sx._op()(lhs_dense, fill_rhs)
        # Result fill is the op-combined per-side fill (L3), matching the data;
        # combined_fill keeps a None (statically-zero) fill None.
        return _dense_pair_result(tensor, dense, K, fill_value=sx.combined_fill(raw_fill))

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
        return _dense_pair_result(tensor, dense, K)

    raise NotImplementedError(
        f"densify of compressed dims: unhandled shape — {len(comp)} compressed "
        f"dims across {len(tensor.out_dims)} out / {len(tensor.primal_dims)} "
        f"primal dims, val type {type(tensor.val).__name__}. Handled: a pure "
        "K=1 SetIndex / banded pair, or an all-BandedIndex K≥2 tensor. A "
        "compressed pair coexisting with Dense/Diagonal dims is not yet "
        "supported (no producer emits it today)."
    )


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
        # Compressed pairs (BandedIndex / ToeplitzIndex) legitimately link axes
        # of DIFFERENT sizes (e.g. a conv's out-pos P vs kernel-tap K) and carry
        # their own densify-time validation, so they are exempt from the
        # size-equality sparse-pair invariant below.
        if d.is_sparse and not d.is_compressed:
            if not _check_sparse_dim_pair(d, dim_map):
                raise ValueError(
                    f"Topology Error: Invalid sparse dimension pair configuration for dimension {d.id}"
                )
            if getattr(d, "block_axis", None) is not None:
                _check_block_axis(d, dim_map, block_axiss)


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
