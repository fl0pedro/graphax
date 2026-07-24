"""Atomic SparseTensor micro-actions: DIAG, COMPRESS, and QUANT.

Operations the RL policy can emit per sub-step:

* :class:`Diag` — block-diagonalise a pair of *logical* indices ``(i, j)`` with
  an explicit positive integer factor. gcd-collapse is *not* a sentinel here —
  the caller passes the actual integer it picked, even if that integer happens
  to equal ``gcd(N_i, N_j)``.
* :class:`Compress` — reduce one or more *physical* axes of the val array via
  one of six elementwise reductions (``mean`` / ``min`` / ``max`` / ``median``
  / ``abs_min`` / ``abs_max``), and mark every Index that pointed at those
  axes as ``axis=None``. The per-step semantics: ``val ← reduce(val,
  axis=axes)``, then physical-axis bookkeeping shifts the surviving indices
  down. ``abs_min`` / ``abs_max`` pick the entry whose absolute value is
  smallest / largest (closest to zero / furthest from zero), preserving the
  original sign.

The two operations are atomic and order-dependent:
``DIAG ∘ COMPRESS ≠ COMPRESS ∘ DIAG`` in general. :func:`apply_micro_actions`
applies an ordered sequence; multi-axis :class:`Compress` is the natural way
to batch several physical-axis reductions into one ``jnp.mean`` call when the
policy emits them in the same coordinate frame.

Legality
--------
The atomic helpers raise :class:`ValueError` on structural illegality —
``i == j``, axes out of range, duplicate physical axes, mismatched
``DiagonalIndex`` pairings, etc. Policy code is expected to either mask these
choices out before sampling or catch the ValueError at rollout time.

This module's atomic helpers raise on structural illegality rather than
silently filtering (as the now-removed ``apply_dynamic_sparsity`` did) so
policy bugs surface immediately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Sequence, Union

import jax.numpy as jnp

from graphax.sparse.indexes import DenseIndex, Index, DiagonalIndex, CompressedIndex
from graphax.sparse.tensor import SparseTensor, _apply_block_diagonal, _subdivide_coupled_blockdiag
from graphax.sparse.dtype_compute import _scaled_mul

import os as _os
from functools import wraps as _wraps
from graphax.sparse.ops.utils import _squeeze_unreferenced_val_axes
# Default ON; set GRAPHAX_KEEP_BLOCKDIAG=0 to force the legacy densify path.
_KEEP_BLOCKDIAG = _os.environ.get("GRAPHAX_KEEP_BLOCKDIAG", "1") != "0"


def _squeeze_result(fn):
    """Fold away non-data size-1 ``val`` axes from a micro-action's result.

    DIAG (block-diagonal split / subdivide) and COMPRESS both INSERT a fresh
    physical axis per meta / block side and never reclaim the leftovers, so an
    approx edge's ``val.ndim`` creeps upward step-by-step toward the numpy/XLA
    32-axis cap while its logical rank stays tiny (measured on ViT approx: a
    logical-rank-4 edge carried ``val.ndim`` 11-12, the extra axes all size 1).
    Those axes carry no data — every consumer works off ``dim.axis`` /
    ``dim.block_axis`` pointers, not the raw ``val.ndim`` — so squeezing them is
    a byte-identical reshape. A no-op when there are none, so it never perturbs a
    result that was already minimal (``apply_*`` that returned ``st`` unchanged,
    or the EXACT-AD path, which never invokes these actions)."""
    @_wraps(fn)
    def _wrapper(st, action, *a, **k):
        out = fn(st, action, *a, **k)
        if isinstance(out, SparseTensor):
            return _squeeze_unreferenced_val_axes(out)
        return out
    return _wrapper



# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diag:
    """A single block-diagonalisation of the logical index pair (i, j).

    ``i`` and ``j`` index into ``st.out_dims + st.primal_dims`` (the
    concatenated logical axis list). ``factor`` is the block size — the
    resulting :class:`DiagonalIndex` will have ``size=factor`` and
    ``block_size = N // factor`` for each side. ``factor`` must be a positive
    divisor of both logical sizes. If you want gcd-collapse, pass
    ``factor = math.gcd(N_i, N_j)`` explicitly.
    """

    i: int
    j: int
    factor: int

    def __post_init__(self):
        if self.factor <= 0:
            raise ValueError(
                f"Diag.factor must be a positive integer, got {self.factor!r}. "
                "Use math.gcd(...) for gcd-collapse and pass it explicitly; "
                "Pass an explicit positive divisor (the -1 / 0 sentinels "
                "from the legacy sparsity_map API are not accepted)."
            )
        if self.i == self.j:
            raise ValueError(f"Diag pair must be distinct, got i = j = {self.i}.")


COMPRESS_KINDS: tuple[str, ...] = (
    "mean", "min", "max", "median", "abs_min", "abs_max",
)
NUM_COMPRESS_KINDS = len(COMPRESS_KINDS)
COMPRESS_KIND_INDEX: dict[str, int] = {k: i for i, k in enumerate(COMPRESS_KINDS)}


@dataclass(frozen=True)
class Compress:
    """Reduce one or more *physical* axes of the underlying val array.

    ``axes`` is a tuple of physical-axis positions (``0 <= a < val.ndim``).
    Each axis must appear at most once. The physical axes are interpreted in
    the SparseTensor's current frame — if you intend to compress axes that
    were renumbered by an earlier micro-action in the same sub-episode, use
    the *current* numbering, not the numbering at sub-episode start.

    ``kind`` picks the reduction:

    * ``"mean"`` — arithmetic mean (default; the legacy behaviour).
    * ``"min"`` / ``"max"`` — elementwise min / max.
    * ``"median"`` — elementwise median.
    * ``"abs_min"`` — the entry whose absolute value is smallest along the
      axis (closest to zero); the original sign is preserved.
    * ``"abs_max"`` — the entry whose absolute value is largest (furthest
      from zero); the original sign is preserved.

    All :class:`Index` instances whose ``axis`` or ``block_axis`` referenced a
    compressed physical axis have that pointer set to ``None``; remaining
    pointers are shifted down to account for the dropped axes.
    """

    axes: tuple[int, ...]
    kind: str = "mean"

    def __post_init__(self):
        if len(self.axes) != len(set(self.axes)):
            raise ValueError(
                f"Compress.axes must be unique, got {self.axes!r}."
            )
        for a in self.axes:
            if a < 0:
                raise ValueError(
                    f"Compress.axes entries must be non-negative, got {a!r}."
                )
        if self.kind not in COMPRESS_KIND_INDEX:
            raise ValueError(
                f"Compress.kind must be one of {COMPRESS_KINDS!r}, "
                f"got {self.kind!r}."
            )


# Canonical JAX dtype names accepted by :class:`Quant`. Strings (not
# ``jnp.dtype`` objects) keep :class:`Quant` hashable and match the
# ``Compress.kind: str`` pattern. The int index into :data:`QUANT_DTYPE_INDEX`
# is the wire-format contract between a policy head (samples an int) and the
# env-side translator (looks the int up to construct a ``Quant``) — same
# convention as :data:`COMPRESS_KINDS`.
def _get_quant_dtypes():
    import jax
    import jax.numpy as jnp
    # 64-bit dtypes are only available if explicitly enabled.
    x64_enabled = jax.config.jax_enable_x64
    
    dtypes = []
    # 64-bit block
    if x64_enabled:
        dtypes.extend([jnp.float64, jnp.int64, jnp.uint64])
        
    # Standard 32/16 bit
    dtypes.extend([
        jnp.float32, jnp.float16, jnp.bfloat16,
        jnp.int32, jnp.uint32,
        jnp.int16, jnp.uint16,
        jnp.int8, jnp.uint8,
    ])
    
    # FP8 variants (only include if jax natively supports them to avoid AttributeError)
    if hasattr(jnp, "float8_e4m3fn"):
        dtypes.append(jnp.float8_e4m3fn)
    if hasattr(jnp, "float8_e4m3b11fnuz"):
        dtypes.append(jnp.float8_e4m3b11fnuz)
    if hasattr(jnp, "float8_e5m2"):
        dtypes.append(jnp.float8_e5m2)
    if hasattr(jnp, "float8_e5m2fnuz"):
        dtypes.append(jnp.float8_e5m2fnuz)
        
    # Sub-byte variants
    if hasattr(jnp, "int4"):
        dtypes.append(jnp.int4)
    if hasattr(jnp, "uint4"):
        dtypes.append(jnp.uint4)
        
    return tuple(jnp.dtype(d).name for d in dtypes)

QUANT_DTYPES = _get_quant_dtypes()

def verify_hardware_compat():
    """Dynamically verify jnp.dot compatibility across all QUANT_DTYPES."""
    import jax
    import jax.numpy as jnp
    
    avail_mask = []
    for dt in QUANT_DTYPES:
        try:
            x = jnp.zeros((2, 2), dtype=dt)
            jnp.dot(x, x)
            avail_mask.append(1.0)
        except Exception:
            avail_mask.append(0.0)
    
    N = len(QUANT_DTYPES)
    compat_matrix = jnp.zeros((N, N), dtype=jnp.float32)
    for i, dt1 in enumerate(QUANT_DTYPES):
        for j, dt2 in enumerate(QUANT_DTYPES):
            if not avail_mask[i] or not avail_mask[j]:
                continue
            try:
                x = jnp.zeros((2, 2), dtype=dt1)
                y = jnp.zeros((2, 2), dtype=dt2)
                jnp.dot(x, y)
                compat_matrix = compat_matrix.at[i, j].set(1.0)
            except Exception:
                pass
                
    return jnp.array(avail_mask, dtype=jnp.float32), compat_matrix

NUM_QUANT_DTYPES = len(QUANT_DTYPES)
QUANT_DTYPE_INDEX: dict[str, int] = {d: i for i, d in enumerate(QUANT_DTYPES)}


@dataclass(frozen=True)
class Quant:
    """Cast :attr:`SparseTensor.val` to a chosen JAX dtype.

    Multiple Quant actions in a sub-episode are applied sequentially in
    order — last one wins, no special rounding. Equivalent to chained
    ``val.astype(d_1).astype(d_2)...``; the chain matters when intermediate
    dtypes are lossy (e.g. ``int8`` then ``float32``). Only ``val`` is cast;
    ``scalar_mult`` and ``fill_value`` keep their native dtype.

    A ``val=None`` SparseTensor (uniform grid) is returned unchanged.

    ``dtype`` is the canonical name (e.g. ``"float16"``, ``"float8_e4m3fn"``,
    ``"bfloat16"``) and must be a member of :data:`QUANT_DTYPES`. Narrow
    dtypes (float8 / float4 / sub-byte int / complex) have no implicit JAX
    promotion path; the ops upcast operands to their highest common dtype
    before arithmetic (see :func:`graphax.sparse.ops.utils._compute_dtype`),
    so a quantized ``val`` is stored narrow but computable everywhere.

    ``scale_sign`` ∈ ``{+1, -1}`` selects which arm of the value range fills an
    UNSIGNED target. The quantizer stays symmetric-about-zero (no zero-point):
    for ``+1`` the positive half of the values maps onto ``[0, dtype_max]`` and
    the negative half clips to 0; for ``-1`` the negative half's magnitudes map
    on and the positive half clips. The chosen sign is folded into
    ``scalar_mult`` so ``val * scalar_mult`` dequantizes the kept arm exactly.
    For signed / float targets the sign is a benign symmetry (they already
    represent both arms), so it only materially changes unsigned quantization.
    """

    dtype: str
    scale_sign: int = 1

    def __post_init__(self):
        if self.dtype not in QUANT_DTYPE_INDEX:
            raise ValueError(
                f"Quant.dtype must be one of {QUANT_DTYPES!r}, "
                f"got {self.dtype!r}."
            )
        if self.scale_sign not in (1, -1):
            raise ValueError(
                f"Quant.scale_sign must be +1 or -1, got {self.scale_sign!r}."
            )


MicroAction = Union[Diag, Compress, Quant]


# ---------------------------------------------------------------------------
# Atomic DIAG
# ---------------------------------------------------------------------------


@_squeeze_result
def apply_diag(st: SparseTensor, action: Diag) -> SparseTensor:
    """Apply a single block-diagonalisation rule to ``st``.

    Validates structurally before delegating to the existing
    :func:`graphax.sparse.tensor._apply_block_diagonal` primitive — the
    primitive does the val-axis reshape + ``jnp.diagonal`` and metadata
    rebuild. Returns ``st`` unchanged for the degenerate ``factor == 1`` case
    (no real diagonalisation).
    """
    out_len = len(st.out_dims)
    total_ndim = out_len + len(st.primal_dims)

    if not (0 <= action.i < total_ndim):
        raise ValueError(
            f"Diag.i = {action.i} out of range [0, {total_ndim}); "
            f"out_dims={out_len}, primal_dims={total_ndim - out_len}."
        )
    if not (0 <= action.j < total_ndim):
        raise ValueError(
            f"Diag.j = {action.j} out of range [0, {total_ndim}); "
            f"out_dims={out_len}, primal_dims={total_ndim - out_len}."
        )

    is_out1 = action.i < out_len
    rel_i = action.i if is_out1 else action.i - out_len
    is_out2 = action.j < out_len
    rel_j = action.j if is_out2 else action.j - out_len

    d1 = st.out_dims[rel_i] if is_out1 else st.primal_dims[rel_i]
    d2 = st.out_dims[rel_j] if is_out2 else st.primal_dims[rel_j]

    # A Jacobian diagonal ties an OUT axis to a PRIMAL axis (nonzero only where
    # out_idx == in_idx). out<->out and primal<->primal pairs are therefore
    # meaningless. This was never validated: is_out1/is_out2 were computed and
    # never compared, so such a pair was ACCEPTED and produced a dim whose
    # other_id points into the primal range — which later detonated as
    # `IndexError: list index out of range` in inverse_transpose_transform
    # (ViT + Diag). Reject it at the source instead of tolerating the bad state.
    if is_out1 == is_out2:
        side = "out_dims" if is_out1 else "primal_dims"
        raise ValueError(
            f"Diag pair ({action.i}, {action.j}) is not split across out/primal: "
            f"both indices are in {side}. A diagonal must tie one out axis to one "
            f"primal axis."
        )

    if d1.is_sparse and d1.other_id != d2.id:
        raise ValueError(
            f"Diag pair conflict: logical index {action.i} is already paired "
            f"with another index (other_id={d1.other_id}, but d2.id={d2.id})."
        )
    if d2.is_sparse and d2.other_id != d1.id:
        raise ValueError(
            f"Diag pair conflict: logical index {action.j} is already paired "
            f"with another index (other_id={d2.other_id}, but d1.id={d1.id})."
        )

    N1, N2 = d1.logical_size, d2.logical_size
    factor = action.factor

    # GRAPHAX_KEEP_BLOCKDIAG: (i, j) may ALREADY be a coupled block-diagonal (the
    # keep-sparse re-mask path). Re-masking rules against the CURRENT meta count:
    #   * factor == cur_meta        -> identical structure: no-op.
    #   * factor  > cur_meta, and factor is a MULTIPLE of cur_meta AND divides both
    #     logical sizes -> FINER blocks: subdivide each current block into
    #     (factor // cur_meta) meta-diagonal sub-blocks (stays block-sparse).
    #   * anything else (coarser / non-multiple / non-divisor block) -> raise
    #     (a non-nestable re-mask; correctness over a silent wrong structure).
    if _KEEP_BLOCKDIAG and d1.is_sparse and d2.is_sparse:
        coupled = (
            getattr(d1, "other_id", None) == d2.id
            and getattr(d2, "other_id", None) == d1.id
            and getattr(d1, "size", None) == getattr(d2, "size", None)
        )
        cur_meta = getattr(d1, "size", None)
        if coupled and cur_meta == factor:
            return st
        if (
            coupled
            and cur_meta
            and factor > cur_meta
            and factor % cur_meta == 0
            and N1 % factor == 0
            and N2 % factor == 0
        ):
            return _subdivide_coupled_blockdiag(
                st, is_out1, rel_i, d1, is_out2, rel_j, d2, factor
            )
        # COARSER factor on a finer coupled block-diagonal: the existing structure
        # already satisfies the coarser constraint (blocks of cur_meta automatically
        # satisfy any factor that divides cur_meta, or factor==1 which is the trivial
        # "full diagonal" already implied by the coupling). Treat as no-op.
        if (
            coupled
            and cur_meta
            and factor < cur_meta
            and (factor == 1 or cur_meta % factor == 0)
        ):
            return st
        # Coupled but NOT same-factor, NOT a valid finer subdivision, and NOT a
        # coarser-subsumed factor. This is a non-nestable re-mask: the legacy path
        # would silently return st unchanged (v1==v2 no-op) => UNDER-mask.
        # Reject explicitly so the caller drops it rather than applying a wrong
        # (lighter) approximation. The implicit pure-diagonal pair (axis is None) is
        # NOT rejected here — it is handled as a genuine no-op just below.
        if (
            coupled
            and factor != cur_meta
            and getattr(d1, "axis", None) is not None
            and getattr(d2, "axis", None) is not None
        ):
            raise ValueError(
                f"Diag: cannot re-mask a coupled block-diagonal (meta={cur_meta}) "
                f"by factor={factor}: not the same factor, not a finer subdivision "
                f"(factor must be a multiple of cur_meta and divide both logical "
                f"sizes), and not a coarser-subsumed factor (cur_meta must be "
                f"divisible by factor)."
            )

    if N1 % factor != 0 or N2 % factor != 0:
        raise ValueError(
            f"Diag.factor = {factor} does not divide both logical sizes "
            f"({N1}, {N2})."
        )
    if factor == 1:
        # Degenerate: no real diagonalisation. Keep st as-is.
        return st

    b1, b2 = N1 // factor, N2 // factor

    # A PURE-DIAGONAL pair (d1.other_id == d2.id) that is ALSO implicit (both
    # ``axis is None`` — the pair carries no physical val axis, i.e. it is
    # ``scalar·I`` over this logical index) is its own block-diagonal: the mask
    # ``floor(i/b1) == floor(j/b2)`` is satisfied for every diagonal entry
    # ``i == j``, so block-diagonalising it changes no value. ``_apply_block_
    # diagonal`` would here emit a DiagonalIndex with ``block_size > 1`` but
    # ``block_axis = None`` — an unrepresentable, un-densifiable dim (the
    # ``(2,3,1,16,...)`` malformed dense() on the slice/concat multi-head ViT
    # under reverse / non-canonical orders). Treat it as the no-op it is and keep
    # the plain diagonal pair untouched.
    if (
        d1.is_sparse and d2.is_sparse
        and getattr(d1, "other_id", None) == d2.id
        and getattr(d2, "other_id", None) == d1.id
        and getattr(d1, "axis", None) is None
        and getattr(d2, "axis", None) is None
    ):
        return st

    # AUDIT FIX: a mixed implicit/physical (or both-implicit non-paired) Diag
    # cannot be represented by _apply_block_diagonal, which assumes BOTH sides
    # are materialized val axes. Applying it anyway corrupts the dim list
    # (physical-axis collision -> downstream transpose/IndexError crash).
    # Raise ValueError so the per-vertex best-effort handler skips this
    # transform on this edge (leaves it exact).
    if getattr(d1, "axis", None) is None or getattr(d2, "axis", None) is None:
        raise ValueError(
            "Diag: cannot block-diagonalise a pair involving an implicit "
            "(axis=None) dim — its physical axis was dropped (e.g. by a prior "
            "Compress) and _apply_block_diagonal requires materialized axes."
        )

    return _apply_block_diagonal(
        st, is_out1, rel_i, is_out2, rel_j, factor, b1, b2,
    )


# ---------------------------------------------------------------------------
# Atomic COMPRESS
# ---------------------------------------------------------------------------


def _reduce_along_axes(val: jnp.ndarray, axes: tuple[int, ...], kind: str):
    """Reduce ``val`` along ``axes`` using the named elementwise rule.

    ``abs_min`` / ``abs_max`` pick the entry whose absolute value is smallest
    / largest along the reduction axes, preserving the original sign. Both
    are implemented as a take_along_axis over the argmin / argmax of ``|val|``
    after merging the reduction axes into a single trailing one — this keeps
    the reduction to a single XLA op even for multi-axis Compress and avoids
    a Python-level fold over the axes.
    """
    axes = tuple(sorted(set(axes)))
    if kind == "mean":
        return jnp.mean(val, axis=axes)
    if kind == "min":
        return jnp.min(val, axis=axes)
    if kind == "max":
        return jnp.max(val, axis=axes)
    if kind == "median":
        return jnp.median(val, axis=axes)
    if kind in ("abs_min", "abs_max"):
        # Move every reduction axis to the trailing positions, flatten them
        # into a single axis, then argmin / argmax over |·|. take_along_axis
        # against the original (flattened) values keeps the sign.
        keep = [a for a in range(val.ndim) if a not in axes]
        perm = keep + list(axes)
        moved = jnp.transpose(val, perm)
        flat_shape = moved.shape[: len(keep)] + (-1,)
        flat = moved.reshape(flat_shape)
        abs_flat = jnp.abs(flat)
        if kind == "abs_min":
            idx = jnp.argmin(abs_flat, axis=-1, keepdims=True)
        else:
            idx = jnp.argmax(abs_flat, axis=-1, keepdims=True)
        picked = jnp.take_along_axis(flat, idx, axis=-1)
        return jnp.squeeze(picked, axis=-1)
    raise ValueError(f"Unknown Compress.kind {kind!r}")


@_squeeze_result
def apply_compress(st: SparseTensor, action: Compress) -> SparseTensor:
    """Reduce the listed physical axes via ``action.kind`` (default ``mean``).

    The reduction happens as a single ``jnp.<reduce>(val, axis=sorted_axes)``
    call — multi-axis Compress is the efficient form because all axes are
    dropped in one XLA op rather than a sequence of single-axis reductions.
    """
    if st.val is None:
        # A ``val is None`` tensor is UNIFORM: every logical cell equals the
        # (post-scaled) ``scalar_mult``, and every dim is already stored-once
        # (implicit). Compress reduces a dim to a single representative and
        # marks it implicit — but for a uniform tensor that representative is
        # the value the dim already holds, and the dim is already implicit, so
        # the reduction is a NO-OP by construction (``return st``). This is
        # exact for EVERY supported ``Compress.kind`` because all of them —
        # mean, min, max, median, abs_min, abs_max — are IDEMPOTENT on a set of
        # identical values (there is no sum/prod kind). Previously this branch
        # raised on ``action.axes``; that raise only ever fired when a later
        # transform met an already-uniform edge (e.g. one the sparsity-retaining
        # elementwise path collapsed earlier than the materializing path would),
        # surfacing as core.py's "TRANSFORM DID NOT FIT". Returning the uniform
        # tensor unchanged is the correct result the materializing path also
        # reaches (mean of N identical cells == that cell), so no approximation
        # semantics change — only the spurious failure is removed.
        return st
    if not action.axes:
        return st

    val_ndim = st.val.ndim
    for a in action.axes:
        if a >= val_ndim:
            raise ValueError(
                f"Compress.axes entry {a} out of range for val.ndim = {val_ndim}."
            )

    # COMPRESS may only reduce FREE physical axes. A *structural* axis — the
    # block_axis of a sparse (Diagonal) dim, or ANY axis of a still-compressed
    # Banded/Set/Toeplitz band buffer — encodes block/band geometry that the
    # densify kernels read positionally, while `logical_size` is derived from
    # `size * block_size`. Dropping such an axis via `jnp.mean` shifts the
    # buffer layout AND leaves `logical_size` stale, producing the core.py
    # edge-shape AssertionError and the matmul.py "Contraction size mismatch"
    # (16 vs 4 / 16 vs 10) on high-rank edges (ViT attention, MoE experts).
    # Raise ValueError so the elimination loop's best-effort handler skips this
    # COMPRESS on this edge (leaving it exact) instead of corrupting it.
    # DenseIndex axes (NN/ConvNet) are unaffected: not sparse, no block_axis,
    # not a CompressedIndex.
    # REMOVED 2026-07-15: over-conservative structural-block-axis / CompressedIndex guard.
    # It raised ValueError *so the elimination loop would silently SKIP the COMPRESS*, which
    # desyncs the two edges of a shared var -> the documented root cause of the very
    # "Contraction size mismatch" family it claimed to prevent. Block-axis compress is
    # legitimate (all blocks identical == batched along the sparse component); _remap below
    # already handles block_axis -> None. Measured: -50% storage, .dense() identical to the
    # uncompressed edge (logical_size NOT stale), zero added downstream contraction failures,
    # 153/153 axes fine unguarded. Invalid actions must be masked UP FRONT by the caller.
    drops = sorted(set(action.axes))
    new_val = _reduce_along_axes(st.val, tuple(drops), action.kind)

    drop_set = set(drops)

    def _shift_after_drops(p: int | None) -> int | None:
        if p is None:
            return None
        if p in drop_set:
            return None
        return p - sum(1 for d in drops if d < p)

    def _remap(d: Index) -> Index:
        new_axis = _shift_after_drops(getattr(d, "axis", None))
        if d.is_sparse:
            new_block_axis = _shift_after_drops(d.block_axis)
            return replace(d, axis=new_axis, block_axis=new_block_axis)
        if not d.is_sparse:
            return replace(d, axis=new_axis)
        return replace(d, axis=new_axis)

    new_out = tuple(_remap(d) for d in st.out_dims)
    new_primal = tuple(_remap(d) for d in st.primal_dims)

    new_scalar_mult = st.scalar_mult
    # FULL-REDUCTION CANONICALIZATION. When the reduction drops EVERY physical
    # axis the result is a 0-dim ``val`` scalar — the tensor is now uniform
    # (every logical cell equals that one reduced value). The bare 0-dim-``val``
    # form is non-canonical: ``.dense()`` would broadcast ``_scaled_mul(val,
    # scalar_mult)`` up, leaving a dangling rank-0 array where the canonical
    # "uniform grid" representation is ``val=None`` with the uniform magnitude
    # carried by ``scalar_mult`` (see ``dense_for_matmul``'s ``val is None``
    # fast path: it returns ``broadcast_to(scalar_mult * 1.0, shape)``). Fold
    # the reduced scalar into ``scalar_mult`` and set ``val=None``:
    #     new_scalar_mult = scalar_mult * reduced_scalar
    # so ``.dense()`` reconstructs EXACTLY the same uniform tensor the 0-dim
    # path densified (scalar_mult * reduced_scalar, broadcast) — a clean
    # canonicalization, not a behaviour change.
    #
    # Guard: only fold to ``val=None`` when NO sparse dim remains. ``val=None``
    # is read by ``_structural_val_size`` / ``dense()`` as an all-ones grid
    # whose stored size divides out each sparse pair's meta count; a tensor
    # that still carries a Diagonal pair would mis-densify under that reading.
    # The COMPRESS guard already forbids dropping a sparse dim's structural
    # block_axis, so a tensor with sparse dims cannot legitimately reach a
    # 0-dim val here — but keep the bare scalar val for that case rather than
    # silently corrupt it.
    if new_val is not None and new_val.ndim == 0 and not any(
        d.is_sparse for d in (*new_out, *new_primal)
    ):
        new_scalar_mult = _scaled_mul(st.scalar_mult, new_val)
        new_val = None

    return SparseTensor(
        new_out,
        new_primal,
        new_val,
        scalar_mult=new_scalar_mult,
        fill_value=st.fill_value,
        # Forward the deferred-transform queues (the bare constructor defaults
        # them to ()): dropping them silently erases pending reshape/slice/...
        # relabels — the same defect fixed in apply_quant. Compress only changes
        # the at-rest val STORAGE (drops size-1/compressed axes); the transforms
        # act on the .dense() form, whose logical shape is unchanged, so they
        # ride through correctly.
        pre_transforms=st.pre_transforms,
        post_transforms=st.post_transforms,
        check_consistency=False,
    )


# ---------------------------------------------------------------------------
# Atomic QUANT
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Scaled-quant dtype helpers
# ---------------------------------------------------------------------------

# Integer / unsigned / sub-byte targets get SCALED quantization (a per-tensor
# float scale folded into scalar_mult). Everything else (float8/float4/bf16/
# fp16/fp32/fp64/complex/bool) keeps a plain astype.
def _is_scaled_quant_target(target) -> bool:
    name = jnp.dtype(target).name
    if name == "bool":
        return False
    return name.startswith("int") or name.startswith("uint")


def _int_dtype_range(target) -> tuple[int, int]:
    """(min, max) representable integer values for an int/uint target, including
    the sub-byte ints (int2/int4/uint2/uint4) which JAX exposes via iinfo even
    though they are stored in an int8/uint8 container."""
    ii = jnp.iinfo(jnp.dtype(target))
    return int(ii.min), int(ii.max)


def _int_dtype_max(target) -> float:
    """Symmetric positive scale denominator: the largest representable
    magnitude of the target (``iinfo.max``). For signed ints this is the
    positive arm (e.g. int8 -> 127, int4 -> 7, int2 -> 1); for unsigned it is
    the full max (uint8 -> 255). Guaranteed >= 1 so the scale is well-defined."""
    ii = jnp.iinfo(jnp.dtype(target))
    return float(max(int(ii.max), 1))


@_squeeze_result
def apply_quant(st: SparseTensor, action: Quant) -> SparseTensor:
    """Cast ``st.val`` to ``action.dtype``.

    Returns ``st`` unchanged if ``val is None`` or the target dtype already
    matches the current one. Only ``val`` is cast — ``scalar_mult`` and
    ``fill_value`` keep their native dtype. Narrow targets (float8 /
    sub-byte int / float4 / complex) have no implicit JAX promotion path, so
    downstream densify / matmul / elementwise upcast to the highest common
    dtype before arithmetic (see :func:`graphax.sparse.ops.utils._compute_dtype`);
    the narrow dtype only sets the at-rest ``val.dtype``.
    """
    if st.val is None:
        return st
    target = jnp.dtype(action.dtype)
    if st.val.dtype == target:
        return st

    new_val = st.val
    new_scalar_mult = st.scalar_mult
    if _is_scaled_quant_target(target):
        # SCALED PER-TENSOR QUANTIZATION (int / uint / sub-byte targets).
        # A bare ``val.astype(int8)`` TRUNCATES every fractional Jacobian entry
        # toward zero — a |J|<1 Jacobian collapses to all-zeros (the cossim-0
        # degenerate). Instead store a symmetric per-tensor scale ``s`` and the
        # rounded integer codes, folding ``s`` into ``scalar_mult`` so every
        # downstream consumer auto-dequantizes (``.dense()`` / matmul /
        # elementwise all apply ``scalar_mult`` to ``val`` before arithmetic —
        # verified: no consumer reads ``val`` as a value without scalar_mult):
        #     s   = max(|val|) / dtype_max(target)        # symmetric scale
        #     q   = round(val / s).astype(target)         # integer codes
        #     sm' = scalar_mult * s                        # val*sm' ~= val
        # All-zero ``val`` -> s would be 0; guard to a plain astype no-op
        # (the zeros quantize to zeros, scalar_mult unchanged).
        # Work from the DEQUANTIZED logical value (``val * scalar_mult``) so
        # re-quantization composes and the sign is authoritative (a val already
        # narrow from a prior Quant carries its polarity in scalar_mult):
        #     logical = val * scalar_mult                  # true signed values
        #     s       = max(|logical|) / dtype_max(target) # scale to the TARGET's
        #                                                  # max, not the value's
        #     q       = round(logical * sign / s)          # integer codes
        #     sm'     = s * sign                            # q * sm' ~= logical
        # For a SIGNED / float-scaled target ``sign`` is a benign symmetry and
        # this reduces to the previous symmetric quantizer. For an UNSIGNED
        # target only the ``sign`` arm of ``logical`` is kept (the other clips to
        # 0) and ``sign`` folds into ``sm'`` so the kept arm dequantizes with its
        # original polarity — no zero-point / offset is ever stored.
        sm = 1.0 if st.scalar_mult is None else st.scalar_mult
        logical = st.val.astype(jnp.float32) * jnp.asarray(sm, jnp.float32)
        sign = jnp.float32(action.scale_sign)
        absmax = jnp.max(jnp.abs(logical))
        dmax = _int_dtype_max(target)                # target's max magnitude
        s = absmax / dmax
        # All-zero ``val`` -> absmax 0 -> keep s=1 (codes stay zeros, no div-by-0).
        s_safe = jnp.where(s > 0, s, jnp.ones_like(s))
        if jnp.issubdtype(target, jnp.unsignedinteger):
            # Sign-flip half-range: keep the arm selected by ``sign``; clip the
            # other to 0. The magnitudes fill [0, dtype_max].
            kept = jnp.clip(logical * sign, 0.0, None)
            q = jnp.round(kept / s_safe)
            q = jnp.clip(q, 0.0, jnp.float32(dmax))
        else:
            q = jnp.round(logical * sign / s_safe)
            # Clamp to the target's logical range before the narrow cast so a
            # round-half-away at the extreme can't wrap; use FLOAT bounds (raw
            # int64 bounds overflow JAX's jit argument parser).
            lo, hi = _int_dtype_range(target)
            q = jnp.clip(q, jnp.float32(lo), jnp.float32(hi))
        new_val = q.astype(target)
        # sm' REPLACES scalar_mult (its old value is already folded into the
        # quantized codes via ``logical``).
        new_scalar_mult = s_safe * sign
    else:
        # Float targets (bfloat16 / float16 / float8_* / float4 / complex) carry
        # a fraction natively — a plain astype preserves relative magnitudes, so
        # no scale is needed (and ``bool`` has no meaningful scaled form).
        new_val = st.val.astype(target)

    return SparseTensor(
        st.out_dims,
        st.primal_dims,
        new_val,
        scalar_mult=new_scalar_mult,
        fill_value=st.fill_value,
        # The cast is shape-preserving, so any deferred Jacobian transform queued
        # on this edge (a reshape/slice/concatenate relabel awaiting drain) must
        # ride along — dropping it leaves ``val`` at its pre-drain shape and later
        # desyncs the contraction size / shape assert (jit-latent under canonical
        # orders, surfaced by non-canonical elimination orders).
        pre_transforms=st.pre_transforms,
        post_transforms=st.post_transforms,
        check_consistency=False,
    )


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Copy-pastable factory helpers
# ---------------------------------------------------------------------------
#
# These return a callable ``(SparseTensor) -> SparseTensor`` so the policy
# logs can render a sub-episode as a list of evaluable Python expressions
# (e.g. ``[diag(0, 1, 2), compress("abs_min", 3)]``) that round-trip through
# the typed-transform API in :mod:`graphax.core`.


def diag(i: int, j: int, factor: int) -> Callable[[SparseTensor], SparseTensor]:
    """Return a function that applies :class:`Diag` with the given args.

    Equivalent to ``lambda st: apply_diag(st, Diag(i, j, factor))``; the
    closure is hashable in graphax's `transforms` cache because the inner
    ``Diag`` is a frozen dataclass.
    """
    action = Diag(i=int(i), j=int(j), factor=int(factor))

    def _apply(st: SparseTensor) -> SparseTensor:
        return apply_diag(st, action)

    _apply.__name__ = f"diag({i}, {j}, {factor})"
    return _apply


def compress(
    kind: str, *axes: int,
) -> Callable[[SparseTensor], SparseTensor]:
    """Return a function that applies :class:`Compress` with the given kind/axes.

    ``compress("mean", 3)`` is the single-axis form; multi-axis is
    ``compress("abs_max", 0, 2)``. Mirrors the API of :func:`diag`.
    """
    action = Compress(axes=tuple(int(a) for a in axes), kind=kind)

    def _apply(st: SparseTensor) -> SparseTensor:
        return apply_compress(st, action)

    _apply.__name__ = f"compress({kind!r}, {', '.join(str(a) for a in axes)})"
    return _apply


def quant(dtype: str) -> Callable[[SparseTensor], SparseTensor]:
    """Return a function that applies :class:`Quant` with the given dtype.

    Equivalent to ``lambda st: apply_quant(st, Quant(dtype))``; the closure
    is hashable in graphax's ``transforms`` cache because the inner
    ``Quant`` is a frozen dataclass. Mirrors the API of :func:`diag` /
    :func:`compress`.
    """
    action = Quant(dtype=str(dtype))

    def _apply(st: SparseTensor) -> SparseTensor:
        return apply_quant(st, action)

    _apply.__name__ = f"quant({dtype!r})"
    return _apply


def apply_micro_actions(
    st: SparseTensor,
    actions: Sequence[MicroAction],
) -> SparseTensor:
    """Apply an ordered sequence of :class:`Diag` and :class:`Compress` actions.

    Order matters: ``DIAG ∘ COMPRESS ≠ COMPRESS ∘ DIAG`` in general because
    DIAG sees logical indices (which depend on which axes have been compressed
    away) and COMPRESS sees physical axes (which depend on prior shape edits).

    No coalescing: if you want to batch several Compress actions into one
    ``jnp.mean`` call, emit a single :class:`Compress` with multiple axes —
    that's exactly what the multi-axis form is for. Coalescing across
    intervening DIAGs is not generally sound because the physical-axis
    numbering shifts.
    """
    for action in actions:
        if isinstance(action, Diag):
            st = apply_diag(st, action)
        elif isinstance(action, Compress):
            st = apply_compress(st, action)
        elif isinstance(action, Quant):
            st = apply_quant(st, action)
        else:
            raise TypeError(
                f"apply_micro_actions expected Diag, Compress, or Quant, "
                f"got {type(action).__name__}."
            )
    return st
