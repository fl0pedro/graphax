r"""Composition + dispatch layer wiring the elemental kernels into the real
vertex-elimination matmul / elementwise.

================================================================================
WHAT THIS IS
================================================================================
``src/graphax/sparse/elemental/`` ships six *pairwise* kernels, each owning ONE
elemental operation between a specific pair of structured dims:

  * ``contract_dense_block_diagonal``  — ``D @ B`` / ``B @ D`` (one Diagonal pair)
  * ``contract_B_B``                   — ``B @ B``            (two Diagonal pairs)
  * ``contract_implicit``              — an implicit (Compress-away) contracted dim
  * ``elementwise_dense_block_diagonal``— ``D (+|*) B`` / ``B op B``
  * ``elementwise_implicit``           — an implicit dim under an elementwise op
  * ``materialize_compressed``         — expand a CompressedIndex → ``{D, B}``

Each kernel handles EXACTLY ONE structured contracted pair plus ride-through free
dims.  A real jacve contraction, however, may contract SEVERAL dims at once of
MIXED type (a block-diagonal pair AND a dense pair AND/OR an implicit pair).
This module is the COMPOSITION layer: it identifies all contracted pairs, routes
the clean single-structured-pair cases to the nnz-optimal pairwise kernel, and
composes everything else into a single correct, canonically-id'd contraction.

================================================================================
DIM-ID CONVENTION (the thing that caused prior permutation bugs)
================================================================================
graphax aligns a downstream contraction by dim id and the matmul ``finalize``
renumbers the output ids to ``range(n)`` — out dims get ``0 .. n_out-1``, primal
dims get ``n_out .. n_out+n_primal-1``, in the dot_general output-axis order
(batch dims, then lhs kept dims, then rhs kept dims).  EVERY result this module
emits MUST carry ids in that convention so a LATER multi-edge / all-vertices
contraction aligns correctly.  The pairwise kernels' ``_emit`` helpers already
follow it, and the composed-dense path reuses matmul's own ``_matmul_via_densify``
(which builds ``range(0, n)`` ids by construction), so the convention is honoured
on every branch.

================================================================================
SCOPE / FALLBACK POLICY
================================================================================
* PURE-DENSE contraction (no Diagonal / implicit / compressed contracted dim
  after materialization) ⇒ return ``None`` IMMEDIATELY.  The existing dense path
  then handles it, so the EXACT-AD (``transforms=()``) edge is byte-identical —
  this module is never even entered for it.
* A single structured contracted pair in the pairwise kernels' 2-D-core scope ⇒
  the matching nnz-optimal kernel (stays sparse where it can).
* Anything beyond a single kernel's scope — MULTIPLE structured contracted pairs
  simultaneously, or a structured pair coexisting with extra contracted dims the
  kernel can't ride through — is composed via a single correct ``dot_general``
  (``_matmul_via_densify``) with the canonical id convention.  These are counted
  in ``DISPATCH_STATS`` so the multi-structured frequency is reported, never
  silently capped.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp

from graphax.sparse.ops.utils import _is_zero_fill

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


# --------------------------------------------------------------------------- #
# Approximation-active gate.
#
# DESIGN INVARIANT: with NO approximation (``Diag``/``Compress``) active, reverse
# (and forward) vertex elimination must be EXACT and byte-identical to the
# pre-elemental path — the existing matmul/elementwise already exploit block-
# diagonal / zero-fill structure correctly there. The elemental kernels exist
# only to make the APPROXIMATION edges (rectangular Diag blocks, Compress implicit
# dims) contract legally; firing them on the intrinsic-diagonal edges that arise
# in plain exact AD restructures those edges and breaks a downstream contraction
# (e.g. a broadcast-bias Jacobian -> ``size mismatch 1 vs N``). So the dispatch is
# a hard no-op unless ``core`` has flagged that this elimination carries a
# Diag/Compress transform. Thread-local so concurrent traces don't race.
# --------------------------------------------------------------------------- #
_approx_state = threading.local()


def set_approx_active(active: bool) -> None:
    _approx_state.active = bool(active)


def approx_active() -> bool:
    return getattr(_approx_state, "active", False)


# --------------------------------------------------------------------------- #
# Telemetry — honest accounting of which path fired (gate 4 / blocker report).
# --------------------------------------------------------------------------- #
DISPATCH_STATS: dict[str, int] = {
    "matmul_pure_dense_skip": 0,        # returned None (exact-AD / dense path)
    "matmul_kernel_D_B": 0,             # contract_dense_block_diagonal
    "matmul_kernel_B_B": 0,             # contract_B_B
    "matmul_kernel_implicit": 0,        # contract_implicit
    "matmul_kernel_multi_D_B": 0,       # contract_dense_multi_block_diagonal
    "matmul_kernel_fallback": 0,        # kernel precondition miss → composed-dense
    "matmul_composed_dense": 0,         # multi-structured → dot_general compose
    "matmul_nonzero_fill_skip": 0,      # let existing densify path handle fills
    "elementwise_pure_dense_skip": 0,
    "elementwise_kernel_D_B": 0,
    "elementwise_kernel_implicit": 0,
    "elementwise_nonzero_fill_skip": 0,
}


def reset_stats() -> None:
    for k in DISPATCH_STATS:
        DISPATCH_STATS[k] = 0


def _bump(key: str) -> None:
    DISPATCH_STATS[key] = DISPATCH_STATS.get(key, 0) + 1


# --------------------------------------------------------------------------- #
# Structural predicates
# --------------------------------------------------------------------------- #
# A genuine meta-block-diagonal (Diagonal) dim — sparse, not compressed.
# ``block_size in {None, 1}`` is a *plain* diagonal (a permutation / scaled-
# identity factor); ``block_size > 1`` is a rectangular block.
from graphax.sparse.elemental._common import is_block_diagonal as _is_block_diagonal


def _is_implicit(d) -> bool:
    """A Compress-away dim: a plain Dense dim whose physical val axis was dropped
    (``axis is None``) while its logical size is the broadcast size > 1."""
    return (
        not d.is_sparse
        and not d.is_compressed
        and d.axis is None
        and int(d.logical_size) > 1
    )


def _has_structured_dim(st: "SparseTensor") -> bool:
    """Whether ANY dim of ``st`` is block-diagonal or implicit — i.e. the
    operand is structured and a dense path would be wasteful / wrong-shaped."""
    return any(_is_block_diagonal(d) or _is_implicit(d) for d in st.dims)


def _has_compressed_dim(st: "SparseTensor") -> bool:
    return any(d.is_compressed for d in st.dims)


# --------------------------------------------------------------------------- #
# Contracted-pair classification
# --------------------------------------------------------------------------- #
def _classify_pair(ld, rd) -> str:
    """Classify a single contracted pair ``(lhs primal dim, rhs out dim)``.

    Returns one of ``"dense"`` / ``"D_B"`` / ``"B_B"`` / ``"implicit"``.  A pair
    is *structured* iff it is not ``"dense"``.
    """
    l_b, r_b = _is_block_diagonal(ld), _is_block_diagonal(rd)
    l_i, r_i = _is_implicit(ld), _is_implicit(rd)
    if l_i or r_i:
        return "implicit"
    if l_b and r_b:
        return "B_B"
    if l_b or r_b:
        return "D_B"
    return "dense"


def _contracted_pairs(lhs: "SparseTensor", rhs: "SparseTensor"):
    """The aligned contracted pairs ``(lhs_dim, rhs_dim)`` — the SAME topology
    resolution matmul's tiled / densify paths use (``_align_contract_dims`` with
    ``embed=True``), so this module can't disagree with them on what contracts."""
    from graphax.sparse.ops.matmul import _align_contract_dims

    return _align_contract_dims(lhs.primal_dims, rhs.out_dims, embed=True)


# --------------------------------------------------------------------------- #
# Kernel-scope predicates (the 2-D-per-operand core the pairwise kernels own)
# --------------------------------------------------------------------------- #
def _is_2d_core(st: "SparseTensor") -> bool:
    return len(st.out_dims) == 1 and len(st.primal_dims) == 1


def _is_rectangular(d) -> bool:
    """A meta-block-diagonal dim whose block is wider than 1×1 — the rectangular
    case the tiled path's reshape can't consume (the genuine kernel gap)."""
    return _is_block_diagonal(d) and (d.block_size or 1) > 1


def _needs_elemental(pairs, kinds, lhs, rhs) -> bool:
    """Whether a contraction GENUINELY needs the elemental layer rather than the
    existing matmul path.

    The existing tiled / densify paths already contract PLAIN diagonals
    (``block_size in {None, 1}``) and pure-dense pairs correctly and
    byte-identically — intercepting those would only reassociate the float ops
    (a spurious EXACT-AD drift). We therefore intercept only the cases the
    existing path can't do: a RECTANGULAR block-diagonal contracted pair, an
    IMPLICIT (Compress-away) contracted pair, a ``B @ B`` pair, or an operand
    that still carries a rectangular / implicit structural dim (compressed dims
    were already expanded by the caller)."""
    for (ld, rd), k in zip(pairs, kinds):
        if k in ("implicit", "B_B"):
            return True
        if k == "D_B" and (_is_rectangular(ld) or _is_rectangular(rd)):
            return True
    # A structural rectangular / implicit dim that is NOT itself contracted (it
    # rides through to the output) still needs nominal-aware handling.
    for st in (lhs, rhs):
        for d in st.dims:
            if _is_rectangular(d) or _is_implicit(d):
                return True
    return False


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def try_elemental_matmul(lhs: "SparseTensor", rhs: "SparseTensor", count: bool = False):
    r"""First fast-path for ``matmul``: route a STRUCTURED contraction (one whose
    contracted dims include a block-diagonal or implicit dim, or whose operands
    carry compressed dims) through the elemental kernels / composed contraction.

    Returns ``None`` when the contraction is pure-dense (no structured contracted
    dim) — the existing matmul path then handles it, keeping exact-AD byte
    identical.  Otherwise returns the contracted ``SparseTensor`` (or, with
    ``count=True``, ``(result, (adds, muls, fmas))``) built with the canonical
    output-id convention so downstream contractions align.
    """
    # EXACT-AD GUARD: no approximation active -> defer entirely to the existing
    # (block-diagonal/zero-fill-efficient) path so reverse/forward stay exact and
    # byte-identical. The elemental kernels only handle approximation edges.
    if not approx_active():
        _bump("matmul_pure_dense_skip")
        return None

    # Operands with a non-zero fill: the structured kernels assume zero fill (the
    # canonical Jacobian case). Let the existing densify path own non-zero fills.
    if not (_is_zero_fill(lhs) and _is_zero_fill(rhs)):
        _bump("matmul_nonzero_fill_skip")
        return None

    # Expand any compressed operand into {D, B} so the kernels can consume it.
    lhs_m, rhs_m = _materialize_both(lhs, rhs)

    # Find the contracted pairs and classify them. If NONE is structured the
    # contraction is pure-dense — bail so the existing path stays byte-identical.
    try:
        pairs = _contracted_pairs(lhs_m, rhs_m)
    except Exception:
        return None
    kinds = [_classify_pair(ld, rd) for ld, rd in pairs]
    structured = [k for k in kinds if k != "dense"]

    if not structured and not _has_structured_dim(lhs_m) and not _has_structured_dim(rhs_m):
        _bump("matmul_pure_dense_skip")
        return None

    # Only intercept contractions the existing path can't do byte-identically:
    # plain-diagonal / pure-dense contractions stay on the existing path (no
    # float reassociation), preserving EXACT-AD.
    if not _needs_elemental(pairs, kinds, lhs_m, rhs_m):
        _bump("matmul_pure_dense_skip")
        return None

    result = _dispatch_matmul(lhs_m, rhs_m, pairs, kinds)
    if result is None:
        return None
    if count:
        from graphax.sparse.ops.matmul import _compute_matmul_count

        # Count from the operands ACTUALLY contracted (post-materialization):
        # for a compressed operand, lhs/rhs have a different shape/topology than
        # the expanded lhs_m/rhs_m the kernel ran on, so the un-materialized
        # operands give a wrong op count.
        return result, _compute_matmul_count(lhs_m, rhs_m, result)
    return result


def _dispatch_matmul(lhs, rhs, pairs, kinds):
    """Pick the kernel / composition for a structured contraction."""
    structured_idx = [i for i, k in enumerate(kinds) if k != "dense"]
    n_struct = len(structured_idx)

    # ---- exactly ONE structured pair: try the nnz-optimal pairwise kernel ---- #
    if n_struct == 1:
        si = structured_idx[0]
        kind = kinds[si]
        ld, rd = pairs[si]
        n_dense = len(kinds) - 1

        # The pairwise kernels own the 2-D-per-operand core (one out + one primal
        # dim each): a single contracted pair and NO other contracted dims.
        # contract_implicit additionally rides through arbitrary free dims (its
        # _emit handles multi-dim operands), so an implicit pair routes directly
        # whenever it is the sole structured contracted pair (n_dense == 0). When
        # neither applies (extra dense pairs / higher-rank), compose below.
        single_pair_core = (
            n_dense == 0 and _is_2d_core(lhs) and _is_2d_core(rhs)
        )
        if single_pair_core or (kind == "implicit" and n_dense == 0):
            out = _route_single_kernel(lhs, rhs, ld, rd, kind)
            if out is not None:
                return out

    # ---- multiple block-diagonal pairs against a fully-dense operand -------- #
    # Dense (x) N-block-diagonal-pairs (the attention / ViT structured
    # contraction): several mixed-type dims contracted at once where exactly ONE
    # operand is fully dense and the other carries >=1 DiagonalIndex pair plus
    # dense dims. Stays nnz-sparse via a single batched einsum over all meta axes
    # (no full prod_p N_p dense intermediate). Returns None when out of its
    # scope, so the composed-dense fallback below still owns the remainder.
    multiB = _try_multi_block_diagonal(lhs, rhs, kinds)
    if multiB is not None:
        _bump("matmul_kernel_multi_D_B")
        return multiB

    # ---- composed contraction (multi-structured / extra dims) --------------- #
    # MULTIPLE structured contracted pairs (e.g. two block-diagonal pairs + a
    # dense pair) are beyond any single pairwise kernel. We compose by expanding
    # ONLY the block-diagonal / implicit STRUCTURE of each operand to a plain
    # dense factor — keeping every dim's id and logical order — and re-running
    # the contraction through the existing tiled topology resolver. The resolver
    # aligns the contracted axes BY ID (``_align_tensor_ids`` + the topology
    # builder), which is what gets the surviving free-dim ORDER right; the
    # rectangular-block reshape that crashes the tiled path on a structured
    # operand no longer fires because the operands are now plain dense. The
    # result carries the canonical ``range(0, n)`` output ids (the tiled path's
    # ``finalize``), so downstream multi-edge contractions align.
    _bump("matmul_composed_dense")
    lhs_d = _to_dense_st(lhs)
    rhs_d = _to_dense_st(rhs)
    from graphax.sparse.ops.matmul import matmul as _matmul

    return _matmul(lhs_d, rhs_d)


def _try_multi_block_diagonal(lhs, rhs, kinds):
    """Attempt the Dense (x) N-block-diagonal-pair kernel.

    Only worth attempting when there are MULTIPLE structured contracted pairs
    (the case the single pairwise kernels decline) and at least one is a genuine
    block-diagonal pair. Returns the contracted SparseTensor or ``None`` (the
    kernel itself returns ``None`` for anything outside its scope, e.g. both
    operands structured, a B@B-style pair, or a surviving free diagonal pair —
    those stay on the composed-dense fallback)."""
    n_bd = sum(1 for k in kinds if k == "D_B")
    if n_bd == 0:
        return None
    from graphax.sparse.elemental.contract_D_B import (
        contract_dense_multi_block_diagonal,
    )

    try:
        return contract_dense_multi_block_diagonal(lhs, rhs)
    except (ValueError, NotImplementedError):
        # Precondition miss -> composed-dense fallback (correct), never abort.
        _bump("matmul_kernel_fallback")
        return None


def _route_single_kernel(lhs, rhs, ld, rd, kind):
    """Route a single-structured-pair 2-D-core contraction to its kernel.

    All three kernels are handed the dispatcher's already-resolved ``(ld, rd)``
    contracted pair (from ``_align_contract_dims``) so they don't re-derive it with
    their own positional heuristic — which could pick a DIFFERENT pair than the
    dispatcher classified and yield a wrong/transposed result. A kernel still
    raises ``ValueError`` if a precondition isn't met (e.g. ``contract_B_B`` can't
    resolve the ``other_id`` partner); that must NOT abort the whole gradient, so
    catch it and return ``None`` for ``_dispatch_matmul`` to compose via the
    (always-correct) dense fallback. ``_bump`` only on success so telemetry counts
    real kernel use; the ``matmul_kernel_fallback`` counter records the misses.
    """
    try:
        if kind == "implicit":
            from graphax.sparse.elemental.produce_compress import contract_implicit

            out = contract_implicit(lhs, rhs, ld, rd)
            _bump("matmul_kernel_implicit")
            return out
        if kind == "B_B":
            from graphax.sparse.elemental.contract_B_B import contract_B_B

            out = contract_B_B(lhs, rhs, ld, rd)
            _bump("matmul_kernel_B_B")
            return out
        if kind == "D_B":
            from graphax.sparse.elemental.contract_D_B import (
                contract_dense_block_diagonal,
            )

            out = contract_dense_block_diagonal(lhs, rhs, ld, rd)
            _bump("matmul_kernel_D_B")
            return out
    except (ValueError, NotImplementedError):
        _bump("matmul_kernel_fallback")
        return None
    return None


def try_elemental_elementwise(
    lhs: "SparseTensor",
    rhs: "SparseTensor",
    op: Callable,
    is_intersection: bool = False,
    count: bool = False,
):
    r"""First fast-path for ``elementwise``: route a structured (block-diagonal /
    implicit) elementwise op through the elemental kernels.

    Returns ``None`` when neither operand is structured (pure-dense elementwise)
    so the existing path handles it (exact-AD untouched), or when the structured
    shape is outside the pairwise kernels' 2-D core (the general tiled path then
    owns it).
    """
    # EXACT-AD GUARD (see try_elemental_matmul): no-op unless approximation active.
    if not approx_active():
        _bump("elementwise_pure_dense_skip")
        return None
    if not (_is_zero_fill(lhs) and _is_zero_fill(rhs)):
        _bump("elementwise_nonzero_fill_skip")
        return None

    lhs_m, rhs_m = _materialize_both(lhs, rhs)
    if lhs_m.shape != rhs_m.shape:
        return None

    l_struct = _has_structured_dim(lhs_m)
    r_struct = _has_structured_dim(rhs_m)
    if not l_struct and not r_struct:
        _bump("elementwise_pure_dense_skip")
        return None

    out = _dispatch_elementwise(lhs_m, rhs_m, op, is_intersection)
    if out is None:
        return None
    if count:
        from graphax.sparse.ops.elementwise import _ew_op_count

        n = _ew_op_count(lhs, rhs, is_intersection)
        return out, n
    return out


def _dispatch_elementwise(lhs, rhs, op, is_intersection):
    has_impl = any(_is_implicit(d) for d in lhs.dims) or any(
        _is_implicit(d) for d in rhs.dims
    )
    if has_impl:
        from graphax.sparse.elemental.produce_compress import elementwise_implicit

        _bump("elementwise_kernel_implicit")
        return elementwise_implicit(lhs, rhs, op, is_intersection=is_intersection)

    # Block-diagonal elementwise: the kernel owns the 2-D-per-operand core.
    if _is_2d_core(lhs) and _is_2d_core(rhs):
        from graphax.sparse.elemental.elementwise_D_B import (
            elementwise_dense_block_diagonal,
        )

        l_b = _is_block_diagonal(lhs.out_dims[0]) or _is_block_diagonal(
            lhs.primal_dims[0]
        )
        r_b = _is_block_diagonal(rhs.out_dims[0]) or _is_block_diagonal(
            rhs.primal_dims[0]
        )
        if l_b or r_b:
            _bump("elementwise_kernel_D_B")
            return elementwise_dense_block_diagonal(
                lhs, rhs, op, is_intersection=is_intersection
            )
    # Outside the 2-D core (multi-pair block elementwise) — let the general
    # tiled elementwise path own it (it handles the LCM / multi-axis machinery).
    return None


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _materialize_both(lhs, rhs):
    """Expand compressed operands to ``{D, B}`` (no-op when none are compressed)."""
    from graphax.sparse.elemental.materialize_C import materialize_compressed

    if _has_compressed_dim(lhs):
        lhs = materialize_compressed(lhs)
    if _has_compressed_dim(rhs):
        rhs = materialize_compressed(rhs)
    return lhs, rhs


def _to_dense_st(st: "SparseTensor") -> "SparseTensor":
    """Expand a structured operand to a plain ``DenseIndex``-only SparseTensor,
    PRESERVING each dim's id and logical order so the downstream id-based
    topology resolver pairs the contracted axes correctly.

    ``.dense()`` lays the array out in logical ``(out_dims..., primal_dims...)``
    order (each block-diagonal / implicit dim expanded to its full logical
    size), so a fresh ``DenseIndex`` per logical dim — same ids, same sizes,
    physical axis = logical position — describes that array exactly. The result
    has no block-diagonal / implicit / compressed dims, so a recursive matmul on
    it cannot re-enter the structured fast path (no infinite recursion).
    """
    from graphax.sparse.indexes import DenseIndex
    from graphax.sparse.tensor import SparseTensor

    arr = st.dense()  # (out_logical..., primal_logical...), scale/fill folded
    n_out = len(st.out_dims)
    out_dims = tuple(
        DenseIndex(d.id, int(d.logical_size), axis=i)
        for i, d in enumerate(st.out_dims)
    )
    primal_dims = tuple(
        DenseIndex(d.id, int(d.logical_size), axis=n_out + i)
        for i, d in enumerate(st.primal_dims)
    )
    return SparseTensor(
        out_dims, primal_dims, arr,
        fill_value=None, check_consistency=False,
    )


__all__ = [
    "try_elemental_matmul",
    "try_elemental_elementwise",
    "DISPATCH_STATS",
    "reset_stats",
]
