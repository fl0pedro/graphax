r"""Elemental materialize: CompressedIndex -> {Dense (D), Block-diagonal (B)}.

This module owns the single elemental operation that turns a *compressed*
operand — one whose dims are :class:`~graphax.sparse.indexes.CompressedIndex`
subclasses (:class:`BandedIndex` / :class:`ToeplitzIndex` / :class:`SetIndex`) —
into a closed ``{D, B}`` :class:`~graphax.sparse.tensor.SparseTensor` whose dims
are only ``DenseIndex`` (D) or non-compressed ``DiagonalIndex`` (B).  The
``contract_*`` and ``elementwise_*`` kernels consume only D and B, so this is the
boundary adapter that lets a compressed Jacobian factor enter the elemental
algebra.

================================================================================
1. MATH  —  what each compressed type expands to, and WHEN it yields B vs D
================================================================================
A compressed pair (two ``CompressedIndex`` dims linked by ``other_id``) is a
COMPACT encoding of a structured 2-D factor ``J`` of logical shape
``(R, C) = (out.logical_size, primal.logical_size)``.  ``J`` is zero off its
structural support and carries the operand's (statically-zero, in the canonical
Jacobian case) ``fill_value`` there.  Materialization expands the compact storage
into the FULL support that ``J`` represents, choosing the tightest closed form:

  * If the pair ``reduces_to_diagonal()`` — its support is exactly the
    meta-block diagonal of ``N`` rectangular ``(B_row x B_col)`` blocks — the
    materialized factor is **Block-diagonal (B)**: a ``DiagonalIndex`` pair
    carrying the ``(N, B_row, B_col)`` meta blocks.  ``N x`` smaller storage than
    a dense ``(R, C)``; off-diagonal cells are structurally zero, so nothing is
    lost.
  * Otherwise the support is genuinely 2-D (a band wider than one block, a
    windowed/Toeplitz incidence, a set-combination over an LCM grid) and the
    materialized factor is **Dense (D)**: a ``DenseIndex`` pair carrying the
    expanded ``(R, C)`` grid.  Denseness is correct, not wasteful — the support
    really does spread off any single diagonal.

----  BandedIndex (a finite block-band)  --------------------------------------
``J`` is non-zero only within a block-band of width ``W`` around the meta
diagonal: primary meta-row ``i`` couples to secondary meta-cols
``offset[i] .. offset[i]+W-1``, each coupling an explicit ``(B_row x B_col)``
sub-block (see :meth:`BandedIndex.densify_axis`).

  * ``W == 1`` with a centered (identity) offset and a SQUARE per-batch meta
    count ⇒ ``reduces_to_diagonal()`` ⇒ **B** (the single in-band slot per row
    IS the diagonal block; :meth:`to_meta_blocks` reads it directly).
  * ``W > 1`` (or a non-identity offset / rectangular meta) ⇒ **D**
    (off-diagonal band content cannot live on a single meta diagonal).

----  ToeplitzIndex (a windowed / convolutional incidence)  -------------------
``J`` is the rank-3 conv incidence ``M[p, q, k]`` (output ``p`` / input ``q`` /
tap ``k``; :meth:`ToeplitzIndex.indicator`) contracted over the one role stored
compactly in ``val`` (taps for ``d out/d lhs``, inputs for ``d out/d rhs``),
leaving the other two roles as the dense ``(R, C)`` pair.  A genuine window
always spreads off the diagonal, so ``reduces_to_diagonal()`` is ``False`` and
the result is **D** (the conv rule emits a true identity directly as a
``DiagonalIndex``, never as a ToeplitzIndex, so a ToeplitzIndex is only ever
built for a genuinely windowed axis).

----  SetIndex (an elementwise union / intersection)  -------------------------
``J`` places each side's per-meta blocks on a per-meta block diagonal over an LCM
grid ``(LCM_h x LCM_w)``, combines the two sides with the stored op, and the
combined per-meta grid sits on the M-meta diagonal of the full
``(M*LCM_h, M*LCM_w)`` form (see :meth:`SetIndex.densify_axis`).  Because all
content lives on the M-meta diagonal, ``reduces_to_diagonal()`` is ALWAYS
``True`` ⇒ **B** (a ``DiagonalIndex`` pair carrying the ``(M, LCM_h, LCM_w)``
per-meta grid, via :meth:`to_meta_blocks`); the dense ``M x`` form is only built
when full materialization is explicitly requested.

----  CLOSED RESULT (why it stays in {D, B})  ---------------------------------
Each materialized factor is either a ``DiagonalIndex`` pair (B) or a
``DenseIndex`` pair (D); neither carries a ``CompressedIndex`` dim, so the
elemental algebra is closed.  ``materialize_C(C)`` equals ``C.dense()`` cell for
cell — the B form is a loss-free compact view of the same dense factor.

================================================================================
2. ALGORITHM  (fusion-friendly, cost ~ nnz)
================================================================================
One dispatch on the pair's compressed type + ``reduces_to_diagonal()``:

  * B path (``reduces_to_diagonal()``):  read the compact meta blocks directly —
    ``primary.to_meta_blocks(val)`` (banded, ``(N, B_row, B_col)``) or
    ``sx.to_meta_blocks(val, fill)`` (set, ``(M, LCM_h, LCM_w)``) — and emit a
    ``DiagonalIndex`` pair.  No dense ``(R, C)`` grid is ever formed; cost and
    storage are ``~ nnz``.

  * D path:  call the index's own ``densify_axis`` — :meth:`BandedIndex.\
    densify_axis` (broadcast band placement, gather-free for a centered offset),
    :meth:`SetIndex.densify_axis` (per-meta block-diag + meta-stitch + op), or,
    for Toeplitz, contract the scatter-free :meth:`ToeplitzIndex.indicator`
    ``M[p,q,k]`` against ``val`` in a single ``einsum`` (the conv pair plus any
    coexisting Dense/Diagonal dims ride through that one einsum).  Emit a
    ``DenseIndex`` pair (Toeplitz: Dense/Diagonal dims pass through unchanged).

Every primitive is broadcast / where / reshape / einsum — XLA-fusable, no
``lax.gather`` / ``lax.scatter`` / python loop over blocks.  The fill is consumed
as a concrete array (``_eff_fill`` ⇒ 0 for the statically-zero ``None`` marker);
the RESULT tensor preserves the ``None`` marker (or the op-combined fill for a
SetIndex) so the materialized operand stays fast-path eligible downstream.

================================================================================
3. SCOPE / DISPATCH (see INTEGRATION NOTE at bottom)
================================================================================
``materialize_compressed(tensor)`` materializes a tensor whose compressed dims
are a single pure pair (one out + one primal ``BandedIndex`` / ``SetIndex``), or
a tensor carrying ``ToeplitzIndex`` pairs possibly coexisting with Dense/Diagonal
dims.  No-op on a tensor with no compressed dims.  Delegates the structural
expansion to the index methods + ``ops/utils._densify_compressed_dims`` (the
shared, CR-fixed densify orchestrator) so there is exactly one definition of each
compressed type's expansion; this module is the thin, documented elemental
boundary around it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphax.sparse.indexes import (
    BandedIndex,
    SetIndex,
    ToeplitzIndex,
)
from graphax.sparse.ops.utils import (
    _compressed_dims,
    _densify_compressed_dims,
)

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


def materialize_compressed(tensor: "SparseTensor", compact: bool = True) -> "SparseTensor":
    r"""Materialize a compressed operand into the closed ``{D, B}`` algebra.

    Returns a :class:`SparseTensor` whose dims are only ``DenseIndex`` (D) or
    non-compressed ``DiagonalIndex`` (B) — the contract / elementwise kernels
    consume exactly these.  The result equals ``tensor.dense()`` cell for cell
    (the B form is a loss-free compact view of the same dense factor).

    ``compact=True`` (the default, fusion-friendly form): a pair that
    :meth:`reduces_to_diagonal` materializes to the compact ``DiagonalIndex``
    pair (B) at ``N x`` lower storage; everything else materializes to a dense
    ``DenseIndex`` pair (D).  ``compact=False`` forces FULL dense materialization
    (every compressed pair becomes a ``DenseIndex`` pair) — used when a caller
    wants the raw dense grid rather than the tightest closed form.

    No-op (returns ``tensor`` unchanged) when the tensor carries no compressed
    dims.  Raises ``NotImplementedError`` for compressed shapes no producer emits
    yet (e.g. a Banded/Set pair coexisting with Dense/Diagonal dims), matching
    the shared densify orchestrator.
    """
    if not _compressed_dims(tensor):
        return tensor
    return _densify_compressed_dims(tensor, compact=compact)


def expansion_kind(tensor: "SparseTensor") -> str:
    r"""Statically report whether ``tensor``'s compressed pair materializes to a
    Block-diagonal (``"B"``) or a Dense (``"D"``) factor — the closed-form choice
    made by :func:`materialize_compressed` with ``compact=True``, decided purely
    from the pair's structure (``reduces_to_diagonal``), no data touched.

    ``"none"`` when the tensor carries no compressed dims; ``"mixed"`` for a
    Toeplitz tensor (its conv pair is always Dense, but it may coexist with
    pass-through Diagonal dims, so the tensor as a whole is not a single kind).
    """
    comp = _compressed_dims(tensor)
    if not comp:
        return "none"
    if any(isinstance(d, ToeplitzIndex) for d in comp):
        # The conv pair is always Dense; Diagonal dims may ride through.
        return "mixed"
    banded = [d for d in comp if isinstance(d, BandedIndex)]
    set_dims = [d for d in comp if isinstance(d, SetIndex)]
    if set_dims:
        # A SetIndex output is always meta-block-diagonal -> B.
        return "B"
    if banded:
        primary = next((d for d in banded if d.primary), banded[0])
        return "B" if primary.reduces_to_diagonal() else "D"
    return "D"


__all__ = ["materialize_compressed", "expansion_kind"]
