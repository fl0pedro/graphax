r"""Contract Block-Diagonal <-> Block-Diagonal  (``B @ B``).

This module owns the single elemental operation: contract a meta-block-diagonal
(:class:`~graphax.sparse.indexes.DiagonalIndex`, "B") dim of the left operand
against a meta-block-diagonal dim of the right operand, where BOTH contracted
dims are block-diagonal. It is the closed, fusion-friendly kernel that
``matmul`` should dispatch to for the ``(B, B)`` contracted-pair case.


1. MATH
=======

A meta-block-diagonal factor is a matrix that is non-zero only on a diagonal of
rectangular blocks. The left operand ``A`` has logical shape
``(M_out, K)`` and is block-diagonal with ``Na`` meta blocks::

    A = blockdiag( A_0, A_1, ..., A_{Na-1} ),   A_i  shape (P, Ka)

so ``M_out = Na * P`` and ``K = Na * Ka``. Concretely
``A[i*P + r, i*Ka + c] = A_i[r, c]`` and ``0`` elsewhere (the *support* of A is
exactly the ``Na`` diagonal blocks — ``Na * P * Ka`` non-zeros instead of
``M_out * K``).

The right operand ``B`` has logical shape ``(K, M_prim)`` and is block-diagonal
with ``Nb`` meta blocks::

    B = blockdiag( B_0, ..., B_{Nb-1} ),        B_j  shape (Kb, Q)

with ``K = Nb * Kb`` and ``M_prim = Nb * Q``.

The two contracted dims share the same *logical* size ``K`` but may have
DIFFERENT block geometry (``Na, Ka``) vs (``Nb, Kb``); only ``Na*Ka == Nb*Kb``
is required.

The contraction is the ordinary matrix product ``C = A @ B`` (shape
``(M_out, M_prim)``), summing over the shared logical axis ``K``.

CLOSED RESULT.  Let ``G = gcd(Na, Nb)``, ``ra = Na / G``, ``rb = Nb / G``.
Group the meta index of each operand into ``G`` outer groups of ``ra`` (resp.
``rb``) inner blocks.  A's group ``g`` spans contracted columns
``[g*(K/G), (g+1)*(K/G))`` and rows ``[g*ra*P, (g+1)*ra*P)``; B's group ``g``
spans contracted rows ``[g*(K/G), (g+1)*(K/G))`` and columns
``[g*rb*Q, (g+1)*rb*Q)``.  Because both operands are block-diagonal, A's row
group ``g`` only has support inside contracted-column group ``g`` and B's column
group ``g`` only inside contracted-row group ``g`` — so the product couples ONLY
matching groups::

    C = blockdiag( C_0, ..., C_{G-1} ),   C_g  shape (ra*P, rb*Q)
    C_g = Adiag_g @ Bdiag_g

where ``Adiag_g`` is the ``(ra*P) x (K/G)`` block-diagonal of A's ``ra`` blocks
in group ``g`` and ``Bdiag_g`` the ``(K/G) x (rb*Q)`` block-diagonal of B's
``rb`` blocks in group ``g``.

Hence **the result is again meta-block-diagonal** with ``G`` meta blocks of
shape ``(ra*P) x (rb*Q)`` — it stays in ``{D, B}``:

  * ``G > 1``  ⇒ a ``DiagonalIndex`` pair (a "B"), block size ``(ra*P, rb*Q)``.
  * ``G == 1`` ⇒ a single dense block ⇒ a ``DenseIndex`` pair (a "D").

Non-zero support of ``C``: ``G * (ra*P) * (rb*Q)`` cells — never the full
``M_out * M_prim`` (unless ``G == 1``, where the single group block is
genuinely dense, which is the correct dense answer).

Special / aligned case (``Na == Nb == G``, so ``ra == rb == 1``):
``C_g = A_g @ B_g`` — a pure per-meta-block matmul. This is the nnz-optimal
common case (attention heads, grouped MLPs).


2. ALGORITHM
============

Two regimes, both gather-free / loop-free (broadcast + reshape + ``einsum`` /
``dot_general`` only — XLA-fusable):

(a) ALIGNED (``Na == Nb``):  one batched einsum over the meta axis::

        C_meta = einsum('n p k, n k q -> n p q', A_blocks, B_blocks)

    Cost ``~ Na * P * Ka * Q`` == result nnz times the contracted block width —
    proportional to nnz, never ``N^2``.  No intermediate densification.

(b) MISALIGNED (``Na != Nb``):  refine to ``G = gcd`` groups. Reshape A's meta
    axis to ``(G, ra)`` and B's to ``(G, rb)``.  Within each group scatter the
    ``ra`` (resp ``rb``) blocks onto the group's block-diagonal using an
    identity (``eye``) mask broadcast — NOT ``lax.scatter`` — producing
    ``Adiag`` ``(G, ra*P, K/G)`` and ``Bdiag`` ``(G, K/G, rb*Q)``, then one
    batched ``einsum('g p k, g k q -> g p q')``.

    Cost ``~ G * (ra*P) * (K/G) * (rb*Q)`` — result-nnz times ``K/G``; only the
    *group-local* contracted width ``K/G`` is materialized, never the full ``K``
    nor the full ``M_out * M_prim`` dense product.  The aligned regime is the
    ``ra == rb == 1`` special case (``K/G == Ka == Kb``) and could share this
    path, but the dedicated einsum in (a) skips the eye-scatter for the
    overwhelmingly common attention/MLP geometry.

``scalar_mult`` of both operands folds into the output ``scalar_mult``
(linear); ``val is None`` densifies to an all-ones block buffer (matching
``dense()`` / the matmul ``val=None`` convention).  Both inputs are assumed
zero-fill (the structured ``{D,B}`` algebra — off-block-diagonal cells are
structurally zero); a non-zero ``fill_value`` is out of scope for this kernel
(matmul reroutes that to the densify path before reaching here).


3. RESULT INDEX TYPES
=====================

``contract_B_B`` returns a ``SparseTensor`` whose single out dim carries A's
out block (size ``ra*P``) and single primal dim carries B's primal block
(size ``rb*Q``):

  * ``G > 1``  → ``DiagonalIndex`` pair (meta count ``G``).
  * ``G == 1`` → ``DenseIndex`` pair.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp

from graphax.sparse.elemental._common import canonical_block_buffer
from graphax.sparse.indexes import DenseIndex, Index, DiagonalIndex

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


def _block_buffer(st: "SparseTensor", dim: Index, other: Index) -> jnp.ndarray:
    """Canonicalize ``st.val`` for a single block-diagonal pair ``(dim, other)``
    into ``(meta, block_row, block_col)`` order, where ``block_row`` is
    ``dim``'s block and ``block_col`` is ``other``'s block.

    Reads ``.axis`` (meta) and ``.block_axis`` (per-meta block) for indirection
    — the physical ``val`` axis order is implementation-defined, so we never
    assume positions.  ``val is None`` densifies to an all-ones buffer (matching
    ``dense()`` / matmul's ``val=None`` convention).

    Uses the STRICT present-axis check (an axis counts only if in range with the
    expected length) — historically tighter than its ``contract_D_B`` /
    ``elementwise_D_B`` siblings; preserved via ``strict_present=True``.
    """
    N = dim.size
    Br = dim.block_size or 1
    Bc = other.block_size or 1
    meta_ax = dim.axis if dim.axis is not None else other.axis
    slots = [(meta_ax, N), (dim.block_axis, Br), (other.block_axis, Bc)]
    return canonical_block_buffer(
        st.val, slots, dtype=st.dtype, strict_present=True
    )


def _scatter_block_diag(blocks: jnp.ndarray) -> jnp.ndarray:
    """Scatter ``(G, r, b_row, b_col)`` per-group blocks onto each group's
    block-diagonal, producing ``(G, r*b_row, r*b_col)``.

    Gather-free: an ``(r, r)`` identity mask broadcast against the blocks places
    block ``i`` at rows ``[i*b_row]`` / cols ``[i*b_col]`` and zeros elsewhere —
    XLA fuses this into the surrounding einsum (no ``lax.scatter``)."""
    G, r, b_row, b_col = blocks.shape
    eye = jnp.eye(r, dtype=blocks.dtype)  # (r, r): block-index i -> diag target j
    # The diagonal places block i at rows [i*b_row, (i+1)*b_row) and cols
    # [i*b_col, (i+1)*b_col): row index = i*b_row + p, col index = i*b_col + q.
    # placed[g, i, p, j, q] = blocks[g, i, p, q] * eye[i, j]
    #   blocks → (G, i, p, 1, q);  eye → (1, i, 1, j, 1)
    placed = blocks[:, :, :, None, :] * eye[None, :, None, :, None]
    # rows = (i, p) → i*b_row + p ; cols = (j, q) → j*b_col + q. The eye makes
    # j == i on-diagonal, so this is exactly the block-diagonal placement.
    return placed.reshape(G, r * b_row, r * b_col)


def contract_B_B(
    lhs: "SparseTensor",
    rhs: "SparseTensor",
    lhs_contract: Index,
    rhs_contract: Index,
) -> "SparseTensor":
    r"""Contract block-diagonal ``lhs_contract`` (a primal dim of ``lhs``)
    against block-diagonal ``rhs_contract`` (an out dim of ``rhs``), where BOTH
    are :class:`DiagonalIndex` (the ``B @ B`` case).

    ``lhs`` is ``(out_dim, lhs_contract)`` block-diagonal, ``rhs`` is
    ``(rhs_contract, primal_dim)`` block-diagonal.  Returns the closed
    meta-block-diagonal product (see the module docstring).

    Requirements (raise ``ValueError`` otherwise):
      * ``lhs_contract`` and ``rhs_contract`` are both sparse (``DiagonalIndex``)
        and not compressed.
      * equal contracted logical size.
      * each operand has exactly one out and one primal dim, paired with the
        contracted dim (the canonical 2-D ``B @ B`` shape).
    """
    if not (lhs_contract.is_sparse and not lhs_contract.is_compressed):
        raise ValueError("contract_B_B: lhs_contract must be a DiagonalIndex (B).")
    if not (rhs_contract.is_sparse and not rhs_contract.is_compressed):
        raise ValueError("contract_B_B: rhs_contract must be a DiagonalIndex (B).")
    if lhs_contract.logical_size != rhs_contract.logical_size:
        raise ValueError(
            f"contract_B_B: contracted logical size mismatch "
            f"{lhs_contract.logical_size} vs {rhs_contract.logical_size}"
        )

    # Resolve the partner (out) dim of lhs and (primal) dim of rhs.
    lhs_out = _partner(lhs, lhs_contract, "out")
    rhs_primal = _partner(rhs, rhs_contract, "primal")
    if lhs_out is None or rhs_primal is None:
        raise ValueError(
            "contract_B_B: each operand must carry the contracted dim's partner "
            "(lhs out-side / rhs primal-side) — canonical 2-D B@B shape."
        )

    # Block buffers, canonicalized to (meta, block_row, block_col).
    #   A: rows = lhs_out block (P), cols = lhs_contract block (Ka)
    #   B: rows = rhs_contract block (Kb), cols = rhs_primal block (Q)
    A = _block_buffer(lhs, lhs_out, lhs_contract)        # (Na, P, Ka)
    B = _block_buffer(rhs, rhs_contract, rhs_primal)     # (Nb, Kb, Q)

    Na, P, Ka = A.shape
    Nb, Kb, Q = B.shape
    K = Na * Ka  # == Nb * Kb (logical contracted size, checked above)

    G = math.gcd(Na, Nb)
    ra, rb = Na // G, Nb // G

    if Na == Nb:
        # Aligned: pure per-meta-block matmul (nnz-optimal common case).
        C = jnp.einsum("npk,nkq->npq", A, B)             # (G, P, Q)
        block_row, block_col = P, Q
    else:
        # Misaligned: refine to G groups, scatter each group's blocks onto its
        # block-diagonal, then one batched einsum over the group-local width.
        A4 = A.reshape(G, ra, P, Ka)
        B4 = B.reshape(G, rb, Kb, Q)
        Adiag = _scatter_block_diag(A4)                  # (G, ra*P, ra*Ka=K/G)
        Bdiag = _scatter_block_diag(B4)                  # (G, rb*Kb=K/G, rb*Q)
        C = jnp.einsum("gpk,gkq->gpq", Adiag, Bdiag)     # (G, ra*P, rb*Q)
        block_row, block_col = ra * P, rb * Q

    scalar_mult = lhs.scalar_mult * rhs.scalar_mult
    return _wrap_result(C, G, block_row, block_col, scalar_mult)


def _partner(st: "SparseTensor", dim: Index, side: str) -> Index | None:
    """The dim on ``side`` ('out'/'primal') of ``st`` paired with ``dim`` via
    ``other_id`` (the block-diagonal partner)."""
    pool = st.out_dims if side == "out" else st.primal_dims
    for d in pool:
        if d.id == dim.other_id:
            return d
    return None


def _wrap_result(
    C: jnp.ndarray,
    G: int,
    block_row: int,
    block_col: int,
    scalar_mult: jnp.ndarray,
) -> "SparseTensor":
    """Wrap the ``(G, block_row, block_col)`` meta-block buffer into a closed
    ``{D, B}`` ``SparseTensor``.  ``G == 1`` collapses to a ``DenseIndex`` pair
    (a single dense block); ``G > 1`` stays a ``DiagonalIndex`` pair."""
    from graphax.sparse.tensor import SparseTensor

    if G == 1:
        # Single dense block → plain dense axes; drop the meta axis.
        dense_val = C[0]  # (block_row, block_col)
        out_dim = DenseIndex(0, block_row, axis=0)
        primal_dim = DenseIndex(1, block_col, axis=1)
        return SparseTensor(
            (out_dim,),
            (primal_dim,),
            dense_val,
            scalar_mult=scalar_mult,
            fill_value=None,  # structured {D,B} → zero off-block-diagonal
            check_consistency=False,
        )

    out_dim = DiagonalIndex(
        0, G, axis=0, other_id=1, block_size=block_row, block_axis=1
    )
    primal_dim = DiagonalIndex(
        1, G, axis=0, other_id=0, block_size=block_col, block_axis=2
    )
    return SparseTensor(
        (out_dim,),
        (primal_dim,),
        C,  # (G, block_row, block_col)
        scalar_mult=scalar_mult,
        fill_value=None,
        check_consistency=False,
    )
