r"""Producer micro-action: ``Diag``  (``D -> B``).

This module owns the single vertex-elimination *producer* that turns a pair of
plain **Dense** (``DenseIndex`` / ``D``) logical dims of a ``SparseTensor`` into
a meta-block-diagonal (``DiagonalIndex`` / ``B``) pair — the structured
approximation the RL policy emits to throw away a Jacobian's off-block-diagonal
mass before the next contraction.  It is a *producer* (one operand in, one
operand out), not a binary contraction; the resulting ``B`` must then contract /
add correctly via the ``contract_D_B`` / ``contract_B_B`` / elementwise kernels.

================================================================================
1. MATH
================================================================================
Let ``st`` be a ``SparseTensor`` and pick two *Dense* logical dims, ``i`` (logical
size ``Ni``) and ``j`` (logical size ``Nj``).  Pick a block ``factor`` ``f`` with
``f | Ni`` and ``f | Nj``.  Write ``bi = Ni / f`` and ``bj = Nj / f``.

``Diag(i, j, f)`` keeps the ``f`` meta-diagonal blocks of the ``(Ni, Nj)`` slab
spanned by axes ``i, j`` and zeroes everything else.  Reshape the two logical
axes into a (meta, in-block) pair,

    i  ->  (gi, r)   with gi in [0, f), r in [0, bi)   (index  i = gi*bi + r)
    j  ->  (gj, c)   with gj in [0, f), c in [0, bj)   (index  j = gj*bj + c)

then the produced tensor ``T`` is, cell-for-cell against the input ``S``,

    T[..., i, ..., j, ...] = S[..., i, ..., j, ...]   if  gi == gj          (1)
    T[..., i, ..., j, ...] = 0                        if  gi != gj.

I.e. ``T`` masks ``S`` with the *meta-block-diagonal indicator* ``[gi == gj]``.
Equivalently, defining the per-meta block

    block[..., g, r, c, ...] = S[..., (g*bi + r), ..., (g*bj + c), ...],       (2)

``T`` is the block-diagonal matrix whose ``g``-th diagonal block is
``block[g]`` (an ``bi x bj`` rectangular block when ``bi != bj``) and whose
off-diagonal blocks are zero.  This is EXACTLY the ``(N, B_row, B_col)`` per-meta
layout that :func:`contract_D_B._diag_blocks` consumes (with ``N = f``,
``B_row = bi``, ``B_col = bj``), so the produced ``B`` is a first-class operand
of the existing block-diagonal contraction / densify machinery.

----  NON-ZERO SUPPORT & CLOSURE (why it stays in ``{D, B}``) -------------------
The mask ``[gi == gj]`` keeps only ``f * bi * bj`` cells of the original
``Ni * Nj = f^2 * bi * bj`` (per setting of the OTHER, untouched axes) — an
``f``-fold reduction.  Those surviving cells are precisely the meta-diagonal
blocks, which is by definition a ``DiagonalIndex`` (``B``) pair coupling axis
``i`` (block ``bi``) to axis ``j`` (block ``bj``).  Every OTHER dim of ``st`` is
untouched and keeps its kind.  Hence ``Diag`` maps ``D, D -> B`` on the chosen
pair and is closed in ``{D, B}``.  A degenerate ``f == 1`` produces ``bi = Ni``,
``bj = Nj``, a single block = the whole slab = no masking; we return ``st``
unchanged (no spurious ``B`` of meta-count 1).

================================================================================
2. ALGORITHM
================================================================================
We never materialize the ``Ni x Nj`` (or any ``N^2``) dense mask.  Working on the
``val`` array directly (cost ~ nnz of the result):

  1. Reshape the physical val axis carrying ``i`` from ``Ni`` to ``(f, bi)`` and
     the one carrying ``j`` from ``Nj`` to ``(f, bj)`` — pure ``reshape``, free.
  2. Extract the meta-diagonal blocks ``block[..., g, r, c, ...]`` (eq. (2))
     WITHOUT a gather: contract the two meta axes ``gi, gj`` against an identity
     ``eye(f)`` in one ``einsum`` —

         block[..., g, r, c, ...]
             = sum_{gi, gj} eye[g, gi] * eye[g, gj] * V[..., gi, r, gj, c, ...].

     XLA lowers this to a ``dot_general`` (broadcast + multiply + reduce); there
     is no ``lax.gather`` / ``scatter`` / python loop.  The result keeps ONE
     meta axis ``g`` (size ``f``), the two block axes ``r`` (``bi``), ``c``
     (``bj``), and all the untouched axes in place.
  3. Re-emit dims: axes ``i`` and ``j`` become a coupled ``DiagonalIndex`` pair
     (``size = f``, shared meta ``block_axis``-less ``axis = g``, block axes ``r``
     / ``c``); every other dim's physical ``axis`` / ``block_axis`` is shifted to
     account for the meta-axis collapse (``f, bi`` and ``f, bj`` -> one shared
     ``g`` plus ``r`` and ``c``).

A ``val is None`` (uniform all-ones) tensor stays ``val is None``: masking ones
to the meta-diagonal yields all-ones diagonal blocks, which is exactly the
``val=None`` ``DiagonalIndex`` reading (``dense()`` paints ones on the meta-block
diagonal, fill off it) — so we only rewrite the dims, not the data.

Complexity: the einsum touches ``f^2 * bi * bj * (other_free)`` MACs to PRODUCE
``f * bi * bj * (other_free)`` = ``nnz(B) * other_free`` output cells — i.e.
``O(f) * nnz`` work, with ``f`` the (small) meta count, versus the ``O(Ni*Nj) =
O(f^2 bi bj)`` of a dense mask.  All ops are ``reshape`` / ``einsum`` /
``transpose`` — HLO-fusion-friendly, no gather / scatter / python loop.

================================================================================
3. SCOPE / DISPATCH (see INTEGRATION NOTE at bottom)
================================================================================
``produce_diag(st, i, j, factor)`` requires the two chosen logical dims to be
plain ``DenseIndex`` (the ``D -> B`` producer).  It rejects already-sparse or
compressed dims (those are not the producer's job) and validates ``f | Ni`` and
``f | Nj``.  Extra/untouched dims of any kind (Dense, other Diagonal pairs) ride
through unchanged; only the chosen ``i, j`` pair is block-diagonalised.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax.numpy as jnp

from graphax.sparse.indexes import DiagonalIndex, Index

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


# --------------------------------------------------------------------------- #
# Logical-index resolution
# --------------------------------------------------------------------------- #
def _resolve(st: "SparseTensor", i: int, j: int):
    """Resolve logical positions ``i`` / ``j`` (into ``out_dims + primal_dims``)
    to their ``(is_out, rel_pos, Index)`` triples, validating range + identity."""
    out_len = len(st.out_dims)
    total = out_len + len(st.primal_dims)
    if not (0 <= i < total):
        raise ValueError(f"produce_diag: i={i} out of range [0, {total}).")
    if not (0 <= j < total):
        raise ValueError(f"produce_diag: j={j} out of range [0, {total}).")
    if i == j:
        raise ValueError(f"produce_diag: i and j must be distinct, got {i}.")

    def _one(p):
        if p < out_len:
            return True, p, st.out_dims[p]
        return False, p - out_len, st.primal_dims[p - out_len]

    return _one(i), _one(j)


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
def produce_diag(
    st: "SparseTensor", i: int, j: int, factor: int
) -> "SparseTensor":
    """Block-diagonalise the Dense logical pair ``(i, j)`` of ``st`` into a
    coupled ``DiagonalIndex`` (``B``) pair with meta-count ``factor``.

    ``i`` / ``j`` index into ``st.out_dims + st.primal_dims``; both must be plain
    ``DenseIndex``.  ``factor`` must divide both logical sizes.  Returns a new
    ``SparseTensor`` whose ``val`` holds ONLY the ``factor`` meta-diagonal blocks
    (the ``D->B`` producer of the module docstring); a ``factor == 1`` (no
    masking) returns ``st`` unchanged.

    The masking is computed by a single gather-free ``einsum`` over ``val`` (cost
    ~ ``factor * nnz(B)``), never an ``Ni x Nj`` dense mask.
    """
    from graphax.sparse.tensor import SparseTensor

    if factor <= 0:
        raise ValueError(f"produce_diag: factor must be positive, got {factor}.")

    (is_out_i, rel_i, di), (is_out_j, rel_j, dj) = _resolve(st, i, j)

    if di.is_sparse or di.is_compressed:
        raise ValueError(
            f"produce_diag: dim i={i} must be a plain DenseIndex (D->B producer), "
            f"got {type(di).__name__} (sparse={di.is_sparse}, "
            f"compressed={di.is_compressed})."
        )
    if dj.is_sparse or dj.is_compressed:
        raise ValueError(
            f"produce_diag: dim j={j} must be a plain DenseIndex (D->B producer), "
            f"got {type(dj).__name__} (sparse={dj.is_sparse}, "
            f"compressed={dj.is_compressed})."
        )

    Ni, Nj = di.logical_size, dj.logical_size
    if Ni % factor != 0 or Nj % factor != 0:
        raise ValueError(
            f"produce_diag: factor={factor} must divide both logical sizes "
            f"({Ni}, {Nj})."
        )

    if factor == 1:
        # f == 1: a single block spanning the whole slab = no off-diagonal mass
        # to drop = identity. Returning st (rather than a meta-count-1 B) keeps
        # the producer from fabricating a degenerate pair.
        return st

    bi, bj = Ni // factor, Nj // factor

    # ----------------------------------------------------------------------- #
    # val=None (uniform all-ones): the meta-diagonal of an all-ones slab is an
    # all-ones set of diagonal blocks, which is precisely the val=None reading
    # of a DiagonalIndex pair (dense() paints ones on the meta-block diagonal,
    # fill off it). So only the dims change; val stays None.
    # ----------------------------------------------------------------------- #
    if st.val is None:
        new_di, new_dj = _diag_pair(
            di, dj, factor, bi, bj, axis=None, bi_axis=None, bj_axis=None
        )
        return _rebuild(st, is_out_i, rel_i, is_out_j, rel_j, new_di, new_dj, None)

    # ----------------------------------------------------------------------- #
    # Concrete val: reshape the i / j physical axes to (f, b) and contract the
    # two meta axes against eye(f) to keep the meta-diagonal blocks (gather-free).
    # ----------------------------------------------------------------------- #
    ax_i = di.axis
    ax_j = dj.axis
    if ax_i is None or ax_j is None:
        # An implicit (axis is None) chosen dim isn't carried by val yet — make
        # it explicit so the reshape/einsum has a physical axis to act on. dense()
        # with hard=True broadcasts implicit dims into val.
        from graphax.sparse.ops.dense import dense as _dense_op

        needs = [p for p, d in ((i, di), (j, dj)) if d.axis is None]
        st = _dense_op(st, axes=needs, hard=True)
        # Re-resolve against the materialized tensor (ids are stable).
        (is_out_i, rel_i, di), (is_out_j, rel_j, dj) = _resolve(st, i, j)
        ax_i, ax_j = di.axis, dj.axis

    val = st.val
    new_val, g_axis, bi_axis, bj_axis = _meta_diagonal(val, ax_i, ax_j, factor, bi, bj)

    new_di, new_dj = _diag_pair(
        di, dj, factor, bi, bj, axis=g_axis, bi_axis=bi_axis, bj_axis=bj_axis
    )
    return _rebuild(
        st, is_out_i, rel_i, is_out_j, rel_j, new_di, new_dj, new_val,
        old_ax_i=ax_i, old_ax_j=ax_j,
    )


# --------------------------------------------------------------------------- #
# Gather-free meta-diagonal extraction
# --------------------------------------------------------------------------- #
def _meta_diagonal(val, ax_i: int, ax_j: int, f: int, bi: int, bj: int):
    """Keep the ``f`` meta-diagonal blocks of ``val`` along physical axes
    ``ax_i`` (size ``f*bi``) and ``ax_j`` (size ``f*bj``).

    Returns ``(new_val, g_axis, bi_axis, bj_axis)`` where ``new_val`` carries a
    SINGLE shared meta axis ``g`` (size ``f``) plus block axes of size ``bi`` /
    ``bj``, and the three returned ints are their physical positions in
    ``new_val``.  Pure reshape + one ``einsum`` (``dot_general``) — no gather.

    Layout convention: we move the chosen axes to the front as
    ``(gi, r, gj, c, *rest)``, contract ``gi, gj`` with ``eye`` to ``(g, r, c,
    *rest)``, so ``g_axis=0, bi_axis=1, bj_axis=2`` and the untouched ``*rest``
    axes follow in their original relative order.  The caller maps every other
    dim's old physical axis through :func:`_remap_other_axis` accordingly.
    """
    ndim = val.ndim
    rest = [a for a in range(ndim) if a not in (ax_i, ax_j)]
    # Bring (i, j) to the front: (i, j, *rest).
    v = jnp.transpose(val, [ax_i, ax_j] + rest)
    rest_shape = list(v.shape[2:])
    # Split each meta axis: (i, j, *rest) -> (gi, r, gj, c, *rest).
    v = v.reshape([f, bi, f, bj] + rest_shape)
    eye = jnp.eye(f, dtype=v.dtype)
    # block[g, r, c, *rest] = sum_{gi,gj} eye[g,gi] eye[g,gj] v[gi,r,gj,c,*rest]
    #   = the (gi==gj==g) meta-diagonal blocks. dot_general, gather-free.
    new_val = jnp.einsum("gi,gj,irjc...->grc...", eye, eye, v)
    return new_val, 0, 1, 2


# --------------------------------------------------------------------------- #
# Dim emission
# --------------------------------------------------------------------------- #
def _diag_pair(di: Index, dj: Index, f: int, bi: int, bj: int,
               axis, bi_axis, bj_axis):
    """Build the coupled ``DiagonalIndex`` pair replacing ``di`` / ``dj``.

    Both sides share meta ``size = f`` and the same physical meta ``axis``
    (the diagonal stores ONE meta axis, per :func:`contract_D_B._diag_blocks`);
    side ``i`` carries block ``bi`` (axis ``bi_axis``), side ``j`` block ``bj``
    (axis ``bj_axis``).  A size-1 block collapses to ``block_size=None`` /
    ``block_axis=None`` (no physical block axis), matching the test builder /
    ``contract_D_B`` conventions.
    """
    new_di = DiagonalIndex(
        id=di.id, size=f, axis=axis, other_id=dj.id,
        block_size=bi if bi > 1 else None,
        block_axis=bi_axis if bi > 1 else None,
    )
    new_dj = DiagonalIndex(
        id=dj.id, size=f, axis=axis, other_id=di.id,
        block_size=bj if bj > 1 else None,
        block_axis=bj_axis if bj > 1 else None,
    )
    return new_di, new_dj


def _remap_other_axis(p, old_ax_i, old_ax_j):
    """Map an untouched dim's old physical axis ``p`` to its position in the
    new val layout ``(g, r, c, *rest)``.

    ``_meta_diagonal`` moves ``ax_i, ax_j`` to the front (collapsed to ``g`` at
    axis 0, with block axes ``r``/``c`` at 1/2) and keeps every other axis in
    its original relative order at positions ``3, 4, ...``.  So an old axis ``p``
    (neither ``ax_i`` nor ``ax_j``) lands at ``3 + (#untouched axes before p)``.
    """
    if p is None:
        return None
    lower = {old_ax_i, old_ax_j}
    # untouched axes before p, in original order
    n_before = sum(1 for a in range(p) if a not in lower)
    return 3 + n_before


def _rebuild(st, is_out_i, rel_i, is_out_j, rel_j, new_di, new_dj, new_val,
             *, old_ax_i=None, old_ax_j=None):
    """Re-assemble the SparseTensor: swap dims ``i`` / ``j`` for the new
    DiagonalIndex pair and (for the concrete-val path) remap every untouched
    dim's physical ``axis`` / ``block_axis`` through :func:`_remap_other_axis`.

    For the ``val is None`` path (``new_val is None``) the untouched dims keep
    their (axis-less or original) pointers — only ``i`` / ``j`` change — and the
    new pair is emitted axis-less too (handled by the caller passing axis=None).
    """
    from graphax.sparse.tensor import SparseTensor

    remap = new_val is not None

    def _map_dim(d, slot_is_out, slot_rel):
        # Is THIS slot one of the two we are replacing?
        if slot_is_out == is_out_i and slot_rel == rel_i:
            return new_di
        if slot_is_out == is_out_j and slot_rel == rel_j:
            return new_dj
        if not remap:
            return d
        new_axis = _remap_other_axis(getattr(d, "axis", None), old_ax_i, old_ax_j)
        if d.is_sparse:
            new_block = _remap_other_axis(d.block_axis, old_ax_i, old_ax_j)
            return replace(d, axis=new_axis, block_axis=new_block)
        return replace(d, axis=new_axis)

    new_out = tuple(
        _map_dim(d, True, p) for p, d in enumerate(st.out_dims)
    )
    new_primal = tuple(
        _map_dim(d, False, p) for p, d in enumerate(st.primal_dims)
    )

    return SparseTensor(
        new_out,
        new_primal,
        new_val,
        scalar_mult=st.scalar_mult,
        fill_value=st.fill_value,  # None (statically-zero) preserved
        check_consistency=False,
    )
