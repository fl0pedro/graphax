r"""Elemental contraction: Dense <-> BlockDiagonal  (``D @ B`` and ``B @ D``).

This module owns the single vertex-elimination contraction where ONE operand
has a meta-block-diagonal (``DiagonalIndex`` / ``B``) *contracted* dim and the
OTHER operand is plain Dense (``D``) on its contracted side.

================================================================================
1. MATH
================================================================================
Vertex elimination contracts ``lhs.primal_dims`` against ``rhs.out_dims`` over a
shared logical axis (the "post @ pre" of two edge-Jacobians).  We treat the case
where exactly one operand's contracted dim is a ``DiagonalIndex`` block-diagonal
factor and the other operand's contracted dim is a plain ``DenseIndex``.

A ``DiagonalIndex`` pair (linked by ``other_id``) encodes a *meta-block-diagonal*
matrix.  Let the pair have meta-count ``N``.  One side (the contracted side) has
block size ``B_c`` so its logical extent is ``K = N * B_c``; the partner (the
free side that survives the contraction) has block size ``B_f`` so its logical
extent is ``F = N * B_f``.  The pair is RECTANGULAR when ``B_c != B_f``.  The
matrix it represents, ``J``, is non-zero ONLY on the meta-diagonal:

    J[g*B_row + r, h*B_col + c] = block[g, r, c]   if g == h, else 0.

so as a contraction kernel it restricts the sum to within-block index pairs.

----  D @ B  (block-diagonal on the RIGHT / contracted out-side) ---------------
``lhs`` is dense with contracted axis ``K``; ``rhs`` is the block-diagonal pair
with contracted out-side ``K`` (meta ``N``, block ``B_c``) coupled to free
primal-side ``F`` (meta ``N``, block ``B_f``).  Write a free output column
``m = g*B_f + j`` (meta-block ``g``, in-block ``j``).  Because ``rhs`` is
block-diagonal, only contracting positions ``k = g*B_c + i`` in the SAME meta
block ``g`` are non-zero, so:

    out[..., m] = sum_{k}  lhs[..., k] * J[k, m]
                = sum_{i}  lhs[..., g*B_c + i] * block[g, i, j].          (D@B)

----  B @ D  (block-diagonal on the LEFT / contracted primal-side) -------------
``lhs`` is the block-diagonal pair with free out-side ``F`` (meta ``N``, block
``B_f``) coupled to contracted primal-side ``K`` (meta ``N``, block ``B_c``);
``rhs`` is dense with contracted out-axis ``K``.  A free output row
``f = g*B_f + j`` only draws on contracting positions ``k = g*B_c + i`` in its
own meta block ``g``:

    out[f, ...] = sum_{k}  J[f, k] * rhs[k, ...]
                = sum_{i}  block[g, j, i] * rhs[g*B_c + i, ...].          (B@D)

----  CLOSED RESULT (why it stays in {D, B}) -----------------------------------
The dense operand carries NO block structure across the contraction, so each
output free position mixes the dense operand's OTHER (non-contracted) free axes
freely.  Consequently:

  * The block-diagonal's free dim ``F = N*B_f`` survives as a single
    *dense* axis of the output (each free position ``m`` is computed
    independently; there is no surviving meta-pairing because the partner
    contracted dim was consumed).
  * The dense operand's non-contracted dims pass through as dense axes.

Hence the result is **Dense (D)** in every axis: ``D @ B -> D`` and
``B @ D -> D``.  The non-zero support of the result is the full
(``lhs_free`` x ``rhs_free``) grid — denseness is correct, not wasteful: a dense
row times a block-diagonal genuinely produces a dense row.  The *savings* are in
the CONTRACTION COST, not the output storage: we never form the ``K x F`` dense
``J``; we contract block-by-block over the meta diagonal.

================================================================================
2. ALGORITHM
================================================================================
Reshape the dense operand's contracted axis ``K`` into ``(N, B_c)`` and the
block-diagonal ``val`` into its canonical ``(N, B_row, B_col)`` per-meta blocks,
then contract the shared ``(N, B_c)`` with a SINGLE batched ``einsum`` (one
``dot_general`` with ``N`` as a batch axis and ``B_c`` as the contract axis).
The free block axis ``B_f`` of the diagonal becomes a new dense output axis;
fold ``(N, B_f) -> F`` by a reshape.

  D@B:  out[*L, g, j] = einsum('...gi,gij->...gj', lhs_NBc, blocks)   then (g,j)->F
  B@D:  out[g, j, *R] = einsum('gji,gi...->gj...', blocks, rhs_NBc)   then (g,j)->F

Complexity:  the einsum touches ``N * B_c * B_f * (other_free)`` MACs — i.e.
``nnz(B) * other_free`` — versus the dense ``K * F * other_free = N^2 * B_c * B_f
* other_free``.  An ``N x`` win, exactly the block-diagonal sparsity factor.  All
ops are reshape / transpose / dot_general — XLA-fusion-friendly, no gather /
scatter / python loop over blocks (the ``N`` meta loop is a batched dot_general
axis).

scalar_mult and a possible non-zero ``fill_value`` on the DENSE operand are
folded in:  a dense operand with uniform fill ``f`` over its contracted axis
adds ``f * (block column sums)`` — handled by densify-fold fallback only when a
non-zero fill is present (the fast path assumes zero fill, the common case).

================================================================================
3. SCOPE / DISPATCH (see INTEGRATION NOTE at bottom)
================================================================================
``contract_dense_block_diagonal(lhs, rhs)`` handles the 2-D-per-operand core:
each operand has exactly one out dim and one primal dim, the contracted pair is
(Dense, DiagonalIndex) in either order, and the diagonal pair's two sides are
the contracted + the surviving free dim.  Extra batch/leftover axes on the dense
operand are carried through as leading/trailing dense axes.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING

import jax.numpy as jnp

from graphax.sparse.dtype_compute import _scaled_mul
from graphax.sparse.indexes import DenseIndex, DiagonalIndex, Index
from graphax.sparse.ops.utils import _compute_dtype

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


# --------------------------------------------------------------------------- #
# Canonicalization helpers
# --------------------------------------------------------------------------- #
def _diag_blocks(st: "SparseTensor", contract_dim: Index, free_dim: Index) -> jnp.ndarray:
    """Extract a DiagonalIndex pair's per-meta blocks as ``(N, B_contract,
    B_free)``.

    ``st.val`` carries (in some physical-axis order given by each dim's ``axis``
    / ``block_axis``) a meta axis of size ``N`` shared by both sides, the
    contracted side's block axis (size ``B_contract``) and the free side's block
    axis (size ``B_free``).  We transpose those three physical axes to the
    canonical ``(N, B_contract, B_free)`` order.  Any size-1 (absent) block axis
    is broadcast in.  ``val is None`` (uniform-ones structure) materializes to
    ones of the canonical shape.
    """
    N = contract_dim.size
    B_c = contract_dim.block_size or 1
    B_f = free_dim.block_size or 1

    if st.val is None:
        # val=None ⇒ all-ones structure (matches dense()); the meta-diagonal
        # blocks are all ones of the canonical shape.
        return jnp.ones((N, B_c, B_f), dtype=st.dtype)

    val = st.val
    # The shared meta axis: both sides advertise the same physical ``axis``
    # (the diagonal stores ONE meta axis, not two). Prefer the contracted
    # side's, fall back to the free side's.
    meta_axis = contract_dim.axis if contract_dim.axis is not None else free_dim.axis
    bc_axis = contract_dim.block_axis
    bf_axis = free_dim.block_axis

    # Gather the PRESENT canonical axes (meta, B_c, B_f order) to the front,
    # dropping leftover axes (which are size-1 for a clean diagonal val).
    target_present = (meta_axis is not None, bc_axis is not None, bf_axis is not None)
    present = [a for a in (meta_axis, bc_axis, bf_axis) if a is not None]
    leftover = [a for a in range(val.ndim) if a not in present]
    perm = present + leftover
    v = val.transpose(perm) if perm != list(range(val.ndim)) else val
    # Keep only the present canonical axes (leftover are size-1).
    present_sizes = tuple(sz for p, sz in zip(target_present, (N, B_c, B_f)) if p)
    v = v.reshape(present_sizes)
    # Re-insert singleton axes for the absent canonical axes, then broadcast to
    # the full (N, B_c, B_f) block layout.
    expand_shape = tuple(sz if p else 1 for p, sz in zip(target_present, (N, B_c, B_f)))
    v = v.reshape(expand_shape)
    return jnp.broadcast_to(v, (N, B_c, B_f))


def _dense_with_contract_last(
    st: "SparseTensor", contract_dim: Index
) -> tuple[jnp.ndarray, list[Index]]:
    """Materialize a dense operand to ``(*free, K)`` — the contracted axis last,
    the surviving free dims (in their original order) leading.

    Returns the array plus the list of surviving free ``Index`` objects (those
    other than ``contract_dim``), in their logical order, so the caller can
    re-emit them as output dims.
    """
    K = contract_dim.logical_size
    free_dims = [d for d in st.dims if d.id != contract_dim.id]

    if st.val is None:
        free_sizes = tuple(d.logical_size for d in free_dims)
        arr = jnp.ones(free_sizes + (K,), dtype=st.dtype)
        return arr, free_dims

    val = st.val
    c_axis = contract_dim.axis
    if c_axis is None:
        # Contracted axis is implicit (val doesn't carry it) ⇒ broadcast it in.
        free_axes = [d.axis for d in free_dims if d.axis is not None]
        leftover = [a for a in range(val.ndim) if a not in free_axes]
        perm = free_axes + leftover
        v = val.transpose(perm) if perm != list(range(val.ndim)) else val
        # Drop leftover size-1 axes, then append a broadcast K axis.
        v = v.reshape(tuple(v.shape[: len(free_axes)]))
        v = jnp.broadcast_to(v[..., None], v.shape + (K,))
        return v, free_dims

    free_axes = [d.axis for d in free_dims if d.axis is not None]
    # Order the free axes by the logical dim order (free_dims order), putting the
    # contract axis last.
    ordered_free_axes = []
    for d in free_dims:
        if d.axis is not None:
            ordered_free_axes.append(d.axis)
    perm = ordered_free_axes + [c_axis]
    leftover = [a for a in range(val.ndim) if a not in perm]
    perm = perm + leftover
    v = val.transpose(perm) if perm != list(range(val.ndim)) else val
    # Collapse any leftover size-1 trailing axes into nothing.
    n_keep = len(ordered_free_axes) + 1
    if v.ndim > n_keep:
        v = v.reshape(v.shape[:n_keep])
    # Insert singleton axes for any implicit (axis is None) free dims so the
    # array carries one axis per free dim, contract axis last.
    out_shape = []
    src = iter(v.shape[:-1])
    for d in free_dims:
        out_shape.append(next(src) if d.axis is not None else 1)
    out_shape.append(K)
    v = v.reshape(tuple(out_shape))
    full = tuple(d.logical_size for d in free_dims) + (K,)
    if v.shape != full:
        v = jnp.broadcast_to(v, full)
    return v, free_dims


# --------------------------------------------------------------------------- #
# Topology resolution
# --------------------------------------------------------------------------- #
def _find_contract_pair(lhs: "SparseTensor", rhs: "SparseTensor"):
    """Return ``(lhs_contract_dim, rhs_contract_dim)`` — the single contracted
    pair (lhs primal-side vs rhs out-side) of equal logical size."""
    if not lhs.primal_dims or not rhs.out_dims:
        raise ValueError(
            "contract_dense_block_diagonal needs one primal dim on lhs and one "
            "out dim on rhs to contract."
        )
    lc = lhs.primal_dims[-1]
    rc = rhs.out_dims[-1]
    if lc.logical_size != rc.logical_size:
        raise ValueError(
            f"Contraction size mismatch: lhs primal {lc.logical_size} vs "
            f"rhs out {rc.logical_size}."
        )
    return lc, rc


def _free_partner(st: "SparseTensor", contract_dim: Index) -> Index:
    """For a DiagonalIndex contracted dim, return its coupled free partner dim
    (the ``other_id`` side, which survives the contraction)."""
    for d in st.dims:
        if d.id == contract_dim.other_id:
            return d
    raise ValueError(
        f"DiagonalIndex contracted dim {contract_dim.id} has no partner "
        f"other_id={contract_dim.other_id} in the tensor."
    )


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
def contract_dense_block_diagonal(
    lhs: "SparseTensor", rhs: "SparseTensor"
) -> "SparseTensor":
    """Contract ``lhs @ rhs`` where exactly ONE operand's contracted dim is a
    ``DiagonalIndex`` (block-diagonal) and the other's is a plain ``DenseIndex``.

    Implements ``D @ B`` and ``B @ D`` (see module docstring).  The result is a
    fully-dense ``SparseTensor`` (the closed form), computed by a single batched
    ``dot_general`` over the meta-diagonal — cost ~ ``nnz`` not ``N**2``.

    Requires zero-fill operands (the common vertex-elim case).  A non-zero fill
    on either operand falls back to the dense oracle (``lhs.dense() @
    rhs.dense()``) so correctness is never compromised.
    """
    from graphax.sparse.ops.utils import _is_zero_fill

    lc, rc = _find_contract_pair(lhs, rhs)

    lhs_is_diag = lc.is_sparse
    rhs_is_diag = rc.is_sparse
    if lhs_is_diag == rhs_is_diag:
        raise ValueError(
            "contract_dense_block_diagonal requires EXACTLY ONE of the "
            "contracted dims to be a DiagonalIndex (block-diagonal) and the "
            f"other Dense; got lhs_diag={lhs_is_diag}, rhs_diag={rhs_is_diag}. "
            "(D@D, B@B and compressed inputs route to other kernels.)"
        )

    # Non-zero fill: fall back to the dense oracle (correctness over speed).
    if not (_is_zero_fill(lhs) and _is_zero_fill(rhs)):
        out = lhs.dense() @ rhs.dense()
        from graphax.sparse.ops.utils import _arr2st

        return _arr2st(out, len(lhs.out_dims), out.ndim - len(lhs.out_dims))

    out_dtype = _compute_dtype(lhs.dtype, rhs.dtype)
    scalar = (lhs.scalar_mult * rhs.scalar_mult).astype(out_dtype)

    if rhs_is_diag:
        # ---- D @ B -----------------------------------------------------------
        # lhs dense, contracted axis K = lc.logical_size.
        # rhs block-diagonal: contracted out-side rc (meta N, block B_c),
        # free primal-side rf (meta N, block B_f).
        rf = _free_partner(rhs, rc)
        N = rc.size
        B_c = rc.block_size or 1
        B_f = rf.block_size or 1

        lhs_arr, lhs_free = _dense_with_contract_last(lhs, lc)  # (*L, K)
        blocks = _diag_blocks(rhs, rc, rf)  # (N, B_c, B_f)

        # Reshape lhs contracted axis K -> (N, B_c).
        L_shape = lhs_arr.shape[:-1]
        lhs_nbc = lhs_arr.reshape(L_shape + (N, B_c))  # (*L, N, B_c)

        # einsum '...gi,gij->...gj' : batch over g=N, contract i=B_c.
        out = jnp.einsum("...gi,gij->...gj", lhs_nbc, blocks)  # (*L, N, B_f)
        out = out.reshape(L_shape + (N * B_f,))  # (*L, F)
        out = _scaled_mul(out.astype(out_dtype), scalar)

        # Output dims: lhs free dims keep their sides; the diagonal's free dim
        # ``F`` joins the primal side (replacing the consumed contracted primal).
        lhs_out_free = [d for d in lhs.out_dims if d.id != lc.id]
        lhs_primal_free = [d for d in lhs.primal_dims if d.id != lc.id]
        F = N * B_f
        out_dims, primal_dims = _assemble_dims(lhs_out_free, lhs_primal_free, F)
        return _emit(out, out_dims, primal_dims, dtype=out_dtype)

    else:
        # ---- B @ D -----------------------------------------------------------
        # lhs block-diagonal: free out-side lf (meta N, block B_f), contracted
        # primal-side lc (meta N, block B_c). rhs dense, contracted out-axis K.
        lf = _free_partner(lhs, lc)
        N = lc.size
        B_c = lc.block_size or 1
        B_f = lf.block_size or 1

        rhs_arr, rhs_free = _dense_with_contract_last(rhs, rc)  # (*R, K)
        blocks = _diag_blocks(lhs, lc, lf)  # (N, B_c, B_f)  (contract, free)

        R_shape = rhs_arr.shape[:-1]
        rhs_nbc = rhs_arr.reshape(R_shape + (N, B_c))  # (*R, N, B_c)

        # out[*R, g, j] = sum_i rhs[*R, g, i] * block[g, i, j]
        # einsum 'gij,...gi->...gj'
        out = jnp.einsum("gij,...gi->...gj", blocks, rhs_nbc)  # (*R, N, B_f)
        out = out.reshape(R_shape + (N * B_f,))  # (*R, F)
        # The block-diagonal's free out-side becomes the leading OUT dim; the
        # rhs free dims are primal (their original side preserved below).
        F = N * B_f

        rhs_out_free = [d for d in rhs.out_dims if d.id != rc.id]
        rhs_primal_free = [d for d in rhs.primal_dims if d.id != rc.id]

        # Current layout: (*R_free, F). R_free is rhs's free dims in logical
        # order; move F to the FRONT (it is the surviving out dim of lhs).
        out = jnp.moveaxis(out, -1, 0)  # (F, *R_free)
        out = _scaled_mul(out.astype(out_dtype), scalar)

        # out dims: the lhs free out dim (F) + rhs out-free dims; primal: rhs
        # primal-free dims.
        out_dims, primal_dims = _assemble_dims_bd(F, rhs_out_free, rhs_primal_free)
        return _emit(out, out_dims, primal_dims, dtype=out_dtype)


# --------------------------------------------------------------------------- #
# Output dim assembly
# --------------------------------------------------------------------------- #
def _assemble_dims(lhs_out_free, lhs_primal_free, F):
    """D@B output: lhs free dims keep their sides; the diagonal's free dim ``F``
    joins the primal side (the contracted lhs primal dim was consumed, F takes
    its place as the new contracted-replacement free axis on the primal side).

    Physical axis order of the produced array is ``(*lhs_out_free,
    *lhs_primal_free, F)``.
    """
    next_id = 0
    axis = 0
    out_dims = []
    primal_dims = []
    for d in lhs_out_free:
        out_dims.append(DenseIndex(next_id, d.logical_size, axis))
        next_id += 1
        axis += 1
    for d in lhs_primal_free:
        primal_dims.append(DenseIndex(next_id, d.logical_size, axis))
        next_id += 1
        axis += 1
    # The surviving diagonal free dim lands on the primal side, last axis.
    primal_dims.append(DenseIndex(next_id, F, axis))
    return out_dims, primal_dims


def _assemble_dims_bd(F, rhs_out_free, rhs_primal_free):
    """B@D output: physical axis order ``(F, *rhs_out_free, *rhs_primal_free)``.
    ``F`` (lhs free out-side) leads as an out dim, followed by rhs free dims on
    their original sides.
    """
    out_dims = []
    primal_dims = []
    next_id = 0
    axis = 0
    out_dims.append(DenseIndex(next_id, F, axis))
    next_id += 1
    axis += 1
    for d in rhs_out_free:
        out_dims.append(DenseIndex(next_id, d.logical_size, axis))
        next_id += 1
        axis += 1
    for d in rhs_primal_free:
        primal_dims.append(DenseIndex(next_id, d.logical_size, axis))
        next_id += 1
        axis += 1
    return out_dims, primal_dims


def _emit(values, out_dims, primal_dims, dtype):
    """Wrap the contracted array into a fully-dense SparseTensor.

    ``scalar_mult`` is already folded into ``values`` upstream, so it is set to
    1.  Primal dims are re-id'd to follow the out dims for id contiguity.
    """
    from graphax.sparse.tensor import SparseTensor

    n_out = len(out_dims)
    out_dims = [DenseIndex(i, d.logical_size, d.axis) for i, d in enumerate(out_dims)]
    primal_dims = [
        DenseIndex(n_out + i, d.logical_size, d.axis)
        for i, d in enumerate(primal_dims)
    ]
    return SparseTensor(
        tuple(out_dims),
        tuple(primal_dims),
        values.astype(dtype),
        scalar_mult=jnp.array(1, dtype=dtype),
        fill_value=None,
        check_consistency=False,
    )


# =========================================================================== #
# Dense <-> MULTIPLE-block-diagonal-pair contraction  (the generalization)
# =========================================================================== #
r"""
``contract_dense_multi_block_diagonal`` extends the single-pair kernel above to
the case where ONE operand is fully Dense (``D`` on every dim) and the OTHER
carries N >= 1 meta-block-diagonal (``DiagonalIndex`` / ``B``) *pairs*
simultaneously, coexisting with plain Dense dims, with SEVERAL dims contracted at
once of mixed type.  This is the structured contraction that an attention /
ViT jacve produces (e.g. ``lhs.dims=[D,D,D,D,D]`` @
``rhs.dims=[B,B,D,B,B,D]`` — two coupled DiagonalIndex pairs plus dense dims,
several dims contracted together), and which the single-pair kernel declines and
the dispatcher previously routed to the densifying ``composed_dense`` fallback.

------------------------------------------------------------------------------
MATH (general closed form)
------------------------------------------------------------------------------
Let the DENSE operand be ``Dn`` and the structured operand ``Sr``.  Vertex
elimination contracts ``Dn``'s contracted side against ``Sr``'s contracted side
over a set of aligned dims.  Each contracted dim of ``Sr`` is one of:

  * a DIAGONAL contracted dim of a pair ``p`` (meta ``N_p``, contracted block
    ``Bc_p``) whose partner — meta ``N_p``, free block ``Bf_p`` — survives on
    ``Sr``'s OTHER side as a free output dim, OR
  * a plain DENSE contracted dim (an ordinary contraction index ``K_c``).

Each *free* (non-contracted) dim of ``Sr`` is either a free dense dim, or the
free side of a diagonal pair whose contracted side IS contracted (rides through
as a dense output axis of size ``N_p*Bf_p``).  ``Sr``'s ``val`` stores, per meta
group ``g_p`` of each pair, the block ``W_p[g_p, i_p, j_p]`` (contract index
``i_p`` in ``[0,Bc_p)``, free index ``j_p`` in ``[0,Bf_p)``) together with the
dense axes — non-zero ONLY on the shared meta-diagonal ``g_p`` of each pair.

Because each pair is block-diagonal, a contracted position only couples LHS
positions in the SAME meta group ``g_p``.  Writing the dense operand's contracted
axis for pair ``p`` as ``(g_p, i_p)`` and a free output position as ``(g_p, j_p)``
(SAME ``g_p`` — that is the diagonal restriction), the contraction is the single
batched einsum (batch over every meta ``g_p``, contract every block axis ``i_p``
and every dense-contracted axis ``k_c``):

    out[*O, {g_p, j_p}_p, *Fd]
        = sum_{ {i_p}_p, {k_c}_c }
              Dn[*O, {g_p, i_p}_p, {k_c}_c]
            * Sr_val[ {g_p, i_p, j_p}_p, *Fd, {k_c}_c ]

NO full ``prod_p N_p`` (the ``N^2`` of the single-pair case, to the N-th power)
dense intermediate is ever built: the meta axes ``g_p`` stay BATCH axes of one
``dot_general`` and the contracted block widths ``Bc_p`` are the only contracted
extent per pair.  Cost ``~ nnz(Sr) * (Dn free)`` — the product of the per-pair
block-diagonal sparsity factors, exactly the savings.

The result is fully Dense in ``{D}`` (each free output position is computed
independently; the surviving partner free axes ``N_p*Bf_p`` are plain dense
axes, like the single-pair ``D@B -> D`` closure).

------------------------------------------------------------------------------
OUTPUT-ID CONVENTION
------------------------------------------------------------------------------
Mirrors the composed-dense / matmul ``finalize`` convention so downstream
multi-edge contractions align: physical axis order is
``(*Dn_free_out, *Dn_free_primal_other_than_contracted, *Sr_free)`` with ids
renumbered ``0..n_out-1`` for out dims and ``n_out..`` for primal dims.  When the
DENSE operand is the LHS (the ``D @ multi-B`` orientation, the attention case),
that is ``out = Dn.out_free`` and ``primal = Sr_free`` (the diagonal partners +
dense free dims, in ``Sr``'s primal logical order) — byte-for-byte the layout the
``composed_dense`` densify fallback produced (verified against it).
"""


def _structured_operand(lhs, rhs):
    """Identify which operand is fully Dense and which carries block-diagonal
    structure.  Returns ``(dense_st, struct_st, dense_is_lhs)`` or ``None`` when
    the orientation isn't "one fully-dense, one structured" (the only shape this
    kernel owns)."""
    l_struct = any(_is_block_diag(d) for d in lhs.dims)
    r_struct = any(_is_block_diag(d) for d in rhs.dims)
    if l_struct and not r_struct:
        return rhs, lhs, False  # dense=rhs, struct=lhs, dense_is_lhs=False
    if r_struct and not l_struct:
        return lhs, rhs, True
    return None


def _is_block_diag(d) -> bool:
    return d.is_sparse and not getattr(d, "is_compressed", False)


def _diag_pairs(st: "SparseTensor"):
    """All meta-block-diagonal pairs of ``st`` as ``{frozenset(ids): (a, b)}``
    keyed by the id-pair, ``a`` the lower-id side.  A pair links ``a.other_id ==
    b.id``."""
    by_id = {d.id: d for d in st.dims}
    seen = set()
    pairs = {}
    for d in st.dims:
        if not _is_block_diag(d) or d.id in seen:
            continue
        partner = by_id.get(d.other_id)
        if partner is None:
            continue
        a, b = (d, partner) if d.id < partner.id else (partner, d)
        pairs[frozenset((a.id, b.id))] = (a, b)
        seen.add(a.id)
        seen.add(b.id)
    return pairs


def _dense_full(st: "SparseTensor") -> jnp.ndarray:
    """Materialize a fully-Dense operand to its logical ``(out..., primal...)``
    array with scalar_mult / fill folded in (``st.dense()`` does exactly this)."""
    return st.dense()


def contract_dense_multi_block_diagonal(
    lhs: "SparseTensor", rhs: "SparseTensor"
) -> "SparseTensor | None":
    """Contract ``lhs @ rhs`` where ONE operand is fully Dense and the OTHER
    carries ``N >= 1`` meta-block-diagonal pairs (plus dense dims), with several
    mixed-type dims contracted at once (see the module-level docstring above).

    Stays nnz-sparse via a SINGLE batched ``einsum`` over all meta axes (no
    gather/scatter, no python loop, no full ``prod_p N_p`` dense intermediate).
    Returns ``None`` when the contraction is outside this kernel's scope (the
    dispatcher then keeps the ``composed_dense`` densify fallback), so it is
    always safe to attempt.
    """
    from graphax.sparse.ops.utils import _is_zero_fill

    if not (_is_zero_fill(lhs) and _is_zero_fill(rhs)):
        return None

    pick = _structured_operand(lhs, rhs)
    if pick is None:
        return None
    dense_st, struct_st, dense_is_lhs = pick

    # ----- resolve the contracted pairs (lhs primal vs rhs out), aligned BY the
    # same topology the matmul uses (id-based), so we agree on what contracts.
    from graphax.sparse.ops.matmul import _align_contract_dims

    try:
        cpairs = _align_contract_dims(lhs.primal_dims, rhs.out_dims, embed=True)
    except Exception:
        return None
    if not cpairs:
        return None

    # Sets of contracted dim ids on each side.
    lhs_contract_ids = {ld.id for ld, _ in cpairs}
    rhs_contract_ids = {rd.id for _, rd in cpairs}

    if dense_is_lhs:
        dense_contract_ids = lhs_contract_ids
        struct_contract_ids = rhs_contract_ids
    else:
        dense_contract_ids = rhs_contract_ids
        struct_contract_ids = lhs_contract_ids

    # The map from a struct contracted dim id -> the dense dim id it pairs with,
    # so we can line up the dense operand's contracted axes with struct's.
    if dense_is_lhs:
        # lhs (dense) primal contracts rhs (struct) out
        contract_map = {rd.id: ld.id for ld, rd in cpairs}
    else:
        # lhs (struct) primal contracts rhs (dense) out
        contract_map = {ld.id: rd.id for ld, rd in cpairs}

    diag_pairs = _diag_pairs(struct_st)

    # Classify each diagonal pair: is its contracted side actually contracted?
    # We need EXACTLY one side of each pair to be on the struct contracted set
    # for a "contracted pair", or NEITHER side contracted for a "free pair".
    # A pair with BOTH sides contracted, or a rectangular/aligned mismatch we
    # can't express, falls out of scope -> return None.
    contracted_pairs = []  # (contract_dim, free_dim)
    free_pairs = []        # (a, b) both surviving (ride-through block structure)
    for key, (a, b) in diag_pairs.items():
        a_c = a.id in struct_contract_ids
        b_c = b.id in struct_contract_ids
        if a_c and b_c:
            return None  # both sides contracted (B@B-like): out of scope here
        if a_c:
            contracted_pairs.append((a, b))
        elif b_c:
            contracted_pairs.append((b, a))
        else:
            free_pairs.append((a, b))

    # Free ride-through diagonal pairs are not yet supported by this kernel's
    # output assembly (they would survive as a DiagonalIndex pair in the output).
    # Keep them for the fallback rather than silently densifying wrong.
    if free_pairs:
        return None

    # ----- gather dim partitions -------------------------------------------- #
    # struct side: contracted diag dims, their free partners (survive), dense
    # contracted dims, dense free dims.
    struct_by_id = {d.id: d for d in struct_st.dims}
    contracted_diag = [c for (c, f) in contracted_pairs]
    surviving_partner = [f for (c, f) in contracted_pairs]
    surviving_partner_ids = {f.id for f in surviving_partner}
    contracted_diag_ids = {c.id for c in contracted_diag}

    # dense contracted dims of struct (contracted, not part of a diagonal pair)
    struct_dense_contracted = [
        struct_by_id[i] for i in struct_contract_ids
        if i not in contracted_diag_ids and not _is_block_diag(struct_by_id[i])
    ]
    # Any *implicit-style* contracted struct dim with axis=None but logical>1 is
    # also a plain contraction index — handled by densifying that axis below.

    # struct free dims = everything not contracted and not a surviving partner
    # handled separately; surviving partners DO survive (ride through).
    struct_free = [
        d for d in struct_st.dims
        if d.id not in struct_contract_ids and d.id not in surviving_partner_ids
        and not _is_block_diag(d)
    ]
    # surviving partners are also "free output" but carry block structure size
    # N*Bf which we fold into a dense axis.

    # ----- build the dense operand array, contracted axes last -------------- #
    # Order of contracted axes on the dense side MUST match the order we feed the
    # struct factors.  We choose: [diag pairs in contracted_pairs order],
    # [struct dense contracted in struct_contract order].
    dense_arr = _dense_full(dense_st)
    if dense_is_lhs:
        dense_out_free = list(dense_st.out_dims)
        dense_contract_side = list(dense_st.primal_dims)
        dense_other_free = []  # lhs: out dims are free out, no extra primal free
    else:
        dense_out_free = []
        dense_contract_side = list(dense_st.out_dims)
        dense_other_free = list(dense_st.primal_dims)

    # Map each dense contracted dim id -> its struct counterpart id.
    if dense_is_lhs:
        dense_to_struct = {ld.id: rd.id for ld, rd in cpairs}
    else:
        dense_to_struct = {rd.id: ld.id for ld, rd in cpairs}
    struct_to_dense = {v: k for k, v in dense_to_struct.items()}

    return _multiB_einsum(
        dense_arr=dense_arr,
        dense_st=dense_st,
        struct_st=struct_st,
        dense_is_lhs=dense_is_lhs,
        dense_out_free=dense_out_free,
        dense_contract_side=dense_contract_side,
        dense_other_free=dense_other_free,
        contracted_pairs=contracted_pairs,
        struct_dense_contracted=struct_dense_contracted,
        struct_free=struct_free,
        struct_to_dense=struct_to_dense,
        lhs=lhs,
        rhs=rhs,
    )


def _multiB_einsum(
    *,
    dense_arr,
    dense_st,
    struct_st,
    dense_is_lhs,
    dense_out_free,
    dense_contract_side,
    dense_other_free,
    contracted_pairs,
    struct_dense_contracted,
    struct_free,
    struct_to_dense,
    lhs,
    rhs,
):
    """Compose the dense operand and struct's packed block factors into ONE
    batched einsum.  The struct ``val`` is consumed in its PACKED form (per-meta
    blocks, never the dense ``prod N_p`` matrix); the dense operand's contracted
    axes are reshaped into ``(meta, block)`` per diagonal pair so meta becomes a
    shared batch axis."""
    out_dtype = _compute_dtype(lhs.dtype, rhs.dtype)
    # The DENSE operand's scalar_mult is ALREADY folded by ``dense_st.dense()``
    # into ``dense_arr``; only the STRUCT operand's scalar_mult (it is read from
    # the raw ``val`` buffer below) still needs folding — applying both here
    # would double-count the dense side's scale.
    scalar = struct_st.scalar_mult.astype(out_dtype)

    # --- letter pools for einsum subscripts (distinct symbols) --------------- #
    pool = iter(string.ascii_letters)

    def fresh():
        return next(pool)

    # Per-pair symbols: g (meta), i (contract block), j (free block).
    pair_syms = []  # list of (g, i, j, N, Bc, Bf, contract_dim, free_dim)
    for (c, f) in contracted_pairs:
        N = c.size
        Bc = c.block_size or 1
        Bf = f.block_size or 1
        if f.size != N:
            return None  # partner meta mismatch — out of scope
        pair_syms.append((fresh(), fresh(), fresh(), N, Bc, Bf, c, f))

    # Dense-contracted (struct dense contracted) symbols.
    dctr_syms = []  # (sym, struct_dim, K)
    for sd in struct_dense_contracted:
        dctr_syms.append((fresh(), sd, int(sd.logical_size)))

    # struct free (dense) dim symbols (ride through to output).
    sfree_syms = []  # (sym, dim, size)
    for d in struct_free:
        sfree_syms.append((fresh(), d, int(d.logical_size)))

    # dense operand free symbols (out side + other free side).
    dout_syms = [(fresh(), d, int(d.logical_size)) for d in dense_out_free]
    dother_syms = [(fresh(), d, int(d.logical_size)) for d in dense_other_free]

    # --- build the struct block tensor (val packed) ------------------------- #
    struct_factor, struct_sub = _pack_struct_factor(
        struct_st, pair_syms, dctr_syms, sfree_syms, out_dtype
    )
    if struct_factor is None:
        return None

    # --- build the dense operand factor + its subscript --------------------- #
    dense_factor, dense_sub = _pack_dense_factor(
        dense_arr, dense_st, dense_is_lhs,
        dout_syms, dother_syms, pair_syms, dctr_syms,
        struct_to_dense, out_dtype,
    )
    if dense_factor is None:
        return None

    # --- output subscript --------------------------------------------------- #
    # Output physical order:
    #   dense LHS:  (*dout, [g_p j_p per pair], *sfree)   -> out=dout, primal=rest
    #   dense RHS:  ([g_p j_p per pair], *sfree, *dother) -> out=struct-survive, primal=dother
    out_pair_part = "".join(g + j for (g, i, j, *_rest) in pair_syms)
    sfree_part = "".join(s for (s, _d, _sz) in sfree_syms)
    dout_part = "".join(s for (s, _d, _sz) in dout_syms)
    dother_part = "".join(s for (s, _d, _sz) in dother_syms)

    if dense_is_lhs:
        out_sub = dout_part + out_pair_part + sfree_part
    else:
        out_sub = out_pair_part + sfree_part + dother_part

    eq = f"{dense_sub},{struct_sub}->{out_sub}"
    out = jnp.einsum(eq, dense_factor, struct_factor)
    out = _scaled_mul(out.astype(out_dtype), scalar)

    # --- collapse each (g_p, j_p) meta+block into a single dense axis -------- #
    # einsum produced them as adjacent axes in out_pair_part order. Reshape.
    out, out_dims, primal_dims = _assemble_multiB_dims(
        out, dense_is_lhs, dout_syms, pair_syms, sfree_syms, dother_syms,
    )
    return _emit(out, out_dims, primal_dims, dtype=out_dtype)


def _pack_struct_factor(struct_st, pair_syms, dctr_syms, sfree_syms, out_dtype):
    """Canonicalize the single ``struct_st.val`` buffer into the packed axis
    order ``([g_p i_p j_p]*, [k_c]*, [free]*)`` using each dim's ``axis`` /
    ``block_axis`` for indirection (the val physical layout is
    implementation-defined).  Returns ``(array, subscript)`` or ``(None, None)``
    when an axis can't be located (out of scope).

    The struct ``val`` is a SINGLE buffer coupling all pairs + dense axes (it is
    NOT per-pair concatenable), so we read it directly and place every axis —
    never materializing the full ``prod_p N_p`` dense matrix."""
    """Canonicalize the single ``struct_st.val`` buffer into the packed axis
    order ``([g_p i_p j_p]*, [k_c]*, [free]*)`` using each dim's ``axis`` /
    ``block_axis`` for indirection (the val physical layout is
    implementation-defined).  Returns ``(array, subscript)`` or ``(None, None)``
    when an axis can't be located (out of scope)."""
    val = struct_st.val

    # Build the ordered list of (physical_axis_or_None, size, sym) we want, in
    # the packed order.
    want = []  # (axis, size, sym)
    for (g, i, j, N, Bc, Bf, c, f) in pair_syms:
        meta_axis = c.axis if c.axis is not None else f.axis
        want.append((meta_axis, N, g))
        want.append((c.block_axis, Bc, i))
        want.append((f.block_axis, Bf, j))
    for (sym, sd, K) in dctr_syms:
        want.append((sd.axis, K, sym))
    for (sym, d, sz) in sfree_syms:
        want.append((d.axis, sz, sym))

    if val is None:
        shape = tuple(sz for (_ax, sz, _s) in want)
        arr = jnp.ones(shape, dtype=out_dtype)
        sub = "".join(s for (_ax, _sz, s) in want)
        return arr, sub

    # Locate each wanted axis in val; absent (None) axes are broadcast singletons.
    present = [ax for (ax, _sz, _s) in want if ax is not None]
    # Detect duplicate physical axes (a shared meta axis used by >1 want slot is
    # fine ONLY if it is the SAME meta for the pair's three slots; here each pair
    # has distinct block axes, the meta axis is unique per pair). If a physical
    # axis is requested by two DIFFERENT logical slots we can't express it.
    if len(present) != len(set(present)):
        return None, None
    leftover = [a for a in range(val.ndim) if a not in present]
    perm = present + leftover
    v = jnp.transpose(val, perm) if perm != list(range(val.ndim)) else val
    # Now present axes are at the front in `want` order (skipping None slots).
    # Re-expand to full packed shape, inserting singleton axes for None slots and
    # collapsing leftover trailing axes (should be size-1 for a clean buffer).
    n_present = len(present)
    if v.ndim > n_present:
        # collapse leftover (size-1) into nothing
        lead = v.shape[:n_present]
        rest = v.shape[n_present:]
        import math as _m
        if _m.prod(rest) != 1:
            return None, None
        v = v.reshape(lead)
    # Build target packed shape, inserting singletons for absent axes.
    final_shape = []
    src = iter(range(n_present))
    sub = []
    for (ax, sz, s) in want:
        sub.append(s)
        if ax is not None:
            final_shape.append(v.shape[next(src)])
        else:
            final_shape.append(1)
    v = v.reshape(tuple(final_shape))
    full = tuple(sz for (_ax, sz, _s) in want)
    if v.shape != full:
        v = jnp.broadcast_to(v, full)
    return v.astype(out_dtype), "".join(sub)


def _pack_dense_factor(
    dense_arr, dense_st, dense_is_lhs,
    dout_syms, dother_syms, pair_syms, dctr_syms,
    struct_to_dense, out_dtype,
):
    """Reshape the dense operand array into the packed axis order
    ``(*dfree_out, [g_p i_p]* , [k_c]*, *dfree_other)`` and return
    ``(array, subscript)``.  ``dense_arr`` is ``dense_st.dense()`` laid out as
    ``(out_logical..., primal_logical...)``."""
    # Identify which dense-side dims are contracted vs free.
    # Map: struct contracted dim id -> dense dim id.
    if dense_is_lhs:
        out_dims = list(dense_st.out_dims)
        contract_pool = list(dense_st.primal_dims)
        other_free = []  # none; out dims already the only free
    else:
        out_dims = []
        contract_pool = list(dense_st.out_dims)
        other_free = list(dense_st.primal_dims)

    # The dense_arr axis order is logical (out..., primal...). Build a per-dim
    # logical-axis index.
    logical_dims = list(dense_st.out_dims) + list(dense_st.primal_dims)
    axis_of = {d.id: k for k, d in enumerate(logical_dims)}

    # We must place dense's contracted axes in the SAME order the struct factor
    # advertises them: first the diagonal pairs (in pair_syms order, splitting
    # each into (g_p, i_p)), then the dense-contracted dims (dctr_syms order).
    # Find the dense dim id matching each struct contracted dim.
    dense_id_for_struct = struct_to_dense  # struct_id -> dense_id

    # Subscript per logical axis of dense_arr, plus reshape splitting contracted
    # diagonal axes into (meta, block).
    # Strategy: transpose dense_arr to [free_out..., contracted(in pair+dctr order)..., free_other...]
    # then reshape each diagonal-contracted axis K -> (N, Bc).
    free_out_axes = [axis_of[d.id] for d in (out_dims if dense_is_lhs else [])]
    free_other_axes = [axis_of[d.id] for d in other_free]

    # contracted axes in the required order
    contracted_axes = []
    contracted_meta = []  # (is_diag, N, Bc) parallel to contracted_axes
    for (g, i, j, N, Bc, Bf, c, f) in pair_syms:
        sid = c.id  # struct contracted dim id
        did = dense_id_for_struct.get(sid)
        if did is None or did not in axis_of:
            return None, None
        contracted_axes.append(axis_of[did])
        contracted_meta.append((True, N, Bc, g, i))
    for (sym, sd, K) in dctr_syms:
        did = dense_id_for_struct.get(sd.id)
        if did is None or did not in axis_of:
            return None, None
        contracted_axes.append(axis_of[did])
        contracted_meta.append((False, K, None, sym, None))

    perm = free_out_axes + contracted_axes + free_other_axes
    if sorted(perm) != list(range(dense_arr.ndim)):
        return None, None
    arr = jnp.transpose(dense_arr, perm)

    # Now reshape: free_out kept, each diagonal-contracted axis K -> (N, Bc),
    # dense-contracted kept, free_other kept.
    new_shape = []
    sub = []
    for (s, _d, sz) in dout_syms:
        new_shape.append(sz)
        sub.append(s)
    for meta in contracted_meta:
        is_diag = meta[0]
        if is_diag:
            _isd, N, Bc, g, i = meta
            new_shape.extend([N, Bc])
            sub.append(g)
            sub.append(i)
        else:
            _isd, K, _none, sym, _n = meta
            new_shape.append(K)
            sub.append(sym)
    for (s, _d, sz) in dother_syms:
        new_shape.append(sz)
        sub.append(s)

    arr = arr.reshape(tuple(new_shape))
    return arr.astype(out_dtype), "".join(sub)


def _assemble_multiB_dims(out, dense_is_lhs, dout_syms, pair_syms, sfree_syms, dother_syms):
    """Collapse each adjacent ``(g_p, j_p)`` meta+free-block axis pair (size
    ``N_p*Bf_p``) into one dense output axis, and split out vs primal dims per the
    output-id convention."""
    # The einsum out_sub put axes in order:
    #   dense LHS: dout..., [g_p j_p]..., sfree...
    #   dense RHS: [g_p j_p]..., sfree..., dother...
    # Build the new (collapsed) shape and the dim split.
    out_sizes = []
    # leading dout (LHS only)
    n_dout = len(dout_syms) if dense_is_lhs else 0
    # walk and collapse pairs
    shape = list(out.shape)
    new_shape = []
    idx = 0
    # dout part
    for _ in range(n_dout):
        new_shape.append(shape[idx]); idx += 1
    pair_sizes = []
    for (g, i, j, N, Bc, Bf, c, f) in pair_syms:
        # two adjacent axes g (N) and j (Bf)
        new_shape.append(shape[idx] * shape[idx + 1])
        pair_sizes.append(shape[idx] * shape[idx + 1])
        idx += 2
    sfree_sizes = []
    for (s, _d, sz) in sfree_syms:
        new_shape.append(shape[idx]); sfree_sizes.append(shape[idx]); idx += 1
    dother_sizes = []
    if not dense_is_lhs:
        for (s, _d, sz) in dother_syms:
            new_shape.append(shape[idx]); dother_sizes.append(shape[idx]); idx += 1
    out = out.reshape(tuple(new_shape))

    # Now split into out_dims / primal_dims per convention.
    out_dims = []
    primal_dims = []
    ax = 0
    if dense_is_lhs:
        for (s, d, sz) in dout_syms:
            out_dims.append(DenseIndex(0, int(d.logical_size), ax)); ax += 1
        # surviving partners + dense free become primal dims
        for ps in pair_sizes:
            primal_dims.append(DenseIndex(0, int(ps), ax)); ax += 1
        for sz in sfree_sizes:
            primal_dims.append(DenseIndex(0, int(sz), ax)); ax += 1
    else:
        # dense RHS: struct survivors are the OUT dims, dother are primal.
        for ps in pair_sizes:
            out_dims.append(DenseIndex(0, int(ps), ax)); ax += 1
        for sz in sfree_sizes:
            out_dims.append(DenseIndex(0, int(sz), ax)); ax += 1
        for sz in dother_sizes:
            primal_dims.append(DenseIndex(0, int(sz), ax)); ax += 1
    return out, out_dims, primal_dims
