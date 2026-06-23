r"""Elemental elementwise op: Dense <-> BlockDiagonal  (``D (+|*) B`` and friends).

This module owns the vertex-elimination MERGE (``+``) and the ``*``-style
elementwise combinations between a plain Dense (``D``) operand and a
meta-block-diagonal (``DiagonalIndex`` / ``B``) operand, plus the ``B (+|*) B``
matched-structure case.  Unlike a *contraction* (``contract_D_B`` / ``@``), an
elementwise op pairs the two operands cell-for-cell on the SAME logical shape;
no axis is summed away.

================================================================================
1. MATH
================================================================================
Both operands share the logical shape ``(R, C)`` (``R`` rows = out side, ``C``
cols = primal side; trailing leftover/batch dense axes ride along untouched).
A ``DiagonalIndex`` pair with meta-count ``N``, out-side block ``B_o`` and
primal-side block ``B_i`` has ``R = N*B_o``, ``C = N*B_i`` and is non-zero ONLY
on the META-BLOCK DIAGONAL:

    B[g*B_o + r, h*B_i + c] = block[g, r, c]      if g == h
                            = fill_B               otherwise            (*)

where ``fill_B`` is the operand's (post-scaled) ``fill_value`` — ``0`` for the
canonical zero-fill Jacobian.

Let ``op`` be the binary op (``add`` / ``multiply`` / ...).  The elementwise
result is, cell-by-cell, ``op(X[i,j], Y[i,j])`` with each operand read through
its structure (a B operand reads (*); a D operand reads its dense ``val``).

----  D op B  /  B op D  -------------------------------------------------------
Split the grid into the block-diagonal SUPPORT ``S = {(i,j): block(i)==block(j)}``
and its complement ``S^c``:

  * On ``S`` (the ``N`` diagonal ``(B_o x B_i)`` blocks):
        out_block[g, r, c] = op( D_block[g, r, c], block[g, r, c] )
    where ``D_block[g, r, c] = D[g*B_o+r, g*B_i+c]`` is D's diagonal sub-block.
  * On ``S^c`` (off the meta-diagonal):
        out[i, j] = op( D[i, j], fill_B ).

So:

  * ``add`` (and any op with ``op(x, 0) == x``, the UNION ops):  on ``S^c`` the
    result is ``D[i,j] + 0 = D[i,j]``, i.e. D passes through untouched and the
    blocks get ``D_block + block`` added on the diagonal.  The output support is
    the UNION of the two supports = D's full grid ⇒ the result is **Dense**.
    ``D + B -> D``,  ``B + D -> D``.

  * ``multiply`` (and any op with ``op(x, 0) == 0`` and ``fill_B == 0``, the
    INTERSECTION ops):  on ``S^c`` the result is ``D[i,j] * 0 = 0``, so the
    output is non-zero only on ``S`` — the SAME meta-block-diagonal support as
    B.  The output support is the INTERSECTION ⇒ the result stays **Block-
    diagonal** with the same ``(N, B_o, B_i)`` grid.  Its blocks are
    ``D_block * block`` and its fill is ``0``.
    ``D * B -> B``,  ``B * D -> B``.

(A non-zero ``fill_B`` breaks the support narrowing for ``*`` — then ``D*B`` is
dense again; this kernel falls back to the dense oracle there, see ALGORITHM.)

----  B op B  (matched block structure) ----------------------------------------
Two B operands with the SAME meta grid (``N`` and matching ``B_o`` / ``B_i``)
share the same support ``S``.  Then for ANY op:

    out_block[g, r, c] = op( blockA[g, r, c], blockB[g, r, c] )           on S
    out[i, j]          = op( fill_A, fill_B )                             on S^c

The support is unchanged, so the result is **Block-diagonal** with the same grid
and fill ``op(fill_A, fill_B)``.  For zero fills and a union/intersection op the
fill stays ``0`` and the result is a pure block-diagonal whose blocks are the
elementwise op of the two block buffers.   ``B op B -> B``.

----  WHY THE RESULT STAYS IN {D, B}  ------------------------------------------
Every case above produces output whose non-zero support is one of: D's full grid
(union with a B) ⇒ Dense; or B's meta-block-diagonal (intersection / B-op-B) ⇒
Block-diagonal with the SAME ``(N, B_o, B_i)`` grid.  No new off-diagonal
structure is ever created, so {D, B} is closed under these ops.

================================================================================
2. ALGORITHM (cost ~ nnz, fusion-friendly)
================================================================================
The one structural primitive is *extracting / scattering D's diagonal
sub-blocks*, done with reshape + ``jnp.diagonal`` (a pure stride view; NO gather/
scatter/loop) and, for the union (Dense) result, a single broadcast ``where`` /
add over an eye mask.

  D op B  (union, -> Dense):
    1. Reshape D's grid ``(N*B_o, N*B_i)`` -> ``(N, B_o, N, B_i)``.
    2. ``diag = jnp.diagonal(D4, axis1=0, axis2=2)`` -> ``(B_o, B_i, N)`` ->
       move N to front -> ``(N, B_o, B_i)`` = D's diagonal sub-blocks.
    3. ``new_blocks = op(diag, blocks)``  (cost ~ ``N*B_o*B_i`` = nnz(B)).
    4. Scatter ``new_blocks`` back onto D's diagonal blocks (broadcast + ``where``
       over an ``N x N`` eye mask), leaving the off-diagonal D entries untouched
       (since ``op(D, 0) = D``).  Result is the full dense grid.

  D op B  (intersection, -> Block-diagonal):
    1-3 as above, then EMIT a ``DiagonalIndex`` pair carrying ``new_blocks`` —
    no scatter, no dense grid is ever formed.  Cost ~ nnz(B), output storage ~
    nnz(B).  This is the real sparsity win: ``N x`` less compute AND storage than
    densifying.

  B op B  (matched):  ``op(blocksA, blocksB)`` then emit the DiagonalIndex pair.
    Cost ~ nnz, pure block-buffer op.

Complexity: the diagonal extraction is ``O(R*C)`` to *read* D (intrinsic — D is
dense and must be read), but the op itself touches only the ``N*B_o*B_i`` block
cells.  The intersection / B-op-B paths NEVER materialize the ``N^2`` dense grid.
All ops are reshape / transpose / diagonal / broadcast / where / elementwise —
XLA fuses them; there is no ``lax.gather`` / ``lax.scatter`` / python block loop.

``scalar_mult`` is folded into each operand's values up front; a non-zero fill
on either operand routes to the dense-oracle fallback (correctness over speed).

================================================================================
3. SCOPE / DISPATCH (see INTEGRATION NOTE at bottom)
================================================================================
``elementwise_dense_block_diagonal(lhs, rhs, op, is_intersection=False)`` handles
the 2-D-per-operand core: each operand has exactly one out dim and one primal
dim; the pair is (D, B) / (B, D) / (B, B).  Leftover/batch dense axes on the D
operand (and matching ones on B) ride through as trailing axes.  Mismatched B/B
block grids and non-zero fills fall back to the dense oracle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp

from graphax.sparse.dtype_compute import _scaled_mul
from graphax.sparse.indexes import DenseIndex, DiagonalIndex, Index
from graphax.sparse.ops.utils import _compute_dtype, _is_zero_fill

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


# --------------------------------------------------------------------------- #
# Canonicalization helpers
# --------------------------------------------------------------------------- #
def _diag_grid(st: "SparseTensor", out_dim: Index, primal_dim: Index):
    """Extract a DiagonalIndex pair's per-meta blocks as ``(N, B_o, B_i, *L)``,
    with ``scalar_mult`` folded in.

    ``st.val`` carries (in some physical-axis order) a shared meta axis of size
    ``N``, the out-side block axis (size ``B_o``), the primal-side block axis
    (size ``B_i``) and any leftover/batch axes ``*L``.  Returns those reordered
    to the canonical ``(N, B_o, B_i, *L)``.  Absent (size-1) block axes are
    broadcast in.  ``val is None`` (uniform-ones structure) yields ones.
    """
    N = out_dim.size
    B_o = out_dim.block_size or 1
    B_i = primal_dim.block_size or 1

    if st.val is None:
        return jnp.ones((N, B_o, B_i), dtype=st.dtype)

    val = _scaled_mul(st.val, st.scalar_mult)
    meta_axis = out_dim.axis if out_dim.axis is not None else primal_dim.axis
    bo_axis = out_dim.block_axis
    bi_axis = primal_dim.block_axis

    present = [a for a in (meta_axis, bo_axis, bi_axis) if a is not None]
    leftover = [a for a in range(val.ndim) if a not in present]
    perm = present + leftover
    v = val.transpose(perm) if perm != list(range(val.ndim)) else val

    # v now leads with the present canonical axes; reshape inserts singleton
    # slots for any absent (size-1) block axis, then broadcast to the block grid.
    n_present = len(present)
    L = v.shape[n_present:]
    canon_present = (meta_axis is not None, bo_axis is not None, bi_axis is not None)
    expand = tuple(sz if p else 1 for p, sz in zip(canon_present, (N, B_o, B_i))) + L
    v = v.reshape(expand)
    return jnp.broadcast_to(v, (N, B_o, B_i) + L)


def _dense_grid(st: "SparseTensor", out_dim: Index, primal_dim: Index):
    """Materialize a dense operand to ``(R, C, *L)`` — out axis first, primal
    axis second, leftover/batch axes ``*L`` trailing — with ``scalar_mult`` and
    any ``fill_value`` folded in (via ``.dense()``).  Returns the array and the
    leftover-axis sizes ``L``."""
    R = out_dim.logical_size
    C = primal_dim.logical_size
    arr = st.dense()  # (R, C, *L), scalar_mult + fill folded
    L = arr.shape[2:]
    return arr, L


def _diag_subblocks(dense_grid, N, B_o, B_i):
    """Extract the ``N`` meta-diagonal ``(B_o, B_i)`` sub-blocks of a dense
    ``(R=N*B_o, C=N*B_i, *L)`` grid as ``(N, B_o, B_i, *L)``.

    Pure reshape + ``jnp.diagonal`` (a strided view — no gather): reshape to
    ``(N, B_o, N, B_i, *L)`` then take the diagonal over the two ``N`` meta axes.
    ``jnp.diagonal`` appends the diagonalized axis last, so move it back to front.
    """
    L = dense_grid.shape[2:]
    g4 = dense_grid.reshape((N, B_o, N, B_i) + L)
    # diagonal over the two meta axes (0 and 2) -> (B_o, B_i, *L, N)
    diag = jnp.diagonal(g4, axis1=0, axis2=2)
    # move the trailing diag (N) axis to the front -> (N, B_o, B_i, *L)
    return jnp.moveaxis(diag, -1, 0)


def _scatter_blocks_onto_grid(dense_grid, new_blocks, N, B_o, B_i):
    """Write ``new_blocks`` ``(N, B_o, B_i, *L)`` onto the meta-diagonal blocks
    of ``dense_grid`` ``(N*B_o, N*B_i, *L)``, leaving the off-diagonal entries
    untouched.  Single broadcast + ``where`` over an ``N x N`` eye mask (no
    scatter)."""
    L = dense_grid.shape[2:]
    g4 = dense_grid.reshape((N, B_o, N, B_i) + L)  # (g, r, h, c, *L)
    # new_blocks (N, B_o, B_i, *L) -> place on the (g == h) diagonal:
    # broadcast to (g, r, h, c, *L) by inserting the h axis.
    nb = new_blocks.reshape((N, B_o, 1, B_i) + L)
    nb = jnp.broadcast_to(nb, (N, B_o, N, B_i) + L)
    eye = jnp.eye(N, dtype=jnp.bool_).reshape((N, 1, N, 1) + (1,) * len(L))
    out4 = jnp.where(eye, nb, g4)
    return out4.reshape((N * B_o, N * B_i) + L)


# --------------------------------------------------------------------------- #
# Topology resolution
# --------------------------------------------------------------------------- #
def _split_2d(st: "SparseTensor"):
    """Return ``(out_dim, primal_dim)`` for a 2-D-per-operand tensor."""
    if len(st.out_dims) != 1 or len(st.primal_dims) != 1:
        raise ValueError(
            "elementwise_dense_block_diagonal handles the 2-D core (one out + "
            f"one primal dim); got {len(st.out_dims)} out / "
            f"{len(st.primal_dims)} primal."
        )
    return st.out_dims[0], st.primal_dims[0]


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
def elementwise_dense_block_diagonal(
    lhs: "SparseTensor",
    rhs: "SparseTensor",
    op: Callable,
    is_intersection: bool = False,
) -> "SparseTensor":
    """Elementwise ``op(lhs, rhs)`` for the (Dense, BlockDiagonal) component
    pairings — ``D op B``, ``B op D``, ``B op B`` (matched grid).

    ``is_intersection`` declares ``op`` an intersection op (``op(x, 0) == 0``,
    e.g. ``multiply``): then a ``D op B`` / ``B op D`` result stays on B's
    meta-block-diagonal support (closed as ``B``).  Otherwise (union, e.g.
    ``add``) a D-involving result is Dense.

    Cost ~ ``nnz`` for the block op; the intersection / B-op-B paths never
    materialize the ``N^2`` dense grid.  Non-zero fills or a mismatched B/B grid
    fall back to the dense oracle (correctness over speed).
    """
    from graphax.sparse.tensor import SparseTensor

    l_out, l_pri = _split_2d(lhs)
    r_out, r_pri = _split_2d(rhs)

    if lhs.shape != rhs.shape:
        raise ValueError(f"Shape mismatch: {lhs.shape} != {rhs.shape}")

    l_is_b = l_out.is_sparse
    r_is_b = r_out.is_sparse

    out_dtype = _compute_dtype(lhs.dtype, rhs.dtype)

    # Non-zero fill on either operand: the support-narrowing arguments above
    # assume zero fill (op(x, 0) leaves the union/intersection clean). Fall back
    # to the dense oracle, which is always correct.
    if not (_is_zero_fill(lhs) and _is_zero_fill(rhs)):
        return _dense_fallback(lhs, rhs, op, out_dtype)

    # ---- B op B (matched grid) -------------------------------------------- #
    if l_is_b and r_is_b:
        if not _grids_match(l_out, l_pri, r_out, r_pri):
            # Mismatched block grids: no shared simple support — defer to the
            # general LCM machinery via the dense oracle.
            return _dense_fallback(lhs, rhs, op, out_dtype)
        N = l_out.size
        B_o = l_out.block_size or 1
        B_i = l_pri.block_size or 1
        a = _diag_grid(lhs, l_out, l_pri)  # (N, B_o, B_i, *L)
        b = _diag_grid(rhs, r_out, r_pri)
        new_blocks = op(a, b).astype(out_dtype)
        return _emit_block_diagonal(new_blocks, N, B_o, B_i, out_dtype)

    # ---- exactly one B ---------------------------------------------------- #
    b_t, b_out, b_pri = (lhs, l_out, l_pri) if l_is_b else (rhs, r_out, r_pri)
    d_t, d_out, d_pri = (rhs, r_out, r_pri) if l_is_b else (lhs, l_out, l_pri)

    N = b_out.size
    B_o = b_out.block_size or 1
    B_i = b_pri.block_size or 1

    blocks = _diag_grid(b_t, b_out, b_pri)              # (N, B_o, B_i, *L)
    d_grid, L = _dense_grid(d_t, d_out, d_pri)          # (R, C, *L)
    d_sub = _diag_subblocks(d_grid, N, B_o, B_i)        # (N, B_o, B_i, *L)

    # Apply op block-by-block IN OPERAND ORDER (op may be non-commutative, e.g.
    # subtract / divide): lhs-block first.
    if l_is_b:
        new_blocks = op(blocks, d_sub)
    else:
        new_blocks = op(d_sub, blocks)
    new_blocks = new_blocks.astype(out_dtype)

    if is_intersection:
        # Result stays on B's support (off-diagonal op(D, 0) == 0). Emit B.
        return _emit_block_diagonal(new_blocks, N, B_o, B_i, out_dtype)

    # Union: off the meta-block-diagonal B is implicitly ZERO, so each off-diagonal
    # cell is ``op`` applied with that zero on B's side — and ``op`` may be
    # NON-COMMUTATIVE (subtract / divide), so the operand order matters: op(D, 0)
    # when D is the lhs, op(0, D) when B is the lhs. (Scattering D's raw grid, as
    # before, is only correct for D-as-lhs add/subtract; B-as-lhs ``B - D`` /
    # ``B / D`` need ``-D`` / ``0`` off-diagonal.) The diagonal cells carry the
    # real per-block op (new_blocks).
    zero = jnp.zeros_like(d_grid)
    off_grid = op(zero, d_grid) if l_is_b else op(d_grid, zero)
    out = _scatter_blocks_onto_grid(off_grid, new_blocks, N, B_o, B_i)
    out = out.astype(out_dtype)
    return _emit_dense(out, N * B_o, N * B_i, out_dtype)


# --------------------------------------------------------------------------- #
# Helpers: grid match / fallback / emit
# --------------------------------------------------------------------------- #
def _grids_match(a_out, a_pri, b_out, b_pri) -> bool:
    """Two DiagonalIndex pairs share the same meta-block-diagonal support iff
    their meta counts and both block sizes agree."""
    return (
        a_out.size == b_out.size
        and (a_out.block_size or 1) == (b_out.block_size or 1)
        and (a_pri.block_size or 1) == (b_pri.block_size or 1)
    )


def _dense_fallback(lhs, rhs, op, out_dtype) -> "SparseTensor":
    """Materialize both operands and run the plain dense op — the always-correct
    oracle, used for non-zero fills and mismatched B/B grids."""
    from graphax.sparse.ops.utils import _arr2st

    out = op(lhs.dense(), rhs.dense()).astype(out_dtype)
    return _arr2st(out, out_ndim=len(lhs.out_dims))


def _emit_block_diagonal(blocks, N, B_o, B_i, dtype) -> "SparseTensor":
    """Wrap ``blocks`` ``(N, B_o, B_i, *L)`` as a DiagonalIndex-pair SparseTensor
    (zero fill, scalar_mult folded -> 1).  Block axes are dropped (set to None)
    when the block size is 1, matching the canonical compact layout."""
    from graphax.sparse.tensor import SparseTensor

    blocks = blocks.astype(dtype)
    L = blocks.shape[3:]
    # Canonical val layout: (N, [B_o], [B_i], *L) — squeeze size-1 block axes.
    has_o = B_o > 1
    has_i = B_i > 1
    val = blocks
    axis = 0
    bo_axis = None
    bi_axis = None
    next_ax = 1
    if has_o:
        bo_axis = next_ax
        next_ax += 1
    else:
        val = val[:, 0]  # drop B_o axis
    if has_i:
        bi_axis = next_ax
        next_ax += 1
    else:
        # B_i axis sits right after the (already-maybe-dropped) B_o axis.
        drop = 2 if has_o else 1
        val = jnp.take(val, 0, axis=drop)

    out_dim = DiagonalIndex(
        0, N, axis=axis, other_id=1,
        block_size=B_o if has_o else None, block_axis=bo_axis,
    )
    primal_dim = DiagonalIndex(
        1, N, axis=axis, other_id=0,
        block_size=B_i if has_i else None, block_axis=bi_axis,
    )
    return SparseTensor(
        (out_dim,), (primal_dim,), val,
        scalar_mult=jnp.array(1, dtype=dtype),
        fill_value=None,
        check_consistency=False,
    )


def _emit_dense(grid, R, C, dtype) -> "SparseTensor":
    """Wrap a dense ``(R, C, *L)`` grid as a fully-Dense SparseTensor (zero fill,
    scalar_mult folded -> 1).  Leftover ``*L`` axes become trailing primal-side
    dense dims."""
    from graphax.sparse.tensor import SparseTensor

    grid = grid.astype(dtype)
    L = grid.shape[2:]
    out_dims = (DenseIndex(0, R, 0),)
    primal_dims = [DenseIndex(1, C, 1)]
    for k, sz in enumerate(L):
        primal_dims.append(DenseIndex(2 + k, sz, 2 + k))
    return SparseTensor(
        out_dims, tuple(primal_dims), grid,
        scalar_mult=jnp.array(1, dtype=dtype),
        fill_value=None,
        check_consistency=False,
    )
