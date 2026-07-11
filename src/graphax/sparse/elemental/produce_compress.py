r"""Producer micro-action consumer: the IMPLICIT (compressed) dim algebra.

This module owns the contraction / elementwise SEMANTICS of an **implicit**
logical dim — a plain ``DenseIndex`` (``D``) whose physical ``val`` axis has been
collapsed away by :class:`~graphax.sparse.micro_actions.Compress`
(``mean`` / ``min`` / ``max`` / ``median`` / ``abs_min`` / ``abs_max``).  The
producer :func:`~graphax.sparse.micro_actions.apply_compress` ALREADY does the
reduction and the bookkeeping (verified): it keeps the dim's ``logical_size`` and
marks the reduced dim ``axis=None`` (implicit), so ``val`` no longer carries that
axis.  This module defines how such an implicit dim CONTRACTS and ADDS — the
math, an nnz-cost fusion-friendly algorithm, and the closed ``{D, B}`` result —
WITHOUT ever materializing the dropped axis of logical size ``N``.

================================================================================
1. MATH
================================================================================
After ``Compress`` an implicit dim ``d`` is a ``DenseIndex`` with ``axis=None``
and ``logical_size = N``.  Reading it through ``.dense()`` BROADCASTS the stored
constant back along that logical axis: the edge is CONSTANT ``c`` along ``d`` —

    edge[..., n, ...] = c[..., ...]     for every  n in [0, N).            (impl)

``c`` is exactly what ``val`` holds for the surviving (non-implicit) axes; the
implicit axis simply repeats it ``N`` times.

----  IMPLICIT CONTRACTION  (``c * sum``)  -------------------------------------
Contract an implicit dim of ``lhs`` (logical size ``N``, constant ``c`` over it)
against a partner dim of ``rhs`` (logical size ``N``, values ``p[n, ...]``):

    out[...] = sum_{n=0}^{N-1} edge_lhs[..., n] * p[n, ...]
             = sum_{n=0}^{N-1} c[...] * p[n, ...]
             = c[...] * ( sum_{n} p[n, ...] ).                            (C)

The constant pulls OUT of the sum — the contraction is a *partner reduction*
``Σ_n p[n, ...]`` times the implicit constant ``c``.  We never build the length-
``N`` constant vector; we reduce the partner over its contracted axis (a single
``jnp.sum`` / ``dot_general`` against a ``ones`` row, fusion-friendly) and scale.
The partner may itself be Dense (``D``) or block-diagonal (``B``) — its reduction
is a plain sum either way (a ``B`` partner's contracted axis sums each meta block
and lays the per-meta sums on the surviving free side).

----  IMPLICIT ELEMENTWISE  (broadcast)  ---------------------------------------
An elementwise ``op`` pairs an implicit dim ``d`` (size ``N``, constant ``c``)
cell-for-cell with the OTHER operand's matching dim of size ``N``:

    out[..., n, ...] = op( c[...], q[..., n, ...] )      n in [0, N).      (E)

``c`` simply broadcasts along the logical axis; the result is whatever ``op``
produces against the (full) other operand.  For the canonical zero-fill
union op ``add`` the result is Dense (``D``) — the implicit constant is added to
every one of the ``N`` partner cells.  For an intersection op ``multiply`` the
result keeps the partner's support (``c`` scales each partner cell).

----  CLOSED RESULT (why it stays in {D, B} / implicit)  -----------------------
*Contraction (C)* consumes the implicit dim entirely (summed away), leaving the
partner's surviving free structure scaled by ``c``: a ``D`` partner's free axes
stay ``D``; a ``B`` partner's free (meta-block) side stays ``B`` — the per-meta
sums are independent, so the meta-block-diagonal support is preserved.  Hence
``impl @ D -> D`` and ``impl @ B -> B`` (and symmetrically with the implicit dim
on the right).  No off-diagonal structure is created.

*Elementwise (E)* either broadcasts ``c`` over a Dense partner (stays ``D``) or
scales a ``B`` partner's blocks (stays ``B``) — again no new structure.  In every
case the implicit axis is never enumerated, so the constant-along-``N`` property
is honoured exactly while paying only nnz(partner) cost.

================================================================================
2. ALGORITHM  (cost ~ nnz(partner), fusion-friendly)
================================================================================
The single structural primitive is "READ the implicit constant ``c`` from the
surviving val axes" and "REDUCE / BROADCAST the partner over the matched axis".
Both are reshape / transpose / sum / broadcast / multiply — XLA fuses them; there
is NO gather / scatter / fori_loop / python loop and the dropped ``N`` axis is
NEVER materialized.

  IMPLICIT CONTRACTION  (eq. C):
    1. Materialize the partner operand to ``(*free, N)`` with the contracted
       axis last (the partner is the FULL operand — its values are real data, so
       reading it is intrinsic, not waste).
    2. ``red = partner.sum(axis=-1)``                       (cost ~ nnz(partner))
    3. Read the implicit operand's constant ``c`` over its OWN surviving free
       axes (its ``val``; the implicit contracted axis is simply absent).
    4. ``out = c (outer/broadcast) red`` over the free axes, scaled by the two
       ``scalar_mult``.  Emit Dense (partner Dense) — the common producer case.

  IMPLICIT ELEMENTWISE  (eq. E):
    Broadcast the implicit constant ``c`` (its val, the implicit axis absent) up
    to the partner's logical shape via ``.dense()`` broadcasting and run ``op``.
    The implicit side's ``.dense()`` is a pure broadcast (no ``N`` buffer is
    built beyond the unavoidable output), so this is the dense oracle WITHOUT a
    separate length-``N`` intermediate for the constant.

``scalar_mult`` folds linearly; a non-zero ``fill_value`` on either operand (rare
for a Jacobian) routes to the dense oracle (correctness over speed).

================================================================================
3. SCOPE / DISPATCH (see INTEGRATION NOTE at bottom)
================================================================================
``contract_implicit(lhs, rhs)`` handles the contraction where the single
contracted pair has at least one IMPLICIT (``axis=None``) side; the partner may
be Dense or block-diagonal.  ``elementwise_implicit(lhs, rhs, op,
is_intersection)`` handles a cell-for-cell op where at least one operand carries
an implicit dim.  Both keep the implicit axis un-materialized.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp

from graphax.sparse.dtype_compute import _scaled_mul
from graphax.sparse.elemental._common import dense_op_fallback, emit_dense_result
from graphax.sparse.indexes import DenseIndex, Index
from graphax.sparse.ops.utils import _compute_dtype, _is_zero_fill

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


# --------------------------------------------------------------------------- #
# Implicit-dim predicates
# --------------------------------------------------------------------------- #
def _is_implicit(d: Index) -> bool:
    """A dim is IMPLICIT iff it is a plain Dense dim (not sparse / compressed)
    whose physical ``val`` axis was dropped by Compress — ``axis is None`` while
    its ``logical_size`` is the (broadcast) size ``N``.

    A ``DiagonalIndex`` may also carry ``axis=None`` (the meta axis lives on its
    partner), so we explicitly require a NON-sparse, non-compressed dim: only a
    compressed-away Dense dim is "implicit" in the (broadcast-constant) sense
    this kernel handles."""
    return (
        not d.is_sparse
        and not d.is_compressed
        and d.axis is None
    )


# --------------------------------------------------------------------------- #
# Materialization helpers (partner side only — implicit side stays un-built)
# --------------------------------------------------------------------------- #
def _partner_with_contract_last(
    st: "SparseTensor", contract_dim: Index
) -> tuple[jnp.ndarray, list[Index]]:
    """Materialize a (non-implicit) partner operand to ``(*free, N)`` — the
    contracted axis last, the surviving free dims (logical order) leading.

    Uses ``.dense()`` (which folds scalar_mult + any block structure + fill) and
    then moves the contracted logical axis to the end.  Returns the array plus
    the surviving free ``Index`` objects in logical order, so the caller can
    re-emit them.  ``.dense()`` lays axes out as ``(out..., primal...)`` in the
    operand's logical dim order, so the contracted dim's logical position drives
    the moveaxis."""
    dims = list(st.dims)
    pos = dims.index(contract_dim)
    arr = st.dense()  # (*logical_dims,) with scalar_mult / blocks / fill folded
    arr = jnp.moveaxis(arr, pos, -1)  # (*free, N)
    free_dims = [d for i, d in enumerate(dims) if i != pos]
    return arr, free_dims


def _implicit_constant(
    st: "SparseTensor", implicit_dim: Index
) -> tuple[jnp.ndarray, list[Index]]:
    """Read the IMPLICIT operand's constant ``c`` over its surviving free dims —
    the contracted (implicit) axis is simply ABSENT from ``val``, so no length-N
    buffer is created.

    Returns ``(c_arr, free_dims)`` where ``c_arr`` carries one physical axis per
    surviving free dim (logical order), with ``scalar_mult`` folded in.  Any
    OTHER implicit free dim of the operand (also ``axis=None``) is broadcast in
    as a singleton then expanded to its logical size — only the genuinely-present
    val axes are physical.  ``val is None`` (uniform) yields a ``scalar_mult * ones`` buffer.
    """
    free_dims = [d for d in st.dims if d.id != implicit_dim.id]

    if st.val is None:
        free_sizes = tuple(d.logical_size for d in free_dims)
        c = _scaled_mul(jnp.ones(free_sizes, dtype=st.dtype), st.scalar_mult)
        return c, free_dims

    val = _scaled_mul(st.val, st.scalar_mult)

    # Order the PRESENT free axes by logical dim order; implicit free dims carry
    # no physical axis (broadcast singleton).
    present_axes = [d.axis for d in free_dims if d.axis is not None]
    leftover = [a for a in range(val.ndim) if a not in present_axes]
    perm = present_axes + leftover
    v = val.transpose(perm) if perm != list(range(val.ndim)) else val
    # Drop any leftover (size-1) trailing axes.
    n_present = len(present_axes)
    if v.ndim > n_present:
        v = v.reshape(v.shape[:n_present])

    # Re-insert singleton axes for implicit free dims, in logical order, then
    # broadcast every dim up to its logical size.
    out_shape = []
    src = iter(v.shape)
    for d in free_dims:
        out_shape.append(next(src) if d.axis is not None else 1)
    v = v.reshape(tuple(out_shape)) if out_shape else v
    full = tuple(d.logical_size for d in free_dims)
    if v.shape != full:
        v = jnp.broadcast_to(v, full)
    return v, free_dims


# --------------------------------------------------------------------------- #
# Topology resolution
# --------------------------------------------------------------------------- #
def _find_contract_pair(lhs: "SparseTensor", rhs: "SparseTensor"):
    """Return ``(lhs_contract_dim, rhs_contract_dim)`` — the single contracted
    pair (lhs trailing primal dim vs rhs leading out dim) of equal logical
    size, matching the vertex-elim convention of the sibling kernels."""
    if not lhs.primal_dims or not rhs.out_dims:
        raise ValueError(
            "contract_implicit needs one primal dim on lhs and one out dim on "
            "rhs to contract."
        )
    lc = lhs.primal_dims[-1]
    rc = rhs.out_dims[0]
    if lc.logical_size != rc.logical_size:
        raise ValueError(
            f"Contraction size mismatch: lhs primal {lc.logical_size} vs rhs "
            f"out {rc.logical_size}."
        )
    return lc, rc


# --------------------------------------------------------------------------- #
# Contraction kernel
# --------------------------------------------------------------------------- #
def contract_implicit(
    lhs: "SparseTensor", rhs: "SparseTensor", lc=None, rc=None
) -> "SparseTensor":
    r"""Contract ``lhs @ rhs`` where the single contracted pair has at least one
    IMPLICIT (compressed-away, ``axis=None`` Dense) side.

    Implements eq. (C): the implicit constant ``c`` pulls out of the sum, so the
    contraction is ``c * (partner reduced over the contracted axis)``.  The
    partner may be Dense or block-diagonal.  The result is a fully-Dense
    ``SparseTensor`` (the closed form for a Dense partner / the common producer
    case), computed by a single partner-reduction + broadcast — cost ~
    nnz(partner), the implicit ``N`` axis is never materialized.

    If BOTH contracted sides are implicit, the contraction is
    ``c_lhs * c_rhs * N`` (each constant times the length, summed): both pull out
    and the ``Σ_n 1 = N`` factor remains.  Non-zero fills fall back to the dense
    oracle.

    ``lc`` / ``rc`` are the contracted (lhs-primal, rhs-out) dims; when the caller
    has resolved them (the dispatcher's authoritative pair) they are used directly,
    else resolved locally. They are consumed by IDENTITY, never by position.
    """
    if lc is None or rc is None:
        lc, rc = _find_contract_pair(lhs, rhs)

    l_impl = _is_implicit(lc)
    r_impl = _is_implicit(rc)
    if not (l_impl or r_impl):
        raise ValueError(
            "contract_implicit requires at least one contracted side to be an "
            f"IMPLICIT (axis=None Dense) dim; got lhs_implicit={l_impl}, "
            f"rhs_implicit={r_impl}. (D@D, D@B, B@B route to other kernels.)"
        )

    # Non-zero fill: the c*sum factorization assumes the implicit constant is the
    # WHOLE edge value along N (zero fill off it). Fall back to the dense oracle.
    if not (_is_zero_fill(lhs) and _is_zero_fill(rhs)):
        return dense_op_fallback(lhs, rhs, jnp.matmul)

    # AUDIT FIX: `_implicit_constant` reads the implicit operand's val directly
    # and drops every leftover physical axis as if size-1. A SPARSE (block-
    # diagonal) free dim on the implicit operand has a size-B>1 block_axis that
    # is NOT droppable -> reshape crash. This 2-D-core kernel cannot represent a
    # sparse free dim on the implicit side; raise so _route_single_kernel routes
    # the contraction to the (always-correct) composed-dense fallback, which
    # densifies both operands and re-runs the tiled topology resolver.
    if l_impl and any(d.is_sparse for d in lhs.dims if d.id != lc.id):
        raise ValueError(
            "contract_implicit: implicit lhs has a sparse (block-diagonal) free "
            "dim its constant-read cannot represent; defer to composed-dense.")
    if r_impl and any(d.is_sparse for d in rhs.dims if d.id != rc.id):
        raise ValueError(
            "contract_implicit: implicit rhs has a sparse (block-diagonal) free "
            "dim its constant-read cannot represent; defer to composed-dense.")

    out_dtype = _compute_dtype(lhs.dtype, rhs.dtype)
    N = lc.logical_size

    if l_impl and r_impl:
        # Both implicit: out = (c_lhs ⊗ c_rhs) * N  over the two free grids.
        c_l, l_free = _implicit_constant(lhs, lc)  # (*Lfree,)
        c_r, r_free = _implicit_constant(rhs, rc)  # (*Rfree,)
        # Outer product over free axes, times the contracted length N.
        out = (
            c_l.reshape(c_l.shape + (1,) * c_r.ndim)
            * c_r.reshape((1,) * c_l.ndim + c_r.shape)
        ) * jnp.asarray(N, dtype=out_dtype)
        out = out.astype(out_dtype)
        return _emit_contract(out, lhs, lc, l_free, rhs, rc, r_free, out_dtype)

    if l_impl:
        # lhs implicit: out = c_lhs ⊗ (Σ_n rhs[n, ...]).
        c, l_free = _implicit_constant(lhs, lc)            # (*Lfree,)
        r_arr, r_free = _partner_with_contract_last(rhs, rc)  # (*Rfree, N)
        red = r_arr.sum(axis=-1)                            # (*Rfree,)
    else:
        # rhs implicit: out = (Σ_n lhs[..., n]) ⊗ c_rhs.
        l_arr, l_free = _partner_with_contract_last(lhs, lc)  # (*Lfree, N)
        red = l_arr.sum(axis=-1)                            # (*Lfree,)
        c, r_free = _implicit_constant(rhs, rc)             # (*Rfree,)

    if l_impl:
        out = (
            c.reshape(c.shape + (1,) * red.ndim)
            * red.reshape((1,) * c.ndim + red.shape)
        )
    else:
        out = (
            red.reshape(red.shape + (1,) * c.ndim)
            * c.reshape((1,) * red.ndim + c.shape)
        )
    out = out.astype(out_dtype)
    return _emit_contract(out, lhs, lc, l_free, rhs, rc, r_free, out_dtype)


def _emit_contract(out, lhs, lc, l_free, rhs, rc, r_free, dtype) -> "SparseTensor":
    """Wrap the contracted ``(*lhs_free, *rhs_free)`` array as a fully-Dense
    ``SparseTensor``.  ``lhs``'s surviving free dims keep their out/primal side;
    ``rhs``'s surviving free dims keep theirs.  Physical axis order matches the
    array: ``lhs_free`` (its logical order) then ``rhs_free``.

    Side assignment: a surviving lhs free dim is an OUT dim iff it sits in
    ``lhs.out_dims``; a surviving rhs free dim is a PRIMAL dim iff it sits in
    ``rhs.primal_dims`` — mirroring the standard vertex-elim out/primal split.
    """
    from graphax.sparse.tensor import SparseTensor

    lhs_out_ids = {d.id for d in lhs.out_dims}
    rhs_out_ids = {d.id for d in rhs.out_dims}

    out_dims: list[Index] = []
    primal_dims: list[Index] = []
    axis = 0
    next_id = 0

    # lhs free dims: out-side dims become OUT, primal-side become PRIMAL.
    for d in l_free:
        idx = DenseIndex(next_id, d.logical_size, axis)
        (out_dims if d.id in lhs_out_ids else primal_dims).append(idx)
        next_id += 1
        axis += 1
    # rhs free dims: out-side dims become OUT, primal-side become PRIMAL.
    for d in r_free:
        idx = DenseIndex(next_id, d.logical_size, axis)
        (out_dims if d.id in rhs_out_ids else primal_dims).append(idx)
        next_id += 1
        axis += 1

    # Re-id: out dims first, primal dims following, for id contiguity. The
    # physical axis order in ``out`` is (lhs_free, rhs_free); we must reorder the
    # array so all OUT axes precede all PRIMAL axes (the dense() convention).
    out_axes = [d.axis for d in out_dims]
    primal_axes = [d.axis for d in primal_dims]
    perm = out_axes + primal_axes
    if perm != list(range(out.ndim)):
        out = jnp.transpose(out, perm)

    # After the transpose the array is laid out (out..., primal...), so the
    # contiguous physical axes are 0..n_out-1 (out) then n_out.. (primal).
    n_out = len(out_dims)
    out_specs = [(d.logical_size, i) for i, d in enumerate(out_dims)]
    primal_specs = [(d.logical_size, n_out + i) for i, d in enumerate(primal_dims)]
    return emit_dense_result(out, out_specs, primal_specs, dtype=dtype)


# --------------------------------------------------------------------------- #
# Elementwise kernel
# --------------------------------------------------------------------------- #
def _has_implicit(st: "SparseTensor") -> bool:
    """True iff ``st`` carries at least one implicit (compressed-away) dim."""
    return any(_is_implicit(d) for d in st.dims)


def elementwise_implicit(
    lhs: "SparseTensor",
    rhs: "SparseTensor",
    op: Callable,
    is_intersection: bool = False,
) -> "SparseTensor":
    r"""Elementwise ``op(lhs, rhs)`` where at least one operand carries an
    IMPLICIT (compressed-away) dim — eq. (E): the implicit constant ``c``
    broadcasts along its logical axis and is combined cell-for-cell with the
    other operand.

    The implicit operand's ``.dense()`` is a pure broadcast of its stored
    constant along the implicit axis (no separate length-``N`` buffer is built
    for the constant), so running ``op`` against the materialized operands is the
    dense oracle with the implicit axis honoured as a constant.  The result's
    structure follows the standard union (``add`` → Dense) / intersection
    (``multiply`` → partner support) rules; here we emit the always-correct Dense
    form (the closed result for an implicit operand under add, the common case).

    ``is_intersection`` is accepted for dispatch-signature parity with the
    sibling ``elementwise_D_B`` kernel; an implicit operand carries no narrowing
    support of its own (it is dense-constant), so the result rides on the OTHER
    operand's support — captured exactly by materializing both and running
    ``op``.
    """
    from graphax.sparse.ops.utils import _arr2st

    if lhs.shape != rhs.shape:
        raise ValueError(f"Shape mismatch: {lhs.shape} != {rhs.shape}")
    if not (_has_implicit(lhs) or _has_implicit(rhs)):
        raise ValueError(
            "elementwise_implicit requires at least one operand to carry an "
            "implicit (axis=None Dense) dim."
        )

    out_dtype = _compute_dtype(lhs.dtype, rhs.dtype)
    out = op(lhs.dense(), rhs.dense()).astype(out_dtype)
    return _arr2st(out, out_ndim=len(lhs.out_dims))
