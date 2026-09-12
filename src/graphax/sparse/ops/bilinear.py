"""The bilinear accumulators of a comparison, WITHOUT materializing either
operand (ticket dsnn-3qm.62, owner ruling (c), 2026-09-12).

WHY THIS EXISTS. ``alphagrad``'s reward path scores an approximated gradient
against the exact one with four numbers -- ``<e, a>``, ``||e||^2``, ``||a||^2``
and ``||e - a||^2``. All four are bilinear forms, so neither side has to be a
dense buffer: they contract from the two STORAGE forms directly. The comparison
used to call ``SparseTensor.dense()`` on a structured Jacobian leaf and on a
diagonal-paired approximated leaf -- allocating the full logical tensor for the
sole purpose of summing it, on the path that runs for EVERY terminal
measurement. The owner's ruling: "the comparison should remain lazy, should not
need to materialize, and we definitely don't want to densify".

WHY NOT THE OBVIOUS TWO-LINE VERSION. Handing the ``SparseTensor`` straight to
the existing arithmetic instead of its ``.dense()`` does not work, and every way
it fails is a densify hiding one layer down. Measured on CPU, 2026-09-12,
graphax cdabc9e, on a 4x4 diagonal pair stored as ``val (4, 1, 1)``:

  * ``jnp.sum(e_arr * a)`` -> ``TypeError: sum requires ndarray or scalar
    arguments, got graphax.sparse.tensor.SparseTensor``. ``jnp.sum`` does not
    dispatch on a ``SparseTensor``.
  * ``(e_arr * a).sum()`` DOES run and gives the right number (100.0), but the
    elementwise op promotes the dense operand against the pair as a ONE-BLOCK
    synthetic pair: ``val (4, 1, 1)`` comes back ``(4, 4)``. That is the
    densification, moved inside ``elementwise``.
  * ``abs(a) ** 2`` -> ``AttributeError: 'int' object has no attribute
    'astype'`` (``__pow__`` routes the python int through ``elementwise``).
  * ``SparseTensor.sum()`` is WRONG for a tensor with an IMPLICIT dim: it
    counts the broadcast cells as FILL cells, so a logical ``(4, 3)`` tensor
    storing one ``(3,)`` representative sums to 6.0 where its own ``dense()``
    sums to 24.0. So nothing here is built on ``sum()``.

THE MODEL. A ``SparseTensor`` without compressed dims has exactly three kinds
of dim, and its dense form is determined by them:

  * a MATERIALIZED dense dim owns one ``val`` axis;
  * an IMPLICIT dim (``axis is None``) owns none -- its extent is a BROADCAST
    of the stored value, not a fill;
  * a DIAGONAL PAIR (two dims with ``is_sparse``, each carrying the other's
    ``id``) shares a meta axis of extent ``N`` and owns one block axis each,
    ``B_out`` / ``B_in``; cell ``(i, j)`` is on-structure iff
    ``i // B_out == j // B_in``, and off-structure cells hold ``fill_value``.

Write ``G`` for the COMPACT array of STORED values in the canonical frame
``(N_1, ..., N_P) + (per-dim compact extent)`` -- 1 for an implicit dim,
``B_out`` / ``B_in`` for a pair member, the logical size for a dense dim -- and
``s`` for ``scalar_mult``, which is factored OUT of ``G`` and folded into the
scalar result at the end so no scaled copy of ``val`` is ever allocated. With
``n_bcast`` the product of the implicit extents and ``n_fill`` the number of
off-structure logical cells:

    ||t||^2 = |s|^2 * ( n_bcast * sum |G|^2  +  n_fill * |fill|^2 )
    <t, x>  = s * ( sum G * gather(reduce_implicit(x))
                    +  fill * ( sum x  -  sum gather(reduce_implicit(x)) ) )

``gather`` reads ``x`` only at the structure positions -- it allocates
``G.size``, not ``t.size`` -- and ``reduce_implicit`` folds ``t``'s broadcast
dims out of ``x`` first, which is the identity
``<e, bcast(a)> = <sum_over_implicit(e), a>`` that the implicit-dim branch of
``_gradient_similarity`` has used since ticket .62 landed. A ``val is None``
tensor keeps ``G`` UNBUILT (``uniform=True``, every stored cell is 1), so a
uniform leaf of logical shape ``(256, 784)`` costs nothing at all -- the
degenerate dead leaf the owner named ("only implicit dims and
``scalar_mult = 0`` is cheap and equal to a dense zero tensor") never
materializes. Nothing here builds an array of the logical shape.

A PAIR AXIS MAY ITSELF BE A BROADCAST, and that is not an exotic case -- it is
how Helmholtz's exact Jacobian arrives. Two degenerate forms occur and both are
handled rather than refused (measured 2026-09-12, job 65010):

  * ``size=1, axis=None, block_size=4, block_axis=0/1`` -- ONE meta block, i.e.
    a plain dense 4x4 matrix that happens to be stored as a pair. Its structure
    covers every logical cell with no broadcast, so it is a GATHER SOURCE.
  * ``size=4, axis=0, block_size=None, block_axis=None`` -- a pure diagonal,
    ``val (4,)`` for a logical ``(4, 4)``. The 1x1 blocks carry no axis.

Refusing these (which an earlier draft did, on the grounds that graphax itself
only reaches them through ``dense(hard=True)``) turned
``tests/landscape_map_sweep_test.py::test_measure_singleton_and_stacks`` red on
all four plan classes -- a REGRESSION against a previously-working measurement.
Generally: a frame axis with no ``val`` axis is a BROADCAST along that axis (the
value is constant, not absent), and it is summed out of the other operand first.

WHAT IS REFUSED, LOUDLY. Two operands that BOTH carry a broadcast or a fill and
whose structure frames differ share no frame, and there is no honest lazy
contraction for them here, so :func:`bilinear_accumulators` RAISES
:class:`LazyContractionUnsupported` naming both. It does NOT fall back to
``.dense()``: a comparison that materializes one layer down is the same defect
with a longer stack trace. A compressed (``BandedIndex`` / ``SetIndex``) dim is
refused too -- ``_gradient_similarity`` already rejects one in a returned
gradient outright.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

import jax.numpy as jnp
from jax import Array


class LazyContractionUnsupported(Exception):
    """The two operands' storage forms have no common compact frame, or one of
    them carries structure this module cannot contract lazily. Deliberately NOT
    a silent densification -- see the module doc."""


def _is_sparse_tensor(x) -> bool:
    from graphax.sparse.tensor import SparseTensor
    return isinstance(x, SparseTensor)


@dataclass(frozen=True)
class _View:
    """One operand in its STRUCTURE FRAME.

    The frame has one coordinate axis per diagonal pair (its meta, extent ``N``)
    followed by one per logical dim (a pair member's block extent ``B``, a
    non-pair dim's logical extent). ``frame`` holds those full extents and
    enumerates EXACTLY the on-structure logical cells, one to one. ``stored``
    is ``frame`` with every BROADCAST axis collapsed to 1 -- an axis is
    broadcast when no ``val`` axis carries it (``axis is None`` on a dim, a
    pair with no meta axis, a pair member with no block axis), which means the
    value is constant along it rather than absent. ``G`` has shape ``stored``
    and holds the STORED values unscaled; ``scale`` is folded into the scalar
    result so no scaled copy of ``val`` is built.
    """
    logical: tuple[int, ...]
    pair_pos: tuple[tuple[int, int], ...]   # (out dim position, in dim position)
    frame: tuple[int, ...]                  # len == n_pairs + ndim
    stored: tuple[int, ...]                 # same length; 1 where broadcast
    G: Array | None                         # shape == stored; None <=> uniform
    uniform: bool                           # val is None: every stored cell is 1
    scale: Array
    fill: Array | None                      # raw fill_value; None <=> statically 0
    n_fill: int                             # off-structure logical cells

    @property
    def n_struct(self) -> int:
        """On-structure logical cells."""
        return prod(self.frame)

    @property
    def n_stored(self) -> int:
        return prod(self.stored)

    @property
    def n_bcast(self) -> int:
        return self.n_struct // max(self.n_stored, 1)

    @property
    def n_logical(self) -> int:
        return prod(self.logical)

    @property
    def is_gather_source(self) -> bool:
        """The structure covers every logical cell exactly once and nothing is
        broadcast, so ``G`` reshaped to ``logical`` IS the dense form and this
        operand can be read at the other's structure.

        ``n_fill == 0`` with pairs present forces every meta to 1 (a pair's
        logical extents are ``N*B_out`` by ``N*B_in`` while its structure holds
        ``N*B_out*B_in``, so they agree only at ``N == 1``), and the frame's
        meta axes are then all extent 1 -- dropping them leaves the per-dim
        extents in dim order, which is the logical shape. That is why a
        DEGENERATE pair (one meta block, i.e. a plain dense matrix that happens
        to be stored as a pair) lands here instead of being refused: it is the
        form Helmholtz's exact Jacobian arrives in.
        """
        return self.n_fill == 0 and self.n_bcast == 1

    @property
    def key(self):
        """Structural identity: equal keys share a frame."""
        return (self.logical, self.pair_pos, self.frame, self.stored)

    def describe(self) -> str:
        return ("logical=%s pairs=%s frame=%s stored=%s n_fill=%d G=%s fill=%s"
                % (self.logical, self.pair_pos, self.frame, self.stored,
                   self.n_fill, "uniform(val=None)" if self.uniform
                   else tuple(self.G.shape), "0" if self.fill is None else "set"))

    # --- the reductions every formula is built from -----------------------
    def sum_abs2_G(self) -> Array:
        if self.uniform:
            return jnp.asarray(self.n_stored, self.scale.dtype)
        return jnp.sum(jnp.abs(self.G) ** 2)

    def sum_G(self) -> Array:
        if self.uniform:
            return jnp.asarray(self.n_stored, self.scale.dtype)
        return jnp.sum(self.G)

    def sum_G_times(self, other: Array) -> Array:
        """``sum G * other`` for an ``other`` already in this stored frame."""
        if self.uniform:
            return jnp.sum(other)
        return jnp.sum(self.G * other)

    def dense_form(self) -> Array:
        """Only valid when :attr:`is_gather_source`."""
        return self.G.reshape(self.logical)


def _view_of_array(arr, dtype) -> _View:
    a = jnp.asarray(arr).astype(dtype)
    shape = tuple(int(v) for v in a.shape)
    return _View(logical=shape, pair_pos=(), frame=shape, stored=shape, G=a,
                 uniform=False, scale=jnp.ones((), dtype), fill=None, n_fill=0)


def _view_of_sparse(st, dtype) -> _View:
    dims = st.dims
    if any(getattr(d, "is_compressed", False) for d in dims):
        raise LazyContractionUnsupported(
            f"a compressed (Banded/Set) dim has no compact contraction frame: "
            f"dims {dims}")

    logical = tuple(int(d.logical_size) for d in dims)
    ndim = len(dims)

    # --- pair the diagonal dims -------------------------------------------
    pair_pos, seen = [], set()
    for pos, d in enumerate(dims):
        if not d.is_sparse:
            continue
        key = frozenset((int(d.id), int(d.other_id)))
        if key in seen:
            continue
        partner = next((q for q, x in enumerate(dims)
                        if q != pos and x.is_sparse
                        and int(x.id) == int(d.other_id)), None)
        if partner is None:
            raise LazyContractionUnsupported(
                f"dim {d} is half of a diagonal pair whose partner is not in "
                f"this tensor: dims {dims}")
        seen.add(key)
        pair_pos.append((pos, partner))
    n_pair = len(pair_pos)
    pair_of = {}
    for k, (po, pi) in enumerate(pair_pos):
        pair_of[po] = k
        pair_of[pi] = k

    # --- the frame, and which of its axes val actually carries -------------
    # val_axis[i] is the val axis carrying frame axis i, or None (broadcast).
    frame: list[int] = [0] * (n_pair + ndim)
    val_axis: list[int | None] = [None] * (n_pair + ndim)
    for k, (po, pi) in enumerate(pair_pos):
        d_o, d_i = dims[po], dims[pi]
        if int(d_o.size) != int(d_i.size):
            raise LazyContractionUnsupported(
                f"diagonal pair meta sizes disagree: {d_o} vs {d_i}")
        frame[k] = int(d_o.size)
        if d_o.axis is not None and d_i.axis is not None and int(d_o.axis) != int(d_i.axis):
            raise LazyContractionUnsupported(
                f"diagonal pair members name different meta axes: {d_o} / {d_i}")
        # A pair with NO meta axis in val is a BROADCAST along the meta: the
        # same block sits on every diagonal position. At N == 1 that is the
        # degenerate "one block" pair, i.e. a plain dense matrix.
        meta_ax = d_o.axis if d_o.axis is not None else d_i.axis
        val_axis[k] = None if meta_ax is None else int(meta_ax)
    for p, d in enumerate(dims):
        i = n_pair + p
        if p in pair_of:
            frame[i] = int(d.block_size or 1)
            # block_axis None with block_size 1 carries no extent; with
            # block_size > 1 it is a broadcast along the block.
            val_axis[i] = None if d.block_axis is None else int(d.block_axis)
        else:
            frame[i] = int(d.logical_size)
            val_axis[i] = None if d.axis is None else int(d.axis)
    stored = [1 if val_axis[i] is None else frame[i] for i in range(len(frame))]

    n_struct = prod(frame)
    n_fill = prod(logical) - n_struct
    if n_fill < 0:
        raise LazyContractionUnsupported(
            f"structure cells {n_struct} exceed the logical size "
            f"{prod(logical)}: dims {dims} val "
            f"{None if st.val is None else tuple(st.val.shape)}")

    # --- the stored values, in frame order ---------------------------------
    sm = jnp.asarray(st.scalar_mult).astype(dtype)
    G, uniform = None, True
    if st.val is not None:
        uniform = False
        val = jnp.asarray(st.val).astype(dtype)
        order = [val_axis[i] for i in range(len(frame)) if val_axis[i] is not None]
        want = [frame[i] for i in range(len(frame)) if val_axis[i] is not None]
        if len(set(order)) != len(order):
            raise LazyContractionUnsupported(
                f"two frame axes claim the same val axis: {order}, dims {dims}")
        if order and max(order) >= val.ndim:
            raise LazyContractionUnsupported(
                f"a dim names val axis {max(order)} but val has rank "
                f"{val.ndim}: dims {dims}")
        got = [int(val.shape[a]) for a in order]
        if want != got:
            raise LazyContractionUnsupported(
                f"stale layout metadata: the dims claim val axes {order} of "
                f"extents {want} but val {tuple(val.shape)} has {got}; "
                f"dims {dims}")
        leftover = [a for a in range(val.ndim) if a not in order]
        bad = [a for a in leftover if int(val.shape[a]) != 1]
        if bad:
            raise LazyContractionUnsupported(
                f"val axes {bad} (extents {[int(val.shape[a]) for a in bad]}) "
                f"are named by no dim and are not size-1: refusing to drop "
                f"data; val {tuple(val.shape)} dims {dims}")
        G = val.transpose(order + leftover)
        G = G.reshape(G.shape[:len(order)]).reshape(tuple(stored))

    fill = None
    if n_fill and st.fill_value is not None:
        fill = jnp.asarray(st.fill_value).astype(dtype)

    return _View(logical=logical, pair_pos=tuple(pair_pos), frame=tuple(frame),
                 stored=tuple(stored), G=G, uniform=uniform, scale=sm,
                 fill=fill, n_fill=int(n_fill))


def view_of(x, dtype) -> _View:
    """The structure frame of an Array or a ``SparseTensor``."""
    return _view_of_sparse(x, dtype) if _is_sparse_tensor(x) else _view_of_array(x, dtype)


def _gather_into(view: _View, x: Array) -> Array:
    """``x`` (an array of the logical shape) read at ``view``'s structure and
    reduced onto ``view``'s STORED frame.

    The gather enumerates the on-structure cells only -- ``n_struct`` elements,
    never ``n_logical`` -- from an index array built by broadcasting. The sum
    over the broadcast axes afterwards is the identity
    ``<bcast(G), x> = <G, sum_over_bcast(x)>``, which is what the implicit-dim
    comparison has used since ticket .62 landed, now applied to a pair's meta
    and block axes as well.
    """
    n_pair = len(view.pair_pos)
    rank = len(view.frame)
    pair_of = {}
    for k, (po, pi) in enumerate(view.pair_pos):
        pair_of[po] = k
        pair_of[pi] = k

    strides, acc = [1] * len(view.logical), 1
    for p in range(len(view.logical) - 1, -1, -1):
        strides[p] = acc
        acc *= int(view.logical[p])

    flat = jnp.zeros((), jnp.int32)
    for p in range(len(view.logical)):
        i = n_pair + p
        ext = int(view.frame[i])
        sh = [1] * rank
        sh[i] = ext
        coord = jnp.arange(ext, dtype=jnp.int32).reshape(sh)
        if p in pair_of:
            k = pair_of[p]
            N = int(view.frame[k])
            shn = [1] * rank
            shn[k] = N
            coord = jnp.arange(N, dtype=jnp.int32).reshape(shn) * ext + coord
        flat = flat + coord * jnp.asarray(strides[p], jnp.int32)

    flat = jnp.broadcast_to(flat, view.frame)
    g = x.reshape(-1)[flat.reshape(-1)].reshape(view.frame)
    bax = tuple(i for i in range(rank) if view.stored[i] != view.frame[i])
    if bax:
        g = jnp.sum(g, axis=bax, keepdims=True)
    return g.reshape(view.stored)


def squared_norm(x, dtype) -> Array:
    """``sum |x|^2`` over the DENSE form of ``x``, without materializing it."""
    return _norm2(view_of(x, dtype))


def _norm2(v: _View) -> Array:
    tot = v.n_bcast * v.sum_abs2_G()
    if v.n_fill and v.fill is not None:
        tot = tot + v.n_fill * jnp.abs(v.fill) ** 2
    return jnp.abs(v.scale) ** 2 * tot


def _dot_dense_struct(dense: _View, other: _View) -> Array:
    """``<dense, other>`` with ``dense`` a gather source."""
    if dense.uniform:
        # the dense form is all ones: the gather is the constant n_bcast per
        # stored cell, and the complement is exactly the fill-cell count
        on = other.n_bcast * other.sum_G()
        g_sum = jnp.asarray(other.n_struct, other.scale.dtype)
        x_sum = jnp.asarray(other.n_logical, other.scale.dtype)
    else:
        x = dense.dense_form()
        g = _gather_into(other, x)
        on = other.sum_G_times(g)
        g_sum = jnp.sum(g)
        x_sum = jnp.sum(x)
    if other.n_fill and other.fill is not None:
        on = on + other.fill * (x_sum - g_sum)
    return dense.scale * other.scale * on


def _dot_same_frame(u: _View, v: _View) -> Array:
    """``<u, v>`` for two operands sharing one frame."""
    if u.uniform and v.uniform:
        inner = jnp.asarray(u.n_stored, u.scale.dtype)
    elif u.uniform:
        inner = v.sum_G()
    elif v.uniform:
        inner = u.sum_G()
    else:
        inner = jnp.sum(u.G * v.G)
    tot = u.n_bcast * inner
    if u.n_fill and (u.fill is not None or v.fill is not None):
        zero = jnp.zeros((), u.scale.dtype)
        fu = u.fill if u.fill is not None else zero
        fv = v.fill if v.fill is not None else zero
        tot = tot + u.n_fill * fu * fv
    return u.scale * v.scale * tot


def bilinear_accumulators(e, a, *, dtype=None):
    """``(dot, ||e||^2, ||a||^2)`` for two operands, NEITHER materialized.

    ``e`` / ``a`` are each an Array or a ``SparseTensor``. The residual
    ``||e - a||^2 = ||e||^2 - 2 <e, a> + ||a||^2`` is left to the caller -- the
    same algebraic form the implicit-dim comparison has used since ticket .62
    landed.

    Raises :class:`LazyContractionUnsupported` when the two storage forms share
    no frame. It never densifies as a fallback.
    """
    if dtype is None:
        dtype = jnp.promote_types(
            jnp.promote_types(getattr(e, "dtype", jnp.float32),
                              getattr(a, "dtype", jnp.float32)), jnp.float32)
    ve, va = view_of(e, dtype), view_of(a, dtype)
    if ve.logical != va.logical:
        raise LazyContractionUnsupported(
            f"logical shapes differ: {ve.logical} vs {va.logical}")
    if ve.is_gather_source:
        dot = _dot_dense_struct(ve, va)
    elif va.is_gather_source:
        dot = _dot_dense_struct(va, ve)
    elif ve.key == va.key:
        dot = _dot_same_frame(ve, va)
    else:
        raise LazyContractionUnsupported(
            "neither operand covers its logical shape without a broadcast or a "
            "fill, and their structure frames differ, so they share no frame. "
            "Densifying one of them here would defeat the no-materialize "
            "contract (ticket dsnn-3qm.62 ruling (c)), so this raises instead. "
            "Extend this module rather than calling .dense().\n"
            f"  lhs: {ve.describe()}\n"
            f"  rhs: {va.describe()}")
    return dot, _norm2(ve), _norm2(va)
