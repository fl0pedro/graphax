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

WHAT IS REFUSED, LOUDLY. Two structured operands whose structures are neither
equal nor "one of them fully materialized" have no common compact frame, and
there is no honest lazy contraction for them here, so
:func:`bilinear_accumulators` RAISES :class:`LazyContractionUnsupported` naming
both structures. It does NOT fall back to ``.dense()``: a comparison that
materializes one layer down is the same defect with a longer stack trace. The
same applies to a compressed (``BandedIndex`` / ``SetIndex``) dim and to a pair
whose meta or block axis is itself implicit -- both of which graphax reaches
only through ``dense(hard=True)``.
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
    """One operand's compact structural view. ``G`` holds the STORED (unscaled)
    values; ``scale`` is folded in at the end."""
    logical: tuple[int, ...]
    metas: tuple[int, ...]                  # N per diagonal pair, in pair order
    pair_pos: tuple[tuple[int, int], ...]   # (out dim position, in dim position)
    implicit: tuple[int, ...]               # dim positions with axis None
    compact: tuple[int, ...]                # per dim position; 1 for implicit
    G: Array | None                         # (metas..., *compact); None <=> uniform
    uniform: bool                           # val is None: every stored cell is 1
    scale: Array                            # scalar_mult
    fill: Array | None                      # raw fill_value; None <=> statically 0
    n_fill: int                             # off-structure logical cells

    @property
    def n_bcast(self) -> int:
        return prod(int(self.logical[p]) for p in self.implicit) if self.implicit else 1

    @property
    def n_struct(self) -> int:
        """Stored cells in the compact frame (``G.size``, built or not)."""
        return prod(self.metas) * prod(self.compact)

    @property
    def n_logical(self) -> int:
        return prod(self.logical)

    @property
    def is_materialized(self) -> bool:
        """No pairs and no implicit dims, so the compact frame IS the logical
        shape: this operand can serve as the ``x`` of a gather."""
        return not self.metas and not self.implicit

    @property
    def key(self):
        """Structural identity: equal keys share a compact frame."""
        return (self.logical, self.metas, self.pair_pos, self.implicit, self.compact)

    def describe(self) -> str:
        return ("logical=%s pairs=%s metas=%s implicit=%s compact=%s "
                "stored=%s fill=%s" % (
                    self.logical, self.pair_pos, self.metas, self.implicit,
                    self.compact,
                    "uniform(val=None)" if self.uniform else tuple(self.G.shape),
                    "0" if self.fill is None else "set"))

    # --- the three reductions every formula is built from -----------------
    def sum_abs2_G(self) -> Array:
        if self.uniform:
            return jnp.asarray(self.n_struct, self.scale.dtype)
        return jnp.sum(jnp.abs(self.G) ** 2)

    def sum_G(self) -> Array:
        if self.uniform:
            return jnp.asarray(self.n_struct, self.scale.dtype)
        return jnp.sum(self.G)

    def sum_G_times(self, other: Array) -> Array:
        """``sum G * other`` for an ``other`` already in this compact frame."""
        if self.uniform:
            return jnp.sum(other)
        return jnp.sum(self.G * other)


def _view_of_array(arr, dtype) -> _View:
    a = jnp.asarray(arr).astype(dtype)
    shape = tuple(int(v) for v in a.shape)
    return _View(logical=shape, metas=(), pair_pos=(), implicit=(), compact=shape,
                 G=a, uniform=False, scale=jnp.ones((), dtype), fill=None, n_fill=0)


def _view_of_sparse(st, dtype) -> _View:
    dims = st.dims
    if any(getattr(d, "is_compressed", False) for d in dims):
        raise LazyContractionUnsupported(
            f"a compressed (Banded/Set) dim has no compact contraction frame: "
            f"dims {dims}")

    logical = tuple(int(d.logical_size) for d in dims)
    implicit, pair_pos, seen = [], [], set()
    for pos, d in enumerate(dims):
        if d.is_sparse:
            key = frozenset((int(d.id), int(d.other_id)))
            if key in seen:
                continue
            partner = next((q for q, x in enumerate(dims)
                            if q != pos and x.is_sparse
                            and int(x.id) == int(d.other_id)), None)
            if partner is None:
                raise LazyContractionUnsupported(
                    f"dim {d} is half of a diagonal pair whose partner is not "
                    f"in this tensor: dims {dims}")
            seen.add(key)
            pair_pos.append((pos, partner))
        elif d.axis is None:
            implicit.append(pos)

    metas, compact = [], [0] * len(dims)
    for pos in implicit:
        compact[pos] = 1
    for pos, d in enumerate(dims):
        if not d.is_sparse and d.axis is not None:
            compact[pos] = int(d.logical_size)
    for (po, pi) in pair_pos:
        d_o, d_i = dims[po], dims[pi]
        if int(d_o.size) != int(d_i.size):
            raise LazyContractionUnsupported(
                f"diagonal pair meta sizes disagree: {d_o} vs {d_i}")
        metas.append(int(d_o.size))
        compact[po] = int(d_o.block_size or 1)
        compact[pi] = int(d_i.block_size or 1)

    sm = jnp.asarray(st.scalar_mult).astype(dtype)
    target = tuple(metas) + tuple(compact)

    G, uniform = None, True
    if st.val is not None:
        uniform = False
        val = jnp.asarray(st.val).astype(dtype)
        order: list[int] = []
        for (po, pi) in pair_pos:
            d_o, d_i = dims[po], dims[pi]
            if d_o.axis is None or d_i.axis is None:
                raise LazyContractionUnsupported(
                    f"a diagonal pair with an IMPLICIT meta axis is reachable "
                    f"only through dense(hard=True): {d_o} / {d_i}")
            if int(d_o.axis) != int(d_i.axis):
                raise LazyContractionUnsupported(
                    f"diagonal pair members do not share their meta axis: "
                    f"{d_o} / {d_i}")
            order.append(int(d_o.axis))
        for pos, d in enumerate(dims):
            if pos in implicit:
                continue
            ax = d.block_axis if d.is_sparse else d.axis
            if ax is None:
                raise LazyContractionUnsupported(
                    f"dim {d} names no val axis and is not implicit; reachable "
                    f"only through dense(hard=True)")
            order.append(int(ax))
        if len(set(order)) != len(order):
            raise LazyContractionUnsupported(
                f"two dims claim the same val axis: order {order}, dims {dims}")
        if max(order, default=-1) >= val.ndim:
            raise LazyContractionUnsupported(
                f"a dim names val axis {max(order)} but val has rank "
                f"{val.ndim}: dims {dims}")
        # each claim must match the extent it claims -- the same check
        # ops/dense.py makes before it trusts a dim's .axis
        want = list(metas) + [compact[p] for p in range(len(dims)) if p not in implicit]
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
        G = G.reshape(G.shape[:len(order)]).reshape(target)

    n_bcast = prod(int(logical[p]) for p in implicit) if implicit else 1
    n_fill = prod(logical) - prod(target) * n_bcast
    if n_fill < 0:
        raise LazyContractionUnsupported(
            f"structure cells {prod(target) * n_bcast} exceed the logical size "
            f"{prod(logical)}: dims {dims} val "
            f"{None if st.val is None else tuple(st.val.shape)}")
    fill = None
    if n_fill and st.fill_value is not None:
        fill = jnp.asarray(st.fill_value).astype(dtype)

    return _View(logical=logical, metas=tuple(metas), pair_pos=tuple(pair_pos),
                 implicit=tuple(implicit), compact=tuple(compact), G=G,
                 uniform=uniform, scale=sm, fill=fill, n_fill=int(n_fill))


def view_of(x, dtype) -> _View:
    """The compact structural view of an Array or a ``SparseTensor``."""
    return _view_of_sparse(x, dtype) if _is_sparse_tensor(x) else _view_of_array(x, dtype)


def _gather_into(view: _View, x: Array) -> Array:
    """``x`` (an array of the logical shape) read at ``view``'s structure, in
    ``view``'s compact frame ``(metas..., *compact)``.

    ``view``'s implicit dims are SUMMED OUT of ``x`` first: ``view``'s value is
    constant along them, so ``<bcast(G), x> = <G, sum_implicit(x)>``. What
    remains is one flat gather of ``n_struct`` elements, its index array built
    by broadcasting -- no array of the logical shape is created.
    """
    imp = set(view.implicit)
    if view.implicit:
        x = jnp.sum(x, axis=view.implicit, keepdims=True)
    red = tuple(1 if p in imp else int(view.logical[p])
                for p in range(len(view.logical)))
    x = x.reshape(red)
    target = tuple(view.metas) + tuple(view.compact)
    if not view.metas:
        return x.reshape(target)

    strides, acc = [1] * len(red), 1
    for p in range(len(red) - 1, -1, -1):
        strides[p] = acc
        acc *= int(red[p])

    n_ax = len(view.metas)
    rank = n_ax + len(view.logical)
    pair_of = {}
    for k, (po, pi) in enumerate(view.pair_pos):
        pair_of[po] = k
        pair_of[pi] = k

    flat = jnp.zeros((), jnp.int32)
    for p in range(len(view.logical)):
        if p in imp:
            continue                       # coordinate 0: contributes nothing
        if p in pair_of:
            k = pair_of[p]
            N, B = int(view.metas[k]), int(view.compact[p])
            sh = [1] * rank
            sh[k] = N
            n_idx = jnp.arange(N, dtype=jnp.int32).reshape(sh)
            sh2 = [1] * rank
            sh2[n_ax + p] = B
            b_idx = jnp.arange(B, dtype=jnp.int32).reshape(sh2)
            coord = n_idx * B + b_idx
        else:
            ext = int(view.compact[p])
            sh = [1] * rank
            sh[n_ax + p] = ext
            coord = jnp.arange(ext, dtype=jnp.int32).reshape(sh)
        flat = flat + coord * jnp.asarray(strides[p], jnp.int32)

    flat = jnp.broadcast_to(flat, target)
    return x.reshape(-1)[flat.reshape(-1)].reshape(target)


def squared_norm(x, dtype) -> Array:
    """``sum |x|^2`` over the DENSE form of ``x``, without materializing it."""
    return _norm2(view_of(x, dtype))


def _norm2(v: _View) -> Array:
    tot = v.n_bcast * v.sum_abs2_G()
    if v.n_fill and v.fill is not None:
        tot = tot + v.n_fill * jnp.abs(v.fill) ** 2
    return jnp.abs(v.scale) ** 2 * tot


def _dot_dense_struct(dense: _View, other: _View) -> Array:
    """``<dense, other>`` with ``dense`` fully materialized (no pairs, no
    implicit dims), so it can be read at ``other``'s structure."""
    if dense.uniform:
        # dense form is all-ones: sum over structure is sum_G, sum over the
        # complement is exactly the fill-cell count
        on = other.sum_G()
        if other.n_fill and other.fill is not None:
            on = on + other.fill * other.n_fill
    else:
        x = dense.G.reshape(dense.logical)
        g = _gather_into(other, x)
        on = other.sum_G_times(g)
        if other.n_fill and other.fill is not None:
            on = on + other.fill * (jnp.sum(x) - jnp.sum(g))
    return dense.scale * other.scale * on


def _dot_same_frame(u: _View, v: _View) -> Array:
    """``<u, v>`` for two operands sharing one compact frame."""
    if u.uniform and v.uniform:
        inner = jnp.asarray(u.n_struct, u.scale.dtype)
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

    Raises :class:`LazyContractionUnsupported` when the two storage forms have
    no common compact frame. It never densifies as a fallback.
    """
    if dtype is None:
        dtype = jnp.promote_types(
            jnp.promote_types(getattr(e, "dtype", jnp.float32),
                              getattr(a, "dtype", jnp.float32)), jnp.float32)
    ve, va = view_of(e, dtype), view_of(a, dtype)
    if ve.logical != va.logical:
        raise LazyContractionUnsupported(
            f"logical shapes differ: {ve.logical} vs {va.logical}")
    if ve.is_materialized:
        dot = _dot_dense_struct(ve, va)
    elif va.is_materialized:
        dot = _dot_dense_struct(va, ve)
    elif ve.key == va.key:
        dot = _dot_same_frame(ve, va)
    else:
        raise LazyContractionUnsupported(
            "neither operand is fully materialized and their structures "
            "differ, so they share no compact frame. Densifying one of them "
            "here would defeat the no-materialize contract (ticket "
            "dsnn-3qm.62 ruling (c)), so this raises instead. Extend this "
            "module rather than calling .dense().\n"
            f"  lhs: {ve.describe()}\n"
            f"  rhs: {va.describe()}")
    return dot, _norm2(ve), _norm2(va)
