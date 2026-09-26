"""A symbolic shape view: an array plus the reshapes and transposes still
pending on it (dsnn-dfw.250, .252).

The contraction and the join lay each edge out in their own frames and back.
Nearly every step of that only adds, drops or moves size-1 axes, or permutes
axes a consumer can read in any order. A view records the steps and emits them
only where values are consumed: at most one reshape, one transpose and one
reshape, none of them when the composition is the identity on the data.

A slot is the list of atoms it merges, in order; an atom is a factor of one
base axis, and ``order`` lists the atoms in the base's own layout. An empty
slot is a size-1 axis. ``one`` marks the placeholder of an operand that stores
nothing.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp


class _LazyProduct:
    """``x * y`` over the union of their axes, not emitted yet (dsnn-dfw.272).

    Axis ``k`` of the product is axis ``xa[k]`` of ``x`` and axis ``ya[k]`` of
    ``y``, or ``None`` where the operand does not carry it. ``emit`` writes the
    product with its axes in the order the reader asks for. When the reader
    puts the shared axes first, the product is the ``dot_general`` over them,
    and a transpose of the operands' own axes when the reader orders them
    otherwise. When it does not, a dot would have to move its batch axes
    behind the others, and the product is a multiply of the two operands
    broadcast in the reader's order, with no transpose after it."""

    __slots__ = ("x", "y", "xa", "ya", "shape", "dtype", "_out")

    def __init__(self, x, xa, y, ya, shape):
        self.x, self.y, self.xa, self.ya = x, y, list(xa), list(ya)
        self.shape = tuple(int(d) for d in shape)
        self.dtype = jnp.result_type(x, y)
        self._out = {}

    @property
    def ndim(self):
        return len(self.shape)

    def emit(self, perm):
        perm = tuple(int(p) for p in perm)
        if perm not in self._out:
            out = self._dot(perm)
            if out is None:
                shape = tuple(self.shape[p] for p in perm)
                out = jax.lax.mul(_side(self.x, self.xa, perm, shape),
                                  _side(self.y, self.ya, perm, shape))
            self._out[perm] = out
        return self._out[perm]

    def _dot(self, perm):
        both = [p for p in perm if self.xa[p] is not None and self.ya[p] is not None]
        k = len(both)
        if k == len(perm) or not self.x.ndim or not self.y.ndim or list(perm[:k]) != both:
            return None
        own_x = sorted((p for p in perm if self.ya[p] is None), key=lambda p: self.xa[p])
        own_y = sorted((p for p in perm if self.xa[p] is None), key=lambda p: self.ya[p])
        bx, by = tuple(self.xa[p] for p in both), tuple(self.ya[p] for p in both)
        if list(perm[k:]) == own_y + own_x:
            return jax.lax.dot_general(self.y, self.x, (((), ()), (by, bx)))
        out = jax.lax.dot_general(self.x, self.y, (((), ()), (bx, by)))
        made = both + own_x + own_y
        if list(perm) != made:
            out = jax.lax.transpose(out, [made.index(p) for p in perm])
        return out


def _side(arr, axes, perm, shape):
    # One operand laid along the product's axes in ``perm`` order.
    if not arr.ndim:
        return arr
    carried = [(pos, axes[p]) for pos, p in enumerate(perm) if axes[p] is not None]
    src = [a for _, a in carried]
    if len(src) != arr.ndim:
        raise ValueError(f"an operand of rank {arr.ndim} carries {len(src)} product axes")
    if src != list(range(len(src))):
        arr = jax.lax.transpose(arr, src)
    dims = tuple(pos for pos, _ in carried)
    if dims == tuple(range(len(shape))):
        return arr
    return jax.lax.broadcast_in_dim(arr, shape, dims)


def _force(base):
    return base.emit(range(base.ndim)) if isinstance(base, _LazyProduct) else base


class _View:
    __slots__ = ("base", "order", "extent", "slots", "one")

    def __init__(self, base, *, order=None, extent=None, slots=None, one=False):
        self.base = base
        if order is None:
            order, extent, slots = [], {}, []
            for k, n in enumerate(base.shape):
                n = int(n)
                if n == 1:
                    slots.append([])
                else:
                    order.append(k)
                    extent[k] = n
                    slots.append([k])
        self.order, self.extent, self.slots, self.one = order, extent, slots, one

    def _clone(self, slots, order=None, extent=None):
        return _View(
            self.base,
            order=list(self.order) if order is None else order,
            extent=dict(self.extent) if extent is None else extent,
            slots=slots,
            one=self.one,
        )

    @property
    def shape(self):
        return tuple(math.prod(self.extent[t] for t in s) for s in self.slots)

    @property
    def ndim(self):
        return len(self.slots)

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def dtype(self):
        return self.base.dtype

    def transpose(self, perm):
        perm = list(perm)
        if perm == list(range(len(self.slots))):
            return self
        return self._clone([self.slots[p] for p in perm])

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        shape = tuple(int(d) for d in shape)
        if shape == self.shape:
            return self
        if math.prod(shape) != self.size:
            raise ValueError(f"cannot reshape a view of shape {self.shape} to {shape}")
        seq = [t for s in self.slots for t in s]
        order, extent = list(self.order), dict(self.extent)
        nxt = max(extent, default=-1) + 1
        slots, i = [], 0
        for d in shape:
            if d == 1:
                slots.append([])
                continue
            run, group = 1, []
            while run < d:
                t = seq[i]
                s = extent[t]
                if d % (run * s) == 0:
                    group.append(t)
                    run *= s
                    i += 1
                elif d % run == 0 and s % (d // run) == 0:
                    # Split the atom at the boundary; the minor part is a new
                    # atom right after it in the base order.
                    d1 = d // run
                    t2 = nxt
                    nxt += 1
                    extent[t] = d1
                    extent[t2] = s // d1
                    order.insert(order.index(t) + 1, t2)
                    seq.insert(i + 1, t2)
                    group.append(t)
                    run *= d1
                    i += 1
                else:
                    # A regrouping the atoms cannot express: a real reshape of
                    # the materialized buffer.
                    return _View(self._materialize(shape), one=self.one)
            slots.append(group)
        return self._clone(slots, order, extent)

    def _atoms(self):
        arr = self.base
        atoms = tuple(self.extent[t] for t in self.order)
        if tuple(int(n) for n in arr.shape) != atoms:
            arr = _force(arr).reshape(atoms)
        return arr

    def _materialize(self, target=None):
        arr = self._atoms()
        seq = [t for s in self.slots for t in s]
        pos = {t: i for i, t in enumerate(self.order)}
        perm = [pos[t] for t in seq]
        if isinstance(arr, _LazyProduct):
            arr = arr.emit(perm)
        elif perm != list(range(len(perm))):
            arr = arr.transpose(perm)
        target = self.shape if target is None else tuple(int(d) for d in target)
        if tuple(int(n) for n in arr.shape) != target:
            arr = arr.reshape(target)
        return arr

    def materialize(self):
        arr = self._materialize()
        self.base = arr
        self.order = [t for s in self.slots for t in s]
        return arr

    def natural(self):
        """``(array, rank)``: the non-unit slots as axes in the base's own atom
        order, so no transpose is emitted; ``rank[k]`` is the slot at axis
        ``k``. Falls back to the slot order when a slot's atoms are not one
        contiguous run of the base."""
        full = [i for i, s in enumerate(self.slots) if s]
        if isinstance(self.base, _LazyProduct):
            return self._clone([self.slots[j] for j in full])._materialize(), full
        pos = {t: i for i, t in enumerate(self.order)}
        first = {}
        for i in full:
            idx = [pos[t] for t in self.slots[i]]
            if idx != list(range(idx[0], idx[0] + len(idx))):
                arr = self._clone([self.slots[j] for j in full])._materialize()
                return arr, full
            first[i] = idx[0]
        rank = sorted(full, key=first.__getitem__)
        shape = tuple(math.prod(self.extent[t] for t in self.slots[i]) for i in rank)
        arr = self.base
        if tuple(int(n) for n in arr.shape) != shape:
            arr = arr.reshape(shape)
        return arr, rank

    def broadcast_to(self, shape):
        shape = tuple(int(d) for d in shape)
        v = self
        if len(shape) > v.ndim:
            v = v.insert_units(len(shape), range(len(shape) - v.ndim))
        if v.shape == shape:
            return v
        dims = v._base_dims()
        if dims is not None:
            out = jax.lax.broadcast_in_dim(v.base, shape, dims)
        else:
            keep = [i for i, s in enumerate(v.slots) if s]
            core = v._clone([v.slots[i] for i in keep])._materialize()
            out = jax.lax.broadcast_in_dim(core, shape, tuple(keep))
        return _View(out, one=self.one)

    def base_axes(self):
        """``{base axis: slot}`` when every non-unit slot is one whole base
        axis and the slots keep the base's order, else ``None``: then an op
        that maps axes can read the base as it is."""
        if isinstance(self.base, _LazyProduct):
            return None
        base = tuple(int(n) for n in self.base.shape)
        at = {}
        for p, s in enumerate(self.slots):
            if s:
                if len(s) != 1 or s[0] >= len(base) or self.extent[s[0]] != base[s[0]]:
                    return None
                at[s[0]] = p
        if [at[k] for k in sorted(at)] != sorted(at.values()):
            return None
        return at

    def _base_dims(self):
        """The slot of every base axis, strictly increasing, when each size-1
        base axis can take a slot between its neighbours: then
        ``broadcast_in_dim`` reads the base with no reshape or transpose."""
        base = tuple(int(n) for n in self.base.shape)
        at = self.base_axes()
        if at is None:
            return None
        fixed = sorted(at.values())
        dims, last = [], -1
        for k, n in enumerate(base):
            if n != 1:
                p = at[k]
            else:
                nxt = min((at[j] for j in at if j > k), default=len(self.slots))
                p = next((q for q in range(last + 1, nxt) if q not in fixed), None)
                if p is None:
                    return None
            if p <= last:
                return None
            dims.append(p)
            last = p
        return tuple(dims)

    def sum_keepdims(self, axes):
        axes = tuple(sorted(set(int(a) for a in axes)))
        out = _View(jnp.sum(self.materialize(), axis=axes))
        return out.insert_units(self.ndim, axes)

    def __getitem__(self, idx):
        return _View(self.materialize()[idx])

    def drop_units(self, labels):
        keep = [i for i, s in enumerate(self.slots) if s]
        return self._clone([self.slots[i] for i in keep]), [labels[i] for i in keep]

    def drop_slots(self, positions):
        positions = set(positions)
        for p in positions:
            if self.slots[p]:
                raise ValueError(f"slot {p} of a view of shape {self.shape} is not a unit axis")
        return self._clone([s for i, s in enumerate(self.slots) if i not in positions])

    def insert_units(self, n_total, positions):
        positions = set(positions)
        rest = iter(self.slots)
        return self._clone([[] if i in positions else next(rest) for i in range(n_total)])


def _as_view(x):
    return x if isinstance(x, _View) else _View(x)


def _mat(x):
    return x.materialize() if isinstance(x, _View) else x


def _bcast(x, shape):
    if isinstance(x, _View):
        return x.broadcast_to(shape)
    return jnp.broadcast_to(x, tuple(shape))
