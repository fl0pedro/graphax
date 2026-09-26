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
            arr = arr.reshape(atoms)
        return arr

    def _materialize(self, target=None):
        arr = self._atoms()
        seq = [t for s in self.slots for t in s]
        pos = {t: i for i, t in enumerate(self.order)}
        perm = [pos[t] for t in seq]
        if perm != list(range(len(perm))):
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
        keep = [i for i, s in enumerate(v.slots) if s]
        core = v._clone([v.slots[i] for i in keep])._materialize()
        out = jax.lax.broadcast_in_dim(core, shape, tuple(keep))
        return _View(out, one=self.one)

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


def _stored(v):
    """``(array, axis_of)`` for a view stored as an edge's ``val``: the
    non-unit slots follow the base's own order, the unit slots come last, and
    ``axis_of[k]`` is the stored axis of slot ``k``. The caller points its dims
    at these axes, so the store emits no transpose."""
    v = _as_view(v)
    arr, rank = v.natural()
    units = [i for i, s in enumerate(v.slots) if not s]
    if units:
        arr = arr.reshape(tuple(arr.shape) + (1,) * len(units))
    axis_of = {s: k for k, s in enumerate(rank + units)}
    return arr, [axis_of[k] for k in range(v.ndim)]
