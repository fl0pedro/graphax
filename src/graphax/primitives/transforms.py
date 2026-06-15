import copy
from dataclasses import replace
from functools import partial
from typing import Callable

import jax.lax as lax
import jax._src.lax.lax as lax_src
import jax.numpy as jnp

from ..sparse.tensor import (
    DenseIndex,
    DiagonalIndex,
    SparseTensor,
    _materialize_indexes,
    _swap_back_axes,
)
from .base import (
    elemental_only_rules,
    elemental_rules,
    multi_output_elemental_only_rules,
)

Transform = Callable[[SparseTensor], SparseTensor]


class JacobianTransform:
    transform: Transform
    inverse_transform: Transform

    def __init__(
        self, transform: Transform, inverse_transform: Transform = None
    ) -> None:
        self.transform = transform
        self.inverse_transform = inverse_transform

    def __repr__(self) -> str:
        return (
            f"JacobianTransform(transform={self.transform}, "
            f"inverse_transform={self.inverse_transform})"
        )

    def apply(self, tensor: SparseTensor) -> SparseTensor:
        if self.transform is None:
            raise NotImplementedError("Transform not implemented!")
        return self.transform(tensor)

    def apply_inverse(self, tensor: SparseTensor) -> SparseTensor:
        if self.inverse_transform is None:
            raise NotImplementedError("Inverse transform not implemented!")
        return self.inverse_transform(tensor)


def _inverse_permutation(permutation):
    inverse = [0] * len(permutation)
    for i, p in enumerate(permutation):
        inverse[p] = i
    return inverse


def _is_scalar_identity_post(post) -> bool:
    """A post edge that is the bare identity seed: no out/primal dims and no
    stored ``val`` (the structure is the all-ones identity up to
    ``scalar_mult``). This is what an inverse transform receives in forward
    ('fwd') elimination order when the identity output-seed is drained through a
    transform-only edge BEFORE any val-carrying edge has shaped it. In that
    state ``post.dense()`` is rank-0, so the inverse rules below cannot run
    their scatter/embedding logic — they must first re-expand the seed into the
    explicit identity Jacobian over the op's OUTPUT shape."""
    return (
        not post.out_dims
        and not post.primal_dims
        and post.val is None
    )


def _identity_post_over(post, out_shape, dtype=jnp.float32):
    """Re-expand a bare scalar-identity ``post`` (see ``_is_scalar_identity_post``)
    into the explicit dense identity Jacobian ``s * I`` over ``out_shape`` —
    ``out_dims`` and ``primal_dims`` both equal ``out_shape`` and ``val`` is the
    reshaped identity matrix scaled by ``post.scalar_mult``. The resulting tensor
    feeds the regular (non-scalar) branch of each inverse rule unchanged, so the
    'fwd' path reconstructs exactly the same op Jacobian the 'rev' path builds.

    ``scalar_mult`` is folded into ``val``; the synthesized tensor keeps the
    default ``scalar_mult`` so the downstream rule (which copies ``scalar_mult``
    through verbatim) does not double-apply it."""
    out_shape = tuple(int(s) for s in out_shape)
    n = 1
    for s in out_shape:
        n *= s
    eye = jnp.eye(n, dtype=dtype)
    scalar = post.scalar_mult
    if scalar is not None:
        eye = eye * scalar
    val = eye.reshape(out_shape + out_shape)
    counter = 0
    new_out_dims = []
    for s in out_shape:
        new_out_dims.append(DenseIndex(counter, s, counter))
        counter += 1
    new_primal_dims = []
    for s in out_shape:
        new_primal_dims.append(DenseIndex(counter, s, counter))
        counter += 1
    return SparseTensor(
        new_out_dims,
        new_primal_dims,
        val,
        fill_value=post.fill_value,
    )


# ---------- transpose ----------


def _transpose_elementals(primals, val_out, **params):
    permutation = params["permutation"]

    def transpose_transform(pre):
        new_out_dims = []
        new_primal_dims = list(pre.primal_dims)
        counter = 0
        l = len(pre.out_dims)

        for p in permutation:
            new_out_dims.append(pre.out_dims[p])
            new_out_dims[-1] = replace(new_out_dims[-1], id=counter)
            if new_out_dims[-1].is_sparse:
                other_id = new_out_dims[-1].other_id
                new_primal_dims[other_id - l] = replace(
                    new_primal_dims[other_id - l], other_id=counter
                )
            counter += 1

        return _swap_back_axes(
            SparseTensor(
                new_out_dims,
                new_primal_dims,
                pre.val,
                scalar_mult=pre.scalar_mult,
                fill_value=pre.fill_value,
            )
        )

    def inverse_transpose_transform(post):
        new_out_dims = list(post.out_dims)
        new_primal_dims = []
        counter = len(post.out_dims)

        # This implementation is faulty!
        inv_permutation = _inverse_permutation(permutation)
        for p in inv_permutation:
            new_primal_dims.append(post.primal_dims[p])
            new_primal_dims[-1] = replace(new_primal_dims[-1], id=counter)
            if new_primal_dims[-1].is_sparse:
                other_id = new_primal_dims[-1].other_id
                new_out_dims[other_id] = replace(
                    new_out_dims[other_id], other_id=counter
                )
            counter += 1

        return _swap_back_axes(
            SparseTensor(
                new_out_dims,
                new_primal_dims,
                post.val,
                scalar_mult=post.scalar_mult,
                fill_value=post.fill_value,
            )
        )

    transform = JacobianTransform(transpose_transform, inverse_transpose_transform)
    return [SparseTensor([], [], None, pre_transforms=[transform])]


# Should work for high-dimensional stuff
def transpose_elemental_rule(primals, **params):
    val_out = lax.transpose_p.bind(*primals, **params)
    return val_out, _transpose_elementals(primals, val_out, **params)


def transpose_elemental_only(primal_out, primals, **params):
    return _transpose_elementals(primals, primal_out, **params)


elemental_rules[lax.transpose_p] = transpose_elemental_rule
elemental_only_rules[lax.transpose_p] = transpose_elemental_only


# ---------- reshape ----------


def _reshape_elementals(primals, val_out, **params):
    # TODO: dimensional collapse is not covered here!
    # Implement sparsity-aware version for significant speedup!

    def reshape_transform(pre):
        # NOTE array is not correctly materialized sometimes!
        full_val = pre.dense()
        new_shape = []
        new_out_dims = []
        new_primal_dims = []
        counter = 0

        for s in val_out.shape:
            new_out_dims.append(DenseIndex(counter, s, counter))
            new_shape.append(s)
            counter += 1

        for d in pre.primal_dims:
            new_primal_dims.append(DenseIndex(counter, d.size, counter))
            new_shape.append(d.size)
            counter += 1

        full_val = full_val.reshape(new_shape)
        return SparseTensor(new_out_dims, new_primal_dims, full_val)

    def inverse_reshape_transform(post):
        full_val = post.dense()
        new_shape = []
        new_out_dims = []
        new_primal_dims = []
        counter = 0
        for d in post.out_dims:
            new_out_dims.append(DenseIndex(counter, d.size, counter))
            new_shape.append(d.size)
            counter += 1
        for s in primals[0].shape:
            new_primal_dims.append(DenseIndex(counter, s, counter))
            new_shape.append(s)
            counter += 1
        full_val = full_val.reshape(new_shape)
        return SparseTensor(new_out_dims, new_primal_dims, full_val)

    transform = JacobianTransform(reshape_transform, inverse_reshape_transform)
    return [SparseTensor([], [], None, pre_transforms=[transform])]


def reshape_elemental_rule(primals, **params):
    val_out = lax.reshape_p.bind(*primals, **params)
    return val_out, _reshape_elementals(primals, val_out, **params)


def reshape_elemental_only(primal_out, primals, **params):
    return _reshape_elementals(primals, primal_out, **params)


elemental_rules[lax.reshape_p] = reshape_elemental_rule
elemental_only_rules[lax.reshape_p] = reshape_elemental_only


# ---------- slice ----------


def _slice_elementals(primals, val_out, **params):
    # The slice primitive is written in such a way that it just densifies the
    # Jacobian and then slices it. This is not efficient and there might be ways
    # to make this more efficient by checking if sparse dimensions are untouched
    # how this changes the Jacobian.

    def slice_transform(pre):
        start_indices = list(params["start_indices"])
        limit_indices = list(params["limit_indices"])
        strides = params.get("strides")
        strides = [1] * len(start_indices) if strides is None else list(strides)
        full_val = pre.dense()
        new_out_dims = []
        new_primal_dims = []
        counter = 0

        for s in val_out.shape:
            new_out_dims.append(DenseIndex(counter, s, counter))
            counter += 1

        for d in pre.primal_dims:
            new_primal_dims.append(DenseIndex(counter, d.size, counter))
            start_indices.append(0)
            limit_indices.append(d.size)
            strides.append(1)  # primal dims are not sliced
            counter += 1

        new_val = lax.slice(full_val, start_indices, limit_indices, strides)
        return SparseTensor(new_out_dims, new_primal_dims, new_val)

    def inverse_slice_transform(post):
        start_indices = list(params["start_indices"])
        strides = params.get("strides")
        strides = [1] * len(start_indices) if strides is None else list(strides)
        # In 'fwd' elimination order this inverse can be drained against the bare
        # identity output-seed (no out/primal dims, val=None). Re-expand that seed
        # into the explicit identity Jacobian over the slice's OUTPUT shape so the
        # embedding below — which assumes post.dense() is (out..., slice_out...)
        # with the slice-output dims trailing — runs exactly as in 'rev' order.
        if _is_scalar_identity_post(post):
            post = _identity_post_over(post, val_out.shape, val_out.dtype)
        full_val = post.dense()
        new_out_dims = []
        new_primal_dims = []
        counter = 0

        # The transpose of a (possibly strided) slice is a pad: each slice-output
        # element gradient lands at input position ``start + k*stride``, i.e. low
        # pad = start and interior pad = stride-1. (The old version used a
        # contiguous scatter that ignored ``strides`` AND hardcoded the input
        # dtype via jnp.zeros, mis-placing strided slices and crashing on f64.)
        pad_config = []
        for d in post.out_dims:
            new_out_dims.append(DenseIndex(counter, d.size, counter))
            pad_config.append((0, 0, 0))
            counter += 1

        in_shape = primals[0].shape
        slice_out_sizes = full_val.shape[len(post.out_dims):]
        for ax, L in enumerate(in_shape):
            new_primal_dims.append(DenseIndex(counter, L, counter))
            n = slice_out_sizes[ax]
            st, stride = start_indices[ax], strides[ax]
            high = L - st - (n - 1) * stride - 1
            pad_config.append((st, high, stride - 1))
            counter += 1

        new_val = lax.pad(
            full_val, jnp.array(0.0, dtype=full_val.dtype), pad_config
        )
        return SparseTensor(new_out_dims, new_primal_dims, new_val)

    transform = JacobianTransform(slice_transform, inverse_slice_transform)
    return [SparseTensor([], [], None, pre_transforms=[transform])]


def slice_elemental_rule(primals, **params):
    val_out = lax.slice_p.bind(*primals, **params)
    return val_out, _slice_elementals(primals, val_out, **params)


def slice_elemental_only(primal_out, primals, **params):
    return _slice_elementals(primals, primal_out, **params)


elemental_rules[lax.slice_p] = slice_elemental_rule
elemental_only_rules[lax.slice_p] = slice_elemental_only


# ---------- broadcast_in_dim ----------


def _broadcast_elementals(primals, val_out, **params):
    # Materialize the broadcast Jacobian as a SparseTensor instead of a deferred
    # JacobianTransform. The Jacobian is the Kronecker delta on matched axes and
    # constant (1.0) along broadcasted axes, so this costs only dimension
    # metadata — no tensor data is allocated.
    #
    # The previous transform-based approach worked in forward mode (transforms
    # got consumed during composition with a val-carrying pre), but broke in
    # reverse mode: when a transform-only edge was composed as `pre` with a
    # val-carrying `post`, the transform was appended to the resulting edge's
    # pre_transforms. Later attempts to apply it to that composed edge mutated
    # out_dims (the transform's design), but the broadcast actually needed to
    # be resolved on the primal side of the composed edge.
    dims = params["broadcast_dimensions"]
    shape = params["shape"]
    primal = primals[0]
    primal_shape = primal.shape if hasattr(primal, "shape") else ()

    l = len(shape)
    n = len(primal_shape)
    # dims[j] is the output axis where primal axis j is placed; any remaining
    # output axes are new broadcast axes.
    # Pair an output axis with its primal axis only when their sizes match
    # (a Kronecker-delta on that axis). For stretch broadcasts (primal size 1
    # → output size N) the contribution is a constant 1 across the entire
    # axis pair, not a diagonal — represent both ends as DenseIndex so the
    # sparse-pair topology check (size match) holds.
    new_out_dims = []
    for i in range(l):
        if i in dims:
            primal_idx = dims.index(i)
            if primal_shape[primal_idx] == shape[i]:
                new_out_dims.append(DiagonalIndex(i, shape[i], None, l + primal_idx))
            else:
                new_out_dims.append(DenseIndex(i, shape[i], None))
        else:
            new_out_dims.append(DenseIndex(i, shape[i], None))
    new_primal_dims = []
    for j in range(n):
        out_pos = dims[j]
        if primal_shape[j] == shape[out_pos]:
            new_primal_dims.append(DiagonalIndex(l + j, primal_shape[j], None, out_pos))
        else:
            new_primal_dims.append(DenseIndex(l + j, primal_shape[j], None))
    return [SparseTensor(new_out_dims, new_primal_dims, 1.0)]


def broadcast_elemental_rule(primals, **params):
    val_out = lax.broadcast_in_dim_p.bind(*primals, **params)
    return val_out, _broadcast_elementals(primals, val_out, **params)


def broadcast_elemental_only(primal_out, primals, **params):
    return _broadcast_elementals(primals, primal_out, **params)


elemental_rules[lax.broadcast_in_dim_p] = broadcast_elemental_rule
elemental_only_rules[lax.broadcast_in_dim_p] = broadcast_elemental_only


# ---------- squeeze ----------


def _squeeze_elementals(primals, val_out, **params):
    # NOTE: squeeze is basically just the inverse operation to broadcast_in_dim
    # since it just adds a DenseIndex of size 1

    def squeeze_transform(pre):
        dims = sorted(params["dimensions"])
        new_out_dims = list(copy.deepcopy(pre.out_dims))
        new_primal_dims = list(copy.deepcopy(pre.primal_dims))
        squeeze_dims = []
        counter = 0

        for id in dims:
            idx = [j for j, d in enumerate(new_out_dims) if d.id == id][0]
            axis = new_out_dims[idx].axis
            squeeze_dims.append(axis)

            if new_out_dims[idx].is_sparse:

                def _check(d, id):
                    if d.is_sparse:
                        return d.other_id == id
                    else:
                        return False

                other_idx = [j for j, d in enumerate(new_primal_dims) if _check(d, id)][
                    0
                ]
                other_dim = new_primal_dims[other_idx]
                new_primal_dims[other_idx] = DenseIndex(
                    other_dim.id, other_dim.size, None
                )

            del new_out_dims[idx]
            counter += 1

        out_ids = [d.id for d in new_out_dims]
        primal_ids = [d.id for d in new_primal_dims]
        new_val_axes = [d.axis for d in new_out_dims if d.axis is not None]
        new_val_axes += [
            d.axis
            for d in new_primal_dims
            if not d.is_sparse and d.axis is not None
        ]

        for i, d in enumerate(new_out_dims):
            updates = {"id": out_ids.index(d.id)}
            if d.axis is not None:
                updates["axis"] = new_val_axes.index(d.axis)
            if d.is_sparse:
                updates["other_id"] = len(new_out_dims) + primal_ids.index(d.other_id)
            new_out_dims[i] = replace(d, **updates)

        for i, d in enumerate(new_primal_dims):
            updates = {"id": len(new_out_dims) + primal_ids.index(d.id)}
            if d.axis is not None:
                updates["axis"] = new_val_axes.index(d.axis)
            if d.is_sparse:
                updates["other_id"] = out_ids.index(d.other_id)
            new_primal_dims[i] = replace(d, **updates)

        squeeze_dims = [d for d in squeeze_dims if d is not None]
        if len(squeeze_dims) > 0:
            new_val = jnp.squeeze(pre.val, axis=squeeze_dims)
        else:
            new_val = pre.val
        return SparseTensor(
            new_out_dims,
            new_primal_dims,
            new_val,
            scalar_mult=pre.scalar_mult,
            fill_value=pre.fill_value,
        )

    def inverse_squeeze_transform(post):
        # Re-insert each squeezed (size-1) primal axis at its ORIGINAL input
        # position. Insert in ascending order so earlier insertions don't shift
        # later positions. Each inserted dim gets a fresh size-1 val axis at the
        # correct VAL position (after the out-dense and preceding primal-dense
        # axes) — NOT at the input-axis index, which was the old bug. Ids are
        # renumbered contiguously at the end (the old code collided ids and
        # expanded val at the wrong axis).
        new_dims = sorted(params["dimensions"])
        out_dims = list(copy.deepcopy(post.out_dims))
        primal_dims = list(copy.deepcopy(post.primal_dims))
        num_out = len(out_dims)
        val = post.val

        # Each re-inserted size-1 primal axis gets a FRESH val axis appended at
        # the end; _swap_back_axes then permutes val into canonical dim order.
        # (Computing the exact insertion axis by hand is unsound because diagonal
        # pairs share a val axis, so a dim-count overshoots val.ndim.)
        for dim in new_dims:
            if val is not None:
                new_ax = val.ndim
                val = jnp.expand_dims(val, axis=new_ax)
            else:
                new_ax = None
            primal_dims.insert(dim, DenseIndex(-1, 1, new_ax))  # id fixed below

        old_to_new = {d.id: num_out + pos for pos, d in enumerate(primal_dims) if d.id != -1}
        new_primal = [replace(d, id=num_out + pos) for pos, d in enumerate(primal_dims)]
        new_out = [
            replace(d, other_id=old_to_new[d.other_id]) if d.is_sparse else d
            for d in out_dims
        ]
        return _swap_back_axes(SparseTensor(
            new_out, new_primal, val,
            scalar_mult=post.scalar_mult, fill_value=post.fill_value,
        ))

    transform = JacobianTransform(squeeze_transform, inverse_squeeze_transform)
    return [SparseTensor([], [], None, pre_transforms=[transform])]


def squeeze_elemental_rule(primals, **params):
    val_out = lax.squeeze_p.bind(*primals, **params)
    return val_out, _squeeze_elementals(primals, val_out, **params)


def squeeze_elemental_only(primal_out, primals, **params):
    return _squeeze_elementals(primals, primal_out, **params)


elemental_rules[lax.squeeze_p] = squeeze_elemental_rule
elemental_only_rules[lax.squeeze_p] = squeeze_elemental_only


# ---------- concatenate ----------


def _concatenate_elementals(primals, val_out, **params):
    # This gradient transformation is designed to take an post edge and
    # decompose it into the pre edges. This is done by densifying the post along
    # the respective axes and then use jnp.split to split the tensor.
    # TODO DynamicJaxprTracer is now a unhashable type, so we can no longer use
    # it as a key in the dict. We need to find another way of doing this.
    dim = params["dimension"]

    offset = primals[0].shape[dim]
    slices = {0: [0, offset]}
    for i, val in enumerate(primals[1:], start=1):
        slices[i] = [offset, offset + val.shape[dim]]
        offset += val.shape[dim]

    def concatenate_transform(primal_idx, pre):
        new_out_dims = list(copy.deepcopy(pre.out_dims))
        new_primal_dims = list(copy.deepcopy(pre.primal_dims))
        l = len(pre.out_dims)

        d = new_out_dims[dim]
        dim_id = d.id
        idx, _idx = slices[primal_idx]

        if not d.is_sparse:
            if d.axis is not None:
                lshape = list(pre.val.shape)
                rshape = list(pre.val.shape)
                lshape[d.axis] = idx
                rshape[d.axis] = val_out.shape[dim] - _idx
                lcat_zeros = jnp.zeros(lshape)
                rcat_zeros = jnp.zeros(rshape)

                new_val = jnp.concatenate(
                    [lcat_zeros, pre.val, rcat_zeros], axis=d.axis
                )

                new_out_dims[dim] = replace(
                    new_out_dims[dim], size=new_val.shape[d.axis]
                )
            else:
                # axis=None: this output dimension is a Kronecker factor not stored in val.
                # Materialize it: broadcast pre.val to the primal's slice size (zero-copy),
                # then pad with zeros in a single lax.pad pass.
                val_size = _idx - idx
                new_axis = sum(1 for dd in new_out_dims[:dim] if dd.axis is not None)

                new_val = jnp.expand_dims(pre.val, axis=new_axis)
                new_val = jnp.broadcast_to(
                    new_val,
                    (*pre.val.shape[:new_axis], val_size, *pre.val.shape[new_axis:]),
                )

                pad_config = [(0, 0, 0)] * new_val.ndim
                pad_config[new_axis] = (idx, val_out.shape[dim] - _idx, 0)
                new_val = lax.pad(
                    new_val, jnp.zeros((), dtype=new_val.dtype), pad_config
                )

                # Inserting a new axis shifts all subsequent val_axes up by 1
                for j in range(dim + 1, len(new_out_dims)):
                    _dim = new_out_dims[j]
                    if _dim.axis is not None:
                        new_out_dims[j] = replace(_dim, axis=_dim.axis + 1)
                for j, _dim in enumerate(new_primal_dims):
                    if not _dim.is_sparse and _dim.axis is not None:
                        new_primal_dims[j] = replace(_dim, axis=_dim.axis + 1)

                new_out_dims[dim] = replace(
                    new_out_dims[dim], axis=new_axis, size=val_out.shape[dim]
                )
        else:
            other_id = d.other_id
            if d.axis is not None:
                _d = new_primal_dims[other_id - l]

                # Calculate the new axis of the primal dimension
                axis = sum(1 for dd in new_out_dims if dd.axis is not None)
                axis += sum(
                    1
                    for dd in new_primal_dims[: other_id - l]
                    if dd.axis is not None and not dd.is_sparse
                )

                # Update the axis of all following dimensions
                for j in range(dim + 1, len(new_primal_dims)):
                    _dim = new_primal_dims[j]
                    if not _dim.is_sparse and _dim.axis is not None:
                        new_primal_dims[j] = replace(_dim, axis=_dim.axis + 1)

                # Materialize the sparse dimensions related to the concatenation dimension
                new_val = _materialize_indexes(pre, [d.id])

                sub_iota = jnp.eye(d.size, dtype=new_val.dtype)

                shape = [1 for _ in range(pre.val.ndim)]
                shape[_d.axis] = _d.size
                shape.insert(axis, d.size)
                sub_iota = sub_iota.reshape(shape)

                new_val = new_val * sub_iota

                # Make zeros for insertion (match new_val's dtype: the scatter below
                # requires identical operand/update dtypes — was hardcoded float32,
                # crashing under jax_enable_x64 / non-float32 Jacobian vals).
                _size = val_out.shape[dim]
                _shape = list(new_val.shape)
                _shape[d.axis] = _size
                _shape[axis] = d.size
                zeros = jnp.zeros(_shape, dtype=new_val.dtype)

                # scatter_indices: where in `zeros` to place `new_val`
                scatter_indices = [0 for _ in _shape]
                scatter_indices[d.axis] = idx
                scatter_indices[axis] = 0

                update_window_dims = tuple(range(len(_shape)))
                scatter_dims_to_operand_dims = tuple(range(len(_shape)))

                scatter_dims = lax.ScatterDimensionNumbers(
                    update_window_dims, (), scatter_dims_to_operand_dims
                )
                new_val = lax.scatter(
                    zeros,
                    jnp.array(scatter_indices),
                    new_val,
                    scatter_dims,
                    indices_are_sorted=True,
                    unique_indices=True,
                )

                new_out_dims[dim_id] = DenseIndex(dim_id, val_out.shape[dim], d.axis)
                new_primal_dims[other_id - l] = DenseIndex(other_id, d.size, axis)
            else:
                _d = new_primal_dims[other_id - l]
                _size = val_out.shape[dim]

                # Calculate the new axis of the out dimension
                out_axis = sum(1 for dd in new_out_dims[:dim] if dd.axis is not None)

                # Calculate the new axis of the primal dimension
                primal_axis = sum(1 for dd in new_out_dims if dd.axis is not None)
                primal_axis += sum(
                    1
                    for dd in new_primal_dims[: other_id - l]
                    if dd.axis is not None and not dd.is_sparse
                )
                primal_axis = max(1, primal_axis)

                # Update the axis of all following dimensions
                for j in range(dim + 1, len(new_primal_dims)):
                    _dim = new_primal_dims[j]
                    if not _dim.is_sparse and _dim.axis is not None:
                        new_primal_dims[j] = replace(_dim, axis=_dim.axis + 1)

                # Materialize the sparse dimensions related to the concatenation dimension
                if pre.val.shape != ():
                    new_val = _materialize_indexes(pre, [d.id, d.other_id])
                else:
                    new_val = pre.val

                sub_iota = jnp.eye(d.size, dtype=new_val.dtype)

                shape = [1 for _ in range(pre.val.ndim)]
                shape.insert(out_axis, _d.size)
                shape.insert(primal_axis, d.size)

                new_val = new_val * sub_iota

                # Make zeros for insertion (dtype must match new_val for the scatter)
                _shape = list(pre.val.shape)
                _shape.insert(out_axis, _size)
                _shape.insert(primal_axis, _d.size)
                zeros = jnp.zeros(_shape, dtype=new_val.dtype)

                scatter_dims = lax.ScatterDimensionNumbers(
                    (out_axis, primal_axis), (), (out_axis, primal_axis)
                )
                new_val = lax.scatter(
                    zeros,
                    jnp.array([idx, 0]),
                    new_val,
                    scatter_dims,
                    indices_are_sorted=True,
                    unique_indices=True,
                )

                new_out_dims[dim_id] = DenseIndex(dim_id, val_out.shape[dim], out_axis)
                new_primal_dims[other_id - l] = DenseIndex(
                    other_id, d.size, primal_axis
                )

        return SparseTensor(
            new_out_dims,
            new_primal_dims,
            new_val,
            scalar_mult=pre.scalar_mult,
            fill_value=pre.fill_value,
        )

    def inverse_concatenate_transform(primal_idx, post):
        # In 'fwd' elimination order this inverse can be drained against the bare
        # identity output-seed (no out/primal dims, val=None). The seed has lost
        # the embedding structure (which concat slot maps to which output rows),
        # so re-expand it into the explicit identity Jacobian over the concat
        # OUTPUT shape. Then post.primal_dims[dim] is the concat-axis dimension and
        # the regular slicing branch extracts this slot's embedding block exactly
        # as in 'rev' order, instead of returning the structureless scalar.
        if _is_scalar_identity_post(post):
            post = _identity_post_over(post, val_out.shape, val_out.dtype)
        new_out_dims = list(copy.deepcopy(post.out_dims))
        new_primal_dims = list(copy.deepcopy(post.primal_dims))

        d = None
        if len(new_primal_dims) > 0:
            d = new_primal_dims[dim]
        if d is None:
            # post is a pure transform with no primal dims; nothing to slice.
            new_val = post.val
        elif not d.is_sparse:
            if d.axis is not None:
                new_val = lax.slice_in_dim(post.val, *slices[primal_idx], axis=d.axis)
                new_primal_dims[dim] = replace(d, size=new_val.shape[d.axis])
            else:
                # axis=None: the primal dimension is a Kronecker factor not stored in val.
                # There is no axis to slice — just narrow the size to this primal's contribution.
                new_primal_dims[dim] = replace(
                    d, size=slices[primal_idx][1] - slices[primal_idx][0]
                )
                new_val = post.val
        else:
            _d = new_out_dims[d.other_id]
            if d.axis is not None:
                new_out_dims[d.other_id] = DenseIndex(_d.id, _d.size, _d.axis)
                size = slices[primal_idx][1] - slices[primal_idx][0]

                # Calculate the new axis of the primal dimension
                axis = sum(1 for dd in new_out_dims if dd.axis is not None)
                axis += sum(
                    1
                    for dd in new_primal_dims[:dim]
                    if dd.axis is not None and not dd.is_sparse
                )
                new_primal_dims[dim] = DenseIndex(_d.other_id, size, axis)

                # Update the axis of all following dimensions
                for j in range(dim + 1, len(new_primal_dims)):
                    _dim = new_primal_dims[j]
                    if not _dim.is_sparse and _dim.axis is not None:
                        new_primal_dims[j] = replace(_dim, axis=_dim.axis + 1)

                # Materialize the sparse dimensions related to the concatenation dimension
                new_val = _materialize_indexes(post, [d.id])

                sub_iota = jnp.eye(d.size, dtype=new_val.dtype)

                shape = [1 for _ in range(post.val.ndim)]
                shape[_d.axis] = _d.size
                shape.insert(axis, d.size)
                sub_iota = sub_iota.reshape(shape)

                new_val = new_val * sub_iota

                new_val = lax.slice_in_dim(new_val, *slices[primal_idx], axis=axis)
                # d and _d were already replaced in the lists above; the prior
                # d.size/_d.size mutations targeted orphaned objects (no-op).
            else:
                # d is DiagonalIndex with axis=None:
                # Both d and its partner _d are implicit Kronecker factors not stored
                # in val. Slicing the primal axis at [s, e] yields columns [s:e] of
                # the implicit identity, which in general is not a square Kronecker
                # block (e.g. for a single-element slot of a multi-element output).
                # We must materialize: build the full identity slice as new val and
                # convert both indices to DenseIndex.
                s, e = slices[primal_idx]
                size = e - s

                # Position to insert the new out axis: among existing val-axis-
                # bearing dims, before the partner _d's position in out_dims.
                out_axis = sum(
                    1 for dd in new_out_dims[: d.other_id] if dd.axis is not None
                )
                # Position to insert the new primal axis: after all out val-axes
                # (including the one we are inserting) plus DenseIndex primal val-axes
                # before our position.
                primal_axis = sum(1 for dd in new_out_dims if dd.axis is not None) + 1
                primal_axis += sum(
                    1
                    for dd in new_primal_dims[:dim]
                    if not dd.is_sparse and dd.axis is not None
                )

                # Shift val_axes of all existing val-axis-bearing dims for the two
                # axis insertions (out then primal).
                def _shifted_axis(axis):
                    if axis is None:
                        return None
                    if axis >= out_axis:
                        axis += 1
                    if axis >= primal_axis:
                        axis += 1
                    return axis

                for j, _dim in enumerate(new_out_dims):
                    if _dim.axis is not None:
                        new_out_dims[j] = replace(_dim, axis=_shifted_axis(_dim.axis))
                for j, _dim in enumerate(new_primal_dims):
                    if not _dim.is_sparse and _dim.axis is not None:
                        new_primal_dims[j] = replace(
                            _dim, axis=_shifted_axis(_dim.axis)
                        )

                # Build the column slice of identity: shape (_d.size, size).
                _dt = post.val.dtype if post.val is not None else jnp.float32
                sub_iota = jnp.eye(_d.size, dtype=_dt)
                sub_iota = lax.slice_in_dim(sub_iota, s, e, axis=1)

                base_val = (
                    post.val
                    if post.val is not None
                    else jnp.array(1.0, dtype=_dt)
                )
                new_val = jnp.expand_dims(base_val, axis=out_axis)
                new_val = jnp.expand_dims(new_val, axis=primal_axis)

                iota_shape = [1] * new_val.ndim
                iota_shape[out_axis] = _d.size
                iota_shape[primal_axis] = size
                new_val = new_val * sub_iota.reshape(iota_shape)

                new_out_dims[d.other_id] = DenseIndex(_d.id, _d.size, out_axis)
                new_primal_dims[dim] = DenseIndex(d.id, size, primal_axis)
        return SparseTensor(
            new_out_dims,
            new_primal_dims,
            new_val,
            scalar_mult=post.scalar_mult,
            fill_value=post.fill_value,
        )

    # Group primal slot indices by primal identity. When the same primal feeds
    # multiple slots of one concatenate (e.g. concat([pl, pl, pl])), the
    # graph stores a single edge per (invar, outvar) pair, so the elemental
    # for that invar must represent the sum of all slot contributions.
    groups = {}
    for i, p in enumerate(primals):
        groups.setdefault(id(p), []).append(i)

    def _make_elemental(slot_indices):
        if len(slot_indices) == 1:
            (i,) = slot_indices
            transform = JacobianTransform(
                partial(concatenate_transform, i),
                partial(inverse_concatenate_transform, i),
            )
        else:

            def combined_fwd(pre, _slots=tuple(slot_indices)):
                acc = None
                for i in _slots:
                    r = concatenate_transform(i, pre)
                    acc = r if acc is None else acc + r
                return acc

            def combined_inv(post, _slots=tuple(slot_indices)):
                acc = None
                for i in _slots:
                    r = inverse_concatenate_transform(i, post)
                    acc = r if acc is None else acc + r
                return acc

            transform = JacobianTransform(combined_fwd, combined_inv)
        return SparseTensor([], [], None, pre_transforms=[transform])

    elementals_per_slot = [None] * len(primals)
    for slot_indices in groups.values():
        elemental = _make_elemental(slot_indices)
        for i in slot_indices:
            elementals_per_slot[i] = elemental
    return elementals_per_slot


def concatenate_elemental_rule(primals, **params):
    val_out = lax.concatenate_p.bind(*primals, **params)
    return val_out, _concatenate_elementals(primals, val_out, **params)


def concatenate_elemental_only(primal_out, primals, **params):
    return _concatenate_elementals(primals, primal_out, **params)


elemental_rules[lax.concatenate_p] = concatenate_elemental_rule
elemental_only_rules[lax.concatenate_p] = concatenate_elemental_only


# ---------- convert_element_type ----------


def _convert_element_type_elementals(primals, val_out, **params):
    new_dtype = params["new_dtype"]

    def convert_element_type_transform(pre):
        if pre.val is None:
            return pre.copy()
        new_pre_val = lax.convert_element_type(pre.val, new_dtype)
        new_out_dims = copy.deepcopy(pre.out_dims)
        new_primal_dims = copy.deepcopy(pre.primal_dims)
        return SparseTensor(
            new_out_dims,
            new_primal_dims,
            new_pre_val,
            scalar_mult=lax.convert_element_type(pre.scalar_mult, new_dtype),
            # None (statically-zero) fill stays None across a dtype convert.
            fill_value=(None if pre.fill_value is None
                        else lax.convert_element_type(pre.fill_value, new_dtype)),
        )

    def inverse_convert_element_type_transform(post):
        if post.val is None:
            return post.copy()
        new_post_val = lax.convert_element_type(post.val, new_dtype)
        new_out_dims = copy.deepcopy(post.out_dims)
        new_primal_dims = copy.deepcopy(post.primal_dims)
        return SparseTensor(
            new_out_dims,
            new_primal_dims,
            new_post_val,
            scalar_mult=lax.convert_element_type(post.scalar_mult, new_dtype),
            # None (statically-zero) fill stays None across a dtype convert.
            fill_value=(None if post.fill_value is None
                        else lax.convert_element_type(post.fill_value, new_dtype)),
        )

    transform = JacobianTransform(
        convert_element_type_transform, inverse_convert_element_type_transform
    )
    return [SparseTensor([], [], None, pre_transforms=[transform])]


def convert_element_type_rule(primals, **params):
    val_out = lax.convert_element_type_p.bind(*primals, **params)
    return val_out, _convert_element_type_elementals(primals, val_out, **params)


def convert_element_type_only(primal_out, primals, **params):
    return _convert_element_type_elementals(primals, primal_out, **params)


elemental_rules[lax.convert_element_type_p] = convert_element_type_rule
elemental_only_rules[lax.convert_element_type_p] = convert_element_type_only


# ---------- split (each output is a lax.slice of the input) ----------

def _split_elemental_for_output(primal, primal_out_k, start_k, end_k, axis):
    """Elemental partial for the k-th output of split w.r.t. the input.

    This is identical to the lax.slice elemental with start/limit indices
    chosen to select the k-th chunk along the split axis.
    """
    ndim = primal.ndim
    start_indices = tuple(start_k if i == axis else 0 for i in range(ndim))
    limit_indices = tuple(end_k if i == axis else primal.shape[i] for i in range(ndim))
    slice_params = {
        'start_indices': start_indices,
        'limit_indices': limit_indices,
        'strides': None,
    }
    # _slice_elementals returns list[SparseTensor] with one entry per invar
    return _slice_elementals([primal], primal_out_k, **slice_params)


def split_elemental_only(primal_outs, primals, **params):
    """Multi-output elemental rule for lax.split_p.

    Returns elementals[outvar_idx][invar_idx].  split has one invar and N
    outvars, so the outer list has N entries, each a length-1 list.
    """
    primal = primals[0]
    sizes = params['sizes']
    axis = params['axis']

    result = []
    start = 0
    for size, primal_out_k in zip(sizes, primal_outs):
        end = start + int(size)
        result.append(_split_elemental_for_output(primal, primal_out_k, start, end, axis))
        start = end
    return result


multi_output_elemental_only_rules[lax_src.split_p] = split_elemental_only
