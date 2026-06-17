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


def _factor_match(in_sizes, out_sizes):
    """Map output axes to input axes for a reshape that ONLY inserts/removes
    size-1 axes (a 1:1 alignment of the non-1 axes). Returns ``assign`` where
    ``assign[k]`` is the input-axis index that output axis ``k`` came from, or
    ``None`` for a freshly-inserted size-1 output axis. Returns ``None`` (caller
    must dense-fallback) when a non-1 axis is split or merged across the reshape,
    i.e. when the ordered non-1 sizes differ — the only regime in which each
    input dim survives as a single equal-size output dim, so any structure
    (diagonal pairing) attached to an input dim carries through untouched."""
    in_nontriv = [(i, s) for i, s in enumerate(in_sizes) if s != 1]
    out_nontriv = [(k, s) for k, s in enumerate(out_sizes) if s != 1]
    if [s for _, s in in_nontriv] != [s for _, s in out_nontriv]:
        return None
    assign = [None] * len(out_sizes)
    for (i, _), (k, _) in zip(in_nontriv, out_nontriv):
        assign[k] = i
    return assign


def _reshape_elementals(primals, val_out, **params):
    # Structure-preserving reshape: reshape only relabels the side it acts on
    # (OUTPUT for the forward transform, PRIMAL for the inverse); the opposite
    # side — where a diagonal pairing's partner lives — is untouched. When the
    # reshape merely inserts/removes size-1 axes and aligns the non-1 axes 1:1
    # (``_factor_match``), every dim — INCLUDING a diagonal pair — maps to a
    # single equal-size axis and is carried through with NO densification (val
    # stays O(n) for an incoming diagonal; cf. ``transpose_transform``). Any
    # split/merge of a non-1 axis, or a block-diagonal dim, falls back to the
    # dense path below, which is always correct.
    def _structural(in_dims, in_is_out, new_sizes, val, scalar_mult, fill_value,
                    other_dims):
        in_sizes = [d.logical_size for d in in_dims]
        new_sizes = [int(s) for s in new_sizes]
        assign = _factor_match(in_sizes, new_sizes)
        if assign is None:
            return None

        assigned_in = [a for a in assign if a is not None]
        assigned_set = set(assigned_in)
        # A split would reuse one input axis for several output axes.
        if len(assigned_in) != len(assigned_set):
            return None
        # A sparse (diagonal) in-dim must map 1:1 to a single equal-size output
        # axis and carry no block factor (logical_size folds a block into the
        # axis — splitting it is unsafe). Reject if any sparse in-dim is dropped.
        for i, d in enumerate(in_dims):
            if d.is_sparse:
                if d.block_size is not None:
                    return None
                if i not in assigned_set:
                    return None

        # ids: out_dims occupy [0, n_out); primal follow.
        if in_is_out:
            in_id_base, other_id_base = 0, len(new_sizes)
        else:
            in_id_base, other_id_base = len(other_dims), 0

        old_in_id_to_new = {}
        new_in_dims = []
        for k, src in enumerate(assign):
            new_id = in_id_base + k
            size = new_sizes[k]
            if src is None:
                # inserted size-1 axis — not stored in val (axis=None)
                new_in_dims.append(DenseIndex(new_id, size, None))
            else:
                d = in_dims[src]
                old_in_id_to_new[d.id] = new_id
                new_in_dims.append(replace(d, id=new_id))

        old_other_id_to_new = {d.id: other_id_base + j
                               for j, d in enumerate(other_dims)}
        new_other_dims = []
        for j, d in enumerate(other_dims):
            upd = {"id": other_id_base + j}
            if d.is_sparse:
                if d.other_id not in old_in_id_to_new:
                    return None
                upd["other_id"] = old_in_id_to_new[d.other_id]
            new_other_dims.append(replace(d, **upd))
        for idx, d in enumerate(new_in_dims):
            if d.is_sparse:
                if d.other_id not in old_other_id_to_new:
                    return None
                new_in_dims[idx] = replace(
                    d, other_id=old_other_id_to_new[d.other_id]
                )

        if in_is_out:
            out_dims, primal_dims = new_in_dims, new_other_dims
        else:
            out_dims, primal_dims = new_other_dims, new_in_dims

        return _swap_back_axes(
            SparseTensor(
                out_dims, primal_dims, val,
                scalar_mult=scalar_mult, fill_value=fill_value,
                check_consistency=False,
            )
        )

    def reshape_transform(pre):
        res = _structural(
            list(pre.out_dims), True, val_out.shape, pre.val,
            pre.scalar_mult, pre.fill_value, list(pre.primal_dims),
        )
        if res is not None:
            return res
        # Dense fallback: densify then reshape the OUTPUT side.
        full_val = pre.dense()
        new_shape, new_out_dims, new_primal_dims, counter = [], [], [], 0
        for s in val_out.shape:
            new_out_dims.append(DenseIndex(counter, s, counter))
            new_shape.append(s)
            counter += 1
        for d in pre.primal_dims:
            new_primal_dims.append(DenseIndex(counter, d.size, counter))
            new_shape.append(d.size)
            counter += 1
        return SparseTensor(new_out_dims, new_primal_dims,
                            full_val.reshape(new_shape))

    def inverse_reshape_transform(post):
        res = _structural(
            list(post.primal_dims), False, primals[0].shape, post.val,
            post.scalar_mult, post.fill_value, list(post.out_dims),
        )
        if res is not None:
            return res
        # Dense fallback: densify then reshape the PRIMAL side.
        full_val = post.dense()
        new_shape, new_out_dims, new_primal_dims, counter = [], [], [], 0
        for d in post.out_dims:
            new_out_dims.append(DenseIndex(counter, d.size, counter))
            new_shape.append(d.size)
            counter += 1
        for s in primals[0].shape:
            new_primal_dims.append(DenseIndex(counter, s, counter))
            new_shape.append(s)
            counter += 1
        return SparseTensor(new_out_dims, new_primal_dims,
                            full_val.reshape(new_shape))

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
    # slice selects a sub-range of the OUTPUT axes only (primal dims are never
    # sliced). When every sliced OUTPUT axis is a plain DenseIndex (not a
    # DiagonalIndex partner, not compressed) the slice acts on the val axis (or,
    # when that dim is implicit, only on the dim metadata) and the primal-side
    # diagonal structure is untouched and carried through WITHOUT materializing.
    # Only when a sliced axis is a sparse/compressed dim (slicing breaks the
    # square Kronecker structure) do we fall back to densify-then-slice. The
    # inverse (transpose-of-slice = interior-pad embedding) is likewise
    # structure-preserving when every slice-output primal dim is plain dense
    # (pad its own axis in place, n -> L); a diagonal-partner / compressed /
    # implicit primal dim falls back to densify-then-pad (_inverse_dense).
    start_indices = list(params["start_indices"])
    limit_indices = list(params["limit_indices"])
    _strides = params.get("strides")
    strides = [1] * len(start_indices) if _strides is None else list(_strides)

    # An output axis is "actually sliced" iff its range is not the trivial
    # full-axis stride-1 selection.
    def _is_sliced(ax, full):
        return not (start_indices[ax] == 0 and limit_indices[ax] == full
                    and strides[ax] == 1)

    def slice_transform(pre):
        # Structure-preserving fast path: every sliced output dim is a plain
        # DenseIndex (not sparse / not compressed).
        preservable = True
        for ax, d in enumerate(pre.out_dims):
            if _is_sliced(ax, d.size) and (d.is_sparse or d.is_compressed):
                preservable = False
                break

        if preservable:
            new_out_dims = []
            val = pre.val
            for ax, d in enumerate(pre.out_dims):
                if _is_sliced(ax, d.size):
                    # Plain dense dim: slice its val axis if materialized,
                    # otherwise just shrink the metadata (constant along it).
                    n = (limit_indices[ax] - 1 - start_indices[ax]) // strides[ax] + 1
                    if d.axis is not None and val is not None:
                        val = lax.slice_in_dim(
                            val, start_indices[ax], limit_indices[ax],
                            stride=strides[ax], axis=d.axis,
                        )
                    new_out_dims.append(replace(d, size=n))
                else:
                    new_out_dims.append(d)
            return SparseTensor(
                new_out_dims,
                list(pre.primal_dims),
                val,
                scalar_mult=pre.scalar_mult,
                fill_value=pre.fill_value,
            )

        # Dense fallback (a sliced axis is a diagonal partner / compressed):
        # densify, then slice. primal dims are appended as full-range, stride-1.
        s_idx = list(start_indices)
        l_idx = list(limit_indices)
        s_strides = list(strides)
        full_val = pre.dense()
        new_out_dims = []
        new_primal_dims = []
        counter = 0
        for s in val_out.shape:
            new_out_dims.append(DenseIndex(counter, s, counter))
            counter += 1
        for d in pre.primal_dims:
            new_primal_dims.append(DenseIndex(counter, d.size, counter))
            s_idx.append(0)
            l_idx.append(d.size)
            s_strides.append(1)  # primal dims are not sliced
            counter += 1
        new_val = lax.slice(full_val, s_idx, l_idx, s_strides)
        return SparseTensor(new_out_dims, new_primal_dims, new_val)

    def _inverse_dense(post):
        # The always-correct transpose-of-slice = interior-pad embedding on a
        # freshly-densified (out..., slice_out...) grid. Out axes 0..K-1 then the
        # rebuilt input (primal) axes follow, each padded low=start,
        # interior=stride-1, high=L-start-(n-1)*stride-1. (The old version used a
        # contiguous scatter that ignored ``strides`` AND hardcoded the input
        # dtype via jnp.zeros, mis-placing strided slices and crashing on f64.)
        full_val = post.dense()
        new_out_dims = []
        new_primal_dims = []
        counter = 0
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

    def inverse_slice_transform(post):
        # In 'fwd' elimination order this inverse can be drained against the bare
        # identity output-seed (no out/primal dims, val=None). Re-expand that seed
        # into the explicit identity Jacobian over the slice's OUTPUT shape so the
        # embedding below runs exactly as in 'rev' order.
        if _is_scalar_identity_post(post):
            post = _identity_post_over(post, val_out.shape, val_out.dtype)

        in_shape = primals[0].shape

        # Structure-preserving fast path: each primal (slice-output) dim is a
        # plain DenseIndex carried on a distinct physical val axis. The pad
        # embedding grows that axis in place (n -> L); the out-side dims keep
        # their own distinct axes and structure verbatim, so no axis collides.
        # A sparse (DiagonalIndex partner) / compressed / implicit (axis is None)
        # primal dim, or a primal-rank mismatch, routes to the dense fallback:
        # padding a diagonal partner's axis would break the equal-size pair
        # invariant (out size n != padded input size L), which the index
        # vocabulary cannot represent.
        preservable = (
            post.val is not None
            and len(post.primal_dims) == len(in_shape)
        )
        if preservable:
            for d in post.primal_dims:
                if d.is_sparse or d.is_compressed or d.axis is None:
                    preservable = False
                    break

        if not preservable:
            return _inverse_dense(post)

        # Renumber ids: out_dims occupy [0, K), primal follow [K, K+rank). The
        # out-side dims are carried with their structure intact (a diagonal pair
        # living entirely within out_dims is relinked); the rebuilt primal dims
        # are fresh DenseIndex on the SAME (now grown) physical axis. We build the
        # dims by hand and let _swap_back_axes restore the canonical layout
        # (check_consistency=False, exactly the reshape structural discipline).
        K = len(post.out_dims)
        old_to_new = {}
        for i, d in enumerate(post.out_dims):
            old_to_new[d.id] = i
        for i, d in enumerate(post.primal_dims):
            old_to_new[d.id] = K + i

        new_out_dims = []
        for i, d in enumerate(post.out_dims):
            nd = replace(d, id=i)
            if nd.is_sparse:
                nd = replace(nd, other_id=old_to_new[d.other_id])
            new_out_dims.append(nd)

        val = post.val
        new_primal_dims = []
        for ax, d in enumerate(post.primal_dims):
            L = in_shape[ax]
            n = d.size
            st, stride = start_indices[ax], strides[ax]
            high = L - st - (n - 1) * stride - 1
            val = lax.pad(
                val, jnp.array(0.0, dtype=val.dtype),
                [(st, high, stride - 1) if a == d.axis else (0, 0, 0)
                 for a in range(val.ndim)],
            )
            new_primal_dims.append(DenseIndex(K + ax, L, d.axis))

        return _swap_back_axes(
            SparseTensor(
                new_out_dims,
                new_primal_dims,
                val,
                scalar_mult=post.scalar_mult,
                fill_value=post.fill_value,
                check_consistency=False,
            )
        )

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
    # This gradient transformation takes a post edge and decomposes it into the
    # pre edges (forward), or vice versa (inverse). ``dim`` is the concat axis;
    # ``slices[i]`` is the [start, end) span of slot ``i`` along that axis.
    dim = params["dimension"]

    offset = primals[0].shape[dim]
    slices = {0: [0, offset]}
    for i, val in enumerate(primals[1:], start=1):
        slices[i] = [offset, offset + val.shape[dim]]
        offset += val.shape[dim]

    def concatenate_transform(primal_idx, pre):
        # ``pre.out_dims`` spans the concat OUTPUT shape (the ``dim``-th out dim
        # is this slot's concat axis). The transform embeds this slot's Jacobian
        # block into the full concat-output axis, zero-padding the other slots.
        new_out_dims = list(copy.deepcopy(pre.out_dims))
        new_primal_dims = list(copy.deepcopy(pre.primal_dims))
        d = new_out_dims[dim]
        idx, _idx = slices[primal_idx]

        if not d.is_sparse:
            if d.axis is not None:
                # Plain materialized output axis: concat zeros | block | zeros.
                lshape = list(pre.val.shape)
                rshape = list(pre.val.shape)
                lshape[d.axis] = idx
                rshape[d.axis] = val_out.shape[dim] - _idx
                lcat_zeros = jnp.zeros(lshape, dtype=pre.val.dtype)
                rcat_zeros = jnp.zeros(rshape, dtype=pre.val.dtype)

                new_val = jnp.concatenate(
                    [lcat_zeros, pre.val, rcat_zeros], axis=d.axis
                )

                new_out_dims[dim] = replace(
                    new_out_dims[dim], size=new_val.shape[d.axis]
                )
            else:
                # axis=None: this output dimension is a Kronecker factor not
                # stored in val. Materialize it: broadcast pre.val to the slot
                # size (zero-copy), then pad to the full concat axis.
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

                # Inserting a new axis shifts all subsequent val_axes up by 1.
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
            return SparseTensor(
                new_out_dims,
                new_primal_dims,
                new_val,
                scalar_mult=pre.scalar_mult,
                fill_value=pre.fill_value,
            )

        # SPARSE concat axis (DiagonalIndex). The structured scatter path mis-
        # shapes the embedding band whenever the concat axis is not val-axis 0
        # and the incoming Jacobian carries extra dense axes (e.g. from a
        # preceding matmul): a square DiagonalIndex cannot carry concat's
        # offset/rectangular selection band, so the val-axis bookkeeping breaks.
        # Robust route: densify, embed the slot's dense block into the full
        # concat-output axis with zero padding, and return a fully-dense tensor.
        full = pre.dense()  # (out_shape..., primal_shape...) in dim order
        pad_config = [(0, 0, 0)] * full.ndim
        pad_config[dim] = (idx, val_out.shape[dim] - _idx, 0)
        new_val = lax.pad(full, jnp.array(0.0, dtype=full.dtype), pad_config)

        counter = 0
        new_out = []
        new_primal = []
        out_shape = list(pre.out_shape)
        out_shape[dim] = val_out.shape[dim]
        for s in out_shape:
            new_out.append(DenseIndex(counter, s, counter))
            counter += 1
        for s in pre.primal_shape:
            new_primal.append(DenseIndex(counter, s, counter))
            counter += 1
        return SparseTensor(
            new_out,
            new_primal,
            new_val,
            scalar_mult=pre.scalar_mult,
            fill_value=pre.fill_value,
        )

    def inverse_concatenate_transform(primal_idx, post):
        # In 'fwd' elimination order this inverse can be drained against the bare
        # identity output-seed (no out/primal dims, val=None). Re-expand it into
        # the explicit identity Jacobian over the concat OUTPUT shape so the
        # slicing below extracts this slot's block exactly as in 'rev' order.
        if _is_scalar_identity_post(post):
            post = _identity_post_over(post, val_out.shape, val_out.dtype)
        new_out_dims = list(copy.deepcopy(post.out_dims))
        new_primal_dims = list(copy.deepcopy(post.primal_dims))

        d = None
        if len(new_primal_dims) > 0:
            d = new_primal_dims[dim]
        if d is None:
            # post is a pure transform with no primal dims; nothing to slice.
            return SparseTensor(
                new_out_dims,
                new_primal_dims,
                post.val,
                scalar_mult=post.scalar_mult,
                fill_value=post.fill_value,
            )

        if not d.is_sparse:
            if d.axis is not None:
                new_val = lax.slice_in_dim(post.val, *slices[primal_idx], axis=d.axis)
                new_primal_dims[dim] = replace(d, size=new_val.shape[d.axis])
            else:
                # axis=None: the primal dimension is a Kronecker factor not stored
                # in val. There is no axis to slice — narrow the size only.
                new_primal_dims[dim] = replace(
                    d, size=slices[primal_idx][1] - slices[primal_idx][0]
                )
                new_val = post.val
            return SparseTensor(
                new_out_dims,
                new_primal_dims,
                new_val,
                scalar_mult=post.scalar_mult,
                fill_value=post.fill_value,
            )

        # SPARSE concat axis (DiagonalIndex). Same intrinsic limitation as the
        # forward sparse branch (a square DiagonalIndex cannot represent concat's
        # rectangular selection slice). Robust route: densify, slice the primal
        # concat axis to this slot's span, return a fully-dense tensor.
        full = post.dense()  # (out_shape..., primal_shape...) in dim order
        concat_ax = len(post.out_dims) + dim
        new_val = lax.slice_in_dim(full, *slices[primal_idx], axis=concat_ax)

        counter = 0
        new_out = []
        new_primal = []
        for s in post.out_shape:
            new_out.append(DenseIndex(counter, s, counter))
            counter += 1
        primal_shape = list(post.primal_shape)
        primal_shape[dim] = slices[primal_idx][1] - slices[primal_idx][0]
        for s in primal_shape:
            new_primal.append(DenseIndex(counter, s, counter))
            counter += 1
        return SparseTensor(
            new_out,
            new_primal,
            new_val,
            scalar_mult=post.scalar_mult,
            fill_value=post.fill_value,
        )

    # Group primal slot indices by primal identity. When the same primal feeds
    # multiple slots of one concatenate (e.g. concat([pl, pl, pl])), the graph
    # stores a single edge per (invar, outvar) pair, so the elemental for that
    # invar must represent the sum of all slot contributions.
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
