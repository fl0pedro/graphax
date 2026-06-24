"""Densification of ``SparseTensor`` axes.

THE DENSIFICATION MAP
=====================
Densification is one *ladder* from most-compressed to fully-materialized::

    compressed (BandedIndex / SetIndex)
        │  _densify_compressed_dims / Index.densify_axis
        ▼
    diagonal (DiagonalIndex block-diagonal pairs)   ← matmul / elementwise consume here
        │  dense() / _densify_diagonal_select
        ▼
    dense grid (plain DenseIndex, NxN with fill off-block-diagonal)

Two questions pick the entry point: (1) *how far down the ladder*, and (2)
*what you get back* — a ``SparseTensor`` (structure preserved, ``scalar_mult``
deferred) or a raw ``Array`` (fully dense, ``scalar_mult`` folded in).

Entry points (where they live -> what they return):

* ``dense(tensor, axes=None, hard=False)``  [this module]  -> ``SparseTensor``
    The workhorse (~97 call sites). Materializes sparse pairs into dense axes.
    ``axes`` selects *which* logical dims (None = all); ``hard=True`` also
    materializes *implicit* dims (``axis is None``, i.e. not yet carried by
    ``val``). Does NOT understand BandedIndex/SetIndex — densify those first.

* ``dense_for_matmul(tensor)``  [this module]  -> ``Array``
    Fusion-friendly full densify for feeding ``jax.lax.dot_general``. Two fast
    paths — fully-dense (``val`` IS the answer, modulo a permutation) and a
    single sparse pair (one broadcast+select) — that XLA folds into the matmul
    kernel (stays in SMEM, no HBM spill). Everything else falls back to
    ``dense(tensor, hard=True)`` then ``* scalar_mult`` — i.e. it is exactly
    ``SparseTensor.dense()`` minus the compressed-dim handling, plus fast paths.

* ``SparseTensor.dense()``  [tensor.py]  -> ``Array``
    The public "give me the dense array" method, used everywhere (``flat``,
    ``__getitem__``, ``float()``, and as the test-suite's correctness oracle).
    = ``_densify_compressed_dims`` (if any) -> ``dense(hard=True)`` -> ``* scalar_mult``.

* ``_densify_compressed_dims(tensor, compact=False)``  [ops/utils.py]  -> ``SparseTensor``
    Replaces BandedIndex/SetIndex dims with their expanded equivalents.
    ``compact=True`` stops at the DiagonalIndex rung (M× less storage) when the
    pair ``reduces_to_diagonal``; ``compact=False`` goes to the full dense grid.

* ``_materialize_for_op(tensor)``  [ops/utils.py]  -> ``SparseTensor``
    The matmul/elementwise boundary: strips compressed dims so those ops only
    ever see Dense/Diagonal. == ``_densify_compressed_dims(compact=True)``.
    Exposed as the ``SparseTensor._materialize_compressed()`` method (via the
    ``@_on_materialized`` decorator) for reductions / non-linear unary ops,
    which must run on the dense element multiset, not the raw band/set buffer.

Internal kernels (not entry points): ``_apply_dense_scattering`` ->
``_densify_diagonal_select`` (broadcast + ``jnp.where`` over an eye-mask — *no*
``lax.scatter``, despite the surrounding "scatter" vocabulary, so XLA can fuse
it); compressed expansion goes through ``Index.densify_axis`` -> the band
kernels in ``ops/block_storage.py``.

Architectural invariant: ops consume only {Dense, Diagonal}; {Banded, Set} are
densified at every boundary except transpose-as-view.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Sequence

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax import Array

from graphax.sparse.indexes import DenseIndex, Index, DiagonalIndex
from graphax.sparse.dtype_compute import _scaled_mul

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


# --- Public API ----------------------------------------------------------
def dense(
    tensor: SparseTensor, axes: Sequence[int] | None = None, hard: bool = False
) -> SparseTensor:
    """Materialize sparse pairs into dense axes on a ``SparseTensor``.

    Pipeline (each step is one helper below):

      1. ``_get_implicit_indices`` — find dims whose ``axis`` (or ``block_axis``
         for sparse) is ``None``, i.e. the val doesn't yet carry that axis.
         Under ``hard=False`` these are excluded from the materialization set;
         under ``hard=True`` they're forced in.
      2. ``_broadcast_and_append_dimensions`` — grow ``val`` so every requested
         (and implicit, when ``hard``) axis has a physical slot. Reuses any
         orphan val axes via ``_collect_free_val_axes`` before appending new ones.
      3. ``_collect_scatter_indices`` — for each requested dim, also pull in
         its sparse-pair sibling (the diagonal needs both ends).
      4. ``_apply_dense_scattering`` → ``_prepare_values_for_scattering`` →
         ``_densify_diagonal_select`` — for each sparse pair to materialize,
         emit one ``(N, N, …) where eye_mask`` broadcast/select; loop the pairs
         independently (linearizing them would corrupt block-axis layouts and
         waste memory in the trailing axes).
      5. Wrap the resulting array + relabeled dims into a fresh ``SparseTensor``.

    Worked example — a single sparse pair with ``N=2`` blocks of size ``B=3``::

        a = SparseTensor(
            (DiagonalIndex(0, 2, axis=0, other_id=1, block_size=3, block_axis=1),),
            (DiagonalIndex(1, 2, axis=0, other_id=0, block_size=3, block_axis=2),),
            val,  # shape (2, 3, 3) — two 3×3 blocks
        )
        a.dense()  # SparseTensor with val shape (6, 6) — blocks on the diagonal,
                   # fill_value elsewhere; dims become two ``DenseIndex``-of-size-6.

    Internally: implicit=∅ (every dim already has an axis), so step 3 is a
    no-op. Step 4 expands the requested ``{0, 1}`` to itself (siblings already
    in). Step 5 calls ``_densify_diagonal_select`` once on the leading axis,
    producing shape ``(2, 2, 3, 3)``; ``_apply_dense_scattering`` then
    transposes to ``(2, 3, 2, 3)`` and reshapes to ``(6, 6)``.

    ``axes`` selects which logical dim positions to densify (``None`` = all).
    ``hard=True`` also materializes dims whose val axis is implicit (``None``).
    """
    logical_indices = set(range(tensor.ndim)) if axes is None else set(axes)
    id_to_idx = {dim.id: i for i, dim in enumerate(tensor.dims)}
    implicit = _get_implicit_indices(tensor, logical_indices, hard)
    if not hard:
        logical_indices -= implicit

    values = jnp.array(1.0, dtype=tensor.dtype) if tensor.val is None else tensor.val
    values, updated_dims = _broadcast_and_append_dimensions(tensor, values, implicit)

    actual_scatter = _collect_scatter_indices(logical_indices, updated_dims, id_to_idx)
    phys_to_scatter = sorted(
        {
            updated_dims[i].axis
            for i in actual_scatter
            if updated_dims[i].is_sparse
            and updated_dims[i].axis is not None
        }
    )

    values, result_dims = _apply_dense_scattering(
        values, tensor._eff_fill, updated_dims, actual_scatter, phys_to_scatter
    )
    from graphax.sparse.tensor import SparseTensor

    return SparseTensor(
        tuple(result_dims[: len(tensor.out_dims)]),
        tuple(result_dims[len(tensor.out_dims) :]),
        values,
        scalar_mult=tensor.scalar_mult,
        fill_value=tensor.fill_value,  # None (statically zero) preserved
        check_consistency=False,
    )


def dense_for_matmul(tensor: SparseTensor) -> Array:
    """Fusion-friendly dense form (broadcast+select; no scatter / no gather where possible).

    See module docstring. Falls back to the scatter-based ``dense()`` for shapes the
    simple builder doesn't cover.
    """
    # Fast path: fully-dense tensor — val IS the dense form (modulo permutation).
    if all(not d.is_sparse for d in tensor.dims):
        if tensor.val is None:
            # val=None ⇒ the structure is all-ones (× scalar_mult) — same as
            # dense(); fill_value paints only a sparse pair's off-diagonal, of
            # which a fully-dense tensor has none. (Using _eff_fill here was the
            # dense()/dense_for_matmul inconsistency: it materialised 0 instead
            # of 1 for a fully-dense val=None operand.)
            return jnp.broadcast_to(
                _scaled_mul(jnp.array(1.0, dtype=tensor.dtype), tensor.scalar_mult),
                tensor.shape,
            )
        v = tensor.val
        perm = [d.axis for d in tensor.dims if d.axis is not None]
        # A Diag-split / Compress can leave the physical ``val`` with EXTRA size-1
        # axes that no dim references (the ``(...,1,1,1)`` tails) while its
        # referenced axes are PERMUTED relative to the logical (out..., primal...)
        # order. Reconstruct the logical layout explicitly: pull each dim's physical
        # axis into dim order, drop the unreferenced size-1 axes, and broadcast the
        # implicit (``axis is None``) dims up. Without this, ``broadcast_to`` below
        # sees a higher-rank, mis-ordered array and fails ("Cannot broadcast to
        # shape with fewer dimensions") — the ViT seq/embed densify case.
        if perm and len(set(perm)) == len(perm) and len(perm) < v.ndim:
            extra = [ax for ax in range(v.ndim) if ax not in perm]
            if all(int(v.shape[ax]) == 1 for ax in extra):
                v = _scaled_mul(jnp.transpose(v, perm + extra), tensor.scalar_mult)
                v = v.reshape(v.shape[: len(perm)])  # drop trailing size-1 extras
                built, v_iter = [], iter(v.shape)
                for d in tensor.dims:
                    built.append(next(v_iter) if d.axis is not None else 1)
                return jnp.broadcast_to(v.reshape(built), tensor.shape)
        if (
            perm
            and len(perm) == v.ndim
            and sorted(perm) == list(range(len(perm)))
            and perm != list(range(len(perm)))
        ):
            v = v.transpose(perm)
        v = _scaled_mul(v, tensor.scalar_mult)
        # Broadcast back up to the logical shape: a SparseTensor can carry a
        # rank-0 (or otherwise rank-reduced) ``val`` while its dims advertise
        # a larger structural shape (e.g. concat-transformed Jacobians where
        # the val landed as a scalar but the structure says ``(N,)``).
        if v.ndim != len(tensor.shape):
            # If v has one axis per non-None-axis dim, insert singleton axes at
            # None-axis positions so broadcast_to lines them up correctly.
            # (broadcast_to right-aligns smaller-rank arrays, which fails when
            # the singleton dim is interior or trailing — e.g. (4,) → (4, 1).)
            n_non_none = sum(1 for d in tensor.dims if d.axis is not None)
            if v.ndim == n_non_none and n_non_none < len(tensor.shape):
                expanded_shape = []
                v_iter = iter(v.shape)
                for d in tensor.dims:
                    expanded_shape.append(1 if d.axis is None else next(v_iter))
                v = v.reshape(expanded_shape)
            v = jnp.broadcast_to(v, tensor.shape)
        return v

    # Single-sparse-pair fast path: emit a where over a 1-fusion dense form.
    # Requires every NON-pair (dense) dim to be physically present in ``val``
    # (``axis is not None``): the layout step below maps each such dim to a
    # leftover ``val`` axis, so an implicit (``axis is None``) dense dim has no
    # axis to map and would break the layout. Those tensors fall through to the
    # generic ``dense(hard=True)`` path below, which broadcasts implicit dims in.
    sparse_dims = [d for d in tensor.dims if d.is_sparse]
    if (
        tensor.val is not None
        and len(sparse_dims) == 2
        and sparse_dims[0].other_id == sparse_dims[1].id
        and sparse_dims[1].other_id == sparse_dims[0].id
        and sparse_dims[0].axis == sparse_dims[1].axis
        and all(d.axis is not None for d in tensor.dims if not d.is_sparse)
    ):
        d_o, d_i = sparse_dims
        B_o, B_i = d_o.block_size or 1, d_i.block_size or 1
        N = d_o.size
        logical_outer, logical_inner = N * B_o, N * B_i
        gather_axes = [d_o.axis, d_o.block_axis, d_i.block_axis]
        if all(a is not None for a in gather_axes):
            v = _scaled_mul(tensor.val, tensor.scalar_mult)
            leftover = [a for a in range(v.ndim) if a not in gather_axes]
            v = v.transpose(gather_axes + leftover)
            # v.shape: (N, B_o, B_i, *leftover_sizes). Collapse (N, B_o) → logical_outer
            # so that v_2d[i, k, *l] == v[i // B_o, i % B_o, k, *l].
            leftover_sizes = list(v.shape[3:])
            v_2d = v.reshape(logical_outer, B_i, *leftover_sizes)
            # Tile across the inner axis via broadcast+reshape (pure shape ops, fold into
            # the consuming kernel). gathered[i, j, *l] == v_2d[i, j % B_i, *l].
            v_3d = jnp.broadcast_to(
                v_2d[:, None, ...], (logical_outer, N, B_i, *leftover_sizes)
            )
            gathered = v_3d.reshape(logical_outer, logical_inner, *leftover_sizes)
            blk_o = jnp.arange(logical_outer) // B_o
            blk_i = jnp.arange(logical_inner) // B_i
            mask = blk_o[:, None] == blk_i[None, :]
            mask_b = mask[(..., *((None,) * len(leftover_sizes)))]
            dense_pair = jnp.where(
                mask_b, gathered, _scaled_mul(tensor._eff_fill, tensor.scalar_mult)
            )
            # Reorder dense_pair's axes to match tensor.dims order. ``target_axes[i]`` is
            # the dense_pair axis that should land at result position ``i``, so the
            # transpose permutation is ``target_axes`` directly (NOT its inverse).
            leftover_iter = iter(range(2, 2 + len(leftover_sizes)))
            target_axes = [
                0 if d is d_o else 1 if d is d_i else next(leftover_iter)
                for d in tensor.dims
            ]
            return dense_pair.transpose(target_axes)

    densified = dense(tensor, hard=True)
    val = densified.val
    if val is None:
        # Purely structural after hard densify ⇒ all-ones × scalar_mult (val=None
        # semantics; see the fully-dense branch above).
        return jnp.broadcast_to(
            _scaled_mul(jnp.array(1.0, dtype=tensor.dtype), tensor.scalar_mult),
            densified.shape,
        )
    return _scaled_mul(val, tensor.scalar_mult)


# --- Internals -----------------------------------------------------------
def _is_dimension_implicit(dim, id_to_idx):
    needs_val = dim.axis is None
    needs_block = (
        dim.is_sparse and dim.block_size and dim.block_axis is None
    )
    if needs_val or needs_block:
        to_add = {id_to_idx[dim.id]}
        if dim.is_sparse:
            to_add.add(id_to_idx[dim.other_id])
        return True, to_add
    return False, set()


def _get_implicit_indices(tensor, requested_axes, hard):
    id_to_idx = {dim.id: i for i, dim in enumerate(tensor.dims)}
    implicit = set()
    for i in requested_axes:
        is_impl, indices = _is_dimension_implicit(tensor.dims[i], id_to_idx)
        if is_impl:
            implicit.update(indices)
    return implicit


def _calculate_target_shape(val_shape, dims):
    target = list(val_shape)
    for dim in dims:
        if dim.axis is not None:
            target[dim.axis] = max(target[dim.axis], dim.size)
        if dim.is_sparse and dim.block_axis is not None:
            target[dim.block_axis] = max(target[dim.block_axis], dim.block_size or 1)
    return tuple(target)


def _append_primary_dimension(
    i, dim, implicit, current_ndim, dims_to_append, sparse_pair_map, free_axes
):
    if i in implicit and dim.axis is None:
        if dim.is_sparse:
            pair_key = tuple(sorted((dim.id, dim.other_id)))
            if pair_key in sparse_pair_map:
                new_idx = sparse_pair_map[pair_key]
            else:
                new_idx = _claim_free_axis(free_axes, dim.size)
                if new_idx is None:
                    new_idx = current_ndim + len(dims_to_append)
                    dims_to_append.append(dim.size)
                sparse_pair_map[pair_key] = new_idx
        else:
            new_idx = _claim_free_axis(free_axes, dim.size)
            if new_idx is None:
                new_idx = current_ndim + len(dims_to_append)
                dims_to_append.append(dim.size)
        return replace(dim, axis=new_idx)
    return dim


def _claim_free_axis(free_axes, size):  # TODO this is a workaround*
    bucket = free_axes.get(size)
    if bucket:
        return bucket.pop(0)
    return None


def _append_block_dimension(i, dim, implicit, current_ndim, dims_to_append):
    if (
        i in implicit
        and dim.is_sparse
        and dim.block_axis is None
        and dim.block_size is not None
    ):
        new_idx = current_ndim + len(dims_to_append)
        dims_to_append.append(dim.block_size)
        return replace(dim, block_axis=new_idx)
    return dim


def _collect_free_val_axes(tensor, val_shape):  # TODO this is a workaround*
    claimed = set()
    for d in tensor.dims:
        if d.axis is not None:
            claimed.add(d.axis)
        if d.is_sparse and d.block_axis is not None:
            claimed.add(d.block_axis)
    free = {}
    for ax in range(len(val_shape)):
        if ax not in claimed:
            free.setdefault(val_shape[ax], []).append(ax)
    return free


def _broadcast_and_append_dimensions(tensor, values, implicit):
    if tensor.val is not None:
        target = _calculate_target_shape(values.shape, tensor.dims)
        if target != values.shape:
            values = jnp.broadcast_to(values, target)
    free_axes = _collect_free_val_axes(tensor, values.shape)
    dims_to_append, sparse_pair_map, current_ndim = [], {}, values.ndim
    new_dims = [
        _append_primary_dimension(
            i, d, implicit, current_ndim, dims_to_append, sparse_pair_map, free_axes
        )
        for i, d in enumerate(tensor.dims)
    ]
    new_dims = [
        _append_block_dimension(i, d, implicit, current_ndim, dims_to_append)
        for i, d in enumerate(new_dims)
    ]
    if dims_to_append:
        values = lax.broadcast_in_dim(
            values, values.shape + tuple(dims_to_append), tuple(range(current_ndim))
        )
    return values, new_dims


def _collect_scatter_indices(logical_indices, updated_dims, id_to_idx):
    scatter_logical = set()
    for i in logical_indices:
        scatter_logical.add(i)
        if updated_dims[i].is_sparse:
            scatter_logical.add(id_to_idx[updated_dims[i].other_id])
    return scatter_logical


def _prepare_values_for_scattering(values, scatter_axes, fill_value):
    if not scatter_axes:
        return values, {idx: idx for idx in range(values.ndim)}
    other_axes = [i for i in range(values.ndim) if i not in scatter_axes]
    unique_perm = list(dict.fromkeys(scatter_axes + other_axes))
    values = values.transpose(unique_perm)
    num_scatter = len(scatter_axes)
    # Densify each sparse pair INDEPENDENTLY by scattering its axis to
    # ``(N_i, N_i)`` one at a time. Linearizing all pairs into a single
    # ``B = prod(N_i)`` diagonal would (a) waste memory in the trailing
    # block-diag axes and (b) corrupt block-axis layouts because the
    # resulting reshape can't simultaneously place the same physical
    # block axis at both an out-side and a primal-side position.
    #
    # Invariant after densifying ``i`` pairs:
    #   axes[0..i-1]                     = out copies (in scatter_axes order)
    #   axes[i..num_scatter-1]           = pairs not yet densified (original order)
    #   axes[num_scatter..num_scatter+i-1] = primal copies (in scatter_axes order)
    #   axes[num_scatter+i..]            = trailing axes (original order)
    for i in range(num_scatter):
        # The next pair to densify currently sits at axis ``i`` (its "out"
        # slot). Move it to position 0 so ``_densify_diagonal_select`` can
        # operate on the leading axis.
        if i != 0:
            perm = [i] + [a for a in range(values.ndim) if a != i]
            values = values.transpose(perm)
        values = _densify_diagonal_select(values, fill_value)
        # ``_densify_diagonal_select`` produced shape ``(N, N, *rest)``.
        # Place the new "out" copy at position ``i`` (its final slot) and
        # the new "primal" copy at position ``num_scatter + i`` (slot for
        # the i-th primal copy) without disturbing the relative order of
        # the as-yet-undensified scatter axes or the trailing axes.
        ndim = values.ndim
        # Current layout: [N_out (axis 0), N_primal (axis 1), <i-1 already-done out copies that got pushed to indices 2..i>, <undensified pairs at i+1..num_scatter>, <i-1 already-done primal copies>, <trailing>].
        # We want: [<already-done out 0..i-1 at 0..i-1>, N_out at i, <undensified at i+1..num_scatter>, <already-done primal 0..i-1>, N_primal at num_scatter+i, <trailing>].
        # Build the target permutation explicitly.
        # After the transpose at the start of this iteration, values has the
        # invariant for "i pairs processed before this iteration":
        #   pre-densify shape: [pair_i (was axis i, moved to 0), out_0..out_{i-1} (at 1..i), undensified_{i+1}..{num_scatter-1} (at i+1..num_scatter-1), primal_0..primal_{i-1} (at num_scatter..num_scatter+i-1), trailing (at num_scatter+i..)].
        # _densify produced: [out_i (0), primal_i (1), out_0..out_{i-1} (2..i+1), undensified (i+2..num_scatter), primal_0..primal_{i-1} (num_scatter+1..num_scatter+i), trailing (num_scatter+i+1..)].
        # Target order: [out_0..out_{i-1} (2..i+1), out_i (0), undensified (i+2..num_scatter), primal_0..primal_{i-1} (num_scatter+1..num_scatter+i), primal_i (1), trailing (num_scatter+i+1..)]
        target = (
            list(range(2, i + 2))  # out_0..out_{i-1}
            + [0]  # out_i
            + list(range(i + 2, num_scatter + 1))  # undensified pairs
            + list(
                range(num_scatter + 1, num_scatter + i + 1)
            )  # primal_0..primal_{i-1}
            + [1]  # primal_i
            + list(range(num_scatter + i + 1, ndim))  # trailing
        )
        if target != list(range(ndim)):
            values = values.transpose(target)
    phys_map = {old: (i, num_scatter + i) for i, old in enumerate(scatter_axes)}
    for i, old in enumerate(other_axes):
        phys_map[old] = 2 * num_scatter + i
    return values, phys_map


def _apply_dense_scattering(
    values, fill_value, logical_dims, scatter_logical_indices, scatter_phys_axes
):
    values, phys_map = _prepare_values_for_scattering(
        values, scatter_phys_axes, fill_value
    )
    final_perm, final_shape, active_axes = [], [], set()
    log_to_phys, visited_pairs, sparse_phys_to_final = {}, set(), {}
    curr_f_idx = 0

    for i, dim in enumerate(logical_dims):
        pair_key = (
            tuple(sorted((dim.id, dim.other_id)))
            if dim.is_sparse
            else (dim.id,)
        )
        if i in scatter_logical_indices and dim.is_sparse:
            idx = 1 if pair_key in visited_pairs else 0
            visited_pairs.add(pair_key)
            p_idx = phys_map[dim.axis][idx]
            final_perm.append(p_idx)
            active_axes.add(p_idx)
            l_size = dim.size
            if dim.block_axis is not None:
                b_p = phys_map[dim.block_axis]
                final_perm.append(b_p)
                active_axes.add(b_p)
                l_size *= dim.block_size
            final_shape.append(l_size)
            log_to_phys[i] = (curr_f_idx, None, l_size)
            curr_f_idx += 1
        elif dim.is_sparse:
            new_v, new_b = None, None
            if pair_key not in visited_pairs:
                if dim.axis is not None:
                    p_idx = phys_map[dim.axis]
                    final_perm.append(p_idx)
                    active_axes.add(p_idx)
                    sparse_phys_to_final[dim.axis] = curr_f_idx
                    final_shape.append(dim.size)
                    curr_f_idx += 1
                visited_pairs.add(pair_key)
            new_v = sparse_phys_to_final.get(dim.axis)
            if dim.block_axis is not None:
                b_p = phys_map[dim.block_axis]
                final_perm.append(b_p)
                active_axes.add(b_p)
                new_b = curr_f_idx
                final_shape.append(dim.block_size)
                curr_f_idx += 1
            log_to_phys[i] = (new_v, new_b, None)
        else:
            new_v = None
            if dim.axis is not None:
                p_idx = phys_map[dim.axis]
                final_perm.append(p_idx)
                active_axes.add(p_idx)
                new_v = curr_f_idx
                final_shape.append(dim.size)
                curr_f_idx += 1
            log_to_phys[i] = (new_v, None, None)

    final_perm.extend(i for i in range(values.ndim) if i not in active_axes)
    unique_perm = list(dict.fromkeys(final_perm))
    values = values.transpose(unique_perm).reshape(
        tuple(final_shape) + values.shape[len(unique_perm) :]
    )
    return values, _reconstruct_logical_dimensions(logical_dims, log_to_phys)


def _reconstruct_logical_dimensions(logical_dims, logical_to_physical):
    res = []
    for i, dim in enumerate(logical_dims):
        v_ax, b_ax, d_size = logical_to_physical[i]
        if d_size is not None:
            res.append(DenseIndex(dim.id, d_size, v_ax))
        elif dim.is_sparse:
            res.append(replace(dim, axis=v_ax, block_axis=b_ax))
        else:
            res.append(replace(dim, axis=v_ax))
    return res


def _densify_diagonal_select(val: Array, fill_value: Array) -> Array:
    """Place ``val[i, ...]`` on the diagonal of a single sparse pair.

    Returns a ``(n_diag, n_diag, *trailing)`` grid where the leading axis of
    ``val`` (size ``n_diag``) is replicated to a diagonal and ``fill_value``
    is placed everywhere off-diagonal. Implemented as a single broadcast+select
    fusion (no scatter) so XLA can fold it into a downstream consumer.

    Called once per sparse pair from ``_prepare_values_for_scattering`` — each
    call materializes ONE pair's diagonal independently, so callers with
    multiple sparse pairs invoke this in a loop rather than linearizing all
    pairs into a single combined diagonal."""
    n_diag = val.shape[0]
    fv = jnp.asarray(fill_value, dtype=val.dtype)
    eye_mask = jnp.eye(n_diag, dtype=jnp.bool_)
    eye_mask = eye_mask[(slice(None), slice(None)) + (None,) * (val.ndim - 1)]
    val_b = jnp.broadcast_to(val[:, None], (n_diag, n_diag) + val.shape[1:])
    return jnp.where(eye_mask, val_b, fv)
