import jax
import jax.lax as lax
import jax.numpy as jnp

from ..sparse.tensor import (
    DenseIndex,
    DiagonalIndex,
    SparseTensor,
)
from .base import elemental_rules, get_ndim, get_shape
from .transforms import (
    JacobianTransform,
    _dense_grid,
    _identity_post_over,
    _is_scalar_identity_post,
    _slice_elementals,
)

# ---------------------------------------------------------------------------
# Shared helpers: replay a gather / scatter over a LEADING shape-block while a
# trailing block of axes rides along as a batch.
#
# A Jacobian SparseTensor densifies to ``(out_shape..., primal_shape...)``. The
# gather / scatter / dynamic-update-slice rules below act on ONE of those two
# blocks (the OUT block in the forward transform, the PRIMAL block in the
# adjoint) and must leave the opposite block untouched — that opposite block is
# exactly the contracted cotangent axes, so leaving it alone is what keeps the
# adjoint O(operand) rather than O(operand x output).
#
# We replay the *original* XLA primitive (same ``dimension_numbers``) per
# trailing slice via ``jax.vmap`` and never call ``int()`` / ``float()`` on the
# index or value arrays, so every path is jit-safe.
# ---------------------------------------------------------------------------


def _apply_over_leading(arr, lead_ndim, fn):
    """Apply ``fn`` (a single-array op on the LEADING ``lead_ndim`` axes) to
    ``arr`` while vmapping over the TRAILING axes as a flat batch. Returns the
    result with the (possibly reshaped) leading block restored to the front and
    the trailing batch restored to the back."""
    nb = arr.ndim - lead_ndim
    if nb == 0:
        return fn(arr)
    # bring the trailing batch axes to the front
    perm = list(range(lead_ndim, lead_ndim + nb)) + list(range(lead_ndim))
    x = jnp.transpose(arr, perm)                 # (*batch, lead...)
    flat = x.reshape((-1,) + x.shape[nb:])       # (B, lead...)
    res = jax.vmap(fn)(flat)                      # (B, newlead...)
    res = res.reshape(x.shape[:nb] + res.shape[1:])
    new_lead = res.ndim - nb
    # move the batch axes back to the end
    perm2 = list(range(nb, nb + new_lead)) + list(range(nb))
    return jnp.transpose(res, perm2)


def _scatter_dn_from_gather(dn):
    """The scatter ``dimension_numbers`` that is the ADJOINT of a gather with
    ``dn`` (used by the gather rule's inverse = scatter-add of the cotangent)."""
    return lax.ScatterDimensionNumbers(
        update_window_dims=dn.offset_dims,
        inserted_window_dims=dn.collapsed_slice_dims,
        scatter_dims_to_operand_dims=dn.start_index_map,
        operand_batching_dims=dn.operand_batching_dims,
        scatter_indices_batching_dims=dn.start_indices_batching_dims,
    )


def _gather_dn_from_scatter(dn):
    """The gather ``dimension_numbers`` that is the ADJOINT of a scatter with
    ``dn`` (used by the scatter/segment rules' update Jacobian inverse = gather
    of the cotangent)."""
    return lax.GatherDimensionNumbers(
        offset_dims=dn.update_window_dims,
        collapsed_slice_dims=dn.inserted_window_dims,
        start_index_map=dn.scatter_dims_to_operand_dims,
        operand_batching_dims=dn.operand_batching_dims,
        start_indices_batching_dims=dn.scatter_indices_batching_dims,
    )


def _do_gather(operand, indices, gdn, slice_sizes, params):
    """High-level ``lax.gather`` (NOT ``gather_p.bind``). Used for synthesized
    adjoint/forward selections so non-unique indices behave correctly."""
    return lax.gather(
        operand, indices, gdn, tuple(slice_sizes),
        indices_are_sorted=params.get("indices_are_sorted", False),
        unique_indices=params.get("unique_indices", False),
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def _do_scatter_add(operand, indices, updates, sdn, params):
    """High-level ``lax.scatter_add`` — ACCUMULATES on repeated indices (the raw
    ``scatter_add_p.bind`` with ``update_jaxpr=None`` overwrites, dropping the
    repeated-index contributions an adjoint scatter-add must sum)."""
    return lax.scatter_add(
        operand, indices, updates, sdn,
        indices_are_sorted=params.get("indices_are_sorted", False),
        unique_indices=params.get("unique_indices", False),
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def _do_scatter_set(operand, indices, updates, sdn, params):
    """High-level ``lax.scatter`` (overwrite). Used only for building the
    operand-side coefficient/mask grids where targets are treated as set."""
    return lax.scatter(
        operand, indices, updates, sdn,
        indices_are_sorted=params.get("indices_are_sorted", False),
        unique_indices=params.get("unique_indices", False),
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def _adjoint_gather_slice_sizes(dn, out_ndim, up_shape):
    """``slice_sizes`` for the adjoint gather of a scatter: 1 on every operand
    dim that is inserted (a point) or a batch dim, and the UPDATE window-dim
    size on every operand dim that carries a window. Each scatter
    ``update_window_dim`` maps (in order) to a non-inserted, non-batch operand
    dim; the window's extent along it is that update dim's size."""
    inserted = set(dn.inserted_window_dims) | set(dn.operand_batching_dims)
    non_inserted = [d for d in range(out_ndim) if d not in inserted]
    slice_sizes = [1] * out_ndim
    for w_i, w_dim in enumerate(dn.update_window_dims):
        slice_sizes[non_inserted[w_i]] = up_shape[w_dim]
    return slice_sizes


# ============================= dynamic_slice ===============================


def dynamic_slice_elemental_rule(primals, **params):
    # dynamic_slice with CONCRETIZED start indices is exactly a static lax.slice
    # over [start, start+size). Route through ``_slice_elementals`` to reuse its
    # structure-preserving forward path (a diagonal through a dense-axis slice
    # stays O(n) instead of densifying), interior-pad inverse, dense fallback,
    # and terminal-output identity-seed guard — no duplicated slice logic.
    val_out = lax.dynamic_slice_p.bind(*primals, **params)
    operand = primals[0]
    start_list = [int(s) for s in primals[1:]]
    slice_sizes = params["slice_sizes"]
    limit_list = [s + sz for s, sz in zip(start_list, slice_sizes)]

    # ``_slice_elementals`` returns one elemental for its single primal
    # (``[operand]``); the start-index invars carry no gradient (positions past
    # the returned list are skipped by ``_build_graph``).
    return val_out, _slice_elementals(
        [operand], val_out,
        start_indices=tuple(start_list),
        limit_indices=tuple(limit_list),
        strides=None,
    )


elemental_rules[lax.dynamic_slice_p] = dynamic_slice_elemental_rule


# ========================= dynamic_update_slice ============================
#
# out = operand with `update` written into the window [start, start+up_shape).
#   d out / d operand  : identity, zeroed inside the window (a Diagonal `mask` —
#                        already structural & O(n), kept verbatim).
#   d out / d update   : an EMBEDDING of the update into the output window. Its
#                        forward is "pad the update block into the output", its
#                        adjoint is "slice the output cotangent back to the
#                        window" — both O(update), seed-drainable.


def _dus_update_elementals(operand, update, val_out, start_list):
    op_shape = get_shape(operand)
    up_shape = get_shape(update)
    out_shape = get_shape(val_out)
    ndim = get_ndim(val_out)

    # ---- d out / d update : seed-drainable embedding transform -------------
    # forward(pre): pre.out_dims span the UPDATE shape -> pad each out axis to
    #   the output window (low=start, high=out-start-up).
    # inverse(post): post.primal_dims span the OUTPUT shape -> slice each primal
    #   axis to the window [start, start+up).
    def up_forward(pre):
        if _is_scalar_identity_post(pre):
            pre = _identity_post_over(pre, up_shape, val_out.dtype)
        full = pre.dense()                       # (up_shape..., primal...)
        pad_config = [(start_list[ax], out_shape[ax] - start_list[ax] - up_shape[ax], 0)
                      for ax in range(ndim)]
        pad_config += [(0, 0, 0)] * (full.ndim - ndim)
        new_val = lax.pad(full, jnp.array(0.0, dtype=full.dtype), pad_config)
        new_out, new_primal = _dense_grid(out_shape, pre.primal_shape)
        return SparseTensor(new_out, new_primal, new_val,
                            scalar_mult=pre.scalar_mult, fill_value=pre.fill_value)

    def up_inverse(post):
        if _is_scalar_identity_post(post):
            post = _identity_post_over(post, out_shape, val_out.dtype)
        full = post.dense()                      # (out_dims..., out_shape...)
        K = len(post.out_dims)
        sl = [slice(None)] * full.ndim
        for ax in range(ndim):
            sl[K + ax] = slice(start_list[ax], start_list[ax] + up_shape[ax])
        new_val = full[tuple(sl)]
        new_out, new_primal = _dense_grid(post.out_shape, up_shape)
        return SparseTensor(new_out, new_primal, new_val,
                            scalar_mult=post.scalar_mult, fill_value=post.fill_value)

    transform = JacobianTransform(up_forward, up_inverse, seed_drainable=True)
    return SparseTensor([], [], None, pre_transforms=[transform])


def dynamic_update_slice_elemental_rule(primals, **params):
    val_out = lax.dynamic_update_slice_p.bind(*primals, **params)
    operand = primals[0]
    update = primals[1]
    start_indices = primals[2:]

    op_shape = get_shape(operand)
    up_shape = get_shape(update)
    out_shape = get_shape(val_out)
    ndim = get_ndim(val_out)

    start_list = [int(s) for s in start_indices]

    # d out / d operand: identity except in the updated region (where it's 0).
    # This stays a structural Diagonal `mask` — already O(n). (The mask is a
    # static boolean grid over the operand shape; no index/value tracers.)
    mask = jnp.ones(op_shape, dtype=jnp.float32)
    slices_obj = tuple(slice(s, s + sz) for s, sz in zip(start_list, up_shape))
    mask = mask.at[slices_obj].set(0.0)
    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, mask)

    # invars are (operand=0, update=1, start_0=2, ...). The start indices carry
    # no gradient; the returned list is shorter than invars, so positions past
    # the update (the integer starts) are skipped by ``_build_graph``.
    up_tensor = _dus_update_elementals(operand, update, val_out, start_list)
    return val_out, [op_tensor, up_tensor]


elemental_rules[lax.dynamic_update_slice_p] = dynamic_update_slice_elemental_rule


# ================================ gather ===================================
#
# out = gather(operand, indices). The Jacobian is a pure SELECTION:
#   forward(pre): gather pre's OUT block (operand axes) by `indices`.
#   inverse(post): SCATTER-ADD post's PRIMAL block (output axes) back to the
#                  operand positions — the adjoint, O(operand).


def gather_elemental_rule(primals, **params):
    val_out = lax.gather_p.bind(*primals, **params)
    operand = primals[0]
    indices = primals[1]

    out_shape = get_shape(val_out)
    op_shape = get_shape(operand)
    out_ndim = len(out_shape)
    op_ndim = len(op_shape)
    dtype = jnp.result_type(operand.dtype if hasattr(operand, "dtype") else jnp.float32,
                            jnp.float32)

    dn = params["dimension_numbers"]
    gather_slice_sizes = params["slice_sizes"]

    # scatter (adjoint) dimension_numbers; the operand block is the FULL gather
    # operand, the updates block is the gather OUTPUT block.
    sdn = _scatter_dn_from_gather(dn)

    def gather_forward(pre):
        if _is_scalar_identity_post(pre):
            pre = _identity_post_over(pre, op_shape, dtype)
        full = pre.dense()                       # (op_shape..., primal...)

        def g(o):
            return _do_gather(o, indices, dn, gather_slice_sizes, params)

        new_val = _apply_over_leading(full, op_ndim, g)
        new_out, new_primal = _dense_grid(out_shape, pre.primal_shape)
        return SparseTensor(new_out, new_primal, new_val,
                            scalar_mult=pre.scalar_mult, fill_value=pre.fill_value)

    def gather_inverse(post):
        if _is_scalar_identity_post(post):
            post = _identity_post_over(post, out_shape, dtype)
        full = post.dense()                      # (out_dims..., out_shape...)
        K = len(post.out_dims)
        # the PRIMAL block (gather output) is the TRAILING block of `full`; move
        # it to the front, scatter-add into operand-sized zeros, restore.
        nlead = K
        # bring trailing (output) block to front, leading (out_dims) to back
        perm = list(range(nlead, full.ndim)) + list(range(nlead))
        x = jnp.transpose(full, perm)            # (out_shape..., out_dims...)

        def s(u_arr):
            zeros = jnp.zeros(op_shape, dtype=u_arr.dtype)
            return _do_scatter_add(zeros, indices, u_arr, sdn, params)

        scattered = _apply_over_leading(x, out_ndim, s)   # (op_shape..., out_dims...)
        # move out_dims block back to the front
        nb = scattered.ndim - op_ndim
        perm2 = list(range(op_ndim, op_ndim + nb)) + list(range(op_ndim))
        new_val = jnp.transpose(scattered, perm2)         # (out_dims..., op_shape...)
        new_out, new_primal = _dense_grid(post.out_shape, op_shape)
        return SparseTensor(new_out, new_primal, new_val,
                            scalar_mult=post.scalar_mult, fill_value=post.fill_value)

    transform = JacobianTransform(gather_forward, gather_inverse, seed_drainable=True)
    return val_out, [SparseTensor([], [], None, pre_transforms=[transform])]


elemental_rules[lax.gather_p] = gather_elemental_rule


# ================================ scatter ==================================
#
# out = scatter*(operand, indices, updates).
#   d out / d operand : a Diagonal identity carrying a per-output coefficient
#                       (1 / 1-mask / updates / tie-split) — structural & O(n),
#                       built with array ops (jit-safe), kept as a SparseTensor.
#   d out / d updates : an EMBEDDING of `updates` into the output. Its forward
#                       scatters the update block into out-sized zeros; its
#                       adjoint GATHERS the output cotangent at the scatter
#                       positions — both O(updates), seed-drainable.


def _scatter_op_coeff_tensors(val_out, operand, updates, params, op_coeff):
    """Build the ``d out / d operand`` Diagonal tensor from a precomputed
    ``op_coeff`` array (shape == output shape), all via array ops."""
    out_shape = get_shape(val_out)
    op_shape = get_shape(operand)
    ndim = get_ndim(val_out)
    out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    primal_dims = [DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)]
    return SparseTensor(out_dims, primal_dims, op_coeff)


def _scatter_update_transform(val_out, operand, updates, indices, params,
                              up_value_fn):
    """Seed-drainable ``d out / d updates`` transform.

    ``up_value_fn(updates_zeros_like)`` returns a per-update coefficient array
    (== updates shape): 1 for add, -1 for sub, the operand-at-target value for
    mul, a tie-split weight for min/max. Forward scatters ``coeff * update``
    into the output window; adjoint gathers ``coeff * cotangent`` back."""
    out_shape = get_shape(val_out)
    up_shape = get_shape(updates)
    out_ndim = len(out_shape)
    up_ndim = len(up_shape)
    dtype = jnp.result_type(
        operand.dtype if hasattr(operand, "dtype") else jnp.float32, jnp.float32)
    dn = params["dimension_numbers"]

    coeff = up_value_fn()                          # (up_shape...) jit-safe array

    # adjoint gather dimension_numbers + slice_sizes (1 on inserted dims, the
    # update window-dim size on window dims).
    gdn = _gather_dn_from_scatter(dn)
    slice_sizes = _adjoint_gather_slice_sizes(dn, out_ndim, up_shape)

    def up_forward(pre):
        if _is_scalar_identity_post(pre):
            pre = _identity_post_over(pre, up_shape, dtype)
        full = pre.dense()                          # (up_shape..., primal...)
        c = coeff.reshape(up_shape + (1,) * (full.ndim - up_ndim))
        full = full * c

        def s(u_arr):
            zeros = jnp.zeros(out_shape, dtype=u_arr.dtype)
            return _do_scatter_add(zeros, indices, u_arr, dn, params)

        new_val = _apply_over_leading(full, up_ndim, s)
        new_out, new_primal = _dense_grid(out_shape, pre.primal_shape)
        return SparseTensor(new_out, new_primal, new_val,
                            scalar_mult=pre.scalar_mult, fill_value=pre.fill_value)

    def up_inverse(post):
        if _is_scalar_identity_post(post):
            post = _identity_post_over(post, out_shape, dtype)
        full = post.dense()                         # (out_dims..., out_shape...)
        K = len(post.out_dims)
        # PRIMAL block (output shape) is the trailing block; gather it down to
        # the update shape, then scale by coeff.
        perm = list(range(K, full.ndim)) + list(range(K))
        x = jnp.transpose(full, perm)               # (out_shape..., out_dims...)

        def g(o):
            return _do_gather(o, indices, gdn, slice_sizes, params)

        gathered = _apply_over_leading(x, out_ndim, g)   # (up_shape..., out_dims...)
        nb = gathered.ndim - up_ndim
        c = coeff.reshape(up_shape + (1,) * nb)
        gathered = gathered * c
        perm2 = list(range(up_ndim, up_ndim + nb)) + list(range(up_ndim))
        new_val = jnp.transpose(gathered, perm2)         # (out_dims..., up_shape...)
        new_out, new_primal = _dense_grid(post.out_shape, up_shape)
        return SparseTensor(new_out, new_primal, new_val,
                            scalar_mult=post.scalar_mult, fill_value=post.fill_value)

    transform = JacobianTransform(up_forward, up_inverse, seed_drainable=True)
    return SparseTensor([], [], None, pre_transforms=[transform])


def _scatter_op_mask(val_out, operand, updates, indices, params):
    """Per-output indicator (1 where a scatter writes, 0 elsewhere), via a
    scatter-set of ones — jit-safe, no index ``int()``."""
    out_shape = get_shape(val_out)
    up_shape = get_shape(updates)
    dn = params["dimension_numbers"]
    ones = jnp.ones(up_shape, dtype=jnp.float32)
    zeros = jnp.zeros(out_shape, dtype=jnp.float32)
    return _do_scatter_set(zeros, indices, ones, dn, params)


def _scatter_gather_at_targets(val_out, operand, updates, indices, params, source):
    """Gather ``source`` (== output shape) at the scatter target positions,
    returning an array of UPDATE shape. Used to read the operand value at each
    update's target (for mul/min/max coeffs), jit-safe."""
    out_shape = get_shape(val_out)
    up_shape = get_shape(updates)
    out_ndim = len(out_shape)
    dn = params["dimension_numbers"]
    gdn = _gather_dn_from_scatter(dn)
    slice_sizes = _adjoint_gather_slice_sizes(dn, out_ndim, up_shape)
    return _do_gather(source, indices, gdn, slice_sizes, params)


def scatter_add_elemental_rule(primals, **params):
    val_out = lax.scatter_add_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]

    # d/d(operand) = identity; d/d(updates) = one-hot embedding (coeff = 1).
    op_tensor = _scatter_op_coeff_tensors(
        val_out, operand, updates, params,
        jnp.ones(get_shape(val_out), dtype=jnp.float32),
    )
    up_tensor = _scatter_update_transform(
        val_out, operand, updates, indices, params,
        lambda: jnp.ones(get_shape(updates), dtype=jnp.float32),
    )
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_add_p] = scatter_add_elemental_rule


def scatter_sub_elemental_rule(primals, **params):
    val_out = lax.scatter_sub_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]

    # d/d(operand) = identity; d/d(updates) = -1 x one-hot embedding.
    op_tensor = _scatter_op_coeff_tensors(
        val_out, operand, updates, params,
        jnp.ones(get_shape(val_out), dtype=jnp.float32),
    )
    up_tensor = _scatter_update_transform(
        val_out, operand, updates, indices, params,
        lambda: -jnp.ones(get_shape(updates), dtype=jnp.float32),
    )
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_sub_p] = scatter_sub_elemental_rule


def scatter_set_elemental_rule(primals, **params):
    val_out = lax.scatter_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]

    # d/d(operand) = identity * (1 - mask) — zeroed at overwritten positions.
    mask = _scatter_op_mask(val_out, operand, updates, indices, params)
    op_tensor = _scatter_op_coeff_tensors(
        val_out, operand, updates, params, 1.0 - mask
    )
    # d/d(updates) = one-hot embedding (coeff = 1).
    up_tensor = _scatter_update_transform(
        val_out, operand, updates, indices, params,
        lambda: jnp.ones(get_shape(updates), dtype=jnp.float32),
    )
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_p] = scatter_set_elemental_rule


def scatter_mul_elemental_rule(primals, **params):
    val_out = lax.scatter_mul_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]

    # Product rule on out = operand * updates at the scattered positions:
    #   d/d(operand) = the updates value there (identity elsewhere),
    #   d/d(updates) = the operand value there.
    # d/d(operand): start from identity, overwrite scattered positions with the
    # updates value. Scatter-set the per-target updates value into a ones grid.
    out_shape = get_shape(val_out)
    dn = params["dimension_numbers"]
    ones_grid = jnp.ones(out_shape, dtype=jnp.float32)
    up_f32 = updates.astype(jnp.float32)
    op_coeff = _do_scatter_set(ones_grid, indices, up_f32, dn, params)
    op_tensor = _scatter_op_coeff_tensors(val_out, operand, updates, params, op_coeff)

    # d/d(updates) coeff = operand value at each update's target.
    op_at_target = _scatter_gather_at_targets(
        val_out, operand, updates, indices, params,
        operand.astype(jnp.float32),
    )
    up_tensor = _scatter_update_transform(
        val_out, operand, updates, indices, params,
        lambda: op_at_target,
    )
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_mul_p] = scatter_mul_elemental_rule


def _tie_split(a, b, greater):
    """Balanced subgradient weight (jit-safe, array-valued): 1 where a strictly
    wins, 0.5 on a tie, 0 where b wins. ``greater`` selects max (a wins when
    a>b) vs min (a wins when a<b)."""
    a = a.astype(jnp.float32)
    b = b.astype(jnp.float32)
    strict = (a > b) if greater else (a < b)
    tie = (a == b)
    return jnp.where(strict, 1.0, jnp.where(tie, 0.5, 0.0)).astype(jnp.float32)


def _scatter_minmax_rule(prim, greater):
    def rule(primals, **params):
        val_out = prim.bind(*primals, **params)
        operand, indices, updates = primals[0], primals[1], primals[2]
        out_shape = get_shape(val_out)
        dn = params["dimension_numbers"]

        # operand value at each update target, and update value broadcast onto
        # the output grid — both via the structural gather/scatter (jit-safe).
        op_at_target = _scatter_gather_at_targets(
            val_out, operand, updates, indices, params, operand.astype(jnp.float32)
        )

        # d/d(operand): identity weight everywhere, replaced at scattered
        # targets by the operand-side tie weight. Build the operand-side weight
        # on the OUTPUT grid: scatter the per-update operand-tie weight (operand
        # wins => 1 - update_side_weight at that target) over an identity grid.
        op_side_w = _tie_split(op_at_target, updates, greater)   # (up_shape)
        # where multiple updates hit one cell jax composes them sequentially;
        # for distinct indices (the common case) this scatter-set is exact.
        ones_grid = jnp.ones(out_shape, dtype=jnp.float32)
        op_coeff = _do_scatter_set(ones_grid, indices, op_side_w, dn, params)
        op_tensor = _scatter_op_coeff_tensors(
            val_out, operand, updates, params, op_coeff
        )

        # d/d(updates): update-side tie weight (update wins).
        up_side_w = _tie_split(updates, op_at_target, greater)   # (up_shape)
        up_tensor = _scatter_update_transform(
            val_out, operand, updates, indices, params,
            lambda: up_side_w,
        )
        return val_out, [op_tensor, None, up_tensor]
    return rule


scatter_min_elemental_rule = _scatter_minmax_rule(lax.scatter_min_p, greater=False)
scatter_max_elemental_rule = _scatter_minmax_rule(lax.scatter_max_p, greater=True)

elemental_rules[lax.scatter_min_p] = scatter_min_elemental_rule
elemental_rules[lax.scatter_max_p] = scatter_max_elemental_rule
