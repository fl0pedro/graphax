import itertools

import jax.lax as lax
import jax.numpy as jnp

from ..sparse.tensor import (
    DenseIndex,
    DiagonalIndex,
    SparseTensor,
    _swap_back_axes,
)
from .base import elemental_rules, get_ndim, get_shape
from .transforms import JacobianTransform


def make_slice_transform(start_indices, limit_indices, out_shape):
    def slice_transform(pre):
        s_idx = list(start_indices)
        l_idx = list(limit_indices)
        full_val = jnp.array(pre)
        new_out_dims = []
        new_primal_dims = []
        counter = 0

        for s in out_shape:
            new_out_dims.append(DenseIndex(counter, s, counter))
            counter += 1

        for d in pre.primal_dims:
            new_primal_dims.append(DenseIndex(counter, d.size, counter))
            s_idx.append(0)
            l_idx.append(d.size)
            counter += 1

        new_val = lax.slice(full_val, s_idx, l_idx)
        return SparseTensor(new_out_dims, new_primal_dims, new_val)

    return slice_transform


def make_inverse_slice_transform(start_indices, limit_indices, primal0_shape):
    def inverse_slice_transform(post):
        full_val = jnp.array(post)
        new_out_dims = []
        new_primal_dims = []
        counter = 0

        for d in post.out_dims:
            new_out_dims.append(DenseIndex(counter, d.size, counter))
            counter += 1

        for s in primal0_shape:
            new_primal_dims.append(DenseIndex(counter, s, counter))
            counter += 1

        new_shape = [d.size for d in new_out_dims] + [d.size for d in new_primal_dims]
        zeros = jnp.zeros(new_shape, dtype=full_val.dtype)
        start_indices_full = [0] * len(post.out_dims) + list(start_indices)
        new_val = lax.dynamic_update_slice(zeros, full_val, start_indices_full)

        return SparseTensor(new_out_dims, new_primal_dims, new_val)

    return inverse_slice_transform


def dynamic_slice_elemental_rule(primals, **params):
    val_out = lax.dynamic_slice_p.bind(*primals, **params)
    operand = primals[0]
    start_indices = primals[1:]
    slice_sizes = params["slice_sizes"]

    start_list = [int(s) for s in start_indices]
    limit_list = [s + sz for s, sz in zip(start_list, slice_sizes)]

    transform = JacobianTransform(
        make_slice_transform(start_list, limit_list, val_out.shape),
        make_inverse_slice_transform(start_list, limit_list, operand.shape),
    )
    return val_out, [SparseTensor([], [], None, pre_transforms=[transform])]


elemental_rules[lax.dynamic_slice_p] = dynamic_slice_elemental_rule


def _build_dus_update_jac(start_list, out_shape, up_shape):
    """``d out/d update`` for dynamic_update_slice: the update is embedded into
    the output window, so ``J[out_idx, up_idx] = 1`` iff
    ``out_idx == start + up_idx`` (within the window). Built as the outer product
    of per-axis shifted-identity indicators ``E_d[i, j] = (i == start_d + j)`` via
    one einsum (vectorized, scatter-free). Shape: ``out_shape + up_shape``."""
    n = len(up_shape)
    if n == 0:
        return jnp.array(1.0, dtype=jnp.float32)
    letters = "abcdefghijklmnopqrstuvwxyz"
    Es, subs, o_letters, u_letters = [], [], [], []
    for d in range(n):
        i = jnp.arange(out_shape[d])[:, None]
        j = jnp.arange(up_shape[d])[None, :]
        Es.append((i == start_list[d] + j).astype(jnp.float32))
        o, u = letters[2 * d], letters[2 * d + 1]
        subs.append(o + u); o_letters.append(o); u_letters.append(u)
    eq = ",".join(subs) + "->" + "".join(o_letters) + "".join(u_letters)
    return jnp.einsum(eq, *Es)


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

    # Jacobian for operand: identity except in updated region (where it's 0)
    mask = jnp.ones(op_shape, dtype=jnp.float32)
    slices_obj = tuple(slice(s, s + sz) for s, sz in zip(start_list, up_shape))
    mask = mask.at[slices_obj].set(0.0)
    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, mask)

    # Jacobian for update: the update is embedded into the output window, so
    # d out/d update is an (out_shape x up_shape) embedding — NOT an operand-
    # shaped identity. The old slice-transform built it in the operand direction
    # (wrong shape/values for the update arg).
    jac = _build_dus_update_jac(start_list, out_shape, up_shape)
    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    return val_out, [op_tensor, up_tensor]


elemental_rules[lax.dynamic_update_slice_p] = dynamic_update_slice_elemental_rule


def gather_elemental_rule(primals, **params):
    val_out = lax.gather_p.bind(*primals, **params)
    operand = primals[0]
    indices = primals[1]

    out_shape = get_shape(val_out)
    op_shape = get_shape(operand)
    out_ndim = len(out_shape)
    op_ndim = len(op_shape)

    dn = params["dimension_numbers"]
    slice_sizes = params["slice_sizes"]
    offset_dims = dn.offset_dims
    collapsed_slice_dims = dn.collapsed_slice_dims
    start_index_map = dn.start_index_map

    batch_dims = [d for d in range(out_ndim) if d not in offset_dims]
    non_collapsed = [d for d in range(op_ndim) if d not in collapsed_slice_dims]

    for nc_dim in non_collapsed:
        if slice_sizes[nc_dim] != op_shape[nc_dim]:
            raise NotImplementedError(
                f"gather: partial slice along non-collapsed dim {nc_dim} not supported"
            )

    out_dims = []
    primal_dims = []
    axis_count = 0

    batch_axiss = {}
    for bd in batch_dims:
        batch_axiss[bd] = axis_count
        axis_count += 1

    collapsed_axiss = {}
    for cd in collapsed_slice_dims:
        collapsed_axiss[cd] = axis_count
        axis_count += 1

    for i in range(out_ndim):
        if i in offset_dims:
            offset_pos = list(offset_dims).index(i)
            paired_op = non_collapsed[offset_pos]
            out_dims.append(
                DiagonalIndex(i, out_shape[i], None, out_ndim + paired_op)
            )
        else:
            out_dims.append(DenseIndex(i, out_shape[i], batch_axiss[i]))

    for j in range(op_ndim):
        if j in collapsed_slice_dims:
            primal_dims.append(
                DenseIndex(out_ndim + j, op_shape[j], collapsed_axiss[j])
            )
        else:
            nc_pos = non_collapsed.index(j)
            paired_out = offset_dims[nc_pos]
            primal_dims.append(
                DiagonalIndex(out_ndim + j, op_shape[j], None, paired_out)
            )

    val_shape = [out_shape[bd] for bd in batch_dims] + [
        op_shape[cd] for cd in collapsed_slice_dims
    ]

    if len(val_shape) == 0:
        val = jnp.array(1.0, dtype=jnp.float32)
    else:
        val = jnp.zeros(val_shape, dtype=jnp.float32)

        batch_sizes = [out_shape[bd] for bd in batch_dims]
        for batch_idx in itertools.product(*(range(s) for s in batch_sizes)):
            idx_val = indices[batch_idx]
            if hasattr(idx_val, "ndim") and idx_val.ndim == 0:
                idx_val = idx_val.reshape(1)

            op_pos = []
            for cd in collapsed_slice_dims:
                if cd in start_index_map:
                    k = list(start_index_map).index(cd)
                    pos = (
                        int(idx_val[k])
                        if hasattr(idx_val, "__getitem__")
                        else int(idx_val)
                    )
                else:
                    pos = 0
                op_pos.append(pos)

            full_idx = tuple(batch_idx) + tuple(op_pos)
            val = val.at[full_idx].set(1.0)

    tensor = _swap_back_axes(SparseTensor(out_dims, primal_dims, val))
    return val_out, [tensor]


elemental_rules[lax.gather_p] = gather_elemental_rule


def _update_to_output_index(up_idx, indices, dn, operand_ndim, up_ndim):
    """Map an update-array multi-index to the operand position it writes to.

    XLA scatter places each window at ``start + window_offset``: the start comes
    from the scatter index vector (selected by the update's non-window dims) and
    the offset from the update's window dims. The previous version set
    out_idx = start OR offset (never both) and skipped the start entirely when
    every update dim was a window dim (a slice scatter, e.g. ``x.at[1:2].max(u)``),
    so the whole Jacobian was mis-placed at offset 0 instead of the slice start."""
    update_window_dims = dn.update_window_dims
    inserted_window_dims = dn.inserted_window_dims
    scatter_dims_to_operand_dims = dn.scatter_dims_to_operand_dims

    # The update's non-window dims select WHICH window (its index vector).
    scatter_dims = [d for d in range(up_ndim) if d not in update_window_dims]
    scatter_idx = tuple(up_idx[d] for d in scatter_dims)
    idx_vec = jnp.reshape(indices[scatter_idx], (-1,))

    out_idx = [0] * operand_ndim
    # Base: scatter start for each operand dim named by scatter_dims_to_operand_dims.
    for k, op_dim in enumerate(scatter_dims_to_operand_dims):
        out_idx[op_dim] = int(idx_vec[k])
    # Offset: add each window dim's position within the window.
    non_inserted = [d for d in range(operand_ndim) if d not in inserted_window_dims]
    for w_i, w_dim in enumerate(update_window_dims):
        out_idx[non_inserted[w_i]] += int(up_idx[w_dim])

    return tuple(out_idx)


def _build_scatter_update_jac(indices, out_shape, up_shape, params, coeff=None):
    ndim_out = len(out_shape)
    ndim_up = len(up_shape)
    dn = params["dimension_numbers"]
    jac_shape = list(out_shape) + list(up_shape)
    jac = jnp.zeros(jac_shape, dtype=jnp.float32)


    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(up_idx, indices, dn, ndim_out, ndim_up)
        full_idx = tuple(out_idx) + tuple(up_idx)
        val = (
            1.0
            if coeff is None
            else float(coeff[up_idx])
            if hasattr(coeff, "__getitem__")
            else float(coeff)
        )
        jac = jac.at[full_idx].set(val)

    return jac


def _build_scatter_mask(indices, out_shape, up_shape, params):
    ndim_out = len(out_shape)
    ndim_up = len(up_shape)
    dn = params["dimension_numbers"]
    mask = jnp.zeros(out_shape, dtype=jnp.float32)


    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(up_idx, indices, dn, ndim_out, ndim_up)
        mask = mask.at[out_idx].set(1.0)

    return mask


def scatter_add_elemental_rule(primals, **params):
    val_out = lax.scatter_add_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]
    out_shape, op_shape, up_shape = (
        get_shape(val_out),
        get_shape(operand),
        get_shape(updates),
    )
    ndim = get_ndim(val_out)

    # d/d(operand) = identity
    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(
        op_out_dims, op_primal_dims, jnp.ones(out_shape, dtype=jnp.float32)
    )

    # d/d(updates) = one-hot embedding (coeff = 1)
    jac = _build_scatter_update_jac(indices, out_shape, up_shape, params)
    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    # invars are (operand=0, indices=1, updates=2); core indexes elementals by
    # eqn.invars position, so the updates Jacobian must sit at slot 2 with a None
    # for the non-differentiable integer indices (else it is silently dropped).
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_add_p] = scatter_add_elemental_rule


def scatter_sub_elemental_rule(primals, **params):
    val_out = lax.scatter_sub_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]
    out_shape, op_shape, up_shape = (
        get_shape(val_out),
        get_shape(operand),
        get_shape(updates),
    )
    ndim = get_ndim(val_out)

    # d/d(operand) = identity
    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(
        op_out_dims, op_primal_dims, jnp.ones(out_shape, dtype=jnp.float32)
    )

    # d/d(updates) = -1 × one-hot embedding
    jac = _build_scatter_update_jac(indices, out_shape, up_shape, params, coeff=-1.0)
    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    # invars are (operand=0, indices=1, updates=2); core indexes elementals by
    # eqn.invars position, so the updates Jacobian must sit at slot 2 with a None
    # for the non-differentiable integer indices (else it is silently dropped).
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_sub_p] = scatter_sub_elemental_rule


def scatter_set_elemental_rule(primals, **params):
    val_out = lax.scatter_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]
    out_shape, op_shape, up_shape = (
        get_shape(val_out),
        get_shape(operand),
        get_shape(updates),
    )
    ndim = get_ndim(val_out)

    # d/d(operand) = identity * (1 - mask) — zeroed at overwritten positions
    mask = _build_scatter_mask(indices, out_shape, up_shape, params)
    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, 1.0 - mask)

    # d/d(updates) = one-hot embedding (coeff = 1)
    jac = _build_scatter_update_jac(indices, out_shape, up_shape, params)
    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    # invars are (operand=0, indices=1, updates=2); core indexes elementals by
    # eqn.invars position, so the updates Jacobian must sit at slot 2 with a None
    # for the non-differentiable integer indices (else it is silently dropped).
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_p] = scatter_set_elemental_rule


def scatter_mul_elemental_rule(primals, **params):
    val_out = lax.scatter_mul_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]
    out_shape, op_shape, up_shape = (
        get_shape(val_out),
        get_shape(operand),
        get_shape(updates),
    )
    ndim = get_ndim(val_out)
    dn = params["dimension_numbers"]

    # d/d(operand): identity at non-scattered; updates value at scattered
    # Build per-element operand coefficient
    op_coeff = jnp.ones(out_shape, dtype=jnp.float32)

    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        op_coeff = op_coeff.at[out_idx].set(float(updates[up_idx]))

    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, op_coeff)

    # d/d(updates) = operand value at the scattered output positions (product
    # rule: out = operand * updates there, so d out/d updates = operand).
    jac_shape = list(out_shape) + list(up_shape)
    jac2 = jnp.zeros(jac_shape, dtype=jnp.float32)
    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        full_idx = tuple(out_idx) + tuple(up_idx)
        jac2 = jac2.at[full_idx].set(float(operand[out_idx]))

    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac2)

    # See scatter_add: updates Jacobian at slot 2, None for the integer indices.
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_mul_p] = scatter_mul_elemental_rule


def _min_tie_split(a, b):
    """Subgradient weight for ``a`` in ``min(a, b)``: 1 if a<b, 0.5 if a==b
    (balanced, matching JAX), 0 if a>b."""
    return 1.0 if a < b else (0.5 if a == b else 0.0)


def _max_tie_split(a, b):
    """Subgradient weight for ``a`` in ``max(a, b)``: 1 if a>b, 0.5 if a==b, 0."""
    return 1.0 if a > b else (0.5 if a == b else 0.0)


def scatter_min_elemental_rule(primals, **params):
    val_out = lax.scatter_min_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]
    out_shape, op_shape, up_shape = (
        get_shape(val_out),
        get_shape(operand),
        get_shape(updates),
    )
    ndim = get_ndim(val_out)
    dn = params["dimension_numbers"]

    # d/d(operand): 1 where operand <= updates at scattered pos, 1 elsewhere
    op_coeff = jnp.ones(out_shape, dtype=jnp.float32)

    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        # Balanced subgradient at a tie (operand == update): 0.5 to each side,
        # matching JAX. Was operand<=update -> 1.0 (operand took all the credit).
        op_coeff = op_coeff.at[out_idx].set(
            _min_tie_split(operand[out_idx], updates[up_idx])
        )

    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, op_coeff)

    # d/d(updates): 1 where updates < operand, 0.5 at a tie
    jac_shape = list(out_shape) + list(up_shape)
    jac = jnp.zeros(jac_shape, dtype=jnp.float32)
    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        indicator = _min_tie_split(updates[up_idx], operand[out_idx])
        full_idx = tuple(out_idx) + tuple(up_idx)
        jac = jac.at[full_idx].set(indicator)

    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    # invars are (operand=0, indices=1, updates=2); core indexes elementals by
    # eqn.invars position, so the updates Jacobian must sit at slot 2 with a None
    # for the non-differentiable integer indices (else it is silently dropped).
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_min_p] = scatter_min_elemental_rule


def scatter_max_elemental_rule(primals, **params):
    val_out = lax.scatter_max_p.bind(*primals, **params)
    operand, indices, updates = primals[0], primals[1], primals[2]
    out_shape, op_shape, up_shape = (
        get_shape(val_out),
        get_shape(operand),
        get_shape(updates),
    )
    ndim = get_ndim(val_out)
    dn = params["dimension_numbers"]

    # d/d(operand): 1 where operand >= updates at scattered pos, 1 elsewhere
    op_coeff = jnp.ones(out_shape, dtype=jnp.float32)

    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        # Balanced subgradient at a tie (operand == update): 0.5 each (was 1.0
        # operand / 0.0 update).
        op_coeff = op_coeff.at[out_idx].set(
            _max_tie_split(operand[out_idx], updates[up_idx])
        )

    op_out_dims = [DiagonalIndex(i, s, i, ndim + i) for i, s in enumerate(out_shape)]
    op_primal_dims = [
        DiagonalIndex(ndim + i, s, i, i) for i, s in enumerate(op_shape)
    ]
    op_tensor = SparseTensor(op_out_dims, op_primal_dims, op_coeff)

    # d/d(updates): 1 where updates > operand, 0.5 at a tie
    jac_shape = list(out_shape) + list(up_shape)
    jac = jnp.zeros(jac_shape, dtype=jnp.float32)
    for up_idx in itertools.product(*(range(s) for s in up_shape)):
        out_idx = _update_to_output_index(
            up_idx, indices, dn, len(out_shape), len(up_shape)
        )
        indicator = _max_tie_split(updates[up_idx], operand[out_idx])
        full_idx = tuple(out_idx) + tuple(up_idx)
        jac = jac.at[full_idx].set(indicator)

    up_out_dims = [DenseIndex(i, s, i) for i, s in enumerate(out_shape)]
    up_primal_dims = [
        DenseIndex(ndim + i, s, ndim + i) for i, s in enumerate(up_shape)
    ]
    up_tensor = SparseTensor(up_out_dims, up_primal_dims, jac)

    # invars are (operand=0, indices=1, updates=2); core indexes elementals by
    # eqn.invars position, so the updates Jacobian must sit at slot 2 with a None
    # for the non-differentiable integer indices (else it is silently dropped).
    return val_out, [op_tensor, None, up_tensor]


elemental_rules[lax.scatter_max_p] = scatter_max_elemental_rule
