import jax.lax as lax

from .base import elemental_rules, elemental_only_rules, get_shape
from ..sparse.tensor import (
    DenseIndex,
    DiagonalIndex,
    SparseTensor,
    _swap_back_axes,
)


def _dot_general_elementals(primals, out_shape, **params):
    """Build lhs_tensor and rhs_tensor given already-known output shape."""
    lhs, rhs = primals

    # Which dimensions of the tensors are contracted
    dimension_numbers = params["dimension_numbers"][0]
    batch_dims = params["dimension_numbers"][1]
    # NOTE: Batch dimensions are just treated as DiagonalIndex.

    lhs_contracting_dims = dimension_numbers[0]
    rhs_contracting_dims = dimension_numbers[1]

    lhs_batch_dims = batch_dims[0]
    rhs_batch_dims = batch_dims[1]

    lhs_shape = list(get_shape(lhs))
    rhs_shape = list(get_shape(rhs))

    lhs_out_dims, rhs_out_dims = [], []
    lhs_primal_dims, rhs_primal_dims = [], []

    num_out_dims = len(out_shape)
    num_batch = len(lhs_batch_dims)

    # The output of ``dot_general`` always lays the batch axes out first, in the
    # order they appear in the batch tuples (``lhs_batch_dims[p]`` pairs with
    # ``rhs_batch_dims[p]`` at output position ``p``). We therefore emit the
    # batch DiagonalIndex pairs FIRST, in canonical ``p`` order, so output
    # positions 0..num_batch-1 are correct even when the batch / contracting
    # axes are permuted between lhs and rhs. (The previous encounter-order build
    # placed batch axes in lhs-/rhs-axis order, corrupting permuted-batch
    # Jacobians.)
    for p in range(num_batch):
        l_ax = lhs_batch_dims[p]
        r_ax = rhs_batch_dims[p]
        size = lhs_shape[l_ax]
        # lhs_tensor (val = rhs): out batch axis indexes the rhs val at ``r_ax``.
        lhs_out_dims.append(DiagonalIndex(p, size, r_ax, num_out_dims + l_ax))
        # rhs_tensor (val = lhs): out batch axis indexes the lhs val at ``l_ax``.
        rhs_out_dims.append(DiagonalIndex(p, size, l_ax, num_out_dims + r_ax))

    # Pre-size primal lists so contracting / free dims can be assigned by their
    # natural primal slot (``num_out_dims + axis``); batch primals are filled in
    # at the matching slot too. Placeholders are overwritten in every loop below.
    lhs_primal_dims = [None] * len(lhs_shape)
    rhs_primal_dims = [None] * len(rhs_shape)

    for p in range(num_batch):
        l_ax = lhs_batch_dims[p]
        r_ax = rhs_batch_dims[p]
        size = lhs_shape[l_ax]
        lhs_primal_dims[l_ax] = DiagonalIndex(num_out_dims + l_ax, size, r_ax, p)
        rhs_primal_dims[r_ax] = DiagonalIndex(num_out_dims + r_ax, size, l_ax, p)

    for lid, ld in enumerate(lhs_shape):
        other_lid = lid + num_out_dims
        if lid in lhs_contracting_dims:
            # Contracting dimension. Pair ``lid`` with its rhs partner
            # *positionally* — ``lhs_contracting_dims[p]`` contracts with
            # ``rhs_contracting_dims[p]`` — by looking ``lid`` up in
            # ``lhs_contracting_dims`` rather than relying on encounter order
            # (which only matches when the contracting dims are listed ascending;
            # breaks for permuted contractions). The DenseIndex carries the lhs
            # axis's own size ``ld`` and an ``axis`` pointing at the partner rhs
            # axis (the val_dim into the ``rhs`` val).
            dim = rhs_contracting_dims[lhs_contracting_dims.index(lid)]
            lhs_primal_dims[lid] = DenseIndex(other_lid, ld, dim)
        elif lid not in lhs_batch_dims:
            # Free lhs axis: appears in the output (after all batch axes) and is
            # the diagonal of the lhs-Jacobian; on the rhs side it is a plain
            # dense axis indexing the rhs val.
            _lid = len(lhs_out_dims)
            lhs_out_dims.append(DiagonalIndex(_lid, ld, None, other_lid))
            lhs_primal_dims[lid] = DiagonalIndex(other_lid, ld, None, _lid)
            rhs_out_dims.append(DenseIndex(len(rhs_out_dims), ld, lid))

    for rid, rd in enumerate(rhs_shape):
        other_rid = rid + num_out_dims
        if rid in rhs_contracting_dims:
            # Contracting dimension. Symmetric to the lhs loop: pair ``rid`` with
            # its lhs partner positionally via its index in ``rhs_contracting_dims``.
            # Carries the rhs axis's own size ``rd`` and an ``axis`` pointing at the
            # partner lhs axis (the val_dim into the ``lhs`` val).
            dim = lhs_contracting_dims[rhs_contracting_dims.index(rid)]
            rhs_primal_dims[rid] = DenseIndex(other_rid, rd, dim)
        elif rid not in rhs_batch_dims:
            # Free rhs axis: diagonal of the rhs-Jacobian, dense on the lhs side.
            _rid = len(rhs_out_dims)
            rhs_out_dims.append(DiagonalIndex(_rid, rd, None, other_rid))
            rhs_primal_dims[rid] = DiagonalIndex(other_rid, rd, None, _rid)
            lhs_out_dims.append(DenseIndex(len(lhs_out_dims), rd, rid))

    lhs_tensor = SparseTensor(lhs_out_dims, lhs_primal_dims, rhs)
    rhs_tensor = SparseTensor(rhs_out_dims, rhs_primal_dims, lhs)

    lhs_tensor = _swap_back_axes(lhs_tensor)
    rhs_tensor = _swap_back_axes(rhs_tensor)
    return [lhs_tensor, rhs_tensor]


def dot_general_elemental_rule(primals, **params):
    val_out = lax.dot_general_p.bind(*primals, **params)
    out_shape = list(get_shape(val_out))
    return val_out, _dot_general_elementals(primals, out_shape, **params)


def dot_general_elemental_only(primal_out, primals, **params):
    out_shape = list(get_shape(primal_out))
    return _dot_general_elementals(primals, out_shape, **params)


elemental_rules[lax.dot_general_p] = dot_general_elemental_rule
elemental_only_rules[lax.dot_general_p] = dot_general_elemental_only
