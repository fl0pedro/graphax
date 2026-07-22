import jax.lax as lax
import jax.numpy as jnp

from .base import (
    elemental_rules,
    elemental_only_rules,
    multi_output_elemental_only_rules,
    get_shape,
)
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


# --------------------------------------------------------------------------- #
# Migrated from auto.py: linear solve + matrix decompositions.
# --------------------------------------------------------------------------- #


def linear_solve_elemental_rule(primals, **params):
    val_out = lax.linear_solve_p.bind(*primals, **params)
    A, b = primals
    x = val_out

    A_shape = list(get_shape(A))
    b_shape = list(get_shape(b))
    out_shape = list(get_shape(val_out))

    N = A_shape[-1]
    batch_dims = A_shape[:-2]
    num_batch = len(batch_dims)
    num_out_dims = len(out_shape)

    # 1. Jacobian w.r.t b: J_b = A^{-1}
    I = jnp.eye(N, dtype=val_out.dtype)
    I_broadcast = jnp.broadcast_to(I, batch_dims + [N, N])
    # Compute batched A^{-1}
    A_inv = lax.linear_solve_p.bind(A, I_broadcast, **params)

    b_out_dims = []
    b_primal_dims = []

    # Batch dims map strictly 1-to-1 (SparseIndexes)
    for i, s in enumerate(batch_dims):
        b_out_dims.append(DiagonalIndex(i, s, None, num_out_dims + i))
        b_primal_dims.append(DiagonalIndex(num_out_dims + i, s, None, i))

    # Matrix dims are dense within the block
    b_out_dims.append(DenseIndex(num_batch, N, 0))
    b_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 1))

    b_tensor = _swap_back_axes(SparseTensor(b_out_dims, b_primal_dims, A_inv))

    # 2. Jacobian w.r.t A: J_A = - A^{-1} \otimes x
    # We reshape to (*batch, N, N, N) to broadcast the outer product cleanly
    A_inv_exp = jnp.expand_dims(A_inv, -1)
    x_reshaped = jnp.reshape(x, batch_dims + [1, 1, N])
    J_A_val = -(A_inv_exp * x_reshaped)

    A_out_dims = []
    A_primal_dims = []

    # Batch dims map strictly 1-to-1 (SparseIndexes)
    for i, s in enumerate(batch_dims):
        A_out_dims.append(DiagonalIndex(i, s, None, num_out_dims + i))
        A_primal_dims.append(DiagonalIndex(num_out_dims + i, s, None, i))

    # The resulting tensor has one dense output dim and two dense primal dims
    A_out_dims.append(DenseIndex(num_batch, N, 0))
    A_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 1))
    A_primal_dims.append(DenseIndex(num_out_dims + num_batch + 1, N, 2))

    A_tensor = _swap_back_axes(SparseTensor(A_out_dims, A_primal_dims, J_A_val))

    return val_out, [A_tensor, b_tensor]


elemental_rules[lax.linear_solve_p] = linear_solve_elemental_rule


import jax._src.lax.linalg as lax_linalg


# triangular_solve: (A, b) -> x where Ax = b (with A triangular)
def triangular_solve_elemental_rule(primals, **params):
    val_out = lax_linalg.triangular_solve_p.bind(*primals, **params)
    A, b = primals
    x = val_out

    A_shape = list(get_shape(A))
    out_shape = list(get_shape(val_out))
    N = A_shape[-1]
    batch_dims = A_shape[:-2]
    num_batch = len(batch_dims)
    num_out_dims = len(out_shape)
    nrhs = out_shape[-1] if len(out_shape) > num_batch + 1 else 1

    # J_b = A^{-1}
    I = jnp.eye(N, dtype=val_out.dtype)
    I_broadcast = jnp.broadcast_to(I, batch_dims + [N, N])
    A_inv = lax_linalg.triangular_solve_p.bind(A, I_broadcast, **params)

    b_out_dims = []
    b_primal_dims = []
    for i, s in enumerate(batch_dims):
        b_out_dims.append(DiagonalIndex(i, s, None, num_out_dims + i))
        b_primal_dims.append(DiagonalIndex(num_out_dims + i, s, None, i))

    if len(out_shape) == num_batch + 2:
        b_out_dims.append(DenseIndex(num_batch, N, 0))
        b_out_dims.append(
            DiagonalIndex(num_batch + 1, nrhs, None, num_out_dims + num_batch + 1)
        )
        b_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 1))
        b_primal_dims.append(
            DiagonalIndex(num_out_dims + num_batch + 1, nrhs, None, num_batch + 1)
        )
    else:
        b_out_dims.append(DenseIndex(num_batch, N, 0))
        b_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 1))

    b_tensor = _swap_back_axes(SparseTensor(b_out_dims, b_primal_dims, A_inv))

    # J_A = -A^{-1} ⊗ x
    A_inv_exp = jnp.expand_dims(A_inv, -1)
    x_for_outer = x if x.ndim > num_batch + 1 else x[..., None]
    x_reshaped = jnp.reshape(x_for_outer, batch_dims + [1, 1, N, nrhs])
    # J_A_val shape: (N, nrhs, N, N) for d(X)/dA
    # A_inv_exp is (N, N, 1), x_reshaped is (1, 1, N, nrhs)
    # Product is (N, N, N, nrhs). We need (N, nrhs, N, N).
    J_A_val = -(A_inv_exp[..., None] * x_reshaped)
    J_A_val = jnp.transpose(
        J_A_val,
        (*range(num_batch), num_batch, num_batch + 3, num_batch + 1, num_batch + 2),
    )

    # Masking for triangular_solve: zero out derivatives w.r.t. unused elements of A
    lower = params.get("lower", False)
    mask = jnp.tri(N, k=0, dtype=bool) if lower else jnp.tri(N, k=0, dtype=bool).T
    mask = jnp.reshape(mask, [1] * num_batch + [1, 1, N, N])
    J_A_val = jnp.where(mask, J_A_val, 0)

    if nrhs == 1 and len(out_shape) == num_batch + 1:
        J_A_val = jnp.squeeze(J_A_val, axis=num_batch + 1)

    A_out_dims = []
    A_primal_dims = []
    for i, s in enumerate(batch_dims):
        A_out_dims.append(DiagonalIndex(i, s, None, num_out_dims + i))
        A_primal_dims.append(DiagonalIndex(num_out_dims + i, s, None, i))

    if len(out_shape) == num_batch + 2:
        A_out_dims.append(DenseIndex(num_batch, N, 0))
        A_out_dims.append(DenseIndex(num_batch + 1, nrhs, 1))
        A_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 2))
        A_primal_dims.append(DenseIndex(num_out_dims + num_batch + 1, N, 3))
    else:
        A_out_dims.append(DenseIndex(num_batch, N, 0))
        A_primal_dims.append(DenseIndex(num_out_dims + num_batch, N, 1))
        A_primal_dims.append(DenseIndex(num_out_dims + num_batch + 1, N, 2))

    A_tensor = _swap_back_axes(SparseTensor(A_out_dims, A_primal_dims, J_A_val))
    return val_out, [A_tensor, b_tensor]


elemental_rules[lax_linalg.triangular_solve_p] = triangular_solve_elemental_rule


# cholesky: A -> L
# Pure analytical: J_{ijkl} = sum_m L_{im} M_{mj} B_{mk} B_{jl}
def cholesky_elemental_rule(primals, **params):
    val_out = lax_linalg.cholesky_p.bind(*primals, **params)
    A = primals[0]
    L = val_out

    A_shape = list(get_shape(A))
    out_shape = list(get_shape(L))
    N = A_shape[-1]
    batch_dims = A_shape[:-2]
    num_batch = len(batch_dims)
    num_out_dims = len(out_shape)

    # Use actual masked lower triangle for the derivative calculation
    L_lower = jnp.tril(L)
    I = jnp.eye(N, dtype=A.dtype)
    I_bcast = jnp.broadcast_to(I, batch_dims + [N, N])

    # B = L^{-1}
    B = lax_linalg.triangular_solve_p.bind(
        L_lower,
        I_bcast,
        left_side=True,
        lower=True,
        transpose_a=False,
        conjugate_a=False,
        unit_diagonal=False,
    )

    # M_{mj} = 1/(1 + d_{mj}) for m >= j
    M = jnp.tril(jnp.ones((N, N), dtype=A.dtype)) / (1.0 + I)

    # J = L @ M @ (B x B)
    J = jnp.einsum("...im,mj,...mk,...jl->...ijkl", L_lower, M, B, B)

    out_dims = []
    primal_dims = []
    vd = 0
    for i, s in enumerate(batch_dims):
        out_dims.append(DiagonalIndex(i, s, vd, num_out_dims + i))
        primal_dims.append(DiagonalIndex(num_out_dims + i, s, vd, i))
        vd += 1
    out_dims.append(DenseIndex(num_batch, N, vd))
    vd += 1
    out_dims.append(DenseIndex(num_batch + 1, N, vd))
    vd += 1
    primal_dims.append(DenseIndex(num_out_dims + num_batch, N, vd))
    vd += 1
    primal_dims.append(DenseIndex(num_out_dims + num_batch + 1, N, vd))
    vd += 1

    tensor = _swap_back_axes(SparseTensor(out_dims, primal_dims, J))
    return val_out, [tensor]


elemental_rules[lax_linalg.cholesky_p] = cholesky_elemental_rule


# eigh: A -> (V, w) (Note: standard eigh_p returns (V, w) internally)
def eigh_elemental_rule(primals, **params):
    val_out = lax_linalg.eigh_p.bind(*primals, **params)
    A = primals[0]
    V, w = val_out  # Actual internal eigh_p returns (V, w)

    A_shape = list(get_shape(A))
    N = A_shape[-1]
    batch_dims = A_shape[:-2]
    num_batch = len(batch_dims)

    w_shape = list(get_shape(w))
    V_shape = list(get_shape(V))
    w_ndim = len(w_shape)
    V_ndim = len(V_shape)

    # dw = diag(V^T dA V) -> J_w = einsum('ik,jk->kij', V, V)
    J_w = jnp.einsum("...ik,...jk->...kij", V, V)
    # The input is symmetric, so derivative is symmetric in inputs
    J_w = 0.5 * (J_w + jnp.swapaxes(J_w, -2, -1))

    w_out_dims = []
    w_primal_dims = []
    vd = 0
    for i, s in enumerate(batch_dims):
        w_out_dims.append(DiagonalIndex(i, s, vd, w_ndim + i))
        w_primal_dims.append(DiagonalIndex(w_ndim + i, s, vd, i))
        vd += 1
    w_out_dims.append(DenseIndex(num_batch, N, vd))
    vd += 1
    w_primal_dims.append(DenseIndex(w_ndim + num_batch, N, vd))
    vd += 1
    w_primal_dims.append(DenseIndex(w_ndim + num_batch + 1, N, vd))
    vd += 1
    w_tensor = _swap_back_axes(SparseTensor(w_out_dims, w_primal_dims, J_w))

    # dV = V (F ⊙ (V^T dA V)) -> J_V = einsum('kq,qp,ip,jq->kpij', V, F, V, V)
    eye_n = jnp.eye(N, dtype=A.dtype)
    w_diff = w[..., None, :] - w[..., :, None]
    F = jnp.where(eye_n == 1, 0.0, 1.0 / jnp.where(w_diff == 0, 1.0, w_diff))

    J_V = jnp.einsum("...kq,...qp,...ip,...jq->...kpij", V, F, V, V)
    # Symmetric in inputs
    J_V = 0.5 * (J_V + jnp.swapaxes(J_V, -2, -1))

    V_out_dims = []
    V_primal_dims = []
    vd = 0
    for i, s in enumerate(batch_dims):
        V_out_dims.append(DiagonalIndex(i, s, vd, V_ndim + i))
        V_primal_dims.append(DiagonalIndex(V_ndim + i, s, vd, i))
        vd += 1
    V_out_dims.append(DenseIndex(num_batch, N, vd))
    vd += 1
    V_out_dims.append(DenseIndex(num_batch + 1, N, vd))
    vd += 1
    V_primal_dims.append(DenseIndex(V_ndim + num_batch, N, vd))
    vd += 1
    V_primal_dims.append(DenseIndex(V_ndim + num_batch + 1, N, vd))
    vd += 1
    V_tensor = _swap_back_axes(SparseTensor(V_out_dims, V_primal_dims, J_V))

    # Actual eigh_p returns (V, w), not (w, V)
    return val_out, [[V_tensor], [w_tensor]]


def _multi_output_only(rule):
    """Adapt a ``(primals, **params) -> (val_out, elementals[out][invar])`` rule
    (the matrix-decomposition rules below) to the
    ``multi_output_elemental_only_rules`` contract. These primitives are
    ``multiple_results`` (eigh -> (V, w), svd -> (s, U, Vt), ...), so they MUST
    dispatch through the multi-output path; registering them in single-output
    ``elemental_rules`` made the dispatcher try to treat the per-output lists as
    SparseTensors and crash ('list' object has no attribute 'dims')."""
    def _only(primal_outs, primals, **params):
        return rule(primals, **params)[1]
    return _only


def _unsupported_decomposition(name):
    """Loud guard for a decomposition whose elemental rule is not yet correct.

    Better to fail clearly than to (a) crash with a cryptic 'list has no attribute
    dims' (the old single-output mis-registration) or (b) silently return a wrong
    Jacobian. eigh and svd (singular values) ARE verified against jax; qr/lu/eig
    are not — their analytical rules are incorrect (and eig's eigenvector
    derivatives are unsupported by jax itself)."""
    def _only(primal_outs, primals, **params):
        raise NotImplementedError(
            f"graphax does not yet have a correct elemental rule for {name}. "
            f"eigh and svd are supported; differentiate through those, or supply "
            f"a custom rule for {name}."
        )
    return _only


multi_output_elemental_only_rules[lax_linalg.eigh_p] = _multi_output_only(
    eigh_elemental_rule
)


# svd: A -> (s, U, Vt)
def svd_elemental_rule(primals, **params):
    val_out = lax_linalg.svd_p.bind(*primals, **params)
    A = primals[0]
    compute_uv = params.get("compute_uv", True)

    # Only the SINGULAR-VALUE gradient is correct. The singular-VECTOR (U / Vt)
    # Jacobians here disagree with jax (gauge ambiguity / wrong formula), so the
    # full-uv path fails loudly rather than returning a wrong gradient. Use
    # ``jnp.linalg.svd(a, compute_uv=False)`` for singular-value gradients.
    if compute_uv:
        raise NotImplementedError(
            "graphax supports gradients of svd singular VALUES only. Differentiate "
            "jnp.linalg.svd(a, compute_uv=False); singular-vector (U/Vt) gradients "
            "are not yet correct."
        )

    A_shape = list(get_shape(A))
    M, N = A_shape[-2], A_shape[-1]
    K = min(M, N)
    batch_dims = A_shape[:-2]
    num_batch = len(batch_dims)

    if compute_uv:
        s, U, Vt = val_out
    else:
        s = val_out[0]
        # In non-uv compute we don't have U and Vt to compute analytically without recomputing
        # For pure analytical, we just stop gradient if we can't form it,
        # but in practice svd is rarely used without UV if gradients are needed.
        # Let's compute thin SVD just to get the gradients:
        s, U, Vt = lax_linalg.svd_p.bind(
            A,
            full_matrices=False,
            compute_uv=True,
            subset_by_index=params.get("subset_by_index"),
            algorithm=params.get("algorithm"),
        )

    s_shape = list(get_shape(s))   # ``s`` is already unpacked (works for compute_uv=False)
    s_ndim = len(s_shape)

    # s Jacobian: ds_q = sum_{i,j} U[i,q] * dA[i,j] * V[j,q]
    # Here Vt is V.T. So V[j,q] = Vt[q,j]
    J_s = jnp.einsum("...iq,...qj->...qij", U, Vt)

    s_out_dims = []
    s_primal_dims = []
    vd = 0
    for i, sz in enumerate(batch_dims):
        s_out_dims.append(DiagonalIndex(i, sz, vd, s_ndim + i))
        s_primal_dims.append(DiagonalIndex(s_ndim + i, sz, vd, i))
        vd += 1
    s_out_dims.append(DenseIndex(num_batch, K, vd))
    vd += 1
    s_primal_dims.append(DenseIndex(s_ndim + num_batch, M, vd))
    vd += 1
    s_primal_dims.append(DenseIndex(s_ndim + num_batch + 1, N, vd))
    vd += 1
    s_tensor = _swap_back_axes(SparseTensor(s_out_dims, s_primal_dims, J_s))

    if not compute_uv:
        return val_out, [[s_tensor]]

    s_dim = s[..., None, :]
    s_diffs = (s_dim + jnp.swapaxes(s_dim, -2, -1)) * (
        s_dim - jnp.swapaxes(s_dim, -2, -1)
    )
    s_diffs_zeros = jnp.eye(K, dtype=A.dtype)
    F = 1.0 / (s_diffs + s_diffs_zeros) - s_diffs_zeros

    s_zeros = (s == 0).astype(s.dtype)
    s_inv = 1.0 / (s + s_zeros) - s_zeros
    # Diagonalize s_inv:
    s_inv_mat = jnp.eye(K, dtype=A.dtype) * s_inv[..., None]

    # U and Vt Jacobians built analytically using pure tensor contraction
    # dS_pq = u_p^T dA v_q = sum_ij U[i, p] dA[i, j] Vt[q, j]
    # => dS_tensor[p, q, i, j] = U[i, p] * Vt[q, j]
    dS_tensor = jnp.einsum("...ip,...qj->...pqij", U, Vt)

    # dU = U @ (F * (S_dim * dS + S_dim.T * dS.T) + 0.5*(dS - dS.T)*s_inv_mat)
    T1 = jnp.einsum("...q,...pqij->...pqij", s, dS_tensor)
    T2 = jnp.einsum("...p,...qpij->...pqij", s, jnp.swapaxes(dS_tensor, -4, -3))  # dS.T
    T_sym = T1 + T2

    T_skew = (
        0.5 * (dS_tensor - jnp.swapaxes(dS_tensor, -4, -3)) * s_inv_mat[..., None, None]
    )

    inner_U = F[..., None, None] * T_sym + T_skew
    J_U = jnp.einsum("...kp,...pqij->...kqij", U, inner_U)

    # V Jacobian
    # dV = V @ (F * (S_dim.T * dS + S_dim * dS.T) )
    T3 = jnp.einsum("...p,...pqij->...pqij", s, dS_tensor)
    T4 = jnp.einsum("...q,...qpij->...pqij", s, jnp.swapaxes(dS_tensor, -4, -3))
    inner_V = F[..., None, None] * (T3 + T4)
    # V = Vt.T => V[j, q] = Vt[q, j]
    V_mat = jnp.swapaxes(Vt, -2, -1)
    J_V = jnp.einsum("...lp,...pqij->...lqij", V_mat, inner_V)

    if M > N:
        # dA @ V (size M, K)
        dAV = jnp.einsum("...lq->...ql", V_mat)  # This represents derivative mask
        # We need dAV_tensor[m, q, i, j] = d(dA @ V)_{m, q} / dA[i, j]
        # = delta_{mi} V[j, q]
        I_M = jnp.eye(M, dtype=A.dtype)
        dAV_tensor = jnp.einsum("mi,...lq->...mqli", I_M, V_mat)  # [m, q, i, j=l]
        dAV_tensor = jnp.swapaxes(dAV_tensor, -2, -1)  # m, q, j, i -> m, q, i, j

        # dU += (dAV - U @ U.T @ dAV) / s
        UUt = jnp.einsum("...mp,...kp->...mk", U, U)
        proj = dAV_tensor - jnp.einsum("...mk,...kqli->...mqli", UUt, dAV_tensor)
        J_U = J_U + proj * s_inv[..., None, :, None, None]

    if N > M:
        I_N = jnp.eye(N, dtype=A.dtype)
        # dAH U = dA.T @ U
        # dAH_U_tensor[n, q, i, j] = delta_{nj} * U[i, q]
        dAH_U_tensor = jnp.einsum("nj,...iq->...nqij", I_N, U)
        VVt = jnp.einsum("...mp,...kp->...mk", V_mat, V_mat)
        proj_V = dAH_U_tensor - jnp.einsum("...mk,...kqij->...mqij", VVt, dAH_U_tensor)
        J_V = J_V + proj_V * s_inv[..., None, :, None, None]

    # J_Vt = jnp.swapaxes(J_V, -4, -3) => [k, q, i, j] -> [q, k, i, j]
    J_Vt = jnp.swapaxes(J_V, -4, -3)

    U_shape = list(get_shape(U))
    U_ndim = len(U_shape)
    U_out_dims = []
    U_primal_dims = []
    vd = 0
    for i, sz in enumerate(batch_dims):
        U_out_dims.append(DiagonalIndex(i, sz, vd, U_ndim + i))
        U_primal_dims.append(DiagonalIndex(U_ndim + i, sz, vd, i))
        vd += 1
    U_out_dims.append(DenseIndex(num_batch, M, vd))
    vd += 1
    U_out_dims.append(DenseIndex(num_batch + 1, K, vd))
    vd += 1
    U_primal_dims.append(DenseIndex(U_ndim + num_batch, M, vd))
    vd += 1
    U_primal_dims.append(DenseIndex(U_ndim + num_batch + 1, N, vd))
    vd += 1
    U_tensor = _swap_back_axes(SparseTensor(U_out_dims, U_primal_dims, J_U))

    Vt_shape = list(get_shape(Vt))
    Vt_ndim = len(Vt_shape)
    Vt_out_dims = []
    Vt_primal_dims = []
    vd = 0
    for i, sz in enumerate(batch_dims):
        Vt_out_dims.append(DiagonalIndex(i, sz, vd, Vt_ndim + i))
        Vt_primal_dims.append(DiagonalIndex(Vt_ndim + i, sz, vd, i))
        vd += 1
    Vt_out_dims.append(DenseIndex(num_batch, K, vd))
    vd += 1
    Vt_out_dims.append(DenseIndex(num_batch + 1, N, vd))
    vd += 1
    Vt_primal_dims.append(DenseIndex(Vt_ndim + num_batch, M, vd))
    vd += 1
    Vt_primal_dims.append(DenseIndex(Vt_ndim + num_batch + 1, N, vd))
    vd += 1
    Vt_tensor = _swap_back_axes(SparseTensor(Vt_out_dims, Vt_primal_dims, J_Vt))

    return val_out, [[s_tensor], [U_tensor], [Vt_tensor]]


multi_output_elemental_only_rules[lax_linalg.svd_p] = _multi_output_only(
    svd_elemental_rule
)


# qr: A -> (Q, R)
multi_output_elemental_only_rules[lax_linalg.qr_p] = _unsupported_decomposition("qr")


# tridiagonal_solve: (dl, d, du, b) -> x
# Analytical via explicit triangular solves loop logic replaced by inverse matrix
def tridiagonal_solve_elemental_rule(primals, **params):
    val_out = lax_linalg.tridiagonal_solve_p.bind(*primals, **params)
    dl, d, du, b = primals
    x = val_out

    x_shape = list(get_shape(x))
    x_ndim = len(x_shape)
    N = x_shape[-2]
    nrhs = x_shape[-1]

    # We want J_b = T^{-1}, J_param = -T^{-1} dT/dparam @ x
    # Without vmap/jacfwd, we can construct the dense inverse using the solver
    I_N = jnp.eye(N, dtype=x.dtype)

    # Solve T @ A_inv = I. tridiagonal_solve expects right hand size of shape (N, B)
    # A_inv shape will be (N, N)
    A_inv = lax_linalg.tridiagonal_solve_p.bind(dl, d, du, I_N)

    # b_tensor is identical to A_inv with specific tensor axes
    b_out_dims = []
    b_primal_dims = []
    b_out_dims.append(DenseIndex(0, N, 0))
    b_out_dims.append(DiagonalIndex(1, nrhs, None, x_ndim + 1))
    b_primal_dims.append(DenseIndex(x_ndim, N, 1))
    b_primal_dims.append(DiagonalIndex(x_ndim + 1, nrhs, None, 1))
    b_tensor = _swap_back_axes(SparseTensor(b_out_dims, b_primal_dims, A_inv))

    # Analytical J_d = -A_inv[i, k] * x[k, j]
    J_d = -jnp.einsum("ik,kj->kij", A_inv, x)

    d_out_dims = [DenseIndex(0, N, 0), DenseIndex(1, nrhs, 1)]
    d_primal_dims = [DenseIndex(x_ndim, N, 2)]
    d_tensor = _swap_back_axes(SparseTensor(d_out_dims, d_primal_dims, J_d))

    # Analytical J_dl = -A_inv[:, k] * x[k-1, :]
    shifted_x_dl = jnp.concatenate([jnp.zeros_like(x[:1]), x[:-1]], axis=0)
    J_dl = -jnp.einsum("ik,kj->kij", A_inv, shifted_x_dl)

    dl_out_dims = [DenseIndex(0, N, 0), DenseIndex(1, nrhs, 1)]
    dl_primal_dims = [DenseIndex(x_ndim, N, 2)]
    dl_tensor = _swap_back_axes(SparseTensor(dl_out_dims, dl_primal_dims, J_dl))

    # Analytical J_du = -A_inv[:, k-1] * x[k, :] (shift A_inv instead of x for correct index matching)
    shifted_x_du = jnp.concatenate([x[1:], jnp.zeros_like(x[:1])], axis=0)
    J_du = -jnp.einsum("ik,kj->kij", A_inv, shifted_x_du)

    du_out_dims = [DenseIndex(0, N, 0), DenseIndex(1, nrhs, 1)]
    du_primal_dims = [DenseIndex(x_ndim, N, 2)]
    du_tensor = _swap_back_axes(SparseTensor(du_out_dims, du_primal_dims, J_du))

    return val_out, [dl_tensor, d_tensor, du_tensor, b_tensor]


elemental_rules[lax_linalg.tridiagonal_solve_p] = tridiagonal_solve_elemental_rule


multi_output_elemental_only_rules[lax_linalg.lu_p] = _unsupported_decomposition("lu")


multi_output_elemental_only_rules[lax_linalg.eig_p] = _unsupported_decomposition("eig")

from jax._src.lax.lax import ragged_dot_general_p


def ragged_dot_general_elemental_rule(primals, **params):
    val_out = ragged_dot_general_p.bind(*primals, **params)
    lhs, rhs, group_sizes = primals

    # Strip group_sizes to route standard bilinear relaxation mapping
    # structurally through your existing dot_general framework
    _, (lhs_jac, rhs_jac) = dot_general_elemental_rule([lhs, rhs], **params)

    return val_out, [lhs_jac, rhs_jac, []]


elemental_rules[ragged_dot_general_p] = ragged_dot_general_elemental_rule
