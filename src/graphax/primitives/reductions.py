
import numpy as np
import jax.lax as lax
import jax.numpy as jnp

from .base import (
    NO_EDGE,
    elemental_rules,
    elemental_only_rules,
    multi_output_elemental_only_rules,
    get_ndim,
    get_shape,
)
from ..sparse.tensor import (
    DenseIndex,
    DiagonalIndex,
    SparseTensor,
    _swap_back_axes,
)


# ---------- select_n ----------

def _select_elementals(primals, val_out, **params):
    # select_n(which, *cases) picks element-wise from `cases` indexed by `which`.
    # Jacobian wrt case_k is the identity masked by (which == k). `which` is
    # integer-valued and non-differentiable: when it comes from a Var we emit
    # NO_EDGE so the dispatcher keeps alignment but adds no edge; when it comes
    # from a Literal the dispatcher already drops its slot, so we omit it here.
    which, *cases = primals
    out_shape = get_shape(val_out)
    out_ndim = len(out_shape)
    out_dtype = val_out.dtype

    def _masked_identity(mask):
        if out_ndim == 0:
            return SparseTensor([], [], mask)
        # Diagonal Jacobian whose diagonal carries the per-element mask values, so
        # each DiagonalIndex must read its own axis of ``mask`` (axis=i). axis=None
        # marked the dim as a pure (value-less) identity, which DISCARDED the mask
        # -> select_n/where Jacobians were silently wrong (and mis-shaped when the
        # same input fed multiple branches).
        out_dims = [
            DiagonalIndex(i, s, i, out_ndim + i) for i, s in enumerate(out_shape)
        ]
        primal_dims = [
            DiagonalIndex(out_ndim + i, s, i, i) for i, s in enumerate(out_shape)
        ]
        return SparseTensor(out_dims, primal_dims, mask)

    elementals = [NO_EDGE]
    for k, _ in enumerate(cases):
        mask = (which == k).astype(out_dtype)
        elementals.append(_masked_identity(mask))
    return elementals


def select_elemental_rule(primals, **params):
    val_out = lax.select_n_p.bind(*primals, **params)
    return val_out, _select_elementals(primals, val_out, **params)


def select_elemental_only(primal_out, primals, **params):
    return _select_elementals(primals, primal_out, **params)


elemental_rules[lax.select_n_p] = select_elemental_rule
elemental_only_rules[lax.select_n_p] = select_elemental_only


# ---------- reduce_sum ----------

# TODO Create a general reduce rule with a custom derivative!
def _reduce_sum_elementals(primals, val_out_ndim, **params):
    primal = primals[0]
    axes = params["axes"]

    new_out_dims, new_primal_dims, shape = [], [], []
    reduce_all = axes is None
    if reduce_all:
        # Full reduction -> scalar output (one size-1 out axis). The append used
        # to run before this list existed -> UnboundLocalError (latent: jnp.sum
        # always lowers explicit axes, never None).
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseIndex(0, 1, 0))
    elif isinstance(axes, int):
        axes = (axes,)

    l = val_out_ndim
    base = 1 if reduce_all else l  # contiguous ids; see _reduce_extremum_elementals
    count = 0
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            new_primal_dims.append(DenseIndex(base + i, size, count))
            shape.append(size)
            count += 1
        else:
            ll = len(new_out_dims)
            new_out_dims.append(DiagonalIndex(ll, size, None, l + i))
            new_primal_dims.append(DiagonalIndex(l + i, size, None, ll))

    val = jnp.ones(shape, dtype=jnp.float32)
    return [SparseTensor(new_out_dims, new_primal_dims, val)]


def reduce_sum_elemental_rule(primals, **params):
    val_out = lax.reduce_sum_p.bind(*primals, **params)
    return val_out, _reduce_sum_elementals(primals, get_ndim(val_out), **params)


def reduce_sum_elemental_only(primal_out, primals, **params):
    return _reduce_sum_elementals(primals, get_ndim(primal_out), **params)


elemental_rules[lax.reduce_sum_p] = reduce_sum_elemental_rule
elemental_only_rules[lax.reduce_sum_p] = reduce_sum_elemental_only


# ---------- reduce_max / reduce_min ----------

def _reduce_extremum_elementals(primals, val_out, **params):
    """Shared elemental for BOTH reduce_max and reduce_min.

    The subgradient is the indicator of the positions that achieved the extremum
    — ``primal == val_out`` — normalized so a non-unique extremum splits the
    gradient equally across the ties. This is identical for max and min because
    ``val_out`` already IS the achieved extremum, so equality selects the
    argmax / argmin positions either way."""
    primal = primals[0]
    axes = params["axes"]
    shape = list(get_shape(val_out))

    new_out_dims, new_primal_dims, _shape = [], [], []
    reduce_all = axes is None
    if reduce_all:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseIndex(0, 1, 0, True))
    elif isinstance(axes, int):
        axes = (axes,)

    l = get_ndim(val_out)  # number of kept (out) axes
    # Contiguous ids: every primal axis gets ``base + i`` (base = #out dims),
    # matching the kept DiagonalIndex pairs. The old ``len(out)+len(primal)``
    # counter collided with a kept axis following a reduced one on >=3D
    # (ids like [0,1,2,2,4] -> Topology Error).
    base = 1 if reduce_all else l
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            shape.insert(i, 1)
            new_primal_dims.append(DenseIndex(base + i, size, i))
            _shape.append(size)
        else:
            ll = len(new_out_dims)
            new_out_dims.append(DiagonalIndex(ll, size, i, l + i))
            new_primal_dims.append(DiagonalIndex(l + i, size, i, ll))

    # Reshape val_out with size-1 at reduced axes so the equality broadcasts
    # against the full-shape primal.
    _val_out = val_out.reshape(shape)
    new_val = jnp.where(primal == _val_out, 1, 0)
    # NOTE: Normalization is important if the extremum is not unique (ties split).
    norm = jnp.sum(new_val, axis=axes, keepdims=True)
    new_val = new_val / norm
    return [_swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))]


def reduce_max_elemental_rule(primals, **params):
    val_out = lax.reduce_max_p.bind(*primals, **params)
    return val_out, _reduce_extremum_elementals(primals, val_out, **params)


def reduce_max_elemental_only(primal_out, primals, **params):
    return _reduce_extremum_elementals(primals, primal_out, **params)


elemental_rules[lax.reduce_max_p] = reduce_max_elemental_rule
elemental_only_rules[lax.reduce_max_p] = reduce_max_elemental_only


def reduce_min_elemental_rule(primals, **params):
    val_out = lax.reduce_min_p.bind(*primals, **params)
    return val_out, _reduce_extremum_elementals(primals, val_out, **params)


def reduce_min_elemental_only(primal_out, primals, **params):
    return _reduce_extremum_elementals(primals, primal_out, **params)


elemental_rules[lax.reduce_min_p] = reduce_min_elemental_rule
elemental_only_rules[lax.reduce_min_p] = reduce_min_elemental_only


# --------------------------------------------------------------------------- #
# Migrated from auto.py: reduce_prod, cumulative ops, sort, rev.
# --------------------------------------------------------------------------- #


def reduce_prod_elemental_rule(primals, **params):
    """d prod / d x_i = product of the OTHER reduced elements. Same dim layout as
    reduce_max (kept axes DiagonalIndex, reduced axes DenseIndex); the val is the
    per-element product-of-others, robust to zeros."""
    val_out = lax.reduce_prod_p.bind(*primals, **params)
    primal = primals[0]
    axes = params["axes"]
    shape = list(get_shape(val_out))

    new_out_dims, new_primal_dims = [], []
    reduce_all = axes is None
    if reduce_all:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseIndex(0, 1, 0))
    elif isinstance(axes, int):
        axes = (axes,)

    l = get_ndim(val_out)
    base = 1 if reduce_all else l
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            shape.insert(i, 1)
            new_primal_dims.append(DenseIndex(base + i, size, i))
        else:
            ll = len(new_out_dims)
            new_out_dims.append(DiagonalIndex(ll, size, i, l + i))
            new_primal_dims.append(DiagonalIndex(l + i, size, i, ll))

    # prod_others, robust to zeros: tot/x_i for non-zero x_i; for x_i == 0,
    # the product of the rest when it is the UNIQUE zero, else 0.
    tot = val_out.reshape(shape)                       # full product (0 if any 0)
    n_zeros = jnp.sum(primal == 0, axis=axes, keepdims=True)
    prod_nonzero = jnp.prod(jnp.where(primal == 0, 1.0, primal), axis=axes, keepdims=True)
    safe = jnp.where(primal != 0, primal, 1.0)
    new_val = jnp.where(
        primal != 0, tot / safe,
        jnp.where(n_zeros == 1, prod_nonzero, 0.0),
    ).astype(jnp.float32)
    return val_out, [
        _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))
    ]


elemental_rules[lax.reduce_prod_p] = reduce_prod_elemental_rule


def _cumsum_elementals(primals, val_out, **params):
    """Cumulative sum (``out[i] = sum_{j<=i} x[j]``) as a SEED-DRAINABLE deferred
    transform. cumsum is LINEAR with a triangular Jacobian ``L`` whose adjoint
    ``Lᵀ`` is the REVERSE-direction cumsum, so ``post @ cumsum(pre) ==
    rev_cumsum(post) @ pre``: in reverse the elimination drains it onto the
    cotangent VECTOR (one reverse-cumsum, O(n)) instead of materialising the full
    n×n triangular block. The forward / non-drainable path densifies."""
    from .transforms import (
        JacobianTransform, _dense_grid,
        _is_scalar_identity_post, _identity_post_over,
    )
    axis = params["axis"]
    reverse = params.get("reverse", False)

    def cumsum_transform(pre):                   # forward: cumsum along the OUT axis
        full = lax.cumsum(pre.dense(), axis=axis, reverse=reverse)
        new_out, new_primal = _dense_grid(pre.out_shape, pre.primal_shape)
        return SparseTensor(new_out, new_primal, full,
                            scalar_mult=pre.scalar_mult, fill_value=pre.fill_value)

    def inverse_cumsum_transform(post):          # adjoint: REVERSE-cumsum, PRIMAL axis
        if _is_scalar_identity_post(post):
            post = _identity_post_over(post, val_out.shape, val_out.dtype)
        n_out = len(post.out_dims)
        full = lax.cumsum(post.dense(), axis=n_out + axis, reverse=not reverse)
        new_out, new_primal = _dense_grid(post.out_shape, post.primal_shape)
        return SparseTensor(new_out, new_primal, full,
                            scalar_mult=post.scalar_mult, fill_value=post.fill_value)

    transform = JacobianTransform(cumsum_transform, inverse_cumsum_transform,
                                  seed_drainable=True)
    return [SparseTensor([], [], None, pre_transforms=[transform])]


def cumsum_elemental_rule(primals, **params):
    val_out = lax.cumsum_p.bind(*primals, **params)
    return val_out, _cumsum_elementals(primals, val_out, **params)


def cumsum_elemental_only(primal_out, primals, **params):
    return _cumsum_elementals(primals, primal_out, **params)


elemental_rules[lax.cumsum_p] = cumsum_elemental_rule
elemental_only_rules[lax.cumsum_p] = cumsum_elemental_only


def sort_elemental_rule(primals, **params):
    """Single-operand ``sort``: ``out = x[argsort(x)]`` along ``dimension``. The
    Jacobian is the selection (permutation) matrix from the argsort indices on
    the sort axis; other axes are independent (DiagonalIndex). ``sort_p`` is
    multiple_results, hence the ``[tensor]`` (one per output) return shape."""
    operand = primals[0]
    dim = params["dimension"]
    num_keys = params.get("num_keys", 1)
    if len(primals) != 1 or num_keys != 1:
        raise NotImplementedError(
            "sort_elemental_rule supports the single-operand case only "
            f"(got {len(primals)} operands, num_keys={num_keys})."
        )
    val_out_list = lax.sort_p.bind(*primals, **params)
    shape = get_shape(operand)
    n = len(shape)
    S = shape[dim]

    perm = jnp.argsort(operand, axis=dim)
    perm_e = jnp.expand_dims(perm, dim + 1)
    j_idx = jnp.arange(S).reshape([S if k == dim + 1 else 1 for k in range(n + 1)])
    indicator = (perm_e == j_idx).astype(jnp.float32)  # out[i] picks in[perm[i]]

    new_out_dims, new_primal_dims = [], []
    for i, size in enumerate(shape):
        if i == dim:
            new_out_dims.append(DenseIndex(i, size, dim))
            new_primal_dims.append(DenseIndex(n + i, size, dim + 1))
        else:
            vd = i if i < dim else i + 1   # val axis (shifted past the extra in-axis)
            new_out_dims.append(DiagonalIndex(i, size, vd, n + i))
            new_primal_dims.append(DiagonalIndex(n + i, size, vd, i))
    st = _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, indicator))
    return val_out_list, [st]


elemental_rules[lax.sort_p] = sort_elemental_rule


def _rev_elementals(primals, val_out, **params):
    """``rev`` (``jnp.flip``) as a SEED-DRAINABLE deferred transform. flip is a
    reverse permutation (``out[i] = x[n-1-i]``) and is SELF-ADJOINT (R = Rᵀ), so
    ``post @ flip(pre) == flip(post) @ pre``: in reverse the elimination drains it
    onto the cotangent VECTOR (flip the vector, O(n)) instead of materialising the
    full n×n anti-diagonal R. The forward / non-drainable path densifies."""
    from .transforms import (
        JacobianTransform, _dense_grid,
        _is_scalar_identity_post, _identity_post_over,
    )
    dims = params["dimensions"]

    def rev_transform(pre):
        full = pre.dense()                       # (out..., primal...) in dim order
        for ax in dims:                          # flip the OUT axes (0..n_out-1)
            full = jnp.flip(full, axis=ax)
        new_out, new_primal = _dense_grid(pre.out_shape, pre.primal_shape)
        return SparseTensor(new_out, new_primal, full,
                            scalar_mult=pre.scalar_mult, fill_value=pre.fill_value)

    def inverse_rev_transform(post):
        if _is_scalar_identity_post(post):
            post = _identity_post_over(post, val_out.shape, val_out.dtype)
        full = post.dense()
        n_out = len(post.out_dims)
        for ax in dims:                          # self-adjoint: flip the PRIMAL axes
            full = jnp.flip(full, axis=n_out + ax)
        new_out, new_primal = _dense_grid(post.out_shape, post.primal_shape)
        return SparseTensor(new_out, new_primal, full,
                            scalar_mult=post.scalar_mult, fill_value=post.fill_value)

    transform = JacobianTransform(rev_transform, inverse_rev_transform,
                                  seed_drainable=True)
    return [SparseTensor([], [], None, pre_transforms=[transform])]


def rev_elemental_rule(primals, **params):
    val_out = lax.rev_p.bind(*primals, **params)
    return val_out, _rev_elementals(primals, val_out, **params)


def rev_elemental_only(primal_out, primals, **params):
    return _rev_elementals(primals, primal_out, **params)


elemental_rules[lax.rev_p] = rev_elemental_rule
elemental_only_rules[lax.rev_p] = rev_elemental_only


def _cumulative_dims(shape, axis, N):
    """Dim lists for a value-DEPENDENT cumulative op (cumprod/cummax/cummin):
    the scan axis is a dense (out i, in j) pair (val axes ``axis``/``axis+1``);
    every other axis is a DiagonalIndex reading its own (shifted) val axis — the
    val carries the full per-position matrix, unlike cumsum's broadcast block."""
    out_dims, primal_dims = [], []
    for i, size in enumerate(shape):
        if i == axis:
            out_dims.append(DenseIndex(i, size, axis))
            primal_dims.append(DenseIndex(N + i, size, axis + 1))
        else:
            vd = i if i < axis else i + 1
            out_dims.append(DiagonalIndex(i, size, vd, N + i))
            primal_dims.append(DiagonalIndex(N + i, size, vd, i))
    return out_dims, primal_dims


def _cumulative_mask(n, axis, N, reverse):
    """Triangular ``(j<=i)`` mask (``j>=i`` if reverse), broadcastable to the
    full val with the scan-out axis at ``axis`` and scan-in axis at ``axis+1``."""
    idx = jnp.arange(n)
    m = (idx[None, :] >= idx[:, None]) if reverse else (idx[None, :] <= idx[:, None])
    mshape = [1] * (N + 1)
    mshape[axis] = n; mshape[axis + 1] = n
    return m.astype(jnp.float32).reshape(mshape)


def cumprod_elemental_rule(primals, **params):
    """``out[i] = prod_{j<=i} x[j]``; d out[i]/d x[j] (for j<=i) = product of the
    OTHER factors up to ``i``. For ``x[j] != 0`` that is ``out[i]/x[j]``; for
    ``x[j] == 0`` it is the cumulative product of the non-zeros up to ``i`` when
    ``x[j]`` is the UNIQUE zero in that window, else 0 (a second zero ≤ i makes
    every partial-product-of-others vanish). The naive ``out[i]/x[j]`` was a 0/0
    at zeros and dropped that contribution — wrong for any input with a zero
    (e.g. a masked / post-ReLU activation)."""
    val_out = lax.cumprod_p.bind(*primals, **params)
    x = primals[0]; axis = params["axis"]; reverse = params.get("reverse", False)
    shape = get_shape(x); N = len(shape)
    out_dims, primal_dims = _cumulative_dims(shape, axis, N)
    mask = _cumulative_mask(shape[axis], axis, N, reverse)
    out_e = jnp.expand_dims(val_out, axis + 1)         # out[i] at scan-out axis
    x_e = jnp.expand_dims(x, axis)                      # x[j] at scan-in axis
    # Cumulative (same direction as the op) zero-count and non-zero product up to i.
    is_zero = (x == 0).astype(jnp.float32)
    nz_count = lax.cumsum(is_zero, axis=axis, reverse=reverse)
    prod_nz = lax.cumprod(jnp.where(x == 0, 1.0, x).astype(jnp.float32),
                          axis=axis, reverse=reverse)
    nz_count_i = jnp.expand_dims(nz_count, axis + 1)    # zeros up to i
    prod_nz_i = jnp.expand_dims(prod_nz, axis + 1)      # prod of non-zeros up to i
    safe = jnp.where(x_e != 0, x_e, 1.0)
    deriv = jnp.where(
        x_e != 0,
        out_e / safe,                                  # 0 if another zero <= i
        jnp.where(nz_count_i == 1, prod_nz_i, 0.0),    # x[j] is the unique zero
    )
    V = (mask * deriv).astype(jnp.float32)
    return val_out, [_swap_back_axes(SparseTensor(out_dims, primal_dims, V))]


elemental_rules[lax.cumprod_p] = cumprod_elemental_rule


def cumlogsumexp_elemental_rule(primals, **params):
    """``out[i] = log(sum_{j<=i} exp(x[j]))``; d out[i]/d x[j] (for j<=i) is the
    softmax weight ``exp(x[j] - out[i])`` over the prefix window. Same triangular
    value-dependent layout as cumprod. (CTC loss / HMM-CRF forward passes.)"""
    val_out = lax.cumlogsumexp_p.bind(*primals, **params)
    x = primals[0]; axis = params["axis"]; reverse = params.get("reverse", False)
    shape = get_shape(x); N = len(shape)
    out_dims, primal_dims = _cumulative_dims(shape, axis, N)
    mask = _cumulative_mask(shape[axis], axis, N, reverse)
    out_e = jnp.expand_dims(val_out, axis + 1)         # out[i] at scan-out axis
    x_e = jnp.expand_dims(x, axis)                      # x[j] at scan-in axis
    V = (mask * jnp.exp(x_e - out_e)).astype(jnp.float32)
    return val_out, [_swap_back_axes(SparseTensor(out_dims, primal_dims, V))]


elemental_rules[lax.cumlogsumexp_p] = cumlogsumexp_elemental_rule


def _cum_extremum_rule(prim, primals, params):
    """Shared cummax/cummin: d out[i]/d x[j] = (j<=i)·[x[j]==out[i]] normalised by
    the number of tie positions <= i (matches jax's tie convention)."""
    val_out = prim.bind(*primals, **params)
    x = primals[0]; axis = params["axis"]; reverse = params.get("reverse", False)
    shape = get_shape(x); N = len(shape)
    out_dims, primal_dims = _cumulative_dims(shape, axis, N)
    mask = _cumulative_mask(shape[axis], axis, N, reverse)
    out_e = jnp.expand_dims(val_out, axis + 1)
    x_e = jnp.expand_dims(x, axis)
    hit = mask * (x_e == out_e).astype(jnp.float32)
    norm = jnp.sum(hit, axis=axis + 1, keepdims=True)  # #ties up to i
    V = (hit / norm).astype(jnp.float32)
    return val_out, [_swap_back_axes(SparseTensor(out_dims, primal_dims, V))]


def cummax_elemental_rule(primals, **params):
    return _cum_extremum_rule(lax.cummax_p, primals, params)


def cummin_elemental_rule(primals, **params):
    return _cum_extremum_rule(lax.cummin_p, primals, params)


elemental_rules[lax.cummax_p] = cummax_elemental_rule
elemental_rules[lax.cummin_p] = cummin_elemental_rule


# ---------- top_k / argmax (selection: one-hot value Jacobian; integer indices carry no grad) ----------

# top_k: returns (values, indices); values' Jacobian wrt x is a one-hot selection
# matrix, indices are integer (no gradient). multiple_results -> handled by the
# multi-output rule below.
def top_k_elemental_only(primal_outs, primals, **params):
    """Multi-output elemental rule for ``top_k`` (returns values + indices).

    ``top_k`` is ``multiple_results`` so it must go through
    ``multi_output_elemental_only_rules``: the return is
    ``elementals[outvar_idx][invar_idx]``. The single input ``x`` (invar 0) flows
    to the VALUES output via the one-hot selection Jacobian; the integer INDICES
    output carries no gradient (None). (The cache-key fix in core.py makes this
    value-dependent indicator safe across repeated jacve calls.)"""
    x = primals[0]
    values, indices = primal_outs
    k = params["k"]
    x_shape = get_shape(x)
    val_ndim = len(get_shape(values))
    x_ndim = len(x_shape)
    n = x_shape[-1]

    out_dims, primal_dims = [], []
    axis_count = 0
    for i, s in enumerate(x_shape[:-1]):  # batch dims: diagonal
        out_dims.append(DiagonalIndex(i, s, axis_count, val_ndim + i))
        primal_dims.append(DiagonalIndex(val_ndim + i, s, axis_count, i))
        axis_count += 1
    out_dims.append(DenseIndex(val_ndim - 1, k, axis_count)); axis_count += 1
    primal_dims.append(DenseIndex(val_ndim + x_ndim - 1, n, axis_count))
    indicator = (indices[..., :, None] == jnp.arange(n)[None, :]).astype(x.dtype)
    tensor = _swap_back_axes(SparseTensor(out_dims, primal_dims, indicator))
    # outputs = [values, indices]; one invar (x). indices -> no edge (None).
    return [[tensor], [None]]


multi_output_elemental_only_rules[lax.top_k_p] = top_k_elemental_only


# argmax: returns integer indices, derivative is zero everywhere
def argmax_elemental_rule(primals, **params):
    val_out = lax.argmax_p.bind(*primals, **params)
    return val_out, []


elemental_rules[lax.argmax_p] = argmax_elemental_rule
