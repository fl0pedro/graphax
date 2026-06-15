"""Correctness of the elemental Jacobian rules across the ``graphax.primitives``
category modules (``indexing`` / ``conv`` / ``reductions`` / ``custom`` and the
elementwise rules in ``math``) against JAX, with an emphasis on ML-core
primitives: convolution,
``dot_general`` (matmul / attention contractions), windowed pooling,
activations, reductions, and gather/scatter.

Each primitive is exercised over SEVERAL configs (shapes / params) and, for
each config, over 10 RANDOM input draws — the ground truth is
``jax.jacfwd`` / ``jax.jacrev``. ``graphax.jacve`` must match both 'fwd' and
'rev' elimination orders (vertex elimination is order-invariant for an exact
Jacobian).

These tests were reconstructed after the original ``primitive_test.py`` suite
was almost entirely commented out (15/17 methods). Configs that currently
disagree with JAX are marked ``xfail`` with a precise reason so the gap is
tracked rather than silent; remove the marker when the underlying rule is
fixed. See the module-level BUGS list for the catalogue.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

import graphax
from graphax import jacve, tree_allclose

jax.config.update("jax_platform_name", "cpu")

N_DRAWS = 10  # random input arrays per config


# --------------------------------------------------------------------------- #
# Catalogue of confirmed rule bugs (xfail reasons reference these).
# --------------------------------------------------------------------------- #
BUG_CONV = (
    "conv_general_dilated rule never uses _build_windowed_jacobian_1d; declares "
    "independent dense out/in spatial dims -> val axis-count mismatch (IndexError) "
    "and, even patched, ignores stride/kernel/pad/dilation. auto.py:770"
)
BUG_DG_PERM = (
    "dot_general rule conflates val-read-axis with output size/position for "
    "permuted contracting/batch dims (auto.py:716) -> wrong shape / broadcast error"
)
BUG_REDUCE_TOPO = (
    "reduce_max/reduce_min assign non-contiguous index IDs (max(idx,1) hack) for a "
    "kept axis following a reduced axis on >=3D -> Topology Error. auto.py:538,578"
)
BUG_REDUCE_MIN_BCAST = (
    "reduce_min omits the keepdims reshape reduce_max has -> broadcast crash for "
    "any reduction not over a leading axis. auto.py:588"
)
BUG_SCATTER_UPDATES = (
    "scatter_* return a length-2 elemental list but core indexes by absolute invar "
    "position (updates=2) -> updates Jacobian silently dropped (grad=0)"
)
BUG_DUS_UPDATES = (
    "dynamic_update_slice builds the updates Jacobian in the operand direction -> "
    "wrong shape/values for the updates argument. auto.py:1546"
)
BUG_SQUEEZE_IDS = (
    "squeeze_transform ID compaction produces duplicate IDs when the squeezed "
    "primal already carries a Jacobian -> Topology Error. auto.py:2197"
)
BUG_ABS_ZERO = "abs rule primal/out = 0/0 = NaN at x==0 (JAX returns subgradient 1.0)"
BUG_EXP2 = "exp2 elemental lambda lacks accuracy=None param -> TypeError under jacve"


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def _keys(n, seed):
    return jax.random.split(jax.random.PRNGKey(seed), n)


def check(f, argnums, arg_factory, *, seed=0, n=N_DRAWS, rtol=1e-4, atol=1e-5,
          modes=("fwd", "rev")):
    """Assert graphax.jacve == jax.jac{fwd,rev} over ``n`` random draws.

    ``arg_factory(key) -> tuple_of_args`` produces one random input tuple.
    """
    for i, key in enumerate(_keys(n, seed)):
        args = arg_factory(key)
        ref_fwd = jax.jacfwd(f, argnums)(*args)
        for mode in modes:
            got = jacve(f, mode, argnums)(*args)
            ok = tree_allclose(got, ref_fwd, rtol=rtol, atol=atol)
            assert ok, (
                f"jacve('{mode}') != jax.jac for draw {i}: "
                f"argnums={argnums}, shapes={[np.shape(a) for a in args]}"
            )


def randn(key, shape):
    return jax.random.normal(key, shape)


def pos(key, shape):  # strictly positive (for log/sqrt domains)
    return jnp.abs(jax.random.normal(key, shape)) + 0.5


# =========================================================================== #
# 1. Elementwise activations & unary math  (ML: tanh/sigmoid/gelu building blocks)
# =========================================================================== #
# (fn, input-domain factory) — all expected to MATCH jax.
_UNARY_OK = [
    ("tanh", jnp.tanh, randn),
    ("logistic", jax.nn.sigmoid, randn),
    ("exp", jnp.exp, randn),
    ("log", jnp.log, pos),
    ("log1p", jnp.log1p, pos),
    ("sqrt", jnp.sqrt, pos),
    ("rsqrt", lax.rsqrt, pos),
    ("square", jnp.square, randn),
    ("sin", jnp.sin, randn),
    ("cos", jnp.cos, randn),
    ("tan", jnp.tan, lambda k, s: 0.5 * jax.random.normal(k, s)),
    ("atan", jnp.arctan, randn),
    ("sinh", jnp.sinh, randn),
    ("cosh", jnp.cosh, randn),
    ("tanh2", lambda x: jnp.tanh(x) ** 2, randn),
    ("erf", lax.erf, randn),
    ("asinh", jnp.arcsinh, randn),          # FIXED (math.py:33): 1/sqrt(1+x^2)
    ("acosh", jnp.arccosh, lambda k, s: jnp.abs(jax.random.normal(k, s)) + 1.5),
    ("atanh", jnp.arctanh, lambda k, s: 0.5 * jax.random.normal(k, s)),
    ("gelu", jax.nn.gelu, randn),           # composite ML activation
    # softplus via primitives (jax.nn.softplus is a custom_jvp primitive that
    # jacve cannot trace — a framework limitation, not an elemental-rule bug).
    ("softplus", lambda x: jnp.log1p(jnp.exp(x)), randn),
]


@pytest.mark.parametrize("name,fn,dom", _UNARY_OK, ids=[c[0] for c in _UNARY_OK])
@pytest.mark.parametrize("shape", [(8,), (3, 4)], ids=["1d", "2d"])
def test_unary_activations(name, fn, dom, shape):
    check(lambda x: fn(x), (0,), lambda k: (dom(k, shape),),
          seed=hash((name, shape)) & 0xFFFF)


def test_abs_nonzero_matches_jax():
    check(lambda x: jnp.abs(x), (0,),
          lambda k: (jax.random.normal(k, (8,)) + 2.0,))  # away from 0


def test_abs_at_zero_matches_jax():  # was BUG_ABS_ZERO: primal/out = 0/0 = NaN
    g = jacve(lambda x: jnp.abs(x), "rev", (0,))(jnp.array([0.0, 1.0, -1.0]))
    ref = jax.jacrev(lambda x: jnp.abs(x), (0,))(jnp.array([0.0, 1.0, -1.0]))
    assert tree_allclose(g, ref)


def test_exp2_matches_jax():  # was BUG_EXP2: lambda lacked accuracy=None
    check(lambda x: jnp.exp2(x), (0,), lambda k: (randn(k, (8,)),))


# =========================================================================== #
# 2. Binary / multi-arg elementwise  (add/mul/div/pow/max/min/clamp/select)
# =========================================================================== #
def test_add():
    check(lambda x, y: x + y, (0, 1), lambda k: tuple(randn(k_, (3, 4)) for k_ in jax.random.split(k)))


def test_mul():
    check(lambda x, y: x * y, (0, 1), lambda k: tuple(randn(k_, (3, 4)) for k_ in jax.random.split(k)))


def test_div():
    def fac(k):
        a, b = jax.random.split(k)
        return (randn(a, (3, 4)), pos(b, (3, 4)) + 0.5)
    check(lambda x, y: x / y, (0, 1), fac)


def test_pow():
    def fac(k):
        a, b = jax.random.split(k)
        return (pos(a, (6,)) + 0.5, 0.5 + jax.random.uniform(b, (6,)) * 2)
    check(lambda x, y: x ** y, (0, 1), fac)


def test_clamp_wrt_x():
    check(lambda x: jnp.clip(x, -0.5, 0.5), (0,), lambda k: (randn(k, (10,)),))


def test_clamp_all_three_args():
    # gradient w.r.t. lo / x / hi (was: 3-input parallel Jacobian unsupported +
    # lo/hi hard-zeroed). lo<hi guaranteed so the clamp is well-formed.
    def f(lo, x, hi):
        return lax.clamp(lo, x, hi)
    def fac(k):
        a, b, c = jax.random.split(k, 3)
        return (-jnp.abs(randn(a, (8,))) - 0.2, randn(b, (8,)),
                jnp.abs(randn(c, (8,))) + 0.2)
    check(f, (0, 1, 2), fac)


def test_select_n():
    def fac(k):
        a, b, c = jax.random.split(k, 3)
        return (jax.random.bernoulli(a, shape=(8,)).astype(jnp.int32),
                randn(b, (8,)), randn(c, (8,)))
    check(lambda p, x, y: lax.select_n(p, x, y), (1, 2), fac)


def test_max_min_no_ties():
    # distinct draws -> no ties -> matches jax exactly
    def fac(k):
        a, b = jax.random.split(k)
        return (randn(a, (8,)), randn(b, (8,)) + 5.0)
    check(lambda x, y: jnp.maximum(x, y), (0, 1), fac)
    check(lambda x, y: jnp.minimum(x, y), (0, 1), fac)


# =========================================================================== #
# 3. Reductions  (ML: softmax denominators, pooling, norms)
# =========================================================================== #
@pytest.mark.parametrize("axes", [None, (0,), (1,), (0, 1)], ids=str)
def test_reduce_sum(axes):
    check(lambda x: jnp.sum(x, axis=axes), (0,), lambda k: (randn(k, (4, 5)),))


@pytest.mark.parametrize("axis", [0, 1, None], ids=["ax0", "ax1", "all"])
def test_reduce_max(axis):
    check(lambda x: jnp.max(x, axis=axis), (0,), lambda k: (randn(k, (4, 5)),))


@pytest.mark.parametrize("axis", [0, 1, 2, None], ids=["ax0", "ax1", "ax2", "all"])
def test_reduce_max_3d(axis):  # was BUG_REDUCE_TOPO: ids collided for a kept axis after a reduced one
    check(lambda x: jnp.max(x, axis=axis), (0,), lambda k: (randn(k, (2, 3, 4)),), n=3)


@pytest.mark.parametrize("axis", [0, 1, 2, None], ids=["ax0", "ax1", "ax2", "all"])
def test_reduce_min_3d(axis):  # was BUG_REDUCE_MIN_BCAST + BUG_REDUCE_TOPO
    check(lambda x: jnp.min(x, axis=axis), (0,), lambda k: (randn(k, (2, 3, 4)),), n=3)


# --- newly added rules: reduce_prod / cumulative / sort / flip ---
@pytest.mark.parametrize("axis", [0, 1, None], ids=["ax0", "ax1", "all"])
def test_reduce_prod(axis):  # ADDED rule
    check(lambda x: jnp.prod(x, axis=axis), (0,), lambda k: (pos(k, (4, 5)),))


@pytest.mark.parametrize("axis,rev", [(0, False), (1, False), (0, True)], ids=["ax0", "ax1", "rev"])
def test_cumsum(axis, rev):  # ADDED rule
    check(lambda x: lax.cumsum(x, axis, reverse=rev), (0,), lambda k: (randn(k, (4, 5)),))


@pytest.mark.parametrize("axis,rev", [(0, False), (1, False), (0, True)], ids=["ax0", "ax1", "rev"])
def test_cumprod(axis, rev):  # ADDED rule (value-dependent)
    check(lambda x: lax.cumprod(x, axis, reverse=rev), (0,), lambda k: (pos(k, (4, 5)),))


@pytest.mark.parametrize("x", [
    [2.0, 0.0, 3.0], [0.0, 3.0, 4.0], [2.0, 3.0, 0.0],
    [2.0, 0.0, 3.0, 0.0, 5.0],  # two zeros: every partial product-of-others = 0
], ids=["zmid", "zfirst", "zlast", "twozeros"])
def test_cumprod_with_zeros(x):  # zero-robust: out[i]/x[j] was 0/0, dropped the edge
    arr = jnp.asarray(x)
    check(lambda z: lax.cumprod(z, 0), (0,), lambda k: (arr,), n=1)


@pytest.mark.parametrize("fn,name", [(lax.cummax, "cummax"), (lax.cummin, "cummin")], ids=["max", "min"])
def test_cum_extremum(fn, name):  # ADDED rule (value-dependent, tie-normalized)
    check(lambda x: jnp.sin(fn(x, 1)), (0,), lambda k: (randn(k, (3, 5)),))


@pytest.mark.parametrize("axis,rev", [(0, False), (1, False), (0, True)], ids=["ax0", "ax1", "rev"])
def test_cumlogsumexp(axis, rev):  # ADDED rule (CTC/HMM log-domain prefix)
    check(lambda x: lax.cumlogsumexp(x, axis, reverse=rev), (0,), lambda k: (randn(k, (4, 5)),))


def test_reduce_precision():  # ADDED rule (identity under AD; mixed-precision sim)
    check(lambda x: lax.reduce_precision(x, 8, 7) * 2.0, (0,), lambda k: (randn(k, (6,)),))


@pytest.mark.parametrize("axis", [0, 1], ids=["ax0", "ax1"])
def test_sort(axis):  # ADDED rule (argsort permutation)
    check(lambda x: jnp.sort(x, axis=axis), (0,), lambda k: (randn(k, (4, 5)),))


@pytest.mark.parametrize("axes", [(0,), (1,), (0, 1)], ids=["ax0", "ax1", "both"])
def test_flip(axes):  # ADDED rule (rev / reverse permutation)
    check(lambda x: jnp.sin(jnp.flip(x, axes)), (0,), lambda k: (randn(k, (4, 5)),))


# =========================================================================== #
# 4. dot_general  (ML: matmul, batched matmul, attention scores)
# =========================================================================== #
def _dg(dn):
    return lambda x, y: lax.dot_general(x, y, dn)


_DG_OK = [
    ("matmul", (((1,), (0,)), ((), ())), (4, 5), (5, 3)),
    ("batched", (((2,), (1,)), ((0,), (0,))), (2, 4, 5), (2, 5, 3)),
    ("multi_contract", (((1, 2), (0, 1)), ((), ())), (4, 5, 6), (5, 6, 3)),
    ("multi_batch", (((3,), (2,)), ((0, 1), (0, 1))), (2, 3, 4, 5), (2, 3, 5, 6)),
    ("matvec", (((1,), (0,)), ((), ())), (4, 5), (5,)),
]


@pytest.mark.parametrize("name,dn,sl,sr", _DG_OK, ids=[c[0] for c in _DG_OK])
def test_dot_general_aligned(name, dn, sl, sr):
    def fac(k):
        a, b = jax.random.split(k)
        return (randn(a, sl), randn(b, sr))
    check(_dg(dn), (0, 1), fac, seed=hash(name) & 0xFFFF)


@pytest.mark.parametrize("name,dn,sl,sr", [
    ("ijk,lkj", (((1, 2), (2, 1)), ((), ())), (3, 4, 5), (6, 5, 4)),
    ("ijk,kjl", (((1, 2), (1, 0)), ((), ())), (3, 4, 5), (5, 4, 6)),
], ids=["ijk,lkj", "ijk,kjl"])
def test_dot_general_permuted_contract(name, dn, sl, sr):  # was BUG_DG_PERM
    check(_dg(dn), (0, 1),
          lambda k: tuple(randn(kk, s) for kk, s in zip(jax.random.split(k), [sl, sr])), n=3)


def test_dot_general_permuted_batch():  # was BUG_DG_PERM
    dn = (((3,), (2,)), ((0, 1), (1, 0)))  # permuted batch mapping
    check(_dg(dn), (0, 1),
          lambda k: tuple(randn(kk, s) for kk, s in zip(jax.random.split(k), [(2, 3, 4, 5), (3, 2, 5, 6)])),
          n=3)


# =========================================================================== #
# 5. Convolution  (ML core) — exact windowed-Jacobian rule
# =========================================================================== #
def _conv(strides, padding, labels, sl, sr, lhs_dil=None, rhs_dil=None):
    dn = lax.conv_dimension_numbers(sl, sr, labels)

    def f(lhs, rhs):
        return lax.conv_general_dilated(lhs, rhs, strides, padding,
                                        lhs_dilation=lhs_dil, rhs_dilation=rhs_dil,
                                        dimension_numbers=dn)
    return f


# (name, strides, padding, labels, lhs_shape, rhs_shape, lhs_dil, rhs_dil)
_CONV_CFG = [
    # --- 1D (NCH) over kernel / stride / padding / dilation / channels ---
    ("1d_1x1_s1", (1,), "VALID", ("NCH", "OIH", "NCH"), (1, 1, 5), (1, 1, 1), None, None),
    ("1d_k3_valid", (1,), "VALID", ("NCH", "OIH", "NCH"), (2, 1, 6), (1, 1, 3), None, None),
    ("1d_k3_s2", (2,), "VALID", ("NCH", "OIH", "NCH"), (1, 1, 7), (1, 1, 3), None, None),
    ("1d_k3_same", (1,), "SAME", ("NCH", "OIH", "NCH"), (2, 1, 6), (1, 1, 3), None, None),
    ("1d_multichan", (1,), "VALID", ("NCH", "OIH", "NCH"), (2, 3, 6), (4, 3, 3), None, None),
    # pointwise / 1x1 conv (pass-through spatial -> DiagonalIndex path)
    ("1d_pointwise", (1,), "VALID", ("NCH", "OIH", "NCH"), (2, 3, 7), (5, 3, 1), None, None),
    ("2d_pointwise", (1, 1), "VALID", ("NCHW", "OIHW", "NCHW"), (2, 3, 4, 4), (5, 3, 1, 1), None, None),
    ("1d_rdil2", (1,), "VALID", ("NCH", "OIH", "NCH"), (1, 2, 9), (3, 2, 3), None, (2,)),
    ("1d_ldil2", (1,), [(2, 2)], ("NCH", "OIH", "NCH"), (1, 2, 5), (3, 2, 3), (2,), None),
    # --- 2D (NCHW / OIHW) ---
    ("2d_k3", (1, 1), "VALID", ("NCHW", "OIHW", "NCHW"), (2, 3, 5, 5), (4, 3, 3, 3), None, None),
    ("2d_s2_same", (2, 2), "SAME", ("NCHW", "OIHW", "NCHW"), (1, 3, 7, 7), (4, 3, 3, 3), None, None),
    # --- 2D permuted dimension_numbers (NHWC, the flax/keras default) ---
    ("2d_NHWC", (1, 1), "VALID", ("NHWC", "HWIO", "NHWC"), (2, 5, 5, 3), (3, 3, 3, 4), None, None),
]


@pytest.mark.parametrize("name,strides,pad,labels,sl,sr,ld,rd", _CONV_CFG,
                         ids=[c[0] for c in _CONV_CFG])
def test_conv(name, strides, pad, labels, sl, sr, ld, rd):
    def fac(k):
        a, b = jax.random.split(k)
        return (randn(a, sl), randn(b, sr))
    check(_conv(strides, pad, labels, sl, sr, ld, rd), (0, 1), fac, n=3,
          seed=hash(name) & 0xFFFF)


def test_conv_grouped_raises_clearly():
    # grouped / depthwise conv is not yet supported — must fail LOUDLY, not
    # silently produce a wrong Jacobian.
    sl, sr = (1, 4, 6), (4, 2, 3)  # feature_group_count=2
    dn = lax.conv_dimension_numbers(sl, sr, ("NCH", "OIH", "NCH"))
    f = lambda a, b: lax.conv_general_dilated(a, b, (1,), "VALID",
                                              dimension_numbers=dn, feature_group_count=2)
    with pytest.raises(NotImplementedError, match="group"):
        jacve(f, "rev", (0, 1))(randn(jax.random.PRNGKey(0), sl),
                                randn(jax.random.PRNGKey(1), sr))


# =========================================================================== #
# 6. Windowed pooling — reduce_window_sum  (ML: average/sum pooling)  [PASSES]
# =========================================================================== #
def _avgsum_pool(window, strides, pad):
    return lambda x: lax.reduce_window(x, 0.0, lax.add, window, strides, pad)


_POOL_CFG = [
    ("1d_w3_s1", (3,), (1,), "VALID", (8,)),
    ("1d_w3_s2", (3,), (2,), "VALID", (9,)),
    ("2d_w2_s2", (1, 2, 2), (1, 2, 2), "VALID", (1, 4, 4)),
]


@pytest.mark.parametrize("name,w,s,pad,shape", _POOL_CFG, ids=[c[0] for c in _POOL_CFG])
def test_reduce_window_sum(name, w, s, pad, shape):
    check(_avgsum_pool(w, s, pad), (0,), lambda k: (randn(k, shape),),
          n=5, seed=hash(name) & 0xFFFF)


# =========================================================================== #
# 7. Structural / indexing  (broadcast, gather, pad, transpose, reshape, concat)
# =========================================================================== #
def test_broadcast_in_dim():
    check(lambda x: jnp.broadcast_to(x[None, :], (5, 4)).sum(0), (0,),
          lambda k: (randn(k, (4,)),))


def test_gather():
    idx = jnp.array([0, 2, 2, 4, 1])
    check(lambda x: x[idx], (0,), lambda k: (randn(k, (6,)),))


# Matrix decompositions: eigh/svd(singular values)/cholesky go through the
# multi-output path (were mis-registered in single-output elemental_rules and
# crashed). qr/lu/eig rules are not yet correct, so they fail loudly.
def _sympd(k, n=3):
    M = randn(k, (n, n))
    return M @ M.T + n * jnp.eye(n)


def _sym(k, n=3):
    M = randn(k, (n, n))
    return 0.5 * (M + M.T)


def test_eigh_eigenvalues():
    check(lambda a: jnp.linalg.eigh(a)[0], (0,), lambda k: (_sym(k),), n=2, atol=1e-3)


def test_eigh_eigenvectors():
    check(lambda a: jnp.sum(jnp.sin(jnp.linalg.eigh(a)[1])), (0,),
          lambda k: (_sym(k),), n=2, atol=1e-3)


@pytest.mark.parametrize("shape", [(3, 3), (2, 4), (4, 2)], ids=["sq", "wide", "tall"])
def test_svd_singular_values(shape):
    check(lambda a: jnp.linalg.svd(a, compute_uv=False), (0,),
          lambda k: (randn(k, shape),), n=2, atol=1e-3)


def test_cholesky():
    check(lambda a: jnp.linalg.cholesky(a), (0,), lambda k: (_sympd(k),), n=2, atol=1e-3)


@pytest.mark.parametrize("fn", [
    lambda a: jnp.linalg.qr(a)[1],
    lambda a: jax.scipy.linalg.lu_factor(a)[0],
], ids=["qr", "lu"])
def test_unsupported_decompositions_raise(fn):
    # Loud failure beats a cryptic crash or a silently-wrong Jacobian.
    A = jnp.array([[1.0, 2.0], [3.0, 5.0]])
    with pytest.raises(NotImplementedError):
        jacve(fn, "rev", (0,))(A)


@pytest.mark.parametrize("shape,kk", [((8,), 3), ((4, 6), 2)], ids=["1d", "batched"])
def test_top_k(shape, kk):  # multi-output rule; safe now the cache keys by content
    check(lambda x: lax.top_k(x, kk)[0], (0,), lambda key: (randn(key, shape),))


def test_pad_and_crop():
    check(lambda x: lax.pad(x, 0.0, ((1, 2, 0),)), (0,), lambda k: (randn(k, (5,)),))
    check(lambda x: lax.pad(x, 0.0, ((-1, -1, 0),)), (0,), lambda k: (randn(k, (5,)),))


# Pure-structural ops (transpose/reshape/concat/slice) return transform-only
# Jacobians (val=None) that don't materialize standalone — compose with a
# nonlinearity (as in any real differentiated function) so the chain has a
# concrete Jacobian to compare.
def test_transpose_reshape():
    check(lambda x: jnp.sin(jnp.transpose(x, (1, 0))).reshape(-1), (0,),
          lambda k: (randn(k, (3, 4)),))


def test_concatenate():
    def fac(k):
        a, b = jax.random.split(k)
        return (randn(a, (3, 2)), randn(b, (3, 2)))
    check(lambda x, y: jnp.sin(jnp.concatenate([x, y], axis=1)), (0, 1), fac)


def test_slice():
    check(lambda x: jnp.sin(x[1:5]), (0,), lambda k: (randn(k, (8,)),))


@pytest.mark.parametrize("start,limit,stride", [(0, 8, 2), (1, 8, 2), (0, 9, 3), (2, 8, 1)],
                         ids=["s2", "s2off", "s3", "s1"])
def test_slice_strided(start, limit, stride):
    # strides were ignored -> the Jacobian selected contiguous rows. The inverse
    # (fwd order) embeds via interior padding.
    check(lambda x: lax.slice(jnp.sin(x), (start,), (limit,), (stride,)), (0,),
          lambda k: (randn(k, (9,)),))


def test_concat_float64():
    # concatenate hardcoded float32 eye/zeros -> lax.scatter dtype crash on f64.
    jax.config.update("jax_enable_x64", True)
    try:
        x = jnp.linspace(0.1, 0.3, 3).astype(jnp.float64)
        f = lambda z: jnp.concatenate([jnp.sin(z), jnp.cos(z)])
        ref = jax.jacrev(f, (0,))(x)
        for m in ("fwd", "rev"):
            assert tree_allclose(jacve(f, m, (0,))(x), ref, rtol=1e-9, atol=1e-9)
    finally:
        jax.config.update("jax_enable_x64", False)


# =========================================================================== #
# 8. Scatter — updates-direction Jacobian currently dropped
# =========================================================================== #
def test_scatter_add_operand():
    # operand-direction Jacobian IS correct
    idx = jnp.array([[0], [2], [4]])
    dn = lax.ScatterDimensionNumbers((), (0,), (0,))
    up = jnp.array([10.0, 20.0, 30.0])
    check(lambda o: lax.scatter_add(o, idx, up, dn), (0,),
          lambda k: (randn(k, (5,)),))


@pytest.mark.parametrize("name,sf", [("add", lax.scatter_add), ("sub", lax.scatter_sub)],
                         ids=["add", "sub"])
def test_scatter_updates(name, sf):  # was BUG_SCATTER_UPDATES: dropped at slot 1
    idx = jnp.array([[0], [2], [4]])
    dn = lax.ScatterDimensionNumbers((), (0,), (0,))

    def f(o, u):
        return jnp.sin(sf(o, idx, u, dn))
    # both argnums -> the updates Jacobian (slot 2) must be present and correct
    check(f, (0, 1), lambda k: tuple(randn(kk, s) for kk, s in
                                     zip(jax.random.split(k), [(5,), (3,)])), n=3)


_O1 = jnp.array([5.0, 2.0, 3.0, 7.0])
_U1 = jnp.array([2.0, 4.0])           # u[0]==o[1] -> a tie for max/min
_O2 = jnp.arange(12.0).reshape(3, 4)


@pytest.mark.parametrize("op,args", [
    (lambda o, u: o.at[1:3].max(u), (_O1, _U1)),
    (lambda o, u: o.at[1:3].min(u), (_O1, _U1)),
    (lambda o, u: o.at[1:3].add(u), (_O1, _U1)),
    (lambda o, u: o.at[1:3].set(u), (_O1, _U1)),
    (lambda o, u: o.at[0:2, 1:3].max(u), (_O2, jnp.ones((2, 2)))),
], ids=["max", "min", "add", "set", "max2d"])
def test_scatter_slice(op, args):
    # Slice scatters (window-dim dimension_numbers) — _update_to_output_index
    # used to drop the scatter start, mis-placing the whole Jacobian at offset 0.
    # max/min also exercise the balanced 0.5/0.5 tie convention.
    check(op, (0, 1), lambda k: args, n=1)


@pytest.mark.parametrize("shape,ax", [((1, 4, 5), 0), ((4, 1, 5), 1), ((4, 5, 1), 2),
                                       ((1, 1, 5), (0, 1))], ids=["ax0", "ax1", "ax2", "ax01"])
def test_squeeze_carrying_jacobian(shape, ax):  # was BUG_SQUEEZE_IDS (inverse-transform axis/id)
    check(lambda x: jnp.sin(jnp.squeeze(x, ax)), (0,), lambda k: (randn(k, shape),), n=3)


def test_gather_dynamic_index():  # argmax-driven single-element gather (was squeeze-transform crash)
    check(lambda x: x[jnp.argmax(x)], (0,), lambda k: (randn(k, (8,)),))


def test_jit_inlined():
    # jit/pjit bodies are inlined before elimination (was a macro-vertex that
    # mis-shaped the inner Jacobian).
    check(lambda x: jax.jit(lambda y: jnp.sin(y) * 2.0)(x), (0,), lambda k: (randn(k, (6,)),))
    check(lambda x: jax.jit(lambda y: jax.jit(lambda z: jnp.tanh(z))(y) + 1.0)(x),
          (0,), lambda k: (randn(k, (6,)),))


def test_custom_jvp_inlined():
    # custom_jvp_call body is inlined (differentiate the primal decomposition).
    # hard_sigmoid's body has no shared-input select fan-out, so it composes.
    check(lambda x: jax.nn.hard_sigmoid(x), (0,), lambda k: (randn(k, (6,)),))


def _ste(x):
    # Straight-through estimator: round forward, identity gradient.
    f = jax.custom_vjp(lambda y: jnp.round(y))
    f.defvjp(lambda y: (jnp.round(y), ()), lambda res, ct: (ct,))
    return f(x)


def _surrogate(x):
    # Deliberately non-primal gradient (2*cos instead of cos).
    f = jax.custom_vjp(lambda y: jnp.sin(y))
    f.defvjp(lambda y: (jnp.sin(y), (y,)), lambda res, ct: (2 * jnp.cos(res[0]) * ct,))
    return f(x)


def _clipgrad(x):
    f = jax.custom_vjp(lambda y: y)
    f.defvjp(lambda y: (y, ()), lambda res, ct: (jnp.clip(ct, -1.0, 1.0),))
    return f(x)


@pytest.mark.parametrize("name,fn", [("ste", _ste), ("surrogate", _surrogate),
                                     ("clipgrad", _clipgrad)],
                         ids=["ste", "surrogate", "clipgrad"])
def test_custom_vjp_honored(name, fn):
    # custom_vjp is NOT inlined: its bwd rule may differ from the primal
    # derivative (straight-through / surrogate / clipped gradients). jacve must
    # honor bwd in BOTH elimination orders. jax.jacfwd can't trace a custom_vjp
    # (no fwd rule), so jax.jacrev is the reference for both graphax modes.
    for i, key in enumerate(_keys(N_DRAWS, hash(name) & 0xFFFF)):
        x = randn(key, (6,))
        ref = jax.jacrev(fn, (0,))(x)
        for mode in ("fwd", "rev"):
            got = jacve(fn, mode, (0,))(x)
            assert tree_allclose(got, ref, rtol=1e-4, atol=1e-5), (
                f"custom_vjp '{name}' jacve('{mode}') != jax.jacrev, draw {i}"
            )


# Conditional / jit-wrapped activations: each uses select_n/where with the SAME
# input feeding multiple branches — was crashing/silently-wrong until the
# select_n masked-identity Jacobian carried its values (axis=i, not None).
_COND_ACTS = [
    ("relu", jax.nn.relu), ("elu", jax.nn.elu), ("selu", jax.nn.selu),
    ("celu", jax.nn.celu), ("leaky_relu", jax.nn.leaky_relu),
    ("softplus", jax.nn.softplus), ("mish", jax.nn.mish),
]


@pytest.mark.parametrize("name,fn", _COND_ACTS, ids=[c[0] for c in _COND_ACTS])
@pytest.mark.parametrize("shape", [(8,), (3, 4)], ids=["1d", "2d"])
def test_conditional_activations(name, fn, shape):
    check(lambda x: fn(x), (0,), lambda k: (randn(k, shape),), seed=hash((name, shape)) & 0xFFFF)


def test_where_shared_input():
    # x feeds predicate AND both branches
    check(lambda x: jnp.where(x > 0, jnp.sin(x), 2 * x), (0,), lambda k: (randn(k, (8,)),))


@pytest.mark.parametrize("shift,axis,shape", [(2, None, (6,)), (-1, None, (6,)), (9, None, (6,)),
                                              (1, 1, (3, 5))],
                         ids=["s2", "s-1", "big", "2d"])
def test_roll(shift, axis, shape):
    # roll = slice+slice+concatenate; fwd mode used to crash on the scalar-identity
    # seed in the slice/concat inverse transforms. check() asserts fwd==rev==jax
    # (vertex elimination is order-invariant).
    check(lambda x: jnp.sin(jnp.roll(x, shift, axis=axis)), (0,), lambda k: (randn(k, shape),))


@pytest.mark.parametrize("oshape,ushape,start", [((8,), (3,), (2,)), ((5, 6), (2, 3), (1, 2))],
                         ids=["1d", "2d"])
def test_dynamic_update_slice(oshape, ushape, start):  # was BUG_DUS_UPDATES (wrong-direction)
    def f(o, u):
        return jnp.sin(lax.dynamic_update_slice(o, u, start))
    # both argnums -> operand AND update Jacobians must be correct
    check(f, (0, 1), lambda k: tuple(randn(kk, s) for kk, s in
                                     zip(jax.random.split(k), [oshape, ushape])), n=3)
