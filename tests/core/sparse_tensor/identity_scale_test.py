# The identity scalar_mult is a Python float and multiplies nothing (dsnn-dfw.253).
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.dtype_compute import _cast_scalar, _compute_dtype, _scaled_mul
from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.ops.matmul import scale_by_scalar
from graphax.sparse.tensor import SparseTensor

KEY = jax.random.PRNGKey(11)
A = jax.random.normal(KEY, (2, 8), jnp.float32)
B = jax.random.normal(jax.random.fold_in(KEY, 1), (2, 8), jnp.float32)


def _diag(val):
    n, m = val.shape
    return SparseTensor(
        (DiagonalIndex(0, n, 0, 2), DiagonalIndex(1, m, 1, 3)),
        (DiagonalIndex(2, n, 0, 0), DiagonalIndex(3, m, 1, 1)),
        val,
    )


def _dense(val, n_out):
    dims = tuple(DenseIndex(i, s, i) for i, s in enumerate(val.shape))
    return SparseTensor(dims[:n_out], dims[n_out:], val)


def _names(jaxpr):
    return [e.primitive.name for e in jaxpr.eqns]


def test_the_identity_scale_is_a_python_float():
    st = _diag(A)
    assert isinstance(st.scalar_mult, float) and st.scalar_mult == 1.0
    assert jnp.dtype(st.dtype) == jnp.dtype(jnp.float32)
    narrow = _diag(A.astype(jnp.bfloat16))
    assert isinstance(narrow.scalar_mult, float)
    structural = SparseTensor((), (), None)
    assert isinstance(structural.scalar_mult, float)
    assert jnp.dtype(structural.dtype) == jnp.dtype(jnp.float32)
    mask = _dense(A > 0, 1)
    assert not isinstance(mask.scalar_mult, float)
    assert jnp.dtype(mask.scalar_mult.dtype) == jnp.dtype(jnp.bool_)


def test_scaled_mul_by_the_identity_is_the_value_itself():
    assert _scaled_mul(A, 1.0) is A
    assert _scaled_mul(1.0, A) is A
    assert _scaled_mul(1.0, 1.0) == 1.0
    narrow = A.astype(jnp.bfloat16)
    up = _scaled_mul(narrow, 1.0)
    assert jnp.dtype(up.dtype) == jnp.dtype(jnp.float32)
    np.testing.assert_array_equal(np.asarray(up), np.asarray(narrow.astype(jnp.float32)))
    assert _scaled_mul(narrow, 1.0, keep_narrow=True) is narrow
    two = _scaled_mul(A, 2.0)
    np.testing.assert_array_equal(np.asarray(two), np.asarray(A * jnp.float32(2.0)))


def test_a_product_of_two_diagonal_partials_is_exactly_one_equation():
    # Fails on the array identity: two dead scalar multiplies follow the product.
    jaxpr = jax.make_jaxpr(lambda x, y: (_diag(x) @ _diag(y)).val)(A, B)
    assert _names(jaxpr) == ["mul"], _names(jaxpr)
    res = _diag(A) @ _diag(B)
    assert isinstance(res.scalar_mult, float) and res.scalar_mult == 1.0


def test_a_join_of_two_edges_scales_nothing():
    jaxpr = jax.make_jaxpr(lambda x, y: (_diag(x) + _diag(y)).val)(A, B)
    assert "mul" not in _names(jaxpr), _names(jaxpr)
    np.testing.assert_array_equal(np.asarray((_diag(A) + _diag(B)).val), np.asarray(A + B))


def test_the_dense_read_of_an_unscaled_tensor_is_the_value():
    jaxpr = jax.make_jaxpr(lambda x: _dense(x, 1).dense())(A)
    assert jaxpr.eqns == [], _names(jaxpr)
    np.testing.assert_array_equal(np.asarray(_dense(A, 1).dense()), np.asarray(A))


def test_a_narrow_join_still_reads_the_edges_at_full_precision():
    a, b = A.astype(jnp.bfloat16), B.astype(jnp.bfloat16)
    s = _diag(a) + _diag(b)
    assert jnp.dtype(s.val.dtype) == jnp.dtype(jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(s.val), np.asarray(a.astype(jnp.float32) + b.astype(jnp.float32)))


def test_a_narrow_product_keeps_its_narrow_array_scale():
    res = _diag(A.astype(jnp.bfloat16)) @ _diag(B.astype(jnp.bfloat16))
    assert jnp.dtype(res.val.dtype) == jnp.dtype(jnp.bfloat16)
    assert not isinstance(res.scalar_mult, float)
    assert jnp.dtype(res.scalar_mult.dtype) == jnp.dtype(jnp.bfloat16)


def test_scales_fold_into_one_python_float():
    two = SparseTensor((), (), None, scalar_mult=jnp.asarray(2.0, jnp.float32))
    three = SparseTensor((), (), None, scalar_mult=3.0)
    out = scale_by_scalar(scale_by_scalar(_diag(A), two), three)
    assert jnp.ndim(out.scalar_mult) == 0
    np.testing.assert_allclose(np.asarray(out.dense()), 6.0 * np.asarray(_diag(A).dense()), rtol=1e-6)
    neg = -_diag(A)
    assert neg.scalar_mult == -1.0
    np.testing.assert_array_equal(np.asarray(neg.dense()), -np.asarray(_diag(A).dense()))


def _old_scaled_mul(value, scalar_mult, keep_narrow=False):
    # The body before dsnn-dfw.253, where every scale was an array.
    vdt, sdt = value.dtype, scalar_mult.dtype
    if keep_narrow and jnp.dtype(vdt) != jnp.dtype(sdt):
        return value * jnp.asarray(scalar_mult).astype(vdt)
    cdt = _compute_dtype(vdt, sdt)
    return value.astype(cdt) * scalar_mult.astype(cdt)


def _bits(x):
    return np.atleast_1d(np.asarray(x)).view(np.uint8)


def _same(new, old):
    assert jnp.dtype(jnp.result_type(new)) == jnp.dtype(old.dtype), (new, old)
    np.testing.assert_array_equal(_bits(jnp.asarray(new, old.dtype)), _bits(old))


DTYPES = (jnp.float32, jnp.bfloat16, jnp.float16, jnp.int8, jnp.int32, jnp.bool_)
SCALES = (1.0, -1.0, 2.0, 0.5, 3.0, 1.0)


@pytest.mark.parametrize("seed", range(36))
def test_scaled_mul_equals_the_array_scale_bit_for_bit(seed):
    rng = np.random.default_rng(seed)
    dt = DTYPES[seed % len(DTYPES)]
    shape = tuple(int(n) for n in rng.integers(1, 5, size=int(rng.integers(0, 3))))
    x = rng.standard_normal(shape) * 4
    value = jnp.asarray(x > 0) if dt == jnp.bool_ else jnp.asarray(x, jnp.float32).astype(dt)
    s = float(SCALES[int(rng.integers(0, len(SCALES)))])
    sa = jnp.asarray(s, jnp.float32)
    for keep in (False, True):
        _same(_scaled_mul(value, s, keep_narrow=keep), _old_scaled_mul(value, sa, keep_narrow=keep))
    _same(_scaled_mul(s, value), _old_scaled_mul(sa, value))
    other = jnp.asarray(rng.standard_normal(()), jnp.float32)
    _same(_scaled_mul(s, other), _old_scaled_mul(sa, other))
    if dt != jnp.bool_:
        _same(_cast_scalar(s, dt), sa.astype(dt))


def _pair(rng, dt, kind, n=2, m=4):
    v = jnp.asarray(rng.standard_normal((n, m)), jnp.float32).astype(dt)
    if kind == "diag":
        mk = lambda sm: SparseTensor(
            (DiagonalIndex(0, n, 0, 2), DiagonalIndex(1, m, 1, 3)),
            (DiagonalIndex(2, n, 0, 0), DiagonalIndex(3, m, 1, 1)), v, scalar_mult=sm)
    else:
        w = jnp.asarray(rng.standard_normal((n, m, n, m)), jnp.float32).astype(dt)
        dims = tuple(DenseIndex(i, s, i) for i, s in enumerate(w.shape))
        mk = lambda sm: SparseTensor(dims[:2], dims[2:], w, scalar_mult=sm)
    return mk(None), mk(jnp.asarray(1.0, jnp.float32))


@pytest.mark.parametrize("seed", range(24))
def test_the_python_identity_equals_the_array_identity_bit_for_bit(seed):
    rng = np.random.default_rng(100 + seed)
    dt = (jnp.float32, jnp.bfloat16)[(seed // 4) % 2]
    kinds = [("diag", "diag"), ("diag", "dense"), ("dense", "diag"), ("dense", "dense")][seed % 4]
    a_new, a_old = _pair(rng, dt, kinds[0])
    b_new, b_old = _pair(rng, dt, kinds[1])
    three = SparseTensor((), (), None, scalar_mult=jnp.asarray(3.0, jnp.float32))
    ops = [
        lambda a, b: a.dense(),
        lambda a, b: (a @ b).dense(),
        lambda a, b: (a + b).dense(),
        lambda a, b: (a * b).dense(),
        lambda a, b: (-a).dense(),
        lambda a, b: scale_by_scalar(a, three).dense(),
        lambda a, b: (a.astype(jnp.bfloat16) @ b.astype(jnp.bfloat16)).dense(),
        lambda a, b: a.astype(jnp.float32).dense(),
    ]
    for op in ops:
        new, old = op(a_new, b_new), op(a_old, b_old)
        assert jnp.dtype(new.dtype) == jnp.dtype(old.dtype)
        np.testing.assert_array_equal(_bits(new), _bits(old))
