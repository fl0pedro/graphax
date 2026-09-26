# The contraction engine's shape view and what a face emits (dsnn-dfw.250).
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.ops.matmul import _View, matmul
from graphax.sparse.tensor import SparseTensor

SHAPE_PRIMS = ("reshape", "transpose", "broadcast_in_dim", "squeeze", "expand_dims")


def _prims(jaxpr):
    names = []
    for e in jaxpr.eqns:
        names.append(e.primitive.name)
        for p in e.params.values():
            if hasattr(p, "jaxpr"):
                names.extend(_prims(p.jaxpr))
    return names


def _eqns(jaxpr, name):
    return [e for e in jaxpr.eqns if e.primitive.name == name]


def _reads_an_input(e, jaxpr):
    return any(v is x for v in e.invars for x in jaxpr.jaxpr.invars)


def _reads_the_inputs(e, jaxpr):
    ins = jaxpr.jaxpr.invars
    return len(e.invars) == len(ins) and all(any(v is x for v in e.invars) for x in ins)


def _factor(rng, n):
    out = []
    while n > 1:
        divs = [d for d in range(2, n + 1) if n % d == 0]
        d = int(rng.choice(divs)) if rng.random() < 0.7 else n
        out.append(d)
        n //= d
    return out


def _random_shape_like(rng, shape, total):
    if rng.random() < 0.15:
        # An arbitrary regrouping of the flat buffer.
        dims, n = [], total
        while n > 1 and len(dims) < 4:
            d = int(rng.choice([d for d in range(2, n + 1) if n % d == 0]))
            dims.append(d)
            n //= d
        if n > 1:
            dims.append(n)
    else:
        atoms = [a for d in shape for a in _factor(rng, int(d))]
        dims, i = [], 0
        while i < len(atoms):
            k = int(rng.integers(1, min(3, len(atoms) - i) + 1))
            dims.append(math.prod(atoms[i:i + k]))
            i += k
    for _ in range(int(rng.integers(0, 4))):
        dims.insert(int(rng.integers(0, len(dims) + 1)), 1)
    return tuple(dims)


@pytest.mark.parametrize("seed", range(40))
def test_a_random_chain_of_reshapes_and_transposes_is_bit_identical(seed):
    rng = np.random.default_rng(seed)
    base_shape = tuple(int(d) for d in rng.choice([1, 2, 3, 4, 6], size=int(rng.integers(1, 5))))
    arr = jnp.asarray(rng.standard_normal(base_shape).astype(np.float32))
    view, eager = _View(arr), arr
    for _ in range(int(rng.integers(1, 9))):
        if rng.random() < 0.4:
            perm = [int(p) for p in rng.permutation(eager.ndim)]
            view, eager = view.transpose(perm), eager.transpose(perm)
        else:
            shape = _random_shape_like(rng, eager.shape, eager.size)
            view, eager = view.reshape(shape), eager.reshape(shape)
        assert view.shape == eager.shape
        assert view.ndim == eager.ndim and view.size == eager.size
    got = view.materialize()
    assert got.shape == eager.shape
    np.testing.assert_array_equal(np.asarray(got), np.asarray(eager))


def test_a_view_of_unit_shuffles_emits_nothing():
    x = jnp.arange(6.0, dtype=jnp.float32).reshape(2, 3)

    def f(a):
        v = _View(a).reshape(2, 1, 1, 3, 1).transpose([0, 2, 3, 1, 4]).reshape(2, 3, 1)
        return v.transpose([2, 0, 1]).reshape(2, 3).materialize()

    jaxpr = jax.make_jaxpr(f)(x)
    assert jaxpr.eqns == []
    np.testing.assert_array_equal(np.asarray(f(x)), np.asarray(x))


def test_a_view_emits_at_most_reshape_transpose_reshape():
    x = jnp.arange(24.0, dtype=jnp.float32).reshape(2, 12)

    def f(a):
        v = _View(a).reshape(2, 1, 3, 4, 1).transpose([4, 3, 1, 0, 2]).reshape(4, 1, 2, 3)
        return v.transpose([1, 0, 2, 3]).reshape(8, 3).materialize()

    jaxpr = jax.make_jaxpr(f)(x)
    names = [e.primitive.name for e in jaxpr.eqns]
    assert names == ["reshape", "transpose", "reshape"], names
    ref = x.reshape(2, 1, 3, 4, 1).transpose([4, 3, 1, 0, 2]).reshape(4, 1, 2, 3).transpose([1, 0, 2, 3]).reshape(8, 3)
    np.testing.assert_array_equal(np.asarray(f(x)), np.asarray(ref))


def test_a_regrouping_the_atoms_cannot_express_materializes_once():
    x = jnp.arange(24.0, dtype=jnp.float32).reshape(2, 3, 4)

    def f(a):
        v = _View(a).reshape(2, 1, 12).transpose([2, 1, 0]).reshape(4, 3, 1, 2)
        return v.transpose([1, 3, 0, 2]).reshape(6, 4).materialize()

    ref = x.reshape(2, 1, 12).transpose([2, 1, 0]).reshape(4, 3, 1, 2).transpose([1, 3, 0, 2]).reshape(6, 4)
    np.testing.assert_array_equal(np.asarray(f(x)), np.asarray(ref))
    names = [e.primitive.name for e in jax.make_jaxpr(f)(x).eqns]
    assert names.count("reshape") <= 3 and names.count("transpose") <= 2, names


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


KEY = jax.random.PRNGKey(7)
A = jax.random.normal(KEY, (2, 8), jnp.float32)
B = jax.random.normal(jax.random.fold_in(KEY, 1), (2, 8), jnp.float32)


def test_a_product_of_two_diagonal_partials_is_one_mul():
    # Fails on the frame emission: a batch-only dot_general in unit-axis reshapes.
    f = lambda x, y: (_diag(x) @ _diag(y)).val
    jaxpr = jax.make_jaxpr(f)(A, B)
    names = _prims(jaxpr)
    assert not [n for n in names if n in SHAPE_PRIMS or n == "dot_general"], names
    assert any(_reads_the_inputs(e, jaxpr) for e in _eqns(jaxpr, "mul")), names
    np.testing.assert_array_equal(np.asarray(f(A, B)), np.asarray(A * B))
    res = _diag(A) @ _diag(B)
    np.testing.assert_array_equal(np.asarray(res.dense()), np.asarray(_diag(A * B).dense()))


def test_a_diagonal_against_a_dense_partial_contracts_without_unit_axes():
    w = jax.random.normal(jax.random.fold_in(KEY, 2), (2, 8, 3), jnp.float32)
    f = lambda x, y: (_diag(x) @ _dense(y, 2)).val
    jaxpr = jax.make_jaxpr(f)(A, w)
    dots = _eqns(jaxpr, "dot_general")
    assert len(dots) == 1, _prims(jaxpr)
    for v in dots[0].invars:
        assert 1 not in tuple(v.aval.shape), v.aval.shape
    assert not [e for e in jaxpr.eqns if e.primitive.name in SHAPE_PRIMS], _prims(jaxpr)
    np.testing.assert_allclose(np.asarray(f(A, w)), np.asarray(A[:, :, None] * w), rtol=1e-6, atol=1e-6)


def test_a_real_contraction_matches_the_dense_reference():
    x = jax.random.normal(jax.random.fold_in(KEY, 3), (4, 2, 8), jnp.float32)
    y = jax.random.normal(jax.random.fold_in(KEY, 4), (2, 8, 3), jnp.float32)
    res = _dense(x, 1) @ _dense(y, 2)
    want = jnp.einsum("abc,bcd->ad", x, y)
    np.testing.assert_allclose(np.asarray(res.dense()), np.asarray(want), rtol=1e-5, atol=1e-5)
    jaxpr = jax.make_jaxpr(lambda a, b: (_dense(a, 1) @ _dense(b, 2)).val)(x, y)
    for v in _eqns(jaxpr, "dot_general")[0].invars:
        assert 1 not in tuple(v.aval.shape), v.aval.shape


def test_a_narrow_pair_multiplies_narrow_and_matches_the_f32_product_rounded():
    a, b = A.astype(jnp.bfloat16), B.astype(jnp.bfloat16)
    res = _diag(a) @ _diag(b)
    assert jnp.dtype(res.val.dtype) == jnp.dtype(jnp.bfloat16)
    want = (a.astype(jnp.float32) * b.astype(jnp.float32)).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(res.val, np.float32), np.asarray(want, np.float32))
    jaxpr = jax.make_jaxpr(lambda x, y: (_diag(x) @ _diag(y)).val)(a, b)
    assert not [e for e in jaxpr.eqns if e.primitive.name == "convert_element_type"
                and _reads_an_input(e, jaxpr)], _prims(jaxpr)
    muls = [e for e in _eqns(jaxpr, "mul") if _reads_the_inputs(e, jaxpr)]
    assert muls and jnp.dtype(muls[0].outvars[0].aval.dtype) == jnp.dtype(jnp.bfloat16), _prims(jaxpr)


def test_an_operand_that_stores_nothing_is_not_multiplied():
    ident = SparseTensor(
        (DiagonalIndex(0, 2, None, 2), DiagonalIndex(1, 8, None, 3)),
        (DiagonalIndex(2, 2, None, 0), DiagonalIndex(3, 8, None, 1)),
        None, scalar_mult=jnp.asarray(3.0, jnp.float32),
    )
    f = lambda x: (ident @ _diag(x)).val
    jaxpr = jax.make_jaxpr(f)(A)
    names = _prims(jaxpr)
    assert not [n for n in names if n in SHAPE_PRIMS or n == "dot_general"], names
    assert not any(_reads_an_input(e, jaxpr) for e in jaxpr.eqns), names
    res = ident @ _diag(A)
    np.testing.assert_array_equal(np.asarray(res.val), np.asarray(A))
    np.testing.assert_allclose(np.asarray(res.dense()), 3.0 * np.asarray(_diag(A).dense()), rtol=1e-6)


@pytest.mark.parametrize("seed", range(30))
def test_broadcasting_a_view_equals_broadcasting_the_array(seed):
    rng = np.random.default_rng(1000 + seed)
    base_shape = tuple(int(d) for d in rng.choice([1, 2, 3], size=int(rng.integers(1, 5))))
    arr = jnp.asarray(rng.standard_normal(base_shape).astype(np.float32))
    view, eager = _View(arr), arr
    if rng.random() < 0.5:
        perm = [int(p) for p in rng.permutation(eager.ndim)]
        view, eager = view.transpose(perm), eager.transpose(perm)
    if rng.random() < 0.5:
        shape = list(eager.shape)
        shape.insert(int(rng.integers(0, len(shape) + 1)), 1)
        view, eager = view.reshape(shape), eager.reshape(shape)
    target = tuple(int(rng.integers(1, 4)) if d == 1 else d for d in eager.shape)
    got = view.broadcast_to(target).materialize()
    np.testing.assert_array_equal(np.asarray(got), np.asarray(jnp.broadcast_to(eager, target)))


def test_a_stand_in_axis_broadcasts_without_a_reshape():
    x = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 1, 4)
    f = lambda a: _View(a).broadcast_to((3, 5, 4)).materialize()
    assert _prims(jax.make_jaxpr(f)(x)) == ["broadcast_in_dim"]
    np.testing.assert_array_equal(np.asarray(f(x)), np.asarray(jnp.broadcast_to(x, (3, 5, 4))))
