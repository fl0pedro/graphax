# The join on the shape view: the pad embedding and a diagonal meeting a dense edge (dsnn-dfw.252).
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.ops.elementwise import _promote_to_unified
from graphax.sparse.ops.layout import generate_block_permutation
from graphax.sparse.ops.view import _View
from graphax.sparse.indexes import static_eye
from graphax.sparse.tensor import SparseTensor

SHAPE_PRIMS = ("reshape", "transpose", "broadcast_in_dim", "squeeze", "expand_dims")


def _old_promote(value, metrics, is_left, fill):
    # The eye-mask select this module used before dsnn-dfw.252.
    in_shape, exp_shape, out_shape = [], [], []
    needs = False
    for m in metrics:
        b1, b2 = (m["left_b1"], m["left_b2"]) if is_left else (m["right_b1"], m["right_b2"])
        exp = m["common_b1"] // b1
        in_shape += [m["unified_size"], exp, b1, b2]
        exp_shape += [m["unified_size"], exp, 1, b1, b2]
        out_shape += [m["unified_size"], m["common_b1"], m["common_b2"]]
        needs = needs or exp > 1
    rem = list(value.shape[3 * len(metrics):])
    value = value.reshape(in_shape + rem).reshape(exp_shape + rem)
    if needs:
        fill = jnp.asarray(fill, value.dtype)
        mask = None
        for i, m in enumerate(metrics):
            b1 = m["left_b1"] if is_left else m["right_b1"]
            exp = m["common_b1"] // b1
            if exp > 1:
                ms = [1] * len(exp_shape + rem)
                ms[5 * i + 1] = ms[5 * i + 2] = exp
                em = static_eye(exp, bool).reshape(ms)
                mask = em if mask is None else mask & em
        value = jnp.where(mask, value, fill)
    perm = generate_block_permutation(len(metrics), 5, [0, 1, 3, 2, 4])
    perm.extend(range(5 * len(metrics), len(exp_shape) + len(rem)))
    return value.transpose(perm).reshape(out_shape + rem)


def _bits(x):
    return np.atleast_1d(np.asarray(x)).view(np.uint8)


@pytest.mark.parametrize("seed", range(40))
def test_the_pad_embedding_writes_the_cells_of_the_eye_mask_select(seed):
    rng = np.random.default_rng(seed)
    metrics, shape = [], []
    for _ in range(int(rng.integers(1, 3))):
        M, exp = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        b1, b2 = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        metrics.append({"unified_size": M, "common_b1": b1 * exp, "common_b2": b2 * exp,
                        "left_b1": b1, "left_b2": b2, "right_b1": b1, "right_b2": b2})
        shape += [M * exp, b1, b2]
    shape += [int(n) for n in rng.integers(2, 4, size=int(rng.integers(0, 2)))]
    x = rng.standard_normal(shape).astype(np.float32)
    flat = x.reshape(-1)
    for k in rng.choice(flat.size, size=min(3, flat.size), replace=False):
        flat[k] = (np.inf, -np.inf, np.nan)[int(k) % 3]
    value = jnp.asarray(x)
    fill = jnp.float32(0.0 if seed % 2 else 1.5)
    got = _promote_to_unified(_View(value), metrics, True, fill)
    got = got.materialize() if isinstance(got, _View) else got
    want = _old_promote(value, metrics, True, fill)
    assert got.shape == want.shape and got.dtype == want.dtype
    np.testing.assert_array_equal(_bits(got), _bits(want))


def _diag(v):
    n = v.shape[0]
    return SparseTensor((DiagonalIndex(0, n, 0, 1),), (DiagonalIndex(1, n, 0, 0),), v)


def _dense(w):
    return SparseTensor((DenseIndex(0, w.shape[0], 0),), (DenseIndex(1, w.shape[1], 1),), w)


def _batched_diag(v):
    b, n = v.shape
    return SparseTensor((DiagonalIndex(0, b, 0, 2), DiagonalIndex(1, n, 1, 3)),
                        (DiagonalIndex(2, b, 0, 0), DiagonalIndex(3, n, 1, 1)), v)


def _batched_dense(w):
    b, n, _ = w.shape
    return SparseTensor((DiagonalIndex(0, b, 0, 2), DenseIndex(1, n, 1)),
                        (DiagonalIndex(2, b, 0, 0), DenseIndex(3, n, 2)), w)


KEY = jax.random.PRNGKey(3)


@pytest.mark.parametrize("seed,order", list(itertools.product(range(6), ("dd", "Dd"))))
def test_a_diagonal_joined_to_a_dense_edge_is_the_dense_sum_bit_for_bit(seed, order):
    k1, k2 = jax.random.split(jax.random.fold_in(KEY, seed))
    n = 3 + seed % 4
    v = jax.random.normal(k1, (n,), jnp.float32)
    w = jax.random.normal(k2, (n, n), jnp.float32)
    a, b = (_diag(v), _dense(w)) if order == "dd" else (_dense(w), _diag(v))
    got = (a + b).dense()
    np.testing.assert_array_equal(_bits(got), _bits(a.dense() + b.dense()))
    bv = jax.random.normal(k1, (2, n), jnp.float32)
    bw = jax.random.normal(k2, (2, n, n), jnp.float32)
    a, b = ((_batched_diag(bv), _batched_dense(bw)) if order == "dd"
            else (_batched_dense(bw), _batched_diag(bv)))
    np.testing.assert_array_equal(_bits((a + b).dense()), _bits(a.dense() + b.dense()))
    np.testing.assert_allclose(np.asarray((a * b).dense()), np.asarray(a.dense() * b.dense()),
                               rtol=1e-6, atol=1e-6)


def _prims(jaxpr):
    out = []
    for e in jaxpr.eqns:
        out.append(e.primitive.name)
        for p in e.params.values():
            if hasattr(p, "jaxpr"):
                out.extend(_prims(p.jaxpr))
    return out


def test_a_diagonal_meeting_a_dense_edge_is_one_pad_and_one_add():
    v = jnp.arange(4.0, dtype=jnp.float32) + 1
    w = jnp.ones((4, 4), jnp.float32)
    names = _prims(jax.make_jaxpr(lambda x, y: (_dense(y) + _diag(x)).val)(v, w))
    assert names.count("pad") == 1 and names.count("add") == 1, names
    assert "select_n" not in names, names
    assert sum(names.count(p) for p in SHAPE_PRIMS) <= 1, names


def test_a_join_of_equal_structures_in_different_layouts_is_one_transpose():
    x = jax.random.normal(KEY, (3, 5), jnp.float32)
    y = jax.random.normal(jax.random.fold_in(KEY, 9), (5, 3), jnp.float32)
    a = SparseTensor((DenseIndex(0, 3, 0),), (DenseIndex(1, 5, 1),), x)
    b = SparseTensor((DenseIndex(0, 3, 1),), (DenseIndex(1, 5, 0),), y)
    names = _prims(jax.make_jaxpr(lambda p, q: (SparseTensor(a.out_dims, a.primal_dims, p)
                                                 + SparseTensor(b.out_dims, b.primal_dims, q)).val)(x, y))
    assert [n for n in names if n in SHAPE_PRIMS] == ["transpose"], names
    np.testing.assert_array_equal(_bits((a + b).dense()), _bits(a.dense() + b.dense()))


def test_a_contraction_transposes_at_most_once():
    # The product is written in the order its result is stored (dsnn-dfw.272):
    # no transpose, and the one broadcast lays the diagonal along the dense axis
    # for the multiply.
    x = jax.random.normal(KEY, (4, 2, 8), jnp.float32)
    d = jax.random.normal(jax.random.fold_in(KEY, 1), (2, 8), jnp.float32)
    lhs = lambda p: SparseTensor((DenseIndex(0, 4, 0),), (DenseIndex(1, 2, 1), DenseIndex(2, 8, 2)), p)
    rhs = lambda q: SparseTensor((DiagonalIndex(0, 2, 0, 2), DiagonalIndex(1, 8, 1, 3)),
                                 (DiagonalIndex(2, 2, 0, 0), DiagonalIndex(3, 8, 1, 1)), q)
    names = _prims(jax.make_jaxpr(lambda p, q: (lhs(p) @ rhs(q)).val)(x, d))
    assert names.count("transpose") <= 1, names
    assert not [n for n in names if n in ("reshape", "squeeze")], names
    assert names.count("broadcast_in_dim") <= 1 and names.count("mul") == 1, names
    want = np.asarray(x) * np.asarray(d)[None]
    np.testing.assert_allclose(np.asarray((lhs(x) @ rhs(d)).dense()), want, rtol=1e-6, atol=1e-6)
