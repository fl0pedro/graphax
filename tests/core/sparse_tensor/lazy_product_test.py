"""A product with no label summed between its operands stays lazy, and the
reader that materializes it gets it in its own axis order (dsnn-dfw.272).

The NN256 Markowitz step multiplies W2[o, h] by tanh'[b, h] over the shared h
and stores the product as (b, o, h). Emitted as a dot_general, the product came
out as (h, o, b), and the 33 GB Jacobian fusion after it read that layout with a
stride of 40960 floats (1.79x slower on pgi15-gpu19, job 68352)."""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from graphax.sparse.ops.matmul import _emit_einsum
from graphax.sparse.ops.view import _LazyProduct, _View

KEY = jax.random.PRNGKey(11)


def _n(shape, i):
    return jax.random.normal(jax.random.fold_in(KEY, i), shape, jnp.float32)


def _prims(fn, *args):
    return [e.primitive.name for e in jax.make_jaxpr(fn)(*args).jaxpr.eqns]


def test_the_lazy_product_is_the_outer_product_in_every_order():
    x, y = _n((5, 3), 0), _n((4, 3), 1)                  # x[o, h], y[b, h]
    p = _LazyProduct(x, [1, None, 0], y, [1, 0, None], (3, 4, 5))   # axes (h, b, o)
    want = np.einsum("oh,bh->hbo", np.asarray(x), np.asarray(y))
    for perm in itertools.permutations(range(3)):
        np.testing.assert_array_equal(np.asarray(p.emit(perm)), np.transpose(want, perm))


def test_a_view_of_a_lazy_product_emits_no_transpose():
    x, y = _n((5, 3), 2), _n((4, 3), 3)

    def f(x, y):
        v = _View(_LazyProduct(x, [1, None, 0], y, [1, 0, None], (3, 4, 5)))
        return v.transpose([1, 2, 0]).reshape(20, 3).materialize()

    assert "transpose" not in _prims(f, x, y), _prims(f, x, y)
    want = np.einsum("oh,bh->boh", np.asarray(x), np.asarray(y)).reshape(20, 3)
    np.testing.assert_array_equal(np.asarray(f(x, y)), want)


def test_a_batch_only_product_is_written_in_the_order_it_is_read():
    w, t = _n((10, 256), 4), _n((64, 256), 5)             # w[o, h], t[b, h]

    def f(w, t):
        return _emit_einsum(w, [1, 2], t, [0, 2], [0, 1, 2]).materialize()   # (b, o, h)

    prims = _prims(f, w, t)
    assert "dot_general" not in prims and "transpose" not in prims, prims
    assert prims.count("mul") == 1, prims
    want = np.einsum("oh,bh->boh", np.asarray(w), np.asarray(t))
    np.testing.assert_array_equal(np.asarray(f(w, t)), want)


def test_a_narrow_batch_only_product_is_a_plain_bf16_multiply():
    w, t = _n((10, 256), 6).astype(jnp.bfloat16), _n((64, 256), 7).astype(jnp.bfloat16)

    def f(w, t):
        return _emit_einsum(w, [1, 2], t, [0, 2], [0, 1, 2]).materialize()

    got = f(w, t)
    assert jnp.dtype(got.dtype) == jnp.dtype(jnp.bfloat16)
    want = (w.astype(jnp.float32)[None, :, :] * t.astype(jnp.float32)[:, None, :]).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(want, np.float32))


def test_a_product_read_in_a_dot_generals_order_is_that_dot():
    # The shared axis first, then each operand's own axes in its order: one
    # dot_general makes exactly that, with no broadcast and no transpose.
    x, y = _n((3, 4), 8), _n((3, 5), 9)                  # x[a, b], y[a, d]

    def f(x, y):
        return _emit_einsum(x, [0, 1], y, [0, 2], [0, 1, 2]).materialize()   # (a, b, d)

    prims = _prims(f, x, y)
    assert prims == ["dot_general"], prims
    np.testing.assert_array_equal(np.asarray(f(x, y)), np.einsum("ab,ad->abd", np.asarray(x), np.asarray(y)))


def test_a_product_read_with_the_shared_axis_first_is_the_dot_and_one_transpose():
    # The reader puts the shared axis first and interleaves the operands' own
    # axes: the dot over the shared axis, then one transpose of the own axes.
    # This is the form 50e3aed emitted. As two sibling multiplies, two such
    # products of TLM free_w8_s2_exact were fused into one kernel and doubled
    # its temp bytes (job 68372).
    x, y = _n((3, 4), 10), _n((3, 5, 2), 11)             # x[a, b], y[a, d, e]

    def f(x, y):
        return _emit_einsum(x, [0, 1], y, [0, 2, 3], [0, 2, 1, 3]).materialize()   # (a, d, b, e)

    prims = _prims(f, x, y)
    assert prims == ["dot_general", "transpose"], prims
    want = np.einsum("ab,ade->adbe", np.asarray(x), np.asarray(y))
    np.testing.assert_array_equal(np.asarray(f(x, y)), want)
