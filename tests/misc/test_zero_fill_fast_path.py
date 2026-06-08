"""Perf-regression guard: a zero-fill matmul must take the fast (``tiled``)
path, NEVER the ``densify`` path.

History: ``_is_zero_fill`` once re-probed the (traced) ``fill_value`` inside
``jit``, where a tracer can't prove it zero, so ``has_nonzero_fill`` came out
True and almost every matmul was force-routed to ``_matmul_via_densify`` —
a massive slowdown. The static-zero marker is now ``fill_value is None``: None
lives in the pytree treedef (static), so ``_is_zero_fill`` is a compile-time
``fill_value is None`` test that survives jit. These tests pin that invariant,
including that it survives chaining and the compressed-output densify boundary
(so a BandedIndex/SetIndex result materialized before the next op stays
fast-path-eligible). The standard test suite is all zero-fill, so without this
guard a regression here would pass silently.
"""
import jax
import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.tensor import SparseTensor
from graphax.sparse.indexes import DiagonalIndex
from graphax.sparse.ops.matmul import matmul
from graphax.sparse.ops.elementwise import elementwise
from graphax.sparse.ops.utils import _is_zero_fill, _materialize_for_op
from graphax.sparse.ops._path_tracking import track_paths
import graphax.sparse.ops._path_tracking as pt


def _n(shape, key=0):
    return jr.normal(jr.PRNGKey(key), shape).astype(jnp.float32)


def _bd(M, B, key):
    return SparseTensor(
        (DiagonalIndex(0, M, 0, 1, B, 1),), (DiagonalIndex(1, M, 0, 0, B, 2),),
        _n((M, B, B), key),
    )


def test_zero_fill_matmul_takes_tiled_not_densify():
    a, b = _bd(4, 3, 1), _bd(4, 3, 2)
    assert a.fill_value is None and _is_zero_fill(a)
    with track_paths() as paths:
        matmul(a, b)
    assert paths[-1] == "tiled", f"zero-fill matmul force-densified: {paths[-1]}"


def test_zero_fill_survives_jit():
    a, b = _bd(4, 3, 1), _bd(4, 3, 2)

    @jax.jit
    def f(x, y):
        return matmul(x, y).val

    _ = f(a, b)
    assert pt.last_path == "tiled", f"jit matmul force-densified: {pt.last_path}"


def test_zero_fill_propagates_through_matmul_chain():
    a, b = _bd(4, 3, 1), _bd(4, 3, 2)
    r = matmul(a, b)
    assert r.fill_value is None and _is_zero_fill(r)
    with track_paths() as paths:
        matmul(r, b)
    assert paths[-1] == "tiled", f"chained matmul force-densified: {paths[-1]}"


def test_compressed_output_stays_fast_path_eligible():
    # A misaligned elementwise → SetIndex output, materialized at the next op
    # boundary, must keep fill_value=None so the consumer stays on the fast path.
    au = SparseTensor((DiagonalIndex(0, 11, 0, 1, 5, 1),),
                      (DiagonalIndex(1, 11, 0, 0, 5, 2),), _n((11, 5, 5), 1))
    bu = SparseTensor((DiagonalIndex(0, 5, 0, 1, 11, 1),),
                      (DiagonalIndex(1, 5, 0, 0, 11, 2),), _n((5, 11, 11), 2))
    s = au + bu
    assert s.fill_value is None
    mat = _materialize_for_op(s)
    assert mat.fill_value is None and _is_zero_fill(mat)
