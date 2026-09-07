"""The tiled executor keeps a one-sided primal dim's logical extent.

Finding 61, verdict 6 (ticket dsnn-3qm.67): on NeuralNetwork, reverse order,
Reduce(mean) on slot ``new`` of every face, the tiled engine returned ``W1``
and ``Wout`` with logical sizes ``[1, 1]``.  The last contraction of each of
those chains is ``X @ s``, where ``X`` carries only primal dims and ``s`` is a
rank-0 SparseTensor.  Every primal dim of ``X`` is then an unmatched
``spatial_primal_lhs`` pair, whose extent the topology stores in
``PairData.shared_block_len`` — that is, on the RHS side of the output grid.
``_build_pair_dims`` read it from the LHS side, which is always 1 for that
pairing, so both the size and the stored values collapsed.

These are value tests, not layout tests: the contraction is a scale, so the
result must equal ``X.dense() * s.dense()`` element for element.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.indexes import DenseIndex
from graphax.sparse.tensor import SparseTensor


def _dims(t):
    return tuple(t.out_dims) + tuple(t.primal_dims)


def test_implicit_primal_dims_survive_a_scalar_contraction():
    """Two implicit primal dims times a rank-0 scalar: the extents stay, the
    axes stay implicit, and no buffer is created."""
    lhs = SparseTensor(
        (), (DenseIndex(0, 7, axis=None), DenseIndex(1, 5, axis=None)), None
    )
    rhs = SparseTensor((), (), None, scalar_mult=jnp.asarray(3.0))
    out = lhs @ rhs
    assert [int(d.logical_size) for d in _dims(out)] == [7, 5]
    assert [d.axis for d in _dims(out)] == [None, None]
    assert out.val is None
    np.testing.assert_allclose(
        np.asarray(out.dense()), np.asarray(lhs.dense()) * 3.0, rtol=0, atol=0
    )


def test_stored_primal_dim_survives_a_scalar_contraction():
    """The same pairing with a stored (physical) primal dim: the values must
    ride through, not collapse to the first element."""
    lhs = SparseTensor((), (DenseIndex(0, 5, axis=0),), jnp.arange(5.0))
    rhs = SparseTensor((), (), None, scalar_mult=jnp.asarray(3.0))
    out = lhs @ rhs
    assert [int(d.logical_size) for d in _dims(out)] == [5]
    np.testing.assert_allclose(
        np.asarray(out.dense()), np.arange(5.0) * 3.0, rtol=0, atol=0
    )


def test_mixed_stored_and_implicit_primal_dims():
    """One stored and one implicit primal dim: both extents survive and the
    stored one keeps its values."""
    lhs = SparseTensor(
        (),
        (DenseIndex(0, 4, axis=0), DenseIndex(1, 6, axis=None)),
        jnp.arange(4.0) + 1.0,
    )
    rhs = SparseTensor((), (), None, scalar_mult=jnp.asarray(2.0))
    out = lhs @ rhs
    assert [int(d.logical_size) for d in _dims(out)] == [4, 6]
    np.testing.assert_allclose(
        np.asarray(out.dense()), np.asarray(lhs.dense()) * 2.0, rtol=0, atol=0
    )


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_out_side_mirror_is_unaffected(n_dims):
    """The mirror pairing (a one-sided OUT dim) was always right; pin it so the
    fix cannot break it."""
    dims = tuple(DenseIndex(i, 3 + i, axis=i) for i in range(n_dims))
    val = jnp.arange(float(np.prod([3 + i for i in range(n_dims)]))).reshape(
        [3 + i for i in range(n_dims)]
    )
    rhs = SparseTensor(dims, (), val)
    s = SparseTensor((), (), None, scalar_mult=jnp.asarray(5.0))
    out = s @ rhs
    assert [int(d.logical_size) for d in _dims(out)] == [3 + i for i in range(n_dims)]
    np.testing.assert_allclose(
        np.asarray(out.dense()), np.asarray(rhs.dense()) * 5.0, rtol=0, atol=0
    )


# --------------------------------------------------------------------------
# Differential test: the lazy frame against the incumbent tiled executor of
# graphax 1f3d404, reached in the same process under GRAPHAX_TILED_LEGACY=1
# (ticket dsnn-3qm.67). Values must agree; the candidate may keep MORE
# structure symbolic, so shapes are compared on the dense form.
# --------------------------------------------------------------------------
import os

from graphax.sparse.indexes import DiagonalIndex


def _legacy_dense(lhs, rhs):
    saved = os.environ.get("GRAPHAX_TILED_LEGACY")
    os.environ["GRAPHAX_TILED_LEGACY"] = "1"
    try:
        return np.asarray((lhs @ rhs).dense())
    finally:
        if saved is None:
            os.environ.pop("GRAPHAX_TILED_LEGACY", None)
        else:
            os.environ["GRAPHAX_TILED_LEGACY"] = saved


def _cases():
    """(name, lhs, rhs) pairs that exercise each lazy rule."""
    out = []
    # a diagonal pair neither side stores, contracted with a stored diagonal
    out.append((
        "unstored_diag_x_stored_diag",
        SparseTensor((DiagonalIndex(0, 6, None, 1),),
                     (DiagonalIndex(1, 6, None, 0),), None),
        SparseTensor((DiagonalIndex(0, 6, 0, 1),),
                     (DiagonalIndex(1, 6, 0, 0),), jnp.arange(6.0) + 1.0),
    ))
    # both diagonals unstored: the whole meta axis stays symbolic
    out.append((
        "unstored_diag_x_unstored_diag",
        SparseTensor((DiagonalIndex(0, 5, None, 1),),
                     (DiagonalIndex(1, 5, None, 0),), None),
        SparseTensor((DiagonalIndex(0, 5, None, 1),),
                     (DiagonalIndex(1, 5, None, 0),), None),
    ))
    # a dense contraction against an unstored diagonal
    out.append((
        "dense_x_unstored_diag",
        SparseTensor((DenseIndex(0, 4, axis=0),), (DenseIndex(1, 6, axis=1),),
                     jnp.arange(24.0).reshape(4, 6)),
        SparseTensor((DiagonalIndex(0, 6, None, 1),),
                     (DiagonalIndex(1, 6, None, 0),), None),
    ))
    # an implicit (Reduce-shaped) primal dim riding through a dense contraction
    out.append((
        "implicit_primal_rides_through",
        SparseTensor((DenseIndex(0, 3, axis=0),), (DenseIndex(1, 4, axis=1),),
                     jnp.arange(12.0).reshape(3, 4)),
        SparseTensor((DenseIndex(0, 4, axis=0),),
                     (DenseIndex(1, 7, axis=None),), jnp.arange(4.0) + 1.0),
    ))
    # a contracted extent neither side stores: an analytic scale
    out.append((
        "unstored_contracted_extent",
        SparseTensor((DenseIndex(0, 3, axis=None),),
                     (DenseIndex(1, 5, axis=None),), None),
        SparseTensor((DenseIndex(0, 5, axis=None),),
                     (DenseIndex(1, 2, axis=None),), None),
    ))
    # plain dense x dense: the lazy frame must not fire at all
    out.append((
        "plain_dense",
        SparseTensor((DenseIndex(0, 3, axis=0),), (DenseIndex(1, 4, axis=1),),
                     jnp.arange(12.0).reshape(3, 4)),
        SparseTensor((DenseIndex(0, 4, axis=0),), (DenseIndex(1, 2, axis=1),),
                     jnp.arange(8.0).reshape(4, 2)),
    ))
    return out


@pytest.mark.parametrize("name,lhs,rhs", _cases(), ids=[c[0] for c in _cases()])
def test_lazy_frame_matches_the_incumbent_dense_form(name, lhs, rhs):
    got = np.asarray((lhs @ rhs).dense())
    want = _legacy_dense(lhs, rhs)
    assert got.shape == want.shape, f"{name}: {got.shape} vs {want.shape}"
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
