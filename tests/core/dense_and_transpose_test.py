import jax
import jax.numpy as jnp
import pytest
from jax import vmap
from jax.scipy.linalg import block_diag

from graphax.sparse.block import (
    BlockSparseTensor,
    DenseDimension,
    SparseDimension,
)


@pytest.fixture
def sta():
    x = jnp.arange(100).reshape(100, 1, 1)
    return BlockSparseTensor(
        [SparseDimension(0, 100, 0, 1, 1)], [SparseDimension(1, 100, 1, 0, 1)], x
    )


@pytest.fixture
def stb():
    x = jnp.arange(4 * 5 * 6).reshape(4, 5, 6)
    return BlockSparseTensor(
        [SparseDimension(0, 4, 0, 1, 5)], [SparseDimension(1, 4, 1, 0, 6)], x
    )


@pytest.fixture
def stc():
    x = jnp.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    return BlockSparseTensor(
        [SparseDimension(0, 3, 0, 2, 4)],
        [DenseDimension(1, 5, 1), SparseDimension(2, 3, 2, 0, 6)],
        x,
    )


@pytest.fixture
def std():
    x = jnp.arange(3 * 4).reshape(3, 4, 1, 1, 1, 1)
    return BlockSparseTensor(
        [SparseDimension(0, 3, 0, 2, 1), SparseDimension(1, 4, 1, 3, 1)],
        [SparseDimension(2, 3, 2, 0, 1), SparseDimension(3, 4, 3, 1, 1)],
        x,
    )


@pytest.fixture
def ste():
    x = jnp.arange(3 * 4 * 5 * 6 * 7 * 8).reshape(3, 4, 5, 6, 7, 8)
    return BlockSparseTensor(
        [SparseDimension(0, 3, 0, 2, 5), SparseDimension(1, 4, 1, 3, 6)],
        [SparseDimension(2, 3, 2, 0, 7), SparseDimension(3, 4, 3, 1, 8)],
        x,
    )


# TODO make a permuter for dimensions too.
# make dense
@pytest.fixture
def stf():
    x = jnp.arange(3 * 4 * 5 * 6 * 7).reshape(3, 4, 5, 6, 7)
    return BlockSparseTensor(
        [SparseDimension(0, 3, 0, 2, 4), DenseDimension(1, 5, 1)],
        [SparseDimension(2, 3, 2, 0, 6), DenseDimension(3, 7, 3)],
        x,
    )


# TODO make dense and None cases.
# TODO canonicalize only one dimension ??


def test_block_diagonal_transpose(stc):
    assert (stc.dense().T == stc.T.dense()).all()


def test_transpose_one_sparse(stb):
    n, x, y = stb.val.shape
    assert stb.T.val.shape == (n, y, x)
    assert (stb.transpose((1,), (0,)) == stb.T).all()


@pytest.mark.xfail
def test_bad_transpose_one_sparse(stb):
    stb.transpose((1, 0))


def test_transpose_mixed(stc):
    n, x, a, y = stc.val.shape
    assert stc.T.val.shape == (n, y, a, x)
    assert (stc.transpose((2, 1), (0,)) == stc.T).all()
    assert stc.transpose((1, 0), (2,)).val.shape == (n, a, x, y)
    assert stc.transpose((0, 1), (2,)).val.shape == (n, x, a, y)
    assert stc.transpose((0,), (1, 2)).val.shape == (n, x, a, y)
    assert stc.transpose((0,), (2, 1)).val.shape == (n, x, y, a)


@pytest.mark.xfail
def test_bad_transpose_mixed(stb):
    stb.transpose((1,), (0, 2))


def test_transpose_two_sparse(ste):
    n, m, x, a, y, b = ste.val.shape
    assert ste.T.val.shape == (m, n, b, y, a, x)
    assert ste.swapdims().val.shape == (n, m, y, b, x, a)
    assert ste.transpose(out_transpose=(1, 0)).val.shape == (m, n, a, x, y, b)
    assert ste.transpose(primal_transpose=(3, 2)).val.shape == (n, m, x, a, b, y)
    assert ste.transpose((0, 3), (2, 1)).val.shape == (n, m, x, b, y, a)


@pytest.mark.xfail
def test_bad_transpose_two_sparse(stb):
    stb.transpose((0, 2), (1, 3))


@pytest.mark.parametrize("st_name", ["sta", "stb", "stc", "std", "ste", "stf"])
def test_transpose(st_name, request):
    st = request.getfixturevalue(st_name)
    assert (st == st.T.T).all()
    assert (st.T == st.T.T.T).all()
    assert (st == st.swapdims().swapdims()).all()
    assert (st.swapdims() == st.swapdims().swapdims().swapdims()).all()


@pytest.mark.parametrize("st_name", ["sta", "stb", "stc", "std", "ste", "stf"])
def test_dense_transpose(st_name, request):
    st = request.getfixturevalue(st_name)
    assert (st.T.dense() == st.dense().T).all()
    assert (st.T.T.dense() == st.dense().T.T).all()


@pytest.mark.parametrize("st_name", ["sta", "stb", "stc", "std", "ste", "stf"])
def test_transpose_einsum(st_name, request):
    st = request.getfixturevalue(st_name)
    alphas = "abcdefghijklmnopqrstuvwxyz"

    n = len(st.out_dims)
    m = len(st.primal_dims)

    primal_idxs1 = alphas[:n]
    contract_idxs = alphas[n : n + m]
    primal_idxs2 = alphas[n + m : n + m + n]

    lhs_idxs = primal_idxs1 + contract_idxs
    rhs_idxs = contract_idxs + primal_idxs2
    res_idxs = primal_idxs1 + primal_idxs2

    einsum_statement = f"{lhs_idxs},{rhs_idxs}->{res_idxs}"

    res = jnp.einsum(einsum_statement, st.dense(), st.swapdims().dense())

    assert jnp.allclose(res, (st @ st.swapdims()).dense())
    assert jnp.allclose(res, st @ st.swapdims().dense())
    assert jnp.allclose(res, st.dense() @ st.swapdims())


def test_non_blocK_consistency_one_sparse(sta):
    val = sta.val.reshape(-1)
    b1 = sta.dense()
    b2 = jnp.diag(val).reshape(sta.shape)
    assert b1.shape == b2.shape
    assert (b1 == b2).all()


def test_non_blocK_consistency_two_sparse(std):
    val = std.val.reshape(-1)
    b1 = std.dense()
    b2 = jnp.diag(val).reshape(std.shape)
    assert b1.shape == b2.shape
    assert (b1 == b2).all()


def test_2d_block_consistency(stb):
    b1 = stb.dense()
    b2 = block_diag(*stb.val)
    assert b1.shape == b2.shape
    assert (b1 == b2).all()


def _block_diag_star(val):
    return block_diag(*val)


def test_3d_block_consistency(stc):
    val = stc.val
    b1 = stc.dense()
    diag_over_dense = vmap(_block_diag_star, in_axes=2, out_axes=1)
    b2 = diag_over_dense(val)
    assert b1.shape == b2.shape
    assert (b1 == b2).all()


def test_4d_block_consistency_one_sparse(stf):
    val = stf.val
    b1 = stf.dense()
    diag_over_dense_1 = vmap(_block_diag_star, in_axes=2, out_axes=1)
    diag_over_dense_2 = vmap(diag_over_dense_1, in_axes=4, out_axes=3)
    b2 = diag_over_dense_2(val)
    assert b1.shape == b2.shape
    assert (b1 == b2).all()


def test_4d_block_consistency_two_sparse(ste):
    val = ste.val
    b1 = ste.dense()
    diag_over_sparse_1 = vmap(_block_diag_star, in_axes=1, out_axes=0)
    diag_over_block_dim_1 = vmap(diag_over_sparse_1, in_axes=3, out_axes=2)
    diag_over_block_dim_2 = vmap(diag_over_block_dim_1, in_axes=5, out_axes=4)
    t = diag_over_block_dim_2(val)

    diag_over_block_dim_1 = vmap(_block_diag_star, in_axes=1, out_axes=0)
    diag_over_block_dim_2 = vmap(diag_over_block_dim_1, in_axes=3, out_axes=2)
    b2 = diag_over_block_dim_2(t)

    assert b1.shape == b2.shape
    assert (b1 == b2).all()


# TODO dense transpose

# TODO add block diagonal tests
#  - diag @ diag (symetric)
#  - mixed
#  - pure dense
#  - shuffled
#  - fully dense w/ bst
#  - pure dense w/ bst
#  - block diag w/ non-bloc diag.
