import jax.numpy as jnp
import numpy as np
import pytest

from graphax.sparse.tensor import (
    BlockSparseTensor,
    DenseDimension,
    SparseDimension,
)

# this is too complicated for a test...
# def to_dense_reference(st):
#     shape = st.shape
#     res = jnp.zeros(shape)

#     dims = st.out_dims + st.primal_dims
#     val = st.val if st.val is not None else jnp.array(1.0)

#     for l_idx in np.ndindex(shape):
#         v_idx = [0] * val.ndim
#         possible = True
#         for i, (li, d) in enumerate(zip(l_idx, dims)):
#             if isinstance(d, SparseDimension):
#                 b_idx = li // d.block_size
#                 blk_idx = li % d.block_size
#                 partner_idx = next(j for j, o in enumerate(dims) if o.id == d.other_id)
#                 p_d = dims[partner_idx]
#                 if l_idx[partner_idx] // p_d.block_size != b_idx:
#                     possible = False
#                     break
#                 if d.val_dim is not None:
#                     v_idx[d.val_dim] = b_idx
#                 if d.block_val_dim is not None:
#                     v_idx[d.block_val_dim] = blk_idx
#             else:
#                 if d.val_dim is not None:
#                     v_idx[d.val_dim] = li
#         if possible:
#             res = res.at[l_idx].set(val[tuple(v_idx)] * st.scalar_mult)
#     return res


@pytest.fixture
def sta():
    x = jnp.arange(100).reshape(100)
    return BlockSparseTensor(
        [SparseDimension(0, 100, 0, 1)], [SparseDimension(1, 100, 0, 0)], x
    )


@pytest.fixture
def stb():
    x = jnp.arange(4 * 5 * 6).reshape(4, 5, 6)
    return BlockSparseTensor(
        [SparseDimension(0, 4, 0, 1, 5, 1)], [SparseDimension(1, 4, 0, 0, 6, 2)], x
    )


@pytest.fixture
def stc():
    x = jnp.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    return BlockSparseTensor(
        [SparseDimension(0, 3, 0, 2, 4, 1)],
        [DenseDimension(1, 5, 2), SparseDimension(2, 3, 0, 0, 6, 3)],
        x,
    )


@pytest.fixture
def std():
    x = jnp.arange(3 * 4).reshape(3, 4)
    return BlockSparseTensor(
        [SparseDimension(0, 3, 0, 2), SparseDimension(1, 4, 1, 3)],
        [SparseDimension(2, 3, 0, 0), SparseDimension(3, 4, 1, 1)],
        x,
    )


@pytest.fixture
def ste():
    x = jnp.arange(3 * 4 * 5 * 6 * 7 * 8).reshape(3, 4, 5, 6, 7, 8)
    return BlockSparseTensor(
        [SparseDimension(0, 3, 0, 2, 5, 2), SparseDimension(1, 4, 1, 3, 6, 3)],
        [SparseDimension(2, 3, 0, 0, 7, 4), SparseDimension(3, 4, 1, 1, 8, 5)],
        x,
    )


@pytest.fixture
def stf():
    x = jnp.arange(3 * 4 * 5 * 6 * 7).reshape(3, 4, 5, 6, 7)
    return BlockSparseTensor(
        [SparseDimension(0, 3, 0, 2, 4, 1), DenseDimension(1, 5, 2)],
        [SparseDimension(2, 3, 0, 0, 6, 3), DenseDimension(3, 7, 4)],
        x,
    )


@pytest.mark.parametrize("st_name", ["sta", "stb", "stc", "std", "ste", "stf"])
def test_transpose_involutive(st_name, request):
    st = request.getfixturevalue(st_name)
    assert jnp.allclose(jnp.array(st), jnp.array(st.T.T))
    assert jnp.allclose(jnp.array(st.T), jnp.array(st.T.T.T))
    assert jnp.allclose(jnp.array(st), jnp.array(st.swapdims().swapdims()))


@pytest.mark.parametrize("st_name", ["sta", "stb", "stc", "std", "ste", "stf"])
def test_dense_transpose_match(st_name, request):
    st = request.getfixturevalue(st_name)
    n_out = len(st.out_dims)
    n_primal = len(st.primal_dims)
    perm = tuple(range(n_out, n_out + n_primal)) + tuple(range(n_out))
    assert jnp.allclose(jnp.array(st.T), jnp.array(st).transpose(perm))


def test_transpose_one_sparse(stb):
    assert jnp.allclose(jnp.array(stb.T), jnp.array(stb).T)
    assert jnp.allclose(
        jnp.array(stb.transpose(out_perm=(1,), primal_perm=(0,))), jnp.array(stb.T)
    )


def test_transpose_mixed(stc):
    assert jnp.allclose(jnp.array(stc.T), jnp.array(stc).transpose(1, 2, 0))
    assert jnp.allclose(
        jnp.array(stc.transpose((2, 1), (0,))), jnp.array(stc).transpose(2, 1, 0)
    )


def test_transpose_two_sparse(ste):
    assert jnp.allclose(jnp.array(ste.T), jnp.array(ste).transpose(2, 3, 0, 1))
    assert jnp.allclose(jnp.array(ste.swapdims()), jnp.array(ste.T))


# @pytest.mark.parametrize("st_name", ["sta", "stb", "stc", "std", "ste", "stf"])
# def test_consistency(st_name, request):
#     st = request.getfixturevalue(st_name)
#     assert jnp.allclose(jnp.array(st), to_dense_reference(st))
