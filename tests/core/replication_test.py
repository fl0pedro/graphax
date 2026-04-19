import jax
import jax.numpy as jnp
import jax.random as jrand
import pytest
from jax.scipy.linalg import block_diag

from graphax.sparse.tensor import BlockSparseTensor, DenseDimension, SparseDimension
from graphax.sparse.tensor import BlockSparseTensor as SparseTensor


class TestReplicationMul:
    ### Replication tests
    def test_simple_replication(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4,))
        _x = jnp.expand_dims(x, 1)
        _x = jnp.tile(_x, (1, 3))
        y = jrand.normal(ykey, (3, 2))
        res = _x @ y

        stx = SparseTensor([DenseDimension(0, 4, 0)], [DenseDimension(1, 3, None)], x)
        sty = SparseTensor([DenseDimension(0, 3, 0)], [DenseDimension(1, 2, 1)], y)
        stres = stx @ sty
        assert isinstance(stres, SparseTensor) and stres.val is not None

        assert jnp.allclose(res, stres.val)

    def test_block_simple_replication(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)

        x_val = jrand.normal(xkey, (2, 2, 2))
        x_bd = block_diag(*x_val)

        x_full = jnp.expand_dims(x_bd, 2)
        x_full = jnp.tile(x_full, (1, 1, 3))

        y = jrand.normal(ykey, (3, 2))

        res = jnp.einsum("ijk,kl->ijl", x_full, y)

        stx = BlockSparseTensor(
            [SparseDimension(0, 2, 0, 1, 2, 1), SparseDimension(1, 2, 0, 0, 2, 2)],
            [DenseDimension(2, 3, None)],
            x_val,
        )

        sty = BlockSparseTensor([DenseDimension(0, 3, 0)], [DenseDimension(1, 2, 1)], y)

        stres = stx @ sty

        assert jnp.allclose(res, jnp.array(stres))

    def test_double_replication(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4,))
        _x = jnp.expand_dims(x, 1)
        _x = jnp.tile(_x, (1, 3))

        y = jrand.normal(ykey, (2,))
        _y = jnp.expand_dims(y, 0)
        _y = jnp.tile(_y, (3, 1))
        res = _x @ _y

        stx = SparseTensor([DenseDimension(0, 4, 0)], [DenseDimension(1, 3, None)], x)
        sty = SparseTensor([DenseDimension(0, 3, None)], [DenseDimension(1, 2, 0)], y)
        stres = stx @ sty

        assert jnp.allclose(res, jnp.array(stres))

    def test_replication_2d(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)

        x = jrand.normal(xkey, (4,))
        _x = jnp.eye(4) * x
        _x = jnp.expand_dims(_x, 2)
        _x = jnp.tile(_x, (1, 1, 5))

        y = jrand.normal(ykey, (4, 5))
        _y = jnp.einsum("ij,jk->ijk", y, jnp.eye(5))

        res = jnp.einsum("ijk,jkl->il", _x, _y)

        stx = SparseTensor(
            [SparseDimension(0, 4, 0, 1)],
            [SparseDimension(1, 4, 0, 0), DenseDimension(2, 5, None)],
            x,
        )
        sty = SparseTensor(
            [DenseDimension(0, 4, 0), SparseDimension(1, 5, 1, 2)],
            [SparseDimension(2, 5, 1, 1)],
            y,
        )
        stres = stx @ sty

        assert jnp.allclose(res, jnp.array(stres))

    def test_replication_2d_2nd(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)

        x = jrand.normal(xkey, (4, 5))
        _x = jnp.einsum("ij,ik->ikj", x, jnp.eye(4))

        y = jrand.normal(ykey, (5,))
        _y = jnp.eye(5) * y
        _y = jnp.expand_dims(_y, 0)
        _y = jnp.tile(_y, (4, 1, 1))

        res = jnp.einsum("ijk,jkl->il", _x, _y)

        stx = SparseTensor(
            [SparseDimension(0, 4, 0, 1)],
            [SparseDimension(1, 4, 0, 0), DenseDimension(2, 5, 1)],
            x,
        )
        sty = SparseTensor(
            [DenseDimension(0, 4, None), SparseDimension(1, 5, 0, 2)],
            [SparseDimension(2, 5, 0, 1)],
            y,
        )
        stres = stx @ sty

        assert jnp.allclose(res, jnp.array(stres))

    def test_4d_replication(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (3, 4, 5))
        d = jnp.eye(4)
        _x = jnp.einsum("ijk,jl->ijlk", x, d)

        y = jrand.normal(ykey, (5, 2))
        _y = jnp.expand_dims(y, 0)
        _y = jnp.tile(_y, (4, 1, 1))
        res = jnp.einsum("ijkl,klm->ijm", _x, _y)

        stx = SparseTensor(
            [DenseDimension(0, 3, 0), SparseDimension(1, 4, 1, 2)],
            [SparseDimension(2, 4, 1, 1), DenseDimension(3, 5, 2)],
            x,
        )
        sty = SparseTensor(
            [DenseDimension(0, 4, None), DenseDimension(1, 5, 0)],
            [DenseDimension(2, 2, 1)],
            y,
        )
        stres = stx @ sty

        assert jnp.allclose(res, jnp.array(stres))

    def test_block_replication_mixed(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)

        # Replicate a block sparse tensor along a new dimension
        # x: (4, 4) Block Diag (blocks 2x2).
        # Broadcast to (5, 4, 4).

        x_val = jrand.normal(xkey, (2, 2, 2))
        x_bd = block_diag(*x_val)  # (4, 4)

        # y: (4, 4) Block Diag (blocks 2x2).
        # Batched? No.
        # We want x (5, 4, 4) @ y (4, 4) -> (5, 4, 4).

        # But replication test usually involves None val_dims.

        # Case:
        # x: [Dense(0, 5, None), Sparse(1, 4, b=2), Sparse(2, 4, b=2)]
        # y: [Sparse(0, 4, b=2), Sparse(1, 4, b=2)]

        # x_val (2, 2, 2).
        # y_val (2, 2, 2).

        y_val = jrand.normal(ykey, (2, 2, 2))
        y_bd = block_diag(*y_val)

        x_full = jnp.broadcast_to(x_bd, (5, 4, 4))

        res = x_full @ y_bd  # (5, 4, 4) @ (4, 4) -> (5, 4, 4)

        stx = BlockSparseTensor(
            [DenseDimension(0, 5, None), SparseDimension(1, 2, 0, 2, 2, 1)],
            [SparseDimension(2, 2, 0, 1, 2, 2)],
            x_val,
        )
        # Note: SparseDimension 1 pairs with 2.

        sty = BlockSparseTensor(
            [SparseDimension(0, 2, 0, 1, 2, 1)],
            [SparseDimension(1, 2, 0, 0, 2, 2)],
            y_val,
        )

        stres = stx @ sty
        assert isinstance(stres, SparseTensor) and stres.val is not None

        res_val = stres.val
        # Result should be (5, 2, 2, 2).
        # We need to densify block dims to compare with res.
        # Or compare block vals.

        # Manually compute block product
        res_blocks = jnp.einsum("ijk,ikl->ijl", x_val, y_val)  # (2, 2, 2)
        # Broadcasted over 5?
        # The result val will have shape (5, 2, 2, 2) ?

        # Wait, stx @ sty.
        # x Out: Dense(0, 5), Sparse(1, 4).
        # x Primal: Sparse(2, 4).
        # y Out: Sparse(0, 4).
        # x Primal matches y Out.
        # Result Out: Dense(0, 5), Sparse(1, 4)
        # Result Primal: Sparse(1, 4) from y?

        # Contraction:
        # x(b, i, j). y(j, k) -> (b, i, k).
        # i, j, k are block-sparse.
        # Block arithmetic:
        # For each block b: x_block[b] @ y_block[b]
        # x has dim 0 (5).
        # Result val should definitely have dim 0 size 5.
        # And dim 1 size 2 (num blocks).
        # And dim 2, 3 size 2 (block size).

        # Contraction:
        # x(b, i, j). y(j, k) -> (b, i, k).
        # i, j, k are block-sparse.
        # Block arithmetic:
        # For each block b: x_block[b] @ y_block[b]
        # x has dim 0 (5).
        # Result val should definitely have dim 0 size 5.
        # And dim 1 size 2 (num blocks).
        # And dim 2, 3 size 2 (block size).

        # Verify dense result
        assert jnp.allclose(res, jnp.array(stres))
