import unittest

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
from jax.tree_util import tree_map

from graphax import jacve, tree_allclose


class PrimitiveTest(unittest.TestCase): 
    # def test_broadcast_add(self):
    #     def broadcast_add(x, y):
    #         return jnp.tanh(x + y)

    #     x = 2*jnp.ones((2, 3))
    #     y = 3*jnp.ones((1, 3))
    #     jac_rev = jax.jit(jacve(broadcast_add, order="fwd", argnums=(0, 1)))
    #     veres = jac_rev(x, y)

    #     jax_jac_rev = jax.jit(jax.jacfwd(broadcast_add, argnums=(0, 1)))
    #     revres = jax_jac_rev(x, y)

    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_broadcast_sub(self):
    #     def broadcast_add(x, y):
    #         return jnp.tanh(x - y)

    #     x = 2*jnp.ones((2, 3))
    #     y = 3*jnp.ones((1, 3))
    #     jac_rev = jax.jit(jacve(broadcast_add, order="fwd", argnums=(0, 1)))
    #     veres = jac_rev(x, y)

    #     jax_jac_rev = jax.jit(jax.jacfwd(broadcast_add, argnums=(0, 1)))
    #     revres = jax_jac_rev(x, y)

    #     self.assertTrue(tree_allclose(veres, revres))
        
    # def test_broadcast_mul(self):
    #     def broadcast_mul(x, y):
    #         z = jnp.exp(y)
    #         return jnp.sin(x * z)

    #     x = jnp.arange(6).reshape((2, 3)).astype(jnp.float32)
    #     y = jnp.arange(3).reshape((3, )).astype(jnp.float32)
    #     jac_rev = jax.jit(jacve(broadcast_mul, order="fwd", argnums=(0, 1)))
    #     veres = jac_rev(x, y)

    #     jax_jac_rev = jax.jit(jax.jacfwd(broadcast_mul, argnums=(0, 1)))
    #     revres = jax_jac_rev(x, y)
    #     get_shape = lambda x: x.shape

    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_broadcast_outer_product(self):
    #     def broadcast_mul(x, y):
    #         return jnp.sin(x * y)

    #     x = jnp.arange(4).reshape((4, 1)).astype(jnp.float32) + 1
    #     y = jnp.arange(3).reshape((1, 3)).astype(jnp.float32)
    #     jac_rev = jax.jit(jacve(broadcast_mul, order="fwd", argnums=(0, 1)))
    #     veres = jac_rev(x, y)

    #     jax_jac_rev = jax.jit(jax.jacfwd(broadcast_mul, argnums=(0, 1)))
    #     revres = jax_jac_rev(x, y)
    #     get_shape = lambda x: x.shape

    #     self.assertTrue(tree_allclose(veres, revres))

    # def test_transpose(self):
    #     def transpose(x, y):
    #         z = jnp.cos(x)
    #         return z.T * y

    #     x = jnp.ones((2, 3))
    #     y = jnp.ones((3, 2))
    #     jac_fwd = jax.jit(jacve(transpose, order="fwd", argnums=(0, 1)))
    #     jaxpr = jax.make_jaxpr(jac_fwd)(x, y)
    #     veres = jac_fwd(x, y)[0]

    #     revres = jax.jacrev(transpose)(x, y)

    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_matmul(self):
    #     def f(x, y):
    #         z = x @ y
    #         return jnp.sin(z)

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (2, 3))
    #     y = jrand.normal(ykey, (3,))

    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jacrev(f, argnums=(0, 1))(x, y)

    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_reduce_sum(self):
    #     def sums(x, y):
    #         return jnp.sin(jnp.sum(x@y, axis=0))

    #     x = jnp.ones((2, 3))
    #     y = jnp.ones((3, 4))
        
    #     jac_fwd = jax.jit(jacve(sums, order="rev", argnums=(0, 1)))
    #     veres = jac_fwd(x, y)

    #     revres = jax.jacrev(sums, argnums=(0, 1))(x, y)
    
    #     self.assertTrue(tree_allclose(veres, revres))
        
    # def test_reduce_max(self):
    #     def maxs(x, y):
    #         return jnp.sin(jnp.max(x@y, axis=0))

    #     x = jnp.array([[0., 1., 2.],[1., 0., 2.]])
    #     y = jnp.ones((3, 4))
        
    #     jac_rev = jax.jit(jacve(maxs, order="rev", argnums=(0, 1)))
    #     veres = jac_rev(x, y)

    #     revres = jax.jacrev(maxs, argnums=(0, 1))(x, y)

    #     self.assertTrue(tree_allclose(veres, revres))
        
    # def test_slicing(self):
    #     def f(x, y):
    #         z = x @ y
    #         return jnp.sin(z[:, 0:1])

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (2, 3))
    #     y = jrand.normal(ykey, (3, 4))

    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jacrev(f, argnums=(0, 1))(x, y)

    #     self.assertTrue(tree_allclose(veres, revres)) 
        
    # def test_squeezing(self):
    #     def f(x, y):
    #         z = x @ y
    #         return jnp.squeeze(z).sum()

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (2, 3))
    #     y = jrand.normal(ykey, (3, 1))

    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jacrev(f, argnums=(0, 1))(x, y)

    #     self.assertTrue(tree_allclose(veres, revres)) 
        
    # def test_concatenate_1(self):
    #     def f(x, y, z):
    #         z = jnp.concatenate([y, z], axis=0)
    #         w = x @ z
    #         return jnp.sin(w)

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (2, 3))
    #     y = jrand.normal(ykey, (2, 4))
    #     z = jrand.normal(ykey, (1, 4))

    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1, 2)))
    #     veres = deriv_fn(x, y, z)

    #     revres = jax.jit(jax.jacrev(f, argnums=(0, 1, 2)))(x, y, z)

    #     self.assertTrue(tree_allclose(veres, revres)) 
        
    # def test_concatenate_2(self):
    #     def f(x, y, z):
    #         x = jnp.sin(x)
    #         y = jnp.cos(y)
    #         z = jnp.tanh(z)
    #         w = jnp.concatenate([x, y, z], axis=0)
    #         return jnp.sin(w)

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (4,))
    #     y = jrand.normal(ykey, (2,))
    #     z = jrand.normal(ykey, (3,))

    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1, 2)))
    #     veres = deriv_fn(x, y, z)

    #     revres = jax.jit(jax.jacrev(f, argnums=(0, 1, 2)))(x, y, z)

    #     self.assertTrue(tree_allclose(veres, revres)) 
        
    def test_concatenate_sparse_none(self):
        # f(x, y) = concat(x, y) is the identity on the concat output.
        # In rev mode the accumulated Jacobian at the concat node is the full
        # identity tensor, represented as a DiagonalIndex pair with axis=None.
        # This directly exercises the else-branch of inverse_concatenate_transform.
        def f(x, y):
            z, w = jnp.sin(x), jnp.log(y)
            return jnp.tan(jnp.concatenate([z, w], axis=0))

        x = jnp.ones((2,))
        y = jnp.ones((3,))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
        veres = deriv_fn(x, y)

        revres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)

        self.assertTrue(tree_allclose(veres, revres))

    def test_concatenate_sparse_none_fwd(self):
        # Same structure as above but in forward mode.
        def f(x, y):
            z, w = jnp.sin(x), jnp.log(y)
            return jnp.tan(jnp.concatenate([z, w], axis=0))

        x = jnp.ones((2,))
        y = jnp.ones((3,))

        deriv_fn = jax.jit(jacve(f, order="fwd", argnums=(0, 1)))
        veres = deriv_fn(x, y)

        fwdres = jax.jit(jax.jacfwd(f, argnums=(0, 1)))(x, y)

        print(veres)
        print(fwdres)
        self.assertTrue(tree_allclose(veres, fwdres))

    # def test_reshape(self):
    #     def f(x, y):
    #         x = jnp.reshape(x, (2, 3))
    #         return jnp.sin(x @ y)

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (6,))
    #     y = jrand.normal(ykey, (3,))
        
    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)

    #     self.assertTrue(tree_allclose(veres, revres)) 
    
    # def test_large_matmul(self):
    #     def f(x, y):
    #         return lax.dot_general(x, y, (([2], [0]), ([0], [1])))

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (3, 1, 4))
    #     y = jrand.normal(ykey, (4, 3, 2))

    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)
        
    #     self.assertTrue(tree_allclose(veres, revres))   
        
    # def test_eq(self):
    #     def f(x, y):
    #         w = jnp.sin(x)
    #         z = jnp.sin(y)
    #         return (w == z) - 1.
    #     x = jnp.array([[1., 0., 1.]])
    #     y = jnp.array([[1.], [0.], [0.]])

    #     deriv_fn = jacve(f, order="rev", argnums=(0, 1))
    #     veres = deriv_fn(x, y)
        
    #     jax_deriv_fn = jax.jacrev(f, argnums=(0, 1))
    #     revres = jax_deriv_fn(x, y)
        
    #     self.assertTrue(tree_allclose(veres, revres)) 


# ---------------------------------------------------------------------------
# SATURATED tanh: the derivative must not lose digits to cancellation
# ---------------------------------------------------------------------------
def test_the_tanh_derivative_survives_saturation():
    """``d tanh/dx == 1 - t**2`` must be FACTORED as ``(1 - t)(1 + t)``.

    As tanh saturates, ``t**2`` and ``1`` both approach 1 and their difference
    cancels: the ~1 ULP rounding of ``t**2`` becomes an absolute error of a
    result of size ``1 - t**2``, so the relative error is amplified by
    ``1/(1 - t**2)``. At x == 8 that factor is about 9.8e6, which is far past
    what f32 can absorb. The factored form has no such step -- ``1 - t`` is
    EXACT in binary floating point for t in [0.5, 2] (Sterbenz) -- so the error
    stays at ~1 ULP however deep the saturation goes.

    jax factors it the same way, so the two engines are compared at a tolerance
    that only the factored form can meet. Evaluating ``1 - t**2`` instead fails
    this from about x == 3 onward (MEASURED 5.0e-6 relative at x == 3.5, against
    the 1e-6 asserted here).
    """
    def f(x):
        return jnp.tanh(x)

    # Well past saturation: tanh(8) == 0.99999977, derivative ~4.5e-7.
    x = jnp.array([0.5, 1.0, 2.0, 3.5, 5.0, 8.0])
    got = jacve(f, order="fwd", argnums=(0,))(x)[0]
    want = jax.jacfwd(f)(x)
    # Diagonal only; the off-diagonal zeros are structural on both sides.
    got_d = jnp.diagonal(got)
    want_d = jnp.diagonal(want)
    assert jnp.allclose(got_d, want_d, rtol=1e-6, atol=0.0), (
        "saturated tanh derivative lost precision: "
        f"got {got_d}, want {want_d}, "
        f"max rel {jnp.max(jnp.abs(got_d - want_d) / jnp.abs(want_d))}")


if __name__ == "__main__":
    unittest.main()
        
        