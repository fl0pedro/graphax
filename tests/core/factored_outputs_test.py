"""#46 GRAPHAX_FACTORED_OUTPUTS / DeferredOutputProduct tests.

The flag defers the FINAL contraction onto a (graph input -> pure output)
edge as a factor pair instead of materializing the (batch-wise rank-1)
product. Contract under test:

* flag OFF: byte-identical to the historic path (dense AND sparse drains);
* flag ON, dense drain: byte-identical (the deferred pair spills through
  the same ``sparse_matmul`` the eager path would have run);
* flag ON, sparse drain: returns a ``DeferredOutputProduct`` whose
  ``.dense()`` matches the dense reference, and whose stored factor sizes
  are much smaller than the dense block;
* NOT-low-rank / below-threshold blocks fall back to the dense path;
* a second contribution merging onto a deferred edge spills correctly;
* toggling the env var mid-process is honored (the VertexEliminator prefix
  cache keys flag-on runs into a disjoint subtree).

Each test uses its OWN hidden sizes so ``pytree_hash_cache`` cannot alias
eliminator state between tests.
"""
import os
import unittest

import numpy as np

import jax
import jax.numpy as jnp

from graphax import jacve
from graphax.core import DEFERRED_OUTPUT_STATS, DeferredOutputProduct


def _mlp(hidden):
    def mlp(x, W1, b1, W2, b2):
        h = jnp.tanh(W1 @ x + b1)
        return W2 @ h + b2
    return mlp


def _mlp_args(seed, n_in, hidden, n_out):
    ks = jax.random.split(jax.random.PRNGKey(seed), 5)
    return (
        jax.random.normal(ks[0], (n_in,)),
        jax.random.normal(ks[1], (hidden, n_in)) * 0.05,
        jax.random.normal(ks[2], (hidden,)),
        jax.random.normal(ks[3], (n_out, hidden)) * 0.05,
        jax.random.normal(ks[4], (n_out,)),
    )


def _bytes(a):
    return np.asarray(a).tobytes()


class FactoredOutputsTest(unittest.TestCase):
    def setUp(self):
        self._prev = os.environ.get("GRAPHAX_FACTORED_OUTPUTS")
        os.environ["GRAPHAX_FACTORED_OUTPUTS"] = "0"
        DEFERRED_OUTPUT_STATS.clear()

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("GRAPHAX_FACTORED_OUTPUTS", None)
        else:
            os.environ["GRAPHAX_FACTORED_OUTPUTS"] = self._prev

    def _flag(self, on):
        os.environ["GRAPHAX_FACTORED_OUTPUTS"] = "1" if on else "0"

    # ---- (a) MNIST-style MLP ------------------------------------------

    def test_mlp_dense_flag_on_byte_identical(self):
        mlp = _mlp(96)
        args = _mlp_args(0, 784, 96, 10)
        f = jacve(mlp, order="rev", argnums=(0, 1, 2, 3, 4))
        ref = f(*args)
        self._flag(True)
        on = f(*args)
        for i, (a, b) in enumerate(zip(ref, on)):
            self.assertEqual(_bytes(a), _bytes(b), f"argnum {i} differs")

    def test_mlp_sparse_returns_deferred_and_densifies(self):
        mlp = _mlp(112)
        args = _mlp_args(1, 784, 112, 10)
        ref = jacve(mlp, order="rev", argnums=(1,))(*args)
        jr = jax.jacrev(mlp, argnums=1)(*args)
        np.testing.assert_allclose(np.asarray(ref), np.asarray(jr),
                                   rtol=1e-5, atol=1e-6)
        self._flag(True)
        sp = jacve(mlp, order="rev", argnums=(1,),
                   sparse_representation=True)(*args)
        self.assertIsInstance(sp, DeferredOutputProduct)
        self.assertEqual(_bytes(sp.dense()), _bytes(ref))
        dense_n = int(np.asarray(ref).size)
        fac_n = int(sp.post.val.size) + int(sp.pre.val.size)
        self.assertLess(fac_n * 8, dense_n,
                        "factored storage not at least 8x smaller")

    # ---- (b) attention / einsum-shaped case ---------------------------

    def test_attention_einsum_case(self):
        d = 16

        def attn(q, k, v):
            a = jax.nn.softmax(q @ k.T / jnp.sqrt(jnp.float32(d)), axis=-1)
            return jnp.einsum("ij,jk->ik", a, v)

        ks = jax.random.split(jax.random.PRNGKey(2), 3)
        q = jax.random.normal(ks[0], (8, d))
        k = jax.random.normal(ks[1], (8, d))
        v = jax.random.normal(ks[2], (8, d))
        f = jacve(attn, order="rev", argnums=(0, 1, 2))
        ref = f(q, k, v)
        jr = jax.jacrev(attn, argnums=(0, 1, 2))(q, k, v)
        for a, b in zip(ref, jr):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                       rtol=1e-4, atol=1e-5)
        self._flag(True)
        on = f(q, k, v)
        for i, (a, b) in enumerate(zip(ref, on)):
            self.assertEqual(_bytes(a), _bytes(b), f"argnum {i} differs")
        sp = jacve(attn, order="rev", argnums=(0, 1, 2),
                   sparse_representation=True)(q, k, v)
        for i, (t, r) in enumerate(zip(sp, ref)):
            self.assertEqual(_bytes(t.dense()), _bytes(r),
                             f"sparse argnum {i} differs")

    # ---- (c) NOT-low-rank block: dense fallback -----------------------

    def test_full_rank_block_falls_back_dense(self):
        # dy/dx of the MLP: the final contraction is (n_out, h) @ (h, n_in)
        # with a DENSE W1 factor -- dense product (n_out * n_in) is far
        # below 8x the factor sizes, so the deferral guard must reject it.
        mlp = _mlp(80)
        args = _mlp_args(3, 784, 80, 10)
        ref = jacve(mlp, order="rev", argnums=(0,))(*args)
        self._flag(True)
        sp = jacve(mlp, order="rev", argnums=(0,),
                   sparse_representation=True)(*args)
        self.assertNotIsInstance(sp, DeferredOutputProduct)
        self.assertEqual(_bytes(sp.dense()), _bytes(ref))
        on = jacve(mlp, order="rev", argnums=(0,))(*args)
        self.assertEqual(_bytes(on), _bytes(ref))

    # ---- merge onto a deferred edge spills ----------------------------

    def test_second_contribution_spills_deferred_edge(self):
        def two_path(x, W):
            return W @ x + W @ (x * x)

        ks = jax.random.split(jax.random.PRNGKey(4), 2)
        x = jax.random.normal(ks[0], (300,))
        W = jax.random.normal(ks[1], (11, 300)) * 0.1
        ref = jacve(two_path, order="rev", argnums=(1,))(x, W)
        jr = jax.jacrev(two_path, argnums=1)(x, W)
        np.testing.assert_allclose(np.asarray(ref), np.asarray(jr),
                                   rtol=1e-5, atol=1e-6)
        self._flag(True)
        on = jacve(two_path, order="rev", argnums=(1,))(x, W)
        self.assertEqual(_bytes(on), _bytes(ref))
        sp = jacve(two_path, order="rev", argnums=(1,),
                   sparse_representation=True)(x, W)
        np.testing.assert_allclose(np.asarray(sp.dense()), np.asarray(ref),
                                   rtol=1e-6, atol=1e-7)

    # ---- flag toggling vs the eliminator prefix cache -----------------

    def test_flag_toggle_mid_process_cache_isolation(self):
        mlp = _mlp(104)
        args = _mlp_args(5, 784, 104, 10)

        def sparse_jac():
            return jacve(mlp, order="rev", argnums=(1,),
                         sparse_representation=True)(*args)

        off1 = sparse_jac()
        self.assertNotIsInstance(off1, DeferredOutputProduct)
        self._flag(True)
        on = sparse_jac()
        self.assertIsInstance(
            on, DeferredOutputProduct,
            "flag-on call reused a flag-off cached prefix (cache-key bug)")
        self._flag(False)
        off2 = sparse_jac()
        self.assertNotIsInstance(
            off2, DeferredOutputProduct,
            "flag-off call reused a flag-on cached prefix (cache-key bug)")
        self.assertEqual(_bytes(off1.dense()), _bytes(off2.dense()))
        self.assertEqual(_bytes(on.dense()), _bytes(off1.dense()))

    # ---- pytree / jit transparency ------------------------------------

    def test_jit_sparse_flag_on(self):
        mlp = _mlp(88)
        args = _mlp_args(6, 784, 88, 10)
        ref = jacve(mlp, order="rev", argnums=(1,))(*args)
        self._flag(True)

        jf = jax.jit(lambda *a: jacve(mlp, order="rev", argnums=(1,),
                                      sparse_representation=True)(*a))
        out = jf(*args)
        self.assertIsInstance(out, DeferredOutputProduct)
        np.testing.assert_allclose(np.asarray(out.dense()), np.asarray(ref),
                                   rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
