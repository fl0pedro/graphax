import os
import unittest
import jax.numpy as jnp
import jax.random as jrand
from utils import (
    assert_matmul_result,
    generate_tensors,
    matmul_reference,
    run_matmul_blocks_test,
    idfn,
)

from graphax.sparse.tensor import SparseTensor

EXHAUSTIVE = os.getenv("EXHAUSTIVE", "0") == "1"


def gen_blocks(key, n_blocks, block_shape):
    return jrand.normal(key, (n_blocks,) + block_shape)


def get_valid_pairs(blocks_a, blocks_b, ndim_a, ndim_b):
    tensors_a = list(generate_tensors(blocks_a, ndim_a))
    tensors_b = list(generate_tensors(blocks_b, ndim_b))

    pairs = []
    for tensor_a in tensors_a:
        for tensor_b in tensors_b:
            if len(tensor_a.primal_dims) != len(tensor_b.out_dims):
                continue

            if any(
                d_a.logical_size != d_b.logical_size
                for d_a, d_b in zip(tensor_a.primal_dims, tensor_b.out_dims)
            ):
                continue

            pairs.append((tensor_a, tensor_b))
    return pairs


def generate_cases(seed, n_blocks_a, base_shape_a, n_blocks_b, base_shape_b):
    cases = []
    for ndim in [2, 3]:
        key = jrand.PRNGKey(seed)
        k1, k2 = jrand.split(key)

        shape_a = (base_shape_a,) * ndim
        shape_b = (base_shape_b,) * ndim

        blocks_a = gen_blocks(k1, n_blocks_a, shape_a)
        blocks_b = gen_blocks(k2, n_blocks_b, shape_b)

        for ta, tb in get_valid_pairs(blocks_a, blocks_b, ndim, ndim):
            cases.append((ndim, "A@B", ta, tb))
        for tb, ta in get_valid_pairs(blocks_b, blocks_a, ndim, ndim):
            cases.append((ndim, "B@A", tb, ta))
    return cases


class TestMatmulDiffBlocks(unittest.TestCase):
    def _run_matmul_test(self, ndim, direction, tensor_a, tensor_b):
        with self.subTest(
            ndim=ndim,
            direction=direction,
            tensor_a=idfn(tensor_a),
            tensor_b=idfn(tensor_b),
        ):
            run_matmul_blocks_test(self, tensor_a, tensor_b)

    @unittest.skipUnless(EXHAUSTIVE, "set EXHAUSTIVE=1 to run exhaustive sweeps")
    def test_exhaustive_divisible_blocks(self):
        cases = generate_cases(
            seed=0, n_blocks_a=4, base_shape_a=2, n_blocks_b=2, base_shape_b=4
        )
        for ndim, direction, ta, tb in cases:
            self._run_matmul_test(ndim, direction, ta, tb)

    @unittest.skipUnless(EXHAUSTIVE, "set EXHAUSTIVE=1 to run exhaustive sweeps")
    def test_exhaustive_gcd_common_factor(self):
        cases = generate_cases(
            seed=1, n_blocks_a=3, base_shape_a=4, n_blocks_b=2, base_shape_b=6
        )
        for ndim, direction, ta, tb in cases:
            self._run_matmul_test(ndim, direction, ta, tb)

    @unittest.skipUnless(EXHAUSTIVE, "set EXHAUSTIVE=1 to run exhaustive sweeps")
    def test_exhaustive_coprime_blocks(self):
        cases = generate_cases(
            seed=2, n_blocks_a=4, base_shape_a=3, n_blocks_b=3, base_shape_b=4
        )
        for ndim, direction, ta, tb in cases:
            self._run_matmul_test(ndim, direction, ta, tb)


if __name__ == "__main__":
    unittest.main()
