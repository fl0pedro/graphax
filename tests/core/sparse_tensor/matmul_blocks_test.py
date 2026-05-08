import os
import unittest
from itertools import product

import jax.numpy as jnp
import jax.random as jrand
from jax import Array
from utils import (
    create_block_test_data,
    generate_tensors,
    get_block_configs,
    matmul_reference,
    validate_sparse_tensor,
    assert_matmul_result,
    get_keys,
    idfn,
    run_matmul_blocks_test,
)

from graphax.sparse.tensor import (
    SparseIndex,
    SparseTensor,
    _arr2st,
    _matmul,
)

from graphax.sparse.ops.matmul import matmul

EXHAUSTIVE = os.getenv("EXHAUSTIVE", "0") == "1"

DATA = create_block_test_data()


def get_test_params():
    params = []
    ids = []

    configs = []
    for ndim in range(1, 4):
        bs = get_block_configs(DATA, ndim)
        for name, blocks in bs:
            if isinstance(blocks, list):
                continue
            configs.append((ndim, name))

    for (na, name_a), (nb, name_b) in product(configs, configs):
        params.append((na, name_a, nb, name_b))
        ids.append(f"{name_a}_x_{name_b}")

    return params, ids


TEST_PARAMS, TEST_IDS = get_test_params()


class TestMatmulBlocks(unittest.TestCase):
    def test_dense_matmul(self):
        dense_params = [
            ((4, 5), (5, 6), 1),  # 2D @ 2D (1 CD)
            ((4, 5), (5, 6, 7), 1),  # 2D @ 3D (1 CD)
            ((4, 5), (5, 6, 7, 8), 1),  # 2D @ 4D (1 CD)
            ((4, 5, 6), (6, 7, 8), 1),  # 3D @ 3D (1 CD)
            ((4, 5, 6), (5, 6, 7), 2),  # 3D @ 3D (2 CD)
            ((4, 5, 6), (6, 7, 8, 9), 1),  # 3D @ 4D (1 CD)
            ((4, 5, 6), (5, 6, 7, 8), 2),  # 3D @ 4D (2 CD)
            ((4, 5, 6, 7), (6, 7, 8, 9), 2),  # 4D @ 4D (2 CD)
            ((4, 5, 6, 7), (5, 6, 7, 8), 3),  # 4D @ 4D (3 CD)
            ((5, 6, 7), (7, 4), 1),  # 3D @ 2D (1 CD)
            ((5, 6, 7, 8), (8, 4), 1),  # 4D @ 2D (1 CD)
            ((5, 6, 7, 8), (8, 9, 4), 1),  # 4D @ 3D (1 CD)
            ((5, 6, 7, 8), (7, 8, 4), 2),  # 4D @ 3D (2 CD)
        ]
        types = ["array", "sparse"]

        for lhs_shape, rhs_shape, n_contract in dense_params:
            for lhs_type in types:
                for rhs_type in types:
                    if lhs_type == "array" and rhs_type == "array":
                        continue

                    with self.subTest(
                        lhs_shape=lhs_shape,
                        rhs_shape=rhs_shape,
                        n_contract=n_contract,
                        lhs_type=lhs_type,
                        rhs_type=rhs_type,
                    ):
                        keys = get_keys()
                        l_key, r_key = keys[0], keys[1]

                        l_arr = jrand.normal(l_key, lhs_shape)
                        r_arr = jrand.normal(r_key, rhs_shape)

                        lhs = (
                            _arr2st(l_arr, out_ndim=len(lhs_shape) - n_contract)
                            if lhs_type == "sparse"
                            else l_arr
                        )
                        rhs = (
                            _arr2st(r_arr, out_ndim=n_contract)
                            if rhs_type == "sparse"
                            else r_arr
                        )

                        res_matmul = _matmul(lhs, rhs)
                        reference = matmul_reference(lhs, rhs)

                        expected_out = lhs_shape[:-n_contract]
                        expected_primal = rhs_shape[n_contract:]

                        assert_matmul_result(
                            res_matmul,
                            reference.dense(),
                            expected_out,
                            expected_primal,
                            res_matmul.val.shape
                            if res_matmul.val is not None
                            else None,
                        )

    def test_scalar_matmul(self):
        key = jrand.PRNGKey(0)
        arr = jrand.normal(key, (3, 3))

        st = _arr2st(arr, out_ndim=1)

        lhs = jnp.array([[2.0, 3.0, 4.0]])
        res_matmul = _matmul(lhs, st)
        reference = matmul_reference(lhs, st)

        assert_matmul_result(
            res_matmul,
            reference.dense(),
            (1,),
            (3,),
            res_matmul.val.shape if res_matmul.val is not None else None,
        )

    @unittest.skipUnless(EXHAUSTIVE, "set EXHAUSTIVE=1 to run exhaustive sweeps")
    def test_exhaustive_matmul_blocks(self):
        def generate_exhaustive_cases():
            cases = []
            configs = []
            for ndim in range(1, 4):
                for name, blocks in get_block_configs(DATA, ndim):
                    if not isinstance(blocks, list):
                        configs.append((ndim, name, blocks))

            for (ndim_a, name_a, blocks_a), (ndim_b, name_b, blocks_b) in product(
                configs, configs
            ):
                for tensor_a in generate_tensors(blocks_a, ndim_a):
                    for tensor_b in generate_tensors(blocks_b, ndim_b):
                        if len(tensor_a.primal_dims) != len(tensor_b.out_dims):
                            continue

                        if any(
                            da.logical_size != db.logical_size
                            for da, db in zip(tensor_a.primal_dims, tensor_b.out_dims)
                        ):
                            continue

                        cases.append((name_a, name_b, tensor_a, tensor_b))
            return cases

        for name_a, name_b, tensor_a, tensor_b in generate_exhaustive_cases():
            with self.subTest(
                name_a=name_a,
                name_b=name_b,
                tensor_a=idfn(tensor_a),
                tensor_b=idfn(tensor_b),
            ):
                run_matmul_blocks_test(self, tensor_a, tensor_b)

    def test_matmul_dense_dense(self):
        a = jnp.arange(6).reshape(2, 3)
        b = jnp.arange(12).reshape(3, 4)
        res = matmul(a, b)
        self.assertTrue(jnp.allclose(res, a @ b))

    def test_matmul_implicit_value(self):
        st1 = _arr2st(jnp.arange(4).reshape(2, 2))
        st1.val = None
        st2 = _arr2st(jnp.arange(4).reshape(2, 2))
        res = matmul(st1, st2)
        self.assertTrue(res.val is not None)


if __name__ == "__main__":
    unittest.main()
