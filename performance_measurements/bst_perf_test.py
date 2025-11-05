from jax import tree_util
from graphax.sparse.block import BlockSparseTensor, SparseDimension, DenseDimension, new_block_sparse_tensor
import jax.random as jrand
import jax.numpy as jnp
from jax import jit, lax, profiler, clear_caches
import argparse

def flatten_block_sparse_tensor(tensor):
    children = (tensor.blocks,)
    aux_data = (tensor.out_dims, tensor.primal_dims, tensor.out_shape, tensor.primal_shape, tensor.sparse_dims, tensor.pre_transforms, tensor.post_transforms)
    return children, aux_data

def unflatten_block_sparse_tensor(aux_data, children):
    blocks, = children
    out_dims, primal_dims, out_shape, primal_shape, sparse_dims, pre_transforms, post_transforms = aux_data
    return BlockSparseTensor(out_dims, primal_dims, out_shape, primal_shape, blocks, sparse_dims, pre_transforms, post_transforms)

tree_util.register_pytree_node(BlockSparseTensor, flatten_block_sparse_tensor, unflatten_block_sparse_tensor)

jit_matmul = jit(lambda a, b: a @ b)
jit_dot = jit(lax.dot, static_argnames=["dimension_numbers"])

matmul_args = {
    "2d-1c-1s": {
        "SparseTensor": lambda block_nums, block_size: (
            (block_nums, block_size, block_size),
            ([SparseDimension(0, block_nums, 0, 1, block_size)],
             [SparseDimension(1, block_nums, 1, 0, block_size)]),
            ([SparseDimension(0, block_nums, 0, 1, block_size)],
             [SparseDimension(1, block_nums, 1, 0, block_size)])
        ),
        "Array": lambda block_nums, block_size: (
            (block_nums, block_size, block_size),
            ([SparseDimension(0, block_nums, 0, 1, block_size)],
             [SparseDimension(1, block_nums, 1, 0, block_size)]),
            (block_size*block_nums,) * 2
        )
    },
    "3d-1c-1s": {
        "SparseTensor": lambda block_nums, block_size: (
            (block_nums, block_size, block_size, block_size),
            ([SparseDimension(0, block_nums, 0, 2, block_size), DenseDimension(1, block_size, 1)],
             [SparseDimension(2, block_nums, 2, 0, block_size)]),
            ([SparseDimension(0, block_nums, 0, 1, block_size)],
             [SparseDimension(1, block_nums, 1, 0, block_size), DenseDimension(2, block_size, 2)])
        ),
        "Array": lambda block_nums, block_size: (
            (block_nums, block_size, block_size, block_size),
            ([SparseDimension(0, block_nums, 0, 2, block_size), DenseDimension(1, block_size, 1)],
             [SparseDimension(2, block_nums, 2, 0, block_size)]),
            (block_size*block_nums,) * 2
        )
    },
    "3d-2c-1s": {
        "SparseTensor": lambda block_nums, block_size: (
            (block_nums, block_size, block_size, block_size),
            ([SparseDimension(0, block_nums, 0, 1, block_size)],
             [SparseDimension(1, block_nums, 1, 0, block_size), DenseDimension(2, block_size, 2)]),
            ([SparseDimension(0, block_nums, 0, 2, block_size), DenseDimension(1, block_size, 1)],
             [SparseDimension(2, block_nums, 2, 0, block_size)])
        ),
        "Array": lambda block_nums, block_size: (
            (block_nums, block_size, block_size, block_size),
            ([SparseDimension(0, block_nums, 0, 1, block_size)],
             [SparseDimension(1, block_nums, 1, 0, block_size), DenseDimension(2, block_size, 2)]),
            (block_size*block_nums, block_size, block_nums*block_size)
        )
    },
    "4d-1c-1s": {
        "SparseTensor": lambda block_nums, block_size: (
            (block_nums, block_size, block_size, block_size, block_size),
            ([SparseDimension(0, block_nums, 0, 2, block_size), DenseDimension(1, block_size, 1)],
             [SparseDimension(2, block_nums, 2, 0, block_size), DenseDimension(3, block_size, 3)]),
            ([SparseDimension(0, block_nums, 0, 2, block_size), DenseDimension(1, block_size, 1)],
             [SparseDimension(2, block_nums, 2, 0, block_size), DenseDimension(3, block_size, 3)])
        ),
        "Array": lambda block_nums, block_size: (
            (block_nums, block_size, block_size, block_size, block_size),
            ([SparseDimension(0, block_nums, 0, 2, block_size), DenseDimension(1, block_size, 1)],
             [SparseDimension(2, block_nums, 2, 0, block_size), DenseDimension(3, block_size, 3)]),
            (block_size*block_nums, block_size, block_size*block_nums)
        )
    },
    "4d-1c-2s": {
        "SparseTensor": lambda block_nums, block_size: (
            (block_nums, block_nums, block_size, block_size, block_size, block_size),
            ([SparseDimension(0, block_nums, 0, 2, block_size), SparseDimension(1, block_nums, 1, 3, block_size)],
             [SparseDimension(2, block_nums, 2, 0, block_size), SparseDimension(3, block_nums, 3, 1, block_size)]),
            ([SparseDimension(0, block_nums, 0, 2, block_size), SparseDimension(1, block_nums, 1, 3, block_size)],
             [SparseDimension(2, block_nums, 2, 0, block_size), SparseDimension(3, block_nums, 3, 1, block_size)])
        ),
        "Array": lambda block_nums, block_size: (
            (block_nums, block_nums, block_size, block_size, block_size, block_size),
            ([SparseDimension(0, block_nums, 0, 2, block_size), SparseDimension(1, block_nums, 1, 3, block_size)],
             [SparseDimension(2, block_nums, 2, 0, block_size), SparseDimension(3, block_nums, 3, 1, block_size)]),
            (block_size*block_nums,) * 3
        )
    }
}

def calc(size, lhs_params, rhs_params, k1, k2, rhs_is_sparse, matmul_is_sparse, skip_matmul, is_test=False):
    print("generating lhs sparse tensor")
    lhs = new_block_sparse_tensor(*lhs_params, jrand.normal(k1, size))
    d1 = tuple(d.id for d in lhs.primal_dims)

    if rhs_is_sparse:
        print("generating rhs sparse tensor")
        rhs = new_block_sparse_tensor(*rhs_params, jrand.normal(k2, size))
        d2 = tuple(d.id for d in rhs.out_dims)
    else:
        print("generating rhs dense tensor")
        rhs = jrand.normal(k2, rhs_params)
        d2 = tuple(range(len(d1)))
    
    dnums = ((d1, d2), ((), ()))

    if skip_matmul:
        return
    
    if matmul_is_sparse or is_test:
        print("compile sparse matmul")
        res_sparse = jit_matmul(lhs, rhs).block_until_ready()
        print("running sparse matmul")
        if not is_test:
            for _ in range(20):
                _ = jit_matmul(lhs, rhs).block_until_ready()

    if not matmul_is_sparse or is_test:
        print("generating dense tensor(s)")
        lhs = lhs.dense()
        rhs = rhs.dense() if rhs_is_sparse else rhs
    
        print("compile dense matmul")
        res_dense = jit_dot(lhs, rhs, dimension_numbers=dnums).block_until_ready()
        print("running dense matmul")
        if not is_test:
            for _ in range(20):
                _ = jit_dot(lhs, rhs, dimension_numbers=dnums).block_until_ready()

    if is_test and res_sparse is not None and res_dense is not None:
        print("checking correctness")
        if rhs_is_sparse:
            assert res_sparse.shape == res_sparse.dense().shape, f"{res_sparse.shape=} is not equal to {res_sparse.dense().shape=}"
        
        norm = ...
        assert res_sparse.shape == res_dense.shape, f"{res_sparse.shape=} is not equal to {res_dense.shape}"
        assert jnp.allclose(res_sparse, res_dense, 1e2, 1e3), f"tensor a is not equal to b, with a normed delta of {norm}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bn", "--block_numbers", type=int, required=True)
    parser.add_argument("-bs", "--block_sizes", type=int, required=True)
    parser.add_argument("-t", "--matmul_type", choices=["2d-1c-1s", "3d-1c-1s", "3d-2c-1s", "4d-1c-1s", "4d-1c-2s"], required=True)
    parser.add_argument("--sparse-rhs", action="store_true")
    parser.add_argument("--sparse-matmul", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-p", "--prof", action="store_true")
    parser.add_argument("-nop", "--skip-matmul", action="store_true")

    args = parser.parse_args()

    if args.prof:
        clear_caches()

    k1, k2 = jrand.split(jrand.PRNGKey(args.seed))
    object_name = "SparseTensor" if args.sparse_rhs else "Array"

    calc(
        *matmul_args[args.matmul_type][object_name](args.block_numbers, args.block_sizes),
        k1, k2, args.sparse_rhs, args.sparse_matmul, args.skip_matmul
    )

    if args.prof:
        profiler.save_device_memory_profile(f"memory_bn{args.block_numbers}_bs{args.block_sizes}_t{args.matmul_type.replace('-','')}{'_sparse_rhs' if args.sparse_rhs else ''}{'_sparse_matmul' if args.sparse_matmul else ''}{'_baseline' if args.skip_matmul else ''}_s{args.seed}.prof")

