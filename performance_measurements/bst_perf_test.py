from jax import tree_util
from graphax.sparse.block import BlockSparseTensor, SparseDimension, DenseDimension, new_block_sparse_tensor
import jax.random as jrand
import jax.numpy as jnp
from jax import jit, lax, clear_caches
import time
from jax_peak_memory_monitor import PeakMemoryMonitor
from itertools import product
from collections import defaultdict
import json
from tqdm import tqdm

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

def bench(
    shape,
    lhs_params,
    rhs_params,
    rhs_is_sparse,
    matmul_is_sparse,
    iters,
    key=None,
    compile_only=False,
    skip_matmul=False,
    is_test=False
):
    assert not (skip_matmul and is_test), "skip_matmul and is_test is mutually exclusive"

    if key is None:
        key = jrand.PRNGKey(0)
    key, key1, key2 = jrand.split(key, 3)

    #print("generating lhs sparse tensor")
    lhs = new_block_sparse_tensor(*lhs_params, jrand.uniform(key1, shape))
    d1 = tuple(d.id for d in lhs.primal_dims)

    if rhs_is_sparse:
        #print("generating rhs sparse tensor")
        rhs = new_block_sparse_tensor(*rhs_params, jrand.uniform(key2, shape))
        d2 = tuple(d.id for d in rhs.out_dims)
    else:
        #print("generating rhs dense tensor")
        rhs = jrand.uniform(key2, rhs_params)
        d2 = tuple(range(len(d1)))
    
    dnums = ((d1, d2), ((), ()))

    if skip_matmul:
        if not matmul_is_sparse:
            #print("generating dense tensor(s)")
            lhs = lhs.dense()
            rhs = rhs.dense() if rhs_is_sparse else rhs
    
    if matmul_is_sparse or is_test:
        #print("compile sparse matmul")
        res_sparse = jit_matmul(lhs, rhs)
        #print("execute sparse matmul")
        if not is_test and not compile_only:
            times = []
            with PeakMemoryMonitor(interval=0) as monitor:
                for i in range(iters):
                    if i > 0:
                        key, key1, key2 = jrand.split(key, 3)
                        lhs.blocks = jrand.uniform(key1, shape)
                        if rhs_is_sparse:
                            rhs.blocks = jrand.uniform(key2, shape)
                        else:
                            rhs = jrand.uniform(key2, shape)
                    # regenerate blocks?
                    t1 = time.perf_counter()
                    jit_matmul(lhs, rhs)
                    t2 = time.perf_counter()
                    times.append(t2-t1)

    if not matmul_is_sparse or is_test:
        #print("generating dense tensor(s)")
        lhs = lhs.dense()
        rhs = rhs.dense() if rhs_is_sparse else rhs
    
        #print("compile dense matmul")
        res_dense = jit_dot(lhs, rhs, dimension_numbers=dnums)
        #print("execute dense matmul")
        if not is_test and not compile_only:
            times = []
            with PeakMemoryMonitor(interval=0) as monitor:
                for _ in range(iters):
                    if i > 0:
                        key, key1, key2 = jrand.split(key, 3)
                        lhs.blocks = jrand.uniform(key1, shape)
                        if rhs_is_sparse:
                            rhs.blocks = jrand.uniform(key2, shape)
                        else:
                            rhs = jrand.uniform(key2, shape)
                    t1 = time.perf_counter()
                    jit_dot(lhs, rhs, dimension_numbers=dnums)
                    t2 = time.perf_counter()
                    times.append(t2-t1)

    if is_test and res_sparse is not None and res_dense is not None:
        #print("checking correctness")
        if rhs_is_sparse:
            assert res_sparse.shape == res_sparse.dense().shape, f"{res_sparse.shape=} is not equal to {res_sparse.dense().shape=}"
        
        norm = ...
        assert res_sparse.shape == res_dense.shape, f"{res_sparse.shape=} is not equal to {res_dense.shape}"
        assert jnp.allclose(res_sparse, res_dense, 1e2, 1e3), f"tensor a is not equal to b, with a normed delta of {norm}"
        return None
    else:
        return times, monitor.peak

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-bn", "--block_numbers", type=int, required=True)
#     parser.add_argument("-bs", "--block_sizes", type=int, required=True)
#     parser.add_argument("-t", "--matmul_type", choices=["2d-1c-1s", "3d-1c-1s", "3d-2c-1s", "4d-1c-1s", "4d-1c-2s"], required=True)
#     parser.add_argument("--sparse-rhs", action="store_true")
#     parser.add_argument("--sparse-matmul", action="store_true")
#     parser.add_argument("-s", "--seed", type=int, default=0)
#     parser.add_argument("-nop", "--skip-matmul", action="store_true")
#     parser.add_argument("--compile-only", action="store_true")
#     parser.add_argument("--iters", type=int, default=20)
# 
#     args = parser.parse_args()
# 
#     clear_caches()
# 
#     k1, k2 = jrand.split(jrand.PRNGKey(args.seed))
#     object_name = "SparseTensor" if args.sparse_rhs else "Array"
# 
#     path_components = [
#         f"bn{args.block_numbers}",
#         f"bs{args.block_sizes}",
#         f"t{args.matmul_type.replace('-','')}",
#         "sparse_rhs" if args.sparse_rhs else "",
#         "sparse_matmul" if args.sparse_matmul else "",
#     ]
#     path = "_".join([x for x in path_components if x])
# 
#     res = bench(
#         *matmul_args[args.matmul_type][object_name](args.block_numbers, args.block_sizes), k1, k2,
#         args.sparse_rhs, args.sparse_matmul, args.compile_only, args.skip_matmul, path, args.iters
#     )
# 
#     if res is None:
#         ...
#     else:
#         print(sum(res[0])/args.iters, res[1])

def nested_dict():
    return defaultdict(nested_dict)

if __name__ == "__main__":
    clear_caches()

    if True:
        L2 = 5
        L10 = 15
        iters = 5
    else:
        L2 = 8
        L10 = 20
        iters = 20

    test_cases = {2 ** i for i in range(L2)} | {(i%9+1)*10**(i//9) for i in range(L10)}
    params = product(
        matmul_args.keys(), test_cases, test_cases, {0, 1}, {0, 1}
    )
    params_len = len(matmul_args)*len(test_cases)**2*4

    res = nested_dict()
    # note block_sizes comes first as that increments slower.
    for matmul_type, block_sizes, block_numbers, sparse_rhs, sparse_matmul in tqdm(params, total=params_len):
        object_name = "SparseTensor" if sparse_rhs else "Array"
        implementation = "informed" if sparse_matmul else "naive",

        times, pmem = bench(
            *matmul_args[matmul_type][object_name](block_numbers, block_sizes),
            sparse_rhs, sparse_matmul, iters
        )
        
        res[matmul_type][block_numbers][block_sizes][matmul_type][object_name][implementation] = {
            "times": times,
            "peak memory": pmem
        }

    with open("r3.json", "w") as f:
        json.dump(res, f)
