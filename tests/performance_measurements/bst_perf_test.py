from jax import tree_util
from itertools import product
from jax import jit, make_jaxpr
from graphax.sparse.block import BlockSparseTensor, SparseDimension, DenseDimension, new_block_sparse_tensor
import jax.random as jrand
import timeit
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from functools import reduce
import operator
import jax
import time
import threading
from jax import lax

def profile_jax(fn, *args, device=None, warmup=True, poll_ms=0, **kwargs):
    if device is None:
        device = jax.devices()[0]

    if warmup:
        tmp = fn(*args, **kwargs)
        tmp.block_until_ready()
        del tmp

    mem_before = device.memory_stats()
    start = time.perf_counter()
    peak = mem_before["bytes_in_use"]
    stop_flag = False

    def poll():
        nonlocal peak
        while not stop_flag:
            used = device.memory_stats()["bytes_in_use"]
            peak = max(peak, used)
            time.sleep(poll_ms / 1000.0)

    thread = None
    if poll_ms > 0:
        thread = threading.Thread(target=poll, daemon=True)
        thread.start()

    out = fn(*args, **kwargs)
    out.block_until_ready()

    end = time.perf_counter()
    stop_flag = True
    if thread:
        thread.join()

    mem_after = device.memory_stats()

    stats = {
        "wall_s": end - start,
        "bytes_in_use_before": mem_before["bytes_in_use"],
        "bytes_in_use_after": mem_after["bytes_in_use"],
        "net_bytes_in_use": mem_after["bytes_in_use"] - mem_before["bytes_in_use"],
        "process_peak_bytes": mem_after["peak_bytes_in_use"],
        "peak_bytes_during": (
            max(0, peak - mem_before["bytes_in_use"])
            if poll_ms > 0
            else max(
                0, mem_after["peak_bytes_in_use"] - mem_before["peak_bytes_in_use"]
            )
        ),
    }

    return out, stats

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
range_ = [(i%9+1)*10**(i//9) for i in range(100)]
res = {}

for i, (block_nums, block_size) in enumerate(product(range_, range_)):
    res.setdefault(block_nums, {})
    res[block_nums].setdefault(block_size, {
        "2d, 1c, 1s": [],
        "3d, 1c, 1s": [],
        "3d, 2c, 1s": [],
        "4d, 1c, 1s": [],
        "4d, 1c, 2s": []
    })

    k1, k2 = jrand.split(jrand.PRNGKey(i), 2)

    # 2D
    x = jrand.normal(k1, (block_nums, block_size, block_size))
    y = jrand.normal(k2, (block_nums, block_size, block_size))

    stx = new_block_sparse_tensor(
        [SparseDimension(0, block_nums, 0, 1, block_size)], 
        [SparseDimension(1, block_nums, 1, 0, block_size)], 
        x)
    sty = new_block_sparse_tensor(
        [SparseDimension(0, block_nums, 0, 1, block_size)], 
        [SparseDimension(1, block_nums, 1, 0, block_size)], 
        y)
    
    for _ in range(20):
        _, r = profile_jax(jit_matmul, stx, sty, warmup=True, poll_ms=1)
        res[block_nums][block_size]["2d, 1c, 1s"].append(r)
        

    # 3D
    x = jrand.normal(k1, (block_nums, block_size, block_size, block_size))
    y = jrand.normal(k2, (block_nums, block_size, block_size, block_size))

    stx = new_block_sparse_tensor(
        [SparseDimension(0, block_nums, 0, 1, block_size)], 
        [
            SparseDimension(1, block_nums, 1, 0, block_size),
            DenseDimension(2, block_size, 2)
        ], 
        x)
    sty = new_block_sparse_tensor(
        [
            SparseDimension(0, block_nums, 0, 2, block_size),
            DenseDimension(1, block_size, 1)
        ], 
        [SparseDimension(2, block_nums, 2, 0, block_size)], 
        y)
    
    # 3D - 1
    for _ in range(20):
        _, r = profile_jax(jit_matmul, stx, sty, warmup=True, poll_ms=1)
        res[block_nums][block_size]["3d, 1c, 1s"].append(r)

    # 3D - 2
    for _ in range(20):
        _, r = profile_jax(jit_matmul, sty, stx, warmup=True, poll_ms=1)
        res[block_nums][block_size]["3d, 2c, 1s"].append(r)

    # 4D - 1
    x = jrand.normal(k1, (block_nums, block_size, block_size, block_size, block_size))
    y = jrand.normal(k2, (block_nums, block_size, block_size, block_size, block_size))

    stx = new_block_sparse_tensor(
        [
            SparseDimension(0, block_nums, 0, 2, block_size),
            DenseDimension(1, block_size, 1)
        ], [
            SparseDimension(2, block_nums, 2, 0, block_size),
            DenseDimension(3, block_size, 3)
        ], 
        x)
    sty = new_block_sparse_tensor(
        [
            SparseDimension(0, block_nums, 0, 2, block_size),
            DenseDimension(1, block_size, 1)
        ], [
            SparseDimension(2, block_nums, 2, 0, block_size),
            DenseDimension(3, block_size, 3)
        ], 
        y)
    
    for _ in range(20):
        _, r = profile_jax(jit_matmul, stx, sty, warmup=True, poll_ms=1)
        res[block_nums][block_size]["4d, 1c, 1s"].append(r)

    # 4D - 2
    x = jrand.normal(k1, (block_nums, block_nums, block_size, block_size, block_size))
    y = jrand.normal(k2, (block_nums, block_nums, block_size, block_size, block_size))

    stx = new_block_sparse_tensor(
        [
            SparseDimension(0, block_nums, 0, 2, block_size),
            SparseDimension(1, block_nums, 1, 3, block_size)
        ], [
            SparseDimension(2, block_nums, 2, 0, block_size),
            SparseDimension(3, block_nums, 3, 1, block_size)
        ], 
        x)
    sty = new_block_sparse_tensor(
        [
            SparseDimension(0, block_nums, 0, 2, block_size),
            SparseDimension(1, block_nums, 1, 3, block_size)
        ], [
            SparseDimension(2, block_nums, 2, 0, block_size),
            SparseDimension(3, block_nums, 3, 1, block_size)
        ], 
        y)
    
    for _ in range(20):
        _, r = profile_jax(jit_matmul, stx, sty, warmup=True, poll_ms=1)
        res[block_nums][block_size]["4d, 1c, 2s"].append(r)
