import json
from random import shuffle
import multiprocessing
import signal
import os
from matplotlib import colormaps
from jax import tree_util
from itertools import product
from graphax.sparse.block import BlockSparseTensor, SparseDimension, DenseDimension, new_block_sparse_tensor
import jax.random as jrand
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import time
import threading
from jax import lax

MAX_MEMORY = int(os.environ.get("MAX_MEMORY", 9216000000))

def profile_jax(fn, *args, device=None, warmup=True, poll_ms=0, **kwargs):
    if device is None:
        device = jax.devices()[0]

    if warmup:
        tmp = fn(*args, **kwargs)
        tmp.block_until_ready()
        del tmp

    mem_before = device.memory_stats()

    if mem_before is not None:
        peak = mem_before["bytes_in_use"] or None
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

    start = time.perf_counter()
    
    out = fn(*args, **kwargs)
    out.block_until_ready()

    end = time.perf_counter()
   
    if mem_before is None:
        stats = { "wall_s": end - start }
    else:
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
jit_dot = jit(lax.dot, static_argnames=["dimension_numbers"])


# TODO WIP.
def matshow3d(a, ax = None):
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')
    
    ax.voxels(a, facecolors=colormaps["viridis"](a), shade=False)
    
    ax.set_aspect("equal")
    
    x=jnp.arange(a.shape[0])
    y=jnp.arange(a.shape[1])
    z=jnp.arange(a.shape[2])
    
    ax.set_xticks(x+0.5)
    ax.set_yticks(y+0.5)
    ax.set_zticks(z+0.5)
    
    ax.set_xticklabels(x)
    ax.set_yticklabels(y)
    ax.set_zticklabels(z)

    ax.grid(False)

def matshowXd(t):
    match t.ndim:
        case 1:
            plt.matshow(t[:, None])
        case 2:
            plt.matshow(t)
        case 3:
            matshow3d(t)
        case 4:
            fig, ax = plt.subplots(len(t), subplot_kw=dict(projection="3d"))
            print(type(ax))
            for i in range(len(t)):
                if not isinstance(ax, np.array):
                    ax = np.array([ax])
                matshow3d(t[i], ax)
        case _:
            raise ValueError("`t` must be a tensor of dimension 4 or less")

# TODO rename... these are shit :I
def test(size, k1, k2, stx_dims, sty_dims, res = None):
    stx = new_block_sparse_tensor(*stx_dims, jrand.normal(k1, size))
    sty = new_block_sparse_tensor(*sty_dims, jrand.normal(k2, size))

    # print(
    #     f"({", ".join([str(i) for i in stx.out_shape])} | "
    #     f"{", ".join([str(i) for i in stx.primal_shape])}) @ "
    #     f"({", ".join([str(i) for i in sty.out_shape])} | "
    #     f"{", ".join([str(i) for i in sty.primal_shape])})"
    # )
    
    if res is None:
        res = {}

        res["sparse"] = {}
        res["sparse"]["estimate"] = jit_matmul.lower(stx, sty).cost_analysis()
        
        res["vals_estimate"] = jit(lambda a, b: (a.dense(), b.dense())).lowewr(stx, sty).cost_analysis()

        x = stx.dense()
        y = sty.dense()
        
        dnums = ((tuple(d.id for d in stx.primal_dims), tuple(d.id for d in sty.out_dims)), ((), ()))
        
        res["dense"] = {}
        res["dense"]["estimate"] = jit_dot.lower(x, y, dimension_numbers=dnums).cost_analysis()
    else:
    
        a = None
        if res["sparse"]["estimate"]["bytes accessed"] <= MAX_MEMORY:
            res["sparse"]["measured"] = []
            for _ in range(20):
                a, r = profile_jax(jit_matmul, stx, sty, poll_ms=1)
                res["sparse"]["measured"].append(r)
        else:
            return res
        
        if res["vals_estimate"]["bytes accessed"] <= MAX_MEMORY:
            x = stx.dense()
            y = sty.dense()
        
        dnums = ((tuple(d.id for d in stx.primal_dims), tuple(d.id for d in sty.out_dims)), ((), ()))
    
        b = None
        if res["dense"]["estimate"]["bytes accessed"] <= MAX_MEMORY:
            res["dense"]["measured"] = []
            for _ in range(20):
                b, r = profile_jax(jit_dot, x, y, dimension_numbers=dnums, poll_ms=1)
                res["dense"]["measured"].append(r)
        else: 
            return res
    
        if a is not None and b is not None:
            assert a.shape == a.dense().shape, f"{a.shape=} is not equal to {a.dense().shape=}"
            assert a.shape == b.shape, f"{a.shape=} is not equal to {b.shape}"
            assert jnp.allclose(a.dense(), b, 1e2, 1e3), f"tensor a is not equal to b, with a normed delta of {jnp.linalg.norm(jnp.abs(a-b))}"

    return res

def _calc(x, res=None):

    i, (block_nums, block_size) = x

    bn = str(block_nums)
    bs = str(block_size)

    if res is None:
        res = {}
        res.setdefault(bn, {})
        res[bn].setdefault(bs, {})
    
    k1, k2 = jrand.split(jrand.PRNGKey(i), 2)

    # 2D
    #print("2d, 1c, 1s")
    res[bn][bs]["2d, 1c, 1s"] = test(
        (block_nums, block_size, block_size), 
        k1, k2, 
        (
            [SparseDimension(0, block_nums, 0, 1, block_size)], 
            [SparseDimension(1, block_nums, 1, 0, block_size)] 
        ), (
            [SparseDimension(0, block_nums, 0, 1, block_size)], 
            [SparseDimension(1, block_nums, 1, 0, block_size)] 
        )
    , res[bn][bs].get("2d, 1c, 1s", None))

    # 3D - 1
    #print("3d, 1c, 1s")
    res[bn][bs]["3d, 1c, 1s"] = test(
        (block_nums, block_size, block_size, block_size),
        k1, k2,
        (
            [
                SparseDimension(0, block_nums, 0, 2, block_size),
                DenseDimension(1, block_size, 1)
            ], 
            [SparseDimension(2, block_nums, 2, 0, block_size)]
        ),(
            [SparseDimension(0, block_nums, 0, 1, block_size)], 
            [
                SparseDimension(1, block_nums, 1, 0, block_size),
                DenseDimension(2, block_size, 2)
            ] 
        )
    , res[bn][bs].get("3d, 1c, 1s", None))
    
    # 3D - 2
    #print("3d, 2c, 1s")
    res[bn][bs]["3d, 2c, 1s"] = test(
        (block_nums, block_size, block_size, block_size),
        k1, k2,
        (
            [SparseDimension(0, block_nums, 0, 1, block_size)], 
            [
                SparseDimension(1, block_nums, 1, 0, block_size),
                DenseDimension(2, block_size, 2)
            ] 
        ),(
            [
                SparseDimension(0, block_nums, 0, 2, block_size),
                DenseDimension(1, block_size, 1)
            ], 
            [SparseDimension(2, block_nums, 2, 0, block_size)]
        )
    , res[bn][bs].get("3d, 2c, 1s", None))
    
    # 4D - 1
    #print("4d, 1c, 1s")
    res[bn][bs]["4d, 1c, 1s"] = test(
        (block_nums, block_size, block_size, block_size, block_size),
        k1, k2,
        (
            [
                SparseDimension(0, block_nums, 0, 2, block_size),
                DenseDimension(1, block_size, 1)
            ], [
                SparseDimension(2, block_nums, 2, 0, block_size),
                DenseDimension(3, block_size, 3)
            ] 
        ), (
            [
                SparseDimension(0, block_nums, 0, 2, block_size),
                DenseDimension(1, block_size, 1)
            ], [
                SparseDimension(2, block_nums, 2, 0, block_size),
                DenseDimension(3, block_size, 3)
            ] 
        )
    , res[bn][bs].get("4d, 1c, 1s", None))

    # 4D - 2
    #print("4d, 1c, 2s")
    res[bn][bs]["4d, 1c, 2s"] = test(
        (block_nums, block_nums, block_size, block_size, block_size, block_size),
        k1, k2,
        (
            [
                SparseDimension(0, block_nums, 0, 2, block_size),
                SparseDimension(1, block_nums, 1, 3, block_size)
            ], [
                SparseDimension(2, block_nums, 2, 0, block_size),
                SparseDimension(3, block_nums, 3, 1, block_size)
            ], 
        ), (
            [
                SparseDimension(0, block_nums, 0, 2, block_size),
                SparseDimension(1, block_nums, 1, 3, block_size)
            ], [
                SparseDimension(2, block_nums, 2, 0, block_size),
                SparseDimension(3, block_nums, 3, 1, block_size)
            ], 
        )
    , res[bn][bs].get("4d, 1c, 2s", None))

    return res

small = False
if small:
    n = m = k = 4
else:
    n = 19
    m = 8
    k = 23

range_ = set([(i%9+1)*10**(i//9) for i in range(n)] + [2**i for i in range(m)])
d = list(product(range_, range_))
shuffle(d)

if not os.path.isfile("res.json"):
    print("running estimates")

    res = {}

    pool = multiprocessing.Pool(k)

    for re in tqdm(pool.imap_unordered(_calc, enumerate(d)), total=len(d)):
        for bn in re.keys():
            if bn not in res:
                res.update(re)
            else:
                res[bn].update(re[bn])

    pool.close()
    pool.join()
else:
    print("running measurements")
    
    with open("res.json", "r") as f:
        res = json.load(f)

    for x in enumerate(t:=tqdm(d)):
        res = _calc(x, res)

with open("res.json", "w") as f:
    json.dump(res, f)

