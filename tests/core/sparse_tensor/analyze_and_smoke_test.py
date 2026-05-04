import math
import builtins
import unittest
import jax
import jax.numpy as jnp
import jax.random as jr
from graphax.sparse.tensor import SparseTensor, DenseIndex, SparseIndex, _arr2st
from graphax.sparse.ops.elementwise import elementwise

from jax_memory_monitor.jax_peak_memory_monitor import PeakMemoryMonitor
import timeit
import time

import re
from typing import NamedTuple
import os

ANALYZE = os.getenv("ANALYZE", "0") == "1"
SCALE = int(os.getenv("SCALE", "1"))


def core_plus(a, b):
    return a + b


def core_minus(a, b):
    return a - b


def core_lor(a, b):
    return a | b


def core_max(a, b):
    return elementwise(a, b, jnp.maximum)


def core_mul(a, b):
    return a * b


def core_power(a, b):
    return a**b


def core_matmul(a, b):
    return a @ b


def core_matmul_plus(a, b, c):
    return a @ b + c


# --- Dummy Object for Dense Assertions ---
class DummyDense(NamedTuple):
    arr: jax.Array

    def dense(self):
        return self.arr

    @property
    def shape(self):
        return self.arr.shape


# --- Fixed Dense Implementations ---
def densify(*args):
    return tuple(a.dense() for a in args)


def plus(a, b):
    return DummyDense(a + b)


def minus(a, b):
    return DummyDense(a - b)


def lor(a, b):
    return DummyDense(a.astype(jnp.bool_) | b.astype(jnp.bool_))


def max(a, b):
    return DummyDense(jnp.maximum(a, b))


def mul(a, b):
    return DummyDense(a * b)


def power(a, b):
    return DummyDense(a**b)


def matmul(
    a,
    b,
    lhs_contract_axes=None,
    rhs_contract_axes=None,
    lhs_batch=None,
    rhs_batch=None,
    perm=None,
):
    if lhs_contract_axes is None:
        # Fallback to simple matmul behavior if no metadata provided
        return DummyDense(a @ b)

    res_arr = jax.lax.dot_general(
        a, b, ((lhs_contract_axes, rhs_contract_axes), (lhs_batch, rhs_batch))
    )

    if perm is not None:
        res_arr = jnp.transpose(res_arr, axes=perm)

    return DummyDense(res_arr)


def matmul_plus(a, b, c, **kwargs):
    return DummyDense(matmul(a, b, **kwargs).arr + c)


def _get_matmul_kwargs(a, b):
    num_contract = min(len(a.primal_dims), len(b.out_dims))

    lhs_contract_axes = list(range(a.ndim - num_contract, a.ndim))
    rhs_contract_axes = list(range(len(b.out_dims) - num_contract, len(b.out_dims)))

    a_contract_dims = a.primal_dims[-num_contract:] if num_contract > 0 else []
    b_contract_dims = b.out_dims[-num_contract:] if num_contract > 0 else []

    a_contract_ids = {d.id for d in a_contract_dims}
    b_contract_ids = {d.id for d in b_contract_dims}
    a_sibling_ids = {
        d.other_id for d in a_contract_dims if getattr(d, "other_id", None) is not None
    }
    b_sibling_ids = {
        d.other_id for d in b_contract_dims if getattr(d, "other_id", None) is not None
    }

    lhs_batch = []
    rhs_batch = []

    for i, da in enumerate(a.dims):
        if da.id in a_contract_ids:
            continue
        for j, db in enumerate(b.dims):
            if db.id in b_contract_ids:
                continue
            if da.id == db.id and da.logical_size == db.logical_size:
                if da.id in a_sibling_ids or db.id in b_sibling_ids:
                    continue
                if i not in lhs_batch and j not in rhs_batch:
                    lhs_batch.append(i)
                    rhs_batch.append(j)
                    break

    rhs_id_offset = builtins.max([d.id for d in a.dims] + [-1]) + 1

    out_axes_ids = []
    for ax_idx in lhs_batch:
        out_axes_ids.append(a.dims[ax_idx].id)

    for ax_idx, da in enumerate(a.dims):
        if ax_idx not in lhs_batch and ax_idx not in lhs_contract_axes:
            out_axes_ids.append(da.id)

    for ax_idx, db in enumerate(b.dims):
        if ax_idx not in rhs_batch and ax_idx not in rhs_contract_axes:
            out_axes_ids.append(db.id + rhs_id_offset)

    out_ids_pool = []
    primal_ids_pool = []

    for da in a.out_dims:
        if da.id not in a_contract_ids:
            out_ids_pool.append(da.id)
    for db in b.out_dims:
        if db.id not in b_contract_ids and b.dims.index(db) not in rhs_batch:
            out_ids_pool.append(db.id + rhs_id_offset)

    for da in a.primal_dims:
        if da.id not in a_contract_ids:
            primal_ids_pool.append(da.id)
    for db in b.primal_dims:
        if db.id not in b_contract_ids and b.dims.index(db) not in rhs_batch:
            primal_ids_pool.append(db.id + rhs_id_offset)
    target_ids = sorted(out_ids_pool) + sorted(primal_ids_pool)
    perm = tuple(out_axes_ids.index(i) for i in target_ids)

    return {
        "lhs_contract_axes": tuple(lhs_contract_axes),
        "rhs_contract_axes": tuple(rhs_contract_axes),
        "lhs_batch": tuple(lhs_batch),
        "rhs_batch": tuple(rhs_batch),
        "perm": tuple(perm),
    }


def wrap_dense(fn):
    def wrapped(*args):
        if fn.__name__ in ("matmul", "matmul_plus"):
            kwargs = _get_matmul_kwargs(*args[:2])
            return fn(*densify(*args), **kwargs)
        return fn(*densify(*args))

    return wrapped


# --- Manual Implementations ---


def manual_01(a, b):
    return SparseTensor(a.out_dims, a.primal_dims, a.val + b.val, sort_val=False)


def manual_02(a, b):
    return SparseTensor(a.out_dims, a.primal_dims, a.val - b.val, sort_val=False)


def manual_03(a, b):
    s1, s2, s3 = a.val.shape
    s4, s5, s6 = b.val.shape
    a_blocks = jnp.zeros((s4, s5, s6), dtype=jnp.bool_)
    a_val = a.val.reshape(s4, s1 // s4, s2, s3)
    a_blocks = a_blocks.at[:, 0:s2, 0:s3].set(a_val[:, 0])
    a_blocks = a_blocks.at[:, s2 : s2 * 2, s3 : s3 * 2].set(a_val[:, 1])
    return SparseTensor(
        b.out_dims, b.primal_dims, a_blocks | b.val, dtype=jnp.bool_, sort_val=False
    )


def manual_04(a, b):
    return SparseTensor(
        a.out_dims, a.primal_dims, jnp.maximum(a.val, b.val), sort_val=False
    )


def manual_05(a, b):
    return SparseTensor(a.out_dims, a.primal_dims, a.val * b.val, sort_val=False)


def manual_06(a, b):
    s1 = a.dims[0].size  # number of a-blocks on diagonal
    s2 = a.dims[0].block_size  # a block size
    s3 = b.dims[0].size  # number of b-blocks on diagonal
    s4 = b.dims[0].block_size  # b block size

    lcm_block = math.lcm(s2, s4)
    n_a_sub = lcm_block // s2  # a sub-blocks per unified block
    n_b_sub = lcm_block // s4  # b sub-blocks per unified block
    n_unified = s1 // n_a_sub  # number of unified blocks

    # Expand a micro-blocks into unified block-diagonal
    a_micro = a.val.reshape(n_unified, n_a_sub, s2, s2)
    idx_a = jnp.arange(n_a_sub)
    a_macro = (
        jnp.zeros((n_unified, n_a_sub, n_a_sub, s2, s2))
        .at[:, idx_a, idx_a]
        .set(a_micro)
    )
    a_macro = a_macro.transpose(0, 1, 3, 2, 4).reshape(n_unified, lcm_block, lcm_block)

    # Expand b micro-blocks and extract matching positions
    b_micro = b.val.reshape(n_unified, n_b_sub, s4, s4)
    idx_b = jnp.arange(n_b_sub)

    a_view = a_macro.reshape(n_unified, n_b_sub, s4, n_b_sub, s4)
    a_at_b = a_view[:, idx_b, :, idx_b, :]

    b_micro_t = b_micro.transpose(1, 0, 2, 3)
    res_micro = a_at_b**b_micro_t

    out_macro = jnp.ones((n_unified, lcm_block, lcm_block))
    out_view = out_macro.reshape(n_unified, n_b_sub, s4, n_b_sub, s4)
    out_view = out_view.at[:, idx_b, :, idx_b, :].set(res_micro)
    out_macro = out_view.reshape(n_unified, lcm_block, lcm_block)

    return SparseTensor(
        (
            SparseIndex(
                0,
                n_unified,
                axis=0,
                other_id=1,
                block_size=lcm_block,
                block_axis=1,
            ),
        ),
        (
            SparseIndex(
                1,
                n_unified,
                axis=0,
                other_id=0,
                block_size=lcm_block,
                block_axis=2,
            ),
        ),
        out_macro,
        fill_value=1,
        sort_val=False,
    )


def manual_07(a, b):
    return SparseTensor(a.out_dims, b.primal_dims, a.val * b.val, sort_val=False)


def manual_08(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[1].size
    s3 = b.primal_dims[1].size
    R_val = jnp.expand_dims(a.val, 2) * jnp.expand_dims(b.val, 1)  # (s1, s2, s3)
    return SparseTensor(
        (DenseIndex(0, s1, 0), SparseIndex(1, s2, axis=1, other_id=2)),
        (SparseIndex(2, s2, axis=1, other_id=1), DenseIndex(3, s3, 2)),
        R_val,
        sort_val=False,
    )


def manual_09(a, b):
    s1 = a.dims[0].size
    s2 = a.primal_dims[1].size
    s3 = b.primal_dims[1].size
    R_val = jnp.expand_dims(a.val, 2) * jnp.expand_dims(b.val, 0)
    return SparseTensor(
        (SparseIndex(0, s1, axis=0, other_id=1),),
        (
            SparseIndex(1, s1, axis=0, other_id=0),
            DenseIndex(2, s2, 1),
            DenseIndex(3, s3, 2),
        ),
        R_val,
        sort_val=False,
    )


def manual_10(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[1].size
    s3 = b.primal_dims[1].size
    R_val = jnp.expand_dims(a.val, 2) * jnp.expand_dims(b.val, 1)  # (s1, s2, s3)
    return SparseTensor(
        (SparseIndex(0, s1, axis=0, other_id=2), DenseIndex(1, s2, 1)),
        (SparseIndex(2, s1, axis=0, other_id=0), DenseIndex(3, s3, 2)),
        R_val,
        sort_val=False,
    )


def manual_11(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[0].block_size
    R_val = a.val @ b.val
    return SparseTensor(
        (
            SparseIndex(
                0, s1, axis=0, other_id=1, block_size=s2, block_axis=1
            ),
        ),
        (
            SparseIndex(
                1, s1, axis=0, other_id=0, block_size=s2, block_axis=2
            ),
        ),
        R_val,
        sort_val=False,
    )


def manual_12(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[1].size
    s3 = b.dims[0].size
    R_val = jnp.expand_dims(a.val, 2) * jnp.expand_dims(b.val, 1)  # (s1, s2, s3)
    return SparseTensor(
        (
            SparseIndex(0, s1, axis=0, other_id=3),
            DenseIndex(1, s2, 1),
            DenseIndex(2, s3, 2),
        ),
        (SparseIndex(3, s1, axis=0, other_id=0),),
        R_val,
        sort_val=False,
    )


def manual_13(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[1].size
    s3 = a.dims[2].size
    R_val = jax.lax.dot_general(a.val, b.val, (((3,), (3,)), ((0, 1, 2), (0, 1, 2))))
    return SparseTensor(
        (
            DenseIndex(0, s1, 0),
            DenseIndex(1, s2, 1),
            SparseIndex(2, s3, axis=2, other_id=3),
        ),
        (SparseIndex(3, s3, axis=2, other_id=2),),
        R_val,
        sort_val=False,
    )


def manual_14(a, b):
    s1 = b.dims[0].size
    s2 = a.dims[1].size
    s3 = b.primal_dims[1].size
    return SparseTensor(
        (SparseIndex(0, s1, axis=0, other_id=2), DenseIndex(1, s2, None)),
        (SparseIndex(2, s1, axis=0, other_id=0), DenseIndex(3, s3, 1)),
        b.val,
        sort_val=False,
    )


def manual_15(a, b):
    s1 = a.dims[0].size
    R_val = jnp.sum(a.val * b.val, axis=1)
    return SparseTensor(
        (SparseIndex(0, s1, axis=0, other_id=1),),
        (SparseIndex(1, s1, axis=0, other_id=0),),
        R_val,
        sort_val=False,
    )


def manual_16(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[1].size
    s3 = a.dims[2].size
    R_val = jax.lax.dot_general(a.val, b.val, (((3,), (3,)), ((0, 1, 2), (0, 1, 2))))
    return SparseTensor(
        (
            DenseIndex(0, s1, 0),
            DenseIndex(1, s2, 1),
            SparseIndex(2, s3, axis=2, other_id=3),
        ),
        (SparseIndex(3, s3, axis=2, other_id=2),),
        R_val,
        sort_val=False,
    )


def manual_17(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[0].block_size
    s3 = a.primal_dims[0].block_size
    s4 = b.primal_dims[0].block_size
    R_val = a.val @ b.val
    return SparseTensor(
        (
            SparseIndex(
                0, s1, axis=0, other_id=1, block_size=s2, block_axis=1
            ),
        ),
        (
            SparseIndex(
                1, s1, axis=0, other_id=0, block_size=s4, block_axis=2
            ),
        ),
        R_val,
        sort_val=False,
    )


def manual_18(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[1].size
    s3 = b.primal_dims[1].size
    R_val = jnp.expand_dims(a.val, 2) * jnp.expand_dims(b.val, 1)  # (s1, s2, s3)
    return SparseTensor(
        (DenseIndex(0, s1, 0), SparseIndex(1, s2, axis=1, other_id=2)),
        (SparseIndex(2, s2, axis=1, other_id=1), DenseIndex(3, s3, 2)),
        R_val,
        sort_val=False,
    )


def manual_19(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[1].size
    s3 = a.dims[2].size
    s4 = a.dims[3].size
    s6 = b.primal_dims[1].size
    R_val = jax.lax.dot_general(a.val, b.val, (((4,), (3,)), ((0, 1, 2), (0, 1, 2))))
    return SparseTensor(
        (
            DenseIndex(0, s1, 0),
            DenseIndex(1, s2, 1),
            SparseIndex(2, s3, axis=2, other_id=4),
            DenseIndex(3, s4, 3),
        ),
        (SparseIndex(4, s3, axis=2, other_id=2), DenseIndex(5, s6, 4)),
        R_val,
        sort_val=False,
    )


def manual_matmul_aligned(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[1].size
    s4 = b.primal_dims[1].size
    res_val = jax.lax.dot_general(a.val, b.val, (((2,), (1,)), ((0,), (0,))))
    return SparseTensor(
        (SparseIndex(0, s1, axis=0, other_id=2), DenseIndex(1, s2, 1)),
        (SparseIndex(2, s1, axis=0, other_id=0), DenseIndex(3, s4, 2)),
        res_val,
        sort_val=False,
    )


def manual_plus_aligned(a, b):
    return SparseTensor(a.out_dims, a.primal_dims, a.val + b.val, sort_val=False)


def manual_matmul_plus_aligned(a, b, c):
    res = manual_matmul_aligned(a, b)
    return manual_plus_aligned(res, c)


def manual_matmul_unaligned(a, b):
    # a.val: (s3, s2, s1), b.val: (s1, s3, s4)
    # R[i, j, m, p] = a.val[i, j, m] * b.val[m, i, p]
    b_val_t = jnp.transpose(b.val, (1, 0, 2))  # (s3, s1, s4) -> (i, m, p)
    res_val = a.val[:, :, :, None] * b_val_t[:, None, :, :]  # (s3, s2, s1, s4)

    s3, s2, s1 = a.val.shape
    s4 = b.val.shape[2]

    # Unaligned matmul results in a dense tensor in this case
    return SparseTensor(
        (DenseIndex(0, s3, 0), DenseIndex(1, s2, 1)),
        (DenseIndex(2, s1, 2), DenseIndex(3, s4, 3)),
        res_val,
        sort_val=False,
    )


def manual_plus_unaligned(a, b):
    # a: (s3, s2, s1, s4), b: (s5, s2, s7, s4)
    s5, s2, s7, s4 = b.val.shape
    s6 = a.val.shape[0] // s5

    val_reshaped = a.val.reshape(s5, s6, s2, s5, s7, s4)
    idx = jnp.arange(s5)
    res_val = val_reshaped.at[idx, :, :, idx, :, :].add(b.val[:, None, :, :, :])
    return a.copy(val=res_val.reshape(a.val.shape))


def manual_matmul_plus_unaligned(a, b, c):
    res = manual_matmul_unaligned(a, b)
    return manual_plus_unaligned(res, c)


class TestSmokeScreen(unittest.TestCase):
    def setUp(self):
        self.key = jr.PRNGKey(42)

    def _n(self, shape, key_idx=0, dtype=jnp.float32):
        key = jr.PRNGKey(key_idx)
        if dtype == jnp.bool_:
            return jr.normal(key, shape) > 0.0
        return jr.normal(key, shape).astype(dtype)

    def _assert_equivalence(self, res, res_dense, res_manual):
        tol = {"atol": 1e-3} if ANALYZE else {}
        if res_dense is not None:
            self.assertTrue(jnp.allclose(res.dense(), res_dense.dense(), **tol))
        self.assertEqual(res.shape, res_manual.shape)
        if res.val is not None and res_manual.val is not None:
            self.assertEqual(res.val.shape, res_manual.val.shape)
            self.assertTrue(jnp.allclose(res.val, res_manual.val, **tol))

        # JAX equality across structural SparseTensors
        if res.val is not None and res_manual.val is not None:
            self.assertTrue(res.val.shape == res_manual.val.shape)
        else:
            self.assertTrue(res.val is None and res_manual.val is None)

    def _analyze(self, func, *args, **kwargs):
        name = func.__name__
        static_argnames = tuple(kwargs.keys())
        jitted_func = jax.jit(func, static_argnames=static_argnames)

        lowered = jitted_func.lower(*args, **kwargs)
        compiled = lowered.compile()
        cost = compiled.cost_analysis()

        flops = cost.get("flops", 0.0)
        mbytes_acc = cost.get("bytes accessed", 0.0) / 1024**2
        # bytes_acc = cost.get('bytes accessed', 0.0)

        hlo = compiled.as_text()
        entry = re.search(r"ENTRY\s+.*?\{(.*?)\n\}", hlo, re.DOTALL)
        ops = (
            sum(
                1
                for line in entry.group(1).split("\n")
                if "=" in line
                and "parameter(" not in line
                and "constant(" not in line
                and "tuple(" not in line
                and "copy(%constant" not in line
            )
            if entry
            else 0
        )

        # 2. Compilation / First-Call Latency & Memory
        peak_mem_str = "N/A"
        compile_time = 0

        with PeakMemoryMonitor() as monitor:
            t0 = time.perf_counter()
            _ = jax.block_until_ready(jitted_func(*args, **kwargs))
            compile_time = time.perf_counter() - t0
        peak_mem_str = f"{monitor.peak / 1024**2:.2f} MB"
        # peak_mem_str = monitor.peak

        # 3. Execution Latency (Autorange)
        timer = timeit.Timer(
            lambda: jax.block_until_ready(jitted_func(*args, **kwargs))
        )
        number, time_taken = timer.autorange()
        exec_time_ms = (time_taken / number) * 1000  # Convert to milliseconds

        print(
            f"[STATS] {name.split('_')[0]:<6} | "
            f"FLOPs: {flops:.0e} | "
            f"Mem: {mbytes_acc:.2f} MB | "
            f"Ops: {ops:^3} | "
            f"Compile: {compile_time:<6.3f}s | "
            f"Exec: {exec_time_ms:<6.3f}ms | "
            f"Peak Mem: {peak_mem_str} | "
            f"HLO len: {len(hlo)}"
        )

    def _deep_analysis(self, fn, core_fn, manual_fn, *args):
        if not ANALYZE:
            return

        print()

        self._analyze(densify, *args)
        dargs = densify(*args)
        kwargs = (
            _get_matmul_kwargs(*args[:2])
            if fn.__name__ in ("matmul", "matmul_plus")
            else {}
        )
        self._analyze(fn, *dargs, **kwargs)
        self._analyze(wrap_dense(fn), *args)
        self._analyze(core_fn, *args)
        self._analyze(manual_fn, *args)

        print("core jaxpr =", jax.make_jaxpr(core_fn)(*args))
        print("manual jaxpr =", jax.make_jaxpr(manual_fn)(*args))

        print("core hlo =", jax.jit(core_fn).lower(*args).compile().as_text())
        print("manual hlo =", jax.jit(manual_fn).lower(*args).compile().as_text())

    # --- Elementwise Tests (6) ---

    def test_01_add_sparse_sparse_union(self):
        s1 = 4
        s2 = 3
        s3 = 5
        if ANALYZE:
            s1 *= 26 * SCALE
            s2 *= 26 * SCALE
            s3 *= 26 * SCALE

        a = SparseTensor(
            (
                SparseIndex(0, s1, axis=0, other_id=3),
                DenseIndex(1, s2, 1),
                DenseIndex(2, s3, 2),
            ),
            (SparseIndex(3, s1, axis=0, other_id=0),),
            self._n((s1, s2, s3), 1),
        )
        b = SparseTensor(
            (
                SparseIndex(0, s1, axis=0, other_id=3),
                DenseIndex(1, s2, 1),
                DenseIndex(2, s3, 2),
            ),
            (SparseIndex(3, s1, axis=0, other_id=0),),
            self._n((s1, s2, s3), 2),
        )

        self._deep_analysis(plus, core_plus, manual_01, a, b)
        res = core_plus(a, b)
        self.assertEqual(res.shape, (s1, s2, s3, s1))
        self._assert_equivalence(res, wrap_dense(plus)(a, b), manual_01(a, b))

    def test_02_sub_sparse_sparse_union(self):
        s1 = 4
        s2 = 2
        s3 = 2
        s4 = 3
        s5 = 5
        if ANALYZE:
            s1 *= 7 * SCALE
            s2 *= 7 * SCALE
            s3 *= 7 * SCALE
            s4 *= 7 * SCALE
            s5 *= 7 * SCALE

        a = SparseTensor(
            (
                SparseIndex(
                    0, s1, axis=0, other_id=1, block_size=s2, block_axis=1
                ),
                DenseIndex(2, s3, 2),
            ),
            (
                SparseIndex(
                    1, s1, axis=0, other_id=0, block_size=s4, block_axis=3
                ),
                DenseIndex(3, s5, 4),
            ),
            self._n((s1, s2, s3, s4, s5), 1),
        )
        b = SparseTensor(
            (
                SparseIndex(
                    0, s1, axis=0, other_id=1, block_size=s2, block_axis=1
                ),
                DenseIndex(2, s3, 2),
            ),
            (
                SparseIndex(
                    1, s1, axis=0, other_id=0, block_size=s4, block_axis=3
                ),
                DenseIndex(3, s5, 4),
            ),
            self._n((s1, s2, s3, s4, s5), 2),
        )

        self._deep_analysis(minus, core_minus, manual_02, a, b)
        res = core_minus(a, b)
        self.assertEqual(res.shape, (s1 * s2, s3, s1 * s4, s5))
        self._assert_equivalence(res, wrap_dense(minus)(a, b), manual_02(a, b))

    def test_03_logical_or_unaligned_union(self):
        s1 = 4
        s2 = 2
        s3 = 3
        s4 = 2
        s5 = 4
        s6 = 6
        if ANALYZE:
            s1 *= 45 * SCALE
            s2 *= 45 * SCALE
            s3 *= 45 * SCALE
            s4 *= 45 * SCALE
            s5 *= 45 * SCALE
            s6 *= 45 * SCALE

        a = SparseTensor(
            (
                SparseIndex(
                    0, s1, axis=0, other_id=1, block_size=s2, block_axis=1
                ),
            ),
            (
                SparseIndex(
                    1, s1, axis=0, other_id=0, block_size=s3, block_axis=2
                ),
            ),
            self._n((s1, s2, s3), 1, dtype=jnp.bool_),
            dtype=jnp.bool_,
        )
        b = SparseTensor(
            (
                SparseIndex(
                    0, s4, axis=0, other_id=1, block_size=s5, block_axis=1
                ),
            ),
            (
                SparseIndex(
                    1, s4, axis=0, other_id=0, block_size=s6, block_axis=2
                ),
            ),
            self._n((s4, s5, s6), 2, dtype=jnp.bool_),
            dtype=jnp.bool_,
        )

        self._deep_analysis(lor, core_lor, manual_03, a, b)
        res = core_lor(a, b)
        self.assertEqual(res.shape, (s1 * s2, s1 * s3))
        self._assert_equivalence(res, wrap_dense(lor)(a, b), manual_03(a, b))

    def test_04_max_sparse_sparse_union(self):
        s1 = 2
        s2 = 5
        if ANALYZE:
            s1 *= 125 * SCALE
            s2 *= 125 * SCALE

        a = SparseTensor(
            (DenseIndex(0, s1, 0), SparseIndex(1, s2, axis=1, other_id=2)),
            (SparseIndex(2, s2, axis=1, other_id=1),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (DenseIndex(0, s1, 0), SparseIndex(1, s2, axis=1, other_id=2)),
            (SparseIndex(2, s2, axis=1, other_id=1),),
            self._n((s1, s2), 2),
        )

        self._deep_analysis(max, core_max, manual_04, a, b)
        res = core_max(a, b)
        self.assertEqual(res.shape, (s1, s2, s2))
        self._assert_equivalence(res, wrap_dense(max)(a, b), manual_04(a, b))

    def test_05_mul_sparse_sparse_intersection(self):
        s1 = 6
        s2 = 3
        if ANALYZE:
            s1 *= 100 * SCALE
            s2 *= 100 * SCALE

        a = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=2), DenseIndex(1, s2, 1)),
            (SparseIndex(2, s1, axis=0, other_id=0),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=2), DenseIndex(1, s2, 1)),
            (SparseIndex(2, s1, axis=0, other_id=0),),
            self._n((s1, s2), 2),
        )

        self._deep_analysis(mul, core_mul, manual_05, a, b)
        res = core_mul(a, b)
        self.assertEqual(res.shape, (s1, s2, s1))
        self._assert_equivalence(res, wrap_dense(mul)(a, b), manual_05(a, b))

    def test_06_power_mismatch_blocks_intersection(self):
        s1 = 4
        s2 = 3
        s3 = 6
        s4 = 2
        if ANALYZE:
            s1 *= 30 * SCALE
            s2 *= 30 * SCALE
            s3 *= 30 * SCALE
            s4 *= 30 * SCALE

        a = SparseTensor(
            (
                SparseIndex(
                    0, s1, axis=0, other_id=1, block_size=s2, block_axis=1
                ),
            ),
            (
                SparseIndex(
                    1, s1, axis=0, other_id=0, block_size=s2, block_axis=2
                ),
            ),
            jnp.abs(self._n((s1, s2, s2), 1)),
        )
        b = SparseTensor(
            (
                SparseIndex(
                    0, s3, axis=0, other_id=1, block_size=s4, block_axis=1
                ),
            ),
            (
                SparseIndex(
                    1, s3, axis=0, other_id=0, block_size=s4, block_axis=2
                ),
            ),
            jnp.abs(self._n((s3, s4, s4), 2)),
        )

        self._deep_analysis(power, core_power, manual_06, a, b)

        res = core_power(a, b)
        self.assertEqual(res.shape, (s1 * s2, s3 * s4))

        tol = {"atol": 1e-3} if ANALYZE else {}
        self.assertTrue(
            jnp.allclose(manual_06(a, b).dense(), a.dense() ** b.dense(), **tol)
        )
        self.assertTrue(jnp.allclose(res.dense(), a.dense() ** b.dense(), **tol))
        self._assert_equivalence(res, wrap_dense(power)(a, b), manual_06(a, b))

    # --- Matmul Tests (14) ---

    def test_07_matmul_sparse_sparse_batched(self):
        s1 = 2
        s2 = 6
        if ANALYZE:
            s1 *= 110 * SCALE
            s2 *= 110 * SCALE

        a = SparseTensor(
            (DenseIndex(0, s1, 0), SparseIndex(1, s2, axis=1, other_id=2)),
            (SparseIndex(2, s2, axis=1, other_id=1),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (DenseIndex(0, s1, 0), SparseIndex(1, s2, axis=1, other_id=2)),
            (SparseIndex(2, s2, axis=1, other_id=1),),
            self._n((s1, s2), 2),
        )

        self._deep_analysis(matmul, core_matmul, manual_07, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s2))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_07(a, b))

    def test_08_matmul_sparse_sparse_3d(self):
        s1 = 2
        s2 = 4
        s3 = 5
        if ANALYZE:
            s1 *= 28 * SCALE
            s2 *= 28 * SCALE
            s3 *= 28 * SCALE

        a = SparseTensor(
            (DenseIndex(0, s1, 0), SparseIndex(1, s2, axis=1, other_id=2)),
            (SparseIndex(2, s2, axis=1, other_id=1),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, s2, axis=0, other_id=1),),
            (SparseIndex(1, s2, axis=0, other_id=0), DenseIndex(2, s3, 1)),
            self._n((s2, s3), 2),
        )

        self._deep_analysis(matmul, core_matmul, manual_08, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s2, s3))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_08(a, b))

    def test_09_matmul_sparse_sparse_coverage(self):
        s1 = 4
        s2 = 6
        s3 = 4
        s4 = 6
        s5 = 3
        if ANALYZE:
            s1 *= 25 * SCALE
            s2 *= 25 * SCALE
            s3 *= 25 * SCALE
            s4 *= 25 * SCALE
            s5 *= 25 * SCALE

        a = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=1),),
            (SparseIndex(1, s1, axis=0, other_id=0), DenseIndex(2, s2, 1)),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, s4, axis=0, other_id=1),),
            (SparseIndex(1, s4, axis=0, other_id=0), DenseIndex(2, s5, 1)),
            self._n((s4, s5), 2),
        )

        self._deep_analysis(matmul, core_matmul, manual_09, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s1, s4, s5))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_09(a, b))

    def test_10_matmul_sparse_sparse_aligned(self):
        s1 = 6
        s2 = 2
        s3 = 6
        s4 = 3
        if ANALYZE:
            s1 *= 26 * SCALE
            s2 *= 26 * SCALE
            s3 *= 26 * SCALE
            s4 *= 26 * SCALE

        a = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=2), DenseIndex(1, s2, 1)),
            (SparseIndex(2, s1, axis=0, other_id=0),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, s3, axis=0, other_id=1),),
            (SparseIndex(1, s3, axis=0, other_id=0), DenseIndex(2, s4, 1)),
            self._n((s3, s4), 2),
        )

        self._deep_analysis(matmul, core_matmul, manual_10, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s3, s4))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_10(a, b))

    def test_11_matmul_misaligned_blocks(self):
        s1 = 4
        s2 = 2
        s3 = 4
        if ANALYZE:
            s1 *= 10 * SCALE
            s2 *= 10 * SCALE
            s3 *= 10 * SCALE

        a = SparseTensor(
            (
                SparseIndex(
                    0, s1, axis=0, other_id=1, block_size=s2, block_axis=1
                ),
            ),
            (
                SparseIndex(
                    1, s1, axis=0, other_id=0, block_size=s3, block_axis=2
                ),
            ),
            self._n((s1, s2, s3), 1),
        )
        b = SparseTensor(
            (
                SparseIndex(
                    0, s1, axis=0, other_id=1, block_size=s3, block_axis=1
                ),
            ),
            (
                SparseIndex(
                    1, s1, axis=0, other_id=0, block_size=s2, block_axis=2
                ),
            ),
            self._n((s1, s3, s2), 2),
        )

        self._deep_analysis(matmul, core_matmul, manual_11, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1 * s2, s1 * s2))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_11(a, b))

    def test_12_matmul_batch_simple(self):
        s1 = 6
        s2 = 4
        s3 = 5
        if ANALYZE:
            s1 *= 19 * SCALE
            s2 *= 19 * SCALE
            s3 *= 19 * SCALE

        a = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=2), DenseIndex(1, s2, 1)),
            (SparseIndex(2, s1, axis=0, other_id=0),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (DenseIndex(2, s3, 1), SparseIndex(0, s1, axis=0, other_id=1)),
            (SparseIndex(1, s1, axis=0, other_id=0),),
            self._n((s1, s3), 2),
        )

        self._deep_analysis(matmul, core_matmul, manual_12, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s3, s1))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_12(a, b))

    def test_13_matmul_batch_complex(self):
        s1 = 2
        s2 = 3
        s3 = 5
        s4 = 4
        if ANALYZE:
            s1 *= 8 * SCALE
            s2 *= 8 * SCALE
            s3 *= 8 * SCALE
            s4 *= 8 * SCALE

        a = SparseTensor(
            (
                DenseIndex(0, s1, 0),
                DenseIndex(1, s2, 1),
                SparseIndex(2, s3, axis=2, other_id=3),
            ),
            (SparseIndex(3, s3, axis=2, other_id=2), DenseIndex(4, s4, 3)),
            self._n((s1, s2, s3, s4), 1),
        )
        b = SparseTensor(
            (
                DenseIndex(0, s1, 0),
                DenseIndex(1, s2, 1),
                SparseIndex(2, s3, axis=2, other_id=4),
                DenseIndex(3, s4, 3),
            ),
            (SparseIndex(4, s3, axis=2, other_id=2),),
            self._n((s1, s2, s3, s4), 2),
        )

        self._deep_analysis(matmul, core_matmul, manual_13, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s3, s3))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_13(a, b))

    def test_14_matmul_unmaterialized(self):
        s1 = 5
        s2 = 4
        s3 = 2
        if ANALYZE:
            s1 *= 27 * SCALE
            s2 *= 27 * SCALE
            s3 *= 27 * SCALE

        a = SparseTensor(
            (
                SparseIndex(0, s1, axis=None, other_id=2),
                DenseIndex(1, s2, None),
            ),
            (SparseIndex(2, s1, axis=None, other_id=0),),
            val=None,
        )
        b = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=1),),
            (SparseIndex(1, s1, axis=0, other_id=0), DenseIndex(2, s3, 1)),
            self._n((s1, s3), 2),
        )

        self._deep_analysis(matmul, core_matmul, manual_14, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s1, s3))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_14(a, b))

    def test_15_matmul(self):
        s1 = 6
        s2 = 3
        if ANALYZE:
            s1 *= 60 * SCALE
            s2 *= 60 * SCALE

        a = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=2),),
            (DenseIndex(1, s2, 1), SparseIndex(2, s1, axis=0, other_id=0)),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (DenseIndex(0, s2, 1), SparseIndex(1, s1, axis=0, other_id=2)),
            (SparseIndex(2, s1, axis=0, other_id=1),),
            self._n((s1, s2), 2),
        )

        self._deep_analysis(matmul, core_matmul, manual_15, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s1))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_15(a, b))

    def test_16_matmul_high_rank(self):
        s1 = 2
        s2 = 2
        s3 = 4
        s4 = 2
        if ANALYZE:
            s1 *= 15 * SCALE
            s2 *= 15 * SCALE
            s3 *= 15 * SCALE
            s4 *= 15 * SCALE

        a = SparseTensor(
            (
                DenseIndex(0, s1, 0),
                DenseIndex(1, s2, 1),
                SparseIndex(2, s3, axis=2, other_id=3),
            ),
            (SparseIndex(3, s3, axis=2, other_id=2), DenseIndex(4, s4, 3)),
            self._n((s1, s2, s3, s4), 1),
        )
        b = SparseTensor(
            (
                DenseIndex(0, s1, 0),
                DenseIndex(1, s2, 1),
                SparseIndex(2, s3, axis=2, other_id=4),
                DenseIndex(3, s4, 3),
            ),
            (SparseIndex(4, s3, axis=2, other_id=2),),
            self._n((s1, s2, s3, s4), 2),
        )

        self._deep_analysis(matmul, core_matmul, manual_16, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s3, s3))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_16(a, b))

    def test_17_matmul_aligned_blocks(self):
        s1 = 6
        s2 = 2
        s3 = 3
        s4 = 4
        if ANALYZE:
            s1 *= 15 * SCALE
            s2 *= 15 * SCALE
            s3 *= 15 * SCALE
            s4 *= 15 * SCALE

        a = SparseTensor(
            (
                SparseIndex(
                    0, s1, axis=0, other_id=1, block_size=s2, block_axis=1
                ),
            ),
            (
                SparseIndex(
                    1, s1, axis=0, other_id=0, block_size=s3, block_axis=2
                ),
            ),
            self._n((s1, s2, s3), 1),
        )
        b = SparseTensor(
            (
                SparseIndex(
                    0, s1, axis=0, other_id=1, block_size=s3, block_axis=1
                ),
            ),
            (
                SparseIndex(
                    1, s1, axis=0, other_id=0, block_size=s4, block_axis=2
                ),
            ),
            self._n((s1, s3, s4), 2),
        )

        self._deep_analysis(matmul, core_matmul, manual_17, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1 * s2, s1 * s4))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_17(a, b))

    def test_18_matmul_sparse_primal_pairs(self):
        s1 = 4
        s2 = 8
        s3 = 5
        if ANALYZE:
            s1 *= 10 * SCALE
            s2 *= 10 * SCALE
            s3 *= 10 * SCALE

        a = SparseTensor(
            (DenseIndex(0, s1, 0), SparseIndex(1, s2, axis=1, other_id=2)),
            (SparseIndex(2, s2, axis=1, other_id=1),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, s2, axis=0, other_id=1),),
            (SparseIndex(1, s2, axis=0, other_id=0), DenseIndex(2, s3, 1)),
            self._n((s2, s3), 2),
        )

        self._deep_analysis(matmul, core_matmul, manual_18, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s2, s3))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_18(a, b))

    def test_19_matmul_multi_batch(self):
        s1 = 2
        s2 = 3
        s3 = 4
        s4 = 5
        s5 = 6
        s6 = 7
        if ANALYZE:
            s1 *= 4 * SCALE
            s2 *= 4 * SCALE
            s3 *= 4 * SCALE
            s4 *= 4 * SCALE
            s5 *= 4 * SCALE
            s6 *= 4 * SCALE

        a = SparseTensor(
            (
                DenseIndex(0, s1, 0),
                DenseIndex(1, s2, 1),
                SparseIndex(2, s3, axis=2, other_id=4),
                DenseIndex(3, s4, 3),
            ),
            (SparseIndex(4, s3, axis=2, other_id=2), DenseIndex(5, s5, 4)),
            self._n((s1, s2, s3, s4, s5), 1),
        )
        b = SparseTensor(
            (
                DenseIndex(0, s1, 0),
                DenseIndex(1, s2, 1),
                SparseIndex(2, s3, axis=2, other_id=4),
                DenseIndex(3, s5, 3),
            ),
            (SparseIndex(4, s3, axis=2, other_id=2), DenseIndex(5, s6, 4)),
            self._n((s1, s2, s3, s5, s6), 2),
        )

        self._deep_analysis(matmul, core_matmul, manual_19, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s3, s4, s3, s6))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_19(a, b))

    def _get_chained_dims(self):
        s1 = 4
        s2 = 5
        s3 = 6
        s4 = 7
        s5 = 2
        s6 = 3
        s7 = 2
        if ANALYZE:
            s5 *= 4 * SCALE
            s6 *= 4 * SCALE
            s7 *= 4 * SCALE
            s2 *= 4 * SCALE
            s4 *= 4 * SCALE
            s3 = s5 * s6
            s1 = s5 * s7
        return s1, s2, s3, s4, s5, s6, s7

    def _get_chained_tensors(self, s1, s2, s3, s4, s5, s6, s7):
        a1 = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=2), DenseIndex(1, s2, 1)),
            (SparseIndex(2, s1, axis=0, other_id=0), DenseIndex(3, s3, 2)),
            self._n((s1, s2, s3), 1),
        )
        a2 = SparseTensor(
            (SparseIndex(0, s3, axis=0, other_id=3), DenseIndex(1, s2, 1)),
            (DenseIndex(2, s1, 2), SparseIndex(3, s3, axis=0, other_id=0)),
            self._n((s3, s2, s1), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=2), DenseIndex(1, s3, 1)),
            (SparseIndex(2, s1, axis=0, other_id=0), DenseIndex(3, s4, 2)),
            self._n((s1, s3, s4), 2),
        )
        c1 = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=2), DenseIndex(1, s2, 1)),
            (SparseIndex(2, s1, axis=0, other_id=0), DenseIndex(3, s4, 2)),
            self._n((s1, s2, s4), 3),
        )
        c2 = SparseTensor(
            (
                SparseIndex(0, s5, axis=0, other_id=2, block_size=s6),
                DenseIndex(1, s2, 1),
            ),
            (
                SparseIndex(
                    2, s5, axis=0, other_id=0, block_size=s7, block_axis=2
                ),
                DenseIndex(3, s4, 3),
            ),
            self._n((s5, s2, s7, s4), 3),
        )
        return a1, a2, b, c1, c2

    def test_20_1_matmul(self):
        s1, s2, s3, s4, s5, s6, s7 = self._get_chained_dims()
        a1, _, b, _, _ = self._get_chained_tensors(s1, s2, s3, s4, s5, s6, s7)

        self._deep_analysis(matmul, core_matmul, manual_matmul_aligned, a1, b)
        res = core_matmul(a1, b)
        self.assertEqual(res.shape, (s1, s2, s1, s4))
        self._assert_equivalence(
            res, wrap_dense(matmul)(a1, b), manual_matmul_aligned(a1, b)
        )

    def test_20_1_plus(self):
        s1, s2, s3, s4, s5, s6, s7 = self._get_chained_dims()
        a1, _, b, c1, _ = self._get_chained_tensors(s1, s2, s3, s4, s5, s6, s7)
        matmul_res = core_matmul(a1, b)
        self._deep_analysis(plus, core_plus, manual_plus_aligned, matmul_res, c1)
        res = core_plus(matmul_res, c1)
        self.assertEqual(res.shape, (s1, s2, s1, s4))
        self._assert_equivalence(
            res, wrap_dense(plus)(matmul_res, c1), manual_plus_aligned(matmul_res, c1)
        )

    def test_20_1_matmul_plus(self):
        s1, s2, s3, s4, s5, s6, s7 = self._get_chained_dims()
        a1, _, b, c1, _ = self._get_chained_tensors(s1, s2, s3, s4, s5, s6, s7)
        self._deep_analysis(
            matmul_plus, core_matmul_plus, manual_matmul_plus_aligned, a1, b, c1
        )
        res = core_matmul_plus(a1, b, c1)
        self.assertEqual(res.shape, (s1, s2, s1, s4))
        self._assert_equivalence(
            res,
            wrap_dense(matmul_plus)(a1, b, c1),
            manual_matmul_plus_aligned(a1, b, c1),
        )

    def test_20_2_matmul(self):
        s1, s2, s3, s4, s5, s6, s7 = self._get_chained_dims()
        _, a2, b, _, _ = self._get_chained_tensors(s1, s2, s3, s4, s5, s6, s7)

        self._deep_analysis(matmul, core_matmul, manual_matmul_unaligned, a2, b)
        res = core_matmul(a2, b)
        self.assertEqual(res.shape, (s3, s2, s1, s4))
        self._assert_equivalence(
            res, wrap_dense(matmul)(a2, b), manual_matmul_unaligned(a2, b)
        )

    def test_20_2_plus(self):
        s1, s2, s3, s4, s5, s6, s7 = self._get_chained_dims()
        _, a2, b, _, c2 = self._get_chained_tensors(s1, s2, s3, s4, s5, s6, s7)
        matmul_res = core_matmul(a2, b)
        self._deep_analysis(plus, core_plus, manual_plus_unaligned, matmul_res, c2)
        res = core_plus(matmul_res, c2)
        self.assertEqual(res.shape, (s3, s2, s1, s4))
        self._assert_equivalence(
            res, wrap_dense(plus)(matmul_res, c2), manual_plus_unaligned(matmul_res, c2)
        )

    def test_20_2_matmul_plus(self):
        s1, s2, s3, s4, s5, s6, s7 = self._get_chained_dims()
        _, a2, b, _, c2 = self._get_chained_tensors(s1, s2, s3, s4, s5, s6, s7)
        self._deep_analysis(
            matmul_plus, core_matmul_plus, manual_matmul_plus_unaligned, a2, b, c2
        )
        res = core_matmul_plus(a2, b, c2)
        self.assertEqual(res.shape, (s3, s2, s1, s4))
        self._assert_equivalence(
            res,
            wrap_dense(matmul_plus)(a2, b, c2),
            manual_matmul_plus_unaligned(a2, b, c2),
        )


if __name__ == "__main__":
    unittest.main()
