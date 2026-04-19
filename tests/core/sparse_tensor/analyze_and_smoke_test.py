import math
import unittest
import jax
import jax.numpy as jnp
import jax.random as jr
from graphax.sparse.tensor import SparseTensor, DenseDimension, SparseDimension, _arr2st
from graphax.sparse.ops.elementwise import elementwise

from jax_memory_monitor import PeakMemoryMonitor
import timeit
import time

import re
from typing import NamedTuple
import os

ANALYZE = os.getenv("ANALYZE", "0") == "1"


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
def dense_plus(a, b):
    return DummyDense(a.dense() + b.dense())


def dense_minus(a, b):
    return DummyDense(a.dense() - b.dense())


def dense_lor(a, b):
    return DummyDense(a.dense().astype(jnp.bool_) | b.dense().astype(jnp.bool_))


def dense_max(a, b):
    return DummyDense(jnp.maximum(a.dense(), b.dense()))


def dense_mul(a, b):
    return DummyDense(a.dense() * b.dense())


def dense_power(a, b):
    return DummyDense(a.dense() ** b.dense())


def _dense_matmul_impl(a, b):
    A = a.dense()
    B = b.dense()
    num_contract = min(len(a.primal_dims), len(b.out_dims))

    lhs_contract_axes = list(range(A.ndim - num_contract, A.ndim))
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

    res_arr = jax.lax.dot_general(
        A, B, ((lhs_contract_axes, rhs_contract_axes), (lhs_batch, rhs_batch))
    )

    rhs_id_offset = max([d.id for d in a.dims] + [-1]) + 1

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
    perm = [out_axes_ids.index(i) for i in target_ids]

    return DummyDense(jnp.transpose(res_arr, axes=perm))


def dense_matmul(a, b):
    return _dense_matmul_impl(a, b)


def dense_matmul_plus(a, b, c):
    return DummyDense(_dense_matmul_impl(a, b).dense() + c.dense())


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
    a_macro = jnp.zeros((n_unified, n_a_sub, n_a_sub, s2, s2)).at[:, idx_a, idx_a].set(
        a_micro
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
            SparseDimension(
                0, n_unified, val_dim=0, other_id=1, block_size=lcm_block, block_val_dim=1
            ),
        ),
        (
            SparseDimension(
                1, n_unified, val_dim=0, other_id=0, block_size=lcm_block, block_val_dim=2
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
        (DenseDimension(0, s1, 0), SparseDimension(1, s2, val_dim=1, other_id=2)),
        (SparseDimension(2, s2, val_dim=1, other_id=1), DenseDimension(3, s3, 2)),
        R_val,
        sort_val=False,
    )


def manual_09(a, b):
    s1 = a.dims[0].size
    s2 = a.primal_dims[1].size
    s3 = b.primal_dims[1].size
    R_val = jnp.expand_dims(a.val, 2) * jnp.expand_dims(b.val, 0)
    return SparseTensor(
        (SparseDimension(0, s1, val_dim=0, other_id=1),),
        (
            SparseDimension(1, s1, val_dim=0, other_id=0),
            DenseDimension(2, s2, 1),
            DenseDimension(3, s3, 2),
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
        (SparseDimension(0, s1, val_dim=0, other_id=2), DenseDimension(1, s2, 1)),
        (SparseDimension(2, s1, val_dim=0, other_id=0), DenseDimension(3, s3, 2)),
        R_val,
        sort_val=False,
    )


def manual_11(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[0].block_size
    R_val = a.val @ b.val
    return SparseTensor(
        (
            SparseDimension(
                0, s1, val_dim=0, other_id=1, block_size=s2, block_val_dim=1
            ),
        ),
        (
            SparseDimension(
                1, s1, val_dim=0, other_id=0, block_size=s2, block_val_dim=2
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
            SparseDimension(0, s1, val_dim=0, other_id=3),
            DenseDimension(1, s2, 1),
            DenseDimension(2, s3, 2),
        ),
        (SparseDimension(3, s1, val_dim=0, other_id=0),),
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
            DenseDimension(0, s1, 0),
            DenseDimension(1, s2, 1),
            SparseDimension(2, s3, val_dim=2, other_id=3),
        ),
        (SparseDimension(3, s3, val_dim=2, other_id=2),),
        R_val,
        sort_val=False,
    )


def manual_14(a, b):
    s1 = b.dims[0].size
    s2 = a.dims[1].size
    s3 = b.primal_dims[1].size
    return SparseTensor(
        (SparseDimension(0, s1, val_dim=0, other_id=2), DenseDimension(1, s2, None)),
        (SparseDimension(2, s1, val_dim=0, other_id=0), DenseDimension(3, s3, 1)),
        b.val,
        sort_val=False,
    )


def manual_15(a, b):
    s1 = a.dims[0].size
    R_val = jnp.sum(a.val * b.val, axis=1)
    return SparseTensor(
        (SparseDimension(0, s1, val_dim=0, other_id=1),),
        (SparseDimension(1, s1, val_dim=0, other_id=0),),
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
            DenseDimension(0, s1, 0),
            DenseDimension(1, s2, 1),
            SparseDimension(2, s3, val_dim=2, other_id=3),
        ),
        (SparseDimension(3, s3, val_dim=2, other_id=2),),
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
            SparseDimension(
                0, s1, val_dim=0, other_id=1, block_size=s2, block_val_dim=1
            ),
        ),
        (
            SparseDimension(
                1, s1, val_dim=0, other_id=0, block_size=s4, block_val_dim=2
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
        (DenseDimension(0, s1, 0), SparseDimension(1, s2, val_dim=1, other_id=2)),
        (SparseDimension(2, s2, val_dim=1, other_id=1), DenseDimension(3, s3, 2)),
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
            DenseDimension(0, s1, 0),
            DenseDimension(1, s2, 1),
            SparseDimension(2, s3, val_dim=2, other_id=4),
            DenseDimension(3, s4, 3),
        ),
        (SparseDimension(4, s3, val_dim=2, other_id=2), DenseDimension(5, s6, 4)),
        R_val,
        sort_val=False,
    )


def manual_20_1(a, b, c):
    s1 = a.dims[0].size
    s2 = a.dims[1].size
    s4 = b.primal_dims[1].size
    R_val = jax.lax.dot_general(a.val, b.val, (((2,), (1,)), ((0,), (0,)))) + c.val
    return SparseTensor(
        (SparseDimension(0, s1, val_dim=0, other_id=2), DenseDimension(1, s2, 1)),
        (SparseDimension(2, s1, val_dim=0, other_id=0), DenseDimension(3, s4, 2)),
        R_val,
        sort_val=False,
    )

def manual_20_2(a, b, c):
    s5 = c.dims[0].size
    s6 = c.dims[0].block_size
    s7 = c.primal_dims[0].block_size

    b_val_t = jnp.transpose(b.val, (1, 0, 2))
    R_val = a.val[..., None] * jnp.expand_dims(b_val_t, 1)

    for i in range(s5): 
        R_val = R_val.at[i*s6:(i+1)*s6, :, i*s7:(i+1)*s7, :].add(jnp.expand_dims(c.val[i], 0))

    return _arr2st(R_val)


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

    def _analyze(self, func, *args):
        name = func.__name__

        lowered = jax.jit(func).lower(*args)
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
                if "=" in line and "parameter(" not in line
            )
            if entry
            else 0
        )

        # 2. Compilation / First-Call Latency & Memory
        jitted_func = jax.jit(func)
        peak_mem_str = "N/A"
        compile_time = 0

        with PeakMemoryMonitor() as monitor:
            t0 = time.perf_counter()
            _ = jax.block_until_ready(jitted_func(*args))
            compile_time = time.perf_counter() - t0
        peak_mem_str = f"{monitor.peak / 1024**2:.2f} MB"
        # peak_mem_str = monitor.peak

        # 3. Execution Latency (Autorange)
        timer = timeit.Timer(lambda: jax.block_until_ready(jitted_func(*args)))
        number, time_taken = timer.autorange()
        exec_time_ms = (time_taken / number) * 1000  # Convert to milliseconds

        print(
            f"[STATS] {name.split('_')[0]:<6} | "
            f"FLOPs: {flops:.0e} | "
            f"Mem: {mbytes_acc:.2f} MB | "
            f"Ops: {ops:^3} | "
            f"Compile: {compile_time:<6.3f}s | "
            f"Exec: {exec_time_ms:<6.3f}ms | "
            f"Peak Mem: {peak_mem_str}"
        )

    def _deep_analysis(self, dense_fn, core_fn, manual_fn, *args):
        if not ANALYZE:
            return

        print()
        self._analyze(dense_fn, *args)
        self._analyze(core_fn, *args)
        self._analyze(manual_fn, *args)

    # --- Elementwise Tests (6) ---

    def test_01_add_sparse_sparse_union(self):
        s1 = 4
        s2 = 3
        s3 = 5
        if ANALYZE:
            s1 *= 26
            s2 *= 26
            s3 *= 26

        a = SparseTensor(
            (
                SparseDimension(0, s1, val_dim=0, other_id=3),
                DenseDimension(1, s2, 1),
                DenseDimension(2, s3, 2),
            ),
            (SparseDimension(3, s1, val_dim=0, other_id=0),),
            self._n((s1, s2, s3), 1),
        )
        b = SparseTensor(
            (
                SparseDimension(0, s1, val_dim=0, other_id=3),
                DenseDimension(1, s2, 1),
                DenseDimension(2, s3, 2),
            ),
            (SparseDimension(3, s1, val_dim=0, other_id=0),),
            self._n((s1, s2, s3), 2),
        )

        self._deep_analysis(dense_plus, core_plus, manual_01, a, b)
        res = core_plus(a, b)
        self.assertEqual(res.shape, (s1, s2, s3, s1))
        self._assert_equivalence(res, dense_plus(a, b), manual_01(a, b))

    def test_02_sub_sparse_sparse_union(self):
        s1 = 4
        s2 = 2
        s3 = 2
        s4 = 3
        s5 = 5
        if ANALYZE:
            s1 *= 7
            s2 *= 7
            s3 *= 7
            s4 *= 7
            s5 *= 7

        a = SparseTensor(
            (
                SparseDimension(
                    0, s1, val_dim=0, other_id=1, block_size=s2, block_val_dim=1
                ),
                DenseDimension(2, s3, 2),
            ),
            (
                SparseDimension(
                    1, s1, val_dim=0, other_id=0, block_size=s4, block_val_dim=3
                ),
                DenseDimension(3, s5, 4),
            ),
            self._n((s1, s2, s3, s4, s5), 1),
        )
        b = SparseTensor(
            (
                SparseDimension(
                    0, s1, val_dim=0, other_id=1, block_size=s2, block_val_dim=1
                ),
                DenseDimension(2, s3, 2),
            ),
            (
                SparseDimension(
                    1, s1, val_dim=0, other_id=0, block_size=s4, block_val_dim=3
                ),
                DenseDimension(3, s5, 4),
            ),
            self._n((s1, s2, s3, s4, s5), 2),
        )

        self._deep_analysis(dense_minus, core_minus, manual_02, a, b)
        res = core_minus(a, b)
        self.assertEqual(res.shape, (s1 * s2, s3, s1 * s4, s5))
        self._assert_equivalence(res, dense_minus(a, b), manual_02(a, b))

    def test_03_logical_or_unaligned_union(self):
        s1 = 4
        s2 = 2
        s3 = 3
        s4 = 2
        s5 = 4
        s6 = 6
        if ANALYZE:
            s1 *= 45
            s2 *= 45
            s3 *= 45
            s4 *= 45
            s5 *= 45
            s6 *= 45

        a = SparseTensor(
            (
                SparseDimension(
                    0, s1, val_dim=0, other_id=1, block_size=s2, block_val_dim=1
                ),
            ),
            (
                SparseDimension(
                    1, s1, val_dim=0, other_id=0, block_size=s3, block_val_dim=2
                ),
            ),
            self._n((s1, s2, s3), 1, dtype=jnp.bool_),
            dtype=jnp.bool_,
        )
        b = SparseTensor(
            (
                SparseDimension(
                    0, s4, val_dim=0, other_id=1, block_size=s5, block_val_dim=1
                ),
            ),
            (
                SparseDimension(
                    1, s4, val_dim=0, other_id=0, block_size=s6, block_val_dim=2
                ),
            ),
            self._n((s4, s5, s6), 2, dtype=jnp.bool_),
            dtype=jnp.bool_,
        )

        self._deep_analysis(dense_lor, core_lor, manual_03, a, b)
        res = core_lor(a, b)
        self.assertEqual(res.shape, (s1 * s2, s1 * s3))
        self._assert_equivalence(res, dense_lor(a, b), manual_03(a, b))

    def test_04_max_sparse_sparse_union(self):
        s1 = 2
        s2 = 5
        if ANALYZE:
            s1 *= 125
            s2 *= 125

        a = SparseTensor(
            (DenseDimension(0, s1, 0), SparseDimension(1, s2, val_dim=1, other_id=2)),
            (SparseDimension(2, s2, val_dim=1, other_id=1),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (DenseDimension(0, s1, 0), SparseDimension(1, s2, val_dim=1, other_id=2)),
            (SparseDimension(2, s2, val_dim=1, other_id=1),),
            self._n((s1, s2), 2),
        )

        self._deep_analysis(dense_max, core_max, manual_04, a, b)
        res = core_max(a, b)
        self.assertEqual(res.shape, (s1, s2, s2))
        self._assert_equivalence(res, dense_max(a, b), manual_04(a, b))

    def test_05_mul_sparse_sparse_intersection(self):
        s1 = 6
        s2 = 3
        if ANALYZE:
            s1 *= 100
            s2 *= 100

        a = SparseTensor(
            (SparseDimension(0, s1, val_dim=0, other_id=2), DenseDimension(1, s2, 1)),
            (SparseDimension(2, s1, val_dim=0, other_id=0),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (SparseDimension(0, s1, val_dim=0, other_id=2), DenseDimension(1, s2, 1)),
            (SparseDimension(2, s1, val_dim=0, other_id=0),),
            self._n((s1, s2), 2),
        )

        self._deep_analysis(dense_mul, core_mul, manual_05, a, b)
        res = core_mul(a, b)
        self.assertEqual(res.shape, (s1, s2, s1))
        self._assert_equivalence(res, dense_mul(a, b), manual_05(a, b))

    def test_06_power_mismatch_blocks_intersection(self):
        s1 = 4
        s2 = 3
        s3 = 6
        s4 = 2
        if ANALYZE:
            s1 *= 30
            s2 *= 30
            s3 *= 30
            s4 *= 30

        a = SparseTensor(
            (
                SparseDimension(
                    0, s1, val_dim=0, other_id=1, block_size=s2, block_val_dim=1
                ),
            ),
            (
                SparseDimension(
                    1, s1, val_dim=0, other_id=0, block_size=s2, block_val_dim=2
                ),
            ),
            jnp.abs(self._n((s1, s2, s2), 1)),
        )
        b = SparseTensor(
            (
                SparseDimension(
                    0, s3, val_dim=0, other_id=1, block_size=s4, block_val_dim=1
                ),
            ),
            (
                SparseDimension(
                    1, s3, val_dim=0, other_id=0, block_size=s4, block_val_dim=2
                ),
            ),
            jnp.abs(self._n((s3, s4, s4), 2)),
        )

        self._deep_analysis(dense_power, core_power, manual_06, a, b)

        res = core_power(a, b)
        self.assertEqual(res.shape, (s1 * s2, s3 * s4))

        tol = {"atol": 1e-3} if ANALYZE else {}
        self.assertTrue(
            jnp.allclose(manual_06(a, b).dense(), a.dense() ** b.dense(), **tol)
        )
        self.assertTrue(jnp.allclose(res.dense(), a.dense() ** b.dense(), **tol))
        self._assert_equivalence(res, dense_power(a, b), manual_06(a, b))

    # --- Matmul Tests (14) ---

    def test_07_matmul_sparse_sparse_batched(self):
        s1 = 2
        s2 = 6
        if ANALYZE:
            s1 *= 110
            s2 *= 110

        a = SparseTensor(
            (DenseDimension(0, s1, 0), SparseDimension(1, s2, val_dim=1, other_id=2)),
            (SparseDimension(2, s2, val_dim=1, other_id=1),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (DenseDimension(0, s1, 0), SparseDimension(1, s2, val_dim=1, other_id=2)),
            (SparseDimension(2, s2, val_dim=1, other_id=1),),
            self._n((s1, s2), 2),
        )

        self._deep_analysis(dense_matmul, core_matmul, manual_07, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s2))
        self._assert_equivalence(res, dense_matmul(a, b), manual_07(a, b))

    def test_08_matmul_sparse_sparse_3d(self):
        s1 = 2
        s2 = 4
        s3 = 5
        if ANALYZE:
            s1 *= 28
            s2 *= 28
            s3 *= 28

        a = SparseTensor(
            (DenseDimension(0, s1, 0), SparseDimension(1, s2, val_dim=1, other_id=2)),
            (SparseDimension(2, s2, val_dim=1, other_id=1),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (SparseDimension(0, s2, val_dim=0, other_id=1),),
            (SparseDimension(1, s2, val_dim=0, other_id=0), DenseDimension(2, s3, 1)),
            self._n((s2, s3), 2),
        )

        self._deep_analysis(dense_matmul, core_matmul, manual_08, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s2, s3))
        self._assert_equivalence(res, dense_matmul(a, b), manual_08(a, b))

    def test_09_matmul_sparse_sparse_coverage(self):
        s1 = 4
        s2 = 6
        s3 = 4
        s4 = 6
        s5 = 3
        if ANALYZE:
            s1 *= 25
            s2 *= 25
            s3 *= 25
            s4 *= 25
            s5 *= 25

        a = SparseTensor(
            (SparseDimension(0, s1, val_dim=0, other_id=1),),
            (SparseDimension(1, s1, val_dim=0, other_id=0), DenseDimension(2, s2, 1)),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (SparseDimension(0, s4, val_dim=0, other_id=1),),
            (SparseDimension(1, s4, val_dim=0, other_id=0), DenseDimension(2, s5, 1)),
            self._n((s4, s5), 2),
        )

        self._deep_analysis(dense_matmul, core_matmul, manual_09, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s1, s4, s5))
        self._assert_equivalence(res, dense_matmul(a, b), manual_09(a, b))

    def test_10_matmul_sparse_sparse_aligned(self):
        s1 = 6
        s2 = 2
        s3 = 6
        s4 = 3
        if ANALYZE:
            s1 *= 26
            s2 *= 26
            s3 *= 26
            s4 *= 26

        a = SparseTensor(
            (SparseDimension(0, s1, val_dim=0, other_id=2), DenseDimension(1, s2, 1)),
            (SparseDimension(2, s1, val_dim=0, other_id=0),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (SparseDimension(0, s3, val_dim=0, other_id=1),),
            (SparseDimension(1, s3, val_dim=0, other_id=0), DenseDimension(2, s4, 1)),
            self._n((s3, s4), 2),
        )

        self._deep_analysis(dense_matmul, core_matmul, manual_10, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s3, s4))
        self._assert_equivalence(res, dense_matmul(a, b), manual_10(a, b))

    def test_11_matmul_misaligned_blocks(self):
        s1 = 4
        s2 = 2
        s3 = 4
        if ANALYZE:
            s1 *= 35
            s2 *= 35
            s3 *= 35

        a = SparseTensor(
            (
                SparseDimension(
                    0, s1, val_dim=0, other_id=1, block_size=s2, block_val_dim=1
                ),
            ),
            (
                SparseDimension(
                    1, s1, val_dim=0, other_id=0, block_size=s3, block_val_dim=2
                ),
            ),
            self._n((s1, s2, s3), 1),
        )
        b = SparseTensor(
            (
                SparseDimension(
                    0, s1, val_dim=0, other_id=1, block_size=s3, block_val_dim=1
                ),
            ),
            (
                SparseDimension(
                    1, s1, val_dim=0, other_id=0, block_size=s2, block_val_dim=2
                ),
            ),
            self._n((s1, s3, s2), 2),
        )

        self._deep_analysis(dense_matmul, core_matmul, manual_11, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1 * s2, s1 * s2))
        self._assert_equivalence(res, dense_matmul(a, b), manual_11(a, b))

    def test_12_matmul_batch_simple(self):
        s1 = 6
        s2 = 4
        s3 = 5
        if ANALYZE:
            s1 *= 19
            s2 *= 19
            s3 *= 19

        a = SparseTensor(
            (SparseDimension(0, s1, val_dim=0, other_id=2), DenseDimension(1, s2, 1)),
            (SparseDimension(2, s1, val_dim=0, other_id=0),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (DenseDimension(2, s3, 1), SparseDimension(0, s1, val_dim=0, other_id=1)),
            (SparseDimension(1, s1, val_dim=0, other_id=0),),
            self._n((s1, s3), 2),
        )

        self._deep_analysis(dense_matmul, core_matmul, manual_12, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s3, s1))
        self._assert_equivalence(res, dense_matmul(a, b), manual_12(a, b))

    def test_13_matmul_batch_complex(self):
        s1 = 2
        s2 = 3
        s3 = 5
        s4 = 4
        if ANALYZE:
            s1 *= 28
            s2 *= 28
            s3 *= 28
            s4 *= 28

        a = SparseTensor(
            (
                DenseDimension(0, s1, 0),
                DenseDimension(1, s2, 1),
                SparseDimension(2, s3, val_dim=2, other_id=3),
            ),
            (SparseDimension(3, s3, val_dim=2, other_id=2), DenseDimension(4, s4, 3)),
            self._n((s1, s2, s3, s4), 1),
        )
        b = SparseTensor(
            (
                DenseDimension(0, s1, 0),
                DenseDimension(1, s2, 1),
                SparseDimension(2, s3, val_dim=2, other_id=4),
                DenseDimension(3, s4, 3),
            ),
            (SparseDimension(4, s3, val_dim=2, other_id=2),),
            self._n((s1, s2, s3, s4), 2),
        )

        self._deep_analysis(dense_matmul, core_matmul, manual_13, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s3, s3))
        self._assert_equivalence(res, dense_matmul(a, b), manual_13(a, b))

    def test_14_matmul_unmaterialized(self):
        s1 = 5
        s2 = 4
        s3 = 2
        if ANALYZE:
            s1 *= 27
            s2 *= 27
            s3 *= 27

        a = SparseTensor(
            (
                SparseDimension(0, s1, val_dim=None, other_id=2),
                DenseDimension(1, s2, None),
            ),
            (SparseDimension(2, s1, val_dim=None, other_id=0),),
            val=None,
        )
        b = SparseTensor(
            (SparseDimension(0, s1, val_dim=0, other_id=1),),
            (SparseDimension(1, s1, val_dim=0, other_id=0), DenseDimension(2, s3, 1)),
            self._n((s1, s3), 2),
        )

        self._deep_analysis(dense_matmul, core_matmul, manual_14, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s1, s3))
        self._assert_equivalence(res, dense_matmul(a, b), manual_14(a, b))

    def test_15_matmul(self):
        s1 = 6
        s2 = 3
        if ANALYZE:
            s1 *= 600
            s2 *= 600

        a = SparseTensor(
            (SparseDimension(0, s1, val_dim=0, other_id=2),),
            (DenseDimension(1, s2, 1), SparseDimension(2, s1, val_dim=0, other_id=0)),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (DenseDimension(0, s2, 1), SparseDimension(1, s1, val_dim=0, other_id=2)),
            (SparseDimension(2, s1, val_dim=0, other_id=1),),
            self._n((s1, s2), 2),
        )

        self._deep_analysis(dense_matmul, core_matmul, manual_15, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s1))
        self._assert_equivalence(res, dense_matmul(a, b), manual_15(a, b))

    def test_16_matmul_high_rank(self):
        s1 = 2
        s2 = 2
        s3 = 4
        s4 = 2
        if ANALYZE:
            s1 *= 35
            s2 *= 35
            s3 *= 35
            s4 *= 35

        a = SparseTensor(
            (
                DenseDimension(0, s1, 0),
                DenseDimension(1, s2, 1),
                SparseDimension(2, s3, val_dim=2, other_id=3),
            ),
            (SparseDimension(3, s3, val_dim=2, other_id=2), DenseDimension(4, s4, 3)),
            self._n((s1, s2, s3, s4), 1),
        )
        b = SparseTensor(
            (
                DenseDimension(0, s1, 0),
                DenseDimension(1, s2, 1),
                SparseDimension(2, s3, val_dim=2, other_id=4),
                DenseDimension(3, s4, 3),
            ),
            (SparseDimension(4, s3, val_dim=2, other_id=2),),
            self._n((s1, s2, s3, s4), 2),
        )

        self._deep_analysis(dense_matmul, core_matmul, manual_16, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s3, s3))
        self._assert_equivalence(res, dense_matmul(a, b), manual_16(a, b))

    def test_17_matmul_aligned_blocks(self):
        s1 = 6
        s2 = 2
        s3 = 3
        s4 = 4
        if ANALYZE:
            s1 *= 25
            s2 *= 25
            s3 *= 25
            s4 *= 25

        a = SparseTensor(
            (
                SparseDimension(
                    0, s1, val_dim=0, other_id=1, block_size=s2, block_val_dim=1
                ),
            ),
            (
                SparseDimension(
                    1, s1, val_dim=0, other_id=0, block_size=s3, block_val_dim=2
                ),
            ),
            self._n((s1, s2, s3), 1),
        )
        b = SparseTensor(
            (
                SparseDimension(
                    0, s1, val_dim=0, other_id=1, block_size=s3, block_val_dim=1
                ),
            ),
            (
                SparseDimension(
                    1, s1, val_dim=0, other_id=0, block_size=s4, block_val_dim=2
                ),
            ),
            self._n((s1, s3, s4), 2),
        )

        self._deep_analysis(dense_matmul, core_matmul, manual_17, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1 * s2, s1 * s4))
        self._assert_equivalence(res, dense_matmul(a, b), manual_17(a, b))

    def test_18_matmul_sparse_primal_pairs(self):
        s1 = 4
        s2 = 8
        s3 = 5
        if ANALYZE:
            s1 *= 16
            s2 *= 16
            s3 *= 16

        a = SparseTensor(
            (DenseDimension(0, s1, 0), SparseDimension(1, s2, val_dim=1, other_id=2)),
            (SparseDimension(2, s2, val_dim=1, other_id=1),),
            self._n((s1, s2), 1),
        )
        b = SparseTensor(
            (SparseDimension(0, s2, val_dim=0, other_id=1),),
            (SparseDimension(1, s2, val_dim=0, other_id=0), DenseDimension(2, s3, 1)),
            self._n((s2, s3), 2),
        )

        self._deep_analysis(dense_matmul, core_matmul, manual_18, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s2, s3))
        self._assert_equivalence(res, dense_matmul(a, b), manual_18(a, b))

    def test_19_matmul_multi_batch(self):
        s1 = 2
        s2 = 3
        s3 = 4
        s4 = 5
        s5 = 6
        s6 = 7
        if ANALYZE:
            s1 *= 5
            s2 *= 5
            s3 *= 5
            s4 *= 5
            s5 *= 5
            s6 *= 5

        a = SparseTensor(
            (
                DenseDimension(0, s1, 0),
                DenseDimension(1, s2, 1),
                SparseDimension(2, s3, val_dim=2, other_id=4),
                DenseDimension(3, s4, 3),
            ),
            (SparseDimension(4, s3, val_dim=2, other_id=2), DenseDimension(5, s5, 4)),
            self._n((s1, s2, s3, s4, s5), 1),
        )
        b = SparseTensor(
            (
                DenseDimension(0, s1, 0),
                DenseDimension(1, s2, 1),
                SparseDimension(2, s3, val_dim=2, other_id=4),
                DenseDimension(3, s5, 3),
            ),
            (SparseDimension(4, s3, val_dim=2, other_id=2), DenseDimension(5, s6, 4)),
            self._n((s1, s2, s3, s5, s6), 2),
        )
        self._deep_analysis(dense_matmul, core_matmul, manual_19, a, b)
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s3, s4, s3, s6))
        self._assert_equivalence(res, dense_matmul(a, b), manual_19(a, b))

    def test_20_chained_smoke(self):
        s1 = 4
        s2 = 5
        s3 = 6
        s4 = 7
        s5 = 2
        s6 = 3
        s7 = 2
        if ANALYZE:
            s5 *= 6
            s6 *= 5
            s7 *= 5
            s2 *= 6
            s4 *= 6

            s3 = s5 * s6  
            s1 = s5 * s7

        a1 = SparseTensor(
            (SparseDimension(0, s1, val_dim=0, other_id=2), DenseDimension(1, s2, 1)),
            (SparseDimension(2, s1, val_dim=0, other_id=0), DenseDimension(3, s3, 2)),
            self._n((s1, s2, s3), 1),
        )
        a2 = SparseTensor(
            (SparseDimension(0, s3, val_dim=0, other_id=3), DenseDimension(1, s2, 1)),
            (DenseDimension(2, s1, 2), SparseDimension(3, s3, val_dim=0, other_id=0)),
            self._n((s3, s2, s1), 1),
        )
        b = SparseTensor(
            (SparseDimension(0, s1, val_dim=0, other_id=2), DenseDimension(1, s3, 1)),
            (SparseDimension(2, s1, val_dim=0, other_id=0), DenseDimension(3, s4, 2)),
            self._n((s1, s3, s4), 2),
        )
        c1 = SparseTensor(
            (SparseDimension(0, s1, val_dim=0, other_id=2), DenseDimension(1, s2, 1)),
            (SparseDimension(2, s1, val_dim=0, other_id=0), DenseDimension(3, s4, 2)),
            self._n((s1, s2, s4), 3),
        )
        c2 = SparseTensor(
            (
                SparseDimension(0, s5, val_dim=0, other_id=2, block_size=s6),
                DenseDimension(1, s2, 1),
            ),
            (
                SparseDimension(
                    2, s5, val_dim=0, other_id=0, block_size=s7, block_val_dim=2
                ),
                DenseDimension(3, s4, 3),
            ),
            self._n((s5, s2, s7, s4), 3),
        )

        self._deep_analysis(dense_matmul_plus, core_matmul_plus, manual_20_1, a1, b, c1)
        with self.subTest(name="Dimension Type Aligned"):
            res1 = core_matmul_plus(a1, b, c1)
            self.assertEqual(res1.shape, (s1, s2, s1, s4))
            self._assert_equivalence(
                res1, dense_matmul_plus(a1, b, c1), manual_20_1(a1, b, c1)
            )

        self._deep_analysis(dense_matmul_plus, core_matmul_plus, manual_20_2, a2, b, c2)
        with self.subTest(name="Dimension Type Unaligned"):
            res2 = core_matmul_plus(a2, b, c2)
            self.assertEqual(res2.shape, (s3, s2, s1, s4))
            
            self._assert_equivalence(
                res2, dense_matmul_plus(a2, b, c2), manual_20_2(a2, b, c2)
            )



if __name__ == "__main__":
    unittest.main()
