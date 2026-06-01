import builtins
import importlib
import math
import os
import re
import time
import timeit
import unittest
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.ops.elementwise import elementwise
from graphax.sparse.ops.matmul import matmul as _sparse_matmul
from graphax.sparse.tensor import DenseIndex, SparseIndex, SparseTensor
from jax_memory_monitor import PeakMemoryMonitor

ANALYZE = os.getenv("ANALYZE", "0") == "1"
SCALE = int(os.getenv("SCALE", "1"))


# --- Path / optimality assertion helpers --------------------------------
# Each test that wants to assert which fast-path fired passes
# ``expected_path=...`` to ``_deep_analysis``; under the hood we run the
# core fn inside a ``track_paths()`` context manager and check the last
# recorded path. Production runs pay nothing — outside the context
# manager, ``record_path()`` is a no-op.
from graphax.sparse.ops._path_tracking import track_paths


def _expect_path(core_fn, *args, expected_path: str) -> None:
    """Trace ``core_fn(*args)`` once and assert which dispatcher fired.

    Runs inside a ``track_paths()`` block so the recorded path is scoped
    to this assertion (no global state leaks). Tests use this to assert
    that a given input class actually hits the optimization rather than
    silently bailing to the slow path.
    """
    with track_paths() as paths:
        core_fn(*args)
    actual = paths[-1] if paths else None
    if actual != expected_path:
        raise AssertionError(
            f"Expected {core_fn.__name__} to use path {expected_path!r}, "
            f"got {actual!r} (full sequence: {paths})"
        )


def core_plus(a, b):
    return elementwise(a, b, jax.lax.add)


def core_minus(a, b):
    return elementwise(a, b, jax.lax.sub)


def core_lor(a, b):
    return elementwise(a, b, jnp.logical_or)


def core_max(a, b):
    return elementwise(a, b, jax.lax.max)


def core_mul(a, b):
    return elementwise(a, b, jax.lax.mul)


def core_power(a, b):
    return elementwise(a, b, jax.lax.pow)


def core_matmul(a, b):
    return _sparse_matmul(a, b)


def core_matmul_plus(a, b, c):
    return core_plus(core_matmul(a, b), c)


def core_plus_matmul(a, b, c):
    return core_matmul(core_plus(a, b), c)


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


def plus_matmul(a, b, c, **kwargs):
    return DummyDense(matmul(a + b, c, **kwargs).arr)


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
        if fn.__name__ == "plus_matmul":
            kwargs = _get_matmul_kwargs(args[0] + args[1], args[2])
            return fn(*densify(*args), **kwargs)
        return fn(*densify(*args))

    return wrapped


# --- Manual Implementations ---


def manual_01(a, b):
    return SparseTensor(a.out_dims, a.primal_dims, a.val + b.val)


def manual_02(a, b):
    return SparseTensor(a.out_dims, a.primal_dims, a.val - b.val)


def manual_03(a, b):
    s1, s2, s3 = a.val.shape
    s4, s5, s6 = b.val.shape
    a_blocks = jnp.zeros((s4, s5, s6), dtype=jnp.bool_)
    a_val = a.val.reshape(s4, s1 // s4, s2, s3)
    a_blocks = a_blocks.at[:, 0:s2, 0:s3].set(a_val[:, 0])
    a_blocks = a_blocks.at[:, s2 : s2 * 2, s3 : s3 * 2].set(a_val[:, 1])
    return SparseTensor(
        b.out_dims, b.primal_dims, a_blocks | b.val, dtype=jnp.bool_
    )


def manual_04(a, b):
    return SparseTensor(
        a.out_dims, a.primal_dims, jnp.maximum(a.val, b.val)
    )


def manual_05(a, b):
    return SparseTensor(a.out_dims, a.primal_dims, a.val * b.val)


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
    )


def manual_07(a, b):
    return SparseTensor(a.out_dims, b.primal_dims, a.val * b.val)


def manual_08(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[1].size
    s3 = b.primal_dims[1].size
    # a.val[i, j] sits at dense position [i, j, j]; b.val[j, m] at [j, j, m].
    # Matmul contracts the inner sparse pair (j == j), so R_val[i, j, m] =
    # a.val[i, j] * b.val[j, m] (the surviving s2 pair stays diagonal in val).
    R_val = jnp.einsum("ij,jm->ijm", a.val, b.val)
    return SparseTensor(
        (DenseIndex(0, s1, 0), SparseIndex(1, s2, axis=1, other_id=2)),
        (SparseIndex(2, s2, axis=1, other_id=1), DenseIndex(3, s3, 2)),
        R_val,
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
    )


def manual_11(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[0].block_size
    R_val = a.val @ b.val
    return SparseTensor(
        (SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),),
        (SparseIndex(1, s1, axis=0, other_id=0, block_size=s2, block_axis=2),),
        R_val,
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
    )


def manual_14(a, b):
    s1 = b.dims[0].size
    s2 = a.dims[1].size
    s3 = b.primal_dims[1].size
    return SparseTensor(
        (SparseIndex(0, s1, axis=0, other_id=2), DenseIndex(1, s2, None)),
        (SparseIndex(2, s1, axis=0, other_id=0), DenseIndex(3, s3, 1)),
        b.val,
    )


def manual_15(a, b):
    s1 = a.dims[0].size
    R_val = jnp.sum(a.val * b.val, axis=1)
    return SparseTensor(
        (SparseIndex(0, s1, axis=0, other_id=1),),
        (SparseIndex(1, s1, axis=0, other_id=0),),
        R_val,
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
    )


def manual_17(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[0].block_size
    s3 = a.primal_dims[0].block_size
    s4 = b.primal_dims[0].block_size
    R_val = a.val @ b.val
    return SparseTensor(
        (SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),),
        (SparseIndex(1, s1, axis=0, other_id=0, block_size=s4, block_axis=2),),
        R_val,
    )


def manual_18(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[1].size
    s3 = b.primal_dims[1].size
    # See manual_08 — same diagonal-collapse shortcut, just different sizes.
    R_val = jnp.einsum("ij,jm->ijm", a.val, b.val)
    return SparseTensor(
        (DenseIndex(0, s1, 0), SparseIndex(1, s2, axis=1, other_id=2)),
        (SparseIndex(2, s2, axis=1, other_id=1), DenseIndex(3, s3, 2)),
        R_val,
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
    )


def manual_21(a, b):
    s1 = a.dims[0].size  # number of blocks on the diagonal
    s2 = a.dims[0].block_size  # outer block size
    s3 = a.dims[1].block_size  # inner block size
    s4 = b.primal_dims[-1].size

    f = a.fill_value * a.scalar_mult
    v = a.val * a.scalar_mult  # shape (s1, s2, s3)

    NB_o = s1 * s2
    NB_i = s1 * s3
    v_2d = v.reshape(NB_o, s3)
    # Tile via broadcast+reshape — pure shape ops, fold straight into the consumer.
    gathered = jnp.broadcast_to(v_2d[:, None, :], (NB_o, s1, s3)).reshape(NB_o, NB_i)
    blk_o = jnp.arange(NB_o) // s2
    blk_i = jnp.arange(NB_i) // s3
    mask = blk_o[:, None] == blk_i[None, :]
    a_dense = jnp.where(mask, gathered, f)

    R_val = a_dense @ b.val

    return SparseTensor(
        (DenseIndex(0, NB_o, 0),),
        (DenseIndex(1, s4, 1),),
        R_val,
    )


def manual_22(a, b):
    s1 = a.dims[0].size
    s2 = b.primal_dims[0].size
    R_val = a.val[:, None] * b.val
    return SparseTensor(
        (DenseIndex(0, s1, 0),),
        (DenseIndex(1, s2, 1),),
        R_val,
    )


def manual_23(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[1].size
    s3 = a.dims[2].size
    s5 = b.primal_dims[-1].size
    R_val = jax.lax.dot_general(a.val, b.val, (((3,), (3,)), ((0, 1, 2), (0, 1, 2))))
    return SparseTensor(
        (
            SparseIndex(0, s1, axis=0, other_id=3),
            SparseIndex(1, s2, axis=1, other_id=4),
            SparseIndex(2, s3, axis=2, other_id=5),
        ),
        (
            SparseIndex(3, s1, axis=0, other_id=0),
            SparseIndex(4, s2, axis=1, other_id=1),
            SparseIndex(5, s3, axis=2, other_id=2),
            DenseIndex(6, s5, 3),
        ),
        R_val,
    )


def manual_24(a, b):
    s1 = a.dims[0].size  # M (number of blocks)
    s2 = a.dims[0].block_size  # B (block size)
    s3 = b.primal_dims[0].size  # K (output cols)
    b_reshaped = b.val.reshape(s1, s2, s3)
    R_val = jax.lax.dot_general(
        a.val,
        b_reshaped,
        (((2,), (1,)), ((0,), (0,))),
    ).reshape(s1 * s2, s3)
    return SparseTensor(
        (DenseIndex(0, s1 * s2, 0),),
        (DenseIndex(1, s3, 1),),
        R_val,
    )


def manual_25(a, _b):
    return SparseTensor(a.out_dims, a.primal_dims, a.val * a.val)


def manual_26(a, _b):
    s1 = a.dims[0].size
    s2 = a.dims[0].block_size
    R_val = jnp.matmul(a.val, a.val)  # (s1, s2, s2)
    return SparseTensor(
        (SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),),
        (SparseIndex(1, s1, axis=0, other_id=0, block_size=s2, block_axis=2),),
        R_val,
    )


def manual_27(a, b, c):
    s1 = a.dims[0].size
    s2 = a.dims[0].block_size
    s3 = b.dims[0].size
    s4 = b.dims[0].block_size
    s5 = c.primal_dims[0].size
    NB = s1 * s2  # == s3 * s4

    v_2d_a = a.val.reshape(NB, s2)
    g_a = jnp.broadcast_to(v_2d_a[:, None, :], (NB, s1, s2)).reshape(NB, NB)
    blk_a = jnp.arange(NB) // s2
    a_dense = jnp.where(blk_a[:, None] == blk_a[None, :], g_a, 0.0)

    v_2d_b = b.val.reshape(NB, s4)
    g_b = jnp.broadcast_to(v_2d_b[:, None, :], (NB, s3, s4)).reshape(NB, NB)
    blk_b = jnp.arange(NB) // s4
    b_dense = jnp.where(blk_b[:, None] == blk_b[None, :], g_b, 0.0)

    R_val = (a_dense + b_dense) @ c.val
    return SparseTensor(
        (DenseIndex(0, NB, 0),),
        (DenseIndex(1, s5, 1),),
        R_val,
    )


def manual_28(a, b):
    s1 = a.dims[0].size
    s2 = a.dims[0].block_size
    s3 = b.primal_dims[0].size
    NB = s1 * s2

    f = a.fill_value * a.scalar_mult
    v = a.val * a.scalar_mult

    v_2d = v.reshape(NB, s2)
    gathered = jnp.broadcast_to(v_2d[:, None, :], (NB, s1, s2)).reshape(NB, NB)
    blk = jnp.arange(NB) // s2
    a_dense = jnp.where(blk[:, None] == blk[None, :], gathered, f)

    R_val = a_dense @ b.val
    return SparseTensor(
        (DenseIndex(0, NB, 0),),
        (DenseIndex(1, s3, 1),),
        R_val,
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
    )


def manual_plus_aligned(a, b):
    return SparseTensor(a.out_dims, a.primal_dims, a.val + b.val)


def manual_matmul_plus_aligned(a, b, c):
    s1 = a.dims[0].size
    s2 = a.dims[1].size
    s4 = b.primal_dims[1].size
    res_val = jax.lax.dot_general(a.val, b.val, (((2,), (1,)), ((0,), (0,))))
    return SparseTensor(
        (SparseIndex(0, s1, axis=0, other_id=2), DenseIndex(1, s2, 1)),
        (SparseIndex(2, s1, axis=0, other_id=0), DenseIndex(3, s4, 2)),
        res_val + c.val,
    )


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
    )


def manual_plus_unaligned(a, b):
    # a: (s3, s2, s1, s4), b: (s5, s2, s7, s4)
    s5, s2, s7, s4 = b.val.shape
    s6 = a.val.shape[0] // s5

    val_reshaped = a.val.reshape(s5, s6, s2, s5, s7, s4)
    idx = jnp.arange(s5)
    res_val = val_reshaped.at[idx, :, :, idx, :, :].add(b.val[:, None, :, :, :])
    return SparseTensor(
        a.out_dims, a.primal_dims, res_val.reshape(a.val.shape)
    )


def manual_matmul_plus_unaligned(a, b, c):
    # Inlined matmul_unaligned: produces a fully-dense (s3, s2, s1, s4) result.
    b_val_t = jnp.transpose(b.val, (1, 0, 2))  # (s3, s1, s4)
    matmul_val = a.val[:, :, :, None] * b_val_t[:, None, :, :]  # (s3, s2, s1, s4)
    s3, s2, s1 = a.val.shape
    s4 = b.val.shape[2]

    # Inlined plus_unaligned: scatter-add c into matching diagonal positions.
    # c.val has shape (s5, s2, s7, s4) with s3 == s5*s6 and s1 == s5*s7.
    s5, _, s7, _ = c.val.shape
    s6 = s3 // s5
    val_reshaped = matmul_val.reshape(s5, s6, s2, s5, s7, s4)
    idx = jnp.arange(s5)
    summed = val_reshaped.at[idx, :, :, idx, :, :].add(c.val[:, None, :, :, :])

    return SparseTensor(
        (DenseIndex(0, s3, 0), DenseIndex(1, s2, 1)),
        (DenseIndex(2, s1, 2), DenseIndex(3, s4, 3)),
        summed.reshape(s3, s2, s1, s4),
    )


class TestSmokeScreen(unittest.TestCase):
    def setUp(self):
        self.key = jr.PRNGKey(42)

    def _n(self, shape, key_idx=0, dtype=jnp.float32):
        key = jr.PRNGKey(key_idx)
        if dtype == jnp.bool_:
            return jr.normal(key, shape) > 0.0
        return jr.normal(key, shape).astype(dtype)

    def _assert_equivalence(self, res, res_dense, res_manual):
        # Phase 2 dropped sort_val, so ``res.val.shape`` and ``manual.val.shape``
        # may carry equivalent data in different physical layouts (each dim's
        # ``axis`` / ``block_axis`` still points to its own physical slot). The
        # invariant that matters is ``res.dense() == manual.dense()`` — verify
        # that, not the storage layout.
        tol = {"atol": 1e-3} if ANALYZE else {}
        if res_dense is not None:
            self.assertTrue(jnp.allclose(res.dense(), res_dense.dense(), **tol))
        self.assertEqual(res.shape, res_manual.shape)
        self.assertTrue(jnp.allclose(res.dense(), res_manual.dense(), **tol))
        if res.val is None or res_manual.val is None:
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
        peak_mem = 0
        compile_time = 0

        with PeakMemoryMonitor() as monitor:
            t0 = time.perf_counter()
            _ = jax.block_until_ready(jitted_func(*args, **kwargs))
            compile_time = time.perf_counter() - t0
        peak_mem = monitor.peak / 1024**2  # MB
        peak_mem_str = f"{peak_mem:.2f} MB"

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
        return {
            "flops": flops,
            "bytes_acc_mb": mbytes_acc,
            "peak_mem_mb": peak_mem,
            "ops": ops,
            "hlo_len": len(hlo),
            "exec_ms": exec_time_ms,
        }

    def _deep_analysis(
        self,
        fn,
        core_fn,
        manual_fn,
        *args,
        expected_path: str | None = None,
        flops_ratio_max: float | None = None,
        mem_ratio_max: float | None = None,
        hlo_ratio_max: float | None = None,
    ):
        """Run the four-way analysis on densify/wrap/core/manual and optionally
        assert that:
          * the recorded fast-path identifier matches ``expected_path``
          * core's FLOPs / peak-memory / HLO-size are within the given ratio
            of manual's (use ``None`` to skip a particular check)

        ``expected_path`` is asserted regardless of ``ANALYZE`` (cheap — a
        single call records the path). The bound assertions only fire under
        ``ANALYZE`` since they need the cost-analysis data, which is
        otherwise not collected.
        """
        # Path assertion — runs even when ANALYZE is off, since path tracking
        # is cheap (one extra global write per dispatch). Catches silent
        # bails of fast paths in any test environment.
        if expected_path is not None:
            _expect_path(core_fn, *args, expected_path=expected_path)

        if not ANALYZE:
            return

        print()

        self._analyze(densify, *args)
        dargs = densify(*args)
        if fn.__name__ in ("matmul", "matmul_plus"):
            kwargs = _get_matmul_kwargs(*args[:2])
        elif fn.__name__ == "plus_matmul":
            kwargs = _get_matmul_kwargs(args[0] + args[1], args[2])
        else:
            kwargs = {}
        self._analyze(fn, *dargs, **kwargs)
        self._analyze(wrap_dense(fn), *args)
        core_stats = self._analyze(core_fn, *args)
        manual_stats = self._analyze(manual_fn, *args)

        def _check_ratio(key, max_ratio, label):
            if max_ratio is None:
                return
            cv, mv = core_stats[key], manual_stats[key]
            if mv == 0:
                # Avoid div by zero; require core to also be zero or near-zero.
                if cv > 1.0:
                    raise AssertionError(
                        f"{label}: manual reports 0 but core reports {cv}"
                    )
                return
            ratio = cv / mv
            if ratio > max_ratio:
                raise AssertionError(
                    f"{label}: core/manual = {cv}/{mv} = {ratio:.2f}× "
                    f"exceeds the {max_ratio:.2f}× bound"
                )

        _check_ratio("flops", flops_ratio_max, "FLOPs ratio")
        _check_ratio("peak_mem_mb", mem_ratio_max, "Peak memory ratio")
        _check_ratio("hlo_len", hlo_ratio_max, "HLO size ratio")

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

        self._deep_analysis(
            plus, core_plus, manual_01, a, b, expected_path="general", mem_ratio_max=1.5
        )
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
                SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),
                DenseIndex(2, s3, 2),
            ),
            (
                SparseIndex(1, s1, axis=0, other_id=0, block_size=s4, block_axis=3),
                DenseIndex(3, s5, 4),
            ),
            self._n((s1, s2, s3, s4, s5), 1),
        )
        b = SparseTensor(
            (
                SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),
                DenseIndex(2, s3, 2),
            ),
            (
                SparseIndex(1, s1, axis=0, other_id=0, block_size=s4, block_axis=3),
                DenseIndex(3, s5, 4),
            ),
            self._n((s1, s2, s3, s4, s5), 2),
        )

        self._deep_analysis(
            minus,
            core_minus,
            manual_02,
            a,
            b,
            expected_path="general",
            mem_ratio_max=1.5,
        )
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
            (SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),),
            (SparseIndex(1, s1, axis=0, other_id=0, block_size=s3, block_axis=2),),
            self._n((s1, s2, s3), 1, dtype=jnp.bool_),
            dtype=jnp.bool_,
        )
        b = SparseTensor(
            (SparseIndex(0, s4, axis=0, other_id=1, block_size=s5, block_axis=1),),
            (SparseIndex(1, s4, axis=0, other_id=0, block_size=s6, block_axis=2),),
            self._n((s4, s5, s6), 2, dtype=jnp.bool_),
            dtype=jnp.bool_,
        )

        self._deep_analysis(
            lor,
            core_lor,
            manual_03,
            a,
            b,
            expected_path="divisor_fast",
            mem_ratio_max=2.0,
        )
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

        self._deep_analysis(
            max, core_max, manual_04, a, b, expected_path="general", mem_ratio_max=1.5
        )
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

        self._deep_analysis(
            mul, core_mul, manual_05, a, b, expected_path="general", mem_ratio_max=1.5
        )
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
            (SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),),
            (SparseIndex(1, s1, axis=0, other_id=0, block_size=s2, block_axis=2),),
            jnp.abs(self._n((s1, s2, s2), 1)),
        )
        b = SparseTensor(
            (SparseIndex(0, s3, axis=0, other_id=1, block_size=s4, block_axis=1),),
            (SparseIndex(1, s3, axis=0, other_id=0, block_size=s4, block_axis=2),),
            jnp.abs(self._n((s3, s4, s4), 2)),
        )

        self._deep_analysis(power, core_power, manual_06, a, b, expected_path="general")

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

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_07,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
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

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_08,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
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

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_09,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
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

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_10,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
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
            (SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),),
            (SparseIndex(1, s1, axis=0, other_id=0, block_size=s3, block_axis=2),),
            self._n((s1, s2, s3), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=1, block_size=s3, block_axis=1),),
            (SparseIndex(1, s1, axis=0, other_id=0, block_size=s2, block_axis=2),),
            self._n((s1, s3, s2), 2),
        )

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_11,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
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

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_12,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
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

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_13,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
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

        self._deep_analysis(matmul, core_matmul, manual_14, a, b, expected_path="tiled")
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

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_15,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
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

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_16,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
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
            (SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),),
            (SparseIndex(1, s1, axis=0, other_id=0, block_size=s3, block_axis=2),),
            self._n((s1, s2, s3), 1),
        )
        b = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=1, block_size=s3, block_axis=1),),
            (SparseIndex(1, s1, axis=0, other_id=0, block_size=s4, block_axis=2),),
            self._n((s1, s3, s4), 2),
        )

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_17,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
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

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_18,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
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

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_19,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
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
                SparseIndex(2, s5, axis=0, other_id=0, block_size=s7, block_axis=2),
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

    def test_21_matmul_nonzero_fill(self):
        """Smoke + analyze the late-densification matmul path (non-zero ``fill_value``).

        ``a`` is a block-diagonal sparse tensor whose unstored positions carry a non-zero
        fill, so the tiled algorithm's "implicit zero" assumption no longer holds and the
        backend must reroute to a dense ``dot_general``. ``manual_21`` is the optimal
        hand-coded reference: a single broadcast+select densify (one ``kLoop`` fusion in
        HLO) feeding directly into ``jnp.matmul``. With ``IMPL=7``, the core path emits
        the same fusion shape; with ``IMPL=0`` the canonical implementation runs the
        scatter-based ``.dense()`` path before the matmul (more HBM traffic, more ops).
        """
        s1 = 4  # number of blocks on the diagonal
        s2 = 2  # outer block size
        s3 = 2  # inner block size
        s4 = 5  # rhs primal size
        if ANALYZE:
            s1 *= 18 * SCALE
            s2 *= 18 * SCALE
            s3 *= 18 * SCALE
            s4 *= 18 * SCALE

        a = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),),
            (SparseIndex(1, s1, axis=0, other_id=0, block_size=s3, block_axis=2),),
            self._n((s1, s2, s3), 1),
            fill_value=jnp.array(0.5, dtype=jnp.float32),
        )
        b = SparseTensor(
            (DenseIndex(0, s1 * s3, 0),),
            (DenseIndex(1, s4, 1),),
            self._n((s1 * s3, s4), 2),
        )

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_21,
            a,
            b,
            expected_path="densify",
            mem_ratio_max=1.5,
        )
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1 * s2, s4))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_21(a, b))

    def test_22_pure_diagonal_x_dense(self):
        """Pure diagonal × dense matrix.

        ``a`` is a pure diagonal (single sparse pair, *no block*) — val.shape =
        (s1,). ``b`` is a dense matrix (s1, s2). The matmul collapses to a
        broadcast-multiply ``a.val[:, None] * b.val`` — no reduction, no
        gather, just elementwise. Stresses the fast-path dispatcher's ability
        to recognize that a sib-pair contracting axis with no block on lhs is
        equivalent to batching when the partner side has the matched length
        as a dense axis.
        """
        s1 = 8
        s2 = 5
        if ANALYZE:
            s1 *= 80 * SCALE
            s2 *= 80 * SCALE

        a = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=1),),
            (SparseIndex(1, s1, axis=0, other_id=0),),
            self._n((s1,), 1),
        )
        b = SparseTensor(
            (DenseIndex(0, s1, 0),),
            (DenseIndex(1, s2, 1),),
            self._n((s1, s2), 2),
        )

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_22,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_22(a, b))

    def test_23_deep_batched_matmul(self):
        """Deep-batched 2-D matmul (3 sib-pair batches + 1 dense contract).

        Stress-tests the dot_general fast path with a 3-deep batching
        topology: every sib pair on lhs has a same-id sib pair on rhs (they
        match as ``batch_sparse``), the trailing dense axis on lhs.primal
        contracts with the dense axis on rhs.out, and rhs has a final
        ``spatial_primal_rhs`` dense axis.
        """
        s1 = 3
        s2 = 4
        s3 = 5
        s4 = 6
        s5 = 4
        if ANALYZE:
            # Modest scaling here — densify of the 5-D batched val grows as
            # (s1·s2·s3)² × s4 × s5, which blows up fast. The scaled core
            # values are still much larger than the un-scaled defaults, so
            # ANALYZE-mode FLOPs / memory metrics are still meaningful.
            s1 *= 2 * SCALE
            s2 *= 2 * SCALE
            s3 *= 2 * SCALE
            s4 *= 2 * SCALE
            s5 *= 2 * SCALE

        a = SparseTensor(
            (
                SparseIndex(0, s1, axis=0, other_id=3),
                SparseIndex(1, s2, axis=1, other_id=4),
                SparseIndex(2, s3, axis=2, other_id=5),
            ),
            (
                SparseIndex(3, s1, axis=0, other_id=0),
                SparseIndex(4, s2, axis=1, other_id=1),
                SparseIndex(5, s3, axis=2, other_id=2),
                DenseIndex(6, s4, 3),
            ),
            self._n((s1, s2, s3, s4), 1),
        )
        b = SparseTensor(
            (
                SparseIndex(0, s1, axis=0, other_id=3),
                SparseIndex(1, s2, axis=1, other_id=4),
                SparseIndex(2, s3, axis=2, other_id=5),
                DenseIndex(6, s4, 3),
            ),
            (
                SparseIndex(3, s1, axis=0, other_id=0),
                SparseIndex(4, s2, axis=1, other_id=1),
                SparseIndex(5, s3, axis=2, other_id=2),
                DenseIndex(7, s5, 4),
            ),
            self._n((s1, s2, s3, s4, s5), 2),
        )

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_23,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1, s2, s3, s1, s2, s3, s5))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_23(a, b))

    def test_24_block_diagonal_x_dense(self):
        """Block-diagonal × dense-matrix.

        ``a`` is block-diagonal: ``M`` square blocks of size ``B`` × ``B``
        with the rest of the implicit (``M*B``, ``M*B``) tensor being zero.
        ``b`` is a dense (``M*B``, ``K``) matrix. The matmul reduces to
        a single batched ``dot_general(a.val, reshape(b.val, (M, B, K)), ...)``
        followed by a reshape — no LCM mismatch, no segment_sum, just
        per-block matmuls with the correct b-slices.
        """
        s1 = 4
        s2 = 3
        s3 = 5
        if ANALYZE:
            s1 *= 18 * SCALE
            s2 *= 18 * SCALE
            s3 *= 18 * SCALE

        a = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),),
            (SparseIndex(1, s1, axis=0, other_id=0, block_size=s2, block_axis=2),),
            self._n((s1, s2, s2), 1),
        )
        b = SparseTensor(
            (DenseIndex(0, s1 * s2, 0),),
            (DenseIndex(1, s3, 1),),
            self._n((s1 * s2, s3), 2),
        )

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_24,
            a,
            b,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1 * s2, s3))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_24(a, b))

    # --- Correctness gap tests (added as part of optimality assertions) ---

    def test_25_self_elementwise_mul(self):
        """Self-elementwise multiplication (a * a). Output should equal
        elementwise square; tests that the dispatcher handles repeated-input
        cases correctly (a known JIT footgun where args alias each other)."""
        s1 = 6
        s2 = 4
        if ANALYZE:
            s1 *= 30 * SCALE
            s2 *= 30 * SCALE

        a = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=2),),
            (DenseIndex(1, s2, 1), SparseIndex(2, s1, axis=0, other_id=0)),
            self._n((s1, s2), 7),
        )

        # Pass ``a`` twice — same Python object on both sides.
        self._deep_analysis(mul, core_mul, manual_25, a, a)
        res = core_mul(a, a)
        self.assertEqual(res.shape, a.shape)
        self._assert_equivalence(res, wrap_dense(mul)(a, a), manual_25(a, a))

    def test_26_self_matmul_gram(self):
        """Self-matmul ``a @ a`` where ``a`` is a square sib-pair tensor.
        The contracting axis aliases the kept axis on the other side — a
        non-trivial topology pattern. Tests the Gram-matrix-like idiom."""
        s1 = 4
        s2 = 3
        if ANALYZE:
            s1 *= 25 * SCALE
            s2 *= 25 * SCALE

        a = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),),
            (SparseIndex(1, s1, axis=0, other_id=0, block_size=s2, block_axis=2),),
            self._n((s1, s2, s2), 8),
        )

        self._deep_analysis(
            matmul,
            core_matmul,
            manual_26,
            a,
            a,
            expected_path="tiled",
            mem_ratio_max=1.5,
        )
        res = core_matmul(a, a)
        self.assertEqual(res.shape, (s1 * s2, s1 * s2))
        self._assert_equivalence(res, wrap_dense(matmul)(a, a), manual_26(a, a))

    def test_27_chained_compressed_union_then_matmul(self):
        """``(a + b) @ c`` where ``a + b`` produces a ``compressed_val=
        UnionBlocks``. Verifies the lazy form composes — the matmul should
        materialize the union inline (XLA folds it into the matmul kernel)
        rather than eagerly densifying first. Correctness check is the
        primary goal; memory should also stay bounded."""
        s1 = 5  # a's outer
        s2 = 3  # a's block (5×3 = 15 logical)
        s3 = 3  # b's outer
        s4 = 5  # b's block (3×5 = 15 logical, matches a's logical size)
        s5 = 4  # c's primal dim
        if ANALYZE:
            s1 *= 8 * SCALE
            s2 *= 8 * SCALE
            s3 *= 8 * SCALE
            s4 *= 8 * SCALE
            s5 *= 8 * SCALE

        a = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),),
            (SparseIndex(1, s1, axis=0, other_id=0, block_size=s2, block_axis=2),),
            self._n((s1, s2, s2), 9),
        )
        b = SparseTensor(
            (SparseIndex(0, s3, axis=0, other_id=1, block_size=s4, block_axis=1),),
            (SparseIndex(1, s3, axis=0, other_id=0, block_size=s4, block_axis=2),),
            self._n((s3, s4, s4), 10),
        )
        c = SparseTensor(
            (DenseIndex(0, s1 * s2, 0),),
            (DenseIndex(1, s5, 1),),
            self._n((s1 * s2, s5), 11),
        )

        # Correctness only — no path/bound assertion since the result of
        # ``(a + b)`` here may or may not be UnionBlocks depending on whether
        # the storage check triggers. The key invariant: the final dense
        # result equals the manually composed (a + b) @ c.
        self._deep_analysis(plus_matmul, core_plus_matmul, manual_27, a, b, c)
        res = core_plus_matmul(a, b, c)
        self.assertEqual(res.shape, (s1 * s2, s5))
        self._assert_equivalence(
            res, wrap_dense(plus_matmul)(a, b, c), manual_27(a, b, c)
        )

    def test_28_nonzero_fill_both_sides_matmul(self):
        """Both sides have non-zero ``fill_value`` — exercises the densify
        path with a fully-implicit-fill product. Previously untested in the
        analyze suite; ``test_21`` only had non-zero fill on the lhs."""
        s1 = 4  # block count
        s2 = 2  # block size
        s3 = 5  # rhs primal
        if ANALYZE:
            s1 *= 18 * SCALE
            s2 *= 18 * SCALE
            s3 *= 18 * SCALE

        a = SparseTensor(
            (SparseIndex(0, s1, axis=0, other_id=1, block_size=s2, block_axis=1),),
            (SparseIndex(1, s1, axis=0, other_id=0, block_size=s2, block_axis=2),),
            self._n((s1, s2, s2), 12),
            fill_value=jnp.array(0.3, dtype=jnp.float32),
        )
        b = SparseTensor(
            (DenseIndex(0, s1 * s2, 0),),
            (DenseIndex(1, s3, 1),),
            self._n((s1 * s2, s3), 13),
            fill_value=jnp.array(-0.7, dtype=jnp.float32),
        )

        self._deep_analysis(
            matmul, core_matmul, manual_28, a, b, expected_path="densify"
        )
        res = core_matmul(a, b)
        self.assertEqual(res.shape, (s1 * s2, s3))
        self._assert_equivalence(res, wrap_dense(matmul)(a, b), manual_28(a, b))

    # --- Phase 1a additions: refinement spectrum + band scenarios ---
    # These exercise the cases the unified-kernel work in Phase 5/6 needs to
    # collapse into one algorithm. Correctness only here — dense ground truth
    # via ``a.dense() + b.dense()`` / ``a.dense() @ b.dense()``. Phase 1b will
    # wrap these in ``_expect_path(...)`` once the path taxonomy is locked in.

    def _mul_intersection(self, a, b):
        return elementwise(a, b, jax.lax.mul, is_intersection=True)

    # --- Elementwise refinement spectrum (union: add) -------------------

    def test_29_add_1refine_aligned(self):
        """Union add on two sparse pairs with identical block_size on the
        contracting axis (gcd == block_size — degenerate refinement)."""
        n = 4
        b = 4  # logical 16
        if ANALYZE:
            n *= 8 * SCALE
            b *= 8 * SCALE

        a = SparseTensor(
            (SparseIndex(0, n, axis=0, other_id=1, block_size=b, block_axis=1),),
            (SparseIndex(1, n, axis=0, other_id=0, block_size=b, block_axis=2),),
            self._n((n, b, b), 29),
        )
        rhs = SparseTensor(
            (SparseIndex(0, n, axis=0, other_id=1, block_size=b, block_axis=1),),
            (SparseIndex(1, n, axis=0, other_id=0, block_size=b, block_axis=2),),
            self._n((n, b, b), 129),
        )
        _expect_path(core_plus, a, rhs, expected_path="general")
        expected = a.dense() + rhs.dense()
        res = core_plus(a, rhs)
        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(jnp.allclose(res.dense(), expected, atol=1e-5))

    def test_30_add_multi_refine(self):
        """Union add with non-trivial GCD on the contracting axis.
        lhs block 4 over 6 outer (logical 24), rhs block 6 over 4 outer
        (logical 24). gcd == 2, LCM == 12."""
        n_lhs, b_lhs = 6, 4
        n_rhs, b_rhs = 4, 6
        if ANALYZE:
            n_lhs *= 4 * SCALE
            b_lhs *= 4 * SCALE
            n_rhs *= 4 * SCALE
            b_rhs *= 4 * SCALE

        a = SparseTensor(
            (SparseIndex(0, n_lhs, axis=0, other_id=1, block_size=b_lhs, block_axis=1),),
            (SparseIndex(1, n_lhs, axis=0, other_id=0, block_size=b_lhs, block_axis=2),),
            self._n((n_lhs, b_lhs, b_lhs), 30),
        )
        rhs = SparseTensor(
            (SparseIndex(0, n_rhs, axis=0, other_id=1, block_size=b_rhs, block_axis=1),),
            (SparseIndex(1, n_rhs, axis=0, other_id=0, block_size=b_rhs, block_axis=2),),
            self._n((n_rhs, b_rhs, b_rhs), 130),
        )
        _expect_path(core_plus, a, rhs, expected_path="compressed_union")
        expected = a.dense() + rhs.dense()
        res = core_plus(a, rhs)
        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(jnp.allclose(res.dense(), expected, atol=1e-5))

    def test_31_add_very_unaligned(self):
        """Union add with coprime block sizes (gcd == 1) — LCM == full
        logical size, so the LCM-grid degenerates to a single big block.
        lhs block 5 over 7 outer (logical 35), rhs block 7 over 5 outer."""
        n_lhs, b_lhs = 7, 5
        n_rhs, b_rhs = 5, 7
        if ANALYZE:
            n_lhs *= 3 * SCALE
            b_lhs *= 3 * SCALE
            n_rhs *= 3 * SCALE
            b_rhs *= 3 * SCALE

        a = SparseTensor(
            (SparseIndex(0, n_lhs, axis=0, other_id=1, block_size=b_lhs, block_axis=1),),
            (SparseIndex(1, n_lhs, axis=0, other_id=0, block_size=b_lhs, block_axis=2),),
            self._n((n_lhs, b_lhs, b_lhs), 31),
        )
        rhs = SparseTensor(
            (SparseIndex(0, n_rhs, axis=0, other_id=1, block_size=b_rhs, block_axis=1),),
            (SparseIndex(1, n_rhs, axis=0, other_id=0, block_size=b_rhs, block_axis=2),),
            self._n((n_rhs, b_rhs, b_rhs), 131),
        )
        _expect_path(core_plus, a, rhs, expected_path="compressed_union")
        expected = a.dense() + rhs.dense()
        res = core_plus(a, rhs)
        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(jnp.allclose(res.dense(), expected, atol=1e-5))

    # --- Elementwise refinement spectrum (intersection: mul) ------------

    def test_32_mul_1refine_aligned(self):
        """Intersection mul on two sparse pairs with identical block_size."""
        n = 4
        b = 4
        if ANALYZE:
            n *= 8 * SCALE
            b *= 8 * SCALE

        a = SparseTensor(
            (SparseIndex(0, n, axis=0, other_id=1, block_size=b, block_axis=1),),
            (SparseIndex(1, n, axis=0, other_id=0, block_size=b, block_axis=2),),
            self._n((n, b, b), 32),
        )
        rhs = SparseTensor(
            (SparseIndex(0, n, axis=0, other_id=1, block_size=b, block_axis=1),),
            (SparseIndex(1, n, axis=0, other_id=0, block_size=b, block_axis=2),),
            self._n((n, b, b), 132),
        )
        _expect_path(self._mul_intersection, a, rhs, expected_path="general")
        expected = a.dense() * rhs.dense()
        res = self._mul_intersection(a, rhs)
        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(jnp.allclose(res.dense(), expected, atol=1e-5))

    def test_33_mul_multi_refine(self):
        """Intersection mul with non-trivial GCD (gcd == 2, LCM == 12)."""
        n_lhs, b_lhs = 6, 4
        n_rhs, b_rhs = 4, 6
        if ANALYZE:
            n_lhs *= 4 * SCALE
            b_lhs *= 4 * SCALE
            n_rhs *= 4 * SCALE
            b_rhs *= 4 * SCALE

        a = SparseTensor(
            (SparseIndex(0, n_lhs, axis=0, other_id=1, block_size=b_lhs, block_axis=1),),
            (SparseIndex(1, n_lhs, axis=0, other_id=0, block_size=b_lhs, block_axis=2),),
            self._n((n_lhs, b_lhs, b_lhs), 33),
        )
        rhs = SparseTensor(
            (SparseIndex(0, n_rhs, axis=0, other_id=1, block_size=b_rhs, block_axis=1),),
            (SparseIndex(1, n_rhs, axis=0, other_id=0, block_size=b_rhs, block_axis=2),),
            self._n((n_rhs, b_rhs, b_rhs), 133),
        )
        _expect_path(self._mul_intersection, a, rhs, expected_path="general")
        expected = a.dense() * rhs.dense()
        res = self._mul_intersection(a, rhs)
        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(jnp.allclose(res.dense(), expected, atol=1e-5))

    def test_34_mul_very_unaligned(self):
        """Intersection mul with coprime block sizes (gcd == 1, LCM == 35)."""
        n_lhs, b_lhs = 7, 5
        n_rhs, b_rhs = 5, 7
        if ANALYZE:
            n_lhs *= 3 * SCALE
            b_lhs *= 3 * SCALE
            n_rhs *= 3 * SCALE
            b_rhs *= 3 * SCALE

        a = SparseTensor(
            (SparseIndex(0, n_lhs, axis=0, other_id=1, block_size=b_lhs, block_axis=1),),
            (SparseIndex(1, n_lhs, axis=0, other_id=0, block_size=b_lhs, block_axis=2),),
            self._n((n_lhs, b_lhs, b_lhs), 34),
        )
        rhs = SparseTensor(
            (SparseIndex(0, n_rhs, axis=0, other_id=1, block_size=b_rhs, block_axis=1),),
            (SparseIndex(1, n_rhs, axis=0, other_id=0, block_size=b_rhs, block_axis=2),),
            self._n((n_rhs, b_rhs, b_rhs), 134),
        )
        _expect_path(self._mul_intersection, a, rhs, expected_path="general")
        expected = a.dense() * rhs.dense()
        res = self._mul_intersection(a, rhs)
        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(jnp.allclose(res.dense(), expected, atol=1e-5))

    # --- Matmul refinement spectrum ------------------------------------

    def test_35_matmul_1refine_aligned(self):
        """Matmul on two sparse pairs with identical contracting block size
        (gcd == block_size). The aligned fast path is expected to fire."""
        n = 4
        b = 4
        if ANALYZE:
            n *= 8 * SCALE
            b *= 8 * SCALE

        a = SparseTensor(
            (SparseIndex(0, n, axis=0, other_id=1, block_size=b, block_axis=1),),
            (SparseIndex(1, n, axis=0, other_id=0, block_size=b, block_axis=2),),
            self._n((n, b, b), 35),
        )
        rhs = SparseTensor(
            (SparseIndex(0, n, axis=0, other_id=1, block_size=b, block_axis=1),),
            (SparseIndex(1, n, axis=0, other_id=0, block_size=b, block_axis=2),),
            self._n((n, b, b), 135),
        )
        _expect_path(core_matmul, a, rhs, expected_path="tiled")
        expected = a.dense() @ rhs.dense()
        res = core_matmul(a, rhs)
        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(jnp.allclose(res.dense(), expected, atol=1e-4))

    def test_36_matmul_multi_refine(self):
        """Matmul with non-trivial GCD on the contracting axis.
        lhs block 4 over 6 outer (logical 24), rhs block 6 over 4 outer
        (logical 24). gcd == 2 — the tiled path's LCM-grid is expected."""
        n_lhs, b_lhs = 6, 4
        n_rhs, b_rhs = 4, 6
        if ANALYZE:
            n_lhs *= 4 * SCALE
            b_lhs *= 4 * SCALE
            n_rhs *= 4 * SCALE
            b_rhs *= 4 * SCALE

        a = SparseTensor(
            (SparseIndex(0, n_lhs, axis=0, other_id=1, block_size=b_lhs, block_axis=1),),
            (SparseIndex(1, n_lhs, axis=0, other_id=0, block_size=b_lhs, block_axis=2),),
            self._n((n_lhs, b_lhs, b_lhs), 36),
        )
        rhs = SparseTensor(
            (SparseIndex(0, n_rhs, axis=0, other_id=1, block_size=b_rhs, block_axis=1),),
            (SparseIndex(1, n_rhs, axis=0, other_id=0, block_size=b_rhs, block_axis=2),),
            self._n((n_rhs, b_rhs, b_rhs), 136),
        )
        _expect_path(core_matmul, a, rhs, expected_path="tiled")
        expected = a.dense() @ rhs.dense()
        res = core_matmul(a, rhs)
        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(jnp.allclose(res.dense(), expected, atol=1e-4))

    def test_37_matmul_very_unaligned(self):
        """Matmul with coprime contracting block sizes (gcd == 1, LCM == 35).
        Every output element pulls data from every input block on the
        contracting axis — stress test for the tiled path."""
        n_lhs, b_lhs = 7, 5
        n_rhs, b_rhs = 5, 7
        if ANALYZE:
            n_lhs *= 3 * SCALE
            b_lhs *= 3 * SCALE
            n_rhs *= 3 * SCALE
            b_rhs *= 3 * SCALE

        a = SparseTensor(
            (SparseIndex(0, n_lhs, axis=0, other_id=1, block_size=b_lhs, block_axis=1),),
            (SparseIndex(1, n_lhs, axis=0, other_id=0, block_size=b_lhs, block_axis=2),),
            self._n((n_lhs, b_lhs, b_lhs), 37),
        )
        rhs = SparseTensor(
            (SparseIndex(0, n_rhs, axis=0, other_id=1, block_size=b_rhs, block_axis=1),),
            (SparseIndex(1, n_rhs, axis=0, other_id=0, block_size=b_rhs, block_axis=2),),
            self._n((n_rhs, b_rhs, b_rhs), 137),
        )
        _expect_path(core_matmul, a, rhs, expected_path="tiled")
        expected = a.dense() @ rhs.dense()
        res = core_matmul(a, rhs)
        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(jnp.allclose(res.dense(), expected, atol=1e-4))

    # --- Band scenarios (uniform blocks per Phase 5d deferral) ---------

    def test_38_dense_x_block_diag(self):
        """Dense × block-diagonal (uniform blocks). Output's column axis
        is the block-diag's logical size. Complement to test_24 (which is
        block-diag × dense). 'Horizontal band' in the user's framing —
        the block-diag is on the right and has many small blocks."""
        m = 6   # rows of dense lhs
        n = 8   # block count on rhs (more, smaller blocks)
        b = 4   # block size on rhs
        if ANALYZE:
            m *= 10 * SCALE
            n *= 10 * SCALE
            b *= 10 * SCALE

        a = SparseTensor(
            (DenseIndex(0, m, 0),),
            (DenseIndex(1, n * b, 1),),
            self._n((m, n * b), 38),
        )
        rhs = SparseTensor(
            (SparseIndex(0, n, axis=0, other_id=1, block_size=b, block_axis=1),),
            (SparseIndex(1, n, axis=0, other_id=0, block_size=b, block_axis=2),),
            self._n((n, b, b), 138),
        )
        _expect_path(core_matmul, a, rhs, expected_path="tiled")
        expected = a.dense() @ rhs.dense()
        res = core_matmul(a, rhs)
        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(jnp.allclose(res.dense(), expected, atol=1e-4))

    def test_39_block_diag_x_dense_fewer_larger(self):
        """Block-diagonal × dense, with fewer, larger blocks on the lhs.
        Companion to test_24 (which uses many small blocks). 'Vertical
        band' in the user's framing — output's row axis is the block-diag
        side, large per-block rows."""
        n = 2   # block count on lhs (fewer, larger blocks)
        b = 16  # block size on lhs
        k = 5   # cols of dense rhs
        if ANALYZE:
            n *= 8 * SCALE
            b *= 8 * SCALE
            k *= 8 * SCALE

        a = SparseTensor(
            (SparseIndex(0, n, axis=0, other_id=1, block_size=b, block_axis=1),),
            (SparseIndex(1, n, axis=0, other_id=0, block_size=b, block_axis=2),),
            self._n((n, b, b), 39),
        )
        rhs = SparseTensor(
            (DenseIndex(0, n * b, 0),),
            (DenseIndex(1, k, 1),),
            self._n((n * b, k), 139),
        )
        _expect_path(core_matmul, a, rhs, expected_path="tiled")
        expected = a.dense() @ rhs.dense()
        res = core_matmul(a, rhs)
        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(jnp.allclose(res.dense(), expected, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
