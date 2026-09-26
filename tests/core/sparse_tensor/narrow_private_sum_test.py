"""A private sum of a narrow edge reads the edge as it is stored (dsnn-dfw.273).

The sum upcasts the bf16 edge to f32. XLA sees the Quant cast f32 -> bf16 and that
upcast as a pair and drops both, so it summed the unrounded f32 value: NN256 pilot
plan compress_e993 read 4 MB in place of 2 MB and ran 1.8% slower than core-v2 (job
68378). An optimization barrier on the edge keeps the stored value for the sum.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from graphax.sparse.ops.matmul import _emit_einsum


def _sum_of_quantized(p):
    # The Quant cast of an f32 edge, then a contraction whose batch label only it carries.
    a = p.astype(jnp.bfloat16)
    w = jnp.ones((p.shape[1],), jnp.bfloat16)
    return _emit_einsum(a, [0, 1], w, [1], [1]).materialize()


def test_the_private_sum_of_a_narrow_edge_reads_the_stored_edge():
    jaxpr = jax.make_jaxpr(_sum_of_quantized)(jnp.zeros((2, 3), jnp.float32)).jaxpr
    names = [e.primitive.name for e in jaxpr.eqns]
    assert "optimization_barrier" in names, names
    barrier = jaxpr.eqns[names.index("optimization_barrier")]
    assert jnp.dtype(barrier.invars[0].aval.dtype) == jnp.dtype(jnp.bfloat16), barrier


def test_the_sum_is_of_the_rounded_values():
    # 1 + 3 * 2^-9 rounds to 1 + 2^-7 in bf16. Against -1 the sum of the rounded
    # values is 2^-7, and of the unrounded ones 3 * 2^-9; both are exact in bf16.
    p = jnp.asarray([[1 + 3 * 2.0 ** -9] * 3, [-1.0] * 3], jnp.float32)
    got = jax.jit(_sum_of_quantized)(p)
    assert jnp.dtype(got.dtype) == jnp.dtype(jnp.bfloat16)
    assert [float(x) for x in got] == [2.0 ** -7] * 3, [float(x) for x in got]
