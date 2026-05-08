"""Regression tests for ``ops.block_storage`` JAX-safety and large-M paths.

Two bugs covered:

1. ``UnionBlocks.from_combined`` used ``int(jnp.prod(jnp.array(lhs_shape)))``
   to recover the lhs split offset. ``lhs_shape`` is a static Python tuple
   carried in ``meta``, so wrapping it in ``jnp.array`` uselessly traced it,
   and the surrounding ``int(...)`` raised ``ConcretizationTypeError`` the
   moment the call sat inside ``@jax.jit``. The fix is ``math.prod(lhs_shape)``.

2. ``BlockBanded.to_dense`` builds an intermediate of static shape
   ``(M, M, W, B, B, *L)``. For ``M`` in the thousands this exceeds practical
   buffer limits at trace time even though XLA fuses the where+sum at runtime.
   The fix is a per-band ``lax.fori_loop`` fallback that writes blocks via
   ``dynamic_update_slice`` once the broadcast intermediate exceeds
   ``_BLOCK_BANDED_BROADCAST_LIMIT``. This test exercises the fallback path
   on a synthetic shape that crosses the threshold and confirms the result
   matches a direct reference build.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from graphax.sparse.ops.block_storage import (
    BlockBanded,
    UnionBlocks,
    _BLOCK_BANDED_BROADCAST_LIMIT,
)


def _n(shape, key_idx, dtype=jnp.float32):
    return jr.normal(jr.PRNGKey(key_idx), shape).astype(dtype)


def test_union_from_combined_inside_jit():
    """``UnionBlocks.from_combined`` must work under ``@jax.jit``.

    Pre-fix this raised ``ConcretizationTypeError`` because the static-tuple
    ``lhs_shape`` was being wrapped in ``jnp.array`` then ``int(...)``-cast.
    """
    lhs = _n((2, 3, 5, 5), 1)
    rhs = _n((2, 5, 3, 3), 2)
    fl = jnp.array(0.0, dtype=jnp.float32)
    fr = jnp.array(0.0, dtype=jnp.float32)
    ub = UnionBlocks(lhs, rhs, fl, fr, op=jnp.add)
    flat, meta = ub.combined()

    @jax.jit
    def _round_trip(flat_):
        ub2 = UnionBlocks.from_combined(flat_, meta, fl, fr, op=jnp.add)
        return ub2.to_dense()

    got = _round_trip(flat)
    expected = ub.to_dense()
    assert got.shape == expected.shape
    assert jnp.allclose(got, expected, atol=1e-5)


def test_block_banded_large_m_uses_fallback():
    """``BlockBanded.to_dense`` on a shape that overflows the broadcast limit
    must take the per-band ``fori_loop`` fallback and still equal the
    reference build.

    We pick a shape whose ``M*M*W*B*B`` is comfortably above
    ``_BLOCK_BANDED_BROADCAST_LIMIT`` so the dispatch hits the fallback
    deterministically. The reference is a hand-rolled python build of the
    band layout — no ``BlockBanded`` involved.
    """
    M, w, B = 512, 1, 64
    W = 2 * w + 1
    # Sanity-check the threshold is actually crossed for this shape.
    assert M * M * W * B * B > _BLOCK_BANDED_BROADCAST_LIMIT

    data = _n((M, W, B, B), 7)
    fill = jnp.array(0.0, dtype=jnp.float32)
    bb = BlockBanded(data=data, fill_value=fill)

    got = bb.to_dense()
    assert got.shape == (M * B, M * B)

    # Spot-check: pick a few representative meta-positions and verify they
    # agree with the band-storage convention. A full dense reference would
    # itself OOM at this size.
    def _ref_block(k, b):
        col = k + b - w
        if 0 <= col < M:
            return data[k, b]
        return jnp.full((B, B), fill, dtype=data.dtype)

    for k, b in [(0, w), (0, w + 1), (M - 1, w - 1), (M - 1, w), (M // 2, w)]:
        col = k + b - w
        if 0 <= col < M:
            block = got[k * B : (k + 1) * B, col * B : (col + 1) * B]
            assert jnp.allclose(block, _ref_block(k, b), atol=1e-5), (
                f"band block mismatch at (k={k}, b={b})"
            )
    # Off-band positions must be ``fill_value`` (sample a few far-from-band cells).
    for k in (0, M // 3, M - 1):
        far_col = (k + w + 5) % M  # well outside the band
        block = got[k * B : (k + 1) * B, far_col * B : (far_col + 1) * B]
        assert jnp.allclose(block, fill, atol=1e-5)


def test_block_banded_fallback_matches_broadcast_path():
    """Cross-check fallback vs broadcast path on a smaller-but-equivalent shape.

    For correctness we want the fallback to produce the *same* output as the
    broadcast path, not just a plausible one. The broadcast path is too
    expensive at the threshold-crossing shape, so we use a small ``M`` where
    both paths are tractable, force the fallback by temporarily lowering the
    module-level threshold, and compare.
    """
    import graphax.sparse.ops.block_storage as bs

    M, w, B = 8, 1, 4
    W = 2 * w + 1
    data = _n((M, W, B, B), 11)
    fill = jnp.array(0.0, dtype=jnp.float32)
    bb = BlockBanded(data=data, fill_value=fill)

    broadcast_out = bb.to_dense()

    saved = bs._BLOCK_BANDED_BROADCAST_LIMIT
    try:
        bs._BLOCK_BANDED_BROADCAST_LIMIT = 0
        fallback_out = bb.to_dense()
    finally:
        bs._BLOCK_BANDED_BROADCAST_LIMIT = saved

    assert fallback_out.shape == broadcast_out.shape
    assert jnp.allclose(fallback_out, broadcast_out, atol=1e-5), (
        f"max diff {float(jnp.max(jnp.abs(fallback_out - broadcast_out))):.3e}"
    )


def test_block_banded_fallback_with_leftover_dims():
    """Fallback must thread leftover ``L`` axes through ``dynamic_update_slice``."""
    import graphax.sparse.ops.block_storage as bs

    M, w, B = 6, 2, 3
    W = 2 * w + 1
    L = (4,)
    data = _n((M, W, B, B, *L), 13)
    fill = jnp.array(0.0, dtype=jnp.float32)
    bb = BlockBanded(data=data, fill_value=fill)

    broadcast_out = bb.to_dense()

    saved = bs._BLOCK_BANDED_BROADCAST_LIMIT
    try:
        bs._BLOCK_BANDED_BROADCAST_LIMIT = 0
        fallback_out = bb.to_dense()
    finally:
        bs._BLOCK_BANDED_BROADCAST_LIMIT = saved

    assert fallback_out.shape == broadcast_out.shape
    assert jnp.allclose(fallback_out, broadcast_out, atol=1e-5)
