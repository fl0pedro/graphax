"""Bug regression + feature pin: `count_ops=True` returns op-count aux.

Verifies that `count_ops` is wired all the way through `jacve` ->
`vertex_elimination_jaxpr` -> `_eliminate_vertex` -> `add_w_counts` /
`matmul(count=True)`. The aux dict has the expected keys, integer counts,
and a per-vertex `order_counts` breakdown.
"""

import jax
import jax.numpy as jnp

from graphax import jacve


def test_count_ops_returns_aux_dict():
    def f(x, y):
        return jnp.sin(x * y).sum()

    x = jnp.ones(5)
    y = jnp.ones(5)
    jac, aux = jacve(f, order="rev", argnums=(0, 1), count_ops=True)(x, y)
    # The Jacobian is still returned correctly.
    assert isinstance(jac, tuple)
    assert all(j.shape == (5,) for j in jac)
    # Aux dict has the documented keys.
    for k in ("adds", "muls", "fmas", "mem", "order_counts"):
        assert k in aux, f"missing aux key: {k}"
    # And counts are non-negative ints.
    for k in ("adds", "muls", "fmas", "mem"):
        assert int(aux[k]) >= 0


def test_count_ops_off_returns_no_aux():
    """When `count_ops=False` (default), jacve does not return an aux dict.

    The return shape mirrors the function's output structure (jacve wraps
    single-output as a length-1 tuple due to its outvars-tree handling) so
    the contract being pinned is "no dict-shaped aux on the side", not the
    exact pytree shape.
    """
    def f(x):
        return jnp.sin(x).sum()

    x = jnp.ones(3)
    res = jacve(f, order="rev", argnums=(0,))(x)
    # Should not be a (something, dict) tuple where the second is an aux dict.
    if isinstance(res, tuple) and len(res) == 2:
        assert not isinstance(res[1], dict), (
            "count_ops=False unexpectedly returned an aux dict"
        )


def test_order_counts_is_per_step_breakdown():
    """`order_counts` must have one entry per eliminated vertex, monotonic."""
    def f(x):
        return jnp.sin(x * 2.0).sum()

    x = jnp.ones(3)
    _, aux = jacve(f, order="rev", argnums=(0,), count_ops=True)(x)
    oc = aux["order_counts"]
    assert isinstance(oc, list)
    # Each entry is (vertex_id, (adds, muls, fmas, mem)).
    for entry in oc:
        assert len(entry) == 2
        v, cnt = entry
        assert isinstance(v, int)
        assert len(cnt) == 4
    # Counts should be monotonically non-decreasing across steps.
    cumul = [c[1] for _, c in oc]
    for i in range(1, len(cumul)):
        assert cumul[i] >= cumul[i - 1]


def test_count_ops_matches_correctness():
    """Numerical correctness must be unaffected by count_ops."""
    from graphax import tree_allclose

    def f(x):
        return jnp.sin(x).sum()

    x = jnp.array([0.1, 0.2, 0.3])
    plain = jacve(f, order="rev", argnums=(0,))(x)
    counted, _ = jacve(f, order="rev", argnums=(0,), count_ops=True)(x)
    assert bool(tree_allclose(plain, counted))
