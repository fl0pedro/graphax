"""Bug: `_sort_val` crashed when `val` was a Python float.

Some elemental rules return Python floats — e.g. `lax.neg_p` returns
`lambda x: -1.0` and `lax.sub_p` returns `(1.0, -1.0)`. `make_parallel_jacobian`
in the singleton path does `SparseTensor([], [], elemental)` where
`elemental` is the bare float, and the SparseTensor constructor calls
`_sort_val` which accesses `val.ndim` -> `AttributeError`.

Fix: in `_sort_val`, wrap non-array vals via `jnp.asarray(val)` first.
"""

import jax.numpy as jnp

from graphax.sparse.tensor import SparseTensor


def test_sparse_tensor_accepts_python_float_val():
    st = SparseTensor([], [], 1.0)
    assert st.val is not None
    assert float(st.val) == 1.0


def test_sparse_tensor_accepts_python_int_val():
    st = SparseTensor([], [], 7)
    assert st.val is not None
    assert int(st.val) == 7


def test_jacve_through_neg_does_not_crash():
    """`neg` returns the float -1.0 as elemental — exercises the singleton path."""
    import jax

    from graphax import jacve, tree_allclose

    def f(x):
        return jnp.sum(-x)

    x = jnp.ones(4)
    veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
    refres = jax.jit(jax.jacrev(f, argnums=(0,)))(x)
    assert bool(tree_allclose(veres, refres))
