import jax.numpy as jnp
import jax.tree_util as jtu


def tree_allclose(
    tree1, tree2, equal_nan: bool = False, atol: float = 1e-5, rtol: float = 1e-4
) -> bool:
    allclose = lambda a, b: jnp.allclose(
        a, b, equal_nan=equal_nan, atol=atol, rtol=rtol
    )
    is_equal = jtu.tree_map(allclose, tree1, tree2)
    return jtu.tree_reduce(jnp.logical_and, is_equal)
