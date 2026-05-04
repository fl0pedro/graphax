"""Bug: transforms used the obsolete `lax.ScatterIndexNumbers`.

JAX renamed `ScatterIndexNumbers` -> `ScatterDimensionNumbers`. A few transforms
in `transforms.py` and `auto.py` (concatenate/inverse_concatenate, slice
inverse) hadn't been updated, raising

    AttributeError: module 'jax.lax' has no attribute 'ScatterIndexNumbers'.

Pin: a `lax.ScatterDimensionNumbers` exists and the obsolete name doesn't
appear in any of our transform sources.
"""

from pathlib import Path

import jax.lax as lax


def test_jax_lax_has_scatter_dimension_numbers():
    """Sanity: the new name exists in the JAX we depend on."""
    assert hasattr(lax, "ScatterDimensionNumbers")


def test_no_obsolete_scatter_index_numbers_in_transforms():
    """grep our transform code for the obsolete name."""
    src_root = Path(__file__).resolve().parents[2] / "src" / "graphax"
    bad = []
    for path in (
        src_root / "primitives" / "transforms.py",
        src_root / "primitives" / "auto.py",
    ):
        if path.exists():
            text = path.read_text()
            if "ScatterIndexNumbers" in text:
                bad.append(str(path))
    assert not bad, f"Obsolete ScatterIndexNumbers in: {bad}"


def test_concatenate_with_complex_structure_via_jacve():
    """Exercise the concatenate transforms that use ScatterDimensionNumbers."""
    import jax

    from graphax import jacve, tree_allclose
    import jax.numpy as jnp

    def f(x, y):
        # Forces the concat's transform into the materialize-then-scatter path.
        return jnp.concatenate([jnp.sin(x), jnp.cos(y)], axis=0).sum()

    x = jnp.ones((3,))
    y = jnp.ones((2,))
    veres = jax.jit(jacve(f, order="rev", argnums=(0, 1)))(x, y)
    refres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)
    assert bool(tree_allclose(veres, refres))
