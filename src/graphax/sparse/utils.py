import jax.numpy as jnp

from jax._src.core import ShapedArray


def zeros_like(invar: ShapedArray, outvar: ShapedArray) -> jnp.ndarray:
    """
    Function that creates an array of zeros. The shape of the array is the
    concatenation of the shapes of the input and output dimensions.

    Args:
        invar (ShapedArray): The input variable.
        outvar (ShapedArray): The output variable.

    Returns:
        jnp.ndarray: An array of zeros with the shape of the concatenation of the
                    shapes of the input and output dimensions.
    """
    in_shape = invar.aval.shape
    out_shape = outvar.aval.shape

    if in_shape == () and out_shape == ():
        return 0.
    else:
        shape = (*in_shape, *out_shape)
        return jnp.zeros(shape)
