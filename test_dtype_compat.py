import jax
import jax.numpy as jnp
from graphax.sparse.micro_actions import QUANT_DTYPES

def test_compat():
    print(f"Device: {jax.devices()[0].device_kind}")
    
    available_dtypes = []
    
    print("\n--- DType Availability ---")
    for dtype in QUANT_DTYPES:
        try:
            name = jnp.dtype(dtype).name
            _ = jnp.zeros((2, 2), dtype=dtype)
            available_dtypes.append(dtype)
            print(f"{name}: AVAILABLE")
        except Exception as e:
            try:
                name = dtype.__name__
            except:
                name = str(dtype)
            print(f"{name}: UNAVAILABLE ({type(e).__name__})")
            
    print("\n--- Dot Product Compatibility ---")
    names = [jnp.dtype(d).name for d in available_dtypes]
    print("," + ",".join(names))
    
    for d1 in available_dtypes:
        n1 = jnp.dtype(d1).name
        row = [n1]
        a = jnp.ones((2, 2), dtype=d1)
        for d2 in available_dtypes:
            n2 = jnp.dtype(d2).name
            b = jnp.ones((2, 2), dtype=d2)
            try:
                _ = jnp.dot(a, b).block_until_ready()
                row.append("OK")
            except Exception:
                row.append("FAIL")
        print(",".join(row))

if __name__ == "__main__":
    test_compat()
