import jax
import jax.numpy as jnp
from graphax.core import vertex_elimination_jaxpr, jacve
import numpy as np

def measure_orders(fun, *args, argnums=(0,)):
    """
    Measures the computational and memory costs of forward and reverse mode 
    elimination orders for a given function.
    """
    # Extract jaxpr and constants
    closed_jaxpr = jax.make_jaxpr(fun)(*args)
    jaxpr = closed_jaxpr.jaxpr
    consts = closed_jaxpr.literals

    results = {}

    for order_str in ["fwd", "rev"]:
        # 1. Analytical measurement via Graphax
        # This uses Graphax's internal counting mechanism during vertex elimination.
        _, aux = vertex_elimination_jaxpr(
            jaxpr,
            order_str,
            consts,
            *args,
            argnums=argnums,
            count_ops=True
        )
        analytical_cmp = aux["adds"] + aux["muls"] + aux["fmas"]
        analytical_mem = aux["mem"]

        # 2. Estimated measurement via XLA (JIT-compiled)
        # This uses the JIT-compiled executable's cost analysis.
        jac_fun = jacve(fun, order_str, argnums=argnums)
        lowered = jax.jit(jac_fun).lower(*args)
        compiled = lowered.compile()
        cost = compiled.cost_analysis()
        
        estimated_flops = cost.get("flops", 0)
        estimated_mem = cost.get("bytes accessed", 0)

        results[order_str] = {
            "analytical": {"cmp": analytical_cmp, "mem": analytical_mem},
            "estimated": {"flops": estimated_flops, "mem": estimated_mem}
        }

    # 3. Standard JAX measurements
    for name, jac_fn_gen in [("jax_fwd", jax.jacfwd), ("jax_rev", jax.jacrev)]:
        jac_fun = jac_fn_gen(fun, argnums=argnums)
        lowered = jax.jit(jac_fun).lower(*args)
        compiled = lowered.compile()
        cost = compiled.cost_analysis()
        
        results[name] = {
            "estimated": {"flops": cost.get("flops", 0), "mem": cost.get("bytes accessed", 0)}
        }

    return results

if __name__ == "__main__":
    # Example: VmappedNeuralNetwork
    def neural_network(x, y, W1, b1, W2, b2):
        a1 = jnp.tanh(x @ W1.T + b1)
        return 0.5 * (jnp.tanh(a1 @ W2.T + b2) - y) ** 2

    # Vmap over the first two arguments (x and y)
    vmapped_nn = jax.vmap(neural_network, in_axes=(0, 0, None, None, None, None))

    # Setup inputs based on shapes from scratch/inspect_jaxpr.py
    # shapes = [(16, 4), (16, 4), (8, 4), (8,), (4, 8), (4,)]
    batch_size = 16
    in_dim = 4
    hidden_dim = 8
    out_dim = 4

    x_in = jnp.ones((batch_size, in_dim))
    y_in = jnp.ones((batch_size, out_dim))
    W1_val = jnp.ones((hidden_dim, in_dim))
    b1_val = jnp.ones((hidden_dim,))
    W2_val = jnp.ones((out_dim, hidden_dim))
    b2_val = jnp.ones((out_dim,))
    
    args = (x_in, y_in, W1_val, b1_val, W2_val, b2_val)
    
    print("Measuring costs for VmappedNeuralNetwork (forward vs reverse mode)...")
    # Differentiate with respect to parameters W1, b1, W2, b2 (argnums 2, 3, 4, 5)
    
    # Check for GPUs
    try:
        has_gpu = len(jax.devices("gpu")) > 0
    except (RuntimeError, ValueError):
        has_gpu = False
    
    platforms = ["cpu"]
    if has_gpu:
        platforms.append("gpu")
    
    for platform in platforms:
        device = jax.devices(platform)[0]
        platform_label = f" ({platform.upper()})" if has_gpu else ""
        
        # Ensure arguments are on the correct device
        device_args = jax.device_put(args, device)
        
        with jax.default_device(device):
            results = measure_orders(vmapped_nn, *device_args, argnums=(2, 3, 4, 5))
        
        for name, data in results.items():
            if name == "fwd":
                mode = "Forward Mode Elimination"
            elif name == "rev":
                mode = "Reverse Mode Elimination"
            elif name == "jax_fwd":
                mode = "JAX Forward"
            elif name == "jax_rev":
                mode = "JAX Reverse"
            else:
                mode = name
                
            print(f"\n[{mode}]{platform_label}")
            if "analytical" in data:
                print(f"  Analytical (Graphax): ops={data['analytical']['cmp']}, mem={data['analytical']['mem']} bytes")
            print(f"  Estimated  (XLA):     flops={data['estimated']['flops']}, mem={data['estimated']['mem']} bytes")
