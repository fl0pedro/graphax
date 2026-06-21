"""A/B benchmark: approximation on the OUTPUT edge (current/OLD) vs on the
MATMUL (NEW, GRAPHAX_APPROX_MATMUL=1 — Diag/Compress on pre_val, Quant on the
contraction operands). Same elimination order, same approximation sequence, no
seed vertices. Metrics via JAX: cost_analysis()['flops'] and
memory_analysis().peak_memory_in_bytes.
"""
import os, sys
import jax, jax.numpy as jnp, jax.random as jr, jax.tree_util as jtu
from graphax import jacve
from graphax.sparse.micro_actions import Quant, Compress


def measure(fn, args):
    c = jax.jit(fn).lower(*args).compile()
    flops = float(c.cost_analysis().get("flops", float("nan")))
    peak = int(c.memory_analysis().peak_memory_in_bytes)
    return flops, peak


def cos(a, b):
    a = jnp.concatenate([jnp.ravel(x) for x in jtu.tree_leaves(a)])
    b = jnp.concatenate([jnp.ravel(x) for x in jtu.tree_leaves(b)])
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30))


def run_variant(flag, fn, order, argnums, transforms, args):
    if flag is None:
        os.environ.pop("GRAPHAX_APPROX_MATMUL", None)
    else:
        os.environ["GRAPHAX_APPROX_MATMUL"] = flag
    g = jacve(fn, order, argnums, transforms=transforms)
    flops, peak = measure(g, args)
    val = g(*args)            # eager, for the cos-vs-jax accuracy check
    return flops, peak, val


def bench(name, fn, argnums, args, order, dtype="bfloat16", compress=False):
    nv = len(jax.make_jaxpr(fn)(*args).jaxpr.eqns)
    # Same approximation sequence for both placements: Quant on every vertex,
    # optionally a Compress of the trailing physical axis too.
    seq = [Quant(dtype)] + ([Compress((1,), "mean")] if compress else [])
    transforms = [(v, list(seq)) for v in range(1, nv + 1)]

    ref = jax.jacrev(fn, argnums)(*args)
    base_f, base_p, _ = run_variant(None, fn, order, argnums, (), args)          # no approx
    old_f, old_p, old_v = run_variant(None, fn, order, argnums, transforms, args)  # approx on edge
    new_f, new_p, new_v = run_variant("1", fn, order, argnums, transforms, args)   # approx on matmul

    print(f"== {name}  (nv={nv}, order={order if isinstance(order,str) else 'perm'}, "
          f"seq={[type(t).__name__ for t in seq]}) ==")
    print(f"   baseline      flops={base_f:>12.0f}  peak={base_p:>10d} B")
    print(f"   OLD (edge)    flops={old_f:>12.0f}  peak={old_p:>10d} B  cos={cos(old_v,ref):.4f}")
    print(f"   NEW (matmul)  flops={new_f:>12.0f}  peak={new_p:>10d} B  cos={cos(new_v,ref):.4f}")
    df = (old_f - new_f) / old_f * 100 if old_f else 0.0
    dp = (old_p - new_p) / old_p * 100 if old_p else 0.0
    print(f"   IMPROVEMENT   flops {df:+6.1f}%   peak {dp:+6.1f}%   (NEW vs OLD)")
    print()


def main():
    k = jr.PRNGKey(0)
    # matmul chain
    x = jr.normal(k, (64,)); W = jr.normal(k, (48, 64)); V = jr.normal(k, (32, 48))
    bench("matmul-chain V@tanh(W@x)", lambda x, W, V: V @ jnp.tanh(W @ x),
          (0, 1, 2), (x, W, V), "rev")
    # MLP with bias
    W1 = jr.normal(k, (64, 100)); b1 = jr.normal(k, (64,))
    W2 = jr.normal(k, (32, 64)); b2 = jr.normal(k, (32,))
    xi = jr.normal(k, (100,))
    bench("MLP tanh(W2 tanh(W1 x+b1)+b2)",
          lambda x, W1, b1, W2, b2: jnp.tanh(W2 @ jnp.tanh(W1 @ x + b1) + b2),
          (0, 1, 2, 3, 4), (xi, W1, b1, W2, b2), "rev")
    # vision models (no seeds)
    try:
        from graphax.examples import vision
        bs = 8
        xb = jr.normal(k, (bs, 784)); yb = jr.normal(jr.PRNGKey(1), (bs, 10))
        for nm in ["ConvNet", "MoE", "ViT"]:
            w = getattr(vision, {"ConvNet": "conv_weights", "MoE": "moe_weights",
                                 "ViT": "vit_weights"}[nm])(k)
            fnn = getattr(vision, nm)
            fn = (lambda *ww, _f=fnn: jax.vmap(lambda xi, yi: _f(xi, yi, *ww),
                                               in_axes=(0, 0))(xb, yb))
            bench(f"vmap {nm}", fn, tuple(range(len(w))), w, "rev")
    except Exception as e:
        print("vision bench skipped:", type(e).__name__, str(e)[:80])


if __name__ == "__main__":
    main()
