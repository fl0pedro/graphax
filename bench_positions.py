"""Sweep: place each approximation transform at each of 6 structural positions
(matmul pre/post/out, add pre/post/out) plus the historical final placement,
and measure flops (cost_analysis) + peak memory (memory_analysis) + accuracy
(cos vs jax.jacrev). Same order, no seed vertices. Position is selected via the
GRAPHAX_APPROX_POS env var read inside core._eliminate_vertex.

Usage: python bench_positions.py [model ...]   (default: all)
"""
import os, sys
import jax, jax.numpy as jnp, jax.random as jr, jax.tree_util as jtu
from graphax import jacve
from graphax.sparse.micro_actions import Quant, Compress, Diag

POSITIONS = [None, "matmul_pre", "matmul_post", "matmul_out",
             "add_pre", "add_post", "add_out"]
TRANSFORMS = [
    ("Quant(bf16)", Quant("bfloat16")),
    ("Compress(0)", Compress((0,), "mean")),
    ("Compress(1)", Compress((1,), "mean")),
    ("Diag(0,1,2)", Diag(0, 1, 2)),
]


def measure(fn, args):
    try:
        c = jax.jit(fn).lower(*args).compile()
        return (float(c.cost_analysis().get("flops", float("nan"))),
                int(c.memory_analysis().peak_memory_in_bytes), None)
    except Exception as e:
        return None, None, type(e).__name__ + ": " + str(e)[:46]


def cos(a, b):
    a = jnp.concatenate([jnp.ravel(x) for x in jtu.tree_leaves(a)])
    b = jnp.concatenate([jnp.ravel(x) for x in jtu.tree_leaves(b)])
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30))


def variant(pos, fn, order, argnums, tr, args, ref):
    if pos is None:
        os.environ.pop("GRAPHAX_APPROX_POS", None)
    else:
        os.environ["GRAPHAX_APPROX_POS"] = pos
    g = jacve(fn, order, argnums, transforms=tr)
    fl, pk, err = measure(g, args)
    if err:
        return None, None, None, err
    try:
        c = cos(g(*args), ref)
    except Exception:
        c = float("nan")
    return fl, pk, c, None


def model_matmul_chain():
    k = jr.PRNGKey(0)
    x = jr.normal(k, (64,)); W = jr.normal(k, (48, 64)); V = jr.normal(k, (32, 48))
    return (lambda x, W, V: V @ jnp.tanh(W @ x)), (0, 1, 2), (x, W, V), "rev"


def model_mlp():
    k = jr.PRNGKey(0)
    W1 = jr.normal(k, (64, 100)); b1 = jr.normal(k, (64,))
    W2 = jr.normal(k, (32, 64)); b2 = jr.normal(k, (32,)); xi = jr.normal(k, (100,))
    return ((lambda x, W1, b1, W2, b2: jnp.tanh(W2 @ jnp.tanh(W1 @ x + b1) + b2)),
            (0, 1, 2, 3, 4), (xi, W1, b1, W2, b2), "rev")


def _vision(nm):
    from graphax.examples import vision
    k = jr.PRNGKey(0); bs = 8
    xb = jr.normal(k, (bs, 784)); yb = jr.normal(jr.PRNGKey(1), (bs, 10))
    w = getattr(vision, {"ConvNet": "conv_weights", "MoE": "moe_weights",
                         "ViT": "vit_weights"}[nm])(k)
    fnn = getattr(vision, nm)
    fn = (lambda *ww, _f=fnn: jax.vmap(lambda a, b: _f(a, b, *ww),
                                       in_axes=(0, 0))(xb, yb))
    return fn, tuple(range(len(w))), w, "rev"


MODELS = {
    "matmul-chain": model_matmul_chain,
    "MLP": model_mlp,
    "ConvNet": lambda: _vision("ConvNet"),
    "MoE": lambda: _vision("MoE"),
    "ViT": lambda: _vision("ViT"),
}


def run(names):
    for name in names:
        fn, argnums, args, order = MODELS[name]()
        nv = len(jax.make_jaxpr(fn)(*args).jaxpr.eqns)
        ref = jax.jacrev(fn, argnums)(*args)
        bf, bp, _, _ = variant(None, fn, order, argnums, (), args, ref)
        print(f"#### {name}  (nv={nv}, order={order})  "
              f"BASELINE flops={bf:.0f} peak={bp}")
        for tname, T in TRANSFORMS:
            tr = [(v, [T]) for v in range(1, nv + 1)]
            print(f"  -- {tname} --")
            for pos in POSITIONS:
                fl, pk, c, err = variant(pos, fn, order, argnums, tr, args, ref)
                tag = pos if pos else "final(edge)"
                if err:
                    print(f"     {tag:12} ERR {err}")
                else:
                    df = (bf - fl) / bf * 100 if bf else 0.0
                    dp = (bp - pk) / bp * 100 if bp else 0.0
                    print(f"     {tag:12} flops={fl:>12.0f} ({df:+6.1f}%)  "
                          f"peak={pk:>10d} ({dp:+6.1f}%)  cos={c:.3f}")
        print()


if __name__ == "__main__":
    run(sys.argv[1:] or list(MODELS))
