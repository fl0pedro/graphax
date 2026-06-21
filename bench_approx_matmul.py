"""A/B: approximation on the OUTPUT edge (OLD) vs on the MATMUL (NEW,
GRAPHAX_APPROX_MATMUL=1 — Diag/Compress on pre_val, Quant on the contraction
operands). Same elimination order, same approximation sequence, no seed
vertices. Metrics: cost_analysis()['flops'] and memory_analysis().peak_memory.
"""
import os
import jax, jax.numpy as jnp, jax.random as jr, jax.tree_util as jtu
from graphax import jacve
from graphax.sparse.micro_actions import Quant, Compress, Diag


def measure(fn, args):
    try:
        c = jax.jit(fn).lower(*args).compile()
        return (float(c.cost_analysis().get("flops", float("nan"))),
                int(c.memory_analysis().peak_memory_in_bytes), None)
    except Exception as e:
        return None, None, type(e).__name__ + ": " + str(e)[:50]


def cos(a, b):
    a = jnp.concatenate([jnp.ravel(x) for x in jtu.tree_leaves(a)])
    b = jnp.concatenate([jnp.ravel(x) for x in jtu.tree_leaves(b)])
    return float(a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-30))


def variant(flag, fn, order, argnums, transforms, args, ref):
    if flag:
        os.environ["GRAPHAX_APPROX_MATMUL"] = "1"
    else:
        os.environ.pop("GRAPHAX_APPROX_MATMUL", None)
    g = jacve(fn, order, argnums, transforms=transforms)
    fl, pk, err = measure(g, args)
    if err:
        return None, None, None, err
    try:
        c = cos(g(*args), ref)
    except Exception:
        c = float("nan")
    return fl, pk, c, None


def bench(name, fn, argnums, args, order, seqs):
    nv = len(jax.make_jaxpr(fn)(*args).jaxpr.eqns)
    ref = jax.jacrev(fn, argnums)(*args)
    print(f"#### {name}  (nv={nv}, order={order})")
    bf, bp, _, berr = variant(False, fn, order, argnums, (), args, ref)
    print(f"   baseline (no approx)   flops={bf:>12.0f}  peak={bp:>10d} B")
    for label, seq in seqs:
        tr = [(v, list(seq)) for v in range(1, nv + 1)]
        of, op, oc, oerr = variant(False, fn, order, argnums, tr, args, ref)
        nf, npk, nc, nerr = variant(True, fn, order, argnums, tr, args, ref)
        if oerr or nerr:
            print(f"   [{label}] OLD={'ERR '+oerr if oerr else 'ok'} | "
                  f"NEW={'ERR '+nerr if nerr else 'ok'}")
            continue
        df = (of - nf) / of * 100 if of else 0.0
        dp = (op - npk) / op * 100 if op else 0.0
        print(f"   [{label:16}] OLD flops={of:>11.0f} peak={op:>9d}  | "
              f"NEW flops={nf:>11.0f} peak={npk:>9d}  | "
              f"Δflops {df:+6.1f}% Δpeak {dp:+6.1f}%  cos={nc:.3f}")
    print()


SEQS = [
    ("Quant(bf16)", [Quant("bfloat16")]),
    ("Compress(0)", [Compress((0,), "mean")]),
    ("Compress(1)", [Compress((1,), "mean")]),
    ("Diag(0,1,2)", [Diag(0, 1, 2)]),
    ("Diag+Compress", [Diag(0, 1, 2), Compress((0,), "mean")]),
]


def main():
    k = jr.PRNGKey(0)
    x = jr.normal(k, (64,)); W = jr.normal(k, (48, 64)); V = jr.normal(k, (32, 48))
    bench("matmul-chain V@tanh(W@x)", lambda x, W, V: V @ jnp.tanh(W @ x),
          (0, 1, 2), (x, W, V), "rev", SEQS)
    W1 = jr.normal(k, (64, 100)); b1 = jr.normal(k, (64,))
    W2 = jr.normal(k, (32, 64)); b2 = jr.normal(k, (32,)); xi = jr.normal(k, (100,))
    bench("MLP", lambda x, W1, b1, W2, b2: jnp.tanh(W2 @ jnp.tanh(W1 @ x + b1) + b2),
          (0, 1, 2, 3, 4), (xi, W1, b1, W2, b2), "rev", SEQS)
    try:
        from graphax.examples import vision
        bs = 8
        xb = jr.normal(k, (bs, 784)); yb = jr.normal(jr.PRNGKey(1), (bs, 10))
        for nm in ["ConvNet", "MoE", "ViT"]:
            w = getattr(vision, {"ConvNet": "conv_weights", "MoE": "moe_weights",
                                 "ViT": "vit_weights"}[nm])(k)
            fnn = getattr(vision, nm)
            fn = (lambda *ww, _f=fnn: jax.vmap(lambda a, b: _f(a, b, *ww),
                                               in_axes=(0, 0))(xb, yb))
            bench(f"vmap {nm}", fn, tuple(range(len(w))), w, "rev", SEQS)
    except Exception as e:
        print("vision skipped:", type(e).__name__, str(e)[:80])


if __name__ == "__main__":
    main()
