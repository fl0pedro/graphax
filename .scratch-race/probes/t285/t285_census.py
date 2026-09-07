"""Ticket dsnn-3qm.28.5 — run every case under every mode, dump JSON + tables.

Usage:  python t285_census.py [out.json]
"""
import json
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp

import t285_lib as L


def run_one(name, mode):
    builder = L.CASES[name]
    rec = dict(case=name, mode=mode)
    rec["optimum"] = list(L.OPTIMUM[name])
    with L.mode_env(mode):
        lhs, rhs, note = builder()
        rec["note"] = note
        # 1. eager run with the census installed
        with L.Census() as cen:
            try:
                res = lhs @ rhs
                jax.block_until_ready(res.val) if res.val is not None else None
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["traceback"] = traceback.format_exc()[-1500:]
                return rec
            rec["growing_calls"] = len(cen.growing)
            rec["growing_elements"] = sum(g["grew"] for g in cen.growing)
            rec["growing_axes"] = [a for g in cen.growing for a in g["axes"]]
            rec["growing_raw"] = [
                dict(before=g["before_shape"], target=g["target"]) for g in cen.growing
            ]
            rec["frames"] = cen.frames

        rec["val_shape"] = None if res.val is None else list(res.val.shape)
        rec["stored"] = 0 if res.val is None else int(res.val.size)
        rec["dims"] = L.dims_report(res)
        rec["logical"] = L.logical_size(res)
        rec["out_shape"] = list(res.out_shape)
        rec["primal_shape"] = list(res.primal_shape)
        rec["scalar_mult"] = float(res.scalar_mult) if res.scalar_mult is not None else None

        # 2. values against the dense oracle
        try:
            oracle = L.ORACLES[name](lhs.dense(), rhs.dense())
            got = res.dense()
            rec["oracle_shape"] = list(oracle.shape)
            rec["got_shape"] = list(got.shape)
            rec["value_max_abs_err"] = float(jnp.max(jnp.abs(got - oracle)))
            rec["value_scale"] = float(jnp.max(jnp.abs(oracle)))
        except Exception as e:
            rec["value_error"] = f"{type(e).__name__}: {e}"

        # 3. jaxpr
        def f(a, b):
            return (a @ b).val

        try:
            cj = jax.make_jaxpr(f)(lhs, rhs)
            n, grown, shapes = L.jaxpr_growing_broadcasts(cj)
            rec["jaxpr_growing"] = n
            rec["jaxpr_grown_elements"] = grown
            rec["jaxpr_growing_shapes"] = [[list(a), list(b)] for a, b in shapes]
            rec["jaxpr_eqns"] = len(cj.jaxpr.eqns)
        except Exception as e:
            rec["jaxpr_error"] = f"{type(e).__name__}: {e}"

        # 4. CPU HLO
        try:
            lowered = jax.jit(f).lower(lhs, rhs)
            comp = lowered.compile()
            text = comp.as_text()
            rec["hlo"] = L.hlo_census(text)
            rec["hlo_text_len"] = len(text)
            an = comp.memory_analysis()
            rec["temp_bytes"] = int(getattr(an, "temp_size_in_bytes", -1))
        except Exception as e:
            rec["hlo_error"] = f"{type(e).__name__}: {e}"
            text = ""
        rec["hlo_dump"] = text
    return rec


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "t285_census.json"
    dump_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), "hlo")
    os.makedirs(dump_dir, exist_ok=True)
    results = []
    for name in L.CASES:
        for mode in L.MODES:
            r = run_one(name, mode)
            text = r.pop("hlo_dump", "")
            if text:
                with open(os.path.join(dump_dir, f"{name}__{mode}.txt"), "w") as fh:
                    fh.write(text)
            results.append(r)
            tag = r.get("error", "")
            print(
                f"{name:34s} {mode:14s} "
                f"stored={r.get('stored','-'):>6} "
                f"logical={r.get('logical','-'):>7} "
                f"bcast={r.get('growing_calls','-')}/{r.get('growing_elements','-')} "
                f"jaxpr={r.get('jaxpr_growing','-')}/{r.get('jaxpr_grown_elements','-')} "
                f"dots={len(r.get('hlo',{}).get('dot',[]))} "
                f"err={r.get('value_max_abs_err','-')} {tag}",
                flush=True,
            )
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
