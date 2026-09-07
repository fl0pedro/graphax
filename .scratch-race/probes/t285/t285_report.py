"""Turn the census JSON into the markdown tables of finding 64."""
import json
import re
import sys

MODES = ["legacy", "lazy_off", "lazy_nodemote", "lazy_full", "planner"]
SHORT = {"legacy": "incumbent", "lazy_off": "lazy off",
         "lazy_nodemote": "lazy nodemote", "lazy_full": "lazy full",
         "planner": "planner"}


def load(path):
    d = json.load(open(path))
    return {(r["case"], r["mode"]): r for r in d}, [r["case"] for r in d]


def dot_macs(shapes):
    """A dot's shape string like f32[4,3,5] does not carry the contracted
    extent, so this is the OUTPUT element count, not the product count."""
    tot = 0
    for sh in shapes:
        m = re.search(r"\[([\d,]*)\]", sh)
        if not m:
            continue
        dims = [int(x) for x in m.group(1).split(",") if x]
        p = 1
        for x in dims:
            p *= x
        tot += p
    return tot


def axis_summary(r):
    ax = r.get("growing_axes") or []
    if not ax:
        return "-"
    parts = {}
    for a in ax:
        k = f"{a['side']}.{a['slot']}[{a['pairing_type']}]"
        parts[k] = parts.get(k, 1) * a["factor"]
    return " ".join(f"{k}x{v}" for k, v in parts.items())


def implicit_pattern(r):
    d = r.get("dims")
    if not d:
        return "-"
    out = []
    for x in d:
        if x["sparse"]:
            out.append("S" if x["axis"] is not None else "s")
        elif x["axis"] is None and x["logical"] > 1:
            out.append("I")
        else:
            out.append("D")
    return "".join(out)


def main():
    path = sys.argv[1]
    rec, cases = load(path)
    seen = []
    for c in cases:
        if c not in seen:
            seen.append(c)

    print("## Table A — stored elements, growing broadcasts, dots\n")
    print("| case | optimum stored | mode | stored | val.shape | growing calls/elements | jaxpr growing | dots | temp B | dims |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in seen:
        opt = rec[(c, "legacy")].get("optimum", ["?", "?"])[0]
        for m in MODES:
            r = rec.get((c, m))
            if r is None:
                continue
            if "error" in r:
                print(f"| {c} | {opt} | {SHORT[m]} | RAISED | | | | | | {r['error'][:60]} |")
                continue
            h = r.get("hlo", {})
            print(
                f"| {c} | {opt} | {SHORT[m]} | {r['stored']} | "
                f"{tuple(r['val_shape']) if r['val_shape'] else 'None'} | "
                f"{r['growing_calls']}/{r['growing_elements']} | "
                f"{r.get('jaxpr_growing','?')}/{r.get('jaxpr_grown_elements','?')} | "
                f"{len(h.get('dot', []))} | {r.get('temp_bytes','?')} | "
                f"{implicit_pattern(r)} |"
            )
    print()
    print("## Table B — which axis grew\n")
    print("| case | mode | grown axes (side.slot[pairing] x factor) | raw before -> target |")
    print("|---|---|---|---|")
    for c in seen:
        for m in MODES:
            r = rec.get((c, m))
            if r is None or "error" in r or not r.get("growing_calls"):
                continue
            raw = "; ".join(
                f"{tuple(g['before'])}->{tuple(g['target'])}" for g in r["growing_raw"]
            )
            print(f"| {c} | {SHORT[m]} | {axis_summary(r)} | {raw} |")
    print()
    print("## Table C — CPU HLO kernel census\n")
    print("| case | mode | dot shapes | reduce | copy | transpose | top-level broadcast | fusion kinds |")
    print("|---|---|---|---|---|---|---|---|")
    for c in seen:
        for m in MODES:
            r = rec.get((c, m))
            if r is None or "hlo" not in r:
                continue
            h = r["hlo"]
            print(
                f"| {c} | {SHORT[m]} | {' '.join(h['dot']) or '-'} | {h['reduce']} | "
                f"{h['copy']} | {h['transpose']} | "
                f"{' '.join(h['top_broadcast']) or '-'} | {h['fusion_kinds'] or '-'} |"
            )
    print()
    print("## Table D — the frame decision per pair\n")
    print("| case | mode | pairing | ol/orr | T/G | aligned | m_l/m_r | meta_lazy | lhs_lazy | rhs_lazy | demote |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in seen:
        for m in MODES:
            r = rec.get((c, m))
            if r is None or not r.get("frames"):
                continue
            for f in r["frames"][0]:
                print(
                    f"| {c} | {SHORT[m]} | {f['pairing_type']} | {f['ol']}/{f['orr']} | "
                    f"{f['T']}/{f['G']} | {f['aligned']} | {f['m_l']}/{f['m_r']} | "
                    f"{f['meta_lazy']} | {f['lhs_lazy']} | {f['rhs_lazy']} | {f['demote']} |"
                )
    print()
    print("## Table E — values against the dense oracle\n")
    print("| case | mode | max abs err | oracle scale |")
    print("|---|---|---|---|")
    for c in seen:
        for m in MODES:
            r = rec.get((c, m))
            if r is None:
                continue
            print(f"| {c} | {SHORT[m]} | {r.get('value_max_abs_err', r.get('value_error','-'))} "
                  f"| {r.get('value_scale','-')} |")


if __name__ == "__main__":
    main()
