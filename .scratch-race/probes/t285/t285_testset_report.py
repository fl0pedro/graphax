"""Aggregate the test-set census into the tables of finding 64 section (a)."""
import json
import sys
from collections import defaultdict

CASES = [
    "single_implicit_sparse", "single_implicit_dense_block",
    "single_implicit_dense_contracted", "single_implicit_dense_carried",
    "double_implicit_batch", "double_implicit_contracted",
    "no_implicit_axis", "lcm_grid", "spatial_sparse", "partially_stored",
]


def main():
    rows = json.load(open(sys.argv[1]))
    per_file = defaultdict(lambda: defaultdict(int))
    per_file_tot = defaultdict(lambda: [0, 0, 0, 0])  # tests, contractions, calls, elements
    hits = defaultdict(list)
    for r in rows:
        f = r["file"]
        t = per_file_tot[f]
        t[0] += 1
        t[1] += r["contractions"]
        t[2] += r["growing_calls"]
        t[3] += r["growing_elements"]
        for c in r["cases"]:
            per_file[f][c] += 1
            hits[c].append(r["nodeid"])

    print("## Which case does each module reach\n")
    print("| set | module | tests | contractions | growing calls | growing elements | cases reached |")
    print("|---|---|---|---|---|---|---|")
    seen = []
    for r in rows:
        if r["file"] not in seen:
            seen.append(r["file"])
    for f in seen:
        t = per_file_tot[f]
        s = next(r["set"] for r in rows if r["file"] == f)
        cs = ", ".join(f"{c} ({n})" for c, n in sorted(per_file[f].items()))
        print(f"| {s} | {f} | {t[0]} | {t[1]} | {t[2]} | {t[3]} | {cs or '-'} |")

    print("\n## The three cases finding 63 could not reach\n")
    for c in ("lcm_grid", "spatial_sparse", "partially_stored"):
        n = hits.get(c, [])
        print(f"\n### {c}: {len(n)} tests")
        for x in n[:40]:
            print("  -", x)

    print("\n## Every case, tests that reach it\n")
    print("| case | tests |")
    print("|---|---|")
    for c in CASES:
        print(f"| {c} | {len(hits.get(c, []))} |")


if __name__ == "__main__":
    main()
