"""Ticket dsnn-3qm.28.5 deliverable (a): which implicit-axis case does each test
of the three sets actually exercise?

Runs the three test sets under the instrumented engine (one pytest session per
module, the trustworthy suite shape) and records, per test function, every
contraction's pairing type and its (m_l, m_r, T, aligned, can) signature, plus
the growing operand broadcasts. A test whose signature list contains
``contract`` with ``m_l == T and m_r == 1`` reaches the single implicit sparse
axis; ``spatial_sparse_*`` reaches the spatial-sparse pairing; ``aligned=False``
reaches the genuine LCM grid.

Usage:  python t285_testset_census.py <out.json> [test_file ...]
"""
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pytest

import t285_lib as L


SETS = {
    "aligned_and_matched": [
        "matmul_blocks_test.py",
        "matmul_diff_blocks_test.py",
        "matmul_replication_test.py",
        "diag_block_diagonal_test.py",
        "coarsen_blockdiag_test.py",
    ],
    "mismatched": ["misaligned_blocks_test.py"],
    "explicit_vs_implicit": [
        "explicit_matmul_test.py",
        "implicit_matmul_test.py",
        "explicit_dense_test.py",
        "implicit_dense_test.py",
        "implicit_dense_self_test.py",
    ],
}


class CensusPlugin:
    """One Census per test function."""

    def __init__(self):
        self.rows = []
        self.cen = None

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        cen = L.Census()
        cen.install()
        try:
            yield
        finally:
            cen.uninstall()
            sig = []
            for frame in cen.frames:
                for f in frame:
                    sig.append(
                        (f["pairing_type"], f["T"], f["m_l"], f["m_r"],
                         bool(f["aligned"]), bool(f["can"]))
                    )
            self.rows.append(
                dict(
                    nodeid=item.nodeid,
                    contractions=len(cen.frames),
                    growing_calls=len(cen.growing),
                    growing_elements=sum(g["grew"] for g in cen.growing),
                    growing_axes=[a for g in cen.growing for a in g["axes"]],
                    signatures=sorted(set(sig)),
                )
            )


def classify(row):
    """Which of the implicit-axis cases the test's contractions touch."""
    tags = set()
    for pt, T, m_l, m_r, aligned, can in row["signatures"]:
        if pt.startswith("spatial_sparse"):
            tags.add("spatial_sparse")
        if not aligned:
            tags.add("lcm_grid")
        if T > 1 and can and aligned:
            if m_l == 1 and m_r == 1:
                tags.add("double_implicit_" + ("batch" if pt != "contract" else "contracted"))
            elif (m_l == T and m_r == 1) or (m_r == T and m_l == 1):
                tags.add(
                    "single_implicit_sparse" if pt == "contract" else "single_implicit_dense_carried"
                )
            elif m_l == T and m_r == T:
                tags.add("no_implicit_axis")
            else:
                tags.add("partially_stored")
    for a in row["growing_axes"]:
        if a["slot"] == "block" or a["slot"] == "shared":
            tags.add("single_implicit_dense_block")
        if a["slot"] == "split":
            tags.add("single_implicit_dense_contracted")
    return sorted(tags)


def main():
    out = sys.argv[1]
    tests_dir = os.environ.get(
        "T285_TESTS",
        os.path.abspath(os.path.join(HERE, "..", "..", "..", "tests", "core", "sparse_tensor")),
    )
    files = sys.argv[2:] or [f for fs in SETS.values() for f in fs]
    sys.path.insert(0, tests_dir)
    all_rows = []
    for f in files:
        plug = CensusPlugin()
        path = os.path.join(tests_dir, f)
        code = pytest.main(["-q", "-p", "no:cacheprovider", path], plugins=[plug])
        for r in plug.rows:
            r["file"] = f
            r["set"] = next((k for k, v in SETS.items() if f in v), "?")
            r["cases"] = classify(r)
        all_rows.extend(plug.rows)
        print(f"== {f}: exit {code}, {len(plug.rows)} tests", flush=True)
    with open(out, "w") as fh:
        json.dump(all_rows, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
