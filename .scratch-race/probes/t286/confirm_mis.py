import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
G = ("/tmp/claude-1001/-home-claude-dsnn--claude-worktrees-alpha-pprox/"
     "22ed5f10-dee2-44f0-b385-b953fa5bc6a1/scratchpad/race/G/graphax")
sys.path[:0] = [G + "/src", G + "/tests/core/sparse_tensor"]

import jax, jax.numpy as jnp, jax.random as jr
from graphax.sparse.indexes import DenseIndex, DiagonalIndex
from graphax.sparse.tensor import SparseTensor
from graphax.sparse.ops.matmul import matmul
from utils import class_covers, replication_factor, stored_elements

def _n(shape, k):
    return jr.normal(jr.PRNGKey(k), shape).astype(jnp.float32)

def pair2d(N, B, k):
    return SparseTensor(
        (DiagonalIndex(0, N, axis=0, other_id=1, block_size=B, block_axis=1),),
        (DiagonalIndex(1, N, axis=0, other_id=0, block_size=B, block_axis=2),),
        _n((N, B, B), k))

CASES = {}
CASES["2d_coprime_3x5"] = (pair2d(10, 3, 1), pair2d(6, 5, 2), None)
CASES["2d_divisor_2x4"] = (pair2d(8, 2, 1), pair2d(4, 4, 2), None)
CASES["2d_shared_factor_4x6"] = (pair2d(6, 4, 1), pair2d(4, 6, 2), None)

s1 = 3
CASES["3d_one_misalignment"] = (
    SparseTensor((DenseIndex(0, s1, 0),
                  DiagonalIndex(1, 6, axis=1, other_id=2, block_size=2, block_axis=2)),
                 (DiagonalIndex(2, 6, axis=1, other_id=1, block_size=2, block_axis=3),),
                 _n((s1, 6, 2, 2), 1)),
    SparseTensor((DenseIndex(0, s1, 0),
                  DiagonalIndex(1, 4, axis=1, other_id=2, block_size=3, block_axis=2)),
                 (DiagonalIndex(2, 4, axis=1, other_id=1, block_size=3, block_axis=3),),
                 _n((s1, 4, 3, 3), 2)), None)

t1, t2 = 2, 3
CASES["4d_one_misalignment"] = (
    SparseTensor((DenseIndex(0, t1, 0), DenseIndex(1, t2, 1),
                  DiagonalIndex(2, 6, axis=2, other_id=3, block_size=2, block_axis=3)),
                 (DiagonalIndex(3, 6, axis=2, other_id=2, block_size=2, block_axis=4),),
                 _n((t1, t2, 6, 2, 2), 1)),
    SparseTensor((DenseIndex(0, t1, 0), DenseIndex(1, t2, 1),
                  DiagonalIndex(2, 4, axis=2, other_id=3, block_size=3, block_axis=3)),
                 (DiagonalIndex(3, 4, axis=2, other_id=2, block_size=3, block_axis=4),),
                 _n((t1, t2, 4, 3, 3), 2)), None)

W = 22
print(f"{'case'.ljust(W)} | supp | cover | best | repl | stored | honest | verdict")
print("-" * (W + 60))
for name, (a, b, _) in CASES.items():
    got = matmul(a, b)
    ref = a.dense() @ b.dense() if a.dense().ndim == 2 else jnp.matmul(a.dense(), b.dense())
    cov = class_covers(ref, got)
    r = replication_factor(got)
    honest = cov["declared"] // r
    stored = stored_elements(got)
    per = cov["per_class"]
    v = "OK" if stored == honest else f"MISMATCH != {honest}"
    if honest * r != cov["best"]:
        v += f"; class costs {honest * r / cov['best']:.2f}x"
    print(f"{name.ljust(W)} | {cov['support']:4} | {cov['declared']:5} | "
          f"{cov['best']:4} | {r:4} | {stored:6} | {honest:6} | {v}")
    for k, c in per.items():
        print(f"{'':{W}} |   pair {k}: " +
              " ".join(f"{n}={val}" for n, val in sorted(c.items())) +
              f"  declared={[type(got.dims[k[0]]).__name__]}")

print()
print("Partition optimality (the finest partition the operands allow):")
from utils import partition_covers
PART = {"2d_coprime_3x5": (3, 5), "2d_divisor_2x4": (2, 4),
        "2d_shared_factor_4x6": (4, 6),
        "3d_one_misalignment": (3, 2, 3), "4d_one_misalignment": (2, 3, 2, 3)}
for name, (a, b, _) in CASES.items():
    got = matmul(a, b)
    ref = a.dense() @ b.dense() if a.dense().ndim == 2 else jnp.matmul(a.dense(), b.dense())
    pc = partition_covers(ref, PART[name]); best = pc["best"]; floor = pc["set_floor"]
    st = stored_elements(got)
    print(f"  {name:22} stored {st:5}  operand-partition best {best:5}  "
          f"ratio {st / best:.2f}  (set-index floor {floor})")
