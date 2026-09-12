#!/usr/bin/env python
"""Closed-form support, class cover and honest minimum. No engine, no JAX.

For a contraction along a logical axis of length L that the lhs partitions into
n_a blocks of b_a and the rhs into n_b blocks of b_b:

  occupancy O(i,j) = 1  iff  [b_a i, b_a i + b_a) meets [b_b j, b_b j + b_b)

The output row-block i has extent r (the lhs out block) and col-block j has
extent c (the rhs out block). Four covers:

  support     |{(i,j): O}| * r * c              the non-zero entries
  row band    n_a * max_i |{j: O}| * r * c      one uniform bandwidth per row
  col band    n_b * max_j |{i: O}| * r * c      one uniform bandwidth per col
  meta grid   (L/l) * (l/b_a * r) * (l/b_b * c) with l = lcm(b_a, b_b)
  dense       (n_a r) * (n_b c)

honest minimum = the chosen cover / product of the extents kept implicit.
"""
from math import gcd


def lcm(a, b):
    return a * b // gcd(a, b)


def occupancy(L, b_a, b_b):
    n_a, n_b = L // b_a, L // b_b
    return n_a, n_b, [[int(not (b_a * i + b_a <= b_b * j or b_b * j + b_b <= b_a * i))
                       for j in range(n_b)] for i in range(n_a)]


def covers(L, b_a, b_b, r, c):
    n_a, n_b, O = occupancy(L, b_a, b_b)
    hits = sum(map(sum, O))
    l = lcm(b_a, b_b)
    return {
        "support": hits * r * c,
        "row_band": n_a * max(sum(row) for row in O) * r * c,
        "col_band": n_b * max(sum(O[i][j] for i in range(n_a)) for j in range(n_b)) * r * c,
        "meta_grid": (L // l) * (l // b_a * r) * (l // b_b * c),
        "dense": (n_a * r) * (n_b * c),
    }


ROWS = [
    # name,                        L,  b_a, b_b,  r,  c, batch, engine_stored
    ("2d_coprime_2x3_contract",   12,   2,  12,  2, 6,     1,   72),
    ("2d_coprime_3x5_contract",   30,   3,   5,  3, 5,     1,  270),
    ("2d_divisor_2x4_contract",   16,   2,   4,  2, 4,     1,   64),
    ("2d_shared_factor_4x6",      24,   4,   6,  4, 6,     1,  192),
    ("3d_one_misalignment",       12,   2,   3,  2, 3,     3,  216),
    ("4d_one_misalignment",       12,   2,   3,  2, 3,     6,  432),
    ("explicit_block_block_gcd",  12,   3,   2,  5, 7,     1,  280),
    ("isolated_lcm_grid",         12,   3,   2,  5, 7,     1,  280),
]

W = max(len(r[0]) for r in ROWS)
head = ("case".ljust(W) + " | support | row band | col band | meta grid |"
        "   dense | engine | best | verdict")
print(head)
print("-" * len(head))
for name, L, b_a, b_b, r, c, batch, stored in ROWS:
    cv = {k: v * batch for k, v in covers(L, b_a, b_b, r, c).items()}
    best = min(cv["row_band"], cv["col_band"], cv["meta_grid"], cv["dense"])
    verdict = ("at the class optimum" if stored == best
               else f"excess {stored / best:.2f}x over {best}")
    print(f"{name.ljust(W)} | {cv['support']:7d} | {cv['row_band']:8d} | "
          f"{cv['col_band']:8d} | {cv['meta_grid']:9d} | {cv['dense']:7d} | "
          f"{stored:6d} | {best:4d} | {verdict}")

# 4d_two_misalignments: two independent contracted pairs multiply.
p1 = covers(12, 2, 3, 2, 3)
p2 = covers(16, 2, 4, 2, 4)
print()
print("4d_two_misalignments (two independent pairs, covers multiply)")
for k in ("support", "row_band", "col_band", "meta_grid", "dense"):
    print(f"  {k:10s} {p1[k]:5d} x {p2[k]:5d} = {p1[k] * p2[k]:7d}")
best2 = min(p1[k] * p2[k] for k in ("row_band", "col_band", "meta_grid", "dense"))
print(f"  engine 4608, best {best2}, excess {4608 / best2:.2f}x")
