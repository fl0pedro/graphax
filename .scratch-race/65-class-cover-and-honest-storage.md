# 65. Class cover, honest storage, and the correction to finding 64

Plain English. Short sentences. Every cost is a counted element.

## The question

Finding 64 asserted storage against the SUPPORT, the count of non-zero entries
of the dense result. The owner rejected that ceiling on 2026-09-07: a
`SparseTensor` cannot store an arbitrary set of entries, so a correct emission
can legitimately store more than the support. The question became: what is the
real floor, and which of the measured excesses are defects?

## Verdict

1. The floor is the CLASS COVER, not the support. See CONTEXT.md, "Structural
   class". An implicit axis then divides the cover, because it stores one copy
   of an extent-n axis.
2. Waste has TWO independent levels. A result can be minimal for the block
   partition it chose while that partition is coarser than the operands force.
   The two need two assertions.
3. All eleven isolated cases of finding 64 are at the class optimum AND at the
   best cover over every class the engine emits. The isolated set holds no
   storage defect.
4. `2d_coprime_3x5_contract` is NOT a defect. It stores 270 against a support
   of 210, and 270 is the exact column-band cover. Finding 64 recorded a ratio
   of 1.29 without saying it was legal. My planned expected-failure on that test
   would have been wrong.
5. Three tests are defects: `3d_one_misalignment`, `4d_one_misalignment` and
   `4d_two_misalignments`, each storing 1.50 times the operand-partition
   optimum.
6. The cause is sharper than finding 64 stated. The engine has no band path
   above rank 2. At rank 2 it picks the better of the row band and the column
   band. At rank 3 and above it falls to the least-common-multiple meta grid.

## The three quantities

For a contraction along a logical axis of length L that the lhs partitions into
n_a blocks of b_a and the rhs into n_b blocks of b_b, with output block extents
r and c:

```
occupancy   O(i, j) = 1  iff  [b_a i, b_a i + b_a)  meets  [b_b j, b_b j + b_b)

support     |{(i, j) : O}| * r * c
row band    n_a * max_i |{j : O}| * r * c
col band    n_b * max_j |{i : O}| * r * c
meta grid   (L / l) * (l / b_a * r) * (l / b_b * c),   l = lcm(b_a, b_b)
dense       (n_a r) * (n_b c)

class cover      = the cover of the class the result declares
honest minimum   = class cover / product of the extents kept implicit
```

`set` reaches the support exactly. It is a legal `SparseTensor` class and no
contraction path emits one, so the support is headroom for a future emission,
never a bound on the one running. `EMITTABLE_CLASSES` in the test utilities
names the four the engine does emit.

## The closed form, per test

Computed with no engine and no JAX (`probes/t286/analytic.py`).

| case | support | row band | col band | meta grid | dense | engine | best emittable |
|---|---|---|---|---|---|---|---|
| 2d_coprime_2x3_contract | 72 | 72 | 72 | 72 | 72 | 72 | 72 |
| 2d_coprime_3x5_contract | 210 | 300 | **270** | 450 | 900 | 270 | 270 |
| 2d_divisor_2x4_contract | 64 | 64 | 64 | 64 | 256 | 64 | 64 |
| 2d_shared_factor_4x6 | 192 | 288 | **192** | 288 | 576 | 192 | 192 |
| 3d_one_misalignment | 144 | 216 | **144** | 216 | 432 | 216 | 144 |
| 4d_one_misalignment | 288 | 432 | **288** | 432 | 864 | 432 | 288 |
| 4d_two_misalignments | 3072 | 4608 | **3072** | 4608 | 36864 | 4608 | 3072 |
| explicit_block_block_gcd | 280 | **280** | 420 | 420 | 840 | 280 | 280 |

The owner's hand-written 420 in `explicit_matmul_test.test_block_block_gcd` is
the column band. The engine takes the row band at 280. On that shape the row
band is the better one, so the engine is right and the hand value is a legal
but larger member of the same class family.

## The engine, measured

graphax `wip/t285-20260907`, CPU, `GRAPHAX_TILED_LAZY=nodemote`
(`probes/t286/confirm.py`, `probes/t286/confirm_mis.py`).

Eleven isolated cases. Every one matches the closed form and the hand-written
optimum of `t285_lib.OPTIMUM`, cell for cell.

| case | support | cover | replication | stored | honest minimum |
|---|---|---|---|---|---|
| single_implicit_sparse | 60 | 60 | 1 | 60 | 60 |
| single_implicit_sparse_rhs | 60 | 60 | 1 | 60 | 60 |
| single_implicit_dense_block | 60 | 60 | 3 | 20 | 20 |
| single_implicit_dense_contracted | 30 | 30 | 1 | 30 | 30 |
| single_implicit_dense_carried | 30 | 30 | 1 | 30 | 30 |
| double_implicit_contracted | 30 | 30 | 1 | 30 | 30 |
| double_implicit_batch | 30 | 30 | 2 | 15 | 15 |
| uniform_operand | 30 | 30 | 3 | 10 | 10 |
| lcm_grid | 280 | 280 | 1 | 280 | 280 |
| spatial_sparse | 48 | 48 | 1 | 48 | 48 |
| no_implicit_control | 60 | 60 | 1 | 60 | 60 |

The mismatched set, at the operand partition:

| case | stored | operand-partition optimum | ratio | set-index floor |
|---|---|---|---|---|
| 2d_coprime_3x5 | 270 | 270 | 1.00 | 210 |
| 2d_divisor_2x4 | 64 | 64 | 1.00 | 64 |
| 2d_shared_factor_4x6 | 192 | 192 | 1.00 | 192 |
| 3d_one_misalignment | 216 | 144 | **1.50** | 144 |
| 4d_one_misalignment | 432 | 288 | **1.50** | 288 |

## The assertions

Five names in `tests/core/sparse_tensor/utils.py`.

`class_covers(dense_ref, st)` builds the occupancy grid at the partition the
RESULT declares and returns the cover of every class there.

`partition_covers(dense_ref, block_sizes)` does the same at a partition given
as an argument, the finest the operands allow, and returns the best over
`EMITTABLE_CLASSES` plus the set-index floor.

`replication_factor(st)` is the product of every extent kept implicit. A sparse
pair shares one meta axis, so an implicit meta counts once for the pair; the two
block extents count separately.

`assert_constant_along_implicit(dense_ref, st)` proves each implicit axis is
genuinely replicated, so the division is lossless. Without it an engine can
declare an axis implicit, discard the variation along it, and pass.

`assert_honest_storage(st, dense_ref)` asserts `stored == cover / replication`
as an EQUALITY, after the constancy guard. Storing more means the result
materialised structure its class did not force. Storing less means it found a
better class than it declared, which is a change to the declaration and has to
be made on purpose.

`assert_partition_optimal(st, dense_ref, block_sizes)` is the second level.
This is the one the three 3-D and 4-D tests fail.

## What this retires

`support_size` stays as a REPORTED number. It is no longer a bound anywhere.
`max_stored` as a ceiling is replaced by the two equalities above.

## Corrections to finding 64

* Verdict 5 of finding 64 says the mismatched 3-D and 4-D shapes "store 1.50
  times the structural support" and that "the band form is better". Both hold.
  The reason given, "they fall back to the LCM meta grid", is right and
  incomplete: there is no band path at all above rank 2.
* The 1.29 of `2d_coprime_3x5_contract` appears in finding 64's table beside the
  three 1.50 rows and reads as the same kind of number. It is not. It is the
  class optimum.
