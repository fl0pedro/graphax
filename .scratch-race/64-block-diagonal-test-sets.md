# 64. The block-diagonal and implicit-dimension test sets (ticket dsnn-3qm.28.5)

Plain English. Short sentences. Every cost is a ratio or a counted element.

Vocabulary is CONTEXT.md's. An axis of a contraction is a no implicit axis, a
single implicit axis, or a double implicit axis. A single implicit axis is
sparse or dense. A sparse one is the meta axis of a diagonal pair. The engine's
choice for it is to broadcast the implicit side inside the contraction, or to
keep the axis on the storing operand. The retired word is not used here.

## The question

The owner hand-wrote the expected result of every contraction in three test
sets. Are those hand-written expectations the optimal emission? Can the engine
do better? And how do we test the real implementation against them?

## Verdict

1. The hand-written expectation is the optimum in eight of the eleven isolated
   cases. The engine reaches it under the landed default in six of the eight.
2. A better emission exists in two cases. The engine already emits it.
3. On a single implicit dense block axis the lazy frame stores 20 elements.
   The incumbent, the hand reference and the planner all store 60. The lazy
   frame performs 40 products. The others perform 120. The ratio is 0.33 on
   both channels.
4. On the genuine LCM grid the engine stores 280 elements. That is exactly the
   structural support of the dense result. The owner's hand-written reference
   in `explicit_matmul_test.test_block_block_gcd` declares a meta-2 pair with
   10 by 21 blocks. That is 420 elements, a ratio of 1.50 over the optimum.
   The planner stores the full 840, a ratio of 3.00.
5. The mismatched 3-D and 4-D matmuls store 1.50 times the structural support.
   They fall back to the LCM meta grid. The band form is better and is not new.
   The 2-D tests of the same file already reach it.
6. The `physical_shape` defect is real and total. Of the eleven modules,
   exactly one passes a hand-written `physical_shape`. The shared driver passed
   the result's own shape, so the check compared the result against itself.
7. Two of the three unreached cases of finding 63 are reached by these sets.
   Those two are the genuine LCM grid and the spatial-sparse pairing. The
   third, a partially stored extent, is not reached. The frame algebra says it
   cannot be reached. That is a stronger statement than "no target reaches it".

## Apparatus

graphax `wip/t285-20260907`, based on lane B's `f826da2`. The probes live in
`graphax/.scratch-race/probes/t285/`. Nothing under `src/` changed. This ticket
is measurement and tests.

Five modes run in one process. The incumbent executor is
`GRAPHAX_TILED_LEGACY=1`. The lazy frame runs under `GRAPHAX_TILED_LAZY` in
off, nodemote and full. The planner runs under `GRAPHAX_EINSUM_GENERAL=1` with
`GRAPHAX_PLANNER_EXACT=1`.

The probe records seven things per case and mode. The pairing census read out
of `_lazy_frame`. Every `_as_shape(mode="broadcast")` call that grows a buffer,
attributed to the axis that grew. The growing `broadcast_in_dim` equations of
the jaxpr. The CPU HLO kernel census. The XLA static temp. The stored element
count of the result. Which output dims are implicit.

There is no GPU job. GPU behaviour is read off the saved HLO that grill2/F2
analyzed. Its rule is short. A dot whose batch dims are degenerate or one-sided
becomes a fused reduce kernel on GPU. A clean 2-D dot becomes a cuBLAS call and
a fusion barrier.

## (a) The census of the three sets

### The defect, confirmed

`utils.assert_matmul_result` takes a `physical_shape`. Its own comment says the
argument catches unintended densification. The shared driver
`run_matmul_blocks_test` passed `result.val.shape` for all three operand forms.
That compares the result against itself. It can never fail.
`matmul_blocks_test.py` did the same at two more call sites.
`utils.matmul_reference` densifies both operands and calls `jnp.tensordot`, so
its own `val.shape` is the dense shape and holds no expectation either.

Of the eleven modules exactly one passed a hand-written `physical_shape`. That
is `matmul_replication_test.py`, in all five of its tests.

### What each module constructs and asserts

| set | module | tests | what it constructs | what it asserts |
|---|---|---|---|---|
| 1 | matmul_blocks_test.py | 5 | 13 dense and `_arr2st` shape pairs, one scalar case | values, out_shape, primal_shape, the self-comparing storage pin. The exhaustive sweep is skipped unless `EXHAUSTIVE=1` |
| 1 | matmul_diff_blocks_test.py | 3 | the divisor, shared-factor and coprime block generators | nothing. All three tests are skipped unless `EXHAUSTIVE=1` |
| 1 | matmul_replication_test.py | 5 | hand-built replication and diagonal pairs | values, both logical shapes, a real hand-written storage pin |
| 1 | diag_block_diagonal_test.py | 4 | `apply_diag` block masks on a dense pair | exact dense masks, one `assert_structure(expect_block=True)` |
| 1 | coarsen_blockdiag_test.py | 6 | `_coarsen_coupled_blockdiag` and one reconciled meta-16 against meta-2 matmul | bit-identical dense forms, the coarsened meta size, values against a dense oracle |
| 2 | misaligned_blocks_test.py | 18 | coprime, shared-factor and divisor block pairs, elementwise and matmul, 2-D to 4-D | the elementwise classes assert the structure the docstring claims. The matmul class asserted values only |
| 3 | explicit_matmul_test.py | 19 | hand-built tiled-layout references over every pairing kind | dense values plus, under `_PIN_LAYOUT`, byte identity of the layout at 15 of 19 sites. The three misaligned tests were relaxed to dense values |
| 3 | implicit_matmul_test.py | 8 | one operand with an implicit dense dim, in every position | dense values and `jnp.allclose` on `val`. `allclose` broadcasts, so this is an accidental storage check |
| 3 | explicit_dense_test.py | 25 | `dense(st)` and `dense(st, axes=...)` round trips | shapes and values. Never calls the matmul engine |
| 3 | implicit_dense_test.py | 11 | the same for implicit dims | shapes and values. Never calls the matmul engine |
| 3 | implicit_dense_self_test.py | 5 | `dense(st, hard=True)` | shapes and dim counts. Never calls the matmul engine |

### Which of the three unreached cases these sets reach

The genuine LCM grid is reached, by ten tests across three modules.
`explicit_matmul_test.test_block_block_gcd` uses meta 4 against meta 6. Its two
coprime siblings use 2 against 3. All seven matmul tests of
`misaligned_blocks_test.py` are misaligned by construction.
`coarsen_blockdiag_test` reconciles meta 16 against meta 2. Those contractions
return a `BandedIndex` pair, and only the LCM path builds one.

The spatial-sparse pairing is reachable and is exercised by the new isolated
case. Whether the existing sets reach it is answered by the per-test signature
census of job 63863.

A partially stored extent is not reached, and the frame algebra says it cannot
be. On an aligned pair `m_l = lo_p * (T // ol if ls_p != 1 else 1)`, and `lo_p`
is either 1 or `ol`. So `m_l` and `m_r` only ever take the values 1 or T. A
value in between needs `T > ol`, which forces `ol != orr`, which is the
not-aligned branch, which is the LCM grid. So ticket dsnn-3qm.28.2 needs a
target for two cases, not three. The third does not exist.

## (b) The optimum on paper, and the isolated experiment

Eleven hand-built contractions, one per case. Diagonal cases use meta M = 4,
out block P = 3, contracted block K = 2, primal block Q = 5. Dense cases use
a = 2, b = 3, c = 4, d = 5. The LCM case uses `explicit_matmul_test`'s own
a, b, c, d, e, f = 4, 6, 2, 5, 3, 7.

### The optimum

| case | buffer | products |
|---|---|---|
| single implicit sparse, lhs stores meta | 60 = M P Q | 120 = M P K Q |
| single implicit sparse, rhs stores meta | 60 | 120 |
| single implicit dense, block kind | 20 = M Q | 40 = M K Q |
| single implicit dense, contracted kind | 30 = a b d | 40 adds plus 30 products |
| single implicit dense, carried kind | 30 | 120 = a b c d |
| double implicit contracted | 30 | 30, c folds into scalar_mult |
| double implicit batch | 15 = b d | 60 = b c d |
| uniform operand | 10 = a d | 40 = a c d |
| genuine LCM grid | 280 | 420 = 12 x 5 x 7 |
| spatial-sparse pairing | 48 = b s b_r | 240 = b s c b_r |
| no implicit axis, the control | 60 | 120 |

The LCM grid's 280 is not a guess. The dense result has exactly 280
structurally non-zero entries, and no emission can store fewer. The block
case's 20 is smaller than its 60 non-zeros because the block axis is a
replication of 20 distinct values.

### What the engine stores and spends

Read as stored / products / grown operand elements.

| case | incumbent | nodemote | full | planner |
|---|---|---|---|---|
| single implicit sparse (lhs) | 60/120/30 | 60/120/30 | 60/120/0 | 60/120/0 |
| single implicit sparse (rhs) | 60/120/18 | 60/120/18 | 60/120/0 | 60/120/0 |
| single implicit dense block | 60/120/16 | **20/40/0** | 20/40/0 | 60/120/0 |
| single implicit dense contracted | 30/120/18 | 30/70/0 | 30/70/0 | 30/70/0 |
| single implicit dense carried | 30/120/12 | 30/120/12 | 30/120/0 | 30/120/0 |
| double implicit contracted | 30/30/0 | 30/30/0 | 30/30/0 | 30/30/0 |
| double implicit batch | 15/120/32 | 15/60/0 | 15/60/0 | 15/60/0 |
| uniform operand | 10/120/23 | 10/40/1 | 10/40/0 | 10/40/0 |
| genuine LCM grid | 280/420/0 | 280/420/0 | 280/420/0 | **840/10080/0** |
| spatial-sparse pairing | 48/240/40 | 48/240/40 | 48/240/40 | **48/240/0** |
| no implicit axis | 60/120/0 | 60/120/0 | 60/120/0 | 60/120/0 |

The jaxpr growing-broadcast count agrees with the `_as_shape` census call for
call and element for element in every cell but one. The exception is the LCM
grid, where the tiled path's growth is 3 calls and 2087 elements inside the LCM
reduce and does not pass through `_as_shape` at all.

### The CPU HLO

The broadcast modes of the sparse, carried and spatial-sparse cases emit one
batched dot with a degenerate or one-sided batch dim: `f32[4,3,5]`,
`f32[2,3,5]`, `f32[3,4,4]`. The `full` mode and the planner emit a clean 2-D
dot instead: `f32[12,5]`, `f32[3,20]`, `f32[4,5]`, `f32[3,10]`, `f32[12,4]`.
The contracted, double-contracted and uniform cases emit no dot at all under
the default: one reduce and some elementwise work, with 0 to 40 bytes of static
temp. The LCM grid emits `f32[12,35]` plus a reduce on the tiled path and
`f32[20,42]` on the planner.

Static temp falls sharply where the lazy rules fire. Incumbent against
nodemote: 288 to 40 bytes on the contracted case, 416 to 48 on the double
implicit batch case, 416 to 0 on the uniform operand, 96 to 0 on the block
case.

### The GPU fusion judgement

No GPU job was run. This is read off the saved HLO patterns of grill2/F2.

The batched dots have degenerate or one-sided batch dims. XLA GPU cannot call
cuBLAS on them. It expands them into broadcast, broadcast, multiply, reduce and
fuses the whole thing into one kInput kernel. The operand broadcast becomes an
index remap there and costs no buffer. That is the GPU win of the broadcast
form, and it is why the default keeps it.

The clean 2-D dots map to one cuBLASLt call each. A custom call is a fusion
barrier. It also forces a copy kernel for every sliced operand. The carried
case already shows the CPU shadow of that: 7 copies and 6 transposes under
`full` against 4 and 4 under the default.

The contracted, double-contracted and uniform cases emit no dot under the
default. They reserve no cuBLAS workspace on GPU and stay fusible on both
devices. Those three are the only cases where the two devices cannot disagree.

## (c) The verdict per case

The hand-written expectation is the optimum, and the default engine reaches it,
in these cases. The single implicit dense contracted kind stores 30 and spends
70 operations with no dot. The double implicit contracted axis stores 30 and
spends 30, with c folded into `scalar_mult`. The double implicit batch axis
stores 15 and spends 60. The uniform operand stores 10 and spends 40. The
no-implicit control stores 60 and spends 120 with zero growth in every mode.
The genuine LCM grid stores 280 on the engine side, which is the structural
support, so it is provably optimal.

The hand-written expectation is the optimum on storage and products, but the
default pays an operand broadcast, in these cases. Both single implicit sparse
directions store 60 and spend 120 in every mode; the default writes 30 or 18
extra operand elements, and only `full` and the planner reach zero. That trade
is finding 63's subject and its rule already covers it. The single implicit
dense carried kind stores 30 and spends 120 in every mode; the default writes
12 extra operand elements.

That last row is a correction to finding 63 deliverable (f). Its table lists
the carried case as broadcast-free under the default. That holds for the
block-slot variant of a carried axis. The meta-slot variant, a `batch_out`
pairing whose meta axis only one side stores, is governed by the demote rule
and still grows under the default. The isolated case here is that variant. The
totals of finding 63 are unaffected.

A better emission exists in two cases.

The single implicit dense block kind. The optimum is 20 stored and 40 products.
The lazy frame reaches it exactly. The incumbent, the lazy-off frame and the
planner all store 60 and spend 120. The ratio is 0.33 on both channels in
favour of the lazy frame. The planner is not the reference here.

The genuine LCM grid, on the test side. The hand-written reference in
`explicit_matmul_test.test_block_block_gcd` builds a meta-2 pair with 10 by 21
blocks, which stores 420 elements. The engine stores 280 as a `BandedIndex`
pair with `val` of shape (4, 2, 5, 7). The ratio is 1.50 in favour of the
engine. The planner stores 840 and spends 10080 products against the engine's
420, ratios of 3.00 and 24.0. On this case the tiled path beats both the hand
reference and the planner.

One case where the engine is worse than it could be, and the fix is one line.
The spatial-sparse pairing reaches the optimal 48 stored and 240 products in
all four tiled modes, but writes 40 extra operand elements, a 3.00 blow-up of a
20-element operand. The planner writes none. The cause is that
`spatial_sparse_lhs` and `spatial_sparse_rhs` are not in `_LAZY_PAIRINGS`
(`ops/matmul.py` lines 923 to 934), so `can` is False and no lazy rule fires.
The frame's own numbers for that pair are `ol/orr` 3/1, `T/G` 3/1, aligned
True, `m_l/m_r` 3/1. That is exactly the shape the demote rule already handles
for a contract pairing. Adding the two literals to `_LAZY_PAIRINGS` is the
whole change, and it belongs to ticket dsnn-3qm.28.2.

### The mismatched set, on the real tests

Stored, structural support, and the logical size, on the default engine.

| test | stored | support | logical | stored/support |
|---|---|---|---|---|
| 2d_coprime_2x3_contract | 72 | 72 | 72 | 1.00 |
| 2d_coprime_3x5_contract | 270 | 210 | 900 | 1.29 |
| 2d_divisor_2x4_contract | 64 | 64 | 256 | 1.00 |
| 2d_shared_factor_4x6_contract | 192 | 192 | 576 | 1.00 |
| 3d_one_misalignment | 216 | 144 | 432 | 1.50 |
| 4d_one_misalignment | 432 | 288 | 864 | 1.50 |
| 4d_two_misalignments | 4608 | 3072 | 36864 | 1.50 |
| explicit block_block_gcd | 280 | 280 | 840 | 1.00 |
| explicit coprime_a_greater_b | 140 | 140 | 210 | 1.00 |
| explicit coprime_a_less_b | 140 | 140 | 210 | 1.00 |

The 2-D shapes take the band path and store the support exactly. The 3-D and
4-D shapes fall back to the LCM meta grid and store 1.50 times the support. A
better emission exists and it is not new: it is the band form the same file
already gets at 2-D. That is the structure expectation
`misaligned_blocks_test.py`'s own docstring states and never asserted for
matmul.

## (d) The assertion design

Four assertion kinds, all in `tests/core/sparse_tensor/utils.py`.

`physical_shape` stays what it was: the exact stored extents, order-free, hand
written. It is now never derived from the result.

`max_stored` is new. It is the structural support of the dense reference. An
emission may store fewer elements than the support, because a replicated axis
kept implicit stores one copy of many equal entries. Storing more means it
materialised structure it did not need. This is the engine-independent ceiling
and it replaces the self-comparing pin in the shared driver, where no single
hand-written shape can cover a generated tensor pair.

`axis_pattern` is new. One character per output dim, in `dims` order. D is a
stored dense dim, I an implicit dense dim, S a stored sparse pair, s an
implicit sparse pair. Physical axis order stays unpinned. Whether an axis
exists at all is the contract.

The growing-broadcast count is asserted exactly in the new module
`implicit_axis_storage_test.py`, including where it is not zero today. A change
that removes one of those broadcasts must edit that file on purpose.

### What changed, file by file

`utils.py` gains `stored_elements`, `support_size`, `assert_axis_pattern`, and
two keyword arguments on `assert_matmul_result`. `run_matmul_blocks_test` now
passes `max_stored = support_size(reference_dense)` for all three operand forms
instead of the result's own shape.

`matmul_blocks_test.py` loses its two self-comparing pins. The scalar test
gains a hand-written `physical_shape` of (1, 3).

`matmul_replication_test.py` keeps its five existing hand-written pins and
gains the axis pattern and the support ceiling.

`implicit_matmul_test.py` gains `assert_storage` on all seven real tests. It
checks the hand reference `R_st` against the expected numbers first, then the
engine's result against the same numbers, then the axis pattern of both. The
expected pairs are (30, DDD), (10, DID), (30, DDD), (30, DDD), (10, DID),
(2, DII) and (0, III). Every one equals what the owner's own `R_st` carries.

`misaligned_blocks_test.py` gains `_assert_storage` on all seven matmul tests.
It pins the stored count and the hand-written support, refuses a densified
output where the product is structurally sparse, caps `stored / support` at
1.50, and pins whether the output carries a `BandedIndex` pair.

`explicit_matmul_test.py` gains `assert_band_storage` on its three relaxed
misaligned tests. It asserts the engine's 280, 140 and 140, checks that each
equals the structural support, and requires a `BandedIndex` pair. It is gated
on `_PIN_LAYOUT` like the rest of that file, because the planner returns the
dense product instead.

`implicit_axis_storage_test.py` is new. Eleven tests, one per case, each
pinning stored count, axis pattern, growing-broadcast count, and where it
applies the structural support and the index class.

### Red before, green after

Nothing here is red under the default engine. The engine already reaches the
optimum in every case where an optimum was claimed. The value of the change is
that the claims are now checked. Three of them would go red on a regression
that no existing test could see: the shared driver's densification ceiling, the
misaligned matmul's stored counts, and the band form of the three explicit LCM
tests.

## Two things the census showed that the ticket did not ask for

`matmul_blocks_test.py` runs 41 contractions and every one of them is dense
against dense. It exercises no implicit-axis case at all. Its name suggests
block coverage. The block coverage lives in `matmul_diff_blocks_test.py`, and
all three tests of that file are skipped unless `EXHAUSTIVE=1`. So the aligned
and matched set has no default block coverage beyond the five replication
tests.

`explicit_dense_test.py`, `implicit_dense_test.py` and
`implicit_dense_self_test.py` perform zero contractions between them. They test
`dense()` and `dense(st, axes=...)`. They cannot catch a matmul storage defect
and should not be counted as part of the implicit-dims matmul coverage.

## What I did not do

I ran no GPU job. The ticket says none is needed and the GPU statements above
are read off the saved HLO of grill2/F2.

I changed nothing under `src/`. The one-line fix the spatial-sparse case needs
(`spatial_sparse_lhs` and `spatial_sparse_rhs` in `_LAZY_PAIRINGS`) is named
here and left to ticket dsnn-3qm.28.2, which owns the engine.

I did not measure latency. Every number here is a counted element, a counted
kernel, or a static temp figure. Finding 63 owns the latency of the same
choices and its rule already covers them.

I did not turn on `EXHAUSTIVE=1`. `matmul_diff_blocks_test.py` stays skipped by
default and I did not change that. Whether the three block regimes it generates
should run by default is an owner decision, not mine.

I did not change the hand-written `R_st` of
`explicit_matmul_test.test_block_block_gcd`, although it asks for 1.50 times
the optimal buffer. It is a correct value oracle and the ticket did not ask me
to rewrite the owner's references.

The census classifier under-counts two cases in its first run. The block-slot
and split-slot kinds of a single implicit dense axis were detectable only
through a growing broadcast there, and the landed default already removes those
growths. The re-instrumented classifier reads the lazy flags directly.

## (e) The runs

| job | node | tree | what |
|---|---|---|---|
| 63859 | pgi15-cpu1 | b4ba2ac | cancelled. The node was fully allocated by another group for the whole session, with a long queue behind it |
| 63863 | pgi15-cpu2 | b4ba2ac | the baseline. The isolated experiment, the test-set census, the whole graphax pytest |
| 63866 | pgi15-cpu2 | 5637c7d | the green run. The same three stages with the assertions in place |

pgi15-cpu2 has no home mount. It did not need one. The job runs out of
`/Scratch/assmuth/t57/stack` with `HOME` redirected to node-local `/tmp`, which
is finding 57's route, and it writes nothing to the home directory. That makes
pgi15-cpu2 usable for CPU work today, which is worth recording for the fleet.

The baseline, job 63863: 1343 passed, 71 skipped, 2 xfailed, 441 subtests
passed, in 1741 seconds, exit 0.

### The per-test case census, measured

| set | module | tests | contractions | growing calls / elements | cases reached |
|---|---|---|---|---|---|
| 1 | matmul_blocks_test.py | 5 | 41 | 0 / 0 | single implicit dense carried (1) |
| 1 | matmul_diff_blocks_test.py | 3 | 0 | 0 / 0 | none, all three skip |
| 1 | matmul_replication_test.py | 5 | 5 | 3 / 61 | single implicit sparse (3), no implicit axis (2) |
| 1 | diag_block_diagonal_test.py | 4 | 0 | 0 / 0 | none, no contraction |
| 1 | coarsen_blockdiag_test.py | 6 | 1 | 0 / 0 | no implicit axis (1) |
| 2 | misaligned_blocks_test.py | 18 | 14 | 0 / 0 | LCM grid (6), no implicit axis (3) |
| 3 | explicit_matmul_test.py | 19 | 19 | 0 / 0 | LCM grid (12), no implicit axis (10) |
| 3 | implicit_matmul_test.py | 8 | 8 | 3 / 17 | single implicit dense carried (6), double implicit batch (1), no implicit axis (3) |
| 3 | explicit_dense_test.py | 25 | 0 | 0 / 0 | none |
| 3 | implicit_dense_test.py | 11 | 0 | 0 / 0 | none |
| 3 | implicit_dense_self_test.py | 5 | 0 | 0 / 0 | none |

Totals per case over the three sets: single implicit sparse 3, single implicit
dense carried 7, double implicit batch 1, no implicit axis 19, LCM grid 18,
spatial-sparse 0, partially stored 0, single implicit dense block 0, single
implicit dense contracted 0, double implicit contracted 0.

The block kind reaches zero tests because no module of the three sets builds a
diagonal pair whose block axis has no physical axis. The contracted kind reads
zero because the census detects it only through a growing split-slot broadcast,
which the landed default already removes; the isolated experiment covers it
directly. Both numbers are lower bounds, and both cases are now covered by
`implicit_axis_storage_test.py`.

### The green run

Job 63866, graphax 5637c7d. Whole graphax pytest: 1354 passed, 71 skipped,
2 xfailed, 441 subtests passed, in 1755 seconds, exit 0.

Against the baseline that is plus 11 passed, which is exactly the new module
`implicit_axis_storage_test.py`. Skips and xfails are unchanged. Nothing went
red. All eleven modules of the three sets pass one process each.

The isolated experiment reproduces byte for byte between the two cluster runs.
Against the local run it agrees in every cell but one static-temp figure, the
planner on the LCM grid, 3008 bytes locally against 3376 on the node. That is
XLA temp accounting, not structure.

### A note for any lane that writes a finding into graphax

`graphax/.gitignore` line 69 is `*.md`. A finding placed under
`.scratch-race/` is silently ignored by `git add -A`. It needs `git add -f`.
