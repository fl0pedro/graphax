# 66. What sparsity is actually there (ticket dsnn-3qm.28.6)

**Status: the first version of this finding was wrong and is retracted below.**
Two adversarial reviews on 2026-09-07 found a correctness bug in the oracle and
a classifier that over-reported structure in 85 percent of cases. What survives
is stated here; what does not is listed with its correction.

## Method

A dense vertex-elimination oracle written from the definition
(`probes/t287/dense_elim.py`), using no graphax storage. Edge `u -> v` holds the
full Jacobian `d(v)/d(u)` of shape `shape(v) + shape(u)`, built with
`jax.jacfwd` on one primitive. Eliminating `v` composes every `(u -> v, v -> w)`
pair with `jnp.tensordot` over `v`'s rank and adds the product onto `u -> w`.

Ten targets, both orders (reverse and static minimum Markowitz degree), CPU,
float32. 273 records: 99 elemental partials and 174 accumulated Jacobians.

## The defects found by review, all confirmed by reproduction

1. **The oracle returned half the gradient whenever a variable appeared twice
   in one equation.** `trace` assigned the elemental partial per operand
   position instead of summing over positions. Measured: `sum(x*x)` returned
   `[1,2,3]` against the true `[2,4,6]`; `sum(x+x)` the same; `sum(x*x*x)` a
   ratio of 0.667. `x**2` lowers to `integer_pow`, one operand, and was
   correct, so the defect hid from every validation target. Fixed. Any
   target with a squared term, a Gram matrix or a variance would have been
   silently halved.
2. **The pair classifier reported an `any` projection as if it were the
   structure.** It reduced every other axis with `any` before classifying an
   axis pair, which is a union, so a tensor whose every slice was sparse could
   read `full`. Measured after the fix: the projection disagrees with the
   slices in **487 of 640 pair entries, 76 percent**.
3. **Empty and permutation patterns were counted as bands.** A row with zero
   or one non-zero is trivially a contiguous run. 32 of the 45 reported band
   pairs were one of those two. The band conclusion of the first version is
   void.
4. **The census was not reproducible.** Markowitz ties broke on `str(Var)`,
   which prints a heap address, and the `edge` field recorded the same address,
   so records could not be joined across runs. Fixed by breaking ties on the
   equation index and naming vars by position. Three runs now agree byte for
   byte.
5. **`block_period` was identically 1.** It searched upward for the SMALLEST
   block size making the pattern constant, and 1 satisfies that for every
   array. The field is deleted. The first version's claim that no block
   structure exists rested on it and is withdrawn.
6. **All-zero tensors were reported as replicated on every axis and as bands.**
   Six records, all from `attention`'s `stop_gradient` edge. Excluded now.
7. **`replicated_axes` used numpy's default `rtol=1e-5`.** Slices differing by
   2.5e-6 relative were called replicated. Now `rtol=0`.
8. **The two stages were not comparable populations.** Every target returns a
   scalar, so 115 of 174 accumulated records (66 percent) are gradients into
   that scalar, against 11 of 99 elemental records (11 percent). A gradient is
   dense by nature. The first version's headline compared the two directly.

## What survives

The oracle, after the fix. It reproduces `jax.grad` on all ten targets, both
orders, and on six targets built to trigger the repeated-operand bug. It also
passes a **path-sum check on 508 intermediate edges**: after eliminating a set
S, edge `u -> w` must equal the sum over every `u -> w` path with interior in S.
That check is what the first validation lacked (`probes/t287/validate2.py`).

One measurement survives because it does not depend on the classifier at all.
Density is read straight off the non-zero count.

| population | n | fully dense | mean density |
|---|---|---|---|
| elemental, NOT into the scalar output | 88 | 0 (0.0%) | 0.132 |
| accumulated, NOT into the scalar output | 59 | 4 (6.8%) | 0.209 |
| elemental, into the scalar output | 11 | 11 (100%) | 1.000 |
| accumulated, into the scalar output | 115 | 109 (94.8%) | — |

Read it as: **fill-in is real and mild.** On comparable tensors, mean density
rises from 0.132 to 0.209 and the fully-dense share from 0 to 6.8 percent. The
first version claimed 11 percent to 68 percent. That gap was the gradient
population, not fill-in.

## What is NOT claimed

Every count of diagonal, banded, set or block structure is withdrawn. The
classifier has been rewritten three times and the counts move each time, which
means the classification is not designed yet. Naming a structure needs a
definition that is sound on a tensor of rank above 2, and the two candidates
tried so far are each unsound in one direction: the `any` projection
over-reports, and per-slice agreement under-reports (a genuine double diagonal
reads as "mixed" because its slices differ). Until that is settled, no
structure count from this census is usable.

The misalignment verdict of the first version is withdrawn outright. The check
compared diagonal extents, and the classifier returns `diagonal` only when the
axis pair is square, so two diagonal pairs sharing an axis have equal extents by
construction. The check could not fire on any input.

## Apparatus

graphax `wip/t285-20260907`. `probes/t287/`: `dense_elim.py` (the oracle),
`t287_shape_census.py` (the classifier), `targets2.py` (seven targets),
`run_census.py`, `validate.py` (the weak first check, kept as record),
`validate2.py` (the path-sum check), `t287.jsonl` (the records).

## Known limits

* Ten small targets, CPU, float32. Not the campaign shapes.
* The oracle cannot handle a multi-output primitive, `where`, `scan`, or a
  `custom_jvp` such as `relu`: those crash or sever the chain. That excludes
  most real networks and bounds what this census can ever cover.
* A `jit` block counts as one primitive, so granularity is not uniform.
* Structural and numerical zeros are not separated. One `attention` tensor is
  mathematically zero and reads as density 0.5 in float32.
* One random input per target. One point cannot establish structure.
