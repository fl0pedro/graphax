# T69 design note: a dense-contraction mode in graphax

Ticket dsnn-3qm.69. Branch `wip/t69-20260906`, base `wip/t62-20260905` plus .68, 3a4b181.
Written before implementation. I wait for the owner's answer before step (c).

Words are the ones in `CONTEXT.md`: face, Reduce (code says `compress`), Quant, Diag,
implicit axis, plan, quality = grad-cosine.

---

## 1. The problem, restated

`jacve(..., sparse_representation=False)` runs the same `SparseTensor` contractions as
`sparse_representation=True`. The flag changes the RETURN form only. `core.py:3199` appends
the tensor as it is. `core.py:3215` calls `.dense()` on it. So the two settings are the
same contractions packed two ways. Finding 61 verdict 4 records this. The toy of section 6
confirms it again. On the 2-layer MLP the two are BIT-IDENTICAL on the exact plan and on
every approximated plan I ran.

So an approximated plan has no value oracle today. `jax.grad` is an oracle for the exact
plan only. A plan with a Quant, Reduce or Diag face has nothing dense to compare against
except itself.

The fix is a second engine. Every edge is a plain `jnp` array. Every contraction is a plain
dense contraction. The three approximations are defined on arrays. Sparse against dense on
the same plan is then a real value check. Oracle B of ticket .63 then becomes meaningful.

## 2. Where the initial elemental Jacobians are built

Three sites, all reached from `_build_graph` (`core.py:2445`):

* `multi_output_elemental_only_rules` (`core.py:2607-2624`). The elemental is computed
  eagerly and stored as a `SparseTensor`.
* `elemental_only_rules` (`core.py:2626-2665`). The elemental is deferred into a `LazyEdge`
  thunk (`core.py:259`) and forced when the edge is first consumed.
* the fallback `elemental_rules` path (`core.py:2666-2683`).

Every one of them produces a `SparseTensor`. The shapes come from the primitives' partial
rules under `src/graphax/primitives/`. `make_parallel_jacobian` (`primitives/base.py:37`)
builds the diagonal and broadcast-singleton forms. The structural primitives
(`transforms.py`, `indexing.py`, `reductions.py`) build a tensor plus a QUEUE of pre- and
post-transforms. A later `_drain_transforms` (`core.py:804`) folds that queue into the data.

**Decision: do not touch the partial rules.** The dense mode calls `_build_graph` unchanged.
It then densifies each edge once, at graph-build time:

```python
arr = _drain_transforms(_force(edge).copy(), post_first=False).dense()
```

`_drain_transforms` resolves the relabel queue. `.dense()` materialises the array. This is
one line. It covers reshape, transpose, slice, concatenate, reductions, broadcasts and the
repeated-operand (`x*x`) sum for free. Section 6.5 verifies it on ten targets. A rewrite of
the roughly 40 partial rules to emit arrays is a far larger change. It also needs a second
definition of every elemental, and the definition is not what this oracle tests.

Draining order does not matter for a single edge. A `pre_transform` relabels the primal
side. A `post_transform` relabels the out side. The two act on disjoint axes. I use the
final-output order (`post_first=False`), the same one `core.py:3191` uses.

## 3. What a dense edge is

**A dense edge from variable `u` to variable `v` is a `jnp` array of shape
`v.aval.shape + u.aval.shape`.** No wrapper. No metadata.

The out/primal split is not stored on the array. It does not need to be. It is
`v.aval.ndim`, and the elimination always knows `u` and `v`. This is the same invariant the
exact-AD assert already states (`core.py:2084-2087`,
`edge_shape = out_edge.aval.shape + in_edge.aval.shape`).

The invariant holds through the WHOLE dense elimination, approximated plans included. Each
approximation keeps the logical shape:

| approximation | on a dense edge |
| --- | --- |
| Quant | `arr.astype(dtype)`. Shape kept, dtype narrowed. |
| Reduce | the configured kind (default mean) over the axis, broadcast back to full extent. Shape kept. |
| Diag(i, j, f) | zero everything outside the block-diagonal blocks of the (i, j) pair. Shape kept. |

So the dense mode asserts the nominal shape at every store. That assert is the mode's own
correctness gate, and it is cheap.

Cost: a dense edge is `out_size * primal_size` numbers. This is the mode's hard limit. See
section 8.

## 4. The face hooks: a thin adapter, not dense-native hooks

**Proposal: a thin adapter.** Wrap the array as a fully dense `SparseTensor`: one
`DenseIndex` per axis, `axis == position`, the out/primal split at `v.aval.ndim`. Hand that
to the hook. Unwrap with `st.dense(keep_quantization=True)`.

```python
def _wrap(arr, out_ndim):
    dims = tuple(DenseIndex(i, int(s), i) for i, s in enumerate(arr.shape))
    return SparseTensor(dims[:out_ndim], dims[out_ndim:], arr, check_consistency=False)

def _unwrap(st):
    return st.dense(keep_quantization=True)
```

Three reasons for the adapter instead of dense-native hooks.

1. **The unwrap already IS the specified dense semantics.** `apply_compress`
   (`micro_actions.py:794`) reduces the physical axis and marks the Index implicit.
   `.dense()` broadcasts it back. That is exactly "the mean over the axis, broadcast back to
   the full extent". `apply_diag` (`micro_actions.py:511`) builds the `DiagonalIndex` pair.
   `.dense()` re-embeds it with zeros off the blocks. That is exactly "zeroing everything
   outside the block-diagonal blocks". `apply_quant` (`micro_actions.py:1039`) casts the
   buffer. `keep_quantization=True` keeps the narrow dtype instead of promoting it back.
   Verified in section 6.
2. **One legality predicate, not two.** alphagrad's hooks are choosers, not plain
   transforms (`common/masks.py::make_live_masked_hook`). They call `hook_rule_is_legal`,
   `project_rule_to_face` and `compress_to_graphax` on the live operand. Dense-native hooks
   need a second copy of all of that in alphagrad. Two copies drift. After a drift a plan is
   legal on one side and illegal on the other. The two sides then stop running the same
   plan. That is exactly what an oracle must not do.
3. **The shared code is the definition of the approximation, not the accumulation.** The
   thing under test is the contraction chain, and that stays fully independent. State this
   plainly in the finding when the mode lands. The oracle is independent in the
   accumulation and shared in the definition.

On a fully dense wrap, `canonical_axis_order` (`micro_actions.py:757`) gives slot `k` equal
to physical axis `k`. So alphagrad's physical-to-canonical `Compress` boundary is the
identity there and needs no special case.

The adapter also accepts a hook that returns a plain array. A dense-native hook stays
possible later without an engine change.

**The hooks do not always choose the same action on the two sides.** Measured, section 6.4.
On the MLP, `make_live_masked_hook` on every face applied this:

| rule | sparse | dense |
| --- | --- | --- |
| Quant bf16, slot lhs | 11 applied, 0 skipped | 11 applied, 0 skipped |
| Reduce mean axis 0, slot lhs | 8 applied, 3 skipped | 10 applied, 1 skipped |
| Diag(0, 2, 4), slot res | 2 applied, 9 skipped | 6 applied, 5 skipped |

The cause is structural. A sparse edge already carries `DiagonalIndex` pairs and implicit
axes from the primitives. A dense edge carries none. So their legal sets differ. On Diag the
four extra dense applications were value no-ops, and the values still agreed to 3.9e-8. On
Reduce they were not no-ops. The gap was 2.7 relative, a genuinely different plan, not a
defect.

**Consequence for oracle B, and a proposal.** A value comparison is meaningful only when
both sides applied the same actions. So:

* The dense mode records every dispatched micro-action through the always-on `TransformLog`
  (`sparse/tracer.py:186`), using `core._record_micro` unchanged. Both engines then produce
  the same census format.
* Oracle B compares the two censuses FIRST. It compares values only when the censuses match.
  A census mismatch is reported as "not comparable", never as a defect.
* The sweep tool hands the oracle literal micro-actions, that is already-decoded `Diag`,
  `Compress` and `Quant` per slot, not the masked chooser. A mismatch is then rare.
* The dense mode takes `on_illegal={"skip","raise"}`, default `raise`. A literal action that
  does not fit is then loud. The sparse path keeps its documented silent skip
  (`_apply_face_transform`, `except ValueError`). So the census comparison stays necessary.

## 5. Join, contraction and boundary in dense mode

**Contraction.** `post` is the `central -> out_edge` edge, of shape
`out.shape + central.shape`. `pre` is the `in_edge -> central` edge, of shape
`central.shape + in.shape`. The contraction runs over the eliminated variable's axes:

```python
contract = jnp.tensordot(post, pre, axes=central.aval.ndim)
```

**One correction to the ticket.** The ticket names "the `dense_dense` path of
ops/matmul.py::matmul". That path (`matmul.py:2697-2699`) is `jnp.matmul`, a MATRIX product
with numpy broadcasting. It is right only when the eliminated variable has exactly one axis
and each side has one. Vertices here have 0, 1, 2 or 3 axes. `jnp.tensordot` over
`central.aval.ndim` is the correct general form. It reduces to `jnp.matmul` in the 2-D case.
For a rank-0 vertex it becomes an outer product, which is also the .68 ruling
`X @ scalar == scalar * X`, because `tensordot(..., axes=0)` is the scale.
**Recommendation: call `jnp.tensordot` directly in the dense engine and leave
`ops/matmul.py` alone.** A generalised `dense_dense` changes a path the sparse engine can
reach, and it needs its own regression. Owner call.

**Join.** `edge = edge + contract`, a plain `+` on two arrays of the same shape, asserted
equal. Same site as the sparse `+` (`core.py:2102`). Same two-op hook order: `jl` on the new
contribution, `jr` on the existing edge, `jres` on the sum. Same merge-free-face rule from
`_unpack_face_slots` (`core.py:1216`).

**Boundary.** No `.dense()`, and no layout contract. The edges already ARE arrays in
parameter layout. The output collection returns `D[invar][outvar]`, or
`jnp.zeros(out.shape + in.shape)` for an absent edge, in the same outvar-major and
invar-minor order as `core.py:3199-3240`.

**Dtype.** `jnp` promotion decides. Verified in section 6.3. With Quant on ONE operand the
contraction is f32, so the engine never narrows on its own. With Quant on BOTH operands
every contraction and every returned gradient stays bf16. That is the `CONTEXT.md` ruling of
2026-09-06, satisfied without a special case.

## 6. Evidence (deliverable (b), already run)

Toy: `.scratch-race/t69_dense_toy.py`, a standalone prototype of the whole mode. CPU, local,
under a minute. Target: the 2-layer MLP of
`tests/core/sparse_tensor/output_layout_test.py`, static minimum Markowitz degree order, 11
live faces.

### 6.1 Sparse against its own dense return form: bit-identical

For every plan below, `sparse_representation=True` against `False` is bit-identical, rel_l2
exactly 0. That flag is not an oracle. This repeats finding 61 verdict 4.

### 6.2 Dense mode against jax.grad and against the sparse engine

| plan | sparse vs jax.grad | dense vs jax.grad | dense vs sparse |
| --- | --- | --- | --- |
| exact | 1.3e-07 | 9.3e-08 | 1.1e-07 |
| Quant bf16, slot lhs, 1 face | 1.255e-03 | 1.255e-03 | 1.5e-07 |
| Reduce mean, slot lhs, 1 face | 5.700e-01 | 5.700e-01 | 1.2e-07 |
| Diag(0,2,4), slot res, 1 face | 6.268e-01 | 6.268e-01 | 5.4e-08 |
| all three, on three faces | 7.837e-01 | 7.837e-01 | 5.0e-08 |
| Quant bf16, slot lhs, every face | 2.931e-03 | 2.931e-03 | 5.3e-08 |

Read it as the ticket asks. The dense result differs from `jax.grad` BY THE APPROXIMATION:
1e-3 for Quant, 0.57 to 0.78 for Reduce and Diag. On the exact plan both engines equal
`jax.grad` at 1e-7. The dense-against-sparse column is float32 reduction-order noise. So on
this toy the sparse engine is confirmed correct on approximated plans for the first time.

### 6.3 The oracle has resolution

Quant bf16 on BOTH operands of every face gives dense against sparse **1.8e-03**, with both
sides 1e-3 from `jax.grad`. That is the "bf16 rounds different intermediates" family.
Finding 61 verdict 4 measured it at 5e-3 to 1.2e-2 between the two sparse engines. So the
tolerance of oracle B is set PER CLASS: about 1e-6 for exact, Reduce and Diag, about 1e-2
for Quant.

### 6.4 The masked-hook divergence

See the table in section 4. This is the one real obstacle. Section 4 proposes the fix.

### 6.5 The build-and-densify step is general

Dense mode against `jax.grad`, exact plan, ten targets. reshape 1.4e-07, transpose 0.0,
slice 9.9e-08, concatenate 1.4e-07, sum over an axis 8.7e-08, broadcast 1.1e-07, `x*x` 0.0,
MLP on Markowitz 9.3e-08, MLP reverse 8.6e-08, MLP forward 9.9e-08. It is also correct under
`jax.jit`, at 1.3e-07.

## 7. The argument name

**Proposal: `jacve(..., dense_edges=True)`.** It names what changes. Every edge is a plain
array. `sparse_representation` names the RETURN form and keeps that meaning.

How the two relate:

* `dense_edges=True` implies plain-array outputs. No `SparseTensor` is left to return.
* `dense_edges=True` with `sparse_representation=True` raises `ValueError`. A silently
  ignored return form is how a measurement lies.
* `dense_edges=True` with `sparse_representation=False` is accepted. It is the only
  combination, and `False` is the default.
* `dense_edges=False` is today's behaviour, reached by exactly today's code.

Alternatives considered. `engine={"sparse","dense"}` is better long-term. Ticket .65 and .28
are about to redefine the word "engine" for the tiled and planner merge, so the collision
is confusing. `dense_contraction=True` is accurate but says less, because the edges are what
changes and the contraction follows.

`dense_edges` is also carried through `vertex_elimination_jaxpr` (`core.py:3042`). It joins
the topology cache key (`core.py:3325`) beside `sparse_representation`.

## 8. What the mode is NOT

* **Not a measurement path.** A dense edge is `out_size * primal_size` numbers. On TLM at
  the campaign shape that is out of reach by orders of magnitude. It is a small-shape value
  oracle only. The implementation raises when the total dense-edge bytes exceed a budget.
  Proposed default 2 GiB, as an argument. It never OOMs a node.
  **Open question. Oracle B of .63 asks for sparse against dense on a handful of TLM plans.
  At the campaign shape that does not fit. Reduced shape, or NeuralNetwork instead of TLM?**
* **Not a cost model.** `count_ops=True` with `dense_edges=True` raises
  `NotImplementedError`. The counts describe the oracle, not the engine.
* **Not on any training path.** Sweep tool only, as owner Q21 already rules for oracle B.

## 9. Where the code goes, and how big it is

A new module, `src/graphax/dense_edges.py`, plus a thin route in `core.py`. NOT a flag
threaded through `_eliminate_vertex` (`core.py:1485`, about 830 lines carrying lazy edges,
transform queues, the reconciler peel, demand-emit, deferred output products, op counting,
path sinks and block-storage materialisation). Two reasons.

1. **Risk.** Threading dense through it means about twenty branch points inside the code the
   whole project runs. The rule is minimal diffs and a flag-off bit-identity test. A separate
   module makes flag-off identity true BY CONSTRUCTION, because no existing line changes.
2. **The oracle must be independent.** An oracle that shares the contraction code with the
   engine proves only the packing. That is finding 61 verdict 4, the mistake this ticket
   exists to fix. Reuse of `_eliminate_vertex` repeats it.

What IS shared, deliberately: `_build_graph`, `_prune_graph`, `_checkify_order`, `_vidx_for`
and `_stable_var_index`, so both engines see the same graph, the same face keys and the same
order. Also `_unpack_face_slots` and `_record_micro`, for the same plan format and the same
census. Also the three `apply_*` micro-actions through the adapter, for the same definition
of each approximation.

Estimated size:

| file | lines |
| --- | --- |
| `src/graphax/dense_edges.py` (new): elimination 90, adapter 35, build and densify 40, output 25, guards and errors 40 | 230 |
| `src/graphax/core.py`: the `dense_edges` argument on `jacve` and `vertex_elimination_jaxpr`, the combination check, the route, docstrings | 35 |
| `tests/core/dense_edges_test.py` (new): the section-6 matrix, the shape and dtype contracts, the census, the raises | 200 |
| total | **about 470 lines, no existing line changed** |

The prototype of section 6 already carries most of the 230. Work left after the go:
SKIP_FACE, the two-op face form, the per-vertex `transforms` list, `has_aux`, the byte
budget, the census wiring, the error paths, and the tests.

## 10. What I need from the orchestrator and the owner

1. `dense_edges=True` as the argument name, with `ValueError` on `dense_edges=True` plus
   `sparse_representation=True`. Agreed?
2. `jnp.tensordot` in a new module, with `ops/matmul.py`'s `dense_dense` left alone.
   Agreed? See section 5.
3. The masked-hook divergence: census compare first, then value compare, plus
   `on_illegal="raise"` for literal actions. Agreed? See section 4.
4. Oracle B on TLM at the campaign shape does not fit in memory. Reduced shape, or a smaller
   target? See section 8.
5. Per-class tolerances for oracle B: about 1e-6 for exact, Reduce and Diag, about 1e-2 for
   Quant. Agreed? See section 6.3.
