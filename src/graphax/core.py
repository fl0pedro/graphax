import functools
import os
import threading
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict, NamedTuple, Sequence, Set, Tuple, Union, cast

import immutables
import jax
import numpy as np
import jax._src.core as core
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax._src.core import ShapeDtypeStruct
from jax._src.pjit import jit_p
from jax._src.lax.control_flow.conditionals import cond_p
from jax._src.util import safe_map

from .jaxpr import VEJaxpr
from .primitives import (
    NO_EDGE,
    elemental_only_rules,
    elemental_rules,
    jit_name_rules,
    jit_named_elemental_only,
    multi_output_elemental_only_rules,
)
from .sparse.ops import add_w_counts
from .sparse.dtype_compute import _scaled_mul as _scaled_mul_promote
from .sparse.ops.matmul import matmul as sparse_matmul
from .sparse.ops.utils import (
    _is_approx,
    _squeeze_unreferenced_val_axes,
)
from .sparse.tensor import _assert_sparse_tensor_consistency
from .sparse.micro_actions import (
    Compress, Diag, Quant, apply_compress, apply_diag, apply_quant,
)
from .sparse.ops.join import FaceJoinPolicy
from .sparse.utils import zeros_like
from .sparse.tracer import (
    get_face_sink as _get_face_sink,
    get_transform_log as _get_transform_log,
)

EliminationOrder = Union[Sequence[int], str]
ComputationalGraph = Dict[core.Var, Dict[core.Var, jnp.ndarray]]


# Toggle caching of jaxpr-derived structures (e.g. computational graph). Set
# GX_ENABLE_CACHE=0 to disable caching when iterating on tracing logic or
# diagnosing graph-state corruption.
ENABLE_CACHE = os.environ.get("GX_ENABLE_CACHE", "1") != "0"


# ---------------------------------------------------------------------------
# STORED-BYTE ACCOUNTING for the accumulated Jacobians.
#
# WHAT IT COUNTS. Every accumulated-Jacobian edge this elimination WRITES back
# into the graph -- ``edge_outval`` at the single store site in
# ``_eliminate_vertex``, read AFTER ``_squeeze_unreferenced_val_axes`` so the
# number is the buffer that is really carried, not the pre-squeeze shape. A
# SKIPped face stores nothing and so contributes nothing, which is the point.
# The base elemental partials built by ``_build_graph`` are NOT counted: they
# are identical in an exact and an approximated run of the same order, so
# including them would only dilute a ratio between the two.
#
# WHY STORED BYTES AND NOT A DECLARED CLASS. A dim can DECLARE itself sparse
# and still materialize dense, and ``val is None`` declares nothing while
# storing nothing, so counting declared classes overstates sparsity in both
# directions. ``val.size * itemsize`` is what is actually allocated -- the same
# definition the offline structure audit uses. ``_structural_val_size`` is
# tallied beside it as the on-structure CELL count so a ``val is None`` edge is
# visible as structure instead of silently absent.
#
# WHEN IT FIRES. ``_eliminate_vertex`` is plain Python that runs while
# ``jacve`` is TRACED, so the tally fills once per trace and costs nothing at
# runtime. Callers must reset-and-read around ONE trace: alphagrad traces the
# same order three or four times per measurement (count pass, face-enum replay,
# tokenizer replay, the measured lower()), and an ungated sink would report
# several times the truth -- the trap ``masks.arm_face_counts`` documents.
#
# COST WHEN DISARMED: one dict lookup and a branch per edge store. The tally
# reads shapes, never values, and mutates no tensor, so an armed trace and a
# disarmed trace emit the same jaxpr.
_STORE_ACCT: Dict[str, int] = {
    "armed": 0, "bytes": 0, "cells": 0, "elems": 0, "edges": 0,
    "logical": 0,
    # ELIMINATION WALKS performed while armed. This is what says "the
    # tally is a measurement", NOT `edges`: an all-SKIP plan walks the
    # whole order and stores nothing, and a caller that read `edges == 0`
    # as "nothing walked" would refuse to record the very plan the
    # sparsity channel most needs to score.
    "walks": 0,
}


def arm_store_accounting() -> None:
    """Enter a scope whose edge stores ARE the tally. Depth-counted, so a
    nested scope cannot disarm an outer one."""
    _STORE_ACCT["armed"] += 1


def disarm_store_accounting() -> None:
    _STORE_ACCT["armed"] = max(0, _STORE_ACCT["armed"] - 1)


def store_accounting_armed() -> bool:
    return _STORE_ACCT["armed"] > 0


def reset_store_accounting() -> None:
    for _k in ("bytes", "cells", "elems", "edges", "logical", "walks"):
        _STORE_ACCT[_k] = 0


def store_accounting_totals() -> Dict[str, int]:
    """The tally since the last reset. ``bytes`` is the headline number."""
    return {_k: int(_v) for _k, _v in _STORE_ACCT.items() if _k != "armed"}


def _record_edge_store(t) -> None:
    """Tally ONE accumulated-Jacobian edge write. Inert unless armed."""
    if not _STORE_ACCT["armed"] or t is None:
        return
    if isinstance(t, DeferredOutputProduct):
        # #46 deferred outputs: the edge IS the factor pair, so the pair is
        # what is stored. Counting it as one unmeasurable edge would make a
        # FACTORED_OUTPUTS run look free.
        _record_edge_store(t.post)
        _record_edge_store(t.pre)
        return
    try:
        _v = getattr(t, "val", None)
        _it = int(np.dtype(t.dtype).itemsize)
        _n = int(_v.size) if _v is not None else 0
        _STORE_ACCT["elems"] += _n
        _STORE_ACCT["bytes"] += _n * _it
        _STORE_ACCT["cells"] += int(t._structural_val_size)
        _STORE_ACCT["logical"] += int(t.size)
    except Exception:
        # An edge we cannot measure must never kill an elimination. It is
        # still counted as an edge, so ``edges`` vs the measurable tallies
        # says how much of the graph the number actually covers.
        pass
    _STORE_ACCT["edges"] += 1


def _leaf_cache_key(leaf):
    """Hashable cache key for one pytree leaf.

    Array leaves are keyed by (shape, dtype, content-digest) rather than by
    ``id(leaf)``. Keying by object identity is unsound here: a cached result may
    bake the leaf's *values* into a value-dependent structure (e.g. the eager
    elemental edges in :func:`_build_graph`) while NOT retaining the leaf array
    itself. Once such a leaf is garbage-collected, a later array allocated at the
    same address reuses its ``id()`` -> the cache returns the stale, wrong-valued
    result. A content digest makes the key correct regardless of value
    dependence: distinct values yield distinct keys, identical values reuse the
    entry. ``_get_eliminator`` is consulted once per ``jacve`` call (and once at
    trace time under ``jax.jit``), so the O(size) digest is off the per-vertex
    hot path.
    """
    if hasattr(leaf, "shape"):
        # Abstract tracers have no concrete buffer; under jax.jit the leaf
        # values are deferred to runtime, so keying by (shape, dtype) is both
        # all we can do and correct (no concrete value is baked at trace time).
        if isinstance(leaf, core.Tracer):
            return (leaf.shape, str(leaf.dtype))
        # Concrete array: hash the raw buffer. ``hash(bytes)`` is a fast C
        # routine; combined with shape + dtype the collision probability is
        # negligible for caching.
        buf = np.asarray(leaf).tobytes()
        return (leaf.shape, str(leaf.dtype), hash(buf))
    # Non-array leaf: the key is the LEAF ITSELF, not ``hash(leaf)``. For
    # identity-hashed objects (``core.Jaxpr`` above all) ``hash(leaf)`` is
    # ``id(leaf)//16``, and CPython recycles addresses -- a later jaxpr
    # allocated where a dead one lived reused its key and the cache returned
    # the dead jaxpr's eliminator: a silently WRONG JACOBIAN, reproduced at
    # ~8/20 on tests/core/{named_activation,prune}_test.py. Keeping the leaf
    # in the key pins it (an entry's id can never be recycled while the entry
    # lives) and dict equality falls back to ``==`` -- identity for jaxprs,
    # value equality for ints/strings/argnums -- so a stale hit is impossible.
    return leaf


def pytree_hash_cache(maxsize: int | None = None):
    """Decorator that memoizes a function on its (args, kwargs) pytree content.

    Disabled at call-time when ``ENABLE_CACHE`` is False so the wrapped
    function is invoked directly without consulting the cache.
    """

    def decorator(func):
        cache: dict = {}
        lock = threading.Lock()
        pending: dict = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLE_CACHE:
                return func(*args, **kwargs)

            leaves, treedef = jtu.tree_flatten((args, kwargs))
            # Under an abstract trace (jax.jit / vmap) the leaves are tracers with
            # no concrete value, so they can only be keyed by (shape, dtype) —
            # which COLLIDES across distinct traces of equally-shaped inputs. A
            # hit would then return a cached result whose SparseTensor edges hold
            # a PRIOR trace's tracers, which escape the current trace
            # (UnexpectedTracerError). Always compute fresh while tracing; jit's
            # own compilation cache already memoizes repeated calls.
            if any(isinstance(leaf, core.Tracer) for leaf in leaves):
                return func(*args, **kwargs)
            leaf_hashes = tuple(_leaf_cache_key(leaf) for leaf in leaves)
            # The TUPLE is the key, never ``hash(tuple)``: an int key gives
            # the dict no ``==`` to fall back on, so any hash collision --
            # including the id-reuse one _leaf_cache_key documents -- returned
            # a stale entry instead of missing.
            key = (treedef, leaf_hashes)

            must_compute = False
            event = None
            with lock:
                if key in cache:
                    return cache[key]
                if key in pending:
                    event = pending[key]
                else:
                    event = threading.Event()
                    pending[key] = event
                    must_compute = True

            if not must_compute:
                event.wait()
                with lock:
                    return cache[key]

            try:
                result = func(*args, **kwargs)
                with lock:
                    if maxsize is not None and len(cache) >= maxsize:
                        cache.pop(next(iter(cache)))
                    cache[key] = result
                return result
            finally:
                with lock:
                    pending.pop(key, None)
                event.set()

        return wrapper

    return decorator


_UNSET = (
    object()
)  # sentinel: distinguishes "thunk not yet run" from "thunk returned None"


class LazyEdge:
    """Deferred SparseTensor — evaluated on first read.

    Both ``graph[u][v]`` and ``transpose_graph[v][u]`` store the *same*
    ``LazyEdge`` object for a given edge, so the underlying thunk fires at
    most once regardless of which direction reads it first.

    If the thunk returns ``None`` (e.g. for stop_gradient inputs where there
    is no Jacobian), ``value`` is ``None`` and callers must guard accordingly.
    """

    __slots__ = ("_thunk", "_value")

    def __init__(self, thunk):
        self._thunk = thunk
        self._value = _UNSET  # cached after first evaluation

    @property
    def value(self):
        if self._value is _UNSET:
            self._value = self._thunk()
        return self._value


def _force(edge):
    """Return the concrete SparseTensor, evaluating a LazyEdge if necessary."""
    return edge.value if isinstance(edge, LazyEdge) else edge


def _jit_kept_as_vertex(eqn) -> bool:
    """A jit is kept as a vertex (NOT inlined) iff graphax has its own Jacobian
    for its name. This single predicate ties the two halves of the invariant
    together: the inline gate (``_call_body``) leaves these eqns in place, and the
    jit elemental rule (``_make_jit_elemental_rule``) dispatches them by name.
    Every other jit is inlined into the parent graph."""
    return eqn.primitive is jit_p and eqn.params.get("name") in jit_name_rules


def _has_recursive_macro_vertex(jaxpr) -> bool:
    """True if the jaxpr contains a value-dependent macro-vertex kept AS a vertex
    — cond/switch (differentiates the taken branch) or a named jit (name-keyed
    Jacobian). Their Jacobians depend on runtime values, so such a top-level
    jaxpr must bypass the id-keyed eliminator cache (like inlined jaxprs) to
    avoid returning a stale eliminator built for a different, GC'd function whose
    object id was reused."""
    return any(eqn.primitive is cond_p or _jit_kept_as_vertex(eqn)
               for eqn in jaxpr.eqns)


def inline_call_primitives(jaxpr, consts):
    """Public alias of :func:`_inline_call_primitives`.

    Anything that numbers vertices for a jaxpr MUST number them on the inlined
    form, because that is what ``jacve`` / the AOJ actually eliminate: they
    splice jit/pjit bodies in, which adds equations. An order built from the
    raw ``jax.make_jaxpr`` output therefore addresses the wrong vertices and
    leaves the spliced-in ones un-eliminated -- which the elimination then
    rejects rather than silently returning a Jacobian missing every path
    through them. Callers that tokenize a jaxpr or build an elimination order
    should run it through here first. Returns ``(new_jaxpr, new_consts)``.
    """
    return _inline_call_primitives(jaxpr, consts)


def _inline_call_primitives(jaxpr, consts):
    """Splice jit/pjit (and nested) bodies into the parent jaxpr so vertex
    elimination only ever sees primitives with registered elemental rules.

    This is the PRIMARY path for jit: its body is a closed sub-jaxpr — pure
    structural plumbing — so we re-point the body's invars onto the outer call
    args, recurse into the body's equations, and re-point the outer outvars onto
    the body's results. Inlining composes far better than the *macro-vertex*
    alternative (recursively differentiating the body and re-wrapping its
    Jacobian, which mis-shaped the inner Jacobian during the OUTER composition —
    e.g. a (6,6,6) -> (6,6) reshape for ``jax.nn`` activations).

    The ONE exception is a jit graphax has its own Jacobian for
    (``_jit_kept_as_vertex`` — jax.nn.elu/selu/...): that one is kept as a vertex
    so the named rule can apply the exact analytic Jacobian (the macro-vertex
    survives only as that rule's non-default-args fallback). Returns ``(new_jaxpr,
    new_consts)``; inner consts are hoisted to top-level constvars. A no-op (the
    original objects) when there are no call primitives to inline.

    Inner bodies are renamed into FRESH variables per call site: JAX reuses the
    *same* inner-jaxpr object (hence the same ``Var`` instances) for every call of
    a given jitted/custom function, so a single global substitution keyed by those
    shared Vars would let a second call site overwrite the first's mapping and
    silently corrupt the graph (e.g. a shared MLP block applied twice). ``gensym``
    gives every inlined Var a fresh identity local to its call site."""
    has_call = [False]
    newvar = core.gensym()

    new_eqns = []
    constvars = list(jaxpr.constvars)
    new_consts = list(consts)

    def _call_body(eqn):
        """The closed inner jaxpr to inline for this eqn, else None.

        Only ordinary jits are inlined. custom_jvp / custom_vjp are NOT inlined:
        their bespoke jvp/bwd rules can differ from the primal decomposition's
        derivative (kink subgradients, straight-through / surrogate gradients),
        so each is handled by a dedicated elemental rule that HONORS the user
        rule (``custom_jvp_elemental_only`` / ``custom_vjp_elemental_only`` in
        primitives/custom.py). A named jit graphax has its own Jacobian for
        (``_jit_kept_as_vertex``) is likewise NOT inlined — it is dispatched by
        name by the jit elemental rule."""
        if eqn.primitive is jit_p and not _jit_kept_as_vertex(eqn):
            return eqn.params["jaxpr"]
        return None

    def process(eqns, env, fresh):
        """Emit ``eqns`` with vars resolved through ``env``. ``fresh`` is True for
        equations that came from an inlined body — their outvars are renamed to
        fresh Vars so repeated call sites of the same shared inner jaxpr never
        collide. Top-level equations (``fresh=False``) keep their already-unique
        Vars untouched."""
        def resolve(v):
            if isinstance(v, core.Literal):
                return v
            return env.get(v, v)

        for eqn in eqns:
            inner_closed = _call_body(eqn)
            if inner_closed is not None:
                has_call[0] = True
                inner = inner_closed.jaxpr
                # Each inlining gets its own child scope, seeded with the outer
                # env so inner invars can resolve onto outer call args.
                child = dict(env)
                for cv, cval in zip(inner.constvars, inner_closed.consts):
                    nv = newvar(cv.aval)
                    child[cv] = nv
                    constvars.append(nv)
                    new_consts.append(cval)
                for iv, ov in zip(inner.invars, eqn.invars):
                    child[iv] = resolve(ov)
                process(inner.eqns, child, True)
                for oov, iov in zip(eqn.outvars, inner.outvars):
                    if isinstance(oov, core.DropVar):
                        continue
                    env[oov] = iov if isinstance(iov, core.Literal) else child.get(iov, iov)
            elif fresh:
                new_invars = [resolve(v) for v in eqn.invars]
                new_outvars = []
                for ov in eqn.outvars:
                    if isinstance(ov, core.DropVar):
                        new_outvars.append(ov)
                    else:
                        nv = newvar(ov.aval)
                        env[ov] = nv
                        new_outvars.append(nv)
                new_eqns.append(eqn.replace(invars=new_invars, outvars=new_outvars))
            else:
                new_eqns.append(eqn.replace(invars=[resolve(v) for v in eqn.invars]))

    top_env: Dict[Any, Any] = {}
    process(jaxpr.eqns, top_env, False)
    if not has_call[0]:
        return jaxpr, consts

    def resolve_out(v):
        return v if isinstance(v, core.Literal) else top_env.get(v, v)

    new_outvars = [resolve_out(v) for v in jaxpr.outvars]
    new_jaxpr = jaxpr.replace(constvars=constvars, eqns=new_eqns, outvars=new_outvars)
    return new_jaxpr, new_consts


def jacve(
    fun: Callable,
    order: EliminationOrder,
    argnums: Sequence[int] = (0,),
    has_aux: bool = False,
    count_ops: bool = False,
    sparse_representation: bool = False,
    dense_edges: bool = False,
    dense_max_bytes: int = None,
    transforms: Sequence[
        Tuple[
            int,
            Sequence[Union[Diag, Compress, Callable[["SparseTensor"], "SparseTensor"]]],
        ]
    ] = None,
    face_transforms: dict = None,
) -> Callable:
    """
    Jacobian `fun` with respect to the `argnums` using the vertex elimination method.
    The vertex elimination order can be specified as a sequence of integers or
    as a string "forward" or "fwd" for forward elimination and "reverse" or
    "rev" for reverse elimination. The forward order basically corresponds to
    the elimination order [1, 2, 3, ...] while the reverse order corresponds to
    [..., 3, 2, 1]. For custom orders, just pass the sequence of  integers in
    the desired order. Additionally, the `count_ops` flag can be set
    to True to count the number of multiplications and additions during the
    elimination process, i.e. the Jacobian accumulation. The
    `sparse_representation` flag can be set to `True` to return the Jacobian in
    a sparse representation using the `SparseTensor` class.

    Args:
        fun (Callable): Function to differentiate.
        order (Union[Sequence[int], str]): Vertex elimination order. Either pass
            the desired order directly or specify a string. Allows options are
            "forward", "fwd", "reverse" and "rev".
        argnums (Sequence[int], optional): Argument numbers to differentiate
                                            with respect to. Defaults to (0,).
        has_aux (bool): _description_
        count_ops (bool, optional): Track adds/muls/fmas/peak-mem during the
                                    elimination. When True, the returned
                                    callable yields ``(jacobian, aux)`` (with
                                    ``aux`` being a dict of counts) instead of
                                    just ``jacobian``. Defaults to False.
        sparse_representation (bool, optional): Return the Jacobian in a sparse
                                            representation. Defaults to `False`.
        dense_edges (bool, optional): Run the DENSE-CONTRACTION mode of
            :mod:`graphax.dense_edges` instead of the sparse engine: every edge
            is a plain array of shape ``out_shape + primal_shape`` and every
            contraction is a ``jnp.tensordot`` over the eliminated variable's
            axes. This is the VALUE ORACLE for an approximated plan
            (ticket dsnn-3qm.69) -- ``sparse_representation`` changes the return
            form only, so it cannot serve as one. Independent of
            ``sparse_representation``, whose ``True`` setting has nothing to
            return here and therefore raises. Not a measurement path: a dense
            edge is ``out_size * primal_size`` numbers. Defaults to `False`.
        dense_max_bytes (int, optional): Ceiling on the total bytes of the live
            dense edges under ``dense_edges=True``; ``None`` takes
            :data:`graphax.dense_edges.DEFAULT_MAX_BYTES` (2 GiB). Over the
            ceiling raises instead of letting the job be killed.
        face_transforms (dict, optional): PER-FACE approximations, as a
            mapping with TWO levels::

                {vertex: {face_key: (lhs, rhs, res)}}

            The outer key is the 1-based vertex; the inner key is the pair
            :func:`faces_of` returns for that vertex, enumerated on the graph
            state IMMEDIATELY before that vertex is eliminated. A flat
            ``{face_key: slots}`` dict is the natural mistake -- it is what
            ``faces_of`` hands you -- and it matches no vertex, so it RAISES
            here rather than dropping every approximation in silence. A
            request that matches no face raises as well, after the
            elimination. The per-vertex ``transforms`` argument also accepts a
            dict per vertex, but that one is keyed by the ELIMINATOR's own
            ``(in_eqn_id, out_eqn_id)`` pair, which is a different index;
            ``faces_of``'s keys belong to ``face_transforms``.

    Returns:
        Callable: The function that returns the Jacobian of `fun`.
    """
    if dense_edges and sparse_representation:
        raise ValueError(
            "dense_edges=True with sparse_representation=True: the dense mode "
            "has no SparseTensor to return, every edge is a plain array. "
            "Silently ignoring the requested return form is how a measurement "
            "lies, so this combination raises. Drop sparse_representation.")

    @wraps(fun)
    def jacfun(*args, **kwargs):
        # TODO Make repackaging work properly with one input value only
        flattened_args, in_tree = jtu.tree_flatten(args)
        closed_jaxpr = jax.make_jaxpr(fun)(*flattened_args, **kwargs)
        inlined_jaxpr, inlined_consts = _inline_call_primitives(
            closed_jaxpr.jaxpr, closed_jaxpr.literals
        )
        # Bypass the id-keyed eliminator cache when the jaxpr is a fresh inlined
        # object, OR contains a value-dependent macro-vertex (cond / named jit) —
        # both are exposed to GC id-reuse staleness (see vertex_elimination_jaxpr).
        was_inlined = (inlined_jaxpr is not closed_jaxpr.jaxpr
                       or _has_recursive_macro_vertex(inlined_jaxpr))

        out = vertex_elimination_jaxpr(
            inlined_jaxpr,
            order,
            inlined_consts,
            *args,
            has_aux=has_aux,
            argnums=argnums,
            count_ops=count_ops,
            sparse_representation=sparse_representation,
            dense_edges=dense_edges,
            dense_max_bytes=dense_max_bytes,
            fresh_eliminator=was_inlined,
            transforms=transforms,
            face_transforms=face_transforms,
        )

        # When count_ops is True, vertex_elimination_jaxpr returns (out, aux).
        aux_data: dict = {}
        if count_ops:
            out, aux_data = out

        if has_aux:
            primal_out, grads = out
            out_tree = jtu.tree_structure(tuple(closed_jaxpr.jaxpr.outvars))
            if (
                len(closed_jaxpr.jaxpr.outvars) == 1
                and len(closed_jaxpr.jaxpr.invars) > 1
            ):
                res = (primal_out[0], grads[0])
            else:
                res = (
                    jtu.tree_unflatten(out_tree, primal_out),
                    jtu.tree_unflatten(out_tree, grads),
                )
        else:
            out_tree = jtu.tree_structure(tuple(closed_jaxpr.jaxpr.outvars))
            if (
                len(closed_jaxpr.jaxpr.outvars) == 1
                and len(closed_jaxpr.jaxpr.invars) > 1
            ):
                res = out[0]
            else:
                res = jtu.tree_unflatten(out_tree, out)

        if count_ops:
            return res, aux_data
        return res

    return jacfun


# Per-vertex Jacobian-transform spec — the shared type of the ``transforms``
# argument on jacve / grad / value_and_grad. Two per-vertex forms:
#   * legacy list ``[(vertex_id, [transform, ...])]`` — transforms applied to
#     each merged edge (normalized to nominal shape).
#   * face-like per-path dict ``[(vertex_id, {(primal_id, out_id): hooks})]`` —
#     ``hooks`` is ``(pre, post, new)`` [contraction] or
#     ``((pre, post, new), (lhs, rhs, res))`` [contraction + join]; each hook is a
#     callable ``SparseTensor -> SparseTensor`` or ``None``. Applied at the op
#     boundary, so the edge is never normalized (the sparse ops reconcile shapes).
_Hook = Callable[["SparseTensor"], "SparseTensor"]
_PathHooks = Union[
    Tuple[_Hook, _Hook, _Hook],
    Tuple[Tuple[_Hook, _Hook, _Hook], Tuple[_Hook, _Hook, _Hook]],
]
TransformSpec = Sequence[
    Tuple[int, Union[
        Sequence[Union[Diag, Compress, _Hook]],
        Dict[Tuple[int, int], _PathHooks],
    ]]
]


def _leaf_dtype(x):
    return x.dtype if hasattr(x, "dtype") else jnp.result_type(x)


def _validate_grad_io(fun, args, kwargs, argnums):
    """jax.grad-style validation for grad / value_and_grad. Fail LOUDLY and
    accurately — rather than crash deep in vertex elimination or silently return
    a wrong gradient — for anything jax.grad refuses OR that jacve cannot yet
    handle:

    * keyword arguments to ``fun`` (vertex elimination threads only positional
      args, so a kwarg would mismatch the traced invars);
    * pytree / multi-leaf inputs, e.g. a dict of parameters (not yet supported —
      jacve flattens args into positional invars);
    * non-floating *differentiated* inputs (integer / complex);
    * a non-scalar, or non-floating (integer / complex), output.

    The output check uses ``jax.eval_shape`` (abstract — no FLOPs) and is
    jit-trace-safe. Complex differentiation is rejected outright (graphax does
    not implement the holomorphic/conjugation handling jax.grad gates behind
    ``holomorphic=True``)."""
    if kwargs:
        raise TypeError(
            "graphax.grad / value_and_grad do not accept keyword arguments to "
            f"the differentiated function (got {sorted(kwargs)}). Vertex "
            "elimination threads only positional args; bind keyword args with "
            "functools.partial or pass them positionally."
        )
    argnums = set(argnums)
    for j, arg in enumerate(args):
        leaves = jtu.tree_leaves(arg)
        if len(leaves) != 1:
            raise NotImplementedError(
                f"graphax.grad: argument {j} is a pytree with {len(leaves)} "
                "leaves; pytree inputs (e.g. a dict of parameters) are not yet "
                "supported. Flatten with jax.tree_util and differentiate the "
                "leaves as positional arguments."
            )
        if j in argnums and not jnp.issubdtype(_leaf_dtype(leaves[0]), jnp.floating):
            raise TypeError(
                "grad requires real (floating) inputs, but the differentiated "
                f"argument {j} has dtype {_leaf_dtype(leaves[0])} "
                "(integer / complex differentiation is not supported)."
            )
    leaves = jtu.tree_leaves(jax.eval_shape(fun, *args))
    if len(leaves) != 1 or leaves[0].shape != ():
        shapes = [getattr(l, "shape", "?") for l in leaves]
        raise TypeError(
            "Gradient only defined for scalar-output functions; the output had "
            f"{len(leaves)} leaves with shapes {shapes}."
        )
    if not jnp.issubdtype(leaves[0].dtype, jnp.floating):
        raise TypeError(
            "grad requires a real (floating) scalar output, but the output "
            f"dtype was {leaves[0].dtype}. Complex output would need holomorphic "
            "differentiation (unsupported); integer output is not differentiable."
        )


def _unwrap_single(x):
    """jacve packs single-output / single-argnum results as a length-1
    tuple-or-list in some arities; normalize to the bare value (jax.grad
    convention for an int ``argnums`` and for the single scalar primal)."""
    if isinstance(x, (tuple, list)) and len(x) == 1:
        return x[0]
    return x


def _normalize_grads(out, scalar_argnums):
    """Match jax.grad argnums conventions: an int ``argnums`` yields the bare
    gradient; a *sequence* ``argnums`` always yields a tuple. jacve unwraps a
    length-1 sequence (and single-arg functions) inconsistently, so re-impose
    the convention here."""
    if scalar_argnums:
        return _unwrap_single(out)
    return out if isinstance(out, tuple) else (out,)


def _make_grad(fun, order, argnums, count_ops, transforms, *, return_value):
    """Shared implementation of :func:`grad` (``return_value=False``) and
    :func:`value_and_grad` (``return_value=True``). The latter rides jacve's
    ``has_aux=True`` path, which returns ``(primal_outputs, jacobians)`` from the
    same elimination pass — no second forward evaluation."""
    scalar_argnums = isinstance(argnums, int)
    _argnums = (argnums,) if scalar_argnums else tuple(argnums)
    jac_fn = jacve(
        fun, order, _argnums, has_aux=return_value,
        count_ops=count_ops, transforms=transforms,
    )

    @wraps(fun)
    def diff_fn(*args, **kwargs):
        _validate_grad_io(fun, args, kwargs, _argnums)
        out = jac_fn(*args)
        aux_data = None
        if count_ops:
            out, aux_data = out
        if return_value:
            primal, grads = out
            result = (_unwrap_single(primal), _normalize_grads(grads, scalar_argnums))
        else:
            result = _normalize_grads(out, scalar_argnums)
        return (result, aux_data) if count_ops else result

    return diff_fn


def grad(
    fun: Callable,
    order: EliminationOrder = "rev",
    argnums: Union[int, Sequence[int]] = 0,
    count_ops: bool = False,
    transforms: TransformSpec = None,
) -> Callable:
    """``jax.grad`` analogue computed by vertex elimination (a thin wrapper
    over :func:`jacve` for scalar-output functions).

    Beyond ``jax.grad``'s single reverse-mode VJP, this forwards the
    vertex-elimination degrees of freedom to :func:`jacve`:

    * ``order`` — ``"rev"`` reproduces classical reverse-mode / jax.grad;
      ``"fwd"`` forward elimination; an explicit vertex sequence gives
      cross-country / partial elimination. For an EXACT gradient the order is
      value-invariant (it changes only FLOP/memory cost; ``"rev"`` is near
      optimal for a scalar output).
    * ``transforms`` — per-vertex Jacobian transforms (``Diag`` / ``Compress`` /
      callable) applied DURING elimination: structured gradient *approximations*
      jax.grad cannot express (the result is then inexact by construction).
    * ``count_ops`` — when True the callable returns ``(grads, aux)`` with the
      adds/muls/fmas/peak-mem accounting.

    Returns gradients with jax.grad conventions: a bare array for an int
    ``argnums``, a tuple for a sequence. Inputs must be positional floating
    arrays (no kwargs / pytree / integer / complex — see :func:`_validate_grad_io`);
    there is no ``has_aux`` parameter (see :func:`value_and_grad`)."""
    return _make_grad(fun, order, argnums, count_ops, transforms, return_value=False)


def value_and_grad(
    fun: Callable,
    order: EliminationOrder = "rev",
    argnums: Union[int, Sequence[int]] = 0,
    count_ops: bool = False,
    transforms: TransformSpec = None,
) -> Callable:
    """``jax.value_and_grad`` analogue via vertex elimination — returns
    ``(value, grads)`` from a single elimination pass (jacve's ``has_aux=True``
    path yields ``(primal_outputs, jacobians)``; the primal is evaluated while
    building the graph, so there is no second forward pass). Same ``order`` /
    ``transforms`` / ``count_ops`` semantics and input restrictions as
    :func:`grad`; with ``count_ops`` the callable returns ``((value, grads), aux)``."""
    return _make_grad(fun, order, argnums, count_ops, transforms, return_value=True)


def unload_post_transforms(post, pre):
    new_post = post.copy()
    for transform in pre.post_transforms:
        new_post = transform.apply_inverse(new_post)
    _assert_sparse_tensor_consistency(new_post)
    return new_post


def unload_pre_transforms(post, pre):
    new_pre = pre.copy()
    for transform in post.pre_transforms:
        new_pre = transform.apply(new_pre)
    _assert_sparse_tensor_consistency(new_pre)
    return new_pre


def _drain_or_unload_pre(post_val, pre_val, _post_val):
    """Resolve ``post_val``'s pre_transforms against ``pre_val``; returns the
    updated ``(_post_val, _pre_val)``. SEED-AWARE DRAINING: a pure-relabel
    pre_transform (reshape / transpose / squeeze / slice / ...) on ``post_val``
    relabels the CONTRACTED dimension; applying it FORWARD onto a sparse
    ``pre_val`` diagonal densifies it (the O(n^2) conv-head blow-up). When every
    transform is seed_drainable and ``pre_val`` is a sparse diagonal, the
    bijective identity ``post @ apply(pre) == apply_inverse(post) @ pre`` lets us
    instead apply the INVERSE relabel to the cotangent VECTOR ``_post_val`` (free)
    and keep ``pre_val`` diagonal; otherwise densify via ``apply`` (unload), or
    pass ``pre_val`` through. Used only by ``_eliminate_vertex``."""
    _pre_transforms = post_val.pre_transforms
    if (
        len(_pre_transforms) > 0
        and pre_val.val is not None
        and post_val.val is not None
        # cheap attribute check first: short-circuits the dim scan for the
        # common non-relabel transforms (slice/concat/...).
        and all(getattr(t, "seed_drainable", False) for t in _pre_transforms)
        and _has_sparse_dim(pre_val)
    ):
        for _t in _pre_transforms[::-1]:
            _post_val = _t.apply_inverse(_post_val)
        _assert_sparse_tensor_consistency(_post_val)
        _pre_val = pre_val.copy()
    elif len(_pre_transforms) > 0 and (pre_val.val is not None or pre_val.dims):
        # ``pre_val.val is None`` (a UNIFORM operand) used to fall through to
        # the pass-through below, which DROPS the transform. That is not a
        # cheaper route, it is a wrong one: ``post_val``'s pre_transform carries
        # the contracted dimension's relabelling, and the tensor it sits on is a
        # rank-0 uniform stand-in with no dims of its own, so once the transform
        # is gone nothing states the edge's shape. The store then writes a
        # rank-0 tensor for an edge whose nominal shape is, for example, (1, 3),
        # and core.py's nominal-shape assertion fires.
        #
        # It stayed hidden because a uniform ``pre_val`` was rare: the
        # materializing elementwise path wrote a buffer for almost every edge,
        # so ``val is not None`` held and the transform was resolved. Turning
        # ``_lazy_uu`` on makes uniform operands common and the hole shows
        # immediately. The fault is here, not in the lazy rule.
        # The transform reshapes and slices ``val``, so a uniform operand needs
        # a buffer first. ``materialize_uniform`` gives it the block-diagonal
        # storage it would occupy, not the dense one.
        #
        # A DIMS-LESS uniform operand is excluded above and keeps the
        # pass-through. It has no contracted axis for the transform to relabel,
        # so there is nothing to unload onto: materializing it yields a rank-0
        # buffer and the slice/concat transforms raise on it (IndexError on
        # roll / gather / multi-head attention). The case this branch exists
        # for is the one with dims and no buffer.
        if pre_val.val is None:
            from .sparse.tensor import materialize_uniform
            pre_val = materialize_uniform(pre_val)
        _pre_val = unload_pre_transforms(post_val, pre_val)
    else:
        _pre_val = pre_val.copy()
    return _post_val, _pre_val


def prepend_post_transforms(post, out):
    transforms = post.post_transforms + out.post_transforms
    out.post_transforms = transforms
    return out


def append_pre_transforms(pre, out):
    transforms = pre.pre_transforms + out.pre_transforms
    out.pre_transforms = transforms
    return out


def _identity_passthrough(keep, ident, side: str):
    """Pass ``keep`` through an IDENTITY ``ident`` operand (the ``_need_contract
    is False`` shortcut), folding the identity's ``scalar_mult``.

    Both contraction paths re-attach ``post_val.post_transforms`` /
    ``pre_val.pre_transforms`` to the emitted edge immediately below. That is
    correct after a REAL contraction, because ``sparse_matmul`` / ``unload_*``
    rebuild the tensor and DROP its queued transforms -- the re-attach restores
    them exactly once. The pass-through shortcut is a ``copy()``, which KEEPS
    the queue, so the very same transform object got queued a SECOND time
    (observed: ``pre_transforms = (transpose, transpose)`` on a ViT edge -- the
    relabel applied twice, cos 1/32). Clear the side that is about to be
    re-attached so the shortcut has the same postcondition as the contract path.

    ``side="pre"``  -- ``keep`` is the pre operand; its ``pre_transforms``
                      are re-appended by ``append_pre_transforms``.
    ``side="post"`` -- ``keep`` is the post operand; its ``post_transforms``
                      are re-prepended by ``prepend_post_transforms``.
    The OTHER side is deliberately preserved: it lives on the surviving
    (non-contracted) dims and nothing re-attaches it.
    """
    out = keep.copy(scalar_mult=_scaled_mul_promote(keep.scalar_mult, ident.scalar_mult))
    if side == "pre":
        out.pre_transforms = ()
    else:
        out.post_transforms = ()
    return out


def _drain_transforms(tensor, post_first: bool = True):
    """Fold a tensor's queued Jacobian transforms into its data: apply each
    ``post_transform`` forward (``apply``) and each ``pre_transform`` in reverse
    (``apply_inverse``). ``post_first`` (the edge-merge order) drains post then
    pre; ``post_first=False`` (the final-output drain) drains pre then post.
    Empty transform lists are no-ops."""
    def _post(t):
        for transform in t.post_transforms:
            t = transform.apply(t)
        return t

    def _pre(t):
        for transform in t.pre_transforms[::-1]:
            t = transform.apply_inverse(t)
        return t

    return _pre(_post(tensor)) if post_first else _post(_pre(tensor))


def _peel_reconciler_transforms(tensor):
    """Materialise the SHAPE-RECONCILING transforms queued on ``tensor`` into its
    data BEFORE it enters a contraction, leaving pure RELABELS queued for the
    post-contraction re-attach. Returns ``(reduced_tensor, remaining_pre,
    remaining_post)`` where the remaining lists are the un-peeled transforms
    (still to be re-attached to the contraction output).

    A *reconciler* (concatenate slot embed/slice, broadcast reduce, slice) folds
    an INFLATED operand axis back to its nominal size — its ``apply`` /
    ``apply_inverse`` acts on the operand's OWN dims and STRICTLY shrinks the
    logical size. A *relabel* (transpose / reshape / squeeze) is size-preserving
    and is defined against the CONTRACTION-OUTPUT dim list, so applying it to the
    operand indexes a dim it does not have (``IndexError``); it must ride the
    re-attach path unchanged.

    Why this matters: the matmul is CORRECT on nominal operands, but a diagonal
    ``dW/dV`` contracted against a ``dV/dU`` whose ``U`` axis is stored INFLATED
    (a concat slot at full width) emits a block-diagonal coupling that pins the
    inflated size onto ``U`` — a transform-free NON-NOMINAL edge that then
    broadcasts through every sibling merge until an ``N`` vs ``M`` (neither 1)
    collision crashes ``_reconcile_broadcast_dims``. Folding the reconciler into
    the operand first feeds the contraction a nominal ``U`` so no coupling forms.
    Peels in the exact drain order (post forward, then pre reversed) and STOPS at
    the first non-reconciler so the kept remainder is a valid transform prefix.
    """
    import math as _math

    def _sz(x):
        s = x.shape
        return _math.prod(s) if s else 1

    post = list(tensor.post_transforms)
    pre = list(tensor.pre_transforms)
    cur = tensor.copy()
    cur.post_transforms = ()
    cur.pre_transforms = ()

    for k in range(len(post)):
        before = _sz(cur)
        try:
            cand = post[k].apply(cur)
            _assert_sparse_tensor_consistency(cand)
        except Exception:
            cand = None
        if cand is None or _sz(cand) >= before:
            cur.post_transforms = tuple(post[k:])
            cur.pre_transforms = tuple(pre)
            return cur, tuple(pre), tuple(post[k:])
        cur = cand

    for j in range(len(pre) - 1, -1, -1):
        before = _sz(cur)
        try:
            cand = pre[j].apply_inverse(cur)
            _assert_sparse_tensor_consistency(cand)
        except Exception:
            cand = None
        if cand is None or _sz(cand) >= before:
            cur.pre_transforms = tuple(pre[: j + 1])
            cur.post_transforms = ()
            return cur, tuple(pre[: j + 1]), ()
        cur = cand

    cur.pre_transforms = ()
    cur.post_transforms = ()
    return cur, (), ()


# Face-like per-path approximation: the `transforms` entry for a vertex may be a
# dict keyed by (primal_vertex_id, out_vertex_id) -> per-path hooks. This applies
# the approximation at the OP boundary (contraction operands + result, join
# operands + result) exactly like the face engine, so an edge NEVER needs
# `_normalize_approx_edge` (the sparse matmul/+ reconcile shapes locally). Vertex
# ids follow the eqn-position scheme (1-based, matching `order`); graph inputs
# take a negative id -(invar_index+1). See `_parse_path_hooks` for the value
# shape. This path is byte-identical to exact AD when `transforms == ()`.
_NULL3 = (None, None, None)


def _parse_path_hooks(value):
    """Normalize a per-path transforms value into ``((pre, post, new),
    (lhs, rhs, res))`` — the contraction hooks and the join/add hooks. The two
    ops are ordered ``(contraction, join)``; an EMPTY tuple ``()`` (or ``None``)
    in either slot means "no approximation for that op".

    Dispatch is by outer length, so the two ops are never ambiguous:
      * ``None`` / ``()``                        -> no approximation at all
      * ``(pre, post, new)``      (length 3)     -> contraction only, join = none
      * ``((pre,post,new),)``     (length 1)     -> contraction only, join = none
      * ``((pre,post,new), (lhs,rhs,res))`` (length 2) -> both ops
      * ``((), (lhs,rhs,res))``   (length 2)     -> join only (contraction skipped)
      * ``((pre,post,new), ())``  (length 2)     -> contraction only

    ``pre``/``post`` apply to the two contraction operands, ``new`` to the
    contraction result; ``lhs``/``rhs`` apply to the two join operands (the new
    contribution and the existing edge), ``res`` to the summed edge. Each hook is
    a callable ``SparseTensor -> SparseTensor`` or ``None`` (identity).
    """
    if value is None:
        return _NULL3, _NULL3

    def _as3(h):
        # None or an empty tuple -> no approximation for this op.
        if h is None or (hasattr(h, "__len__") and len(h) == 0):
            return _NULL3
        h = tuple(h)
        if len(h) != 3:
            raise ValueError(
                f"per-path op hooks must be a 3-tuple (a, b, res) or () / None; "
                f"got length {len(h)}"
            )
        return h

    n = len(value)
    if n == 0:                                   # ()  -> no approximation
        return _NULL3, _NULL3
    if n == 3:                                    # bare (pre, post, new)
        return _as3(value), _NULL3
    if n == 1:                                    # ((pre,post,new),)
        return _as3(value[0]), _NULL3
    if n == 2:                                    # (contraction, join)
        return _as3(value[0]), _as3(value[1])
    raise ValueError(
        "per-path transforms value must be (pre,post,new), ((pre,post,new),), "
        f"or ((pre,post,new),(lhs,rhs,res)); got {value!r}"
    )


def _is_scalar_st(t) -> bool:
    return not t.out_dims and not t.primal_dims


def _has_sparse_dim(t) -> bool:
    """Whether the tensor carries a sparse (diagonal) dim that a forward
    relabel transform would densify. Used by seed-aware draining."""
    return any(d.is_sparse for d in t.out_dims) or any(
        d.is_sparse for d in t.primal_dims
    )


def _acts_as_identity(t) -> bool:
    """Whether a structural (``val is None``) edge Jacobian is the PURE-DIAGONAL
    IDENTITY — up to its ``scalar_mult`` — so that ``t @ pre`` is a pass-through
    returning ``pre`` scaled by ``t.scalar_mult`` (the caller folds the
    ``scalar_mult`` in).

    Grounded in the representation's semantics of ``val is None`` / ``axis is
    None`` (not a shape heuristic):
      * a DENSE dim (``other_id is None``) is a BROADCAST over that axis — NOT
        the identity;
      * a DIAGONAL pair (``other_id`` set, linking out↔primal of equal size)
        with ``block_size in {None, 1}`` is a PURE DIAGONAL = identity;
        ``block_size > 1`` is a block-local reduction — NOT the identity;
      * a non-zero ``fill_value`` (anything but the statically-zero ``None``)
        paints the off-structure cells, so it is not a pure identity;
      * a 0-rank scalar (no dims) is the identity up to ``scalar_mult``.

    ``scalar_mult`` is deliberately NOT inspected here — it can't be proven
    ``== 1`` inside ``jit`` (it is a tracer) — so the test is purely structural
    and the caller multiplies it into the passed-through value, which is correct
    for any ``scalar_mult``."""
    if t.fill_value is not None:
        return False
    if not t.out_dims and not t.primal_dims:
        return True  # scalar: identity up to scalar_mult
    if len(t.out_dims) != len(t.primal_dims):
        return False
    primal_by_id = {d.id: d for d in t.primal_dims}
    for d in t.out_dims:
        if d.other_id is None:  # dense dim => broadcast, not identity
            return False
        p = primal_by_id.get(d.other_id)
        if p is None or p.other_id != d.id or p.logical_size != d.logical_size:
            return False
        if (d.block_size or 1) != 1 or (p.block_size or 1) != 1:
            return False  # block_size > 1 => block-local reduction, not identity
    return True


def _stable_var_index(jaxpr):
    """Var -> first-appearance position in the jaxpr (constvars, invars, then
    each eqn's outvars). Gives a deterministic key for ordering the id-hashed
    edge maps during path tokenization."""
    idx = {}
    for v in jaxpr.constvars:
        idx.setdefault(v, len(idx))
    for v in jaxpr.invars:
        idx.setdefault(v, len(idx))
    for e in jaxpr.eqns:
        for v in e.outvars:
            idx.setdefault(v, len(idx))
    return idx


def _apply_micro(edge_outval, _t):
    """Dispatch one typed micro-action; traced in isolation by a PathSink."""
    if isinstance(_t, Diag):
        return apply_diag(edge_outval, _t)
    if isinstance(_t, Compress):
        return apply_compress(edge_outval, _t)
    return apply_quant(edge_outval, _t)


def _approx_meta(_t):
    """(TYPE token, params dict) for a typed micro-action -> approx block head."""
    if isinstance(_t, Diag):
        return "DIAG", {"i": int(_t.i), "j": int(_t.j), "factor": int(_t.factor)}
    if isinstance(_t, Compress):
        return "COMPRESS", {"kind": _t.kind, "axes": tuple(int(a) for a in _t.axes)}
    return "QUANT", {"dtype": _t.dtype}


def _micro_applied(before, after) -> bool:
    """Did a micro-action actually CHANGE the tensor?

    ``apply_quant`` returns its input UNCHANGED (the same object) for a
    structural ``val is None`` edge or an already-matching dtype, and
    ``apply_*`` may likewise short-circuit — so a
    dispatched micro-action is not evidence that an approximation happened. The
    identity check is the reliable signal (every real micro-action builds a new
    :class:`SparseTensor`); the field-wise fallback additionally catches a fresh
    wrapper around an untouched payload.

    This is a TRACE-TIME fact, not a numerical one: a value-EXACT approximation
    (int8 quantization of a uniform-magnitude structural Jacobian round-trips
    bit-exactly) still counts as applied, because two traced tensors cannot be
    compared by value while tracing. See :class:`TransformRecord`.
    """
    if after is before:
        return False
    return not (
        getattr(after, "val", None) is getattr(before, "val", "x")
        and getattr(after, "scalar_mult", None)
        is getattr(before, "scalar_mult", "x")
        and getattr(after, "fill_value", None)
        is getattr(before, "fill_value", "x")
        and getattr(after, "out_dims", None) == getattr(before, "out_dims", "x")
        and getattr(after, "primal_dims", None)
        == getattr(before, "primal_dims", "x")
        and getattr(after, "pre_transforms", None)
        == getattr(before, "pre_transforms", "x")
        and getattr(after, "post_transforms", None)
        == getattr(before, "post_transforms", "x")
    )


def _eqn_count(_face_sink, _xlog) -> int:
    """Current persistent-frame equation count, from whichever recorder is
    installed (``0`` when neither is — nothing consumes the range then)."""
    if _face_sink is not None:
        return _face_sink.n_eqns()
    if _xlog is not None:
        return _xlog.n_eqns()
    return 0


class _SkipFace:
    """Sentinel ``face_transforms`` VALUE (in place of the ``(lhs, rhs, res)``
    3-tuple): the SKIP approximation — this path's contraction is NOT
    performed, so its contribution never reaches the ``(in_edge, out_edge)``
    accumulation (an absent addend, not a zero edge). Recorded on the face
    sink as an ``approx SKIP`` block and on the TransformLog, so the token
    stream and the record stay truthful."""

    __slots__ = ()

    def __repr__(self):
        return "graphax.SKIP_FACE"


SKIP_FACE = _SkipFace()


class FaceRequest(dict):
    """ONE vertex's ``{face_key: slots}`` request, with a HIT RECORD.

    :func:`vertex_elimination_jaxpr` wraps every inner dict of a
    ``face_transforms`` mapping in this class before the elimination runs, and
    checks the hit record afterwards. The wrapper exists for one reason: an
    approximation that is REQUESTED and never APPLIED must not pass in
    silence. See :func:`check_face_transforms`.
    """

    __slots__ = ("hit",)

    def __init__(self, mapping=()):
        super().__init__(mapping)
        self.hit = set()

    def get(self, key, default=None):
        if dict.__contains__(self, key):
            self.hit.add(key)
            return dict.__getitem__(self, key)
        return default


def _is_face_key(key) -> bool:
    """Is ``key`` an INNER (face) key -- a ``(in_vidx, out_vidx)`` pair?"""
    return (isinstance(key, (tuple, list)) and len(key) == 2
            and all(isinstance(_x, (int, np.integer)) or _x is None
                    for _x in key))


def check_face_transforms(face_transforms, *, site: str = "jacve"):
    """Validate the SHAPE of a ``face_transforms`` argument, or raise.

    THE NESTING IS THE TRAP, and it used to be a silent one.
    ``face_transforms`` is ``{vertex: {face_key: slots}}`` -- TWO levels. But
    the thing a caller has in hand is :func:`faces_of`'s output, which is the
    list of INNER keys, so the natural mistake is to build the FLAT
    ``{face_key: slots}`` dict and hand that to :func:`jacve`. Every lookup in
    the elimination is then ``face_transforms.get(vertex)``, no vertex is ever
    a pair, and the whole request evaporates: measured on 2026-09-16
    (job 65975) twelve requested ``Diag`` hooks produced ZERO hook calls and a
    bit-identical gradient. An approximation asked for and not applied is
    exactly what "nothing skips silently" forbids, so the flat form raises
    here instead.

    Returns the mapping with every inner dict wrapped in :class:`FaceRequest`,
    or ``None`` / the original falsy value unchanged.
    """
    if not face_transforms:
        return face_transforms
    if not isinstance(face_transforms, dict):
        raise TypeError(
            f"{site}: face_transforms must be a dict "
            f"{{vertex: {{face_key: slots}}}}, got "
            f"{type(face_transforms).__name__}.")
    out = {}
    for _v, _faces in face_transforms.items():
        if _is_face_key(_v):
            raise ValueError(
                f"{site}: face_transforms is keyed by a FACE KEY {_v!r}, not "
                f"by a vertex. The mapping has TWO levels -- "
                f"{{vertex: {{face_key: slots}}}} -- and `faces_of` returns "
                f"the INNER keys only. A flat {{face_key: slots}} dict "
                f"matches no vertex, so every approximation in it would be "
                f"dropped in silence. Group the keys by the vertex they were "
                f"enumerated on.")
        if not (isinstance(_v, (int, np.integer)) and not isinstance(_v, bool)):
            raise ValueError(
                f"{site}: face_transforms key {_v!r} "
                f"({type(_v).__name__}) is not a vertex id. The outer key is "
                f"the 1-based vertex the faces belong to.")
        if _faces is None:
            continue
        if _faces is SKIP_FACE or not isinstance(_faces, dict):
            raise ValueError(
                f"{site}: face_transforms[{int(_v)}] is {_faces!r}, not a "
                f"{{face_key: slots}} dict. A vertex maps to its FACES; the "
                f"slots triple sits one level further in. Pass "
                f"{{{int(_v)}: {{face_key: slots}}}}, or use the per-vertex "
                f"`transforms` argument for a whole-vertex transform.")
        for _k in _faces:
            if not _is_face_key(_k):
                raise ValueError(
                    f"{site}: face_transforms[{int(_v)}] has key {_k!r}, "
                    f"which is not a face key. A face key is the pair "
                    f"`(vidx[in_edge], vidx[out_edge])` that `faces_of` "
                    f"returns for this vertex, on the graph state "
                    f"IMMEDIATELY before it is eliminated.")
        out[int(_v)] = FaceRequest(_faces)
    return out


def report_unapplied_face_transforms(face_transforms, *, site: str = "jacve"):
    """Raise when a whole vertex's face request matched NOTHING.

    Called after the elimination, on the mapping :func:`check_face_transforms`
    returned. A vertex whose request is non-empty and whose hit record is
    empty was never looked up at all: either it is not in the order, or its
    keys were enumerated on a different graph state (``faces_of`` must be
    called immediately before that vertex's own elimination, because every
    elimination rewires the graph).

    PARTIAL misses are NOT an error. :func:`faces_of` is a documented SUPERSET
    of the faces the elimination visits -- an unevaluated ``LazyEdge`` is
    listed optimistically and may force to ``None`` -- so a request that lands
    on some of a vertex's faces and not all of them is legitimate.
    """
    if not face_transforms:
        return
    dead = [int(_v) for _v, _faces in face_transforms.items()
            if isinstance(_faces, FaceRequest) and _faces and not _faces.hit]
    if not dead:
        return
    _v0 = dead[0]
    raise ValueError(
        f"{site}: the face transforms requested for "
        f"{'vertices' if len(dead) > 1 else 'vertex'} {dead} were never "
        f"applied -- the elimination looked up none of their face keys. "
        f"Vertex {_v0} asked for "
        f"{sorted(face_transforms[_v0].keys())[:8]}. Either the vertex is "
        f"not in the elimination order, or the keys come from a different "
        f"graph state: `faces_of` has to be called on the graph the vertex "
        f"is about to be eliminated on, because every elimination rewires "
        f"the graph. Nothing is applied in silence here (owner rule).")



def _record_micro(_t, before, after, vertex, slot, in_edge, out_edge,
                  start, _face_sink, _xlog, log_slot=None):
    """Record ONE dispatched micro-action, truthfully.

    Writes to the always-on :class:`TransformLog` (installed by
    ``IncrementalJaxpr`` for every elimination, so the record exists even with
    ``track_faces=False``) with an explicit ``applied`` flag, and — only when the
    action really changed the tensor — to the opt-in :class:`FaceSink`'s
    ``approx`` block list, so the tokenizer never renders an empty block for a
    no-op. Both sinks are optional; with neither installed this is a no-op.

    ``log_slot`` (default: ``slot``) is the tag written to the TRANSFORM LOG
    only. The two sinks read the slot differently: the FaceSink's tag is a
    POSITION in a fixed-width per-face layout (``FACE_SLOT_INDEX`` is closed —
    an unknown tag would be rendered in ``SKIP``'s slot-less position and shift
    the whole stream), while the transform log's tag is free-form text nothing
    matches on. So the two-op join sites, which all act on the ``"res"``
    operand, keep ``slot="res"`` for the sink and pass a finer
    ``log_slot="res:jr"`` etc. here — distinguishable in the log, invisible to
    the tokenizer. See :func:`_unpack_face_slots`.
    """
    if _face_sink is None and _xlog is None:
        return
    applied = _micro_applied(before, after)
    end = _eqn_count(_face_sink, _xlog)
    _atype, _params = _approx_meta(_t)
    if _face_sink is not None and applied:
        # SLOT-TAGGED. ``slot`` says WHICH OPERAND of the face was approximated
        # ("lhs" / "rhs" / "res", or "vertex" for a per-vertex transform, which
        # hits the contraction result -- the same operand as "res"). A slot that
        # declined records nothing at all, so the emitter cannot recover the
        # slot from the ORDER of the records; it has to be carried here.
        _face_sink.approx(_atype, _params, start, end, slot)
    if _xlog is not None:
        _xlog.record("transform", vertex,
                     slot if log_slot is None else log_slot,
                     _atype, _params,
                     in_edge, out_edge, start, end, applied)


# ---------------------------------------------------------------------------
# Per-FACE (per local path) Jacobian transforms
# ---------------------------------------------------------------------------
#
# A vertex elimination contracts one FACE per (in_edge -> central_var ->
# out_edge) path. The per-vertex ``transforms`` argument is applied uniformly to
# every one of those faces, which is exactly what an RL policy that must choose
# an approximation PER PATH cannot express. ``face_transforms`` adds that: a
# mapping FACE KEY -> ``(lhs, rhs, res)`` where
#
#   * FACE KEY is ``(vidx[in_edge], vidx[out_edge])`` under the SAME stable var
#     index (:func:`_stable_var_index`) the face sink caches — the vertex is
#     implicit because the mapping is per-vertex-elimination;
#   * ``lhs`` transforms the in_edge Jacobian (``pre_val``) and ``rhs`` the
#     out_edge Jacobian (``post_val``), BOTH before the contraction;
#   * ``res`` transforms the contraction result (``edge_outval``) at the same
#     site as the per-vertex ``transforms``, immediately AFTER them.
#
# Naming follows the slot semantics of the local path ``res = op(lhs, rhs)``.

# ONE-ENTRY memo for the stable var index. An elimination sequence walks the
# same jaxpr for every vertex, so rebuilding the index per ``_eliminate_vertex``
# call would be O(V^2); the face sink already caches it for the tracked path,
# this covers the untracked ``face_transforms`` path. Thread-local and bounded
# to a single jaxpr, so it can't leak across threads or grow.
_VIDX_MEMO = threading.local()


def _vidx_for(jaxpr):
    """``_stable_var_index(jaxpr)``, memoized on the last jaxpr seen."""
    cached = getattr(_VIDX_MEMO, "entry", None)
    if cached is not None and cached[0] is jaxpr:
        return cached[1]
    idx = _stable_var_index(jaxpr)
    _VIDX_MEMO.entry = (jaxpr, idx)
    return idx


def _known_none_edge(edge) -> bool:
    """True iff ``_force(edge)`` is KNOWN to be ``None`` WITHOUT forcing.

    A :class:`LazyEdge` whose thunk has not run yet emits jax equations when
    forced, so a read-only enumeration (:func:`faces_of`) must not touch it:
    such an edge reports ``False`` ("not known to be None") and its face is
    listed even though the elimination may later skip it.
    """
    if isinstance(edge, LazyEdge):
        return edge._value is not _UNSET and edge._value is None
    return edge is None


def _is_two_op_slots(slots) -> bool:
    """Is this ``face_transforms`` entry the TWO-OP form (a pair of triples)?

    The single predicate both :func:`_unpack_face_slots` and the elimination
    loop dispatch on, so "which form is this" is decided in exactly one place.
    """
    return (isinstance(slots, (tuple, list)) and len(slots) == 2
            and all(isinstance(_s, (tuple, list)) and len(_s) == 3
                    for _s in slots))


def _iter_face_hooks(slots):
    """Every hook object inside one ``face_transforms`` entry, flattened.

    The TWO-OP form nests its hooks one level deep
    (``((lhs, rhs, new), (jl, jr, jres))``), so a naive ``for _t in slots``
    sees TUPLES — which are neither ``Diag``/``Compress`` nor callable — and an
    "is this an approximation?" test built on it silently answers *no* for a
    two-op entry that carries the very same micro-action a flat 3-tuple would
    arm. Both arming sites (the per-vertex ``_is_approx_cfg`` and the global
    dispatch flag in :func:`jacve`) iterate through here so they cannot drift.

    A :class:`~graphax.sparse.ops.join.FaceJoinPolicy` at the ``jr`` position
    is yielded AS ITSELF and its ``pre`` hook yielded beside it. The policy is
    not callable (it takes two tensors), so the arming tests name its type
    explicitly; a ``lossy`` policy DROPS values, so it must arm the approx
    config exactly as a ``Diag`` does.
    """
    from .sparse.ops.join import FaceJoinPolicy as _FJP

    def _one(_t):
        yield _t
        if isinstance(_t, _FJP) and _t.pre is not None:
            yield _t.pre

    for _t in (slots if isinstance(slots, (tuple, list)) else ()):
        if isinstance(_t, (tuple, list)):
            for _u in _t:
                yield from _one(_u)
        else:
            yield from _one(_t)


def _unpack_face_slots(slots, vertex):
    """Validate one ``face_transforms`` entry -> ``(lhs, rhs, res, new, join)``.

    TWO forms are accepted:

    * the FLAT ``(lhs, rhs, res)`` triple — ``lhs``/``rhs`` hook the two
      contraction operands, ``res`` hooks the edge at the POST-JOIN site
      (returned as ``(lhs, rhs, res, None, None)``);
    * the TWO-OP ``((lhs, rhs, new), (jl, jr, jres))`` pair of triples, which
      splits the single legacy ``res`` slot into the four sites a face's
      result actually passes through::

          contract = op(lhs(pre_val), rhs(post_val))     # the contraction
          fresh    = new(contract)                       # PRE-join
          old      = jr(old) + jl(fresh)                 # the MERGE
          old      = jres(old)                           # post-join (== res)

      i.e. the join semantics ``old = approx(old) + approx(new)``: BOTH
      addends of a merge are hooked, which a flat triple cannot express (its
      ``res`` lands on the SUM, approximating the two addends' merge instead
      of the addends).

    Returned as ``(lhs, rhs, jres, new, (jl, jr))`` so the elimination loop's
    ``res``-slot variable carries ``jres`` unchanged and the flat path needs no
    branch of its own.

    THE ``jr`` POSITION MAY HOLD A POLICY, not a hook. Every hook position sees
    ONE tensor, so none of them can express "make these two addends share one
    container" -- that is inherently a BINARY operation on the pair. A
    :class:`~graphax.sparse.ops.join.FaceJoinPolicy` placed at ``jr`` is handed
    BOTH addends by the elimination loop and returns both. ``jr`` is the right
    position for it because ``jr`` is the old edge's slot and the old edge is
    what moves. The policy carries its own single-tensor ``pre`` hook for the
    old edge, so a learned approximation of the old edge still has a slot;
    ``jl`` and ``jres`` are untouched and stay ordinary hook positions.

    WHY THIS IS NOT A HOOK WITH A SECOND ARGUMENT. The asymmetry the policy
    removes is the reason it exists: the two addends of a merge share their
    logical dims but not their STORAGE, so a single approximation rule applied
    at both sites is a legal subdivision on one tensor and an idempotent no-op
    on the other, and one legality mask cannot describe both (alphagrad finding
    72 / ticket dsnn-3qm.59 fault 1). Expressing that as two independent
    single-tensor hooks is what produced the defect; the binary form is the
    fix, not a convenience.

    MERGE-FREE FACES (documented, pinned by
    ``tests/misc/test_face_two_op_form.py``). ``jl``/``jr`` are applied ONLY
    inside the ``graph[in_edge][out_edge] is not None`` branch, so a face whose
    contraction creates a BRAND-NEW edge degenerates to ``jres(new(contract))``
    with both join hooks skipped and NO record emitted. That is
    correct-by-construction, not an oversight:

    * ``jr`` hooks the EXISTING edge. There is no existing edge, so it has no
      operand at all — applying it to anything else would be a fabrication.
    * ``jl`` hooks the fresh contribution *as it enters the merge*. With no
      merge, the tensor at that site is bit-identical to the one ``new``
      already hooked, with no intervening op — so anything ``jl`` could
      express on a merge-free face is expressible via ``new``, and no
      expressive power is lost. Applying it anyway would make the SAME face's
      effective approximation depend on whether a sibling contribution
      happened to be stored first, i.e. on elimination order — which would
      make a plan's measured cost irreproducible across replays.
    * The silence is truthful under this module's recording contract: a slot
      that did not run records nothing (see :func:`_record_micro`), so the
      telemetry never claims an approximation that never happened.

    SLOT TAGS. ``new``/``jl``/``jr``/``jres`` all report the ``"res"`` operand
    slot to a :class:`~graphax.sparse.tracer.FaceSink`, so the sink cannot tell
    them apart. That is DELIBERATE: ``FACE_SLOT_INDEX`` is a closed map and the
    tokenizer emits exactly ``N_FACE_SLOTS`` equation blocks per face at FIXED
    positions, so an unknown tag would be classed "unslotted" and rendered in
    ``SKIP``'s slot-less position, corrupting the stream every downstream head
    reads by position. All four sites act on the same operand of the local path
    (``res``/``new``, the contraction result and its merge), so slot 2 is the
    correct index for every one of them; within it the records are ordered by
    application (``new`` -> ``jl`` -> ``jr`` -> ``jres``). The always-on
    :class:`~graphax.sparse.tracer.TransformLog`, which has no positional
    consumer, receives the FINE-GRAINED tags ``"res:new"`` / ``"res:jl"`` /
    ``"res:jr"`` / ``"res:jres"`` instead, so the four are distinguishable
    there (the flat form keeps a bare ``"res"``).

    A malformed entry is a structural programming error, so it raises
    ``TypeError`` — which the per-face dispatch deliberately does NOT catch.
    """
    if _is_two_op_slots(slots):
        # TWO-OP form ((lhs, rhs, new), (jl, jr, jres)) -- the join
        # semantics ``new = approx(new_existing) + approx(contract)``:
        # triple 1 hooks this face's operands and its contraction result
        # PRE-join; triple 2 hooks the JOIN (jl -> new contribution,
        # jr -> the EXISTING edge, jres -> the merged sum, landing at the
        # same post-site the legacy ``res`` uses).
        (lhs, rhs, new), (jl, jr, jres) = slots
        return lhs, rhs, jres, new, (jl, jr)
    try:
        lhs, rhs, res = slots
    except (TypeError, ValueError):
        raise TypeError(
            f"face_transforms entry at vertex {vertex} must be a 3-tuple "
            f"(lhs, rhs, res) or a pair of 3-tuples "
            f"((lhs, rhs, new), (jl, jr, jres)); got {slots!r}."
        ) from None
    return lhs, rhs, res, None, None


class FaceTransformIllegal(ValueError):
    """A per-face transform the caller asked for cannot be applied to that
    operand (ticket dsnn-3qm.70).

    Raised instead of skipping, so a measured plan is always the plan that was
    asked for. It subclasses ``ValueError`` so an existing ``except ValueError``
    higher up still catches it, but the message names the vertex, the slot, the
    action and the operand's structure.
    """


def _apply_face_transform(val, _t, slot, vertex, _face_sink, in_edge=None,
                          out_edge=None, _xlog=None, log_slot=None):
    """Apply ONE per-face slot transform to ONE Jacobian operand.

    Mirrors the per-vertex ``transforms`` dispatch in :func:`_eliminate_vertex`
    exactly: ``Diag`` / ``Compress`` / ``Quant`` go through :func:`_apply_micro`
    (and are recorded by :func:`_record_micro` — on the currently open face as a
    labelled ``approx`` block when a face sink is installed, and ALWAYS on the
    :class:`TransformLog` when one is), any other callable is handed the tensor
    directly, and anything else raises ``TypeError``.

    An action that does not fit THIS operand's geometry RAISES
    :class:`FaceTransformIllegal` (ticket dsnn-3qm.70, owner ruling D11). It used
    to be swallowed and the operand returned unchanged, which made a measured
    plan differ from the plan the caller asked for with no sign in the record:
    on the CPU toy under the Markowitz order a literal ``Diag`` was dropped on
    5 of 6 planned faces.

    The caller decides what to do about an illegal action. A caller that cannot
    know the operand's index structure until this moment — which is every
    policy, because these operands are join intermediates — must pass a CHOOSER
    callable instead of a literal action, and return ``None`` from it to skip.
    That is the one legal way to decline, and it is recorded as a decline.
    :func:`~graphax.sparse.micro_actions.action_is_legal` answers the same
    question without applying anything.

    THE CHOOSER PROTOCOL, in full. A chooser is handed the live operand and
    returns exactly ONE of: a ``Diag`` / ``Compress`` / ``Quant`` (applied and
    recorded here), a ``SparseTensor`` (taken as the new operand and recorded by
    nobody — the historical opaque form), or ``None`` (decline; the operand is
    returned untouched). A sequence of actions RAISES. A chooser that also
    carries a ``chosen_applied(action, applied)`` attribute is called back with
    the outcome, so a caller keeping applied / skipped counters reads the same
    ``_micro_applied`` verdict that decides whether a block is emitted.

    ``slot`` is the FaceSink's positional tag; ``log_slot`` (default: ``slot``)
    the transform log's finer one — see :func:`_record_micro`.
    """
    if _t is None:
        return val
    try:
        if isinstance(_t, (Diag, Compress, Quant)):
            _as = _eqn_count(_face_sink, _xlog)
            out = _apply_micro(val, _t)
            _record_micro(_t, val, out, vertex, slot, in_edge, out_edge,
                          _as, _face_sink, _xlog, log_slot)
        elif callable(_t):
            # A slot callable may act as a CHOOSER: handed the live operand, it
            # returns the micro-action it picked (or None to skip) instead of a
            # tensor. The operands here are join intermediates -- lhs is the
            # fresh contraction, rhs the existing edge -- so a policy cannot
            # know their index structure until this moment; masking legal
            # actions requires seeing the tensor. Routing a chosen action back
            # through _apply_micro/_record_micro keeps it as visible to the
            # transform log as a literal one, which a plain tensor-returning
            # callable is NOT.
            _chosen = _t(val)
            if _chosen is None:
                return val
            if isinstance(_chosen, (Diag, Compress, Quant)):
                _as = _eqn_count(_face_sink, _xlog)
                out = _apply_micro(val, _chosen)
                _record_micro(_chosen, val, out, vertex, slot, in_edge,
                              out_edge, _as, _face_sink, _xlog, log_slot)
                # THE OUTCOME, back to the chooser that asked for it. A chooser
                # decides BEFORE the action runs, so it cannot know by itself
                # whether the action changed the tensor -- and "changed" is the
                # only truthful reading of "applied" while tracing
                # (:func:`_micro_applied`), the same reading that decides
                # whether a block is recorded at all. A chooser that keeps
                # applied / skipped counters would otherwise have to guess, and
                # a guess is how a counter and the token stream drift apart.
                # OPTIONAL: a chooser without the attribute is not told.
                _notify = getattr(_t, "chosen_applied", None)
                if _notify is not None:
                    _notify(_chosen, _micro_applied(val, out))
            elif isinstance(_chosen, (tuple, list)):
                # ONE action per chooser call. graphax applies and records what
                # the chooser hands back, and a sequence would have to be
                # legality-checked against the INTERMEDIATE tensors this call
                # never sees. Raising names the caller; silently taking the
                # first would drop the rest of what the caller asked for.
                raise TypeError(
                    f"The chooser in slot {slot!r} at vertex {vertex} returned "
                    f"{len(_chosen)} actions; a chooser returns exactly ONE "
                    "micro-action, a SparseTensor, or None to decline. A "
                    "caller with several actions for one operand must install "
                    "them as separate slots or as per-vertex transforms."
                )
            else:
                out = _chosen
        else:
            raise TypeError(
                f"Unknown per-face transform of type {type(_t).__name__} in "
                f"slot {slot!r} at vertex {vertex}; expected None, Diag, "
                "Compress, Quant, or a callable taking a SparseTensor and "
                "returning either a SparseTensor or a chosen micro-action."
            )
    except ValueError as exc:
        raise FaceTransformIllegal(
            f"Face transform {_t!r} cannot be applied to the operand in slot "
            f"{slot!r} at vertex {vertex}: {exc}. The operand's dims are "
            f"{tuple(type(d).__name__ for d in val.dims)} with logical sizes "
            f"{tuple(int(d.logical_size) for d in val.dims)} and out/primal "
            f"split {len(val.out_dims)}/{len(val.primal_dims)}. This used to be "
            "skipped silently (ticket dsnn-3qm.70). Fix the caller: check with "
            "action_is_legal, or pass a chooser callable that returns None to "
            "decline."
        ) from exc
    _assert_sparse_tensor_consistency(out)
    return out


def face_config_is_approx(face_transforms) -> bool:
    """Does this ``face_transforms`` dict arm the APPROX contraction path?

    :func:`_eliminate_vertex` calls this to set its ``_is_approx_cfg``, which
    gates the reconciler peel and the re-evaluation of ``need_contract`` from
    the PEELED operands -- i.e. the ``approx`` argument of
    :func:`prepare_face_operands`. It is public because a caller that computes a
    face's result STRUCTURE outside the elimination (alphagrad's per-vertex
    approximation mask) has to pass the flag the elimination will use, and
    guessing it is how a mask goes stale.

    A ``SKIP_FACE`` arms it (a dropped path deviates from the exact structure at
    least as much as a ``Diag`` does) and so does a ``Diag`` / ``Compress`` /
    :class:`FaceJoinPolicy` anywhere in any face's hooks. A plain CALLABLE hook
    does NOT: a chooser or a frame-decoding hook is opaque here, which is why a
    per-vertex callable transform (``transforms=(fn,)``) is the thing that arms
    it on the recording-probe path.

    Iterated through ``_iter_face_hooks`` so the TWO-OP form
    ``((lhs, rhs, new), (jl, jr, jres))`` arms this exactly as its flat
    equivalent does.
    """
    if not face_transforms:
        return False
    return any(
        _slots is SKIP_FACE or any(
            isinstance(_t, (Diag, Compress, FaceJoinPolicy))
            for _t in _iter_face_hooks(_slots)
        )
        for _slots in face_transforms.values()
    )


def faces_of(graph, transpose_graph, vertex, jaxpr):
    """The FACE KEYS of ``vertex``, in the order the elimination will visit them.

    A policy that must pick an approximation per local path needs the path list
    BEFORE the vertex is eliminated (so the choice loop can be unrolled
    statically). This returns exactly the keys
    :func:`_eliminate_vertex` looks up in ``face_transforms``::

        [(vidx[in_edge], vidx[out_edge]), ...]

    under the same stable var index (:func:`_stable_var_index`) and the same
    ``out_edge``-major / ``in_edge``-minor nested product the elimination loop
    walks, with the same ordering (first-appearance position in ``jaxpr``).

    Args:
        graph: the computational graph (``graph[u][v]`` = edge Jacobian).
        transpose_graph: its transpose.
        vertex (int): the vertex about to be eliminated (1-based, as everywhere
            else — its equation is ``jaxpr.eqns[vertex - 1]``).
        jaxpr (core.Jaxpr): the traced jaxpr the graph was built from.

    Returns:
        list[tuple[int, int]]: face keys in elimination order.

    Notes:
        * Call this IMMEDIATELY before eliminating ``vertex``. Every elimination
          rewires the graph (a predecessor's elimination replaces this vertex's
          in-edges with ITS predecessors), so keys enumerated earlier describe a
          graph that no longer exists.
        * The elimination SKIPS a face whose edge Jacobian forces to ``None``
          (e.g. a ``stop_gradient`` blocked path). Dead edges from non-differentiable
          paths (``stop_gradient``, ``iota``, ``device_put``, ``select_n`` predicate)
          are pruned at graph build time (ticket dsnn-3qm.74). An edge that is already
          concrete (or an already-evaluated ``LazyEdge``) is filtered out here
          too, but an *unevaluated* ``LazyEdge`` is NOT forced — forcing emits
          jax equations into whatever trace happens to be current, which would
          corrupt the append-only jaxpr. The returned list is thus a clean inventory
          of reachable faces.
        * A multi-output vertex contributes the faces of every one of its
          output variables; the central variable is not part of the key (the
          mapping is per-vertex-elimination), so in the rare case where two
          output variables share the same ``(in_edge, out_edge)`` pair their key
          collides and one entry configures both faces.
    """
    return [fs.key for fs in face_specs_of(graph, transpose_graph, vertex,
                                           jaxpr)]


class FaceSpec(NamedTuple):
    """One face of a vertex elimination, named by its three VARIABLES.

    :func:`faces_of` returns only ``key``, which is what ``face_transforms`` is
    indexed by; a caller that has to compute a face's OPERANDS (the two edge
    Jacobians the contraction consumes) needs the variables themselves, and a
    caller that has to detect the multi-output collision documented on
    :func:`faces_of` needs ``central`` as well -- ``key`` omits it, so two
    output variables of one equation that share an ``(in_edge, out_edge)`` pair
    produce the SAME key and one ``face_transforms`` entry configures both
    faces.

    Attributes:
        key: ``(vidx[in_edge], vidx[out_edge])`` -- the ``face_transforms`` key.
        central: the output variable of ``vertex``'s equation this face runs
            through. NOT part of ``key``.
        in_edge: the predecessor variable; ``transpose_graph[central][in_edge]``
            is the face's ``lhs`` operand.
        out_edge: the successor variable; ``graph[central][out_edge]`` is the
            face's ``rhs`` operand.
        f: the face's position in elimination order, i.e. the index
            ``faces_of`` lists it at and the ``f`` every per-face array in
            alphagrad is indexed by.
    """
    key: tuple
    central: Any
    in_edge: Any
    out_edge: Any
    f: int


def face_specs_of(graph, transpose_graph, vertex, jaxpr):
    """:func:`faces_of` with the three VARIABLES of each face, same order.

    THE ONE ENUMERATION. ``faces_of`` is a projection of this onto ``key``, so
    a caller that needs the operands or the central variable cannot drift from
    the key list the elimination looks up -- there is one loop, not two.

    Every caveat on :func:`faces_of` applies verbatim, in particular that an
    unevaluated ``LazyEdge`` is listed OPTIMISTICALLY (forcing it here would
    emit equations into whatever trace is current), so the result is a SUPERSET
    of the faces the elimination visits.

    Returns:
        list[FaceSpec]: one entry per face, in elimination order.
    """
    eqn = jaxpr.eqns[int(vertex) - 1]
    vidx = _vidx_for(jaxpr)

    def _ordered(keys):
        return sorted(keys, key=lambda v: vidx.get(v, 1 << 30))

    out = []
    for central_var in eqn.outvars:
        if central_var not in graph:
            continue  # dead or already-eliminated vertex
        # read-only: never materialize a missing transpose entry (the loop uses
        # a defaultdict, this helper must not mutate the caller's graph)
        _in_edges = transpose_graph.get(central_var) or {}
        for out_edge in _ordered(graph[central_var].keys()):
            if _known_none_edge(graph[central_var][out_edge]):
                continue
            for in_edge in _ordered(_in_edges.keys()):
                if _known_none_edge(_in_edges[in_edge]):
                    continue
                out.append(FaceSpec(
                    key=(vidx.get(in_edge), vidx.get(out_edge)),
                    central=central_var, in_edge=in_edge, out_edge=out_edge,
                    f=len(out)))
    return out


import math


def _factored_outputs_enabled() -> bool:
    """GRAPHAX_FACTORED_OUTPUTS (default OFF): store the FINAL contraction
    onto a pure output head as a factor pair instead of materializing the
    (batch-wise rank-1) product. Read per call so alphagrad can scope it to
    the sparse cost executable's trace only."""
    return os.environ.get("GRAPHAX_FACTORED_OUTPUTS", "0") == "1"


DEFERRED_OUTPUT_STATS: dict = {}


class DeferredOutputProduct:
    """An unevaluated ``post @ pre`` on an (input -> output) edge.

    Only two operations can ever touch it (guaranteed by the deferral
    guards: in_edge is a graph input, out_edge a pure output — neither is
    ever eliminated): a MERGE, which spills via :meth:`materialize`, and
    the output drain — dense drains spill; the sparse drain returns this
    object, whose pytree leaves are the factor vals (the measured form).
    """

    _is_deferred_output = True

    def __init__(self, post, pre):
        self.post = post
        self.pre = pre

    def materialize(self):
        DEFERRED_OUTPUT_STATS["spill"] = (
            DEFERRED_OUTPUT_STATS.get("spill", 0) + 1)
        return sparse_matmul(self.post, self.pre)

    def dense(self):
        return self.materialize().dense()

    def copy(self):
        return self

    def tree_flatten(self):
        return (self.post, self.pre), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


from jax import tree_util as _jtu  # noqa: E402
_jtu.register_pytree_node(
    DeferredOutputProduct,
    lambda t: t.tree_flatten(),
    DeferredOutputProduct.tree_unflatten,
)



class FaceOperands(NamedTuple):
    """One face's two operands, DRAINED and ready for the contraction.

    ``post`` / ``pre`` are the working copies
    :func:`contract_face_operands` consumes; ``need_contract`` is the decision
    between a real contraction and an identity pass-through;
    ``pre_reattach`` / ``post_reattach`` are the Jacobian transforms that ride
    onto the RESULT rather than through it.
    """
    post: Any
    pre: Any
    need_contract: bool
    pre_reattach: Any
    post_reattach: Any
    #: the ``approx`` flag the operands were prepared under, and the STORED
    #: (un-drained) ``pre_val``. Both are needed by
    #: :func:`contract_face_operands`'s identity-pass-through branch, which
    #: asks a DIFFERENT operand's ``val`` depending on the flag -- EXACT AD
    #: keeps the stored tensor's test and is byte-identical.
    approx: bool = False
    stored_pre: Any = None


def prepare_face_operands(post_val, pre_val, *, approx: bool = False,
                          pre_hook=None, post_hook=None) -> FaceOperands:
    """Drain one face's two stored edge Jacobians into contraction operands.

    THE FIRST HALF OF THE FACE CONTRACTION, and the only copy of it:
    :func:`_eliminate_vertex` calls this, so a caller that needs a face's
    RESULT STRUCTURE before the elimination runs (alphagrad's per-vertex
    approximation mask) gets it from the same code the measurement will use.

    Args:
        post_val: the out-edge Jacobian, ``graph[central][out_edge]`` forced.
        pre_val: the in-edge Jacobian, ``transpose_graph[central][in_edge]``
            forced.
        approx: the elimination's ``_perpath or _is_approx_cfg`` -- whether the
            reconciler peel is performed and ``need_contract`` recomputed from
            the PEELED operands. EXACT AD passes ``False`` and is
            byte-identical to the pre-extraction code.
        pre_hook: the per-path ``pre`` hook (``_h_pre``), or None.
        post_hook: the per-path ``post`` hook (``_h_post``), or None.

    Returns:
        FaceOperands
    """
    # Handle stuff like reshape, squeeze etc.
    # Apply Jacobian transforms where applicable. ``unload_*``
    # already returns a fresh tensor, so only copy in the no-
    # transform branch — copying *then* overwriting with the unload
    # result (the old code) wasted a full tensor copy per edge.
    if len(pre_val.post_transforms) > 0 and post_val.val is not None:
        _post_val = unload_post_transforms(post_val, pre_val)
    else:
        _post_val = post_val.copy()

    # Seed-aware draining (shared with the triplet path).
    _post_val, _pre_val = _drain_or_unload_pre(
        post_val, pre_val, _post_val)

    # Multiply the two values of the edges if applicable. The real
    # contraction runs whenever both edges carry values OR a
    # ``val is None`` operand is a non-identity structural Jacobian
    # (a broadcast / reduction — see ``_acts_as_identity``); the
    # pass-through shortcuts below are only valid when the val-less
    # operand truly acts as the identity.
    _need_contract = (
        (pre_val.val is not None and post_val.val is not None)
        or (post_val.val is None and not _acts_as_identity(_post_val))
        or (pre_val.val is None and not _acts_as_identity(_pre_val))
    )
    # Per-path contraction-operand hooks (pre -> in-edge Jacobian,
    # post -> out-edge Jacobian). Applied to the working copies that
    # feed the contraction, mirroring face_env (pre->cf[u], post->cf[w]).
    if pre_hook is not None:
        _pre_val = pre_hook(_pre_val)
    if post_hook is not None:
        _post_val = post_hook(_post_val)

    # Reconciliation drain (2026-07-21). An operand can carry
    # ``seed_drainable`` transforms — concatenate slot embed/slice,
    # head slices, position-embed broadcast — that reconcile its
    # non-nominal STORED shape back to nominal (a concat slot is
    # stored at the FULL concat width and sliced to its own width on
    # drain; the stored edge is a bare identity-seed: empty dims,
    # ``val is None``, only the queued reconciler). The old code rode
    # those transforms THROUGH the contraction and re-attached them to
    # the OUTPUT. That is correct only while the reconcilable axis
    # stays a free dim: a diagonal ``dW/dV`` contracted against such a
    # ``dV/dU`` couples W's axis to U's axis in a block-diagonal pair
    # that PINS the full concat width onto ``U`` — a transform-free
    # NON-NOMINAL edge (logical 32 on a nominal-16/-1 axis) that then
    # broadcasts through every sibling merge until an ``N`` vs ``M``
    # (neither 1) collision crashes ``_reconcile_broadcast_dims`` (the
    # ViT-compress ``(1,32,8,16)`` vs ``(1,32,8,32)`` merge). Folding
    # the reconciler into the OPERAND here feeds the contraction a
    # nominal ``U`` so no coupling forms. Only the RELABEL remainder
    # (``_pre_reattach`` / ``_post_reattach``) rides the re-attach.
    # Gated on the approx config so EXACT AD (``transforms == ()``)
    # keeps the operands' full queues and is byte-identical.
    _pre_reattach = pre_val.pre_transforms
    _post_reattach = post_val.post_transforms
    if approx:
        if _pre_val.pre_transforms or _pre_val.post_transforms:
            _pre_val, _pre_rem_pre, _pre_rem_post = (
                _peel_reconciler_transforms(_pre_val)
            )
            _pre_reattach = _pre_rem_pre
        if _post_val.pre_transforms or _post_val.post_transforms:
            _post_val, _post_rem_pre, _post_rem_post = (
                _peel_reconciler_transforms(_post_val)
            )
            _post_reattach = _post_rem_post
        # Recompute the contraction decision from the PEELED operands.
        # ``_need_contract`` above was computed from the STORED
        # operands, where a concatenate/slice edge is a bare
        # identity-seed — ``_acts_as_identity`` sees the scalar seed
        # and chooses the pass-through. Peeling MATERIALISES that seed
        # into a real RECTANGULAR Jacobian (e.g. ``(8,32,8,16)`` for a
        # concat slot), which must be CONTRACTED, not passed through:
        # the stale decision returns the other operand verbatim
        # (``(8,32,8,32)``), pinning the concat width onto the input
        # axis. Only re-evaluated in the approx path, so EXACT AD is
        # untouched.
        _need_contract = (
            (_pre_val.val is not None and _post_val.val is not None)
            or (_post_val.val is None
                and not _acts_as_identity(_post_val))
            or (_pre_val.val is None
                and not _acts_as_identity(_pre_val))
        )
    return FaceOperands(_post_val, _pre_val, _need_contract,
                        _pre_reattach, _post_reattach,
                        approx=bool(approx), stored_pre=pre_val)


class FaceContraction(NamedTuple):
    """The contracted face edge plus the op counts the count path accumulates."""
    val: Any
    adds: int
    muls: int
    fmas: int
    mem: int


def contract_face_operands(ops: FaceOperands, *,
                           count_ops: bool = False) -> FaceContraction:
    """Contract one face's prepared operands into the new edge Jacobian.

    THE SECOND HALF OF THE FACE CONTRACTION, and the only copy of it. The three
    special cases are all here and stay here: scalar x scalar
    (``sparse_matmul`` rejects 0-rank operands), ONE scalar operand (a scale,
    ticket dsnn-3qm.68), and the ``count_ops`` path.

    The STRUCTURE of the result -- ``val.shape``, ``out_dims``,
    ``primal_dims``, the transform queues -- is a pure function of the two
    operands' structures, which is what lets alphagrad compute a face's
    ``res:new`` approximation mask from the operands alone, with no speculative
    elimination. Calling THIS function is what keeps that mask from being a
    second copy of the structure algebra.

    Args:
        ops: the result of :func:`prepare_face_operands`.
        count_ops: accumulate adds / muls / fmas / mem (the cost path).

    Returns:
        FaceContraction
    """
    _post_val, _pre_val = ops.post, ops.pre
    _need_contract = ops.need_contract
    _pre_reattach, _post_reattach = ops.pre_reattach, ops.post_reattach
    adds = muls = fmas = mem = 0
    if _need_contract:
        # A scalar × scalar contraction is an elementwise multiply:
        # ``sparse_matmul`` rejects 0-rank operands, so it must NEVER
        # be routed through matmul — on either the count or non-count
        # path (the count path used to crash here).
        if _is_scalar_st(_post_val) and _is_scalar_st(_pre_val):
            edge_outval = _post_val * _pre_val
            if count_ops:
                muls += 1
        elif _is_scalar_st(_post_val) or _is_scalar_st(_pre_val):
            # ONE rank-0 edge. A scalar has no axes to contract, so
            # the chain rule here is a SCALE, not a matmul. This
            # used to fall through to ``@``, which rerouted it
            # silently inside matmul; matmul now raises
            # (ScalarMatmul, ticket dsnn-3qm.68) and the routing
            # belongs here, at the site that knows it is composing.
            from graphax.sparse.ops.matmul import scale_by_scalar
            _sc, _tn = ((_post_val, _pre_val)
                        if _is_scalar_st(_post_val)
                        else (_pre_val, _post_val))
            edge_outval = scale_by_scalar(_tn, _sc)
            if count_ops:
                muls += int(edge_outval.size)
        elif count_ops:
            edge_outval, (_a, _m, _f) = sparse_matmul(
                _post_val, _pre_val, count=True
            )
            adds += int(_a)
            muls += int(_m)
            fmas += int(_f)
        else:
            edge_outval = _post_val @ _pre_val
        if count_ops:
            post_size = (
                _post_val.val.size if _post_val.val is not None else 0
            )
            pre_size = _pre_val.val.size if _pre_val.val is not None else 0
            out_size = (
                edge_outval.val.size
                if edge_outval.val is not None
                else 0
            )
            mem += max(
                post_size * _post_val.dtype.itemsize,
                pre_size * _pre_val.dtype.itemsize,
                out_size * edge_outval.dtype.itemsize,
            )

    elif (_pre_val.val is not None if ops.approx
          else ops.stored_pre.val is not None):
        # post is a pure-diagonal identity up to its scalar_mult:
        # pass pre through, FOLDING post's scalar_mult (a scalar /
        # scaled-identity edge multiplies by it; dropping it was the
        # ``sum(z*sum(z))`` bug — 10·pre became pre). In the approx
        # path a peeled operand's own ``val`` decides which side is the
        # identity (the stored ``pre_val`` may be a since-materialised
        # seed); EXACT AD keeps the original ``pre_val.val`` test and
        # is byte-identical.
        edge_outval = _identity_passthrough(_pre_val, _post_val, "pre")
        if count_ops:
            muls += 1
    else:
        # pre is the identity (up to scalar_mult): pass post through.
        edge_outval = _identity_passthrough(_post_val, _pre_val, "post")
        if count_ops:
            muls += 1
    # Offload the remaining (un-peeled) Jacobian transforms to the
    # output tensor. ``_post_reattach`` / ``_pre_reattach`` are the
    # operands' full queues on EXACT AD (byte-identical to the old
    # ``prepend_post_transforms`` / ``append_pre_transforms``) and the
    # RELABEL remainder after a reconciler peel in an approx config.
    if len(_post_reattach) > 0:
        edge_outval.post_transforms = (
            tuple(_post_reattach) + tuple(edge_outval.post_transforms)
        )

    if len(_pre_reattach) > 0:
        edge_outval.pre_transforms = (
            tuple(_pre_reattach) + tuple(edge_outval.pre_transforms)
        )
    return FaceContraction(edge_outval, adds, muls, fmas, mem)


def contract_face(post_val, pre_val, *, approx: bool = False,
                  count_ops: bool = False,
                  pre_hook=None, post_hook=None) -> FaceContraction:
    """:func:`prepare_face_operands` then :func:`contract_face_operands`.

    The whole face contraction in one call, for a caller that does not need to
    interpose between the two halves. :func:`_eliminate_vertex` does need to
    (the deferred-output fast path of #46 sits between them), so it calls the
    two halves; everything else should call this.
    """
    return contract_face_operands(
        prepare_face_operands(post_val, pre_val, approx=approx,
                              pre_hook=pre_hook, post_hook=post_hook),
        count_ops=count_ops)


def _eliminate_vertex(
    vertex: int,
    jaxpr: core.Jaxpr,
    graph: ComputationalGraph,
    transpose_graph: ComputationalGraph,
    vo_vertices: Set[core.Var],
    count_ops: bool = False,
    transforms: Sequence[
        Union[Diag, Compress, Callable[["SparseTensor"], "SparseTensor"]]
    ] = (),
    face_transforms: Union[
        Dict[
            Tuple[int, int],
            Tuple[
                Union[None, Diag, Compress, Quant,
                      Callable[["SparseTensor"], "SparseTensor"]], ...
            ],
        ],
        None,
    ] = None,
    var_vid: Dict[core.Var, int] = None,
) -> Tuple[int, int, int, int]:
    """
    Function that eliminates a vertex from the computational graph.
    everything that has a _val in its name is a `SparseTensor` object

    Args:
        vertex (int): The vertex we want to eliminate from the computational graph
                    according to the vertex elimination rule as described in
                    cross-country elimination.
        jaxpr (core.Jaxpr): The jaxpression derived by tracing the input function
                            whose Jacobian we intend to calculate.
        graph (ComputationalGraph): Computational graph representation derived
                                    from `jaxpr`.
        transpose_graph (ComputationalGraph): Transpose computational graph
                                                derived from `jaxpr`.
        vo_vertices (Set[core.Var]): A `set` containing all the output vertices.
        count_ops (bool): If True, track adds/muls/fmas/peak-mem during the
                          elimination and return them; otherwise return zeros.
        transforms (Sequence[Union[Diag, Compress, Callable]]): Per-vertex
            Jacobian transforms applied IN ORDER to each ``edge_outval``
            before it's wired back into the graph. Each transform is one
            of:
              * :class:`Diag` — block-diagonalise a logical-index pair
                (``Diag(i, j, factor)``) via :func:`apply_diag`.
              * :class:`Compress` — mean-compress one or more physical
                axes (``Compress(axes)``) via :func:`apply_compress`.
              * a callable ``(SparseTensor) -> SparseTensor`` — escape
                hatch for arbitrary user-defined transforms.
            ``DIAG ∘ COMPRESS ≠ COMPRESS ∘ DIAG``, so the sequence order
            is preserved exactly. Defaults to ``()`` (no transforms).
        face_transforms (Dict[Tuple[int, int], Tuple]): PER-FACE (per local
            path) transforms, i.e. one choice per ``in_edge -> central_var ->
            out_edge`` path rather than one choice for the whole vertex. Maps a
            FACE KEY ``(vidx[in_edge], vidx[out_edge])`` — the same stable var
            index :func:`_stable_var_index` produces, enumerable up front with
            :func:`faces_of` — to a 3-tuple ``(lhs, rhs, res)`` of slots named
            after the local path ``res = op(lhs, rhs)``:

              * ``lhs`` is applied to ``pre_val``, the in_edge Jacobian, and
                ``rhs`` to ``post_val``, the out_edge Jacobian, BOTH before the
                contraction;
              * ``res`` is applied to ``edge_outval``, the contraction result,
                at the same site as ``transforms`` above and immediately AFTER
                them (so a per-vertex rule composes before a per-face one).

            Each slot is ``None`` (leave that operand exact), a :class:`Diag` /
            :class:`Compress` / :class:`Quant`, or a callable
            ``(SparseTensor) -> SparseTensor``. Slots follow the same
            best-effort semantics as ``transforms``: a ``ValueError`` (the
            transform does not fit that operand's geometry) skips just that
            slot, a ``TypeError`` propagates. Keys with no matching face are
            silently unused. ``None`` (the default) leaves the exact-AD /
            per-vertex path byte-identical.

            An entry may instead be the TWO-OP form
            ``((lhs, rhs, new), (jl, jr, jres))``, which splits the single
            ``res`` slot into the four sites the result passes through —
            ``new`` (the fresh contraction, pre-join), ``jl``/``jr`` (the two
            addends of the merge into an EXISTING edge) and ``jres`` (the
            merged sum, exactly where ``res`` lands). It exists to express
            ``old = approx(old) + approx(new)``, which a flat triple cannot:
            its ``res`` approximates the SUM, not the two addends. See
            :func:`_unpack_face_slots` for the full semantics, the merge-free
            degeneration, and the slot tags the sinks see.

    Returns:
        Tuple[int, int, int, int]: ``(adds, muls, fmas, mem)`` accumulated
            during this vertex elimination, all zero unless ``count_ops``.
    """
    eqn = jaxpr.eqns[vertex - 1]
    adds = muls = fmas = mem = 0

    # Whether this vertex carries a Diag/Compress approximation — invariant over
    # every edge, so compute it ONCE here rather than per (in_edge, out_edge).
    # Gates the approx-edge normalization below; ``transforms == ()`` (the EXACT
    # AD path) gives ``any([]) == False`` so that path stays byte-identical.
    # A transform is an approximation if it is a Diag/Compress instance OR a CALLABLE —
    # the `transforms` API documents "a callable (SparseTensor) -> SparseTensor — escape
    # hatch for arbitrary user-defined transforms", but the old isinstance-only test did
    # not match callables, so a callable transform set _is_approx_cfg=False and SILENTLY
    # BYPASSED the whole approx path: every `_normalize_approx_edge` gate is keyed off this
    # (and `_approx_elim`), so a non-nominal approx edge sailed into the merge-path shape
    # assert => "Computed edge shape (10,4,16,16) does not match expected shape (4,16,16)"
    # (MoE) / "matmul: mismatch in core dimension 0" (NN). Callables are exactly what a
    # MASK-AWARE policy must use (the mask needs the real edge, only known mid-elimination),
    # so this silently broke the one correct way to apply approximations.
    # Per-path (face-like) mode: `transforms` is a dict keyed by
    # (primal_vertex_id, out_vertex_id). Hooks are applied at the op boundary; the
    # legacy list form applies its transforms post-merge. NEITHER normalizes to a
    # nominal dense edge any more (the norm was deleted) — the sparse matmul/+
    # reconcile every layout. Exact AD (`transforms == ()`) leaves `_is_approx_cfg`
    # False so it stays byte-identical.
    _perpath = isinstance(transforms, dict)
    _is_approx_cfg = (not _perpath) and any(
        isinstance(_t, (Diag, Compress)) or callable(_t) for _t in transforms
    )
    # A PER-FACE Diag/Compress approximates this vertex as much as a per-vertex
    # one, so it must arm the same non-shortcut contraction path. Guarded on
    # ``face_transforms`` so the None path keeps the value above.
    if face_transforms:
        # SKIP_FACE is a bare sentinel value (not a 3-tuple of slots) and a
        # skipped path deviates from the exact structure at least as much as
        # a Diag/Compress does — it must arm the approx config (the nominal
        # shape asserts are exact-only).
        #
        # Iterated through ``_iter_face_hooks`` so the TWO-OP form
        # ``((lhs, rhs, new), (jl, jr, jres))`` arms this exactly as its flat
        # equivalent does: without the flatten the loop sees the two TRIPLES,
        # neither of which is a Diag/Compress, so the identical approximation
        # armed the approx config in one form and not the other — and this
        # flag gates the reconciler peel and the pre-Diag drain, so the two
        # forms took DIFFERENT code paths for the same request.
        _is_approx_cfg = _is_approx_cfg or face_config_is_approx(
            face_transforms)

    # Path tokenization sink (None on the exact-AD hot path -> zero overhead,
    # every contraction/accumulation runs inline exactly as before).
    # Face sink: records per-face edge identities + equation ranges into the
    # PRESERVED trace's frame (contraction/join vs each approximation), so the
    # tokenizer can label "which elimination, which path, which approx". None on
    # the exact-AD path -> zero overhead.
    _face_sink = _get_face_sink()
    # Always-on transform log (``sparse.tracer.TransformLog``): records EVERY
    # dispatched micro-action with a truthful ``applied`` flag, independent of
    # ``track_faces``. One thread-local lookup per vertex elimination; ``None``
    # (zero cost) unless a builder installed one.
    _xlog = _get_transform_log()

    # #46 deferred outputs: static per-call sets for the deferral guard.
    _FACTORED = _factored_outputs_enabled()
    if _FACTORED:
        _graph_input_vars = frozenset(jaxpr.invars)
        _pure_out_vars = frozenset(jaxpr.outvars) - set(vo_vertices)

    # In tokenize mode, iterate the edge maps in a STABLE order (they are keyed
    # by id-hashed core.Var, so their native iteration order varies per trace,
    # which would make the emitted op stream — and its first-appearance names —
    # nondeterministic). Sorting by first-appearance position in the jaxpr fixes
    # the order without touching the exact-AD path (numeric result is
    # order-invariant; only token determinism needs it). The index is invariant
    # over the whole elimination, so build it ONCE and cache it on the sink
    # (rebuilding per vertex would be O(V^2)).
    _vidx = None
    if _face_sink is not None:
        _vidx = _face_sink.vidx
        if _vidx is None:
            _vidx = _face_sink.vidx = _stable_var_index(jaxpr)
    elif face_transforms is not None:
        # Per-face transforms are keyed by the SAME index, so it has to exist
        # even with face tracking OFF (no sink to cache it on) -> memoized on
        # the last jaxpr instead. This also stabilizes ``_ordered`` below, so
        # ``faces_of`` enumerates in exactly the order used here.
        _vidx = _vidx_for(jaxpr)

    def _ordered(keys):
        if _vidx is None:
            return keys
        return sorted(keys, key=lambda v: _vidx.get(v, 1 << 30))

    for central_var in eqn.outvars:
        if central_var not in graph:
            continue  # dead or already-eliminated vertex

        for out_edge in _ordered(graph[central_var].keys()):
            _post_raw = _force(graph[central_var][out_edge])
            if _post_raw is None:
                continue  # no Jacobian for this out-edge; skip
            # ``_post_raw`` / ``_pre_raw`` come straight from the memoized
            # ``_force`` cache; ``post_val`` / ``pre_val`` only READ them (their
            # transform lists + ``.val``) and are never used after the working
            # values are built, so they can alias the cache directly — the
            # private working copies below absorb every in-place mutation
            # (a defensive ``.copy()`` of these read-only bases was pure
            # per-edge overhead on the O(E²) AD hot path).
            post_val = _post_raw
            for in_edge in _ordered(transpose_graph[central_var].keys()):
                _pre_raw = _force(transpose_graph[central_var][in_edge])
                if _pre_raw is None:
                    continue  # no Jacobian (e.g. stop_gradient blocks grad); skip
                pre_val = _pre_raw
                if _face_sink is not None:
                    # one FACE = this (in_edge -> central_var -> out_edge) path
                    _face_sink.open_face(vertex, in_edge, central_var, out_edge)

                # ---- PER-FACE transforms: the ``lhs`` / ``rhs`` slots --------
                # This face's local path is ``res = op(lhs, rhs)`` with
                # ``lhs = pre_val`` (the in_edge Jacobian) and ``rhs = post_val``
                # (the out_edge Jacobian), so both slots land HERE, before the
                # contraction below consumes them; the ``res`` slot is carried in
                # ``_face_res_t`` down to the per-vertex transform site.
                #
                # ``post_val`` is bound ONCE per out_edge in the enclosing loop
                # and shared by every in_edge, so it is re-seeded from the
                # read-only ``_post_raw`` base here — an ``rhs`` transform must
                # affect THIS face only, not the rest of the out_edge's fan-in.
                # ``pre_val`` is already re-seeded per face just above.
                #
                # The whole block is skipped when ``face_transforms is None``,
                # so the exact-AD / per-vertex path is untouched.
                _face_res_t = None
                _face_new_t = None
                _face_join_t = None
                # Transform-LOG tag for the ``res``-site hook: the flat form's
                # own ``res``, or the two-op form's ``jres`` (same site, same
                # FaceSink slot -- see _unpack_face_slots "SLOT TAGS").
                _face_res_log = "res"
                if face_transforms is not None:
                    post_val = _post_raw
                    _slots = face_transforms.get(
                        (_vidx.get(in_edge), _vidx.get(out_edge))
                    )
                    if _slots is SKIP_FACE:
                        # SKIP: drop this path outright — no contraction, no
                        # join, no counts. Recorded (and the face CLOSED — the
                        # ``continue`` bypasses the loop-bottom close) so the
                        # face renders as ``approx SKIP`` rather than silence.
                        _n = _eqn_count(_face_sink, _xlog)
                        if _face_sink is not None:
                            _face_sink.approx("SKIP", {}, _n, _n)
                            _face_sink.close_face()
                        if _xlog is not None:
                            _xlog.record("transform", vertex, "face", "SKIP",
                                         {}, in_edge, out_edge, _n, _n, True)
                        continue
                    if _slots is not None:
                        (_lhs_t, _rhs_t, _face_res_t, _face_new_t,
                         _face_join_t) = _unpack_face_slots(_slots, vertex)
                        if _is_two_op_slots(_slots):
                            _face_res_log = "res:jres"
                        pre_val = _apply_face_transform(
                            pre_val, _lhs_t, "lhs", vertex, _face_sink,
                            in_edge, out_edge, _xlog)
                        post_val = _apply_face_transform(
                            post_val, _rhs_t, "rhs", vertex, _face_sink,
                            in_edge, out_edge, _xlog)

                # Resolve this path's per-op hooks. The path (in_edge -> vertex ->
                # out_edge) is keyed by its neighbour vertex ids; ``_h_*`` default
                # to None (identity) for any unaddressed path or role.
                (_h_pre, _h_post, _h_new), (_h_lhs, _h_rhs, _h_res) = _NULL3, _NULL3
                if _perpath:
                    _pid = var_vid.get(in_edge) if var_vid is not None else None
                    _oid = var_vid.get(out_edge) if var_vid is not None else None
                    (_h_pre, _h_post, _h_new), (_h_lhs, _h_rhs, _h_res) = (
                        _parse_path_hooks(transforms.get((_pid, _oid)))
                    )

                # TODO implement a process that discards unnecessary edges from the computation

                # THE FACE CONTRACTION, FIRST HALF (``prepare_face_operands``).
                # Extracted so the structure algebra has exactly ONE
                # implementation: alphagrad's per-vertex approximation mask
                # needs a face's RESULT STRUCTURE before this elimination runs,
                # and a second copy of this arithmetic is what produced
                # finding 72's fault 1.
                _ops = prepare_face_operands(
                    post_val, pre_val,
                    approx=bool(_perpath or _is_approx_cfg),
                    pre_hook=_h_pre if _perpath else None,
                    post_hook=_h_post if _perpath else None)
                _post_val, _pre_val = _ops.post, _ops.pre
                _need_contract = _ops.need_contract
                _pre_reattach, _post_reattach = (_ops.pre_reattach,
                                                 _ops.post_reattach)
                if (
                    _FACTORED
                    and _need_contract
                    and not count_ops
                    and out_edge in _pure_out_vars
                    and in_edge in _graph_input_vars
                    and graph.get(in_edge).get(out_edge) is None
                    and _post_val.val is not None
                    and _pre_val.val is not None
                    and not (_post_val.pre_transforms or _post_val.post_transforms)
                    and not (_pre_val.pre_transforms or _pre_val.post_transforms)
                    and not _pre_reattach
                    and not _post_reattach
                    and _h_new is None
                    and _h_lhs is None
                    and _h_res is None
                    and _face_res_t is None
                    and _face_new_t is None
                    # NOTE: ``_face_join_t`` (the two-op ``jl``/``jr``) is
                    # deliberately NOT tested here. Those two hooks run ONLY
                    # in the merge branch below, which needs an existing
                    # ``graph[in_edge][out_edge]`` -- and this fast path
                    # already requires that edge to be None (above). So on
                    # every face this branch can fire, the join hooks would
                    # not have run on the slow path either: deferring is
                    # behaviour-preserving, and adding the test would only
                    # disable the optimisation for hooks that are dead here.
                    and (_perpath or not transforms)
                    and math.prod(out_edge.aval.shape)
                    * math.prod(in_edge.aval.shape)
                    >= 8 * (_post_val.val.size + _pre_val.val.size)
                ):
                    # DEFERRED FINAL CONTRACTION (#46): this (input -> pure
                    # output) edge can only be merged (spills) or drained.
                    # Store the factor pair; the sparse drain returns it, the
                    # dense drain materializes byte-identically. Operand
                    # hooks (_h_pre/_h_post) were already applied above, so
                    # the deferred product IS the hooked contraction.
                    DEFERRED_OUTPUT_STATS["defer"] = (
                        DEFERRED_OUTPUT_STATS.get("defer", 0) + 1)
                    _dp = DeferredOutputProduct(_post_val, _pre_val)
                    _record_edge_store(_dp)
                    _set_inner(graph, in_edge, out_edge, _dp)
                    _set_inner(transpose_graph, out_edge, in_edge, _dp, is_transpose=True)
                    if _face_sink is not None:
                        _face_sink.close_face()
                    continue
                # THE FACE CONTRACTION, SECOND HALF
                # (``contract_face_operands``) -- the same function alphagrad's
                # mask composes face structures with.
                # ``_ops`` verbatim: the deferred-output guard between the two
                # halves only READS the operands, so the second half consumes
                # exactly what the first produced.
                _fc = contract_face_operands(_ops, count_ops=count_ops)
                edge_outval = _fc.val
                if count_ops:
                    adds += _fc.adds
                    muls += _fc.muls
                    fmas += _fc.fmas
                    mem += _fc.mem

                # Per-path contraction-RESULT hook (``new``). The face engine
                # applies its ``new`` to the product; here we apply it to the
                # freshly-contracted edge before the join. A join ``lhs`` hook
                # (below) then composes on top for the merge case.
                if _perpath and _h_new is not None:
                    edge_outval = _h_new(edge_outval)
                    _assert_sparse_tensor_consistency(edge_outval)
                # PER-FACE two-op form: this face's ``new`` hook on the
                # fresh contraction BEFORE any join -- the
                # ``approx(contract)`` half of
                # ``new = approx(new) + approx(contract)``.
                if _face_new_t is not None:
                    edge_outval = _apply_face_transform(
                        edge_outval, _face_new_t, "res", vertex,
                        _face_sink, in_edge, out_edge, _xlog,
                        log_slot="res:new")

                _assert_sparse_tensor_consistency(edge_outval)
                # If there is already an edge between the two vertices, add the new
                # edge to the existing one
                if graph.get(in_edge).get(out_edge) is not None:
                    _edge = _force(transpose_graph[out_edge][in_edge])
                    if getattr(_edge, "_is_deferred_output", False):
                        # a later contribution reached a deferred output
                        # edge: spill (always correct) and merge densely.
                        _edge = _edge.materialize()
                    _assert_sparse_tensor_consistency(_edge)

                    # Offload the remaining Jacobian transforms to each tensor
                    edge_outval = _drain_transforms(edge_outval)
                    _assert_sparse_tensor_consistency(edge_outval)

                    _edge = _drain_transforms(_edge)
                    _assert_sparse_tensor_consistency(_edge)

                    # Per-path JOIN operand hooks: ``lhs`` -> the new contribution,
                    # ``rhs`` -> the existing edge. Applied to the drained addends
                    # before the sparse ``+`` (which reconciles their layouts).
                    if _perpath and _h_lhs is not None:
                        edge_outval = _h_lhs(edge_outval)
                    if _perpath and _h_rhs is not None:
                        _edge = _h_rhs(_edge)
                    # PER-FACE two-op form JOIN hooks: ``jl`` on the new
                    # contribution (usually None -- hooked pre-join),
                    # ``jr`` on the EXISTING edge: the ``approx(new)``
                    # half of ``new = approx(new) + approx(contract)``.
                    # BOTH live inside this ``existing edge`` branch on
                    # purpose -- a merge-free face has no operand for ``jr``
                    # and no site for ``jl`` distinct from ``new``'s (see
                    # _unpack_face_slots, "MERGE-FREE FACES").
                    if _face_join_t is not None:
                        _jl_t, _jr_t = _face_join_t
                        if _jl_t is not None:
                            edge_outval = _apply_face_transform(
                                edge_outval, _jl_t, "res", vertex,
                                _face_sink, in_edge, out_edge, _xlog,
                                log_slot="res:jl")
                        if isinstance(_jr_t, FaceJoinPolicy):
                            # A JOIN POLICY, not a hook. "Make these two
                            # addends share one container" is inherently a
                            # BINARY operation: every hook position sees one
                            # tensor, so none of them can express it, and the
                            # asymmetry it removes is what made one legality
                            # mask unable to describe both addend sites
                            # (finding 72 / dsnn-3qm.59 fault 1). The ``jr``
                            # position carries it because ``jr`` is the old
                            # edge's slot and the old edge is what moves; the
                            # policy's own ``pre`` hook is the single-tensor
                            # slot a learned approximation of the old edge
                            # occupies, and it runs inside ``reconcile`` so the
                            # reconciliation still has the last word on the
                            # structure. ``jl`` (above) and ``jres`` (below)
                            # stay ordinary hook positions.
                            edge_outval, _edge = _jr_t.reconcile(
                                edge_outval, _edge)
                            _assert_sparse_tensor_consistency(edge_outval)
                            _assert_sparse_tensor_consistency(_edge)
                        elif _jr_t is not None:
                            _edge = _apply_face_transform(
                                _edge, _jr_t, "res", vertex,
                                _face_sink, in_edge, out_edge, _xlog,
                                log_slot="res:jr")

                    # Nominal-shape asserts hold only for EXACT AD (no approx of
                    # any kind): an approximation (per-path OR legacy list) can
                    # leave an edge sparse/permuted, and the sparse ``+`` reconciles
                    # it — there is no normalization to nominal any more.
                    # The GLOBAL approx flag must gate too: under the keep-sparse
                    # redesign a permuted edge from an approximated vertex legally
                    # reaches a NON-approx vertex's merge — the per-vertex gate
                    # alone is stale there (ViT layer_norm case).
                    if not _perpath and not _is_approx_cfg:
                        edge_shape = tuple(
                            list(out_edge.aval.shape) + list(in_edge.aval.shape)
                        )
                        if not (edge_shape == edge_outval.shape):
                            raise RuntimeError(f"Computed edge shape {edge_outval.shape} does not match expected shape {edge_shape}!")
                        if not (edge_shape == _edge.shape):
                            raise RuntimeError(f"Existing edge shape {_edge.shape} does not match expected shape {edge_shape}!")
                    if count_ops:
                        edge_outval, (_a, _m, _f) = add_w_counts(edge_outval, _edge)
                        adds += int(_a)
                        muls += int(_m)
                        fmas += int(_f)
                        mem += (
                            edge_outval.val.size
                            if edge_outval.val is not None
                            else 0
                        ) * edge_outval.dtype.itemsize
                    else:
                        edge_outval += _edge

                    # Per-path JOIN result hook (``res`` -> the summed edge).
                    if _perpath and _h_res is not None:
                        edge_outval = _h_res(edge_outval)

                # Drain queued Jacobian transforms (slice / concatenate / reshape
                # / transpose relabels awaiting embed) into the edge BEFORE the
                # per-vertex Diag / Compress. A head-slice edge
                # (``q[:, h*dh:(h+1)*dh]``) carries a queued embed whose
                # ``apply_inverse`` grows the head-sliced free axis ``dh`` back to
                # the full graph-variable size ``D``. Diag must NOT run while that
                # transform is still queued: the block split mutates the dim list
                # (ids / axes / pair count), so a later drain of a transpose
                # relabel indexes a now-stale ``other_id`` (IndexError) and a
                # slice embed lands on block-structured axes it can't represent
                # (malformed dense / size mismatch) — both surfaced by Diag on the
                # slice/concat multi-head ViT under non-canonical orders. Restrict
                # to a MATERIALIZED, non-scalar edge: a ``val is None`` / 0-rank
                # structural-identity edge (e.g. a reshape seed) can't be embedded
                # and its transform is irrelevant to Diag (which ValueError-skips
                # a non-fitting edge anyway). Gated on the approx config so
                # EXACT-AD (``transforms == ()``) stays byte-identical.
                if (
                    _is_approx_cfg
                    and (edge_outval.pre_transforms or edge_outval.post_transforms)
                    and edge_outval.val is not None
                    and (edge_outval.out_dims or edge_outval.primal_dims)
                ):
                    # Drain queued transforms (NO densify) so the legacy per-vertex
                    # Diag/Compress below sees a clean edge; the sparse ops reconcile
                    # downstream — there is no normalization to nominal any more.
                    edge_outval = _drain_transforms(edge_outval)
                    _assert_sparse_tensor_consistency(edge_outval)

                # Apply per-vertex transforms in order. Diag / Compress are
                # dispatched to the atomic helpers in micro_actions; any
                # other callable is given the edge_outval directly. The
                # dispatch happens at Python time; each helper produces
                # traced JAX ops so the resulting jaxpr is statically
                # determined.
                #
                # A transform may legitimately not fit the current edge
                # even though it fit the nominal `(out_dims, primal_dims)`
                # signature — the SparseTensor's physical `val.ndim` can
                # be smaller than the nominal axis count (sparse
                # representations omit dims of size 1 / diagonal axes), and
                # earlier transforms in the same sequence may shrink it
                # further. apply_diag / apply_compress raise ValueError in
                # that case; we catch and skip the offending transform so
                # the caller's "best-effort" intent — apply what fits, drop
                # what doesn't — round-trips through to JAX correctly. The
                # alternative is to push the full edge geometry up to the
                # caller so it can pre-filter, which couples the typed
                # transform API to internal sparse representations.
                #
                # Per-path (dict) mode applied its hooks at the op boundaries
                # above, so the legacy per-vertex list loop is skipped here.
                for _t in (() if _perpath else transforms):
                    try:
                        if isinstance(_t, (Diag, Compress, Quant)):
                            # ONE dispatch (``_apply_micro``); the recorders are
                            # pure instrumentation logging the eqn range around
                            # it (a labelled ``approx`` block on the face sink,
                            # a TransformRecord on the always-on log).
                            _as = _eqn_count(_face_sink, _xlog)
                            _before = edge_outval
                            edge_outval = _apply_micro(edge_outval, _t)
                            _record_micro(_t, _before, edge_outval, vertex,
                                          "vertex", in_edge, out_edge, _as,
                                          _face_sink, _xlog)
                        elif callable(_t):
                            edge_outval = _t(edge_outval)
                        else:
                            raise TypeError(
                                f"Unknown transform of type "
                                f"{type(_t).__name__} at vertex {vertex}; "
                                "expected Diag, Compress, Quant, or a callable "
                                "(SparseTensor) -> SparseTensor."
                            )
                    except ValueError as _exc:
                        # LOUD BY DEFAULT (2026-07-15). This used to `continue`,
                        # silently dropping the transform on this edge. That silent
                        # skip is the single defect behind this whole bug family:
                        # two edges meeting at a shared variable receive DIFFERENT
                        # effective approximations and desync in rank/extent (the
                        # "Contraction size mismatch" family), and a 100%-skipped
                        # Diag reports cos=1.0 — indistinguishable from a working
                        # approximation. Measured: blind Diag failed 47/47 on
                        # NN/MoE/ViT, every one swallowed here.
                        #
                        # The old note said this skip was sound ONLY because
                        # _normalize_approx_edge densifies every approx edge back to
                        # nominal so the NEXT transform always fits, and warned:
                        # "don't weaken the densify without first making every
                        # transform structure-invariant". That is precisely what
                        # up-front masking now does (diag_mask / compress_mask):
                        # an invalid action is never PROPOSED, so this handler
                        # should be unreachable. If it fires, that is a REAL bug
                        # (or an unmasked caller) and must be seen, not buried.
                        #
                        # Escape hatch for the legacy best-effort contract:
                        #   GRAPHAX_BEST_EFFORT_TRANSFORMS=1
                        if os.environ.get("GRAPHAX_BEST_EFFORT_TRANSFORMS", "0") == "1":
                            continue
                        _f = lambda ds: [
                            (d.size, getattr(d, "other_id", None),
                             getattr(d, "block_size", None), getattr(d, "axis", None))
                            for d in ds
                        ]
                        raise ValueError(
                            f"TRANSFORM DID NOT FIT at vertex {vertex}: {_t!r} on edge "
                            f"(in={in_edge}, out={out_edge}) -> {type(_exc).__name__}: {_exc}. "
                            f"edge out_dims={_f(edge_outval.out_dims)} "
                            f"primal_dims={_f(edge_outval.primal_dims)} "
                            f"val={None if edge_outval.val is None else tuple(edge_outval.val.shape)}. "
                            "This transform was previously SKIPPED SILENTLY, which desyncs the two "
                            "edges of a shared variable and makes a no-op approximation report "
                            "cos=1.0. Mask invalid actions up front (see diag_mask/compress_mask) "
                            "instead of discovering them by throwing. Set "
                            "GRAPHAX_BEST_EFFORT_TRANSFORMS=1 to restore the legacy silent skip."
                        ) from _exc
                    _assert_sparse_tensor_consistency(edge_outval)

                # ---- PER-FACE transforms: the ``res`` slot -------------------
                # ``res`` is this face's contraction result, so it is applied at
                # the SAME site as the per-vertex ``transforms`` above and
                # immediately AFTER them: a per-vertex rule (uniform over every
                # face) composes first, then this face's own choice. Same
                # best-effort ValueError skip, same approx recording.
                if _face_res_t is not None:
                    edge_outval = _apply_face_transform(
                        edge_outval, _face_res_t, "res", vertex, _face_sink,
                        in_edge, out_edge, _xlog, log_slot=_face_res_log)

                # Post-transform edges stay SPARSE: normalization to nominal dense
                # form was deleted with the rest of the norm. A freshly Diag-split /
                # Compress-implicit edge rides into the next vertex's contraction and
                # any later merge as-is, and the sparse matmul/+ reconcile its layout
                # (the same guarantee the face engine relies on).

                # NOTE: the previous KNOWN-INCOMPLETE "densify approx edge to
                # nominal" band-aid that lived here was removed — the structured
                # rectangular-Diag / implicit-Compress contractions it papered
                # over are now handled at the op boundary by the elemental
                # composition layer (graphax.sparse.elemental.dispatch), wired as
                # the first fast path in matmul() / elementwise(). The band-aid's
                # fresh-id rebuild mis-aligned downstream multi-edge contractions
                # (permuted edge_outval → merge shape-assert), which the kernels'
                # canonical output-id convention now avoids.

                # Fold away non-data size-1 val axes that no dim references
                # before the edge is stored / accumulated. Block-diagonal
                # restructuring and Diag/Compress insert a fresh physical axis
                # per meta/block side and never reclaim the leftovers, so an
                # approx edge's ``val.ndim`` creeps toward the 32-axis cap while
                # its logical rank stays tiny. A pure reshape (byte-identical);
                # a strict no-op on the EXACT-AD path (tiled output is already
                # squeezed), so it never perturbs exact Jacobians.
                if os.environ.get("GX_NO_SQUEEZE", "0") != "1":
                    edge_outval = _squeeze_unreferenced_val_axes(edge_outval)

                _record_edge_store(edge_outval)
                _set_inner(graph, in_edge, out_edge, edge_outval)
                _set_inner(transpose_graph, out_edge, in_edge, edge_outval, is_transpose=True)
                if _face_sink is not None:
                    _face_sink.close_face()

        # Cleanup of input and output edges for this output variable
        if central_var not in vo_vertices:
            for in_vertex in list(transpose_graph[central_var].keys()):
                _del_inner(graph, in_vertex, central_var)
        for out_vertex in list(graph[central_var].keys()):
            _del_inner(transpose_graph, out_vertex, central_var)

        # Cleanup the eliminated vertex
        graph.pop(central_var, None)
        if central_var not in vo_vertices:
            transpose_graph.pop(central_var, None)

    return adds, muls, fmas, mem


def _is_persistent(obj) -> bool:
    """True for `immutables.Map` and its `MapMutation` proxy."""
    return isinstance(obj, immutables.Map) or hasattr(obj, "finish")


class StoredEdgeShapeMismatch(ValueError):
    """A stored edge's ``SparseTensor.shape`` is not ``out_edge.aval.shape +
    in_edge.aval.shape`` (ticket dsnn-3qm.71). The check is UNGATED -- an
    approximation never changes an edge's logical shape (measured, see
    ``tests/misc/test_nominal_shape_invariant.py``) -- and it RAISES rather
    than ``assert``-ing, because ``python -O`` deletes an ``assert``. An edge
    with a queued JacobianTransform is exempt: its shape is nominal only once
    drained."""


def _set_inner(outer, k1, k2, v, is_transpose=False):
    """Set ``outer[k1][k2] = v`` for both nested-defaultdict and immutables.Map proxies."""
    out_var, in_var = (k1, k2) if is_transpose else (k2, k1)
    if hasattr(out_var, "aval") and hasattr(in_var, "aval") and hasattr(v, "shape"):
        if not (getattr(v, "pre_transforms", ()) or getattr(v, "post_transforms", ())):
            expected = tuple(out_var.aval.shape) + tuple(in_var.aval.shape)
            if tuple(v.shape) != expected:
                raise StoredEdgeShapeMismatch(
                    f"Stored edge shape {tuple(v.shape)} does not match the "
                    f"nominal {expected} (out {out_var} x in {in_var})")

    inner = outer.get(k1)
    if _is_persistent(outer) or _is_persistent(inner):
        if inner is None:
            inner = immutables.Map()
        outer[k1] = inner.set(k2, v)
    else:
        if inner is None:
            outer[k1] = {k2: v}
        else:
            inner[k2] = v


def _del_inner(outer, k1, k2):
    """Delete ``outer[k1][k2]`` for both nested-defaultdict and immutables.Map proxies."""
    inner = outer.get(k1)
    if inner is None:
        return
    if _is_persistent(outer) or _is_persistent(inner):
        if k2 in inner:
            outer[k1] = inner.delete(k2)
    else:
        inner.pop(k2, None)


def _checkify_order(
    order: EliminationOrder, jaxpr: core.Jaxpr, vo_vertices: Set[core.Var]
) -> EliminationOrder:
    """
    Function that checks if the supplied elimination order is valid for the
    given computational graph/jaxpr. In the case of an elimination order that
    has been provided as a string, it first maps the string to the respective
    order:
    - "fwd", "forward": [1, 2, 3, ...]
    - "rev", "reverse": [..., 3, 2, 1]

    For explicit (numeric) orders, the input is filtered to keep only valid
    vertex IDs in their supplied relative order. Partial orders are
    supported: any eliminable vertex not listed in ``order`` is simply left
    in the graph after elimination — useful for staged / triplet-based
    elimination strategies. JAX scalar arrays are converted to Python ints.

    Args:
        order (EliminationOrder): The elimination order to check.
        jaxpr (core.Jaxpr): The jaxpr we want to differentiate.
        vo_vertices (Set[core.Var]): A `set` containing all the output vertices.

    Returns:
        EliminationOrder: A valid elimination order.
    """

    def _should_eliminate(eqn):
        """Include equation in the order if any of its outvars needs elimination."""
        return any(
            ov not in jaxpr.outvars or ov in vo_vertices
            for ov in eqn.outvars
            if isinstance(ov, core.Var)
        )

    if isinstance(order, str):
        if order == "forward" or order == "fwd":
            return [
                i for i, eqn in enumerate(jaxpr.eqns, start=1) if _should_eliminate(eqn)
            ]
        elif order == "reverse" or order == "rev":
            return [
                i for i, eqn in enumerate(jaxpr.eqns, start=1) if _should_eliminate(eqn)
            ][::-1]
        else:
            raise ValueError(f"{order} is not a valid order identifier!")

    # Numeric (explicit) order. Coerce array-like / JAX-scalar inputs to a
    # plain list of Python ints so downstream comparisons against the
    # eliminable-vertex set work uniformly.
    if hasattr(order, "tolist"):
        order = order.tolist()
    order = [int(o) for o in order]

    vertex_set = {
        i for i, eqn in enumerate(jaxpr.eqns, start=1) if _should_eliminate(eqn)
    }
    # Keep the supplied relative order, dropping anything that isn't an
    # eliminable vertex. Partial orders are valid: any vertex absent from
    # ``order`` is simply not eliminated.
    return [o for o in order if o in vertex_set]


def _eval_primal(eqn, invals):
    """Compute an equation's primal output value(s).

    Most primitives bind directly. ``custom_jvp_call_p`` / ``custom_vjp_call_p``
    cannot be re-bound from ``eqn.params`` (their sub-functions live in a
    ``subfuns`` slot the generic bind pops), so we evaluate the primal
    ``call_jaxpr`` instead — the gradient is supplied separately by the
    jvp/bwd-honoring elemental rules."""
    name = getattr(eqn.primitive, "name", "")
    if name in ("custom_jvp_call", "custom_vjp_call"):
        cj = eqn.params["call_jaxpr"]
        return core.eval_jaxpr(cj.jaxpr, cj.consts, *invals)
    if eqn.primitive is jit_p:  # named jit kept for by-name dispatch
        cj = eqn.params["jaxpr"]
        return core.eval_jaxpr(cj.jaxpr, cj.consts, *invals)
    if eqn.primitive is cond_p:  # evaluate the TAKEN branch (index is concrete)
        branches = eqn.params["branches"]
        index = max(0, min(int(invals[0]), len(branches) - 1))
        b = branches[index]
        return core.eval_jaxpr(b.jaxpr, b.consts, *invals[1:])
    return eqn.primitive.bind(*invals, **eqn.params)


def _build_graph(
    jaxpr: core.Jaxpr,
    args: Sequence[jnp.ndarray],
    consts: Sequence[core.Literal],
    argnums: Tuple[int, ...] = None,
    *,
    eqn_provenance=None,
    n_eqns_fn=None,
) -> Tuple[Dict, ComputationalGraph, ComputationalGraph, Set[core.Var]]:
    """
    This function performs the `tracing` of the jaxpression into a computational
    graph representation that is amenable to the vertex elimination procedure.
    The computational graph is stored as a dict of dicts where basically every
    item can be accessed through `graph[source_vertex][dest_vertex]` and yields
    the corresponding "partial Jacobian". The transpose computational graph stores
    the same information in reverse order, i.e. \n

    \t ``graph[sv][dv] == transpose_graph[dv][sv]`` \n

    where sv is the source and dv is the destination vertex. The computational
    graph will later evolve by applying the vertex elimination rule. In addition
    to the two graph obejects, this function also generates a `set` containing
    all intermediate and output vertices. This is necessary in order to later be
    able to determine ...

    Args:
        jaxpr (core.Jaxpr): The jaxpr we want to differentiate.
        args (Sequence[jnp.ndarray]): The input arguments of the function as a
                                        flattened PyTree.
        consts (Sequence[core.Literal]): The constant arguments of the function.
        argnums (Tuple[int, ...], optional): Positions in `jaxpr.invars` that
            are differentiable. When provided, edges are only emitted along
            paths reachable from these inputs (forward pruning), avoiding
            wasted work on dead branches. When ``None`` (default), all invars
            are treated as active.

    Returns:
        (env, graph, transpose_graph, vo_vertices)

    """
    env = {}  # env stores the primal value associated with the core.Var object

    graph = defaultdict(lambda: defaultdict())  # Input connectivity
    transpose_graph = defaultdict(lambda: defaultdict())  # Output connectivity

    vo_vertices = (
        set()
    )  # Set[core.Var]: outvars that are both intermediate and final outputs

    # active_vars tracks which Vars carry differentiable signal. When
    # ``argnums`` is given we seed it with the differentiable inputs and
    # propagate forward through eqns; non-active invars don't get edges.
    if argnums is not None:
        active_vars: Set[core.Var] = {jaxpr.invars[i] for i in argnums}
    else:
        active_vars = None  # disabled: treat every Var as active

    def is_active(var) -> bool:
        return active_vars is None or var in active_vars

    # Reads variable and corresponding traced shaped array
    def read(var):
        if isinstance(var, core.Literal):
            return var.val
        return env[var]

    # Adds new variable and corresponding traced shaped array
    def write(var, val):
        env[var] = val

    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    # NOTE: this is essentially the tracing part. Probably should write a proper
    # tracing system with lift etc. for better compatibility with JAX
    # Loop though elemental partials and create an abstract representation of
    # the computational graph
    # PROVENANCE (optional). Every traced base equation is emitted while
    # processing exactly one ORIGINAL equation, and original equation i is
    # vertex i+1 (1-based, as everywhere else). Recording the traced-equation
    # index at the START of each iteration yields consecutive spans that a
    # consumer can invert into `traced eqn index -> vertex`. Costs one list
    # append per equation and nothing at all when the hooks are absent.
    for _vidx0, eqn in enumerate(jaxpr.eqns):
        if eqn_provenance is not None and n_eqns_fn is not None:
            eqn_provenance.append((int(n_eqns_fn()), _vidx0 + 1))
        # Detect intermediate variables that are also final outputs
        for invar in eqn.invars:
            if invar in jaxpr._outvars:
                vo_vertices.add(invar)

        invals = safe_map(read, eqn.invars)

        if (
            eqn.primitive not in elemental_rules
            and eqn.primitive not in elemental_only_rules
            and eqn.primitive not in multi_output_elemental_only_rules
        ):
            raise NotImplementedError(
                f"{eqn.primitive} does not have registered elemental partial."
            )

        invals_snapshot = list(invals)
        # Pairs of (eqn.invars position, Var) for differentiable inputs. The
        # elemental rules return one entry per primal (i.e. per eqn.invars
        # position); we only wire edges for Var positions, but must index into
        # the elemental list using the original position so Literals don't
        # shift the indexing. When `active_vars` is enabled, we additionally
        # skip Var positions whose invar isn't active (forward pruning).
        var_positions = [
            (i, invar)
            for i, invar in enumerate(eqn.invars)
            if isinstance(invar, core.Var) and is_active(invar)
        ]

        # Group differentiable positions by invar. When the SAME Var feeds
        # MULTIPLE input positions of one eqn (e.g. ``x * x``, ``x + x``), its
        # edge to the outvar is the SUM of the per-position elementals — writing
        # them one-per-position would overwrite ``graph[invar][outvar]`` (keeping
        # only the last) and silently HALVE the gradient (``d(x*x)/dx`` = 2x, not
        # x). For the common distinct-operand case each invar has one position,
        # so the sum is a no-op. Insertion order is preserved for determinism.
        pos_by_invar = defaultdict(list)
        for pos, invar in var_positions:
            pos_by_invar[invar].append(pos)

        def _sum_elementals(elemental_list, positions):
            """Sum the elementals at ``positions`` (skipping out-of-range / None);
            returns the combined SparseTensor or None if every position is empty.

            Distinct per-position elementals are summed (``x*x`` -> ``arg1 + arg0``
            = 2x). A rule that ALREADY combines its repeated-operand contributions
            and returns the SAME object at several positions (e.g. concatenate's
            same-primal slot grouping) is counted ONCE — identity-dedup avoids
            double counting and never ``+``-s the deferred transform-only tensors
            those rules emit (which don't support add)."""
            acc = None
            seen = []
            for pos in positions:
                if pos >= len(elemental_list):
                    continue
                e = elemental_list[pos]
                if e is None or any(e is s for s in seen):
                    continue
                seen.append(e)
                acc = e if acc is None else (acc + e)
            return acc

        # If none of the eqn's invars are active, this eqn produces no edges
        # and its outvars stay non-active. Skip the elemental computation
        # entirely — but still bind the primitive so `env` carries the primal
        # for downstream use (output value selection, vo_vertices accounting).
        if active_vars is not None and not var_positions:
            primal_outvals = _eval_primal(eqn, invals_snapshot)
            if eqn.primitive.multiple_results:
                safe_map(write, eqn.outvars, primal_outvals)
            else:
                safe_map(write, eqn.outvars, [primal_outvals])
            continue

        # Stop-gradient and zero-gradient pass-through primitives produce no Jacobian
        # elementals ([]). They carry no differentiable signal forward and must not
        # insert LazyEdges into graph or transpose_graph (ticket dsnn-3qm.74).
        if eqn.primitive in (lax.stop_gradient_p, lax.iota_p, lax.device_put_p):
            primal_outvals = _eval_primal(eqn, invals_snapshot)
            if eqn.primitive.multiple_results:
                safe_map(write, eqn.outvars, primal_outvals)
            else:
                safe_map(write, eqn.outvars, [primal_outvals])
            continue

        # Activate downstream: any outvar of this eqn becomes active because it
        # carries differentiable signal forward.
        if active_vars is not None:
            for ov in eqn.outvars:
                if isinstance(ov, core.Var):
                    active_vars.add(ov)

        if eqn.primitive in multi_output_elemental_only_rules:
            # Multi-output path: primitive produces multiple output variables.
            # The rule returns elementals[outvar_idx][invar_idx].
            primal_outvals = _eval_primal(eqn, invals_snapshot)
            safe_map(write, eqn.outvars, primal_outvals)

            fn = multi_output_elemental_only_rules[eqn.primitive]
            elementals_per_output = fn(primal_outvals, invals_snapshot, **eqn.params)

            for outvar, elementals_for_outvar in zip(
                eqn.outvars, elementals_per_output
            ):
                for invar, positions in pos_by_invar.items():
                    elemental = _sum_elementals(elementals_for_outvar, positions)
                    if elemental is None:
                        continue  # no dependency: zero Jacobian, omit edge
                    _assert_sparse_tensor_consistency(elemental)
                    graph[invar][outvar] = elemental
                    transpose_graph[outvar][invar] = elemental

        elif eqn.primitive in elemental_only_rules:
            # Deferred dispatch path: bind primal eagerly, defer all elemental
            # JAX ops to lazy thunks that fire only when the edge is consumed.
            outvar = eqn.outvars[0]
            primal_outvals = _eval_primal(eqn, invals_snapshot)
            if eqn.primitive.multiple_results:
                safe_map(write, eqn.outvars, primal_outvals)
            else:
                safe_map(write, eqn.outvars, [primal_outvals])

            elemental_only_fn = elemental_only_rules[eqn.primitive]
            _elemental_cache = []

            def _get_elementals(
                _fn=elemental_only_fn,
                _pout=primal_outvals,
                _snap=invals_snapshot,
                _params=eqn.params,
                _cache=_elemental_cache,
            ):
                if not _cache:
                    _cache.append(_fn(_pout, _snap, **_params))
                return _cache[0]

            for invar, positions in pos_by_invar.items():
                # select_n position 0 ('which') is integer-valued / non-differentiable
                # (NO_EDGE = None). Do not create a LazyEdge for an invar feeding only which.
                if eqn.primitive is lax.select_n_p and all(p == 0 for p in positions):
                    continue

                def _make_thunk(positions=tuple(positions), _get=_get_elementals):
                    def thunk():
                        res = _get()
                        # SUM over all positions this invar feeds (x*x etc.).
                        # Returns None when no elemental exists for any of them
                        # (e.g. stop_gradient, iota, device_put return []).
                        # _eliminate_vertex guards against None values.
                        return _sum_elementals(res, positions)

                    return thunk

                edge = LazyEdge(_make_thunk())
                graph[invar][outvar] = edge
                transpose_graph[outvar][invar] = edge
        else:
            # Fallback path for custom rules not yet split into elemental_only_rules.
            # Call cce once and store elementals directly — no double-dispatch.
            outvar = eqn.outvars[0]
            cce = elemental_rules[eqn.primitive]
            primal_outvals, elemental_outvals = cce(invals_snapshot, **eqn.params)
            if eqn.primitive.multiple_results:
                safe_map(write, eqn.outvars, primal_outvals)
            else:
                safe_map(write, eqn.outvars, [primal_outvals])

            for invar, positions in pos_by_invar.items():
                elemental = _sum_elementals(elemental_outvals, positions)
                if elemental is None:
                    continue
                _assert_sparse_tensor_consistency(elemental)
                graph[invar][outvar] = elemental
                transpose_graph[outvar][invar] = elemental

    return env, graph, transpose_graph, vo_vertices


_PRUNE_CACHE = None


def prune_enabled() -> bool:
    """``GRAPHAX_PRUNE=0`` -> skip the dead-vertex / non-argnum sweep.

    Pruning silently removes vertices from the graph before the policy ever
    sees them: inputs we do not differentiate for, and dead intermediates with
    no input or no output edges (typically ``stop_gradient`` outputs). That is
    the right default for plain AD, but for the RL setting it decides part of
    the problem on the agent's behalf -- those vertices and the paths through
    them are exactly the kind of structure we may want the policy to learn to
    drop, or to approximate rather than drop.

    Disabling it leaves the full graph, including edges the eliminator would
    otherwise have deleted for free. The caller then owns the cost of dealing
    with them. Lazy + cached so it stays a compile-time constant.
    """
    global _PRUNE_CACHE
    if _PRUNE_CACHE is None:
        _PRUNE_CACHE = os.environ.get("GRAPHAX_PRUNE", "1") != "0"
    return _PRUNE_CACHE


def _prune_graph(
    graph: ComputationalGraph,
    transpose_graph: ComputationalGraph,
    jaxpr: core.Jaxpr,
    argnums: Sequence[int],
) -> None:
    """
    Function that prunes a given computational graph based on the argnums we
    give it, i.e. for argnums that we do not differentiate for we can just ignore
    them and all edges solely connected to them. This might incur significant
    savings. It also checks for dead intermediate vertices that have either no
    input or no output edges. These typically arise from a lax.stop_grad operation
    somewhere in the function we want to differentiate. These dead vertices and
    all associated edges are deleted as well.
    """
    argnums_set = set(argnums)
    # Identify non-differentiated input invars for pruning.
    # Only prune invars that are actual user arguments (indexed by argnums),
    # not constvars which are stored separately in jaxpr.constvars.
    pruned_invars = set()
    for i, invar in enumerate(jaxpr.invars):
        if i not in argnums_set:
            pruned_invars.add(invar)

    # Remove pruned inputs from ALL vertices' edge dictionaries in both
    # graph and transpose_graph to maintain the invariant:
    #   graph[u][v] == transpose_graph[v][u]
    # This must be done before vertex elimination, otherwise stale edges
    # pointing to pruned inputs can cause KeyError or incorrect Jacobians.
    for invar in pruned_invars:
        # Remove edges from pruned invar -> other vertices in graph
        graph.pop(invar, None)
        # Remove edges from other vertices -> pruned invar in transpose_graph
        transpose_graph.pop(invar, None)
        # Remove references to the pruned invar from all other vertices' edges.
        # Use .get() to avoid defaultdict auto-creation.
        for outvar in list(transpose_graph.keys()):
            if invar in transpose_graph.get(outvar, {}):
                del transpose_graph[outvar][invar]
        for inother in list(graph.keys()):
            if invar in graph.get(inother, {}):
                del graph[inother][invar]

    # Iteratively remove dead intermediate vertices (no incoming or outgoing edges).
    # Use regular dict .get() to avoid defaultdict auto-creation.
    # Track deleted vertices in a set to avoid re-checking.
    outvars_set = set(jaxpr.outvars)
    deleted = set()
    changed = True
    while changed:
        changed = False
        to_delete = []
        for eqn in jaxpr.eqns:
            for ov in eqn.outvars:
                if (
                    isinstance(ov, core.Var)
                    and ov not in outvars_set
                    and ov not in deleted
                    and (ov in graph or ov in transpose_graph)
                ):
                    if (
                        len(graph.get(ov, {})) == 0
                        or len(transpose_graph.get(ov, {})) == 0
                    ):
                        to_delete.append(ov)

        if to_delete:
            for ov in to_delete:
                deleted.add(ov)
                # Remove edges pointing to ov from graph
                for in_edge in list(transpose_graph.get(ov, {}).keys()):
                    graph[in_edge].pop(ov, None)
                # Remove edges from ov in transpose_graph
                for out_edge in list(graph.get(ov, {}).keys()):
                    transpose_graph[out_edge].pop(ov, None)
                # Remove the vertex itself
                graph.pop(ov, None)
                transpose_graph.pop(ov, None)
                changed = True


def _to_persistent(graph) -> immutables.Map:
    """Convert a nested defaultdict-style graph to nested immutables.Map.

    Used as a one-shot conversion at the boundary between `_build_graph` (which
    produces a nested defaultdict) and `VertexEliminator` (which caches
    intermediate states using persistent maps for cheap O(log N) snapshots).
    """
    if isinstance(graph, immutables.Map):
        return graph
    return immutables.Map(
        {k: immutables.Map(inner) for k, inner in graph.items()}
    )


class GraphState:
    """A node in the elimination prefix-cache tree.

    Each node stores the (graph, transpose_graph) snapshot reached by applying
    the prefix of the elimination order leading to it, plus the cumulative
    op counts. ``children`` is keyed by ``(vertex, sp_rules)`` so different
    elimination orders sharing a prefix reuse the same nodes.
    """

    __slots__ = (
        "children",
        "graph",
        "transpose_graph",
        "adds",
        "muls",
        "fmas",
        "mem",
        "lock",
    )

    def __init__(
        self,
        graph: immutables.Map,
        transpose_graph: immutables.Map,
        adds: int = 0,
        muls: int = 0,
        fmas: int = 0,
        mem: int = 0,
    ) -> None:
        self.children: Dict[tuple, "GraphState"] = {}
        self.graph = graph
        self.transpose_graph = transpose_graph
        self.adds = adds
        self.muls = muls
        self.fmas = fmas
        self.mem = mem
        self.lock = threading.Lock()


class VertexEliminator:
    """Caches intermediate elimination states keyed by (vertex, transforms).

    When two elimination plans share a prefix, the cached `GraphState` for the
    longest matching prefix is reused — only the suffix is re-executed. This
    is the main reason the graph uses ``immutables.Map``: snapshots cost
    O(log N) instead of O(N) deep copies.

    The cache key includes the per-vertex transforms tuple — different
    transforms for the same vertex produce different sub-trees.
    """

    def __init__(self, initial_graph, initial_transpose_graph) -> None:
        self.root = GraphState(
            _to_persistent(initial_graph), _to_persistent(initial_transpose_graph)
        )

    def eliminate(
        self,
        order: Sequence[int],
        jaxpr: core.Jaxpr,
        transforms: Sequence[
            Tuple[int, Sequence[Union[Diag, Compress, Callable]]]
        ],
        vo_vertices: Set[core.Var],
        count_ops: bool,
        face_transforms: dict = None,
    ):
        """Run elimination, reusing any cached prefix in the GraphState tree.

        ``transforms`` is the new typed-transform API — a sequence of
        ``(vertex, (transform1, transform2, ...))`` pairs where each
        transform is :class:`Diag`, :class:`Compress`, or a callable.
        See :func:`_eliminate_vertex` for the per-vertex dispatch.
        """
        node = self.root
        prefix_length = 0

        # Build a per-vertex transforms dict for fast lookup during the
        # elimination scan. Vertices missing from `transforms` get an
        # empty tuple (no transforms applied). A per-vertex value that is a
        # dict is the face-like per-path spec (kept as-is); a sequence is the
        # legacy per-vertex transform list.
        t_dict: Dict[int, object] = {}
        _t_requests: Dict[int, object] = {}
        for v, ts in (transforms or ()):
            if isinstance(ts, dict):
                # THE PER-VERTEX DICT FORM IS KEYED DIFFERENTLY FROM
                # ``face_transforms``. Its keys are
                # ``(var_vid[in_edge], var_vid[out_edge])`` -- the eqn-position
                # ids built just below, with a graph input at
                # ``-(invar_index + 1)`` -- NOT the stable-var-index pair
                # ``faces_of`` returns. Handing it ``faces_of``'s keys matches
                # nothing. Wrapped so a request that matches nothing raises
                # instead of running exact in silence.
                t_dict[int(v)] = _t_requests[int(v)] = FaceRequest(ts)
            else:
                t_dict[int(v)] = tuple(ts)

        # Var -> integer vertex-id map so per-path dicts keyed by
        # (primal_vertex_id, out_vertex_id) resolve during elimination: a
        # produced var takes its eqn-position id (1-based, matching `order`); a
        # graph input takes a negative id -(invar_index+1).
        # face_transforms must ALSO disable the prefix cache: its dicts are
        # unhashable and invisible to the (vertex, v_transforms) cache key, so
        # a face-transformed call could silently REUSE an exact run's cached
        # elimination (measured: an all-SKIP jacve returned the exact
        # Jacobian bit-for-bit because the whole order was a cache hit).
        _has_perpath = any(
            isinstance(_x, dict) for _x in t_dict.values()
        ) or bool(face_transforms)
        _var_vid: Dict[core.Var, int] = {}
        if _has_perpath:
            for _i, _eqn in enumerate(jaxpr.eqns, start=1):
                for _ov in _eqn.outvars:
                    _var_vid[_ov] = _i
            for _j, _iv in enumerate(jaxpr.invars):
                _var_vid.setdefault(_iv, -(_j + 1))

        # #46 deferred outputs: the flag changes what a cached GraphState
        # CONTAINS (DeferredOutputProduct edges vs materialized products), and
        # it is read per call so alphagrad can scope it to one trace. Fold it
        # into the prefix-cache key: flag-off runs keep the historic 2-tuple
        # key (byte-identical behaviour and warm-cache reuse), flag-on runs
        # key a DISJOINT subtree -- neither state can leak into the other.
        # Without this, a flag-off prefix was silently reused by a flag-on
        # call (deferral never fired) and a flag-on prefix would hand
        # DeferredOutputProduct edges to a flag-off caller.
        _f46_on = _factored_outputs_enabled()

        # When counting, never reuse the cached prefix: a node's stored counts
        # are only real if the run that created it had count_ops=True. A prior
        # count_ops=False run caches zeros (GraphState defaults), so reusing the
        # prefix here would report muls/adds=0 and an empty per-step breakdown.
        # Re-run the full order so the counts are honest (count_ops is an
        # analysis path, not the hot path); the graph result is identical.
        # The STORED-BYTE tally must see every vertex of the order. A
        # replayed prefix skips `_eliminate_vertex` entirely, so an
        # exact run (no face transforms => cache eligible) would tally
        # only its suffix while the approximated run (perpath =>
        # ineligible) tallied all of it, and the ratio between them
        # would be an artifact of the cache. Armed runs re-walk the
        # order; the graph result is identical, exactly as it is for
        # `count_ops` on the line above.
        if (ENABLE_CACHE and not count_ops and not _has_perpath
                and not store_accounting_armed()):
            for vertex in order:
                v_transforms = t_dict.get(vertex, ())
                key = ((vertex, v_transforms) if not _f46_on
                       else (vertex, v_transforms, "#46-factored"))
                with node.lock:
                    if key in node.children:
                        node = node.children[key]
                        prefix_length += 1
                    else:
                        break

        adds = node.adds
        muls = node.muls
        fmas = node.fmas
        mem = node.mem
        counts: list = []
        m_graph = node.graph.mutate()
        m_transpose_graph = node.transpose_graph.mutate()
        if _STORE_ACCT["armed"]:
            _STORE_ACCT["walks"] += 1

        for vertex in order[prefix_length:]:
            v_transforms = t_dict.get(vertex, ())
            # PER-FACE slots. ``face_transforms`` is {vertex: {face_key:
            # (lhs, rhs, res)}}; _eliminate_vertex already supports the inner
            # dict, it was simply never reachable from jacve. lhs/rhs land on
            # pre_val/post_val BEFORE the contraction, so this is what lets a
            # policy approximate pre1 differently from pre2 -- something the
            # per-vertex ``transforms`` list structurally cannot express.
            _v_faces = (face_transforms or {}).get(vertex)
            _adds, _muls, _fmas, _mem = _eliminate_vertex(
                vertex,
                jaxpr,
                m_graph,
                m_transpose_graph,
                vo_vertices,
                count_ops=count_ops,
                transforms=v_transforms,
                face_transforms=_v_faces,
                var_vid=_var_vid,
            )
            adds += _adds
            muls += _muls
            fmas += _fmas
            mem += _mem
            if count_ops:
                counts.append((adds, muls, fmas, mem))

            # Per-path transform dicts are unhashable and path-specific, so they
            # are never memoized in the prefix tree (like the count path).
            if ENABLE_CACHE and not _has_perpath:
                key = ((vertex, v_transforms) if not _f46_on
                       else (vertex, v_transforms, "#46-factored"))
                with node.lock:
                    if key not in node.children:
                        cur_graph = m_graph.finish()
                        cur_transpose_graph = m_transpose_graph.finish()
                        m_graph = cur_graph.mutate()
                        m_transpose_graph = cur_transpose_graph.mutate()
                        node.children[key] = GraphState(
                            cur_graph,
                            cur_transpose_graph,
                            adds,
                            muls,
                            fmas,
                            mem,
                        )
                    node = node.children[key]

        graph = m_graph.finish()
        transpose_graph = m_transpose_graph.finish()
        report_unapplied_face_transforms(
            _t_requests, site="transforms (the per-vertex dict form)")
        return graph, transpose_graph, adds, muls, fmas, mem, counts


@pytree_hash_cache()
def _get_eliminator(
    jaxpr: core.Jaxpr,
    args: tuple,
    consts: tuple,
    argnums: tuple,
) -> VertexEliminator:
    """Cached factory: one VertexEliminator per (jaxpr, args, consts, argnums).

    Builds the graph with ``argnums`` enabled so dead branches reachable only
    through non-differentiable inputs are pruned during construction. We still
    run ``_prune_graph`` afterward for the dead-intermediate-vertex sweep
    (e.g. stop_gradient outputs) -- unless ``GRAPHAX_PRUNE=0``, which leaves
    the full graph so a policy can decide for itself what to drop or
    approximate (see :func:`prune_enabled`).
    """
    _, graph, transpose_graph, _ = _build_graph(jaxpr, args, consts, argnums)
    if prune_enabled():
        _prune_graph(graph, transpose_graph, jaxpr, argnums)
    return VertexEliminator(graph, transpose_graph)


def vertex_elimination_jaxpr(
    jaxpr: core.Jaxpr,
    order: Union[Sequence[int], str],
    consts: Sequence[core.Literal],
    *args,
    has_aux: bool = False,
    argnums: Sequence[int] = (0,),
    count_ops: bool = False,
    sparse_representation: bool = False,
    dense_edges: bool = False,
    dense_max_bytes: int = None,
    fresh_eliminator: bool = False,
    transforms: Sequence[
        Tuple[
            int,
            Sequence[Union[Diag, Compress, Callable[["SparseTensor"], "SparseTensor"]]],
        ]
    ] = None,
    face_transforms: dict = None,
) -> Sequence[Sequence[jnp.ndarray]]:
    """
    Function that generates a new vertex elimination jaxpression based on the
    vertex elimination jaxpression `jaxpr` found by JAX through tracing the
    function `fun` we intend to differentiate. The function operates in three
    stages:\n
    1.) It creates a computational graph representation amenable to the vertex
    elimination rule. This is mainly facilitated through `_build_graph`.\n
    2.) It applies the vertex elimination rule to every vertex following the
    given `order` using `_eliminate_vertex`.\n
    3.) It performs post processing. This includes the application of several
    Jacobian transformation, densifying sparse tensors and reordering output
    values.

    Args:
        jaxpr (core.Jaxpr): The jaxpr we want to differentiate.
        order (Union[Sequence[int], str]): Vertex elimination order. Either pass
                                        the desired order directly or specify a
                                        string. Allows options are "forward",
                                        "fwd", "reverse" and "rev".
        consts (Sequence[core.Literal]): The constant arguments of the function.
        *args (Any): The input arguments of the function as a flattened PyTree.
        argnums (Sequence[int], optional): Argument numbers to differentiate
                                            with respect to. Defaults to (0,).
        has_aux (bool): _description_
        count_ops (bool, optional): Track adds/muls/fmas/peak-mem during the
                                    elimination. When True, return ``(out, aux)``
                                    where ``aux`` is a dict of cumulative
                                    counts and a per-step breakdown.
                                    Defaults to False.
        sparse_representation (bool, optional): Return the Jacobian in a sparse
                                            representation. Defaults to `False`.

    Returns:
        Sequence[Sequence[jnp.ndarray]]: The Jacobian of the function `fun`.
                                        The output is a list of lists which
                                        corresponds to a flattened PyTree of the
                                        actual input parameters and will be
                                        reassambled into the correct PyTree
                                        by `jacve`.
    """

    # THE DENSE-CONTRACTION MODE (ticket dsnn-3qm.69). A whole separate engine
    # in ``graphax.dense_edges``: every edge a plain array, every contraction a
    # ``jnp.tensordot``. It is the VALUE ORACLE for an approximated plan, and an
    # oracle that shared the contraction code with the engine would prove only
    # the output packing (finding 61 verdict 4) -- so this is an early return,
    # not a flag threaded through ``_eliminate_vertex``. Nothing below runs, and
    # no line below changed, so ``dense_edges=False`` is bit-identical by
    # construction.
    # THE NESTING, CHECKED BEFORE ANYTHING RUNS. A flat {face_key: slots}
    # dict matches no vertex and used to evaporate in silence (job 65975).
    face_transforms = check_face_transforms(
        face_transforms, site="vertex_elimination_jaxpr")

    if dense_edges:
        if sparse_representation:
            raise ValueError(
                "dense_edges=True with sparse_representation=True: the dense "
                "mode has no SparseTensor to return, every edge is a plain "
                "array. Drop sparse_representation.")
        from .dense_edges import dense_vertex_elimination

        _dense_out = dense_vertex_elimination(
            jaxpr,
            order,
            consts,
            *args,
            has_aux=has_aux,
            argnums=argnums,
            count_ops=count_ops,
            transforms=transforms,
            face_transforms=face_transforms,
            max_bytes=dense_max_bytes,
        )
        report_unapplied_face_transforms(
            face_transforms, site="vertex_elimination_jaxpr(dense_edges=True)")
        return _dense_out

    jaxpr_invars = [invar for i, invar in enumerate(jaxpr.invars) if i in argnums]
    env, _, _, vo_vertices = _build_graph(jaxpr, args, consts)

    # A freshly INLINED jaxpr (from jit/pjit/custom_jvp inlining) is a brand-new
    # object whose ``__hash__`` is identity-based; the eliminator cache would key
    # it by object id, and GC id-reuse then yields stale, wrong-valued hits
    # (nondeterministically). Inlined jaxprs are rebuilt every call anyway, so the
    # cache offers them nothing — bypass it (fresh eliminator). Non-inlined jaxprs
    # use the cache as before.
    if fresh_eliminator:
        eliminator = _get_eliminator.__wrapped__(jaxpr, args, consts, tuple(argnums))
    else:
        eliminator = _get_eliminator(jaxpr, args, consts, tuple(argnums))
    order = _checkify_order(order, jaxpr, vo_vertices)
    graph, _, adds, muls, fmas, mem, counts = eliminator.eliminate(
        order, jaxpr, transforms, vo_vertices, count_ops,
        face_transforms=face_transforms,
    )
    # EVERY REQUEST IS ACCOUNTED FOR. A vertex whose faces were never looked
    # up asked for an approximation that did not happen, and that must not
    # pass for a clean run.
    report_unapplied_face_transforms(
        face_transforms, site="vertex_elimination_jaxpr")

    # Offloading all remaining Jacobian transforms to the output variables
    # before densification! Mutate via a single .mutate() proxy on the outer
    # immutables.Map so we don't pay the rebuild cost per (invar, outvar).
    m_graph = graph.mutate()
    for invar in jaxpr_invars:
        invar_inner = m_graph.get(invar)
        if invar_inner is None:
            continue
        m_inner = invar_inner.mutate()
        updated = False
        for outvar in jaxpr.outvars:
            edge = m_inner.get(outvar)
            if edge is None:
                continue
            tensor = _force(edge)
            if tensor is None:
                continue  # null edge (e.g. stop_gradient); treat as zero
            if getattr(tensor, "_is_deferred_output", False):
                continue  # deferred factor pair: no queued transforms by construction
            tensor = _drain_transforms(tensor.copy(), post_first=False)
            m_inner[outvar] = tensor
            updated = True
        if updated:
            m_graph[invar] = m_inner.finish()
    graph = m_graph.finish()

    # Collect outputs
    if sparse_representation:
        # THE OUTPUT-LAYOUT CONTRACT (ticket dsnn-3qm.62): every returned
        # SparseTensor is stored in PARAMETER LAYOUT (its val axes in the
        # order of its dims, axis == position once dense). The tiled engine
        # leaves a 2-D weight gradient transposed in storage while the
        # planner does not; the Index tuple is pytree aux data, so two
        # gradients that differ only in axis assignment have different
        # pytree structure and a consumer cannot tree_map them (finding 60).
        # One transpose at the boundary, or nothing when the layout already
        # holds; asserted, never silently skipped.
        from .sparse.ops.output_layout import (
            canonical_output_layout, require_parameter_layout)
        from .sparse.tensor import SparseTensor

        jac_vals = []
        for outvar in jaxpr.outvars:
            for invar in jaxpr_invars:
                inner = graph.get(invar)
                edge = inner.get(outvar) if inner is not None else None
                tensor = _force(edge) if edge is not None else None
                if isinstance(tensor, SparseTensor):
                    tensor = canonical_output_layout(tensor)
                    require_parameter_layout(
                        tensor, f"the gradient d{outvar}/d{invar}")
                jac_vals.append(tensor)
    else:
        jac_vals = []
        for outvar in jaxpr.outvars:
            for invar in jaxpr_invars:
                inner = graph.get(invar)
                edge = inner.get(outvar) if inner is not None else None
                tensor = _force(edge) if edge is not None else None
                jac_vals.append(
                    tensor.dense() if tensor is not None else zeros_like(outvar, invar)
                )

    # Restructure Jacobians for more complicated pytrees
    n = len(jaxpr_invars)
    if n > 1:
        ratio = len(jac_vals) // n
        jac_vals = [tuple(jac_vals[i * n : i * n + n]) for i in range(0, ratio)]

    if has_aux:
        out = ([env[var] for var in jaxpr.outvars], jac_vals)
    else:
        out = jac_vals

    if count_ops:
        aux = {
            "adds": adds,
            "muls": muls,
            "fmas": fmas,
            "mem": mem,
            "order_counts": [(int(o), c) for o, c in zip(order, counts)],
        }
        return out, aux

    return out


# ---------------------------------------------------------------------------
# extract_jaxpr — JIT-trace the entire vertex elimination process and wrap the
# resulting jaxpr as a VEJaxpr for downstream consumers (e.g. alphagrad). The
# topology cache memoizes by (jaxpr, argnums, order, sparse_representation) so
# repeated calls with the same elimination plan are O(1) after the first build.
# ---------------------------------------------------------------------------

_topology_cache: dict = {}
_topology_lock = threading.Lock()
_topology_pending: dict = {}


def extract_jaxpr(
    jaxpr: core.Jaxpr,
    argnums: Sequence[int],
    order: Sequence[int],
    sparse_representation: bool,
    args: Sequence,
    consts: Sequence,
    transforms: Sequence[
        Tuple[
            int,
            Sequence[Union[Diag, Compress, Callable[["SparseTensor"], "SparseTensor"]]],
        ]
    ] = None,
) -> VEJaxpr:
    """Build a `VEJaxpr` capturing the full vertex-elimination computation.

    The returned `VEJaxpr` is the closed jaxpr of `vertex_elimination_jaxpr`
    applied with the given `order` and per-vertex `transforms`. Cached by
    ``(jaxpr, argnums, order, transforms, sparse_representation)`` so
    subsequent calls with the same plan return the cached object without
    re-tracing.

    ``transforms`` is the typed-transform API: a sequence of
    ``(vertex, (transform1, transform2, ...))`` pairs where each transform
    is a :class:`Diag`, a :class:`Compress`, or a callable
    ``(SparseTensor) -> SparseTensor``. See :func:`_eliminate_vertex` for
    the per-vertex dispatch.
    """
    if isinstance(order, str):
        env, graph, transpose_graph, vo_vertices = _build_graph(jaxpr, args, consts)
        _order = tuple(_checkify_order(order, jaxpr, vo_vertices))
    elif hasattr(order, "tolist"):
        _order = tuple(map(int, order.tolist()))
    else:
        _order = tuple(map(int, order))

    # Cache-key normalisation: each transform is either a frozen dataclass
    # (Diag / Compress — hashable by value) or a plain callable (hashable
    # by identity). Outer structure is a tuple of (vertex, tuple-of-transforms).
    _transforms = (
        tuple(
            (int(v), tuple(ts))
            for v, ts in transforms
        )
        if transforms is not None
        else ()
    )

    cache_key = (jaxpr, tuple(argnums), _order, _transforms, sparse_representation)
    # #46 deferred outputs: the traced VEJaxpr BAKES IN the flag state (its
    # outputs are factor leaves vs the materialized product), and the flag is
    # toggled per trace by alphagrad. Flag-on topologies key a disjoint entry;
    # flag-off keys keep their historic shape.
    if _factored_outputs_enabled():
        cache_key = cache_key + ("#46-factored-topo",)

    # A per-path transforms entry is a dict of opaque per-path hook callables;
    # ``tuple(ts)`` above captures only its (primal_id, out_id) KEYS, never the
    # hooks, so two different hook sets share a cache_key. Exclude per-path runs
    # from the topology cache entirely (same rule the eliminate-tree cache uses),
    # so a cached topology is never served for a different set of hooks.
    _perpath = any(isinstance(ts, dict) for _, ts in (transforms or ()))
    _use_cache = ENABLE_CACHE and not _perpath

    must_compute = False
    event = None
    with _topology_lock:
        if _use_cache and cache_key in _topology_cache:
            return _topology_cache[cache_key]

        if _use_cache and cache_key in _topology_pending:
            event = _topology_pending[cache_key]
        else:
            event = threading.Event()
            if _use_cache:
                _topology_pending[cache_key] = event
            must_compute = True

    if not must_compute:
        event.wait()
        with _topology_lock:
            return _topology_cache[cache_key]

    try:

        def eval_graph(*tracer_args):
            tracer_map = {num: tracer for num, tracer in zip(argnums, tracer_args)}
            full_args = [tracer_map.get(i, args[i]) for i in range(len(args))]

            res = vertex_elimination_jaxpr(
                jaxpr,
                _order,
                consts,
                *full_args,
                argnums=argnums,
                sparse_representation=sparse_representation,
                transforms=_transforms,
            )
            # vertex_elimination_jaxpr returns just jac_vals when has_aux=False.
            # Flatten so the resulting jaxpr has all jacobians as outputs.
            return tuple(jtu.tree_leaves(res))

        # APPEND-ONLY STATE tokenization (opt-in via GRAPHAX_STATE_TOKENS=1):
        # skip the per-step Jacobian RE-TRACE entirely. The token stream becomes
        # <original-graph tokens> | <elimination-order prefix>, which is a pure
        # prefix-extension step to step (incrementally cacheable) and a lossless
        # encoding of the partial-elimination state (original graph + order are a
        # sufficient statistic). This also avoids the expensive make_jaxpr trace.
        import os as _os
        if _os.environ.get("GRAPHAX_STATE_TOKENS", "0") == "1":
            # Pass the per-vertex micro-actions (DIAG/COMPRESS/QUANT) too, so
            # the state stream losslessly encodes the APPROXIMATED state, not
            # just the exact elimination order. ``_transforms`` is already the
            # normalised ((vertex, (transform_obj, ...)), ...) structure.
            ve_jaxpr = VEJaxpr(jaxpr, elim_order=_order, transforms=_transforms)
            if _use_cache:
                with _topology_lock:
                    _topology_cache[cache_key] = ve_jaxpr
            return ve_jaxpr

        dummy_args = [
            ShapeDtypeStruct(v.aval.shape, v.aval.dtype)
            for i, v in enumerate(jaxpr.invars)
            if i in argnums
        ]

        closed = jax.make_jaxpr(eval_graph)(*dummy_args)
        ve_jaxpr = VEJaxpr(closed.jaxpr)

        if _use_cache:
            with _topology_lock:
                _topology_cache[cache_key] = ve_jaxpr
        return ve_jaxpr
    finally:
        if _use_cache:
            with _topology_lock:
                _topology_pending.pop(cache_key, None)
            event.set()


# ---------------------------------------------------------------------------
# The jit_p elemental rule (the ONLY registration for jit_p; ordinary jits never
# reach it — they are inlined by _inline_call_primitives). It is reached only for
# a jit KEPT as a vertex (``_jit_kept_as_vertex``: a jax.nn activation graphax has
# its own Jacobian for) and dispatches:
#
#   1. name in jit_name_rules + default static args -> graphax's own clean
#      Jacobian (jit_named_elemental_only), exact analytic subgradient at kinks;
#   2. otherwise (non-default alpha / negative_slope) -> fall back to the
#      macro-vertex: recursively differentiate params["jaxpr"] with
#      vertex_elimination_jaxpr. Its order is configurable via
#      set_jit_fallback_order() (default "reverse").
#
# Registered here (not in a primitives/pjit.py) to avoid a circular import:
# core.py needs vertex_elimination_jaxpr, which lives here.
# ---------------------------------------------------------------------------


def _make_jit_elemental_rule(order):
    def jit_elemental_rule(primal_outs, primals, **params):
        # 1. graphax's own named-activation Jacobian (clean kink subgradient).
        #    Raises on non-default static args -> fall through to the body diff.
        if params.get("name") in jit_name_rules:
            try:
                return jit_named_elemental_only(primal_outs, primals, **params)
            except NotImplementedError:
                pass

        # 2. Macro-vertex fallback: Jacobian of the inner jaxpr by recursion.
        inner_closed = params["jaxpr"]
        inner_jaxpr = inner_closed.jaxpr
        consts = inner_closed.literals
        n = len(primals)
        argnums = tuple(range(n))

        jac_vals = vertex_elimination_jaxpr(
            inner_jaxpr,
            order,
            consts,
            *primals,
            argnums=argnums,
            sparse_representation=True,
            fresh_eliminator=True,  # sub-jaxpr: bypass the id-keyed eliminator cache
        )

        # vertex_elimination_jaxpr output layout (sparse_representation=True):
        #   n=1, M outputs -> [J(out0,in0), J(out1,in0), ..., J(outM,in0)]
        #   n>1, M outputs -> [(J(out0,in0),...,J(out0,inN)), ..., (J(outM,in0),...)]
        #
        # We must return result[outvar_idx][invar_idx] for multi_output_elemental_only_rules.
        if n == 1:
            # Each entry is a single SparseTensor for one output; wrap in a list.
            return [[jac] for jac in jac_vals]
        else:
            # Each entry is a tuple of N SparseTensors (one per input) for one output.
            return [list(jac_tuple) for jac_tuple in jac_vals]

    return jit_elemental_rule


def set_jit_fallback_order(order: str = "reverse") -> None:
    """Set the vertex elimination order for the jit_p macro-vertex fallback.

    Ordinary jits are inlined and named jax.nn activations are dispatched by
    name, so this order only affects the rare FALLBACK path: a named activation
    called with NON-default static args (where graphax's name-keyed Jacobian
    doesn't apply), whose body is then differentiated by recursion.

    Args:
        order: Any elimination order accepted by jacve — ``"forward"``, ``"fwd"``,
               ``"reverse"``, ``"rev"``, or an explicit integer sequence.
               Defaults to ``"reverse"``.
    """
    elemental_only_rules.pop(jit_p, None)  # remove any prior single-output registration
    multi_output_elemental_only_rules[jit_p] = _make_jit_elemental_rule(order)


# Back-compat alias: this used to be the only public name, but the order governs
# the jit_p macro-vertex FALLBACK (an inlined / name-dispatched jit never reaches
# it) and "pjit" is legacy JAX terminology, so ``set_jit_fallback_order`` is now
# the preferred name. Both refer to the same function.
set_pjit_elimination_order = set_jit_fallback_order


# Register with the default order at import time.
set_jit_fallback_order()


# ---------------------------------------------------------------------------
# cond_p / switch elemental rule
#
# lax.cond / lax.switch select one of `branches` by an integer index. That index
# is concrete during graphax's forward pass, so we differentiate the TAKEN branch
# (recursively, like the jit macro-vertex) — matching jax, which also only
# differentiates the taken branch. The integer index carries no gradient.
# (The old auto.py stub returned `[]`, silently zeroing the gradient of ALL
# control flow.) Registered here because it needs vertex_elimination_jaxpr.
# ---------------------------------------------------------------------------
def cond_elemental_rule(primal_outs, primals, **params):
    branches = params["branches"]
    index = max(0, min(int(primals[0]), len(branches) - 1))
    branch = branches[index]
    operands = primals[1:]
    n = len(operands)

    jac_vals = vertex_elimination_jaxpr(
        branch.jaxpr,
        "reverse",
        branch.consts,
        *operands,
        argnums=tuple(range(n)),
        sparse_representation=True,
        fresh_eliminator=True,  # sub-jaxpr: bypass the id-keyed eliminator cache
    )

    # jac_vals[outvar_idx] is a single SparseTensor (n==1) or a per-operand tuple
    # (n>1). eqn invars are (index, *operands), so prepend None for the
    # non-differentiable index, then one entry per operand.
    if n == 1:
        return [[None, jac] for jac in jac_vals]
    return [[None, *jac_tuple] for jac_tuple in jac_vals]


multi_output_elemental_only_rules[cond_p] = cond_elemental_rule
