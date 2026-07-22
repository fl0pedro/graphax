import functools
import os
import threading
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict, Sequence, Set, Tuple, Union, cast

import immutables
import jax
import numpy as np
import jax._src.core as core
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
    _compressed_dims, _materialize_for_op, _is_approx,
    _squeeze_unreferenced_val_axes,
)
from .sparse.tensor import _assert_sparse_tensor_consistency
from .sparse.micro_actions import (
    Compress, Diag, Quant, apply_compress, apply_diag, apply_quant,
)
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
    return hash(leaf)


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
            key = hash((hash(treedef), leaf_hashes))

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
    transforms: Sequence[
        Tuple[
            int,
            Sequence[Union[Diag, Compress, Callable[["SparseTensor"], "SparseTensor"]]],
        ]
    ] = None,
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

    Returns:
        Callable: The function that returns the Jacobian of `fun`.
    """

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
            fresh_eliminator=was_inlined,
            transforms=transforms,
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
    elif len(_pre_transforms) > 0 and pre_val.val is not None:
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
    ``_materialize_compressed`` / ``apply_*`` may likewise short-circuit — so a
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


def _record_micro(_t, before, after, vertex, slot, in_edge, out_edge,
                  start, _face_sink, _xlog):
    """Record ONE dispatched micro-action, truthfully.

    Writes to the always-on :class:`TransformLog` (installed by
    ``IncrementalJaxpr`` for every elimination, so the record exists even with
    ``track_faces=False``) with an explicit ``applied`` flag, and — only when the
    action really changed the tensor — to the opt-in :class:`FaceSink`'s
    ``approx`` block list, so the tokenizer never renders an empty block for a
    no-op. Both sinks are optional; with neither installed this is a no-op.
    """
    if _face_sink is None and _xlog is None:
        return
    applied = _micro_applied(before, after)
    end = _eqn_count(_face_sink, _xlog)
    _atype, _params = _approx_meta(_t)
    if _face_sink is not None and applied:
        _face_sink.approx(_atype, _params, start, end)
    if _xlog is not None:
        _xlog.record("transform", vertex, slot, _atype, _params,
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


def _unpack_face_slots(slots, vertex):
    """Validate one ``face_transforms`` entry -> ``(lhs, rhs, res)``.

    A malformed entry is a structural programming error, so it raises
    ``TypeError`` — which the per-face dispatch deliberately does NOT catch.
    """
    try:
        lhs, rhs, res = slots
    except (TypeError, ValueError):
        raise TypeError(
            f"face_transforms entry at vertex {vertex} must be a 3-tuple "
            f"(lhs, rhs, res); got {slots!r}."
        ) from None
    return lhs, rhs, res


def _apply_face_transform(val, _t, slot, vertex, _face_sink, in_edge=None,
                          out_edge=None, _xlog=None):
    """Apply ONE per-face slot transform to ONE Jacobian operand.

    Mirrors the per-vertex ``transforms`` dispatch in :func:`_eliminate_vertex`
    exactly: ``Diag`` / ``Compress`` / ``Quant`` go through :func:`_apply_micro`
    (and are recorded by :func:`_record_micro` — on the currently open face as a
    labelled ``approx`` block when a face sink is installed, and ALWAYS on the
    :class:`TransformLog` when one is), any other callable is handed the tensor
    directly, and anything else raises ``TypeError``.

    A ``ValueError`` is the documented best-effort miss — the transform does not
    fit THIS operand's geometry — so the transform is skipped, the operand is
    returned unchanged, and NOTHING is recorded, which keeps the record
    truthful. ``TypeError`` is deliberately NOT caught (see the per-vertex loop).
    """
    if _t is None:
        return val
    try:
        if isinstance(_t, (Diag, Compress, Quant)):
            _as = _eqn_count(_face_sink, _xlog)
            out = _apply_micro(val, _t)
            _record_micro(_t, val, out, vertex, slot, in_edge, out_edge,
                          _as, _face_sink, _xlog)
        elif callable(_t):
            out = _t(val)
        else:
            raise TypeError(
                f"Unknown per-face transform of type {type(_t).__name__} in "
                f"slot {slot!r} at vertex {vertex}; expected None, Diag, "
                "Compress, Quant, or a callable (SparseTensor) -> SparseTensor."
            )
    except ValueError:
        return val
    _assert_sparse_tensor_consistency(out)
    return out


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
          (e.g. a ``stop_gradient`` blocked path). An edge that is already
          concrete (or an already-evaluated ``LazyEdge``) is filtered out here
          too, but an *unevaluated* ``LazyEdge`` is NOT forced — forcing emits
          jax equations into whatever trace happens to be current, which would
          corrupt the append-only jaxpr. Such faces are therefore listed
          optimistically; if the elimination later skips one, its key simply
          never matches and the transform is a no-op. The returned list is thus
          a superset of the visited faces, never a subset.
        * A multi-output vertex contributes the faces of every one of its
          output variables; the central variable is not part of the key (the
          mapping is per-vertex-elimination), so in the rare case where two
          output variables share the same ``(in_edge, out_edge)`` pair their key
          collides and one entry configures both faces.
    """
    eqn = jaxpr.eqns[int(vertex) - 1]
    vidx = _vidx_for(jaxpr)

    def _ordered(keys):
        return sorted(keys, key=lambda v: vidx.get(v, 1 << 30))

    keys = []
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
                keys.append((vidx.get(in_edge), vidx.get(out_edge)))
    return keys


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
        _is_approx_cfg = _is_approx_cfg or any(
            isinstance(_t, (Diag, Compress))
            for _slots in face_transforms.values()
            for _t in _slots
        )

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
                if face_transforms is not None:
                    post_val = _post_raw
                    _slots = face_transforms.get(
                        (_vidx.get(in_edge), _vidx.get(out_edge))
                    )
                    if _slots is not None:
                        _lhs_t, _rhs_t, _face_res_t = _unpack_face_slots(
                            _slots, vertex)
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
                if _perpath and _h_pre is not None:
                    _pre_val = _h_pre(_pre_val)
                if _perpath and _h_post is not None:
                    _post_val = _h_post(_post_val)

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
                if _perpath or _is_approx_cfg:
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
                if _need_contract:
                    # A scalar × scalar contraction is an elementwise multiply:
                    # ``sparse_matmul`` rejects 0-rank operands, so it must NEVER
                    # be routed through matmul — on either the count or non-count
                    # path (the count path used to crash here).
                    if _is_scalar_st(_post_val) and _is_scalar_st(_pre_val):
                        edge_outval = _post_val * _pre_val
                        if count_ops:
                            muls += 1
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

                elif (_pre_val.val is not None if (_perpath or _is_approx_cfg)
                      else pre_val.val is not None):
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

                # A misaligned-contract matmul can emit a compressed output
                # (BandedIndex / SetIndex). The consistency check and the
                # Diag / Compress micro-actions below consume only plain
                # {Dense, Diagonal} dims, so densify the compressed pair to
                # its compact equivalent here (keeps the M× meta-block-diagonal
                # form where the structure reduces to a diagonal).
                if _compressed_dims(edge_outval):
                    edge_outval = _materialize_for_op(edge_outval)

                # Per-path contraction-RESULT hook (``new``). The face engine
                # applies its ``new`` to the product; here we apply it to the
                # freshly-contracted edge before the join. A join ``lhs`` hook
                # (below) then composes on top for the merge case.
                if _perpath and _h_new is not None:
                    edge_outval = _h_new(edge_outval)
                    _assert_sparse_tensor_consistency(edge_outval)

                _assert_sparse_tensor_consistency(edge_outval)
                # If there is already an edge between the two vertices, add the new
                # edge to the existing one
                if graph.get(in_edge).get(out_edge) is not None:
                    _edge = _force(transpose_graph[out_edge][in_edge])
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

                    # Nominal-shape asserts hold only for EXACT AD (no approx of
                    # any kind): an approximation (per-path OR legacy list) can
                    # leave an edge sparse/permuted, and the sparse ``+`` reconciles
                    # it — there is no normalization to nominal any more.
                    if not _perpath and not _is_approx_cfg:
                        edge_shape = tuple(
                            list(out_edge.aval.shape) + list(in_edge.aval.shape)
                        )
                        assert edge_shape == edge_outval.shape, (
                            f"Computed edge shape {edge_outval.shape} does not match expected shape {edge_shape}!"
                        )
                        assert edge_shape == _edge.shape, (
                            f"Existing edge shape {_edge.shape} does not match expected shape {edge_shape}!"
                        )
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
                        in_edge, out_edge, _xlog)

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

                _set_inner(graph, in_edge, out_edge, edge_outval)
                _set_inner(transpose_graph, out_edge, in_edge, edge_outval)
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


def _set_inner(outer, k1, k2, v):
    """Set ``outer[k1][k2] = v`` for both nested-defaultdict and immutables.Map proxies."""
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
    for eqn in jaxpr.eqns:
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
        for v, ts in (transforms or ()):
            t_dict[int(v)] = ts if isinstance(ts, dict) else tuple(ts)

        # Var -> integer vertex-id map so per-path dicts keyed by
        # (primal_vertex_id, out_vertex_id) resolve during elimination: a
        # produced var takes its eqn-position id (1-based, matching `order`); a
        # graph input takes a negative id -(invar_index+1).
        _has_perpath = any(isinstance(_x, dict) for _x in t_dict.values())
        _var_vid: Dict[core.Var, int] = {}
        if _has_perpath:
            for _i, _eqn in enumerate(jaxpr.eqns, start=1):
                for _ov in _eqn.outvars:
                    _var_vid[_ov] = _i
            for _j, _iv in enumerate(jaxpr.invars):
                _var_vid.setdefault(_iv, -(_j + 1))

        # When counting, never reuse the cached prefix: a node's stored counts
        # are only real if the run that created it had count_ops=True. A prior
        # count_ops=False run caches zeros (GraphState defaults), so reusing the
        # prefix here would report muls/adds=0 and an empty per-step breakdown.
        # Re-run the full order so the counts are honest (count_ops is an
        # analysis path, not the hot path); the graph result is identical.
        if ENABLE_CACHE and not count_ops and not _has_perpath:
            for vertex in order:
                v_transforms = t_dict.get(vertex, ())
                key = (vertex, v_transforms)
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

        for vertex in order[prefix_length:]:
            v_transforms = t_dict.get(vertex, ())
            _adds, _muls, _fmas, _mem = _eliminate_vertex(
                vertex,
                jaxpr,
                m_graph,
                m_transpose_graph,
                vo_vertices,
                count_ops=count_ops,
                transforms=v_transforms,
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
                key = (vertex, v_transforms)
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
    (e.g. stop_gradient outputs).
    """
    _, graph, transpose_graph, _ = _build_graph(jaxpr, args, consts, argnums)
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
    fresh_eliminator: bool = False,
    transforms: Sequence[
        Tuple[
            int,
            Sequence[Union[Diag, Compress, Callable[["SparseTensor"], "SparseTensor"]]],
        ]
    ] = None,
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
    # Flag whether this elimination carries a Diag/Compress approximation. The
    # elemental sparse dispatch is a hard no-op unless this is set, so plain
    # exact AD never routes through it (see dispatch.set_approx_active).
    # vertex_elimination_jaxpr RECURSES (topology build, jit/cond macro-vertices),
    # so SAVE+RESTORE the prior value rather than hard-resetting to False — a
    # nested non-approx elimination must not clear an outer approx elimination's
    # flag mid-flight (that would silently route the outer's remaining approx
    # edges onto the existing path).
    from .sparse.elemental.dispatch import approx_active, set_approx_active
    # Same rule as `_is_approx_cfg`: a transform is an approximation if it is a
    # Diag/Compress instance OR a CALLABLE (the documented
    # "(SparseTensor) -> SparseTensor escape hatch"). The isinstance-only test made a
    # callable set _approx_on=False -> set_approx_active(False) -> try_elemental_matmul's
    # EXACT-AD guard (`if not approx_active(): return None`) short-circuited the ENTIRE
    # elemental dispatch, so contract_implicit was NEVER REACHED (telemetry:
    # matmul_pure_dense_skip=11, DISPATCH_FALLBACK_LOG empty). The implicit (Compress-away)
    # dim was therefore never consumed by the contraction and rode through as a phantom out
    # dim, which _normalize_approx_edge then could not regroup ("per-side logical-extent
    # mismatch — edge (out 10 | primal 8) vs nominal (out () | primal (8,))").
    _approx_on = any(
        isinstance(_t, (Diag, Compress)) or callable(_t)
        for _spec in (transforms or ())
        for _t in (_spec[1] if isinstance(_spec, (tuple, list)) and len(_spec) == 2 else ())
    )
    _prev_approx = approx_active()
    set_approx_active(_approx_on)
    try:
        graph, _, adds, muls, fmas, mem, counts = eliminator.eliminate(
            order, jaxpr, transforms, vo_vertices, count_ops
        )
    finally:
        set_approx_active(_prev_approx)

    # --- AUDIT PROTOTYPE: strict full-order guard -------------------------
    # A numeric order that misses eliminable vertices (e.g. built from the
    # NON-inlined jaxpr's eqn count while jacve inlined a custom_jvp/pjit body)
    # leaves live intermediate vertices in the graph. The dense output path
    # then silently drops every Jacobian path through them (all-zero / partial
    # Jacobian). Fail loudly instead; GRAPHAX_ALLOW_PARTIAL_ORDER=1 restores
    # the silent-partial behaviour for staged/triplet elimination callers.
    if not sparse_representation and os.environ.get(
        "GRAPHAX_ALLOW_PARTIAL_ORDER", "0"
    ) == "0":
        _intermediates = {
            ov
            for eqn in jaxpr.eqns
            for ov in eqn.outvars
            if isinstance(ov, core.Var)
        }
        _live = [
            v
            for v in graph.keys()
            if v in _intermediates and len(graph[v]) > 0
        ]
        if _live:
            raise ValueError(
                "jacve: the elimination order left "
                f"{len(_live)} intermediate vertex/vertices with live edges "
                "un-eliminated — the dense Jacobian would silently drop every "
                "path through them (this typically means the numeric order was "
                "built from the non-inlined jaxpr; jacve inlines jit/pjit/"
                "custom_jvp bodies, which adds equations). Pass a full order "
                "or set GRAPHAX_ALLOW_PARTIAL_ORDER=1 to accept a partial "
                "Jacobian."
            )

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
            tensor = _drain_transforms(tensor.copy(), post_first=False)
            m_inner[outvar] = tensor
            updated = True
        if updated:
            m_graph[invar] = m_inner.finish()
    graph = m_graph.finish()

    # Collect outputs
    if sparse_representation:
        jac_vals = []
        for outvar in jaxpr.outvars:
            for invar in jaxpr_invars:
                inner = graph.get(invar)
                edge = inner.get(outvar) if inner is not None else None
                tensor = _force(edge) if edge is not None else None
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
