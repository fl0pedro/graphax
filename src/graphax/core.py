from __future__ import annotations

import os
import threading
from collections import defaultdict
from dataclasses import replace
from functools import cache, partial, wraps
from math import gcd
from typing import Any, Callable, Sequence, Union, cast

import immutables
import jax
import jax._src.core as core
import jax.numpy as jnp
import jax.tree_util as jtu
from jax._src.core import ShapeDtypeStruct
from jax._src.util import safe_map

from .jaxpr import VEJaxpr
from .primitives import elemental_rules
from .sparse.tensor import (
    DenseDimension,
    SparseDimension,
    SparseTensor,
    _assert_sparse_tensor_consistency,
    _create_optimized,
    apply_dynamic_sparsity,
    elementwise_fmas,
    matmul_fmas,
)
from .sparse.utils import get_largest_tensor, zeros_like

# --- Instrumentation ---
try:
    from alphagrad.utils.profiler import track_activity
except ImportError:
    # Fallback if used outside of alphagrad environment
    from contextlib import contextmanager

    @contextmanager
    def track_activity(label):
        yield


ENABLE_CACHE = os.environ.get("GX_ENABLE_CACHE", "1") != "0"


def tree_allclose(tree1, tree2, equal_nan: bool = False) -> bool:
    allclose = lambda a, b: jnp.allclose(
        a, b, equal_nan=equal_nan, atol=1e-5, rtol=1e-4
    )
    is_equal = jtu.tree_map(allclose, tree1, tree2)
    return jtu.tree_reduce(jnp.logical_and, is_equal)


EliminationOrder = Union[Sequence[int], str]
ComputationalGraph = immutables.Map  # [core.Var, immutables.Map[core.Var, Any]]


def tree_allclose(tree1, tree2, equal_nan: bool = False) -> bool:
    allclose = lambda a, b: jnp.allclose(
        a, b, equal_nan=equal_nan, atol=1e-5, rtol=1e-4
    )
    is_equal = jtu.tree_map(allclose, tree1, tree2)
    return jtu.tree_reduce(jnp.logical_and, is_equal)


EliminationOrder = Union[Sequence[int], str]
ComputationalGraph = immutables.Map  # [core.Var, immutables.Map[core.Var, Any]]

def jacve(  # combine with extract_jaxpr?
    fun: Callable,
    order: EliminationOrder,
    argnums: Sequence[int] = (0,),
    has_aux: bool = False,
    count_ops: bool = False,
    sparse_representation: bool = False,
    sparsity_map: Sequence[tuple[int, tuple[tuple[int, ...], ...]]] = None,
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
        count_ops (bool, optional): Count the number of operations during the
                                    elimination process. Defaults to `False`.
        sparse_representation (bool, optional): Return the Jacobian in a sparse
                                                representation. Defaults to `False`.

    Returns:
        Callable: The function that returns the Jacobian of `fun`.
    """

    @wraps(fun)
    def jacfun(*args: Any, **kwargs: Any) -> Any:
        # TODO Make repackaging work properly with one input value only
        flattened_args, in_tree = jtu.tree_flatten(args)
        closed_jaxpr = jax.make_jaxpr(fun)(*flattened_args, **kwargs)

        outputs = vertex_elimination_jaxpr(
            closed_jaxpr.jaxpr,
            order,
            closed_jaxpr.literals,
            *args,
            argnums=argnums,
            count_ops=count_ops,
            sparse_representation=sparse_representation,
            sparsity_map=sparsity_map,
        )

        aux_data: dict[str, Any] = {}

        if count_ops:
            outputs, aux_data = outputs

        primal_out, jac_out = outputs

        out_tree = jtu.tree_structure(tuple(closed_jaxpr.jaxpr.outvars))
        if len(closed_jaxpr.jaxpr.outvars) == 1 and len(closed_jaxpr.jaxpr.invars) > 1:
            jac_res = jac_out[0]
        else:
            jac_res = jtu.tree_unflatten(out_tree, jac_out)

        primal_res: Any = None
        if has_aux:
            if (
                len(closed_jaxpr.jaxpr.outvars) == 1
                and len(closed_jaxpr.jaxpr.invars) > 1
            ):
                primal_res = primal_out[0]
            else:
                primal_res = jtu.tree_unflatten(out_tree, primal_out)

        if has_aux:
            res: Any = (primal_res, jac_res)
        else:
            res = jac_res

        if count_ops:
            return res, aux_data
        return res

    return jacfun


def unload_post_transforms(post, pre, iota):
    new_post = post.copy()
    for transform in pre.post_transforms:
        new_post = transform.apply_inverse(new_post, iota)
    _assert_sparse_tensor_consistency(new_post)
    return new_post


def unload_pre_transforms(post, pre, iota):
    new_pre = pre.copy()
    for transform in post.pre_transforms:
        new_pre = transform.apply(new_pre, iota)
    _assert_sparse_tensor_consistency(new_pre)
    return new_pre


def prepend_post_transforms(post, out, iota):
    transforms = post.post_transforms + out.post_transforms
    return replace(out, post_transforms=transforms)


def append_pre_transforms(pre, out, iota):
    transforms = pre.pre_transforms + out.pre_transforms
    return replace(out, pre_transforms=transforms)


def _eliminate_vertex(
    vertex: int,
    jaxpr: core.Jaxpr,
    graph: Any,  # MapMutation
    transpose_graph: Any,  # MapMutation
    iota: Any,
    vo_vertices: set[int],
    sp_rules: tuple = (),
) -> int:
    """
    Function that eliminates a vertex from the computational graph.
    everything that has a _val in its name is a `SparseTensor` object
    """

    eqn = jaxpr.eqns[vertex - 1]
    fmas = 0

    out_var = eqn.outvars[0]
    out_edges = graph.get(out_var)
    if out_edges is None:
        return 0

    in_edges_map = transpose_graph.get(out_var)
    if in_edges_map is None:
        # Cleanup if no in-edges but out-edges exist (unlikely in valid graph but be safe)
        if vertex not in vo_vertices:
            graph.pop(out_var, None)
        return 0

    for out_edge, post_val_orig in out_edges.items():
        post_val = post_val_orig.copy()
        for in_edge, pre_val_orig in in_edges_map.items():
            pre_val = pre_val_orig.copy()

            # Handle stuff like reshape, squeeze etc.
            # Apply Jacobian transforms where applicable
            _pre_val = pre_val.copy()
            _post_val = post_val.copy()

            if len(pre_val.post_transforms) > 0 and post_val.val is not None:
                _post_val = unload_post_transforms(post_val, pre_val, iota)

            if len(post_val.pre_transforms) > 0 and pre_val.val is not None:
                _pre_val = unload_pre_transforms(post_val, pre_val, iota)

            # Multiply the two values of the edges if applicable
            if pre_val.val is not None and post_val.val is not None:
                if _post_val.ndim == 0 or _pre_val.ndim == 0:
                    edge_outval = _post_val * _pre_val
                else:
                    edge_outval = _post_val @ _pre_val
                # jax.debug.print("{}\n@\n{}\n=\n{}", _post_val, _pre_val, edge_outval)
                _assert_sparse_tensor_consistency(edge_outval)

                fmas += matmul_fmas(_post_val, _pre_val)
            elif pre_val.val is not None:
                edge_outval = _pre_val
            else:
                edge_outval = _post_val

            if len(post_val.post_transforms) > 0:
                edge_outval = prepend_post_transforms(post_val, edge_outval, iota)

            if len(pre_val.pre_transforms) > 0:
                edge_outval = append_pre_transforms(pre_val, edge_outval, iota)

            # If there is already an edge between the two vertices, add the new
            # edge to the existing one
            existing_inner = graph.get(in_edge)
            existing_edge = existing_inner.get(out_edge) if existing_inner else None

            if existing_edge is not None:
                _edge = existing_edge  # It's już a SparseTensor, no copy needed here for reading

                # Offload the remaining Jacobian transforms to the output tensor
                if len(edge_outval.post_transforms) > 0:
                    for transform in edge_outval.post_transforms:
                        edge_outval = transform.apply(edge_outval, iota)

                if len(edge_outval.pre_transforms) > 0:
                    for transform in edge_outval.pre_transforms[::-1]:
                        edge_outval = transform.apply_inverse(edge_outval, iota)

                # Offload the remain Jacobian transforms to the output tensor
                if len(_edge.post_transforms) > 0:
                    for transform in _edge.post_transforms:
                        _edge = transform.apply(_edge, iota)

                if len(_edge.pre_transforms) > 0:
                    for transform in _edge.pre_transforms[::-1]:
                        _edge = transform.apply_inverse(_edge, iota)

                _assert_sparse_tensor_consistency(edge_outval)
                pre_edge_outval = edge_outval
                edge_outval += _edge
                # jax.debug.print("{}\n+\n{}\n=\n{}", pre_edge_outval, _edge, edge_outval)
                fmas += elementwise_fmas(edge_outval, _edge)

            if sp_rules:
                edge_outval = apply_dynamic_sparsity(edge_outval, sp_rules)
                _assert_sparse_tensor_consistency(edge_outval)

            inner_g = graph.get(in_edge, immutables.Map())

            # Update graph (nested immutables.Map)
            inner_g = graph.get(in_edge, immutables.Map())
            graph[in_edge] = inner_g.update({out_edge: edge_outval})

            inner_tg = transpose_graph.get(out_edge, immutables.Map())
            transpose_graph[out_edge] = inner_tg.update({in_edge: edge_outval})

    # Cleanup of input and output edges
    if vertex not in vo_vertices:
        for in_vertex in in_edges_map.keys():
            inner_g = graph.get(in_vertex)
            if inner_g:
                graph[in_vertex] = inner_g.delete(out_var)

    for out_vertex in out_edges.keys():
        inner_tg = transpose_graph.get(out_vertex)
        if inner_tg:
            transpose_graph[out_vertex] = inner_tg.delete(out_var)

    # Cleanup the eliminated vertex
    graph.pop(out_var, None)
    if vertex not in vo_vertices:
        transpose_graph.pop(out_var, None)

    return fmas


class GraphState:
    __slots__ = ["children", "graph", "transpose_graph", "fmas", "lock"]

    def __init__(
        self,
        graph: ComputationalGraph,
        transpose_graph: ComputationalGraph,
        fmas: int = 0,
    ):
        self.children: dict[int, GraphState] = {}
        self.graph = graph
        self.transpose_graph = transpose_graph
        self.fmas = fmas
        self.lock = threading.Lock()


class VertexEliminator:
    def __init__(self, initial_graph, initial_transpose_graph):
        self.root = GraphState(
            self._copy_graph(initial_graph), self._copy_graph(initial_transpose_graph)
        )

    @staticmethod
    def _copy_graph(g: ComputationalGraph) -> ComputationalGraph:
        # Since we use immutables.Map, "copying" is just returning the same object.
        if isinstance(g, immutables.Map):
            return g

        # Initial conversion from dict to immutables.Map (happens once at root)
        outer_builder = {}
        for k1, v1 in g.items():
            inner_builder = {}
            for k2, v2 in v1.items():
                inner_builder[k2] = v2.copy() if hasattr(v2, "copy") else v2
            outer_builder[k1] = immutables.Map(inner_builder)
        return immutables.Map(outer_builder)

    def eliminate(self, order, jaxpr, sparsity_map, iota, vo_vertices, count_ops):
        node = self.root
        prefix_length = 0

        # Convert to dictionary for quick O(1) lookups
        sp_dict = dict(sparsity_map) if sparsity_map is not None else {}

        if ENABLE_CACHE:
            for vertex in order:
                sp_rules = sp_dict.get(vertex, ())
                key = (vertex, sp_rules)
                with node.lock:
                    if key in node.children:
                        node = node.children[key]
                        prefix_length += 1
                    else:
                        break

        fmas = node.fmas
        counts = []
        m_graph = node.graph.mutate()
        m_transpose_graph = node.transpose_graph.mutate()

        for vertex in order[prefix_length:]:
            sp_rules = sp_dict.get(vertex, ())
            step_fmas = _eliminate_vertex(
                vertex, jaxpr, m_graph, m_transpose_graph, iota, vo_vertices, sp_rules
            )
            fmas += step_fmas
            if count_ops:
                counts.append(fmas)

            if ENABLE_CACHE:
                key = (vertex, sp_rules)
                with node.lock:
                    if key not in node.children:
                        cur_graph = m_graph.finish()
                        cur_transpose_graph = m_transpose_graph.finish()
                        m_graph = cur_graph.mutate()
                        m_transpose_graph = cur_transpose_graph.mutate()
                        node.children[key] = GraphState(
                            cur_graph, cur_transpose_graph, fmas
                        )
                    node = node.children[key]

        graph = m_graph.finish()
        transpose_graph = m_transpose_graph.finish()

        return graph, transpose_graph, fmas, counts


_topology_cache = {}
_topology_lock = threading.Lock()
_topology_pending = {}

_TRACING_CACHE = {}
_TRACING_LOCK = threading.Lock()
_tracing_pending = {}


def vertex_elimination_jaxpr(
    jaxpr: core.Jaxpr,
    order: Union[Sequence[int], str],
    consts: Sequence[core.Literal],
    *args,
    argnums: Sequence[int] = (0,),
    count_ops: bool = False,
    sparse_representation: bool = False,
    sparsity_map: Sequence[tuple[int, tuple[tuple[int, ...], ...]]] = None,
) -> Any:
    # Explicitly clear caches to ensure new logical ID logic is applied
    global _TRACING_CACHE, _topology_cache
    with _TRACING_LOCK:
        _TRACING_CACHE.clear()
    with _topology_lock:
        _topology_cache.clear()
        
    jaxpr_invars = [invar for i, invar in enumerate(jaxpr.invars) if i in argnums]
    env, _, _, jaxpr_graph, vo_vertices = _build_graph(jaxpr, args, consts)

    eliminator = _get_eliminator(jaxpr, args, consts, tuple(argnums))

    iota = _iota_shape(jaxpr, argnums)
    order = _checkify_order(order, jaxpr, vo_vertices)

    graph, _, fmas, counts = eliminator.eliminate(
        order, jaxpr, sparsity_map, iota, vo_vertices, count_ops
    )

    m_final_graph = graph.mutate()
    for invar in jaxpr_invars:
        invar_graph = m_final_graph.get(invar)
        if invar_graph is not None:
            m_invar_graph = invar_graph.mutate()
            updated = False
            for outvar in jaxpr.outvars:
                tensor_val = m_invar_graph.get(outvar)
                if tensor_val is not None:
                    tensor = tensor_val.copy()
                    if len(tensor.pre_transforms) > 0:
                        for transform in tensor.pre_transforms[::-1]:
                            tensor = transform.apply_inverse(tensor, iota)
                    if len(tensor.post_transforms) > 0:
                        for transform in tensor.post_transforms:
                            tensor = transform.apply(tensor, iota)
                    m_invar_graph[outvar] = tensor
                    updated = True
            if updated:
                m_final_graph[invar] = m_invar_graph.finish()
    graph = m_final_graph.finish()

    if sparse_representation:
        jac_vals = []
        for outvar in jaxpr.outvars:
            for invar in jaxpr_invars:
                invar_graph = graph.get(invar)
                if invar_graph is not None and outvar in invar_graph:
                    jac_vals.append(invar_graph[outvar])
                else:
                    jac_vals.append(None)
    else:
        jac_vals = []
        for outvar in jaxpr.outvars:
            for invar in jaxpr_invars:
                invar_graph = graph.get(invar)
                if invar_graph is not None and outvar in invar_graph:
                    jac_vals.append(jnp.array(invar_graph[outvar]))
                else:
                    jac_vals.append(zeros_like(outvar, invar))

    n = len(jaxpr_invars)
    if n > 1:
        ratio = len(jac_vals) // n
        jac_vals = [tuple(jac_vals[i * n : i * n + n]) for i in range(0, ratio)]

    outputs = ([env[var] for var in jaxpr.outvars], jac_vals)

    if count_ops:
        aux = {}
        aux["fmas"] = fmas
        aux["order_counts"] = [
            (int(o), int(c[0]) if isinstance(c, tuple) else int(c))
            for o, c in zip(order, counts)
        ]
        return outputs, aux

    return outputs


def extract_jaxpr(
    jaxpr: core.Jaxpr,
    argnums: Sequence[int],
    order: Sequence[int],
    sparse_representation: bool,
    args: Sequence,
    consts: Sequence,
    sparsity_map: Sequence[tuple[int, tuple[tuple[int, ...], ...]]] = None,
) -> VEJaxpr:
    if isinstance(order, str):
        _, _, _, _, vo_vertices = _build_graph(jaxpr, args, consts)
        _order = tuple(_checkify_order(order, jaxpr, vo_vertices))
    elif hasattr(order, "tolist"):
        _order = tuple(map(int, order.tolist()))
    else:
        _order = tuple(map(int, order))

    _sparsity_map = (
        tuple((int(v), tuple(tuple(int(i) for i in pair) for pair in rules)) 
              for v, rules in sparsity_map)
        if sparsity_map is not None else ()
    )

    cache_key = (jaxpr, tuple(argnums), _order, _sparsity_map, sparse_representation)

    must_compute = False
    event = None
    with _topology_lock:
        if cache_key in _topology_cache:
            return _topology_cache[cache_key]

        if cache_key in _topology_pending:
            event = _topology_pending[cache_key]
            must_compute = False
        else:
            event = threading.Event()
            _topology_pending[cache_key] = event
            must_compute = True

    if not must_compute:
        if event:
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
                sparsity_map=_sparsity_map,
            )
            return tuple(res[0]) + tuple(res[1])

        dummy_args = [
            ShapeDtypeStruct(v.aval.shape, v.aval.dtype)
            for i, v in enumerate(jaxpr.invars)
            if i in argnums
        ]

        closed = jax.make_jaxpr(eval_graph)(*dummy_args)
        ve_jaxpr = VEJaxpr(closed.jaxpr)

        with _topology_lock:
            _topology_cache[cache_key] = ve_jaxpr
        return ve_jaxpr
    finally:
        with _topology_lock:
            del _topology_pending[cache_key]
        event.set()


import functools

import jax


def pytree_hash_cache(maxsize=None):
    def decorator(func):
        cache = {}
        lock = threading.Lock()
        pending = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            leaves, treedef = jax.tree_util.tree_flatten((args, kwargs))
            leaf_hashes = tuple(
                (id(leaf), leaf.shape, leaf.dtype)
                if hasattr(leaf, "shape")
                else hash(leaf)
                for leaf in leaves
            )
            key = hash((hash(treedef), leaf_hashes))

            must_compute = False
            event = None
            with lock:
                if key in cache:
                    return cache[key]

                if key in pending:
                    event = pending[key]
                    must_compute = False
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
                    del pending[key]
                event.set()

        return wrapper

    return decorator


@pytree_hash_cache()
def _get_eliminator(jaxpr: core.Jaxpr, args: tuple, consts: tuple, argnums: tuple):
    _, initial_graph, initial_transpose_graph, _, _ = _build_graph(jaxpr, args, consts)
    return VertexEliminator(initial_graph, initial_transpose_graph)


@pytree_hash_cache()
def _build_graph(
    jaxpr: core.Jaxpr, args: Sequence[jnp.ndarray], consts: Sequence[core.Literal]
):  # -> Tuple[ComputationalGraph, ComputationalGraph, Set[core.Var]]:
    """
        This function performs the `tracing` of the jaxpression into a computational
    graph representation that is amenable to the vertex elimination procedure.
    The computational graph is stored as a dict of dicts where basically every
    item can be accessed through `graph[source_vertex][dest_vertex]` and yields
    the corresponding "partial Jacobian". The
    transpose computational graph stores
    the same information in reverse order, i.e. \n

    \t ``graph[sv][dv] == transpose_graph[dv][sv]`` \n

    In addition
    to the two graph obejects, this function also generates a `set` containing
    all intermediate and output vertices. This is necessary in order to later be
    able to determine ...

    Args:
        jaxpr (core.Jaxpr): The jaxpr we want to differentiate.
        args (Sequence[jnp.ndarray]): The input arguments of the function as a
                                        flattened PyTree.
        consts (Sequence[core.Literal]): The constant arguments of the function.

    Returns:


    """
    env = {}  # env stores the primal value associated with the core.Var object

    counter = 1  # vertex id counter
    var_id = {}  # associates every application of a JaxprEqn with a unique integer
    # identifier that is later used when using the vertex elimination order.

    var_dim_ids = {} # Global mapping from (Var) to tuple of (dimension_ids)
    dim_counter = [1]

    def get_var_dim_ids(var):
        if var not in var_dim_ids:
            if hasattr(var, "aval") and hasattr(var.aval, "shape"):
                rank = len(var.aval.shape)
            else:
                rank = 0
            ids = tuple(range(dim_counter[0], dim_counter[0] + rank))
            var_dim_ids[var] = ids
            dim_counter[0] += rank
        return var_dim_ids[var]

    def set_tensor_logical_ids(st, out_ids, primal_ids):
        from graphax.sparse.tensor import replace
        
        def set_dim_lids(dims, ids):
            return tuple(replace(d, logical_id=ids[i]) for i, d in enumerate(dims))
            
        new_out = set_dim_lids(st.out_dims, out_ids)
        new_primal = set_dim_lids(st.primal_dims, primal_ids)
        return replace(st, out_dims=new_out, primal_dims=new_primal)

    graph = defaultdict(lambda: defaultdict())  # Input connectivity
    transpose_graph = defaultdict(lambda: defaultdict())  # Output connectivity
    jaxpr_graph = defaultdict(lambda: defaultdict())

    vo_vertices = set()  # contains all intermediate and output vertices
    # Writes a new elemental partial to the graph and transpose_graph
    def write_elemental(outvar, invar, val, eqns):
        _assert_sparse_tensor_consistency(val)
        if isinstance(invar, core.Var):
            # Assign global logical IDs to the elemental partial
            out_ids = get_var_dim_ids(outvar)
            in_ids = get_var_dim_ids(invar)
            val = set_tensor_logical_ids(val, out_ids, in_ids)
            
            graph[invar][outvar] = val
            transpose_graph[outvar][invar] = val
            jaxpr_graph[invar][outvar] = eqns

    # Reads variable and corresponding traced shaped array
    def read(var):
        if type(var) is core.Literal:
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
        # Treatment of intermediate variables that are also output variables
        for outvar in eqn.outvars:
            if type(outvar) is core.Var and outvar not in var_id.keys():
                var_id[outvar] = counter
                counter += 1

        # TODO send jamie, compare old approach with individual jaxpr
        # 5 pts for paper and lab interest
        # 5 pts for connection to their things

        for invar in eqn.invars:
            if invar in jaxpr.outvars:
                vertex = var_id[invar]
                vo_vertices.add(vertex)

        # print("eqn:", eqn)
        # print("invars", eqn.invars)
        # print("outvars", eqn.outvars)
        invals = safe_map(read, eqn.invars)

        primitive = eqn.primitive
        params = eqn.params
        invars = eqn.invars

        if primitive in elemental_rules:
            rule = elemental_rules[primitive]

            # Optimization: Memoize jax.make_jaxpr for elemental partials
            # Key by primitive, params, and input abstract shapes/dtypes
            cache_key = (
                primitive,
                tuple(sorted(params.items())),
                tuple(v.aval for v in invars),
            )

            must_compute = False
            event = None
            with _TRACING_LOCK:
                if cache_key in _TRACING_CACHE:
                    closed_jaxpr = _TRACING_CACHE[cache_key]
                    must_compute = False
                elif cache_key in _tracing_pending:
                    event = _tracing_pending[cache_key]
                    must_compute = False
                else:
                    event = threading.Event()
                    _tracing_pending[cache_key] = event
                    must_compute = True

            if not must_compute:
                event.wait() if event else None
                with _TRACING_LOCK:
                    closed_jaxpr = _TRACING_CACHE[cache_key]
            else:
                try:

                    def partial_fn(*args):
                        return rule(args, **params)

                    closed_jaxpr = jax.make_jaxpr(partial_fn)(*[v.aval for v in invars])

                    with _TRACING_LOCK:
                        _TRACING_CACHE[cache_key] = closed_jaxpr
                finally:
                    with _TRACING_LOCK:
                        del _tracing_pending[cache_key]
                    if event:
                        event.set()

            eqn_jaxpr = closed_jaxpr.jaxpr
            cce = partial(rule, **params)  # Re-define cce using the rule and params
        else:
            raise NotImplementedError(
                f"{eqn.primitive} does not have registered elemental partial."
            )

        primal_outvals, elemental_outvals = cce(invals)
        eqns = eqn_jaxpr  # Use the cached or newly created jaxpr
        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, primal_outvals)
        else:
            safe_map(write, eqn.outvars, [primal_outvals])
        invars = eqn.invars
        if eqn.primitive.multiple_results:
            for i, outvar in enumerate(eqn.outvars):
                if elemental_outvals[i] is not None:
                    for invar, tensor in zip(invars, elemental_outvals[i]):
                        if isinstance(invar, core.Var):
                            write_elemental(outvar, invar, tensor, eqns)
        else:
            outvar = eqn.outvars[0]
            if elemental_outvals is not None:
                for invar, tensor in zip(invars, elemental_outvals):
                    if isinstance(invar, core.Var):
                        write_elemental(outvar, invar, tensor, eqns)

    return env, graph, transpose_graph, jaxpr_graph, vo_vertices


def _iota_shape(jaxpr: core.Jaxpr, argnums: Sequence[int]) -> Any:
    """
    Function that computes the largest input and output tensors of the function
    by looking at the invals and outvals of the jaxpression. It then computes
    the corresponding larges Kronecker symbol that would be necessary to
    materialize possibly arising sparse tensors. The Kronecker symbol computed
    here will also be used throughout the vertex elimination computations.

    Args:
        jaxpr (core.Jaxpr): The jaxpr we want to differentiate.
        argnums (Sequence[int]): The argument numbers we want to differentiate
                                with respect to.

    Returns:
        jnp.ndarray: A Kronecker delta/unit matrix that is used for materializing
                    sparse tensors during the vertex elimination process.
    """
    largest_input = get_largest_tensor([jaxpr.invars[arg] for arg in argnums])
    largest_output = get_largest_tensor(jaxpr.outvars)

    # TODO check if this is meaningful
    if largest_input == 1 and largest_output == 1:
        return None
    elif largest_output == 1:
        return jnp.ones((1, largest_input))
    elif largest_input == 1:
        return jnp.ones((largest_output, 1))
    else:
        return jnp.eye(max(largest_output, largest_input), largest_input)


# would be nice to cache aswell :p
def _checkify_order(
    order: EliminationOrder, jaxpr: core.Jaxpr, vo_vertices: set[int]
) -> list[int]:
    """
    Function that checks if the supplied elimination order is valid for the
    given computational graph/jaxpr. In the case of an elimination order that
    has been provided as a string, it first maps the string to the respective
    order:
    - "fwd", "forward": [1, 2, 3, ...]
    - "rev", "reverse": [..., 3, 2, 1]

    Args:
        order (EliminationOrder): The elimination order to check.
        jaxpr (core.Jaxpr): The jaxpr we want to differentiate.
        vo_vertices (Set[core.Var]): A `set` containing all the output vertices.

    Returns:
        EliminationOrder: A valid elimination order.
    """
    if hasattr(order, "tolist"):
        order = cast(Any, order).tolist()

    # if it's a seq of JAX scalars, convert them to ints
    if (
        isinstance(order, (list, tuple))
        and len(order) > 0
        and not isinstance(order[0], (int, str))
    ):
        order = [int(o) for o in order]

    if isinstance(order, str):
        if order == "forward" or order == "fwd":
            return [
                i
                for i, eqn in enumerate(jaxpr.eqns, start=1)
                if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices
            ]
        elif order == "reverse" or order == "rev":
            return [
                i
                for i, eqn in enumerate(jaxpr.eqns, start=1)
                if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices
            ][::-1]
        else:
            raise ValueError(f"{order} is not a valid order identifier!")

    vertex_set = set(
        [
            i
            for i, eqn in enumerate(jaxpr.eqns, start=1)
            if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices
        ]
    )

    # Filter order to only include valid vertices, maintaining original relative order
    new_order: list[int] = [int(o) for o in order if int(o) in vertex_set]
    return new_order

def _accumulate_edge_triplet(
    v_i: int,
    v_j: int,
    v_k: int,
    graph: Any,
    transpose_graph: Any,
    iota: Any,
    sp_rules: tuple = ()
) -> int:
    in_edges_map = transpose_graph.get(v_j)
    out_edges_map = graph.get(v_j)

    if not in_edges_map or not out_edges_map:
        return 0

    pre_val_orig = in_edges_map.get(v_i)
    post_val_orig = out_edges_map.get(v_k)

    if pre_val_orig is None or post_val_orig is None:
        return 0

    pre_val = pre_val_orig.copy()
    post_val = post_val_orig.copy()

    _pre_val = pre_val.copy()
    _post_val = post_val.copy()

    if len(pre_val.post_transforms) > 0 and post_val.val is not None:
        _post_val = unload_post_transforms(post_val, pre_val, iota)

    if len(post_val.pre_transforms) > 0 and pre_val.val is not None:
        _pre_val = unload_pre_transforms(post_val, pre_val, iota)

    fmas = 0
    if pre_val.val is not None and post_val.val is not None:
        if _post_val.ndim == 0 or _pre_val.ndim == 0:
            edge_outval = _post_val * _pre_val
        else:
            edge_outval = _post_val @ _pre_val
            
        _assert_sparse_tensor_consistency(edge_outval)
        fmas += matmul_fmas(_post_val, _pre_val)
    elif pre_val.val is not None:
        edge_outval = _pre_val
    else:
        edge_outval = _post_val

    if len(post_val.post_transforms) > 0:
        edge_outval = prepend_post_transforms(post_val, edge_outval, iota)

    if len(pre_val.pre_transforms) > 0:
        edge_outval = append_pre_transforms(pre_val, edge_outval, iota)

    existing_inner = graph.get(v_i)
    existing_edge = existing_inner.get(v_k) if existing_inner else None

    if existing_edge is not None:
        _edge = existing_edge 

        if len(edge_outval.post_transforms) > 0:
            for transform in edge_outval.post_transforms:
                edge_outval = transform.apply(edge_outval, iota)

        if len(edge_outval.pre_transforms) > 0:
            for transform in edge_outval.pre_transforms[::-1]:
                edge_outval = transform.apply_inverse(edge_outval, iota)

        if len(_edge.post_transforms) > 0:
            for transform in _edge.post_transforms:
                _edge = transform.apply(_edge, iota)

        if len(_edge.pre_transforms) > 0:
            for transform in _edge.pre_transforms[::-1]:
                _edge = transform.apply_inverse(_edge, iota)

        _assert_sparse_tensor_consistency(edge_outval)
        edge_outval += _edge
        fmas += elementwise_fmas(edge_outval, _edge)

    if sp_rules:
        edge_outval = apply_dynamic_sparsity(edge_outval, sp_rules)
        _assert_sparse_tensor_consistency(edge_outval)

    inner_g = graph.get(v_i, immutables.Map())
    graph[v_i] = inner_g.update({v_k: edge_outval})

    inner_tg = transpose_graph.get(v_k, immutables.Map())
    transpose_graph[v_k] = inner_tg.update({v_i: edge_outval})

    return fmas


def execute_edge_accumulation(
    triplets: Sequence[tuple[int, int, int]],
    graph_mut: Any, 
    transpose_graph_mut: Any,
    iota: Any,
    sparsity_map: dict[tuple[int, int, int], tuple] = None,
    count_ops: bool = False
) -> tuple[Any, Any, int, list[int]]:
    
    fmas = 0
    counts = []
    sp_dict = sparsity_map if sparsity_map is not None else {}

    for v_i, v_j, v_k in triplets:
        sp_rules = sp_dict.get((v_i, v_j, v_k), ())
        step_fmas = _accumulate_edge_triplet(
            v_i, v_j, v_k, graph_mut, transpose_graph_mut, iota, sp_rules
        )
        fmas += step_fmas
        
        if count_ops:
            counts.append(fmas)

    return graph_mut.finish(), transpose_graph_mut.finish(), fmas, counts