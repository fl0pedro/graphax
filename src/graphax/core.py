from typing import Callable, Dict, Sequence, Set, Tuple, Union
from functools import wraps, partial
from collections import defaultdict

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from jax._src.util import safe_map
import jax._src.core as core

from .primitives import elemental_rules
from .sparse.tensor import get_num_muls, get_num_adds, _assert_sparse_tensor_consistency
from .sparse.utils import zeros_like, get_largest_tensor


def tree_allclose(tree1, tree2, equal_nan: bool = False) -> bool:
    allclose = lambda a, b: jnp.allclose(a, b, equal_nan=equal_nan, atol=1e-5, rtol=1e-4)
    is_equal = jtu.tree_map(allclose, tree1, tree2)
    return jtu.tree_reduce(jnp.logical_and, is_equal)


EliminationOrder = Union[Sequence[int], str]
ComputationalGraph = Dict[core.Var, Dict[core.Var, jnp.ndarray]]


def jacve(fun: Callable, order: EliminationOrder, argnums: Sequence[int] = (0,), 
          has_aux: bool = False, count_ops: bool = False, 
          sparse_representation: bool = False) -> Callable:
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
    `sparsee_representation` flag can be set to `True` to return the Jacobian in 
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
    def jacfun(*args, **kwargs):

        # TODO Make repackaging work properly with one input value only
        flattened_args, in_tree = jtu.tree_flatten(args)
        closed_jaxpr = jax.make_jaxpr(fun)(*flattened_args, **kwargs)

        out = vertex_elimination_jaxpr(closed_jaxpr.jaxpr, 
                                        order, 
                                        closed_jaxpr.literals, 
                                        *args, 
                                        has_aux=has_aux,
                                        argnums=argnums,
                                        count_ops=count_ops,
                                        sparse_representation=sparse_representation)

        # TODO does not support aux and count_opt simultaneously
        if count_ops: 
            out, op_counts = out
            out_tree = jtu.tree_structure(tuple(closed_jaxpr.jaxpr.outvars))
            if len(closed_jaxpr.jaxpr.outvars) == 1 and len(closed_jaxpr.jaxpr.invars) > 1:
                return out[0], op_counts
            return jtu.tree_unflatten(out_tree, out), op_counts
        elif has_aux:
            primal_out, grads = out
            out_tree = jtu.tree_structure(tuple(closed_jaxpr.jaxpr.outvars))
            if len(closed_jaxpr.jaxpr.outvars) == 1 and len(closed_jaxpr.jaxpr.invars) > 1:
                return primal_out[0], grads[0]
            return jtu.tree_unflatten(out_tree, primal_out), jtu.tree_unflatten(out_tree, grads)
        else:
            out_tree = jtu.tree_structure(tuple(closed_jaxpr.jaxpr.outvars))
            if len(closed_jaxpr.jaxpr.outvars) == 1 and len(closed_jaxpr.jaxpr.invars) > 1:
                return out[0]
            return jtu.tree_unflatten(out_tree, out)
    return jacfun


def _iota_shape(jaxpr: core.Jaxpr, argnums: Sequence[int]) -> jnp.ndarray:
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
    largest_input = get_largest_tensor([jaxpr._invars[arg] for arg in argnums])
    largest_output = get_largest_tensor(jaxpr._outvars)
        
    # TODO check if this is meaningful
    if largest_input == 1 and largest_output == 1:
        return None
    elif largest_output == 1:
        return jnp.ones((1, largest_input))
    elif largest_input == 1:
        return jnp.ones((largest_output, 1))
    else:
        return jnp.eye(max(largest_output, largest_input), largest_input) 
        
        
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
    out.post_transforms = transforms
    return out


def append_pre_transforms(pre, out, iota):
    transforms = pre.pre_transforms + out.pre_transforms
    out.pre_transforms = transforms
    return out
        
    
def _eliminate_vertex(vertex: int, jaxpr: core.Jaxpr, graph: ComputationalGraph, 
                      transpose_graph: ComputationalGraph, iota: jnp.ndarray, 
                      vo_vertices: Set[core.Var]) -> Tuple[int, int]:
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
        iota (jnp.ndarray): A Kronecker delta/unit matrix which is helpful when
                            materializing sparse tensors.
        vo_vertices (Set[core.Var]): A `set` containing all the output vertices.

    Returns:
        Tuple[int, int]: The number of multiplications and additions that were
                        performed during the elimination of the vertex

    TODO: Having the function return the number of multiplications and additions
    is probably not the best idea. Can we do this somehow more beautifully?
    Maybe have a impure function and add an additional argument instead?
    """
    eqn = jaxpr.eqns[vertex-1]
    num_mul, num_add = 0, 0
    for out_edge in graph[eqn.outvars[0]].keys():
        post_val = graph[eqn.outvars[0]][out_edge].copy()
        for in_edge in transpose_graph[eqn.outvars[0]].keys():
            pre_val = transpose_graph[eqn.outvars[0]][in_edge].copy()
            
            # TODO implement a process that discards unnecessary edges from the computation
            
            # Handle stuff like reshape, squeeze etc.            
            # Apply Jacobian transforms where applicable
            _pre_val = pre_val.copy()
            _post_val = post_val.copy()

            # print(in_edge.count, "->", eqn.outvars[0].count, "->", out_edge.count)
            # print("Post:", _post_val)
            # print("Pre:", _pre_val) 
            
            if len(pre_val.post_transforms) > 0 and post_val.val is not None:
                _post_val = unload_post_transforms(post_val, pre_val, iota)
                
            if len(post_val.pre_transforms) > 0 and pre_val.val is not None:
                _pre_val = unload_pre_transforms(post_val, pre_val, iota)
                                
            # Multiply the two values of the edges if applicable
            if pre_val.val is not None and post_val.val is not None:     
                edge_outval = _post_val * _pre_val
                num_mul += get_num_muls(_post_val, _pre_val)
                    
            elif pre_val.val is not None:
                edge_outval = _pre_val  
            else:
                edge_outval = _post_val
                
            # print("Edge_outval:", edge_outval)
            # Offload the remain Jacobian transforms to the output tensor
            if len(post_val.post_transforms) > 0:
                edge_outval = prepend_post_transforms(post_val, edge_outval, iota)

            if len(pre_val.pre_transforms) > 0:
                edge_outval = append_pre_transforms(pre_val, edge_outval, iota)
                                                
            # If there is already an edge between the two vertices, add the new
            # edge to the existing one
            if graph.get(in_edge).get(out_edge) is not None:
                _edge = transpose_graph[out_edge][in_edge]  
                # print("Edge_outval:", edge_outval)      
                # print("Edge:", _edge)  
  
                # Offload the remaining Jacobian transforms to the output tensor
                if len(edge_outval.post_transforms) > 0:
                    for transform in edge_outval.post_transforms:
                        edge_outval = transform.apply(edge_outval, iota)

                if len(edge_outval.pre_transforms) > 0:
                    for transform in edge_outval.pre_transforms[::-1]: # Do we need the [::-1] here?
                        edge_outval = transform.apply_inverse(edge_outval, iota)
                
                # Offload the remain Jacobian transforms to the output tensor
                if len(_edge.post_transforms) > 0:
                    for transform in _edge.post_transforms:
                        _edge = transform.apply(_edge, iota)

                if len(_edge.pre_transforms) > 0:
                    for transform in _edge.pre_transforms[::-1]: # Do we need the [::-1] here?
                        _edge = transform.apply_inverse(_edge, iota)

                _assert_sparse_tensor_consistency(edge_outval)
                edge_outval += _edge
                num_add += get_num_adds(edge_outval, _edge)
                
            _assert_sparse_tensor_consistency(edge_outval)
            # print("Edge_outval:", edge_outval)
            graph[in_edge][out_edge] = edge_outval
            transpose_graph[out_edge][in_edge] = edge_outval
                
    # Cleanup of input and output edges
    if vertex not in vo_vertices:
        for in_vertex in transpose_graph[eqn.outvars[0]].keys():
            del graph[in_vertex][eqn.outvars[0]]
    for out_vertex in graph[eqn.outvars[0]].keys():    
        del transpose_graph[out_vertex][eqn.outvars[0]]
    
    # Cleanup the eliminated vertex
    del graph[eqn.outvars[0]]
    if vertex not in vo_vertices:
        del transpose_graph[eqn.outvars[0]]

    return num_mul, num_add


def _checkify_order(order: EliminationOrder, 
                    jaxpr: core.Jaxpr, 
                    vo_vertices: Set[core.Var]) -> EliminationOrder:
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
    if type(order) is str:
        if order == "forward" or order == "fwd":
            return [i for i, eqn in enumerate(jaxpr.eqns, start=1) 
                    if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices]
        elif order == "reverse" or order == "rev":
            return [i for i, eqn in enumerate(jaxpr.eqns, start=1) 
                    if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices][::-1]
        else:
            raise ValueError(f"{order} is not a valid order identifier!")
    else:
        vertex_set = set([i for i, eqn in enumerate(jaxpr.eqns, start=1) 
                    if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices])
    
        set_from_order = set(order)
        missing_vertices = vertex_set.difference(set_from_order)
        if len(missing_vertices) > 0:
            raise ValueError(f"Supplied order is missing vertices {missing_vertices}!")
    return order


def _build_graph(jaxpr: core.Jaxpr, 
                args: Sequence[jnp.ndarray], 
                consts: Sequence[core.Literal]
    ) -> Tuple[ComputationalGraph, ComputationalGraph, Set[core.Var]]:
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

    Returns:


    """
    env = {} # env stores the primal value associated with the core.Var object

    graph = defaultdict(lambda: defaultdict()) # Input connectivity
    transpose_graph = defaultdict(lambda: defaultdict()) # Output connectivity  
        
    vo_vertices = set() # contains all intermediate and output vertices
    counter = 1 # vertex id counter
    var_id = {} # associates every application of a JaxprEqn with a unique integer
    # identifier that is later used when using the vertex elimination order.
    # NOTE: This only works well if the output is a single value.
    # It is ill-defined when having functions with more than one output!.

    # Reads variable and corresponding traced shaped array
    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    # Adds new variable and corresponding traced shaped array
    def write(var, val):
        env[var] = val
        
    # Writes a new elemental partial to the graph and transpose_graph
    def write_elemental(outvar, invar, val):
        _assert_sparse_tensor_consistency(val)
        if isinstance(invar, core.Var):
            graph[invar][outvar] = val
            transpose_graph[outvar][invar] = val
                            
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
                    
        for invar in eqn.invars:
            if invar in jaxpr._outvars:
                vertex = var_id[invar]
                vo_vertices.add(vertex)
                
        # print("eqn:", eqn)
        # print("invars", eqn.invars)
        # print("outvars", eqn.outvars)
        invals = safe_map(read, eqn.invars)      

        if eqn.primitive not in elemental_rules:
            raise NotImplementedError(f"{eqn.primitive} does not have registered elemental partial.")
        cce = elemental_rules.get(eqn.primitive)
        primal_outvals, elemental_outvals = cce(invals, **eqn.params)
        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, primal_outvals)
        else:
            safe_map(write, eqn.outvars, [primal_outvals])
        invars = [invar for invar in eqn.invars if type(invar) is core.Var]
        # NOTE: Currently only able to treat one output variable

        _write_elemental = partial(write_elemental, eqn.outvars[0])
        if len(invars) == len(elemental_outvals):
            safe_map(_write_elemental, invars, elemental_outvals)
        
    return env, graph, transpose_graph, vo_vertices


def _prune_graph(graph: ComputationalGraph, 
                transpose_graph: ComputationalGraph, 
                jaxpr: core.Jaxpr, 
                argnums: Sequence[int]) -> None:
    """
    Function that prunes a given computational graph based on the argnums we
    give it, i.e. for argnums that we do not differentiate for we can just ignore
    them and all edges solely connected to them. This might incur significant
    savings. It also checks for dead intermediate vertices that have either no
    input or no output edges. These typically arise from a lax.stop_grad operation
    somewhere in the function we want to differentiate. These dead vertices and
    all associated edges are deleted as well.

    Args:
        graph (ComputationalGraph): The computational graph representation of the
                                    jaxpr.
        transpose_graph (ComputationalGraph): The transpose computational graph
                                            representation of the jaxpr.
        jaxpr (core.Jaxpr): The jaxpr we want to differentiate.
        argnums (Sequence[int]): The argument numbers we want to differentiate
                                with respect to.

    TODO: Implement some unit tests for pruning. Maybe disable it for now?
    """
    has_dead_vertices = True
    for i, invar in enumerate(jaxpr.invars):
        if i not in argnums:
            for in_edge in transpose_graph[invar].keys():
                del graph[in_edge][invar]
            for out_edge in graph[invar].keys():   
                del transpose_graph[out_edge][invar]   
                
            del graph[invar]
            del transpose_graph[invar]
            # print("Pruned input variable:", invar)
        
    already_deleted = []
    while has_dead_vertices:
        to_delete = []
        for eqn in jaxpr.eqns:
            ov = eqn.outvars[0]
            if ov not in jaxpr.outvars and ov not in already_deleted:
                if len(graph[ov]) == 0 or len(transpose_graph[ov]) == 0:
                    to_delete.append(ov) 
                    
        if len(to_delete) > 0:
            for ov in to_delete:
                for in_edge in transpose_graph[ov].keys():
                    del graph[in_edge][ov]
                for out_edge in graph[ov].keys():   
                    del transpose_graph[out_edge][ov]   
                    
                del graph[ov]
                del transpose_graph[ov] 
                # print("Pruned output variable:", ov)
            already_deleted.extend(to_delete)
        else:
            has_dead_vertices = False


def vertex_elimination_jaxpr(jaxpr: core.Jaxpr, 
                            order: Union[Sequence[int], str], 
                            consts: Sequence[core.Literal], 
                            *args, 
                            has_aux: bool = False,
                            argnums: Sequence[int] = (0,),
                            count_ops: bool = False,
                            sparse_representation: bool = False
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
        count_ops (bool, optional): Count the number of operations during the
                                    elimination process. Defaults to `False`.
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
    env, graph, transpose_graph, vo_vertices = _build_graph(jaxpr, args, consts)
    # _prune_graph(graph, transpose_graph, jaxpr, argnums) NOTE graph pruning is disabled for now
    
    iota = _iota_shape(jaxpr, argnums)
        
    # Eliminate the vertices
    num_muls, num_adds = 0, 0
    counts = []
    order = _checkify_order(order, jaxpr, vo_vertices)
    for vertex in order:
        num_mul, num_add = _eliminate_vertex(vertex, jaxpr, graph, transpose_graph, iota, vo_vertices)
        if count_ops:
            counts.append((num_mul, num_add))
            num_muls += num_mul
            num_adds += num_add
           
    # Offloading all remaining Jacobian transforms to the output variables 
    # before densification!    
    for invar in jaxpr_invars:
        for outvar in jaxpr.outvars:
            if graph.get(invar) is not None:
                if graph.get(invar).get(outvar) is not None:
                    tensor = graph[invar][outvar].copy()
                    if len(tensor.pre_transforms) > 0:
                        for transform in tensor.pre_transforms[::-1]: # Do we need the [::-1] here?
                            tensor = transform.apply_inverse(tensor, iota)
                    if len(tensor.post_transforms) > 0:
                        for transform in tensor.post_transforms:
                            tensor = transform.apply(tensor, iota)
                    graph[invar][outvar] = tensor
    
    # Collect outputs  
    if sparse_representation:   
        jac_vals = [graph[invar][outvar]
                    if outvar in list(graph[invar].keys()) else None
                    for outvar in jaxpr.outvars for invar in jaxpr_invars]
    else:
        jac_vals = [graph[invar][outvar].dense(iota) 
                    if outvar in list(graph[invar].keys()) else zeros_like(outvar, invar)
                    for outvar in jaxpr.outvars for invar in jaxpr_invars]
        
    # Restructure Jacobians for more complicated pytrees
    n = len(jaxpr_invars)
    if n > 1:
        ratio = len(jac_vals)//n
        jac_vals = [tuple(jac_vals[i*n:i*n+n]) for i in range(0, ratio)]
        
    if count_ops:
        # TODO: this needs to be reworked, aux should contain the primal values
        # so that we can compute stuff like loss_and_grad
        order_counts = [(int(o), int(c[0])) for o, c in zip(order, counts)]
        aux = {"num_muls": num_muls, 
                "num_adds": num_adds, 
                "order_counts": order_counts}
        return jac_vals, aux
    if has_aux:
        return [env[var] for var in jaxpr.outvars], jac_vals

    return jac_vals

