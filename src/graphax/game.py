import copy
from typing import Callable, Tuple
from functools import partial

import jax
import jax.lax as lax

from jaxtyping import Array

from .graph import GraphState, is_bipartite
from .elimination import eliminate, front_eliminate, back_eliminate


# Wrapper to make the vertex-eliminate function usable with ft.partial
def vert_elim(info: Array, gs: GraphState, vertex: int):
    return eliminate(gs, vertex, info)


# Wrapper to make the front-eliminate function usable with ft.partial
def front_elim(info: Array, gs: GraphState, edge: Tuple[int, int]):
    return front_eliminate(gs, edge, info)


# Wrapper to make the back-eliminate function usable with ft.partial
def back_elim(info: Array, gs: GraphState, edge: Tuple[int, int]):
    return back_eliminate(gs, edge, info)


class VertexGame:
    """
    OpenAI-like gymnax environment for a game where to goal is to find the 
    best vertex elimination order with minimal multiplication count.
    This game always has finite termination range.

    The `state` of the game is essentially the matrix containing the edges of the
    computational graph and the array containing the edges that have already been
    eliminated.

    The `reward` is the negative number of multiplications since we want to 
    minimize that.

    The `action space` is equal to the number of remaining vertices that can be
    eliminated. For example, for 10 intermediate variables, there are 10 
    different actions. However, every action can only be executed once. This is
    realized by action masking where the logits of actions that have already 
    been performed are sent to -inf.

    The `termination` of the game is indicated by the is_bipartite feature, i.e.
    the game is over when all intermediate vertices and edges have been eliminated.
    """
    gs: GraphState
    vertex_eliminate: Callable
    
    def __init__(self, gs: GraphState) -> None:
        super().__init__()
        self.gs = copy.deepcopy(gs) # copy of the computational graph for reset
        self.vertex_eliminate = partial(vert_elim, gs.get_info())
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self,
            gs: GraphState,
            action: int) -> Tuple[GraphState, float, bool]:  
        # Actions go from 0 to nintermediates-1 and vertices go from 1 to nintermediates      
        vertex = action + 1
        
        gs, nops = self.vertex_eliminate(gs, vertex)

        # Reward is the negative of the multiplication count
        reward = -nops
        
        # Check if the graph is bipartite, i.e. the game is over
        gs, terminated = lax.cond(is_bipartite(gs),
                    lambda g: (self.reset(), True),
                    lambda g: (g, False),
                    gs)
    
        return gs, reward, terminated
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self) -> GraphState:
        return copy.deepcopy(self.gs)


# TODO add comments and docstring
class EdgeGame:
    """
    OpenAI-like gymnax environment for the game
    
    The game always has finite termination range!
    """
    gs: GraphState
    ninputs: int
    front_eliminate: Callable
    back_eliminate: Callable
    
    def __init__(self, gs: GraphState) -> None:
        super().__init__()
        self.gs = copy.deepcopy(gs)
        self.ninputs = gs.get_info()[0]
        self.front_eliminate = partial(front_elim, gs.get_info())
        self.back_eliminate = partial(back_elim, gs.get_info())
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self,
            gs: GraphState,
            action: Array) -> Tuple[GraphState, float, bool]:
        i, j, mode = action

        i = i - self.ninputs + 1
        j += 1

        gs, nops = lax.cond(mode == 0,
                            lambda g: self.front_eliminate(g, (i,j)),
                            lambda g: self.back_eliminate(g, (i,j)),
                            gs)
        
        reward = -nops
    
        gs, terminated = lax.cond(is_bipartite(gs),
                            lambda g: (self.reset(), True),
                            lambda g: (g, False),
                            gs)
        return gs, reward, terminated
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self) -> GraphState:
        return copy.deepcopy(self.gs)

