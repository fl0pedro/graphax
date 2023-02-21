from typing import Tuple

import chex

from ..core import GraphInfo, make_empty_edges, add_edge


def construct_LIF() -> Tuple[chex.Array, GraphInfo]:
    info = GraphInfo(num_inputs=6,
                    num_intermediates=9,
                    num_outputs=3,
                    num_edges=0)
    edges = make_empty_edges(info)
    
    edges, info = add_edge(edges, (-5,1), info)
    
    edges, info = add_edge(edges, (-4, 1), info)
    edges, info = add_edge(edges, (-4, 2), info)
    
    edges, info = add_edge(edges, (-3, 3), info)
    edges, info = add_edge(edges, (-3, 5), info)
    
    edges, info = add_edge(edges, (-2, 3), info)
    edges, info = add_edge(edges, (-2, 4), info)
    
    edges, info = add_edge(edges, (-1, 6), info)
    
    edges, info = add_edge(edges, (0, 9), info)
    
    edges, info = add_edge(edges, (1, 7), info)
    
    edges, info = add_edge(edges, (2, 5), info)
    
    edges, info = add_edge(edges, (3, 8), info)
    
    edges, info = add_edge(edges, (4, 6), info)
    
    edges, info = add_edge(edges, (5, 7), info)
    
    edges, info = add_edge(edges, (6, 8), info)
    
    edges, info = add_edge(edges, (7, 9), info)
    edges, info = add_edge(edges, (7, 11), info)
    
    edges, info = add_edge(edges, (8, 12), info)
    
    edges, info = add_edge(edges, (9, 10), info)
    edges, info = add_edge(edges, (9, 11), info) # gated reset
    return edges, info
