from typing import Any, Callable, Sequence, Generator, TypeAlias, Iterable, NamedTuple

# import numpy as np
from chex import Array

import jax
from jax import lax, jit
import jax.numpy as jnp
from .tensor import SparseTensor
import operator
from dataclasses import dataclass, KW_ONLY
import copy
from functools import reduce, partial
import numpy as np

# TODO: make parent class, or inherit sparse tensor ??

class DenseDimension(NamedTuple):
    id: int
    size: int
    val_dim: int | None
    #val_axis: int = None

# val_axes: These are two axis of the blocks that we would like to apply the diagonal on.
#   By default pick the first two (0, 1) as the axis). TODO: make negatives work like indeces
class SparseDimension(NamedTuple):
    id: int
    size: int
    val_dim: int
    other_id: int
    block_size: int = None
    #val_axis: int = None

# TODO TODO TODO TODO TODO, the new idea is to setup Sparse dim, such that if two different val_dims are set for a pair,
# then its sparse block...


Dimension = DenseDimension | SparseDimension

# find a better name for this.
MultiSparseDimensionBlocks: TypeAlias = Sequence[Array] | Sequence['MultiSparseDimensionBlocks']

# TODO (somewhere) blocks of blocks with neighbors can be combined to one block
# I believe that tensors with d diagonal then there will be 2d blocks maximum (the rest can be merged into individual blocks)
class BlockSparseTensor:
    out_dims: Any
    primal_dims: Any
    out_shape: tuple[int, ...]
    primal_shape:  tuple[int, ...]
    shape:  tuple[int, ...]
    size: int
    ndim: int
    blocks: MultiSparseDimensionBlocks | Array | None
    pre_transforms: Array
    post_transforms: Array
    sparse_dims: int
    block_shape: tuple[int, ...]
    block_size: int
    _sparse_dim_order: list[tuple[int, int]]

    def __init__(self,
                 out_dims: Sequence[Dimension],
                 primal_dims: Sequence[Dimension],
                 out_shape: Sequence[int],
                 primal_shape: Sequence[int],
                 blocks: MultiSparseDimensionBlocks | Array | None,
                 sparse_dims: int,
                 pre_transforms: Sequence[Callable] = None,
                 post_transforms: Sequence[Callable] = None) -> None:

        self.out_dims = out_dims if isinstance(out_dims, tuple) else tuple(out_dims)
        self.primal_dims = primal_dims if isinstance(primal_dims, tuple) else tuple(primal_dims)

        self.sparse_dims = sparse_dims
        self.block_shape = blocks.shape[self.sparse_dims:]
        self.block_size = reduce(operator.mul, self.block_shape)

        self.out_shape = tuple(out_shape)

        self.primal_shape = tuple(primal_shape)

        # self.out_shape = get_block_shape(out_dims)
        # self.primal_shape = get_block_shape(primal_dims)

        self.shape = tuple(self.out_shape + self.primal_shape) # isn't quite right
        self.size = reduce(operator.mul, self.shape)
        self.ndim = len(self.shape)

        self.blocks = blocks

        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms

        self._sparse_dim_order = [(d.id, d.other_id) for d in self.out_dims if isinstance(d, SparseDimension)] # may not be necessary to include, but must be mentioned in the docs

    def __repr__(self) -> str:
        def map_str(a: Sequence) -> Generator:
            return (str(s) for s in a)

        def multiline_seq(s: Sequence, brackets: str) -> str:
            lb, rb, *_ = brackets
            if s:
                res = f'{lb}\n\t\t' + ',\n\t\t'.join(map_str(s)) + f',\n\t{rb}'
            else:
                res = lb + rb
            return res

        str_out_shape = ', '.join(map_str(self.out_shape))
        str_primal_shape = ', '.join(map_str(self.primal_shape))

        multiline_out_dims = multiline_seq(self.out_dims, '()')
        multiline_primal_dims = multiline_seq(self.primal_dims, '()')
        multiline_pre_transform = multiline_seq(self.pre_transforms, '[]')
        multiline_post_transform = multiline_seq(self.post_transforms, '[]')

        return f"""BlockSparseTensor(
    shape = ({str_out_shape} | {str_primal_shape}),
    out_dims = {multiline_out_dims},
    primal_dims = {multiline_primal_dims},
    blocks = Array(shape={self.blocks.shape}, dtype={self.blocks.dtype}),
    sparse_dims = {self.sparse_dims},
    pre_transforms = {multiline_pre_transform},
    post_transforms = {multiline_post_transform}
)"""

    # This is not a transpose like w/ normal tensors. The order should be completely reversed.
    def transpose(self, *args):
        if len(args) > 0:
            pass # regular transpose, warn on breaking of sparsity
        else:
            out_dims = [
                d._replace(
                    id=d.id - len(self.out_dims),
                    other_id=d.other_id + len(self.primal_dims),
                )
                if isinstance(d,SparseDimension)
                else d._replace(id=d.id - len(self.out_dims))
                for d in self.primal_dims
            ]
            primal_dims = [
                d._replace(
                    id=d.id + len(self.primal_dims),
                    other_id=d.other_id - len(self.out_dims),
                )
                if isinstance(d, SparseDimension) 
                else d._replace(id=d.id + len(self.primal_dims))
                for d in self.out_dims
            ]
            return BlockSparseTensor(
                out_dims,
                primal_dims,
                self.primal_shape,
                self.out_shape,
                self.blocks,
                self.sparse_dims
            )

    def block_until_ready(self):
        self.blocks.block_until_ready()
        return self

    @property
    def T(self):
        return self.transpose()

    def dense(self) -> Array:
        return _dense(self)

    def __add__(rhs, lhs):
        return _add(rhs, lhs)

    def __mul__(rhs, lhs):
        return _mul(rhs, lhs)

    def __matmul__(rhs, lhs):
        return _matmul(rhs, lhs)

def has_equal_depth(node: MultiSparseDimensionBlocks, depth=0):
    if isinstance(node, Array):
        return True, depth
    else:
        equality, depth = zip(*[has_equal_depth(child, depth+1) for child in node])
        local_equality = all(d == depth[0] for d in depth)
        return all(equality) and local_equality, max(depth)

# TODO, this will be quite complex...
def has_equal_shapes(node: MultiSparseDimensionBlocks, dims = Sequence[Dimension], shape=None): 
    ...

def new_block_sparse_tensor(
    out_dims: Sequence[Dimension],
    primal_dims: Sequence[Dimension],
    blocks: MultiSparseDimensionBlocks | Array | None,
    pre_transforms: Sequence[Callable] = None,
    post_transforms: Sequence[Callable] = None
) -> BlockSparseTensor:

    if pre_transforms is None:
        pre_transforms = []
    if post_transforms is None:
        post_transforms = []

    n = sum(isinstance(d, SparseDimension) for d in out_dims)
    assert n == sum(isinstance(d, SparseDimension) for d in primal_dims)

    #assert all(d == i for d, i in zip(sorted_val_dims, list(range(n)))), \
    #    "Value dimensions should be continuous"
    
    #print(sorted_val_dims)
    #print(
    #    {d.val_axis for d in out_dims if isinstance(d, DenseDimension)} 
    #    | {d.val_axis for d in primal_dims}
    #)
    #assert n == len(
    #        {d.val_axis for d in out_dims if isinstance(d, DenseDimension)} 
    #        | {d.val_axis for d in primal_dims}
    #    ), "Value axis should be unique"

    # TODO add checks between primal and out dims.

    if isinstance(blocks, Array):
        assert blocks.ndim > n
        block_shape = blocks.shape[n:]
        # TODO add ones for non mentioned areas

        #print(blocks.shape)
        #print(block_shape)

        out_shape = tuple(
            x.size if isinstance(x, DenseDimension)
            else x.size * block_shape[x.val_dim]
            for x in out_dims
        )
        primal_shape = tuple(
            x.size if isinstance(x, DenseDimension)
            else x.size * block_shape[x.val_dim]
            for x in primal_dims
        )
        #print(out_shape, primal_shape)
    else:
        raise TypeError("blocks as MultiSparseDimensionBlocks is not yet implemented")
        assert has_equal_depth(blocks) == (True, n)
        

    return BlockSparseTensor(
        out_dims,
        primal_dims,
        out_shape,
        primal_shape,
        blocks,
        n,
        pre_transforms,
        post_transforms
    )

def get_ienumerated_blocks(seq: Sequence, cur_idx: list[int] = None) -> Iterable[tuple[list[int], Array]]:
    if cur_idx is None:
        cur_idx = []
    for i, elem in enumerate(seq):
        if isinstance(elem, Sequence):
            for res in get_ienumerated_blocks(elem, cur_idx + [i]):
                yield res
        elif isinstance(elem, Array):
            yield cur_idx + [i], elem

def _calculate_coords_for_one_idx(
    idxs: Array,
    dim1_ids: Array,
    dim2_ids: Array,
    block_sizes: Array,
    sparse_indices: Array,
    shape_len: int
) -> Array:
    """
    JIT-compatible helper to calculate start_coords for a single block index.
    
    This function will be vectorized with vmap.
    """
    # Start with all-zero coordinates
    start_coords = jnp.zeros(shape_len, dtype=jnp.int32)
    
    # Get the specific index values (e.g., idxs[0], idxs[1], ...)
    # for each sparse dimension
    idx_vals = idxs.take(sparse_indices)
    
    # Calculate the coordinate offset (block_size * index)
    coord_vals = block_sizes * idx_vals
    
    # Set the coordinates for both paired dimensions
    start_coords = start_coords.at[dim1_ids].set(coord_vals)
    start_coords = start_coords.at[dim2_ids].set(coord_vals)
    
    return start_coords


def _dense(bst: BlockSparseTensor) -> Array:
    """
    Efficiently converts a BlockSparseTensor to a dense Array 
    using vmap and scan.
    """
    shape = bst.out_shape + bst.primal_shape
    dense_tensor_init = jnp.zeros(shape, dtype=bst.blocks.dtype)

    # --- 1. Original Transpose (unchanged) ---
    blocks = bst.blocks.transpose(
        *range(bst.sparse_dims),
        *(d.val_dim + bst.sparse_dims
          for d in bst.out_dims + bst.primal_dims)
    )
    
    # --- 2. Pre-process Metadata ---
    # Convert Python-level dimension info into JAX arrays
    sparse_dim_info = []
    i = 0
    for dim1 in bst.out_dims:
        if isinstance(dim1, SparseDimension):
            dim2 = bst.primal_dims[dim1.other_id - len(bst.out_dims)]
            # Store (dim1.id, dim2.id, block_size, sparse_axis_index)
            sparse_dim_info.append(
                (dim1.id, dim2.id, dim1.block_size, i)
            )
            i += 1

    if sparse_dim_info:
        info_array = jnp.array(sparse_dim_info, dtype=jnp.int32)
        dim1_ids = info_array[:, 0]
        dim2_ids = info_array[:, 1]
        block_sizes = info_array[:, 2]
        sparse_indices = info_array[:, 3]
    else:
        # Handle case with no sparse dimensions
        dim1_ids = jnp.array([], dtype=jnp.int32)
        dim2_ids = jnp.array([], dtype=jnp.int32)
        block_sizes = jnp.array([], dtype=jnp.int32)
        sparse_indices = jnp.array([], dtype=jnp.int32)
        
    # --- 3. Pre-compute All Indices and Coordinates ---
    
    # Get all multi-dimensional sparse indices
    sparse_shape = bst.blocks.shape[:bst.sparse_dims]
    
    # np.ndindex is fine here, as it runs once during tracing
    all_idxs_np = np.array(list(np.ndindex(sparse_shape))) 
    
    if all_idxs_np.size == 0:
        # Handle edge case: 0 sparse dims (1 block)
        if np.prod(sparse_shape) == 1:
            all_idxs_np = np.empty((1, 0), dtype=int)
        else:
            # No blocks, just return the zero tensor
            return dense_tensor_init
            
    all_idxs = jnp.array(all_idxs_np) # Shape: (num_blocks, bst.sparse_dims)

    # Vectorize the coordinate calculation over all indices
    vmapped_coord_calc = jax.vmap(
        _calculate_coords_for_one_idx,
        in_axes=(0, None, None, None, None, None) # vmap over all_idxs
    )
    
    # Calculate all start coordinates in parallel
    all_start_coords = vmapped_coord_calc(
        all_idxs, dim1_ids, dim2_ids, block_sizes, sparse_indices, len(shape)
    ) # Shape: (num_blocks, len(shape))
    
    # --- 4. Flatten Blocks ---
    num_blocks = all_idxs.shape[0]
    val_shape = blocks.shape[bst.sparse_dims:]
    flat_blocks = blocks.reshape(num_blocks, *val_shape)
    
    # --- 5. Run Sequential Scan ---
    
    def update_step(carry_dense_tensor, xs):
        """One step of the scan loop."""
        start_coords, block_data = xs
        
        new_dense_tensor = lax.dynamic_update_slice(
            carry_dense_tensor, block_data, start_coords
        )
        # Return new carry (tensor) and no y output
        return new_dense_tensor, None 

    # Run the scan over all blocks and their coordinates
    final_dense, _ = lax.scan(
        update_step,
        init=dense_tensor_init,
        xs=(all_start_coords, flat_blocks)
    )
    
    return final_dense

# @partial(jit, static_argnames=('lhs', 'rhs'))
def _add(lhs, rhs):
    assert rhs.shape == lhs.shape, "Tensors must be of equal shape"
    if isinstance(lhs, BlockSparseTensor):
        if lhs.blocks is None:
            pass
        elif isinstance(rhs.blocks, Array) and isinstance(lhs.blocks, Array):
            if rhs.shape == lhs.shape and rhs.primal_dims == lhs.primal_dims and rhs.out_dims == lhs.out_dims:
                return BlockSparseTensor(rhs.out_dims, rhs.primal_dims, rhs.out_shape, rhs.primal_shape, rhs.blocks + lhs.blocks, rhs.sparse_dims)
        elif all(b1.shape == b2.shape for b1, b2 in zip(rhs.blocks, lhs.blocks)):
            pass
    elif isinstance(lhs, SparseTensor):
        pass
    elif isinstance(lhs, Array):
        rhs.dense() + lhs
    else:
        raise TypeError("Expected to add with type BlockSparseTensor, SparseTensor, or Array")

# @partial(jit, static_argnames=('lhs', 'rhs'))
def _mul(lhs, rhs):
    assert lhs.shape == rhs.shape, "Tensors must be of equal shape"
    if isinstance(rhs, BlockSparseTensor):
        if rhs.blocks is None:
            pass
        elif isinstance(lhs.blocks, Array) and isinstance(rhs.blocks, Array):
            if lhs.shape == rhs.shape and lhs.primal_dims == rhs.primal_dims and lhs.out_dims == rhs.out_dims:
                return BlockSparseTensor(lhs.out_dims, lhs.primal_dims, lhs.out_shape, lhs.primal_shape, lhs.blocks * rhs.blocks, lhs.sparse_dims)
        elif all(b1.shape == b2.shape for b1, b2 in zip(lhs.blocks, rhs.blocks)):
            pass
    elif isinstance(rhs, SparseTensor):
        pass
    elif isinstance(rhs, Array):
        lhs.dense() + rhs
    else:
        raise TypeError("Expected to add with type BlockSparseTensor, SparseTensor, or Array")

# @partial(jit, static_argnames=('lhs', 'rhs'))
def _matmul(lhs, rhs):
    #print("--- start matmul ---")
    # TODO assert something
    if isinstance(rhs, BlockSparseTensor):
        if lhs.blocks is None:
            #print("--- end matmul ---")
            return copy.copy(rhs)
        elif rhs.blocks is None:
            #print("--- end matmul ---")
            return copy.copy(lhs)
        elif isinstance(lhs.blocks, Array) and isinstance(rhs.blocks, Array) \
                and lhs.out_shape == rhs.primal_shape and rhs.sparse_dims == lhs.sparse_dims:
            out_dims = [
                d._replace(val_dim=i) 
                for i, d in enumerate(lhs.out_dims)
            ]
            
            primal_dims = [
                d._replace(
                    id=d.id - len(rhs.out_dims) + len(lhs.out_dims),
                    val_dim=i+len(lhs.out_dims)
                )
                for i, d in enumerate(rhs.primal_dims)
            ]
            
            #print("--- end matmul ---")
            return BlockSparseTensor(
                out_dims,
                primal_dims,
                lhs.out_shape,
                rhs.primal_shape,
                lax.dot_general(lhs.blocks, rhs.blocks, (([x.val_dim + lhs.sparse_dims for x in lhs.primal_dims], [x.val_dim + rhs.sparse_dims for x in rhs.out_dims]), (list(range(lhs.sparse_dims)),)*2)),
                lhs.sparse_dims
            )
    elif isinstance(rhs, SparseTensor):
        #print("--- end matmul ---")
        pass
    elif isinstance(rhs, Array):  # TODO: Fix default check
        block_nums = lhs.blocks.shape[:lhs.sparse_dims]
        block_idxs = tuple(range(lhs.sparse_dims))
        block_sizes = [d.block_size for d in lhs.primal_dims if isinstance(d, SparseDimension)]
        val_dims = tuple(d.val_dim+lhs.sparse_dims for d in lhs.primal_dims )
        
        #print(f"{lhs.shape=}, {rhs.shape=}")
        
        #print("--- reshape ---")
        #print(f"{block_nums} + {block_sizes}")
        #print(f"{lhs.sparse_dims=} -> {rhs.shape[lhs.sparse_dims:]}")
        #print(f"{len(lhs.primal_dims)=} -> {rhs.shape[-len(lhs.primal_dims):]}")
        rhs = rhs.reshape(*block_nums, *block_sizes, *rhs.shape[lhs.sparse_dims:]) #*rhs.shape[-len(lhs.primal_dims):])
        
        #print(f"{lhs.blocks.shape=}, {rhs.shape=}")
        #print(f"{val_dims=}, {block_nums=}")
        
        rhs_val_dims = tuple(i+lhs.sparse_dims for i in range(len(val_dims)))
        dim_nums = (
            (val_dims, rhs_val_dims),
            (block_idxs,)*2 # block_idxs repeated twice
        )
        
        #print(f"{dim_nums=}")

        res = lax.dot(lhs.blocks, rhs, dimension_numbers=dim_nums)
        
        #print("--- dot ---")
        #print(f"{lhs.out_shape=}")
        # print(f"{res.shape=}")
        # print(f"{lhs.sparse_dims=}")
        # print(lhs.out_shape, "+", rhs.shape[:len(lhs.primal_dims)])

        #print(res.reshape(*[x for i, x in enumerate(lhs.blocks.shape) if i not in val_dims]+*[x for i, x in enumerate(rhs.shape) if i-lhs.sparse_dims not in ...))
        #res = res.reshape(
        res = res.reshape(lhs.out_shape + rhs.shape[len(lhs.primal_dims)+lhs.sparse_dims:])
        #print(f"{res.shape=}")
        #print("--- end matmul ---")
        return res
    else:
        raise TypeError("Expected to matmul with type BlockSparseTensor, SparseTensor, or Array")

def _rmatmul(lhs, rhs):
    if isinstance(lhs, SparseTensor):
        pass
    elif isinstance(lhs, Array):
        block_nums = rhs.blocks.shape[:rhs.sparse_dims]
        block_idxs = tuple(range(rhs.sparse_dims))
        block_sizes = [d.block_size for d in rhs.primal_dims if isinstance(d, SparseDimension)]
        val_dims = [d.val_dim+rhs.sparse_dims for d in rhs.primal_dims if isinstance(d, SparseDimension)]

        # print(rhs.shape, lhs.shape)

        # note the -1 for arbitrary last dimension (*)
        lhs = lhs.reshape(*block_nums, *block_sizes, -1)
        transposed_axes = [rhs.primal_dims[d.other_id-len(rhs.out_dims)].val_dim for d in rhs.out_dims] \
                            + [rhs.out_dims[d.other_id].val_dim for d in rhs.primal_dims]
        # print(transposed_axes)
        rhs.rhs = rhs.blocks.transpose(0,*[x+1 for x in transposed_axes])

        # print(rhs.blocks.shape, lhs.shape)
        # print(val_dims, block_nums)

        dim_nums = (
            # both axis are the same for the two tensors
            # the -1 is for being one left of the last dimension (*)
            ([x-1 for x in val_dims],)*2,
            (block_idxs,)*2
        )

        res = lax.dot(rhs.blocks, lhs, dimension_numbers=dim_nums)
        
        #print(res.shape)
        return res.reshape(lhs.out_shape + rhs.shape[-lhs.sparse_dims:]).transpose(transposed_axes)
    else:
        raise TypeError("Expected to matmul SparseTensor or Array and BlockSparseTensor")
