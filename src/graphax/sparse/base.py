"""
Sparse base implementation
"""
import copy
from typing import Callable, Sequence, Generator, Any, Generic, TypeVar
from abc import ABC, abstractmethod
import jax
import jax.lax as lax
import jax.numpy as jnp

from jax._src.core import ShapedArray
from jax.tree_util import register_pytree_node_class

from chex import Array

from .utils import eye_like_copy, eye_like
from dataclasses import dataclass

T = TypeVar("T")

# NOTE: a val_dim of None means that we have a possible replication of the tensor
#   along the respective dimension `d.size` times to manage broadcasting
#   operations such as broadcasted additions or multiplications.
# TODO: what do we do when we have a tensor that consists only of DenseDimensions
#   with val_dim=None?
@dataclass
class DenseDimension:
    id: int
    size: int
    val_dim: int | None


# NOTE: a val_dim of None means that we have a factored Kronecker delta in
#   our tensor at the respective dimensions.
#   Also we can have unmatching size and val.shape[d.val_dim] for SparseDimensions
#   if the size is 1. This is necessary to enable broadcasting operations.
@dataclass
class SparseDimension:
    id: int
    size: int
    val_dim: int
    other_id: int


Dimension = DenseDimension | SparseDimension


class SparseBase(ABC, Generic[T]):
    """
    The `SparseTensor object enables` the representation of sparse tensors
    that if out_dims or primal_dims is empty, this implies a scalar dependent or
    independent variable. if both are empty, then we have a scalar value and
    everything becomes trivial and the `val` field contains the value of the
    singleton partial
    """
    out_dims: tuple[Dimension]
    primal_dims: tuple[Dimension]  # input dimensions
    shape: tuple[int]  # True shape of the tensor
    val: T
    pre_transforms: tuple[Callable]
    post_transforms: tuple[Callable]

    # NOTE: Document pre_transforms and post_transforms. what about addition?
    # NOTE: We always assume that the dimensions are ordered in ascending order

    def __init__(self,
                 out_dims: Sequence[Dimension],
                 primal_dims: Sequence[Dimension],
                 val: T,
                 pre_transforms: Sequence[Callable] = None,
                 post_transforms: Sequence[Callable] = None) -> None:

        if pre_transforms is None:
            pre_transforms = []
        if post_transforms is None:
            post_transforms = []

        self.out_dims = out_dims if isinstance(out_dims, tuple) else tuple(out_dims)
        self.primal_dims = primal_dims if isinstance(primal_dims, tuple) else tuple(primal_dims)

        self.out_shape = [d.size for d in out_dims]
        self.primal_shape = [d.size for d in primal_dims]

        self.shape = tuple(self.out_shape + self.primal_shape)

        self.val = val

        self.pre_transforms = tuple(pre_transforms)
        self.post_transforms = tuple(post_transforms)

        self._assert_sparse_tensor_consistency()

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

        return f"""SparseTensor(
    shape = ({str_out_shape} | {str_primal_shape}),
    out_dims = {multiline_out_dims},
    primal_dims = {multiline_primal_dims},
    val = {self.val},
    pre_transforms = {multiline_pre_transform},
    post_transforms = {multiline_post_transform}
)"""

    @abstractmethod
    def __add__(self, _tensor: Any):
        ...

    @abstractmethod
    def __mul__(self, _tensor: Any):
        ...

    @abstractmethod
    def dense(self, iota: Array) -> Array:
        ...

    @abstractmethod
    def copy(self):
        ...

    def _assert_value_consistency(self) -> None:
        self._assert_value_consistency(self)

    @abstractmethod
    @staticmethod
    def _assert_value_consistency(s: Any) -> None:
        if not issubclass(s, SparseBase):
            raise TypeError("Not a Sparse object")
        ...

    def _assert_sparse_tensor_consistency(self) -> None:
        self._assert_sparse_tensor_consistency(self)

    @staticmethod
    def _assert_sparse_tensor_consistency(s: Any) -> None:
        """
        Function that validates the consistency of a `SparseTensor` object,
        i.e. checks if the `val` property has the correct shape and if the dimensions
        are ordered correctly and sizes match the shape of `val`.

        Args:
            s (SparseTensor): SparseTensor object we want to validate.

        Returns:
            bool: True if the `SparseTensor` object is consistent.
        """

        if not issubclass(s, SparseBase):
            raise TypeError("Not a Sparse object")

        # Check if d.size matches val.shape[d.val_dim] for all d
        matching_sparse_sizes = all(
            d.size == s.val.shape[d.val_dim]
            or d.size == 1
            if isinstance(d, SparseDimension)
               and d.val_dim is not None else True
            for d in s.out_dims + s.primal_dims
        )

        matching_dense_sizes = all(
            d.size == s.val.shape[d.val_dim]
            if isinstance(d, DenseDimension)
               and d.val_dim is not None else True
            for d in s.out_dims + s.primal_dims
        )

        matching_sizes = matching_sparse_sizes or matching_dense_sizes

        unique_out_dims = [d.val_dim for d in s.out_dims if d.val_dim is not None]
        unique_primal_dims = [d.val_dim for d in s.primal_dims if d.val_dim is not None]

        is_uniqe_out_dims = len(unique_out_dims) == len(set(unique_out_dims))
        is_uniqe_primal_dims = len(unique_primal_dims) == len(set(unique_primal_dims))
        has_uniqe_dims = is_uniqe_out_dims and is_uniqe_primal_dims

        # Check if IDs in out_dims and primal_dims match their index positions
        matching_id = all(
            od.id == i and pd.id == i + len(s.out_dims)
            for i, (od, pd) in enumerate(zip(s.out_dims, s.primal_dims))
        )

        # Check sparse dimension pairing consistency
        matching_sparse_ids = all(
            s.primal_dims[d.other_id - len(s.out_dims)].other_id == d.id
            if isinstance(d, SparseDimension) else True
            for d in s.out_dims
        )

        assert (matching_sizes
                and has_uniqe_dims
                and matching_id
                and matching_sparse_ids
                ), f"{s} is not self-consistent!"

        s._assert_value_consistency()