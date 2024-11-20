from typing import Sequence
from chex import Array
from .tensor import SparseTensor, SparseDimension, DenseDimension, Dimension
import jax.numpy as jnp


class KroneckerDelta:
    i: int
    j: int
    n: int
    def __init__(self, i: int, n: int | range, j: int):
        """
        A mathematical form to better define and understand sparse tensors.

        \(\delta_{ij}, i, j\in [n]\)


        Parameters:
            i: id representing variable i, this should be the same as the axis id of the value tensor
            j: id representing variable j, this should be the same as the axis id of the value tensor
            n: range (1, ..., n) for variable i
        """
        self.i = i
        self.n = n
        self.j = j

    def __mul__(self, other):
        return self._mul(other)

    def __rmul__(self, other):
        return self._mul(other)

    def _mul(self, other):
        """
        Multiplies two Kronecker deltas or a Kronecker delta with an array.
        Note that the resulting sparse tensor will always be of a shape
        where the prefix is the shape of the value array. So unfortunately
        there will always have to be a final operation with
        transpose(res, axis=...) to get a mathematically correct form.

        Parameters:
            other

        Returns:

        """
        if isinstance(other, KroneckerDelta):
            SparseTensor(out_dims=(SparseDimension(id=0, size=self.n, val_dim=self.i, other_id=2),
                                   SparseDimension(id=1, size=other.n, val_dim=other.i,other_id=3)),
                         primal_dims=(SparseDimension(id=2, size=self.n, val_dim=self.j, other_id=0),
                                      SparseDimension(id=3, size=other.n, val_dim=other.j, other_id=1)),
                         val=jnp.ones((self.n,other.n)))
        elif isinstance(other, Array):
            assert other.shape[self.i] == self.n
            SparseTensor(out_dims=(*(DenseDimension(id=i, size=other.shape[i], val_dim=i)
                                     for i in range(len(other))
                                     if i != self.i and i != self.j),
                                   SparseDimension(id=len(other), size=self.n, val_dim=self.i, other_id=2)),
                         primal_dims=(SparseDimension(id=len(other)+1, size=self.n, val_dim=self.j, other_id=0),),
                         val=other)
        elif isinstance(other, SparseTensor):
            ...
        else:
            raise TypeError("")

    def to_tensor(self):


def transpose(st: SparseTensor, axis=None):
    ...