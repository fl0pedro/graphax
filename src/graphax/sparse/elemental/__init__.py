"""Clean, self-contained elemental contraction/elementwise kernels.

Each module in this package owns ONE elemental operation between a specific
pair of :class:`~graphax.sparse.indexes.Index` component types, implemented
so that the result stays inside the closed ``{Dense, Diagonal}`` algebra
(the CLOSURE LAW) and is validated cell-for-cell against the dense oracle
(materialize operands, run the plain dense op).

Currently exposed:

* :func:`contract_B_B` — contract two meta-block-diagonal (``DiagonalIndex``)
  dims against each other (``B @ B``).
"""

from graphax.sparse.elemental.contract_B_B import contract_B_B
from graphax.sparse.elemental.dispatch import (
    try_elemental_elementwise,
    try_elemental_matmul,
)
__all__ = [
    "contract_B_B",
    "try_elemental_matmul",
    "try_elemental_elementwise",
]
