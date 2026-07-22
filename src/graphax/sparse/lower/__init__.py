"""Structure-lowering layer (``GRAPHAX_STRUCT_LOWER``, default OFF).

Given ``(op, lhs.dims, rhs.dims)`` the lower_* engines COMPILE the minimal
physical computation and construct the output structure SYMBOLICALLY.
Hooked at the top of the elementwise / matmul dispatchers; a case without a
rule falls through to the existing path UNCHANGED and is counted — no silent
behavior change.

Imports only — the engines live in their own modules (``add`` / ``matmul``).
"""
from . import add  # noqa: F401
from . import matmul  # noqa: F401
