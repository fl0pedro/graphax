import sys

from .core import (
    faces_of,
    grad,
    jacve,
    set_jit_fallback_order,
    set_pjit_elimination_order,
    value_and_grad,
)

from .sparse import sparse_tensor_zeros_like
from .utils import tree_allclose

# Primary jaxpr tokenizer (append-only, preserved-trace).
from .jaxpr import IncrementalPathTokenizer
# ``IncrementalJaxpr`` is the current name; ``IncrementalJacobian`` is kept as a
# backward-compat alias (alphagrad imports the old name).
from .incremental import IncrementalJacobian, IncrementalJaxpr

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
