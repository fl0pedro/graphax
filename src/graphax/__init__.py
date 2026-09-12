import sys

from .core import (
    SKIP_FACE,
    FaceContraction,
    FaceOperands,
    FaceSpec,
    contract_face,
    contract_face_operands,
    face_config_is_approx,
    face_specs_of,
    prepare_face_operands,
    faces_of,
    grad,
    jacve,
    set_jit_fallback_order,
    set_pjit_elimination_order,
    value_and_grad,
    inline_call_primitives,
)

from .dense_edges import (
    ActionCensus,
    CensusMismatch,
    DenseBudgetExceeded,
    census_plan,
    compare_censuses,
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
