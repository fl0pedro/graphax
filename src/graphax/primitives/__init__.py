from .base import (
    NO_EDGE,
    elemental_rules,
    elemental_only_rules,
    multi_output_elemental_only_rules,
    jit_name_rules,
)

# Import submodules to trigger elemental rule registrations. Each module owns a
# DISJOINT set of primitives, so the listing order does not affect which rule
# wins (no overwrites). Cross-module imports (e.g. indexing -> transforms for
# JacobianTransform) are resolved by Python's import machinery on demand and do
# NOT depend on this order — there are no import cycles among these modules.
from . import math
from . import linalg
from . import reductions
from . import transforms
from . import indexing
from . import conv
from . import passthrough

# User-supplied-derivative honoring: custom_vjp/custom_jvp call rules + named-jit
# (jax.nn activation) Jacobians dispatched by jit_p params['name'].
from . import custom
from .custom import jit_named_elemental_only

from .transforms import JacobianTransform
