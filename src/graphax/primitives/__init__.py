from .base import (
    NO_EDGE,
    elemental_rules,
    elemental_only_rules,
    multi_output_elemental_only_rules,
    jit_name_rules,
)

# Import submodules to trigger elemental rule registrations. These are all
# non-overlapping primitives, so any order after `base` is fine — except that
# transforms must come before indexing (which imports JacobianTransform from it)
# and before passthrough (which imports _slice_elementals from it).
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
