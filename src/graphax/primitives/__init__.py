from .base import NO_EDGE, elemental_rules, elemental_only_rules, multi_output_elemental_only_rules

# Import submodules to trigger elemental rule registrations
# transforms must come before structural because structural imports _slice_elementals from it
from . import math
from . import linalg
from . import reductions
from . import transforms
from . import structural

# Auto-generated primitive rules (parallelism, advanced linalg, scatter/gather,
# reduce_window, dynamic slicing, etc.). Imported last so registrations for
# overlapping primitives override the manual ones above; non-overlapping
# primitives are added to the registry.
from . import auto

from .transforms import JacobianTransform
