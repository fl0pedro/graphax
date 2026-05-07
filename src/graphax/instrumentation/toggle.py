"""Environment-variable toggle for graphax's Jacobian-structure instrumentation.

Reads ``GRAPHAX_JACOBIAN_INSTRUMENTATION`` lazily on every call to ``is_enabled()``
so tests and notebook sessions can flip the toggle on the fly without reimporting.

Importantly, the rest of graphax's hot path (``jacve``, ``vertex_elimination_jaxpr``,
``extract_jaxpr``) is *never* aware of this toggle. The instrumentation modules
are entirely additive — consumers opt in by calling ``extract_jacobian_features``
or one of the lower-level helpers, and that function is the only place the
toggle is consulted. When the toggle is off it returns ``None`` immediately,
which means there is no JAX trace, no XLA compilation, and no jaxpr mutation
involved on the off-path.
"""

from __future__ import annotations

import os

ENV_VAR_NAME: str = "GRAPHAX_JACOBIAN_INSTRUMENTATION"
_TRUE_LITERALS = frozenset({"1", "true", "on", "yes", "y", "t"})


def is_enabled() -> bool:
    """True iff the env var is set to one of `1 / true / on / yes / y / t`
    (case-insensitive). Anything else (including unset) → False."""
    value = os.environ.get(ENV_VAR_NAME, "")
    return value.strip().lower() in _TRUE_LITERALS
