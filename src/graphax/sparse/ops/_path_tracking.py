"""Test-only fast-path tracker shared across ``ops.matmul`` and ``ops.elementwise``.

Each dispatcher calls ``record_path(name)`` at trace time. By default this
is a no-op — production code pays nothing. Tests opt in via either:

  * ``track_paths()`` context manager (preferred) — collects the path
    sequence into a list, scoped to the ``with`` block:

        with track_paths() as paths:
            res = matmul(a, b)
        assert paths[-1] == "aligned_pair"

  * ``TRACK_PATHS=1`` env var — globally enables tracking; the dispatcher
    writes to ``last_path`` and tests read it back. Useful for ad-hoc
    debugging when wrapping the call in a context manager is awkward.

Implementation: ``ContextVar`` so the tracker is per-task even under JAX
tracing / asyncio / threads. The default is ``None`` (no tracking); the
context manager temporarily binds a list and restores ``None`` on exit.
"""
from __future__ import annotations
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# Per-task list-of-recorded-paths, or None when tracking is disabled.
_tracker: ContextVar[list[str] | None] = ContextVar("path_tracker", default=None)

# Env var: if set to "1", the global ``last_path`` mirror is updated even
# without an active ``track_paths()`` context. Off by default.
TRACK_PATHS_ENV: bool = os.getenv("TRACK_PATHS", "0") == "1"

# Mirror of the most recently recorded path. Always written when an env-var
# tracker or context-manager tracker is active; ``None`` otherwise.
last_path: str | None = None


def record_path(name: str | None) -> None:
    """Record a fast-path name. ``None`` resets the most-recent slot —
    every ``matmul`` / ``elementwise`` call begins with ``record_path(None)``
    so a stale value can't leak between calls.

    Cost when no tracker is active: one ContextVar.get() and one global
    write. Both are constant-time and don't show up under JIT (they run
    at trace time, not in the JAXPR).
    """
    global last_path
    tracker = _tracker.get()
    if tracker is not None:
        if name is not None:
            tracker.append(name)
        last_path = name
    elif TRACK_PATHS_ENV:
        last_path = name


@contextmanager
def track_paths() -> Iterator[list[str]]:
    """Context-managed path collector.

    Inside the ``with`` block, every ``record_path(name)`` call appends
    ``name`` to the yielded list. ``None`` resets are not appended.
    Restores the previous tracker on exit (nesting is supported but the
    inner block sees only its own appends; outer-block appends pause
    while inner is active).

    Returns an empty list at entry; mutated in place by the dispatcher.
    """
    paths: list[str] = []
    token = _tracker.set(paths)
    try:
        yield paths
    finally:
        _tracker.reset(token)
