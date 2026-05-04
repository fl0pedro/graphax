"""Feature pin: `extract_jaxpr` builds a `VEJaxpr` and the topology cache hits.

`extract_jaxpr` JIT-traces the entire vertex elimination process and wraps
the resulting jaxpr as a `VEJaxpr` for downstream consumers (e.g. alphagrad).
The topology cache memoizes by (jaxpr, argnums, order, sparsity_map,
sparse_representation), so repeated calls with the same plan return the
same cached object.
"""

import jax
import jax.numpy as jnp

from graphax.core import _topology_cache, extract_jaxpr
from graphax.jaxpr import VEJaxpr


def _make_closed_jaxpr():
    def f(x, y):
        return jnp.sin(x * y).sum()

    args = (jnp.ones(4), jnp.ones(4))
    closed = jax.make_jaxpr(f)(*args)
    return closed, args


def test_extract_jaxpr_returns_vejaxpr():
    closed, args = _make_closed_jaxpr()
    ve = extract_jaxpr(
        closed.jaxpr,
        argnums=(0, 1),
        order="rev",
        sparse_representation=False,
        args=args,
        consts=closed.literals,
    )
    assert isinstance(ve, VEJaxpr)
    assert ve.jaxpr is not None


def test_topology_cache_hits_on_repeat():
    """Same plan -> same VEJaxpr object (cache hit)."""
    _topology_cache.clear()
    closed, args = _make_closed_jaxpr()
    kw = dict(
        argnums=(0, 1),
        order="rev",
        sparse_representation=False,
        args=args,
        consts=closed.literals,
    )
    ve1 = extract_jaxpr(closed.jaxpr, **kw)
    ve2 = extract_jaxpr(closed.jaxpr, **kw)
    assert ve1 is ve2, "topology cache should return the same VEJaxpr instance"


def test_topology_cache_distinguishes_orders():
    """Different orders -> different VEJaxprs."""
    _topology_cache.clear()
    closed, args = _make_closed_jaxpr()
    base = dict(
        argnums=(0, 1),
        sparse_representation=False,
        args=args,
        consts=closed.literals,
    )
    ve_rev = extract_jaxpr(closed.jaxpr, order="rev", **base)
    ve_fwd = extract_jaxpr(closed.jaxpr, order="fwd", **base)
    assert ve_rev is not ve_fwd


def test_extract_jaxpr_supports_sparsity_map():
    """sparsity_map plumbed into the cache key (no rules == empty tuple)."""
    _topology_cache.clear()
    closed, args = _make_closed_jaxpr()
    base = dict(
        argnums=(0, 1),
        order="rev",
        sparse_representation=False,
        args=args,
        consts=closed.literals,
    )
    ve = extract_jaxpr(closed.jaxpr, **base, sparsity_map=())
    assert isinstance(ve, VEJaxpr)
