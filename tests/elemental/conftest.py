"""Shared fixtures for the elemental kernel suites.

These suites compile MANY distinct jaxprs (one per parametrized shape). Without
clearing JAX's caches between tests, the XLA executable + compilation cache
accumulates across the whole file and OOMs a small machine — which is why the
heavy suites (``test_contract_multi``, ``test_contract_D_multiB``) previously ran
only on the cluster. Clearing after each test bounds peak memory to roughly a
single test's footprint, so the suites run on a laptop too.
"""

import gc

import jax
import pytest


@pytest.fixture(autouse=True)
def _clear_jax_caches():
    yield
    jax.clear_caches()
    gc.collect()
