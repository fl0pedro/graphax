"""Pin: toggling ``GRAPHAX_JACOBIAN_INSTRUMENTATION`` must not change graphax's
compiled artefacts in any way.

The instrumentation subpackage is *purely additive*: importing it registers
default rules into ``STRUCTURE_RULES``, but no graphax core code path consults
that registry. So ``jacve(...)`` should produce a byte-identical jaxpr — and
ultimately byte-identical XLA HLO — regardless of:

* whether ``graphax.instrumentation`` was imported, and
* whether the env-var toggle is on or off.

These two checks together rule out the most common ways an "additive" feature
silently changes compilation downstream.
"""

from __future__ import annotations

import importlib
import os

import jax
import jax.numpy as jnp


def _target_fn(x):
    """Mix of elementwise and contraction ops so the jacve trace has both
    primitives that *do* have analytical structure rules and primitives that
    don't."""
    a = jnp.tanh(x)             # eqn 1
    b = jnp.exp(a)              # eqn 2
    return jnp.sum(a + b)       # eqns 3 (add) + 4 (reduce_sum)


def _trace_jacve_jaxpr():
    import graphax  # local import so each test sees the current import order
    x = jnp.array([0.1, 0.2, 0.3, 0.4])
    grad_fn = graphax.jacve(_target_fn, order=[1, 2, 3, 4], argnums=(0,))
    return jax.make_jaxpr(grad_fn)(x)


def _compile_jacve_cost():
    """Compile via XLA and return the cost-analysis dict.

    Cost analysis (flops, bytes, …) is a deterministic function of the HLO's
    *semantics* — it ignores source-location metadata and other non-deterministic
    debug info that JAX tracing emits between repeated calls. So if the toggle
    is truly not affecting compilation, this dict must be bit-for-bit equal
    on/off.
    """
    import graphax
    x = jnp.array([0.1, 0.2, 0.3, 0.4])
    grad_fn = graphax.jacve(_target_fn, order=[1, 2, 3, 4], argnums=(0,))
    compiled = jax.jit(grad_fn).lower(x).compile()
    return compiled.cost_analysis() or {}


def test_toggle_off_jaxpr_unchanged():
    """Toggle off: jaxpr from `jacve` is identical with or without instrumentation imported."""
    print("\n[toggle] jaxpr identical w/ vs w/o instrumentation imported (toggle off)")
    os.environ.pop("GRAPHAX_JACOBIAN_INSTRUMENTATION", None)
    base_jaxpr = _trace_jacve_jaxpr()
    # Force a re-import to make sure any side-effects from instrumentation
    # have already happened, then capture again.
    import graphax.instrumentation  # noqa: F401
    after_jaxpr = _trace_jacve_jaxpr()
    assert str(base_jaxpr) == str(after_jaxpr), (
        "graphax.jacve produced a different jaxpr after instrumentation was "
        "imported, which means the subpackage has a side-effect on the core "
        "compilation path."
    )
    print(f"  ok: jaxpr length {len(str(base_jaxpr))} bytes, identical.")


def test_toggle_on_jaxpr_unchanged():
    """Toggle on: jaxpr is *still* identical — turning instrumentation on does
    not affect the core path; it just enables an additive pass that consumers
    have to opt into via `extract_jacobian_features`."""
    print("\n[toggle] jaxpr identical with toggle on vs off")
    os.environ.pop("GRAPHAX_JACOBIAN_INSTRUMENTATION", None)
    off_jaxpr = _trace_jacve_jaxpr()
    os.environ["GRAPHAX_JACOBIAN_INSTRUMENTATION"] = "1"
    on_jaxpr = _trace_jacve_jaxpr()
    os.environ.pop("GRAPHAX_JACOBIAN_INSTRUMENTATION", None)
    assert str(off_jaxpr) == str(on_jaxpr)
    print(f"  ok: jaxpr identical with toggle off vs on.")


def test_toggle_off_xla_cost_unchanged():
    """Strongest reproducible check: XLA's cost analysis (flops, bytes
    accessed, …) of the compiled `jacve` is *bit-for-bit identical* with the
    toggle on vs off. Cost analysis is a deterministic function of the HLO's
    semantics and doesn't depend on source-location metadata, so any
    leakage from the instrumentation into the XLA program would show up here.
    """
    print("\n[toggle] compiled XLA cost analysis identical with toggle on vs off")
    os.environ.pop("GRAPHAX_JACOBIAN_INSTRUMENTATION", None)
    off_cost = _compile_jacve_cost()
    os.environ["GRAPHAX_JACOBIAN_INSTRUMENTATION"] = "1"
    on_cost = _compile_jacve_cost()
    os.environ.pop("GRAPHAX_JACOBIAN_INSTRUMENTATION", None)
    assert off_cost == on_cost, (
        f"XLA cost analysis differs between toggle states:\n"
        f"  off = {off_cost}\n  on  = {on_cost}\n"
        "the instrumentation is leaking into the XLA program."
    )
    print(f"  ok: cost analysis {dict(off_cost)} identical.")


def test_extract_returns_none_when_off():
    """Behavioural check: with toggle off, `extract_jacobian_features` must
    return ``None`` regardless of inputs (it should never even *start* doing
    work)."""
    print("\n[toggle] extract_jacobian_features returns None when off")
    os.environ.pop("GRAPHAX_JACOBIAN_INSTRUMENTATION", None)
    import graphax
    x = jnp.array([0.1, 0.2, 0.3])
    closed = jax.make_jaxpr(jnp.tanh)(x)
    res = graphax.instrumentation.extract_jacobian_features(
        closed.jaxpr, tuple(closed.literals), (x,),
    )
    assert res is None
    print("  ok: extract_jacobian_features → None")


def test_extract_returns_features_when_on():
    """Behavioural check: with toggle on, `extract_jacobian_features` returns
    a ``JacobianFeatures`` record with one entry per eqn."""
    print("\n[toggle] extract_jacobian_features returns features when on")
    os.environ["GRAPHAX_JACOBIAN_INSTRUMENTATION"] = "1"
    try:
        import graphax
        x = jnp.array([0.1, 0.2, 0.3])
        closed = jax.make_jaxpr(jnp.tanh)(x)
        res = graphax.instrumentation.extract_jacobian_features(
            closed.jaxpr, tuple(closed.literals), (x,),
        )
        assert res is not None
        n = len(closed.jaxpr.eqns)
        assert res.frob_sq.shape == (n,)
        assert res.off_diag_ratio.shape == (n,)
        assert len(res.source) == n
        print(f"  ok: {n} eqns, sources = {res.source}")
    finally:
        os.environ.pop("GRAPHAX_JACOBIAN_INSTRUMENTATION", None)


def main():
    print("=== graphax.instrumentation toggle no-op tests ===")
    test_toggle_off_jaxpr_unchanged()
    test_toggle_on_jaxpr_unchanged()
    test_toggle_off_xla_cost_unchanged()
    test_extract_returns_none_when_off()
    test_extract_returns_features_when_on()
    print("\nALL TOGGLE TESTS OK")


if __name__ == "__main__":
    main()
