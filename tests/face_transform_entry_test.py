"""jacve's standalone ``face_transforms``: nothing is applied in silence.

``face_transforms`` is a mapping with TWO levels, ``{vertex: {face_key:
slots}}``. The thing a caller has in hand is :func:`graphax.faces_of`'s
output, which is the list of INNER keys, so the natural mistake is to build
the flat ``{face_key: slots}`` dict and hand that to :func:`jacve`. Every
lookup in the elimination is then ``face_transforms.get(vertex)``, no vertex
is ever a pair, and the whole request evaporates.

MEASURED, 2026-09-16, job 65975: twelve requested ``Diag`` hooks through the
standalone path produced ZERO hook calls and a bit-identical gradient, while
the env's own path (the same hooks handed to ``IncrementalJaxpr.eliminate``)
changed the measured plan by a factor of 1.75. The owner's rule is that
nothing skips silently, so both the flat form and a request that matches no
face raise now.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import SKIP_FACE, faces_of, jacve
from graphax.core import _inline_call_primitives
from graphax.incremental import IncrementalJaxpr


def _fun(a, b):
    c = jnp.sin(a) * b
    d = jnp.tanh(c) + a
    return jnp.sum(d * c)


A = jnp.asarray(np.linspace(0.1, 0.9, 6), jnp.float32)
B = jnp.asarray(np.linspace(0.2, 1.1, 6), jnp.float32)


def _order_and_keys():
    cj = jax.make_jaxpr(_fun)(A, B)
    jx, consts = _inline_call_primitives(cj.jaxpr, cj.literals)
    valid = [i for i, e in enumerate(jx.eqns, 1)
             if e.outvars[0] not in jx.outvars]
    order = sorted(valid, reverse=True)
    ij = IncrementalJaxpr(jx, (0, 1), list(consts), [A, B], track_faces=False)
    keys = {}
    for v in order:
        keys[v] = list(faces_of(ij.graph, ij.tgraph, int(v), jx))
        ij.eliminate(v, (), None)
    return order, keys


def _counting_hook(fired, tag):
    def hook(t):
        fired.append(tag)
        return t
    return hook


def test_the_nested_form_reaches_every_face():
    order, keys = _order_and_keys()
    fired: list = []
    ft = {v: {k: (_counting_hook(fired, (v, k)), None, None) for k in ks}
          for v, ks in keys.items() if ks}
    jacve(_fun, order, argnums=(0, 1), face_transforms=ft)(A, B)
    assert len(fired) == sum(len(x) for x in ft.values())
    assert set(fired) == {(v, k) for v, ks in ft.items() for k in ks}


def test_the_flat_form_raises_instead_of_doing_nothing():
    """THE DEFECT. `faces_of` returns the inner keys, so a caller naturally
    builds the flat dict; it matched no vertex and every hook was dropped."""
    order, keys = _order_and_keys()
    fired: list = []
    flat = {k: (_counting_hook(fired, k), None, None)
            for ks in keys.values() for k in ks}
    with pytest.raises(ValueError, match="keyed by a FACE KEY"):
        jacve(_fun, order, argnums=(0, 1), face_transforms=flat)(A, B)
    assert not fired


def test_a_vertex_that_is_not_in_the_order_raises():
    order, _ = _order_and_keys()
    with pytest.raises(ValueError, match="never applied"):
        jacve(_fun, order, argnums=(0, 1),
              face_transforms={9999: {(0, 1): (lambda t: t, None, None)}})(A, B)


def test_a_face_key_from_the_wrong_graph_state_raises():
    """Every elimination rewires the graph, so keys enumerated too early name
    faces that no longer exist."""
    order, _ = _order_and_keys()
    with pytest.raises(ValueError, match="never applied"):
        jacve(_fun, order, argnums=(0, 1),
              face_transforms={order[0]: {(77, 88): (lambda t: t, None,
                                                     None)}})(A, B)


def test_a_vertex_mapped_to_slots_instead_of_faces_raises():
    order, keys = _order_and_keys()
    v = order[0]
    with pytest.raises(ValueError, match="not a .*dict"):
        jacve(_fun, order, argnums=(0, 1),
              face_transforms={v: (lambda t: t, None, None)})(A, B)


def test_a_vertex_mapped_to_the_skip_sentinel_raises():
    """SKIP_FACE is a FACE value, not a vertex value."""
    order, _ = _order_and_keys()
    with pytest.raises(ValueError, match="not a .*dict"):
        jacve(_fun, order, argnums=(0, 1),
              face_transforms={order[0]: SKIP_FACE})(A, B)


def test_a_partial_miss_is_legal():
    """`faces_of` is a documented SUPERSET of the faces the elimination
    visits, so a request that lands on some of a vertex's faces and not all
    of them must NOT raise."""
    order, keys = _order_and_keys()
    v = next(v for v, ks in keys.items() if len(ks) >= 2)
    fired: list = []
    ft = {v: {keys[v][0]: (_counting_hook(fired, "live"), None, None),
              (4242, 4343): (_counting_hook(fired, "dead"), None, None)}}
    jacve(_fun, order, argnums=(0, 1), face_transforms=ft)(A, B)
    assert fired == ["live"]


def test_skip_face_counts_as_applied():
    order, keys = _order_and_keys()
    v = next(v for v, ks in keys.items() if ks)
    g = jacve(_fun, order, argnums=(0, 1),
              face_transforms={v: {keys[v][0]: SKIP_FACE}})(A, B)
    base = jacve(_fun, order, argnums=(0, 1))(A, B)
    flat_g = np.concatenate([np.asarray(x, np.float64).ravel()
                             for x in jax.tree_util.tree_leaves(g)])
    flat_b = np.concatenate([np.asarray(x, np.float64).ravel()
                             for x in jax.tree_util.tree_leaves(base)])
    assert not np.allclose(flat_g, flat_b), "the skip did not land"


def test_the_per_vertex_dict_form_raises_on_a_total_miss():
    """The per-vertex ``transforms`` dict is keyed by the ELIMINATOR's own
    equation-position pair, not by `faces_of`'s stable var index. Handing it
    `faces_of`'s keys matched nothing, and that was silent too."""
    order, keys = _order_and_keys()
    fired: list = []
    tr = [(v, {k: (_counting_hook(fired, (v, k)), None, None) for k in ks})
          for v, ks in keys.items() if ks]
    with pytest.raises(ValueError, match="never applied"):
        jacve(_fun, order, argnums=(0, 1), transforms=tr)(A, B)
    assert not fired


def test_the_exact_path_is_untouched():
    """No face transforms, no change: the validator must not move a value."""
    order, _ = _order_and_keys()
    g = jacve(_fun, order, argnums=(0, 1))(A, B)
    ref = jax.grad(_fun, argnums=(0, 1))(A, B)
    for a, e in zip(jax.tree_util.tree_leaves(g),
                    jax.tree_util.tree_leaves(ref)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(e),
                                   rtol=1e-6, atol=1e-6)
