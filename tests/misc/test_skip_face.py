"""SKIP-face semantics: a skipped path's contraction is NOT performed.

The spec's cheapest approximation — ``face_transforms[key] = SKIP_FACE`` drops
that path's contribution entirely (an absent addend, not a zero edge), leaves
every other face bit-identical to the exact path, and renders as an
``approx SKIP`` block in the incremental token stream.
"""
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from graphax import SKIP_FACE, IncrementalPathTokenizer, faces_of, jacve
from graphax.incremental import IncrementalJacobian
from graphax.jaxpr import get_vocab

_M = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0 + 0.1)
_P = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 13.0 + 0.2)
_Q = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 17.0 + 0.3)
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))


def _square(x):
    """One in-edge, two out-edges -> vertex 1 has two faces."""
    e = _M @ x
    return _P @ e, _Q @ e


def _build(fn, args, argnums):
    closed = jax.make_jaxpr(fn)(*args)
    ij = IncrementalJacobian(closed.jaxpr, argnums, list(closed.literals),
                             list(args), track_faces=True)
    return closed.jaxpr, ij


def _run(fn, args, argnums, face_transforms=None, vertex=1):
    jaxpr, ij = _build(fn, args, argnums)
    for v in range(1, len(jaxpr.eqns) + 1):
        if v != vertex:
            ij.eliminate(v)
            continue
        ft = (face_transforms(ij, jaxpr) if callable(face_transforms)
              else face_transforms)
        ij.eliminate(v, (), ft)
    return jaxpr, ij


def _eval(ij, args):
    jaxpr, consts, _ = ij.current_jaxpr()
    return [np.asarray(v) for v in jax.core.eval_jaxpr(jaxpr, consts, *args)]


def _keys_and_order(fn, args, argnums, vertex):
    """Face keys of ``vertex`` on the INITIAL graph + a full order starting at
    it (the keys are only valid at the vertex's own elimination time)."""
    closed = jax.make_jaxpr(fn)(*args)
    ij = IncrementalJacobian(closed.jaxpr, argnums, list(closed.literals),
                             list(args))
    keys = faces_of(ij.graph, ij.tgraph, vertex, closed.jaxpr)
    nv = len(closed.jaxpr.eqns)
    order = [vertex] + [v for v in range(1, nv + 1) if v != vertex]
    return keys, order


def test_skip_drops_only_that_faces_contribution():
    """Pins the MEASUREMENT path (jacve — what the env compiles/executes)."""
    keys, order = _keys_and_order(_square, (_X4,), (0,), vertex=1)
    assert len(keys) >= 2, "fixture needs a multi-face vertex"

    exact = jtu.tree_leaves(jacve(_square, order, argnums=(0,))(_X4))
    out = jtu.tree_leaves(
        jacve(_square, order, argnums=(0,),
              face_transforms={1: {keys[0]: SKIP_FACE}})(_X4)
    )

    assert len(out) == len(exact), "skip must not change the output STRUCTURE"
    diffs = [not np.array_equal(np.asarray(a), np.asarray(b))
             for a, b in zip(out, exact)]
    assert any(diffs), "skipping a face must change the accumulated Jacobian"
    assert not all(diffs), "untargeted faces must stay bit-identical"


def test_skip_is_recorded_and_tokenized():
    closed = jax.make_jaxpr(_square)(_X4)
    tk = IncrementalPathTokenizer(closed.jaxpr, (0,), list(closed.literals),
                                  [_X4], vocab_size=248)
    stream = list(tk.base_tokens())
    keys = faces_of(tk.ij.graph, tk.ij.tgraph, 1, closed.jaxpr)
    delta = list(tk.eliminate(1, (), {keys[0]: SKIP_FACE}))
    stream += delta

    vocab, _, _ = get_vocab()
    assert vocab["SKIP"] in delta, "the skipped face must render `approx SKIP`"
    # the face record carries the SKIP entry with an EMPTY eqn range
    skip_entries = [a for fr in tk.ij.step_faces(0) for a in fr.approx
                    if a[0] == "SKIP"]
    assert len(skip_entries) == 1
    _atype, _params, s, e = skip_entries[0]
    assert s == e


def test_skip_all_faces_of_a_vertex_zeroes_the_jacobian():
    """Every path through vertex 1 skipped ⇒ no contribution reaches any
    output: the measured Jacobian is all-zeros but keeps its structure."""
    keys, order = _keys_and_order(_square, (_X4,), (0,), vertex=1)
    out = jtu.tree_leaves(
        jacve(_square, order, argnums=(0,),
              face_transforms={1: {k: SKIP_FACE for k in keys}})(_X4)
    )
    exact = jtu.tree_leaves(jacve(_square, order, argnums=(0,))(_X4))
    assert len(out) == len(exact)
    for leaf, ref in zip(out, exact):
        assert np.asarray(leaf).shape == np.asarray(ref).shape
        assert not np.any(np.asarray(leaf)), "all paths skipped ⇒ zeros"
