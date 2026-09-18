"""THE ENUMERATOR AND THE ELIMINATOR SEE ONE GRAPH (ticket dsnn-dfw.25).

A policy enumerates its face keys on :class:`graphax.incremental.IncrementalJaxpr`
and the measurement contracts them on the graph ``_build_graph`` makes for
``_get_eliminator``. A key the elimination never enumerates configures nothing,
so the two builders must list the SAME faces for the SAME vertex -- not only on
the initial graph but at every position of the order, because every elimination
rewires what is left.

``_prune_graph`` is the one thing that can separate them. It is gated by
``GRAPHAX_PRUNE`` for the eliminator (:func:`graphax.core.prune_enabled`), and
the incremental builder used to run it unconditionally: with ``GRAPHAX_PRUNE=0``
the enumerator then dropped the dead ``log_softmax`` stabiliser chain and its
neighbours' faces while the elimination kept them, and the divergence GREW along
the order (measured on the two-copy SNN window: 9 vertices at the start, 36 of 89
positions over a reverse order).
"""

import os
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import graphax.core as gxcore
from graphax import faces_of, inline_call_primitives
from graphax.core import (_build_graph, _eliminate_vertex, _prune_graph,
                          prune_enabled)
from graphax.examples import (RSNN_SHD, RSNN_SHD_W2, Helmholtz, LongChain,
                              Perceptron, RoeFlux_1d, Simple)
from graphax.incremental import IncrementalJaxpr

_N_IN, _H, _N_OUT = 4, 3, 2


def _arr(*shape, lo=0.1, hi=0.9):
    n = int(np.prod(shape)) if shape else 1
    return jnp.asarray(
        np.linspace(lo, hi, n, dtype=np.float32).reshape(shape))


def _rsnn_head():
    """``x, y, S, I, U, a, Uo, W, V, Wo`` plus the six constants, small."""
    return (_arr(_N_IN), _arr(_N_OUT), _arr(_H), _arr(_H), _arr(_H), _arr(_H),
            _arr(_N_OUT), _arr(_H, _N_IN), _arr(_H, _H), _arr(_N_OUT, _H),
            jnp.float32(0.37), jnp.float32(0.6), jnp.float32(0.6),
            jnp.float32(0.95), jnp.float32(1.0), jnp.float32(1.0))


def _rsnn_carry():
    """The eleven carried blocks of the rtrl rule, plus the three reference
    weights, at the exact container's shapes."""
    refs = (_arr(_H, _N_IN), _arr(_H, _H), _arr(_N_OUT, _H))
    hidden = tuple(
        b for _ in range(4)
        for b in (_arr(_H, _H, _N_IN), _arr(_H, _H, _H)))
    readout = (_arr(_N_OUT, _H, _N_IN), _arr(_N_OUT, _H, _H),
               _arr(_N_OUT, _N_OUT, _H))
    return refs + hidden + readout


def _rsnn_adjoints():
    return (_arr(_H), _arr(_H), _arr(_H), _arr(_H), _arr(_N_OUT))


#: ``(name, fn, args, argnums)``. Every registered family the search runs on
#: that fits in a unit test, plus all four temporal rules of the SNN target --
#: those are the ones whose ``log_softmax`` leaves a dead stabiliser chain.
TARGETS = [
    ("Simple", Simple, (_arr(3), _arr(3)), (0, 1)),
    ("Helmholtz", Helmholtz,
     (jnp.asarray(np.array([0.05, 0.1, 0.15, 0.2], dtype=np.float32)),), (0,)),
    ("LongChain", LongChain, (_arr(3), _arr(3), _arr(3)), (0, 1, 2)),
    ("Perceptron", Perceptron,
     (_arr(4), _arr(2), _arr(4, 3), _arr(3), _arr(3, 2), _arr(2),
      _arr(3), _arr(3)), (2, 3, 4, 5)),
    ("RoeFlux_1d", RoeFlux_1d,
     tuple(jnp.float32(v) for v in (0.9, 0.4, 2.1, 1.1, 0.5, 2.3)),
     (0, 1, 2, 3, 4, 5)),
    ("RSNN_SHD.tbptt", RSNN_SHD, _rsnn_head(), (7, 8, 9)),
    ("RSNN_SHD.bptt", RSNN_SHD, _rsnn_head() + _rsnn_adjoints(), (7, 8, 9)),
    ("RSNN_SHD.rtrl", RSNN_SHD, _rsnn_head() + _rsnn_carry(), (7, 8, 9)),
    ("RSNN_SHD_W2", RSNN_SHD_W2,
     (_arr(_N_IN), _arr(_N_IN, lo=0.2, hi=0.8)) + _rsnn_head()[1:], (8, 9, 10)),
]


@contextmanager
def _prune_setting(value: str):
    """``GRAPHAX_PRUNE`` for one block. The flag is cached, so the cache is
    reset on the way in and on the way out."""
    old = os.environ.get("GRAPHAX_PRUNE")
    os.environ["GRAPHAX_PRUNE"] = value
    gxcore._PRUNE_CACHE = None
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("GRAPHAX_PRUNE", None)
        else:
            os.environ["GRAPHAX_PRUNE"] = old
        gxcore._PRUNE_CACHE = None


def _inlined(fn, args):
    """The jaxpr BOTH sides number vertices on."""
    closed = jax.make_jaxpr(fn)(*args)
    jaxpr, consts = inline_call_primitives(closed.jaxpr, closed.literals)
    return jaxpr, list(consts)


def _valid_vertices(jaxpr, args, consts, argnums):
    """The eliminable vertices, by the rule the environment applies."""
    _, _, _, vo = _build_graph(jaxpr, list(args), list(consts), argnums)
    return [i for i, eqn in enumerate(jaxpr.eqns, 1)
            if eqn.outvars[0] not in jaxpr.outvars or i in vo]


def _orders(valid):
    rng = np.random.default_rng(97)
    return {"reverse": sorted(valid, reverse=True),
            "shuffle": [int(v) for v in rng.permutation(np.asarray(valid))]}


def _enumerated(jaxpr, consts, args, argnums, order):
    """What the POLICY asks for: faces_of on the incremental builder, taken
    immediately before each elimination."""
    ij = IncrementalJaxpr(jaxpr, tuple(argnums), list(consts), list(args),
                          track_faces=False)
    out = []
    for v in order:
        out.append(list(faces_of(ij.graph, ij.tgraph, v, jaxpr)))
        ij.eliminate(v, (), None)
    return out


def _walked(jaxpr, consts, args, argnums, order):
    """What the MEASUREMENT contracts: the eliminator's own graph, walked in
    the same order. Inside a trace, because an unforced edge emits equations."""
    out = []

    def run(*traced):
        _, g, tg, vo = _build_graph(jaxpr, list(traced), list(consts), argnums)
        if prune_enabled():
            _prune_graph(g, tg, jaxpr, argnums)
        for v in order:
            out.append(list(faces_of(g, tg, v, jaxpr)))
            _eliminate_vertex(v, jaxpr, g, tg, vo, count_ops=False,
                              transforms=(), face_transforms=None, var_vid={})
        return jnp.zeros(())

    jax.make_jaxpr(run)(*args)
    return out


@pytest.mark.parametrize("prune", ["1", "0"])
@pytest.mark.parametrize("name,fn,args,argnums", TARGETS,
                         ids=[t[0] for t in TARGETS])
def test_the_two_builders_list_the_same_faces(name, fn, args, argnums, prune):
    with _prune_setting(prune):
        jaxpr, consts = _inlined(fn, args)
        valid = _valid_vertices(jaxpr, args, consts, argnums)
        assert valid, f"{name}: no eliminable vertex to compare"
        for kind, order in _orders(valid).items():
            asked = _enumerated(jaxpr, consts, args, argnums, order)
            got = _walked(jaxpr, consts, args, argnums, order)
            for pos, (a, b) in enumerate(zip(asked, got)):
                assert a == b, (
                    f"{name} (GRAPHAX_PRUNE={prune}, {kind} order): at "
                    f"position {pos} the enumerator lists {a} for vertex "
                    f"{order[pos]} and the elimination {b}. A face key the "
                    f"elimination never enumerates configures nothing.")
