import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core as jcore
from jax._src.interpreters import partial_eval as pe

from graphax import IncrementalJaxpr, IncrementalPathTokenizer
from graphax.core import _build_graph, _checkify_order, _eliminate_vertex
from graphax.core import _inline_call_primitives

_M = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0 + 0.1)
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))
_Y4 = jnp.asarray(np.linspace(-0.4, 0.6, 4, dtype=np.float32))


def _mlp2in(x, y):
    h = jnp.tanh(_M @ x + y)
    a = jnp.sin(h) * jnp.exp(h)
    return jnp.sum(a * h), _M @ a


def _small():
    args = (_X4, _Y4)
    closed = jax.make_jaxpr(_mlp2in)(*args)
    jaxpr, consts = _inline_call_primitives(closed.jaxpr, closed.literals)
    _, _, _, vo = _build_graph(jaxpr, list(args), list(consts))
    order = [int(v) for v in _checkify_order("rev", jaxpr, vo)]
    return jaxpr, (0, 1), list(consts), list(args), order


def _tlm():
    if not os.environ.get("DSNN_WIKITEXT_DIR"):
        pytest.skip("the TLM case needs alphagrad and DSNN_WIKITEXT_DIR")
    import alphagrad.approx.tools.landscape_map as lm
    a = lm.make_argparser().parse_args(
        ["--example", "TransformerLM", "--dataset", "wikitext2",
         "--seed", "250197"])
    env, _eval, _cj = lm.build_env(a)
    cfg = env.config
    order = sorted((int(v) for v in env.valid_vertices), reverse=True)
    return cfg.jaxpr, tuple(cfg.argnums), list(env.consts), list(env.args), order


GRAPHS = {"small": _small, "tlm": _tlm}


def _same(got, ref):
    assert len(got) == len(ref)
    for i, (a, b) in enumerate(zip(got, ref)):
        if a is b:
            continue
        assert type(a) is type(b) is jcore.JaxprEqn, i
        assert a.primitive is b.primitive, i
        assert a.params is b.params, i
        assert a.effects is b.effects, i
        assert a.source_info is b.source_info, i
        assert a.ctx is b.ctx, i
        assert a.outvars is b.outvars, i
        assert len(a.invars) == len(b.invars), i
        assert all(x is y for x, y in zip(a.invars, b.invars)), i


def _check_all(ij):
    ref = ij.all_eqns_rebuilt()
    _same(ij.all_eqns(), ref)
    _same(ij.base_eqns(), ref[:ij.n_base])
    for i in range(len(ij.steps)):
        s, e = ij.steps[i][2:4]
        _same(ij.step_eqns(i), ref[s:e])


def _speculate(ij, v, through_eliminate):
    teq = ij.trace.frame.tracing_eqns
    n_eq, n_st, n_x = len(teq), len(ij.steps), len(ij.xlog.records)
    sink = ij.face_sink
    n_fc = len(sink.faces) if sink is not None else 0
    g0, t0, vo0 = ij.graph, ij.tgraph, ij.vo
    ij.graph = {k: dict(x) for k, x in g0.items()}
    ij.tgraph = {k: dict(x) for k, x in t0.items()}
    ij.vo = dict(vo0) if isinstance(vo0, dict) else vo0
    try:
        if through_eliminate:
            got = ij.eliminate(v)
            s, e = ij.steps[-1][2:4]
            _same(got, ij.all_eqns_rebuilt()[s:e])
        else:
            with jcore.set_current_trace(ij.trace), sink, ij.xlog:
                _eliminate_vertex(v, ij.jaxpr, ij.graph, ij.tgraph, ij.vo,
                                  False, transforms=(), face_transforms=None)
    finally:
        ij.graph, ij.tgraph, ij.vo = g0, t0, vo0
        del teq[n_eq:]
        del ij.steps[n_st:]
        if sink is not None:
            del sink.faces[n_fc:]
        del ij.xlog.records[n_x:]


@pytest.mark.parametrize("graph", list(GRAPHS))
def test_eliminate_and_all_eqns_match_the_rebuild(graph):
    jaxpr, argnums, consts, args, order = GRAPHS[graph]()
    ij = IncrementalJaxpr(jaxpr, argnums, consts, args, track_faces=True)
    _check_all(ij)
    for k, v in enumerate(order):
        if k + 1 < len(order):
            _speculate(ij, order[k + 1], through_eliminate=k % 2 == 0)
        got = ij.eliminate(v)
        s, e = ij.steps[-1][2:4]
        ref = ij.all_eqns_rebuilt()
        _same(got, ref[s:e])
        _same(ij.all_eqns(), ref)
    _check_all(ij)
    ij.jacobian_outputs()
    _check_all(ij)


@pytest.mark.parametrize("graph", list(GRAPHS))
def test_tokenizer_stream_matches_the_rebuild(graph):
    jaxpr, argnums, consts, args, order = GRAPHS[graph]()
    new = IncrementalPathTokenizer(jaxpr, argnums, consts, args)
    old = IncrementalPathTokenizer(jaxpr, argnums, consts, args)
    oij = old.ij
    oij.all_eqns = oij.all_eqns_rebuilt
    oij.base_eqns = lambda: list(oij.all_eqns_rebuilt()[:oij.n_base])
    assert [int(t) for t in new.base_tokens()] == [
        int(t) for t in old.base_tokens()]
    for v in order:
        assert [int(t) for t in new.eliminate(v)] == [
            int(t) for t in old.eliminate(v)]


def test_eliminate_and_all_eqns_do_not_rebuild(monkeypatch):
    jaxpr, argnums, consts, args, order = _small()
    ij = IncrementalJaxpr(jaxpr, argnums, consts, args, track_faces=True)
    calls = []
    orig = pe.JaxprStackFrame.get_eqns

    def counted(self):
        calls.append(1)
        return orig(self)

    monkeypatch.setattr(pe.JaxprStackFrame, "get_eqns", counted)
    for v in order:
        ij.eliminate(v)
        ij.all_eqns()
        ij.step_eqns(len(ij.steps) - 1)
    ij.base_eqns()
    assert calls == []
