import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import IncrementalPathTokenizer, faces_of
from graphax.incremental import IncrementalJacobian
from graphax.jaxpr import SLOT_SEPARATOR, get_vocab
from graphax.sparse.micro_actions import QUANT_DTYPE_INDEX, Quant

_W1 = jnp.asarray(np.arange(48, dtype=np.float32).reshape(8, 6) / 47.0 - 0.3)
_W2 = jnp.asarray(np.arange(40, dtype=np.float32).reshape(5, 8) / 39.0 + 0.1)
_W3 = jnp.asarray(np.arange(24, dtype=np.float32).reshape(3, 8) / 23.0 - 0.2)
_X6 = jnp.asarray(np.linspace(-0.5, 0.7, 6, dtype=np.float32))
_XB = jnp.asarray(np.linspace(-0.5, 0.7, 18, dtype=np.float32).reshape(3, 6))

_Q = Quant("bfloat16")
_QI = QUANT_DTYPE_INDEX["bfloat16"]


def _chain(x):
    return _W2 @ (_W1 @ x)


def _fanout(x):
    e = _W1 @ x
    return _W2 @ e, _W3 @ e


_W1T = jnp.asarray(np.asarray(_W1).T)
_W2T = jnp.asarray(np.asarray(_W2).T)


def _batched(x):
    return (x @ _W1T) @ _W2T


def _eliminate_all(fn, args, vertex, slots, faces=None):
    closed = jax.make_jaxpr(fn)(*args)
    ij = IncrementalJacobian(closed.jaxpr, (0,), list(closed.literals),
                             list(args), track_faces=True)
    for v in range(1, len(closed.jaxpr.eqns) + 1):
        if v != vertex:
            ij.eliminate(v)
            continue
        keys = faces_of(ij.graph, ij.tgraph, v, closed.jaxpr)
        picked = keys if faces is None else [keys[i] for i in faces]
        ij.eliminate(v, (), {k: slots for k in picked})
    return ij


def _jacobian(ij, args):
    jaxpr, consts, _ = ij.current_jaxpr()
    outs = jax.core.eval_jaxpr(jaxpr, consts, *args)
    return outs[-1]


def _lowered(ij, args):
    jaxpr, consts, _ = ij.current_jaxpr()
    f = jax.jit(lambda *a: jax.core.eval_jaxpr(jaxpr, consts, *a))
    return f.lower(*args).as_text()


_DOT = re.compile(
    r"stablehlo\.dot_general\b.*?:\s*\((tensor<[^>]+>),\s*(tensor<[^>]+>)\)"
    r"\s*->\s*(tensor<[^>]+>)")
_CONVERT = re.compile(
    r"stablehlo\.convert\b.*?:\s*\((tensor<[^>]+>)\)\s*->\s*(tensor<[^>]+>)")


def _assert_narrow_gemm(hlo):
    dots = _DOT.findall(hlo)
    assert dots, hlo
    narrow = [d for d in dots if d[0].endswith("xbf16>") and d[1].endswith("xbf16>")]
    assert narrow, f"no bf16 x bf16 dot:\n{hlo}"
    for lhs, rhs, out in narrow:
        assert out.endswith("xf32>"), f"bf16 dot without f32 sums: {(lhs, rhs, out)}\n{hlo}"
    mixed = [d for d in dots if ("bf16" in d[0]) != ("bf16" in d[1])]
    assert not mixed, f"mixed-width dot: {mixed}\n{hlo}"
    widened = [c for c in _CONVERT.findall(hlo)
               if c[0].endswith("xbf16>") and c[1].endswith("xf32>")]
    assert not widened, f"a bf16 operand is widened to f32: {widened}\n{hlo}"


def test_a_two_sided_quant_face_is_a_bf16_dot_with_f32_sums_and_a_bf16_result():
    ij = _eliminate_all(_chain, (_X6,), 1, (_Q, _Q, None))
    _assert_narrow_gemm(_lowered(ij, (_X6,)))
    got = _jacobian(ij, (_X6,))
    assert jnp.dtype(got.dtype) == jnp.dtype(jnp.bfloat16)
    want = np.asarray(jax.jacrev(_chain)(_X6), np.float64)
    err = np.abs(np.asarray(got, np.float64) - want).max() / np.abs(want).max()
    assert err < 2e-2, err
    recs = [r for fr in ij.step_faces(0) for r in fr.approx]
    assert [(r.atype, r.slot) for r in recs] == [("QUANT", "lhs"), ("QUANT", "rhs")]


def test_a_two_sided_quant_face_with_a_batch_axis_is_a_bf16_dot():
    ij = _eliminate_all(_batched, (_XB,), 1, (_Q, _Q, None))
    recs = [r for fr in ij.step_faces(0) for r in fr.approx]
    assert [(r.atype, r.slot) for r in recs] == [("QUANT", "lhs"), ("QUANT", "rhs")]
    _assert_narrow_gemm(_lowered(ij, (_XB,)))
    got = _jacobian(ij, (_XB,))
    assert jnp.dtype(got.dtype) == jnp.dtype(jnp.bfloat16)
    want = np.asarray(jax.jacrev(_batched)(_XB), np.float64)
    err = np.abs(np.asarray(got, np.float64) - want).max() / np.abs(want).max()
    assert err < 2e-2, err


_WC = jnp.asarray(np.arange(120, dtype=np.float32).reshape(5, 4, 6) / 119.0 - 0.4)
_WA = jnp.asarray(np.arange(12, dtype=np.float32).reshape(3, 4) / 11.0 - 0.3)


def _summed(x):
    return _WA @ jnp.sum(jnp.einsum("chn,n->ch", _WC, x), axis=0)


def _spread(s):
    return _W1 @ jnp.broadcast_to(s, (6,))


def _eliminate_in_order(fn, args, order, vertex, slots):
    closed = jax.make_jaxpr(fn)(*args)
    ij = IncrementalJacobian(closed.jaxpr, (0,), list(closed.literals),
                             list(args), track_faces=True)
    for v in order:
        if v != vertex:
            ij.eliminate(v)
            continue
        keys = faces_of(ij.graph, ij.tgraph, v, closed.jaxpr)
        ij.eliminate(v, (), {k: slots for k in keys})
    return ij


_DOT_OPERANDS = re.compile(r"(%[\w#]+) = stablehlo\.dot_general (%[\w#]+), (%[\w#]+)")
_WIDENED = re.compile(
    r"(%[\w#]+) = stablehlo\.convert %[\w#]+ : \(tensor<[^>]*xbf16>\) -> tensor<[^>]*xf32>")


@pytest.mark.parametrize("fn, args, order, n", [
    (_summed, (_X6,), (2, 1, 3), 5),
    (_spread, (jnp.float32(0.3),), (1, 2), 6),
], ids=["in-edge-uniform", "out-edge-uniform"])
def test_a_two_sided_quant_face_with_a_real_private_sum_is_one_bf16_dot(fn, args, order, n):
    # The reduce is eliminated first (or the broadcast is the in-edge), so the
    # face's contracted axis sits at extent 1 on one side against n on the
    # other: a real sum over the storing side, not a squeeze.
    ij = _eliminate_in_order(fn, args, order, 1, (_Q, _Q, None))
    hlo = _lowered(ij, args)
    dots = _DOT.findall(hlo)
    narrow = [d for d in dots if d[0].endswith("xbf16>") and d[1].endswith("xbf16>")]
    assert len(narrow) == 1, f"expected one bf16 x bf16 dot, got {dots}\n{hlo}"
    assert narrow[0][2].endswith("xf32>"), narrow
    assert not [d for d in dots if ("bf16" in d[0]) != ("bf16" in d[1])], dots
    widened = [c for c in _CONVERT.findall(hlo)
               if c[0].endswith("xbf16>") and c[1].endswith("xf32>")]
    assert len(widened) == 1, f"only the private sum widens, before the dot: {widened}\n{hlo}"
    assert f"x{n}x" in widened[0][0], widened
    widened_names = {m.group(1) for m in _WIDENED.finditer(hlo)}
    for _res, lhs, rhs in _DOT_OPERANDS.findall(hlo):
        assert lhs not in widened_names and rhs not in widened_names, hlo
    got = _jacobian(ij, args)
    assert jnp.dtype(got.dtype) == jnp.dtype(jnp.bfloat16)
    want = np.asarray(jax.jacrev(fn)(*args), np.float64)
    err = np.abs(np.asarray(got, np.float64).reshape(want.shape) - want).max() / np.abs(want).max()
    assert err < 2e-2, err


@pytest.mark.parametrize("slots", [
    (_Q, None, None),
    (None, _Q, None),
    (lambda st: _Q, None, None),
    (None, lambda st: _Q, None),
    ((_Q, None, None), (None, None, None)),
], ids=["lhs", "rhs", "lhs-chooser", "rhs-chooser", "two-op-lhs"])
def test_a_one_sided_face_quant_is_refused(slots):
    with pytest.raises(ValueError, match="one-sided Quant"):
        _eliminate_all(_chain, (_X6,), 1, slots)


def test_a_face_quant_of_two_dtypes_is_refused():
    with pytest.raises(ValueError, match="ONE dtype"):
        _eliminate_all(_chain, (_X6,), 1, (_Q, Quant("float16"), None))


def test_a_quant_on_the_new_slot_alone_stays_legal():
    ij = _eliminate_all(_chain, (_X6,), 1, (None, None, _Q))
    recs = [r for fr in ij.step_faces(0) for r in fr.approx]
    assert [(r.atype, r.slot) for r in recs] == [("QUANT", "res")]


def _tokenize(fn, args, vertex, slots=None, faces=None):
    closed = jax.make_jaxpr(fn)(*args)
    tk = IncrementalPathTokenizer(closed.jaxpr, (0,), list(closed.literals),
                                  list(args), vocab_size=248)
    tk.base_tokens()
    ft = None
    if slots is not None:
        keys = faces_of(tk.ij.graph, tk.ij.tgraph, vertex, closed.jaxpr)
        picked = keys if faces is None else [keys[i] for i in faces]
        ft = {k: slots for k in picked}
    delta = [int(t) for t in tk.eliminate(vertex, (), ft)]
    return tk, delta, tk.last_face_segments()


def _head(tk, toks):
    vocab, _, _ = get_vocab()
    stop = {vocab["{"], vocab["fns"]}
    cut = next((i for i, t in enumerate(toks) if t in stop), len(toks))
    return tk.decode(toks[:cut])


def test_the_stream_carries_the_face_quant_once_before_the_slots():
    vocab, _, _ = get_vocab()
    tk, delta, segs = _tokenize(_chain, (_X6,), 1, (_Q, _Q, None))
    _start, split, end = segs[0]
    part = delta[split:end]
    assert _head(tk, part) == f"approx~{_QI}^^"
    assert part.count(vocab["~"]) == 1
    assert delta.count(vocab["~"]) == 1
    assert vocab["QUANT"] not in part
    assert part.count(vocab[SLOT_SEPARATOR]) == 2
    assert part.count(vocab["{"]) == 3 == part.count(vocab["}"])
    blocks = [i for i, t in enumerate(part) if t == vocab["{"]]
    assert part[blocks[0] + 1] != vocab["}"]
    assert part[blocks[1] + 1] != vocab["}"]
    assert part[blocks[2] + 1] == vocab["}"]


def test_the_face_quant_sits_beside_the_other_slot_decisions():
    tk, delta, segs = _tokenize(_chain, (_X6,), 1, (_Q, _Q, Quant("float16")))
    _start, split, end = segs[0]
    assert _head(tk, delta[split:end]) == f"approx~{_QI}^^QUANTd#float16"


def test_a_new_slot_quant_to_the_result_dtype_is_a_no_op():
    # The two-sided face already stores its result bf16, so a bf16 Quant on
    # the new slot changes nothing and records nothing.
    tk, delta, segs = _tokenize(_chain, (_X6,), 1, (_Q, _Q, _Q))
    _start, split, end = segs[0]
    assert _head(tk, delta[split:end]) == f"approx~{_QI}^^"
    recs = tk.ij.step_faces(0)[0].approx
    assert [r.slot for r in recs] == ["lhs", "rhs"]


def test_an_exact_face_carries_no_tilde():
    vocab, _, _ = get_vocab()
    _tk, delta, segs = _tokenize(_chain, (_X6,), 1)
    _start, split, end = segs[0]
    assert split == end
    assert vocab["~"] not in delta


def test_one_tilde_per_quant_face_and_none_on_its_exact_sibling():
    vocab, _, _ = get_vocab()
    _tk, delta, segs = _tokenize(_fanout, (_X6,), 1, (_Q, _Q, None), faces=[0])
    assert len(segs) == 2
    s0, sp0, e0 = segs[0]
    s1, sp1, e1 = segs[1]
    assert delta[sp0:e0].count(vocab["~"]) == 1
    assert sp1 == e1
    assert vocab["~"] not in delta[s1:e1]
    assert delta.count(vocab["~"]) == 1


def test_the_face_quant_and_a_new_slot_quant_are_different_streams():
    _t0, both, _s0 = _tokenize(_chain, (_X6,), 1, (_Q, _Q, None))
    _t1, new, _s1 = _tokenize(_chain, (_X6,), 1, (None, None, _Q))
    _t2, exact, _s2 = _tokenize(_chain, (_X6,), 1)
    assert len({tuple(both), tuple(new), tuple(exact)}) == 3
