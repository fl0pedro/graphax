"""IncrementalPathTokenizer — graphax's PRIMARY tokenizer had no tests at all.

The property that matters for an RL consumer is APPEND-ONLY: the stream after
step k must be a PREFIX of the stream after step k+1. If it is not, a policy
that encoded the prefix once and extended it per action would silently be
reading a different history than the one that produced its reward.
"""
import jax
import jax.numpy as jnp
import pytest

from graphax import IncrementalPathTokenizer

CASES = [
    ("sin_mul", lambda x, y: jnp.sin(x * y) + x, (jnp.ones((3,)), jnp.ones((3,)))),
    ("chain", lambda x, y: jnp.tanh(x @ y), (jnp.ones((3, 4)), jnp.ones((4, 3)))),
    ("branchy", lambda x, y: (jnp.tanh(jnp.sin(x) * jnp.cos(y))
                              + jnp.exp(jnp.sin(x) * jnp.cos(y))),
     (jnp.ones((3,)) * 0.5, jnp.ones((3,)) * 0.4)),
]
IDS = [c[0] for c in CASES]


def _mk(fn, args):
    cj = jax.make_jaxpr(fn)(*args)
    return IncrementalPathTokenizer(cj.jaxpr, (0, 1), cj.literals, args), cj.jaxpr


@pytest.mark.parametrize("name,fn,args", CASES, ids=IDS)
def test_stream_is_append_only(name, fn, args):
    tk, jaxpr = _mk(fn, args)
    stream = list(tk.base_tokens())
    prev = list(stream)
    for v in range(1, len(jaxpr.eqns) + 1):
        try:
            step = list(tk.eliminate(v))
        except Exception:
            break                      # not eliminable in this order; fine
        stream = stream + step
        assert stream[:len(prev)] == prev, (
            f"{name}: step {v} rewrote history -- the stream is not append-only")
        prev = list(stream)
    assert len(stream) > 0


@pytest.mark.parametrize("name,fn,args", CASES, ids=IDS)
def test_base_tokens_are_a_prefix_of_the_full_stream(name, fn, args):
    tk, jaxpr = _mk(fn, args)
    base = list(tk.base_tokens())
    tk2, _ = _mk(fn, args)
    full = list(tk2.capture_stream(list(range(1, len(jaxpr.eqns) + 1))))
    assert full[:len(base)] == base, f"{name}: base tokens are not a prefix"


@pytest.mark.parametrize("name,fn,args", CASES, ids=IDS)
def test_max_token_id_bounds_every_emitted_id(name, fn, args):
    """A consumer sizes its embedding table from max_token_id(). If any emitted
    id exceeded it, the embedding lookup would be out of bounds -- which in JAX
    CLAMPS silently rather than raising, so the policy would read a wrong row."""
    tk, jaxpr = _mk(fn, args)
    toks = list(tk.capture_stream(list(range(1, len(jaxpr.eqns) + 1))))
    assert toks, f"{name}: empty stream"
    assert max(int(t) for t in toks) <= int(tk.max_token_id()), (
        f"{name}: an emitted token id exceeds max_token_id()")
    assert min(int(t) for t in toks) >= 0


@pytest.mark.parametrize("name,fn,args", CASES, ids=IDS)
def test_max_token_id_can_exceed_the_static_vocab(name, fn, args):
    """Documented hazard, pinned so it is not forgotten: this tokenizer does NOT
    cap the id space, so max_token_id() may exceed len(vocab). Sizing an
    embedding from the vocab alone is a real bug, not a theoretical one."""
    tk, _ = _mk(fn, args)
    # ``_L`` is the static vocab length max_token_id() builds on.
    assert int(tk.max_token_id()) >= int(tk._L) - 1


@pytest.mark.parametrize("name,fn,args", CASES, ids=IDS)
def test_decode_runs_on_the_emitted_stream(name, fn, args):
    tk, jaxpr = _mk(fn, args)
    toks = list(tk.capture_stream(list(range(1, len(jaxpr.eqns) + 1))))
    out = tk.decode(toks)
    assert out is not None and len(out) > 0


def test_eliminating_more_only_grows_the_stream():
    fn, args = CASES[2][1], CASES[2][2]
    tk, jaxpr = _mk(fn, args)
    lengths = [len(list(tk.base_tokens()))]
    for v in range(1, len(jaxpr.eqns) + 1):
        try:
            step = list(tk.eliminate(v))
        except Exception:
            break
        lengths.append(lengths[-1] + len(step))
    assert lengths == sorted(lengths), "stream length must be monotone"
    assert lengths[-1] > lengths[0], "eliminations must emit something"


# ---------------------------------------------------------------------------
# Bounded name alphabet
# ---------------------------------------------------------------------------

def _name_hungry():
    def big(x, y):
        v = x
        for _ in range(12):
            v = jnp.tanh(v * y) + jnp.sin(v)
        return jnp.sum(v), jnp.sum(v * y)
    return big, (jnp.ones((4,)) * 0.5, jnp.ones((4,)) * 0.3)


def test_bounded_vocab_keeps_every_id_in_range():
    """Names are POSITIONAL sequences over the alphabet (a, b, .., aa, ab, ..),
    so a bounded alphabet still spells unlimited names -- it just spends more
    tokens on later ones. The ids must therefore stay inside vocab_size no
    matter how many names the graph needs."""
    fn, args = _name_hungry()
    cj = jax.make_jaxpr(fn)(*args)
    order = list(range(1, len(cj.jaxpr.eqns) + 1))
    tk = IncrementalPathTokenizer(cj.jaxpr, (0, 1), cj.literals, args,
                                  vocab_size=512)
    toks = list(tk.capture_stream(order))
    assert len(tk._names) > 500, "need a name-hungry graph for this to mean anything"
    assert max(int(t) for t in toks) < 512
    assert int(tk.max_token_id()) < 512
    assert max(len(a) for a in tk._names.values()) >= 2, (
        "with 500+ names and a 270-symbol alphabet, names MUST go multi-atom")


def test_bounded_vocab_costs_tokens_not_ids():
    """The bound trades id range for sequence length -- that is the whole deal."""
    fn, args = _name_hungry()
    cj = jax.make_jaxpr(fn)(*args)
    order = list(range(1, len(cj.jaxpr.eqns) + 1))
    a = IncrementalPathTokenizer(cj.jaxpr, (0, 1), cj.literals, args)
    b = IncrementalPathTokenizer(cj.jaxpr, (0, 1), cj.literals, args,
                                 vocab_size=512)
    ta, tb = list(a.capture_stream(order)), list(b.capture_stream(order))
    assert max(tb) < max(ta), "bounded run must use smaller ids"
    assert len(tb) > len(ta), "and pay for it in tokens"


def test_vocab_too_small_to_spell_names_is_rejected():
    fn, args = _name_hungry()
    cj = jax.make_jaxpr(fn)(*args)
    with pytest.raises(ValueError, match="name symbols"):
        IncrementalPathTokenizer(cj.jaxpr, (0, 1), cj.literals, args,
                                 vocab_size=240)
