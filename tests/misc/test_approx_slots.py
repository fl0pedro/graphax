"""Per-face approximation SLOTS are STRUCTURAL in the token stream.

A face has three approximation slots -- ``pre``/``lhs`` (the in_edge Jacobian),
``post``/``rhs`` (the out_edge Jacobian) and ``new``/``res`` (the contraction
result). The old emitter wrote one ``approx <TYPE> <args>`` head per RECORDED
approximation, and a slot that declined recorded nothing, so

    (DIAG, none, none)   (none, DIAG, none)   (none, none, DIAG)

were three BYTE-IDENTICAL streams: the policy was told an approximation
happened but not which operand it applied to. This is the defect class the
``approx SKIP`` marker fixed for whole faces -- identity recoverable only by
position, and position destroyed by absence.

The fix is ONE head per face covering all three slots at fixed positions::

    approx <TYPE args>_pre ^ <TYPE args>_post ^ <TYPE args>_new
    {pre eqns} {post eqns} {new eqns}

so a declining slot contributes no tokens (nothing between two separators) and
an empty ``{}`` block, the separator count is invariantly 2, and the slot is a
POSITION. ``approx SKIP`` keeps its own slot-less form -- a skip drops the whole
contraction, so there are no pre/post/new operands.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from graphax import SKIP_FACE, IncrementalPathTokenizer, faces_of
from graphax.jaxpr import SLOT_SEPARATOR, get_vocab
from graphax.sparse.micro_actions import QUANT_DTYPE_INDEX, Compress, Diag, Quant
from graphax.sparse.tracer import FACE_SLOT_NAMES, N_FACE_SLOTS, face_slot_index

_M = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 15.0 + 0.1)
_P = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 13.0 + 0.2)
_Q = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 17.0 + 0.3)
_R = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4) / 19.0 + 0.4)
_X4 = jnp.asarray(np.linspace(0.1, 0.9, 4, dtype=np.float32))

_DIAG = Diag(0, 1, 2)
_QUANT = Quant("bfloat16")


def _square(x):
    """One in-edge, two out-edges -> vertex 1 has two faces."""
    e = _M @ x
    return _P @ e, _Q @ e


def _cube(x):
    """One in-edge, three out-edges -> vertex 1 has three faces."""
    e = _M @ x
    return _P @ e, _Q @ e, _R @ e


def _tokenize(face_transforms=None, fn=_square, args=(_X4,), vertex=1):
    """Tokenize one elimination of ``vertex`` -> ``(tk, delta, segments)``.

    ``face_transforms`` is a mapping FACE INDEX -> slots value, resolved
    against the live face keys."""
    closed = jax.make_jaxpr(fn)(*args)
    tk = IncrementalPathTokenizer(closed.jaxpr, (0,), list(closed.literals),
                                  list(args), vocab_size=248)
    tk.base_tokens()
    keys = faces_of(tk.ij.graph, tk.ij.tgraph, vertex, closed.jaxpr)
    ft = None if face_transforms is None else {
        keys[i]: v for i, v in face_transforms.items()}
    delta = [int(t) for t in tk.eliminate(vertex, (), ft)]
    return tk, delta, tk.last_face_segments()


def _approx_part(tk, delta, seg):
    """``[split:end]`` of one face -- its whole approximation part."""
    _start, split, end = seg
    return delta[split:end]


def _head(tk, toks):
    """The decoded ``approx`` HEAD of an approximation part: everything before
    the first equation block (``{``) or fn-definition block (``fns``), i.e. the
    slot layout alone, without the equations (whose first-appearance variable
    names legitimately differ between runs)."""
    vocab, _, _ = get_vocab()
    stop = {vocab["{"], vocab["fns"]}
    cut = next((i for i, t in enumerate(toks) if t in stop), len(toks))
    return tk.decode(toks[:cut])


def _slots(**kw):
    """``(pre, post, new)`` from keyword slots, defaulting to ``none``."""
    return (kw.get("pre"), kw.get("post"), kw.get("new"))


# ---------------------------------------------------------------------------
# THE decisive property: same approximation, different slot -> different stream
# ---------------------------------------------------------------------------
def test_the_same_approximation_in_each_slot_gives_three_different_streams():
    _tk_p, tok_pre, _ = _tokenize({0: _slots(pre=_DIAG)})
    _tk_q, tok_post, _ = _tokenize({0: _slots(post=_DIAG)})
    _tk_n, tok_new, _ = _tokenize({0: _slots(new=_DIAG)})

    streams = [tuple(tok_pre), tuple(tok_post), tuple(tok_new)]
    assert len(set(streams)) == 3, (
        "DIAG in pre / post / new must NOT be byte-identical -- the header has "
        "to say WHICH operand was approximated")


@pytest.mark.parametrize("slot,head", [
    ("pre", "approxDIAG012^^"),
    ("post", "approx^DIAG012^"),
    ("new", "approx^^DIAG012"),
])
def test_a_single_slot_lands_at_its_own_position(slot, head):
    """The decoded head, spelled out: the declining slots contribute NO tokens
    -- just nothing between the separators."""
    tk, delta, segs = _tokenize({0: _slots(**{slot: _DIAG})})
    assert _head(tk, _approx_part(tk, delta, segs[0])) == head


def test_two_slots_and_three_slots_keep_the_positions():
    tk, delta, segs = _tokenize({0: _slots(pre=_DIAG, new=_DIAG)})
    assert _head(tk, _approx_part(tk, delta, segs[0])) == \
        "approxDIAG012^^DIAG012"

    tk, delta, segs = _tokenize({0: (_QUANT, _QUANT, _QUANT)})
    assert _head(tk, _approx_part(tk, delta, segs[0])) == (
        f"approx~{QUANT_DTYPE_INDEX['bfloat16']}^^QUANTd#bfloat16")


def test_the_separator_count_is_invariant_and_three_blocks_follow():
    """Whenever a slot head is emitted it carries EXACTLY two separators and is
    followed by exactly three equation blocks -- one per slot, empty for the
    slots that declined. That fixed shape is what makes position readable."""
    vocab, _, _ = get_vocab()
    for slots in [_slots(pre=_DIAG), _slots(post=_DIAG), _slots(new=_DIAG),
                  _slots(pre=_DIAG, new=_DIAG), (_QUANT, _QUANT, _QUANT)]:
        tk, delta, segs = _tokenize({0: slots})
        part = _approx_part(tk, delta, segs[0])
        assert part.count(vocab[SLOT_SEPARATOR]) == 2, slots
        assert part.count(vocab["approx"]) == 1, slots
        assert part.count(vocab["{"]) == N_FACE_SLOTS == part.count(vocab["}"])


# ---------------------------------------------------------------------------
# the two forms that are NOT the slot form
# ---------------------------------------------------------------------------
def test_all_three_slots_none_emits_no_approx_header_at_all():
    vocab, _, _ = get_vocab()
    tk, delta, segs = _tokenize()                    # exact face
    start, split, end = segs[0]
    assert split == end, "an exact face emits nothing after its contraction"
    assert vocab["approx"] not in delta[start:end]
    assert vocab[SLOT_SEPARATOR] not in delta


def test_an_explicit_none_in_every_slot_is_the_same_as_exact():
    """Recording is driven by what was APPLIED, so (None, None, None) is
    indistinguishable from no entry at all -- and emits no header."""
    vocab, _, _ = get_vocab()
    _tk, tok_exact, _ = _tokenize()
    _tk2, tok_none, _ = _tokenize({0: (None, None, None)})
    assert tok_none == tok_exact
    assert vocab["approx"] not in tok_none


def test_a_skipped_face_keeps_the_slotless_skip_form():
    vocab, _, _ = get_vocab()
    tk, delta, segs = _tokenize({0: SKIP_FACE})
    part = _approx_part(tk, delta, segs[0])
    assert tk.decode(part) == "approxSKIP{}"
    assert vocab[SLOT_SEPARATOR] not in part, (
        "SKIP takes no operand slots -- the skip drops the whole contraction")


def test_skip_and_slot_forms_never_mix_across_faces():
    """Face 0 approximated in the POST slot, face 1 skipped: each face carries
    exactly one of the two forms, in its own segment."""
    vocab, _, _ = get_vocab()
    tk, delta, segs = _tokenize({0: _slots(post=_DIAG), 1: SKIP_FACE})
    p0 = _approx_part(tk, delta, segs[0])
    p1 = _approx_part(tk, delta, segs[1])

    assert _head(tk, p0) == "approx^DIAG012^"
    assert vocab["SKIP"] not in p0
    assert tk.decode(p1) == "approxSKIP{}"


# ---------------------------------------------------------------------------
# index alignment (the earlier skip fix) must survive the new head
# ---------------------------------------------------------------------------
def test_slot_heads_do_not_reindex_the_face_segments():
    """Face 1 of 3 skipped while face 0 is approximated: still 3 segments, and
    the SKIP still lands on face 1."""
    vocab, _, _ = get_vocab()
    tk, delta, segs = _tokenize({0: _slots(pre=_DIAG), 1: SKIP_FACE}, fn=_cube)

    assert len(segs) == 3, "a skipped face still gets its own segment entry"
    for f, (s, _sp, e) in enumerate(segs):
        assert (vocab["SKIP"] in delta[s:e]) == (f == 1)
    assert _head(tk, _approx_part(tk, delta, segs[0])) == "approxDIAG012^^"
    assert _approx_part(tk, delta, segs[2]) == [], "face 2 ran exact"


def test_face_headers_are_stable_slot_for_slot_under_a_skip():
    """The identity head of face f must not move when another face is
    skipped -- pinned again here because the approx part now has a new shape."""
    vocab, _, _ = get_vocab()
    stop = {vocab["fns"], vocab["{"]}

    def _ident(tk, delta, seg):
        start, split, _end = seg
        body = delta[start:split]
        cut = next((i for i, t in enumerate(body) if t in stop), len(body))
        return tk.decode(body[:cut])

    tk_x, tok_x, seg_x = _tokenize(fn=_cube)
    tk_s, tok_s, seg_s = _tokenize({1: SKIP_FACE}, fn=_cube)
    assert len(seg_x) == len(seg_s) == 3
    for f in range(3):
        assert _ident(tk_x, tok_x, seg_x[f]) == _ident(tk_s, tok_s, seg_s[f])


# ---------------------------------------------------------------------------
# the record carries the slot; the vocabulary carries the separator
# ---------------------------------------------------------------------------
def test_the_record_carries_the_slot_not_the_position():
    """What makes the head possible: the sink TAGS every approximation with the
    operand it hit, so a declining middle slot cannot shift the others."""
    _tk, _delta, _segs = _tokenize({0: _slots(post=_DIAG)})
    tk, _d, _s = _tokenize({0: _slots(pre=_DIAG, new=_QUANT)})
    recs = tk.ij.step_faces(0)[0].approx
    assert [r.slot for r in recs] == ["lhs", "res"]
    assert [face_slot_index(r.slot) for r in recs] == [0, 2]
    assert FACE_SLOT_NAMES == ("lhs", "rhs", "res")


def test_a_per_vertex_transform_occupies_the_new_slot():
    """A per-vertex ``transforms`` entry is applied to the contraction RESULT,
    the same operand as ``new``/``res``, so it renders in the new slot."""
    closed = jax.make_jaxpr(_square)(_X4)
    tk = IncrementalPathTokenizer(closed.jaxpr, (0,), list(closed.literals),
                                  [_X4], vocab_size=248)
    tk.base_tokens()
    delta = [int(t) for t in tk.eliminate(1, (_QUANT,), None)]
    _s, split, end = tk.last_face_segments()[0]

    assert [r.slot for r in tk.ij.step_faces(0)[0].approx] == ["vertex"]
    assert _head(tk, delta[split:end]) == "approx^^QUANTd#bfloat16"


def test_the_slot_separator_is_its_own_token_appended_at_the_end():
    """``^`` must not reuse ``&`` (the path header's separator): one token, one
    meaning. It is appended LAST so no existing token id moved."""
    vocab, n_vocab, full = get_vocab()
    assert SLOT_SEPARATOR == "^"
    assert vocab[SLOT_SEPARATOR] != vocab["&"]
    assert full[-1] == SLOT_SEPARATOR
    assert vocab[SLOT_SEPARATOR] == len(vocab) - 1
    assert n_vocab[vocab[SLOT_SEPARATOR]] == SLOT_SEPARATOR


# ---------------------------------------------------------------------------
# a CHOOSER'S action is recorded exactly as a literal one is
# ---------------------------------------------------------------------------
#
# A slot callable has two meanings. It may return a TENSOR, which the engine
# takes as the new operand and records for nobody, or it may return the
# micro-action it PICKED, which the engine applies through ``_apply_micro`` and
# records through ``_record_micro``. Only the second reaches the token stream.
# Every policy has to be the second kind: its operands are join intermediates,
# so it cannot know their index structure before this moment, and a decision it
# takes without leaving a block is a decision the encoder never sees.


class _Chooser:
    """A slot chooser that always picks ``action`` and remembers the outcome."""

    def __init__(self, action):
        self.action = action
        self.outcomes = []

    def __call__(self, st):
        return self.action

    def chosen_applied(self, action, applied):
        self.outcomes.append((action, bool(applied)))


_COMPRESS = Compress(axes=(0,), kind="mean")

# A Quant is two-sided (owner ruling 2026-09-23), so its chooser sits on both
# contraction slots and its head is the face's ``~ <dtype index>``.
_CHOOSER_CASES = [
    (_DIAG, ("pre",), "approxDIAG012^^", ["lhs"]),
    (_COMPRESS, ("pre",), "approxCOMPRESSk#mean0^^", ["lhs"]),
    (_QUANT, ("pre", "post"), f"approx~{QUANT_DTYPE_INDEX['bfloat16']}^^",
     ["lhs", "rhs"]),
]


@pytest.mark.parametrize("action,where,head,_slots_unused", _CHOOSER_CASES,
                         ids=["diag", "compress", "quant"])
def test_a_chooser_that_returns_an_action_emits_the_block_a_literal_emits(
        action, where, head, _slots_unused):
    """TYPE, ARGS and the output jaxpr, byte for byte the literal's block.

    The chooser is the only form a policy can use, so if its block differed
    from a literal's in any way the stream would carry two grammars for one
    decision.
    """
    ch = _Chooser(action)
    tk_c, tok_c, segs_c = _tokenize({0: _slots(**{w: ch for w in where})})
    tk_l, tok_l, segs_l = _tokenize({0: _slots(**{w: action for w in where})})

    assert _head(tk_c, _approx_part(tk_c, tok_c, segs_c[0])) == head
    assert _approx_part(tk_c, tok_c, segs_c[0]) == \
        _approx_part(tk_l, tok_l, segs_l[0])
    assert tuple(tok_c) == tuple(tok_l)


@pytest.mark.parametrize("action,where,_head_unused,rec_slots", _CHOOSER_CASES,
                         ids=["diag", "compress", "quant"])
def test_a_chosen_actions_block_carries_the_equations_it_emitted(
        action, where, _head_unused, rec_slots):
    """THE OUTPUT TOKENIZED JAXPR. A block is ``approx <TYPE> <args>`` followed
    by the three slots' equation blocks, and the approximated slot's block is
    the jaxpr the application itself wrote -- a ``convert_element_type`` for
    QUANT, a ``reduce_sum`` and a divide for COMPRESS, the reshape / transpose /
    slice chain for DIAG. The record's equation RANGE is what carves those
    equations out of the face's contraction block, so a non-empty range is the
    same statement as a non-empty block.
    """
    ch = _Chooser(action)
    tk, tok, segs = _tokenize({0: _slots(**{w: ch for w in where})})
    recs = tk.ij.step_faces(0)[0].approx

    assert [r.slot for r in recs] == rec_slots
    assert recs[0].end > recs[0].start, (
        "the approximation emitted no equations, so its block carries no "
        "output jaxpr at all")
    _start, split, end = segs[0]
    part = tok[split:end]
    assert part.count(tk.vocab["{"]) == 3, "three slot blocks, always"
    # The approximated slot's block is the FIRST of the three and it is not
    # empty; the two declining slots emit `{}`.
    first = part.index(tk.vocab["{"])
    assert part[first + 1] != tk.vocab["}"]


def test_the_chooser_is_told_whether_its_action_changed_the_tensor():
    """A chooser decides BEFORE the action runs, so it cannot know by itself
    whether the action was a no-op -- and a no-op emits no block. The callback
    is what lets a caller's ``applied`` counter agree with the stream."""
    ch = _Chooser(_QUANT)
    tk, _tok, _segs = _tokenize({0: _slots(pre=ch, post=ch)})
    assert ch.outcomes == [(_QUANT, True), (_QUANT, True)]
    assert [r.atype for r in tk.ij.step_faces(0)[0].approx] == ["QUANT", "QUANT"]

    # THE INVARIANT the callback exists for: as many blocks as outcomes the
    # engine reported applied, on every face, whatever the slots hold.
    again = _Chooser(_QUANT)
    tk2, _t2, _segs2 = _tokenize({0: _slots(pre=again, post=again, new=again)})
    assert len(again.outcomes) == 3
    n_blocks = len(tk2.ij.step_faces(0)[0].approx)
    assert n_blocks == sum(1 for _a, applied in again.outcomes if applied)


def test_a_chooser_that_declines_emits_no_block_at_all():
    """``None`` is the one legal way to decline, and a decline is silence in
    the stream: nothing was applied, so there is nothing to mark."""
    tk, tok, segs = _tokenize({0: _slots(pre=lambda st: None)})
    _start, split, end = segs[0]
    assert split == end
    assert tk.ij.step_faces(0)[0].approx == []


def test_a_chooser_that_returns_several_actions_raises():
    """graphax applies ONE action per chooser result. Taking the first would
    drop the rest of what the caller asked for, silently."""
    with pytest.raises(TypeError, match="returns exactly ONE"):
        _tokenize({0: _slots(pre=lambda st: (_DIAG, _QUANT))})
