"""Jaxpr tokenization.

PRIMARY: :class:`IncrementalPathTokenizer` -- the append-only path-elimination
tokenizer built on the preserved-trace :class:`graphax.incremental.Incremental
Jacobian`. This is the canonical tokenized-jaxpr representation; new code should
use it (``capture_stream`` for the full jaxpr, ``order_tokens`` for the compact
order, ``pretty_stream`` for a raw-jaxpr view).

LEGACY: :class:`VEJaxpr` -- the older state/order-suffix tokenizer, kept only
because ``graphax.core.extract_jaxpr`` and the alphagrad env still consume it.
Do not build on it. (The HLO tokenizer ``VEHlo`` has been removed.)
"""
import functools
import itertools
import math
import os
import zlib
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import jax.numpy as jnp
from jax._src import core

from graphax.primitives import elemental_rules

# Param NAMES that get their own vocab token (readability: the key is spelled
# as one token instead of an opaque tag). This is NOT a capture filter --
# `get_params_tuple` captures every tokenizable non-None param regardless;
# names absent from this list are rendered as an opaque tag. New names must be
# APPENDED so existing token ids stay stable.
primitive_params = [
    "broadcast_dimensions",  # broadcast_in_dim
    "dimension",  # concatenate
    "dimension_numbers",  # dot_general
    "dimensions",  # reshape, squeeze, reduce
    "limit_indices",  # slice
    "new_sizes",  # reshape
    "permutation",  # transpose
    "shape",  # broadcast_in_dim
    "size",  # iota
    "start_indices",  # slice
    "strides",  # slice
    "y",  # integer_pow
    "axes",  # reduce_sum / reduce_max / ...
    "new_dtype",  # convert_element_type
    "weak_type",  # convert_element_type
    "dtype",  # iota, random bits
]

# TODO test transformer size


# Append-only path-elimination token stream markers (structure + approximation
# vocabulary). These are WORD tokens appended AFTER the base vocabulary so every
# existing token id is preserved. Sections: the base graph, then per elimination
# step an ``elim`` block (contraction + join, one combined jaxpr) followed by
# zero or more ``approx`` blocks. New markers MUST be appended, never inserted.
STRUCTURE_TOKENS = ["inputs", "outputs", "jac", "fns", "elim", "approx",
                    "lhs", "rhs", "res", "---", "path", "face", "in"]

# One special token per approximation TYPE (Diag / Compress / Quant) ...
from graphax.sparse.micro_actions import (  # noqa: E402
    COMPRESS_KINDS as _COMPRESS_KINDS,
    QUANT_DTYPES as _QUANT_DTYPES,
)

APPROX_TYPE_TOKENS = ["DIAG", "COMPRESS", "QUANT"]
# ... and one per STRING argument of Compress (kind) / Quant (dtype). Diag's
# i/j/factor and Compress's axes stay NUMBERS (the jaxpr.py digit scheme). The
# ``d#<dtype>`` tokens double as the rendering of ANY dtype-valued op param.
APPROX_ARG_TOKENS = ["k#" + k for k in _COMPRESS_KINDS] + \
                    ["d#" + d for d in _QUANT_DTYPES]

# Additional op PARAM-KEY tokens (so they spell as one token, not a crc tag).
# Appended at the very END so every existing token id is preserved.
EXTRA_PARAM_KEYS = [
    "padding_config", "rounding_method", "preferred_element_type", "precision",
    "accuracy", "fill_value", "index_vector_dim", "slice_sizes", "offset_dims",
    "collapsed_slice_dims", "start_index_map", "operand_batching_dims",
    "start_indices_batching_dims", "scatter_dims_to_operand_dims",
    "update_window_dims", "inserted_window_dims", "mode", "unique_indices",
    "indices_are_sorted", "num_consts", "jaxpr_type", "update_jaxpr",
]

# Common lax primitive NAMES not covered by ``elemental_rules`` (so a bare op
# spells as one token), plus the non-finite float literals (reduction
# identities: max-init ``-inf``, min-init ``inf``).
EXTRA_OP_NAMES = [
    "and", "or", "xor", "not", "rem", "clamp", "sign", "is_finite", "abs",
    "reduce_and", "reduce_or", "reduce_prod", "reduce_min", "reduce_max",
    "argmax", "argmin", "cumsum", "cumprod", "cummax", "cummin", "rev", "sort",
    "population_count", "clz", "nextafter", "shift_left", "reduce_precision",
    "shift_right_logical", "shift_right_arithmetic", "bitcast_convert_type",
    "real", "imag", "conj", "expand_dims", "copy", "select_n", "gt", "lt",
    "ge", "le", "eq", "ne", "max", "min", "round", "reduce_sum", "pad", "iota",
    "inf", "nan",
]


@functools.lru_cache(maxsize=None)
def _build_vocab(digit_base):
    _raw = (
        list("-½.?<>[](){}\n,;*_|~=:&")
        + [e.name for e in elemental_rules]
        + primitive_params
        + STRUCTURE_TOKENS
        + APPROX_TYPE_TOKENS
        + APPROX_ARG_TOKENS
        + EXTRA_PARAM_KEYS
        + EXTRA_OP_NAMES
    )
    # DEDUPE (keep first occurrence): several EXTRA_OP_NAMES also appear in
    # ``elemental_rules``; duplicate keys would inflate the enumerate() id range
    # past len(vocab), pushing a valid token id above ``_L`` where decode
    # mistakes it for a variable name. Dense, unique ids only.
    seen = set()
    full_params = []
    for p in _raw:
        if p not in seen:
            seen.add(p)
            full_params.append(p)

    # during training we can sample number of variables, nmber of functions, and digits we encounter, to better decide the params
    # TODO: allow to (define and) split the names of vars and fns
    # TODO: this is super slow, write this in c++ or rust.
    # TODO: make the full environment in c++ or rust.
    vocab = {p: i for i, p in enumerate(full_params)}
    n_vocab = {v: k for k, v in vocab.items()}
    return vocab, n_vocab, tuple(full_params)


def get_vocab(digit_base: int = 10):
    """(vocab, n_vocab, full_params) for ``digit_base``. Cached and read-only --
    callers must NOT mutate the returned dicts (they are shared)."""
    return _build_vocab(digit_base)


class VEJaxpr:
    """LEGACY state/order tokenizer -- superseded by
    :class:`IncrementalPathTokenizer`. Retained only for the existing consumers
    (``graphax.core.extract_jaxpr`` and the alphagrad env); do not build on it."""

    def __init__(
        self,
        jaxpr: core.Jaxpr,
        *,
        vocab_size: int = 256,
        vocab_size_fns: int = None,
        digit_base: int = 10,
        elim_order: Sequence[int] = None,
        transforms: Sequence = None,
    ):
        self.jaxpr = jaxpr
        self.digit_base = digit_base
        self.vocab_size = vocab_size
        self.vocab_size_fns = vocab_size_fns
        self.vocab, self.n_vocab, _ = get_vocab(digit_base)
        self._names = self._assign_names(vocab_size)
        self._tokens = None
        # Append-only STATE tokenization: when `elim_order` is provided this
        # VEJaxpr wraps the ORIGINAL (un-eliminated) jaxpr together with the
        # elimination-order prefix that has been applied. See `tokenized()`.
        self.elim_order = tuple(int(v) for v in elim_order) if elim_order is not None else None
        # Append-only per-vertex MICRO-ACTION encoding (DIAG/COMPRESS/QUANT).
        # ``transforms`` is the normalised graphax typed-transform structure
        # ``((vertex_id, (transform_obj, ...)), ...)`` -- the same object the
        # measure path passes to ``vertex_elimination_jaxpr``. Together with
        # ``elim_order`` this is a LOSSLESS sufficient statistic for the
        # APPROXIMATED partial-elimination state the policy sees: the order
        # says WHICH vertices were eliminated and in what sequence, the
        # transforms say WHAT micro-action(s) were applied at each. Stored
        # verbatim (already hashable: Diag/Compress/Quant are frozen
        # dataclasses) and consumed append-only in ``_state_tokenized``.
        self.elim_transforms = tuple(transforms) if transforms else ()

        # Pre-calculate token fragments for hotspots
        self._invar_tokens = self._precalculate_invar_tokens()
        self._eqn_tokens_cache = {}  # (def_fns, show_params, show_shapes) -> tokens

    def _precalculate_invar_tokens(self) -> List[int]:
        tokens = []
        _names = self._names
        for v in self.jaxpr.invars:
            name = _names[v]
            self._tokenize_sequence(name, tokens)
            shape_symbols = self._format_shape(v.aval.shape)
            self._tokenize_sequence(shape_symbols, tokens)
        return tokens

    def _assign_names(self, vocab_size):
        freq_map_vars: dict[Any, int] = {}
        freq_map_fns: dict[Any, int] = {}
        # Single linear stream of (kind, key) in TRUE first-appearance order,
        # used by the stable (append-only) naming scheme. `kind` is "var" or
        # "fn". Insertion order = position in the linearised jaxpr, which is
        # PREFIX-STABLE across vertex-elimination steps (each step appends
        # equations to the end), so the assigned names are append-only.
        appearance: list[tuple[str, Any]] = []

        def _see_var(v):
            if v not in freq_map_vars:
                freq_map_vars[v] = 0
                appearance.append(("var", v))
            freq_map_vars[v] += 1

        def _see_fn(key):
            if key not in freq_map_fns:
                freq_map_fns[key] = 0
                appearance.append(("fn", key))
            freq_map_fns[key] += 1

        def scan_jaxpr(j):
            for v in j.constvars:
                _see_var(v)
            for v in j.invars:
                _see_var(v)
            for eqn in j.eqns:
                pt = get_params_tuple(eqn)
                if pt:
                    _see_fn((eqn.primitive.name, pt))

                for v in eqn.outvars:
                    _see_var(v)

                for p_val in eqn.params.values():
                    if isinstance(p_val, core.Jaxpr):
                        scan_jaxpr(p_val)
                    elif isinstance(p_val, (list, tuple)):
                        for item in p_val:
                            if isinstance(item, core.Jaxpr):
                                scan_jaxpr(item)

        scan_jaxpr(self.jaxpr)

        freq_map_vars = {
            k: v for k, v in freq_map_vars.items() if not isinstance(k, core.DropVar)
        }

        num_extra_tokens = vocab_size - len(self.vocab)
        if num_extra_tokens <= 0:
            raise ValueError(
                f"vocab_size ({vocab_size}) is too small for the base vocabulary ({len(self.vocab)}). "
                f"Increase vocab_size to at least {len(self.vocab) + 1}."
            )

        # ------------------------------------------------------------------ #
        # Naming scheme selection.
        #
        # STABLE (default): assign names by FIRST-APPEARANCE order. A var/fn's
        # name depends only on its position in the linearised jaxpr, never on
        # global frequency. Because vertex elimination only ever APPENDS
        # equations, the appearance prefix is invariant step-to-step, so the
        # token stream becomes append-only -> incrementally cacheable. This is
        # the fix for the per-step re-tokenization churn.
        #
        # FREQUENCY (GRAPHAX_FREQ_NAMING=1): the original scheme -- most
        # frequent var/fn gets the shortest token. Better single-shot token
        # economy, but renames almost everything when an equation is added.
        # ------------------------------------------------------------------ #
        # STABLE naming is the DEFAULT: it is what makes the stream a pure
        # prefix extension across elimination steps (the append-only goal).
        # GRAPHAX_FREQ_NAMING=1 restores legacy frequency naming (A-B knob).
        stable = os.environ.get("GRAPHAX_FREQ_NAMING", "0") != "1"

        if self.vocab_size_fns is None:
            # Shared pool: vars and fns draw names from one token range.
            if stable:
                # Walk the single appearance stream in order. DropVars were
                # filtered from freq_map_vars; skip them here too.
                ordered = [
                    k
                    for kind, k in appearance
                    if kind == "fn" or k in freq_map_vars
                ]
            else:
                freq_map_combined = {**freq_map_vars}
                for k, v in freq_map_fns.items():
                    freq_map_combined[k] = freq_map_combined.get(k, 0) + v
                ordered = [
                    k
                    for k, _ in sorted(
                        freq_map_combined.items(), key=lambda x: x[1], reverse=True
                    )
                ]

            names = {
                k: n
                for k, n in zip(
                    ordered,
                    name_gen_python_style(
                        self.digit_base, self.digit_base + num_extra_tokens
                    ),
                )
            }
        else:
            # Split pool: fns and vars draw from disjoint token ranges. Each
            # pool is ordered independently; in stable mode each pool's
            # appearance order is itself prefix-stable.
            if stable:
                ordered_fns = [k for kind, k in appearance if kind == "fn"]
                ordered_vars = [
                    k for kind, k in appearance if kind == "var" and k in freq_map_vars
                ]
            else:
                ordered_fns = [
                    k
                    for k, _ in sorted(
                        freq_map_fns.items(), key=lambda x: x[1], reverse=True
                    )
                ]
                ordered_vars = [
                    k
                    for k, _ in sorted(
                        freq_map_vars.items(), key=lambda x: x[1], reverse=True
                    )
                ]

            fns_slots = min(self.vocab_size_fns, num_extra_tokens)

            names = {}
            # Assign function names
            for k, n in zip(
                ordered_fns,
                name_gen_python_style(self.digit_base, self.digit_base + fns_slots),
            ):
                names[k] = n

            # Assign variable names
            for k, n in zip(
                ordered_vars,
                name_gen_python_style(
                    self.digit_base + fns_slots, self.digit_base + num_extra_tokens
                ),
            ):
                names[k] = n

        missing_vars = [v for v in freq_map_vars if v not in names]
        missing_fns = [f for f in freq_map_fns if f not in names]

        if missing_vars or missing_fns:
            raise ValueError(
                f"vocab_size ({vocab_size}) is too small to name all variables and functions. "
                f"Variables missing: {len(missing_vars)}, Functions missing: {len(missing_fns)}. "
                "Increase vocab_size or decrease digit_base."
            )

        return names

    def _tokenize_sequence(self, seq: Iterable[str], tokens: List[int]):
        _len_vocab = len(self.vocab)
        _vocab = self.vocab
        for c in seq:
            if c in _vocab:
                tokens.append(_vocab[c])
            else:
                tokens.append(_len_vocab + int(c, 16))

    def _format_shape(self, shape) -> List[str]:
        if not shape:
            return []
        res = ["<"]
        for i, d in enumerate(shape):
            if i > 0:
                res.append("*")
            res.extend(int_to_base(d, self.digit_base))
        res.append(">")
        return res

    def __repr__(self):
        return self._repr()

    def _repr(
        self, *, deliminated=False, human_readable=True, tokenizer_kwargs=None
    ) -> str:
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        _len_vocab = len(self.vocab)
        token_strings = []
        for t in self.tokenized(**tokenizer_kwargs):
            t_int = int(t)
            if t_int < _len_vocab:
                token_strings.append(escape_token(self.n_vocab[t_int]))
            else:
                token_strings.append(hex(t_int - _len_vocab))

        res = "#" if deliminated else ""
        res = res.join(token_strings)
        if human_readable:
            res = res.replace("~", "\n~\n").replace("{", " {\n").replace("}", "\n}\n")

        return res

    def _get_atom_name(self, atom: Any) -> Union[str, Tuple[str, ...]]:
        if isinstance(atom, core.Literal):
            return _literal_symbols(atom.val, self.digit_base)

        ret = self._names[atom]
        return ret

    def __getattr__(self, name: str) -> Any:
        return getattr(self.jaxpr, name)

    def _state_tokenized(
        self, def_fns: bool, show_params: bool, show_shapes: bool
    ) -> jnp.ndarray:
        """Append-only state stream:

            <original-graph tokens> | <order prefix> | <per-vertex micro-actions>

        The original-graph block is produced by the ordinary `tokenized()`
        path over the SAME (un-eliminated) jaxpr -- invariant across steps --
        then we append a separator and one token per eliminated vertex id,
        then a second separator and the per-vertex micro-action (DIAG /
        COMPRESS / QUANT) encoding. Because vertex elimination only ever
        appends one vertex (and its micro-actions) per step, BOTH the order
        suffix and the micro-action suffix grow append-only, so the whole
        stream is a pure prefix extension of the previous step.

        Losslessness: ``(original graph, order prefix, per-vertex
        micro-actions)`` is a sufficient statistic for the approximated
        partial-elimination state, and the encoding below is injective ->
        distinct ``(order, transforms)`` map to distinct streams.
        """
        # The invariant original-graph block, via the plain graph emitter.
        base = self._graph_tokenized(def_fns, show_params, show_shapes)
        tokens = [int(t) for t in base]

        _vocab = self.vocab
        _base = self.digit_base

        def _emit_int(n):
            # Emit a (possibly negative) integer in the same base/overflow
            # scheme tokenized values use, so ids stay in the model vocab.
            for c in int_to_base(int(n), _base):
                if c in _vocab:
                    tokens.append(_vocab[c])
                else:
                    tokens.append(len(_vocab) + int(c, 16))

        # Lazily import the micro-action types (only needed when transforms
        # are present); keeps the no-transform path import-free.
        if self.elim_transforms:
            from .sparse.micro_actions import (
                Compress,
                Diag,
                Quant,
                COMPRESS_KIND_INDEX,
                QUANT_DTYPE_INDEX,
            )
            # vertex -> its transform tuple, for O(1) lookup while we walk the
            # elimination order. (transforms is keyed by vertex id.)
            _micro_by_v = {int(v): ts for v, ts in self.elim_transforms}
        else:
            _micro_by_v = {}

        def _emit_micro(vtransforms):
            # Encode one vertex's micro-actions. Layout per action:
            #   "*" i "_" j "_" factor          DIAG    Diag(i, j, factor)
            #   "_" kind_idx ("_" axis)*        COMPRESS Compress(axes, kind)
            #   "~" dtype_idx                   QUANT   Quant(dtype)
            # The leading char is a TYPE TAG, so a DIAG factor is never
            # confused with a COMPRESS axis -> the encoding is injective.
            for ti, t in enumerate(vtransforms):
                if ti:
                    tokens.append(_vocab[","])
                if isinstance(t, Diag):
                    tokens.append(_vocab["*"])
                    _emit_int(t.i)
                    tokens.append(_vocab["_"])
                    _emit_int(t.j)
                    tokens.append(_vocab["_"])
                    _emit_int(t.factor)
                elif isinstance(t, Compress):
                    tokens.append(_vocab["_"])
                    _emit_int(COMPRESS_KIND_INDEX[t.kind])
                    for a in t.axes:
                        tokens.append(_vocab["_"])
                        _emit_int(int(a))
                elif isinstance(t, Quant):
                    tokens.append(_vocab["~"])
                    _emit_int(QUANT_DTYPE_INDEX[t.dtype])
                else:
                    # Opaque callable transform: encode a stable tag so the
                    # stream still differs from the no-transform state. (Not
                    # used by the alphagrad env, which only passes the three
                    # typed micro-actions above.) crc32, NOT hash(): str hash
                    # is salted per process, and trainer + measure actors must
                    # encode the identical state identically.
                    tokens.append(_vocab["<"])
                    _emit_int(_stable_tag(repr(t)))
                    tokens.append(_vocab[">"])

        # ------------------------------------------------------------------ #
        # APPEND-ONLY interleaved suffix. After a single "|" separator we walk
        # the elimination ORDER and emit, per eliminated vertex, one block:
        #     <vertex-id> [ ":" <micro-actions> ] ";"
        # The micro-action sub-block is present iff that vertex has transforms.
        # Crucially the order id and its micro-actions are emitted TOGETHER, so
        # advancing the elimination by one vertex appends exactly one trailing
        # block and never shifts an earlier one -> the stream is a pure prefix
        # extension step-to-step (CACHEABLE). Interleaving (rather than two
        # separate order/micro segments) is what makes it append-only: a second
        # segment would be pushed right every time the order grows.
        #
        # Losslessness: (original graph, order, per-vertex micro-actions) is a
        # sufficient statistic for the approximated partial-elimination state,
        # and this encoding is injective in (order, transforms).
        # ------------------------------------------------------------------ #
        tokens.append(_vocab["|"])
        for vid in self.elim_order:
            _emit_int(int(vid))
            vmicro = _micro_by_v.get(int(vid))
            if vmicro:
                tokens.append(_vocab[":"])
                _emit_micro(vmicro)
            tokens.append(_vocab[";"])
        if self.elim_order:
            tokens.pop()  # trailing separator after the last vertex block

        res = jnp.array(tokens, dtype=jnp.int32)
        return res

    def tokenized(
        self,
        def_fns: bool = True,
        show_params: bool = True,
        show_shapes: bool = False,
    ) -> jnp.ndarray:
        if self._tokens is not None:
            return self._tokens

        cache_key = (def_fns, show_params, show_shapes)
        if cache_key in self._eqn_tokens_cache:
            return self._eqn_tokens_cache[cache_key]

        # ------------------------------------------------------------------ #
        # APPEND-ONLY STATE tokenization.
        #
        # When this VEJaxpr was built over the ORIGINAL (un-eliminated) jaxpr
        # plus an elimination-order prefix, emit:
        #     <tokens of the original graph>  |  <elimination order prefix>
        # The original-graph block is INVARIANT across elimination steps (it
        # is the same jaxpr every step, named by the prefix-stable scheme), so
        # the only thing that grows step-to-step is the order suffix -> the
        # stream is APPEND-ONLY and incrementally cacheable.
        #
        # Faithfulness: the partial-eliminated Jacobian graph is a determinis-
        # tic function of (original graph, order-prefix); this pair is a loss-
        # less sufficient statistic for the policy state. It is in fact MORE
        # faithful than the re-traced Jacobian jaxpr, which discards the
        # vertex-id correspondence the action space is defined over.
        # ------------------------------------------------------------------ #
        if self.elim_order is not None:
            res = self._state_tokenized(def_fns, show_params, show_shapes)
        else:
            res = self._graph_tokenized(def_fns, show_params, show_shapes)
        self._eqn_tokens_cache[cache_key] = res
        return res

    def _graph_tokenized(
        self,
        def_fns: bool,
        show_params: bool,
        show_shapes: bool,
    ) -> jnp.ndarray:
        """Emit the plain (no elimination-order) graph stream. Uncached --
        `tokenized()` owns the cache."""
        _vocab = self.vocab
        _len_vocab = len(self.vocab)
        _names = self._names
        _base = self.digit_base

        tokens = list(self._invar_tokens)

        # Collect the parameterized-op definitions in a DETERMINISTIC order.
        # Previously this used a `set`, whose iteration order is hash/identity
        # dependent and therefore reshuffled the `~ ...` function-definition
        # block from step to step (a second, independent source of token churn
        # on graphs with parameterized primitives such as dot_general). We
        # dedupe while preserving first-appearance order so the block is stable
        # and append-only as new parameterized ops are introduced.
        parameterized_ops = []
        _seen_ops = set()
        for eqn in self.jaxpr.eqns:
            pt = get_params_tuple(eqn)
            if pt:
                key = (eqn.primitive.name, pt)
                if key not in _seen_ops:
                    _seen_ops.add(key)
                    parameterized_ops.append((_names[key], eqn.primitive.name, pt))

        if def_fns and parameterized_ops:
            tokens.append(_vocab["~"])
            for op_name, prim_name, pt in parameterized_ops:
                self._tokenize_sequence(op_name, tokens)

                if prim_name in _vocab:
                    tokens.append(_vocab[prim_name])
                else:
                    _emit_symbols(_opaque_symbols(prim_name, _base), _vocab, tokens)

                if show_params:
                    tokens.append(_vocab["["])
                    for i, (k, v) in enumerate(pt):
                        if k in _vocab:
                            tokens.append(_vocab[k])
                        else:
                            _emit_symbols(_opaque_symbols(k, _base), _vocab, tokens)

                        tokenize_value(v, _vocab, tokens, _base)
                        if i < len(pt) - 1:
                            tokens.append(_vocab["|"])
                    tokens.append(_vocab["]"])

        tokens.append(_vocab["{"])

        # 3. Equations
        for eqn in self.jaxpr.eqns:
            pt = get_params_tuple(eqn)

            # All non-dropped output vars, '_'-separated. Naming only
            # outvars[0] made multi-output equations lose their remaining
            # outputs from the stream entirely.
            outs = [v for v in eqn.outvars if not isinstance(v, core.DropVar)]
            if not outs:
                continue

            for i, ov in enumerate(outs):
                if i:
                    tokens.append(_vocab["_"])
                self._tokenize_sequence(_names[ov], tokens)

            if show_shapes:
                for v in outs:
                    shape_symbols = self._format_shape(v.aval.shape)
                    self._tokenize_sequence(shape_symbols, tokens)

            if pt and def_fns:
                if self.vocab_size_fns is None:
                    tokens.append(_vocab["="])
                op_name = _names[(eqn.primitive.name, pt)]
                self._tokenize_sequence(op_name, tokens)
            else:
                prim_name = eqn.primitive.name
                if prim_name in _vocab:
                    tokens.append(_vocab[prim_name])
                else:
                    _emit_symbols(_opaque_symbols(prim_name, _base), _vocab, tokens)

                if pt and show_params:
                    tokens.append(_vocab["["])
                    for i, (k, v) in enumerate(pt):
                        if k in _vocab:
                            tokens.append(_vocab[k])
                        else:
                            _emit_symbols(_opaque_symbols(k, _base), _vocab, tokens)
                        tokenize_value(v, _vocab, tokens, _base)
                        if i < len(pt) - 1:
                            tokens.append(_vocab["|"])
                    tokens.append(_vocab["]"])

            if eqn.invars:
                for i, inv in enumerate(eqn.invars):
                    if i != 0:
                        tokens.append(_vocab["_"])
                    else:
                        if pt and def_fns:
                            if self.vocab_size_fns is None:
                                tokens.append(_vocab[":"])

                    atom_name = self._get_atom_name(inv)
                    if isinstance(atom_name, (tuple, list)):
                        self._tokenize_sequence(atom_name, tokens)
                    elif atom_name in _vocab:
                        tokens.append(_vocab[atom_name])
                    else:
                        tokens.extend(_vocab[c] for c in atom_name if c in _vocab)

            tokens.append(_vocab["\n"])

        if tokens and tokens[-1] == _vocab["\n"]:
            tokens.pop()

        tokens.append(_vocab["}"])

        # 4. Outputs
        for v in self.jaxpr.outvars:
            p = self._get_atom_name(v)
            if isinstance(p, (tuple, list)):
                self._tokenize_sequence(p, tokens)
            elif p in _vocab:
                tokens.append(_vocab[p])
            else:
                tokens.extend(_vocab[c] for c in p if c in _vocab)
            tokens.append(_vocab[";"])

        if self.jaxpr.outvars:
            tokens.pop()

        return jnp.array(tokens, dtype=jnp.int32)



def escape_token(t: str) -> str:
    if t == "\n":
        return "\n"
    return t


def name_gen_python_style(start_offset, end_offset):
    vocab_range = end_offset - start_offset
    if vocab_range <= 0:
        return
    for i in itertools.count(1):
        for p in itertools.product(
            [hex(j + start_offset) for j in range(vocab_range)], repeat=i
        ):
            yield tuple(p)


def int_to_base(n: int, base: int) -> Tuple[str, ...]:
    if n == 0:
        return (hex(0),)
    res = []
    is_negative = n < 0
    n = abs(n)
    if base == 1:
        res = [hex(0)] * n
    else:
        while n:
            res.append(hex(n % base))
            n //= base
    if is_negative:
        return ("-",) + tuple(reversed(res))
    return tuple(reversed(res))


# --------------------------------------------------------------------------- #
# Value encoding.
#
# The vocabulary has no letter tokens, so anything the character set cannot
# spell used to be SILENTLY DROPPED -- distinct values tokenized identically
# (e.g. every dtype vanished, `sum(axis=0)` == `sum(axis=1)`). Every encoder
# below is total: a value always leaves a deterministic, distinct trace.
# --------------------------------------------------------------------------- #


def _stable_tag(s: Union[str, bytes]) -> int:
    """Deterministic 32-bit tag. crc32, NOT hash(): str hash is salted per
    process, and trainer + measure actors must tokenize identically."""
    if isinstance(s, str):
        s = s.encode("utf-8")
    return zlib.crc32(s)


_UNCAPTURED = set()   # (kind, string) pairs already warned about


def _opaque_symbols(s: str, base: int, kind: str = "value") -> Tuple[str, ...]:
    """A single ``?`` PLACEHOLDER for something the tokenizer could not spell,
    plus ONE warning per new (kind, string) -- so every distinct kind of gap
    (an uncaptured op name / param key / value / type) is surfaced exactly once
    and is easy to find and fix."""
    key = (kind, s)
    if key not in _UNCAPTURED:
        _UNCAPTURED.add(key)
        import warnings
        warnings.warn(
            f"jaxpr tokenizer: uncaptured {kind} rendered as '?': {s!r}")
    return ("?",)


def _float_symbols(f: float, base: int) -> Tuple[str, ...]:
    """Positional-decimal float encoding: sign, integer digits, ``.`` decimal
    point, fraction digits (leading zeros preserved: 0.05 -> 0.05). ±inf / nan
    are readable tokens. Fraction digits are emitted one decimal digit at a time.
    """
    if not math.isfinite(f):
        if math.isnan(f):
            return ("nan",)
        return ("-", "inf") if f < 0 else ("inf",)
    s = np.format_float_positional(abs(f), trim="-")
    int_part, _, frac_part = s.partition(".")
    syms: List[str] = ["-"] if f < 0 else []
    syms += int_to_base(int(int_part or "0"), base)
    if frac_part:
        syms.append(".")
        for ch in frac_part:
            syms += int_to_base(int(ch), base)
    return tuple(syms)


def _is_dtype_like(v: Any) -> bool:
    return isinstance(v, np.dtype) or (
        isinstance(v, type) and issubclass(v, np.generic)
    )


def _dtype_symbols(dt: Any, base: int) -> Tuple[str, ...]:
    """``~`` bits ``_`` name-tag. Bit width alone collides across the float8 /
    bfloat16 family, so a short (mod 997) name tag disambiguates. Compact and
    define-once amortized: dtypes appear almost only inside fn definitions."""
    dt = np.dtype(dt)
    return (
        ("~",)
        + int_to_base(dt.itemsize * 8, base)
        + ("_",)
        + int_to_base(_stable_tag(dt.name) % 997, base)
    )


def _literal_symbols(val: Any, base: int) -> Tuple[str, ...]:
    """Symbol encoding of a jaxpr Literal's value.

    Scalars keep their actual digits (a policy can relate 2 and 3; it cannot
    relate two opaque tags). Non-scalar arrays -- previously collapsed to
    their FIRST ELEMENT -- become shape + a content tag: injective, bounded.
    """
    arr = np.asarray(val)
    if arr.ndim == 0:
        x = arr.item()
        if isinstance(x, (bool, int)):
            return int_to_base(int(x), base)
        if isinstance(x, float):
            if math.isfinite(x) and x.is_integer():
                return int_to_base(int(x), base)
            return _float_symbols(x, base)
        return _opaque_symbols(repr(x), base, kind="literal")
    syms: List[str] = ["<"]
    for i, d in enumerate(arr.shape):
        if i:
            syms.append("*")
        syms += int_to_base(int(d), base)
    syms.append(">")
    syms += int_to_base(
        _stable_tag(arr.tobytes() + str((arr.shape, str(arr.dtype))).encode()),
        base,
    )
    return tuple(syms)


def _emit_symbols(symbols: Iterable[str], vocab: Dict[str, int], tokens: List[int]):
    """Append token ids for a symbol sequence: vocab tokens by lookup, digit
    symbols (hex strings from `int_to_base`) into the digit range above."""
    _len_vocab = len(vocab)
    for c in symbols:
        if c in vocab:
            tokens.append(vocab[c])
        else:
            tokens.append(_len_vocab + int(c, 16))


def tokenize_value(val: Any, vocab: Dict[str, int], tokens: List[int], base: int = 10):
    """Tokenize one param value. Total by construction (see block comment)."""
    if isinstance(val, (bool, int, np.integer, jnp.integer)):
        _emit_symbols(int_to_base(int(val), base), vocab, tokens)
    elif isinstance(val, (float, np.floating)):
        f = float(val)
        if math.isfinite(f) and f.is_integer():
            _emit_symbols(int_to_base(int(f), base), vocab, tokens)
        else:
            _emit_symbols(_float_symbols(f, base), vocab, tokens)
    elif _is_dtype_like(val):
        # prefer the single ``d#<name>`` token (shared with the QUANT dtypes);
        # fall back to the ``~bits_tag`` encoding for exotic dtypes.
        tok = "d#" + np.dtype(val).name
        if tok in vocab:
            tokens.append(vocab[tok])
        else:
            _emit_symbols(_dtype_symbols(val, base), vocab, tokens)
    elif isinstance(val, (tuple, list)):
        tokens.append(vocab["("])
        for i, v in enumerate(val):
            if i:
                tokens.append(vocab[","])
            tokenize_value(v, vocab, tokens, base)
        tokens.append(vocab[")"])
    else:
        # strings / enums / None-inside-tuples / anything else: spell it IFF every
        # char is representable, else a single clean ``?`` placeholder (no mixed
        # partial-spelling + tag).
        s = str(val).replace(" ", "")
        if all(c in vocab or c.isdigit() for c in s):
            for c in s:
                if c in vocab:
                    tokens.append(vocab[c])
                else:
                    _emit_symbols(int_to_base(int(c), base), vocab, tokens)
        else:
            _emit_symbols(_opaque_symbols(s, base, kind="value"), vocab, tokens)


def _tokenizable_param(v: Any) -> bool:
    """True if a param VALUE can enter the stream (and the fn-identity key).

    Excludes sub-jaxprs and callables (rendered elsewhere / identity-hashed)
    and anything unhashable (fn keys must be dict keys)."""
    if isinstance(v, (core.Jaxpr, core.ClosedJaxpr)):
        return False
    if callable(v) and not isinstance(v, type):
        return False
    if isinstance(v, (tuple, list)):
        return all(_tokenizable_param(x) for x in v)
    try:
        hash(v)
    except TypeError:
        return False
    return True


def get_params_tuple(eqn, tokenized_param_names=None):
    """All tokenizable, non-None params of `eqn`, sorted by key.

    Captures EVERY significant param -- a whitelist here silently erased
    semantics (integer_pow[y], reduce axes, convert_element_type's new_dtype
    all vanished, so e.g. x**2 and x**3 tokenized identically). The
    `tokenized_param_names` arg is kept for call-site compatibility and
    ignored; `primitive_params` only decides which param NAMES get their own
    vocab token."""
    valid = []
    for k in sorted(eqn.params.keys()):
        v = eqn.params[k]
        if v is None or not _tokenizable_param(v):
            continue
        valid.append((k, v))
    return tuple(valid)


# =========================================================================== #
# Append-only PATH-ELIMINATION token stream (see IncrementalPathTokenizer).
#
# The Jacobian is built incrementally on ONE persistent trace
# (graphax.incremental.IncrementalJaxpr); each elimination step APPENDS its
# real SparseTensor contraction + join to the growing jaxpr. This tokenizer
# renders that as base + one ``path`` block per face (+ ``approx`` blocks). Var
# names follow the jaxpr.py scheme (`name_gen_python_style` hex atoms,
# first-appearance order); parameterized ops are defined once as functions.
# =========================================================================== #


def _significant_params(params: Dict[str, Any]):
    """The params that enter the token stream / fn identity: non-None,
    tokenizable, not a sub-jaxpr, sorted by key. Single source of truth for BOTH
    the fn key and rendering, so the two cannot drift."""
    return [(k, v) for k, v in sorted(params.items())
            if v is not None and k not in ("jaxpr", "call_jaxpr")
            and _tokenizable_param(v)]


def _op_fn_key(prim: str, params: Dict[str, Any]):
    """Hashable (op, sorted-repr-params) identity for the define-once fn set;
    empty tuple => parameterless (written inline). Keyed on the SAME params as
    rendering (:func:`_significant_params`) so two ops that render identically
    get the same fn."""
    return tuple((k, repr(v)) for k, v in _significant_params(params))


class IncrementalPathTokenizer:
    """Renders the incremental Jacobian as an append-only token stream.

    NOTE on vocabulary size: unlike :class:`VEJaxpr` this tokenizer does NOT cap
    the id space -- variable/function names are drawn open-endedly from
    ``name_gen_python_style`` and a token id can be as large as
    ``len(vocab) + <num distinct names>`` (see :meth:`max_token_id`). A consumer
    with a fixed embedding table must size it from ``max_token_id()`` (or bound
    the graph), not from ``len(get_vocab())``.
    """

    def __init__(self, jaxpr, argnums, consts, args, digit_base: int = 10,
                 track_faces: bool = True):
        from graphax.incremental import IncrementalJaxpr
        self.ij = IncrementalJaxpr(jaxpr, argnums, consts, args,
                                   track_faces=track_faces)
        self.jaxpr = jaxpr
        self.argnums = tuple(argnums)
        self.digit_base = digit_base
        self.vocab, self.n_vocab, _ = get_vocab(digit_base)
        self._L = len(self.vocab)
        # ONE persistent Var -> hex name map (first appearance over the growing
        # equation list) and ONE append-only fn registry.
        # ONE sequential name pool shared by variables AND functions -- names
        # are drawn from the same generator so a function name is indistinguish-
        # able from a variable name (the model must learn the distinction from
        # context: a name after ``fns`` / before ``=...:`` is a function).
        self._names = {}
        self._namegen = name_gen_python_style(digit_base, digit_base + 400000)
        # (prim, params-key) -> fn name. EVERY op with non-array parameters is
        # defined as a function (dot_general, reshape, ...); parameterless ops
        # (add, max, ...) are written inline.
        self._fns = {}
        self._n_steps = 0
        self._flatten_uid = 0   # unique id per inlined call (see _flatten)
        # GRAPH-NODE names: the original jaxpr variables (inputs, then every eqn
        # outvar in order) get stable low names, so a path RE-STATES the actual
        # variables it connects (central = the eliminated vertex's variable).
        # These share the name pool with the block equations' traced vars but are
        # assigned FIRST, so nodes get the memorable early names.
        for v in jaxpr.invars:
            self._var_name(v)
        for eqn in jaxpr.eqns:
            for ov in eqn.outvars:
                self._var_name(ov)

    # ---- token helpers -----------------------------------------------
    def _emit_atoms(self, atoms, out):
        _emit_symbols(atoms, self.vocab, out)

    def _emit_word(self, w, out):
        out.append(self.vocab[w])

    def _emit_int(self, n, out):
        self._emit_atoms(int_to_base(int(n), self.digit_base), out)

    def _var_name(self, v):
        nm = self._names.get(v)
        if nm is None:
            nm = next(self._namegen)
            self._names[v] = nm
        return nm

    def _emit_atom(self, atom, out):
        if isinstance(atom, core.Literal):
            self._emit_atoms(_literal_symbols(atom.val, self.digit_base), out)
        else:
            self._emit_atoms(self._var_name(atom), out)

    def _fn_key(self, eqn):
        return _op_fn_key(eqn.primitive.name, dict(eqn.params))

    # jnp ops lower to jit-wrapped primitives; flatten those away so a block
    # reads as bare `mul`/`clamp`/`select_n`, not opaque `jit[<shardings...>]`.
    _CALL_PRIMS = {"pjit", "jit", "closed_call", "core_call", "remat_call",
                   "remat2", "custom_jvp_call", "custom_vjp_call", "checkpoint"}

    def _flatten(self, eqns, subst=None, depth=0):
        """Normalize eqns to ``(outs, prim, params, ins)`` tuples, inlining call
        primitives (their sub-jaxpr's inner ops) so nothing renders as an opaque
        ``jit`` wrapper. ``subst`` remaps a call's inner invars/outvars to the
        outer operands so names stay connected across the inline boundary; the
        call's INTERNAL vars are remapped to per-call-unique proxies because jax
        reuses the same inner-jaxpr object (hence the same Var objects) across
        call sites -- without this the same name is assigned to two distinct
        computations."""
        if subst is None:
            subst = {}

        def _sub(v):                      # Literals are unhashable -> pass through
            return subst.get(v, v) if isinstance(v, core.Var) else v

        result = []
        for eqn in eqns:
            prim = eqn.primitive.name
            sub = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
            if prim in self._CALL_PRIMS and sub is not None and depth < 24:
                jx = sub.jaxpr if hasattr(sub, "jaxpr") else sub
                uid = self._flatten_uid
                self._flatten_uid += 1
                inner = dict(subst)
                for iv, a in zip(jx.invars, eqn.invars):
                    inner[iv] = _sub(a)
                for ov_in, ov_out in zip(jx.outvars, eqn.outvars):
                    if isinstance(ov_in, core.Var):
                        inner[ov_in] = _sub(ov_out)
                # every remaining internal var -> a proxy unique to THIS call
                # (leave DropVars alone so they stay filterable downstream)
                for e in jx.eqns:
                    for v in e.outvars:
                        if (isinstance(v, core.Var)
                                and not isinstance(v, core.DropVar)
                                and v not in inner):
                            inner[v] = (uid, v)
                result.extend(self._flatten(jx.eqns, inner, depth + 1))
            else:
                result.append(([_sub(v) for v in eqn.outvars], prim,
                               eqn.params, [_sub(v) for v in eqn.invars]))
        return result

    def _sig_params(self, params):
        return _significant_params(params)

    def _emit_op_params(self, prim, params, out):
        """``op[k0 v0 , k1 v1 , ...]`` from the REAL params (no repr round-trip).
        ``,`` separates parameters (== function arguments)."""
        if prim in self.vocab:
            out.append(self.vocab[prim])
        else:
            self._emit_atoms(_opaque_symbols(prim, self.digit_base, kind="op"), out)
        sig = self._sig_params(params)
        if sig:
            out.append(self.vocab["["])
            for i, (k, v) in enumerate(sig):
                if k in self.vocab:
                    out.append(self.vocab[k])
                else:
                    self._emit_atoms(_opaque_symbols(k, self.digit_base, kind="param-key"), out)
                tokenize_value(v, self.vocab, out, self.digit_base)
                if i < len(sig) - 1:
                    out.append(self.vocab[","])
            out.append(self.vocab["]"])

    def _emit_eqns(self, eqns, out):
        flat = self._flatten(eqns)
        # Pass 1: EVERY parameterized op is a function (defined once, on first
        # sight, from the shared name pool); parameterless ops stay inline.
        render = []          # per eqn: None | ('op', prim) inline | ('fn', name)
        new_defs = []
        for outs, prim, params, ins in flat:
            keep = [v for v in outs if not isinstance(v, core.DropVar)]
            if not keep:
                render.append(None)
                continue
            pk = _op_fn_key(prim, params)
            if not pk:                              # parameterless -> inline
                render.append(("op", prim))
            else:
                key = (prim, pk)
                name = self._fns.get(key)
                if name is None:
                    name = next(self._namegen)      # pooled with variable names
                    self._fns[key] = name
                    new_defs.append((name, prim, params))
                render.append(("fn", name))
        if new_defs:
            self._emit_word("fns", out)
            for name, prim, params in new_defs:
                self._emit_atoms(name, out)
                out.append(self.vocab["="])
                self._emit_op_params(prim, params, out)
        out.append(self.vocab["{"])
        # Equation forms:
        #   parameterless : out op arg0 _ arg1              (op is one token)
        #   fn reference  : out = F : arg0 _ arg1           (=/: mark a CALL)
        for (outs, prim, params, ins), r in zip(flat, render):
            if r is None:
                continue
            keep = [v for v in outs if not isinstance(v, core.DropVar)]
            for i, ov in enumerate(keep):
                if i:
                    out.append(self.vocab["_"])
                self._emit_atom(ov, out)
            if r[0] == "fn":
                out.append(self.vocab["="])
                self._emit_atoms(r[1], out)
                out.append(self.vocab[":"])
            elif prim in self.vocab:
                out.append(self.vocab[prim])
            else:
                self._emit_atoms(_opaque_symbols(prim, self.digit_base, kind="op"), out)
            for j, iv in enumerate(ins):
                if j:
                    out.append(self.vocab["_"])
                self._emit_atom(iv, out)
            out.append(self.vocab["\n"])
        if out and out[-1] == self.vocab["\n"]:
            out.pop()
        out.append(self.vocab["}"])

    # ---- public: base + per-step append ------------------------------
    def base_tokens(self):
        # inputs (+ shapes) then the base equations (primal forward + elemental
        # edge partials). No output/jac list: the Jacobian outputs ARE the
        # input->output edges produced by the path blocks; a reserved-name list
        # here was never referenced, so it is dropped.
        toks = []
        self._emit_word("inputs", toks)
        for ii in self.argnums:
            v = self.jaxpr.invars[ii]
            self._emit_atoms(self._var_name(v), toks)
            self._emit_atoms(self._format_shape(v.aval.shape), toks)
        self._emit_eqns(self.ij.base_eqns(), toks)
        return toks

    def _format_shape(self, shape):
        if not shape:
            return []
        res = ["<"]
        for i, d in enumerate(shape):
            if i:
                res.append("*")
            res.extend(int_to_base(d, self.digit_base))
        res.append(">")
        return res

    # ---- unified PATH emission (vertex- and face-elimination agnostic) ----
    def _emit_approx_head(self, atype, params, out):
        self._emit_word("approx", out)
        self._emit_word(atype, out)               # DIAG / COMPRESS / QUANT
        if atype == "QUANT":
            self._emit_word("d#" + params["dtype"], out)
        elif atype == "COMPRESS":
            self._emit_word("k#" + params["kind"], out)
            for ax in params.get("axes", ()):
                self._emit_int(ax, out)
        elif atype == "DIAG":
            self._emit_int(params["i"], out)
            self._emit_int(params["j"], out)
            self._emit_int(params["factor"], out)

    def eliminate(self, vertex, rules=()):
        """Eliminate one vertex on the PRESERVED trace and emit its PATH blocks.

        There is no ``elim`` header: vertex and face elimination are the same
        thing at this granularity — a sequence of paths. Vertex elimination
        eliminates the whole vertex, then emits its faces one path at a time;
        face elimination emits a single path. Each path is:

            path <central> , <pred> , <succ>   { contraction+join eqns }
            [ approx <TYPE> <args>             { approx eqns } ]*

        ``central`` / ``pred`` / ``succ`` are the actual graph VARIABLE names
        (central = the eliminated vertex's variable).
        """
        step_i = len(self.ij.steps)
        self.ij.eliminate(vertex, rules)
        toks = self._emit_step_paths(step_i, self.ij.all_eqns())
        self._n_steps += 1
        return toks

    @staticmethod
    def _split_face_eqns(fr, eqns):
        """A face -> ``(elim_eqns, [(atype, params, approx_eqns)])``: the
        contraction+join equations (the face range MINUS the approx sub-ranges)
        and each approximation's own equations. Single source of truth for both
        the token and pretty renderers."""
        aranges = [(s, e) for (_, _, s, e) in fr.approx]
        elim = [eqns[i] for i in range(fr.start, fr.end)
                if not any(s <= i < e for s, e in aranges)]
        approx = [(atype, params, [eqns[i] for i in range(s, e)])
                  for (atype, params, s, e) in fr.approx]
        return elim, approx

    def _emit_face_header(self, fr, out):
        """``path <central> & <pred> & <succ>`` (central is the eliminated var)."""
        self._emit_word("path", out)
        self._emit_atoms(self._var_name(fr.central), out)
        out.append(self.vocab["&"])
        self._emit_atoms(self._var_name(fr.in_edge), out)
        out.append(self.vocab["&"])
        self._emit_atoms(self._var_name(fr.out_edge), out)

    def _emit_face(self, fr, eqns, out):
        self._emit_face_header(fr, out)
        elim, approx = self._split_face_eqns(fr, eqns)
        self._emit_eqns(elim, out)
        for atype, params, sub in approx:
            self._emit_approx_head(atype, params, out)
            self._emit_eqns(sub, out)

    def _emit_step_paths(self, step_i, eqns):
        toks = []
        for fr in self.ij.step_faces(step_i):
            self._emit_face(fr, eqns, toks)
        return toks

    # ---- raw-jaxpr (human) rendering of the same structure ----------
    def _pp_name(self, v):
        """Readable name matching the token decode (``#a``/``#10``); literals as
        their value."""
        if isinstance(v, core.Literal):
            val = v.val
            try:
                f = float(val)
                return str(int(f)) if f == int(f) else str(val)
            except Exception:
                return str(val)
        return "".join("#" + format(int(a, 16), "x") for a in self._var_name(v))

    def _pp_params(self, params):
        sig = self._sig_params(params)
        if not sig:
            return ""
        parts = []
        for k, v in sig:
            if _is_dtype_like(v):
                vs = np.dtype(v).name
            else:
                vs = repr(v).replace("\n", " ")
            parts.append(f"{k}={vs}")
        return "[" + ", ".join(parts) + "]"

    def _pp_eqns(self, eqns, indent, out_lines):
        for outs, prim, params, ins in self._flatten(eqns):
            keep = [v for v in outs if not isinstance(v, core.DropVar)]
            if not keep:
                continue
            lhs = ",".join(self._pp_name(v) for v in keep)
            rhs = " ".join(self._pp_name(v) for v in ins)
            out_lines.append(f"{indent}{lhs} = {prim}{self._pp_params(params)} {rhs}")

    def pretty_stream(self, order, transforms=None):
        """The SAME base + per-path structure, but each block rendered as RAW
        jaxpr equations (``out = prim[params] ins``, jit inlined) instead of
        tokens. Returns a pretty-printed string. Var names (``#a``...) match the
        token decode so the two views cross-reference."""
        self.ij.eliminate_order(order, transforms)
        all_eqns = list(self.ij.all_eqns())
        L = []
        inv = "  ".join(f"{self._pp_name(self.jaxpr.invars[ii])}:"
                        f"{self.jaxpr.invars[ii].aval.str_short()}"
                        for ii in self.argnums)
        L.append(f"inputs: {inv}")
        self._pp_eqns(self.ij.base_eqns(), "  ", L)
        # ALL outputs in one place (after the base): value output(s) then the
        # Jacobian block per (output & input); block names are the vars the
        # paths fill.
        self._jac_vars = self._collect_jac_vars(all_eqns)
        vals = " ".join(self._pp_name(v) for v in self._value_output_vars())
        jparts = []
        for ov in self.jaxpr.outvars:
            for ii in self.argnums:
                var = self._jac_vars.get((ov, self.jaxpr.invars[ii]))
                blk = self._pp_name(var) if var is not None else "0"
                jparts.append(f"{self._pp_name(ov)}&"
                              f"{self._pp_name(self.jaxpr.invars[ii])}={blk}")
        L.append(f"outputs: value {vals}   jac  " + "  ".join(jparts))
        for fr in self.ij.all_faces():
            L.append(f"path {self._pp_name(fr.central)} & "
                     f"{self._pp_name(fr.in_edge)} & {self._pp_name(fr.out_edge)}")
            elim, approx = self._split_face_eqns(fr, all_eqns)
            self._pp_eqns(elim, "  ", L)
            for atype, params, sub in approx:
                L.append(f"  approx {atype} {self._pp_approx_args(atype, params)}")
                self._pp_eqns(sub, "    ", L)
        return "\n".join(L)

    def _pp_approx_args(self, atype, params):
        if atype == "QUANT":
            return params["dtype"]
        if atype == "COMPRESS":
            return params["kind"] + " " + ",".join(map(str, params.get("axes", ())))
        if atype == "DIAG":
            return f"{params['i']},{params['j']},{params['factor']}"
        return ""

    def order_tokens(self, order, transforms=None, per_line=False):
        """The elimination ORDER alone -- one ``path <central> & <pred> & <succ>``
        per face (plus any ``approx <TYPE> <args>`` heads), with NO equation
        bodies / fn definitions. This is the compact action sequence. With
        ``per_line`` a newline token separates paths (for readable decoding)."""
        self.ij.eliminate_order(order, transforms)
        toks = []
        for fr in self.ij.all_faces():
            if per_line and toks:
                toks.append(self.vocab["\n"])
            self._emit_face_header(fr, toks)
            for (atype, params, s, e) in fr.approx:
                self._emit_approx_head(atype, params, toks)
        return toks

    def _collect_jac_vars(self, eqns):
        """Map each (output, differentiable-input) to the frame variable that
        holds its FINAL Jacobian block -- the output var of the LAST face that
        produced that input->output edge. ``None`` if the block is zero."""
        in_set = {self.jaxpr.invars[ii] for ii in self.argnums}
        out_set = set(self.jaxpr.outvars)
        jac = {}
        for fr in self.ij.all_faces():
            if (fr.in_edge in in_set and fr.out_edge in out_set
                    and fr.end > fr.start):
                outs = [v for v in eqns[fr.end - 1].outvars
                        if not isinstance(v, core.DropVar)]
                if outs:
                    jac[(fr.out_edge, fr.in_edge)] = outs[-1]
        return jac

    def _value_output_vars(self):
        """Frame variables of the primal jaxpr's VALUE output(s) (e.g. the loss).
        ``to_jaxpr`` names the primal-output tracers; the returned vars are
        identity-matched to base equations, so ``_var_name`` links them."""
        outs = [self.ij.env[ov] for ov in self.jaxpr.outvars]
        jx = self.ij.trace.to_jaxpr(outs, self.ij.dbg, self.ij.si)[0]
        return list(jx.outvars)

    def _emit_outputs(self, toks):
        """State ALL outputs in ONE place, right after the base: first the VALUE
        output(s) (the primal result), then the Jacobian block per (output &
        input). Naming here reserves the block names before the paths run, so
        the path that computes each block references the SAME name (live)."""
        self._emit_word("outputs", toks)
        for vv in self._value_output_vars():        # value output(s)
            self._emit_atom(vv, toks)
        for ov in self.jaxpr.outvars:               # Jacobian blocks
            for ii in self.argnums:
                var = self._jac_vars.get((ov, self.jaxpr.invars[ii]))
                self._emit_atoms(self._var_name(ov), toks)          # output node
                toks.append(self.vocab["&"])
                self._emit_atoms(self._var_name(self.jaxpr.invars[ii]), toks)  # input
                toks.append(self.vocab["="])
                if var is not None:
                    self._emit_atoms(self._var_name(var), toks)     # reserved block
                else:
                    self._emit_int(0, toks)                          # zero block

    def capture_stream(self, order, transforms=None):
        """Full append-only stream for an entire elimination ``order``: run the
        whole elimination first, then emit base, then the output/``jac`` section
        (all output variables in one place), then one path block per face. Every
        parameterized op is defined once as a function. Returns the token list."""
        self.ij.eliminate_order(order, transforms)
        # SNAPSHOT the equation list ONCE: face ranges [start,end) index into it,
        # ``_emit_outputs`` calls ``to_jaxpr`` which may append to the live frame,
        # and ``get_eqns()`` rebuilds the whole list on every call.
        all_eqns = list(self.ij.all_eqns())
        self._jac_vars = self._collect_jac_vars(all_eqns)
        toks = self.base_tokens()
        self._emit_outputs(toks)
        for fr in self.ij.all_faces():
            self._emit_face(fr, all_eqns, toks)
        return toks

    def max_token_id(self):
        """Largest token id an emitted stream can currently use: ``len(vocab)``
        plus the highest name atom allocated so far. Size embeddings from this."""
        top = max((int(a, 16) for atoms in self._names.values() for a in atoms),
                  default=self.digit_base - 1)
        return self._L + top

    def decode(self, toks):
        out = []
        for t in map(int, toks):
            if t < self._L:
                out.append({"\n": " ¶ "}.get(self.n_vocab[t], self.n_vocab[t]))
            else:
                d = t - self._L
                out.append(str(d) if d < self.digit_base else "#" + format(d, "x"))
        return "".join(out)
