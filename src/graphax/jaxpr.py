import itertools
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

import jax.numpy as jnp
from jax._src import core

from graphax.primitives import elemental_rules

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
]

# TODO test transformer size


def get_vocab(digit_base: int = 10):
    full_params = (
        list("-½<>[](){}\n,;*_|~=:")
        + [e.name for e in elemental_rules]
        + primitive_params
    )

    # during training we can sample number of variables, nmber of functions, and digits we encounter, to better decide the params
    # TODO: allow to (define and) split the names of vars and fns
    # TODO: this is super slow, write this in c++ or rust.
    # TODO: make the full environment in c++ or rust.
    vocab = {p: i for i, p in enumerate(full_params)}
    n_vocab = {v: k for k, v in vocab.items()}
    return vocab, n_vocab, full_params


class VEJaxpr:
    def __init__(
        self,
        jaxpr: core.Jaxpr,
        *,
        vocab_size: int = 256,
        vocab_size_fns: int = None,
        digit_base: int = 10,
    ):
        self.jaxpr = jaxpr
        self.digit_base = digit_base
        self.vocab_size = vocab_size
        self.vocab_size_fns = vocab_size_fns
        self.vocab, self.n_vocab, _ = get_vocab(digit_base)
        self._names = self._assign_names(vocab_size)
        self._tokens = None

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

        def scan_jaxpr(j):
            for v in j.constvars:
                freq_map_vars.setdefault(v, 0)
                freq_map_vars[v] += 1
            for v in j.invars:
                freq_map_vars.setdefault(v, 0)
                freq_map_vars[v] += 1
            for eqn in j.eqns:
                pt = get_params_tuple(eqn, primitive_params)
                if pt:
                    key = (eqn.primitive.name, pt)
                    freq_map_fns.setdefault(key, 0)
                    freq_map_fns[key] += 1

                for v in eqn.outvars:
                    freq_map_vars.setdefault(v, 0)
                    freq_map_vars[v] += 1

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

        if self.vocab_size_fns is None:
            # Shared pool behavior
            freq_map_combined = {**freq_map_vars}
            for k, v in freq_map_fns.items():
                freq_map_combined[k] = freq_map_combined.get(k, 0) + v

            freq_list = sorted(
                freq_map_combined.items(), key=lambda x: x[1], reverse=True
            )
            names = {
                k: n
                for (k, _), n in zip(
                    freq_list,
                    name_gen_python_style(
                        self.digit_base, self.digit_base + num_extra_tokens
                    ),
                )
            }
        else:
            # Split pool behavior
            freq_list_fns = sorted(
                freq_map_fns.items(), key=lambda x: x[1], reverse=True
            )
            freq_list_vars = sorted(
                freq_map_vars.items(), key=lambda x: x[1], reverse=True
            )

            fns_slots = min(self.vocab_size_fns, num_extra_tokens)

            names = {}
            # Assign function names
            for (k, _), n in zip(
                freq_list_fns,
                name_gen_python_style(self.digit_base, self.digit_base + fns_slots),
            ):
                names[k] = n

            # Assign variable names
            for (k, _), n in zip(
                freq_list_vars,
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
            val = atom.val
            if isinstance(val, tuple) and len(val) > 0:
                val = val[0]

            is_int_like = False
            if isinstance(val, (int, jnp.integer)):
                is_int_like = True
                int_val = int(val)
            else:
                try:
                    # Generic check for anything that behaves like an integer float
                    f_val = float(val)
                    if f_val.is_integer():
                        is_int_like = True
                        int_val = int(f_val)
                except (TypeError, ValueError, OverflowError):
                    pass

            if is_int_like:
                return int_to_base(int_val, self.digit_base)
            else:
                s = str(val)

            if "(" in s:
                s = s.replace("(", "").replace(")", "")
            if "," in s:
                s = s.split(",")[0]

            s = s.strip()

            if s.endswith(".0"):
                s = s[:-2]

            s = s.replace("0.5", "½")
            return s

        ret = self._names[atom]
        return ret

    def __getattr__(self, name: str) -> Any:
        return getattr(self.jaxpr, name)

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

        _vocab = self.vocab
        _len_vocab = len(self.vocab)
        _names = self._names
        _base = self.digit_base

        tokens = list(self._invar_tokens)

        parameterized_ops = set()
        for eqn in self.jaxpr.eqns:
            pt = get_params_tuple(eqn, primitive_params)
            if pt:
                parameterized_ops.add(
                    (_names[(eqn.primitive.name, pt)], eqn.primitive.name, pt)
                )

        if def_fns and parameterized_ops:
            tokens.append(_vocab["~"])
            for op_name, prim_name, pt in parameterized_ops:
                self._tokenize_sequence(op_name, tokens)

                if prim_name in _vocab:
                    tokens.append(_vocab[prim_name])
                else:
                    tokens.extend(_vocab[c] for c in prim_name if c in _vocab)

                if show_params:
                    tokens.append(_vocab["["])
                    for i, (k, v) in enumerate(pt):
                        if k in _vocab:
                            tokens.append(_vocab[k])
                        else:
                            tokens.extend(_vocab[c] for c in k if c in _vocab)

                        tokenize_value(v, _vocab, tokens, _base)
                        if i < len(pt) - 1:
                            tokens.append(_vocab["|"])
                    tokens.append(_vocab["]"])

        tokens.append(_vocab["{"])

        # 3. Equations
        for eqn in self.jaxpr.eqns:
            pt = get_params_tuple(eqn, primitive_params)

            if not eqn.outvars or isinstance(eqn.outvars[0], core.DropVar):
                continue

            # Output var
            out_name = _names[eqn.outvars[0]]
            self._tokenize_sequence(out_name, tokens)

            if show_shapes:
                for v in eqn.outvars:
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
                    tokens.extend(_vocab[c] for c in prim_name if c in _vocab)

                if pt and show_params:
                    tokens.append(_vocab["["])
                    for i, (k, v) in enumerate(pt):
                        if k in _vocab:
                            tokens.append(_vocab[k])
                        else:
                            tokens.extend(_vocab[c] for c in k if c in _vocab)
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

        res = jnp.array(tokens, dtype=jnp.int32)
        self._eqn_tokens_cache[cache_key] = res
        return res


class VEHlo:
    def __init__(
        self,
        hlo_text: str,
        *,
        vocab_size: int = 256,
        digit_base: int = 10,
    ):
        self.hlo_text = hlo_text
        self.vocab_size = vocab_size
        self.digit_base = digit_base
        self.vocab, self.n_vocab, _ = get_vocab(digit_base)

        try:
            from .graphax_rs import HLOTokenizer

            self.tokenizer = HLOTokenizer(self.vocab, digit_base, vocab_size)
            self._tokens = self.tokenizer.tokenize(hlo_text)
        except ImportError:
            # Fallback or error
            self._tokens = None
            print(
                "Warning: graphax_rs.HLOTokenizer not found. HLO tokenization fallback not implemented."
            )

    def tokenized(self) -> jnp.ndarray:
        return self._tokens

    def __repr__(self):
        if self._tokens is None:
            return "VEHlo(Error: tokens not generated)"

        _len_vocab = len(self.vocab)
        token_strings = []
        for t in self._tokens:
            t_int = int(t)
            if t_int < _len_vocab:
                token_strings.append(self.n_vocab[t_int])
            else:
                token_strings.append(hex(t_int - _len_vocab))

        return (
            "".join(token_strings)
            .replace("{", " {\n")
            .replace("}", "\n}\n")
            .replace("\n\n", "\n")
        )


def extract_hlo(
    fun: Callable, *args, vocab_size: int = 256, digit_base: int = 10, **kwargs
) -> VEHlo:
    import jax

    lowered = jax.jit(fun).lower(*args, **kwargs)
    hlo_text = lowered.as_text()
    return VEHlo(hlo_text, vocab_size=vocab_size, digit_base=digit_base)


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


def tokenize_value(val: Any, vocab: Dict[str, int], tokens: List[int], base: int = 10):
    is_int_like = False
    if isinstance(val, (int, jnp.integer)):
        is_int_like = True
        int_val = int(val)
    else:
        try:
            f_val = float(val)
            if f_val.is_integer():
                is_int_like = True
                int_val = int(f_val)
        except (TypeError, ValueError, OverflowError):
            pass

    if is_int_like:
        _len_vocab = len(vocab)
        for c in int_to_base(int_val, base):
            if c in vocab:
                tokens.append(vocab[c])
            else:
                tokens.append(_len_vocab + int(c, 16))
    elif val is None:
        for c in "none":
            tokens.append(vocab[c])
    elif isinstance(val, tuple):
        if "(" in vocab:
            tokens.append(vocab["("])
        for i, v in enumerate(val):
            tokenize_value(v, vocab, tokens, base)
            if i < len(val) - 1:
                if "," in vocab:
                    tokens.append(vocab[","])
        if ")" in vocab:
            tokens.append(vocab[")"])
    elif isinstance(val, str):
        for c in val:
            if c in vocab:
                tokens.append(vocab[c])
    else:
        s = str(val).replace(" ", "")
        for c in s:
            if c in vocab:
                tokens.append(vocab[c])


def get_params_tuple(eqn, tokenized_param_names):
    keys = sorted(eqn.params.keys())
    valid = []
    for k in keys:
        if k in tokenized_param_names and eqn.params[k] is not None:
            valid.append((k, eqn.params[k]))
    return tuple(valid)
