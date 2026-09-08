"""Phases 1 to 3: pair, frame, decide. No array is touched here.

An axis of a contraction is one of nine cases. With two index classes the table
is small enough to write out, which is why the classes were deleted first.

                  |  contracted        |  carried            |  meta of a pair
  ----------------|--------------------|---------------------|------------------
  both store it   |  a dot axis        |  a batch axis       |  pair kept
  one stores it   |  sum the storer    |  keep on the storer |  keep on the storer
  neither         |  fold into scale   |  implicit out dim   |  implicit out pair

Three of the nine need no dot at all. "Both store it" means both operands carry
a physical ``val`` axis for that extent. "One stores it" is a single implicit
axis, "neither" a double implicit axis (CONTEXT.md).

The decision that follows the frame is one rule (owner ruling D1, 2026-09-07):

    no dot axis                     -> ANALYTIC, a scale or an elementwise product
    multiply grid fits the budget   -> EINSUM, the default
    otherwise                       -> DOT_GENERAL, the fallback

Broadcasting an implicit axis into the physical grid is NOT an option. It is
the only emission that materialises. The budget is a materialisation guard, not
a performance knob, and there is no device branch.

WHY EINSUM AND NOT A HAND-WRITTEN MULTIPLY-THEN-REDUCE (owner ruling
2026-09-08). An einsum with the interleaved INTEGER SUBLIST form states the
contraction and leaves the lowering to XLA, which is the most freedom we can
hand it. Two concrete gains over writing the multiply and the reduce by hand:

  * An implicit axis needs no special case. It is simply ABSENT from that
    operand's index list. An index in one operand and in the output is a free
    axis of that operand; an index in one operand and not in the output is
    summed. Neither needs a broadcast, and neither needs us to choose between a
    multiply-reduce and a dot.
  * XLA keeps the choice. It can lower one einsum to a library GEMM, to a fused
    multiply and reduce, or to something else, per shape and per device. A
    hand-written multiply-then-reduce takes that choice away and pins one
    answer for every shape.

Integer sublists, not letters: `Index.id` is an integer and the sublist form has
no 52-symbol alphabet cap.

The one case einsum cannot state is an output index present in NEITHER operand
(a double implicit axis riding to the output). That axis never enters the
contraction at all: it is carried on the frame and the output dim keeps
`axis=None`, so no value work is needed for it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class AxisCase(Enum):
    """Which of the nine cells an axis falls in."""

    DOT = "dot"                          # both store it, contracted
    BATCH = "batch"                      # both store it, carried
    PAIR_KEPT = "pair_kept"              # both store it, meta of a pair
    SUM_STORER = "sum_storer"            # one stores it, contracted
    KEEP_ON_STORER = "keep_on_storer"    # one stores it, carried
    KEEP_PAIR = "keep_pair"              # one stores it, meta of a pair
    FOLD_SCALE = "fold_scale"            # neither, contracted
    IMPLICIT_OUT = "implicit_out"        # neither, carried
    IMPLICIT_PAIR = "implicit_pair"      # neither, meta of a pair


#: The cases that need no dot. A frame made only of these is ANALYTIC.
NO_DOT = frozenset({AxisCase.SUM_STORER, AxisCase.FOLD_SCALE,
                    AxisCase.IMPLICIT_OUT, AxisCase.IMPLICIT_PAIR})


class EmissionKind(Enum):
    """The three forms a contraction is emitted in. Broadcasting an implicit
    axis into the physical grid is NOT among them (owner ruling D1)."""

    ANALYTIC = "analytic"          # no dot axis survives: a scale or a product
    EINSUM = "einsum"              # the default
    DOT_GENERAL = "dot_general"    # the fallback when the grid will not fit


@dataclass(frozen=True)
class AxisRecord:
    """One logical axis of the contraction, and where it came from.

    ``lhs_pos`` / ``rhs_pos`` are positions in each operand's ``dims``, or
    ``None`` when that operand has no such axis. ``stored_lhs`` / ``stored_rhs``
    say whether that operand gives the axis a physical ``val`` axis: an axis an
    operand declares but keeps implicit is NOT stored.
    """

    extent: int                        # logical size = meta * block
    case: AxisCase
    contracted: bool
    lhs_pos: int | None = None
    rhs_pos: int | None = None
    stored_lhs: bool = False
    stored_rhs: bool = False
    meta: int = 1                      # 1 for a dense axis
    block: int = 0                     # == extent for a dense axis
    group: int = -1                    # meta group id; see Frame.multiply_grid

    @property
    def implicit_count(self) -> int:
        """0, 1 or 2: how many operands leave this axis implicit."""
        present = int(self.lhs_pos is not None) + int(self.rhs_pos is not None)
        stored = int(self.stored_lhs) + int(self.stored_rhs)
        return present - stored


@dataclass(frozen=True)
class Frame:
    """The whole contraction, described without touching a value."""

    axes: tuple[AxisRecord, ...]
    out_arity: int                     # how many leading axes are out dims

    @property
    def dot_axes(self) -> tuple[AxisRecord, ...]:
        return tuple(a for a in self.axes if a.case not in NO_DOT and a.contracted)

    @property
    def multiply_grid(self) -> int:
        """Elements of the intermediate the einsum would form if XLA fuses it
        as a multiply and a reduce rather than a library call.

        NOT the product of the logical extents. A meta-block-diagonal pair
        threads ONE meta extent through several axes: on a block-diagonal
        matmul with meta M and blocks P, K, Q the product is M*P*K*Q, while the
        logical extents multiply to (M*P)*(M*K)*(M*Q), which over-counts M
        twice. Each meta group therefore contributes its meta ONCE, and every
        axis contributes its block.

        This is the number the budget guards: an oversized grid is a
        materialisation, which is the thing the ruling forbids, so counting it
        wrong sends the emission down the wrong branch.
        """
        n = 1
        for a in self.axes:
            n *= int(a.block)
        seen = set()
        for a in self.axes:
            if a.group in seen:
                continue
            seen.add(a.group)
            n *= int(a.meta)
        return n

    def describe(self) -> str:
        """One line per axis. This is what a storage test asserts on."""
        rows = [f"out_arity={self.out_arity} multiply_grid={self.multiply_grid}"]
        for i, a in enumerate(self.axes):
            rows.append(
                f"  [{i}] extent={a.extent:<6} {a.case.value:<16} "
                f"contracted={str(a.contracted):<5} "
                f"lhs={a.lhs_pos}/{'S' if a.stored_lhs else 'i'} "
                f"rhs={a.rhs_pos}/{'S' if a.stored_rhs else 'i'} "
                f"meta={a.meta}x block={a.block} group={a.group}")
        return "\n".join(rows)


def _classify(present_lhs: bool, present_rhs: bool, stored_lhs: bool,
              stored_rhs: bool, contracted: bool, is_pair: bool) -> AxisCase:
    """The nine-cell table, as one function. No arrays, no shapes."""
    stored = int(stored_lhs) + int(stored_rhs)
    if stored == 2:
        if is_pair:
            return AxisCase.PAIR_KEPT
        return AxisCase.DOT if contracted else AxisCase.BATCH
    if stored == 1:
        if is_pair:
            return AxisCase.KEEP_PAIR
        return AxisCase.SUM_STORER if contracted else AxisCase.KEEP_ON_STORER
    if is_pair:
        return AxisCase.IMPLICIT_PAIR
    return AxisCase.FOLD_SCALE if contracted else AxisCase.IMPLICIT_OUT


#: Byte budget on the einsum's intermediate grid. Above it the emission falls
#: back to DOT_GENERAL, because a grid XLA will not fuse is a materialisation.
#: The VALUE is a device property (the fusion window is hardware); the RULE is
#: not, so there is no device branch. Measured and pinned per device class.
DEFAULT_BUDGET_BYTES = 1 << 27          # 128 MB, provisional pending the GPU census


def multiply_budget_bytes() -> int:
    raw = os.environ.get("GRAPHAX_MULTIPLY_BUDGET_BYTES")
    return int(raw) if raw else DEFAULT_BUDGET_BYTES


def decide_emission(frame: Frame, itemsize: int = 4,
                    budget_bytes: int | None = None) -> EmissionKind:
    """Phase 3. One rule, read off the frame alone.

    ``itemsize`` is the operand dtype's byte width, so a narrower dtype fits a
    larger grid. That is the only thing the dtype changes here.
    """
    if not frame.dot_axes:
        return EmissionKind.ANALYTIC
    budget = multiply_budget_bytes() if budget_bytes is None else budget_bytes
    if frame.multiply_grid * itemsize <= budget:
        return EmissionKind.EINSUM
    return EmissionKind.DOT_GENERAL


def build_frame(lhs, rhs, n_contract: int | None = None) -> Frame:
    """Phase 1 and 2: pair the dims positionally, then classify every axis.

    The pairing is the contraction convention, NOT an id match: ``Index.id`` is
    a within-tensor logical position (ids must be contiguous per tensor), so
    the same id in two operands names two different axes. A contraction meets
    ``lhs.primal_dims`` against ``rhs.out_dims``, position by position:

        out axes      lhs.out_dims                   ride to the output
        contracted    lhs.primal_dims x rhs.out_dims meet here
        primal axes   rhs.primal_dims                ride to the output

    ``n_contract`` is how many trailing lhs primal dims meet leading rhs out
    dims; the default contracts all of them, which is what ``matmul`` does.
    Nothing here reads ``val``.
    """
    l_out, l_pri = tuple(lhs.out_dims), tuple(lhs.primal_dims)
    r_out, r_pri = tuple(rhs.out_dims), tuple(rhs.primal_dims)
    k = min(len(l_pri), len(r_out)) if n_contract is None else int(n_contract)
    if k > len(l_pri) or k > len(r_out):
        raise ValueError(
            f"cannot contract {k} axes: lhs has {len(l_pri)} primal dims and "
            f"rhs has {len(r_out)} out dims")

    def stored(d):
        return d is not None and d.axis is not None

    def paired(d):
        return d is not None and d.other_id is not None

    def mb(d):
        """(meta, block) of one dim. A dense axis is one block of its size."""
        if d.other_id is None:
            return 1, int(d.size)
        return int(d.size), int(d.block_size or 1)

    axes: list[AxisRecord] = []
    # (a) lhs out dims, carried
    for i, d in enumerate(l_out):
        axes.append(AxisRecord(
            extent=int(d.logical_size),
            case=_classify(True, False, stored(d), False, False, paired(d)),
            contracted=False, lhs_pos=i, rhs_pos=None,
            stored_lhs=stored(d), stored_rhs=False,
            meta=mb(d)[0], block=mb(d)[1]))
    # (b) the contracted pairs: lhs primal i meets rhs out i
    base = len(l_out)
    for i in range(k):
        ld, rd = l_pri[i], r_out[i]
        if int(ld.logical_size) != int(rd.logical_size):
            raise ValueError(
                f"contracted axis {i} has logical extent {ld.logical_size} on "
                f"the lhs and {rd.logical_size} on the rhs")
        axes.append(AxisRecord(
            extent=int(ld.logical_size),
            case=_classify(True, True, stored(ld), stored(rd), True,
                           paired(ld) or paired(rd)),
            contracted=True, lhs_pos=len(l_out) + i, rhs_pos=i,
            stored_lhs=stored(ld), stored_rhs=stored(rd),
            meta=max(mb(ld)[0], mb(rd)[0]),
            block=int(ld.logical_size) // max(mb(ld)[0], mb(rd)[0])))
    # (c) lhs primal dims that are NOT contracted, carried
    for i in range(k, len(l_pri)):
        d = l_pri[i]
        axes.append(AxisRecord(
            extent=int(d.logical_size),
            case=_classify(True, False, stored(d), False, False, paired(d)),
            contracted=False, lhs_pos=len(l_out) + i, rhs_pos=None,
            stored_lhs=stored(d), stored_rhs=False,
            meta=mb(d)[0], block=mb(d)[1]))
    # (d) rhs out dims that are NOT contracted, then rhs primal dims, carried
    for i in range(k, len(r_out)):
        d = r_out[i]
        axes.append(AxisRecord(
            extent=int(d.logical_size),
            case=_classify(False, True, False, stored(d), False, paired(d)),
            contracted=False, lhs_pos=None, rhs_pos=i,
            stored_lhs=False, stored_rhs=stored(d),
            meta=mb(d)[0], block=mb(d)[1]))
    for i, d in enumerate(r_pri):
        axes.append(AxisRecord(
            extent=int(d.logical_size),
            case=_classify(False, True, False, stored(d), False, paired(d)),
            contracted=False, lhs_pos=None, rhs_pos=len(r_out) + i,
            stored_lhs=False, stored_rhs=stored(d),
            meta=mb(d)[0], block=mb(d)[1]))
    return Frame(tuple(_assign_groups(axes, lhs, rhs, l_out, l_pri, r_out, k)),
                 out_arity=len(l_out))


def _assign_groups(axes, lhs, rhs, l_out, l_pri, r_out, k):
    """Number the META GROUPS: axes that share one diagonal structure.

    Two axes share a meta when they are pair partners inside an operand, or
    when they meet on a contracted axis (the contraction threads the two
    operands' pairs into one). Union-find over those two relations. A dense
    axis is its own group with meta 1, so it costs nothing either way.
    """
    parent = list(range(len(axes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # pair partners inside the lhs: its dims are l_out then l_pri
    l_dims = tuple(l_out) + tuple(l_pri)
    pos_in_frame = {}
    for i, a in enumerate(axes):
        if a.lhs_pos is not None:
            pos_in_frame.setdefault(("l", a.lhs_pos), i)
        if a.rhs_pos is not None:
            pos_in_frame.setdefault(("r", a.rhs_pos), i)
    for j, d in enumerate(l_dims):
        if d.other_id is None:
            continue
        partner = next((q for q, e in enumerate(l_dims) if int(e.id) == int(d.other_id)), None)
        if partner is None:
            continue
        a, b = pos_in_frame.get(("l", j)), pos_in_frame.get(("l", partner))
        if a is not None and b is not None:
            union(a, b)
    r_dims = tuple(r_out) + tuple(rhs.primal_dims)
    for j, d in enumerate(r_dims):
        if d.other_id is None:
            continue
        partner = next((q for q, e in enumerate(r_dims) if int(e.id) == int(d.other_id)), None)
        if partner is None:
            continue
        a, b = pos_in_frame.get(("r", j)), pos_in_frame.get(("r", partner))
        if a is not None and b is not None:
            union(a, b)
    # A contracted axis needs no rule of its own: the lhs pair relation and the
    # rhs pair relation already thread through it, because the contracted axis
    # is a member of a pair in each operand.
    return [
        AxisRecord(extent=a.extent, case=a.case, contracted=a.contracted,
                   lhs_pos=a.lhs_pos, rhs_pos=a.rhs_pos,
                   stored_lhs=a.stored_lhs, stored_rhs=a.stored_rhs,
                   meta=a.meta, block=a.block, group=find(i))
        for i, a in enumerate(axes)
    ]
