"""Structure-lowering layer for matmul (env ``GRAPHAX_STRUCT_LOWER``, default OFF).

Given ``(matmul, lhs.dims, rhs.dims)``, COMPILE the minimal physical computation
(ONE einsum over PHYSICAL axes only) and construct the output structure
SYMBOLICALLY:

  * a free implicit dim (``axis=None``, logical>1 — extent stored once) stays
    implicit in the output; it is never broadcast to a physical axis,
  * a surviving block-diagonal pair stays a ``DiagonalIndex`` pair,
  * a ``val=None`` operand contributes no compute; an all-implicit result keeps
    ``val=None``,
  * a contracted pair that is implicit on BOTH sides is an analytic
    scale-by-N folded into ``scalar_mult`` (no compute at all),
  * a contracted pair implicit on ONE side is a plain SUM over the physical
    side (a reduction, not a matmul).

Topology is NOT re-derived: the planner consumes the exact ``Pair`` list the
existing tiled path builds (``_build_matmul_topology``), so it can never
disagree with the incumbent about WHAT contracts / batches / rides through —
it only replaces HOW the physical buffers combine (the tiled machinery
broadcasts implicit axes to physical vals, which is where output structure
died: census T4 retention 66% off / 5% on).

Hooked at the top of ``ops.matmul.matmul``. Any case without a rule falls
through to the existing path UNCHANGED; every miss is counted in
``LOWER_STATS`` (no silent behavior change). Gates:

  * env ``GRAPHAX_STRUCT_LOWER=1`` (checked by the hook),
  * ``approx_active()`` — the same correctness firewall the elemental dispatch
    uses: an EXACT-AD (no Diag/Compress) elimination NEVER enters here, so the
    exact path stays byte-identical whether the env flag is on or off,
  * zero fill on both operands, no compressed dims, no bool dtype,
  * at least one operand actually carries lowerable structure (an implicit
    dim, a rectangular block, or ``val=None``) — plain dense / plain-diagonal
    contractions stay on the incumbent path,
  * every ``Pair``'s meta/block factorization is letter-compatible; the
    LCM-grid (misaligned) cases stay on the tiled path, which owns them.
"""

# pyright: reportImportCycles=false
from __future__ import annotations

import builtins
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from graphax.sparse.indexes import DenseIndex, DiagonalIndex

if TYPE_CHECKING:
    from graphax.sparse.tensor import SparseTensor


# --------------------------------------------------------------------------- #
# Telemetry — every fall-through is counted, never silent.
# --------------------------------------------------------------------------- #
LOWER_STATS: dict[str, int] = {}

# Factor labels are INTEGERS consumed by jnp.einsum's interleaved
# (operand, sublist, ..., out_sublist) API — no alphabet, no cap. The string
# pool (52 letters -> _NoRule("letters_exhausted")) died here 2026-08-02; a
# ~10-logical-dim fully-coupled contraction needed ~30-40 labels and ViT-with-
# approximations brushed the cap. dot_general has no alphabet either.


def _bump(key: str, n: int = 1) -> None:
    LOWER_STATS[key] = LOWER_STATS.get(key, 0) + n


def reset_stats() -> None:
    LOWER_STATS.clear()


def enabled() -> bool:
    """Env gate, read per call so tests can toggle without re-import."""
    return os.environ.get("GRAPHAX_STRUCT_LOWER", "0") != "0"


# --------------------------------------------------------------------------- #
# L5 demand channel. An edge whose head is a GRAPH OUTPUT is materialized
# dense by the output boundary regardless -- a hard demand. Core sets this
# contextvar around output-headed contractions (they run via the ``@``
# operator, so no kwarg can travel); the demand-emit block in ``_lower``
# reads it and emits the DENSE layout directly from the single einsum, so
# the block form never coexists with the dense output buffer.
# --------------------------------------------------------------------------- #
import contextvars as _contextvars

_DEMAND_DENSE = _contextvars.ContextVar("gx_demand_dense", default=False)


def _demand_emit_on() -> bool:
    return _DEMAND_DENSE.get() and os.environ.get(
        "GRAPHAX_DEMAND_EMIT", "0"
    ) == "1"


class _NoRule(Exception):
    """Planning found no rule for this case — the caller falls through to the
    existing matmul path unchanged (and the reason is counted)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------- #
# Symbolic factors and output-dim specs
# --------------------------------------------------------------------------- #
@dataclass
class _Fac:
    """One symbolic extent (einsum letter): a meta count, a block extent, or a
    plain dense extent. ``axes[side]`` is the operand ``val`` source axis
    (``None`` = implicit on that side; a ``("split", axis, part)`` marker =
    physical via an axis split)."""

    letter: int  # integer einsum label (interleaved-API sublists)
    size: int
    axes: list  # [lhs, rhs]
    in_output: bool = False

    @property
    def phys(self) -> bool:
        return self.axes[0] is not None or self.axes[1] is not None


@dataclass
class _DimSpec:
    """Symbolic output dim (pre-renumbering).

    ``kind="dense"``: ``letters`` multiply out to ``size`` (an empty/implicit
    group keeps ``axis=None``); several letters merge into ONE physical axis.
    ``kind="pair"``: one side of a retained DiagonalIndex pair — ``size`` is
    the META count, ``block_size`` the own-side block; ``meta_fac`` is shared
    with the partner (``other_id``)."""

    is_out: bool
    id: int
    kind: str
    size: int
    letters: list = field(default_factory=list)
    other_id: int = -1
    block_size: int = 1
    meta_fac: Any = None
    block_fac: Any = None

    @property
    def logical(self) -> int:
        return self.size * (self.block_size if self.kind == "pair" else 1)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def try_lower_matmul(lhs: "SparseTensor", rhs: "SparseTensor", count: bool = False):
    """Attempt the structure-lowering matmul. Returns the contracted
    ``SparseTensor`` (or ``(result, (adds, muls, fmas))`` with ``count=True``),
    or ``None`` when no rule applies — the caller then runs the existing path
    unchanged."""
    if not enabled():
        return None
    # EXACT-AD firewall (same as the elemental dispatch): only an elimination
    # that carries a Diag/Compress/Quant transform may be re-associated.
    from graphax.sparse.elemental.dispatch import approx_active

    if not approx_active():
        _bump("skip:exact_ad")
        return None

    from graphax.sparse.ops.utils import _is_approx, _is_zero_fill

    if not (_is_zero_fill(lhs) and _is_zero_fill(rhs)):
        _bump("fallthrough:nonzero_fill")
        return None
    if lhs.dtype == jnp.bool_ or rhs.dtype == jnp.bool_:
        _bump("fallthrough:bool_dtype")
        return None
    if not (_is_approx(lhs) or _is_approx(rhs) or lhs.val is None or rhs.val is None):
        # No implicit dim, no rectangular block, no pure-structure operand:
        # the incumbent path already handles it tightly — don't touch.
        _bump("skip:no_lowerable_structure")
        return None
    from graphax.sparse.ops.matmul import _has_implicit_block_contraction

    if _has_implicit_block_contraction(lhs, rhs):
        # Metadata-stated size-1 embeds are owned by the densify path.
        _bump("fallthrough:block_embed")
        return None

    try:
        out, counts, tags = _lower(lhs, rhs)
    except _NoRule as e:
        _bump(f"fallthrough:{e.reason}")
        return None
    _bump("hit")
    for t in tags:
        _bump(t)
    if count:
        return out, counts
    return out


# --------------------------------------------------------------------------- #
# Planner + executor
# --------------------------------------------------------------------------- #
def _canon_degenerate(st):
    """Canonicalization: a coupled pair with meta size 1 is ONE full block --
    it carries zero sparsity information, but its bookkeeping (a block equal to
    the whole logical extent) trips every downstream rule that reads
    ``(size, block_size)`` as genuine structure (measured: the split rules
    minted a phantom output pair of logical N*N from a ``1xN`` operand -- the
    wrong-shaped-Jacobian bug). Rewrite such pairs to plain dense dims before
    planning; for meta 1 the materialization is a squeeze, so this costs
    nothing at runtime. Failures fall through as _NoRule (never a wrong plan).
    """
    deg = [d for d in st.dims
           if getattr(d, "other_id", None) is not None
           and int(getattr(d, "size", 0) or 0) == 1]
    if not deg:
        return st
    try:
        from dataclasses import replace as _replace

        from graphax.sparse.indexes import DenseIndex
        from graphax.sparse.tensor import SparseTensor

        val = st.val
        # squeeze each PHYSICAL size-1 meta axis (shared by both members)
        drop = sorted({d.axis for d in deg if d.axis is not None}, reverse=True)
        if val is not None:
            for ax in drop:
                val = val.squeeze(ax)

        def _shift(a):
            if a is None:
                return None
            return a - sum(1 for ax in drop if ax < a)

        deg_ids = {d.id for d in deg}

        def _map(d):
            if d.id in deg_ids:
                # the block IS the whole extent: a plain dense dim of the
                # block size, on the (shifted) block axis (None stays implicit)
                return DenseIndex(d.id, int(d.block_size or 1),
                                  axis=_shift(d.block_axis))
            if d.is_sparse:
                return _replace(d, axis=_shift(d.axis),
                                block_axis=_shift(d.block_axis))
            return _replace(d, axis=_shift(d.axis))

        return SparseTensor(
            tuple(_map(d) for d in st.out_dims),
            tuple(_map(d) for d in st.primal_dims),
            val,
            scalar_mult=st.scalar_mult, fill_value=st.fill_value,
            pre_transforms=st.pre_transforms,
            post_transforms=st.post_transforms,
            check_consistency=False,
        )
    except Exception:
        raise _NoRule("canon_degenerate")


def _einsum_accum_dtype(dtypes):
    """``preferred_element_type`` for the planner einsum, or None to keep
    ``jnp.einsum``'s own default. OFF unless GRAPHAX_PLANNER_F32_ACCUM=1.

    The intent mirrors ``ops.matmul._gx_dot_general``: when every operand is
    bfloat16 -- which only a ``Quant`` produces, since a mixed {bf16, f32}
    pair is upcast by ``dtype_compute._unify_operand_dtypes`` before it gets
    here -- ask for float32 product/accumulate, giving a bf16-in / f32-out
    dot. Every other dtype combination returns None, so the EXACT-AD path
    emits a byte-identical einsum.

    DEFAULT OFF, and this is a MEASURED decision, not caution. ``jnp.einsum``
    honours ``preferred_element_type`` as a genuine bf16-in / f32-out
    ``dot_general`` only for the plain ``ij,jk->ik`` form. Every einsum the
    planner actually emits carries BATCH letters and size-1 axes -- e.g.
    ``(4,8)[0,3] x (4,1,8,8)[0,1,3,7] -> [0,3,7]`` -- and for those forms
    ``jnp.einsum`` implements the request by CONVERTING both operands to f32
    up front. Measured on mlp2 / mlp4_multimatmul / attn with both faces of
    every contraction quantized to bf16 (per-face ``lhs``/``rhs`` slots):

        flag off  mlp2  dots {f32:4, bf16:6}   converts 22
        flag on   mlp2  dots {f32:10}          converts 33
        flag off  mlp4  dots {f32:8, bf16:16}  converts 56
        flag on   mlp4  dots {f32:24}          converts 85

    i.e. it DELETES every bf16 dot and adds ~50% more converts, while the
    error against the exact f32 Jacobian is unchanged (relerr 4.207e-3 on
    mlp2 and 4.103e-3 on mlp4 either way -- XLA already accumulates a bf16
    dot in f32 internally and only rounds the output). Getting real bf16-in /
    f32-out accumulation out of the planner needs the contraction lowered to
    ``lax.dot_general`` with explicit dimension numbers (what
    ``_gx_dot_general`` does for the tiled path), not an einsum kwarg.

    GRAPHAX_QUANT_NARROW_GEMM=0 also disables it, same knob as the dot sites.
    """
    if os.environ.get("GRAPHAX_PLANNER_F32_ACCUM", "0") != "1":
        return None
    if os.environ.get("GRAPHAX_QUANT_NARROW_GEMM", "1") == "0":
        return None
    if not dtypes:
        return None
    _bf16 = jnp.dtype(jnp.bfloat16)
    if all(jnp.dtype(d) == _bf16 for d in dtypes):
        return jnp.float32
    return None


def _lower(lhs, rhs):
    from graphax.sparse.dtype_compute import _scaled_mul
    from graphax.sparse.ops.matmul import _align_tensor_ids, _build_matmul_topology
    from graphax.sparse.tensor import SparseTensor

    lhs = _canon_degenerate(lhs)
    rhs = _canon_degenerate(rhs)

    tags: list[str] = []

    # ---- topology: the incumbent's own pairing decisions ------------------ #
    try:
        rhs_out_dims, rhs_primal_dims, off = _align_tensor_ids(lhs, rhs)
        pairs = _build_matmul_topology(lhs, rhs_out_dims, rhs_primal_dims, off)
    except Exception:
        raise _NoRule("topology_error")
    rhs_dims = rhs_out_dims + rhs_primal_dims

    # ---- factor allocation ------------------------------------------------ #
    import itertools as _it

    labels = _it.count()
    facs: list[_Fac] = []

    def fac(size) -> _Fac:
        f = _Fac(next(labels), int(size), [None, None])
        facs.append(f)
        return f

    sts = (lhs, rhs)
    ax_maps: list[dict] = [{}, {}]      # side -> {src_axis: _Fac}
    split_maps: list[dict] = [{}, {}]   # side -> {src_axis: (_Fac, _Fac)}

    def assign(side: int, axis, f: _Fac) -> None:
        if axis is None:
            return
        val = sts[side].val
        if val is None:
            raise _NoRule("axis_without_val")
        if axis >= val.ndim:
            raise _NoRule("axis_shape_mismatch")
        if int(val.shape[axis]) == 1 and f.size != 1:
            # BROADCAST-STORED axis: the buffer is constant along this factor
            # on this side (a keepdims/broadcast representation — the dominant
            # ViT/MoE/Encoder fallthrough family, 3.3k events). Leave the
            # factor implicit here: the einsum-build loop squeezes the
            # undescribed 1-axis and einsum's missing-label semantics ARE
            # broadcast (batch) / constant-times-sum (contract).
            tags.append("rule:broadcast_axis")
            return
        if int(val.shape[axis]) != f.size:
            raise _NoRule("axis_shape_mismatch")
        if axis in ax_maps[side] or axis in split_maps[side]:
            raise _NoRule("axis_collision")
        ax_maps[side][axis] = f
        f.axes[side] = axis

    def assign_split(side: int, axis, f1: _Fac, f2: _Fac) -> None:
        if axis is not None and sts[side].val is not None \
                and axis < sts[side].val.ndim \
                and int(sts[side].val.shape[axis]) == 1 \
                and f1.size * f2.size != 1:
            # broadcast-stored axis under a split: both parts stay implicit
            # on this side (see assign()).
            tags.append("rule:broadcast_axis")
            return
        if axis is None:
            return  # fully implicit on this side
        val = sts[side].val
        if val is None:
            raise _NoRule("axis_without_val")
        if axis >= val.ndim or int(val.shape[axis]) != f1.size * f2.size:
            raise _NoRule("split_shape_mismatch")
        if axis in ax_maps[side] or axis in split_maps[side]:
            raise _NoRule("axis_collision")
        split_maps[side][axis] = (f1, f2)
        f1.axes[side] = ("split", axis, 0)
        f2.axes[side] = ("split", axis, 1)

    # ---- output-id generation: mirror _build_pair_dims exactly ------------ #
    next_id = builtins.max(
        [d.id for d in lhs.dims] + [d.id for d in rhs_dims] + [-1]
    ) + 1

    def gen(dim):
        nonlocal next_id
        if dim is not None:
            return dim.id
        n = next_id
        next_id += 1
        return n

    specs: list[_DimSpec] = []

    def add_dense(is_out, dim_id, size, group):
        specs.append(_DimSpec(is_out, dim_id, "dense", int(size), letters=group))

    def add_pair(out_id, primal_id, meta, u, b1, w, b2, x):
        """Retained DiagonalIndex pair (out side + primal side). ``meta == 1``
        degenerates to two plain dense dims, mirroring ``_build_sparse``."""
        if meta == 1:
            add_dense(True, out_id, b1, [u, w])
            add_dense(False, primal_id, b2, [x])
            return
        specs.append(
            _DimSpec(True, out_id, "pair", int(meta), other_id=primal_id,
                     block_size=int(b1), meta_fac=u, block_fac=w)
        )
        specs.append(
            _DimSpec(False, primal_id, "pair", int(meta), other_id=out_id,
                     block_size=int(b2), meta_fac=u, block_fac=x)
        )
        tags.append("out:pair_retained")

    # ---- per-pair planning ------------------------------------------------ #
    for pm in pairs:
        pt = pm.pairing_type
        l_id = gen(pm.lhs.dim)
        r_id = gen(pm.rhs.dim)
        ls_id = gen(pm.lhs.shared_dim)
        rs_id = gen(pm.rhs.shared_dim)

        if pt == "contract":
            lo, lp = pm.lhs.dim, pm.lhs.shared_dim
            ro, rp = pm.rhs.dim, pm.rhs.shared_dim
            if lp is None or ro is None:
                raise _NoRule("contract_missing_dim")
            if lp.is_sparse and lo is None:
                raise _NoRule("orphan_sparse")
            if ro.is_sparse and rp is None:
                raise _NoRule("orphan_sparse")
            N_l = int(lp.size) if lp.is_sparse else 1
            b_l = int(lp.block_size or 1) if lp.is_sparse else int(lp.size)
            N_r = int(ro.size) if ro.is_sparse else 1
            b_r = int(ro.block_size or 1) if ro.is_sparse else int(ro.size)
            if N_l * b_l != N_r * b_r:
                raise _NoRule("contract_size_mismatch")
            if (N_l, b_l) == (N_r, b_r):
                u, v = fac(N_l), fac(b_l)
                assign(0, pm.lhs.outer_axis, u)
                assign(0, pm.lhs.shared_block_axis, v)
                assign(1, pm.rhs.outer_axis, u)
                assign(1, pm.rhs.block_axis, v)
                tags.append("rule:contract_direct")
            elif (lp.is_sparse and b_l == 1
                  and lo is not None and (lo.block_size or 1) == 1
                  and ro.is_sparse and N_r > 1 and b_r > 1):
                # REFINE-LHS: a plain-diagonal pair (N,1) contracted against a
                # block pair (M, beta) with N == M*beta (guaranteed by logical
                # equality). The delta factorizes: n=(m,r) => d_nn' =
                # d_mm' d_rr', so nothing is summed over r — it SURVIVES as
                # the lhs survivor's block. Split the shared meta axis into
                # (M, beta); output pair meta M, blocks (beta, gamma).
                u, v = fac(N_r), fac(b_r)
                assign_split(0, pm.lhs.outer_axis, u, v)
                assign(1, pm.rhs.outer_axis, u)
                assign(1, pm.rhs.block_axis, v)
                u.in_output = True
                v.in_output = True
                if rp is not None:
                    x = fac(rp.block_size or 1)
                    x.in_output = True
                    assign(1, pm.rhs.shared_block_axis, x)
                    add_pair(l_id, rs_id, N_r, u, b_r, v, rp.block_size or 1, x)
                else:  # ro sparse => rp exists (orphan gate); defensive only
                    add_dense(True, l_id, lo.logical_size, [u, v])
                tags.append("rule:contract_refine_lhs")
                continue
            elif (ro.is_sparse and b_r == 1
                  and rp is not None and (rp.block_size or 1) == 1
                  and lp.is_sparse and N_l > 1 and b_l > 1):
                # REFINE-RHS: mirror — rhs is the plain diagonal (M',1) with
                # M' == N*b; refine it onto lhs's (N, b) grid; b survives as
                # the rhs survivor's block.
                u, v = fac(N_l), fac(b_l)
                assign(0, pm.lhs.outer_axis, u)
                assign(0, pm.lhs.shared_block_axis, v)
                assign_split(1, pm.rhs.outer_axis, u, v)
                u.in_output = True
                v.in_output = True
                if lo is not None:
                    w = fac(lo.block_size or 1)
                    w.in_output = True
                    assign(0, pm.lhs.block_axis, w)
                    add_pair(l_id, rs_id, N_l, u, lo.block_size or 1, w, b_l, v)
                else:  # lp sparse => lo exists (orphan gate); defensive only
                    add_dense(False, rs_id, rp.logical_size, [u, v])
                tags.append("rule:contract_refine_rhs")
                continue
            elif N_l == 1 and N_r > 1:
                # lhs contracted dim is plain; factor it on rhs's (meta, block)
                # grid — a physical lhs axis splits, an implicit one stays put.
                u, v = fac(N_r), fac(b_r)
                assign_split(0, pm.lhs.shared_block_axis, u, v)
                assign(1, pm.rhs.outer_axis, u)
                assign(1, pm.rhs.block_axis, v)
                tags.append("rule:contract_split_lhs")
            elif N_r == 1 and N_l > 1:
                u, v = fac(N_l), fac(b_l)
                assign(0, pm.lhs.outer_axis, u)
                assign(0, pm.lhs.shared_block_axis, v)
                assign_split(1, pm.rhs.block_axis, u, v)
                tags.append("rule:contract_split_rhs")
            else:
                raise _NoRule("lcm_contract")

            w = x = None
            if lo is not None:
                w = fac(lo.block_size or 1)
                w.in_output = True
                u.in_output = True
                assign(0, pm.lhs.block_axis, w)
            if rp is not None:
                x = fac(rp.block_size or 1)
                x.in_output = True
                u.in_output = True
                assign(1, pm.rhs.shared_block_axis, x)
            if lo is not None and rp is not None:
                add_pair(l_id, rs_id, N_l if lp.is_sparse else N_r,
                         u, lo.block_size or 1, w, rp.block_size or 1, x)
            elif lo is not None:
                add_dense(True, l_id, lo.logical_size, [u, w])
            elif rp is not None:
                add_dense(False, rs_id, rp.logical_size, [u, x])

        elif pt == "batch_sparse":
            lout, lprim = pm.lhs.dim, pm.lhs.shared_dim
            rout, rprim = pm.rhs.dim, pm.rhs.shared_dim
            if None in (lout, lprim, rout, rprim):
                raise _NoRule("orphan_sparse")
            if not all(d.is_sparse for d in (lout, lprim, rout, rprim)):
                raise _NoRule("batch_sparse_mixed")
            if (int(lout.size) != int(rout.size)
                    or int(lout.block_size or 1) != int(rout.block_size or 1)
                    or int(lprim.block_size or 1) != int(rprim.block_size or 1)):
                raise _NoRule("batch_sparse_mismatch")
            u = fac(lout.size)
            w = fac(lout.block_size or 1)
            x = fac(lprim.block_size or 1)
            u.in_output = w.in_output = x.in_output = True
            assign(0, pm.lhs.outer_axis, u)
            assign(0, pm.lhs.block_axis, w)
            assign(0, pm.lhs.shared_block_axis, x)
            assign(1, pm.rhs.outer_axis, u)
            assign(1, pm.rhs.block_axis, w)
            assign(1, pm.rhs.shared_block_axis, x)
            add_pair(l_id, rs_id, lout.size, u,
                     lout.block_size or 1, w, lprim.block_size or 1, x)
            tags.append("rule:batch_sparse")

        elif pt == "batch_out":
            lout, rout = pm.lhs.dim, pm.rhs.dim
            if lout is None or rout is None or lout.is_sparse or rout.is_sparse:
                raise _NoRule("batch_odd")
            if int(lout.size) != int(rout.size):
                raise _NoRule("batch_size_mismatch")
            k = fac(lout.size)
            k.in_output = True
            assign(0, pm.lhs.outer_axis, k)
            assign(1, pm.rhs.outer_axis, k)
            add_dense(True, l_id, lout.size, [k])
            tags.append("rule:batch_out")

        elif pt == "batch_primal":
            lp_, rp_ = pm.lhs.shared_dim, pm.rhs.shared_dim
            if lp_ is None or rp_ is None or lp_.is_sparse or rp_.is_sparse:
                raise _NoRule("batch_odd")
            if int(lp_.size) != int(rp_.size):
                raise _NoRule("batch_size_mismatch")
            k = fac(lp_.size)
            k.in_output = True
            assign(0, pm.lhs.shared_block_axis, k)
            assign(1, pm.rhs.shared_block_axis, k)
            add_dense(False, ls_id, lp_.size, [k])
            tags.append("rule:batch_primal")

        elif pt in ("spatial_sparse_lhs", "spatial_sparse_rhs"):
            side = 0 if pt.endswith("lhs") else 1
            pd_side = pm.lhs if side == 0 else pm.rhs
            od, pd_ = pd_side.dim, pd_side.shared_dim
            if od is None or pd_ is None or not (od.is_sparse and pd_.is_sparse):
                raise _NoRule("orphan_sparse")
            u = fac(pd_side.outer_len)
            w = fac(pd_side.block_len)
            x = fac(pd_side.shared_block_len)
            u.in_output = w.in_output = x.in_output = True
            assign(side, pd_side.outer_axis, u)
            assign(side, pd_side.block_axis, w)
            assign(side, pd_side.shared_block_axis, x)
            oid = l_id if side == 0 else r_id
            pid = ls_id if side == 0 else rs_id
            add_pair(oid, pid, pd_side.outer_len, u,
                     pd_side.block_len, w, pd_side.shared_block_len, x)
            tags.append("rule:spatial_sparse")

        elif pt in ("spatial_out_lhs", "spatial_out_rhs"):
            side = 0 if pt.endswith("lhs") else 1
            pd_side = pm.lhs if side == 0 else pm.rhs
            d = pd_side.dim
            if d is None or d.is_sparse:
                raise _NoRule("spatial_odd")
            k = fac(pd_side.block_len)
            k.in_output = True
            assign(side, pd_side.block_axis, k)
            add_dense(True, l_id if side == 0 else r_id, pd_side.block_len, [k])
            tags.append("rule:spatial_dense")

        elif pt in ("spatial_primal_lhs", "spatial_primal_rhs"):
            side = 0 if pt.endswith("lhs") else 1
            pd_side = pm.lhs if side == 0 else pm.rhs
            d = pd_side.shared_dim
            if d is None or d.is_sparse:
                raise _NoRule("spatial_odd")
            k = fac(pd_side.shared_block_len)
            k.in_output = True
            assign(side, pd_side.shared_block_axis, k)
            add_dense(False, ls_id if side == 0 else rs_id,
                      pd_side.shared_block_len, [k])
            tags.append("rule:spatial_dense")

        else:
            raise _NoRule(f"unknown_pairing_{pt}")

    # ---- analytic fold: letters with no physical source and no survivor --- #
    fold = 1
    for f in facs:
        if not f.in_output and not f.phys:
            fold *= f.size
    if fold > 1:
        tags.append("rule:fold_scale")

    # ---- final ordering: mirror _build_output_tensor ---------------------- #
    out_specs = sorted([s for s in specs if s.is_out], key=lambda s: s.id)
    primal_specs = sorted([s for s in specs if not s.is_out], key=lambda s: s.id)
    ordered = out_specs + primal_specs
    n_out = len(out_specs)
    id_map = {s.id: i for i, s in enumerate(ordered)}

    # ---- DEMAND-EMIT (L5): output-headed edge -> dense layout from the ONE
    # einsum. Each surviving coupled pair (meta M, blocks b1/b2) contracts an
    # extra eye(M) operand under a fresh partner label: the delta places the
    # off-diagonal zeros inside the einsum itself, and both pair sides become
    # plain dense dims (merged meta*block, row-major = the DiagonalIndex
    # ``n*block + r`` convention). No block-form buffer is ever emitted, so
    # the boundary's post-hoc dense() (kept as the non-planner fallback)
    # finds nothing left to expand.
    demand_eyes: list = []
    demand_ones: list = []
    if (
        _demand_emit_on()
        and (lhs.val is not None or rhs.val is not None)
        and (
            any(s.kind == "pair" for s in ordered)
            or any(
                s.kind == "dense"
                and any(not f.phys and f.size > 1 for f in s.letters)
                for s in ordered
            )
        )
    ):
        _pair_m2: dict = {}
        _new_specs: list = []
        for s in ordered:
            if s.kind != "pair":
                _new_specs.append(s)
                continue
            _key = tuple(sorted((s.id, s.other_id)))
            if _key not in _pair_m2:
                _m = s.meta_fac
                _m.in_output = True
                if not _m.phys:
                    # rides only on the eye operand (implicit/broadcast meta)
                    _m.axes[0] = "__eye__"
                _m2 = fac(_m.size)
                _m2.in_output = True
                _m2.axes[0] = "__eye__"
                _pair_m2[_key] = _m2
                demand_eyes.append((_m.size, _m, _m2))
                _meta = _m
            else:
                _meta = _pair_m2[_key]
            _grp = [_meta]
            if s.block_fac is not None and s.block_fac.size > 1:
                s.block_fac.in_output = True
                _grp.append(s.block_fac)
            _new_specs.append(
                _DimSpec(
                    s.is_out, s.id, "dense",
                    int(s.size * s.block_size), letters=_grp,
                )
            )
        # Every non-physical output letter rides a ones(N) operand: the
        # broadcast happens INSIDE the single einsum (fused epilogue), never
        # as a post-hoc forced_broadcast / boundary broadcast temp. Covers
        # implicit Compress dims and non-stored pair blocks alike.
        for s in _new_specs:
            if s.kind != "dense":
                continue
            for f in s.letters:
                if not f.phys and f.size > 1:
                    f.in_output = True
                    f.axes[0] = "__ones__"
                    demand_ones.append(f)
        ordered = _new_specs
        tags.append("out:demand_emit")

    # ---- physicality decisions -------------------------------------------- #
    base_val = any(f.in_output and f.phys for f in facs)
    final_phys: dict[str, bool] = {}
    pair_seen: set[tuple] = set()
    for s in ordered:
        if s.kind == "dense":
            anyp = any(f.phys for f in s.letters)
            for f in s.letters:
                final_phys[f.letter] = anyp
        else:
            key = tuple(sorted((s.id, s.other_id)))
            if key in pair_seen:
                continue
            pair_seen.add(key)
            partner = next(
                p for p in ordered
                if p.kind == "pair" and p.id == s.other_id and p.other_id == s.id
            )
            group = [s.meta_fac, s.block_fac, partner.block_fac]
            anyp = any(f.phys for f in group)
            blocks_need = base_val and (s.block_size > 1 or partner.block_size > 1)
            force = anyp or blocks_need
            final_phys[s.meta_fac.letter] = force
            # size-1 blocks never get an axis (mirror _build_sparse).
            final_phys[s.block_fac.letter] = force and s.block_fac.size > 1
            final_phys[partner.block_fac.letter] = (
                force and partner.block_fac.size > 1
            )

    # ---- slot layout (final val axes, pre-merge) --------------------------- #
    slot_facs: list[_Fac] = []
    group_slotcount: list[int] = []  # per ordered spec
    pair_meta_slot: dict[tuple, int] = {}
    for s in ordered:
        cnt = 0
        if s.kind == "dense":
            for f in s.letters:
                if final_phys[f.letter]:
                    slot_facs.append(f)
                    cnt += 1
        else:
            key = tuple(sorted((s.id, s.other_id)))
            if key not in pair_meta_slot:
                if final_phys[s.meta_fac.letter]:
                    pair_meta_slot[key] = len(slot_facs)
                    slot_facs.append(s.meta_fac)
                    cnt += 1
                else:
                    pair_meta_slot[key] = -1
            if final_phys[s.block_fac.letter]:
                slot_facs.append(s.block_fac)
                cnt += 1
        group_slotcount.append(cnt)

    # ---- build the einsum --------------------------------------------------#
    ins, subs = [], []
    for side in (0, 1):
        st = sts[side]
        if st.val is None:
            continue
        val = st.val
        new_shape, sub, changed = [], [], False
        for a in range(val.ndim):
            if a in split_maps[side]:
                f1, f2 = split_maps[side][a]
                new_shape += [f1.size, f2.size]
                sub += [f1.letter, f2.letter]
                changed = True
            elif a in ax_maps[side]:
                new_shape.append(int(val.shape[a]))
                sub.append(ax_maps[side][a].letter)
            elif int(val.shape[a]) == 1:
                changed = True  # squeeze an undescribed 1-axis
            else:
                raise _NoRule("unmapped_val_axis")
        if changed:
            val = val.reshape(new_shape)
        ins.append(val)
        subs.append(list(sub))

    if (demand_eyes or demand_ones) and ins:
        _eye_dt = jnp.result_type(*[_v.dtype for _v in ins])
        for _M, _u, _u2 in demand_eyes:
            ins.append(jnp.eye(_M, dtype=_eye_dt))
            subs.append([_u.letter, _u2.letter])
        for _f in demand_ones:
            ins.append(jnp.ones((_f.size,), dtype=_eye_dt))
            subs.append([_f.letter])

    out_letters = [f for f in slot_facs if f.phys]
    forced = [f for f in slot_facs if not f.phys]
    values = None
    if ins:
        # interleaved integer-label form: einsum(op0, sub0[, op1, sub1], out)
        _args = []
        for _v, _sub in zip(ins, subs):
            _args += [_v, _sub]
        _args.append([f.letter for f in out_letters])
        # OPT-IN f32 accumulation for an all-bf16 contraction
        # (GRAPHAX_PLANNER_F32_ACCUM=1; see _einsum_accum_dtype for the
        # measurement that keeps it off). Default and every other dtype mix
        # take the bare call, byte-identical to before.
        _pet = _einsum_accum_dtype([_v.dtype for _v in ins])
        values = (jnp.einsum(*_args, preferred_element_type=_pet)
                  if _pet is not None else jnp.einsum(*_args))
        if values.ndim == 0 and not forced:
            # fully-contracted physical part: fold the scalar into scalar_mult
            pass  # handled below

    # forced letters: metadata VALIDITY requires a physical axis (a merged
    # dense group with mixed sources, or a >1 block on a pair that carries a
    # val) — broadcast them in. This only happens where the incumbent would
    # have fully densified anyway.
    if forced:
        if values is None:
            # no physical source at all, but a pair block >1 must materialize:
            # only reachable when base_val is False -> forced is empty. Guard.
            raise _NoRule("forced_without_val")
        for f in forced:
            pos = slot_facs.index(f)
            values = jnp.expand_dims(values, pos)
        values = jnp.broadcast_to(values, tuple(f.size for f in slot_facs))
        tags.append("out:forced_broadcast")

    # ---- merge multi-letter dense dims / final reshape --------------------- #
    # Walk the slot layout in final-dim order: a dense group's slots collapse
    # into ONE physical axis (row-major merge — matching the DiagonalIndex
    # ``n*block + r`` logical convention); pair slots stay separate axes.
    if values is not None and values.ndim > 0:
        final_shape: list[int] = []
        slot_i = 0
        axis_of_slot: list[int] = []
        for s, cnt in zip(ordered, group_slotcount):
            if s.kind == "dense" and cnt > 0:
                merged = 1
                for _ in range(cnt):
                    merged *= slot_facs[slot_i].size
                    axis_of_slot.append(len(final_shape))
                    slot_i += 1
                final_shape.append(merged)
            else:
                for _ in range(cnt):
                    axis_of_slot.append(len(final_shape))
                    final_shape.append(slot_facs[slot_i].size)
                    slot_i += 1
        if tuple(values.shape) != tuple(final_shape):
            values = values.reshape(final_shape)
    else:
        axis_of_slot = []

    # ---- scalar_mult / scalar collapse ------------------------------------- #
    final_mult = _scaled_mul(lhs.scalar_mult, rhs.scalar_mult)
    if fold != 1:
        final_mult = final_mult * jnp.asarray(
            fold, dtype=jnp.asarray(final_mult).dtype
        )
    if values is not None and values.ndim == 0:
        final_mult = _scaled_mul(final_mult, values)
        values = None
    if os.environ.get("GRAPHAX_STRUCT_LOWER_SABOTAGE", "0") != "0":
        # TEST-ONLY negative control: deliberately perturb the result so the
        # differential harness proves it can detect a wrong rule.
        final_mult = final_mult * jnp.asarray(
            1.001, dtype=jnp.asarray(final_mult).dtype
        )

    # ---- output dims -------------------------------------------------------#
    def _slot_axis(f: _Fac):
        for i, sf in enumerate(slot_facs):
            if sf is f:
                return axis_of_slot[i] if i < len(axis_of_slot) else None
        return None

    dims_final: list = []
    for pos, s in enumerate(ordered):
        if s.kind == "dense":
            has_slots = any(final_phys[f.letter] for f in s.letters)
            axis = None
            if has_slots and values is not None:
                # merged group -> single axis: the axis of its first slot
                first = next(f for f in s.letters if final_phys[f.letter])
                axis = _slot_axis(first)
            if axis is None and s.size > 1:
                tags.append("out:implicit_kept")
            dims_final.append(DenseIndex(pos, s.size, axis=axis))
        else:
            key = tuple(sorted((s.id, s.other_id)))
            meta_axis = None
            if pair_meta_slot[key] != -1 and values is not None:
                meta_axis = axis_of_slot[pair_meta_slot[key]]
            block_axis = None
            if final_phys[s.block_fac.letter] and values is not None:
                block_axis = _slot_axis(s.block_fac)
            dims_final.append(
                DiagonalIndex(
                    pos,
                    s.size,
                    axis=meta_axis,
                    other_id=id_map[s.other_id],
                    block_size=s.block_size if s.block_size > 1 else None,
                    block_axis=block_axis if s.block_size > 1 else None,
                )
            )
    final_out = tuple(dims_final[:n_out])
    final_primal = tuple(dims_final[n_out:])
    if values is None:
        tags.append("out:val_none")

    out_dtype = values.dtype if values is not None else jnp.asarray(final_mult).dtype
    result = SparseTensor(
        final_out,
        final_primal,
        values,
        scalar_mult=jnp.asarray(final_mult).astype(out_dtype),
        fill_value=None,  # zero-fill in (gated), zero-fill out
    )

    # ---- honest op count: only what the einsum actually reduces ------------ #
    out_size = 1
    for s in ordered:
        out_size *= s.logical
    K = 1
    for f in facs:
        if not f.in_output and f.phys:
            K *= f.size
    if values is None and not ins:
        counts = (0, 0, 0)
    elif K <= 1:
        counts = (0, out_size, 0)
    else:
        counts = (0, out_size, out_size * (K - 1))
    return result, counts, tags
