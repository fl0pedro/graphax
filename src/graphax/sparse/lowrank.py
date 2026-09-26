"""Edge-level LowRank tensor — the factored lattice class (L3).

``LowRankTensor`` represents ``T = U @ V``: ``U`` holds ``(out_dims | rank)``,
``V`` holds ``(rank | primal_dims)``, both plain ``SparseTensor``s whose rank
dims agree in logical extent (ids are per-tensor contiguous in graphax; the
factors contract positionally through the normal matmul alignment). Low rank is NOT a per-dim property — it couples the out side
with the primal side through FACTORED STORAGE — so it is a wrapper around two
``SparseTensor``s, unwrapped at the single ``matmul()`` / ``elementwise()``
entries, NOT an ``Index`` subclass (keeps ``Index``/``val`` untouched and the
closure logic in the op entries; see the lattice module docstring).

Closure (formal, property-tested against the dense oracle):

  contract:  (U V) @ B  = U (V @ B)          factor_right — stays LowRank
             A @ (U V)  = (A U) V            factor_left  — stays LowRank
             (U1 V1) @ (U2 V2) = U1 ((V1 U2) V2)   middle_fold — stays LowRank,
                                             middle is r1 x r2 (small by def)
  add:       LR + LR (same dims, dense-stored factors, r1+r2 <= max_rank)
                        = rank-concat        rank_concat — stays LowRank
             anything else                   spill_dense — materialize U @ V
                                             and delegate (counted)

Every other operation spills. Spilling is ALWAYS correct (``U @ V`` runs
through the proven matmul entry), so the wrapper is total by construction —
the same guarantee shape as the planner's SPILL rule.

OPT-IN by measurement (2026-08-03): three censuses found no production site
where factored storage pays — intermediate edges are never low-rank on any
model family (the per-sample outer-product structure of ``dy/dW`` is already
captured algebraically by the coupled-pair + val storage and never
materializes mid-elimination), and the nn256 final Jacobian blocks are
globally full-rank (effective rank ≈ min dim). Nothing in production
constructs a ``LowRankTensor``; the class exists so the lattice is formally
total over its declared classes, proven by lowrank_property_test.py.

Counting caveat: with ``count=True`` the factor-side ops' counts are summed;
a spill's materialization matmul is counted only on the matmul path (the
elementwise count contract is a single scalar and cannot carry it) — fine for
the formal wrapper, revisit if a production constructor ever lands.
"""
from __future__ import annotations

import os

LOWRANK_STATS: dict[str, int] = {}


def _bump(key: str) -> None:
    LOWRANK_STATS[key] = LOWRANK_STATS.get(key, 0) + 1


def _max_rank() -> int:
    return int(os.environ.get("GRAPHAX_LOWRANK_MAX_RANK", "64"))


class LowRankTensor:
    """``T = U @ V`` with the rank dim shared by id between the factors."""

    _is_lowrank = True  # duck-type marker read by the op entries

    def __init__(self, u, v):
        if len(u.primal_dims) != 1 or len(v.out_dims) != 1:
            raise ValueError(
                "LowRankTensor factors must share exactly ONE rank dim: "
                f"u.primal_dims={len(u.primal_dims)}, v.out_dims={len(v.out_dims)}."
            )
        ur, vr = u.primal_dims[0], v.out_dims[0]
        if ur.logical_size != vr.logical_size:
            # ids are PER-TENSOR contiguous in graphax (consistency assert);
            # the factors contract positionally through the normal matmul
            # alignment, so only the logical extent must agree.
            raise ValueError(
                "Rank dims must match by logical size: "
                f"u {ur.logical_size} vs v {vr.logical_size}."
            )
        self.u = u
        self.v = v

    # ---- SparseTensor-compatible surface (read-only) ----------------------
    @property
    def out_dims(self):
        return self.u.out_dims

    @property
    def primal_dims(self):
        return self.v.primal_dims

    @property
    def dims(self):
        return (*self.out_dims, *self.primal_dims)

    @property
    def rank(self) -> int:
        return int(self.u.primal_dims[0].logical_size)

    @property
    def dtype(self):
        return self.u.dtype

    @property
    def shape(self):
        return (*[d.logical_size for d in self.out_dims],
                *[d.logical_size for d in self.primal_dims])

    def materialize(self, count: bool = False):
        """Spill: contract the factors through the proven matmul entry."""
        from graphax.sparse.ops.matmul import matmul

        return matmul(self.u, self.v, count=count)

    def dense(self):
        return self.materialize().dense()

    def __repr__(self):
        return (f"LowRankTensor(rank={self.rank}, out={self.out_dims}, "
                f"primal={self.primal_dims})")


def _is_lr(t) -> bool:
    return getattr(t, "_is_lowrank", False)


def _combine_counts(a, b):
    return tuple(int(x) + int(y) for x, y in zip(a, b))


def lowrank_matmul(lhs, rhs, count: bool = False):
    """Contract with at least one LowRank operand — factor-side closure."""
    from graphax.sparse.ops.matmul import matmul

    if _is_lr(lhs) and _is_lr(rhs):
        _bump("contract:middle_fold")
        mid = matmul(lhs.v, rhs.u, count=count)          # r1 x r2
        if count:
            mid, c1 = mid
            u1m, c2 = matmul(lhs.u, mid, count=True)
            return LowRankTensor(u1m, rhs.v), _combine_counts(c1, c2)
        return LowRankTensor(matmul(lhs.u, mid), rhs.v)
    if _is_lr(lhs):
        _bump("contract:factor_right")
        out = matmul(lhs.v, rhs, count=count)
        if count:
            vb, c = out
            return LowRankTensor(lhs.u, vb), c
        return LowRankTensor(lhs.u, out)
    _bump("contract:factor_left")
    out = matmul(lhs, rhs.u, count=count)
    if count:
        au, c = out
        return LowRankTensor(au, rhs.v), c
    return LowRankTensor(out, rhs.v)


def _dense_stored(st) -> bool:
    """Factor storable for rank-concat: every dim a plain stored DenseIndex."""
    return st.val is not None and all(
        getattr(d, "other_id", None) is None and d.axis is not None
        for d in st.dims
    )


def _same_logical(a, b) -> bool:
    return ([d.logical_size for d in a.out_dims]
            == [d.logical_size for d in b.out_dims]
            and [d.logical_size for d in a.primal_dims]
            == [d.logical_size for d in b.primal_dims])


def lowrank_elementwise(lhs, rhs, op, is_intersection: bool = False,
                        count: bool = False):
    """Elementwise with at least one LowRank operand.

    ``add``/``subtract`` of two LowRank edges with identical logical dims,
    dense-stored factors and ``r1 + r2 <= GRAPHAX_LOWRANK_MAX_RANK`` is a
    rank-concat (zero compute — subtract negates the second U factor).
    Everything else spills to dense and delegates.
    """
    import jax.numpy as jnp
    from dataclasses import replace
    from graphax.sparse.ops.elementwise import elementwise
    from graphax.sparse.tensor import SparseTensor

    name = getattr(op, "__name__", "")
    if (
        _is_lr(lhs) and _is_lr(rhs)
        and name in ("add", "subtract")
        and _same_logical(lhs, rhs)
        and lhs.rank + rhs.rank <= _max_rank()
        and _dense_stored(lhs.u) and _dense_stored(lhs.v)
        and _dense_stored(rhs.u) and _dense_stored(rhs.v)
        and lhs.u.primal_dims[0].axis == rhs.u.primal_dims[0].axis
        and lhs.v.out_dims[0].axis == rhs.v.out_dims[0].axis
    ):
        _bump(f"add:rank_concat:{name}")
        rax_u = lhs.u.primal_dims[0].axis
        rax_v = lhs.v.out_dims[0].axis
        ru_val = rhs.u.val if name == "add" else -rhs.u.val
        # fold each side's scalar_mult into its own stored factor before the
        # concat — the concatenated buffer can only carry ONE multiplier.
        from graphax.sparse.dtype_compute import _scaled_mul
        lu = _scaled_mul(lhs.u.val, lhs.u.scalar_mult)
        lv = _scaled_mul(lhs.v.val, lhs.v.scalar_mult)
        ru = _scaled_mul(ru_val, rhs.u.scalar_mult)
        rv = _scaled_mul(rhs.v.val, rhs.v.scalar_mult)
        r_new = lhs.rank + rhs.rank
        u_dims = tuple(
            replace(d, size=r_new) if i == len(lhs.u.out_dims) else d
            for i, d in enumerate((*lhs.u.out_dims, *lhs.u.primal_dims))
        )
        v_dims = tuple(
            replace(d, size=r_new) if i == 0 else d
            for i, d in enumerate((*lhs.v.out_dims, *lhs.v.primal_dims))
        )
        n_u_out = len(lhs.u.out_dims)
        u_new = SparseTensor(
            u_dims[:n_u_out], u_dims[n_u_out:],
            jnp.concatenate([lu, ru], axis=rax_u),
        )
        v_new = SparseTensor(
            v_dims[:1], v_dims[1:],
            jnp.concatenate([lv, rv], axis=rax_v),
        )
        out = LowRankTensor(u_new, v_new)
        if count:
            return out, 0
        return out

    _bump(f"add:spill_dense:{name or 'op'}")
    a = lhs.materialize() if _is_lr(lhs) else lhs
    b = rhs.materialize() if _is_lr(rhs) else rhs
    return elementwise(a, b, op, is_intersection=is_intersection, count=count)
