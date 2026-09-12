"""The structure lattice — formal classes and closure rules.

Every ``Index`` on a ``SparseTensor`` belongs to exactly one lattice class;
every binary op's result class is a TABLE entry, not a per-site derivation.
The table is normative: the property suite (lattice_property_test.py) draws
random structured tensors per class pair and fails if an engine's result
class or values disagree with the table + the dense oracle.

Classes (a dim either owns explicit physical axes with sizes, or is implicit
— broadcast to its logical size, no storage):

  DENSE        plain axis (explicit) — DenseIndex with axis set
  IMPLICIT     broadcast axis (axis is None, logical size stored only)
  BLOCKDIAG    one member of a coupled DiagonalIndex pair, meta > 1
  DEGENERATE   coupled pair with meta == 1 — ONE full block, i.e. ZERO
               sparsity information; ≡ DENSE by definition (this table entry
               is what the 2026-08-02 wrong-shaped-Jacobian fix enforces at
               the planner boundary)
  LOWRANK      edge-level factored tensor (U, V) — not an Index property;
               closure lives at the matmul()/elementwise() entry (L3).
"""
from __future__ import annotations

from enum import Enum


class L(Enum):
    DENSE = "dense"
    IMPLICIT = "implicit"
    BLOCKDIAG = "blockdiag"
    DEGENERATE = "degenerate"
    LOWRANK = "lowrank"  # EDGE-level (whole tensor factored), not per-dim


def classify_edge(t) -> L | None:
    """Edge-level lattice class, or ``None`` for a per-dim classified tensor.

    ``LowRankTensor`` is the one edge-level class: the factored storage
    couples the out side with the primal side, so no single dim carries it.
    """
    return L.LOWRANK if getattr(t, "_is_lowrank", False) else None


def classify(d) -> L:
    """Lattice class of one Index.

    GAP, stated not papered over (ticket dsnn-3qm.62): a BLOCKED DENSE dim
    (``other_id is None`` with a ``block_size``, ``block_axis`` None) is DENSE in
    its stored meta and IMPLICIT in its block — one dim in two classes at once,
    which this enum cannot name, and it comes out ``DENSE`` below. The tables
    keyed on ``DENSE`` are still SOUND for it wherever the engine treats its
    block as the implicit extent it is (which is what ``matmul`` and ``dense()``
    do); they are simply silent about the block. Naming it would mean a sixth
    class and a row and column in every table, with the property suite to
    match — an owner decision, not a local one."""
    if getattr(d, "other_id", None) is not None:
        return L.DEGENERATE if int(d.size) == 1 else L.BLOCKDIAG
    return L.IMPLICIT if getattr(d, "axis", None) is None else L.DENSE


# --------------------------------------------------------------------------- #
# CONTRACT closure — the class of the SURVIVING partner when a (lhs_dim,
# rhs_dim) pair is contracted. Keys are (class(lhs_con), class(rhs_con));
# values name the rule family; the property suite checks VALUES against the
# dense oracle and (where stated) the surviving structure.
#
# The entries encode the proven facts:
#  * DEGENERATE behaves as DENSE everywhere (definition).
#  * BLOCKDIAG x BLOCKDIAG with equal meta stays BLOCKDIAG (contract_direct);
#    with commensurable meta the finer side REFINES onto the coarser grid
#    (contract_refine_*) and the pair still survives sparse;
#    with coprime metas the pair spills (dense survivor) — allowed, counted.
#  * BLOCKDIAG x DENSE: the block side's free partner survives DENSE at
#    N*block (the D@B closure).
#  * IMPLICIT x IMPLICIT contracting: analytic scale-by-N fold, no compute.
# --------------------------------------------------------------------------- #
CONTRACT = {
    (L.DENSE, L.DENSE): "dense",
    (L.DENSE, L.IMPLICIT): "sum_reduce",
    (L.IMPLICIT, L.DENSE): "sum_reduce",
    (L.IMPLICIT, L.IMPLICIT): "scale_fold",
    (L.BLOCKDIAG, L.BLOCKDIAG): "block_contract",   # direct/refine/spill
    (L.BLOCKDIAG, L.DENSE): "db_closure",
    (L.DENSE, L.BLOCKDIAG): "db_closure",
    (L.BLOCKDIAG, L.IMPLICIT): "db_closure",
    (L.IMPLICIT, L.BLOCKDIAG): "db_closure",
}
# DEGENERATE aliases to DENSE for every partner:
for _k in list(CONTRACT):
    _a, _b = _k
    if L.DEGENERATE not in _k:
        if _a == L.DENSE:
            CONTRACT[(L.DEGENERATE, _b)] = CONTRACT[_k]
        if _b == L.DENSE:
            CONTRACT[(_a, L.DEGENERATE)] = CONTRACT[_k]
CONTRACT[(L.DEGENERATE, L.DEGENERATE)] = "dense"


# --------------------------------------------------------------------------- #
# ADD closure (union support). DENSE absorbs everything (a union with a dense
# operand is dense — mathematics, not laziness). Aligned BLOCKDIAG pairs stay
# BLOCKDIAG; misaligned commensurable pairs coarsen to the shared meta;
# coprime factorings spill.
# --------------------------------------------------------------------------- #
ADD = {
    (L.DENSE, L.DENSE): "dense",
    (L.DENSE, L.BLOCKDIAG): "dense_absorb",
    (L.BLOCKDIAG, L.DENSE): "dense_absorb",
    (L.BLOCKDIAG, L.BLOCKDIAG): "block_add",        # aligned/coarsen/spill
    (L.IMPLICIT, L.IMPLICIT): "implicit",
    (L.DENSE, L.IMPLICIT): "broadcast_add",
    (L.IMPLICIT, L.DENSE): "broadcast_add",
    (L.BLOCKDIAG, L.IMPLICIT): "broadcast_add",
    (L.IMPLICIT, L.BLOCKDIAG): "broadcast_add",
}
for _k in list(ADD):
    _a, _b = _k
    if L.DEGENERATE not in _k:
        if _a == L.DENSE:
            ADD[(L.DEGENERATE, _b)] = ADD[_k]
        if _b == L.DENSE:
            ADD[(_a, L.DEGENERATE)] = ADD[_k]
ADD[(L.DEGENERATE, L.DEGENERATE)] = "dense"


# --------------------------------------------------------------------------- #
# EDGE-level closure — LowRank (L3). Keys name the operand kinds ("lowrank"
# vs "sparse" = any per-dim classified SparseTensor); values name the rule
# family, property-tested in lowrank_property_test.py. Every non-listed
# combination spills (materialize U @ V through the proven matmul entry and
# delegate) — total by construction, counted in lowrank.LOWRANK_STATS.
# Measured basis for LowRank being formal-only (no production constructor):
# the 2026-08-03 censuses — intermediates are never low-rank on any model
# family and the nn256 final Jacobian blocks are globally full-rank.
# --------------------------------------------------------------------------- #
LOWRANK_CONTRACT = {
    ("lowrank", "sparse"): "factor_right",   # (U V) @ B = U (V @ B)
    ("sparse", "lowrank"): "factor_left",    # A @ (U V) = (A U) V
    ("lowrank", "lowrank"): "middle_fold",   # U1 ((V1 U2) V2), middle r1 x r2
}
LOWRANK_ADD = {
    ("lowrank", "lowrank"): "rank_concat_or_spill",  # concat if same dims,
    # dense-stored factors, r1+r2 <= GRAPHAX_LOWRANK_MAX_RANK; else spill
    ("lowrank", "sparse"): "spill_dense",            # dense absorbs (union)
    ("sparse", "lowrank"): "spill_dense",
}
