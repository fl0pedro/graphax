"""The output-layout contract of ``jacve`` (ticket dsnn-3qm.62).

A returned parameter gradient is stored in PARAMETER LAYOUT: the physical axes
of ``val`` appear in the order of the logical dims they belong to, so a dim
that owns a val axis has ``axis == position`` once every dim is materialized
(the dense case), and in general the sequence of materialized axes read along
``out_dims + primal_dims`` (a pair's ``axis`` before its ``block_axis``) is
``0, 1, 2, ...``. Implicit dims (``axis is None``) own no axis and are skipped.

WHY. The two contraction engines store the same logical tensor differently:
the einsum planner assigns ``axis == position`` while the tiled engine leaves
the axes where the grid permutation dropped them (a 2-D weight gradient
transposed in storage, ``axes (1, 0)``). Both are correct values. But
``SparseTensor`` puts its ``Index`` tuple into the pytree AUX data, so two
gradients that differ only in axis assignment cannot be ``tree_map``ped
together, and the reward path compared them with a silently clamped 0.0
(finding 60). One layout at the boundary removes AXIS ASSIGNMENT as a source
of that mismatch.

WHAT IT DOES NOT DO, measured (ticket dsnn-3qm.72, 2026-09-10). This contract
does NOT make the returned pytree structure a function of the logical tensor
alone, and an earlier revision of this docstring claimed it did. It equalises
``axis``; it cannot equalise the STORED FORM. An approximated plan returns a
gradient whose ``val`` has a different rank or extents than the exact arm's
(a Compress drops an axis, a Diag re-factors one, a dim can come back
implicit), and a contraction a plan SKIPPED returns no tensor at all -- the
sparse branch of ``core.vertex_elimination_jaxpr`` appends ``None`` there
while the dense branch appends ``zeros_like``, so the two return forms do not
even agree on the LEAF COUNT. Measured on ``VmappedNeuralNetwork``/mnist,
twelve random per-face plans per cell, the approximated and exact
``sparse_representation=True`` returns had different pytree structure in 1 of
12 plans on the fwd order (Diag/Compress/Quant) and 11 of 12 on the rev order
once ``SKIP_FACE`` was in the plan family; on the same plans the
``sparse_representation=False`` returns were shape-identical 96 times out of
96.

So a consumer that compares an approximated gradient against the exact one
MUST compare the LOGICAL tensors -- take the dense return form, or densify
both sides -- and must not read the raw pytree children. Comparing the
children is not merely fragile: ``val`` carries neither ``scalar_mult`` nor
the implicit fill, so a residual taken over children is wrong even when the
structures happen to match (measured: relative Frobenius 7.60 where the
logical answer is 0.0000), and a Quant to an 8-bit float makes an
``x.astype`` -free comparison RAISE on dtype promotion.
``tests/core/sparse_tensor/output_layout_test.py`` pins both halves of this:
the structure is one pytree only for exact/quant, and the densified sparse
return equals the dense return for every class.

``canonical_output_layout`` permutes ``val`` (one transpose, or nothing when
the layout already holds, which is the planner's case and every 1-D case) and
renumbers the pointers. ``is_parameter_layout`` is the assertion.
"""
from __future__ import annotations

from dataclasses import replace


def _materialized_axes(st) -> list[int]:
    """The DISTINCT val axes the dims point to, in the order of first
    appearance along ``out_dims + primal_dims`` (a dim's ``axis`` before its
    ``block_axis``). The two members of a diagonal pair share their axes, so
    a pair contributes each axis once; an implicit dim contributes nothing."""
    seen: list[int] = []
    for d in st.dims:
        for a in (d.axis, getattr(d, "block_axis", None) if getattr(d, "is_sparse", False) else None):
            if a is not None and int(a) not in seen:
                seen.append(int(a))
    return seen


def is_parameter_layout(st) -> bool:
    """True iff the distinct materialized axes read ``0, 1, 2, ...`` along
    the dims and cover ``val``."""
    val = getattr(st, "val", None)
    if val is None:
        return True
    axes = _materialized_axes(st)
    return axes == list(range(val.ndim))


def canonical_output_layout(st):
    """Return ``st`` with ``val`` in parameter layout (see the module doc).

    A tensor whose pointers do not cover ``val`` exactly (an orphan val axis)
    is returned unchanged as well: that is a different invariant's business.
    """
    val = getattr(st, "val", None)
    if val is None:
        return st
    old_axes = _materialized_axes(st)
    if sorted(old_axes) != list(range(val.ndim)):
        return st
    if old_axes == list(range(val.ndim)):
        return st
    # new position of each old val axis: the rank of its pointer in dim order
    new_of_old = {a: i for i, a in enumerate(old_axes)}
    new_val = val.transpose(old_axes)

    def _renum(d):
        kw = {}
        if d.axis is not None:
            kw["axis"] = new_of_old[int(d.axis)]
        if getattr(d, "is_sparse", False) and d.block_axis is not None:
            kw["block_axis"] = new_of_old[int(d.block_axis)]
        return replace(d, **kw) if kw else d

    from graphax.sparse.tensor import SparseTensor

    return SparseTensor(
        tuple(_renum(d) for d in st.out_dims),
        tuple(_renum(d) for d in st.primal_dims),
        new_val,
        scalar_mult=st.scalar_mult,
        fill_value=st.fill_value,
        pre_transforms=st.pre_transforms,
        post_transforms=st.post_transforms,
        check_consistency=False,
    )
