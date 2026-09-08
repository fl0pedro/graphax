"""The one contraction, in four phases (ticket dsnn-3qm.72).

    PAIR    match the operands' dims by the contraction spec   metadata only
    FRAME   classify each axis, derive the output dims         metadata only
    DECIDE  choose the emission from the frame alone           metadata only
    EMIT    build the arrays                                   allocates

The first three phases touch no array. That is the point: the ``Frame`` is a
value you can print, assert on and test without running a contraction, so every
storage claim becomes a test on the frame rather than on the output.

This package is built BESIDE the existing engines and proved equivalent to them
before anything switches over. Nothing imports it yet.
"""
from .frame import (AxisCase, AxisRecord, EmissionKind, Frame, build_frame,
                    decide_emission, multiply_budget_bytes)

__all__ = ["AxisCase", "AxisRecord", "EmissionKind", "Frame", "build_frame",
           "decide_emission", "multiply_budget_bytes"]
