"""
Sweep point generation for the MFIA measurements.

Kept separate from :mod:`Breakdown.core.mfia` so the geometry of a sweep can be
tested without ``zhinst`` installed.

Both sweeps are bidirectional, but in different shapes:

* **C(f)** is a there-and-back over a log frequency axis: ``f_min -> f_max ->
  f_min`` at fixed bias.
* **C(V)** is a full hysteresis loop through zero: ``0 -> v_max -> v_min -> 0``,
  linearly spaced. Starting and ending at zero bias matters for a dielectric --
  the loop is what reveals charge trapping.

This replaces ``build_block_path`` (``cv_cf_recipe_ui.py:516-527``), which forces
*every* sweep through zero. That is right for a bias axis and meaningless for a
log frequency axis, where zero is not on the scale at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

#: Two points closer than this are treated as the same setpoint.
_EPS = 1e-12


@dataclass(frozen=True)
class SweepPoint:
    #: Position in the whole sweep, 0-based.
    index: int
    value: float
    #: ``'fwd'`` while the swept quantity increases, ``'rev'`` while it decreases.
    direction: str
    #: Which leg of the sweep this point belongs to. C(f) has 2, C(V) has 3.
    #: Needed because C(V) has two separate increasing branches.
    segment: int


def _linspace(start: float, stop: float, count: int) -> List[float]:
    if count < 2:
        return [float(stop)]
    span = float(stop) - float(start)
    return [float(start) + span * i / (count - 1) for i in range(count)]


def _logspace(start: float, stop: float, count: int) -> List[float]:
    if count < 2:
        return [float(stop)]
    log_a = math.log10(float(start))
    log_b = math.log10(float(stop))
    span = log_b - log_a
    return [10.0 ** (log_a + span * i / (count - 1)) for i in range(count)]


def _assemble(segments: Sequence[Sequence[float]]) -> List[SweepPoint]:
    """Join segments, dropping the duplicate point at each junction.

    Without this the turning points get measured twice, which shows up in
    analysis as a phantom pair of identical readings.
    """
    points: List[SweepPoint] = []
    for segment_index, values in enumerate(segments):
        for value in values:
            if points and abs(points[-1].value - value) <= _EPS:
                continue
            direction = "fwd"
            if points and value < points[-1].value:
                direction = "rev"
            points.append(
                SweepPoint(
                    index=len(points),
                    value=float(value),
                    direction=direction,
                    segment=segment_index,
                )
            )
    return _relabel_first_point(points)


def _relabel_first_point(points: List[SweepPoint]) -> List[SweepPoint]:
    """The first point has no predecessor; give it its segment's direction."""
    if len(points) < 2:
        return points
    first, second = points[0], points[1]
    if first.direction != second.direction:
        points[0] = SweepPoint(
            index=first.index,
            value=first.value,
            direction=second.direction,
            segment=first.segment,
        )
    return points


def build_cf_points(f_min: float, f_max: float, points: int) -> List[SweepPoint]:
    """Log-spaced ``f_min -> f_max -> f_min``.

    ``points`` counts one direction; the returned sweep has ``2*points - 1``
    entries because the turning point is shared.
    """
    if f_min <= 0 or f_max <= 0:
        raise ValueError("C(F) frequencies must be greater than 0 for a log sweep.")
    if f_min >= f_max:
        raise ValueError("C(F) f_min must be less than f_max.")
    count = max(2, int(points))

    forward = _logspace(f_min, f_max, count)
    return _assemble([forward, list(reversed(forward))])


def build_cv_points(v_min: float, v_max: float, points: int) -> List[SweepPoint]:
    """Linear ``0 -> v_max -> v_min -> 0``.

    ``points`` counts each leg of the loop.
    """
    if v_min >= v_max:
        raise ValueError("C(V) v_min must be less than v_max.")
    count = max(2, int(points))

    return _assemble(
        [
            _linspace(0.0, v_max, count),
            _linspace(v_max, v_min, count),
            _linspace(v_min, 0.0, count),
        ]
    )
