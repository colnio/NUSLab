"""
Breakdown detection: absolute current threshold, confirmed over N points.

Why a threshold rather than the compliance flag: compliance is the hardware's
last-resort protection and should sit well above the level that means "this
device has failed". Detecting at a threshold *below* compliance lets the stress
be cut before the device is driven at the full protection current, and it makes
the failure criterion an explicit, recorded experimental parameter instead of an
artefact of how the protection was configured.

Why N consecutive points: a single noisy sample must never end a device. Two
consecutive over-threshold readings cost one extra measurement period -- tens of
milliseconds -- and remove essentially all single-sample false positives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BreakdownCriterion:
    #: Magnitude of current that counts as an excursion.
    i_threshold_A: float
    #: How many consecutive excursions confirm a breakdown.
    consecutive: int = 2


@dataclass(frozen=True)
class BreakdownEvent:
    #: Point where the excursion began. This is the breakdown, so its voltage is
    #: V_BD and its elapsed time is t_BD.
    index: int
    voltage: float
    current: float
    elapsed_s: float
    #: Point at which the run reached ``consecutive`` and the event was declared.
    confirmed_index: int


class BreakdownDetector:
    """Feed it points in order; it returns an event once, then latches."""

    def __init__(self, criterion: BreakdownCriterion):
        if criterion.i_threshold_A <= 0:
            raise ValueError("Breakdown threshold must be greater than 0 A.")
        if int(criterion.consecutive) < 1:
            raise ValueError("Breakdown criterion needs at least 1 point.")
        self.criterion = criterion
        self._run_start: Optional[tuple] = None
        self._run_length = 0
        self._event: Optional[BreakdownEvent] = None

    @property
    def event(self) -> Optional[BreakdownEvent]:
        return self._event

    def reset(self) -> None:
        self._run_start = None
        self._run_length = 0
        self._event = None

    def update(
        self, index: int, voltage: float, current: float, elapsed_s: float
    ) -> Optional[BreakdownEvent]:
        if self._event is not None:
            return self._event

        # A dropped or overflowed reading cannot bridge a "consecutive" run.
        # The stress runner retries once and aborts if it remains invalid.
        if current is None or math.isnan(float(current)):
            self._run_start = None
            self._run_length = 0
            return None

        if abs(float(current)) < self.criterion.i_threshold_A:
            self._run_start = None
            self._run_length = 0
            return None

        if self._run_start is None:
            self._run_start = (int(index), float(voltage), float(current),
                               float(elapsed_s))
        self._run_length += 1

        if self._run_length >= int(self.criterion.consecutive):
            start_index, start_v, start_i, start_t = self._run_start
            self._event = BreakdownEvent(
                index=start_index,
                voltage=start_v,
                current=start_i,
                elapsed_s=start_t,
                confirmed_index=int(index),
            )
        return self._event
