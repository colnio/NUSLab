"""
Choosing the constant-voltage-stress level from accumulated RVS results.

The physics, in short. A CVS run holds a fixed voltage until the dielectric
fails, giving a time-to-breakdown t_BD. The stress level decides whether that
takes a useful amount of time:

* Too close to V_BD and the device dies during the pre-ramp -- you have measured
  the ramp, not the hold.
* Too far below and t_BD runs past any practical session.

Standard practice is ``V_CVS = k * V_BD`` with ``k`` around 0.80-0.92, where
V_BD comes from RVS on sister devices of the *same sample and same crosspoint
size*. The median is used rather than the mean because a single anomalous
device -- a bad probe landing, a particle -- would otherwise drag the whole
campaign's stress level with it.

To extrapolate a lifetime at operating field you need **three or more** distinct
``k`` values, so the batch spans a range of fields; see
``Breakdown/docs/cvs-voltage-selection.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean, median
from typing import Dict, List, Optional, Tuple

#: Below this many devices the median is not yet meaningful.
MIN_RECORDS_FOR_CONFIDENCE = 3

#: Practical band for k. Outside it the run is likely to be a waste of a device.
K_TOO_HIGH = 0.95
K_TOO_LOW = 0.70

#: Ramp rates differing by more than this factor should not be pooled.
RAMP_RATE_SPREAD_LIMIT = 2.0


@dataclass(frozen=True)
class VbdRecord:
    device_index: int
    crosspoint_um: float
    #: Sign is discarded on entry -- statistics work on magnitude, so a
    #: negative-polarity campaign behaves identically to a positive one.
    v_bd: float
    ramp_rate_Vps: float


@dataclass
class Recommendation:
    #: ``None`` when there is nothing to base a recommendation on.
    voltage: Optional[float]
    basis_text: str
    warnings: List[str] = field(default_factory=list)


def _size_key(crosspoint_um: float) -> str:
    return f"{float(crosspoint_um):g}"


class VbdStatistics:
    """Breakdown voltages seen so far, bucketed by crosspoint size."""

    def __init__(self) -> None:
        self._by_size: Dict[str, List[VbdRecord]] = {}

    def add(self, record: VbdRecord) -> None:
        stored = VbdRecord(
            device_index=record.device_index,
            crosspoint_um=record.crosspoint_um,
            v_bd=abs(float(record.v_bd)),
            ramp_rate_Vps=float(record.ramp_rate_Vps),
        )
        self._by_size.setdefault(_size_key(record.crosspoint_um), []).append(stored)

    def records_for(self, crosspoint_um: float) -> List[VbdRecord]:
        return list(self._by_size.get(_size_key(crosspoint_um), ()))

    def count(self, crosspoint_um: float) -> int:
        return len(self._by_size.get(_size_key(crosspoint_um), ()))

    def statistic(self, crosspoint_um: float, kind: str = "median") -> Optional[float]:
        values = [r.v_bd for r in self.records_for(crosspoint_um)]
        if not values:
            return None
        if kind == "mean":
            return float(fmean(values))
        return float(median(values))

    def ramp_rate_span(self, crosspoint_um: float) -> Optional[Tuple[float, float]]:
        rates = [r.ramp_rate_Vps for r in self.records_for(crosspoint_um)
                 if r.ramp_rate_Vps > 0]
        return (min(rates), max(rates)) if rates else None


def recommend(
    stats: VbdStatistics,
    crosspoint_um: float,
    k: float,
    statistic: str = "median",
) -> Recommendation:
    """Suggest a CVS stress level, with every caveat that applies to it."""
    count = stats.count(crosspoint_um)
    if count == 0:
        return Recommendation(
            voltage=None,
            basis_text=(
                f"No RVS breakdown recorded yet for {_size_key(crosspoint_um)} um "
                f"devices on this sample."
            ),
            warnings=[
                "No RVS result to base a stress level on. Run an RVS device "
                "first, or enter a voltage manually."
            ],
        )

    centre = stats.statistic(crosspoint_um, statistic)
    voltage = float(k) * float(centre)
    basis_text = (
        f"{k:g} x {statistic}(V_BD) over {count} RVS "
        f"device{'s' if count != 1 else ''} at {_size_key(crosspoint_um)} um "
        f"= {k:g} x {centre:.4g} V = {voltage:.4g} V"
    )

    warnings: List[str] = []
    if count < MIN_RECORDS_FOR_CONFIDENCE:
        warnings.append(
            f"Only {count} RVS device{'s' if count != 1 else ''} so far -- the "
            f"{statistic} is a weak basis. Expect to revise this level."
        )
    if k > K_TOO_HIGH:
        warnings.append(
            f"k = {k:g} is very close to V_BD. The device may break down during "
            f"the pre-ramp, giving an immediate t_BD that measures the ramp "
            f"rather than the hold."
        )
    elif k < K_TOO_LOW:
        warnings.append(
            f"k = {k:g} is well below V_BD. t_BD may run past any practical "
            f"measurement window."
        )

    span = stats.ramp_rate_span(crosspoint_um)
    if span and span[0] > 0 and span[1] / span[0] > RAMP_RATE_SPREAD_LIMIT:
        warnings.append(
            f"Ramp rate varies from {span[0]:g} to {span[1]:g} V/s across these "
            f"devices. V_BD is ramp rate dependent, so pooling them makes the "
            f"{statistic} hard to interpret."
        )

    return Recommendation(voltage=voltage, basis_text=basis_text, warnings=warnings)
