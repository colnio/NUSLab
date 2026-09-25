"""
Translating a requested ramp rate into a step size and dwell the SMU can hold.

The user asks for volts per second. The instrument delivers points, each costing
integration time plus source delay plus a GPIB round trip. Those two views have
to be reconciled honestly, because **V_BD depends on the ramp rate** -- a run
that silently ramps at 1.7 V/s when 10 V/s was requested produces a number that
cannot be compared with anything.

So the plan reports what is actually achievable and says so when the request
cannot be met, rather than quietly substituting a different experiment.

Kept free of instrument imports so the arithmetic is testable anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

#: Typical GPIB round trip for a `:READ?` on a 2400. Measured behaviour, not spec.
DEFAULT_BUS_OVERHEAD_S = 0.010

#: Below this a voltage step is smaller than the source's own resolution and
#: settling noise, so slow ramps wait longer rather than step finer.
DEFAULT_MIN_STEP_V = 1e-4


@dataclass(frozen=True)
class RampPlan:
    step_V: float
    dwell_s: float
    requested_rate_Vps: float
    achievable_rate_Vps: float
    #: True when the hardware cannot keep up with the requested rate.
    rate_limited: bool
    note: str

    @property
    def points_per_volt(self) -> float:
        return 1.0 / self.step_V if self.step_V else float("inf")


def point_period_s(
    nplc: float,
    source_delay_s: float = 0.0,
    line_freq_hz: float = 50.0,
    bus_overhead_s: float = DEFAULT_BUS_OVERHEAD_S,
) -> float:
    """Shortest time one measured point can take.

    Integration is ``nplc / line_freq`` seconds -- NPLC 1 on 50 Hz mains is 20 ms.
    """
    if line_freq_hz <= 0:
        raise ValueError("line_freq_hz must be greater than 0.")
    integration = max(0.0, float(nplc)) / float(line_freq_hz)
    return integration + max(0.0, float(source_delay_s)) + max(0.0, float(bus_overhead_s))


def plan_ramp(
    rate_Vps: float,
    max_step_V: float,
    nplc: float,
    source_delay_s: float = 0.0,
    line_freq_hz: float = 50.0,
    bus_overhead_s: float = DEFAULT_BUS_OVERHEAD_S,
    min_step_V: float = DEFAULT_MIN_STEP_V,
) -> RampPlan:
    """Work out the step and dwell that best deliver ``rate_Vps``.

    Three regimes:

    * **Normal** -- the ideal step fits under ``max_step_V``; run at the period
      floor and hit the requested rate exactly.
    * **Too fast** -- the ideal step is coarser than ``max_step_V`` allows. The
      step is capped, the dwell stays at the floor, and the achievable rate comes
      out below the request. Flagged.
    * **Very slow** -- the ideal step is below ``min_step_V``. Use the minimum
      step and simply wait longer between points; the requested rate is still met.
    """
    if rate_Vps <= 0:
        raise ValueError("ramp rate must be greater than 0 V/s.")
    if max_step_V <= 0:
        raise ValueError("max_step_V must be greater than 0.")

    period_floor = point_period_s(nplc, source_delay_s, line_freq_hz, bus_overhead_s)
    ideal_step = rate_Vps * period_floor

    step = min(max(ideal_step, min_step_V), max_step_V)
    dwell = max(step / rate_Vps, period_floor)
    achievable = step / dwell
    limited = achievable < rate_Vps * (1.0 - 1e-9)

    if not limited:
        note = f"{rate_Vps:g} V/s at {step:g} V steps every {dwell * 1e3:.1f} ms."
    else:
        note = (
            f"Requested {rate_Vps:g} V/s is not achievable: max_step_V "
            f"({max_step_V:g} V) over the {period_floor * 1e3:.1f} ms point period "
            f"caps the ramp at {achievable:g} V/s. Raise max_step_V or lower NPLC."
        )

    return RampPlan(
        step_V=step,
        dwell_s=dwell,
        requested_rate_Vps=float(rate_Vps),
        achievable_rate_Vps=achievable,
        rate_limited=limited,
        note=note,
    )


def build_ramp_values(start: float, stop: float, step: float) -> List[float]:
    """Voltages from just after ``start`` up to and including ``stop``.

    ``start`` itself is excluded -- the source is already sitting there. The
    final point is snapped exactly onto ``stop`` so the ramp never overshoots
    the ceiling the user set.
    """
    start = float(start)
    stop = float(stop)
    step = abs(float(step))
    if step <= 0:
        raise ValueError("step must be greater than 0.")
    if math.isclose(start, stop, rel_tol=0.0, abs_tol=1e-15):
        return []

    span = stop - start
    sign = 1.0 if span > 0 else -1.0
    count = max(1, int(math.ceil(abs(span) / step)))

    values = []
    for i in range(1, count + 1):
        value = start + sign * step * i
        if (sign > 0 and value >= stop) or (sign < 0 and value <= stop):
            values.append(stop)
            break
        values.append(value)
    else:
        if not math.isclose(values[-1], stop, rel_tol=0.0, abs_tol=1e-12):
            values.append(stop)
    return values


def estimate_duration_s(plan: RampPlan, v_start: float, v_stop: float) -> float:
    """Wall-clock estimate for a ramp, ignoring breakdown."""
    span = abs(float(v_stop) - float(v_start))
    return math.ceil(span / plan.step_V) * plan.dwell_s if plan.step_V else 0.0
