"""
The two destructive measurements: ramped voltage stress and constant voltage stress.

Both are point-by-point software loops rather than the 2400's hardware LIST or
source-memory engines (``Keithley2400FastSweepAdapter``). Those engines are
faster, but a hardware batch runs to completion before returning any data --
there is no way to cut the drive at the moment of breakdown. Here the whole
point is to stop within one measurement period of the failure, so the loop stays
in software and the hardware compliance limit acts only as the backstop.

Written against the :class:`~Breakdown.core.instruments.SourceMeter` protocol,
with the clock and sleep injected, so every path including the error paths is
testable against :mod:`Breakdown.core.mock`.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

from .detect import BreakdownCriterion, BreakdownDetector
from .instruments import SourceMeter, StressResult, Termination
from .models import per_area
from .ramp import build_ramp_values, plan_ramp
from .params import CVSParams, RVSParams

PointCallback = Callable[[dict], None]
StopCheck = Callable[[], bool]


def _row(
    phase: str,
    index: int,
    set_v: float,
    measured_v: float,
    current: float,
    elapsed_s: float,
    area_um2: float,
    bd_flag: bool,
) -> dict:
    return {
        "timestamp_unix": time.time(),
        "elapsed_s": elapsed_s,
        "phase": phase,
        "point_index": index,
        "set_V": set_v,
        "measured_V": measured_v,
        "measured_I_A": current,
        "current_density_A_per_um2": per_area(current, area_um2),
        "bd_flag": bd_flag,
    }


def run_rvs(
    smu: SourceMeter,
    params: RVSParams,
    area_um2: float = 0.0,
    on_point: Optional[PointCallback] = None,
    should_stop: Optional[StopCheck] = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> StressResult:
    """Ramp until breakdown or the ceiling, then drive safely to zero.

    The requested ramp rate may not be achievable; whatever is actually
    delivered is measured and reported, because V_BD depends on it.
    """
    polarity = 1 if params.polarity >= 0 else -1
    plan = plan_ramp(
        rate_Vps=params.ramp_rate_Vps,
        max_step_V=params.max_step_V,
        nplc=params.nplc,
        source_delay_s=params.source_delay_s,
    )
    notes: List[str] = []
    if plan.rate_limited:
        notes.append(plan.note)

    detector = BreakdownDetector(
        BreakdownCriterion(i_threshold_A=params.i_bd_A, consecutive=2)
    )
    setpoints = build_ramp_values(abs(params.v_start), abs(params.v_max), plan.step_V)

    reason = Termination.CEILING_REACHED
    count = 0
    started = clock()
    first_v: Optional[float] = None
    last_v = abs(params.v_start)

    try:
        smu.configure(
            nplc=params.nplc,
            compliance_A=params.compliance_A,
            current_autorange=params.current_autorange,
        )
        for magnitude in setpoints:
            if should_stop is not None and should_stop():
                reason = Termination.STOPPED
                break

            target = polarity * magnitude
            smu.set_voltage(target)
            sleep(plan.dwell_s)
            measured_v, current = smu.read_vi()

            elapsed = clock() - started
            if first_v is None:
                first_v = magnitude
            last_v = magnitude

            event = detector.update(
                index=count, voltage=target, current=current, elapsed_s=elapsed
            )
            if on_point is not None:
                on_point(
                    _row("ramp", count, target, measured_v, current, elapsed,
                         area_um2, event is not None)
                )
            count += 1

            if event is not None:
                reason = Termination.BREAKDOWN
                break
    finally:
        # Runs on the success path, the stop path, and on any instrument error.
        # A device left energised after a failure is the one outcome that must
        # not happen.
        smu.set_voltage(0.0)
        smu.output_off()

    duration = clock() - started
    event = detector.event
    span = abs(last_v - (first_v if first_v is not None else abs(params.v_start)))
    achieved = (span / duration) if duration > 0 else None

    return StressResult(
        stress_type="RVS",
        termination_reason=reason,
        bd_detected=event is not None,
        v_bd=event.voltage if event else None,
        i_bd=event.current if event else None,
        t_bd=event.elapsed_s if event else None,
        v_stress=None,
        requested_rate_Vps=float(params.ramp_rate_Vps),
        achieved_rate_Vps=achieved,
        point_count=count,
        duration_s=duration,
        notes=notes,
    )


def run_cvs(
    smu: SourceMeter,
    params: CVSParams,
    v_stress: float,
    polarity: int = 1,
    area_um2: float = 0.0,
    on_point: Optional[PointCallback] = None,
    should_stop: Optional[StopCheck] = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> StressResult:
    """Ramp quickly to ``v_stress``, then hold until breakdown or the time limit.

    The pre-ramp is fast on purpose. Charge injected on the way up is
    indistinguishable from charge injected during the hold, so a slow approach
    would silently pre-stress the device and shorten the t_BD you measure.

    ``t_BD`` is therefore clocked from the moment the stress level is reached,
    not from the start of the run.
    """
    magnitude = abs(float(v_stress))
    if magnitude <= 0:
        raise ValueError("CVS stress voltage must be greater than 0 V.")
    sign = 1 if polarity >= 0 else -1

    detector = BreakdownDetector(
        BreakdownCriterion(i_threshold_A=params.i_bd_A, consecutive=2)
    )
    pre_ramp_plan = plan_ramp(
        rate_Vps=params.pre_ramp_rate_Vps,
        max_step_V=max(magnitude / 10.0, 1e-3),
        nplc=params.nplc,
    )
    notes: List[str] = []

    reason = Termination.DURATION_REACHED
    count = 0
    started = clock()
    hold_started: Optional[float] = None
    stopped = False

    try:
        smu.configure(
            nplc=params.nplc,
            compliance_A=params.compliance_A,
            current_autorange=params.current_autorange,
        )

        # -- pre-ramp -------------------------------------------------------
        for step in build_ramp_values(0.0, magnitude, pre_ramp_plan.step_V):
            if should_stop is not None and should_stop():
                reason, stopped = Termination.STOPPED, True
                break
            smu.set_voltage(sign * step)
            sleep(pre_ramp_plan.dwell_s)
            measured_v, current = smu.read_vi()
            elapsed = clock() - started
            event = detector.update(
                index=count, voltage=sign * step, current=current, elapsed_s=elapsed
            )
            if on_point is not None:
                on_point(
                    _row("preramp", count, sign * step, measured_v, current,
                         elapsed, area_um2, event is not None)
                )
            count += 1
            if event is not None:
                reason = Termination.BREAKDOWN
                notes.append(
                    "Breakdown occurred during the pre-ramp, before the hold "
                    "began. t_BD is not a constant-voltage lifetime -- lower the "
                    "stress level."
                )
                stopped = True
                break

        # -- hold -----------------------------------------------------------
        if not stopped:
            hold_started = clock()
            while True:
                if should_stop is not None and should_stop():
                    reason = Termination.STOPPED
                    break
                if clock() - hold_started >= params.max_duration_s:
                    reason = Termination.DURATION_REACHED
                    break

                smu.set_voltage(sign * magnitude)
                sleep(params.sample_interval_s)
                measured_v, current = smu.read_vi()
                elapsed = clock() - started
                event = detector.update(
                    index=count,
                    voltage=sign * magnitude,
                    current=current,
                    elapsed_s=clock() - hold_started,
                )
                if on_point is not None:
                    on_point(
                        _row("hold", count, sign * magnitude, measured_v, current,
                             elapsed, area_um2, event is not None)
                    )
                count += 1
                if event is not None:
                    reason = Termination.BREAKDOWN
                    break
    finally:
        smu.set_voltage(0.0)
        smu.output_off()

    duration = clock() - started
    event = detector.event

    return StressResult(
        stress_type="CVS",
        termination_reason=reason,
        bd_detected=event is not None,
        v_bd=event.voltage if event else None,
        i_bd=event.current if event else None,
        t_bd=event.elapsed_s if event else None,
        v_stress=sign * magnitude,
        requested_rate_Vps=None,
        achieved_rate_Vps=None,
        point_count=count,
        duration_s=duration,
        notes=notes,
    )
