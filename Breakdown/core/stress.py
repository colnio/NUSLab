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
import math
import sys
from typing import Callable, List, Optional

from .detect import BreakdownCriterion, BreakdownDetector
from .instruments import SourceMeter, StressResult, Termination
from .models import per_area
from .ramp import RampPlan, build_ramp_values, plan_ramp, point_period_s
from .params import CVSParams, RVSParams

PointCallback = Callable[[dict], None]
StopCheck = Callable[[], bool]


def _safe_output_off(smu: SourceMeter) -> None:
    """Attempt zero and output-off independently so one failure cannot mask the other."""
    zero_error = off_error = None
    try:
        smu.set_voltage(0.0)
    except Exception as exc:
        zero_error = exc
    try:
        smu.output_off()
    except Exception as exc:
        off_error = exc
    cleanup_error = off_error or zero_error
    if cleanup_error is not None and sys.exc_info()[0] is None:
        action = "disable output" if off_error is not None else "drive to zero"
        raise RuntimeError(f"Could not {action}: {cleanup_error}") from cleanup_error


def _read_vi_with_retry(smu: SourceMeter):
    """Retry one failed/invalid current read at the unchanged setpoint."""
    problems = []
    for _attempt in range(2):
        try:
            measured_v, current = smu.read_vi()
            current = float(current)
            measured_v = float(measured_v)
            if not math.isfinite(current):
                raise ValueError("current reading is not finite")
            if not math.isfinite(measured_v) and _attempt == 0:
                raise ValueError("voltage reading is not finite")
            return measured_v, current
        except Exception as exc:
            problems.append(str(exc))
    raise RuntimeError(
        "SMU read failed twice at the same setpoint: " + "; ".join(problems)
    )


def calibrate_smu_point_period(
    smu: SourceMeter,
    nplc: float,
    compliance_A: float,
    current_autorange: bool,
    source_delay_s: float = 0.0,
    clock: Callable[[], float] = time.perf_counter,
    samples: int = 3,
) -> float:
    """Measure the real zero-volt read cycle used to plan a destructive ramp."""
    durations = []
    try:
        smu.configure(
            nplc=nplc,
            compliance_A=compliance_A,
            current_autorange=current_autorange,
            source_delay_s=source_delay_s,
        )
        smu.set_voltage(0.0)
        for _ in range(max(1, int(samples))):
            started = clock()
            _read_vi_with_retry(smu)
            elapsed = clock() - started
            if math.isfinite(elapsed) and elapsed > 0:
                durations.append(elapsed)
    finally:
        _safe_output_off(smu)
    theoretical = point_period_s(nplc, source_delay_s)
    if not durations:
        return theoretical
    durations.sort()
    return max(theoretical, durations[len(durations) // 2])


def plan_for_calibrated_period(
    rate_Vps: float,
    max_step_V: float,
    nplc: float,
    source_delay_s: float,
    calibrated_period_s: float,
) -> RampPlan:
    integration_and_delay = point_period_s(
        nplc, source_delay_s, bus_overhead_s=0.0
    )
    measured_overhead = max(0.0, float(calibrated_period_s) - integration_and_delay)
    return plan_ramp(
        rate_Vps=rate_Vps,
        max_step_V=max_step_V,
        nplc=nplc,
        source_delay_s=source_delay_s,
        bus_overhead_s=measured_overhead,
    )


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
    ramp_plan: Optional[RampPlan] = None,
) -> StressResult:
    """Ramp until breakdown or the ceiling, then drive safely to zero.

    The requested ramp rate may not be achievable; whatever is actually
    delivered is measured and reported, because V_BD depends on it.
    """
    polarity = 1 if params.polarity >= 0 else -1
    plan = ramp_plan or plan_ramp(
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
    last_v = abs(params.v_start)
    ramp_started: Optional[float] = None
    last_set_at: Optional[float] = None

    try:
        smu.configure(
            nplc=params.nplc,
            compliance_A=params.compliance_A,
            current_autorange=params.current_autorange,
            source_delay_s=params.source_delay_s,
        )
        ramp_started = clock()
        for magnitude in setpoints:
            if should_stop is not None and should_stop():
                reason = Termination.STOPPED
                break

            due = ramp_started + abs(magnitude - abs(params.v_start)) / plan.achievable_rate_Vps
            remaining = due - clock()
            if remaining > 0:
                sleep(remaining)
            target = polarity * magnitude
            smu.set_voltage(target)
            last_set_at = clock()
            measured_v, current = _read_vi_with_retry(smu)

            elapsed = clock() - started
            last_v = magnitude

            event = detector.update(
                index=count,
                voltage=measured_v if math.isfinite(measured_v) else target,
                current=current,
                elapsed_s=elapsed,
            )
            if not math.isfinite(measured_v) and not any(
                "measured voltage" in note for note in notes
            ):
                notes.append(
                    "The SMU returned a non-finite measured voltage; V_BD used "
                    "the commanded setpoint for affected points."
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
        _safe_output_off(smu)

    duration = clock() - started
    event = detector.event
    span = abs(last_v - abs(params.v_start))
    ramp_elapsed = (
        last_set_at - ramp_started
        if last_set_at is not None and ramp_started is not None else 0.0
    )
    achieved = (span / ramp_elapsed) if ramp_elapsed > 0 else None
    if (achieved is not None
            and achieved < float(params.ramp_rate_Vps) * 0.9):
        notes.append(
            f"Actual ramp rate was {achieved:.4g} V/s versus the requested "
            f"{params.ramp_rate_Vps:g} V/s."
        )

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
    pre_ramp_plan: Optional[RampPlan] = None,
) -> StressResult:
    """Ramp quickly to ``v_stress``, then hold until breakdown or the time limit.

    The pre-ramp is fast on purpose. Charge injected on the way up is
    indistinguishable from charge injected during the hold, so a slow approach
    would silently pre-stress the device and shorten the t_BD you measure.

    ``t_BD`` is therefore clocked from the moment the stress level is reached,
    not from the start of the run.
    """
    magnitude = abs(float(v_stress))
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("CVS stress voltage must be finite and greater than 0 V.")
    sign = 1 if polarity >= 0 else -1

    detector = BreakdownDetector(
        BreakdownCriterion(i_threshold_A=params.i_bd_A, consecutive=2)
    )
    pre_ramp_plan = pre_ramp_plan or plan_ramp(
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
            source_delay_s=0.0,
        )

        # -- pre-ramp -------------------------------------------------------
        pre_ramp_started = clock()
        for step in build_ramp_values(0.0, magnitude, pre_ramp_plan.step_V):
            if should_stop is not None and should_stop():
                reason, stopped = Termination.STOPPED, True
                break
            due = pre_ramp_started + step / pre_ramp_plan.achievable_rate_Vps
            remaining = due - clock()
            if remaining > 0:
                sleep(remaining)
            smu.set_voltage(sign * step)
            measured_v, current = _read_vi_with_retry(smu)
            elapsed = clock() - started
            event = detector.update(
                index=count,
                voltage=measured_v if math.isfinite(measured_v) else sign * step,
                current=current,
                elapsed_s=elapsed,
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
            # A single excursion at the end of the approach must not combine
            # with a hold sample or give t_BD a pre-ramp time origin.
            detector.reset()
            hold_started = clock()
            hold_sample = 1
            while True:
                if should_stop is not None and should_stop():
                    reason = Termination.STOPPED
                    break
                if clock() - hold_started >= params.max_duration_s:
                    reason = Termination.DURATION_REACHED
                    break
                due = hold_started + hold_sample * params.sample_interval_s
                if due - hold_started > params.max_duration_s:
                    reason = Termination.DURATION_REACHED
                    break

                remaining = due - clock()
                if remaining > 0:
                    sleep(remaining)
                smu.set_voltage(sign * magnitude)
                measured_v, current = _read_vi_with_retry(smu)
                elapsed = clock() - started
                event = detector.update(
                    index=count,
                    voltage=(measured_v if math.isfinite(measured_v)
                             else sign * magnitude),
                    current=current,
                    elapsed_s=clock() - hold_started,
                )
                if on_point is not None:
                    on_point(
                        _row("hold", count, sign * magnitude, measured_v, current,
                             elapsed, area_um2, event is not None)
                    )
                count += 1
                hold_sample += 1
                if event is not None:
                    reason = Termination.BREAKDOWN
                    break
    finally:
        _safe_output_off(smu)

    duration = clock() - started
    event = detector.event

    return StressResult(
        stress_type="CVS",
        termination_reason=reason,
        bd_detected=event is not None,
        v_bd=event.voltage if event else None,
        i_bd=event.current if event else None,
        t_bd=(event.elapsed_s if event is not None and hold_started is not None
              else None),
        v_stress=sign * magnitude,
        requested_rate_Vps=None,
        achieved_rate_Vps=None,
        point_count=count,
        duration_s=duration,
        notes=notes,
    )
