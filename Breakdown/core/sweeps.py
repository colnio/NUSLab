"""
Running the C(f) and C(V) sweeps.

Same shape as ``cv_cf_recipe_ui.py``'s worker -- set the setpoint, wait for it to
settle, poll a sample -- but reduced to the two sweeps this experiment needs and
written against the :class:`~Breakdown.core.instruments.ImpedanceAnalyzer`
protocol so it can be tested without ``zhinst``.

There is no LabOne sweeper module involved anywhere in this repo; timing rests
on the per-point settle time, which is why ``settle_s`` is exposed per sweep.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from typing import Callable, List, Optional

from .instruments import ImpedanceAnalyzer, SweepResult, Termination
from .models import compute_model_values, decompose_impedance, per_area
from .params import CFParams, CVParams, MfiaParams
from .sweep_points import SweepPoint, build_cf_points, build_cv_points

PointCallback = Callable[[dict], None]
StopCheck = Callable[[], bool]

NAN = float("nan")


def _to_float(value, default=NAN) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_valid_sample(analyzer: ImpedanceAnalyzer) -> dict:
    """Retry once when the MFIA sample is missing the core measured values."""
    problems = []
    for _attempt in range(2):
        try:
            sample = analyzer.read_sample() or {}
            required = ("param0", "param1", "frequency", "drive")
            values = [_to_float(sample.get(key)) for key in required]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    "missing/non-finite " + ", ".join(
                        key for key, value in zip(required, values)
                        if not math.isfinite(value)
                    )
                )
            return sample
        except Exception as exc:
            problems.append(str(exc))
            # Concrete adapters own communication retries (including MFIA
            # resubscription).  This layer uses its second attempt only when a
            # returned payload was invalid, avoiding stacked retry loops.
            if not isinstance(exc, ValueError):
                raise RuntimeError(
                    "MFIA sample read failed after its adapter retry: " + str(exc)
                ) from exc
    raise RuntimeError(
        "MFIA sample failed validation twice at the same setpoint: "
        + "; ".join(problems)
    )


def _half_local_point_spacing(points: List[SweepPoint], position: int) -> float:
    """Return half the distance to the nearest adjacent sweep setpoint."""
    distances = []
    value = points[position].value
    if position > 0:
        distances.append(abs(value - points[position - 1].value))
    if position + 1 < len(points):
        distances.append(abs(points[position + 1].value - value))
    positive = [distance for distance in distances if distance > 1e-12]
    return min(positive) / 2.0 if positive else 0.0


def _build_row(
    point: SweepPoint,
    sweep_variable: str,
    bias: float,
    amplitude: float,
    frequency: float,
    sample: dict,
    model: int,
    area_um2: float,
) -> dict:
    param0 = sample.get("param0")
    param1 = sample.get("param1")
    measured_freq = _to_float(sample.get("frequency"), frequency)
    capacitance, resistance = compute_model_values(model, param0, param1, measured_freq)

    row = {
        "timestamp_unix": time.time(),
        "timestamp_iso": dt.datetime.now().isoformat(timespec="seconds"),
        "sweep_variable": sweep_variable,
        "sweep_value": point.value,
        "direction": point.direction,
        "segment": point.segment,
        "point_index": point.index,
        "set_bias_V": bias,
        "set_amplitude_V": amplitude,
        "set_frequency_Hz": frequency,
        "measured_frequency_Hz": measured_freq,
        "measured_drive_V": _to_float(sample.get("drive")),
        "model": model,
        "param0": _to_float(param0),
        "param1": _to_float(param1),
        "C_F": capacitance,
        "C_per_area_F_per_um2": per_area(capacitance, area_um2),
        "R_Ohm": resistance,
    }
    row.update(decompose_impedance(sample.get("z")))
    return row


def _run(
    analyzer: ImpedanceAnalyzer,
    mfia_params: MfiaParams,
    points: List[SweepPoint],
    kind: str,
    sweep_variable: str,
    setpoint_for,
    settle_s: float,
    amplitude_V: float,
    area_um2: float,
    on_point: Optional[PointCallback],
    should_stop: Optional[StopCheck],
    sleep: Callable[[float], None],
) -> tuple:
    """Shared driver for both sweeps. Returns (rows_emitted, reason, duration)."""
    started = time.perf_counter()
    reason = Termination.COMPLETED
    emitted: List[dict] = []

    try:
        analyzer.configure(mfia_params)
        analyzer.set_amplitude(amplitude_V)

        for position, point in enumerate(points):
            if should_stop is not None and should_stop():
                reason = Termination.STOPPED
                break

            bias, frequency = setpoint_for(point)
            analyzer.set_frequency(frequency)
            if sweep_variable == "bias":
                analyzer.set_bias(
                    bias,
                    readback_tolerance_V=_half_local_point_spacing(points, position),
                )
            else:
                analyzer.set_bias(bias)
            if settle_s > 0:
                sleep(settle_s)

            sample = _read_valid_sample(analyzer)
            row = _build_row(point, sweep_variable, bias, amplitude_V, frequency,
                             sample, int(mfia_params.model), area_um2)
            emitted.append(row)
            if on_point is not None:
                on_point(row)
    finally:
        # Never leave a DC bias standing on the device -- the operator is about
        # to unplug these cables by hand.
        try:
            analyzer.set_bias(0.0)
        except Exception:
            pass

    return emitted, reason, time.perf_counter() - started


def run_cf_sweep(
    analyzer: ImpedanceAnalyzer,
    cf: CFParams,
    mfia_params: MfiaParams,
    area_um2: float = 0.0,
    reference_hz: float = 1000.0,
    on_point: Optional[PointCallback] = None,
    should_stop: Optional[StopCheck] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> SweepResult:
    """Log-spaced C(f) from ``f_min`` to ``f_max`` and back, at fixed bias."""
    points = build_cf_points(cf.f_min, cf.f_max, cf.points)

    rows, reason, duration = _run(
        analyzer=analyzer,
        mfia_params=mfia_params,
        points=points,
        kind="CF",
        sweep_variable="frequency",
        setpoint_for=lambda point: (cf.bias_V, point.value),
        settle_s=cf.settle_s,
        amplitude_V=cf.amplitude_V,
        area_um2=area_um2,
        on_point=on_point,
        should_stop=should_stop,
        sleep=sleep,
    )

    c_ref, f_ref = _reference_from_frequency(rows, reference_hz)
    return SweepResult(
        kind="CF",
        termination_reason=reason,
        point_count=len(rows),
        duration_s=duration,
        c_reference_F=c_ref,
        f_reference_Hz=f_ref,
    )


def run_cv_sweep(
    analyzer: ImpedanceAnalyzer,
    cv: CVParams,
    mfia_params: MfiaParams,
    area_um2: float = 0.0,
    on_point: Optional[PointCallback] = None,
    should_stop: Optional[StopCheck] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> SweepResult:
    """Linear C(V) hysteresis loop ``0 -> v_max -> v_min -> 0`` at fixed frequency."""
    points = build_cv_points(cv.v_min, cv.v_max, cv.points)

    rows, reason, duration = _run(
        analyzer=analyzer,
        mfia_params=mfia_params,
        points=points,
        kind="CV",
        sweep_variable="bias",
        setpoint_for=lambda point: (point.value, cv.frequency_Hz),
        settle_s=cv.settle_s,
        amplitude_V=cv.amplitude_V,
        area_um2=area_um2,
        on_point=on_point,
        should_stop=should_stop,
        sleep=sleep,
    )

    return SweepResult(
        kind="CV",
        termination_reason=reason,
        point_count=len(rows),
        duration_s=duration,
        c_reference_F=_reference_at_zero_bias(rows),
        f_reference_Hz=float(cv.frequency_Hz),
    )


def _reference_from_frequency(rows: List[dict], reference_hz: float):
    """Capacitance at whichever measured frequency is nearest ``reference_hz``."""
    if not rows:
        return None, None
    nearest = min(rows, key=lambda r: abs(r["set_frequency_Hz"] - reference_hz))
    return nearest["C_F"], nearest["set_frequency_Hz"]


def _reference_at_zero_bias(rows: List[dict]):
    """Capacitance nearest zero bias -- the headline number for the device."""
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r["sweep_value"]))["C_F"]
