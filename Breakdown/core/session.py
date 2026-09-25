"""
The per-device run state machine.

One device is one pass through::

    1. C(f)                                   (if enabled)
    2. C(V)                                   (if enabled)
    3. MFIA idle ->  PROMPT: connect Keithley        [blocking]
    4. decide RVS or CVS from the mode and the index
    5. if CVS     ->  PROMPT: confirm stress voltage [blocking]
    6. run the stress
    7. SMU off   ->  PROMPT: connect MFIA            [blocking]
    8. write metadata, append the summary row, advance the index

Two rules are load-bearing:

**The Keithley is driven to zero and opened before every prompt; the MFIA is put
at its enabled 10 mV, 0 V bias, 100 kHz idle state.** The operator is about to
handle the wiring by hand, so this happens in ``finally`` blocks, on the error
path as much as the happy path.

**Automatic modes follow their documented sequence strictly.** ``alternating``
uses device-index parity, and ``cvs_only`` ramps device 1 before holding later
devices. Missing V_BD statistics disable the recommendation but never silently
replace a scheduled CVS; the operator can enter a custom voltage or abort.
"""

from __future__ import annotations

import datetime as dt
import copy
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from . import stress as stress_module
from . import sweeps as sweeps_module
from .advisor import VbdRecord, VbdStatistics, recommend
from .events import (
    CvsVoltageContext,
    Instrument,
    Phase,
    Prompter,
    RateLimitContext,
    SessionListener,
    StressType,
)
from .instruments import ImpedanceAnalyzer, SourceMeter, StressResult, SweepResult
from .naming import SamplePaths, measurement_stem
from .params import BreakdownParams, validate
from .storage import (
    MFIA_COLUMNS,
    STRESS_COLUMNS,
    MeasurementWriter,
    SummaryWriter,
    breakdown_field_MV_per_cm,
    read_summary_rows,
    write_json_atomic,
    write_device_meta,
)
from .params import to_dict as params_to_dict
from . import _paths  # noqa: F401


class RunOutcome:
    COMPLETED = "completed"
    STOPPED = "stopped"
    ABORTED = "aborted"


@dataclass
class RunResult:
    devices_completed: int
    reason: str


@dataclass
class DeviceOutcome:
    """Separates a completed device from whether the campaign should continue."""

    completed: bool
    continue_run: bool
    row: Optional[Dict] = None


class BreakdownSession:
    """Drives a campaign device by device. UI-agnostic."""

    def __init__(
        self,
        params: BreakdownParams,
        mfia: ImpedanceAnalyzer,
        smu: SourceMeter,
        prompter: Prompter,
        date: Optional[str] = None,
        timestamp: Optional[str] = None,
        listener: Optional[SessionListener] = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.params = params
        self.mfia = mfia
        self.smu = smu
        self.prompter = prompter
        self.listener = listener or SessionListener()
        self.clock = clock
        self.sleep = sleep

        now = dt.datetime.now()
        self.date = date or now.strftime("%Y-%m-%d")
        self.timestamp = timestamp or now.strftime("%Y-%m-%d_%H-%M-%S")

        self.statistics = VbdStatistics()
        self.summary_rows: List[Dict] = []
        self._stop = False
        self._next_index: Optional[int] = None
        self._previous_cvs_voltage: Optional[float] = None
        self._timing_cache: Dict[tuple, float] = {}

    # -- control ------------------------------------------------------------

    def request_stop(self) -> None:
        self._stop = True

    def should_stop(self) -> bool:
        return self._stop

    def set_next_index(self, index: int) -> None:
        """Override the auto-derived device number (manual set / reset to 1)."""
        self._next_index = max(1, int(index))

    # -- run ----------------------------------------------------------------

    def run(self, device_limit: Optional[int] = None) -> RunResult:
        # A live run owns an immutable snapshot.  GUI widgets and callers may
        # retain and edit their original object without changing the experiment.
        self.params = copy.deepcopy(self.params)
        errors = validate(self.params)
        if errors:
            raise ValueError("Parameters are not valid:\n- " + "\n- ".join(errors))
        if not str(self.params.run.output_dir).strip():
            raise ValueError("No output folder selected.")

        paths = SamplePaths(
            self.params.run.output_dir, self.date, self.params.sample.sample_name
        )
        paths.ensure_sample_dir()
        write_json_atomic(
            paths.params_file(self.timestamp),
            params_to_dict(self.params),
            label="parameter snapshot",
        )
        summary = SummaryWriter(paths.summary_file)

        size = float(self.params.sample.crosspoint_um)
        self._restore_history(paths.summary_file, size)
        index = self._next_index or paths.next_device_index(size)
        completed = 0
        reason = RunOutcome.COMPLETED

        try:
            while device_limit is None or completed < device_limit:
                if self._stop:
                    reason = RunOutcome.STOPPED
                    break
                outcome = self._run_one_device(paths, summary, size, index)
                if outcome.completed:
                    completed += 1
                    index += 1
                    self._next_index = index
                if not outcome.continue_run:
                    reason = RunOutcome.STOPPED if self._stop else RunOutcome.ABORTED
                    break
        finally:
            self._safe_off_both()

        self.listener.on_run_finished(completed, reason)
        return RunResult(devices_completed=completed, reason=reason)

    # -- one device ---------------------------------------------------------

    def _run_one_device(self, paths, summary, size, index) -> DeviceOutcome:
        """Run one physical device and report completion separately from continuation."""
        dirs = paths.allocate_device_run_dirs(size, index)
        if dirs.run_number > 1:
            self.listener.on_log(
                f"Device {index} already has output; preserving it and writing "
                f"this measurement as rerun {dirs.run_number} in {dirs.device_dir}."
            )
        area = self.params.sample.area_um2
        started_iso = dt.datetime.now().isoformat(timespec="seconds")
        metadata = {
            "device_index": index,
            "run_number": dirs.run_number,
            "crosspoint_um": size,
            "area_um2": area,
            "status": "in_progress",
            "last_phase": None,
            "failure_reason": None,
            "started_iso": started_iso,
            "finished_iso": None,
            "run_timestamp": self.timestamp,
            "parameters": params_to_dict(self.params),
        }
        write_device_meta(dirs.meta_file, metadata)
        self.listener.on_device_started(index, size)

        cf_result = cv_result = None
        measured_capacitance = None
        reference_hz = None

        try:
            # -- capacitance ------------------------------------------------
            ran_mfia = False
            try:
                if self.params.run.enable_cf:
                    metadata["last_phase"] = Phase.CF.value
                    cf_result = self._run_sweep(Phase.CF, dirs, size, index, area)
                    ran_mfia = True
                if not self._stop and self.params.run.enable_cv:
                    metadata["last_phase"] = Phase.CV.value
                    cv_result = self._run_sweep(Phase.CV, dirs, size, index, area)
                    ran_mfia = True
            finally:
                mfia_safety_error = self._safe_off(self.mfia) if ran_mfia else None
                if mfia_safety_error is not None:
                    raise RuntimeError(
                        "MFIA could not reach its 10 mV, 0 V bias, 100 kHz idle "
                        f"state before cable handling: {mfia_safety_error}"
                    )

            for result in (cv_result, cf_result):
                if result is not None and result.c_reference_F is not None:
                    measured_capacitance = result.c_reference_F
                    reference_hz = result.f_reference_Hz
                    break

            if self._stop:
                self._finish_metadata(dirs.meta_file, metadata, "aborted", "stop requested")
                return DeviceOutcome(False, False)

        # -- swap to the SMU -------------------------------------------------
            if ran_mfia and not self.prompter.confirm_cable_swap(Instrument.SMU, index):
                self._finish_metadata(dirs.meta_file, metadata, "aborted", "SMU cable swap declined")
                return DeviceOutcome(False, False)
            if self._stop:
                self._finish_metadata(dirs.meta_file, metadata, "aborted", "stop requested")
                return DeviceOutcome(False, False)

        # -- decide and run the stress --------------------------------------
            stress_type = self._resolve_stress_type(index)
            if stress_type is None:
                self._finish_metadata(dirs.meta_file, metadata, "aborted", "stress selection cancelled")
                return DeviceOutcome(False, False)
            metadata["stress_type"] = stress_type.value
            metadata["last_phase"] = stress_type.value

            v_stress = None
            if stress_type is StressType.CVS:
                v_stress = self._resolve_cvs_voltage(index, size)
                if v_stress is None:
                    self._finish_metadata(dirs.meta_file, metadata, "aborted", "CVS voltage selection cancelled")
                    return DeviceOutcome(False, False)

            ramp_plan = self._prepare_stress_plan(stress_type, index, v_stress)
            if ramp_plan is None:
                self._finish_metadata(
                    dirs.meta_file, metadata, "aborted",
                    "hardware timing limitation declined",
                )
                return DeviceOutcome(False, False)

            try:
                stress_result = self._run_stress(
                    stress_type, v_stress, dirs, size, index, area, ramp_plan
                )
            finally:
                smu_safety_error = self._safe_off(self.smu)
                if smu_safety_error is not None:
                    raise RuntimeError(
                        "SMU could not be confirmed safe before cable handling: "
                        f"{smu_safety_error}"
                    )

            if stress_type is StressType.CVS and v_stress is not None:
                self._previous_cvs_voltage = abs(v_stress)
            if stress_type is StressType.RVS and stress_result.bd_detected:
                self.statistics.add(
                    VbdRecord(
                        device_index=index,
                        crosspoint_um=size,
                        v_bd=stress_result.v_bd,
                        ramp_rate_Vps=stress_result.achieved_rate_Vps or 0.0,
                    )
                )

        # -- swap back -------------------------------------------------------
            if ran_mfia and not self.prompter.confirm_cable_swap(Instrument.MFIA, index):
                row = self._on_device_complete(
                    paths, summary, dirs, size, index, stress_type, stress_result,
                    measured_capacitance, reference_hz, started_iso, metadata,
                )
                return DeviceOutcome(True, False, row)

            row = self._on_device_complete(
                paths, summary, dirs, size, index, stress_type, stress_result,
                measured_capacitance, reference_hz, started_iso, metadata,
            )
            return DeviceOutcome(True, True, row)
        except Exception as exc:
            try:
                self._finish_metadata(dirs.meta_file, metadata, "failed", str(exc))
            except Exception as metadata_exc:
                self.listener.on_log(
                    f"Warning: could not finalize failure metadata: {metadata_exc}"
                )
            raise

    # -- phases -------------------------------------------------------------

    def _run_sweep(self, phase: Phase, dirs, size, index, area) -> SweepResult:
        stem = measurement_stem(
            self.params.sample.sample_name, size, index, phase.value, self.timestamp
        )
        path = f"{dirs.data_dir}/{stem}.csv"
        self.listener.on_phase_started(phase, index, None)

        with MeasurementWriter(path, MFIA_COLUMNS) as writer:
            def on_point(row):
                writer.write_row(row)
                self.listener.on_point(phase, row)

            if phase is Phase.CF:
                result = sweeps_module.run_cf_sweep(
                    self.mfia, self.params.cf, self.params.mfia, area_um2=area,
                    reference_hz=self.params.cv.frequency_Hz,
                    on_point=on_point, should_stop=self.should_stop, sleep=self.sleep,
                )
            else:
                result = sweeps_module.run_cv_sweep(
                    self.mfia, self.params.cv, self.params.mfia, area_um2=area,
                    on_point=on_point, should_stop=self.should_stop, sleep=self.sleep,
                )

        self.listener.on_phase_finished(phase, index, result)
        return result

    def _run_stress(
        self, stress_type: StressType, v_stress, dirs, size, index, area,
        ramp_plan,
    ) -> StressResult:
        phase = Phase.RVS if stress_type is StressType.RVS else Phase.CVS
        stem = measurement_stem(
            self.params.sample.sample_name, size, index, phase.value, self.timestamp
        )
        path = f"{dirs.data_dir}/{stem}.csv"
        self.listener.on_phase_started(phase, index, None)

        with MeasurementWriter(path, STRESS_COLUMNS) as writer:
            def on_point(row):
                writer.write_row(row)
                self.listener.on_point(phase, row)

            if stress_type is StressType.RVS:
                result = stress_module.run_rvs(
                    self.smu, self.params.rvs, area_um2=area, on_point=on_point,
                    should_stop=self.should_stop, clock=self.clock, sleep=self.sleep,
                    ramp_plan=ramp_plan,
                )
            else:
                result = stress_module.run_cvs(
                    self.smu, self.params.cvs, v_stress=v_stress,
                    polarity=self.params.rvs.polarity, area_um2=area,
                    on_point=on_point, should_stop=self.should_stop,
                    clock=self.clock, sleep=self.sleep,
                    pre_ramp_plan=ramp_plan,
                )

        self.listener.on_phase_finished(phase, index, result)
        return result

    def _prepare_stress_plan(self, stress_type, index, v_stress):
        if stress_type is StressType.RVS:
            source = self.params.rvs
            rate = source.ramp_rate_Vps
            max_step = source.max_step_V
            source_delay = source.source_delay_s
        else:
            source = self.params.cvs
            rate = source.pre_ramp_rate_Vps
            max_step = max(abs(float(v_stress)) / 10.0, 1e-3)
            source_delay = 0.0

        key = (
            float(source.nplc), float(source.compliance_A),
            bool(source.current_autorange), float(source_delay),
        )
        if key not in self._timing_cache:
            self.listener.on_log(
                f"Calibrating {stress_type.value} timing with three zero-volt reads."
            )
            self._timing_cache[key] = stress_module.calibrate_smu_point_period(
                self.smu,
                nplc=source.nplc,
                compliance_A=source.compliance_A,
                current_autorange=source.current_autorange,
                source_delay_s=source_delay,
                clock=self.clock,
                samples=3,
            )
        period = self._timing_cache[key]
        plan = stress_module.plan_for_calibrated_period(
            rate, max_step, source.nplc, source_delay, period
        )
        achieved_interval = None
        interval_limited = False
        if stress_type is StressType.CVS:
            achieved_interval = max(float(source.sample_interval_s), period)
            interval_limited = achieved_interval > source.sample_interval_s * (1.0 + 1e-9)

        self.listener.on_log(
            f"{stress_type.value} hardware plan: {plan.achievable_rate_Vps:.4g} V/s, "
            f"{plan.step_V:.4g} V steps, {period * 1e3:.1f} ms measured read cycle."
        )
        if plan.rate_limited or interval_limited:
            context = RateLimitContext(
                device_index=index,
                stress_type=stress_type,
                requested_rate_Vps=float(rate),
                achievable_rate_Vps=float(plan.achievable_rate_Vps),
                point_period_s=float(period),
                max_step_V=float(max_step),
                requested_sample_interval_s=(
                    float(source.sample_interval_s)
                    if stress_type is StressType.CVS else None
                ),
                achievable_sample_interval_s=achieved_interval,
            )
            if not self.prompter.confirm_rate_limit(context):
                return None
        return plan

    # -- decisions ----------------------------------------------------------

    def _resolve_stress_type(self, index: int) -> Optional[StressType]:
        mode = self.params.run.mode
        if mode == "manual":
            return self.prompter.choose_stress_type(index)
        if mode == "rvs_only":
            return StressType.RVS
        if mode == "cvs_only":
            return StressType.RVS if index == 1 else StressType.CVS
        return StressType.RVS if index % 2 == 1 else StressType.CVS

    def _resolve_cvs_voltage(self, index: int, size: float) -> Optional[float]:
        recommendation = recommend(
            self.statistics, size, self.params.advisor.k_fraction,
            self.params.advisor.statistic,
        )
        context = CvsVoltageContext(
            device_index=index,
            crosspoint_um=size,
            previous_voltage=self._previous_cvs_voltage,
            recommendation=recommendation,
            max_voltage_V=abs(float(self.params.run.abs_max_voltage_V)),
            records=self.statistics.records_for(size),
        )
        chosen = self.prompter.resolve_cvs_voltage(context)
        if chosen is None:
            return None
        voltage = abs(float(chosen))
        ceiling = abs(float(self.params.run.abs_max_voltage_V))
        if not math.isfinite(voltage) or voltage <= 0 or voltage > ceiling:
            raise ValueError(
                f"CVS stress voltage must be finite, greater than 0 V, and no "
                f"greater than the absolute ceiling of {ceiling:g} V."
            )
        return voltage

    def _restore_history(self, summary_path: str, size: float) -> None:
        rows, warnings = read_summary_rows(summary_path)
        for warning in warnings:
            self.listener.on_log(f"Warning: {warning}")
        sample_name = self.params.sample.sample_name
        ceiling = abs(float(self.params.run.abs_max_voltage_V))
        for line_number, row in enumerate(rows, start=2):
            try:
                if row.get("sample") != sample_name:
                    continue
                row_size = float(row.get("crosspoint_um", ""))
                if not math.isfinite(row_size) or not math.isclose(
                    row_size, size, rel_tol=0.0, abs_tol=1e-12
                ):
                    continue
                stress_type = row.get("stress_type", "").strip().upper()
                if stress_type == StressType.RVS.value:
                    detected = row.get("bd_detected", "").strip().lower() in {
                        "true", "1", "yes"
                    }
                    if not detected:
                        continue
                    v_bd = float(row.get("V_BD_V", ""))
                    rate = float(row.get("achieved_rate_Vps", ""))
                    index = int(row.get("device_index", ""))
                    if not (math.isfinite(v_bd) and math.isfinite(rate) and rate > 0):
                        raise ValueError("non-finite V_BD or achieved rate")
                    self.statistics.add(VbdRecord(index, row_size, v_bd, rate))
                elif stress_type == StressType.CVS.value:
                    voltage = abs(float(row.get("V_stress_V", "")))
                    if math.isfinite(voltage) and 0 < voltage <= ceiling:
                        self._previous_cvs_voltage = voltage
            except (TypeError, ValueError) as exc:
                self.listener.on_log(
                    f"Warning: ignored malformed campaign history row "
                    f"{line_number}: {exc}"
                )

    # -- bookkeeping --------------------------------------------------------

    def _on_device_complete(
        self, paths, summary, dirs, size, index, stress_type, stress_result,
        capacitance, reference_hz, started_iso, metadata,
    ) -> Dict:
        row = self._build_summary_row(
            size, index, stress_type, stress_result, capacitance, reference_hz,
            started_iso, dirs.data_dir,
        )
        metadata.update({
            "stress_type": stress_type.value,
            "stress_notes": list(stress_result.notes),
            "termination_reason": stress_result.termination_reason,
        })
        self._finish_metadata(dirs.meta_file, metadata, "completed", None)
        summary.append(row)
        self.summary_rows.append(row)
        self.listener.on_device_finished(index, row)
        return row

    def _finish_metadata(self, path, metadata, status, failure_reason) -> None:
        metadata["status"] = status
        metadata["failure_reason"] = failure_reason
        metadata["finished_iso"] = dt.datetime.now().isoformat(timespec="seconds")
        write_device_meta(path, metadata)

    def _build_summary_row(
        self, size, index, stress_type, result, capacitance, reference_hz,
        started_iso, data_dir,
    ) -> Dict:
        sample = self.params.sample
        is_rvs = stress_type is StressType.RVS
        source = self.params.rvs if is_rvs else self.params.cvs
        area = sample.area_um2

        return {
            "date": self.date,
            "sample": sample.sample_name,
            "crosspoint_um": size,
            "area_um2": area,
            "thickness_nm": sample.thickness_nm,
            "device_index": index,
            "stress_type": stress_type.value,
            "polarity": self.params.rvs.polarity,
            "C_at_fref_F": capacitance,
            "C_per_area_F_per_um2": (capacitance / area)
            if (capacitance is not None and area > 0) else None,
            "f_ref_Hz": reference_hz,
            "ramp_rate_Vps": self.params.rvs.ramp_rate_Vps if is_rvs else None,
            "achieved_rate_Vps": result.achieved_rate_Vps,
            # V_BD is a property of a ramp. A CVS device fails at the level it
            # was held at by construction, so reporting that as a breakdown
            # voltage would corrupt any statistics taken over this column.
            "V_BD_V": result.v_bd if is_rvs else None,
            "E_BD_MV_per_cm": breakdown_field_MV_per_cm(
                result.v_bd if is_rvs else None, sample.thickness_nm
            ),
            "t_BD_s": result.t_bd,
            "V_stress_V": result.v_stress,
            "E_stress_MV_per_cm": breakdown_field_MV_per_cm(
                result.v_stress, sample.thickness_nm
            ),
            "i_threshold_A": source.i_bd_A,
            "compliance_A": source.compliance_A,
            "bd_detected": result.bd_detected,
            "termination_reason": result.termination_reason,
            "started_iso": started_iso,
            "finished_iso": dt.datetime.now().isoformat(timespec="seconds"),
            "data_dir": data_dir,
        }

    # -- safety -------------------------------------------------------------

    def _safe_off(self, instrument) -> Optional[str]:
        try:
            instrument.safe_off()
            return None
        except Exception as exc:  # never let cleanup mask the original failure
            self.listener.on_log(f"Warning: could not set safe instrument state: {exc}")
            return str(exc)

    def _safe_off_both(self) -> None:
        self._safe_off(self.smu)
        self._safe_off(self.mfia)
