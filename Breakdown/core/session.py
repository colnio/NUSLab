"""
The per-device run state machine.

One device is one pass through::

    1. C(f)                                   (if enabled)
    2. C(V)                                   (if enabled)
    3. MFIA off  ->  PROMPT: connect Keithley        [blocking]
    4. decide RVS or CVS from the mode and the index
    5. if CVS     ->  PROMPT: confirm stress voltage [blocking]
    6. run the stress
    7. SMU off   ->  PROMPT: connect MFIA            [blocking]
    8. write metadata, append the summary row, advance the index

Two rules are load-bearing:

**An instrument is driven to zero and opened before every prompt.** The operator
is about to handle the wiring by hand, so this happens in ``finally`` blocks, on
the error path as much as the happy path.

**The first device of a campaign is always an RVS**, even in ``cvs_only``. A CVS
level cannot be chosen without at least one measured V_BD, so there has to be a
ramp first.
"""

from __future__ import annotations

import datetime as dt
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
    write_device_meta,
)
from .params import to_dict as params_to_dict
from . import _paths  # noqa: F401
from KeithleyGUI.ui_helpers import write_json_file


class RunOutcome:
    COMPLETED = "completed"
    STOPPED = "stopped"
    ABORTED = "aborted"


@dataclass
class RunResult:
    devices_completed: int
    reason: str


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
        errors = validate(self.params)
        if errors:
            raise ValueError("Parameters are not valid:\n- " + "\n- ".join(errors))
        if not str(self.params.run.output_dir).strip():
            raise ValueError("No output folder selected.")

        paths = SamplePaths(
            self.params.run.output_dir, self.date, self.params.sample.sample_name
        )
        paths.ensure_sample_dir()
        write_json_file(
            paths.params_file(self.timestamp),
            params_to_dict(self.params),
            label="parameter snapshot",
        )
        summary = SummaryWriter(paths.summary_file)

        size = float(self.params.sample.crosspoint_um)
        index = self._next_index or paths.next_device_index(size)
        completed = 0
        reason = RunOutcome.COMPLETED

        try:
            while device_limit is None or completed < device_limit:
                if self._stop:
                    reason = RunOutcome.STOPPED
                    break
                outcome = self._run_one_device(paths, summary, size, index)
                if outcome is None:
                    reason = RunOutcome.STOPPED if self._stop else RunOutcome.ABORTED
                    break
                completed += 1
                index += 1
                self._next_index = index
        finally:
            self._safe_off_both()

        self.listener.on_run_finished(completed, reason)
        return RunResult(devices_completed=completed, reason=reason)

    # -- one device ---------------------------------------------------------

    def _run_one_device(self, paths, summary, size, index) -> Optional[Dict]:
        """Return the summary row, or None if the operator aborted."""
        dirs = paths.ensure_device_dirs(size, index)
        area = self.params.sample.area_um2
        started_iso = dt.datetime.now().isoformat(timespec="seconds")
        self.listener.on_device_started(index, size)

        cf_result = cv_result = None
        measured_capacitance = None
        reference_hz = None

        # -- capacitance ----------------------------------------------------
        ran_mfia = False
        try:
            if self.params.run.enable_cf:
                cf_result = self._run_sweep(Phase.CF, dirs, size, index, area)
                ran_mfia = True
            if not self._stop and self.params.run.enable_cv:
                cv_result = self._run_sweep(Phase.CV, dirs, size, index, area)
                ran_mfia = True
        finally:
            if ran_mfia:
                self._safe_off(self.mfia)

        for result in (cv_result, cf_result):
            if result is not None and result.c_reference_F is not None:
                measured_capacitance = result.c_reference_F
                reference_hz = result.f_reference_Hz
                break

        if self._stop:
            return None

        # -- swap to the SMU -------------------------------------------------
        if ran_mfia and not self.prompter.confirm_cable_swap(Instrument.SMU, index):
            return None
        if self._stop:
            return None

        # -- decide and run the stress --------------------------------------
        stress_type = self._resolve_stress_type(index)
        if stress_type is None:
            return None

        v_stress = None
        if stress_type is StressType.CVS:
            v_stress = self._resolve_cvs_voltage(index, size)
            if v_stress is None:
                return None

        try:
            stress_result = self._run_stress(
                stress_type, v_stress, dirs, size, index, area
            )
        finally:
            self._safe_off(self.smu)

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
            # The device itself is finished; record it before leaving.
            self._on_device_complete(
                paths, summary, dirs, size, index, stress_type, stress_result,
                measured_capacitance, reference_hz, started_iso,
            )
            return None

        return self._on_device_complete(
            paths, summary, dirs, size, index, stress_type, stress_result,
            measured_capacitance, reference_hz, started_iso,
        )

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
        self, stress_type: StressType, v_stress, dirs, size, index, area
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
                )
            else:
                result = stress_module.run_cvs(
                    self.smu, self.params.cvs, v_stress=v_stress,
                    polarity=self.params.rvs.polarity, area_um2=area,
                    on_point=on_point, should_stop=self.should_stop,
                    clock=self.clock, sleep=self.sleep,
                )

        self.listener.on_phase_finished(phase, index, result)
        return result

    # -- decisions ----------------------------------------------------------

    def _resolve_stress_type(self, index: int) -> Optional[StressType]:
        mode = self.params.run.mode
        if mode == "manual":
            return self.prompter.choose_stress_type(index)
        if mode == "rvs_only":
            return StressType.RVS
        # The first device of any campaign must be a ramp -- without a measured
        # V_BD there is nothing to base a stress level on.
        if not self.statistics.count(self.params.sample.crosspoint_um):
            return StressType.RVS
        if mode == "cvs_only":
            return StressType.CVS
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
            records=self.statistics.records_for(size),
        )
        chosen = self.prompter.resolve_cvs_voltage(context)
        return None if chosen is None else abs(float(chosen))

    # -- bookkeeping --------------------------------------------------------

    def _on_device_complete(
        self, paths, summary, dirs, size, index, stress_type, stress_result,
        capacitance, reference_hz, started_iso,
    ) -> Dict:
        row = self._build_summary_row(
            size, index, stress_type, stress_result, capacitance, reference_hz,
            started_iso, dirs.data_dir,
        )
        summary.append(row)
        self.summary_rows.append(row)
        write_device_meta(
            dirs.meta_file,
            {
                "device_index": index,
                "crosspoint_um": size,
                "area_um2": self.params.sample.area_um2,
                "stress_type": stress_type.value,
                "started_iso": started_iso,
                "finished_iso": dt.datetime.now().isoformat(timespec="seconds"),
                "run_timestamp": self.timestamp,
                "stress_notes": list(stress_result.notes),
                "termination_reason": stress_result.termination_reason,
                "parameters": params_to_dict(self.params),
            },
        )
        self.listener.on_device_finished(index, row)
        return row

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

    def _safe_off(self, instrument) -> None:
        try:
            instrument.safe_off()
        except Exception as exc:  # never let cleanup mask the original failure
            self.listener.on_log(f"Warning: could not safe-off instrument: {exc}")

    def _safe_off_both(self) -> None:
        self._safe_off(self.smu)
        self._safe_off(self.mfia)
