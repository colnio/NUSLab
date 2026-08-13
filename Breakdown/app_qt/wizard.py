"""
Main window, measurement thread, and the bridge that lets headless code ask
the operator questions.

Threading follows the pattern the newer apps in this repo use (``FastVAC.py``,
``Pulses.py``): a ``QObject`` worker moved onto a ``QThread``, talking to the UI
only through signals, with the instruments opened *inside* the worker so VISA
handles never cross threads.

The interesting part is :class:`QtPrompter`. The core needs to block until a
human answers, but blocking must not deadlock and must not make Stop useless. So
the prompter emits a signal (queued to the GUI thread, which shows the modal) and
then waits on a :class:`threading.Event` in short slices, checking the stop flag
between them.
"""

from __future__ import annotations

import datetime as dt
import os
import threading
import traceback
from typing import Optional

import matplotlib

matplotlib.use("Agg")

from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from Breakdown.core import mfia as mfia_module
from Breakdown.core import smu as smu_module
from Breakdown.core.events import (
    CvsVoltageContext,
    Instrument,
    Phase,
    SessionListener,
    StressType,
)
from Breakdown.core.naming import SamplePaths
from Breakdown.core.params import BreakdownParams, load_params, save_params, validate
from Breakdown.core.ramp import plan_ramp
from Breakdown.core.session import BreakdownSession, RunOutcome

from .dialogs import CVS_GUIDANCE, CableSwapDialog, CvsVoltageDialog, StressTypeDialog
from .panels import ParamsPanel
from .plots import PlotPanel
from .theme import (
    BORDER,
    GO,
    SURFACE,
    TEXT_DIM,
    WAIT,
    Card,
    field_row,
    monospace,
    state_chip_style,
)

PLOT_REFRESH_MS = 100


class QtPrompter(QObject):
    """Implements the core's ``Prompter`` protocol across the thread boundary."""

    swap_requested = pyqtSignal(object, int)
    cvs_requested = pyqtSignal(object)
    stress_type_requested = pyqtSignal(int, bool)

    def __init__(self, should_stop):
        super().__init__()
        self._should_stop = should_stop
        self._answered = threading.Event()
        self._answer = None

    # -- called on the GUI thread -------------------------------------------

    def deliver(self, answer) -> None:
        self._answer = answer
        self._answered.set()

    # -- called on the worker thread ----------------------------------------

    def _wait(self):
        # Short slices rather than one long wait, so a Stop pressed while the
        # dialog is open takes effect instead of hanging the run.
        while not self._answered.wait(0.05):
            if self._should_stop():
                return None
        return self._answer

    def _ask(self, emit) -> object:
        self._answered.clear()
        self._answer = None
        emit()
        return self._wait()

    def confirm_cable_swap(self, target: Instrument, device_index: int) -> bool:
        return bool(self._ask(
            lambda: self.swap_requested.emit(target, device_index)
        ))

    def resolve_cvs_voltage(self, ctx: CvsVoltageContext) -> Optional[float]:
        answer = self._ask(lambda: self.cvs_requested.emit(ctx))
        return None if answer is None else float(answer)

    def choose_stress_type(self, device_index: int) -> Optional[StressType]:
        can_cvs = self._can_run_cvs(device_index)
        return self._ask(
            lambda: self.stress_type_requested.emit(device_index, can_cvs)
        )

    #: Replaced by the worker once the session exists.
    def _can_run_cvs(self, device_index: int) -> bool:
        return True


class SignalListener(SessionListener):
    """Turns the core's progress callbacks into Qt signals.

    Milestones are emitted on ``log`` as well as on their own signal, so they
    reach the run's log file and not just the on-screen pane. Individual points
    are deliberately excluded -- a ramp produces thousands of them.
    """

    def __init__(self, worker: "SessionWorker"):
        self.worker = worker

    def on_log(self, message):
        self.worker.log.emit(message)

    def on_device_started(self, device_index, crosspoint_um):
        self.worker.device_started.emit(device_index, crosspoint_um)
        self.worker.log.emit(f"--- Device {device_index} ({crosspoint_um:g} um) ---")

    def on_phase_started(self, phase, device_index, total_points):
        self.worker.phase_started.emit(phase.value, device_index)
        self.worker.log.emit(f"{phase.value} started.")

    def on_point(self, phase, row):
        self.worker.point.emit(phase.value, row)

    def on_phase_finished(self, phase, device_index, result):
        self.worker.phase_finished.emit(phase.value, device_index, result)
        points = getattr(result, "point_count", "?")
        reason = getattr(result, "termination_reason", "")
        self.worker.log.emit(f"{phase.value} finished: {points} points, {reason}.")

    def on_device_finished(self, device_index, summary_row):
        self.worker.device_finished.emit(device_index, summary_row)
        self.worker.log.emit(
            f"Device {device_index} done "
            f"({summary_row.get('stress_type')}, {_headline(summary_row)})."
        )

    def on_run_finished(self, devices_completed, reason):
        self.worker.log.emit(
            f"Run {reason}. {devices_completed} device(s) completed."
        )


def _headline(row: dict) -> str:
    """The one number that matters for this device."""
    if row.get("stress_type") == "CVS":
        return f"t_BD = {row.get('t_BD_s')} s at {row.get('V_stress_V')} V"
    return f"V_BD = {row.get('V_BD_V')} V"


class SessionWorker(QObject):
    log = pyqtSignal(str)
    device_started = pyqtSignal(int, float)
    phase_started = pyqtSignal(str, int)
    point = pyqtSignal(str, object)
    phase_finished = pyqtSignal(str, int, object)
    device_finished = pyqtSignal(int, object)
    finished = pyqtSignal(int, str)
    failed = pyqtSignal(str)

    def __init__(self, params: BreakdownParams, smu_selection: str,
                 mfia_selection: str, prompter: QtPrompter,
                 start_index: Optional[int] = None):
        super().__init__()
        self.params = params
        self.smu_selection = smu_selection
        self.mfia_selection = mfia_selection
        self.prompter = prompter
        self.start_index = start_index

        self._stop = False
        self.session: Optional[BreakdownSession] = None
        self._log_handle = None

    def stop_requested(self) -> bool:
        return self._stop

    def request_stop(self) -> None:
        """Called DIRECTLY from the GUI thread, not through a queued signal.

        While a run is in progress this object's thread is inside the session
        loop and never returns to its event loop, so a queued slot invocation
        would sit in the queue until the run ended -- which is to say, Stop
        would do nothing. This only sets flags that the loop polls, so calling
        it across threads is safe; it touches no instrument.
        """
        self._stop = True
        if self.session is not None:
            self.session.request_stop()

    @pyqtSlot()
    def run(self) -> None:
        smu = mfia = None
        try:
            timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            date = dt.datetime.now().strftime("%Y-%m-%d")
            self._open_log(date, timestamp)

            self.log.emit(f"Opening MFIA ({self.mfia_selection}) ...")
            mfia = mfia_module.open_impedance_analyzer(
                self.mfia_selection, self.params.mfia
            )
            self.log.emit(f"Opening source meter ({self.smu_selection}) ...")
            smu = smu_module.open_source_meter(
                self.smu_selection, self.params.rvs.nplc
            )

            self.session = BreakdownSession(
                params=self.params, mfia=mfia, smu=smu, prompter=self.prompter,
                date=date, timestamp=timestamp, listener=SignalListener(self),
            )
            if self.start_index is not None:
                self.session.set_next_index(self.start_index)
            # The manual-mode dialog needs to know whether a CVS is possible yet.
            self.prompter._can_run_cvs = lambda _i: bool(
                self.session.statistics.count(self.params.sample.crosspoint_um)
            )
            if self._stop:
                self.session.request_stop()

            self.log.emit("Run started.")
            result = self.session.run(device_limit=None)
            self.finished.emit(result.devices_completed, result.reason)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            for instrument in (smu, mfia):
                if instrument is None:
                    continue
                try:
                    instrument.safe_off()
                    instrument.close()
                except Exception:
                    pass
            self._close_log()

    # -- run log ------------------------------------------------------------

    def _open_log(self, date: str, timestamp: str) -> None:
        try:
            paths = SamplePaths(self.params.run.output_dir, date,
                                self.params.sample.sample_name)
            paths.ensure_sample_dir()
            self._log_handle = open(paths.log_file(timestamp), "a",
                                    encoding="utf-8")
            self.log.connect(self._write_log)
        except Exception:
            self._log_handle = None

    def _write_log(self, message: str) -> None:
        if self._log_handle is None:
            return
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        self._log_handle.write(f"{stamp} {message}\n")
        self._log_handle.flush()

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


class BreakdownWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crosspoint Breakdown Characterization")
        self.resize(1280, 860)

        self.params = BreakdownParams()
        self.worker: Optional[SessionWorker] = None
        self.thread: Optional[QThread] = None
        self.prompter: Optional[QtPrompter] = None
        self._open_dialog = None
        self._devices_done = 0

        self._build_ui()
        self.params_panel.load(self.params)
        # load() blocks signals, so the derived displays are filled in here.
        self._update_ramp_preview()
        self._update_index_label()
        self._set_state("idle", "Idle")

        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.plots.refresh)
        self.plot_timer.start(PLOT_REFRESH_MS)

    # -- construction -------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_header())

        self.tabs = QTabWidget()
        tab_font = self.tabs.tabBar().font()
        tab_font.setBold(True)
        self.tabs.tabBar().setFont(tab_font)
        self.tabs.addTab(self._build_setup_tab(), "Setup")
        self.tabs.addTab(self._build_params_tab(), "Parameters")
        self.tabs.addTab(self._build_run_tab(), "Run")
        self.tabs.addTab(self._build_help_tab(), "Help")
        layout.addWidget(self.tabs, 1)

    def _build_header(self) -> QWidget:
        """Always-visible run state.

        This is the one thing that has to be legible from across the room: what
        the rig is doing, and whether it is waiting for you. Its colour is the
        signal -- amber means a prompt is open, red means a device is energised.
        """
        bar = QWidget()
        # Scoped by object name: a bare `background:` here would cascade into
        # every child and repaint the Start/Stop buttons out of existence.
        bar.setObjectName("HeaderBar")
        bar.setStyleSheet(
            f"QWidget#HeaderBar {{ background: {SURFACE};"
            f" border-bottom: 1px solid {BORDER}; }}"
        )
        row = QHBoxLayout(bar)
        row.setContentsMargins(18, 11, 18, 11)
        row.setSpacing(26)

        self.state_chip = QLabel("IDLE")
        self.state_chip.setObjectName("StateChip")
        row.addWidget(self.state_chip)

        self.header_sample = self._header_item(row, "SAMPLE", "—")
        self.header_size = self._header_item(row, "CROSSPOINT", "—")
        self.header_device = self._header_item(row, "NEXT DEVICE", "—")
        self.header_detail = self._header_item(row, "LAST RESULT", "—")
        row.addStretch(1)

        self.start_button = QPushButton("Start run")
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        row.addWidget(self.start_button)
        row.addWidget(self.stop_button)
        return bar

    def _header_item(self, row: QHBoxLayout, key: str, value: str) -> QLabel:
        holder = QVBoxLayout()
        holder.setSpacing(2)
        key_label = QLabel(key)
        key_label.setObjectName("HeaderKey")
        value_label = QLabel(value)
        value_label.setObjectName("HeaderValue")
        holder.addWidget(key_label)
        holder.addWidget(value_label)
        row.addLayout(holder)
        return value_label

    def _build_setup_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(16)

        columns = QHBoxLayout()
        columns.setSpacing(16)
        left = QVBoxLayout()
        left.setSpacing(16)
        right = QVBoxLayout()
        right.setSpacing(16)

        instruments = Card("Instruments", "opened when the run starts")
        self.smu_combo = QComboBox()
        self.mfia_combo = QComboBox()
        self.mfia_combo.addItems(["MFIA", "Mock"])
        for combo in (self.smu_combo, self.mfia_combo):
            combo.setFixedWidth(300)
        instruments.add(field_row("Stress source", self.smu_combo))
        instruments.add(field_row("Capacitance", self.mfia_combo))
        refresh_row = QHBoxLayout()
        refresh_row.addSpacing(182)
        refresh = QPushButton("Rescan for instruments")
        refresh.clicked.connect(self._refresh_devices)
        refresh_row.addWidget(refresh)
        refresh_row.addStretch(1)
        instruments.add(refresh_row)
        left.addWidget(instruments)

        output = Card("Output", "date / sample / size / device")
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        self.folder_label.setObjectName("FieldUnit")
        browse = QPushButton("Choose folder…")
        browse.clicked.connect(self._choose_folder)
        folder_row = QHBoxLayout()
        folder_row.setSpacing(10)
        folder_row.addWidget(browse)
        folder_row.addWidget(self.folder_label, 1)
        output.add(folder_row)
        left.addWidget(output)

        indexing = Card("Device index", "per sample and crosspoint size")
        self.index_label = QLabel("—")
        self.index_label.setObjectName("HeaderValue")
        indexing.add(field_row("Next device", self.index_label))
        self.index_spin = QSpinBox()
        self.index_spin.setRange(1, 999999)
        self.index_spin.setFixedWidth(110)
        monospace(self.index_spin)
        index_row = QHBoxLayout()
        index_row.setSpacing(8)
        index_row.addSpacing(182)
        set_index = QPushButton("Set")
        set_index.clicked.connect(self._set_index)
        reset_index = QPushButton("Reset to 1")
        reset_index.clicked.connect(lambda: self._set_index(1))
        index_row.addWidget(self.index_spin)
        index_row.addWidget(set_index)
        index_row.addWidget(reset_index)
        index_row.addStretch(1)
        indexing.add(index_row)
        left.addWidget(indexing)
        left.addStretch(1)

        preview = Card("Ramp feasibility", "what the hardware can actually deliver")
        self.ramp_preview = QLabel()
        self.ramp_preview.setWordWrap(True)
        self.ramp_preview.setTextFormat(Qt.RichText)
        preview.add(self.ramp_preview)
        right.addWidget(preview)

        sequence = Card("Per device", "the run repeats this until you stop it")
        steps = QLabel(
            "1  C(f)  — log sweep, fixed bias\n"
            "2  C(V)  — linear loop through zero\n"
            "3  ⏸  reconnect the probes to the Keithley\n"
            "4  RVS or CVS, depending on the mode\n"
            "5  ⏸  reconnect the probes to the MFIA\n"
            "6  write the summary row, advance the index"
        )
        steps.setStyleSheet(f"color: {TEXT_DIM}; line-height: 160%;")
        sequence.add(steps)
        right.addWidget(sequence)
        right.addStretch(1)

        columns.addLayout(left, 1)
        columns.addLayout(right, 1)
        outer.addLayout(columns)

        self._refresh_devices()
        return page

    def _build_params_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        save = QPushButton("Save parameters…")
        save.clicked.connect(self._save_params)
        load = QPushButton("Load parameters…")
        load.clicked.connect(self._load_params)
        buttons.addWidget(save)
        buttons.addWidget(load)
        hint = QLabel("Fields accept expressions — 1e-4 and 1/10000 both work.")
        hint.setObjectName("CardNote")
        buttons.addSpacing(8)
        buttons.addWidget(hint)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.params_panel = ParamsPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.params_panel)
        layout.addWidget(scroll, 1)

        for key in ("rvs.ramp_rate_Vps", "rvs.max_step_V", "rvs.nplc",
                    "rvs.source_delay_s", "rvs.v_max"):
            widget = self.params_panel.widgets.get(key)
            if widget is not None:
                widget.textChanged.connect(self._update_ramp_preview)
        for key in ("sample.sample_name", "sample.crosspoint_um"):
            widget = self.params_panel.widgets.get(key)
            if widget is not None:
                widget.textChanged.connect(self._update_index_label)
        return page

    def _build_run_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        top = QHBoxLayout()
        self.progress_label = QLabel("No run in progress.")
        self.progress_label.setObjectName("FieldLabel")
        top.addWidget(self.progress_label)
        top.addStretch(1)
        layout.addLayout(top)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.plots = PlotPanel()
        layout.addWidget(self.plots, 1)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(150)
        layout.addWidget(self.log_view)
        return page

    def _build_help_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 14, 18, 14)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(CVS_GUIDANCE)
        layout.addWidget(text)
        return page

    # -- run state ----------------------------------------------------------

    def _set_state(self, kind: str, text: str) -> None:
        self.state_chip.setText(text.upper())
        self.state_chip.setStyleSheet(
            "font-size: 11px; font-weight: 700; letter-spacing: 1.4px;"
            "padding: 5px 12px; border-radius: 4px;" + state_chip_style(kind)
        )

    # -- setup actions ------------------------------------------------------

    def _refresh_devices(self) -> None:
        previous = self.smu_combo.currentText()
        self.smu_combo.clear()
        self.smu_combo.addItems(smu_module.list_devices(include_mock=True))
        index = self.smu_combo.findText(previous)
        if index >= 0:
            self.smu_combo.setCurrentIndex(index)

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose the output folder")
        if folder:
            self.params.run.output_dir = folder
            self.folder_label.setText(folder)
            self.params_panel.load(self.params)
            self._update_index_label()

    def _set_index(self, value: Optional[int] = None) -> None:
        self._forced_index = int(value if value else self.index_spin.value())
        self.index_label.setText(f"{self._forced_index}   (set manually)")
        self.header_device.setText(str(self._forced_index))

    def _update_index_label(self) -> None:
        self._forced_index = None
        problems = self.params_panel.apply_to(self.params)
        self.header_sample.setText(self.params.sample.sample_name or "—")
        self.header_size.setText(f"{self.params.sample.crosspoint_um:g} µm")

        if problems or not self.params.run.output_dir:
            self.index_label.setText("—")
            self.header_device.setText("—")
            return
        try:
            paths = SamplePaths(self.params.run.output_dir,
                                dt.datetime.now().strftime("%Y-%m-%d"),
                                self.params.sample.sample_name)
            nxt = paths.next_device_index(self.params.sample.crosspoint_um)
            self.index_label.setText(str(nxt))
            self.header_device.setText(str(nxt))
            self.index_spin.setValue(nxt)
        except Exception:
            self.index_label.setText("—")
            self.header_device.setText("—")

    def _update_ramp_preview(self) -> None:
        problems = self.params_panel.apply_to(self.params)
        rvs = self.params.rvs
        if problems or rvs.ramp_rate_Vps <= 0 or rvs.max_step_V <= 0 or rvs.nplc <= 0:
            self.ramp_preview.setText("Enter valid RVS parameters to see a preview.")
            return
        plan = plan_ramp(rvs.ramp_rate_Vps, rvs.max_step_V, rvs.nplc,
                         rvs.source_delay_s)
        estimate = abs(rvs.v_max) / plan.achievable_rate_Vps \
            if plan.achievable_rate_Vps else 0.0
        colour = WAIT if plan.rate_limited else GO
        headline = (f"{plan.achievable_rate_Vps:.3g} V/s"
                    if not plan.rate_limited
                    else f"{plan.achievable_rate_Vps:.3g} V/s "
                         f"(you asked for {plan.requested_rate_Vps:g})")
        self.ramp_preview.setText(
            f"<div style='font-size:22px;color:{colour};font-weight:600'>"
            f"{headline}</div>"
            f"<div style='color:{TEXT_DIM};margin-top:6px'>"
            f"{plan.step_V:.4g} V steps every {plan.dwell_s * 1e3:.1f} ms &nbsp;·&nbsp; "
            f"{abs(rvs.v_max) / plan.step_V:.0f} points to {abs(rvs.v_max):g} V "
            f"&nbsp;·&nbsp; {estimate:.1f} s if it survives</div>"
            + (f"<div style='color:{WAIT};margin-top:8px'>{plan.note}</div>"
               if plan.rate_limited else "")
        )

    # -- parameters ---------------------------------------------------------

    def _save_params(self) -> None:
        problems = self.params_panel.apply_to(self.params)
        if problems:
            QMessageBox.warning(self, "Cannot save",
                                "Fix these first:\n\n- " + "\n- ".join(problems))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save parameters", "breakdown_params.json", "JSON (*.json)"
        )
        if path:
            save_params(path, self.params)
            self._append_log(f"Parameters saved to {path}")

    def _load_params(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load parameters", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            result = load_params(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load", str(exc))
            return
        self.params = result.params
        self.params_panel.load(self.params)
        self.folder_label.setText(self.params.run.output_dir or "(no folder selected)")
        self._update_index_label()
        self._update_ramp_preview()
        self._append_log(f"Parameters loaded from {path}")
        if result.warnings:
            QMessageBox.information(
                self, "Loaded with notes", "\n".join(f"- {w}" for w in result.warnings)
            )

    # -- run control --------------------------------------------------------

    def _start(self) -> None:
        problems = self.params_panel.apply_to(self.params)
        if problems:
            QMessageBox.warning(self, "Check the parameters",
                                "- " + "\n- ".join(problems))
            return
        if not self.params.run.output_dir:
            QMessageBox.warning(self, "No output folder",
                                "Choose an output folder before starting.")
            return
        errors = validate(self.params)
        if errors:
            QMessageBox.warning(self, "Parameters are not valid",
                                "- " + "\n- ".join(errors))
            return

        self.plots.clear_all()
        self.log_view.clear()
        self._devices_done = 0

        self.prompter = QtPrompter(should_stop=lambda: self.worker is not None
                                   and self.worker.stop_requested())
        self.prompter.swap_requested.connect(self._on_swap_requested)
        self.prompter.cvs_requested.connect(self._on_cvs_requested)
        self.prompter.stress_type_requested.connect(self._on_stress_type_requested)

        self.worker = SessionWorker(
            params=self.params,
            smu_selection=self.smu_combo.currentText(),
            mfia_selection=self.mfia_combo.currentText(),
            prompter=self.prompter,
            start_index=getattr(self, "_forced_index", None),
        )
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.device_started.connect(self._on_device_started)
        self.worker.phase_started.connect(self._on_phase_started)
        self.worker.point.connect(self._on_point)
        self.worker.device_finished.connect(self._on_device_finished)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)

        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._clear_worker)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setVisible(True)
        self._set_state("measuring", "Running")
        self.tabs.setCurrentIndex(2)
        self.thread.start()

    def _stop(self) -> None:
        if self.worker is None:
            return
        self._set_state("waiting", "Stopping")
        self.stop_button.setEnabled(False)
        self.worker.request_stop()
        # A prompt may be on screen; close it so the waiting thread can unwind.
        if self._open_dialog is not None:
            self._open_dialog.reject()

    def _clear_worker(self) -> None:
        if self.thread is not None:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None
        self.prompter = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setVisible(False)
        self._update_index_label()

    # -- prompt handlers (GUI thread) ---------------------------------------

    def _run_dialog(self, dialog, answer_when_accepted):
        self._open_dialog = dialog
        try:
            accepted = dialog.exec_() == dialog.Accepted
            answer = answer_when_accepted(dialog) if accepted else None
        finally:
            self._open_dialog = None
        if self.prompter is not None:
            self.prompter.deliver(answer)

    @pyqtSlot(object, int)
    def _on_swap_requested(self, target: Instrument, device_index: int) -> None:
        self._run_dialog(CableSwapDialog(target, device_index, self), lambda _d: True)

    @pyqtSlot(object)
    def _on_cvs_requested(self, ctx: CvsVoltageContext) -> None:
        self._run_dialog(CvsVoltageDialog(ctx, self),
                         lambda d: d.selected_voltage())

    @pyqtSlot(int, bool)
    def _on_stress_type_requested(self, device_index: int, can_cvs: bool) -> None:
        self._run_dialog(StressTypeDialog(device_index, can_cvs, self),
                         lambda d: d.selected())

    # -- progress handlers --------------------------------------------------

    def _append_log(self, message: str) -> None:
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        self.log_view.append(f"{stamp}  {message}")

    @pyqtSlot(int, float)
    def _on_device_started(self, index: int, size: float) -> None:
        self.plots.reset_device()
        self.progress_label.setText(f"Device {index}  ({size:g} um)")

    @pyqtSlot(str, int)
    def _on_phase_started(self, phase: str, index: int) -> None:
        kind = "stressing" if phase in ("RVS", "CVS") else "measuring"
        self._set_state(kind, f"{phase}  ·  device {index}")

    @pyqtSlot(str, object)
    def _on_point(self, phase: str, row: dict) -> None:
        self.plots.add_point(phase, row)

    @pyqtSlot(int, object)
    def _on_device_finished(self, index: int, row: dict) -> None:
        self._devices_done += 1
        self.progress_label.setText(f"{self._devices_done} device(s) completed.")

    @pyqtSlot(int, str)
    def _on_finished(self, completed: int, reason: str) -> None:
        self._set_state("done" if reason == RunOutcome.COMPLETED else "idle",
                        reason)
        if reason == RunOutcome.ABORTED:
            QMessageBox.information(self, "Run aborted",
                                    f"{completed} device(s) were completed.")

    @pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        self._set_state("stressing", "Failed")
        self._append_log(message)
        QMessageBox.critical(self, "Run failed", message)

    # -- shutdown -----------------------------------------------------------

    def closeEvent(self, event) -> None:
        # Refusing to close while a run is live is not politeness: a CVS hold
        # means the SMU is energised on a probe.
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(
                self, "Run in progress",
                "Stop the run before closing, so both instruments are driven "
                "to zero and their outputs opened.",
            )
            self._stop()
            event.ignore()
            return
        event.accept()
