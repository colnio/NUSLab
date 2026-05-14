from __future__ import annotations

import csv
import datetime
import json
import os
import os.path as op
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from KeithleyGUI.ui_helpers import ProgressEta, apply_standard_window_style, parse_numeric_text
from transport import (
    DeviceCandidate,
    MeasurementResult,
    MockSheetResistanceTransport,
    RestSheetResistanceTransport,
    discover_candidates,
)


class DiscoveryWorker(QObject):
    finished = pyqtSignal(list, str)

    def __init__(self, cidr: str, limit: int, connect_timeout: float, http_timeout: float) -> None:
        super().__init__()
        self.cidr = cidr
        self.limit = limit
        self.connect_timeout = connect_timeout
        self.http_timeout = http_timeout
        self.cancel_event = threading.Event()

    def run(self) -> None:
        try:
            devices = discover_candidates(
                self.cidr,
                limit=self.limit,
                connect_timeout=self.connect_timeout,
                http_timeout=self.http_timeout,
                cancel_event=self.cancel_event,
            )
            message = f"Found {len(devices)} candidate device(s)"
        except Exception as exc:
            devices = []
            message = f"Discovery failed: {exc}"
        self.finished.emit(devices, message)

    def cancel(self) -> None:
        self.cancel_event.set()


class MeasurementWorker(QObject):
    result_ready = pyqtSignal(object)
    finished = pyqtSignal(str)

    def __init__(self, transport: Any, count: int, interval_ms: int) -> None:
        super().__init__()
        self.transport = transport
        self.count = count
        self.interval_ms = interval_ms
        self.cancel_event = threading.Event()

    def run(self) -> None:
        message = "Completed"
        try:
            for _ in range(self.count):
                if self.cancel_event.is_set():
                    message = "Stopped"
                    break
                result = self.transport.measure()
                self.result_ready.emit(result)
                deadline = time.perf_counter() + self.interval_ms / 1000.0
                while time.perf_counter() < deadline:
                    if self.cancel_event.is_set():
                        message = "Stopped"
                        break
                    time.sleep(0.01)
                if self.cancel_event.is_set():
                    break
        except Exception as exc:
            message = f"Measurement failed: {exc}"
        self.finished.emit(message)

    def cancel(self) -> None:
        self.cancel_event.set()


class SheetResistanceApp(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.transport = None
        self.connected_summary: dict[str, Any] | None = None
        self.discovery_thread: QThread | None = None
        self.discovery_worker: DiscoveryWorker | None = None
        self.measurement_thread: QThread | None = None
        self.measurement_worker: MeasurementWorker | None = None
        self.discovered_devices: list[DeviceCandidate] = []
        self.measurements: list[MeasurementResult] = []

        self.date = str(datetime.date.today())
        self.folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if not self.folder:
            self.folder = os.getcwd()
        if not op.exists(op.join(self.folder, self.date)):
            os.makedirs(op.join(self.folder, self.date))

        self.init_ui()

    def init_ui(self) -> None:
        self.setWindowTitle("Sheet Resistance Measurement")
        self.resize(1600, 950)

        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        conn_box = QGroupBox("Connection")
        conn_layout = QGridLayout()
        conn_layout.setHorizontalSpacing(10)
        conn_layout.setVerticalSpacing(6)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Mock", "HTTP/REST"])
        self.device_combo = QComboBox()
        self.device_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.device_combo.currentIndexChanged.connect(self.on_device_selected)
        self.refresh_button = QPushButton("Discover")
        self.refresh_button.clicked.connect(self.start_discovery)
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_selected_device)
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self.disconnect_device)
        self.disconnect_button.setEnabled(False)
        self.connection_status_label = QLabel("Idle")
        self.connection_status_label.setStyleSheet("color: #64748b;")

        self.cidr_input = QLineEdit("169.254.181.0/24")
        self.scan_limit_input = QSpinBox()
        self.scan_limit_input.setRange(1, 65535)
        self.scan_limit_input.setValue(512)
        self.host_input = QLineEdit("")
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(80)
        self.https_checkbox = QCheckBox("HTTPS")
        self.status_path_input = QLineEdit("/api/status")
        self.connection_timeout_input = QLineEdit("3.0")

        conn_layout.addWidget(QLabel("Mode"), 0, 0)
        conn_layout.addWidget(self.mode_combo, 0, 1)
        conn_layout.addWidget(QLabel("Candidates"), 0, 2)
        conn_layout.addWidget(self.device_combo, 0, 3, 1, 3)
        conn_layout.addWidget(self.refresh_button, 0, 6)
        conn_layout.addWidget(self.connect_button, 0, 7)
        conn_layout.addWidget(self.disconnect_button, 0, 8)
        conn_layout.addWidget(QLabel("CIDR"), 1, 0)
        conn_layout.addWidget(self.cidr_input, 1, 1, 1, 2)
        conn_layout.addWidget(QLabel("Scan limit"), 1, 3)
        conn_layout.addWidget(self.scan_limit_input, 1, 4)
        conn_layout.addWidget(QLabel("Manual host"), 1, 5)
        conn_layout.addWidget(self.host_input, 1, 6)
        conn_layout.addWidget(QLabel("Port"), 1, 7)
        conn_layout.addWidget(self.port_input, 1, 8)
        conn_layout.addWidget(self.https_checkbox, 2, 0)
        conn_layout.addWidget(QLabel("Status path"), 2, 1)
        conn_layout.addWidget(self.status_path_input, 2, 2, 1, 3)
        conn_layout.addWidget(QLabel("Timeout (s)"), 2, 5)
        conn_layout.addWidget(self.connection_timeout_input, 2, 6)
        conn_layout.addWidget(self.connection_status_label, 2, 7, 1, 2)
        conn_box.setLayout(conn_layout)

        sample_box = QGroupBox("Sample")
        sample_layout = QFormLayout()
        self.sample_name_input = QLineEdit()
        self.operator_input = QLineEdit()
        self.notes_input = QLineEdit()
        self.folder_input = QLineEdit(self.folder)
        self.folder_button = QPushButton("Browse")
        self.folder_button.clicked.connect(self.choose_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_input)
        folder_row.addWidget(self.folder_button)
        folder_widget = QWidget()
        folder_widget.setLayout(folder_row)
        sample_layout.addRow("Sample name:", self.sample_name_input)
        sample_layout.addRow("Operator:", self.operator_input)
        sample_layout.addRow("Notes:", self.notes_input)
        sample_layout.addRow("Save folder:", folder_widget)
        sample_box.setLayout(sample_layout)

        api_box = QGroupBox("API / Measurement")
        api_layout = QGridLayout()
        api_layout.setHorizontalSpacing(10)
        api_layout.setVerticalSpacing(6)
        self.measure_path_input = QLineEdit("/api/measure")
        self.measure_method_combo = QComboBox()
        self.measure_method_combo.addItems(["GET", "POST"])
        self.payload_input = QTextEdit()
        self.payload_input.setPlaceholderText('{"averages": 3}')
        self.payload_input.setFixedHeight(88)
        self.value_key_input = QLineEdit("sheet_resistance")
        self.unit_key_input = QLineEdit("unit")
        self.measure_count_input = QSpinBox()
        self.measure_count_input.setRange(1, 100000)
        self.measure_count_input.setValue(10)
        self.interval_ms_input = QSpinBox()
        self.interval_ms_input.setRange(50, 600000)
        self.interval_ms_input.setValue(1000)

        api_layout.addWidget(QLabel("Measure path"), 0, 0)
        api_layout.addWidget(self.measure_path_input, 0, 1, 1, 3)
        api_layout.addWidget(QLabel("Method"), 0, 4)
        api_layout.addWidget(self.measure_method_combo, 0, 5)
        api_layout.addWidget(QLabel("Value key"), 1, 0)
        api_layout.addWidget(self.value_key_input, 1, 1)
        api_layout.addWidget(QLabel("Unit key"), 1, 2)
        api_layout.addWidget(self.unit_key_input, 1, 3)
        api_layout.addWidget(QLabel("Count"), 1, 4)
        api_layout.addWidget(self.measure_count_input, 1, 5)
        api_layout.addWidget(QLabel("Interval (ms)"), 2, 4)
        api_layout.addWidget(self.interval_ms_input, 2, 5)
        api_layout.addWidget(QLabel("JSON payload"), 2, 0)
        api_layout.addWidget(self.payload_input, 2, 1, 2, 3)
        api_box.setLayout(api_layout)

        top_row = QHBoxLayout()
        top_row.addWidget(conn_box, 2)
        top_row.addWidget(sample_box, 1)
        root_layout.addLayout(top_row)
        root_layout.addWidget(api_box)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_measurement)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(self.stop_measurement)
        self.stop_button.setEnabled(False)
        self.last_value_label = QLabel("Sheet Resistance: --")
        self.last_value_label.setAlignment(Qt.AlignCenter)
        self.connected_device_label = QLabel("No device connected")
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.last_value_label, 1)
        button_row.addWidget(self.connected_device_label, 1)
        button_row.addWidget(self.stop_button)
        root_layout.addLayout(button_row)

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground("w")
        self.resistance_plot = self.plot_widget.addPlot(title="Sheet Resistance")
        self.resistance_plot.showGrid(x=True, y=True, alpha=0.12)
        self.resistance_plot.setLabel("left", "Sheet Resistance", units="ohm/sq")
        self.resistance_plot.setLabel("bottom", "Measurement #")
        self.resistance_curve = self.resistance_plot.plot([], [], pen=pg.mkPen("#0f766e", width=2), symbol="o")
        self.plot_widget.nextRow()
        self.latency_plot = self.plot_widget.addPlot(title="Response Time")
        self.latency_plot.showGrid(x=True, y=True, alpha=0.12)
        self.latency_plot.setLabel("left", "Latency", units="s")
        self.latency_plot.setLabel("bottom", "Measurement #")
        self.latency_curve = self.latency_plot.plot([], [], pen=pg.mkPen("#dc2626", width=2), symbol="x")
        root_layout.addWidget(self.plot_widget, 1)

        self.status_label = QLabel("")
        self.status_label.setFrameShape(QFrame.StyledPanel)
        self.status_label.setMinimumHeight(24)
        root_layout.addWidget(self.status_label)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(1000)
        self.log_output.setMinimumHeight(180)
        root_layout.addWidget(self.log_output)
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Progress: 0/0")
        self.eta_label = QLabel("ETA: --")
        root_layout.addWidget(self.progress_bar)
        footer = QHBoxLayout()
        footer.addWidget(self.progress_label, 2)
        footer.addWidget(self.eta_label, 1)
        root_layout.addLayout(footer)

        self.setLayout(root_layout)
        apply_standard_window_style(self)
        self.progress_tracker = ProgressEta(self.progress_bar, self.eta_label, self.progress_label)
        self.device_combo.addItem("Mock", None)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        self.on_mode_changed(self.mode_combo.currentText())

    def append_log(self, text: str) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{stamp}] {text}")

    def on_mode_changed(self, mode: str) -> None:
        is_rest = mode == "HTTP/REST"
        self.refresh_button.setEnabled(is_rest and self.discovery_thread is None)
        self.cidr_input.setEnabled(is_rest)
        self.scan_limit_input.setEnabled(is_rest)
        self.host_input.setEnabled(is_rest)
        self.port_input.setEnabled(is_rest)
        self.https_checkbox.setEnabled(is_rest)
        self.status_path_input.setEnabled(is_rest)
        self.measure_path_input.setEnabled(is_rest)
        self.measure_method_combo.setEnabled(is_rest)
        self.payload_input.setEnabled(is_rest)
        self.value_key_input.setEnabled(is_rest)
        self.unit_key_input.setEnabled(is_rest)
        self.device_combo.setEnabled(True)
        self.device_combo.clear()
        if is_rest:
            self.device_combo.addItem("Manual entry", None)
        else:
            self.device_combo.addItem("Mock", None)

    def on_device_selected(self, index: int) -> None:
        candidate = self.device_combo.itemData(index)
        if not isinstance(candidate, DeviceCandidate):
            return
        self.host_input.setText(candidate.host)
        self.port_input.setValue(candidate.port)
        self.https_checkbox.setChecked(candidate.use_https)
        self.status_path_input.setText(candidate.status_path)

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Directory", self.folder_input.text() or self.folder)
        if folder:
            self.folder = folder
            self.folder_input.setText(folder)

    def start_discovery(self) -> None:
        if self.discovery_thread is not None:
            return
        self.connection_status_label.setText("Discovering...")
        self.refresh_button.setEnabled(False)
        self.device_combo.clear()
        self.device_combo.addItem("Scanning...", None)

        self.discovery_thread = QThread(self)
        self.discovery_worker = DiscoveryWorker(
            cidr=self.cidr_input.text().strip(),
            limit=int(self.scan_limit_input.value()),
            connect_timeout=0.2,
            http_timeout=1.0,
        )
        self.discovery_worker.moveToThread(self.discovery_thread)
        self.discovery_thread.started.connect(self.discovery_worker.run)
        self.discovery_worker.finished.connect(self.on_discovery_finished)
        self.discovery_worker.finished.connect(self.discovery_thread.quit)
        self.discovery_thread.finished.connect(self.cleanup_discovery_thread)
        self.discovery_thread.start()

    def cleanup_discovery_thread(self) -> None:
        self.discovery_thread = None
        self.discovery_worker = None
        self.refresh_button.setEnabled(self.mode_combo.currentText() == "HTTP/REST")

    def on_discovery_finished(self, devices: list, message: str) -> None:
        self.discovered_devices = [item for item in devices if isinstance(item, DeviceCandidate)]
        self.device_combo.clear()
        self.device_combo.addItem("Manual entry", None)
        for item in self.discovered_devices:
            self.device_combo.addItem(item.label, item)
        self.connection_status_label.setText(message)
        self.append_log(message)

    def connect_selected_device(self) -> None:
        try:
            timeout = parse_numeric_text(self.connection_timeout_input.text(), "Timeout")
            if self.mode_combo.currentText() == "Mock":
                transport = MockSheetResistanceTransport()
                summary = transport.connect()
                label = "Mock"
            else:
                host = self.host_input.text().strip()
                if not host:
                    raise ValueError("Manual host is required for HTTP/REST mode.")
                transport = RestSheetResistanceTransport(
                    host=host,
                    port=int(self.port_input.value()),
                    use_https=self.https_checkbox.isChecked(),
                    timeout=float(timeout),
                    status_path=self.status_path_input.text().strip(),
                    measure_path=self.measure_path_input.text().strip(),
                    measure_method=self.measure_method_combo.currentText(),
                    payload_text=self.payload_input.toPlainText(),
                    value_key=self.value_key_input.text().strip(),
                    unit_key=self.unit_key_input.text().strip(),
                )
                summary = transport.connect()
                label = f"{host}:{self.port_input.value()}"

            self.transport = transport
            self.connected_summary = summary
            self.connected_device_label.setText(label)
            self.connection_status_label.setText("Connected")
            self.disconnect_button.setEnabled(True)
            self.append_log(f"Connected to {label}: {json.dumps(summary, indent=2, default=str)[:500]}")
        except Exception as exc:
            self.transport = None
            self.connected_summary = None
            QMessageBox.critical(self, "Connection Error", str(exc))
            self.connection_status_label.setText(f"Connect failed: {exc}")

    def disconnect_device(self) -> None:
        if self.transport is not None:
            try:
                self.transport.close()
            except Exception:
                pass
        self.transport = None
        self.connected_summary = None
        self.connected_device_label.setText("No device connected")
        self.connection_status_label.setText("Disconnected")
        self.disconnect_button.setEnabled(False)
        self.append_log("Disconnected")

    def start_measurement(self) -> None:
        if self.measurement_thread is not None:
            return
        if self.transport is None:
            self.connect_selected_device()
            if self.transport is None:
                return

        count = int(self.measure_count_input.value())
        interval_ms = int(self.interval_ms_input.value())

        if self.mode_combo.currentText() == "HTTP/REST":
            try:
                timeout = parse_numeric_text(self.connection_timeout_input.text(), "Timeout")
                self.transport = RestSheetResistanceTransport(
                    host=self.host_input.text().strip(),
                    port=int(self.port_input.value()),
                    use_https=self.https_checkbox.isChecked(),
                    timeout=float(timeout),
                    status_path=self.status_path_input.text().strip(),
                    measure_path=self.measure_path_input.text().strip(),
                    measure_method=self.measure_method_combo.currentText(),
                    payload_text=self.payload_input.toPlainText(),
                    value_key=self.value_key_input.text().strip(),
                    unit_key=self.unit_key_input.text().strip(),
                )
            except Exception as exc:
                QMessageBox.critical(self, "Error", str(exc))
                return

        self.measurements = []
        self.resistance_curve.setData([], [])
        self.latency_curve.setData([], [])
        self.progress_tracker.start(count)
        self.last_value_label.setText("Sheet Resistance: --")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Measurement running...")
        self.append_log(
            f"Starting run: count={count}, interval_ms={interval_ms}, "
            f"sample={self.sample_name_input.text().strip() or 'unnamed'}"
        )

        self.measurement_thread = QThread(self)
        self.measurement_worker = MeasurementWorker(self.transport, count, interval_ms)
        self.measurement_worker.moveToThread(self.measurement_thread)
        self.measurement_thread.started.connect(self.measurement_worker.run)
        self.measurement_worker.result_ready.connect(self.on_measurement_result)
        self.measurement_worker.finished.connect(self.on_measurement_finished)
        self.measurement_worker.finished.connect(self.measurement_thread.quit)
        self.measurement_thread.finished.connect(self.cleanup_measurement_thread)
        self.measurement_thread.start()

    def cleanup_measurement_thread(self) -> None:
        self.measurement_thread = None
        self.measurement_worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def stop_measurement(self) -> None:
        if self.measurement_worker is not None:
            self.measurement_worker.cancel()

    def on_measurement_result(self, result: object) -> None:
        if not isinstance(result, MeasurementResult):
            return
        self.measurements.append(result)
        x = list(range(1, len(self.measurements) + 1))
        self.resistance_curve.setData(x, [item.value for item in self.measurements])
        self.latency_curve.setData(x, [item.latency_s for item in self.measurements])
        self.progress_tracker.step(len(self.measurements))
        self.last_value_label.setText(f"Sheet Resistance: {result.value:.6g} {result.unit}")
        self.status_label.setText(f"Latest: {result.value:.6g} {result.unit} | latency={result.latency_s:.3f} s")
        self.append_log(
            f"#{len(self.measurements)} value={result.value:.6g} {result.unit} "
            f"latency={result.latency_s:.3f}s raw={json.dumps(result.raw, default=str)[:300]}"
        )

    def on_measurement_finished(self, message: str) -> None:
        self.status_label.setText(message)
        self.append_log(message)
        if self.measurements:
            try:
                csv_path = self.save_results()
                self.append_log(f"Saved results to {csv_path}")
            except Exception as exc:
                self.append_log(f"Save failed: {exc}")

    def save_results(self) -> str:
        sample_name = self.sample_name_input.text().strip() or "sheet_resistance"
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = self.folder_input.text().strip() or self.folder
        out_dir = op.join(folder, self.date)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = op.join(out_dir, f"{sample_name}_{stamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp",
                    "sheet_resistance",
                    "unit",
                    "latency_s",
                    "sample_name",
                    "operator",
                    "notes",
                    "raw_json",
                ],
            )
            writer.writeheader()
            for item in self.measurements:
                writer.writerow(
                    {
                        "timestamp": datetime.datetime.fromtimestamp(item.timestamp).isoformat(),
                        "sheet_resistance": item.value,
                        "unit": item.unit,
                        "latency_s": item.latency_s,
                        "sample_name": self.sample_name_input.text().strip(),
                        "operator": self.operator_input.text().strip(),
                        "notes": self.notes_input.text().strip(),
                        "raw_json": json.dumps(item.raw, default=str),
                    }
                )
        return csv_path

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.discovery_worker is not None:
            self.discovery_worker.cancel()
        if self.measurement_worker is not None:
            self.measurement_worker.cancel()
        self.disconnect_device()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = SheetResistanceApp()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
