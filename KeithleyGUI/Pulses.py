import sys
import datetime
import gc
import os
import os.path as op

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import keithley
from ui_helpers import (
    ProgressEta,
    apply_standard_window_style,
    build_device_metadata,
    ensure_directory,
    parse_numeric_text,
    write_json_file,
)

matplotlib.use("Agg")


class PulsesWorker(QObject):
    batch_completed = pyqtSignal(object, object)
    cycle_completed = pyqtSignal(int)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, settings):
        super().__init__()
        self.settings = dict(settings)
        self._stop_requested = False
        self._time_offset_s = 0.0

    def request_stop(self):
        self._stop_requested = True

    def _annotate_rows(self, rows, batch, cycle_index):
        if len(rows) != int(batch["expected_points"]):
            raise RuntimeError(
                f"Expected {int(batch['expected_points'])} pulse readings, received {len(rows)}."
            )
        annotated = []
        point_period_s = float(self.settings["point_period_ms"]) / 1000.0
        for row_index, (template, row) in enumerate(zip(batch["point_templates"], rows), start=1):
            item = dict(row)
            raw_time = item.get("Timestamp_s", np.nan)
            if np.isfinite(raw_time):
                adjusted_time = self._time_offset_s + max(0.0, float(raw_time))
            else:
                adjusted_time = self._time_offset_s + (row_index - 1) * point_period_s
            item.update({
                "CycleIndex": int(cycle_index),
                "BatchIndex": int(batch["batch_index"]),
                "SegmentIndex": int(template["SegmentIndex"]),
                "SegmentLabel": str(template["SegmentLabel"]),
                "SegmentRole": str(template["SegmentRole"]),
                "PointIndexInSegment": int(template["PointIndexInSegment"]),
                "IsBaseline": bool(template["IsBaseline"]),
                "Voltage": float(template["Voltage"]) if np.isfinite(template["Voltage"]) else np.nan,
                "Timestamp_s": adjusted_time,
                "Engine": "Source Memory",
            })
            annotated.append(item)

        if annotated:
            self._time_offset_s = float(annotated[-1]["Timestamp_s"]) + point_period_s
        return annotated

    def run(self):
        device = None
        adapter = None
        try:
            self.status_changed.emit("Opening Keithley 2400...")
            device = keithley.get_device(self.settings["device_address"], nplc=self.settings["nplc"])
            if not isinstance(device, keithley.Keithley2400):
                raise RuntimeError("Selected device is not a Keithley 2400.")
            adapter = keithley.Keithley2400FastSweepAdapter(device)
            planned_batches = list(self.settings["planned_cycle"]["batches"])
            cycles_completed = 0
            while True:
                if not self.settings["continuous"] and cycles_completed >= int(self.settings["cycles_requested"]):
                    break
                cycle_index = cycles_completed + 1
                self.status_changed.emit(f"Cycle {cycle_index} starting...")
                for batch_position, batch in enumerate(planned_batches, start=1):
                    self.status_changed.emit(
                        f"Cycle {cycle_index} | batch {batch_position}/{len(planned_batches)}"
                    )
                    rows = adapter.execute_source_memory_batch(
                        points=batch["points"],
                        compliance_current=self.settings["compliance_current"],
                        nplc=self.settings["nplc"],
                        source_delay_s=self.settings["source_delay_ms"] / 1000.0,
                        current_range=self.settings["current_range"],
                        current_autorange=self.settings["current_autorange"],
                        keep_output_on=True,
                        hold_voltage=self.settings["baseline_voltage"],
                    )
                    annotated = self._annotate_rows(rows, batch, cycle_index)
                    self.batch_completed.emit(annotated, {
                        "cycle_index": int(cycle_index),
                        "batch_index": int(batch["batch_index"]),
                        "batch_position": int(batch_position),
                        "batch_count": len(planned_batches),
                        "point_count": len(annotated),
                    })
                    if any(
                        bool(item.get("InCompliance")) or bool(item.get("RangeCompliance"))
                        for item in annotated
                    ):
                        raise RuntimeError(
                            f"Compliance triggered during cycle {cycle_index}, batch {batch_position}."
                        )
                cycles_completed = cycle_index
                self.cycle_completed.emit(cycle_index)
                if self._stop_requested:
                    break

            if adapter is not None:
                adapter.abort_to_zero(clear_only=False)
            self.finished.emit({
                "status": "stopped" if self._stop_requested else "completed",
                "message": "",
            })
        except Exception as exc:
            if adapter is not None:
                try:
                    adapter.abort_to_zero(clear_only=False)
                except Exception:
                    pass
            self.finished.emit({
                "status": "error",
                "message": str(exc),
            })
        finally:
            if device is not None:
                try:
                    keithley.shutdown_device(device, close=True)
                except Exception:
                    pass


class PulsesRegime(QWidget):
    ROLE_OPTIONS = ["READ", "SET", "RESET", "HOLD", "PULSE"]
    DATA_COLUMNS = [
        "CycleIndex",
        "BatchIndex",
        "SegmentIndex",
        "SegmentLabel",
        "SegmentRole",
        "PointIndexInSegment",
        "IsBaseline",
        "Voltage",
        "Current",
        "Timestamp_s",
        "Status",
        "InCompliance",
        "RangeCompliance",
        "Engine",
    ]

    def __init__(self):
        super().__init__()
        self.folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.date = str(datetime.date.today())
        if self.folder:
            os.makedirs(op.join(self.folder, self.date), exist_ok=True)

        self.sample_name = ""
        self.device_address = ""
        self.measurements = []
        self.measurement_metadata = {}
        self.completed_steps = 0
        self.total_steps = 0
        self.completed_cycles = 0
        self.start_time = datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S")
        self.output_paths = {}
        self.save_requested = True
        self.worker = None
        self.worker_thread = None
        self.planned_cycle = None
        self.sequence_rows = []
        self.progress_tracker = None
        self.source_delay_ms = 0.0
        self.point_period_ms = 0.0
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Keithley 2400 Pulses")
        self.resize(1550, 980)
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        connection_box = QGroupBox("Connection")
        connection_layout = QGridLayout()
        connection_layout.setHorizontalSpacing(10)
        connection_layout.setVerticalSpacing(6)
        self.device_address_input = QComboBox()
        self.device_address_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.refresh_button = QPushButton("Refresh GPIB")
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.refresh_status_label = QLabel("Idle")
        self.refresh_status_label.setStyleSheet("color: #64748b;")
        self.sample_name_input = QLineEdit()
        connection_layout.addWidget(QLabel("2400 device:"), 0, 0)
        connection_layout.addWidget(self.device_address_input, 0, 1, 1, 3)
        connection_layout.addWidget(self.refresh_button, 0, 4)
        connection_layout.addWidget(self.refresh_status_label, 0, 5)
        connection_layout.addWidget(QLabel("Sample name:"), 1, 0)
        connection_layout.addWidget(self.sample_name_input, 1, 1, 1, 3)
        connection_box.setLayout(connection_layout)
        root_layout.addWidget(connection_box)

        sequence_box = QGroupBox("Pulse Sequence")
        sequence_layout = QVBoxLayout()
        self.sequence_table = QTableWidget(0, 5)
        self.sequence_table.setHorizontalHeaderLabels([
            "Enabled",
            "Label",
            "Role",
            "Voltage (V)",
            "Dwell (ms)",
        ])
        self.sequence_table.verticalHeader().setVisible(False)
        self.sequence_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sequence_table.setSelectionMode(QTableWidget.SingleSelection)
        self.sequence_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.sequence_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.sequence_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.sequence_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.sequence_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.sequence_table.itemChanged.connect(self._on_sequence_item_changed)
        sequence_layout.addWidget(self.sequence_table)

        sequence_button_layout = QHBoxLayout()
        self.add_row_button = QPushButton("Add Row")
        self.add_row_button.clicked.connect(self.add_sequence_row)
        self.remove_row_button = QPushButton("Remove Row")
        self.remove_row_button.clicked.connect(self.remove_selected_sequence_row)
        self.move_up_button = QPushButton("Move Up")
        self.move_up_button.clicked.connect(lambda: self.move_selected_sequence_row(-1))
        self.move_down_button = QPushButton("Move Down")
        self.move_down_button.clicked.connect(lambda: self.move_selected_sequence_row(1))
        sequence_button_layout.addWidget(self.add_row_button)
        sequence_button_layout.addWidget(self.remove_row_button)
        sequence_button_layout.addWidget(self.move_up_button)
        sequence_button_layout.addWidget(self.move_down_button)
        sequence_layout.addLayout(sequence_button_layout)
        sequence_box.setLayout(sequence_layout)
        root_layout.addWidget(sequence_box)

        settings_box = QGroupBox("Cycle Settings")
        settings_layout = QGridLayout()
        settings_layout.setHorizontalSpacing(10)
        settings_layout.setVerticalSpacing(6)
        self.baseline_voltage_input = QDoubleSpinBox()
        self.baseline_voltage_input.setRange(-210.0, 210.0)
        self.baseline_voltage_input.setDecimals(4)
        self.baseline_voltage_input.setValue(0.0)
        self.baseline_dwell_input = QDoubleSpinBox()
        self.baseline_dwell_input.setRange(0.0, 100000.0)
        self.baseline_dwell_input.setDecimals(3)
        self.baseline_dwell_input.setValue(20.0)
        self.source_delay_input = QDoubleSpinBox()
        self.source_delay_input.setRange(0.0, 100000.0)
        self.source_delay_input.setDecimals(3)
        self.source_delay_input.setValue(0.0)
        self.cycles_input = QSpinBox()
        self.cycles_input.setRange(1, 1000000)
        self.cycles_input.setValue(100)
        self.continuous_checkbox = QCheckBox("Continuous mode")
        self.continuous_checkbox.toggled.connect(self._on_continuous_toggled)
        self.nplc_combo = QComboBox()
        self.nplc_combo.addItems(["0.01", "0.1", "1", "10"])
        self.compliance_input = QLineEdit("1e-3")
        self.current_range_combo = QComboBox()
        self.current_range_combo.setEditable(True)
        self.current_range_combo.addItems([
            "Auto-range",
            "1e-6",
            "10e-6",
            "100e-6",
            "1e-3",
            "10e-3",
            "100e-3",
            "1",
        ])
        self.current_range_combo.setCurrentText("Auto-range")
        settings_layout.addWidget(QLabel("Baseline voltage (V):"), 0, 0)
        settings_layout.addWidget(self.baseline_voltage_input, 0, 1)
        settings_layout.addWidget(QLabel("Baseline dwell (ms):"), 0, 2)
        settings_layout.addWidget(self.baseline_dwell_input, 0, 3)
        settings_layout.addWidget(QLabel("Additional source delay (ms):"), 0, 4)
        settings_layout.addWidget(self.source_delay_input, 0, 5)
        settings_layout.addWidget(QLabel("Cycles:"), 1, 0)
        settings_layout.addWidget(self.cycles_input, 1, 1)
        settings_layout.addWidget(self.continuous_checkbox, 1, 2, 1, 2)
        settings_layout.addWidget(QLabel("NPLC:"), 1, 4)
        settings_layout.addWidget(self.nplc_combo, 1, 5)
        settings_layout.addWidget(QLabel("Compliance current (A):"), 2, 0)
        settings_layout.addWidget(self.compliance_input, 2, 1)
        settings_layout.addWidget(QLabel("Current range (A):"), 2, 2)
        settings_layout.addWidget(self.current_range_combo, 2, 3)
        self.point_limit_label = QLabel("")
        self.point_limit_label.setStyleSheet("font-weight: 600;")
        self.cycle_points_label = QLabel("")
        self.batch_summary_label = QLabel("")
        self.point_period_label = QLabel("")
        settings_layout.addWidget(self.point_limit_label, 3, 0, 1, 2)
        settings_layout.addWidget(self.cycle_points_label, 3, 2, 1, 2)
        settings_layout.addWidget(self.batch_summary_label, 3, 4, 1, 2)
        settings_layout.addWidget(self.point_period_label, 4, 0, 1, 3)
        settings_box.setLayout(settings_layout)
        root_layout.addWidget(settings_box)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Measurement")
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_measurement)
        self.status_chip = QLabel("Idle")
        self.status_chip.setAlignment(Qt.AlignCenter)
        self.status_chip.setStyleSheet("background-color: lightgray; padding: 4px;")
        self.stop_button = QPushButton("Stop After Cycle")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_measurement)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.status_chip)
        button_layout.addWidget(self.stop_button)
        root_layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Progress: 0/0")
        self.eta_label = QLabel("ETA: --")
        root_layout.addWidget(self.progress_bar)
        root_layout.addWidget(self.progress_label)
        root_layout.addWidget(self.eta_label)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground("w")
        pg.setConfigOption("background", "w")
        self.full_trace_plot = self.plot_widget.addPlot(title="Current / Voltage vs Point")
        self.full_trace_plot.showGrid(x=True, y=True, alpha=0.12)
        self.full_trace_plot.setLabel("left", "Current", units="A")
        self.full_trace_plot.setLabel("bottom", "Point Index")
        self.full_trace_plot.getAxis("left").enableAutoSIPrefix(True)
        self.full_trace_voltage_view = pg.ViewBox()
        self.full_trace_plot.showAxis("right")
        self.full_trace_plot.getAxis("right").setLabel("Voltage", units="V")
        self.full_trace_plot.getAxis("right").enableAutoSIPrefix(True)
        self.full_trace_plot.scene().addItem(self.full_trace_voltage_view)
        self.full_trace_plot.getAxis("right").linkToView(self.full_trace_voltage_view)
        self.full_trace_voltage_view.setXLink(self.full_trace_plot)
        self.full_trace_plot.getViewBox().sigResized.connect(self._sync_voltage_overlay)
        self._sync_voltage_overlay()
        self.cycle_summary_plot = self.plot_widget.addPlot(title="Per-Cycle Segment Summary")
        self.cycle_summary_plot.showGrid(x=True, y=True, alpha=0.12)
        self.cycle_summary_plot.setLabel("left", "Current", units="A")
        self.cycle_summary_plot.setLabel("bottom", "Cycle")
        self.cycle_summary_plot.getAxis("left").enableAutoSIPrefix(True)
        root_layout.addWidget(self.plot_widget)

        self.setLayout(root_layout)
        apply_standard_window_style(self)
        self.progress_tracker = ProgressEta(self.progress_bar, self.eta_label, self.progress_label)
        self._populate_sequence_table(self._default_sequence_rows())
        self._connect_summary_inputs()
        self.refresh_devices()
        self._on_continuous_toggled(self.continuous_checkbox.isChecked())
        self._update_cycle_plan_summary()

    def _connect_summary_inputs(self):
        self.baseline_voltage_input.valueChanged.connect(self._update_cycle_plan_summary)
        self.baseline_dwell_input.valueChanged.connect(self._update_cycle_plan_summary)
        self.source_delay_input.valueChanged.connect(self._update_cycle_plan_summary)
        self.cycles_input.valueChanged.connect(self._update_cycle_plan_summary)
        self.nplc_combo.currentTextChanged.connect(self._update_cycle_plan_summary)
        self.current_range_combo.currentTextChanged.connect(self._update_cycle_plan_summary)

    @staticmethod
    def _default_sequence_rows():
        return [
            {"enabled": True, "label": "READ", "role": "READ", "voltage": 1.0, "dwell_ms": 20.0},
            {"enabled": True, "label": "SET", "role": "SET", "voltage": -1.0, "dwell_ms": 20.0},
            {"enabled": True, "label": "READ", "role": "READ", "voltage": 1.0, "dwell_ms": 20.0},
            {"enabled": True, "label": "RESET", "role": "RESET", "voltage": -1.0, "dwell_ms": 20.0},
        ]

    def _sync_voltage_overlay(self):
        if not hasattr(self, "full_trace_voltage_view"):
            return
        self.full_trace_voltage_view.setGeometry(self.full_trace_plot.getViewBox().sceneBoundingRect())
        self.full_trace_voltage_view.linkedViewChanged(
            self.full_trace_plot.getViewBox(),
            self.full_trace_voltage_view.XAxis,
        )

    def _filtered_2400_devices(self):
        devices = []
        for entry in keithley.get_devices_list():
            resource, model, class_name, _idn = entry
            model_text = str(model).upper()
            class_text = str(class_name)
            if "2400" in model_text or class_text == "Keithley2400":
                devices.append(" - ".join(entry))
        return devices

    def refresh_devices(self):
        previous = self.device_address_input.currentText()
        self.refresh_button.setEnabled(False)
        self.refresh_button.setText("Refreshing...")
        self.refresh_status_label.setText("Scanning GPIB...")
        QApplication.processEvents()
        try:
            devices = self._filtered_2400_devices()
            self.device_address_input.blockSignals(True)
            self.device_address_input.clear()
            self.device_address_input.addItems(devices)
            if previous:
                index = self.device_address_input.findText(previous)
                if index >= 0:
                    self.device_address_input.setCurrentIndex(index)
            self.device_address_input.blockSignals(False)
            self.refresh_status_label.setText(f"Found {len(devices)} device(s)")
        except Exception as exc:
            self.refresh_status_label.setText(f"Refresh failed: {exc}")
            raise
        finally:
            self.refresh_button.setText("Refresh GPIB")
            self.refresh_button.setEnabled(True)

    def _on_continuous_toggled(self, enabled):
        self.cycles_input.setEnabled(not bool(enabled))
        self._update_cycle_plan_summary()

    def _make_enabled_checkbox(self, checked=True):
        checkbox = QCheckBox()
        checkbox.setChecked(bool(checked))
        checkbox.stateChanged.connect(self._update_cycle_plan_summary)
        return checkbox

    def _make_role_combo(self, role):
        combo = QComboBox()
        combo.addItems(self.ROLE_OPTIONS)
        role = str(role or "PULSE").strip().upper()
        index = combo.findText(role)
        if index < 0:
            combo.addItem(role)
            index = combo.findText(role)
        combo.setCurrentIndex(index)
        combo.currentTextChanged.connect(self._update_cycle_plan_summary)
        return combo

    def _make_voltage_spin(self, value):
        spin = QDoubleSpinBox()
        spin.setRange(-210.0, 210.0)
        spin.setDecimals(4)
        spin.setValue(float(value))
        spin.valueChanged.connect(self._update_cycle_plan_summary)
        return spin

    def _make_dwell_spin(self, value):
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 100000.0)
        spin.setDecimals(3)
        spin.setValue(max(0.0, float(value)))
        spin.valueChanged.connect(self._update_cycle_plan_summary)
        return spin

    def _populate_sequence_table(self, rows, selected_row=None):
        rows = list(rows or [])
        self.sequence_table.blockSignals(True)
        self.sequence_table.setRowCount(0)
        for row_index, row in enumerate(rows):
            self.sequence_table.insertRow(row_index)
            self.sequence_table.setCellWidget(row_index, 0, self._make_enabled_checkbox(row.get("enabled", True)))
            label_item = QTableWidgetItem(str(row.get("label", "") or ""))
            self.sequence_table.setItem(row_index, 1, label_item)
            self.sequence_table.setCellWidget(row_index, 2, self._make_role_combo(row.get("role", "PULSE")))
            self.sequence_table.setCellWidget(row_index, 3, self._make_voltage_spin(row.get("voltage", 0.0)))
            self.sequence_table.setCellWidget(row_index, 4, self._make_dwell_spin(row.get("dwell_ms", 20.0)))
        self.sequence_table.blockSignals(False)
        if rows:
            row_to_select = 0 if selected_row is None else max(0, min(int(selected_row), len(rows) - 1))
            self.sequence_table.selectRow(row_to_select)
        self._update_cycle_plan_summary()

    def _snapshot_sequence_rows(self):
        rows = []
        for row_index in range(self.sequence_table.rowCount()):
            enabled_widget = self.sequence_table.cellWidget(row_index, 0)
            role_widget = self.sequence_table.cellWidget(row_index, 2)
            voltage_widget = self.sequence_table.cellWidget(row_index, 3)
            dwell_widget = self.sequence_table.cellWidget(row_index, 4)
            label_item = self.sequence_table.item(row_index, 1)
            rows.append({
                "enabled": bool(enabled_widget.isChecked()) if enabled_widget is not None else True,
                "label": label_item.text().strip() if label_item is not None else "",
                "role": role_widget.currentText().strip() if role_widget is not None else "PULSE",
                "voltage": float(voltage_widget.value()) if voltage_widget is not None else 0.0,
                "dwell_ms": float(dwell_widget.value()) if dwell_widget is not None else 0.0,
            })
        return rows

    def add_sequence_row(self):
        rows = self._snapshot_sequence_rows()
        rows.append({"enabled": True, "label": "PULSE", "role": "PULSE", "voltage": 0.0, "dwell_ms": 20.0})
        self._populate_sequence_table(rows, selected_row=len(rows) - 1)

    def remove_selected_sequence_row(self):
        rows = self._snapshot_sequence_rows()
        if not rows:
            return
        selected = self.sequence_table.currentRow()
        if selected < 0:
            selected = len(rows) - 1
        rows.pop(selected)
        self._populate_sequence_table(rows, selected_row=max(0, selected - 1))

    def move_selected_sequence_row(self, delta):
        rows = self._snapshot_sequence_rows()
        if len(rows) < 2:
            return
        selected = self.sequence_table.currentRow()
        if selected < 0:
            return
        target = selected + int(delta)
        if target < 0 or target >= len(rows):
            return
        rows[selected], rows[target] = rows[target], rows[selected]
        self._populate_sequence_table(rows, selected_row=target)

    def _on_sequence_item_changed(self, _item):
        self._update_cycle_plan_summary()

    def _current_range_setting(self):
        text = self.current_range_combo.currentText().strip()
        if not text or text.lower() == "auto-range":
            return None, True
        return parse_numeric_text(text, "Current range"), False

    def _get_sample_name(self):
        self.sample_name = self.sample_name_input.text().strip() or self.sample_name.strip() or "sample"
        return self.sample_name

    def _get_output_directories(self, leaf_dir=None):
        if not self.folder:
            raise RuntimeError("No output folder selected.")
        date_dir = ensure_directory(op.join(self.folder, self.date), "date")
        sample_name = self._get_sample_name()
        sample_dir = ensure_directory(op.join(date_dir, sample_name), f"sample '{sample_name}'")
        if leaf_dir is None:
            return sample_name, sample_dir
        leaf_path = ensure_directory(op.join(sample_dir, leaf_dir), leaf_dir)
        return sample_name, sample_dir, leaf_path

    def _measurement_file_stem(self):
        sample_name = self._get_sample_name()
        mode_tag = "continuous" if self.continuous_mode else f"{self.cycles_requested}cycles"
        return f"PULSES_{sample_name}_mem_{mode_tag}_{self.start_time}"

    def _build_output_paths(self):
        sample_name, _, data_dir = self._get_output_directories("data")
        _, _, plot_dir = self._get_output_directories("plots")
        stem = self._measurement_file_stem()
        return {
            "sample_name": sample_name,
            "data_file": op.join(data_dir, f"{stem}.data"),
            "metadata_file": op.join(data_dir, f"{stem}.meta.json"),
            "plot_trace": op.join(plot_dir, f"{stem}_trace.png"),
            "plot_summary": op.join(plot_dir, f"{stem}_summary.png"),
        }

    def _update_cycle_plan_summary(self):
        try:
            rows = self._snapshot_sequence_rows()
            self.planned_cycle = keithley.plan_pulse_cycle_batches(
                sequence_rows=rows,
                baseline_voltage=self.baseline_voltage_input.value(),
                baseline_dwell_ms=self.baseline_dwell_input.value(),
                source_delay_ms=self.source_delay_input.value(),
                nplc=float(self.nplc_combo.currentText()),
            )
        except Exception as exc:
            self.planned_cycle = None
            self.point_limit_label.setText(f"Cycle plan invalid: {exc}")
            self.point_limit_label.setStyleSheet("font-weight: 600; color: #b91c1c;")
            self.cycle_points_label.setText("Cycle points: --")
            self.batch_summary_label.setText("Batches: --")
            self.point_period_label.setText("Effective point period: --")
            if self.worker_thread is None:
                self.start_button.setEnabled(False)
            return

        limit = int(self.planned_cycle["memory_limit"])
        batch_points = [int(batch["expected_points"]) for batch in self.planned_cycle["batches"]]
        self.point_limit_label.setText(f"Keithley source-memory limit: {limit} pts per batch")
        self.point_limit_label.setStyleSheet("font-weight: 600; color: #0f172a;")
        self.cycle_points_label.setText(f"Cycle points: {int(self.planned_cycle['total_points'])}")
        self.batch_summary_label.setText(
            f"Batches: {len(batch_points)} | "
            + ", ".join(f"B{index + 1}={count}" for index, count in enumerate(batch_points))
        )
        self.point_period_label.setText(
            f"Effective point period: {float(self.planned_cycle['point_period_ms']):.3f} ms"
        )
        if self.worker_thread is None:
            self.start_button.setEnabled(True)

    def _capture_measurement_metadata(self):
        return {
            "measurement_type": "PULSES",
            "start_time": self.start_time,
            "date_folder": self.date,
            "base_folder": self.folder,
            "sample_name_at_start": self.sample_name,
            "device": build_device_metadata(keithley, self.device_address, None),
            "parameters": {
                "engine": "Source Memory",
                "baseline_voltage_v": self.baseline_voltage,
                "baseline_dwell_ms": self.baseline_dwell_ms,
                "source_delay_ms": self.source_delay_ms,
                "effective_point_period_ms": self.point_period_ms,
                "nplc": self.nplc,
                "cycles_requested": self.cycles_requested,
                "continuous_mode": self.continuous_mode,
                "compliance_current_a": self.compliance_current,
                "current_range_setting": self.current_range_combo.currentText(),
                "sequence_rows": list(self.sequence_rows),
                "planned_segments": [
                    {
                        "segment_index": int(item["segment_index"]),
                        "segment_label": str(item["segment_label"]),
                        "segment_role": str(item["segment_role"]),
                        "is_baseline": bool(item["is_baseline"]),
                        "voltage_v": float(item["voltage"]),
                        "dwell_ms": float(item["dwell_ms"]),
                        "point_count": int(item["point_count"]),
                    }
                    for item in self.planned_cycle["segments"]
                ],
            },
            "progress": {
                "cycle_points": int(self.planned_cycle["total_points"]),
                "planned_batches_per_cycle": len(self.planned_cycle["batches"]),
                "estimated_total_steps": self.total_steps,
                "completed_cycles": self.completed_cycles,
            },
        }

    def _build_metadata_payload(self):
        metadata = dict(self.measurement_metadata) if self.measurement_metadata else self._capture_measurement_metadata()
        metadata.update({
            "sample_name_at_save": self._get_sample_name(),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "measurement_count": len(self.measurements),
            "data_columns": list(self.get_dataframe().columns),
            "data_file": self.output_paths["data_file"],
            "metadata_file": self.output_paths["metadata_file"],
        })
        metadata["progress"] = dict(metadata.get("progress", {}))
        metadata["progress"].update({
            "completed_steps": self.completed_steps,
            "completed_cycles": self.completed_cycles,
        })
        return metadata

    def _write_outputs(self, include_plots=False):
        if not self.measurements:
            return
        df = self.get_dataframe()
        df.to_csv(self.output_paths["data_file"], index=False)
        write_json_file(self.output_paths["metadata_file"], self._build_metadata_payload())
        if include_plots:
            self.make_plots(df)

    def start_measurement(self):
        if self.worker_thread is not None:
            return
        if not self.folder:
            QMessageBox.critical(self, "Error", "No output folder selected.")
            return
        if self.planned_cycle is None:
            QMessageBox.critical(self, "Error", "The pulse cycle plan is invalid.")
            return

        try:
            self.compliance_current = parse_numeric_text(self.compliance_input.text(), "Compliance current")
            self.current_range, self.current_autorange = self._current_range_setting()
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        self.device_address = self.device_address_input.currentText().strip()
        if not self.device_address:
            QMessageBox.critical(self, "Error", "No Keithley 2400 selected.")
            return

        self.sample_name = self._get_sample_name()
        self.sequence_rows = self._snapshot_sequence_rows()
        self.baseline_voltage = float(self.baseline_voltage_input.value())
        self.baseline_dwell_ms = float(self.baseline_dwell_input.value())
        self.source_delay_ms = float(self.source_delay_input.value())
        self.nplc = float(self.nplc_combo.currentText())
        self.continuous_mode = bool(self.continuous_checkbox.isChecked())
        self.cycles_requested = int(self.cycles_input.value())
        self.point_period_ms = float(self.planned_cycle["point_period_ms"])
        self.measurements = []
        self.completed_steps = 0
        self.completed_cycles = 0
        self.start_time = datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S")
        self.output_paths = self._build_output_paths()
        cycle_points = int(self.planned_cycle["total_points"])
        self.total_steps = cycle_points if self.continuous_mode else cycle_points * self.cycles_requested
        self.measurement_metadata = self._capture_measurement_metadata()
        self.full_trace_plot.clear()
        self.full_trace_voltage_view.clear()
        self.cycle_summary_plot.clear()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_chip.setText("Starting...")
        self.save_requested = True

        if self.continuous_mode:
            self.progress_bar.setRange(0, 0)
            self.progress_label.setText(f"Progress: continuous | cycle points={cycle_points}")
            self.eta_label.setText("ETA: --")
        else:
            self.progress_tracker.start(self.total_steps)

        worker_settings = {
            "device_address": self.device_address,
            "planned_cycle": self.planned_cycle,
            "continuous": self.continuous_mode,
            "cycles_requested": self.cycles_requested,
            "nplc": self.nplc,
            "source_delay_ms": self.source_delay_ms,
            "point_period_ms": self.point_period_ms,
            "baseline_voltage": self.baseline_voltage,
            "compliance_current": self.compliance_current,
            "current_range": self.current_range,
            "current_autorange": self.current_autorange,
        }

        self.worker_thread = QThread(self)
        self.worker = PulsesWorker(worker_settings)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.batch_completed.connect(self._on_batch_completed)
        self.worker.cycle_completed.connect(self._on_cycle_completed)
        self.worker.status_changed.connect(self.status_chip.setText)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._clear_worker_refs)
        self.worker_thread.start()

    def stop_measurement(self, save=True):
        self.save_requested = bool(save)
        if self.worker is None:
            return
        self.worker.request_stop()
        self.status_chip.setText("Stopping after current cycle...")
        self.stop_button.setEnabled(False)

    def _on_batch_completed(self, rows, batch_info):
        self.measurements.extend(list(rows))
        self.completed_steps += len(rows)
        if self.continuous_mode:
            self.progress_label.setText(
                f"Progress: continuous | cycles={self.completed_cycles} | points={self.completed_steps}"
            )
        else:
            self.progress_tracker.step(
                self.completed_steps,
                extra_text=(
                    f"C{batch_info['cycle_index']} "
                    f"B{batch_info['batch_position']}/{batch_info['batch_count']}"
                ),
            )
        self.update_plots()

    def _on_cycle_completed(self, cycle_index):
        self.completed_cycles = max(self.completed_cycles, int(cycle_index))
        self.status_chip.setText(f"Cycle {cycle_index} complete")
        if self.continuous_mode:
            self.progress_label.setText(
                f"Progress: continuous | cycles={self.completed_cycles} | points={self.completed_steps}"
            )
        try:
            self._write_outputs(include_plots=False)
        except Exception as exc:
            QMessageBox.critical(self, "Checkpoint Save Error", str(exc))

    def _on_worker_finished(self, result):
        status = str(result.get("status", "error"))
        message = str(result.get("message", "") or "")
        self.stop_button.setEnabled(False)
        save_errors = []
        if self.save_requested and self.measurements:
            try:
                self._write_outputs(include_plots=True)
            except Exception as exc:
                save_errors.append(str(exc))
        if status == "completed":
            self.status_chip.setText("Completed")
        elif status == "stopped":
            self.status_chip.setText("Stopped")
        else:
            self.status_chip.setText("Error")
        if message:
            save_errors.insert(0, message)
        if save_errors:
            QMessageBox.critical(self, "Measurement Error", "\n\n".join(save_errors))

    def _clear_worker_refs(self):
        self.worker = None
        self.worker_thread = None
        self._update_cycle_plan_summary()

    def get_dataframe(self):
        if not self.measurements:
            return pd.DataFrame(columns=self.DATA_COLUMNS)
        df = pd.DataFrame(self.measurements)
        return df.reindex(columns=self.DATA_COLUMNS)

    def _summary_dataframe(self, df):
        if df.empty:
            return df
        active = df[df["IsBaseline"] == False].copy()
        if active.empty:
            return active
        grouped = active.groupby(["CycleIndex", "SegmentIndex"], as_index=False).tail(1)
        return grouped.sort_values(["SegmentIndex", "CycleIndex"])

    def update_plots(self):
        df = self.get_dataframe()
        self.full_trace_plot.clear()
        self.full_trace_voltage_view.clear()
        self.cycle_summary_plot.clear()
        if df.empty:
            return

        x_values = np.arange(1, len(df) + 1, dtype=float)
        currents = df["Current"].to_numpy(dtype=float, copy=True)
        voltages = df["Voltage"].to_numpy(dtype=float, copy=True)
        baseline_mask = df["IsBaseline"].to_numpy(dtype=bool, copy=True)
        active_mask = ~baseline_mask
        if np.any(baseline_mask):
            self.full_trace_plot.plot(
                x_values[baseline_mask],
                currents[baseline_mask],
                pen=pg.mkPen(color=(148, 163, 184, 180), width=1),
                symbol="o",
                symbolSize=3,
                symbolBrush=(148, 163, 184, 180),
            )
        if np.any(active_mask):
            self.full_trace_plot.plot(
                x_values[active_mask],
                currents[active_mask],
                pen=pg.mkPen(color=(37, 99, 235, 220), width=2),
                symbol="o",
                symbolSize=4,
                symbolBrush=(37, 99, 235, 220),
            )
        self.full_trace_voltage_view.addItem(
            pg.PlotCurveItem(
                x_values,
                voltages,
                pen=pg.mkPen(color=(220, 38, 38, 180), width=2),
            )
        )
        self.full_trace_voltage_view.addItem(
            pg.ScatterPlotItem(
                x=x_values,
                y=voltages,
                pen=pg.mkPen(color=(220, 38, 38, 150)),
                brush=pg.mkBrush(220, 38, 38, 150),
                size=3,
            )
        )
        self._sync_voltage_overlay()

        summary_df = self._summary_dataframe(df)
        if summary_df.empty:
            return
        palette = [
            (37, 99, 235),
            (220, 38, 38),
            (22, 163, 74),
            (245, 158, 11),
            (168, 85, 247),
            (14, 165, 233),
        ]
        unique_segments = summary_df[["SegmentIndex", "SegmentLabel", "SegmentRole"]].drop_duplicates()
        for color_index, segment_row in enumerate(unique_segments.itertuples(index=False), start=0):
            mask = (
                (summary_df["SegmentIndex"] == int(segment_row.SegmentIndex))
                & (summary_df["SegmentLabel"] == str(segment_row.SegmentLabel))
            )
            segment_df = summary_df.loc[mask].sort_values("CycleIndex")
            color = palette[color_index % len(palette)]
            label = f"{int(segment_row.SegmentIndex)}: {segment_row.SegmentLabel} ({segment_row.SegmentRole})"
            self.cycle_summary_plot.plot(
                segment_df["CycleIndex"].to_numpy(dtype=float, copy=True),
                segment_df["Current"].to_numpy(dtype=float, copy=True),
                pen=pg.mkPen(color=(*color, 220), width=2),
                symbol="o",
                symbolSize=4,
                symbolBrush=(*color, 220),
                name=label,
            )

    def make_plots(self, df=None):
        if df is None:
            df = self.get_dataframe()
        if df.empty:
            return

        x_values = np.arange(1, len(df) + 1, dtype=float)
        baseline_mask = df["IsBaseline"].to_numpy(dtype=bool, copy=True)
        active_mask = ~baseline_mask
        fig1 = plt.figure(figsize=(12, 6), dpi=300)
        ax_current = fig1.add_subplot(111)
        ax_voltage = ax_current.twinx()
        if np.any(baseline_mask):
            ax_current.plot(
                x_values[baseline_mask],
                df.loc[baseline_mask, "Current"],
                "o-",
                markersize=2,
                linewidth=1.0,
                color="lightgray",
                alpha=0.85,
                label="Baseline",
            )
        if np.any(active_mask):
            ax_current.plot(
                x_values[active_mask],
                df.loc[active_mask, "Current"],
                "o-",
                markersize=2.5,
                linewidth=1.2,
                color="tab:blue",
                alpha=0.8,
                label="Pulse",
            )
        ax_voltage.plot(
            x_values,
            df["Voltage"],
            "-",
            linewidth=1.2,
            color="tab:red",
            alpha=0.7,
            label="Voltage",
        )
        ax_current.set_xlabel("Point Index")
        ax_current.set_ylabel("Current (A)")
        ax_voltage.set_ylabel("Voltage (V)")
        lines_current, labels_current = ax_current.get_legend_handles_labels()
        lines_voltage, labels_voltage = ax_voltage.get_legend_handles_labels()
        ax_current.legend(lines_current + lines_voltage, labels_current + labels_voltage)
        plt.savefig(self.output_paths["plot_trace"], dpi=300)
        fig1.clf()

        summary_df = self._summary_dataframe(df)
        fig2 = plt.figure(figsize=(12, 6), dpi=300)
        palette = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:cyan"]
        if not summary_df.empty:
            unique_segments = summary_df[["SegmentIndex", "SegmentLabel", "SegmentRole"]].drop_duplicates()
            for color_index, segment_row in enumerate(unique_segments.itertuples(index=False), start=0):
                mask = (
                    (summary_df["SegmentIndex"] == int(segment_row.SegmentIndex))
                    & (summary_df["SegmentLabel"] == str(segment_row.SegmentLabel))
                )
                segment_df = summary_df.loc[mask].sort_values("CycleIndex")
                plt.plot(
                    segment_df["CycleIndex"],
                    segment_df["Current"],
                    "o-",
                    markersize=3,
                    linewidth=1.2,
                    color=palette[color_index % len(palette)],
                    alpha=0.8,
                    label=f"{int(segment_row.SegmentIndex)}: {segment_row.SegmentLabel} ({segment_row.SegmentRole})",
                )
        plt.xlabel("Cycle")
        plt.ylabel("Current (A)")
        plt.legend()
        plt.savefig(self.output_paths["plot_summary"], dpi=300)
        fig2.clf()

        matplotlib.pyplot.close(fig1)
        matplotlib.pyplot.close(fig2)
        plt.close("all")
        gc.collect()

    def closeEvent(self, event):
        if self.worker_thread is not None:
            self.stop_measurement(save=False)
            event.ignore()
            return
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PulsesRegime()
    window.show()
    sys.exit(app.exec_())
