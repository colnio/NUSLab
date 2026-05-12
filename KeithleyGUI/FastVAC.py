import sys
import json
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
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
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


class FastVACWorker(QObject):
    batch_completed = pyqtSignal(object, object)
    cycle_completed = pyqtSignal(int)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, settings):
        super().__init__()
        self.settings = dict(settings)
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        device = None
        adapter = None
        try:
            self.status_changed.emit("Opening Keithley 2400...")
            device = keithley.get_device(self.settings["device_address"], nplc=self.settings["nplc"])
            if not isinstance(device, keithley.Keithley2400):
                raise RuntimeError("Selected device is not a Keithley 2400.")
            adapter = keithley.Keithley2400FastSweepAdapter(device)
            batches = list(self.settings["planned_batches"])
            last_cycle = None
            for batch_position, batch in enumerate(batches):
                if self._stop_requested:
                    break
                polarity_key = str(batch["polarity"]).strip().lower()
                polarity_cfg = self.settings["polarity_settings"][polarity_key]
                relaxed = not bool(polarity_cfg["enabled"])
                compliance_current = (
                    float(polarity_cfg["current"])
                    if polarity_cfg["enabled"]
                    else keithley.FAST_VAC_NO_COMPLIANCE_CURRENT
                )
                self.status_changed.emit(
                    f"Cycle {batch['cycle_index']} | {polarity_key} | "
                    f"batch {batch['batch_index']} of subcycle {batch['subcycle_index']}"
                )
                rows = adapter.execute_batch(
                    engine=self.settings["engine"],
                    points=batch["points"],
                    compliance_current=compliance_current,
                    nplc=self.settings["nplc"],
                    source_delay_s=self.settings["source_delay_s"],
                    current_range=self.settings["current_range"],
                    current_autorange=self.settings["current_autorange"],
                    relaxed_current_range=relaxed,
                    holdoff_enabled=self.settings["holdoff_enabled"],
                    holdoff_delay_s=self.settings["holdoff_delay_s"],
                )
                annotated_rows = []
                for step_index, row in enumerate(rows, start=1):
                    item = dict(row)
                    item.update({
                        "CycleIndex": int(batch["cycle_index"]),
                        "SubcycleIndex": int(batch["subcycle_index"]),
                        "BatchIndex": int(batch["batch_index"]),
                        "StepIndex": int(step_index),
                        "Polarity": "positive" if polarity_key.startswith("p") else "negative",
                        "Engine": str(batch["engine"]),
                    })
                    annotated_rows.append(item)
                self.batch_completed.emit(annotated_rows, dict(batch))
                if last_cycle is None:
                    last_cycle = int(batch["cycle_index"])
                next_cycle = int(batches[batch_position + 1]["cycle_index"]) if batch_position + 1 < len(batches) else None
                if next_cycle != last_cycle:
                    self.cycle_completed.emit(last_cycle)
                    last_cycle = next_cycle

            if adapter is not None:
                adapter.abort_to_zero(clear_only=True)
            self.finished.emit({
                "status": "stopped" if self._stop_requested else "completed",
                "message": "",
            })
        except Exception as exc:
            if adapter is not None:
                try:
                    adapter.abort_to_zero(clear_only=True)
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


class FastVAC(QWidget):
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
        self.planned_batches = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Keithley 2400 Fast IV")
        self.resize(1450, 920)
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

        sweep_box = QGroupBox("Sweep")
        sweep_layout = QGridLayout()
        sweep_layout.setHorizontalSpacing(10)
        sweep_layout.setVerticalSpacing(6)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["List", "Source Memory"])
        self.engine_combo.currentTextChanged.connect(self._update_engine_controls)
        self.voltage_min_input = QDoubleSpinBox()
        self.voltage_min_input.setRange(-210.0, 210.0)
        self.voltage_min_input.setDecimals(4)
        self.voltage_min_input.setValue(-1.0)
        self.voltage_max_input = QDoubleSpinBox()
        self.voltage_max_input.setRange(-210.0, 210.0)
        self.voltage_max_input.setDecimals(4)
        self.voltage_max_input.setValue(1.0)
        self.voltage_step_input = QDoubleSpinBox()
        self.voltage_step_input.setRange(0.0001, 210.0)
        self.voltage_step_input.setDecimals(4)
        self.voltage_step_input.setValue(0.01)
        self.cycles_input = QSpinBox()
        self.cycles_input.setRange(1, 100000)
        self.cycles_input.setValue(1)
        self.initial_direction_combo = QComboBox()
        self.initial_direction_combo.addItems(["+ first", "- first"])
        self.nplc_combo = QComboBox()
        self.nplc_combo.addItems(["0.01", "0.1", "1", "10"])
        self.source_delay_input = QDoubleSpinBox()
        self.source_delay_input.setRange(0.0, 100000.0)
        self.source_delay_input.setDecimals(3)
        self.source_delay_input.setValue(0.0)
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
        sweep_layout.addWidget(QLabel("Engine:"), 0, 0)
        sweep_layout.addWidget(self.engine_combo, 0, 1)
        sweep_layout.addWidget(QLabel("Cycles:"), 0, 2)
        sweep_layout.addWidget(self.cycles_input, 0, 3)
        sweep_layout.addWidget(QLabel("Initial direction:"), 0, 4)
        sweep_layout.addWidget(self.initial_direction_combo, 0, 5)
        sweep_layout.addWidget(QLabel("Vmin (V):"), 1, 0)
        sweep_layout.addWidget(self.voltage_min_input, 1, 1)
        sweep_layout.addWidget(QLabel("Vmax (V):"), 1, 2)
        sweep_layout.addWidget(self.voltage_max_input, 1, 3)
        sweep_layout.addWidget(QLabel("Vstep (V):"), 1, 4)
        sweep_layout.addWidget(self.voltage_step_input, 1, 5)
        sweep_layout.addWidget(QLabel("NPLC:"), 2, 0)
        sweep_layout.addWidget(self.nplc_combo, 2, 1)
        sweep_layout.addWidget(QLabel("Source delay (ms):"), 2, 2)
        sweep_layout.addWidget(self.source_delay_input, 2, 3)
        sweep_layout.addWidget(QLabel("Current range (A):"), 2, 4)
        sweep_layout.addWidget(self.current_range_combo, 2, 5)
        self.point_limit_label = QLabel("")
        self.point_limit_label.setStyleSheet("font-weight: 600;")
        self.pos_points_label = QLabel("")
        self.neg_points_label = QLabel("")
        sweep_layout.addWidget(self.point_limit_label, 3, 0, 1, 2)
        sweep_layout.addWidget(self.pos_points_label, 3, 2, 1, 2)
        sweep_layout.addWidget(self.neg_points_label, 3, 4, 1, 2)
        sweep_box.setLayout(sweep_layout)
        root_layout.addWidget(sweep_box)

        polarity_box = QGroupBox("Polarity Compliance")
        polarity_layout = QGridLayout()
        polarity_layout.setHorizontalSpacing(10)
        polarity_layout.setVerticalSpacing(6)
        self.pos_comp_enable = QCheckBox("Enable compliance for V > 0")
        self.pos_comp_enable.setChecked(True)
        self.pos_comp_input = QLineEdit("1e-3")
        self.neg_comp_enable = QCheckBox("Enable compliance for V < 0")
        self.neg_comp_enable.setChecked(True)
        self.neg_comp_input = QLineEdit("1e-3")
        polarity_layout.addWidget(self.pos_comp_enable, 0, 0, 1, 2)
        polarity_layout.addWidget(QLabel("V > 0 compliance (A):"), 0, 2)
        polarity_layout.addWidget(self.pos_comp_input, 0, 3)
        polarity_layout.addWidget(self.neg_comp_enable, 1, 0, 1, 2)
        polarity_layout.addWidget(QLabel("V < 0 compliance (A):"), 1, 2)
        polarity_layout.addWidget(self.neg_comp_input, 1, 3)
        polarity_box.setLayout(polarity_layout)
        root_layout.addWidget(polarity_box)

        self.memory_box = QGroupBox("Source Memory Options")
        memory_layout = QGridLayout()
        memory_layout.setHorizontalSpacing(10)
        memory_layout.setVerticalSpacing(6)
        self.holdoff_enable = QCheckBox("Enable current range holdoff")
        self.holdoff_enable.setChecked(False)
        self.holdoff_delay_input = QDoubleSpinBox()
        self.holdoff_delay_input.setRange(0.0, 100000.0)
        self.holdoff_delay_input.setDecimals(3)
        self.holdoff_delay_input.setValue(0.0)
        memory_layout.addWidget(self.holdoff_enable, 0, 0, 1, 2)
        memory_layout.addWidget(QLabel("Holdoff delay (ms):"), 0, 2)
        memory_layout.addWidget(self.holdoff_delay_input, 0, 3)
        self.memory_box.setLayout(memory_layout)
        root_layout.addWidget(self.memory_box)

        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Measurement")
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_measurement)
        self.status_chip = QLabel("Idle")
        self.status_chip.setAlignment(Qt.AlignCenter)
        self.status_chip.setStyleSheet("background-color: lightgray; padding: 4px;")
        self.stop_button = QPushButton("Stop After Batch")
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
        self.iv_plot = self.plot_widget.addPlot(title="I(V)")
        self.iv_plot.showGrid(x=True, y=True, alpha=0.12)
        self.iv_plot.setLabel("left", "Current", units="A")
        self.iv_plot.setLabel("bottom", "Voltage", units="V")
        self.iv_plot.getAxis("left").enableAutoSIPrefix(True)
        self.iv_plot.getAxis("bottom").enableAutoSIPrefix(True)
        self.abs_iv_plot = self.plot_widget.addPlot(title="|I(V)|")
        self.abs_iv_plot.showGrid(x=True, y=True, alpha=0.12)
        self.abs_iv_plot.setLabel("left", "|Current|", units="A")
        self.abs_iv_plot.setLabel("bottom", "Voltage", units="V")
        self.abs_iv_plot.getAxis("left").enableAutoSIPrefix(True)
        self.abs_iv_plot.getAxis("bottom").enableAutoSIPrefix(True)
        self.abs_iv_plot.setLogMode(False, True)
        root_layout.addWidget(self.plot_widget)

        self.setLayout(root_layout)
        apply_standard_window_style(self)
        self.progress_tracker = ProgressEta(self.progress_bar, self.eta_label, self.progress_label)
        self._point_limit_valid = True
        self.voltage_min_input.valueChanged.connect(self._update_point_limit_summary)
        self.voltage_max_input.valueChanged.connect(self._update_point_limit_summary)
        self.voltage_step_input.valueChanged.connect(self._update_point_limit_summary)
        self.engine_combo.currentTextChanged.connect(self._update_point_limit_summary)
        self.refresh_devices()
        self._update_engine_controls()
        self._update_point_limit_summary()

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
        engine_tag = "mem" if self.engine_combo.currentText().strip().lower().startswith("source") else "list"
        return (
            f"FastVAC_{sample_name}_{engine_tag}_"
            f"{self.voltage_min_input.value():.4g}V_{self.voltage_max_input.value():.4g}V_"
            f"{self.cycles_input.value()}cycles_{self.start_time}"
        )

    def _build_output_paths(self):
        sample_name, _, data_dir = self._get_output_directories("data")
        _, _, plot_dir = self._get_output_directories("plots")
        stem = self._measurement_file_stem()
        return {
            "sample_name": sample_name,
            "data_file": op.join(data_dir, f"{stem}.data"),
            "metadata_file": op.join(data_dir, f"{stem}.meta.json"),
            "plot_linear": op.join(plot_dir, f"{stem}.png"),
            "plot_log": op.join(plot_dir, f"{stem}_logscaleY.png"),
        }

    def _filtered_2400_devices(self):
        devices = []
        for entry in keithley.get_devices_list():
            resource, model, class_name, idn = entry
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

    def _update_engine_controls(self):
        is_memory = self.engine_combo.currentText().strip().lower().startswith("source")
        self.memory_box.setEnabled(is_memory)
        self._update_point_limit_summary()

    def _format_point_count_label(self, prefix, count, limit, base_rgb):
        if count <= 0:
            color = "#94a3b8"
            background = "#f8fafc"
            border = "#cbd5e1"
        elif count > limit:
            color = "#b91c1c"
            background = "#fee2e2"
            border = "#ef4444"
        elif count >= max(1, int(limit * 0.85)):
            color = "#92400e"
            background = "#fef3c7"
            border = "#f59e0b"
        else:
            color = f"rgb({base_rgb[0]}, {base_rgb[1]}, {base_rgb[2]})"
            background = "rgba(241, 245, 249, 0.95)"
            border = "#cbd5e1"
        return (
            f"{prefix}: {count}/{limit} pts",
            "padding: 3px 8px; border-radius: 4px; "
            f"border: 1px solid {border}; color: {color}; background: {background}; font-weight: 600;",
        )

    def _update_point_limit_summary(self):
        try:
            validation = keithley.validate_fast_vac_subcycle_lengths(
                self.voltage_min_input.value(),
                self.voltage_max_input.value(),
                self.voltage_step_input.value(),
                self.engine_combo.currentText(),
            )
        except Exception as exc:
            self._point_limit_valid = False
            self.point_limit_label.setText(f"Point limit check failed: {exc}")
            self.point_limit_label.setStyleSheet("font-weight: 600; color: #b91c1c;")
            self.pos_points_label.setText("V > 0: --")
            self.pos_points_label.setStyleSheet("")
            self.neg_points_label.setText("V < 0: --")
            self.neg_points_label.setStyleSheet("")
            if self.worker_thread is None:
                self.start_button.setEnabled(False)
            return

        limit = int(validation["limit"])
        counts = validation["counts"]
        self._point_limit_valid = bool(validation["is_valid"])
        engine_name = self.engine_combo.currentText()
        if self._point_limit_valid:
            self.point_limit_label.setText(f"Keithley limit: {limit} pts per +/- subcycle ({engine_name})")
            self.point_limit_label.setStyleSheet("font-weight: 600; color: #0f172a;")
        else:
            offending = ", ".join(
                f"{'V > 0' if polarity == 'positive' else 'V < 0'} = {count}"
                for polarity, count in validation["over_limit"].items()
            )
            self.point_limit_label.setText(
                f"Keithley limit exceeded: max {limit} pts per +/- subcycle ({engine_name}); {offending}"
            )
            self.point_limit_label.setStyleSheet("font-weight: 600; color: #b91c1c;")

        pos_text, pos_style = self._format_point_count_label("V > 0", int(counts["positive"]), limit, (37, 99, 235))
        neg_text, neg_style = self._format_point_count_label("V < 0", int(counts["negative"]), limit, (22, 163, 74))
        self.pos_points_label.setText(pos_text)
        self.pos_points_label.setStyleSheet(pos_style)
        self.neg_points_label.setText(neg_text)
        self.neg_points_label.setStyleSheet(neg_style)

        if self.worker_thread is None:
            self.start_button.setEnabled(self._point_limit_valid)

    def _current_range_setting(self):
        text = self.current_range_combo.currentText().strip()
        if not text or text.lower() == "auto-range":
            return None, True
        return parse_numeric_text(text, "Current range"), False

    def _polarity_settings(self):
        return {
            "positive": {
                "enabled": bool(self.pos_comp_enable.isChecked()),
                "current": parse_numeric_text(self.pos_comp_input.text(), "V > 0 compliance"),
            },
            "negative": {
                "enabled": bool(self.neg_comp_enable.isChecked()),
                "current": parse_numeric_text(self.neg_comp_input.text(), "V < 0 compliance"),
            },
        }

    def _capture_measurement_metadata(self):
        return {
            "measurement_type": "FastVAC",
            "start_time": self.start_time,
            "date_folder": self.date,
            "base_folder": self.folder,
            "sample_name_at_start": self.sample_name,
            "device": build_device_metadata(keithley, self.device_address, None),
            "parameters": {
                "engine": self.engine_combo.currentText(),
                "voltage_min_v": self.voltage_min,
                "voltage_max_v": self.voltage_max,
                "voltage_step_v": self.voltage_step,
                "cycles_requested": self.cycles,
                "initial_direction": "positive_first" if self.initial_direction >= 0 else "negative_first",
                "nplc": self.nplc,
                "source_delay_ms": self.source_delay_ms,
                "current_range_setting": self.current_range_combo.currentText(),
                "memory_holdoff_enabled": self.holdoff_enable.isChecked(),
                "memory_holdoff_delay_ms": self.holdoff_delay_input.value(),
                "positive_compliance_enabled": self.polarity_settings["positive"]["enabled"],
                "positive_compliance_a": self.polarity_settings["positive"]["current"],
                "negative_compliance_enabled": self.polarity_settings["negative"]["enabled"],
                "negative_compliance_a": self.polarity_settings["negative"]["current"],
                "no_compliance_current_a": keithley.FAST_VAC_NO_COMPLIANCE_CURRENT,
            },
            "progress": {
                "estimated_total_steps": self.total_steps,
                "planned_batches": len(self.planned_batches),
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

    def _estimate_total_points(self, batches):
        return sum(int(batch.get("expected_points", 0)) for batch in batches)

    def start_measurement(self):
        if self.worker_thread is not None:
            return
        if not self.folder:
            QMessageBox.critical(self, "Error", "No output folder selected.")
            return

        self.voltage_min = self.voltage_min_input.value()
        self.voltage_max = self.voltage_max_input.value()
        self.voltage_step = self.voltage_step_input.value()
        if self.voltage_step <= 0:
            QMessageBox.critical(self, "Error", "Voltage step must be positive.")
            return
        if self.voltage_min > self.voltage_max:
            QMessageBox.critical(self, "Error", "Vmin must be <= Vmax.")
            return
        self.cycles = int(self.cycles_input.value())
        self.initial_direction = 1 if self.initial_direction_combo.currentIndex() == 0 else -1
        self.nplc = float(self.nplc_combo.currentText())
        self.source_delay_ms = float(self.source_delay_input.value())
        self.device_address = self.device_address_input.currentText().strip()
        self.sample_name = self._get_sample_name()
        if not self.device_address:
            QMessageBox.critical(self, "Error", "No Keithley 2400 selected.")
            return
        validation = keithley.validate_fast_vac_subcycle_lengths(
            self.voltage_min,
            self.voltage_max,
            self.voltage_step,
            self.engine_combo.currentText(),
        )
        if not validation["is_valid"]:
            limit = int(validation["limit"])
            detail = ", ".join(
                f"{'V > 0' if polarity == 'positive' else 'V < 0'}: {count} pts"
                for polarity, count in validation["over_limit"].items()
            )
            QMessageBox.critical(
                self,
                "Sweep Too Long",
                f"The selected {self.engine_combo.currentText()} sweep exceeds the Keithley limit "
                f"of {limit} points per +/- subcycle.\n\n{detail}",
            )
            return

        try:
            current_range, current_autorange = self._current_range_setting()
            polarity_settings = self._polarity_settings()
            planned_batches = keithley.plan_fast_vac_batches(
                self.voltage_min,
                self.voltage_max,
                self.voltage_step,
                self.cycles,
                initial_direction=self.initial_direction,
                engine=self.engine_combo.currentText(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        if not planned_batches:
            QMessageBox.critical(self, "Error", "No sweep batches were generated from the selected parameters.")
            return

        self.measurements = []
        self.completed_steps = 0
        self.completed_cycles = 0
        self.start_time = datetime.datetime.today().strftime("%Y-%m-%d %H-%M-%S")
        self.planned_batches = planned_batches
        self.total_steps = self._estimate_total_points(planned_batches)
        self.output_paths = self._build_output_paths()
        self.polarity_settings = polarity_settings
        self.current_range = current_range
        self.current_autorange = current_autorange
        self.measurement_metadata = self._capture_measurement_metadata()
        self.progress_tracker.start(self.total_steps)
        self.iv_plot.clear()
        self.abs_iv_plot.clear()
        self.status_chip.setText("Starting...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_requested = True

        worker_settings = {
            "device_address": self.device_address,
            "engine": self.engine_combo.currentText(),
            "nplc": self.nplc,
            "source_delay_s": self.source_delay_ms / 1000.0,
            "current_range": self.current_range,
            "current_autorange": self.current_autorange,
            "holdoff_enabled": bool(self.holdoff_enable.isChecked()),
            "holdoff_delay_s": float(self.holdoff_delay_input.value()) / 1000.0,
            "polarity_settings": polarity_settings,
            "planned_batches": planned_batches,
        }

        self.worker_thread = QThread(self)
        self.worker = FastVACWorker(worker_settings)
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
        self.status_chip.setText("Stopping after current batch...")
        self.stop_button.setEnabled(False)

    def _on_batch_completed(self, rows, batch):
        self.measurements.extend(list(rows))
        self.completed_steps += len(rows)
        point_text = f"C{batch['cycle_index']} {batch['polarity'][0].upper()} | {len(rows)} pts"
        self.progress_tracker.step(self.completed_steps, extra_text=point_text)
        self.update_plots()

    def _on_cycle_completed(self, cycle_index):
        self.completed_cycles = max(self.completed_cycles, int(cycle_index))
        self.status_chip.setText(f"Cycle {cycle_index} complete")
        try:
            self._write_outputs(include_plots=False)
        except Exception as exc:
            QMessageBox.critical(self, "Checkpoint Save Error", str(exc))

    def _on_worker_finished(self, result):
        status = str(result.get("status", "error"))
        message = str(result.get("message", "") or "")
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)

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

    def get_dataframe(self):
        columns = [
            "CycleIndex",
            "SubcycleIndex",
            "BatchIndex",
            "StepIndex",
            "Polarity",
            "Voltage",
            "Current",
            "Timestamp_s",
            "Status",
            "InCompliance",
            "RangeCompliance",
            "Engine",
        ]
        if not self.measurements:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(self.measurements)
        return df.reindex(columns=columns)

    def _cycle_groups_for_polarity(self, df, polarity):
        subset = df[df["Polarity"] == polarity].copy()
        if subset.empty:
            return []
        groups = []
        cycle_values = sorted(pd.unique(subset["CycleIndex"]))
        for cycle_index in cycle_values:
            cycle_df = subset[subset["CycleIndex"] == cycle_index].sort_values(
                ["SubcycleIndex", "BatchIndex", "StepIndex"]
            )
            groups.append((int(cycle_index), cycle_df))
        return groups

    @staticmethod
    def _latest_cycle_index(groups):
        if not groups:
            return None
        return groups[-1][0]

    def update_plots(self):
        df = self.get_dataframe()
        if df.empty:
            self.iv_plot.clear()
            self.abs_iv_plot.clear()
            return

        self.iv_plot.clear()
        self.abs_iv_plot.clear()
        for polarity, color in (("positive", (37, 99, 235)), ("negative", (22, 163, 74))):
            groups = self._cycle_groups_for_polarity(df, polarity)
            latest_cycle = self._latest_cycle_index(groups)
            for cycle_index, cycle_df in groups:
                voltages = cycle_df["Voltage"].to_numpy(dtype=float, copy=True)
                currents = cycle_df["Current"].to_numpy(dtype=float, copy=True)
                abs_currents = np.abs(currents)
                abs_currents[abs_currents <= 0] = np.nan
                is_latest = cycle_index == latest_cycle
                alpha = 180 if is_latest else 50
                width = 2 if is_latest else 1
                symbol_size = 4 if is_latest else 3
                self.iv_plot.plot(
                    voltages,
                    currents,
                    pen=pg.mkPen(color=(*color, alpha), width=width),
                    symbol="o",
                    symbolSize=symbol_size,
                    symbolBrush=(*color, alpha),
                )
                self.abs_iv_plot.plot(
                    voltages,
                    abs_currents,
                    pen=pg.mkPen(color=(*color, alpha), width=width),
                    symbol="o",
                    symbolSize=symbol_size,
                    symbolBrush=(*color, alpha),
                )

    def make_plots(self, df=None):
        if df is None:
            df = self.get_dataframe()
        if df.empty:
            return

        fig1 = plt.figure(figsize=(10, 6), dpi=300)
        plt.ticklabel_format(axis="y", style="scientific")
        for polarity, color, label in (
            ("positive", "tab:blue", "V > 0"),
            ("negative", "tab:green", "V < 0"),
        ):
            groups = self._cycle_groups_for_polarity(df, polarity)
            latest_cycle = self._latest_cycle_index(groups)
            first = True
            for cycle_index, cycle_df in groups:
                is_latest = cycle_index == latest_cycle
                plt.plot(
                    cycle_df["Voltage"],
                    cycle_df["Current"],
                    "o-",
                    markersize=3 if is_latest else 2,
                    alpha=0.75 if is_latest else 0.18,
                    label=label if first else None,
                    color=color,
                    linewidth=1.5 if is_latest else 1.0,
                )
                first = False
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.legend()
        plt.savefig(self.output_paths["plot_linear"], dpi=300)
        fig1.clf()

        fig2 = plt.figure(figsize=(10, 6), dpi=300)
        plt.ticklabel_format(axis="y", style="scientific")
        for polarity, color, label in (
            ("positive", "tab:blue", "V > 0"),
            ("negative", "tab:green", "V < 0"),
        ):
            groups = self._cycle_groups_for_polarity(df, polarity)
            latest_cycle = self._latest_cycle_index(groups)
            first = True
            for cycle_index, cycle_df in groups:
                abs_current = np.abs(cycle_df["Current"].to_numpy(dtype=float, copy=True))
                abs_current[abs_current <= 0] = np.nan
                is_latest = cycle_index == latest_cycle
                plt.plot(
                    cycle_df["Voltage"],
                    abs_current,
                    "o-",
                    markersize=3 if is_latest else 2,
                    alpha=0.75 if is_latest else 0.18,
                    label=label if first else None,
                    color=color,
                    linewidth=1.5 if is_latest else 1.0,
                )
                first = False
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.yscale("log")
        plt.legend()
        plt.savefig(self.output_paths["plot_log"], dpi=300)
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
    window = FastVAC()
    window.show()
    sys.exit(app.exec_())
