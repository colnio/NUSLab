import sys
import time
import datetime
import os
import os.path as op
import gc

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, QElapsedTimer, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QCheckBox,
    QProgressBar,
)

import keithley
from ui_helpers import (
    ProgressEta,
    refresh_device_combos,
    apply_standard_window_style,
    parse_numeric_text,
    ensure_directory,
    build_device_metadata,
    write_json_file,
)

class FourProbeFET(QWidget):
    def __init__(self):
        super().__init__()
        self.sd_device = None
        self.probe_device = None
        self.gate_device = None
        self.sd_adapter = None
        self.probe_adapter = None
        self.gate_adapter = None
        self.measurements = []
        self.sd_voltage_setpoint = 0.0
        self.gate_direction = 1
        self.gate_remaining_runs = 0
        self.gate_values = []
        self.current_gate_index = 0
        self.current_gate_target = 0.0
        self._eval_warning_shown = False
        self.measurement_metadata = {}
        self.gate_current_range_setting = 'Auto-range'
        self.gate_current_range_effective_a = np.nan
        self.gate_current_range_used_autorange = False
        self.gate_current_range_overflowed = False
        self.gate_current_range_should_lock_after_first_point = False
        self.gate_current_range_lock_settle_ms = 100
        self.gate_current_range_settle_until_ms = 0
        self.gate_current_range_status_text = ''

        self.timer = QTimer()
        self.timer.timeout.connect(self.perform_measurement)
        self.elapsed_timer = QElapsedTimer()

        self.date = str(datetime.date.today())
        self.folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if not self.folder:
            self.folder = os.getcwd()
        if not op.exists(op.join(self.folder, self.date)):
            os.makedirs(op.join(self.folder, self.date))
        self.start_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')

        self.init_ui()

    def _get_sample_name(self):
        return self.sample_name_input.text().strip() or 'sample'

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

    def _capture_measurement_metadata(self):
        return {
            'measurement_type': 'FourProbeFET',
            'start_time': self.start_time,
            'date_folder': self.date,
            'base_folder': self.folder,
            'sample_name_at_start': self._get_sample_name(),
            'devices': {
                'source_drain': build_device_metadata(keithley, self.sd_combo.currentText(), self.sd_device),
                'probe_voltmeter': build_device_metadata(keithley, self.probe_combo.currentText(), self.probe_device),
                'gate': build_device_metadata(keithley, self.gate_combo.currentText(), self.gate_device),
            },
            'parameters': {
                'source_drain_voltage_v': self.sd_voltage_setpoint,
                'source_drain_compliance_current_a': parse_numeric_text(self.sd_compliance_input.text(), 'Source-drain compliance current'),
                'gate_voltage_min_v': parse_numeric_text(self.gate_min_input.text(), 'Gate Vmin'),
                'gate_voltage_max_v': parse_numeric_text(self.gate_max_input.text(), 'Gate Vmax'),
                'gate_voltage_step_v': parse_numeric_text(self.gate_step_input.text(), 'Gate Vstep'),
                'gate_compliance_current_a': parse_numeric_text(self.gate_compliance_input.text(), 'Gate compliance current'),
                'collection_time_ms': self.collection_time,
                'nplc': self.nplc_combo.currentText(),
                'source_drain_voltage_range_setting': self.sd_voltage_range_combo.currentText(),
                'probe_voltage_range_setting': self.probe_voltage_range_combo.currentText(),
                'gate_voltage_range_setting': self.gate_voltage_range_combo.currentText(),
                'gate_current_range_setting': self.gate_current_range_setting,
                'gate_current_range_lock_settle_ms': self.gate_current_range_lock_settle_ms,
                'n_runs_requested': self.nruns_input.value(),
                'average_over_interval': self.average_checkbox.isChecked(),
            },
            'diagnostics': {
                'gate_current_range_effective_a': self._finite_or_none(self.gate_current_range_effective_a),
                'gate_current_range_used_autorange_to_choose_initial_range': self.gate_current_range_used_autorange,
                'gate_current_range_overflowed': self.gate_current_range_overflowed,
            },
            'progress': {
                'estimated_total_steps': self.total_steps,
            },
        }

    def init_ui(self):
        self.setWindowTitle('Keithley Four-Probe FET')
        self.resize(1500, 920)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        device_list = [' - '.join(entry) for entry in keithley.get_devices_list()]
        device_list.append('Mock')

        combo_width = 190
        field_width = 90

        self.sd_combo = QComboBox()
        self.sd_combo.addItems(device_list)
        self.sd_combo.setMaximumWidth(combo_width)
        self.refresh_button = QPushButton('Refresh GPIB')
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.refresh_status_label = QLabel('Idle')
        self.refresh_status_label.setStyleSheet("color: #64748b;")

        self.probe_combo = QComboBox()
        self.probe_combo.addItems(device_list)
        self.probe_combo.setMaximumWidth(combo_width)

        self.gate_combo = QComboBox()
        self.gate_combo.addItems(device_list)
        self.gate_combo.setMaximumWidth(combo_width)

        self.sample_name_input = QLineEdit()
        self.sample_name_input.setMaximumWidth(200)

        self.nplc_combo = QComboBox()
        self.nplc_combo.addItems(['0.01', '0.1', '1', '10'])
        self.nplc_combo.setMaximumWidth(100)

        self.sd_voltage_range_combo = QComboBox()
        self.sd_voltage_range_combo.addItems(['Auto-range', '0.2', '2', '20', '200', '1000'])
        self.sd_voltage_range_combo.setMaximumWidth(110)

        self.sd_voltage_input = QLineEdit("0.1")
        self.sd_voltage_input.setMaximumWidth(field_width)
        self.sd_compliance_input = QLineEdit("1e-3")
        self.sd_compliance_input.setMaximumWidth(field_width)

        self.gate_voltage_range_combo = QComboBox()
        self.gate_voltage_range_combo.addItems(['Auto-range', '2', '20', '200', '1000'])
        self.gate_voltage_range_combo.setMaximumWidth(110)

        self.gate_min_input = QLineEdit("0.0")
        self.gate_min_input.setMaximumWidth(field_width)
        self.gate_max_input = QLineEdit("5.0")
        self.gate_max_input.setMaximumWidth(field_width)
        self.gate_step_input = QLineEdit("0.5")
        self.gate_step_input.setMaximumWidth(field_width)

        self.gate_compliance_input = QLineEdit("1e-6")
        self.gate_compliance_input.setMaximumWidth(field_width)

        gate_current_ranges = ['Auto-range'] + list((2 * 10.0 ** np.arange(-12.0, -1.0, 1.0)).astype(str))
        self.gate_current_range_combo = QComboBox()
        self.gate_current_range_combo.addItems(gate_current_ranges)
        self.gate_current_range_combo.setMaximumWidth(110)

        self.probe_voltage_range_combo = QComboBox()
        self.probe_voltage_range_combo.addItems(['Auto-range', '0.002', '0.02', '0.2', '2', '20', '200'])
        self.probe_voltage_range_combo.setMaximumWidth(110)

        self.nruns_input = QSpinBox()
        self.nruns_input.setRange(1, 1000)
        self.nruns_input.setValue(2)
        self.nruns_input.setMaximumWidth(80)

        self.collection_time_input = QSpinBox()
        self.collection_time_input.setRange(10, 10000)
        self.collection_time_input.setValue(500)
        self.collection_time_input.setMaximumWidth(80)

        self.average_checkbox = QCheckBox('Average readings over interval')
        self.average_checkbox.setChecked(True)

        control_grid = QGridLayout()
        control_grid.setHorizontalSpacing(8)
        control_grid.setVerticalSpacing(4)

        row = 0
        control_grid.addWidget(QLabel('SD Src'), row, 0)
        control_grid.addWidget(self.sd_combo, row, 1)
        control_grid.addWidget(QLabel('Probe Meter'), row, 2)
        control_grid.addWidget(self.probe_combo, row, 3)
        control_grid.addWidget(QLabel('Gate Src'), row, 4)
        control_grid.addWidget(self.gate_combo, row, 5)
        control_grid.addWidget(self.refresh_button, row, 6)
        control_grid.addWidget(self.refresh_status_label, row, 7)

        row += 1
        control_grid.addWidget(QLabel('Sample'), row, 0)
        control_grid.addWidget(self.sample_name_input, row, 1)
        control_grid.addWidget(QLabel('NPLC'), row, 2)
        control_grid.addWidget(self.nplc_combo, row, 3)
        control_grid.addWidget(QLabel('Runs'), row, 4)
        control_grid.addWidget(self.nruns_input, row, 5)

        row += 1
        control_grid.addWidget(QLabel('SD Vset'), row, 0)
        control_grid.addWidget(self.sd_voltage_input, row, 1)
        control_grid.addWidget(QLabel('SD Compliance'), row, 2)
        control_grid.addWidget(self.sd_compliance_input, row, 3)
        control_grid.addWidget(QLabel('SD Range'), row, 4)
        control_grid.addWidget(self.sd_voltage_range_combo, row, 5)

        row += 1
        control_grid.addWidget(QLabel('Avg ms'), row, 0)
        control_grid.addWidget(self.collection_time_input, row, 1)
        control_grid.addWidget(QLabel('Probe Range'), row, 2)
        control_grid.addWidget(self.probe_voltage_range_combo, row, 3)

        row += 1
        control_grid.addWidget(QLabel('Gate Vmin'), row, 0)
        control_grid.addWidget(self.gate_min_input, row, 1)
        control_grid.addWidget(QLabel('Gate Vmax'), row, 2)
        control_grid.addWidget(self.gate_max_input, row, 3)
        control_grid.addWidget(QLabel('Gate Vstep'), row, 4)
        control_grid.addWidget(self.gate_step_input, row, 5)

        row += 1
        control_grid.addWidget(QLabel('Gate Compliance'), row, 0)
        control_grid.addWidget(self.gate_compliance_input, row, 1)
        control_grid.addWidget(QLabel('Gate Range'), row, 2)
        control_grid.addWidget(self.gate_voltage_range_combo, row, 3)
        control_grid.addWidget(QLabel('Gate I Range'), row, 4)
        control_grid.addWidget(self.gate_current_range_combo, row, 5)

        row += 1
        control_grid.addWidget(self.average_checkbox, row, 0, 1, 6)

        layout.addLayout(control_grid)

        button_row = QHBoxLayout()
        self.start_button = QPushButton('Start')
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_measurement)
        self.stop_button = QPushButton('Stop')
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(lambda _checked=False: self.stop_measurement())
        self.stop_button.setEnabled(False)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        layout.addLayout(button_row)

        status_row = QHBoxLayout()
        self.sd_status_label = QLabel('Ids: n/a A')
        self.sd_status_label.setAlignment(Qt.AlignCenter)
        self.probe_status_label = QLabel('Vprobe: n/a V')
        self.probe_status_label.setAlignment(Qt.AlignCenter)
        self.gate_status_label = QLabel('Ig: n/a A')
        self.gate_status_label.setAlignment(Qt.AlignCenter)
        status_row.addWidget(self.sd_status_label)
        status_row.addWidget(self.probe_status_label)
        status_row.addWidget(self.gate_status_label)
        layout.addLayout(status_row)

        pg.setConfigOption('background', 'w')
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('w')
        self.iv_plot = self.plot_widget.addPlot(title='Ids vs Vg')
        self.iv_plot.showGrid(x=True, y=True, alpha=0.12)
        self.iv_plot.setLabel('left', 'Source-Drain current', units='A')
        self.iv_plot.setLabel('bottom', 'Gate voltage', units='V')
        self.iv_plot.getAxis('left').enableAutoSIPrefix(True)
        self.iv_plot.getAxis('bottom').enableAutoSIPrefix(True)
        self.probe_plot = self.plot_widget.addPlot(title='Probe Voltage vs Vg')
        self.probe_plot.showGrid(x=True, y=True, alpha=0.12)
        self.probe_plot.setLabel('left', 'Probe voltage', units='V')
        self.probe_plot.setLabel('bottom', 'Gate voltage', units='V')
        self.probe_plot.getAxis('left').enableAutoSIPrefix(True)
        self.probe_plot.getAxis('bottom').enableAutoSIPrefix(True)
        self.plot_widget.nextRow()
        self.gate_plot = self.plot_widget.addPlot(title='Gate Leakage vs Gate Voltage')
        self.gate_plot.showGrid(x=True, y=True, alpha=0.12)
        self.gate_plot.setLabel('left', 'Gate leakage current', units='A')
        self.gate_plot.setLabel('bottom', 'Gate voltage', units='V')
        self.gate_plot.getAxis('left').enableAutoSIPrefix(True)
        self.gate_plot.getAxis('bottom').enableAutoSIPrefix(True)
        self.resistance_plot = self.plot_widget.addPlot(title='Resistance vs Gate Voltage')
        self.resistance_plot.showGrid(x=True, y=True, alpha=0.12)
        self.resistance_plot.setLabel('left', 'Resistance', units='Ohm')
        self.resistance_plot.setLabel('bottom', 'Gate voltage', units='V')
        self.resistance_plot.getAxis('left').enableAutoSIPrefix(True)
        self.resistance_plot.getAxis('bottom').enableAutoSIPrefix(True)
        layout.addWidget(self.plot_widget)

        self.status_label = QLabel('')
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel('Progress: 0/0')
        self.eta_label = QLabel('ETA: --')
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.eta_label)

        self.setLayout(layout)
        apply_standard_window_style(self)
        self.refresh_devices()
        self.progress_tracker = ProgressEta(self.progress_bar, self.eta_label, self.progress_label)
        self.total_steps = 0
        self.completed_steps = 0

    def refresh_devices(self):
        self.refresh_button.setEnabled(False)
        self.refresh_button.setText('Refreshing...')
        self.refresh_status_label.setText('Scanning GPIB...')
        QApplication.processEvents()
        try:
            devices = refresh_device_combos(
                keithley,
                [self.sd_combo, self.probe_combo, self.gate_combo],
                include_mock=True,
            )
            count = max(0, len(devices) - (1 if 'Mock' in devices else 0))
            self.refresh_status_label.setText(f'Found {count} device(s)')
        except Exception as exc:
            self.refresh_status_label.setText(f'Refresh failed: {exc}')
            raise
        finally:
            self.refresh_button.setText('Refresh GPIB')
            self.refresh_button.setEnabled(True)

    def estimate_total_steps(self):
        return max(1, len(self.gate_values))

    @staticmethod
    def _calculate_resistance(probe_voltage, source_drain_current):
        if (
            probe_voltage is None
            or source_drain_current is None
            or np.isnan(probe_voltage)
            or np.isnan(source_drain_current)
            or np.isclose(source_drain_current, 0.0, atol=1e-18, rtol=1e-12)
        ):
            return np.nan
        return float(abs(probe_voltage / source_drain_current))

    def _eval_numeric_field(self, widget, field_name):
        text = widget.text().strip()
        if not text:
            raise ValueError(f"{field_name} is required.")
        try:
            value = parse_numeric_text(text, field_name)
        except Exception as exc:
            raise ValueError(f"Could not evaluate {field_name}: {text}") from exc
        return value

    def _parse_voltage_range(self, combo):
        selected = combo.currentText()
        if selected == 'Auto-range':
            return None
        try:
            return parse_numeric_text(selected, "Voltage range")
        except Exception:
            try:
                return float(selected)
            except Exception:
                return None

    def _parse_current_range(self, combo, field_name):
        selected = combo.currentText()
        if selected == 'Auto-range':
            return None
        try:
            return parse_numeric_text(selected, field_name)
        except Exception as exc:
            raise ValueError(f"Could not evaluate {field_name}: {selected}") from exc

    @staticmethod
    def _finite_or_none(value):
        try:
            numeric = float(value)
        except Exception:
            return None
        return numeric if np.isfinite(numeric) else None

    @staticmethod
    def _format_value(value):
        return 'n/a' if value is None or np.isnan(value) else f'{value:.3e}'

    def _measurement_status_text(self, suffix=''):
        base = (
            f"Gate target {self._format_value(self.current_gate_target)} V | "
            f"Vsd set {self._format_value(self.sd_voltage_setpoint)} V"
        )
        extra = f"{self.gate_current_range_status_text}{suffix}"
        return f"{base}{extra}"

    def _lock_gate_current_range_after_first_point(self):
        if not self.gate_current_range_should_lock_after_first_point or self.gate_adapter is None:
            return False
        effective_range = self.gate_adapter.get_current_sense_range()
        if not np.isfinite(effective_range) or effective_range <= 0:
            raise RuntimeError("Could not read the effective gate current-sense range in auto mode.")
        self.gate_adapter.set_current_sense_range(effective_range)
        self.gate_current_range_effective_a = float(effective_range)
        self.gate_current_range_should_lock_after_first_point = False
        self.gate_current_range_used_autorange = True
        self.gate_current_range_status_text = f" | Gate I range locked {effective_range:.3e} A"
        return True

    def _gate_current_range_overflow_message(self, gate_currents):
        if self.gate_adapter is None or self.gate_adapter.kind != 'smu':
            return None
        if not np.isfinite(self.gate_current_range_effective_a) or self.gate_current_range_effective_a <= 0:
            return None
        observed = [
            abs(float(value))
            for value in gate_currents
            if value is not None and np.isfinite(value)
        ]
        if not observed:
            return None
        if max(observed) > self.gate_current_range_effective_a * (1 + 1e-6):
            self.gate_current_range_overflowed = True
            return (
                f"Gate current exceeded the locked gate current-sense range "
                f"({self.gate_current_range_effective_a:.3e} A).\n\n"
                "Pick a wider Gate I Range or rerun with Gate I Range set to Auto-range."
            )
        return None

    def _resolve_device(self, text, nplc):
        if text == 'Mock':
            return keithley.get_device('Mock', nplc=nplc)
        return keithley.get_device(text, nplc=nplc)

    def _build_sweep(self, vmin, vmax, step):
        if step <= 0:
            raise ValueError('Step must be positive.')
        values = []
        current = vmin
        eps = abs(step) * 0.5 + 1e-12
        if vmax < vmin:
            raise ValueError('Minimum value must not exceed maximum value.')
        while current <= vmax + eps:
            values.append(current)
            current += step
        if abs(values[-1] - vmax) > eps:
            values.append(vmax)
        return values

    def _segment_points(self, start, stop, step):
        if np.isclose(start, stop, atol=1e-15):
            return [float(start)]
        n = int(np.ceil(abs(stop - start) / max(step, 1e-12)))
        return [float(v) for v in np.linspace(start, stop, max(2, n + 1))]

    def _append_segment(self, out, seg):
        for v in seg:
            if not out or not np.isclose(out[-1], v, atol=1e-12):
                out.append(float(v))

    def _build_gate_path(self, vmin, vmax, step, runs):
        path = []
        self._append_segment(path, self._segment_points(0.0, vmin, step))
        runs = max(1, int(runs))
        current = vmin
        for i in range(runs):
            target = vmax if i % 2 == 0 else vmin
            self._append_segment(path, self._segment_points(current, target, step))
            current = target
        if not np.isclose(current, vmin, atol=1e-12):
            self._append_segment(path, self._segment_points(current, vmin, step))
        self._append_segment(path, self._segment_points(vmin, 0.0, step))
        return path if path else [0.0]

    def start_measurement(self):
        if self.timer.isActive():
            return
        self.measurement_metadata = {}
        try:
            sd_voltage = self._eval_numeric_field(self.sd_voltage_input, 'Source-drain voltage')
            sd_compliance = self._eval_numeric_field(self.sd_compliance_input, 'Source-drain compliance current')
            gate_min = self._eval_numeric_field(self.gate_min_input, 'Gate Vmin')
            gate_max = self._eval_numeric_field(self.gate_max_input, 'Gate Vmax')
            gate_step = self._eval_numeric_field(self.gate_step_input, 'Gate Vstep')
            gate_compliance = self._eval_numeric_field(self.gate_compliance_input, 'Gate compliance current')
            gate_current_range = self._parse_current_range(self.gate_current_range_combo, 'Gate current range')
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        if gate_step <= 0:
            QMessageBox.critical(self, "Error", "Gate voltage step must be positive.")
            return
        if gate_min > gate_max:
            QMessageBox.critical(self, "Error", "Gate Vmin must be <= Vmax.")
            return

        sd_device_text = self.sd_combo.currentText()
        probe_device_text = self.probe_combo.currentText()
        gate_device_text = self.gate_combo.currentText()
        if len({sd_device_text, probe_device_text, gate_device_text}) < 3:
            QMessageBox.critical(self, "Error", "Each role must use a different instrument.")
            return

        nplc_text = self.nplc_combo.currentText()
        try:
            self.sd_device = self._resolve_device(sd_device_text, nplc_text)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to open source-drain device: {exc}")
            return
        try:
            self.probe_device = self._resolve_device(probe_device_text, nplc_text)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to open probe voltmeter: {exc}")
            return
        try:
            self.gate_device = self._resolve_device(gate_device_text, nplc_text)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to open gate device: {exc}")
            return

        try:
            self.sd_adapter = keithley.VoltageSourceAdapter(self.sd_device)
            self.sd_adapter.configure(
                voltage_range=self._parse_voltage_range(self.sd_voltage_range_combo),
                compliance_current=sd_compliance,
                nplc=float(nplc_text),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to configure source-drain device: {exc}")
            self.sd_adapter = None
            return

        try:
            self.probe_adapter = keithley.VoltageMeterAdapter(self.probe_device)
            self.probe_adapter.configure(
                voltage_range=self._parse_voltage_range(self.probe_voltage_range_combo),
                nplc=float(nplc_text),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to configure probe voltmeter: {exc}")
            if self.sd_adapter:
                self.sd_adapter.disable()
            self.sd_adapter = None
            return

        try:
            self.gate_adapter = keithley.VoltageSourceAdapter(self.gate_device)
            gate_current_autorange = None
            gate_current_sense_range = None
            if self.gate_adapter.kind == 'smu':
                if gate_current_range is None:
                    gate_current_autorange = True
                else:
                    gate_current_sense_range = gate_current_range
                    gate_current_autorange = False
            self.gate_adapter.configure(
                voltage_range=self._parse_voltage_range(self.gate_voltage_range_combo),
                compliance_current=gate_compliance,
                nplc=float(nplc_text),
                current_sense_range=gate_current_sense_range,
                current_sense_autorange=gate_current_autorange,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to configure gate device: {exc}")
            if self.sd_adapter:
                self.sd_adapter.disable()
            if self.probe_adapter:
                self.probe_adapter.close()
            self.sd_adapter = None
            self.probe_adapter = None
            return

        self.measurements = []
        try:
            self.gate_values = self._build_gate_path(
                gate_min,
                gate_max,
                gate_step,
                self.nruns_input.value(),
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            self.cleanup_devices()
            return
        self.sd_voltage_setpoint = sd_voltage
        self.gate_direction = 1
        self.gate_remaining_runs = self.nruns_input.value()
        self.current_gate_index = 0
        self.current_gate_target = self.gate_values[0]
        self.gate_current_range_setting = self.gate_current_range_combo.currentText()
        self.gate_current_range_effective_a = np.nan
        self.gate_current_range_used_autorange = False
        self.gate_current_range_overflowed = False
        self.gate_current_range_should_lock_after_first_point = False
        self.gate_current_range_settle_until_ms = 0
        self.gate_current_range_status_text = ''
        if self.gate_adapter.kind == 'smu':
            if gate_current_range is None:
                self.gate_current_range_should_lock_after_first_point = True
                self.gate_current_range_status_text = " | Gate I range auto (pending lock)"
            else:
                effective_gate_range = self.gate_adapter.get_current_sense_range()
                if np.isfinite(effective_gate_range) and effective_gate_range > 0:
                    self.gate_current_range_effective_a = float(effective_gate_range)
                else:
                    self.gate_current_range_effective_a = float(abs(gate_current_range))
                self.gate_current_range_status_text = (
                    f" | Gate I range fixed {self.gate_current_range_effective_a:.3e} A"
                )
        try:
            self.gate_adapter.set_voltage(self.current_gate_target)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to set gate voltage: {exc}")
            self.cleanup_devices()
            return
        try:
            self.sd_adapter.set_voltage(self.sd_voltage_setpoint)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to set source-drain voltage: {exc}")
            self.cleanup_devices()
            return

        self.collection_time = self.collection_time_input.value()
        self.start_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')

        self.iv_plot.clear()
        self.probe_plot.clear()
        self.gate_plot.clear()
        self.resistance_plot.clear()

        self.elapsed_timer.start()
        self.timer.start(max(50, int(self.collection_time / 5)))
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.total_steps = self.estimate_total_steps()
        self.measurement_metadata = self._capture_measurement_metadata()
        self.completed_steps = 0
        self.progress_tracker.start(self.total_steps)
        self.status_label.setText(self._measurement_status_text(" | Measurement running"))

    def cleanup_devices(self):
        if self.sd_adapter:
            self.sd_adapter.disable()
        if self.gate_adapter:
            self.gate_adapter.disable()
        if self.probe_adapter:
            self.probe_adapter.close()
        self.sd_adapter = None
        self.gate_adapter = None
        self.probe_adapter = None

    def perform_measurement(self):
        if not (self.sd_adapter and self.probe_adapter and self.gate_adapter):
            self.stop_measurement(save=False)
            return
        if self.gate_current_range_settle_until_ms:
            remaining = self.gate_current_range_settle_until_ms - self.elapsed_timer.elapsed()
            if remaining > 0:
                self.status_label.setText(
                    self._measurement_status_text(f" | waiting {int(max(0, remaining))} ms after Gate I range lock")
                )
                return
            self.gate_current_range_settle_until_ms = 0

        try:
            self.sd_adapter.set_voltage(self.sd_voltage_setpoint)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to set source-drain voltage: {exc}")
            self.stop_measurement(save=True)
            return
        try:
            self.gate_adapter.set_voltage(self.current_gate_target)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to set gate voltage: {exc}")
            self.stop_measurement(save=True)
            return

        start_mark = self.elapsed_timer.elapsed()
        sd_voltages = []
        sd_currents = []
        probe_voltages = []
        gate_voltages = []
        gate_currents = []
        avg_mode = self.average_checkbox.isChecked()
        sleep_window = max(0.002, self.collection_time / 1000.0 / 10.0)
        while self.elapsed_timer.elapsed() - start_mark < self.collection_time:
            sd_v, sd_i = self.sd_adapter.read_measurement()
            probe_v = self.probe_adapter.read_voltage()
            gate_v, gate_i = self.gate_adapter.read_measurement()
            sd_voltages.append(sd_v)
            sd_currents.append(sd_i)
            probe_voltages.append(probe_v)
            gate_voltages.append(gate_v)
            gate_currents.append(gate_i)
            time.sleep(min(0.05, sleep_window))

        def average_or_last(values):
            vals = [v for v in values if v is not None and not np.isnan(v)]
            if not vals:
                return np.nan
            if avg_mode:
                return float(np.nanmean(vals))
            return float(vals[-1])

        mean_sd_voltage = average_or_last(sd_voltages)
        mean_sd_current = average_or_last(sd_currents)
        mean_probe_voltage = average_or_last(probe_voltages)
        mean_gate_voltage = average_or_last(gate_voltages)
        mean_gate_current = average_or_last(gate_currents)
        locked_this_point = False

        self.measurements.append({
            'SourceDrainVoltage': mean_sd_voltage,
            'SourceDrainCurrent': mean_sd_current,
            'GateVoltage': mean_gate_voltage,
            'GateCurrent': mean_gate_current,
            'ProbeVoltage': mean_probe_voltage,
            'Direction': self.gate_direction,
            'Timestamp': time.time(),
        })
        if self.gate_current_range_should_lock_after_first_point:
            try:
                locked_this_point = self._lock_gate_current_range_after_first_point()
            except Exception as exc:
                self.stop_measurement(save=True)
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to lock the gate current-sense range after the first point:\n{exc}",
                )
                return
        if not locked_this_point:
            overflow_message = self._gate_current_range_overflow_message(gate_currents)
            if overflow_message:
                self.stop_measurement(save=True)
                QMessageBox.critical(self, "Error", overflow_message)
                return

        self.sd_status_label.setText(
            f'Ids: {self._format_value(mean_sd_current)} A @ Vsd {self._format_value(mean_sd_voltage)} V'
        )
        self.probe_status_label.setText(f'Vprobe: {self._format_value(mean_probe_voltage)} V')
        self.gate_status_label.setText(
            f'Ig: {self._format_value(mean_gate_current)} A @ Vg {self._format_value(mean_gate_voltage)} V'
        )
        self.status_label.setText(self._measurement_status_text())

        self.update_plots()
        self.completed_steps += 1
        self.progress_tracker.step(
            self.completed_steps,
            extra_text=f"Vg={self.current_gate_target:.3f} V",
        )
        if not self._advance_gate():
            self.stop_measurement(save=True)
            return
        if locked_this_point:
            self.gate_current_range_settle_until_ms = (
                self.elapsed_timer.elapsed() + self.gate_current_range_lock_settle_ms
            )

    def _advance_gate(self):
        next_index = self.current_gate_index + 1
        if next_index >= len(self.gate_values):
            return False
        self.current_gate_index = next_index
        self.current_gate_target = self.gate_values[self.current_gate_index]
        try:
            self.gate_adapter.set_voltage(self.current_gate_target)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to set gate voltage: {exc}")
            return False
        time.sleep(0.05)
        return True

    def stop_measurement(self, save=True):
        self.timer.stop()
        if self.sd_adapter:
            try:
                self.sd_adapter.set_voltage(0.0)
            except Exception:
                pass
            self.sd_adapter.disable()
        if self.gate_adapter:
            try:
                self.gate_adapter.set_voltage(0.0)
            except Exception as e:
                print(e)
                pass
            self.gate_adapter.disable()
        if self.probe_adapter:
            self.probe_adapter.close()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText('Measurement stopped.')

        if save and self.measurements:
            try:
                self.export_to_csv()
                self.make_plots()
            except Exception as exc:
                QMessageBox.warning(self, "Warning", f"Failed to save data: {exc}")

        self.sd_adapter = None
        self.gate_adapter = None
        self.probe_adapter = None

    def update_plots(self):
        iv_pairs = [
            (m['GateVoltage'], m['SourceDrainCurrent'])
            for m in self.measurements
            if m['GateVoltage'] is not None and not np.isnan(m['GateVoltage'])
            and m['SourceDrainCurrent'] is not None and not np.isnan(m['SourceDrainCurrent'])
        ]
        if iv_pairs:
            xs = [p[0] for p in iv_pairs]
            ys = [p[1] for p in iv_pairs]
            self.iv_plot.plot(xs, ys, pen=pg.mkPen(color='b', width=2), clear=True)
            self.iv_plot.plot(xs, ys, pen=None, symbol='o', symbolSize=4, symbolBrush=(0, 0, 255, 160))
        else:
            self.iv_plot.clear()

        probe_pairs = [
            (m['GateVoltage'], m['ProbeVoltage'])
            for m in self.measurements
            if m['GateVoltage'] is not None and not np.isnan(m['GateVoltage'])
            and m['ProbeVoltage'] is not None and not np.isnan(m['ProbeVoltage'])
        ]
        if probe_pairs:
            xs = [p[0] for p in probe_pairs]
            ys = [p[1] for p in probe_pairs]
            self.probe_plot.plot(xs, ys, pen=pg.mkPen(color='g', width=2), clear=True)
            self.probe_plot.plot(xs, ys, pen=None, symbol='o', symbolSize=4, symbolBrush=(0, 128, 0, 160))
        else:
            self.probe_plot.clear()

        gate_pairs = [
            (m['GateVoltage'], m['GateCurrent'])
            for m in self.measurements
            if m['GateVoltage'] is not None and not np.isnan(m['GateVoltage'])
            and m['GateCurrent'] is not None and not np.isnan(m['GateCurrent'])
        ]
        if gate_pairs:
            xs = [p[0] for p in gate_pairs]
            ys = [p[1] for p in gate_pairs]
            self.gate_plot.plot(xs, ys, pen=pg.mkPen(color='r', width=2), clear=True)
            self.gate_plot.plot(xs, ys, pen=None, symbol='o', symbolSize=4, symbolBrush=(255, 0, 0, 160))
        else:
            self.gate_plot.clear()

        resistance_pairs = []
        for m in self.measurements:
            gate_voltage = m.get('GateVoltage')
            resistance = self._calculate_resistance(
                m.get('ProbeVoltage'),
                m.get('SourceDrainCurrent'),
            )
            if (
                gate_voltage is None
                or np.isnan(gate_voltage)
                or np.isnan(resistance)
            ):
                continue
            resistance_pairs.append((gate_voltage, resistance))
        if resistance_pairs:
            xs = [p[0] for p in resistance_pairs]
            ys = [p[1] for p in resistance_pairs]
            self.resistance_plot.plot(xs, ys, pen=pg.mkPen(color=(128, 64, 0), width=2), clear=True)
            self.resistance_plot.plot(xs, ys, pen=None, symbol='o', symbolSize=4, symbolBrush=(128, 64, 0, 160))
        else:
            self.resistance_plot.clear()

    def get_dataframe(self):
        rows = []
        for m in self.measurements:
            rows.append({
                'SourceDrainVoltage': m['SourceDrainVoltage'],
                'SourceDrainCurrent': m['SourceDrainCurrent'],
                'GateVoltage': m['GateVoltage'],
                'GateCurrent': m['GateCurrent'],
                'ProbeVoltage': m['ProbeVoltage'],
                'Direction': m['Direction'],
                'Timestamp': m['Timestamp'],
            })
        return pd.DataFrame(rows)

    def export_to_csv(self):
        sample_name, _, data_dir = self._get_output_directories('data')
        df = self.get_dataframe()
        name = f'{sample_name}_FETFProbe_{self.start_time}'
        output_path = op.join(data_dir, f'FourProbeFET_{name}.data')
        df.to_csv(output_path, index=False)
        metadata = dict(self.measurement_metadata) if self.measurement_metadata else self._capture_measurement_metadata()
        metadata.update({
            'sample_name_at_save': sample_name,
            'saved_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'measurement_count': len(self.measurements),
            'data_columns': list(df.columns),
            'data_file': output_path,
            'metadata_file': op.join(data_dir, f'FourProbeFET_{name}.meta.json'),
        })
        metadata['parameters'] = dict(metadata.get('parameters', {}))
        metadata['parameters'].update({
            'gate_current_range_setting': self.gate_current_range_setting,
            'gate_current_range_lock_settle_ms': self.gate_current_range_lock_settle_ms,
        })
        metadata['diagnostics'] = dict(metadata.get('diagnostics', {}))
        metadata['diagnostics'].update({
            'gate_current_range_effective_a': self._finite_or_none(self.gate_current_range_effective_a),
            'gate_current_range_used_autorange_to_choose_initial_range': self.gate_current_range_used_autorange,
            'gate_current_range_overflowed': self.gate_current_range_overflowed,
        })
        metadata['progress'] = dict(metadata.get('progress', {}))
        metadata['progress'].update({
            'completed_steps': self.completed_steps,
            'current_gate_index': self.current_gate_index,
        })
        write_json_file(metadata['metadata_file'], metadata)

    def make_plots(self):
        sample_name = self.sample_name_input.text() or 'sample'
        sample_dir = op.join(self.folder, self.date, sample_name)
        plot_dir = op.join(sample_dir, 'plots')
        if not op.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        df = self.get_dataframe()
        name = f'{sample_name}_FETFProbe_{self.start_time}'

        fig1 = plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(df['GateVoltage'], df['SourceDrainCurrent'], 'o-', markersize=3)
        plt.xlabel('Gate Voltage (V)')
        plt.ylabel('Source-Drain Current (A)')
        plt.grid(True)
        plt.savefig(op.join(plot_dir, f'FourProbeFET_IV_{name}.png'), dpi=300)
        fig1.clf()

        fig2 = plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(df['GateVoltage'], df['ProbeVoltage'], 'o-', markersize=3)
        plt.xlabel('Gate Voltage (V)')
        plt.ylabel('Probe Voltage (V)')
        plt.grid(True)
        plt.savefig(op.join(plot_dir, f'FourProbeFET_Probe_{name}.png'), dpi=300)
        fig2.clf()

        fig3 = plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(df['GateVoltage'], df['GateCurrent'], 'o-', markersize=3)
        plt.xlabel('Gate Voltage (V)')
        plt.ylabel('Gate Leakage Current (A)')
        plt.grid(True)
        plt.savefig(op.join(plot_dir, f'FourProbeFET_Gate_{name}.png'), dpi=300)
        fig3.clf()

        source_currents = df['SourceDrainCurrent'].to_numpy(dtype=float, copy=True)
        probe_voltages = df['ProbeVoltage'].to_numpy(dtype=float, copy=True)
        resistance = np.full(source_currents.shape, np.nan, dtype=float)
        valid = (
            ~np.isnan(source_currents)
            & ~np.isnan(probe_voltages)
            & ~np.isclose(source_currents, 0.0, atol=1e-18, rtol=1e-12)
        )
        resistance[valid] = np.abs(probe_voltages[valid] / source_currents[valid])
        fig4 = plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(df['GateVoltage'], resistance, 'o-', markersize=3)
        plt.xlabel('Gate Voltage (V)')
        plt.ylabel('Resistance (Ohm)')
        plt.grid(True)
        plt.savefig(op.join(plot_dir, f'FourProbeFET_Resistance_{name}.png'), dpi=300)
        fig4.clf()

        matplotlib.pyplot.close(fig1)
        matplotlib.pyplot.close(fig2)
        matplotlib.pyplot.close(fig3)
        matplotlib.pyplot.close(fig4)
        plt.close('all')
        gc.collect()

    def closeEvent(self, event):
        try:
            if self.timer.isActive():
                self.timer.stop()
            if self.sd_adapter:
                try:
                    self.sd_adapter.set_voltage(0.0)
                except Exception:
                    pass
                self.sd_adapter.disable()
            if self.gate_adapter:
                try:
                    self.gate_adapter.set_voltage(0.0)
                except Exception:
                    pass
                self.gate_adapter.disable()
            if self.probe_adapter:
                self.probe_adapter.close()
            for dev in (self.sd_device, self.gate_device, self.probe_device):
                if dev and hasattr(dev, 'close'):
                    try:
                        dev.close()
                    except Exception:
                        pass
        finally:
            super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FourProbeFET()
    window.show()
    sys.exit(app.exec_())
