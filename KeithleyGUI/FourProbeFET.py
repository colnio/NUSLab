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
)

import keithley

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
        self.sd_direction = 1
        self.sd_remaining_runs = 0
        self.sd_target = 0.0
        self.gate_values = []
        self.current_gate_index = 0
        self.current_gate_target = 0.0
        self._eval_warning_shown = False

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

    def init_ui(self):
        self.setWindowTitle('Keithley Four-Probe FET')
        layout = QVBoxLayout()

        device_list = [' - '.join(entry) for entry in keithley.get_devices_list()]
        device_list.append('Mock')

        combo_width = 190
        field_width = 90

        self.sd_combo = QComboBox()
        self.sd_combo.addItems(device_list)
        self.sd_combo.setMaximumWidth(combo_width)

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

        self.sd_min_input = QLineEdit("-0.5")
        self.sd_min_input.setMaximumWidth(field_width)
        self.sd_max_input = QLineEdit("0.5")
        self.sd_max_input.setMaximumWidth(field_width)
        self.sd_step_input = QLineEdit("0.05")
        self.sd_step_input.setMaximumWidth(field_width)

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

        row += 1
        control_grid.addWidget(QLabel('Sample'), row, 0)
        control_grid.addWidget(self.sample_name_input, row, 1)
        control_grid.addWidget(QLabel('NPLC'), row, 2)
        control_grid.addWidget(self.nplc_combo, row, 3)
        control_grid.addWidget(QLabel('Runs'), row, 4)
        control_grid.addWidget(self.nruns_input, row, 5)

        row += 1
        control_grid.addWidget(QLabel('SD Vmin'), row, 0)
        control_grid.addWidget(self.sd_min_input, row, 1)
        control_grid.addWidget(QLabel('SD Vmax'), row, 2)
        control_grid.addWidget(self.sd_max_input, row, 3)
        control_grid.addWidget(QLabel('SD Vstep'), row, 4)
        control_grid.addWidget(self.sd_step_input, row, 5)

        row += 1
        control_grid.addWidget(QLabel('SD Compliance'), row, 0)
        control_grid.addWidget(self.sd_compliance_input, row, 1)
        control_grid.addWidget(QLabel('SD Range'), row, 2)
        control_grid.addWidget(self.sd_voltage_range_combo, row, 3)
        control_grid.addWidget(QLabel('Avg ms'), row, 4)
        control_grid.addWidget(self.collection_time_input, row, 5)

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
        control_grid.addWidget(QLabel('Probe Range'), row, 4)
        control_grid.addWidget(self.probe_voltage_range_combo, row, 5)

        row += 1
        control_grid.addWidget(self.average_checkbox, row, 0, 1, 6)

        layout.addLayout(control_grid)

        button_row = QHBoxLayout()
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_measurement)
        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_measurement)
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
        self.iv_plot = self.plot_widget.addPlot(title='I_sd vs V_sd')
        self.iv_plot.showGrid(x=True, y=True)
        self.iv_plot.setLabel('left', 'Source-Drain Current (A)')
        self.iv_plot.setLabel('bottom', 'Source-Drain Voltage (V)')
        self.probe_plot = self.plot_widget.addPlot(title='Probe Voltage vs V_sd')
        self.probe_plot.showGrid(x=True, y=True)
        self.probe_plot.setLabel('left', 'Probe Voltage (V)')
        self.probe_plot.setLabel('bottom', 'Source-Drain Voltage (V)')
        self.plot_widget.nextRow()
        self.gate_plot = self.plot_widget.addPlot(title='Gate Leakage vs Gate Voltage')
        self.gate_plot.showGrid(x=True, y=True)
        self.gate_plot.setLabel('left', 'Gate Leakage Current (A)')
        self.gate_plot.setLabel('bottom', 'Gate Voltage (V)')
        layout.addWidget(self.plot_widget)

        self.status_label = QLabel('')
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def _eval_numeric_field(self, widget, field_name):
        text = widget.text().strip()
        if not text:
            raise ValueError(f"{field_name} is required.")
        if not self._eval_warning_shown:
            QMessageBox.warning(
                self,
                "Warning",
                "Numeric fields accept Python-style expressions (evaluated with eval). "
                "Only enter trusted values.",
            )
            self._eval_warning_shown = True
        try:
            value = float(eval(text, {"__builtins__": {}}, {}))
        except Exception as exc:
            raise ValueError(f"Could not evaluate {field_name}: {text}") from exc
        return value

    def _parse_voltage_range(self, combo):
        selected = combo.currentText()
        if selected == 'Auto-range':
            return None
        try:
            return float(eval(selected))
        except Exception:
            try:
                return float(selected)
            except Exception:
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

    def start_measurement(self):
        if self.timer.isActive():
            return
        try:
            sd_min = self._eval_numeric_field(self.sd_min_input, 'Source-drain Vmin')
            sd_max = self._eval_numeric_field(self.sd_max_input, 'Source-drain Vmax')
            sd_step = self._eval_numeric_field(self.sd_step_input, 'Source-drain Vstep')
            sd_compliance = self._eval_numeric_field(self.sd_compliance_input, 'Source-drain compliance current')
            gate_min = self._eval_numeric_field(self.gate_min_input, 'Gate Vmin')
            gate_max = self._eval_numeric_field(self.gate_max_input, 'Gate Vmax')
            gate_step = self._eval_numeric_field(self.gate_step_input, 'Gate Vstep')
            gate_compliance = self._eval_numeric_field(self.gate_compliance_input, 'Gate compliance current')
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        if sd_step <= 0 or gate_step <= 0:
            QMessageBox.critical(self, "Error", "Voltage steps must be positive.")
            return
        if sd_min > sd_max:
            QMessageBox.critical(self, "Error", "SD Vmin must be <= Vmax.")
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
            self.gate_adapter.configure(
                voltage_range=self._parse_voltage_range(self.gate_voltage_range_combo),
                compliance_current=gate_compliance,
                nplc=float(nplc_text),
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
            self.sd_values = self._build_sweep(sd_min, sd_max, sd_step)
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            self.cleanup_devices()
            return
        try:
            self.gate_values = self._build_sweep(gate_min, gate_max, gate_step)
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            self.cleanup_devices()
            return

        self.sd_min = sd_min
        self.sd_max = sd_max
        self.sd_step = sd_step
        self.sd_direction = 1
        self.sd_remaining_runs = self.nruns_input.value()
        self.sd_target = sd_min
        self.current_gate_index = 0
        self.current_gate_target = self.gate_values[0]
        try:
            self.gate_adapter.set_voltage(self.current_gate_target)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to set gate voltage: {exc}")
            self.cleanup_devices()
            return

        self.collection_time = self.collection_time_input.value()
        self.start_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')

        self.iv_plot.clear()
        self.probe_plot.clear()
        self.gate_plot.clear()

        self.elapsed_timer.start()
        self.timer.start(max(50, int(self.collection_time / 5)))
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText('Measurement running...')

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

        try:
            self.sd_adapter.set_voltage(self.sd_target)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to set source-drain voltage: {exc}")
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

        self.measurements.append({
            'GateTarget': self.current_gate_target,
            'SourceDrainTarget': self.sd_target,
            'SourceDrainVoltage': mean_sd_voltage,
            'SourceDrainCurrent': mean_sd_current,
            'ProbeVoltage': mean_probe_voltage,
            'GateVoltage': mean_gate_voltage,
            'GateLeakageCurrent': mean_gate_current,
            'Direction': self.sd_direction,
            'Timestamp': time.time(),
        })

        def fmt(val):
            return 'n/a' if val is None or np.isnan(val) else f'{val:.3e}'

        self.sd_status_label.setText(f'Ids: {fmt(mean_sd_current)} A @ Vsd {fmt(mean_sd_voltage)} V')
        self.probe_status_label.setText(f'Vprobe: {fmt(mean_probe_voltage)} V')
        self.gate_status_label.setText(f'Ig: {fmt(mean_gate_current)} A @ Vg {fmt(mean_gate_voltage)} V')
        self.status_label.setText(f'Gate target {fmt(self.current_gate_target)} V | SD target {fmt(self.sd_target)} V')

        self.update_plots()
        if not self._advance_sd():
            self.stop_measurement(save=True)

    def _advance_sd(self):
        tol = abs(self.sd_step) * 0.5 + 1e-12
        next_target = self.sd_target + self.sd_step * self.sd_direction

        if self.sd_direction > 0 and next_target > self.sd_max + tol:
            next_target = self.sd_max
            self.sd_direction = -1
            self.sd_remaining_runs -= 1
        elif self.sd_direction < 0 and next_target < self.sd_min - tol:
            next_target = self.sd_min
            self.sd_direction = 1
            self.sd_remaining_runs -= 1

        if self.sd_remaining_runs <= 0 and self.sd_direction == 1 and abs(next_target - self.sd_min) <= tol:
            return self._advance_gate()

        self.sd_target = next_target
        return True

    def _advance_gate(self):
        self.sd_remaining_runs = self.nruns_input.value()
        self.sd_direction = 1
        self.sd_target = self.sd_min
        self.current_gate_index += 1
        if self.current_gate_index >= len(self.gate_values):
            return False
        self.current_gate_target = self.gate_values[self.current_gate_index]
        try:
            self.gate_adapter.set_voltage(self.current_gate_target)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to set gate voltage: {exc}")
            return False
        # Small settle delay
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
            except Exception:
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
        sd_voltages = [m['SourceDrainVoltage'] for m in self.measurements if not np.isnan(m['SourceDrainVoltage'])]
        sd_currents = [m['SourceDrainCurrent'] for m in self.measurements if not np.isnan(m['SourceDrainCurrent'])]
        probe_voltages = [m['ProbeVoltage'] for m in self.measurements if not np.isnan(m['ProbeVoltage'])]
        gate_voltages = [m['GateVoltage'] for m in self.measurements if not np.isnan(m['GateVoltage'])]
        gate_currents = [m['GateLeakageCurrent'] for m in self.measurements if not np.isnan(m['GateLeakageCurrent'])]

        if sd_voltages and sd_currents:
            self.iv_plot.plot(sd_voltages, sd_currents, pen=pg.mkPen(color='b', width=2), clear=True)
            self.iv_plot.plot(sd_voltages, sd_currents, pen=None, symbol='o', symbolSize=4, symbolBrush=(0, 0, 255, 160))
        else:
            self.iv_plot.clear()

        if sd_voltages and probe_voltages:
            self.probe_plot.plot(sd_voltages, probe_voltages, pen=pg.mkPen(color='g', width=2), clear=True)
            self.probe_plot.plot(sd_voltages, probe_voltages, pen=None, symbol='o', symbolSize=4, symbolBrush=(0, 128, 0, 160))
        else:
            self.probe_plot.clear()

        if gate_voltages and gate_currents:
            self.gate_plot.plot(gate_voltages, gate_currents, pen=pg.mkPen(color='r', width=2), clear=True)
            self.gate_plot.plot(gate_voltages, gate_currents, pen=None, symbol='o', symbolSize=4, symbolBrush=(255, 0, 0, 160))
        else:
            self.gate_plot.clear()

    def get_dataframe(self):
        return pd.DataFrame(
            [{
                'SourceDrainCurrent': m['SourceDrainCurrent'],
                'SourceDrainVoltage': m['SourceDrainVoltage'],
                'ProbeVoltage': m['ProbeVoltage'],
                'GateVoltage': m['GateVoltage'],
                'GateLeakageCurrent': m['GateLeakageCurrent'],
                'Direction': m['Direction'],
                'Timestamp': m['Timestamp'],
                'GateTarget': m['GateTarget'],
                'SourceDrainTarget': m['SourceDrainTarget'],
            } for m in self.measurements]
        )

    def export_to_csv(self):
        sample_name = self.sample_name_input.text() or 'sample'
        sample_dir = op.join(self.folder, self.date, sample_name)
        data_dir = op.join(sample_dir, 'data')
        if not op.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        df = self.get_dataframe()
        name = f'{sample_name}_FETFProbe_{self.start_time}'
        df.to_csv(op.join(data_dir, f'FourProbeFET_{name}.csv'), index=False)

    def make_plots(self):
        sample_name = self.sample_name_input.text() or 'sample'
        sample_dir = op.join(self.folder, self.date, sample_name)
        plot_dir = op.join(sample_dir, 'plots')
        if not op.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        df = self.get_dataframe()
        name = f'{sample_name}_FETFProbe_{self.start_time}'

        fig1 = plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(df['SourceDrainVoltage'], df['SourceDrainCurrent'], 'o-', markersize=3)
        plt.xlabel('Source-Drain Voltage (V)')
        plt.ylabel('Source-Drain Current (A)')
        plt.grid(True)
        plt.savefig(op.join(plot_dir, f'FourProbeFET_IV_{name}.png'), dpi=300)
        fig1.clf()

        fig2 = plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(df['SourceDrainVoltage'], df['ProbeVoltage'], 'o-', markersize=3)
        plt.xlabel('Source-Drain Voltage (V)')
        plt.ylabel('Probe Voltage (V)')
        plt.grid(True)
        plt.savefig(op.join(plot_dir, f'FourProbeFET_Probe_{name}.png'), dpi=300)
        fig2.clf()

        fig3 = plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(df['GateVoltage'], df['GateLeakageCurrent'], 'o-', markersize=3)
        plt.xlabel('Gate Voltage (V)')
        plt.ylabel('Gate Leakage Current (A)')
        plt.grid(True)
        plt.savefig(op.join(plot_dir, f'FourProbeFET_Gate_{name}.png'), dpi=300)
        fig3.clf()

        matplotlib.pyplot.close(fig1)
        matplotlib.pyplot.close(fig2)
        matplotlib.pyplot.close(fig3)
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
