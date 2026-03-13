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
    QFormLayout,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QFileDialog,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QCheckBox,
    QProgressBar,
    QGroupBox,
    QSizePolicy,
    QFrame,
)

import keithley
from ui_helpers import (
    ProgressEta,
    refresh_device_combos,
    apply_standard_window_style,
    parse_numeric_text,
)

class FourProbeResistance(QWidget):
    def __init__(self):
        super().__init__()
        self.source_device = None
        self.voltage_device = None
        self.source_adapter = None
        self.voltmeter_adapter = None
        self.measurements = []
        self.direction = 1
        self.remaining_runs = 0
        self.current_target = 0.0
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
        self.setWindowTitle('Keithley Four-Probe Resistance')
        self.resize(1500, 900)
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        device_list = [' - '.join(entry) for entry in keithley.get_devices_list()]
        device_list.append('Mock')

        conn_box = QGroupBox("Connection")
        conn_layout = QGridLayout()
        conn_layout.setHorizontalSpacing(10)
        conn_layout.setVerticalSpacing(6)
        self.source_combo = QComboBox()
        self.source_combo.addItems(device_list)
        self.source_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.refresh_button = QPushButton('Refresh GPIB')
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.refresh_status_label = QLabel('Idle')
        self.refresh_status_label.setStyleSheet("color: #64748b;")
        self.voltmeter_combo = QComboBox()
        self.voltmeter_combo.addItems(device_list)
        self.voltmeter_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        conn_layout.addWidget(QLabel('Current source:'), 0, 0)
        conn_layout.addWidget(self.source_combo, 0, 1, 1, 3)
        conn_layout.addWidget(self.refresh_button, 0, 4)
        conn_layout.addWidget(self.refresh_status_label, 0, 5)
        conn_layout.addWidget(QLabel('Voltmeter:'), 1, 0)
        conn_layout.addWidget(self.voltmeter_combo, 1, 1, 1, 5)
        conn_box.setLayout(conn_layout)

        sample_box = QGroupBox("Sample")
        sample_layout = QFormLayout()
        sample_layout.setHorizontalSpacing(10)
        sample_layout.setVerticalSpacing(6)
        self.sample_name_input = QLineEdit()
        self.contact_label_input = QLineEdit()
        sample_layout.addRow('Sample name:', self.sample_name_input)
        sample_layout.addRow('Contact pair label:', self.contact_label_input)
        sample_box.setLayout(sample_layout)

        sweep_box = QGroupBox("Sweep")
        sweep_layout = QGridLayout()
        sweep_layout.setHorizontalSpacing(10)
        sweep_layout.setVerticalSpacing(6)
        self.nplc_combo = QComboBox()
        self.nplc_combo.addItems(['0.01', '0.1', '1', '10'])
        self.current_range_combo = QComboBox()
        current_ranges = list((2 * 10.0 ** np.arange(-9.0, -1.0, 1.0)).astype(str))
        self.current_range_combo.addItems(['Auto-range'] + current_ranges)
        self.compliance_input = QDoubleSpinBox()
        self.compliance_input.setRange(0.0, 1000.0)
        self.compliance_input.setValue(10.0)
        self.compliance_input.setDecimals(3)
        self.voltage_range_combo = QComboBox()
        voltage_ranges = ['0.002', '0.02', '0.2', '2', '20', '200', '1000']
        self.voltage_range_combo.addItems(['Auto-range'] + voltage_ranges)
        self.current_min_input = QLineEdit("-1e-5")
        self.current_max_input = QLineEdit("1e-5")
        self.current_step_input = QLineEdit("1e-6")
        self.nruns_input = QSpinBox()
        self.nruns_input.setRange(1, 1000)
        self.nruns_input.setValue(2)
        self.collection_time_input = QSpinBox()
        self.collection_time_input.setRange(10, 10000)
        self.collection_time_input.setValue(1000)
        sweep_layout.addWidget(QLabel('NPLC'), 0, 0)
        sweep_layout.addWidget(self.nplc_combo, 0, 1)
        sweep_layout.addWidget(QLabel('Current range (A)'), 0, 2)
        sweep_layout.addWidget(self.current_range_combo, 0, 3)
        sweep_layout.addWidget(QLabel('Compliance voltage (V)'), 1, 0)
        sweep_layout.addWidget(self.compliance_input, 1, 1)
        sweep_layout.addWidget(QLabel('Voltage range (V)'), 1, 2)
        sweep_layout.addWidget(self.voltage_range_combo, 1, 3)
        sweep_layout.addWidget(QLabel('Current min (A)'), 2, 0)
        sweep_layout.addWidget(self.current_min_input, 2, 1)
        sweep_layout.addWidget(QLabel('Current max (A)'), 2, 2)
        sweep_layout.addWidget(self.current_max_input, 2, 3)
        sweep_layout.addWidget(QLabel('Current step (A)'), 3, 0)
        sweep_layout.addWidget(self.current_step_input, 3, 1)
        sweep_layout.addWidget(QLabel('N runs (2 = up/down)'), 3, 2)
        sweep_layout.addWidget(self.nruns_input, 3, 3)
        sweep_layout.addWidget(QLabel('Averaging time (ms)'), 4, 0)
        sweep_layout.addWidget(self.collection_time_input, 4, 1)

        self.average_checkbox = QCheckBox('Average source current and measured voltage')
        self.average_checkbox.setChecked(True)
        sweep_layout.addWidget(self.average_checkbox, 4, 2, 1, 2)
        sweep_box.setLayout(sweep_layout)

        top_row = QHBoxLayout()
        top_row.addWidget(conn_box, 2)
        top_row.addWidget(sample_box, 1)
        root_layout.addLayout(top_row)
        root_layout.addWidget(sweep_box)

        button_row = QHBoxLayout()
        self.start_button = QPushButton('Start')
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_measurement)
        self.stop_button = QPushButton('Stop')
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(self.stop_measurement)
        self.stop_button.setEnabled(False)
        self.current_label = QLabel('Current: 0 A')
        self.current_label.setAlignment(Qt.AlignCenter)
        self.start_button.setMinimumHeight(32)
        self.stop_button.setMinimumHeight(32)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.current_label)
        button_row.addWidget(self.stop_button)
        root_layout.addLayout(button_row)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('w')
        self.iv_plot = self.plot_widget.addPlot(title='Voltage vs Current')
        self.iv_plot.showGrid(x=True, y=True, alpha=0.12)
        self.iv_plot.setLabel('left', 'Source-Drain current', units='A')
        self.iv_plot.setLabel('bottom', 'Probe voltage', units='V')
        self.iv_plot.getAxis('left').enableAutoSIPrefix(True)
        self.iv_plot.getAxis('bottom').enableAutoSIPrefix(True)
        self.current_plot = self.plot_widget.addPlot(title='Current Readings')
        self.current_plot.showGrid(x=True, y=True, alpha=0.12)
        self.current_plot.setLabel('left', 'Measured current', units='A')
        self.current_plot.setLabel('bottom', 'Measurement #')
        self.current_plot.getAxis('left').enableAutoSIPrefix(True)
        self.current_plot.addLegend()
        root_layout.addWidget(self.plot_widget, 1)

        self.status_label = QLabel('')
        self.status_label.setFrameShape(QFrame.StyledPanel)
        self.status_label.setMinimumHeight(24)
        root_layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel('Progress: 0/0')
        self.eta_label = QLabel('ETA: --')
        root_layout.addWidget(self.progress_bar)

        footer = QHBoxLayout()
        footer.addWidget(self.progress_label, 2)
        footer.addWidget(self.eta_label, 1)
        root_layout.addLayout(footer)

        self.setLayout(root_layout)
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
                [self.source_combo, self.voltmeter_combo],
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
        if self.current_step <= 0:
            return 1
        direction = 1
        current = self.current_min
        runs = int(self.remaining_runs)
        steps = 0
        max_iter = 5_000_000
        while steps < max_iter:
            steps += 1
            current += self.current_step * direction
            if current > self.current_max:
                current = self.current_max
                direction *= -1
                runs -= 1
            elif current < self.current_min:
                current = self.current_min
                direction *= -1
                runs -= 1
            if runs <= 0 and abs(current) <= self.current_step / 10:
                break
        return max(1, steps)

    def _parse_current_range(self):
        selected = self.current_range_combo.currentText()
        if selected == 'Auto-range':
            return None
        try:
            return parse_numeric_text(selected, "Current range")
        except Exception:
            try:
                return float(selected)
            except Exception:
                return None

    def _parse_voltage_range(self):
        selected = self.voltage_range_combo.currentText()
        if selected == 'Auto-range':
            return None
        try:
            return parse_numeric_text(selected, "Voltage range")
        except Exception:
            try:
                return float(selected)
            except Exception:
                return None

    def _eval_current_field(self, widget, field_name):
        text = widget.text().strip()
        if not text:
            raise ValueError(f"{field_name} is required.")
        try:
            value = parse_numeric_text(text, field_name)
        except Exception as exc:
            raise ValueError(f"Could not evaluate {field_name}: {text}") from exc
        return value

    def _resolve_device(self, text, nplc):
        if text == 'Mock':
            return keithley.get_device('Mock', nplc=nplc)
        return keithley.get_device(text, nplc=nplc)

    def start_measurement(self):
        if self.timer.isActive():
            return
        try:
            current_min = self._eval_current_field(self.current_min_input, 'Current min')
            current_max = self._eval_current_field(self.current_max_input, 'Current max')
            current_step = self._eval_current_field(self.current_step_input, 'Current step')
        except ValueError as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return
        if current_step <= 0:
            QMessageBox.critical(self, "Error", "Current step must be positive.")
            return
        if current_min > current_max:
            QMessageBox.critical(self, "Error", "Current min must be <= current max.")
            return

        source_text = self.source_combo.currentText()
        meter_text = self.voltmeter_combo.currentText()
        if meter_text == source_text:
            QMessageBox.critical(self, "Error", "Voltmeter must be different from the current source for four-probe measurements.")
            return
        nplc_text = self.nplc_combo.currentText()
        try:
            self.source_device = self._resolve_device(source_text, nplc_text)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to open source device: {exc}")
            return
        try:
            self.voltage_device = self._resolve_device(meter_text, nplc_text)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to open voltmeter: {exc}")
            return

        try:
            self.source_adapter = keithley.CurrentSourceAdapter(self.source_device)
            self.source_adapter.configure(
                current_range=self._parse_current_range(),
                compliance_voltage=self.compliance_input.value(),
                nplc=float(nplc_text),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to configure source: {exc}")
            self.source_adapter = None
            return

        try:
            self.voltmeter_adapter = keithley.VoltageMeterAdapter(self.voltage_device)
            self.voltmeter_adapter.configure(
                voltage_range=self._parse_voltage_range(),
                nplc=float(nplc_text),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to configure voltmeter: {exc}")
            if self.source_adapter:
                self.source_adapter.disable()
            self.source_adapter = None
            return

        self.measurements = []
        self.direction = 1
        self.current_target = current_min
        self.current_min = current_min
        self.current_max = current_max
        self.current_step = current_step
        self.remaining_runs = self.nruns_input.value()
        self.collection_time = self.collection_time_input.value()
        self.start_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')
        self.total_steps = self.estimate_total_steps()
        self.completed_steps = 0
        self.progress_tracker.start(self.total_steps)

        self.iv_plot.clear()
        self.current_plot.clear()
        self.elapsed_timer.start()
        self.timer.start(max(50, int(self.collection_time / 5)))
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText('Measurement running...')

    def perform_measurement(self):
        if not self.source_adapter or not self.voltmeter_adapter:
            self.stop_measurement(save=False)
            return
        try:
            self.source_adapter.set_current(self.current_target)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to set current: {exc}")
            self.stop_measurement(save=True)
            return

        self.current_label.setText(f'Current target: {self.current_target:.3e} A')
        start_mark = self.elapsed_timer.elapsed()
        sd_currents = []
        sd_voltages = []
        probe_voltages = []
        sense_currents = []
        avg_mode = self.average_checkbox.isChecked()
        sleep_window = max(0.002, self.collection_time / 1000.0 / 10.0)
        while self.elapsed_timer.elapsed() - start_mark < self.collection_time:
            sd_v, sd_i = self.source_adapter.read_measurement()
            if sd_i is None or np.isnan(sd_i):
                sd_i = self.source_adapter.read_current()
            if sd_v is None or np.isnan(sd_v):
                sd_v = self.source_adapter.read_voltage()
            probe_v, sense_current = self.voltmeter_adapter.read_measurement()
            sd_currents.append(sd_i)
            sd_voltages.append(sd_v)
            probe_voltages.append(probe_v)
            sense_currents.append(sense_current)
            time.sleep(min(0.05, sleep_window))

        def avg(values):
            vals = [v for v in values if v is not None and not np.isnan(v)]
            if not vals:
                return np.nan
            if avg_mode:
                return float(np.nanmean(vals))
            return float(vals[-1])

        mean_sd_current = avg(sd_currents)
        mean_sd_voltage = avg(sd_voltages)
        mean_probe_voltage = avg(probe_voltages)
        mean_sense_current = avg(sense_currents)

        entry = {
            'SourceDrainVoltage': mean_sd_voltage,
            'SourceDrainCurrent': mean_sd_current,
            'ProbeVoltage': mean_probe_voltage,
            'SenseCurrent': mean_sense_current,
            'Direction': self.direction,
            'Timestamp': time.time(),
        }
        self.measurements.append(entry)

        def fmt(val):
            return 'n/a' if val is None or np.isnan(val) else f'{val:.3e}'

        self.current_label.setText(
            f'Iset {fmt(self.current_target)} A | Isrc {fmt(mean_sd_current)} A | Isense {fmt(mean_sense_current)} A'
        )
        self.status_label.setText(
            f'Latest: Vsd={fmt(mean_sd_voltage)} V | Vprobe={fmt(mean_probe_voltage)} V | Direction {self.direction}'
        )

        self.update_plots()
        self.completed_steps += 1
        self.progress_tracker.step(
            self.completed_steps,
            extra_text=f"Iset={self.current_target:.3e} A",
        )
        self._advance_current()

    def _advance_current(self):
        self.current_target += self.current_step * self.direction
        if self.current_target > self.current_max:
            self.current_target = self.current_max
            self.direction *= -1
            self.remaining_runs -= 1
        elif self.current_target < self.current_min:
            self.current_target = self.current_min
            self.direction *= -1
            self.remaining_runs -= 1

        if self.remaining_runs <= 0 and abs(self.current_target) <= self.current_step / 10:
            self.stop_measurement(save=True)

    def stop_measurement(self, save=True):
        self.timer.stop()
        if self.source_adapter:
            self.source_adapter.disable()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText('Measurement stopped.')

        if save and self.measurements:
            try:
                self.export_to_csv()
                self.make_plots()
            except Exception as exc:
                QMessageBox.warning(self, "Warning", f"Failed to save data: {exc}")

    def update_plots(self):
        iv_pairs = [
            (m['ProbeVoltage'], m['SourceDrainCurrent'])
            for m in self.measurements
            if m['ProbeVoltage'] is not None
            and m['SourceDrainCurrent'] is not None
            and not np.isnan(m['ProbeVoltage'])
            and not np.isnan(m['SourceDrainCurrent'])
        ]
        if iv_pairs:
            iv_sorted = sorted(iv_pairs, key=lambda p: p[0])
            xs = [p[0] for p in iv_sorted]
            ys = [p[1] for p in iv_sorted]
            self.iv_plot.plot(xs, ys, pen=pg.mkPen(color=(25, 95, 185), width=2), clear=True)
            self.iv_plot.plot(xs, ys, pen=None, symbol='o', symbolSize=5, symbolBrush=(25, 95, 185, 120))
        else:
            self.iv_plot.clear()

        source_series = [
            (idx, m['SourceDrainCurrent'])
            for idx, m in enumerate(self.measurements)
            if m['SourceDrainCurrent'] is not None and not np.isnan(m['SourceDrainCurrent'])
        ]
        sense_series = [
            (idx, m.get('SenseCurrent'))
            for idx, m in enumerate(self.measurements)
            if m.get('SenseCurrent') is not None and not np.isnan(m.get('SenseCurrent'))
        ]
        self.current_plot.clear()
        if source_series:
            xs = [p[0] for p in source_series]
            ys = [p[1] for p in source_series]
            self.current_plot.plot(xs, ys, pen=pg.mkPen(color=(0, 150, 70), width=2), clear=False, name='Source current')
            self.current_plot.plot(xs, ys, pen=None, symbol='o', symbolSize=4, symbolBrush=(0, 128, 0, 160))
        if sense_series:
            xs = [p[0] for p in sense_series]
            ys = [p[1] for p in sense_series]
            self.current_plot.plot(xs, ys, pen=pg.mkPen(color='r', width=2), clear=False, name='Sense current')
            self.current_plot.plot(xs, ys, pen=None, symbol='o', symbolSize=4, symbolBrush=(255, 0, 0, 120))

    def get_dataframe(self):
        rows = []
        for m in self.measurements:
            rows.append({
                'SourceDrainVoltage': m['SourceDrainVoltage'],
                'SourceDrainCurrent': m['SourceDrainCurrent'],
                'ProbeVoltage': m['ProbeVoltage'],
                'Direction': m['Direction'],
                'Timestamp': m['Timestamp'],
            })
        return pd.DataFrame(rows)

    def export_to_csv(self):
        sample_name = self.sample_name_input.text() or 'sample'
        contact_label = self.contact_label_input.text() or 'contacts'
        sample_dir = op.join(self.folder, self.date, sample_name)
        data_dir = op.join(sample_dir, 'data')
        if not op.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        df = self.get_dataframe()
        name = f'{sample_name}_{contact_label}_{self.current_min}A_{self.current_max}A_{self.collection_time}ms'
        df.to_csv(op.join(data_dir, f'FourProbe_{name}_{self.start_time}.data'), index=False)

    def make_plots(self):
        sample_name = self.sample_name_input.text() or 'sample'
        contact_label = self.contact_label_input.text() or 'contacts'
        sample_dir = op.join(self.folder, self.date, sample_name)
        plot_dir = op.join(sample_dir, 'plots')
        if not op.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
        df = self.get_dataframe()
        name = f'{sample_name}_{contact_label}_{self.current_min}A_{self.current_max}A_{self.collection_time}ms'

        fig1 = plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(df['SourceDrainVoltage'], df['SourceDrainCurrent'], 'o-', markersize=3)
        plt.xlabel('Source-Drain Voltage (V)')
        plt.ylabel('Source-Drain Current (A)')
        plt.grid(True)
        plt.savefig(op.join(plot_dir, f'FourProbe_IV_{name}_{self.start_time}.png'), dpi=300)
        fig1.clf()

        fig2 = plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(df['SourceDrainVoltage'], df['ProbeVoltage'], 'o-', markersize=3)
        plt.xlabel('Source-Drain Voltage (V)')
        plt.ylabel('Probe Voltage (V)')
        plt.grid(True)
        plt.savefig(op.join(plot_dir, f'FourProbe_Vprobe_{name}_{self.start_time}.png'), dpi=300)
        fig2.clf()
        matplotlib.pyplot.close(fig1)
        matplotlib.pyplot.close(fig2)
        plt.close('all')
        gc.collect()

    def closeEvent(self, event):
        try:
            if self.timer.isActive():
                self.timer.stop()
            if self.source_adapter:
                self.source_adapter.disable()
            if self.voltmeter_adapter:
                self.voltmeter_adapter.close()
            if self.source_device and hasattr(self.source_device, 'close'):
                self.source_device.close()
            if self.voltage_device and hasattr(self.voltage_device, 'close') and self.voltage_device is not self.source_device:
                self.voltage_device.close()
        finally:
            super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FourProbeResistance()
    window.show()
    sys.exit(app.exec_())
