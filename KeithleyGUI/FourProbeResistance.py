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
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QFileDialog,
    QComboBox,
    QLineEdit,
    QMessageBox,
    QCheckBox,
)

import keithley


class CurrentSourceAdapter:
    def __init__(self, device):
        self.device = device
        self.resource = getattr(device, 'device', None)
        self.kind = self._detect_kind(device)
        self.output_enabled = False

    def _detect_kind(self, device):
        if isinstance(device, keithley.Keithley6517B_Mock):
            return 'mock'
        if isinstance(device, keithley.Keithley6517B):
            return '6517B'
        if isinstance(device, keithley.Keithley6430):
            return 'smu'
        return None

    def configure(self, current_range, compliance_voltage=None, nplc=1.0):
        if self.kind is None:
            raise ValueError('Selected device is not supported for current sourcing')
        if self.kind == 'mock':
            return
        if self.resource is None:
            raise RuntimeError('Device resource is not available for configuration')
        try:
            self.resource.write('*CLS')
        except Exception:
            pass

        autorange = current_range is None
        if self.kind == 'smu':
            self.resource.write(":SOUR:FUNC CURR")
            self.resource.write(":SOUR:CURR:MODE FIX")
            if autorange:
                self.resource.write(":SOUR:CURR:RANG:AUTO ON")
            else:
                self.resource.write(f":SOUR:CURR:RANG {current_range}")
            self.resource.write(":SOUR:CURR:LEV 0")
            self.resource.write(":FORM:ELEM VOLT,CURR")
            self.resource.write(":SENS:FUNC 'VOLT'")
            try:
                self.resource.write(f":SENS:VOLT:NPLC {float(nplc)}")
            except Exception:
                pass
            if compliance_voltage is not None:
                try:
                    self.resource.write(f":SENS:VOLT:PROT {compliance_voltage}")
                except Exception:
                    pass
        elif self.kind == '6517B':
            self.resource.write(":SOUR:FUNC CURR")
            self.resource.write(":SOUR:CURR:MODE FIX")
            if autorange:
                self.resource.write(":SOUR:CURR:RANG:AUTO ON")
            else:
                self.resource.write(f":SOUR:CURR:RANG {current_range}")
            self.resource.write(":SOUR:CURR:LEV 0")
            if compliance_voltage is not None:
                try:
                    self.resource.write(f":SENS:VOLT:PROT {compliance_voltage}")
                except Exception:
                    pass
            self.resource.write(":SENS:FUNC 'CURR'")
            try:
                self.resource.write(f":SENS:CURR:NPLC {float(nplc)}")
            except Exception:
                pass

    def set_current(self, value):
        if self.kind == 'mock':
            self.device.voltage_level = value
            if not self.device.output_enabled:
                self.device.output_enabled = True
            return
        if self.resource is None:
            return
        if not self.output_enabled:
            try:
                self.resource.write('OUTP ON')
            except Exception:
                pass
            self.output_enabled = True
        try:
            self.resource.write(f":SOUR:CURR:LEV {value}")
        except Exception as exc:
            raise RuntimeError(f'Failed to program current level: {exc}')

    def read_current(self):
        if hasattr(self.device, 'read_current'):
            try:
                return float(self.device.read_current(autorange=False))
            except Exception:
                pass
        if self.kind == 'mock':
            return float(self.device.read_current())
        if self.resource is not None:
            try:
                self.resource.write(":MEAS:CURR?")
                return float(self.resource.read().split(',')[0])
            except Exception:
                try:
                    self.resource.write(":READ?")
                    return float(self.resource.read().split(',')[1])
                except Exception:
                    return np.nan
        return np.nan

    def read_voltage(self):
        if self.kind == 'mock':
            return 0.0
        if self.resource is not None:
            try:
                self.resource.write(":MEAS:VOLT?")
                return float(self.resource.read().split(',')[0])
            except Exception:
                try:
                    self.resource.write(":READ?")
                    return float(self.resource.read().split(',')[0])
                except Exception:
                    return np.nan
        return np.nan

    def disable(self):
        if self.kind == 'mock':
            try:
                self.device.disable_output()
            except Exception:
                pass
            return
        if self.resource is not None:
            try:
                self.resource.write('OUTP OFF')
            except Exception:
                pass
        if hasattr(self.device, 'disable_output'):
            try:
                self.device.disable_output()
            except Exception:
                pass
        self.output_enabled = False


class VoltageMeterAdapter:
    def __init__(self, device):
        self.device = device
        self.resource = getattr(device, 'device', None)
        self.kind = self._detect_kind(device)

    def _detect_kind(self, device):
        if isinstance(device, keithley.Keithley6517B_Mock):
            return 'mock'
        if isinstance(device, keithley.Keithley6430):
            return 'smu'
        if isinstance(device, keithley.Keithley6514):
            return '6514'
        if isinstance(device, keithley.Keithley2700):
            return '2700'
        if isinstance(device, keithley.Keithley2002):
            return '2002'
        if isinstance(device, keithley.Keithley6517B):
            return '6517B'
        return None

    def configure(self, voltage_range=None, nplc=1.0):
        if self.kind is None:
            raise ValueError('Selected device is not supported for voltage sensing')
        if self.kind == 'mock':
            return
        if self.kind == 'smu':
            if self.resource is None:
                raise RuntimeError('Resource handle missing for voltmeter')
            try:
                self.resource.write('*CLS')
            except Exception:
                pass
            self.resource.write(":SOUR:FUNC CURR")
            self.resource.write(":SOUR:CURR:MODE FIX")
            self.resource.write(":SOUR:CURR:LEV 0")
            self.resource.write(":SENS:FUNC 'VOLT'")
            try:
                self.resource.write(f":SENS:VOLT:NPLC {float(nplc)}")
            except Exception:
                pass
            if voltage_range is None:
                try:
                    self.resource.write(":SENS:VOLT:RANG:AUTO ON")
                except Exception:
                    pass
            else:
                try:
                    self.resource.write(":SENS:VOLT:RANG {}".format(voltage_range))
                except Exception:
                    pass
            self.resource.write(":FORM:ELEM VOLT")
            try:
                self.resource.write('OUTP OFF')
            except Exception:
                pass
        elif self.kind == '6514':
            self.device.set_function('VOLT')
            if voltage_range is None:
                self.device.device.write("VOLT:RANG:AUTO ON")
            else:
                self.device.set_voltage_range(voltage_range)
        elif self.kind == '2700':
            self.device.set_function('VOLT')
            if voltage_range is None:
                self.device.set_voltage_range(auto=True)
            else:
                self.device.set_voltage_range(range_value=voltage_range)
        elif self.kind == '2002':
            self.device.set_function('VOLT:DC')
            try:
                self.device.set_nplc(float(nplc), func='VOLT:DC')
            except Exception:
                pass
            if voltage_range is None:
                self.device.set_range_auto(True, func='VOLT:DC')
            else:
                self.device.set_range(voltage_range, func='VOLT:DC')
        elif self.kind == '6517B':
            if self.resource is None:
                raise RuntimeError('Resource handle missing for voltmeter')
            self.resource.write(":SYST:ZCH OFF")
            self.resource.write(":SENS:FUNC 'VOLT'")
            try:
                self.resource.write(f":SENS:VOLT:NPLC {float(nplc)}")
            except Exception:
                pass
            if voltage_range is None:
                try:
                    self.resource.write("VOLT:RANG:AUTO ON")
                except Exception:
                    pass
            else:
                self.resource.write(f"VOLT:RANG {voltage_range}")
            self.resource.write(":FORM:ELEM VOLT")

    def read_voltage(self):
        if self.kind == 'mock':
            # Mock voltmeter tied to mock source: reuse voltage level
            return getattr(self.device, 'voltage_level', 0.0)
        if self.kind == 'smu':
            if self.resource is None:
                return np.nan
            try:
                self.resource.write(":MEAS:VOLT?")
                return float(self.resource.read().split(',')[0])
            except Exception:
                try:
                    self.resource.write(":READ?")
                    return float(self.resource.read().split(',')[0])
                except Exception:
                    return np.nan
        if self.kind == '6514':
            return float(self.device.read_voltage())
        if self.kind == '2700':
            return float(self.device.read_voltage())
        if self.kind == '2002':
            return float(self.device.meas_voltage_dc())
        if self.kind == '6517B':
            if self.resource is None:
                return np.nan
            try:
                self.resource.write(":READ?")
                return float(self.resource.read().split(',')[0])
            except Exception:
                return np.nan
        return np.nan

    def close(self):
        if hasattr(self.device, 'close'):
            try:
                self.device.close()
            except Exception:
                pass


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
        layout = QVBoxLayout()

        device_list = [' - '.join(entry) for entry in keithley.get_devices_list()]
        device_list.append('Mock')

        source_row = QHBoxLayout()
        self.source_combo = QComboBox()
        self.source_combo.addItems(device_list)
        source_row.addWidget(QLabel('Current source:'))
        source_row.addWidget(self.source_combo)
        layout.addLayout(source_row)

        meter_row = QHBoxLayout()
        self.voltmeter_combo = QComboBox()
        self.voltmeter_combo.addItems(device_list)
        meter_row.addWidget(QLabel('Voltmeter:'))
        meter_row.addWidget(self.voltmeter_combo)
        layout.addLayout(meter_row)

        sample_row = QHBoxLayout()
        self.sample_name_input = QLineEdit()
        sample_row.addWidget(QLabel('Sample name:'))
        sample_row.addWidget(self.sample_name_input)
        layout.addLayout(sample_row)

        contact_row = QHBoxLayout()
        self.contact_label_input = QLineEdit()
        contact_row.addWidget(QLabel('Contact pair label:'))
        contact_row.addWidget(self.contact_label_input)
        layout.addLayout(contact_row)

        source_nplc_row = QHBoxLayout()
        self.source_nplc = QComboBox()
        self.source_nplc.addItems(['0.01', '0.1', '1', '10'])
        source_nplc_row.addWidget(QLabel('Source NPLC:'))
        source_nplc_row.addWidget(self.source_nplc)
        layout.addLayout(source_nplc_row)

        meter_nplc_row = QHBoxLayout()
        self.meter_nplc = QComboBox()
        self.meter_nplc.addItems(['0.01', '0.1', '1', '10'])
        meter_nplc_row.addWidget(QLabel('Meter NPLC:'))
        meter_nplc_row.addWidget(self.meter_nplc)
        layout.addLayout(meter_nplc_row)

        current_range_row = QHBoxLayout()
        self.current_range_combo = QComboBox()
        current_ranges = list((2 * 10.0 ** np.arange(-9.0, -1.0, 1.0)).astype(str))
        self.current_range_combo.addItems(['Auto-range'] + current_ranges)
        current_range_row.addWidget(QLabel('Current range (A):'))
        current_range_row.addWidget(self.current_range_combo)
        layout.addLayout(current_range_row)

        compliance_row = QHBoxLayout()
        self.compliance_input = QDoubleSpinBox()
        self.compliance_input.setRange(0.0, 1000.0)
        self.compliance_input.setValue(10.0)
        self.compliance_input.setDecimals(3)
        compliance_row.addWidget(QLabel('Compliance voltage (V):'))
        compliance_row.addWidget(self.compliance_input)
        layout.addLayout(compliance_row)

        voltage_range_row = QHBoxLayout()
        self.voltage_range_combo = QComboBox()
        voltage_ranges = ['0.002', '0.02', '0.2', '2', '20', '200', '1000']
        self.voltage_range_combo.addItems(['Auto-range'] + voltage_ranges)
        voltage_range_row.addWidget(QLabel('Voltage range (V):'))
        voltage_range_row.addWidget(self.voltage_range_combo)
        layout.addLayout(voltage_range_row)

        current_limits_row = QHBoxLayout()
        self.current_min_input = QDoubleSpinBox()
        self.current_min_input.setRange(-1.0, 1.0)
        self.current_min_input.setDecimals(6)
        self.current_min_input.setValue(-1e-5)
        self.current_max_input = QDoubleSpinBox()
        self.current_max_input.setRange(-1.0, 1.0)
        self.current_max_input.setDecimals(6)
        self.current_max_input.setValue(1e-5)
        current_limits_row.addWidget(QLabel('Current min (A):'))
        current_limits_row.addWidget(self.current_min_input)
        current_limits_row.addWidget(QLabel('Current max (A):'))
        current_limits_row.addWidget(self.current_max_input)
        layout.addLayout(current_limits_row)

        current_step_row = QHBoxLayout()
        self.current_step_input = QDoubleSpinBox()
        self.current_step_input.setRange(1e-9, 1e-1)
        self.current_step_input.setDecimals(9)
        self.current_step_input.setValue(1e-6)
        current_step_row.addWidget(QLabel('Current step (A):'))
        current_step_row.addWidget(self.current_step_input)
        layout.addLayout(current_step_row)

        run_row = QHBoxLayout()
        self.nruns_input = QSpinBox()
        self.nruns_input.setRange(1, 1000)
        self.nruns_input.setValue(2)
        run_row.addWidget(QLabel('N runs (2 = up/down):'))
        run_row.addWidget(self.nruns_input)
        layout.addLayout(run_row)

        collection_row = QHBoxLayout()
        self.collection_time_input = QSpinBox()
        self.collection_time_input.setRange(10, 10000)
        self.collection_time_input.setValue(1000)
        collection_row.addWidget(QLabel('Averaging time (ms):'))
        collection_row.addWidget(self.collection_time_input)
        layout.addLayout(collection_row)

        self.average_checkbox = QCheckBox('Average source current and measured voltage')
        self.average_checkbox.setChecked(True)
        layout.addWidget(self.average_checkbox)

        button_row = QHBoxLayout()
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_measurement)
        self.stop_button = QPushButton('Stop')
        self.stop_button.clicked.connect(self.stop_measurement)
        self.stop_button.setEnabled(False)
        self.current_label = QLabel('Current: 0 A')
        self.current_label.setAlignment(Qt.AlignCenter)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.current_label)
        button_row.addWidget(self.stop_button)
        layout.addLayout(button_row)

        pg.setConfigOption('background', 'w')
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('w')
        self.iv_plot = self.plot_widget.addPlot(title='Voltage vs Current')
        self.iv_plot.showGrid(x=True, y=True)
        self.rh_plot = self.plot_widget.addPlot(title='Resistance vs Current')
        self.rh_plot.showGrid(x=True, y=True)
        self.rh_plot.setLabel('left', 'Resistance (Ohm)')
        self.rh_plot.setLabel('bottom', 'Current (A)')
        layout.addWidget(self.plot_widget)

        self.status_label = QLabel('')
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def _parse_current_range(self):
        selected = self.current_range_combo.currentText()
        if selected == 'Auto-range':
            return None
        try:
            return float(eval(selected))
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

    def start_measurement(self):
        if self.timer.isActive():
            return
        current_min = self.current_min_input.value()
        current_max = self.current_max_input.value()
        current_step = self.current_step_input.value()
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
        try:
            self.source_device = self._resolve_device(source_text, self.source_nplc.currentText())
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to open source device: {exc}")
            return
        try:
            self.voltage_device = self._resolve_device(meter_text, self.meter_nplc.currentText())
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to open voltmeter: {exc}")
            return

        try:
            self.source_adapter = CurrentSourceAdapter(self.source_device)
            self.source_adapter.configure(
                current_range=self._parse_current_range(),
                compliance_voltage=self.compliance_input.value(),
                nplc=float(self.source_nplc.currentText()),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to configure source: {exc}")
            self.source_adapter = None
            return

        try:
            self.voltmeter_adapter = VoltageMeterAdapter(self.voltage_device)
            self.voltmeter_adapter.configure(
                voltage_range=self._parse_voltage_range(),
                nplc=float(self.meter_nplc.currentText()),
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

        self.iv_plot.clear()
        self.rh_plot.clear()
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
        currents = []
        voltages = []
        avg_mode = self.average_checkbox.isChecked()
        sleep_window = max(0.002, self.collection_time / 1000.0 / 10.0)
        while self.elapsed_timer.elapsed() - start_mark < self.collection_time:
            current_value = self.source_adapter.read_current()
            voltage_value = self.voltmeter_adapter.read_voltage()
            currents.append(current_value)
            voltages.append(voltage_value)
            time.sleep(min(0.05, sleep_window))

        if currents and avg_mode:
            mean_current = float(np.nanmean(currents))
        elif currents:
            mean_current = float(currents[-1])
        else:
            mean_current = np.nan
        if voltages and avg_mode:
            mean_voltage = float(np.nanmean(voltages))
        elif voltages:
            mean_voltage = float(voltages[-1])
        else:
            mean_voltage = np.nan

        resistance = np.nan
        if mean_current is not None and not np.isnan(mean_current) and abs(mean_current) > 1e-12:
            resistance = mean_voltage / mean_current

        self.measurements.append(
            (
                self.current_target,
                mean_current,
                mean_voltage,
                resistance,
                self.direction,
                time.time(),
            )
        )
        self.update_plots()
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
        currents = [m[1] for m in self.measurements if not np.isnan(m[1])]
        voltages = [m[2] for m in self.measurements if not np.isnan(m[2])]
        resistances = [m[3] for m in self.measurements if not np.isnan(m[3])]
        if currents and voltages:
            self.iv_plot.plot(currents, voltages, pen=pg.mkPen(color='b', width=2), clear=True)
            self.iv_plot.plot(currents, voltages, pen=None, symbol='o', symbolSize=4, symbolBrush=(0, 0, 255, 160))
            self.iv_plot.setLabel('left', 'Voltage (V)')
            self.iv_plot.setLabel('bottom', 'Current (A)')
        if currents and resistances:
            self.rh_plot.plot(currents, resistances, pen=pg.mkPen(color='r', width=2), clear=True)
            self.rh_plot.plot(currents, resistances, pen=None, symbol='o', symbolSize=4, symbolBrush=(255, 0, 0, 160))

        if resistances:
            self.status_label.setText(f'Latest resistance: {resistances[-1]:.3e} Ohm')

    def get_dataframe(self):
        return pd.DataFrame(
            self.measurements,
            columns=[
                'TargetCurrent',
                'MeasuredCurrent',
                'MeasuredVoltage',
                'Resistance',
                'Direction',
                'Timestamp',
            ],
        )

    def export_to_csv(self):
        sample_name = self.sample_name_input.text() or 'sample'
        contact_label = self.contact_label_input.text() or 'contacts'
        sample_dir = op.join(self.folder, self.date, sample_name)
        data_dir = op.join(sample_dir, 'data')
        if not op.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        df = self.get_dataframe()
        name = f'{sample_name}_{contact_label}_{self.current_min}A_{self.current_max}A_{self.collection_time}ms'
        df.to_csv(op.join(data_dir, f'FourProbe_{name}_{self.start_time}.csv'), index=False)

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
        plt.plot(df['MeasuredCurrent'], df['MeasuredVoltage'], 'o-', markersize=3)
        plt.xlabel('Current (A)')
        plt.ylabel('Voltage (V)')
        plt.grid(True)
        plt.savefig(op.join(plot_dir, f'FourProbe_IV_{name}_{self.start_time}.png'), dpi=300)
        fig1.clf()

        fig2 = plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(df['MeasuredCurrent'], df['Resistance'], 'o-', markersize=3)
        plt.xlabel('Current (A)')
        plt.ylabel('Resistance (Ohm)')
        plt.grid(True)
        plt.savefig(op.join(plot_dir, f'FourProbe_R_{name}_{self.start_time}.png'), dpi=300)
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
