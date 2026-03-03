import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QFileDialog, QComboBox, QLineEdit, QMessageBox, QProgressBar)
from PyQt5.QtCore import QTimer, QElapsedTimer, Qt
import keithley
from ui_helpers import ProgressEta, refresh_device_combos, apply_standard_window_style
import pandas as pd
import time 
import datetime
import matplotlib.pyplot as plt
import os.path as op
import os 
import matplotlib
import gc
matplotlib.use('Agg')

def derivative(arr, step):
    if len(arr) == 1:
        return arr
    outp = np.zeros(arr.shape, dtype=np.float64)
    for i in range(len(arr)):
        if i == 0:
            out = arr[1] - arr[0]
        elif i == len(arr) - 1:
            out = arr[-1] - arr[-2]
        else:
            out = (arr[i + 1] - arr[i - 1]) / 2
        outp[i] = out / step
    return outp

class FETMAPRegime(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize simulation variables
        self.device = None
        self.device_gate = None
        self.device_sd = None
        self.sample_name = ''
        # self.device = keithley.Keithley6517B('GPIB0::27::INSTR')
        self.device_address = ''
        self.voltage_min = 0
        self.voltage_max = 1
        self.compliance_current = 1e-3
        self.voltage_step = 0.1
        self.collection_time = 10  # in milliseconds
        self.settle_time_ms = 0
        self.measurements = []
        self.noise_data = []  # For storing I(t) during measurements
        self.n_runs = 2
        self.current_range = 1e-8
        self.sd_direction = 1
        self.gate_direction = 1
        self.gate_settle_until = 0
        # Setup GUI components
        self.initUI()
        self.sd_direction = 1

        # Setup timer for measurements
        self.timer = QTimer()
        self.timer.timeout.connect(self.perform_measurement)

        # Timer for signal integration
        self.elapsed_timer = QElapsedTimer()
        # date & make folder
        self.date = str(datetime.date.today())
        self.folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        if not self.folder:
            self.folder = os.getcwd()
        if op.exists(op.join(self.folder, self.date)) == False:
            os.makedirs(op.join(self.folder, self.date))
        self.start_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S') 

    def initUI(self):
        
        self.setWindowTitle('Keithley FETMAP')
        self.resize(1500, 920)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        device_list = [' - '.join(i) for i in keithley.get_devices_list()]
        device_list.append('Mock')

        # keithley 
        device_sd_address_layout = QHBoxLayout()
        self.device_sd_address_input = QComboBox()
        self.device_sd_address_input.addItems(device_list)
        self.refresh_button = QPushButton('Refresh GPIB')
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.refresh_status_label = QLabel('Idle')
        self.refresh_status_label.setStyleSheet("color: #64748b;")
        device_sd_address_layout.addWidget(QLabel('Source-Drain Device address:'))
        device_sd_address_layout.addWidget(self.device_sd_address_input)
        device_sd_address_layout.addWidget(self.refresh_button)
        device_sd_address_layout.addWidget(self.refresh_status_label)
        layout.addLayout(device_sd_address_layout)
        # keithley 
        device_gate_address_layout = QHBoxLayout()
        self.device_gate_address_input = QComboBox()
        self.device_gate_address_input.addItems(device_list)
        device_gate_address_layout.addWidget(QLabel('Gate Device address:'))
        device_gate_address_layout.addWidget(self.device_gate_address_input)
        layout.addLayout(device_gate_address_layout)
        # sample name
        sample_name_layout = QHBoxLayout()
        self.sample_name_input = QLineEdit()
        sample_name_layout.addWidget(QLabel('Sample name:'))
        sample_name_layout.addWidget(self.sample_name_input)
        layout.addLayout(sample_name_layout)
        # Current range input
        current_range_layout = QHBoxLayout()
        self.current_range_input = QComboBox()
        self.current_range_input.addItems(list((2 * 10.0**np.arange(-12.0, -4.0, 1.0)).astype(str)) + ['Auto-range'])
        current_range_layout.addWidget(QLabel('Current range (A):'))
        current_range_layout.addWidget(self.current_range_input)
        layout.addLayout(current_range_layout)
        # NPLC input
        nplc_layout = QHBoxLayout()
        self.nplc_input = QComboBox()
        self.nplc_input.addItems(['0.01', '0.1', '1', '10'])
        nplc_layout.addWidget(QLabel('NPLC (1 = 20 ms):'))
        nplc_layout.addWidget(self.nplc_input)
        layout.addLayout(nplc_layout)
        # Voltage range input
        voltage_range_layout = QHBoxLayout()
        self.voltage_min_input = QDoubleSpinBox()
        self.voltage_min_input.setRange(-200, 200)
        self.voltage_min_input.setDecimals(2)
        self.voltage_min_input.setValue(-1)
        self.voltage_max_input = QDoubleSpinBox()
        self.voltage_max_input.setRange(-200, 200)
        self.voltage_max_input.setDecimals(2)
        self.voltage_max_input.setValue(20)
        self.voltage_sd_min_input = QDoubleSpinBox()
        self.voltage_sd_min_input.setRange(-200, 200)
        self.voltage_sd_min_input.setDecimals(4)
        self.voltage_sd_min_input.setValue(0.01)
        self.voltage_sd_max_input = QDoubleSpinBox()
        self.voltage_sd_max_input.setRange(-200, 200)
        self.voltage_sd_max_input.setDecimals(4)
        self.voltage_sd_max_input.setValue(0.01)
        self.voltage_sd_step_input = QDoubleSpinBox()
        voltage_range_layout.addWidget(QLabel('Gate Voltage min (V):'))
        voltage_range_layout.addWidget(self.voltage_min_input)
        voltage_range_layout.addWidget(QLabel('Gate Voltage max (V):'))
        voltage_range_layout.addWidget(self.voltage_max_input)
        voltage_range_layout.addWidget(QLabel('Source-Drain min Voltage (V):'))
        voltage_range_layout.addWidget(self.voltage_sd_min_input)
        voltage_range_layout.addWidget(QLabel('Source-Drain max Voltage (V):'))
        voltage_range_layout.addWidget(self.voltage_sd_max_input)
        layout.addLayout(voltage_range_layout)

        # Voltage step input
        voltage_step_layout = QHBoxLayout()
        self.voltage_step_input = QDoubleSpinBox()
        self.voltage_step_input.setRange(0.0001, 100)
        self.voltage_step_input.setValue(0.01)
        self.voltage_step_input.setDecimals(4)
        self.voltage_sd_step_input.setRange(0.000001, 100)
        self.voltage_sd_step_input.setDecimals(6)
        self.voltage_sd_step_input.setValue(0.01)
        voltage_step_layout.addWidget(QLabel('Gate Voltage step (V):'))
        voltage_step_layout.addWidget(self.voltage_step_input)
        voltage_step_layout.addWidget(QLabel('Source-Drain step Voltage (V):'))
        voltage_step_layout.addWidget(self.voltage_sd_step_input)
        layout.addLayout(voltage_step_layout)

        # nruns input
        nruns_layout = QHBoxLayout()
        self.nruns_input = QSpinBox()
        self.nruns_input.setRange(1, 1000)
        self.nruns_input.setValue(2)
        nruns_layout.addWidget(QLabel('N runs (2 is 1 up 1 down)'))
        nruns_layout.addWidget(self.nruns_input)
        layout.addLayout(nruns_layout)

        # Compliance current input
        compliance_layout = QHBoxLayout()
        self.compliance_input = QLineEdit()
        compliance_layout.addWidget(QLabel('Compliance current (A):'))
        compliance_layout.addWidget(self.compliance_input)
        layout.addLayout(compliance_layout)

        # Collection time input
        collection_time_layout = QHBoxLayout()
        self.collection_time_input = QSpinBox()
        self.collection_time_input.setRange(1, 5000)  # Time in ms
        self.collection_time_input.setValue(1000)
        collection_time_layout.addWidget(QLabel('Collection time (ms):'))
        collection_time_layout.addWidget(self.collection_time_input)
        layout.addLayout(collection_time_layout)

        # Settle time after gate step
        settle_time_layout = QHBoxLayout()
        self.settle_time_input = QSpinBox()
        self.settle_time_input.setRange(0, 20000)
        self.settle_time_input.setValue(0)
        settle_time_layout.addWidget(QLabel('Gate settle time (ms):'))
        settle_time_layout.addWidget(self.settle_time_input)
        layout.addLayout(settle_time_layout)

        # Start/Stop button
        button_layout = QHBoxLayout()
        self.start_button = QPushButton('Start Measurement')
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_measurement)
        self.stop_button = QPushButton('Stop Measurement')
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(self.stop_measurement)
        self.voltage_now = QLabel('0 V')
        self.voltage_now.setAlignment(Qt.AlignCenter)
        self.voltage_now.setStyleSheet("background-color: lightgray")
        self.voltage_now.adjustSize()
        button_layout.addWidget(self.start_button)
        # button_layout.addWidget(QLabel('Voltage now: '))
        button_layout.addWidget(self.voltage_now)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel('Progress: 0/0')
        self.eta_label = QLabel('ETA: --')
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.eta_label)

        # Plot area for I(V), abs(I(V)) 
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('w')

        # I(V) plot
        pg.setConfigOption('background', 'w')
        self.iv_plot = self.plot_widget.addPlot(title="I(Vsd)")
        self.iv_plot.showGrid(x=True, y=True, alpha=0.12)
        self.iv_plot.setLabel('left', 'Current', units='A')
        self.iv_plot.setLabel('bottom', 'Source-drain voltage', units='V')
        self.iv_plot.getAxis('left').enableAutoSIPrefix(True)
        self.iv_plot.getAxis('bottom').enableAutoSIPrefix(True)
        self.leakage_plot = self.plot_widget.addPlot(title="Gate (Leakage) I(t)")
        self.leakage_plot.showGrid(x=True, y=True, alpha=0.12)
        self.leakage_plot.setLabel('left', 'Leakage current', units='A')
        self.leakage_plot.setLabel('bottom', 'Time', units='s')
        self.leakage_plot.getAxis('left').enableAutoSIPrefix(True)
        self.leakage_plot.getAxis('bottom').enableAutoSIPrefix(True)

        layout.addWidget(self.plot_widget)

        # Plot area for I(t)
        self.time_plot_widget = pg.GraphicsLayoutWidget()
        self.time_plot_widget.setBackground('w')
        pg.setConfigOption('background', 'w')
        self.i_plot = self.time_plot_widget.addPlot(title="I(sd, vg)")
        self.i_plot.showGrid(x=True, y=True, alpha=0.12)
        self.i_plot.setLabel('left', 'Current', units='A')
        self.i_plot.setLabel('bottom', 'Source-drain voltage', units='V')
        self.i_plot.getAxis('left').enableAutoSIPrefix(True)
        self.i_plot.getAxis('bottom').enableAutoSIPrefix(True)
        layout.addWidget(self.time_plot_widget)
        self.setLayout(layout)
        apply_standard_window_style(self)
        self.refresh_devices()
        self.progress_tracker = ProgressEta(self.progress_bar, self.eta_label, self.progress_label)
        self.total_steps = 0
        self.completed_steps = 0
        self.gate_path = []
        self.gate_path_index = 0

    def refresh_devices(self):
        self.refresh_button.setEnabled(False)
        self.refresh_button.setText('Refreshing...')
        self.refresh_status_label.setText('Scanning GPIB...')
        QApplication.processEvents()
        try:
            devices = refresh_device_combos(
                keithley,
                [self.device_sd_address_input, self.device_gate_address_input],
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

    def estimate_sd_points(self):
        if self.voltage_sd_step <= 0:
            return 1
        runs = int(self.nruns_input.value())
        direction = 1
        v = self.voltage_sd_min
        steps = 0
        max_iter = 5_000_000
        while steps < max_iter:
            steps += 1
            v += self.voltage_sd_step * direction
            if v > self.voltage_sd_max:
                v = self.voltage_sd_max
                direction *= -1
                runs -= 1
            if v < self.voltage_sd_min:
                v = self.voltage_sd_min
                direction *= -1
                runs -= 1
            if runs <= 0 and abs(v) <= self.voltage_sd_step / 10:
                break
        return max(1, steps)

    def estimate_total_steps(self):
        return max(1, len(self.gate_path) * self.estimate_sd_points())

    def _segment_points(self, start, stop, step):
        if np.isclose(start, stop, atol=1e-15):
            return [float(start)]
        n = int(np.ceil(abs(stop - start) / max(step, 1e-12)))
        return [float(v) for v in np.linspace(start, stop, max(2, n + 1))]

    def _append_segment(self, out, seg):
        for v in seg:
            if not out or not np.isclose(out[-1], v, atol=1e-12):
                out.append(float(v))

    def _build_gate_path(self):
        step = float(self.voltage_step)
        if step <= 0:
            return [0.0]
        path = []
        self._append_segment(path, self._segment_points(0.0, self.voltage_min, step))
        self._append_segment(path, self._segment_points(self.voltage_min, self.voltage_max, step))
        self._append_segment(path, self._segment_points(self.voltage_max, self.voltage_min, step))
        self._append_segment(path, self._segment_points(self.voltage_min, 0.0, step))
        return path if path else [0.0]

    def start_measurement(self):
        self.voltage_min = self.voltage_min_input.value()
        self.voltage_max = self.voltage_max_input.value()
        self.voltage_step = self.voltage_step_input.value()
        # self.voltage_sd = self.voltage_sd_input.value()
        self.voltage_sd_min = self.voltage_sd_min_input.value()
        self.voltage_sd_max = self.voltage_sd_max_input.value()
        self.voltage_sd_step = self.voltage_sd_step_input.value()
        try:
            self.compliance_current = float(eval(self.compliance_input.text()))
        except Exception:
            QMessageBox.critical(None, "Error", f"Compliance current is incorrect : {self.compliance_input.text()}")
            return
        self.nplc = self.nplc_input.currentText()
        self.collection_time = self.collection_time_input.value()
        self.settle_time_ms = int(self.settle_time_input.value())
        self.n_runs = int(self.nruns_input.value())
        self.device_gate_address = self.device_gate_address_input.currentText()
        self.device_sd_address = self.device_sd_address_input.currentText()
        self.sample_name = self.sample_name_input.text().strip() or 'sample'
        self.current_range = self.current_range_input.currentText()
        print(f"Gate device : {self.device_gate_address}")
        print(f"Source-Drain device : {self.device_sd_address}")
        if self.device_gate_address == self.device_sd_address and (self.device_gate_address != 'Mock' or self.device_sd_address != 'Mock'):
            QMessageBox.critical(None, "Error", f"Choose different devices for gate and source-drain.")
            return
        self.device_gate = keithley.get_device(self.device_gate_address, nplc=self.nplc)
        self.device_sd = keithley.get_device(self.device_sd_address, nplc=self.nplc)

        if self.device_gate == None:
            QMessageBox.critical(None, "Error", f"Gate device not found")
            return

        if self.device_sd == None:
            QMessageBox.critical(None, "Error", f"Source-drain device not found")
            return

        if isinstance(self.device_sd, keithley.Keithley6430):
            try:
                self.device_sd.set_complicance_current(self.compliance_current)
            except Exception as exc:
                QMessageBox.critical(None, "Error", f"Failed to set SD compliance: {exc}")
                return

        # Clear previous measurements and noise data
        self.measurements = []
        self.noise_data = []
        self.gate_path = self._build_gate_path()
        self.gate_path_index = 0
        self.current_voltage = float(self.gate_path[0])
        self.current_voltage_sd = self.voltage_sd_min
        self.start_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')
        self.sd_direction = 1
        self.gate_direction = 1
        if type(self.device_gate) == keithley.Keithley6517B:
            self.device_gate.set_voltage_range(max(abs(self.voltage_min), abs(self.voltage_max)))
        if self.current_range != 'Auto-range':
            try:
                current_range_val = float(eval(self.current_range))
            except Exception:
                QMessageBox.critical(None, "Error", f"Invalid current range: {self.current_range}")
                return
            if isinstance(self.device_sd, keithley.Keithley6430) and current_range_val > self.compliance_current:
                QMessageBox.warning(
                    None,
                    "Warning",
                    f"Requested current range {current_range_val} A exceeds compliance "
                    f"{self.compliance_current} A. Capping range to compliance.",
                )
                current_range_val = self.compliance_current
            self.device_sd.set_current_range(current_range_val)
        self.timer.start(50)  # Update every 50 ms
        self.elapsed_timer.start()  # Start the elapsed time for integration

        # Clear plots
        self.iv_plot.clear()
        self.leakage_plot.clear()
        self.sd_direction = 1

        self.device_gate.set_voltage(self.current_voltage)
        self.device_sd.set_voltage(self.current_voltage_sd)
        self.gate_settle_until = self.elapsed_timer.elapsed() + self.settle_time_ms
        self.total_steps = self.estimate_total_steps()
        self.completed_steps = 0
        self.progress_tracker.start(self.total_steps)

    def stop_measurement(self):
        self.current_voltage = 0
        self.current_voltage_sd = 0
        if self.device_gate:
            try:
                self.device_gate.set_voltage(0)
                self.device_gate.disable_output()
            except Exception:
                pass
        if self.device_sd:
            try:
                self.device_sd.set_voltage(0)
                self.device_sd.disable_output()
            except Exception:
                pass
        self.voltage_now.setText('Vg = {:.2f} V; Vsd = {:.2f} V'.format(self.current_voltage, self.current_voltage_sd))
        self.voltage_now.adjustSize()
        self.timer.stop()
        try:
            self.make_plot()
            self.export_to_csv()
        except Exception as exc:
            QMessageBox.warning(self, "Warning", f"Failed to save data: {exc}")

    def perform_measurement(self):
        if self.device_sd is None or self.device_gate is None:
            self.stop_measurement()
            return
        if self.elapsed_timer.elapsed() < self.gate_settle_until:
            remaining = max(0, self.gate_settle_until - self.elapsed_timer.elapsed())
            self.voltage_now.setText(
                'Vg = {:.2f} V; Vsd = {:.2f} V; settling {} ms'.format(
                    self.current_voltage, self.current_voltage_sd, int(remaining)
                )
            )
            self.voltage_now.adjustSize()
            return
        # Perform signal integration over the collection time
        self.device_sd.set_voltage(self.current_voltage_sd)
        start_time = self.elapsed_timer.elapsed()
        total_current = 0
        num_measurements = 0
        noise_currents = []  # Store the current noise

        # self.device_gate.set_voltage(self.current_voltage)
        self.voltage_now.setText('Vg = {:.2f} V; Vsd = {:.2f} V'.format(self.current_voltage, self.current_voltage_sd))
        self.voltage_now.adjustSize()

        p1 = None
        if len(self.measurements) > 0:
            p1 = self.measurements[-1][2]
            if p1 == 0: 
                p1 = None
        if self.current_range == 'Auto-range':
            current = keithley.auto_range(self.device_sd, p1=p1, compl=self.compliance_current)
        total_current += self.device_sd.read_current()
        num_measurements += 1
            # print(f'total current at the start (p0) : {total_current}')
        while self.elapsed_timer.elapsed() - start_time < self.collection_time:
            # Set the voltage and get the current multiple times to average
            current = self.device_sd.read_current(autorange=False)
            total_current += current
            noise_currents.append(current)
            num_measurements += 1
        
        # Calculate the average current during the collection time
        if num_measurements > 0:
            average_current = total_current / num_measurements
        else:
            average_current = 0

        leakage = keithley.auto_range(self.device_gate, p1 = None if len(self.measurements) == 0 else self.measurements[-1][3])
        # Check compliance current
        
        if abs(average_current) >= self.compliance_current or abs(leakage) >= self.compliance_current:
            # self.device.set_voltage(0)  # Set voltage to 0 if compliance exceeded
            self.stop_measurement()
            QMessageBox.critical(None, "Error", "Compliance current or current range exceeded")
            return

        # Store the data (voltage, average current)
        self.measurements.append((self.current_voltage_sd, self.current_voltage, average_current, leakage, self.sd_direction, time.time()))
        self.noise_data.append(noise_currents)

        # Update plots
        self.update_plots()
        self.completed_steps += 1
        self.progress_tracker.step(
            self.completed_steps,
            extra_text=f"Vg={self.current_voltage:.3f} V, Vsd={self.current_voltage_sd:.3f} V",
        )

        # Move to the next voltage step
        self.current_voltage_sd += self.voltage_sd_step * self.sd_direction

        if self.current_voltage_sd > self.voltage_sd_max:
            self.current_voltage_sd = self.voltage_sd_max
            self.sd_direction *= -1
            self.n_runs -= 1

        if self.current_voltage_sd < self.voltage_sd_min:
            self.current_voltage_sd = self.voltage_sd_min
            self.sd_direction *= -1
            self.n_runs -= 1

        if self.n_runs <= 0 and abs(self.current_voltage_sd) <= self.voltage_sd_step/10:
            self.current_voltage_sd = 0
            try:
                self.device_sd.set_voltage(0)
            except Exception:
                pass
            self.gate_voltage_step()


    def gate_voltage_step(self):
        if self.gate_path_index >= len(self.gate_path) - 1:
            self.stop_measurement()
            return self.current_voltage
        self.gate_path_index += 1
        self.current_voltage = float(self.gate_path[self.gate_path_index])
        self.device_gate.set_voltage(self.current_voltage)
        self.gate_settle_until = self.elapsed_timer.elapsed() + self.settle_time_ms
        self.n_runs = int(self.nruns_input.value())
        self.sd_direction = 1
        self.current_voltage_sd = self.voltage_sd_min
        return self.current_voltage

    def update_plots(self):
        # Get I(V) and abs(I(V)) data
        voltages = np.array([m[1] for m in self.measurements])
        voltage_sd = np.array([m[0] for m in self.measurements])
        currents = np.array([m[2] for m in self.measurements])
        currents_leak = np.array([m[3] for m in self.measurements])
        times = np.array([m[5] for m in self.measurements])
        # Update I(Vg) plot
        self.iv_plot.plot(voltage_sd[voltages == self.current_voltage], currents[voltages == self.current_voltage], pen=pg.mkPen(color='b', width=2), clear=True)
        self.iv_plot.plot(voltage_sd[voltages == self.current_voltage], currents[voltages == self.current_voltage], pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0, 0, 255, 255), clear=False)

        # Update |I(V)| plot
        time_axis = times - times[0] if len(times) > 0 else times
        self.leakage_plot.plot(time_axis, currents_leak, pen=pg.mkPen(color='r', width=2), clear=True)
        self.leakage_plot.plot(time_axis, currents_leak, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(255, 0, 0, 255), clear=False)


        # update I(Vsd) plot
        self.i_plot.plot(voltage_sd, currents, pen=pg.mkPen(color='r', width=2), clear=True)
        self.i_plot.plot(voltage_sd, currents, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(255, 0, 0, 255), clear=False)

    def export_to_csv(self):

        sample_dir = op.join(self.folder, self.date, self.sample_name)
        if not op.exists(sample_dir):
            os.makedirs(sample_dir)
        if not op.exists(op.join(sample_dir, 'data')):
            os.makedirs(op.join(sample_dir, 'data'))
        name = f'{self.sample_name}_{self.voltage_min}V_{self.voltage_max}V_{self.voltage_sd_min}V_{self.voltage_sd_max}V_{self.collection_time}ms'
        df = self.get_pandas_data()
        df.to_csv(op.join(sample_dir, 'data', f'FETMAP_{name}_{self.start_time}.data'), index=False)


    def get_pandas_data(self):
        df = pd.DataFrame(self.measurements, columns=['Voltage_sd', 'Voltage_g', 'Current', 'Leakage', 'Direction', 'Timestamp'])
        return df

    def make_plot(self):
        sample_dir = op.join(self.folder, self.date, self.sample_name)
        if not op.exists(sample_dir):
            os.makedirs(sample_dir)
        plot_dir = op.join(sample_dir, 'plots')
        if not op.exists(plot_dir):
            os.makedirs(plot_dir)
        name = f'{self.sample_name}_{self.voltage_min}V_{self.voltage_max}V_{self.voltage_sd_min}V_{self.voltage_sd_max}V_{self.collection_time}ms'
        df = self.get_pandas_data()
        if df.empty:
            return

        unique_vg = sorted(df['Voltage_g'].dropna().unique())
        if unique_vg:
            fig1 = plt.figure(figsize=(10, 6), dpi=300)
            plt.ticklabel_format(axis='y', style='scientific')
            cmap = plt.get_cmap('viridis')
            denom = max(len(unique_vg) - 1, 1)
            for idx, vg in enumerate(unique_vg):
                subset = df[df['Voltage_g'] == vg].sort_values('Voltage_sd')
                color = cmap(idx / denom)
                plt.plot(subset['Voltage_sd'], subset['Current'], 'o-', markersize=2, color=color, alpha=0.7)
            plt.xlabel('Source-Drain Voltage (V)')
            plt.ylabel('Current (A)')
            plt.savefig(op.join(plot_dir, f'FETMAP_IVsd_{name}_{self.start_time}.png'), dpi=300)
            fig1.clf()
            matplotlib.pyplot.close(fig1)

        fig2 = plt.figure(figsize=(10, 6), dpi=300)
        plt.ticklabel_format(axis='y', style='scientific')
        leakage_by_gate = df.groupby('Voltage_g', as_index=False)['Leakage'].mean()
        plt.plot(leakage_by_gate['Voltage_g'], leakage_by_gate['Leakage'], 'o-', markersize=3, color='red', alpha=0.7)
        plt.xlabel('Gate Voltage (V)')
        plt.ylabel('Leakage current (A)')
        plt.savefig(op.join(plot_dir, f'FETMAP_Leakage_{name}_{self.start_time}.png'), dpi=300)
        fig2.clf()

        matplotlib.pyplot.close(fig2)
        plt.close('all')
        gc.collect()
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FETMAPRegime()
    window.show()
    sys.exit(app.exec_())
