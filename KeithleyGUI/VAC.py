import sys
import matplotlib
import matplotlib.pyplot
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QPushButton, QFileDialog, QComboBox,
    QLineEdit, QMessageBox, QCheckBox, QProgressBar
)
from PyQt5.QtCore import QTimer, QElapsedTimer, Qt
import keithley
from ui_helpers import (
    ProgressEta,
    refresh_device_combos,
    apply_standard_window_style,
    parse_numeric_text,
)
import pandas as pd
import time
import datetime
import json
import matplotlib.pyplot as plt
import os.path as op
import os
import gc
matplotlib.use('Agg')


class VACRegime(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize simulation variables
        self.device = None
        self.sample_name = ''
        self.device_address = ''
        self.voltage_min = 0
        self.voltage_max = 1
        self.compliance_current = 1e-3
        self.voltage_step = 0.1
        self.collection_time = 10  # in milliseconds
        self.measurements = []
        self.noise_data = []  # For storing I(t) during measurements
        self.n_runs = 2
        self.current_range = 1e-8
        self.manual_current_range = None
        self.direction = 1
        self.initial_sweep_direction = 1
        self.active_compliance = 1
        # Polarity-aware compliance switching helpers
        self._last_comp_polarity = None  # 'neg' or 'pos'
        self.prev_voltage = 0.0
        self.compliance_relaxed_factor = 10.0  # compliance multiplier when a side is unchecked
        self.measurement_metadata = {}

        # Setup GUI components
        self.initUI()

        # Setup timer for measurements
        self.timer = QTimer()
        self.timer.timeout.connect(self.perform_measurement)

        # Timer for signal integration
        self.elapsed_timer = QElapsedTimer()
        # date & make folder
        self.date = str(datetime.date.today())
        self.folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if self.folder and op.exists(op.join(self.folder, self.date)) == False:
            os.makedirs(op.join(self.folder, self.date))
        self.start_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')

    def _get_sample_name(self):
        sample_name = self.sample_name_input.text().strip() or self.sample_name.strip() or 'sample'
        self.sample_name = sample_name
        return sample_name

    def _ensure_directory(self, path, label):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Failed to create {label} directory:\n{path}\n\n{exc}") from exc
        if not op.isdir(path):
            raise RuntimeError(f"{label} directory was not created:\n{path}")
        return path

    def _get_output_directories(self, leaf_dir=None):
        if not self.folder:
            raise RuntimeError("No output folder selected.")
        date_dir = self._ensure_directory(op.join(self.folder, self.date), "date")
        sample_name = self._get_sample_name()
        sample_dir = self._ensure_directory(op.join(date_dir, sample_name), f"sample '{sample_name}'")
        if leaf_dir is None:
            return sample_name, sample_dir
        leaf_path = self._ensure_directory(op.join(sample_dir, leaf_dir), leaf_dir)
        return sample_name, sample_dir, leaf_path

    def _get_device_metadata(self):
        selection_text = str(self.device_address or self.device_address_input.currentText()).strip()
        device_metadata = {
            'selection_text': selection_text,
            'resource': keithley._extract_resource_token(selection_text) if selection_text else '',
            'reported_model': '',
            'reported_driver_class': '',
            'reported_idn': '',
            'vendor': '',
            'model': '',
            'serial': '',
            'firmware': '',
            'runtime_class': type(self.device).__name__ if self.device is not None else '',
            'runtime_gpib_address': getattr(self.device, 'gpib_address', ''),
            'runtime_resource_name': getattr(getattr(self.device, 'device', None), 'resource_name', ''),
        }
        if selection_text == 'Mock':
            device_metadata['reported_model'] = 'Mock'
            device_metadata['reported_driver_class'] = 'Keithley6517B_Mock'
            device_metadata['reported_idn'] = 'Mock'
        elif selection_text:
            parts = selection_text.split(' - ', 3)
            if len(parts) > 1:
                device_metadata['reported_model'] = parts[1].strip()
            if len(parts) > 2:
                device_metadata['reported_driver_class'] = parts[2].strip()
            if len(parts) > 3:
                device_metadata['reported_idn'] = parts[3].strip()
        vendor, model, serial, firmware = keithley._parse_idn(device_metadata['reported_idn'])
        device_metadata['vendor'] = vendor
        device_metadata['model'] = model or device_metadata['reported_model']
        device_metadata['serial'] = serial
        device_metadata['firmware'] = firmware
        return device_metadata

    def _capture_measurement_metadata(self):
        return {
            'measurement_type': 'VAC',
            'start_time': self.start_time,
            'date_folder': self.date,
            'base_folder': self.folder,
            'sample_name_at_start': self.sample_name,
            'device': self._get_device_metadata(),
            'parameters': {
                'voltage_min_v': self.voltage_min,
                'voltage_max_v': self.voltage_max,
                'voltage_step_v': self.voltage_step,
                'compliance_current_a': self.compliance_current,
                'collection_time_ms': self.collection_time,
                'nplc': self.nplc,
                'current_range_setting': self.current_range,
                'n_runs_requested': self.n_runs,
                'initial_sweep_direction': 'negative_first' if self.initial_sweep_direction < 0 else 'positive_first',
                'polarity_compliance_neg_enabled': self.chk_comp_neg.isChecked(),
                'polarity_compliance_pos_enabled': self.chk_comp_pos.isChecked(),
                'initial_active_compliance_a': self.active_compliance,
            },
            'progress': {
                'estimated_total_steps': self.total_steps,
            },
        }

    def _polarity_for_voltage(self, voltage, direction_hint=None):
        if voltage < 0:
            return 'neg'
        if voltage > 0:
            return 'pos'
        direction = self.direction if direction_hint is None else direction_hint
        if direction == 0:
            direction = self.initial_sweep_direction
        return 'neg' if direction < 0 else 'pos'

    def _compliance_enabled_for_voltage(self, voltage, direction_hint=None):
        side = self._polarity_for_voltage(voltage, direction_hint=direction_hint)
        return self.chk_comp_neg.isChecked() if side == 'neg' else self.chk_comp_pos.isChecked()

    def _apply_measurement_range_for_voltage(self, voltage, direction_hint=None):
        if self.device is None or not hasattr(self.device, 'set_current_range'):
            return

        desired_range = None
        if self.manual_current_range is not None:
            desired_range = self.manual_current_range
            if isinstance(self.device, keithley.Keithley6430) and self._compliance_enabled_for_voltage(voltage, direction_hint=direction_hint):
                desired_range = min(desired_range, float(self.active_compliance))
        elif isinstance(self.device, keithley.Keithley6430) and self._compliance_enabled_for_voltage(voltage, direction_hint=direction_hint):
            desired_range = float(self.active_compliance)

        if desired_range is None:
            return

        try:
            desired_range = max(abs(float(desired_range)), 1e-12)
            self.device.set_current_range(desired_range)
        except Exception as exc:
            print(f"[warn] set_current_range failed: {exc}")

    def _prepare_measurement_range_and_compliance(self, voltage):
        if type(self.device) == keithley.Keithley6517B or type(self.device) == keithley.Keithley6517B_Mock:
            return

        previous_sign = float(self.prev_voltage)
        target_sign = float(np.sign(voltage))
        crossed = target_sign != previous_sign
        if not crossed:
            return

        # Move through 0 V before tightening the sense settings for the next polarity.
        if previous_sign != 0:
            try:
                self.device.set_voltage(0)
            except Exception as exc:
                print(f"[warn] zero-crossing voltage reset failed: {exc}")

        comp = self._apply_compliance_for_voltage(voltage, direction_hint=self.direction)
        if comp is not None:
            self.active_compliance = comp
        self._apply_measurement_range_for_voltage(voltage, direction_hint=self.direction)

    def _build_data_metadata(self, sample_name, output_path, metadata_path):
        metadata = dict(self.measurement_metadata) if self.measurement_metadata else self._capture_measurement_metadata()
        metadata.update({
            'sample_name_at_save': sample_name,
            'saved_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'measurement_count': len(self.measurements),
            'data_columns': ['Voltage', 'Current', 'Direction', 'Timestamp', 'Nruns'],
            'data_file': output_path,
            'metadata_file': metadata_path,
        })
        metadata['progress'] = dict(metadata.get('progress', {}))
        metadata['progress'].update({
            'completed_steps': self.completed_steps,
            'remaining_runs_counter': self.n_runs,
        })
        metadata['parameters'] = dict(metadata.get('parameters', {}))
        metadata['parameters'].update({
            'final_active_compliance_a': self.active_compliance,
        })
        return metadata

    def initUI(self):
        self.setWindowTitle('Keithley IV')
        self.resize(1400, 900)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # keithley
        device_address_layout = QHBoxLayout()
        self.device_address_input = QComboBox()
        self.refresh_button = QPushButton('Refresh GPIB')
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.refresh_status_label = QLabel('Idle')
        self.refresh_status_label.setStyleSheet("color: #64748b;")
        device_address_layout.addWidget(QLabel('Device address:'))
        device_address_layout.addWidget(self.device_address_input)
        device_address_layout.addWidget(self.refresh_button)
        device_address_layout.addWidget(self.refresh_status_label)
        layout.addLayout(device_address_layout)

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
        self.voltage_min_input.setRange(-220, 220)
        self.voltage_min_input.setValue(-1)
        self.voltage_max_input = QDoubleSpinBox()
        self.voltage_max_input.setRange(-220, 220)
        self.voltage_max_input.setValue(1)
        self.voltage_min_input.setDecimals(4)
        self.voltage_max_input.setDecimals(4)
        voltage_range_layout.addWidget(QLabel('Vmin (V):'))
        voltage_range_layout.addWidget(self.voltage_min_input)
        voltage_range_layout.addWidget(QLabel('Vmax (V):'))
        voltage_range_layout.addWidget(self.voltage_max_input)
        layout.addLayout(voltage_range_layout)

        # Voltage step input
        voltage_step_layout = QHBoxLayout()
        self.voltage_step_input = QDoubleSpinBox()
        self.voltage_step_input.setRange(0.0001, 220)
        self.voltage_step_input.setValue(0.01)
        self.voltage_step_input.setDecimals(4)
        voltage_step_layout.addWidget(QLabel('Voltage step (V):'))
        voltage_step_layout.addWidget(self.voltage_step_input)
        layout.addLayout(voltage_step_layout)

        initial_sweep_layout = QHBoxLayout()
        self.initial_sweep_direction_input = QComboBox()
        self.initial_sweep_direction_input.addItems(['+ first', '- first'])
        initial_sweep_layout.addWidget(QLabel('Initial sweep direction:'))
        initial_sweep_layout.addWidget(self.initial_sweep_direction_input)
        layout.addLayout(initial_sweep_layout)

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

        # Polarity-specific compliance checkboxes
        comp_sel_layout = QHBoxLayout()
        self.chk_comp_neg = QCheckBox('Set compliance at V < 0')
        self.chk_comp_neg.setChecked(True)
        self.chk_comp_pos = QCheckBox('Set compliance at V > 0')
        self.chk_comp_pos.setChecked(True)
        comp_sel_layout.addWidget(self.chk_comp_neg)
        comp_sel_layout.addWidget(self.chk_comp_pos)
        layout.addLayout(comp_sel_layout)

        # Start/Stop button
        button_layout = QHBoxLayout()
        self.start_button = QPushButton('Start Measurement')
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_measurement)
        self.voltage_now = QLabel('0 V')
        self.voltage_now.setAlignment(Qt.AlignCenter)
        self.voltage_now.setStyleSheet("background-color: lightgray")
        self.voltage_now.adjustSize()
        self.stop_button = QPushButton('Stop Measurement')
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(lambda _checked=False: self.stop_measurement())
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.voltage_now)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel('Progress: 0/0')
        self.eta_label = QLabel('ETA: --')
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.eta_label)

        # Plot area for I(V), abs(I(V)) and noise using PyQtGraph
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('w')

        # I(V) plot
        pg.setConfigOption('background', 'w')
        self.iv_plot = self.plot_widget.addPlot(title="I(V)")
        self.iv_plot.showGrid(x=True, y=True, alpha=0.12)
        self.iv_plot.setLabel('left', 'Current', units='A')
        self.iv_plot.setLabel('bottom', 'Voltage', units='V')
        self.iv_plot.getAxis('left').enableAutoSIPrefix(True)
        self.iv_plot.getAxis('bottom').enableAutoSIPrefix(True)
        self.abs_iv_plot = self.plot_widget.addPlot(title="|I(V)|")
        self.abs_iv_plot.showGrid(x=True, y=True, alpha=0.12)
        self.abs_iv_plot.setLabel('left', '|Current|', units='A')
        self.abs_iv_plot.setLabel('bottom', 'Voltage', units='V')
        self.abs_iv_plot.getAxis('left').enableAutoSIPrefix(True)
        self.abs_iv_plot.getAxis('bottom').enableAutoSIPrefix(True)

        # Enable logarithmic scale for abs(I(V))
        self.abs_iv_plot.setLogMode(False, True)  # Y-axis in log scale

        layout.addWidget(self.plot_widget)
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
            devices = refresh_device_combos(keithley, [self.device_address_input], include_mock=True)
            count = max(0, len(devices) - (1 if 'Mock' in devices else 0))
            self.refresh_status_label.setText(f'Found {count} device(s)')
        except Exception as exc:
            self.refresh_status_label.setText(f'Refresh failed: {exc}')
            raise
        finally:
            self.refresh_button.setText('Refresh GPIB')
            self.refresh_button.setEnabled(True)

    def estimate_total_steps(self):
        if self.voltage_step <= 0:
            return 1
        runs = int(self.n_runs)
        direction = self.initial_sweep_direction
        voltage = 0.0
        steps = 0
        max_iter = 5_000_000
        while steps < max_iter:
            steps += 1
            voltage += self.voltage_step * direction
            if voltage > self.voltage_max:
                voltage = self.voltage_max
                direction *= -1
                runs -= 1
            if voltage < self.voltage_min:
                voltage = self.voltage_min
                direction *= -1
                runs -= 1
            if runs <= 0 and abs(voltage) <= self.voltage_step / 10:
                break
        return max(1, steps)

    # ===================== Helper: set compliance per polarity =====================
    def _apply_compliance_for_voltage(self, voltage, direction_hint=None):
        """
        Apply instrument compliance for the current voltage side.
        Only applies to non-6517B devices (e.g., 6430).
        If a side is unchecked, we set a 'relaxed' (higher) compliance.
        """
        # Only switch for devices that support explicit compliance setting in your code
        if not isinstance(self.device, keithley.Keithley6430):
            return

        try:
            user_comp = float(self.compliance_current)
        except Exception:
            return

        want_neg = self.chk_comp_neg.isChecked()
        want_pos = self.chk_comp_pos.isChecked()
        print(want_neg, want_pos)
        side = self._polarity_for_voltage(voltage, direction_hint=direction_hint)
        if (side == 'neg' and want_neg) or (side == 'pos' and want_pos):
            comp_to_set = user_comp
        else:
            comp_to_set = 1  # relaxed on unchecked side

        if self._last_comp_polarity == side and self.active_compliance == comp_to_set:
            return comp_to_set

        # Set the compliance on the instrument (method name as in your code)
        try:
            print(f'setting compliance to {comp_to_set}')
            self.device.set_complicance_current(comp_to_set)
            self._last_comp_polarity = side
        except Exception as e:
            print(f"[warn] set_complicance_current failed: {e}")
        return comp_to_set

    # ===================== Measurement lifecycle =====================
    def start_measurement(self):
        if self.timer.isActive():
            return
        self.measurement_metadata = {}
        self.voltage_min = self.voltage_min_input.value()
        self.voltage_max = self.voltage_max_input.value()
        self.voltage_step = self.voltage_step_input.value()
        try:
            self.compliance_current = parse_numeric_text(self.compliance_input.text(), "Compliance current")
        except Exception:
            QMessageBox.critical(None, "Error", f"Compliance current is incorrect : {self.compliance_input.text()}")
            return
        self.nplc = self.nplc_input.currentText()
        self.collection_time = self.collection_time_input.value()
        self.n_runs = int(self.nruns_input.value())
        self.device_address = self.device_address_input.currentText()
        self.sample_name = self._get_sample_name()
        self.current_range = self.current_range_input.currentText()
        self.manual_current_range = None
        self.initial_sweep_direction = 1 if self.initial_sweep_direction_input.currentIndex() == 0 else -1
        self.direction = self.initial_sweep_direction
        print('Device address: ', self.device_address)
        self.device = keithley.get_device(self.device_address, nplc=self.nplc)
        self.active_compliance = self.compliance_current
        self._last_comp_polarity = None
        # Clear previous measurements and noise data
        self.measurements = []
        self.noise_data = []
        self.current_voltage = 0
        self.prev_voltage = 0.0
        self.start_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')
        if self.device == None:
            QMessageBox.critical(None, "Error", f"Device {self.device_address} is unkonw or not found")
            return
        self.device.set_voltage(self.current_voltage)
        if type(self.device) == keithley.Keithley6517B:
            self.device.set_voltage_range(max(abs(self.voltage_min), abs(self.voltage_max)))
        if self.current_range != 'Auto-range':
            try:
                self.manual_current_range = parse_numeric_text(self.current_range, "Current range")
            except Exception:
                QMessageBox.critical(None, "Error", f"Invalid current range: {self.current_range}")
                return
            try:
                self.device.set_current_range(self.manual_current_range)
            except Exception as exc:
                QMessageBox.critical(None, "Error", f"Failed to set current range: {exc}")
                return
        self.total_steps = self.estimate_total_steps()
        self.measurement_metadata = self._capture_measurement_metadata()
        self.completed_steps = 0
        self.progress_tracker.start(self.total_steps)

        self.timer.start(5)  # Update every 50 ms
        self.elapsed_timer.start()  # Start the elapsed time for integration
        # Clear plots
        self.iv_plot.clear()
        self.abs_iv_plot.clear()

    def stop_measurement(self, save=True, close_device=True):
        if self.device:
            try:
                self.device.set_voltage(0)
            except Exception:
                pass
            try:
                self.device.disable_output()
            except Exception:
                pass
            if close_device:
                keithley.shutdown_device(self.device, close=True)
                self.device = None
        self.voltage_now.setText('0 V')
        self.voltage_now.adjustSize()
        self.timer.stop()
        if save and self.measurements:
            save_errors = []
            try:
                self.export_to_csv()
            except Exception as exc:
                save_errors.append(f"Failed to save data:\n{exc}")
            try:
                self.make_plot()
            except Exception as exc:
                save_errors.append(f"Failed to save plots:\n{exc}")
            if save_errors:
                QMessageBox.critical(self, "Save Error", "\n\n".join(save_errors))
        self.direction = self.initial_sweep_direction

    def perform_measurement(self):
        # Perform signal integration over the collection time
        start_time = self.elapsed_timer.elapsed()
        total_current = 0
        num_measurements = 0
        noise_currents = []  # Store the current noise

        self._prepare_measurement_range_and_compliance(self.current_voltage)
        self.device.set_voltage(self.current_voltage)
        self.voltage_now.setText('{:.2f} V, n = {}'.format(self.current_voltage, self.n_runs))
        self.voltage_now.adjustSize()

        p1 = None
        if len(self.measurements) > 0:
            p1 = self.measurements[-1][1]
            if p1 == 0:
                p1 = None
        if self.current_range == 'Auto-range':
            # current = keithley.auto_range(self.device, p1=p1, compl=(1 if type(self.device) == keithley.Keithley6517B else self.compliance_current))
            current = keithley.auto_range(self.device, p1=p1, compl=self.active_compliance)
        total_current += self.device.read_current()
        num_measurements += 1
        while self.elapsed_timer.elapsed() - start_time < self.collection_time:
            # Set the voltage and get the current multiple times to average
            current = self.device.read_current(autorange=False)
            total_current += current
            noise_currents.append(current)  # Collect the noise data
            num_measurements += 1
        # Calculate the average current during the collection time
        if num_measurements > 0:
            average_current = total_current / num_measurements
        else:
            average_current = 0
        # Check compliance current (kept as-is; switching handled separately)
        if abs(average_current) >= self.compliance_current:
            pass
        # Store the data (voltage, average current)
        self.measurements.append((self.current_voltage, average_current, self.direction, time.time(), self.n_runs))
        self.noise_data.append(noise_currents)
        # Update plots
        self.update_plots()
        self.completed_steps += 1
        self.progress_tracker.step(
            self.completed_steps,
            extra_text=f"V={self.current_voltage:.3f} V"
        )
        # Track previous voltage for sign-cross detection
        self.prev_voltage = np.sign(self.current_voltage)
        # Move to the next voltage step
        self.current_voltage += self.voltage_step * self.direction

        if self.current_voltage > self.voltage_max:
            self.current_voltage = self.voltage_max
            self.direction *= -1
            self.n_runs -= 1

        if self.current_voltage < self.voltage_min:
            self.current_voltage = self.voltage_min
            self.direction *= -1
            self.n_runs -= 1

        if self.n_runs <= 0 and abs(self.current_voltage) <= self.voltage_step/10:
            self.current_voltage = 0
            self.device.set_voltage(0)
            self.stop_measurement()



    def update_plots(self):
        # Get I(V) and abs(I(V)) data
        voltages = [m[0] for m in self.measurements][-300::]
        currents = [m[1] for m in self.measurements][-300::]
        abs_currents = [abs(i) for i in currents]

        # Update I(V) plot
        self.iv_plot.plot(voltages, currents, pen=pg.mkPen(color='b', width=2), clear=True)
        self.iv_plot.plot(voltages, currents, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0, 0, 255, 255), clear=False)
        if voltages:
            self.iv_plot.plot([voltages[-1]], [currents[-1]], pen=None, symbol='+', symbolPen=None, symbolSize=10, symbolBrush=(0, 0, 0, 255), clear=False)
        # Update |I(V)| plot
        self.abs_iv_plot.plot(voltages, abs_currents, pen=pg.mkPen(color='r', width=2), clear=True)
        self.abs_iv_plot.plot(voltages, abs_currents, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(255, 0, 0, 255), clear=False)
        if voltages:
            self.abs_iv_plot.plot([voltages[-1]], [abs_currents[-1]], pen=None, symbol='+', symbolPen=None, symbolSize=10, symbolBrush=(0, 0, 0, 255), clear=False)

    def export_to_csv(self):
        sample_name, _, data_dir = self._get_output_directories('data')
        name = f'{sample_name}_{self.voltage_min}V_{self.voltage_max}V_{self.collection_time}ms'
        df = self.get_pandas_data()
        output_path = op.join(data_dir, f'VAC_{name}_{self.start_time}.data')
        try:
            df.to_csv(output_path, index=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to write data file:\n{output_path}\n\n{exc}") from exc
        if not op.isfile(output_path):
            raise RuntimeError(f"Data file was not created:\n{output_path}")
        metadata_path = op.join(data_dir, f'VAC_{name}_{self.start_time}.meta.json')
        metadata = self._build_data_metadata(sample_name, output_path, metadata_path)
        try:
            with open(metadata_path, 'w', encoding='utf-8') as meta_file:
                json.dump(metadata, meta_file, indent=2)
        except Exception as exc:
            raise RuntimeError(f"Failed to write metadata file:\n{metadata_path}\n\n{exc}") from exc
        if not op.isfile(metadata_path):
            raise RuntimeError(f"Metadata file was not created:\n{metadata_path}")
        del df

    def get_pandas_data(self):
        df = pd.DataFrame(self.measurements, columns=['Voltage', 'Current', 'Direction', 'Timestamp', 'Nruns'])
        return df

    def make_plot(self):
        sample_name, _, plot_dir = self._get_output_directories('plots')
        name = f'{sample_name}_{self.voltage_min}V_{self.voltage_max}V_{self.collection_time}ms'
        df = self.get_pandas_data()

        up = df.where(df['Direction'] == 1).dropna().sort_values('Voltage')
        down = df.where(df['Direction'] == -1).dropna().sort_values('Voltage')
        f1 = plt.figure(figsize=(10, 6), dpi=300)
        plt.ticklabel_format(axis='y', style='scientific')
        plt.plot(up[up['Voltage'] < 0]['Voltage'], up[up['Voltage'] < 0]['Current'], 'o-', markersize=3, label='Up', color='blue', alpha=0.6)
        plt.plot(up[up['Voltage'] >= 0]['Voltage'], up[up['Voltage'] >= 0]['Current'], 'o-', markersize=3, color='blue', alpha=0.6)
        plt.plot(down['Voltage'], down['Current'], 'o-', markersize=3, label='Down', color='green', alpha=0.6)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')

        plt.legend()
        linear_plot_path = op.join(plot_dir, f'VAC_{name}_{self.start_time}.png')
        try:
            plt.savefig(linear_plot_path, dpi=300)
        except Exception as exc:
            raise RuntimeError(f"Failed to write plot file:\n{linear_plot_path}\n\n{exc}") from exc
        if not op.isfile(linear_plot_path):
            raise RuntimeError(f"Plot file was not created:\n{linear_plot_path}")
        f1.clf()

        f2 = plt.figure(figsize=(10, 6), dpi=300)
        plt.ticklabel_format(axis='y', style='scientific')
        plt.plot(up[up['Voltage'] < 0]['Voltage'], np.abs(up[up['Voltage'] < 0]['Current']), 'o-', markersize=3, label='Up', color='blue', alpha=0.6)
        plt.plot(up[up['Voltage'] >= 0]['Voltage'], np.abs(up[up['Voltage'] >= 0]['Current']), 'o-', markersize=3, color='blue', alpha=0.6)
        plt.plot(down['Voltage'], np.abs(down['Current']), 'o-', markersize=3, label='Down', color='green', alpha=0.6)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.yscale('log')
        plt.legend()
        log_plot_path = op.join(plot_dir, f'VAC_{name}_{self.start_time}_logscaleY.png')
        try:
            plt.savefig(log_plot_path, dpi=300)
        except Exception as exc:
            raise RuntimeError(f"Failed to write plot file:\n{log_plot_path}\n\n{exc}") from exc
        if not op.isfile(log_plot_path):
            raise RuntimeError(f"Plot file was not created:\n{log_plot_path}")
        del df, up, down
        f2.clf()
        matplotlib.pyplot.close(f1)
        matplotlib.pyplot.close(f2)
        plt.close('all')
        gc.collect()
        # matplotlib.pyplot.clf()

    def closeEvent(self, event):
        try:
            self.stop_measurement(save=False, close_device=True)
        finally:
            super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VACRegime()
    window.show()
    sys.exit(app.exec_())
