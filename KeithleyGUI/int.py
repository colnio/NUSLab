import sys
import numpy as np
import csv
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QFileDialog, QComboBox)
from PyQt5.QtCore import QTimer, QElapsedTimer
from scipy.fft import fft
from pymeasure.instruments.keithley import Keithley6517B
import random 
from pymeasure.instruments.resources import list_resources
import time 

class MeasurementSimulator(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize simulation variables
        self.device = Keithley6517B('GPIB0::27::INSTR')
        self.device_address = ''
        self.voltage_min = 0
        self.voltage_max = 1
        self.compliance_current = 1e-3
        self.voltage_step = 0.1
        self.collection_time = 1000  # in milliseconds
        self.measurements = []
        self.noise_data = []  # For storing I(t) during measurements
        self.n_runs = 2
        # Setup GUI components
        self.initUI()
        self.direction = 1

        # Setup timer for measurements
        self.timer = QTimer()
        self.timer.timeout.connect(self.perform_measurement)

        # Timer for signal integration
        self.elapsed_timer = QElapsedTimer()

    def initUI(self):
        self.setWindowTitle('Simulated Keithley 6517B')
        layout = QVBoxLayout()


        # keithley 
        # device_address_layout = QHBoxLayout()
        self.devce_address_input = QComboBox()
        self.devce_address_input.addItems(list(list_resources()))
        layout.addWidget(self.devce_address_input)
        # Voltage range input
        voltage_range_layout = QHBoxLayout()
        self.voltage_min_input = QDoubleSpinBox()
        self.voltage_min_input.setRange(-10, 10)
        self.voltage_min_input.setValue(0)
        self.voltage_max_input = QDoubleSpinBox()
        self.voltage_max_input.setRange(-10, 10)
        self.voltage_max_input.setValue(1)
        voltage_range_layout.addWidget(QLabel('Vmin (V):'))
        voltage_range_layout.addWidget(self.voltage_min_input)
        voltage_range_layout.addWidget(QLabel('Vmax (V):'))
        voltage_range_layout.addWidget(self.voltage_max_input)
        layout.addLayout(voltage_range_layout)

        # Voltage step input
        voltage_step_layout = QHBoxLayout()
        self.voltage_step_input = QDoubleSpinBox()
        self.voltage_step_input.setRange(0.001, 1)
        self.voltage_step_input.setValue(0.1)
        voltage_step_layout.addWidget(QLabel('Voltage step (V):'))
        voltage_step_layout.addWidget(self.voltage_step_input)
        layout.addLayout(voltage_step_layout)

        # nruns input
        nruns_layout = QHBoxLayout()
        self.nruns_input = QDoubleSpinBox()
        self.nruns_input.setRange(1, 1000)
        self.nruns_input.setValue(2)
        nruns_layout.addWidget(QLabel('N runs (2 is 1 up 1 down)'))
        nruns_layout.addWidget(self.nruns_input)
        layout.addLayout(nruns_layout)

        # Compliance current input
        compliance_layout = QHBoxLayout()
        self.compliance_input = QDoubleSpinBox()
        self.compliance_input.setRange(1e-11, 1e11)
        self.compliance_input.setValue(1e-9)
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

        # Start/Stop button
        button_layout = QHBoxLayout()
        self.start_button = QPushButton('Start Measurement')
        self.start_button.clicked.connect(self.start_measurement)
        self.stop_button = QPushButton('Stop Measurement')
        self.stop_button.clicked.connect(self.stop_measurement)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        # Plot area for I(V), abs(I(V)) and noise using PyQtGraph
        self.plot_widget = pg.GraphicsLayoutWidget()

        # I(V) plot
        self.iv_plot = self.plot_widget.addPlot(title="I(V)")
        self.abs_iv_plot = self.plot_widget.addPlot(title="|I(V)|")

        # Enable logarithmic scale for abs(I(V))
        self.abs_iv_plot.setLogMode(False, True)  # Y-axis in log scale

        # Noise plot
        self.noise_plot = self.plot_widget.addPlot(title="Noise (I(t)) or FFT")

        layout.addWidget(self.plot_widget)

        # Add crosshair (cursor) for the I(V) plot
        self.v_line = pg.InfiniteLine(angle=90, movable=False)
        self.h_line = pg.InfiniteLine(angle=0, movable=False)
        self.iv_plot.addItem(self.v_line, ignoreBounds=True)
        self.iv_plot.addItem(self.h_line, ignoreBounds=True)
        self.proxy = pg.SignalProxy(self.iv_plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

        # Button to export to CSV
        self.export_button = QPushButton('Export to CSV')
        self.export_button.clicked.connect(self.export_to_csv)
        layout.addWidget(self.export_button)

        self.setLayout(layout)

    def start_measurement(self):
        self.voltage_min = self.voltage_min_input.value()
        self.voltage_max = self.voltage_max_input.value()
        self.voltage_step = self.voltage_step_input.value()
        self.compliance_current = self.compliance_input.value()
        self.collection_time = self.collection_time_input.value()
        self.n_runs = int(self.nruns_input.value())
        self.device_address = self.devce_address_input.currentText
        self.device = Keithley6517B('GPIB0::27::INSTR')
        time.sleep(1000)
        # Clear previous measurements and noise data
        self.measurements = []
        self.noise_data = []
        # self.current_voltage = self.voltage_min
        self.current_voltage = 0
        self.timer.start(100)  # Update every 100 ms
        self.elapsed_timer.start()  # Start the elapsed time for integration

        # Clear plots
        self.iv_plot.clear()
        self.abs_iv_plot.clear()
        self.noise_plot.clear()

    def stop_measurement(self):
        self.current_voltage = 0
        self.device.shutdown()
        self.timer.stop()

    def perform_measurement(self):
        # Perform signal integration over the collection time
        # self.device.source_voltage = 0
        self.device.enable_source()
        self.device.apply_voltage(voltage_range=5)
        self.device.measure_current(nplc=1, current=min(self.compliance_current, 21e-3), auto_range=True)
        start_time = self.elapsed_timer.elapsed()
        total_current = 0
        num_measurements = 0

        noise_currents = []  # Store the current noise

        while self.elapsed_timer.elapsed() - start_time < self.collection_time:
            # Set the voltage and get the current multiple times to average
            self.device.ramp_to_voltage(self.current_voltage)
            current = self.device.current
            total_current += current
            noise_currents.append(current)  # Collect the noise data
            num_measurements += 1

        # Calculate the average current during the collection time
        if num_measurements > 0:
            average_current = total_current / num_measurements
        else:
            average_current = 0

        # Check compliance current
        if abs(average_current) >= self.compliance_current:
            self.device.ramp_to_voltage(0)  # Set voltage to 0 if compliance exceeded
            self.timer.stop()

        # Store the data (voltage, average current)
        self.measurements.append((self.current_voltage, average_current))
        self.noise_data.append(noise_currents)

        # Update plots
        self.update_plots()

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
        if self.n_runs <= 0 and abs(self.current_voltage) <= self.voltage_step:
            self.current_voltage = 0
            self.device.ramp_to_voltage(0)
            self.stop_measurement()

    def update_plots(self):
        # Get I(V) and abs(I(V)) data
        voltages = [m[0] for m in self.measurements]
        currents = [m[1] for m in self.measurements]
        abs_currents = [abs(i) for i in currents]

        # Update I(V) plot
        self.iv_plot.plot(voltages, currents, pen=pg.mkPen(color='b', width=2), clear=True)
        self.iv_plot.plot(voltages, currents, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0, 0, 255, 255), clear=False)

        self.iv_plot.setLabel('left', 'Current (A)')
        self.iv_plot.setLabel('bottom', 'Voltage (V)')

        # Update |I(V)| plot
        self.abs_iv_plot.plot(voltages, abs_currents, pen=pg.mkPen(color='r', width=2), clear=True)
        self.abs_iv_plot.plot(voltages, abs_currents, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(255, 0, 0, 255), clear=False)
        self.abs_iv_plot.setLabel('left', '|Current| (A)')
        self.abs_iv_plot.setLabel('bottom', 'Voltage (V)')

        # Update noise plot with I(t)
        if self.noise_data:
            last_noise = self.noise_data[-1]
            time_axis = np.linspace(0, self.collection_time, len(last_noise))
            self.noise_plot.plot(time_axis, last_noise, pen=pg.mkPen(color='g', width=2), clear=True)
            self.noise_plot.setLabel('left', 'Current (A)')
            self.noise_plot.setLabel('bottom', 'Time (ms)')

            # Optional: Uncomment to show FFT of I(t) instead of I(t)
            # fft_data = np.abs(fft(last_noise))
            # freqs = np.fft.fftfreq(len(last_noise), d=(self.collection_time / len(last_noise)) / 1000.0)
            # self.noise_plot.plot(freqs[:len(freqs)//2], fft_data[:len(freqs)//2], pen=pg.mkPen(color='m', width=2), clear=True)
            # self.noise_plot.setLabel('left', 'Amplitude (A)')
            # self.noise_plot.setLabel('bottom', 'Frequency (Hz)')

    def export_to_csv(self):
        # Open file dialog to save CSV
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save CSV', '', 'CSV Files (*.csv)')
        if file_name:
            with open(file_name, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Voltage (V)', 'Current (A)'])
                csvwriter.writerows(self.measurements)

    # def mouse_moved(self, evt):
    #     pos = evt[0]  # Get the mouse position
    #     if self.iv_plot.sceneBoundingRect().contains(pos):
    #         mouse_point = self.iv_plot.plotItem.vb.mapSceneToView(pos)
    #         self.v_line.setPos(mouse_point.x())
    #         self.h_line.setPos(mouse_point.y())

    #         # Optionally, display the value somewhere in the GUI or print it
    #         print(f"Voltage: {mouse_point.x():.2f} V, Current: {mouse_point.y():.6e} A")
    def mouse_moved(self, evt):
        pos = evt[0]  # Get the mouse position
        if self.iv_plot.sceneBoundingRect().contains(pos):
            mouse_point = self.iv_plot.vb.mapSceneToView(pos)  # Use iv_plot.vb directly
            self.v_line.setPos(mouse_point.x())
            self.h_line.setPos(mouse_point.y())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MeasurementSimulator()
    window.show()
    sys.exit(app.exec_())
