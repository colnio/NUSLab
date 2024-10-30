import sys
import matplotlib
import matplotlib.pyplot
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QFileDialog, QComboBox, QLineEdit, QMessageBox)
from PyQt5.QtCore import QTimer, QElapsedTimer
import keithley
import pandas as pd
import time 
import datetime
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
        # Setup GUI components
        self.initUI()
        self.direction = 1

        # Setup timer for measurements
        self.timer = QTimer()
        self.timer.timeout.connect(self.perform_measurement)

        # Timer for signal integration
        self.elapsed_timer = QElapsedTimer()
        # date & make folder
        self.date = str(datetime.date.today())
        self.folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if op.exists(op.join(self.folder, self.date)) == False:
            os.makedirs(op.join(self.folder, self.date))
        self.start_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S') 

    def initUI(self):
        
        self.setWindowTitle('Keithley 6517B IV')
        layout = QVBoxLayout()

        # keithley 
        device_address_layout = QHBoxLayout()
        self.device_address_input = QComboBox()
        self.device_address_input.addItems([' - '.join(i) for i in keithley.get_devices_list()] + ['Mock'])
        device_address_layout.addWidget(QLabel('Device address:'))
        device_address_layout.addWidget(self.device_address_input)
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
        self.voltage_min_input.setRange(-10, 10)
        self.voltage_min_input.setValue(-1)
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
        self.voltage_step_input.setRange(0.0001, 1)
        self.voltage_step_input.setValue(0.01)
        self.voltage_step_input.setDecimals(4)
        voltage_step_layout.addWidget(QLabel('Voltage step (V):'))
        voltage_step_layout.addWidget(self.voltage_step_input)
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
        self.plot_widget.setBackground('w')

        # I(V) plot
        pg.setConfigOption('background', 'w')
        self.iv_plot = self.plot_widget.addPlot(title="I(V)")
        self.iv_plot.showGrid(x=True, y=True)
        self.abs_iv_plot = self.plot_widget.addPlot(title="|I(V)|")
        self.abs_iv_plot.showGrid(x=True, y=True)

        # Enable logarithmic scale for abs(I(V))
        self.abs_iv_plot.setLogMode(False, True)  # Y-axis in log scale

        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

    def start_measurement(self):
        self.voltage_min = self.voltage_min_input.value()
        self.voltage_max = self.voltage_max_input.value()
        self.voltage_step = self.voltage_step_input.value()
        try:
            self.compliance_current = eval(self.compliance_input.text())
        except:
            QMessageBox.critical(None, "Error", f"Compliance current is incorrect : {self.compliance_input.text()}")
            return
        self.nplc = self.nplc_input.currentText()
        self.collection_time = self.collection_time_input.value()
        self.n_runs = int(self.nruns_input.value())
        self.device_address = self.device_address_input.currentText()
        self.sample_name = self.sample_name_input.text()
        self.current_range = self.current_range_input.currentText()
        self.direction = 1
        print('Device address: ', self.device_address)
        self.device = keithley.get_device(self.device_address, nplc=self.nplc)
        # Clear previous measurements and noise data
        self.measurements = []
        self.noise_data = []
        self.current_voltage = 0
        self.start_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')
        if self.device == None:
            QMessageBox.critical(None, "Error", f"Device {self.device_address} is unkonw or not found")
            return
        if type(self.device) == keithley.Keithley6517B:
            self.device.set_voltage_range(max(abs(self.voltage_min), abs(self.voltage_max)))
        if self.current_range != 'Auto-range':
            self.device.set_current_range(self.current_range)
        self.timer.start(50)  # Update every 50 ms
        self.elapsed_timer.start()  # Start the elapsed time for integration
        # Clear plots
        self.iv_plot.clear()
        self.abs_iv_plot.clear()

    def stop_measurement(self):
        self.current_voltage = 0
        self.device.set_voltage(0)
        self.device.disable_output()
        self.timer.stop()
        self.make_plot()
        self.export_to_csv()
        self.direction = 1

    def perform_measurement(self):
        # Perform signal integration over the collection time
        start_time = self.elapsed_timer.elapsed()
        total_current = 0
        num_measurements = 0
        noise_currents = []  # Store the current noise

        self.device.set_voltage(self.current_voltage)
        p1 = None
        if len(self.measurements) > 0:
            p1 = self.measurements[-1][1]
            if p1 == 0: 
                p1 = None
        if self.current_range == 'Auto-range':
            current = keithley.auto_range(self.device, p1=p1)
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
        # Check compliance current
        if abs(average_current) >= self.compliance_current:
            self.stop_measurement()
            QMessageBox.critical(None, "Error", "Compliance current or current range exceeded")
        # Store the data (voltage, average current)
        self.measurements.append((self.current_voltage, average_current, self.direction))
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

        if self.n_runs <= 0 and abs(self.current_voltage) <= self.voltage_step/10:
            self.current_voltage = 0
            self.device.set_voltage(0)
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

    def export_to_csv(self):

        sample_dir = op.join(self.folder, self.date, self.sample_name)
        if not op.exists(sample_dir):
            os.makedirs(sample_dir)
        if not op.exists(op.join(sample_dir, 'data')):
            os.makedirs(op.join(sample_dir, 'data'))
        name = f'{self.sample_name}_{self.voltage_min}V_{self.voltage_max}V_{self.collection_time}ms'
        df = self.get_pandas_data()
        df.to_csv(op.join(sample_dir, 'data', f'VAC_{name}_{self.start_time}.data'), index=False)
        del df


    def get_pandas_data(self):
        df = pd.DataFrame(self.measurements, columns=['Voltage', 'Current', 'Direction'])
        return df

    def make_plot(self):
        sample_dir = op.join(self.folder, self.date, self.sample_name)
        if not op.exists(sample_dir):
            os.makedirs(sample_dir)
        if not op.exists(op.join(sample_dir, 'plots')):
            os.makedirs(op.join(sample_dir, 'plots'))
        name = f'{self.sample_name}_{self.voltage_min}V_{self.voltage_max}V_{self.collection_time}ms'
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
        plt.savefig(op.join(sample_dir, 'plots', f'VAC_{name}_{self.start_time}.png'), dpi=300)
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
        plt.savefig(op.join(sample_dir, 'plots', f'VAC_{name}_{self.start_time}_logscaleY.png'), dpi=300)
        del df, up, down
        f2.clf()
        matplotlib.pyplot.close(f1)
        matplotlib.pyplot.close(f2)
        plt.close('all')
        gc.collect()
        # matplotlib.pyplot.clf()
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VACRegime()
    window.show()
    sys.exit(app.exec_())
