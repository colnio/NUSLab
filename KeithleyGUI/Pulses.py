import sys
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
import matplotlib
matplotlib.use('Agg')

class PulsesRegime(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize simulation variables
        self.device = None
        self.sample_name = ''
        # self.device = keithley.Keithley6517B('GPIB0::27::INSTR')
        self.device_address = ''
        self.v_set = 0 
        self.v_read_set = 0
        self.v_reset = 0
        self.v_read_reset = 0
        self.compliance_current = 1e-3
        self.voltage_step = 0.1
        self.collection_time = 10  # in milliseconds
        self.voltages = []
        self.i_set = []
        self.i_read_set = []
        self.i_reset = []
        self.i_read_reset = []  # For storing I(t) during measurements
        self.n_runs = 2
        self.current_range = 1e-8
        # Setup GUI components
        self.initUI()
        # self.direction = 1

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
        # self.device_address_input.addItems(['dev 1', 'dev 2'])
        device_address_layout.addWidget(QLabel('Device address:'))
        device_address_layout.addWidget(self.device_address_input)
        layout.addLayout(device_address_layout)
        # sample name
        sample_name_layout = QHBoxLayout()
        self.sample_name_input = QLineEdit()
        sample_name_layout.addWidget(QLabel('Sample name:'))
        sample_name_layout.addWidget(self.sample_name_input)
        layout.addLayout(sample_name_layout)
        # NPLC input
        nplc_layout = QHBoxLayout()
        self.nplc_input = QComboBox()
        self.nplc_input.addItems(['0.01', '0.1', '1', '10'])
        nplc_layout.addWidget(QLabel('NPLC (1 = 20 ms):'))
        nplc_layout.addWidget(self.nplc_input)
        layout.addLayout(nplc_layout)
        # Voltage range input
        voltage_input = QHBoxLayout()
        self.voltage_set = QDoubleSpinBox()
        self.voltage_set.setRange(-10, 10)
        self.voltage_set.setValue(-1)
        self.voltage_read_set = QDoubleSpinBox()
        self.voltage_read_set.setRange(-10, 10)
        self.voltage_read_set.setValue(1)
        voltage_input.addWidget(QLabel('V set (V):'))
        voltage_input.addWidget(self.voltage_set)
        voltage_input.addWidget(QLabel('V read set (V):'))
        voltage_input.addWidget(self.voltage_read_set)
        self.voltage_reset = QDoubleSpinBox()
        self.voltage_reset.setRange(-10, 10)
        self.voltage_reset.setValue(-1)
        self.voltage_read_reset = QDoubleSpinBox()
        self.voltage_read_reset.setRange(-10, 10)
        self.voltage_read_reset.setValue(1)
        voltage_input.addWidget(QLabel('V reset (V):'))
        voltage_input.addWidget(self.voltage_reset)
        voltage_input.addWidget(QLabel('V read reset (V):'))
        voltage_input.addWidget(self.voltage_read_reset)
        layout.addLayout(voltage_input)

        # nruns input
        nruns_layout = QHBoxLayout()
        self.nruns_input = QSpinBox()
        self.nruns_input.setRange(1, 1000000)
        self.nruns_input.setValue(100)
        nruns_layout.addWidget(QLabel('N runs (Set -> Read -> Reset -> Read_reset)'))
        nruns_layout.addWidget(self.nruns_input)
        layout.addLayout(nruns_layout)

        # Compliance current input
        compliance_layout = QHBoxLayout()
        self.compliance_input = QLineEdit()
        compliance_layout.addWidget(QLabel('Compliance current (A):'))
        compliance_layout.addWidget(self.compliance_input)
        layout.addLayout(compliance_layout)

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
        # First row
        self.curr_set_plot = self.plot_widget.addPlot(title="Current Set")
        self.curr_read_plot = self.plot_widget.addPlot(title="Current Read")

        # Move to the next row
        self.plot_widget.nextRow()

        # Second row
        self.curr_reset_plot = self.plot_widget.addPlot(title="Current Reset")
        self.curr_read_rest_plot = self.plot_widget.addPlot(title="Current Read Rest")

        # Add the widget to the layout
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)

    def start_measurement(self):
        self.v_set = self.voltage_set.value()
        self.v_read_set = self.voltage_read_set.value()
        self.v_reset = self.voltage_reset.value()
        self.v_read_reset = self.voltage_read_reset.value()
        try:
            self.compliance_current = eval(self.compliance_input.text())
        except:
            QMessageBox.critical(None, "Error", f"Compliance current is incorrect : {self.compliance_input.text()}")
            return
        self.n_runs = int(self.nruns_input.value())
        self.device_address = self.device_address_input.currentText()
        self.sample_name = self.sample_name_input.text()
        self.nplc = self.nplc_input.currentText()
        print('Device address: ', self.device_address)
        self.device = keithley.get_device(self.device_address, nplc=self.nplc)
        time.sleep(1)
        # Clear previous measurements and noise data
        self.voltages = []
        self.i_set = []
        self.i_read_set = []
        self.i_reset = []
        self.i_read_reset = []
        # self.current_voltage = self.voltage_min
        self.current_voltage = 0
        self.start_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')

        if self.device == None:
            QMessageBox.critical(None, "Error", f"Device {self.device_address} is unkonw or not found")
            return
        if type(self.device) == keithley.Keithley6517B:
            self.device.set_voltage_range(max(abs(self.v_set), abs(self.v_reset)))
        self.timer.start(100)  # Update every 100 ms
        self.elapsed_timer.start()  # Start the elapsed time for integration

        # Clear plots
        self.curr_set_plot.clear()
        self.curr_read_plot.clear()
        self.curr_reset_plot.clear()
        self.curr_read_rest_plot.clear()
        # self.noise_plot.clear()

    def stop_measurement(self):
        self.current_voltage = 0
        self.device.set_voltage(0)
        self.device.disable_output()
        self.timer.stop()
        self.make_plot()
        self.export_to_csv()

    def perform_measurement(self):
        # SET
        self.device.set_voltage(self.v_set)
        self.voltages.append(self.v_set)
        self.i_set.append(keithley.auto_range(self.device, p1=None if len(self.i_set) == 0 else self.i_set[-1]))
        self.device.set_voltage(0)
        # READ
        self.device.set_voltage(self.v_read_set)
        self.voltages.append(self.v_read_set)
        self.i_read_set.append(keithley.auto_range(self.device, p1=None if len(self.i_read_set) == 0 else self.i_read_set[-1]))
        self.device.set_voltage(0)
        # RESET
        self.device.set_voltage(self.v_reset)
        self.voltages.append(self.v_reset)
        self.i_reset.append(keithley.auto_range(self.device, p1=None if len(self.i_reset) == 0 else self.i_reset[-1]))
        self.device.set_voltage(0)
        # READ
        self.device.set_voltage(self.v_read_reset)
        self.voltages.append(self.v_read_reset)
        self.i_read_reset.append(keithley.auto_range(self.device, p1=None if len(self.i_read_reset) == 0 else self.i_read_reset[-1]))
        self.device.set_voltage(0)
        # Update plots
        self.update_plots()
        self.n_runs -= 1
        # Move to the next voltage step
        if (self.i_set[-1] > self.compliance_current or self.i_read_set[-1] > self.compliance_current \
            or self.i_reset[-1] > self.compliance_current \
            or self.i_read_reset[-1] > self.compliance_current):
            self.current_voltage = 0
            self.device.set_voltage(0)
            self.stop_measurement()
            QMessageBox.critical(None, "Error", "Compliance current or current range exceeded")

        if self.n_runs <= 0:
            self.current_voltage = 0
            self.device.set_voltage(0)
            self.stop_measurement()

    def update_plots(self):
        g = np.linspace(1, len(self.i_set) + 1, len(self.i_set))
        # Update SET plot
        self.curr_set_plot.plot(g, self.i_set, pen=pg.mkPen(color='b', width=2), clear=True)
        self.curr_set_plot.plot(g, self.i_set, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0, 0, 255, 255), clear=False)
        
        self.curr_set_plot.setLabel('left', 'Current (A)')
        self.curr_set_plot.setLabel('bottom', 'Iteration (N)')
        # Update READ plot
        self.curr_read_plot.plot(g, self.i_read_set, pen=pg.mkPen(color='b', width=2), clear=True)
        self.curr_read_plot.plot(g, self.i_read_set, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0, 0, 255, 255), clear=False)
        
        self.curr_read_plot.setLabel('left', 'Current (A)')
        self.curr_read_plot.setLabel('bottom', 'Iteration (N)')
        # Update RESET plot
        self.curr_reset_plot.plot(g, self.i_reset, pen=pg.mkPen(color='b', width=2), clear=True)
        self.curr_reset_plot.plot(g, self.i_reset, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0, 0, 255, 255), clear=False)
        
        self.curr_reset_plot.setLabel('left', 'Current (A)')
        self.curr_reset_plot.setLabel('bottom', 'Iteration (N)')
        # Update READ RESET plot
        self.curr_read_rest_plot.plot(g, self.i_read_reset, pen=pg.mkPen(color='b', width=2), clear=True)
        self.curr_read_rest_plot.plot(g, self.i_read_reset, pen=None, symbol='o', symbolPen=None, symbolSize=5, symbolBrush=(0, 0, 255, 255), clear=False)
        
        self.curr_read_rest_plot.setLabel('left', 'Current (A)')
        self.curr_read_rest_plot.setLabel('bottom', 'Iteration (N)')


    def export_to_csv(self):

        sample_dir = op.join(self.folder, self.date, self.sample_name)
        if not op.exists(sample_dir):
            os.makedirs(sample_dir)
        if not op.exists(op.join(sample_dir, 'data')):
            os.makedirs(op.join(sample_dir, 'data'))
        name = f'{self.sample_name}_SET{self.v_set}V_READ{self.v_read_set}V_RESET{self.v_reset}V_READ_RESET{self.v_read_reset}V_{self.nruns_input.value()}RUNS'
        df = self.get_pandas_data()
        df.to_csv(op.join(sample_dir, 'data', f'PULSES_{name}_{self.start_time}.data'), index=False)


    def get_pandas_data(self):
        df = pd.DataFrame([[self.i_set[i], self.i_read_set[i], self.i_reset[i], self.i_read_reset[i]] for i in range(len(self.i_set))], columns=[f'Vset={self.v_set}',f'Vread_set={self.v_read_set}', f'Vreset={self.v_reset}', f'Vread_reset={self.v_read_reset}'])
        return df


    def make_plot(self):
        sample_dir = op.join(self.folder, self.date, self.sample_name)
        if not op.exists(sample_dir):
            os.makedirs(sample_dir)
        if not op.exists(op.join(sample_dir, 'plots')):
            os.makedirs(op.join(sample_dir, 'plots'))
        name = f'{self.sample_name}_SET{self.v_set}V_READ{self.v_read_set}V_RESET{self.v_reset}V_READ_RESET{self.v_read_reset}V_{self.nruns_input.value()}RUNS'

        g = np.linspace(1, len(self.i_set) + 1, len(self.i_set))
        
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(g, self.i_set, 'o-', markersize=3, label='SET', color='red', alpha=0.6)
        plt.plot(g, self.i_read_set, 'o-', markersize=3, label='READ', color='green', alpha=0.6)
        plt.plot(g, self.i_reset, 'o-', markersize=3, label='RESET', color='blue', alpha=0.6)
        plt.plot(g, self.i_read_reset, 'o-', markersize=3, label='READ RESET', color='black', alpha=0.6)

        plt.xlabel('Iteration (N)')
        plt.ylabel('Current (A)')

        plt.legend()
        plt.savefig(op.join(sample_dir, 'plots', f'PULSES_{name}_{self.start_time}.png'), dpi=300)
        plt.close('all')
        gc.collect()
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PulsesRegime()
    window.show()
    sys.exit(app.exec_())
