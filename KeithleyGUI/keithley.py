import pyvisa
import time
import logging
import pandas as pd
# import random
import numpy as np


class Keithley6517B:
    def __init__(self, gpib_address='GPIB0::27::INSTR', nplc=1):
        self.rm = pyvisa.ResourceManager()
        self.device = self.rm.open_resource(f'{gpib_address}')
        self.device.timeout = 5000  # Timeout for commands, can be adjusted if needed
        self.clear_buffer()
        self.init_device(nplc)
        
    def init_device(self, nplc):
        self.device.write('*CLS;')
        self.device.write('*RST;')
        self.device.write("SYST:ZCH OFF;")
        self.device.write(":SENS:FUNC 'CURR';")
        self.device.write(f":SENS:CURR:NPLC {nplc}")
        self.device.write(':SOUR:VOLT:MCON 1;')

    def clear_buffer(self):
        """Clears the instrument's input buffer."""
        self.device.write('*CLS')

    def set_voltage_range(self, range_value):
        """
        Set the voltage range.
        :param range_value: Voltage range in volts (e.g., 10, 100, 200).
        """
        self.device.write(f'VOLT:DC:RANG {range_value};')
        print(f"Voltage range set to {range_value} V")

    def set_current_range(self, range_value):
        """
        Set the current range.
        :param range_value: Current range in amps (e.g., 0.01, 0.1, 1).
        """
        self.device.write(f'CURR:RANG {range_value};')
    
    def get_current_range(self):
        self.device.write('CURR:RANG?;')
        return eval(self.device.read())

    def set_voltage(self, voltage_value):
        """
        Set the output voltage.
        :param voltage_value: Voltage to set (in volts).
        """
        self.device.write(f'SOUR:VOLT:LEV {voltage_value};')
        self.device.write('OUTP ON')  # Turn the output on
        # print(f"Voltage set to {voltage_value} V and output enabled")

    def disable_output(self):
        """
        Set the output voltage.
        :param voltage_value: Voltage to set (in volts).
        """
        self.device.write('OUTP OFF')  # Turn the output on

    def read_current(self, autorange=False):
        """
        Read the current from the input.
        :return: The measured current (in amps).
        """
        if autorange:
            self.device.write(':MEAS?;')
        else:
            self.device.write(':READ?;')
        current_value = self.device.read().split(',')[0][:-4:]
        return float(current_value)

    def continuous_current_read(self, delay=1):
        """
        Continuously reads the current at intervals.
        :param delay: Delay between readings in seconds.
        """
        try:
            while True:
                current_value = self.read_current()
                print(f"Current: {current_value} A")
                time.sleep(delay)
        except KeyboardInterrupt:
            print("Continuous current reading stopped.")

    def close(self):
        """Close the GPIB connection."""
        self.set_voltage(0)
        self.device.write('OUTP OFF')  # Turn the output off
        self.device.close()
        print("Connection to Keithley 6517B closed.")

class Keithley6430:
    def __init__(self, gpib_address='GPIB0::16::INSTR', nplc=1):
        self.rm = pyvisa.ResourceManager()
        self.device = self.rm.open_resource(f'{gpib_address}')
        self.device.timeout = 5000  # Timeout for commands, can be adjusted if needed
        self.clear_buffer()
        self.init_device(nplc)
        
    def init_device(self, nplc):
        self.device.write('*CLS;')
        self.device.write('*RST;')
        # self.device.write("SYST:ZCH OFF;")
        self.device.write(":SENS:FUNC 'CURR';")
        self.device.write(f":SENS:CURR:NPLC {nplc}")
        self.device.write(f":SOUR:VOLT:MODE FIX")
        # self.device.write(':SOUR:VOLT:MCON 1;')

    def clear_buffer(self):
        """Clears the instrument's input buffer."""
        self.device.write('*CLS')

    def set_voltage_range(self, range_value):
        """
        Set the voltage range.
        :param range_value: Voltage range in volts (e.g., 10, 100, 200).
        """
        self.device.write(f'VOLT:DC:RANG {range_value};')
        print(f"Voltage range set to {range_value} V")

    def set_current_range(self, range_value):
        """
        Set the current range.
        :param range_value: Current range in amps (e.g., 0.01, 0.1, 1).
        """
        self.device.write(f'CURR:RANG {range_value};')
    
    def get_current_range(self):
        self.device.write('CURR:RANG?;')
        return eval(self.device.read())

    def set_voltage(self, voltage_value):
        """
        Set the output voltage.
        :param voltage_value: Voltage to set (in volts).
        """
        self.device.write(f'SOUR:VOLT:LEV {voltage_value};')
        self.device.write('OUTP ON')  # Turn the output on
        # print(f"Voltage set to {voltage_value} V and output enabled")

    def disable_output(self):
        """
        Set the output voltage.
        :param voltage_value: Voltage to set (in volts).
        """
        self.device.write('OUTP OFF')  # Turn the output on

    def read_current(self, autorange=False):
        """
        Read the current from the input.
        :return: The measured current (in amps).
        """
        if autorange:
            self.device.write(':MEAS?;')
        else:
            self.device.write(':READ?;')
        current_value = self.device.read().split(',')[1]
        return float(current_value)

    def continuous_current_read(self, delay=1):
        """
        Continuously reads the current at intervals.
        :param delay: Delay between readings in seconds.
        """
        try:
            while True:
                current_value = self.read_current()
                print(f"Current: {current_value} A")
                time.sleep(delay)
        except KeyboardInterrupt:
            print("Continuous current reading stopped.")
    

    def close(self):
        """Close the GPIB connection."""
        self.set_voltage(0)
        self.device.write('OUTP OFF')  # Turn the output off
        self.device.close()
        print("Connection to Keithley 6517B closed.")


class Keithley6517B_Mock:
    def __init__(self, gpib_address='GPIB0::27::INSTR', nplc=1, I_s=1e-12, n=1.5, T=300):
        self.gpib_address = gpib_address
        self.nplc = nplc
        self.output_enabled = False
        self.voltage_level = 0.0
        self.voltage_range = 0.0
        self.current_range = 0.0
        self.I_s = I_s  # Saturation current in Amps
        self.n = n      # Ideality factor
        self.T = T      # Temperature in Kelvin
        self.V_T = 25.85e-3  # Thermal voltage (approximately 25.85 mV at 300K)
        print(f"Mock Keithley 6517B initialized at address {gpib_address} with NPLC={nplc}")
        self.clear_buffer()
        self.init_device(nplc)

    def init_device(self, nplc):
        print(f"Initializing mock device with NPLC={nplc}")
        self.output_enabled = False

    def clear_buffer(self):
        """Clears the mock instrument's input buffer."""
        print("Mock buffer cleared")

    def set_voltage_range(self, range_value):
        """
        Set the voltage range.
        :param range_value: Voltage range in volts (e.g., 10, 100, 200).
        """
        self.voltage_range = range_value
        print(f"Mock voltage range set to {range_value} V")

    def set_current_range(self, range_value):
        """
        Set the current range.
        :param range_value: Current range in amps (e.g., 0.01, 0.1, 1).
        """
        self.current_range = range_value
        print(f"Mock current range set to {range_value} A")

    def set_voltage(self, voltage_value):
        """
        Set the output voltage.
        :param voltage_value: Voltage to set (in volts).
        """
        self.voltage_level = voltage_value
        self.output_enabled = True
        print(f"Mock voltage set to {voltage_value} V and output enabled")

    def disable_output(self):
        """
        Disable the output voltage.
        """
        self.output_enabled = False
        print("Mock output disabled")

    def read_current(self, autorange=False):
        """
        Read the current through the diode based on the applied voltage using the diode equation.
        :return: The simulated current (in amps).
        """
        if self.output_enabled:
            V = self.voltage_level
            # Diode I(V) characteristic (Shockley equation)
            I = self.I_s * (np.exp(V / (self.n * self.V_T)) - 1)
            # Simulate noise or rounding in a real device
            I += (1e-9) * (2 * (np.random.random() - 0.5))  # Small random noise
            print(f"Mock current read: {I} A")
            return I
        else:
            print("Output is disabled, returning 0 A")
            return 0.0
    
    def get_current_range(self):
        # self.device.write('CURR:RANG?;')
        return 1e-2

    def continuous_current_read(self, delay=1):
        """
        Continuously reads the current at intervals.
        :param delay: Delay between readings in seconds.
        """
        try:
            while True:
                current_value = self.read_current()
                print(f"Mock continuous current: {current_value} A")
                time.sleep(delay)
        except KeyboardInterrupt:
            print("Mock continuous current reading stopped.")

    def close(self):
        """Close the mock GPIB connection."""
        self.set_voltage(0)
        self.output_enabled = False
        print("Mock connection to Keithley 6517B closed.")
