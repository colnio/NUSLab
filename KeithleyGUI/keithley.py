import pyvisa
import time
import numpy as np
from pyvisa import ResourceManager
import mock

def find_range(current):
    ranges = 2 * 10.0**np.arange(-12.0, -1.0, 1.0)
    idx = np.where(ranges > current)[0]
    if len(idx) > 0:
        return ranges[min(idx)]
    return max(ranges)

def auto_range(device, p1=None):
    if p1 is None:
        c = device.read_current(autorange=True)
        device.set_current_range(find_range(abs(c) * 3))
    else:
        device.set_current_range(find_range(abs(p1) * 3))
        c = device.read_current()
    return c if abs(c) < 20e-3 else device.read_current(autorange=True)

def get_device(gpib_address, nplc):
    device = None
    if gpib_address == 'Mock':
            device = Keithley6517B_Mock()
    else:
        try:
            if '6517B' in gpib_address:
                device = Keithley6517B(gpib_address.split(' ')[0], nplc=nplc)
            if '6430' in gpib_address:
                device = Keithley6430(gpib_address.split(' ')[0], nplc=nplc)
        except:
            pass
    time.sleep(1)
    return device

def get_devices_list():
    l = []
    a = None
    rm = ResourceManager()
    try : 
        a = rm.list_resources()
    except Exception as E:
        print(E)
        pass
    for r in a:
        try:
            res = rm.open_resource(r)
            res.write('*CLS;*RST;*IDN?;')
            id = res.read()
        except:
            id = 'Unkown'
        if 'keithley' in id.lower():
            l.append([r, id.split(',')[1]])
    return l
    

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
        self.output_enabled = False

    def clear_buffer(self):
        """Clears the instrument's input buffer."""
        self.device.write('*CLS;')

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
        if range_value > 20e-3:
            return
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
        if not self.output_enabled:
            self.device.write('OUTP ON')  # Turn the output on
            self.output_enabled = True
            time.sleep(1)

    def disable_output(self):
        """
        Set the output voltage.
        :param voltage_value: Voltage to set (in volts).
        """
        self.output_enabled = False
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
        self.output_enabled = False
        self.clear_buffer()
        self.init_device(nplc)
        
    def init_device(self, nplc):
        self.device.write('*CLS;')
        self.device.write('*RST;')
        self.device.write(":SENS:FUNC 'CURR';")
        self.device.write(f":SENS:CURR:NPLC {nplc};")
        self.device.write(f":SOUR:VOLT:MODE FIX;")
        self.device.write(f":SENS:CURR:PROT 105e-3;")
        self.output_enabled = False

    def set_source_mode(self, mode):
        """Set source mode to either 'voltage' or 'current'"""
        if mode.lower() not in ['voltage', 'current']:
            raise ValueError("Mode must be either 'voltage' or 'current'")
        
        self.source_mode = mode.lower()
        if self.source_mode == 'voltage':
            self.device.write(":SENS:FUNC 'CURR';")
            self.device.write(":SOUR:FUNC VOLT")
        else:
            self.device.write(":SENS:FUNC 'VOLT';")
            self.device.write(":SOUR:FUNC CURR")

    def clear_buffer(self):
        """Clears the instrument's input buffer."""
        self.device.write('*CLS;')

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
        if range_value > 105e-3:
            return
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
        if not self.output_enabled:
            self.device.write('OUTP ON;')  # Turn the output on
            self.output_enabled = True

    def set_current(self, current_value):
        if self.source_mode != 'current':
            raise ValueError("Device is not in current source mode")
        self.device.write(f'SOUR:CURR:LEV {current_value};')
        if not self.output_enabled:
            self.device.write('OUTP ON;')
            self.output_enabled = True
            # time.sleep(1)

    def disable_output(self):
        """
        Set the output voltage.
        :param voltage_value: Voltage to set (in volts).
        """
        self.output_enabled = False
        self.device.write('OUTP OFF;')  # Turn the output on

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
        self.device.write('OUTP OFF;')  # Turn the output off
        self.device.close()
        print("Connection to Keithley 6517B closed.")
    
    def read_voltage(self, autorange=False):
        if self.source_mode != 'current':
            raise ValueError("Device is not in current source mode")
        if autorange:
            self.device.write(':MEAS:VOLT?;')
        else:
            self.device.write(':READ?;')
        voltage_value = self.device.read().split(',')[0][:-4:]
        return float(voltage_value)


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
        self.device = mock.Mock()

    def init_device(self, nplc):
        print(f"Initializing mock device with NPLC={nplc}")
        self.output_enabled = False

    def set_source_mode(self, mode):
        """Set source mode to either 'voltage' or 'current'"""
        if mode.lower() not in ['voltage', 'current']:
            raise ValueError("Mode must be either 'voltage' or 'current'")
        
        self.source_mode = mode.lower()
        if self.source_mode == 'voltage':
            self.device.write(":SENS:FUNC 'CURR';")
            self.device.write(":SOUR:FUNC VOLT")
        else:
            self.device.write(":SENS:FUNC 'VOLT';")
            self.device.write(":SOUR:FUNC CURR")

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

    def set_current(self, current_value):
        if self.source_mode != 'current':
            raise ValueError("Device is not in current source mode")
        self.device.write(f'SOUR:CURR:LEV {current_value};')
        if not self.output_enabled:
            self.device.write('OUTP ON')
            self.output_enabled = True
            # time.sleep(1)

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
    
    def read_voltage(self, autorange=False):
        if self.source_mode != 'current':
            raise ValueError("Device is not in current source mode")
        # if autorange:
        #     self.device.write(':MEAS:VOLT?;')
        # else:
        #     self.device.write(':READ?;')
        voltage_value = 1 + 0.1 * (2 * (np.random.random() - 0.5))
        return float(voltage_value)
    
    
