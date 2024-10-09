import pyvisa
import time

class Keithley6517B:
    def __init__(self, gpib_address='27'):
        self.rm = pyvisa.ResourceManager()
        self.device = self.rm.open_resource(f'GPIB::{gpib_address}::INSTR')
        self.device.timeout = 5000  # Timeout for commands, can be adjusted if needed
        self.clear_buffer()
        self.init_device()
        
    def init_device(self):
        self.device.write('*CLS;')
        # self.device.write('*RST;')
        # self.device.write(':TRAC:FEED:CONT NEV;*RST')
        # selg.de
        # self.device.write('*ESE 60;*SRE 48;*CLS;:FORM:DATA ASC;:FORM:ELEM READ,RNUM,CHAN,TST,UNIT;:SYST:TST:TYPE RTCL;')
        self.device.write("SYST:ZCH OFF;")
        self.device.write(":SENS:FUNC 'CURR';")

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
        print(f"Current range set to {range_value} A")

    def set_voltage(self, voltage_value):
        """
        Set the output voltage.
        :param voltage_value: Voltage to set (in volts).
        """
        self.device.write(f'SOUR:VOLT:LEV {voltage_value};')
        self.device.write('OUTP ON')  # Turn the output on
        print(f"Voltage set to {voltage_value} V and output enabled")

    def read_current(self, autorange=False):
        """
        Read the current from the input.
        :return: The measured current (in amps).
        """
        if autorange:
            self.device.write('MEAS:CURR?;')
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
        self.device.write('OUTP OFF')  # Turn the output off
        self.device.close()
        print("Connection to Keithley 6517B closed.")
