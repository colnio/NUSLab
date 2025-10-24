import pyvisa
import time
import numpy as np
from pyvisa import ResourceManager

def find_range(current):
    ranges = 2 * 10.0**np.arange(-12.0, -1.0, 1.0)
    idx = np.where(ranges > current)[0]
    if len(idx) > 0:
        return ranges[min(idx)]
    return max(ranges)

def auto_range(device, p1=None, compl=1):
    if p1 is None:
        c = device.read_current(autorange=True)
        if find_range(abs(c) * 3) >= compl*0.9:
            pass
        else:
            device.set_current_range(find_range(abs(c) * 3))
    else:
        if find_range(abs(p1) * 3) >= compl*0.9:
            pass
        else:
            device.set_current_range(find_range(abs(p1) * 3))
        c = device.read_current()
    return c if abs(c) < 200e-3 else device.read_current(autorange=True)

def _parse_idn(idn_raw: str):
    """
    Parse SCPI *IDN? reply like 'KEITHLEY INSTRUMENTS INC.,MODEL 2002,1234567,B02 / A02'.
    Returns (vendor, model, serial, firmware) with best-effort cleanup.
    """
    if not idn_raw:
        return ("", "", "", "")
    parts = [p.strip() for p in idn_raw.strip().replace(';', ',').split(',')]
    # Pad to 4 tokens
    parts += [""] * (4 - len(parts))
    vendor, model, serial, fw = parts[:4]
    # Normalize model (often like 'MODEL 2002' or 'KEITHLEY 6517B')
    model_clean = model.upper().replace("MODEL", "").replace("KEITHLEY", "").strip()
    # If empty, try to infer from entire string
    if not model_clean:
        up = idn_raw.upper()
        for tag in ("2002", "2700", "6517B", "6517", "6514", "6430", "2400"):
            if tag in up:
                model_clean = tag
                break
    return (vendor.strip(), model_clean, serial.strip(), fw.strip())


def _model_to_class_name(model_upper: str):
    """
    Map normalized model text to one of our class names (string).
    """
    m = model_upper.upper()
    if "6517" in m:
        return "Keithley6517B"
    if "6430" in m or "2400" in m:
        return "Keithley6430"
    if "6514" in m:
        return "Keithley6514"
    if "2700" in m:
        return "Keithley2700"
    if "2002" in m:
        return "Keithley2002"
    return None

def get_devices_list():
    """
    Enumerate VISA resources, probe *IDN?, and return a list of (resource, model, class_name, idn_string).
    Only Keithley instruments are returned.
    """
    out = []
    rm = ResourceManager()
    try:
        resources = rm.list_resources()
    except Exception as E:
        print(f"VISA list_resources() failed: {E}")
        return out

    for r in resources:
        idn = ""
        try:
            res = rm.open_resource(r)
            # Be tolerant of termination settings
            res.read_termination = '\n'
            res.write_termination = '\n'
            res.timeout = 3000
            res.write('*CLS')
            res.write('*RST')
            # Some models need a moment after *RST
            time.sleep(0.05)
            res.write('*IDN?')
            idn = res.read().strip()
        except Exception as E:
            # Could not query; skip silently but print for debug
            # print(f"IDN failed on {r}: {E}")
            idn = ""
        finally:
            try:
                res.close()
            except Exception:
                pass

        if idn:
            vendor, model, serial, fw = _parse_idn(idn)
            class_name = _model_to_class_name(model)
            if class_name and 'KEITHLEY' in (vendor.upper() if vendor else idn.upper()):
                out.append((r, model, class_name, idn))
                print(f"{r} -> {idn}")
        else:
            # Not Keithley or not responsive; still print what we saw
            # print(f"{r} -> Unknown / no IDN")
            pass

    return out

def get_device(gpib_address, nplc):
    """
    Factory that returns the correct device instance based on:
      - 'Mock' literal (returns Keithley6517B_Mock)
      - A VISA resource string (GPIB0::xx::INSTR, USBx::..., ASRL, TCPIP0::...)
      - A hint string containing the model (e.g. 'GPIB0::16::INSTR 2002' or '2002')
    It tries *IDN? when a VISA resource is provided to auto-select the class.
    """
    if gpib_address == 'Mock':
        return Keithley6517B_Mock()

    # If the user passed a plain model (e.g. "2002") without a resource,
    # we still need a real VISA resource to open. In that case, try to auto-pick
    # the first matching instrument from the bus.
    def _construct_by_class_name(res_str: str, class_name: str):
        if class_name == "Keithley6517B":
            return Keithley6517B(res_str, nplc=nplc)
        if class_name == "Keithley6430":
            return Keithley6430(res_str, nplc=nplc)
        if class_name == "Keithley6514":
            return Keithley6514(res_str, nplc=nplc)
        if class_name == "Keithley2700":
            return Keithley2700(res_str, nplc=nplc)
        if class_name == "Keithley2002":
            return Keithley2002(res_str, nplc=nplc)
        return None

    rm = ResourceManager()

    # Case A: it's a VISA resource string; try to open and query *IDN?
    looks_like_resource = "::" in gpib_address or gpib_address.upper().startswith(("USB", "GPIB", "TCPIP", "ASRL"))

    # Case B: it's a hint string with a model inside (e.g. "... 6517B" or just "2002")
    model_hint = None
    for tag in ("6517B", "6517", "6430", "2400", "6514", "2700", "2002"):
        if tag in gpib_address.upper():
            model_hint = "6517B" if tag == "6517" else tag  # normalize 6517 → 6517B
            break

    try:
        if looks_like_resource:
            # Prefer probing the resource to get an authoritative IDN
            res_str = gpib_address.split()[0]  # in case they appended " 2002" etc.
            inst = rm.open_resource(res_str)
            inst.read_termination = '\n'
            inst.write_termination = '\n'
            inst.timeout = 4000
            try:
                inst.write('*CLS')
                inst.write('*IDN?')
                idn = inst.read().strip()
            except Exception:
                idn = ""
            finally:
                try:
                    inst.close()
                except Exception:
                    pass

            if idn:
                _, model, _, _ = _parse_idn(idn)
                class_name = _model_to_class_name(model)
                if class_name:
                    return _construct_by_class_name(res_str, class_name)

            # If IDN didn’t work, fall back to model hint (if any)
            if model_hint:
                return _construct_by_class_name(res_str, _model_to_class_name(model_hint))

            # Last resort: try 6430 (common for SMUs) to avoid returning None silently
            # but safer is to fail gracefully:
            return None

        else:
            # No VISA resource, but maybe they passed a model hint only
            if model_hint:
                # scan bus, pick first matching *IDN?
                try:
                    for r in rm.list_resources():
                        try:
                            res = rm.open_resource(r)
                            res.read_termination = '\n'
                            res.write_termination = '\n'
                            res.timeout = 2500
                            res.write('*IDN?')
                            idn = res.read().strip()
                        except Exception:
                            idn = ""
                        finally:
                            try:
                                res.close()
                            except Exception:
                                pass

                        if idn and model_hint in idn.upper():
                            class_name = _model_to_class_name(model_hint)
                            if class_name:
                                return _construct_by_class_name(r, class_name)
                except Exception:
                    pass
            # Nothing we can do without a real resource
            return None

    except Exception:
        return None
    

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
        self.gpib_address=gpib_address
        self.nplc=nplc
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
        # self.device.write(f":SENS:CURR:PROT 105e-3;")
        self.output_enabled = False
    def set_complicance_current(self, curr):
        # self.__init__(self.gpib_address, self.nplc)
        self.device.write(f"*RST;")
        # self.device.write(f":SENS:CURR:PROT:STAT ON")
        self.device.write(f':SENS:CURR:RANG:AUTO ON;')
        self.device.write(f":SENS:CURR:PROT {curr}")
        self.device.write(f':SENS:CURR:RANG {curr};')
        self.read_current(autorange=True)
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
        if range_value > 105e-3:
            range_value = 105e-3
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
        self.current_level = 0.0
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


class Keithley6514:
    """
    Minimal 6514 wrapper for low-current/voltage/ohms and charge reads.
    Includes Zero Check / Zero Correct helpers.
    """
    def __init__(self, gpib_address='GPIB0::22::INSTR', nplc=1):
        self.rm = pyvisa.ResourceManager()
        self.device = self.rm.open_resource(f'{gpib_address}')
        self.device.timeout = 5000
        self.clear_status()
        self.init_device(nplc)

    def clear_status(self):
        self.device.write('*CLS')

    def init_device(self, nplc=1):
        # Put into a clean state and default to current function, one-shot style.
        self.device.write('*RST')
        self.device.write("CONF:CURR")  # “one-shot” style configuration (CONF + READ?), see manual
        # NPLC control is rate-related; many uses work fine at default.
        # If you prefer: self.device.write(f"SYST:AZER:STAT ON") to keep autozero on.
        # Zero Check off by default for live measurements
        self.zero_check(False)
        self.zero_correct(False)

    # ---- Zero Check / Zero Correct helpers (6514) ----
    def zero_check(self, enable=True):
        self.device.write(f"SYST:ZCH {'ON' if enable else 'OFF'}")

    def zero_correct(self, enable=True):
        self.device.write(f"SYST:ZCOR {'ON' if enable else 'OFF'}")

    def zero_acquire(self, function='CURR', range_value=None):
        """
        Classic zeroing flow:
          1) ZCHK ON
          2) Select function & range
          3) INIT one reading (offset sampling)
          4) ZCOR:ACQ
          5) ZCHK OFF
          6) ZCOR ON
        """
        self.zero_check(True)
        if function.upper() == 'CURR':
            self.device.write("FUNC 'CURR'")
            if range_value is not None:
                self.device.write(f"CURR:RANG {range_value}")
        elif function.upper() == 'VOLT':
            self.device.write("FUNC 'VOLT'")
            if range_value is not None:
                self.device.write(f"VOLT:RANG {range_value}")
        elif function.upper() == 'RES':
            self.device.write("FUNC 'RES'")
            if range_value is not None:
                self.device.write(f"RES:RANG {range_value}")
        # Trigger one reading to capture offset
        self.device.write("INIT")
        # Acquire the zero-correct value
        self.device.write("SYST:ZCOR:ACQ")
        self.zero_check(False)
        self.zero_correct(True)

    # ---- Basic function + range ----
    def set_function(self, func='CURR'):
        func = func.upper()
        if func not in ('CURR', 'VOLT', 'RES', 'CHARGE'):
            raise ValueError("Function must be one of: CURR, VOLT, RES, CHARGE")
        if func == 'CHARGE':
            self.device.write("CONF:CHAR")
        else:
            self.device.write(f"FUNC '{func}'")

    def set_current_range(self, range_value):
        self.device.write(f"CURR:RANG {range_value}")

    def set_voltage_range(self, range_value):
        self.device.write(f"VOLT:RANG {range_value}")

    def set_resistance_range(self, range_value):
        self.device.write(f"RES:RANG {range_value}")

    # ---- Reading helpers ----
    def read_current(self, autorange=False):
        # For “fresh” readings, READ? is the safest one-shot (INIT + FETCh?)
        if autorange:
            self.device.write("CURR:RANG:AUTO ON")
        self.device.write("READ?")
        # 6514 returns just the reading (or a CSV when math/extra elems on); parse float head
        resp = self.device.read().strip()
        return float(resp.split(',')[0])

    def read_voltage(self):
        self.device.write("CONF:VOLT")
        self.device.write("READ?")
        return float(self.device.read().split(',')[0])

    def read_resistance(self):
        self.device.write("CONF:RES")
        self.device.write("READ?")
        return float(self.device.read().split(',')[0])

    def read_charge(self):
        self.device.write("CONF:CHAR")
        self.device.write("READ?")
        return float(self.device.read().split(',')[0])

    def close(self):
        self.device.close()

# ===================================
# NEW: Keithley 2700 DMM/Switch Sys
# ===================================
class Keithley2700:
    """
    Lightweight 2700 wrapper for basic DCV/DCI/Ohms reads and ranging.
    Designed for single-channel “one-shot” use; scanning/channel list can be added as needed.
    """
    def __init__(self, gpib_address='GPIB0::17::INSTR', nplc=1):
        self.rm = pyvisa.ResourceManager()
        self.device = self.rm.open_resource(f'{gpib_address}')
        self.device.timeout = 5000
        self.clear_status()
        self.init_device(nplc)

    def clear_status(self):
        self.device.write('*CLS')

    def init_device(self, nplc=1):
        self.device.write('*RST')
        # Default to current function
        self.device.write("FUNC 'CURR'")
        # Many setups use NPLC; most 2700 firmwares accept SENS:CURR:NPLC / SENS:VOLT:NPLC
        try:
            self.device.write(f"SENS:CURR:NPLC {nplc}")
        except Exception:
            pass  # keep defaults if model fw differs
        # one-shot measurement model by default after *RST

    # ---- Function + range ----
    def set_function(self, func='CURR'):
        func = func.upper()
        if func not in ('CURR', 'VOLT', 'RES', 'FRES'):
            raise ValueError("Function must be one of: CURR, VOLT, RES, FRES")
        self.device.write(f"FUNC '{func}'")

    def set_current_range(self, range_value=None, auto=False):
        if auto:
            self.device.write("CURR:RANG:AUTO ON")
        elif range_value is not None:
            self.device.write(f"CURR:RANG {range_value}")

    def set_voltage_range(self, range_value=None, ac=False, auto=False):
        root = "VOLT:AC" if ac else "VOLT"
        if auto:
            self.device.write(f"{root}:RANG:AUTO ON")
        elif range_value is not None:
            self.device.write(f"{root}:RANG {range_value}")

    def set_resistance_range(self, range_value=None, four_wire=False, auto=False):
        root = "FRES" if four_wire else "RES"
        if auto:
            self.device.write(f"{root}:RANG:AUTO ON")
        elif range_value is not None:
            self.device.write(f"{root}:RANG {range_value}")

    # ---- Reading helpers ----
    def read_current(self, autorange=False):
        if autorange:
            self.device.write("CURR:RANG:AUTO ON")
        self.device.write("MEAS:CURR?")
        # 2700 “MEAS?” returns a single fresh reading; parse first token
        return float(self.device.read().split(',')[0][:-3])

    def read_voltage(self, ac=False, autorange=False):
        if autorange:
            if ac:
                self.device.write("VOLT:AC:RANG:AUTO ON")
            else:
                self.device.write("VOLT:RANG:AUTO ON")
        cmd = "MEAS:VOLT:AC?" if ac else "MEAS:VOLT?"
        self.device.write(cmd)
        return float(self.device.read().split(',')[0][:-3])

    def read_resistance(self, four_wire=False, autorange=False):
        if autorange:
            if four_wire:
                self.device.write("FRES:RANG:AUTO ON")
            else:
                self.device.write("RES:RANG:AUTO ON")
        cmd = "MEAS:FRES?" if four_wire else "MEAS:RES?"
        self.device.write(cmd)
        return float(self.device.read().split(',')[0])

    def close(self):
        self.device.close()

class Keithley2002:
    """
    Minimal Keithley 2002 DMM wrapper for DCV/DCI/ACV/ACI/2W-Ohm/4W-Ohm/Freq/Temp.
    - Uses :SENSe:FUNC to pick function
    - Uses per-function :...:NPLCycles for integration (PLC)
    - One-shot reads via :MEASure? (or :READ? after a prior :CONFigure/:FUNC)
    """
    def __init__(self, gpib_address='GPIB0::16::INSTR', nplc=1, default_func='VOLT:DC'):
        self.rm = pyvisa.ResourceManager()
        self.device = self.rm.open_resource(f'{gpib_address}')
        self.device.timeout = 5000
        self.clear_status()
        self.init_device(nplc=nplc, func=default_func)

    # ---------- Setup ----------
    def clear_status(self):
        self.device.write('*CLS')

    def init_device(self, nplc=1, func='VOLT:DC'):
        """Put the meter in a clean 'one-shot' friendly state."""
        self.device.write('*RST')
        # Only return the reading value to simplify parsing (optional but handy)
        self.device.write(':FORM:ELE READ')
        # Pick function & nplc
        self.set_function(func)
        self.set_nplc(nplc, func)

    # ---------- Function / NPLC ----------
    def set_function(self, func: str):
        """
        func examples: 'VOLT:DC','VOLT:AC','CURR:DC','CURR:AC','RES','FRES','FREQ','TEMP'
        """
        func = func.upper()
        self.device.write(f":SENS:FUNC '{func}'")

    def set_nplc(self, nplc: float = 1.0, func: str | None = None):
        """
        Set integration time (in power-line cycles) for the active or specified function.
        Valid 0.01 .. 50 PLC on the 2002.
        """
        if func is None:
            # Map current function to the right NPLC path by querying FUNCTION?
            self.device.write(':SENS:FUNC?')
            cur = self.device.read().strip().replace('"', '').replace("'", '').upper()
            func = cur
        root = None
        if func.startswith('VOLT:DC'):
            root = ':SENS:VOLT:DC'
        elif func.startswith('VOLT:AC'):
            root = ':SENS:VOLT:AC'
        elif func.startswith('CURR:DC'):
            root = ':SENS:CURR:DC'
        elif func.startswith('CURR:AC'):
            root = ':SENS:CURR:AC'
        elif func.startswith('RES'):
            root = ':SENS:RES'
        elif func.startswith('FRES'):
            root = ':SENS:FRES'
        elif func.startswith('FREQ'):
            root = ':SENS:FREQ'
        elif func.startswith('TEMP'):
            root = ':SENS:TEMP'
        else:
            # Fallback: if unknown, set the generic NPLC (applies to current function)
            root = ':SENS'
        self.device.write(f'{root}:NPLC {nplc}')

    # ---------- Ranging helpers ----------
    def set_range_auto(self, enable=True, func: str | None = None, four_wire=False, ac=False):
        """Enable/disable autorange for the current or specified function."""
        if func is None:
            self.device.write(':SENS:FUNC?')
            func = self.device.read().strip().replace('"', '').replace("'", '').upper()

        def wr(cmd): self.device.write(cmd)

        if func.startswith('VOLT'):
            root = 'VOLT:AC' if (ac or func.endswith(':AC')) else 'VOLT:DC'
            wr(f':SENS:{root}:RANG:AUTO {"ON" if enable else "OFF"}')
        elif func.startswith('CURR'):
            root = 'CURR:AC' if (ac or func.endswith(':AC')) else 'CURR:DC'
            wr(f':SENS:{root}:RANG:AUTO {"ON" if enable else "OFF"}')
        elif func.startswith('FRES') or (four_wire and func.startswith('RES')):
            wr(f':SENS:FRES:RANG:AUTO {"ON" if enable else "OFF"}')
        elif func.startswith('RES'):
            wr(f':SENS:RES:RANG:AUTO {"ON" if enable else "OFF"}')

    def set_range(self, value: float, func: str | None = None, four_wire=False, ac=False):
        """Manual range set for voltage/current/ohms."""
        if func is None:
            self.device.write(':SENS:FUNC?')
            func = self.device.read().strip().replace('"', '').replace("'", '').upper()

        def wr(cmd): self.device.write(cmd)

        if func.startswith('VOLT'):
            root = 'VOLT:AC' if (ac or func.endswith(':AC')) else 'VOLT:DC'
            wr(f':SENS:{root}:RANG {value}')
        elif func.startswith('CURR'):
            root = 'CURR:AC' if (ac or func.endswith(':AC')) else 'CURR:DC'
            wr(f':SENS:{root}:RANG {value}')
        elif func.startswith('FRES') or (four_wire and func.startswith('RES')):
            wr(f':SENS:FRES:RANG {value}')
        elif func.startswith('RES'):
            wr(f':SENS:RES:RANG {value}')

    # ---------- One-shot reads (preferred) ----------
    def meas_voltage_dc(self):
        self.device.write(':MEAS:VOLT:DC?')
        return float(self.device.read().split(',')[0])

    def meas_voltage_ac(self):
        self.device.write(':MEAS:VOLT:AC?')
        return float(self.device.read().split(',')[0])

    def meas_current_dc(self):
        self.device.write(':MEAS:CURR:DC?')
        return float(self.device.read().split(',')[0])

    def meas_current_ac(self):
        self.device.write(':MEAS:CURR:AC?')
        return float(self.device.read().split(',')[0])

    def meas_resistance_2w(self):
        self.device.write(':MEAS:RES?')
        return float(self.device.read().split(',')[0])

    def meas_resistance_4w(self):
        self.device.write(':MEAS:FRES?')
        return float(self.device.read().split(',')[0])

    def meas_frequency(self):
        self.device.write(':MEAS:FREQ?')
        return float(self.device.read().split(',')[0])

    def meas_temperature(self):
        self.device.write(':MEAS:TEMP?')
        return float(self.device.read().split(',')[0])

    # ---------- Alternate style: READ? after CONFIG/FUNC ----------
    def read_once(self, func='VOLT:DC'):
        """Configure function, then do a single :READ? conversion and return the reading."""
        self.set_function(func)
        self.device.write(':READ?')
        return float(self.device.read().split(',')[0])

    # ---------- Close ----------
    def close(self):
        self.device.close()


# ==========================================================
# Adapter helpers for GUI measurement flows (current/voltage)
# ==========================================================
class CurrentSourceAdapter:
    """
    Wrap a current-source capable instrument (6517B/6430/etc.) to expose
    simple configure/set/read helpers for GUI workflows.
    """
    def __init__(self, device):
        self.device = device
        self.resource = getattr(device, 'device', None)
        self.output_enabled = False
        self.last_set_current = 0.0
        self.kind = self._detect_kind(device)

    def _detect_kind(self, device):
        if isinstance(device, Keithley6517B_Mock):
            return 'mock'
        if isinstance(device, Keithley6517B):
            return '6517B'
        if isinstance(device, Keithley6430):
            return 'smu'
        return None

    def configure(self, current_range=None, compliance_voltage=None, nplc=1.0):
        if self.kind is None:
            raise ValueError('Selected device cannot source current in this mode.')
        if self.kind == 'mock':
            return
        if self.resource is None:
            raise RuntimeError('Instrument VISA resource handle unavailable.')
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
            self.resource.write(":SENS:FUNC 'CURR'")
            try:
                self.resource.write(f":SENS:CURR:NPLC {float(nplc)}")
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
            try:
                self.resource.write(":FORM:ELEM VOLT,CURR")
            except Exception:
                self.resource.write(":FORM:ELEM CURR")

    def set_current(self, value):
        if self.kind == 'mock':
            setattr(self.device, 'current_level', value)
            # keep legacy mock behaviour that uses voltage_level as proxy input
            self.device.voltage_level = value
            if not getattr(self.device, 'output_enabled', False):
                self.device.output_enabled = True
            self.last_set_current = value
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
            self.last_set_current = float(value)
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
            return getattr(self.device, 'voltage_level', 0.0)
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

    def read_measurement(self):
        if self.kind == 'mock':
            voltage = getattr(self.device, 'voltage_level', np.nan)
            try:
                current = float(self.device.read_current())
            except Exception:
                current = float(getattr(self.device, 'current_level', np.nan))
            return (float(voltage) if voltage is not None else np.nan,
                    current if current is not None else np.nan)
        if self.resource is not None:
            try:
                self.resource.write(":READ?")
                resp = self.resource.read().strip().split(',')
                numeric = []
                for token in resp:
                    try:
                        numeric.append(float(token))
                    except Exception:
                        continue
                if len(numeric) >= 2:
                    return numeric[0], numeric[1]
                if len(numeric) == 1:
                    return np.nan, numeric[0]
            except Exception:
                pass
        # Fallback to independent queries if :READ? flow failed
        return self.read_voltage(), self.read_current()

    def disable(self):
        if self.kind == 'mock':
            try:
                self.device.disable_output()
            except Exception:
                pass
            self.last_set_current = 0.0
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
        self.last_set_current = 0.0


class VoltageMeterAdapter:
    """
    Wrap a voltage-measuring instrument (SMU in sense mode, DMM, etc.)
    to provide consistent configure/read helpers.
    """
    def __init__(self, device):
        self.device = device
        self.resource = getattr(device, 'device', None)
        self.kind = self._detect_kind(device)

    def _detect_kind(self, device):
        if isinstance(device, Keithley6517B_Mock):
            return 'mock'
        if isinstance(device, Keithley6430):
            return 'smu'
        if isinstance(device, Keithley6514):
            return '6514'
        if isinstance(device, Keithley2700):
            return '2700'
        if isinstance(device, Keithley2002):
            return '2002'
        if isinstance(device, Keithley6517B):
            return '6517B'
        return None

    def configure(self, voltage_range=None, nplc=1.0):
        if self.kind is None:
            raise ValueError('Selected device cannot measure voltage.')
        if self.kind == 'mock':
            return
        if self.kind == 'smu':
            if self.resource is None:
                raise RuntimeError('Instrument VISA resource handle unavailable.')
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
                    self.resource.write(f":SENS:VOLT:RANG {voltage_range}")
                except Exception:
                    pass
            try:
                self.resource.write(":FORM:ELEM VOLT,CURR")
            except Exception:
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
                raise RuntimeError('Instrument VISA resource handle unavailable.')
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
            try:
                self.resource.write(":FORM:ELEM VOLT,CURR")
            except Exception:
                self.resource.write(":FORM:ELEM VOLT")

    def read_measurement(self):
        if self.kind == 'mock':
            voltage = getattr(self.device, 'voltage_level', 0.0)
            try:
                current = float(self.device.read_current())
            except Exception:
                current = np.nan
            return voltage, current
        if self.kind == 'smu':
            if self.resource is None:
                return np.nan, np.nan
            try:
                self.resource.write(":READ?")
                resp = self.resource.read().strip().split(',')
                voltage = float(resp[0])
                current = float(resp[1]) if len(resp) > 1 else np.nan
                return voltage, current
            except Exception:
                try:
                    self.resource.write(":MEAS:VOLT?")
                    voltage = float(self.resource.read().split(',')[0])
                except Exception:
                    voltage = np.nan
                try:
                    self.resource.write(":MEAS:CURR?")
                    current = float(self.resource.read().split(',')[0])
                except Exception:
                    current = np.nan
                return voltage, current
        if self.kind == '6514':
            try:
                voltage = float(self.device.read_voltage())
            except Exception:
                voltage = np.nan
            return voltage, np.nan
        if self.kind == '2700':
            try:
                voltage = float(self.device.read_voltage())
            except Exception:
                voltage = np.nan
            return voltage, np.nan
        if self.kind == '2002':
            try:
                voltage = float(self.device.meas_voltage_dc())
            except Exception:
                voltage = np.nan
            return voltage, np.nan
        if self.kind == '6517B':
            if self.resource is None:
                return np.nan, np.nan
            try:
                self.resource.write(":READ?")
                resp = self.resource.read().strip().split(',')
                voltage = float(resp[0])
                current = float(resp[1]) if len(resp) > 1 else np.nan
                return voltage, current
            except Exception:
                try:
                    self.resource.write(":MEAS:VOLT?")
                    voltage = float(self.resource.read().split(',')[0])
                except Exception:
                    voltage = np.nan
                try:
                    self.resource.write(":MEAS:CURR?")
                    current = float(self.resource.read().split(',')[0])
                except Exception:
                    current = np.nan
                return voltage, current
        return np.nan, np.nan

    def read_voltage(self):
        voltage, _ = self.read_measurement()
        return voltage

    def close(self):
        if hasattr(self.device, 'close'):
            try:
                self.device.close()
            except Exception:
                pass


class VoltageSourceAdapter:
    """
    Wrap a voltage-source capable instrument, providing set/read helpers and
    compliance control for GUI sweeps.
    """
    def __init__(self, device):
        self.device = device
        self.resource = getattr(device, 'device', None)
        self.output_enabled = False
        self.kind = self._detect_kind(device)

    def _detect_kind(self, device):
        if isinstance(device, Keithley6517B_Mock):
            return 'mock'
        if isinstance(device, Keithley6517B):
            return '6517B'
        if isinstance(device, Keithley6430):
            return 'smu'
        return None

    def configure(self, voltage_range=None, compliance_current=None, nplc=1.0):
        if self.kind is None:
            raise ValueError('Selected device cannot source voltage in this mode.')
        if self.kind == 'mock':
            return
        if self.resource is None:
            raise RuntimeError('Instrument VISA resource handle unavailable.')
        try:
            self.resource.write('*CLS')
        except Exception:
            pass

        autorange = voltage_range is None
        if self.kind == 'smu':
            self.resource.write(":SOUR:FUNC VOLT")
            self.resource.write(":SOUR:VOLT:MODE FIX")
            if autorange:
                self.resource.write(":SOUR:VOLT:RANG:AUTO ON")
            else:
                self.resource.write(f":SOUR:VOLT:RANG {voltage_range}")
            self.resource.write(":SOUR:VOLT:LEV 0")
            self.resource.write(":SENS:FUNC 'CURR'")
            try:
                self.resource.write(f":SENS:CURR:NPLC {float(nplc)}")
            except Exception:
                pass
            if compliance_current is not None:
                try:
                    self.resource.write(f":SENS:CURR:PROT {compliance_current}")
                    self.resource.write(f":SENS:CURR:RANG {abs(compliance_current)}")
                except Exception:
                    pass
            try:
                self.resource.write(":FORM:ELEM VOLT,CURR")
            except Exception:
                self.resource.write(":FORM:ELEM VOLT")
        elif self.kind == '6517B':
            self.resource.write(":SOUR:FUNC VOLT")
            self.resource.write(":SOUR:VOLT:MODE FIX")
            if autorange:
                self.resource.write(":SOUR:VOLT:RANG 1000")
            else:
                self.resource.write(f":SOUR:VOLT:RANG {voltage_range}")
            self.resource.write(":SOUR:VOLT:LEV 0")
            self.resource.write(":SENS:FUNC 'CURR'")
            try:
                self.resource.write(f":SENS:CURR:NPLC {float(nplc)}")
            except Exception:
                pass
            if compliance_current is not None:
                try:
                    self.resource.write(f":SENS:CURR:PROT {compliance_current}")
                except Exception:
                    pass
            try:
                self.resource.write(":FORM:ELEM VOLT,CURR")
            except Exception:
                self.resource.write(":FORM:ELEM VOLT")

    def set_voltage(self, voltage):
        if self.kind == 'mock':
            self.device.voltage_level = voltage
            self.device.output_enabled = True
            return
        if self.resource is None:
            return
        try:
            self.resource.write(f":SOUR:VOLT:LEV {voltage}")
        except Exception as exc:
            raise RuntimeError(f'Failed to program voltage level: {exc}')
        if not self.output_enabled:
            try:
                self.resource.write('OUTP ON')
            except Exception:
                pass
            self.output_enabled = True

    def read_measurement(self):
        if self.kind == 'mock':
            voltage = getattr(self.device, 'voltage_level', np.nan)
            try:
                current = float(self.device.read_current())
            except Exception:
                current = np.nan
            return voltage, current
        if self.resource is not None:
            try:
                self.resource.write(":READ?")
                resp = self.resource.read().strip().split(',')
                voltage = float(resp[0])
                current = float(resp[1]) if len(resp) > 1 else np.nan
                return voltage, current
            except Exception:
                try:
                    self.resource.write(":MEAS:VOLT?")
                    voltage = float(self.resource.read().split(',')[0])
                except Exception:
                    voltage = np.nan
                try:
                    self.resource.write(":MEAS:CURR?")
                    current = float(self.resource.read().split(',')[0])
                except Exception:
                    current = np.nan
                return voltage, current
        return np.nan, np.nan

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
