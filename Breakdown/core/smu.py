"""
Keithley SMU session -- the hardware side of the stress measurements.

Wraps ``KeithleyGUI/keithley.py`` rather than talking to VISA directly, so this
app inherits the same discovery, IDN parsing and safe-shutdown behaviour as the
other instruments in the lab.

``pyvisa`` is imported lazily, inside :meth:`SmuSession.open`. That keeps the
module importable -- and the rest of the core testable -- on a machine with no
VISA backend installed.

**Thread affinity:** open the instrument on the thread that will use it. The
existing apps do this too (``FastVAC.py`` constructs its device inside the
worker's ``run()``), and VISA handles do not travel well between threads.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from . import _paths  # noqa: F401  (puts KeithleyGUI on sys.path)

#: The 2400 reports this when a reading is invalid or overflowed.
OVERFLOW_SENTINEL = 9.9e37

MOCK_SELECTION = "Mock"


def list_devices(include_mock: bool = True) -> List[str]:
    """Discovery strings for the device combo, or just Mock if VISA is absent."""
    try:
        from KeithleyGUI.ui_helpers import build_device_display_list
        from KeithleyGUI import keithley

        return build_device_display_list(keithley, include_mock=include_mock)
    except Exception:
        return [MOCK_SELECTION] if include_mock else []


def _parse_reading(raw: str) -> Tuple[float, float]:
    """Split a ``:READ?`` reply into (voltage, current).

    ``Keithley2400.init_device`` sets ``:FORM:ELEM VOLT,CURR``, so element 0 is
    the sourced voltage and element 1 the measured current. Overflow readings
    become NaN, which the breakdown detector treats as a dropped sample rather
    than as evidence either way.
    """
    values = []
    for token in str(raw).split(","):
        try:
            value = float(token)
        except ValueError:
            continue
        values.append(math.nan if abs(value) >= OVERFLOW_SENTINEL else value)

    if len(values) >= 2:
        return values[0], values[1]
    if values:
        return math.nan, values[0]
    return math.nan, math.nan


class SmuSession:
    """Implements :class:`~Breakdown.core.instruments.SourceMeter`."""

    def __init__(self, selection_text: str, nplc: float = 1.0):
        self.selection_text = str(selection_text)
        self.nplc = float(nplc)
        self.device = None
        self.resource = None
        self.model = ""

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> "SmuSession":
        from KeithleyGUI import keithley

        token = keithley._extract_resource_token(self.selection_text)
        device = keithley.get_device(token, self.nplc)
        if device is None:
            raise RuntimeError(
                f"Could not open the source meter at '{self.selection_text}'. "
                f"Check the VISA connection and that no other program holds it."
            )
        if not isinstance(device, keithley.Keithley6430):
            raise RuntimeError(
                f"{type(device).__name__} cannot source voltage and measure "
                f"current; select a Keithley 2400 (or 6430)."
            )

        self.device = device
        self.resource = getattr(device, "device", None)
        self.model = type(device).__name__
        if self.resource is None:
            raise RuntimeError("Instrument VISA resource handle unavailable.")
        return self

    def close(self) -> None:
        if self.device is None:
            return
        from KeithleyGUI import keithley

        keithley.shutdown_device(self.device, close=True)
        self.device = None
        self.resource = None

    # -- SourceMeter protocol ----------------------------------------------

    def configure(
        self, nplc: float, compliance_A: float, current_autorange: bool = False
    ) -> None:
        self.nplc = float(nplc)
        self.device.nplc = float(nplc)
        self.device.set_source_mode("voltage")
        self.resource.write(f":SENS:CURR:NPLC {float(nplc)}")
        # Sets both :SENS:CURR:PROT and :SENS:CURR:RANG. Pinning the range is
        # wanted here: autorange glitches mid-ramp would look like current steps.
        self.device.set_compliance_current(compliance_A)
        self.resource.write(
            f":SENS:CURR:RANG:AUTO {'ON' if current_autorange else 'OFF'}"
        )

    def set_source_delay(self, seconds: float) -> None:
        self.resource.write(f":SOUR:DEL {max(0.0, float(seconds))}")

    def set_voltage(self, voltage: float) -> None:
        self.device.set_voltage(float(voltage))

    def read_vi(self) -> Tuple[float, float]:
        self.resource.write(":READ?")
        return _parse_reading(self.resource.read())

    def output_off(self) -> None:
        self.device.disable_output()

    def safe_off(self) -> None:
        """Drive to zero and open the output. Must never raise."""
        if self.device is None:
            return
        from KeithleyGUI import keithley

        keithley.shutdown_device(self.device, close=False)


def open_source_meter(selection_text: str, nplc: float = 1.0):
    """Open the selected SMU, or a simulated one when 'Mock' is chosen."""
    if str(selection_text).strip() == MOCK_SELECTION:
        from .mock import MockSourceMeter

        return MockSourceMeter(renew_on_configure=True, weibull_beta=12.0, seed=1)
    return SmuSession(selection_text, nplc=nplc).open()
