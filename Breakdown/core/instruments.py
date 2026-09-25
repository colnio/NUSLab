"""
Instrument interfaces and result types.

The measurement algorithms (:mod:`Breakdown.core.stress`,
:mod:`Breakdown.core.sweeps`) are written against these protocols rather than
against ``pyvisa`` or ``zhinst`` directly. That is what lets the ramp logic, the
breakdown response, and the sweep bookkeeping be tested against
:mod:`Breakdown.core.mock` -- on a machine with no instrument drivers, and
without risking a real device to exercise an error path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class SourceMeter(Protocol):
    """A voltage source with a current measurement -- the Keithley 2400's role."""

    def configure(
        self, nplc: float, compliance_A: float, current_autorange: bool = False,
        source_delay_s: float = 0.0,
    ) -> None:
        ...

    def set_voltage(self, voltage: float) -> None:
        ...

    def read_vi(self) -> Tuple[float, float]:
        """Return ``(measured_V, measured_I)``."""

    def output_off(self) -> None:
        ...

    def safe_off(self) -> None:
        """Attempt zero and output-off; raise only after attempting both."""

    def close(self) -> None:
        ...


@runtime_checkable
class ImpedanceAnalyzer(Protocol):
    """A bias + drive source with an impedance measurement -- the MFIA's role."""

    def configure(self, mfia_params: Any) -> None:
        ...

    def set_bias(
        self, voltage: float, readback_tolerance_V: Optional[float] = None
    ) -> None:
        ...

    def set_amplitude(
        self, voltage: float, readback_tolerance_V: Optional[float] = None
    ) -> None:
        ...

    def set_frequency(self, frequency_hz: float) -> None:
        ...

    def read_sample(self) -> Dict[str, Any]:
        """Return the latest impedance sample.

        Expected keys: ``param0``, ``param1``, ``frequency``, ``drive``, ``z``.
        Missing keys are tolerated by the sweep layer and land as NaN.
        """

    def safe_off(self) -> None:
        """Apply the analyzer's defined safe idle state.

        The MFIA implementation intentionally remains enabled at 10 mV AC,
        0 V DC bias, and 100 kHz.
        """

    def close(self) -> None:
        ...


# --- results ---------------------------------------------------------------

class Termination:
    BREAKDOWN = "breakdown"
    CEILING_REACHED = "v_max_reached"
    DURATION_REACHED = "max_duration"
    STOPPED = "stopped"
    COMPLETED = "completed"


@dataclass
class StressResult:
    stress_type: str
    termination_reason: str
    bd_detected: bool = False
    #: Signed breakdown voltage as sourced (negative for a negative campaign).
    v_bd: Optional[float] = None
    i_bd: Optional[float] = None
    #: Time to breakdown. For CVS this is measured from the moment the stress
    #: level was reached, not from the start of the pre-ramp.
    t_bd: Optional[float] = None
    v_stress: Optional[float] = None
    requested_rate_Vps: Optional[float] = None
    achieved_rate_Vps: Optional[float] = None
    point_count: int = 0
    duration_s: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class SweepResult:
    kind: str
    termination_reason: str
    point_count: int = 0
    duration_s: float = 0.0
    #: Capacitance at the reference frequency, for the campaign summary.
    c_reference_F: Optional[float] = None
    f_reference_Hz: Optional[float] = None
    notes: List[str] = field(default_factory=list)
