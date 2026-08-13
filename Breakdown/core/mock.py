"""
Simulated instruments.

Two uses, both real:

* **Tests.** The stress algorithms and the run state machine are exercised
  against these instead of hardware, so their error paths can be checked without
  destroying devices.
* **Rehearsal.** Selecting "Mock" in the GUI walks the whole per-device workflow
  -- prompts, folder tree, plots, summary -- with no instruments attached. Worth
  doing once before a campaign, since mistakes here are irreversible.

The physics is a caricature chosen to produce plausible-looking curves, not to
predict anything. It follows the repo's existing convention of shipping mock
backends (``keithley.Keithley6517B_Mock``, ``transport.MockSheetResistanceTransport``).
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Callable, Dict, Optional, Tuple

#: Exponent of the power-law lifetime model, t_BD ~ (V_BD / V)^n. A value
#: around 40 is typical for thin dielectrics and gives a usefully steep
#: dependence: 0.85*V_BD breaks down a few hundred times slower than V_BD.
LIFETIME_EXPONENT = 40.0

#: Time to breakdown exactly at V_BD. Sets the overall timescale.
LIFETIME_AT_VBD_S = 0.01


class MockSourceMeter:
    """A leaky capacitor that breaks down.

    Leakage is ``I = i0 * exp(|V| / v0)``. Above ``v_bd`` the device fails
    immediately; below it, damage accumulates at a rate set by the power-law
    lifetime model, so a constant-voltage hold eventually fails too.
    """

    def __init__(
        self,
        v_bd: float = 4.2,
        i0: float = 1e-12,
        v0: float = 0.45,
        seed: Optional[int] = None,
        weibull_beta: float = 0.0,
        renew_on_configure: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self._rng = random.Random(seed)
        # Stress damage accrues against this clock. Passing the same clock the
        # run loop uses is what makes a simulated hold break down on schedule --
        # in virtual time under test, in real time during a GUI rehearsal.
        self._clock = clock
        self._last_read: Optional[float] = None
        self.nominal_v_bd = float(v_bd)
        self.weibull_beta = float(weibull_beta)
        self.i0 = float(i0)
        self.v0 = float(v0)
        # One instrument measures many crosspoints. When this is set, each
        # stress run (which begins with configure()) gets a fresh device --
        # what "move the probes to the next crosspoint" means in simulation.
        self.renew_on_configure = bool(renew_on_configure)

        self.compliance_A = 1e-3
        self.nplc = 1.0
        self.current_autorange = False
        self.voltage = 0.0
        self.output_enabled = False
        self.closed = False
        #: Recorded so tests can assert the instrument was left safe.
        self.safe_off_calls = 0
        #: How many fresh crosspoints have been presented.
        self.devices_presented = 0
        self.renew()

    def renew(self) -> None:
        """Present an undamaged crosspoint, redrawing V_BD if scatter is on."""
        self.v_bd = self._draw_v_bd()
        self.broken = False
        self.damage = 0.0
        self._last_read = None
        self.devices_presented += 1

    def _draw_v_bd(self) -> float:
        # beta = 0 keeps every device identical, which is what deterministic
        # tests need; a positive beta gives Weibull-distributed scatter.
        if self.weibull_beta <= 0:
            return self.nominal_v_bd
        u = max(1e-9, min(1 - 1e-9, self._rng.random()))
        return self.nominal_v_bd * (-math.log(1.0 - u)) ** (1.0 / self.weibull_beta)

    # -- SourceMeter protocol ----------------------------------------------

    def configure(
        self, nplc: float, compliance_A: float, current_autorange: bool = False
    ) -> None:
        if self.renew_on_configure:
            self.renew()
        self.nplc = float(nplc)
        self.compliance_A = float(compliance_A)
        self.current_autorange = bool(current_autorange)

    def set_voltage(self, voltage: float) -> None:
        self.voltage = float(voltage)
        self.output_enabled = True

    def read_vi(self) -> Tuple[float, float]:
        # Charge the device for however long it has been sitting at this
        # voltage since the last reading.
        now = self._clock()
        if self._last_read is not None:
            self.advance_time(now - self._last_read)
        self._last_read = now
        return self.voltage, self._current()

    def output_off(self) -> None:
        self.output_enabled = False

    def safe_off(self) -> None:
        self.safe_off_calls += 1
        self.voltage = 0.0
        self.output_enabled = False

    def close(self) -> None:
        self.closed = True

    # -- simulation ---------------------------------------------------------

    def advance_time(self, dt_s: float) -> None:
        """Accumulate stress damage for a dwell of ``dt_s`` at the present voltage.

        The run loops call this so that a constant-voltage hold fails after a
        realistic time without the test having to wait for it.
        """
        if self.broken or not self.output_enabled or dt_s <= 0:
            return
        magnitude = abs(self.voltage)
        if magnitude <= 0:
            return
        if magnitude >= self.v_bd:
            self.broken = True
            return
        lifetime = LIFETIME_AT_VBD_S * (self.v_bd / magnitude) ** LIFETIME_EXPONENT
        self.damage += float(dt_s) / lifetime
        if self.damage >= 1.0:
            self.broken = True

    def _current(self) -> float:
        if not self.output_enabled:
            return 0.0
        magnitude = abs(self.voltage)
        if magnitude >= self.v_bd:
            self.broken = True
        if self.broken:
            # A shorted device sits at the protection limit.
            return math.copysign(self.compliance_A, self.voltage or 1.0)
        leakage = self.i0 * math.exp(magnitude / self.v0)
        leakage = min(leakage, self.compliance_A)
        return math.copysign(leakage, self.voltage or 1.0)


class MockImpedanceAnalyzer:
    """A capacitor with mild frequency dispersion and a little C(V) hysteresis."""

    def __init__(
        self,
        c0: float = 3.3e-12,
        r_parallel: float = 5e9,
        dispersion: float = 0.03,
        v_coefficient: float = 0.01,
        hysteresis: float = 0.015,
    ):
        self.c0 = float(c0)
        self.r_parallel = float(r_parallel)
        self.dispersion = float(dispersion)
        self.v_coefficient = float(v_coefficient)
        self.hysteresis = float(hysteresis)

        self.bias = 0.0
        self.amplitude = 0.05
        self.frequency = 1000.0
        self.model = 0
        self.configured = False
        self.closed = False
        self.safe_off_calls = 0
        self._previous_bias = 0.0
        self._direction = 1.0

    # -- ImpedanceAnalyzer protocol ----------------------------------------

    def configure(self, mfia_params: Any) -> None:
        self.configured = True
        self.model = int(getattr(mfia_params, "model", 0))

    def set_bias(self, voltage: float) -> None:
        value = float(voltage)
        if value > self._previous_bias:
            self._direction = 1.0
        elif value < self._previous_bias:
            self._direction = -1.0
        self._previous_bias = value
        self.bias = value

    def set_amplitude(self, voltage: float) -> None:
        self.amplitude = float(voltage)

    def set_frequency(self, frequency_hz: float) -> None:
        self.frequency = float(frequency_hz)

    def read_sample(self) -> Dict[str, Any]:
        capacitance = self._capacitance()
        omega = 2.0 * math.pi * self.frequency
        admittance = complex(1.0 / self.r_parallel, omega * capacitance)
        impedance = 1.0 / admittance if admittance != 0 else complex(float("inf"), 0.0)
        return {
            "param0": self.r_parallel,
            "param1": capacitance,
            "frequency": self.frequency,
            "drive": self.amplitude,
            "z": impedance,
            "timestamp": 0,
        }

    def safe_off(self) -> None:
        self.safe_off_calls += 1
        self.bias = 0.0
        self.amplitude = 0.0

    def close(self) -> None:
        self.closed = True

    # -- simulation ---------------------------------------------------------

    def _capacitance(self) -> float:
        # Gentle roll-off with frequency, quadratic bias dependence, and a
        # direction-dependent offset so a C(V) loop actually looks like a loop.
        decade = math.log10(max(self.frequency, 1.0) / 1000.0)
        dispersion = 1.0 - self.dispersion * decade
        bias_term = 1.0 + self.v_coefficient * self.bias ** 2
        hysteresis = 1.0 + self.hysteresis * self._direction
        return self.c0 * dispersion * bias_term * hysteresis
