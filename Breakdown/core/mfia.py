"""
Zurich Instruments MFIA session -- the hardware side of C(f) and C(V).

Uses the low-level ``zhinst.core`` node API, matching ``CV_MAP/``; there is no
``zhinst.toolkit`` and no LabOne sweeper module anywhere in this repo.

Several habits here look defensive because they are, and each one is load-bearing:

* **Node names vary between firmware versions.** Writes go through
  :func:`try_set_double` / :func:`try_set_int`, which walk a list of candidate
  paths until one is accepted.
* **The drive amplitude has no single node.** :func:`set_drive_amplitude` writes
  every plausible node *and* resolves the underlying
  ``sigouts/<i>/amplitudes/<osc>`` pair, enabling the output.
* **DC bias is always ramped, never jumped** -- a step onto a capacitor under
  test is a current spike.

``zhinst`` is imported lazily so the module stays importable without it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .mfia_nodes import extract_sample_payload
from .params import MfiaParams

MOCK_SELECTION = "Mock"


def _zi():
    import zhinst.core as zi

    return zi


# --- node helpers ----------------------------------------------------------

def try_get_double(daq, path: str) -> Optional[float]:
    try:
        return float(daq.getDouble(path))
    except Exception:
        return None


def try_get_int(daq, path: str) -> Optional[int]:
    try:
        return int(daq.getInt(path))
    except Exception:
        return None


def try_set_double(daq, paths: List[str], value: float) -> Optional[str]:
    """Write to the first path the firmware accepts; return which one worked."""
    for path in paths:
        try:
            daq.setDouble(path, float(value))
            return path
        except Exception:
            continue
    return None


def try_set_int(daq, paths: List[str], value: int) -> Optional[str]:
    for path in paths:
        try:
            daq.setInt(path, int(value))
            return path
        except Exception:
            continue
    return None


def ramp_values(current: float, target: float, step: float) -> List[float]:
    if step <= 0:
        return [target]
    if abs(target - current) <= 1e-12:
        return [target]
    steps = max(1, int(abs(target - current) / max(step, 1e-12)))
    span = target - current
    return [current + span * (i + 1) / steps for i in range(steps)]


def ramp_set_double(daq, path: str, target: float, step: float, wait_s: float) -> None:
    try:
        current = float(daq.getDouble(path))
    except Exception:
        daq.setDouble(path, float(target))
        if wait_s > 0:
            time.sleep(wait_s)
        return
    for value in ramp_values(current, float(target), float(step)):
        daq.setDouble(path, float(value))
        if wait_s > 0:
            time.sleep(wait_s)


def find_mfia_device(host: str, port: int, preferred: Optional[str]) -> Tuple[str, str]:
    zi = _zi()
    discovery = zi.ziDiscovery()
    devices = discovery.findAll()
    if not devices:
        raise RuntimeError("No Zurich Instruments devices found via discovery.")

    if preferred:
        name = preferred.upper()
        info = discovery.get(name)
        if info.get("devicetype") != "MFIA":
            raise RuntimeError(f"Device {name} is not an MFIA: {info}")
        return name, info.get("connected", "")

    for name in devices:
        info = discovery.get(name)
        if info.get("devicetype") == "MFIA":
            return name, info.get("connected", "")
    raise RuntimeError("No MFIA found. Check that LabOne can see the instrument.")


def set_drive_amplitude(daq, dev: str, imps: int, value: float) -> List[str]:
    """Set the AC test-signal amplitude, whatever the firmware calls it."""
    base = f"/{dev}/imps/{imps}"
    written: List[str] = []

    for path in (
        f"{base}/drive", f"{base}/drive/value", f"{base}/drive/voltage",
        f"{base}/drive/level", f"{base}/drive/amplitude", f"{base}/amplitude",
        f"{base}/output/amplitude", f"{base}/osc/amp", f"{base}/osc/amplitude",
    ):
        if try_set_double(daq, [path], value):
            written.append(path)

    sigout = _first_int(daq, base, ("sigout", "sigoutselect", "outputselect", "output"))
    osc = _first_int(daq, base, ("oscselect", "oscindex", "oscillator", "osc"))
    if sigout is not None and osc is not None:
        amplitude_path = f"/{dev}/sigouts/{sigout}/amplitudes/{osc}"
        enable_path = f"/{dev}/sigouts/{sigout}/enables/{osc}"
        if try_set_double(daq, [amplitude_path], value):
            written.append(amplitude_path)
        if try_set_int(daq, [enable_path], 1):
            written.append(enable_path)
    return written


def _first_int(daq, base: str, leaves: Tuple[str, ...]) -> Optional[int]:
    for leaf in leaves:
        value = try_get_int(daq, f"{base}/{leaf}")
        if value is not None:
            return value
    return None


@dataclass
class StateSnapshot:
    int_nodes: Dict[str, int] = field(default_factory=dict)
    double_nodes: Dict[str, float] = field(default_factory=dict)


# --- session ---------------------------------------------------------------

class MfiaSession:
    """Implements :class:`~Breakdown.core.instruments.ImpedanceAnalyzer`."""

    def __init__(self, host: str, port: int, device_id: str = "", imps: int = 0):
        self.host = str(host)
        self.port = int(port)
        self.device_id = str(device_id or "")
        self.imps = int(imps)

        self.daq = None
        self.dev = ""
        self.interface = ""
        self._snapshot: Optional[StateSnapshot] = None
        self._subscribed = False
        self._last = {"bias": None, "amplitude": None, "frequency": None}
        self.ramp_step = 0.05
        self.ramp_wait = 0.01

    @property
    def base(self) -> str:
        return f"/{self.dev}/imps/{self.imps}"

    @property
    def sample_path(self) -> str:
        return f"{self.base}/sample"

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> "MfiaSession":
        zi = _zi()
        self.daq = zi.ziDAQServer(self.host, self.port, 6)  # LabOne API level 6
        device, interface = find_mfia_device(self.host, self.port,
                                             self.device_id or None)
        self.dev = device.lower()
        self.interface = interface or "PCIe"
        self.daq.connectDevice(self.dev, self.interface)
        self._snapshot = self._capture_state()
        return self

    def close(self) -> None:
        if self.daq is None:
            return
        try:
            self._unsubscribe()
            if self._snapshot is not None:
                self._restore_state(self._snapshot)
        finally:
            self.daq = None

    # -- ImpedanceAnalyzer protocol ----------------------------------------

    def configure(self, mfia_params: MfiaParams) -> None:
        daq, base = self.daq, self.base
        self.ramp_step = float(mfia_params.ramp_step)
        self.ramp_wait = float(mfia_params.ramp_wait)

        daq.setInt(f"{base}/enable", 1)
        daq.setInt(f"{base}/model", int(mfia_params.model))
        daq.setInt(f"{base}/bias/enable", 1)

        try_set_int(daq, [f"{base}/auto/bw"], int(mfia_params.auto_bw))
        try_set_int(daq, [f"{base}/auto/inputrange"], int(mfia_params.inputrange_mode))
        if mfia_params.inputrange_mode == 0:
            try_set_double(daq, [f"{base}/current/range"],
                           float(mfia_params.manual_current_range))

        try_set_int(
            daq,
            [f"{base}/precision", f"{base}/accuracy", f"{base}/quality"],
            int(mfia_params.quality),
        )
        try_set_int(daq, [f"{base}/demod/order"], int(mfia_params.demod_order))
        try_set_double(daq, [f"{base}/demod/timeconstant"],
                       float(mfia_params.demod_timeconstant))
        try_set_double(daq, [f"{base}/demod/rate"], float(mfia_params.demod_rate))
        try_set_int(daq, [f"{base}/demod/sinc"], int(mfia_params.demod_sinc))

        self._subscribe()

    def set_bias(self, voltage: float) -> None:
        if self._unchanged("bias", voltage):
            return
        ramp_set_double(self.daq, f"{self.base}/bias/value", float(voltage),
                        self.ramp_step, self.ramp_wait)
        self._last["bias"] = float(voltage)

    def set_amplitude(self, voltage: float) -> None:
        if self._unchanged("amplitude", voltage):
            return
        if set_drive_amplitude(self.daq, self.dev, self.imps, float(voltage)):
            self._last["amplitude"] = float(voltage)

    def set_frequency(self, frequency_hz: float) -> None:
        if self._unchanged("frequency", frequency_hz):
            return
        written = try_set_double(
            self.daq,
            [f"{self.base}/freq", f"{self.base}/frequency",
             f"{self.base}/osc/freq", f"{self.base}/osc/frequency"],
            float(frequency_hz),
        )
        if written:
            self._last["frequency"] = float(frequency_hz)

    def read_sample(self, timeout_s: float = 1.0) -> Dict[str, Any]:
        """Poll until the impedance sample arrives, retrying once on timeout."""
        try:
            return self._poll_sample(timeout_s)
        except TimeoutError:
            # One resubscribe cycle recovers the common case where the stream
            # stalled after a settings change.
            self.daq.sync()
            self._unsubscribe()
            self._subscribe()
            return self._poll_sample(timeout_s)

    def safe_off(self) -> None:
        """Zero the bias and drive. Must never raise -- the operator is waiting."""
        if self.daq is None:
            return
        try:
            ramp_set_double(self.daq, f"{self.base}/bias/value", 0.0,
                            self.ramp_step, self.ramp_wait)
        except Exception:
            pass
        for action in (
            lambda: set_drive_amplitude(self.daq, self.dev, self.imps, 0.0),
            lambda: self.daq.setInt(f"{self.base}/bias/enable", 0),
        ):
            try:
                action()
            except Exception:
                continue
        self._last = {"bias": 0.0, "amplitude": 0.0,
                      "frequency": self._last.get("frequency")}

    # -- internals ----------------------------------------------------------

    def _unchanged(self, key: str, value: float) -> bool:
        previous = self._last.get(key)
        return previous is not None and abs(previous - float(value)) <= 1e-15

    def _subscribe(self) -> None:
        if not self._subscribed:
            self.daq.subscribe(self.sample_path)
            self.daq.sync()
            self._subscribed = True

    def _unsubscribe(self) -> None:
        if self._subscribed:
            try:
                self.daq.unsubscribe(self.sample_path)
            except Exception:
                pass
            self._subscribed = False

    def _poll_sample(self, timeout_s: float) -> Dict[str, Any]:
        deadline = time.time() + max(0.2, float(timeout_s))
        seen: List[str] = []
        while time.time() < deadline:
            data = self.daq.poll(0.1, 10, 0, True)
            sample, nodes = extract_sample_payload(data, self.sample_path)
            if sample is not None:
                return sample
            if nodes:
                seen = nodes
        listed = ", ".join(seen[:4]) + (", ..." if len(seen) > 4 else "")
        raise TimeoutError(
            f"No impedance sample on {self.sample_path} within {timeout_s:.2f}s"
            + (f" (saw: {listed})" if seen else "")
        )

    def _capture_state(self) -> StateSnapshot:
        """Record the nodes we are about to change, so they can be put back."""
        base = self.base
        snapshot = StateSnapshot()
        for path in (f"{base}/enable", f"{base}/model", f"{base}/bias/enable",
                     f"{base}/auto/bw", f"{base}/auto/inputrange",
                     f"{base}/demod/order", f"{base}/demod/sinc"):
            value = try_get_int(self.daq, path)
            if value is not None:
                snapshot.int_nodes[path] = value
        for path in (f"{base}/bias/value", f"{base}/current/range",
                     f"{base}/demod/timeconstant", f"{base}/demod/rate",
                     f"{base}/freq", f"{base}/drive"):
            value = try_get_double(self.daq, path)
            if value is not None:
                snapshot.double_nodes[path] = value
        for path in (f"{base}/precision", f"{base}/accuracy", f"{base}/quality"):
            value = try_get_int(self.daq, path)
            if value is not None:
                snapshot.int_nodes[path] = value
                break
        return snapshot

    def _restore_state(self, snapshot: StateSnapshot) -> None:
        for path, value in snapshot.int_nodes.items():
            try:
                self.daq.setInt(path, int(value))
            except Exception:
                continue
        for path, value in snapshot.double_nodes.items():
            try:
                self.daq.setDouble(path, float(value))
            except Exception:
                continue


def open_impedance_analyzer(selection_text: str, params: MfiaParams):
    """Open the MFIA, or a simulated one when 'Mock' is chosen."""
    if str(selection_text).strip() == MOCK_SELECTION:
        from .mock import MockImpedanceAnalyzer

        return MockImpedanceAnalyzer()
    return MfiaSession(
        host=params.host, port=params.port,
        device_id=params.device_id, imps=params.imps,
    ).open()
