from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pyvisa

from .errors import ServiceError


METER_FUNCTIONS = {
    "VOLT:DC": "V", "VOLT:AC": "V", "CURR:DC": "A", "CURR:AC": "A",
    "RES": "Ohm", "FRES": "Ohm", "FREQ": "Hz", "TEMP": "C",
}
OVERLOAD = 9.9e37
LIST_MAX_POINTS = 2500


def finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ServiceError("invalid_value", f"{name} must be a number", 422) from exc
    if not math.isfinite(number):
        raise ServiceError("invalid_value", f"{name} must be finite", 422)
    return number


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def parse_idn(resource: str, raw: str) -> dict:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) < 2 or "KEITHLEY" not in parts[0].upper():
        raise ServiceError("unsupported_device", "Resource is not a recognized Keithley", 422, resource=resource, idn=raw)
    match = re.search(r"\b(2400|6430|2002)\b", parts[1].upper())
    if not match:
        raise ServiceError("unsupported_device", "Keithley model is unsupported", 422, resource=resource, idn=raw)
    return {
        "resource": resource, "model": match.group(1), "serial": parts[2] if len(parts) > 2 else "",
        "firmware": parts[3] if len(parts) > 3 else "", "idn": raw,
    }


def parse_reading(raw: str) -> tuple[float | None, bool]:
    try:
        value = float(raw.strip().split(",")[0])
    except ValueError as exc:
        raise ServiceError("invalid_readback", "Instrument returned a nonnumeric reading", 502, raw=raw[:200]) from exc
    overflow = not math.isfinite(value) or abs(value) >= OVERLOAD
    return (None if overflow else value), overflow


def parse_sweep(raw: str, expected: int) -> list[dict]:
    tokens = [token.strip() for token in raw.strip().split(",") if token.strip()]
    if len(tokens) != expected * 4:
        raise ServiceError("incomplete_readback", "Sweep returned an unexpected number of fields", 502,
                           expected_points=expected, received_fields=len(tokens))
    rows = []
    for index in range(expected):
        voltage, voltage_overload = parse_reading(tokens[4 * index])
        current, current_overload = parse_reading(tokens[4 * index + 1])
        elapsed, time_overload = parse_reading(tokens[4 * index + 2])
        try:
            status = int(float(tokens[4 * index + 3]))
        except ValueError as exc:
            raise ServiceError("invalid_readback", "Sweep status is not numeric", 502, point=index) from exc
        rows.append({
            "voltage_v": voltage, "current_a": current,
            "instrument_time_s": elapsed if elapsed is not None and elapsed >= 0 else None,
            "instrument_time_raw_s": elapsed, "timestamp_invalid": elapsed is None or elapsed < 0,
            "status_word": status, "in_compliance": bool(status & (1 << 3)),
            "range_compliance": bool(status & (1 << 16)),
            "overload": voltage_overload or current_overload or time_overload,
        })
    valid_times = [row["instrument_time_s"] for row in rows]
    if len(valid_times) > 1 and all(value is not None for value in valid_times):
        if any(later <= earlier for earlier, later in zip(valid_times, valid_times[1:])):
            for row in rows:
                row["instrument_time_s"] = None
                row["timestamp_invalid"] = True
    return rows


def visa_error(exc: Exception, resource: str) -> ServiceError:
    text = str(exc)
    is_timeout = "timeout" in text.lower() or getattr(exc, "error_code", None) == -1073807339
    return ServiceError("visa_timeout" if is_timeout else "visa_error", text, 504 if is_timeout else 502,
                        resource=resource)


@dataclass
class Instrument:
    info: dict
    handle: Any
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def model(self) -> str:
        return self.info["model"]

    @property
    def resource(self) -> str:
        return self.info["resource"]

    def query(self, command: str, timeout_ms: int | None = None) -> str:
        previous = self.handle.timeout
        try:
            if timeout_ms is not None:
                self.handle.timeout = timeout_ms
            return str(self.handle.query(command)).strip()
        except Exception as exc:
            raise visa_error(exc, self.resource) from exc
        finally:
            if timeout_ms is not None:
                self.handle.timeout = previous

    def write(self, command: str) -> None:
        try:
            self.handle.write(command)
        except Exception as exc:
            raise visa_error(exc, self.resource) from exc

    def errors(self) -> list[str]:
        found = []
        for _ in range(10):
            response = self.query(":SYST:ERR?")
            if re.match(r"^\s*\+?0\s*[,;]", response) or response.strip() in {"0", "+0"}:
                return found
            found.append(response)
        raise ServiceError("error_queue_overflow", "Instrument error queue did not clear", 502,
                           resource=self.resource, errors=found)

    def check_errors(self) -> None:
        errors = self.errors()
        if errors:
            raise ServiceError("instrument_error", "Instrument rejected a command", 502,
                               resource=self.resource, errors=errors)

    def _smu(self) -> None:
        if self.model not in {"2400", "6430"}:
            raise ServiceError("unsupported_operation", "Operation requires a source meter", 422,
                               resource=self.resource)

    def source_mode(self) -> str:
        self._smu()
        raw = self.query(":SOUR:FUNC?").upper().replace('"', "").replace("'", "")
        return "current" if "CURR" in raw else "voltage"

    def output_enabled(self) -> bool:
        self._smu()
        return self.query(":OUTP?").strip().upper() in {"1", "+1", "ON"}

    def safe_off(self) -> None:
        self._smu()
        failures = []
        for command in (":ABOR", ":OUTP OFF"):
            try:
                self.write(command)
            except ServiceError as exc:
                failures.append(str(exc))
        try:
            mode = self.source_mode()
            self.write(":SOUR:CURR:LEV 0" if mode == "current" else ":SOUR:VOLT:LEV 0")
            if self.output_enabled():
                failures.append("Output still enabled after OUTP OFF")
        except ServiceError as exc:
            failures.append(str(exc))
        if failures:
            raise ServiceError("shutdown_failed", "Could not verify source output is off", 502,
                               resource=self.resource, failures=failures)

    def configuration(self) -> dict:
        if self.model == "2002":
            return {"function": self.query(":SENS:FUNC?").strip("\"'"), "output_enabled": False}
        mode = self.source_mode()
        protection = self.query(":SENS:VOLT:PROT?" if mode == "current" else ":SENS:CURR:PROT?")
        return {"source_mode": mode, "output_enabled": self.output_enabled(),
                "compliance": finite_number(protection, "reported compliance")}

    def configure(self, settings: dict) -> dict:
        allowed = ({"function", "nplc", "range_auto", "range_value"} if self.model == "2002" else
                   {"source_mode", "nplc", "source_range", "sense_autorange", "sense_range", "compliance"})
        unexpected = set(settings) - allowed
        if unexpected:
            raise ServiceError("unsupported_operation", "Setting is unavailable for this model", 422,
                               settings=sorted(unexpected), model=self.model)
        for name in ("nplc", "source_range", "sense_range", "compliance", "range_value"):
            if settings.get(name) is not None and finite_number(settings[name], name) <= 0:
                raise ServiceError("invalid_value", f"{name} must be positive", 422)
        if self.model == "2002":
            func = settings.get("function")
            active_func = str(func or self.query(":SENS:FUNC?").strip("\"'")).upper()
            if active_func not in METER_FUNCTIONS:
                raise ServiceError("invalid_function", "Unsupported 2002 function", 422, function=active_func)
            if settings.get("nplc") is not None and active_func in {"FREQ", "TEMP", "VOLT:AC", "CURR:AC"}:
                raise ServiceError("unsupported_operation", "NPLC is unavailable for this function", 422,
                                   function=active_func)
            if (settings.get("range_auto") is not None or settings.get("range_value") is not None) and active_func not in {
                    "VOLT:DC", "VOLT:AC", "CURR:DC", "CURR:AC", "RES", "FRES"}:
                raise ServiceError("unsupported_operation", "Range is unavailable for this function", 422)
            if func is not None:
                func = str(func).upper()
                if func not in METER_FUNCTIONS:
                    raise ServiceError("invalid_function", "Unsupported 2002 function", 422, function=func)
                self.write(f":SENS:FUNC '{func}'")
            if settings.get("nplc") is not None:
                func = func or self.query(":SENS:FUNC?").strip("\"'").upper()
                if func in {"FREQ", "TEMP", "VOLT:AC", "CURR:AC"}:
                    raise ServiceError("unsupported_operation", "NPLC is unavailable for this function", 422, function=func)
                nplc = finite_number(settings["nplc"], "nplc")
                if nplc <= 0:
                    raise ServiceError("invalid_value", "nplc must be positive", 422)
                self.write(f":SENS:{func}:NPLC {nplc:.12g}")
            if settings.get("range_auto") is not None or settings.get("range_value") is not None:
                func = func or self.query(":SENS:FUNC?").strip("\"'").upper()
                if func not in {"VOLT:DC", "VOLT:AC", "CURR:DC", "CURR:AC", "RES", "FRES"}:
                    raise ServiceError("unsupported_operation", "Range is unavailable for this function", 422)
                root = f":SENS:{func}:RANG"
                if settings.get("range_auto") is not None:
                    self.write(f"{root}:AUTO {'ON' if settings['range_auto'] else 'OFF'}")
                if settings.get("range_value") is not None:
                    value = finite_number(settings["range_value"], "range_value")
                    if value <= 0:
                        raise ServiceError("invalid_value", "range_value must be positive", 422)
                    self.write(f"{root} {value:.12g}")
        else:
            mode = settings.get("source_mode")
            if mode is not None:
                if mode not in {"voltage", "current"}:
                    raise ServiceError("invalid_mode", "source_mode must be voltage or current", 422)
                if self.output_enabled():
                    raise ServiceError("output_active", "Switch output off before changing source mode", 409)
                self.write(f":SOUR:FUNC {'VOLT' if mode == 'voltage' else 'CURR'}")
                self.write(f":SOUR:{'VOLT' if mode == 'voltage' else 'CURR'}:MODE FIX")
                self.write(":SENS:FUNC 'CURR'" if mode == "voltage" else ":SENS:FUNC 'VOLT'")
            mode = mode or self.source_mode()
            sense = "CURR" if mode == "voltage" else "VOLT"
            source = "VOLT" if mode == "voltage" else "CURR"
            if settings.get("nplc") is not None:
                nplc = finite_number(settings["nplc"], "nplc")
                if nplc <= 0:
                    raise ServiceError("invalid_value", "nplc must be positive", 422)
                self.write(f":SENS:{sense}:NPLC {nplc:.12g}")
            if settings.get("source_range") is not None:
                value = finite_number(settings["source_range"], "source_range")
                if value <= 0:
                    raise ServiceError("invalid_value", "source_range must be positive", 422)
                self.write(f":SOUR:{source}:RANG {value:.12g}")
            if settings.get("sense_autorange") is not None:
                self.write(f":SENS:{sense}:RANG:AUTO {'ON' if settings['sense_autorange'] else 'OFF'}")
            if settings.get("sense_range") is not None:
                value = finite_number(settings["sense_range"], "sense_range")
                if value <= 0:
                    raise ServiceError("invalid_value", "sense_range must be positive", 422)
                self.write(f":SENS:{sense}:RANG {value:.12g}")
            if settings.get("compliance") is not None:
                value = finite_number(settings["compliance"], "compliance")
                if value <= 0:
                    raise ServiceError("invalid_value", "compliance must be positive", 422)
                self.write(f":SENS:{sense}:PROT {value:.12g}")
        self.check_errors()
        return self.configuration()

    def setpoint(self, value: float) -> dict:
        self._smu()
        value = finite_number(value, "setpoint")
        mode = self.source_mode()
        self.write(f":SOUR:{'CURR' if mode == 'current' else 'VOLT'}:LEV {value:.12g}")
        self.check_errors()
        return {"source_mode": mode, "setpoint": value, "output_enabled": self.output_enabled()}

    def output(self, enabled: bool) -> dict:
        self._smu()
        if enabled:
            self.write(":OUTP ON")
            self.check_errors()
            if not self.output_enabled():
                raise ServiceError("output_state_mismatch", "Output did not enable", 502)
        else:
            self.safe_off()
        return self.configuration()

    def measure(self, function: str | None = None) -> dict:
        timestamp = utc_now()
        if self.model == "2002":
            func = (function or self.query(":SENS:FUNC?").strip("\"'")).upper()
            if func not in METER_FUNCTIONS:
                raise ServiceError("invalid_function", "Unsupported 2002 function", 422, function=func)
            if function:
                self.write(f":SENS:FUNC '{func}'")
            self.write(":FORM:ELEM READ")
            value, overload = parse_reading(self.query(":READ?"))
            self.check_errors()
            return {"timestamp": timestamp, "function": func, "value": value,
                    "unit": METER_FUNCTIONS[func], "overload": overload}
        self.write(":FORM:ELEM VOLT,CURR,STAT")
        raw = self.query(":READ?")
        tokens = [token.strip() for token in raw.split(",")]
        if len(tokens) != 3:
            raise ServiceError("invalid_readback", "SMU returned unexpected fields", 502,
                               resource=self.resource, raw=raw[:200])
        voltage, voltage_overload = parse_reading(tokens[0])
        current, current_overload = parse_reading(tokens[1])
        try:
            status = int(float(tokens[2]))
        except ValueError as exc:
            raise ServiceError("invalid_readback", "SMU returned invalid status", 502) from exc
        self.check_errors()
        return {"timestamp": timestamp, "voltage_v": voltage, "current_a": current,
                "status_word": status, "in_compliance": bool(status & (1 << 3)),
                "range_compliance": bool(status & (1 << 16)),
                "overload": voltage_overload or current_overload}

    def raw_scpi(self, kind: str, command: str) -> dict:
        if not command or len(command) > 65536:
            raise ServiceError("invalid_command", "SCPI command must contain 1 to 65536 characters", 422)
        if kind == "query":
            response = self.query(command)
        elif kind == "write":
            self.write(command)
            response = None
        else:
            raise ServiceError("invalid_command", "kind must be write or query", 422)
        # The caller receives instrument errors without having to inspect the queue separately.
        errors = [] if ":SYST:ERR?" in command.upper() else self.errors()
        return {"response": response, "instrument_errors": errors}

    def sweep_batch(self, points: list[float], nplc: float, delay_s: float,
                    compliance_a: float | None, sense_range_a: float | None) -> list[dict]:
        self._smu()
        if not 1 <= len(points) <= LIST_MAX_POINTS:
            raise ServiceError("invalid_sweep", "Batch must contain 1 to 2500 points", 422)
        if self.output_enabled():
            raise ServiceError("output_active", "Switch output off before starting a list sweep", 409)
        self.write(":ABOR")
        self.write(":TRAC:CLE")
        self.write(":FORM:DATA ASC")
        self.write(":FORM:ELEM VOLT,CURR,TIME,STAT")
        self.write(":SOUR:FUNC VOLT")
        self.write(":SOUR:VOLT:MODE FIX")
        self.write(":SOUR:VOLT:LEV 0")
        self.write(":SENS:FUNC 'CURR'")
        self.write(f":SENS:CURR:NPLC {nplc:.12g}")
        self.write(f":SOUR:DEL:AUTO OFF")
        self.write(f":SOUR:DEL {delay_s:.12g}")
        if compliance_a is not None:
            self.write(f":SENS:CURR:PROT {compliance_a:.12g}")
        if sense_range_a is None:
            self.write(":SENS:CURR:RANG:AUTO ON")
        else:
            self.write(":SENS:CURR:RANG:AUTO OFF")
            self.write(f":SENS:CURR:RANG {sense_range_a:.12g}")
        self.write(":ARM:COUN 1")
        self.write(f":TRIG:COUN {len(points)}")
        self.write(":SOUR:VOLT:MODE LIST")
        self.write(":SOUR:LIST:VOLT " + ",".join(f"{point:.12g}" for point in points))
        self.write(":SOUR:SWE:RANG BEST")
        self.check_errors()
        estimated_ms = int((len(points) * (nplc / 50 + delay_s + 0.02) + 10) * 1000)
        try:
            self.write(":OUTP ON")
            raw = self.query(":READ?", timeout_ms=max(15000, estimated_ms))
            rows = parse_sweep(raw, len(points))
            self.check_errors()
            return rows
        finally:
            self.safe_off()


class VisaBus:
    def __init__(self, resource_manager: Any | None = None):
        self.rm = resource_manager or pyvisa.ResourceManager()
        self.active: dict[str, Instrument] = {}
        self.identities: dict[str, dict] = {}
        self.discovery_errors: list[dict] = []
        self.lock = threading.RLock()

    def discover(self) -> list[dict]:
        with self.lock:
            try:
                resources = [str(item) for item in self.rm.list_resources() if str(item).upper().startswith("GPIB")]
            except Exception as exc:
                raise ServiceError("visa_discovery_failed", str(exc), 502) from exc
            found = []
            self.discovery_errors = []
            for resource in resources:
                if resource in self.active:
                    found.append(dict(self.active[resource].info, reserved=True))
                    continue
                handle = None
                try:
                    handle = self.rm.open_resource(resource)
                    handle.timeout = 1500
                    handle.read_termination = "\n"
                    handle.write_termination = "\n"
                    info = parse_idn(resource, str(handle.query("*IDN?")).strip())
                    self.identities[resource] = info
                    found.append(dict(info, reserved=False))
                except ServiceError as exc:
                    self.discovery_errors.append({"resource": resource, "error": exc.payload()})
                    continue
                except Exception as exc:
                    self.discovery_errors.append({"resource": resource, "error": visa_error(exc, resource).payload()})
                    continue
                finally:
                    if handle is not None:
                        try:
                            handle.close()
                        except Exception:
                            pass
            return found

    def open(self, resource: str) -> Instrument:
        with self.lock:
            if resource in self.active:
                raise ServiceError("device_busy", "Device is already reserved", 409, resource=resource)
            if resource not in self.identities:
                self.discover()
            if resource not in self.identities:
                raise ServiceError("device_not_found", "Supported device was not found", 404, resource=resource)
            handle = None
            try:
                handle = self.rm.open_resource(resource)
                handle.timeout = 5000
                handle.read_termination = "\n"
                handle.write_termination = "\n"
                info = parse_idn(resource, str(handle.query("*IDN?")).strip())
                if info["model"] != self.identities[resource]["model"] or info["serial"] != self.identities[resource]["serial"]:
                    raise ServiceError("identity_changed", "Resource identity changed since discovery", 409, resource=resource)
                instrument = Instrument(info, handle)
                self.active[resource] = instrument
                return instrument
            except ServiceError:
                if handle is not None:
                    handle.close()
                raise
            except Exception as exc:
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass
                raise visa_error(exc, resource) from exc

    def recover_identity(self, resource: str) -> dict:
        """An explicit operator request may clear a stuck GPIB interface before identification."""
        with self.lock:
            if resource in self.active:
                raise ServiceError("device_busy", "Device is already reserved", 409, resource=resource)
            if not resource.upper().startswith("GPIB") or resource not in self.rm.list_resources():
                raise ServiceError("device_not_found", "GPIB resource was not found", 404, resource=resource)
            handle = None
            try:
                handle = self.rm.open_resource(resource)
                handle.timeout = 3000
                handle.read_termination = "\n"
                handle.write_termination = "\n"
                handle.clear()
                info = parse_idn(resource, str(handle.query("*IDN?")).strip())
                self.identities[resource] = info
                self.discovery_errors = [item for item in self.discovery_errors if item["resource"] != resource]
                return info
            except ServiceError:
                raise
            except Exception as exc:
                raise visa_error(exc, resource) from exc
            finally:
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass

    def close(self, resource: str) -> None:
        with self.lock:
            instrument = self.active.pop(resource, None)
        if instrument is not None:
            instrument.handle.close()

    def startup_shutdown(self) -> list[dict]:
        results = []
        for info in self.discover():
            if info["model"] not in {"2400", "6430"}:
                continue
            resource = info["resource"]
            try:
                device = self.open(resource)
                with device.lock:
                    device.safe_off()
                results.append({"resource": resource, "status": "off"})
            except ServiceError as exc:
                results.append({"resource": resource, "status": "error", "error": exc.payload()})
            finally:
                self.close(resource)
        return results

    def close_all(self) -> None:
        with self.lock:
            resources = list(self.active)
        for resource in resources:
            device = self.active.get(resource)
            if device is not None and device.model in {"2400", "6430"}:
                with device.lock:
                    try:
                        device.safe_off()
                    except ServiceError:
                        pass
            self.close(resource)
        self.rm.close()
