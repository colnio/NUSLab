"""SR830 ASCII driver. Commands/limits: SRS SR830 manual, revision 2.5, chapter 5.

The caller holds Instrument.lock across operations. Sine Out cannot be disabled;
safe_off implements output minimization, not electrical isolation or zero AC.
"""
from __future__ import annotations

import math
import re
from typing import Any

from .errors import ServiceError
from .instruments import Instrument, finite_number, utc_now


ENUMS = {
    "reference_source": ("FMOD", {"external": 0, "internal": 1}),
    "external_trigger": ("RSLP", {"sine": 0, "ttl_rising": 1, "ttl_falling": 2}),
    "input_mode": ("ISRC", {"A": 0, "A-B": 1, "current_1e6": 2, "current_1e8": 3}),
    "grounding": ("IGND", {"float": 0, "ground": 1}),
    "coupling": ("ICPL", {"AC": 0, "DC": 1}),
    "notch_filter": ("ILIN", {"off": 0, "line": 1, "twice_line": 2, "both": 3}),
    "reserve": ("RMOD", {"high": 0, "normal": 1, "low_noise": 2}),
    "filter_slope_db_oct": ("OFSL", {6: 0, 12: 1, 18: 2, 24: 3}),
    "synchronous_filter": ("SYNC", {False: 0, True: 1}),
}
VOLTAGE_SENSITIVITIES = tuple(
    value * 10.0 ** exponent for exponent in range(-9, 0) for value in (1, 2, 5)
)[1:] + (1.0,)
CURRENT_SENSITIVITIES = tuple(value / 1e6 for value in VOLTAGE_SENSITIVITIES)
TIME_CONSTANTS = tuple(value * 10.0 ** exponent for exponent in range(-5, 5) for value in (1, 3))
NUMBERS = {
    "frequency_hz": ("FREQ", 0.001, 102000),
    "phase_deg": ("PHAS", -360, 729.99),
    "harmonic": ("HARM", 1, 19999),
    "sine_amplitude_v_rms": ("SLVL", 0.004, 5.0),
}
LIA_FLAGS = {
    0: "input_overload", 1: "filter_overload", 2: "output_overload",
    3: "reference_unlocked", 4: "frequency_range_changed",
    5: "time_constant_changed", 6: "data_triggered",
}
ESR_ERRORS = {0: "input_queue_overflow", 2: "output_queue_overflow", 4: "execution_error", 5: "command_error"}
HARDWARE_ERRORS = {1: "backup_error", 2: "ram_error", 4: "rom_error", 5: "gpib_error", 6: "dsp_error", 7: "math_error"}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ServiceError("invalid_value", f"{name} must be a number, not a boolean", 422)
    return finite_number(value, name)


def _index(value: Any, choices: tuple, name: str) -> int:
    number = _number(value, name)
    for index, choice in enumerate(choices):
        if math.isclose(number, choice, rel_tol=1e-9, abs_tol=0):
            return index
    raise ServiceError("invalid_value", f"{name} must be a supported discrete value", 422,
                       allowed=list(choices))


class SR830(Instrument):
    @property
    def has_source_outputs(self) -> bool:
        return True

    def _read_number(self, command: str) -> float:
        raw = self.query(command)
        try:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError()
            return value
        except ValueError as exc:
            raise ServiceError("invalid_readback", "SR830 returned a non-finite or nonnumeric value", 502,
                               resource=self.resource, command=command, raw=raw[:200]) from exc

    def _read_int(self, command: str, lower: int, upper: int) -> int:
        value = self._read_number(command)
        if not value.is_integer() or not lower <= value <= upper:
            raise ServiceError("invalid_readback", "SR830 returned an invalid setting or status byte", 502,
                               resource=self.resource, command=command, value=value)
        return int(value)

    def _enum(self, name: str):
        command, values = ENUMS[name]
        code = self._read_int(command + "?", 0, max(values.values()))
        return next(key for key, value in values.items() if value == code)

    def _diagnostics(self, overrides: dict | None = None) -> dict:
        # ESR/ERRS are latched and clear on read. Never poll them twice for the
        # same operation; raw status queries supply their already-read values.
        overrides = overrides or {}
        result = {name: overrides[name] if name in overrides else self._read_int(command, 0, 255)
                  for name, command in (("standard_event_status", "*ESR?"), ("error_status", "ERRS?"))}
        errors = []
        for name, bits in (("standard_event_status", ESR_ERRORS), ("error_status", HARDWARE_ERRORS)):
            byte = result[name]
            if byte is not None:
                errors.extend(label for bit, label in bits.items() if byte & (1 << bit))
        result["instrument_errors"] = errors
        return result

    def _check_diagnostics(self, diagnostics: dict) -> None:
        if diagnostics["instrument_errors"]:
            raise ServiceError("instrument_error", "SR830 reported command or hardware errors", 502,
                               resource=self.resource, diagnostics=diagnostics)

    def errors(self) -> list[str]:
        return self._diagnostics()["instrument_errors"]

    def check_errors(self) -> None:
        self._check_diagnostics(self._diagnostics())

    def _status(self) -> dict:
        byte = self._read_int("LIAS?", 0, 255)
        flags = {label: bool(byte & (1 << bit)) for bit, label in LIA_FLAGS.items()}
        return {"status_word": byte, "status_flags": flags,
                "status_scope": "latched_since_previous_read", "overload": bool(byte & 7),
                "reference_unlocked": flags["reference_unlocked"]}

    def _configuration(self) -> dict:
        result = {name: self._enum(name) for name in ENUMS}
        result.update({name: self._read_number(command + "?") for name, (command, _, _) in NUMBERS.items()})
        if not result["harmonic"].is_integer() or not 1 <= result["harmonic"] <= 19999:
            raise ServiceError("invalid_readback", "SR830 returned an invalid harmonic", 502, resource=self.resource)
        result["harmonic"] = int(result["harmonic"])
        current = result["input_mode"].startswith("current_")
        result["unit"] = "A" if current else "V"
        sens_index = self._read_int("SENS?", 0, 26)
        result["sensitivity"] = (CURRENT_SENSITIVITIES if current else VOLTAGE_SENSITIVITIES)[sens_index]
        result["sensitivity_index"] = sens_index
        result["time_constant_s"] = TIME_CONSTANTS[self._read_int("OFLT?", 0, 19)]
        result["aux_outputs_v"] = {str(channel): self._read_number(f"AUXV? {channel}") for channel in range(1, 5)}
        result["sine_output_active"] = True
        result["supports_output_off"] = False
        result["output_state"] = ("outputs_minimized" if self._are_minimized(result) else "active")
        return result

    @staticmethod
    def _are_minimized(settings: dict) -> bool:
        return (math.isclose(settings["sine_amplitude_v_rms"], .004, rel_tol=0, abs_tol=1e-9)
                and all(abs(value) <= 1e-9 for value in settings["aux_outputs_v"].values()))

    def configuration(self) -> dict:
        result = self._configuration()
        result.update(self._status())
        result["diagnostics"] = self._diagnostics()
        self._check_diagnostics(result["diagnostics"])
        return result

    def configure(self, settings: dict) -> dict:
        allowed = set(ENUMS) | set(NUMBERS) | {"sensitivity", "time_constant_s", "aux_outputs_v"}
        unexpected = set(settings) - allowed
        if unexpected:
            raise ServiceError("unsupported_operation", "Setting is unavailable for SR830", 422,
                               settings=sorted(unexpected), model=self.model)
        requested = dict(settings)
        for name, (_, choices) in ENUMS.items():
            if name in requested:
                value = requested[name]
                if (type(value) not in {type(key) for key in choices} or value not in choices):
                    raise ServiceError("invalid_value", f"Unsupported {name}", 422, allowed=list(choices))
        for name, (_, lower, upper) in NUMBERS.items():
            if name in requested:
                value = _number(requested[name], name)
                if not lower <= value <= upper or (name == "harmonic" and not value.is_integer()):
                    raise ServiceError("invalid_value", f"{name} is outside its supported range", 422,
                                       minimum=lower, maximum=upper)
                requested[name] = int(value) if name == "harmonic" else value
        tc_index = _index(requested["time_constant_s"], TIME_CONSTANTS, "time_constant_s") if "time_constant_s" in requested else None
        if tc_index is not None:
            requested["time_constant_s"] = TIME_CONSTANTS[tc_index]
        if "aux_outputs_v" in requested:
            aux = requested["aux_outputs_v"]
            if not isinstance(aux, dict) or any(type(ch) not in (str, int) or str(ch) not in {"1", "2", "3", "4"} for ch in aux):
                raise ServiceError("invalid_value", "aux_outputs_v must map channels 1 through 4 to voltages", 422)
            normalized = {}
            for channel, raw in aux.items():
                value = _number(raw, "aux voltage")
                if not -10.5 <= value <= 10.5:
                    raise ServiceError("invalid_value", "Auxiliary voltage must be between -10.5 and 10.5 V", 422)
                normalized[str(channel)] = value
            requested["aux_outputs_v"] = normalized

        before = self._configuration()
        target = dict(before, **requested)
        current = target["input_mode"].startswith("current_")
        choices = CURRENT_SENSITIVITIES if current else VOLTAGE_SENSITIVITIES
        sens_index = (_index(requested["sensitivity"], choices, "sensitivity")
                      if "sensitivity" in requested else before["sensitivity_index"])
        if "sensitivity" in requested:
            requested["sensitivity"] = choices[sens_index]
        if target["input_mode"] == "current_1e8" and sens_index > 20:
            raise ServiceError("invalid_value", "current_1e8 requires sensitivity <= 10 nA; include sensitivity in this request", 422)
        if "frequency_hz" in requested and target["reference_source"] != "internal":
            raise ServiceError("invalid_value", "Frequency can only be set with an internal reference", 422)
        switching_internal = before["reference_source"] == "external" and target["reference_source"] == "internal"
        if switching_internal and "frequency_hz" not in requested:
            raise ServiceError("invalid_value", "Include frequency_hz when switching to internal reference; FREQ? currently reports the external frequency", 422)
        detection_hz = target["frequency_hz"] * target["harmonic"]
        if detection_hz > 102000:
            raise ServiceError("invalid_value", "Frequency times harmonic must not exceed 102000 Hz", 422)
        if target["reference_source"] == "external" and target["frequency_hz"] < 1 and target["external_trigger"] == "sine":
            raise ServiceError("invalid_value", "External reference below 1 Hz requires a TTL trigger", 422)
        if tc_index is not None and target["time_constant_s"] > 30 and detection_hz > 200:
            raise ServiceError("invalid_value", "Time constants above 30 s require detection frequency <= 200 Hz", 422)
        # SYNC selects filtering when below 200 Hz; it may remain selected above
        # that frequency. Do not confuse the selected mode with an active filter.
        prior_diagnostics = self._diagnostics()
        self._check_diagnostics(prior_diagnostics)

        # Everything above validates before the first write. Lower HARM before
        # increasing FREQ so the instrument cannot silently clamp the harmonic.
        if "external_trigger" in requested:
            self.write(f"RSLP {ENUMS['external_trigger'][1][target['external_trigger']]}")
        if switching_internal:
            # The stored internal frequency is not queryable in external mode.
            # Avoid harmonic clipping while switching back to that oscillator.
            self.write("HARM 1")
        if "reference_source" in requested:
            self.write(f"FMOD {ENUMS['reference_source'][1][target['reference_source']]}")
        if "frequency_hz" in requested or "harmonic" in requested:
            if not switching_internal and target["harmonic"] < before["harmonic"]:
                self.write(f"HARM {target['harmonic']}")
            if "frequency_hz" in requested:
                self.write(f"FREQ {target['frequency_hz']:.12g}")
            if switching_internal or (target["harmonic"] >= before["harmonic"] and "harmonic" in requested):
                self.write(f"HARM {target['harmonic']}")
        # Selecting high current gain with a wide sensitivity can revert the
        # input gain. Narrow sensitivity first, then select the requested mode.
        early_sens = "input_mode" in requested and target["input_mode"] == "current_1e8"
        if early_sens:
            self.write(f"SENS {sens_index}")
        for name in ("input_mode", "grounding", "coupling", "notch_filter"):
            if name in requested:
                command, values = ENUMS[name]
                self.write(f"{command} {values[requested[name]]}")
        if "sensitivity" in requested and not early_sens:
            self.write(f"SENS {sens_index}")
        for name in ("reserve", "filter_slope_db_oct", "synchronous_filter"):
            if name in requested:
                command, values = ENUMS[name]
                self.write(f"{command} {values[requested[name]]}")
        if tc_index is not None:
            self.write(f"OFLT {tc_index}")
        for name in ("phase_deg", "sine_amplitude_v_rms"):
            if name in requested:
                self.write(f"{NUMBERS[name][0]} {requested[name]:.12g}")
        for channel, value in requested.get("aux_outputs_v", {}).items():
            self.write(f"AUXV {channel},{value:.12g}")

        result = self.configuration()
        expected = {name: requested.get(name, before[name]) for name in allowed}
        expected["sensitivity"] = choices[sens_index]
        expected["aux_outputs_v"] = dict(before["aux_outputs_v"], **requested.get("aux_outputs_v", {}))
        result["adjustments"] = {name: {"requested_or_previous": value, "actual": result[name]}
                                 for name, value in expected.items() if result[name] != value}
        result["diagnostics_before"] = prior_diagnostics
        return result

    def measure(self, function: str | None = None) -> dict:
        if function is not None:
            raise ServiceError("unsupported_operation", "SR830 reads X, Y, magnitude and phase; function is not supported", 422)
        mode = self._enum("input_mode")
        choices = CURRENT_SENSITIVITIES if mode.startswith("current_") else VOLTAGE_SENSITIVITIES
        sensitivity = choices[self._read_int("SENS?", 0, 26)]
        timestamp = utc_now()
        raw = self.query("SNAP? 1,2,3,4,9")
        try:
            values = [float(part) for part in raw.split(",")]
            if len(values) != 5 or not all(math.isfinite(value) for value in values):
                raise ValueError()
        except ValueError as exc:
            raise ServiceError("invalid_readback", "SR830 snapshot must contain five finite numbers", 502,
                               resource=self.resource, raw=raw[:200]) from exc
        result = dict(zip(("x", "y", "magnitude", "phase_deg", "frequency_hz"), values))
        result.update(timestamp=timestamp, input_mode=mode, unit="A" if mode.startswith("current_") else "V",
                      function="lockin", settling_waited=False,
                      snapshot_timing="X/Y coherent; magnitude/phase pair sampled approximately 10 us apart from X/Y")
        # A live 1 Mohm check found native R saturating near 1.09 full scale
        # without a LIAS overload flag. Preserve native flags, but expose this
        # independent quality check so clients can widen sensitivity and reread.
        result["sensitivity"] = sensitivity
        result["range_exceeded"] = max(abs(result[key]) for key in ("x", "y", "magnitude")) > sensitivity
        result.update(self._status())
        result["diagnostics"] = self._diagnostics()
        self._check_diagnostics(result["diagnostics"])
        return result

    def safe_off(self) -> dict:
        failures = []
        for command in ("SLVL 0.004", *(f"AUXV {channel},0" for channel in range(1, 5))):
            try:
                self.write(command)
            except ServiceError as exc:
                failures.append({"command": command, "error": exc.payload()})
        values = {}
        for command, expected in (("SLVL?", .004), *((f"AUXV? {channel}", 0) for channel in range(1, 5))):
            try:
                value = self._read_number(command)
                values[command] = value
                if not math.isclose(value, expected, rel_tol=0, abs_tol=1e-9):
                    failures.append({"command": command, "expected": expected, "actual": value})
            except ServiceError as exc:
                failures.append({"command": command, "error": exc.payload()})
        diagnostics = None
        try:
            diagnostics = self._diagnostics()
            self._check_diagnostics(diagnostics)
        except ServiceError as exc:
            failures.append({"error": exc.payload()})
        if failures:
            raise ServiceError("shutdown_failed", "SR830 output minimization could not be verified", 502,
                               resource=self.resource, failures=failures, readbacks=values, diagnostics=diagnostics)
        return {"status": "outputs_minimized", "sine_amplitude_v_rms": values["SLVL?"],
                "aux_outputs_v": {str(ch): values[f"AUXV? {ch}"] for ch in range(1, 5)},
                "sine_output_active": True, "supports_output_off": False, "diagnostics": diagnostics}

    def minimize_outputs(self) -> dict:
        return self.safe_off()

    def raw_scpi(self, kind: str, command: str) -> dict:
        if not isinstance(command, str) or not command.strip() or len(command) > 65536:
            raise ServiceError("invalid_command", "Command must contain 1 to 65536 characters", 422)
        if kind == "query":
            response = self.query(command)
        elif kind == "write":
            self.write(command)
            response = None
        else:
            raise ServiceError("invalid_command", "kind must be write or query", 422)
        overrides = {}
        for register, key in (("*ESR", "standard_event_status"), ("ERRS", "error_status")):
            if re.search(re.escape(register) + r"\s*\?", command, re.I):
                overrides[key] = None  # Compound replies stay verbatim; do not drain the register again.
                match = re.fullmatch(re.escape(register) + r"\s*\?\s*([0-7])?\s*", command.strip(), re.I)
                if match and response is not None:
                    try:
                        number = int(response)
                        if not 0 <= number <= (1 if match[1] is not None else 255):
                            raise ValueError()
                        overrides[key] = number << int(match[1]) if match[1] is not None else number
                    except ValueError as exc:
                        raise ServiceError("invalid_readback", "Invalid SR830 status reply", 502, raw=response[:200]) from exc
        diagnostics = self._diagnostics(overrides)
        return {"response": response, "instrument_errors": diagnostics["instrument_errors"], "diagnostics": diagnostics}
