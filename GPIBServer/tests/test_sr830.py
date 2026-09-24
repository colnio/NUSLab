from __future__ import annotations

import csv
import json
import math
import time

import pytest
from fastapi.testclient import TestClient

from GPIBServer.app import create_app
from GPIBServer.errors import ServiceError
from GPIBServer.instruments import VisaBus, parse_idn
from GPIBServer.service import LabService
from GPIBServer.sr830 import SR830
from GPIBServer.storage import RunStore
from GPIBServer.tests.test_server import FakeResourceManager


RESOURCE = "GPIB0::8::INSTR"
IDENTITY = "Stanford_Research_Systems,SR830,s/n40423,ver1.07"


class LockinHandle:
    """Independent instrument simulator, including coupled settings and latches."""
    def __init__(self):
        self.timeout = 5000
        self.settings = {"FMOD": 1, "FREQ": 1000., "PHAS": 0., "HARM": 1,
                         "SLVL": .2, "RSLP": 0, "ISRC": 0, "IGND": 0,
                         "ICPL": 0, "ILIN": 0, "SENS": 26, "RMOD": 2,
                         "OFLT": 8, "OFSL": 1, "SYNC": 0}
        self.aux = {1: .5, 2: -.1, 3: 0., 4: .2}
        self.registers = {"*ESR": 128, "ERRS": 0, "LIAS": 0}
        self.commands = []
        self.fail_writes = set()
        self.fail_queries = set()
        self.ignore_writes = set()
        self.replies = {}
        self.closed = False

    @property
    def writes(self):
        return [c for c in self.commands if "?" not in c and c != "VISA_CLEAR"]

    def write(self, command):
        self.commands.append(command)
        if command in self.fail_writes:
            raise TimeoutError("VISA timeout")
        if command in self.ignore_writes:
            return
        root, _, tail = command.partition(" ")
        if root == "AUXV":
            channel, value = tail.split(",")
            self.aux[int(channel)] = round(float(value), 3)
        elif root in self.settings:
            value = float(tail)
            if root in {"FMOD", "HARM", "RSLP", "ISRC", "IGND", "ICPL", "ILIN", "SENS", "RMOD", "OFLT", "OFSL", "SYNC"}:
                value = int(value)
            if root == "FREQ" and (not self.settings["FMOD"] or value * self.settings["HARM"] > 102000):
                self.registers["*ESR"] |= 16
                return
            if root == "ISRC" and value == 3 and self.settings["SENS"] > 20:
                self.registers["*ESR"] |= 16
                return
            if root == "OFLT" and value > 13 and self.settings["FREQ"] * self.settings["HARM"] > 200:
                self.registers["*ESR"] |= 16
                return
            if root == "SLVL":
                value = round(value / .002) * .002
            if root == "PHAS":
                value = (round(value, 2) + 180) % 360 - 180
            if root == "FREQ":
                value = round(value, 4)
            self.settings[root] = value
            if root == "SENS" and value > 20 and self.settings["ISRC"] == 3:
                self.settings["ISRC"] = 2
            if root in {"FREQ", "HARM"}:
                freq = self.settings["FREQ"]
                self.settings["HARM"] = min(self.settings["HARM"], int(102000 / freq))
                if freq * self.settings["HARM"] > 200 and self.settings["OFLT"] > 13:
                    self.settings["OFLT"] = 13
                    self.registers["LIAS"] |= 32
            # Hardware can raise too-short time constants in high reserve.
            if root == "OFLT" and self.settings["RMOD"] == 0 and value == 0:
                self.settings["OFLT"] = 4
                self.registers["LIAS"] |= 32
        else:
            self.registers["*ESR"] |= 32

    def query(self, command):
        self.commands.append(command)
        if command in self.fail_queries:
            raise TimeoutError("VISA timeout")
        if command in self.replies:
            return self.replies[command]
        if command == "*IDN?":
            return IDENTITY
        if command.startswith("AUXV? "):
            return str(self.aux[int(command.split()[1])])
        if command == "SNAP? 1,2,3,4,9":
            x, y = ((3e-12, 4e-12) if self.settings["ISRC"] >= 2 else (.003, .004))
            return f"{x},{y},{math.hypot(x, y)},53.130102,{self.settings['FREQ']}"
        root, _, argument = command.partition("?")
        if root in self.registers:
            byte = self.registers[root]
            if argument.strip():
                mask = 1 << int(argument)
                self.registers[root] &= ~mask
                return str(int(bool(byte & mask)))
            self.registers[root] = 0
            return str(byte)
        if root in self.settings:
            return str(self.settings[root])
        raise AssertionError(f"Unsupported query sent to SR830: {command}")

    def clear(self):
        self.commands.append("VISA_CLEAR")

    def close(self):
        self.closed = True


class MixedManager(FakeResourceManager):
    def __init__(self, lockin=None):
        super().__init__()
        self.lockin = lockin or LockinHandle()

    def list_resources(self):
        return (RESOURCE, *super().list_resources())

    def open_resource(self, resource):
        if resource == RESOURCE:
            self.lockin.closed = False
            return self.lockin
        return super().open_resource(resource)


@pytest.fixture
def driver():
    return SR830(parse_idn(RESOURCE, IDENTITY), LockinHandle())


@pytest.fixture
def server(tmp_path):
    manager = MixedManager()
    store = RunStore(tmp_path / "recovery", tmp_path / "dropbox")
    lab = LabService(VisaBus(manager), store)
    with TestClient(create_app(lab)) as client:
        yield client, lab, manager, store


def session(client, all_devices=False):
    devices = {"lia": RESOURCE}
    if all_devices:
        devices.update(sd="GPIB0::12::INSTR", gate="GPIB0::15::INSTR", probe="GPIB0::16::INSTR")
    response = client.post("/v1/sessions", json={"sample_name": "lockin test", "devices": devices})
    assert response.status_code == 201, response.text
    sid = response.json()["session_id"]
    return sid, f"/v1/sessions/{sid}/devices/lia"


def assert_minimized(handle):
    assert handle.settings["SLVL"] == .004
    assert handle.aux == {1: 0, 2: 0, 3: 0, 4: 0}


def test_identity_discovery_is_only_identification_and_factory_selects_driver():
    manager = MixedManager()
    bus = VisaBus(manager)
    found = bus.discover()
    info = next(item for item in found if item["model"] == "SR830")
    assert info["serial"] == "40423"
    assert info["firmware"] == "1.07"
    assert manager.lockin.commands == ["*IDN?"]
    device = bus.open(RESOURCE)
    assert isinstance(device, SR830)
    bus.close(RESOURCE)
    for raw in ("Other,SR830,1,2", "Stanford_Research_Systems,SR860,1,2"):
        with pytest.raises(ServiceError, match="supported instrument"):
            parse_idn(RESOURCE, raw)


def test_configure_complete_readback_and_rounding(driver):
    result = driver.configure({"reference_source": "internal", "frequency_hz": 17.123456,
                               "phase_deg": 541., "harmonic": 2, "external_trigger": "ttl_rising",
                               "sine_amplitude_v_rms": .0112, "input_mode": "A-B", "coupling": "DC",
                               "grounding": "ground", "notch_filter": "both", "sensitivity": 1e-3,
                               "reserve": "normal", "time_constant_s": 3., "filter_slope_db_oct": 24,
                               "synchronous_filter": True, "aux_outputs_v": {"1": .5004, "4": -.7504}})
    assert result["frequency_hz"] == 17.1235
    assert result["phase_deg"] == -179
    assert result["sine_amplitude_v_rms"] == .012
    assert result["aux_outputs_v"] == {"1": .5, "2": -.1, "3": 0, "4": -.75}
    assert set(result["adjustments"]) == {"frequency_hz", "phase_deg", "sine_amplitude_v_rms", "aux_outputs_v"}
    assert result["diagnostics_before"]["standard_event_status"] == 128  # Power-on is informational.
    assert result["unit"] == "V" and result["input_mode"] == "A-B"
    assert result["sine_output_active"] and not result["supports_output_off"]
    assert not any(c.startswith(":") for c in driver.handle.commands)


@pytest.mark.parametrize("mode", ["current_1e6", "current_1e8"])
def test_current_units_and_no_double_gain_conversion(driver, mode):
    result = driver.configure({"input_mode": mode, "sensitivity": 5e-12})
    assert result["sensitivity"] == pytest.approx(5e-12)
    if mode == "current_1e8":
        assert driver.handle.writes.index("SENS 10") < driver.handle.writes.index("ISRC 3")
    reading = driver.measure()
    assert reading["unit"] == "A"
    assert reading["x"] == 3e-12 and reading["y"] == 4e-12
    assert reading["magnitude"] == pytest.approx(5e-12)
    assert reading["settling_waited"] is False
    assert "SNAP? 1,2,3,4,9" in driver.handle.commands


@pytest.mark.parametrize("settings", [
    {"sine_amplitude_v_rms": 0}, {"sine_amplitude_v_rms": 5.01},
    {"frequency_hz": float("nan")}, {"frequency_hz": 102001}, {"phase_deg": float("inf")},
    {"harmonic": 1.5}, {"harmonic": True}, {"sensitivity": 3e-9},
    {"time_constant_s": .02}, {"filter_slope_db_oct": 9}, {"synchronous_filter": "yes"},
    {"aux_outputs_v": {"5": 0}}, {"aux_outputs_v": {"1": 11}}, {"aux_outputs_v": {"1": True}},
    {"input_mode": "current_1e8"}, {"input_mode": "current_1e8", "sensitivity": 20e-9},
    {"reference_source": "external", "frequency_hz": 17}, {"harmonic": 200},
    {"time_constant_s": 100}, {"source_mode": "voltage"},
    {"phase_deg": 10, "aux_outputs_v": {"1": .5, "2": float("nan")}},
])
def test_invalid_configuration_never_writes(driver, settings):
    with pytest.raises(ServiceError):
        driver.configure(settings)
    assert driver.handle.writes == []


def test_dependent_frequency_harmonic_order_and_automatic_tc_change(driver):
    driver.handle.settings.update(FREQ=10., HARM=100, OFLT=8)
    result = driver.configure({"frequency_hz": 5000., "harmonic": 2})
    assert driver.handle.writes.index("HARM 2") < driver.handle.writes.index("FREQ 5000")
    assert result["harmonic"] == 2
    result = driver.configure({"frequency_hz": 1., "harmonic": 1, "time_constant_s": 100.})
    assert result["time_constant_s"] == 100
    result = driver.configure({"frequency_hz": 1000.})
    assert result["time_constant_s"] == 30
    assert result["adjustments"]["time_constant_s"]["actual"] == 30
    assert result["status_flags"]["time_constant_changed"]
    result = driver.configure({"reserve": "high", "time_constant_s": 1e-5})
    assert result["adjustments"]["time_constant_s"]["actual"] == .001


def test_external_reference_cannot_set_frequency_and_sub_hz_requires_ttl(driver):
    driver.handle.settings.update(FMOD=0, FREQ=.5)
    for payload in ({"frequency_hz": 10}, {"phase_deg": 1}):
        with pytest.raises(ServiceError):
            driver.configure(payload)
    assert not driver.handle.writes
    assert driver.configure({"external_trigger": "ttl_rising"})["external_trigger"] == "ttl_rising"


def test_switching_to_internal_requires_frequency_and_preserves_harmonic(driver):
    driver.handle.settings.update(FMOD=0, FREQ=20, HARM=10)
    with pytest.raises(ServiceError):
        driver.configure({"reference_source": "internal"})
    assert not driver.handle.writes
    result = driver.configure({"reference_source": "internal", "frequency_hz": 100})
    assert result["harmonic"] == 10
    assert driver.handle.writes[:4] == ["HARM 1", "FMOD 1", "FREQ 100", "HARM 10"]


@pytest.mark.parametrize("command,reply", [("SNAP? 1,2,3,4,9", "1,2,3,4"),
                                             ("SNAP? 1,2,3,4,9", "1,2,nan,4,5"),
                                             ("SNAP? 1,2,3,4,9", "garbage"),
                                             ("ISRC?", "2.5"), ("LIAS?", "256")])
def test_malformed_readings_fail_with_structured_error(driver, command, reply):
    driver.handle.replies[command] = reply
    with pytest.raises(ServiceError) as error:
        driver.measure()
    assert error.value.code == "invalid_readback"


def test_latched_measurement_flags_are_not_command_errors(driver):
    driver.handle.registers["LIAS"] = 15
    result = driver.measure()
    assert result["overload"] and result["reference_unlocked"]
    assert result["status_scope"] == "latched_since_previous_read"
    assert result["diagnostics"]["instrument_errors"] == []
    assert not driver.measure()["overload"]
    assert driver.handle.commands.count("LIAS?") == 2


@pytest.mark.parametrize("register,byte,label", [("*ESR", 32, "command_error"), ("ERRS", 64, "dsp_error")])
def test_command_and_hardware_errors_preserve_registers_in_error(driver, register, byte, label):
    driver.handle.registers[register] = byte
    with pytest.raises(ServiceError) as error:
        driver.measure()
    assert error.value.code == "instrument_error"
    assert label in error.value.details["diagnostics"]["instrument_errors"]


@pytest.mark.parametrize("command,register,byte,response", [("*ESR?", "*ESR", 32, "32"),
                                                            ("*ESR? 5", "*ESR", 32, "1"),
                                                            ("ERRS?", "ERRS", 64, "64"),
                                                            ("LIAS?", "LIAS", 8, "8")])
def test_raw_status_queries_are_not_read_twice(driver, command, register, byte, response):
    driver.handle.registers[register] = byte
    result = driver.raw_scpi("query", command)
    assert result["response"] == response
    assert [c for c in driver.handle.commands if c.startswith(register + "?")] == [command]
    if register != "LIAS":
        assert result["instrument_errors"]


def test_raw_bad_command_reports_sr830_errors(driver):
    response = driver.raw_scpi("write", "NOT_A_COMMAND")
    assert response["instrument_errors"] == ["command_error"]
    assert ":SYST:ERR?" not in driver.handle.commands


@pytest.mark.parametrize("failure", ["write", "read", "mismatch"])
def test_cleanup_attempts_all_outputs_and_all_readbacks(driver, failure):
    if failure == "write":
        driver.handle.fail_writes.add("SLVL 0.004")
    elif failure == "read":
        driver.handle.fail_queries.add("SLVL?")
    else:
        driver.handle.ignore_writes.add("SLVL 0.004")
    with pytest.raises(ServiceError) as error:
        driver.safe_off()
    assert error.value.code == "shutdown_failed"
    for ch in range(1, 5):
        assert f"AUXV {ch},0" in driver.handle.writes
        assert f"AUXV? {ch}" in driver.handle.commands
    assert driver.handle.aux == {1: 0, 2: 0, 3: 0, 4: 0}


def test_startup_four_device_session_recording_and_release(server):
    client, lab, manager, store = server
    health = client.get("/v1/health").json()
    assert health["status"] == "ready"
    cleanup = next(r for r in health["startup_shutdown"] if r["resource"] == RESOURCE)
    assert cleanup["status"] == "outputs_minimized" and cleanup["sine_output_active"]
    assert_minimized(manager.lockin)
    manager.lockin.commands.clear()
    devices = client.get("/v1/devices").json()["devices"]
    assert {d["model"] for d in devices} == {"SR830", "2400", "6430", "2002"}
    assert manager.lockin.writes == []
    sid, base = session(client, all_devices=True)
    reserved = client.post("/v1/sessions", json={"sample_name": "other", "devices": {"lia": RESOURCE}})
    assert reserved.status_code == 409
    assert client.post(f"/v1/sessions/{sid}/heartbeat").status_code == 200
    manager.lockin.registers["LIAS"] = 8
    read = client.post(base + "/read", json={}).json()
    assert read["reference_unlocked"]
    response = client.post(f"/v1/sessions/{sid}/samples", json={"devices": ["lia", "sd", "gate", "probe"]})
    assert response.status_code == 200, response.text
    assert len(response.json()["readings"]) == 4
    close = client.delete(f"/v1/sessions/{sid}")
    assert close.status_code == 200
    assert close.json()["output_cleanup"][RESOURCE]["status"] == "outputs_minimized"
    meta = store.load(sid)
    assert meta["schema_version"] == 3 and meta["measurement_count"] == 5
    with (store.recovery_root / meta["files"]["data"]).open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["x"] == "0.003" and rows[0]["phase_deg"] == "53.130102"
    assert rows[0]["reference_unlocked"] == "True"
    assert rows[0]["sensitivity"] == "1.0" and rows[0]["range_exceeded"] == "False"
    assert rows[1]["sensitivity"] == "1.0" and rows[1]["range_exceeded"] == "False"
    assert rows[1]["sample_id"] == rows[2]["sample_id"]
    assert rows[2]["x"] == ""
    assert (store.dropbox_root / meta["files"]["data"]).read_bytes() == (store.recovery_root / meta["files"]["data"]).read_bytes()
    events = [json.loads(line) for line in (store.recovery_root / meta["files"]["events"]).read_text().splitlines()]
    assert any(e["event"] == "outputs_initialized" and e["details"]["lia"]["status"] == "outputs_minimized" for e in events)
    assert not any(command in manager.lockin.commands for command in ("*RST", "*CLS", ":OUTP OFF", ":SYST:ERR?"))


@pytest.mark.parametrize("mode,full_scale", [(0, 1.0), (2, 1e-6), (3, 1e-8)])
@pytest.mark.parametrize("channel", [0, 1, 2])
def test_full_scale_quality_check_independent_of_native_overload(driver, mode, full_scale, channel):
    driver.handle.settings.update(ISRC=mode, SENS=20 if mode == 3 else 26)
    values = [0.0, 0.0, 0.0, 0.0, 1000.0]
    values[channel] = 1.09 * full_scale
    driver.handle.replies["SNAP? 1,2,3,4,9"] = ",".join(map(str, values))
    reading = driver.measure()
    assert reading["sensitivity"] == pytest.approx(full_scale)
    assert reading["range_exceeded"] is True
    assert reading["overload"] is False and reading["status_word"] == 0


def test_range_exceeded_is_preserved_in_individual_and_grouped_recordings(server):
    client, _, manager, store = server
    sid, base = session(client)
    manager.lockin.replies["SNAP? 1,2,3,4,9"] = "0.9,0.8,1.09,40,1000"
    assert client.post(base + "/read", json={}).json()["range_exceeded"]
    assert client.post(f"/v1/sessions/{sid}/samples", json={"devices": ["lia"]}).json()["readings"]["lia"]["range_exceeded"]
    client.delete(f"/v1/sessions/{sid}")
    meta = store.load(sid)
    with (store.recovery_root / meta["files"]["data"]).open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [r["range_exceeded"] for r in rows] == ["True", "True"]


def test_api_configure_minimize_unsupported_and_invalid_fields(server):
    client, _, manager, _ = server
    sid, base = session(client)
    settings = {"frequency_hz": 17, "input_mode": "current_1e8", "sensitivity": 1e-12,
                "sine_amplitude_v_rms": .02, "aux_outputs_v": {"1": .5}}
    response = client.post(base + "/configure", json=settings)
    assert response.status_code == 200, response.text
    assert client.get(base).json()["aux_outputs_v"]["1"] == .5
    for suffix, body in (("/output", {"enabled": False}), ("/setpoint", {"value": 0}), ("/read", {"function": "RES"})):
        assert client.post(base + suffix, json=body).status_code == 422
    assert client.post(f"/v1/sessions/{sid}/sweeps", json={"device": "lia", "voltage_points_v": [0]}).status_code == 422
    for payload in ({"sine_amplitud": .5}, {"harmonic": True}, {"frequency_hz": True}, {"aux_outputs_v": {"1": True}}):
        writes_before = list(manager.lockin.writes)
        assert client.post(base + "/configure", json=payload).status_code == 422
        assert manager.lockin.writes == writes_before
    minimized = client.post(base + "/minimize-outputs")
    assert minimized.status_code == 200
    assert minimized.json()["status"] == "outputs_minimized"
    assert_minimized(manager.lockin)


def test_raw_command_error_also_minimizes_outputs(server):
    client, _, manager, _ = server
    sid, base = session(client)
    client.post(base + "/configure", json={"sine_amplitude_v_rms": .2, "aux_outputs_v": {"3": .3}})
    response = client.post(base + "/scpi", json={"kind": "write", "command": "NOT_A_COMMAND"})
    assert response.status_code == 200
    assert response.json()["instrument_errors"] == ["command_error"]
    assert_minimized(manager.lockin)


def test_failed_cleanup_blocks_active_operations_until_release_and_recovery(server):
    client, lab, manager, _ = server
    sid, base = session(client)
    manager.lockin.settings["SLVL"] = .2
    manager.lockin.ignore_writes.add("SLVL 0.004")
    assert client.post(base + "/minimize-outputs").status_code == 502
    assert RESOURCE in lab.blocked_resources
    assert client.post(base + "/configure", json={"sine_amplitude_v_rms": .01}).status_code == 503
    assert client.post("/v1/devices/recover", json={"resource": RESOURCE}).status_code == 409
    assert client.delete(f"/v1/sessions/{sid}").status_code == 502
    manager.lockin.ignore_writes.clear()
    response = client.post("/v1/devices/recover", json={"resource": RESOURCE})
    assert response.status_code == 200
    assert response.json()["cleanup"]["status"] == "outputs_minimized"
    assert RESOURCE not in lab.blocked_resources
    assert "VISA_CLEAR" in manager.lockin.commands


@pytest.mark.parametrize("cause", ["timeout", "malformed", "command", "storage", "lease"])
def test_lifecycle_minimizes_outputs(server, monkeypatch, cause):
    client, lab, manager, _ = server
    sid, base = session(client)
    client.post(base + "/configure", json={"sine_amplitude_v_rms": .1, "aux_outputs_v": {"2": .5}})
    if cause == "timeout":
        manager.lockin.fail_queries.add("SNAP? 1,2,3,4,9")
        expected = 504
    elif cause == "malformed":
        manager.lockin.replies["SNAP? 1,2,3,4,9"] = "broken"
        expected = 502
    elif cause == "command":
        manager.lockin.registers["*ESR"] = 32
        expected = 502
    elif cause == "storage":
        def full_disk(rows):
            raise OSError("Disk full")
        monkeypatch.setattr(lab.sessions[sid].record, "append_rows", full_disk)
        expected = 507
    else:
        lab.sessions[sid].expires_at = time.monotonic() - 1
        lab._expire_session(sid)
        assert_minimized(manager.lockin)
        return
    response = client.post(base + "/read", json={})
    assert response.status_code == expected, response.text
    assert_minimized(manager.lockin)


def test_startup_failure_and_reservation_failure_are_blocked(tmp_path):
    manager = MixedManager()
    manager.lockin.ignore_writes.add("SLVL 0.004")
    lab = LabService(VisaBus(manager), RunStore(tmp_path / "recovery", tmp_path / "dropbox"))
    with TestClient(create_app(lab)) as client:
        assert RESOURCE in lab.blocked_resources
        assert client.post("/v1/sessions", json={"sample_name": "bad", "devices": {"lia": RESOURCE}}).status_code == 503
        manager.lockin.ignore_writes.clear()
        assert client.post("/v1/devices/recover", json={"resource": RESOURCE}).status_code == 200
        manager.lockin.settings["SLVL"] = .1
        manager.lockin.ignore_writes.add("SLVL 0.004")
        assert client.post("/v1/sessions", json={"sample_name": "bad", "devices": {"lia": RESOURCE}}).status_code == 502
        assert RESOURCE in lab.blocked_resources
        assert RESOURCE not in lab.bus.active


def test_bus_close_all_minimizes_reserved_sr830():
    manager = MixedManager()
    bus = VisaBus(manager)
    bus.discover()
    bus.open(RESOURCE)
    bus.close_all()
    assert_minimized(manager.lockin)
    assert manager.lockin.closed


def test_openapi_exposes_controls(server):
    client, _, _, _ = server
    schema = client.get("/openapi.json").json()
    assert schema["info"]["version"] == "1.1.1"
    assert "/v1/sessions/{session_id}/devices/{alias}/minimize-outputs" in schema["paths"]
    props = schema["components"]["schemas"]["ConfigureRequest"]["properties"]
    assert {"aux_outputs_v", "sine_amplitude_v_rms", "input_mode", "sensitivity"} <= props.keys()


@pytest.mark.parametrize("schema_version,historical_csv", [
    (1, b"timestamp_utc,current_a,voltage_v\r\n2026-09-21,1e-9,0.5\r\n"),
    (2, b"timestamp_utc,x,y,magnitude,unit,input_mode\r\n2026-09-24,3e-12,4e-12,5e-12,A,current_1e8\r\n"),
])
def test_historical_recording_is_loaded_and_mirrored_without_rewriting_csv(tmp_path, schema_version, historical_csv):
    store = RunStore(tmp_path / "recovery", tmp_path / "dropbox")
    record = store.create("historical", {})
    record.metadata["schema_version"] = schema_version
    record.write_metadata()
    record.data_path.write_bytes(historical_csv)
    store.recover_interrupted()
    store.sync_all()
    assert store.load(record.run_id)["schema_version"] == schema_version
    assert record.data_path.read_bytes() == historical_csv
    assert (store.dropbox_root / record.metadata["files"]["data"]).read_bytes() == historical_csv


def test_current_readings_are_preserved_in_grouped_csv(server):
    client, _, _, store = server
    sid, base = session(client)
    configured = client.post(base + "/configure", json={"input_mode": "current_1e8", "sensitivity": 1e-12})
    assert configured.status_code == 200
    read = client.post(f"/v1/sessions/{sid}/samples", json={"devices": ["lia"]})
    assert read.status_code == 200
    assert client.delete(f"/v1/sessions/{sid}").status_code == 200
    metadata = store.load(sid)
    with (store.recovery_root / metadata["files"]["data"]).open(newline="") as file:
        row = next(csv.DictReader(file))
    assert float(row["x"]) == 3e-12 and row["unit"] == "A"
    assert row["input_mode"] == "current_1e8" and row["voltage_v"] == row["current_a"] == ""


def test_reservation_rollback_minimizes_already_opened_lockin(tmp_path, monkeypatch):
    manager = MixedManager()
    lab = LabService(VisaBus(manager), RunStore(tmp_path / "recovery", tmp_path / "dropbox"))
    with TestClient(create_app(lab)) as client:
        manager.lockin.commands.clear()
        def unavailable_storage(*args, **kwargs):
            raise OSError("Cannot create recording")
        monkeypatch.setattr(lab.store, "create", unavailable_storage)
        response = client.post("/v1/sessions", json={"sample_name": "bad", "devices": {"lia": RESOURCE}})
        assert response.status_code == 507
        assert manager.lockin.writes.count("SLVL 0.004") == 2  # Initialization and rollback.
        assert_minimized(manager.lockin)
        assert not lab.owners and RESOURCE not in lab.bus.active
