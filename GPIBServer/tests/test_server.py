from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from GPIBServer.app import create_app
from GPIBServer.instruments import Instrument, VisaBus, parse_idn, parse_sweep
from GPIBServer.service import LabService
from GPIBServer.storage import RunStore


@dataclass
class FakeState:
    model: str
    serial: str
    source_mode: str = "VOLT"
    output: bool = False
    setpoint: float = 0.0
    compliance: float = 0.001
    points: list[float] = field(default_factory=list)
    format: str = "READ"
    function: str = "VOLT:DC"
    commands: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    fail_read: bool = False
    short_read: bool = False
    compliance_point: int | None = None
    read_delay_s: float = 0.0
    idn_stuck: bool = False


class FakeHandle:
    def __init__(self, state: FakeState):
        self.state = state
        self.timeout = 5000
        self.read_termination = "\n"
        self.write_termination = "\n"
        self.closed = False

    def write(self, command: str):
        self.state.commands.append(command)
        upper = command.upper().strip()
        if upper.startswith(":OUTP ON"):
            self.state.output = True
        elif upper.startswith(":OUTP OFF"):
            self.state.output = False
        elif upper.startswith(":SOUR:FUNC "):
            self.state.source_mode = upper.rsplit(" ", 1)[1]
        elif upper.startswith(":SOUR:VOLT:LEV "):
            self.state.setpoint = float(command.split()[-1])
        elif upper.startswith(":SOUR:CURR:LEV "):
            self.state.setpoint = float(command.split()[-1])
        elif upper.startswith(":SENS:CURR:PROT "):
            self.state.compliance = float(command.split()[-1])
        elif upper.startswith(":SENS:CURR:RANG "):
            if self.state.model == "6430" and float(command.split()[-1]) > self.state.compliance:
                self.state.errors.append('824,"Cannot exceed compliance range"')
        elif upper.startswith((":SOUR:LIST:VOLT ", ":SOUR:LIST:VOLT:APP ")):
            points = [float(part) for part in command.split(" ", 1)[1].split(",")]
            if len(points) > 100:
                self.state.errors.append('-223,"Too much data"')
            elif ":APP " in upper:
                self.state.points.extend(points)
            else:
                self.state.points = points
        elif upper.startswith(":FORM:ELEM "):
            self.state.format = upper.split(" ", 1)[1]
        elif upper.startswith(":SENS:FUNC "):
            self.state.function = command.split(" ", 1)[1].strip("'\"")

    def clear(self):
        self.state.commands.append("VISA_CLEAR")
        self.state.idn_stuck = False

    def query(self, command: str) -> str:
        self.state.commands.append(command)
        upper = command.upper().strip()
        if upper == "*IDN?":
            if self.state.idn_stuck:
                raise TimeoutError("VISA timeout")
            return f"KEITHLEY INSTRUMENTS INC.,MODEL {self.state.model},{self.state.serial},B01"
        if upper == ":SOUR:FUNC?":
            return self.state.source_mode
        if upper == ":OUTP?":
            return "1" if self.state.output else "0"
        if upper == ":SENS:CURR:PROT?":
            return str(self.state.compliance)
        if upper == ":SENS:VOLT:PROT?":
            return "10"
        if upper == ":SENS:FUNC?":
            return f"'{self.state.function}'"
        if upper == ":SOUR:LIST:VOLT:POIN?":
            return str(len(self.state.points))
        if upper == ":SYST:ERR?":
            return self.state.errors.pop(0) if self.state.errors else '0,"No error"'
        if upper == ":READ?":
            if self.state.read_delay_s:
                time.sleep(self.state.read_delay_s)
            if self.state.fail_read:
                raise TimeoutError("VISA timeout")
            if self.state.model == "2002":
                return "0.0123"
            points = self.state.points if "TIME" in self.state.format else [self.state.setpoint]
            fields = []
            for index, voltage in enumerate(points):
                status = 8 if index == self.state.compliance_point else 0
                fields.extend([str(voltage), str(voltage / 1_000_000), str(index * 0.01), str(status)])
            if self.state.short_read:
                fields.pop()
            if "TIME" in self.state.format:
                return ",".join(fields)
            return f"{self.state.setpoint},{self.state.setpoint / 1_000_000},0"
        raise AssertionError(f"Unexpected query: {command}")

    def close(self):
        self.closed = True


class FakeResourceManager:
    def __init__(self):
        self.states = {
            "GPIB0::12::INSTR": FakeState("2400", "A2400", output=True),
            "GPIB0::15::INSTR": FakeState("6430", "A6430", output=True),
            "GPIB0::16::INSTR": FakeState("2002", "A2002"),
        }
        self.closed = False

    def list_resources(self):
        return (*self.states, "ASRL1::INSTR")

    def open_resource(self, resource: str):
        return FakeHandle(self.states[resource])

    def close(self):
        self.closed = True


@pytest.fixture
def server(tmp_path):
    manager = FakeResourceManager()
    store = RunStore(tmp_path / "recovery", tmp_path / "dropbox")
    lab = LabService(VisaBus(manager), store)
    with TestClient(create_app(lab)) as client:
        yield client, lab, manager, store


def create_session(client, devices=None):
    response = client.post("/v1/sessions", json={"sample_name": "FET A", "devices": devices or {
        "sd": "GPIB0::12::INSTR", "gate": "GPIB0::15::INSTR", "probe": "GPIB0::16::INSTR"}})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


def await_job(client, job_id):
    for _ in range(100):
        data = client.get(f"/v1/jobs/{job_id}").json()
        if data["finished_at"] is not None and data["status"] != "finalizing":
            return data
        time.sleep(0.05)
    pytest.fail("job did not finish")


@pytest.mark.parametrize("model", ["2400", "6430"])
@pytest.mark.parametrize("count", [100, 101, 241, 2500])
def test_long_list_upload_preserves_order_and_single_output_interval(model, count):
    state = FakeState(model, "test", compliance=1e-5)
    device = Instrument(parse_idn("GPIB0::1::INSTR", f"KEITHLEY,MODEL {model},test,1"), FakeHandle(state))
    points = [((i % 21) - 10) / 10 for i in range(count)]
    rows = device.sweep_batch(points, .1, .05, 1e-5, 1e-5)
    assert state.points == points
    assert [row["voltage_v"] for row in rows] == points
    assert state.commands.count(":OUTP ON") == 1
    assert state.commands.index(":SOUR:LIST:VOLT:POIN?") < state.commands.index(":OUTP ON")
    assert not state.output


def test_6430_widens_compliance_before_requested_sense_range():
    state = FakeState("6430", "test", compliance=1e-6)
    device = Instrument(parse_idn("GPIB0::1::INSTR", "KEITHLEY,MODEL 6430,test,1"), FakeHandle(state))
    result = device.configure({"source_mode": "voltage", "sense_range": 1e-5, "compliance": 1e-5})
    assert result["compliance"] == 1e-5
    assert not result["output_enabled"]


def test_startup_discovery_and_shutdown_are_nonresetting(server):
    client, lab, manager, _ = server
    assert client.get("/v1/health").json()["status"] == "ready"
    devices = client.get("/v1/devices").json()["devices"]
    assert {device["model"] for device in devices} == {"2400", "6430", "2002"}
    assert not manager.states["GPIB0::12::INSTR"].output
    assert not manager.states["GPIB0::15::INSTR"].output
    for state in manager.states.values():
        assert "*CLS" not in state.commands
        assert "*RST" not in state.commands


def test_concurrent_reservations_grouped_sample_raw_and_release(server):
    client, lab, manager, store = server
    sid = create_session(client)
    busy = client.post("/v1/sessions", json={"sample_name": "other", "devices": {"x": "GPIB0::12::INSTR"}})
    assert busy.status_code == 409
    assert busy.json()["error"]["code"] == "device_busy"
    assert client.post(f"/v1/sessions/{sid}/heartbeat").status_code == 200
    assert client.post(f"/v1/sessions/{sid}/devices/sd/configure", json={"source_mode": "voltage", "compliance": 1e-4}).status_code == 200
    status = client.get(f"/v1/sessions/{sid}/devices/sd")
    assert status.status_code == 200
    assert status.json()["source_mode"] == "voltage"
    assert client.post(f"/v1/sessions/{sid}/devices/sd/setpoint", json={"value": 0.1}).status_code == 200
    assert client.post(f"/v1/sessions/{sid}/devices/sd/output", json={"enabled": True}).status_code == 200
    response = client.post(f"/v1/sessions/{sid}/samples", json={"devices": ["sd", "gate", "probe"]})
    assert response.status_code == 200, response.text
    assert len(response.json()["readings"]) == 3
    assert response.json()["readings"]["probe"]["unit"] == "V"
    raw = client.post(f"/v1/sessions/{sid}/devices/sd/scpi", json={"kind": "query", "command": "*IDN?"})
    assert raw.status_code == 200
    assert "2400" in raw.json()["response"]
    closed = client.delete(f"/v1/sessions/{sid}")
    assert closed.status_code == 200
    assert not manager.states["GPIB0::12::INSTR"].output
    metadata = store.load(sid)
    assert metadata["measurement_count"] == 3
    assert metadata["sync"]["state"] == "synced"
    assert (store.dropbox_root / metadata["files"]["data"]).exists()


@pytest.mark.parametrize("model_resource", ["GPIB0::12::INSTR", "GPIB0::15::INSTR"])
def test_voltage_list_sweeps_and_compliance_policy(server, model_resource):
    client, _, manager, store = server
    sid = create_session(client, {"smu": model_resource})
    state = manager.states[model_resource]
    state.compliance_point = 1
    payload = {"device": "smu", "voltage_points_v": [0, 0.1, 0.2, 0],
               "nplc": 0.01, "compliance_a": 1e-5}
    start = client.post(f"/v1/sessions/{sid}/sweeps", json=payload)
    assert start.status_code == 202, start.text
    job = await_job(client, start.json()["job_id"])
    assert job["status"] == "compliance_stop"
    rows = client.get(f"/v1/jobs/{job['job_id']}/readings?after=0&limit=2").json()
    assert rows["next_cursor"] == 2
    assert rows["rows"][1]["in_compliance"] is True
    assert not state.output
    state.compliance_point = 1
    payload["compliance_action"] = "continue"
    start2 = client.post(f"/v1/sessions/{sid}/sweeps", json=payload)
    job2 = await_job(client, start2.json()["job_id"])
    assert job2["status"] == "completed"
    assert job2["completed_points"] == 4
    client.delete(f"/v1/sessions/{sid}")
    metadata = store.load(sid)
    assert metadata["measurement_count"] == 8
    assert len(metadata["files"]["plots"]) >= 2


def test_lease_expiry_zeroes_outputs(server):
    client, lab, manager, _ = server
    sid = create_session(client, {"smu": "GPIB0::12::INSTR"})
    client.post(f"/v1/sessions/{sid}/devices/smu/output", json={"enabled": True})
    lab.sessions[sid].expires_at = time.monotonic() - 1
    for _ in range(40):
        if sid not in lab.sessions:
            break
        time.sleep(0.1)
    assert sid not in lab.sessions
    assert not manager.states["GPIB0::12::INSTR"].output


def test_timeout_and_incomplete_readback_preserve_partial(server):
    client, _, manager, store = server
    sid = create_session(client, {"smu": "GPIB0::15::INSTR"})
    state = manager.states["GPIB0::15::INSTR"]
    state.short_read = True
    start = client.post(f"/v1/sessions/{sid}/sweeps", json={"device": "smu", "voltage_points_v": [0, 0.1]})
    job = await_job(client, start.json()["job_id"])
    assert job["status"] == "failed"
    assert job["error"]["code"] == "incomplete_readback"
    assert not state.output
    state.short_read = False
    state.fail_read = True
    failed = client.post(f"/v1/sessions/{sid}/devices/smu/read", json={})
    assert failed.status_code == 504
    assert failed.json()["error"]["code"] == "visa_timeout"
    assert not state.output
    client.delete(f"/v1/sessions/{sid}")
    assert store.load(sid)["jobs"][job["job_id"]]["status"] == "failed"


def test_dropbox_fallback_and_retry(tmp_path):
    manager = FakeResourceManager()
    blocked_root = tmp_path / "blocked"
    blocked_root.write_text("file", encoding="utf-8")
    store = RunStore(tmp_path / "recovery", blocked_root)
    lab = LabService(VisaBus(manager), store)
    with TestClient(create_app(lab)) as client:
        sid = create_session(client, {"probe": "GPIB0::16::INSTR"})
        assert client.post(f"/v1/sessions/{sid}/devices/probe/read", json={}).status_code == 200
        closed = client.delete(f"/v1/sessions/{sid}").json()
        assert closed["sync"]["state"] == "pending_sync"
        assert store.load(sid)["measurement_count"] == 1
        blocked_root.unlink()
        blocked_root.mkdir()
        assert store.sync_all()[0]["state"] == "synced"
        metadata = store.load(sid)
        assert (blocked_root / metadata["files"]["data"]).exists()


def test_parser_rejects_short_or_overloaded_values():
    with pytest.raises(Exception):
        parse_sweep("0,1,2", 1)
    rows = parse_sweep("9.9e37,1e-6,0,8", 1)
    assert rows[0]["voltage_v"] is None
    assert rows[0]["overload"] is True
    assert rows[0]["in_compliance"] is True
    invalid_time = parse_sweep("0,0,-1851002,0,0.1,1e-7,-1851002,0", 2)
    assert all(row["timestamp_invalid"] and row["instrument_time_s"] is None for row in invalid_time)


def test_sweep_batches_and_cancel_at_boundary(server):
    client, lab, manager, _ = server
    sid = create_session(client, {"smu": "GPIB0::12::INSTR"})
    state = manager.states["GPIB0::12::INSTR"]
    state.read_delay_s = 0.2
    points = [0.1] * 120
    response = client.post(f"/v1/sessions/{sid}/sweeps", json={
        "device": "smu", "voltage_points_v": points, "nplc": 0.01, "compliance_action": "stop"})
    jid = response.json()["job_id"]
    for _ in range(50):
        if ":READ?" in state.commands:
            break
        time.sleep(0.01)
    client.post(f"/v1/jobs/{jid}/cancel")
    job = await_job(client, jid)
    assert job["status"] == "cancelled"
    assert 0 < job["completed_points"] < len(points)
    assert not state.output
    client.delete(f"/v1/sessions/{sid}")


def test_raw_errors_and_rejected_config_have_structured_results(server):
    client, _, manager, _ = server
    sid = create_session(client, {"smu": "GPIB0::12::INSTR"})
    state = manager.states["GPIB0::12::INSTR"]
    state.errors.append('-222,"Data out of range"')
    raw = client.post(f"/v1/sessions/{sid}/devices/smu/scpi", json={"kind": "query", "command": "*IDN?"})
    assert raw.status_code == 200
    assert raw.json()["instrument_errors"] == ['-222,"Data out of range"']
    before = len(state.commands)
    invalid = client.post(f"/v1/sessions/{sid}/devices/smu/configure", json={
        "source_mode": "current", "nplc": -1})
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_value"
    assert not any(command.startswith(":SOUR:FUNC ") for command in state.commands[before:])
    client.delete(f"/v1/sessions/{sid}")


def test_local_recording_failure_switches_source_off(server, monkeypatch):
    client, lab, manager, _ = server
    sid = create_session(client, {"smu": "GPIB0::12::INSTR"})
    client.post(f"/v1/sessions/{sid}/devices/smu/output", json={"enabled": True})

    def full_disk(rows):
        raise OSError("Disk full")

    monkeypatch.setattr(lab.sessions[sid].record, "append_rows", full_disk)
    response = client.post(f"/v1/sessions/{sid}/devices/smu/read", json={})
    assert response.status_code == 507
    assert response.json()["error"]["code"] == "storage_error"
    assert not manager.states["GPIB0::12::INSTR"].output
    client.delete(f"/v1/sessions/{sid}")


def test_interrupted_run_is_preserved_and_retried(tmp_path):
    store = RunStore(tmp_path / "recovery", tmp_path / "dropbox")
    run = store.create("sample", {"smu": {"model": "2400", "resource": "GPIB0::12::INSTR"}})
    run.append_rows([{"run_id": run.run_id, "operation": "read", "voltage_v": 0.1}])
    store.recover_interrupted()
    metadata = store.load(run.run_id)
    assert metadata["status"] == "interrupted"
    assert metadata["sync"]["state"] == "pending_sync"
    assert store.sync_all()[0]["state"] == "synced"


def test_active_run_retries_sync_and_marks_new_data_pending(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("file", encoding="utf-8")
    store = RunStore(tmp_path / "recovery", blocked)
    run = store.create("active", {"meter": {"model": "2002"}})
    run.append_rows([{"run_id": run.run_id, "operation": "read", "value": 1.0}])
    assert store.sync_all()[0]["state"] == "pending_sync"
    blocked.unlink()
    blocked.mkdir()
    assert store.sync_all()[0]["state"] == "synced"
    run.append_event("later_read", {"value": 2.0})
    assert run.metadata["sync"]["state"] == "pending_sync"
    assert store.sync_all()[0]["state"] == "synced"


def test_explicit_recovery_clears_stuck_gpib_without_clearing_discovery(tmp_path):
    manager = FakeResourceManager()
    manager.states["GPIB0::12::INSTR"].idn_stuck = True
    lab = LabService(VisaBus(manager), RunStore(tmp_path / "recovery", tmp_path / "dropbox"))
    with TestClient(create_app(lab)) as client:
        assert client.get("/v1/health").json()["status"] == "degraded"
        assert "VISA_CLEAR" not in manager.states["GPIB0::12::INSTR"].commands
        result = client.post("/v1/devices/recover", json={"resource": "GPIB0::12::INSTR"})
        assert result.status_code == 200, result.text
        assert result.json()["status"] == "ready"
        assert "VISA_CLEAR" in manager.states["GPIB0::12::INSTR"].commands
        assert client.get("/v1/health").json()["status"] == "ready"
