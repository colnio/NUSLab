from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import ServiceError
from .instruments import Instrument, VisaBus, finite_number, utc_now
from .storage import RunRecord, RunStore


LOG = logging.getLogger(__name__)
LEASE_SECONDS = 90
HEARTBEAT_SECONDS = 30


@dataclass
class Session:
    id: str
    devices: dict[str, Instrument]
    record: RunRecord
    expires_at: float
    jobs: set[str] = field(default_factory=set)
    closing: bool = False
    finalizing: bool = False


@dataclass
class SweepJob:
    id: str
    session_id: str
    alias: str
    points: list[float]
    nplc: float
    source_delay_s: float
    compliance_a: float | None
    sense_range_a: float | None
    compliance_action: str
    status: str = "queued"
    error: dict | None = None
    rows: list[dict] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    created_at: str = field(default_factory=utc_now)
    finished_at: str | None = None
    ready: bool = False

    def snapshot(self) -> dict:
        visible_status = self.status if self.ready or self.status in {"queued", "running"} else "finalizing"
        return {"job_id": self.id, "run_id": self.session_id, "device_alias": self.alias,
                "status": visible_status, "total_points": len(self.points), "completed_points": len(self.rows),
                "created_at": self.created_at, "finished_at": self.finished_at if self.ready else None,
                "error": self.error}


class LabService:
    def __init__(self, bus: VisaBus | None = None, store: RunStore | None = None):
        self.bus = bus or VisaBus()
        self.store = store or RunStore()
        self.sessions: dict[str, Session] = {}
        self.jobs: dict[str, SweepJob] = {}
        self.owners: dict[str, str] = {}
        self.blocked_resources: dict[str, dict] = {}
        self.startup_results: list[dict] = []
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.monitor: threading.Thread | None = None
        self.sync_worker: threading.Thread | None = None
        self.started = False
        self._process_lock_file: Any = None

    def _claim_process_lock(self) -> None:
        path = self.store.recovery_root / ".server.lock"
        lock_file = path.open("a+b")
        lock_file.seek(0)
        if path.stat().st_size == 0:
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock_file.close()
            raise ServiceError("server_already_running", "Another GPIB server owns the recovery folder", 409) from exc
        self._process_lock_file = lock_file

    def start(self) -> None:
        if self.started:
            return
        self._claim_process_lock()
        try:
            self.startup_results = self.bus.startup_shutdown()
            self.blocked_resources = {item["resource"]: item["error"] for item in self.startup_results
                                      if item["status"] == "error"}
            self.store.recover_interrupted()
            self.stop_event.clear()
            self.monitor = threading.Thread(target=self._monitor, name="gpib-monitor", daemon=True)
            self.monitor.start()
            self.sync_worker = threading.Thread(target=self._sync_loop, name="gpib-sync", daemon=True)
            self.sync_worker.start()
            self.started = True
        except Exception:
            if self._process_lock_file is not None:
                self._process_lock_file.close()
                self._process_lock_file = None
            raise

    def stop(self) -> None:
        self.stop_event.set()
        if self.monitor is not None:
            self.monitor.join(timeout=3)
        if self.sync_worker is not None:
            self.sync_worker.join(timeout=3)
        for session_id in list(self.sessions):
            try:
                self.close_session(session_id)
            except Exception:
                LOG.exception("Session shutdown failed: %s", session_id)
        self.bus.close_all()
        if self._process_lock_file is not None:
            self._process_lock_file.close()
            self._process_lock_file = None
        self.started = False

    def _monitor(self) -> None:
        while not self.stop_event.wait(1):
            now = time.monotonic()
            with self.lock:
                expired = [sid for sid, session in self.sessions.items()
                           if not session.closing and not session.jobs and session.expires_at <= now]
            for sid in expired:
                LOG.warning("Session lease expired: %s", sid)
                with self.lock:
                    if sid in self.sessions:
                        self.sessions[sid].closing = True
                threading.Thread(target=self._expire_session, args=(sid,), daemon=True).start()

    def _expire_session(self, sid: str) -> None:
        try:
            self.close_session(sid, status="lease_expired")
        except Exception:
            LOG.exception("Lease cleanup failed: %s", sid)

    def _sync_loop(self) -> None:
        while not self.stop_event.wait(10):
            try:
                self.store.sync_all()
            except Exception:
                LOG.exception("Dropbox sync pass failed")

    def health(self) -> dict:
        return {"status": "ready" if self.started and not self.blocked_resources and not self.bus.discovery_errors else "degraded",
                "started": self.started, "startup_shutdown": self.startup_results,
                "blocked_resources": self.blocked_resources, "discovery_errors": self.bus.discovery_errors,
                "active_sessions": len(self.sessions)}

    def discover(self) -> list[dict]:
        devices = self.bus.discover()
        with self.lock:
            for item in devices:
                resource = item["resource"]
                item["owner_session_id"] = self.owners.get(resource)
                item["available"] = resource not in self.blocked_resources and resource not in self.owners
        return devices

    def recover_device(self, resource: str) -> dict:
        with self.lock:
            if resource in self.owners:
                raise ServiceError("device_busy", "Device is reserved", 409, resource=resource)
        self.bus.recover_identity(resource)
        device = self.bus.open(resource)
        try:
            if device.model in {"2400", "6430"}:
                with device.lock:
                    device.safe_off()
            with self.lock:
                self.blocked_resources.pop(resource, None)
            return {"resource": resource, "status": "ready"}
        finally:
            self.bus.close(resource)

    def create_session(self, sample_name: str, aliases: dict[str, str], notes: dict | None = None) -> dict:
        if not aliases or len(aliases) > 3:
            raise ServiceError("invalid_devices", "Reserve one to three devices", 422)
        if len(set(aliases.values())) != len(aliases):
            raise ServiceError("duplicate_device", "Each alias must name a different instrument", 422)
        for alias in aliases:
            if not alias.isidentifier() or len(alias) > 32:
                raise ServiceError("invalid_alias", "Aliases must be short Python-style identifiers", 422, alias=alias)
        with self.lock:
            for resource in aliases.values():
                if resource in self.blocked_resources:
                    raise ServiceError("device_unavailable", "Startup shutdown failed for this device", 503,
                                       resource=resource, cause=self.blocked_resources[resource])
                if resource in self.owners:
                    raise ServiceError("device_busy", "Device is already reserved", 409, resource=resource)
            opened: dict[str, Instrument] = {}
            try:
                for alias, resource in sorted(aliases.items(), key=lambda item: item[1]):
                    device = self.bus.open(resource)
                    opened[alias] = device
                    if device.model in {"2400", "6430"}:
                        with device.lock:
                            device.safe_off()
                record = self.store.create(sample_name, {alias: device.info for alias, device in opened.items()}, notes)
                session = Session(record.run_id, opened, record, time.monotonic() + LEASE_SECONDS)
                self.sessions[session.id] = session
                for resource in aliases.values():
                    self.owners[resource] = session.id
            except Exception:
                for device in opened.values():
                    try:
                        with device.lock:
                            if device.model in {"2400", "6430"}:
                                device.safe_off()
                    except Exception:
                        LOG.exception("Rollback shutdown failed: %s", device.resource)
                    self.bus.close(device.resource)
                raise
        return self.session_snapshot(session)

    def session_snapshot(self, session: Session) -> dict:
        return {"session_id": session.id, "run_id": session.record.run_id,
                "devices": {alias: device.info for alias, device in session.devices.items()},
                "lease_seconds": LEASE_SECONDS, "heartbeat_interval_seconds": HEARTBEAT_SECONDS,
                "expires_in_seconds": max(0, round(session.expires_at - time.monotonic(), 1)),
                "status": "closing" if session.closing else "active",
                "sync": session.record.metadata["sync"]}

    def get_session(self, session_id: str) -> Session:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ServiceError("session_not_found", "Session was not found", 404, session_id=session_id)
            if session.closing:
                raise ServiceError("session_closing", "Session is closing", 409, session_id=session_id)
            if session.expires_at <= time.monotonic() and not session.jobs:
                raise ServiceError("lease_expired", "Session lease expired", 409, session_id=session_id)
            return session

    def heartbeat(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        with self.lock:
            session.expires_at = time.monotonic() + LEASE_SECONDS
        return self.session_snapshot(session)

    def _device(self, session: Session, alias: str) -> Instrument:
        try:
            return session.devices[alias]
        except KeyError as exc:
            raise ServiceError("alias_not_found", "Device alias was not reserved", 404, alias=alias) from exc

    def _operate(self, session_id: str, alias: str, event: str, action: Callable[[Instrument], dict],
                 event_details: dict | None = None) -> dict:
        session = self.get_session(session_id)
        device = self._device(session, alias)
        self._ensure_device_idle(session, alias)
        try:
            with device.lock:
                result = action(device)
        except ServiceError as exc:
            try:
                session.record.append_event("error", {"operation": event, "alias": alias, "error": exc.payload()})
            except OSError as save_exc:
                self._emergency_off(device, session.record)
                raise ServiceError("storage_error", str(save_exc), 507, run_id=session.id) from save_exc
            if exc.code in {"visa_timeout", "visa_error", "instrument_error", "invalid_readback", "incomplete_readback"}:
                self._emergency_off(device, session.record)
            raise
        try:
            session.record.append_event(event, {"alias": alias, "resource": device.resource,
                                                "request": event_details or {}, "result": result})
        except OSError as exc:
            self._emergency_off(device, session.record)
            raise ServiceError("storage_error", str(exc), 507, run_id=session.id) from exc
        return result

    def _ensure_device_idle(self, session: Session, alias: str) -> None:
        with self.lock:
            if any(self.jobs[jid].alias == alias for jid in session.jobs):
                raise ServiceError("device_busy", "Device has an active sweep", 409, alias=alias)

    def _emergency_off(self, device: Instrument, record: RunRecord) -> None:
        if device.model not in {"2400", "6430"}:
            return
        try:
            with device.lock:
                device.safe_off()
        except ServiceError as exc:
            with self.lock:
                self.blocked_resources[device.resource] = exc.payload()
            try:
                record.append_event("shutdown_failed", {"resource": device.resource, "error": exc.payload()})
            except OSError:
                LOG.exception("Could not record shutdown failure: %s", device.resource)

    def configure(self, sid: str, alias: str, settings: dict) -> dict:
        return self._operate(sid, alias, "configure", lambda device: device.configure(settings), settings)

    def device_status(self, sid: str, alias: str) -> dict:
        return self._operate(sid, alias, "device_status", lambda device: device.configuration())

    def setpoint(self, sid: str, alias: str, value: float) -> dict:
        return self._operate(sid, alias, "setpoint", lambda device: device.setpoint(value), {"value": value})

    def output(self, sid: str, alias: str, enabled: bool) -> dict:
        return self._operate(sid, alias, "output", lambda device: device.output(enabled), {"enabled": enabled})

    def read(self, sid: str, alias: str, function: str | None = None) -> dict:
        session = self.get_session(sid)
        result = self._operate(sid, alias, "read", lambda device: device.measure(function), {"function": function})
        device = self._device(session, alias)
        row = self._reading_row(session, alias, device, result, "read", uuid.uuid4().hex)
        try:
            session.record.append_rows([row])
        except OSError as exc:
            self._emergency_off(device, session.record)
            raise ServiceError("storage_error", str(exc), 507, run_id=sid) from exc
        return result

    def sample(self, sid: str, aliases: list[str]) -> dict:
        session = self.get_session(sid)
        if not aliases or len(set(aliases)) != len(aliases):
            raise ServiceError("invalid_sample", "Provide distinct device aliases", 422)
        for alias in aliases:
            self._device(session, alias)
            self._ensure_device_idle(session, alias)
        sample_id = uuid.uuid4().hex
        readings = {}
        rows = []
        rows_saved = False
        try:
            for alias in aliases:
                device = session.devices[alias]
                with device.lock:
                    result = device.measure()
                readings[alias] = result
                rows.append(self._reading_row(session, alias, device, result, "sample", sample_id))
            session.record.append_rows(rows)
            rows_saved = True
            session.record.append_event("grouped_sample", {"sample_id": sample_id, "aliases": aliases})
        except (OSError, ServiceError) as exc:
            if rows and not rows_saved:
                try:
                    session.record.append_rows(rows)
                except OSError:
                    pass
            for device in session.devices.values():
                self._emergency_off(device, session.record)
            if isinstance(exc, OSError):
                raise ServiceError("storage_error", str(exc), 507, run_id=sid) from exc
            raise
        return {"sample_id": sample_id, "readings": readings}

    def _reading_row(self, session: Session, alias: str, device: Instrument, result: dict,
                     operation: str, sample_id: str) -> dict:
        return {"timestamp_utc": result["timestamp"], "run_id": session.id, "sample_id": sample_id,
                "job_id": "", "device_alias": alias, "resource": device.resource, "model": device.model,
                "operation": operation, "voltage_v": result.get("voltage_v"),
                "current_a": result.get("current_a"), "value": result.get("value"),
                "unit": result.get("unit"), "function": result.get("function"),
                "status_word": result.get("status_word"), "in_compliance": result.get("in_compliance"),
                "range_compliance": result.get("range_compliance"), "overload": result.get("overload")}

    def raw_scpi(self, sid: str, alias: str, kind: str, command: str) -> dict:
        return self._operate(sid, alias, "raw_scpi", lambda device: device.raw_scpi(kind, command),
                             {"kind": kind, "command": command})

    def start_sweep(self, sid: str, alias: str, points: list[float], nplc: float,
                    source_delay_s: float = 0.0, compliance_a: float | None = None,
                    sense_range_a: float | None = None, compliance_action: str = "stop") -> dict:
        session = self.get_session(sid)
        device = self._device(session, alias)
        if device.model not in {"2400", "6430"}:
            raise ServiceError("unsupported_operation", "Voltage list sweep requires a source meter", 422)
        if not points:
            raise ServiceError("invalid_sweep", "voltage_points_v must not be empty", 422)
        values = [finite_number(item, "voltage point") for item in points]
        nplc = finite_number(nplc, "nplc")
        source_delay_s = finite_number(source_delay_s, "source_delay_s")
        if nplc <= 0 or source_delay_s < 0:
            raise ServiceError("invalid_sweep", "nplc must be positive and source_delay_s nonnegative", 422)
        if compliance_a is not None:
            compliance_a = finite_number(compliance_a, "compliance_a")
            if compliance_a <= 0:
                raise ServiceError("invalid_sweep", "compliance_a must be positive", 422)
        if sense_range_a is not None:
            sense_range_a = finite_number(sense_range_a, "sense_range_a")
            if sense_range_a <= 0:
                raise ServiceError("invalid_sweep", "sense_range_a must be positive", 422)
        if compliance_action not in {"stop", "continue"}:
            raise ServiceError("invalid_sweep", "compliance_action must be stop or continue", 422)
        with self.lock:
            if any(self.jobs[jid].alias == alias for jid in session.jobs):
                raise ServiceError("device_busy", "Device already has a running sweep", 409)
            with device.lock:
                if device.output_enabled():
                    raise ServiceError("output_active", "Switch output off before starting a list sweep", 409)
            job = SweepJob(uuid.uuid4().hex, sid, alias, values, nplc, source_delay_s,
                           compliance_a, sense_range_a, compliance_action)
            self.jobs[job.id] = job
            session.jobs.add(job.id)
            session.record.metadata["jobs"][job.id] = job.snapshot()
            session.record.write_metadata()
            job.thread = threading.Thread(target=self._run_sweep, args=(job,), name=f"sweep-{job.id[:8]}", daemon=True)
            job.thread.start()
        return job.snapshot()

    def _run_sweep(self, job: SweepJob) -> None:
        session = self.sessions[job.session_id]
        device = session.devices[job.alias]
        job.status = "running"
        outcome = "completed"
        try:
            session.record.append_event("sweep_started", {"job_id": job.id, "alias": job.alias,
                                                           "points": len(job.points), "nplc": job.nplc,
                                                           "source_delay_s": job.source_delay_s,
                                                           "compliance_a": job.compliance_a,
                                                           "compliance_action": job.compliance_action})
            point_s = job.nplc / 50 + job.source_delay_s + 0.02
            max_batch_seconds = 2 if job.compliance_action == "stop" else 30
            batch_size = min(2500, max(1, int(max_batch_seconds / point_s)))
            for start in range(0, len(job.points), batch_size):
                if job.cancel_event.is_set():
                    outcome = "cancelled"
                    break
                points = job.points[start:start + batch_size]
                with device.lock:
                    readings = device.sweep_batch(points, job.nplc, job.source_delay_s,
                                                  job.compliance_a, job.sense_range_a)
                timestamp = utc_now()
                rows = []
                for offset, reading in enumerate(readings):
                    row = {"timestamp_utc": timestamp, "run_id": session.id,
                           "sample_id": f"{job.id}-{start + offset}", "job_id": job.id,
                           "device_alias": job.alias, "resource": device.resource, "model": device.model,
                           "operation": "voltage_list", "point_index": start + offset,
                           "source_voltage_v": points[offset], **reading}
                    rows.append(row)
                job.rows.extend(rows)
                session.record.append_rows(rows)
                session.record.metadata["jobs"][job.id] = job.snapshot()
                session.record.write_metadata()
                with self.lock:
                    session.expires_at = time.monotonic() + LEASE_SECONDS
                if any(row["in_compliance"] or row["range_compliance"] for row in rows):
                    session.record.append_event("compliance", {"job_id": job.id, "first_point": start})
                    if job.compliance_action == "stop":
                        outcome = "compliance_stop"
                        break
        except ServiceError as exc:
            outcome = "failed"
            job.error = exc.payload()
            self._safe_event(session.record, "sweep_error", {"job_id": job.id, "error": job.error})
        except Exception as exc:
            outcome = "failed"
            code = "storage_error" if isinstance(exc, OSError) else "internal_error"
            job.error = ServiceError(code, str(exc), 507 if code == "storage_error" else 500).payload()
            self._safe_event(session.record, "sweep_error", {"job_id": job.id, "error": job.error})
            LOG.exception("Sweep failed: %s", job.id)
        finally:
            job.status = "finalizing"
            self._emergency_off(device, session.record)
            if job.rows:
                try:
                    session.record.plot_sweep(job.id, job.rows)
                except Exception as exc:
                    session.record.metadata["warnings"].append(f"Plot failed for {job.id}: {exc}")
                    self._safe_event(session.record, "plot_error", {"job_id": job.id, "error": str(exc)})
            job.status = outcome
            job.finished_at = utc_now()
            try:
                session.record.metadata["jobs"][job.id] = job.snapshot()
                session.record.write_metadata()
                session.record.append_event("sweep_finished", job.snapshot())
                session.record.sync()
            except Exception as exc:
                job.status = "failed"
                job.error = ServiceError("storage_error", str(exc), 507).payload()
                LOG.exception("Could not finish sweep recording: %s", job.id)
            finally:
                with self.lock:
                    session.jobs.discard(job.id)
                    job.ready = True
                try:
                    session.record.metadata["jobs"][job.id] = job.snapshot()
                    session.record.write_metadata()
                    session.record.sync()
                except Exception:
                    LOG.exception("Could not save final job state: %s", job.id)

    @staticmethod
    def _safe_event(record: RunRecord, event: str, details: dict) -> None:
        try:
            record.append_event(event, details)
        except OSError:
            LOG.exception("Could not append run event: %s", event)

    def get_job(self, job_id: str) -> SweepJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise ServiceError("job_not_found", "Sweep job was not found", 404, job_id=job_id)
        return job

    def cancel_job(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        job.cancel_event.set()
        return job.snapshot()

    def job_rows(self, job_id: str, after: int = 0, limit: int = 1000) -> dict:
        job = self.get_job(job_id)
        if after < 0 or limit < 1 or limit > 10000:
            raise ServiceError("invalid_cursor", "after must be nonnegative and limit 1 to 10000", 422)
        rows = job.rows[after:after + limit]
        return {"job_id": job_id, "rows": rows, "next_cursor": after + len(rows),
                "total_available": len(job.rows), "status": job.status}

    def close_session(self, sid: str, status: str = "closed") -> dict:
        with self.lock:
            session = self.sessions.get(sid)
            if session is None:
                raise ServiceError("session_not_found", "Session was not found", 404, session_id=sid)
            if session.finalizing:
                return {"run_id": sid, "status": "shutdown_pending"}
            session.closing = True
            jobs = [self.jobs[jid] for jid in session.jobs]
            for job in jobs:
                job.cancel_event.set()
        for job in jobs:
            if job.thread is not None and job.thread is not threading.current_thread():
                job.thread.join(timeout=60)
        pending = [job for job in jobs if job.thread is not None and job.thread.is_alive()]
        if pending:
            threading.Thread(target=self._wait_and_close, args=(sid, status, pending),
                             name=f"close-{sid[:8]}", daemon=True).start()
            return {"run_id": sid, "status": "shutdown_pending", "jobs": [job.id for job in pending]}
        return self._finalize_session(sid, status)

    def _wait_and_close(self, sid: str, status: str, jobs: list[SweepJob]) -> None:
        for job in jobs:
            if job.thread is not None:
                job.thread.join()
        try:
            self._finalize_session(sid, status)
        except Exception:
            LOG.exception("Deferred session shutdown failed: %s", sid)

    def _finalize_session(self, sid: str, status: str) -> dict:
        with self.lock:
            session = self.sessions.get(sid)
            if session is None:
                return {"run_id": sid, "status": status}
            if session.finalizing:
                return {"run_id": sid, "status": "shutdown_pending"}
            session.finalizing = True
        failures = []
        for device in session.devices.values():
            try:
                with device.lock:
                    if device.model in {"2400", "6430"}:
                        device.safe_off()
            except ServiceError as exc:
                failures.append(exc.payload())
                self.blocked_resources[device.resource] = exc.payload()
            finally:
                try:
                    self.bus.close(device.resource)
                except Exception as exc:
                    failure = ServiceError("visa_close_failed", str(exc), 502, resource=device.resource).payload()
                    failures.append(failure)
                    self.blocked_resources[device.resource] = failure
                with self.lock:
                    self.owners.pop(device.resource, None)
        sync = None
        save_error = None
        try:
            session.record.update(status=status, closed_at=utc_now())
            session.record.append_event("session_closed", {"status": status, "shutdown_errors": failures})
            sync = session.record.sync()
        except OSError as exc:
            save_error = str(exc)
        finally:
            with self.lock:
                self.sessions.pop(sid, None)
            self.store.forget(sid)
        if failures:
            raise ServiceError("shutdown_failed", "One or more outputs could not be verified off", 502,
                               errors=failures, run_id=sid, sync=sync)
        if save_error:
            raise ServiceError("storage_error", save_error, 507, run_id=sid)
        return {"run_id": sid, "status": status, "sync": sync}
