from __future__ import annotations

import csv
import json
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .errors import ServiceError


SINGAPORE = timezone(timedelta(hours=8))
DEFAULT_DROPBOX = Path(os.environ.get(
    "NUSLAB_GPIB_DROPBOX_ROOT",
    r"C:\Users\MeasurmentStand\NUS Dropbox\Ozyilmaz Group\1_MAC\1_Projects\Iurii\ResistiveSwitches\VacuumProbeData",
))
DEFAULT_RECOVERY = Path(os.environ.get(
    "NUSLAB_GPIB_RECOVERY_ROOT",
    str(Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "NUSLab" / "GPIBServer" / "recovery"),
))

DATA_COLUMNS = [
    "timestamp_utc", "run_id", "sample_id", "job_id", "device_alias", "resource", "model",
    "operation", "point_index", "source_voltage_v", "voltage_v", "current_a", "value", "unit",
    "function", "instrument_time_s", "instrument_time_raw_s", "timestamp_invalid",
    "status_word", "in_compliance", "range_compliance", "overload",
    "x", "y", "magnitude", "phase_deg", "frequency_hz", "input_mode", "status_scope",
    "input_overload", "filter_overload", "output_overload", "reference_unlocked",
    "frequency_range_changed", "time_constant_changed", "data_triggered",
    "standard_event_status", "error_status",
    "sensitivity", "range_exceeded",
]
PLOT_LOCK = threading.Lock()


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", name.strip())
    cleaned = cleaned.strip(" .")[:80]
    if cleaned in {"", ".", ".."}:
        raise ServiceError("invalid_sample_name", "sample_name must contain a usable character", 422)
    return cleaned


def replace_with_retry(source: Path, destination: Path) -> None:
    """Retry brief Windows file locks, preserving atomic replacement and old data.

    Persistent failures still reach the caller's storage-error shutdown path.
    """
    delays = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


class RunStore:
    def __init__(self, recovery_root: Path = DEFAULT_RECOVERY, dropbox_root: Path = DEFAULT_DROPBOX):
        self.recovery_root = Path(recovery_root)
        self.dropbox_root = Path(dropbox_root)
        self.lock = threading.RLock()
        self.records: dict[str, RunRecord] = {}
        self.recovery_root.mkdir(parents=True, exist_ok=True)

    def create(self, sample_name: str, devices: dict[str, dict], notes: dict | None = None) -> "RunRecord":
        run_id = uuid.uuid4().hex
        local_date = datetime.now(SINGAPORE).date().isoformat()
        name = safe_name(sample_name)
        stem = f"REST_{name}_{datetime.now(SINGAPORE):%Y-%m-%d_%H-%M-%S}_{run_id[:8]}"
        relative_base = Path(local_date) / name
        data_rel = relative_base / "data" / f"{stem}.data"
        event_rel = relative_base / "data" / f"{stem}.events.jsonl"
        meta_rel = relative_base / "data" / f"{stem}.meta.json"
        metadata = {
            "schema_version": 3, "run_id": run_id, "sample_name": sample_name,
            "sample_folder": name, "created_at": datetime.now(timezone.utc).isoformat(),
            "devices": devices, "notes": notes or {}, "status": "active",
            "measurement_count": 0, "jobs": {}, "warnings": [],
            "files": {"data": data_rel.as_posix(), "events": event_rel.as_posix(), "metadata": meta_rel.as_posix(), "plots": []},
            "sync": {"state": "pending_sync", "last_error": None},
            "local_root": str(self.recovery_root), "dropbox_root": str(self.dropbox_root),
        }
        record = RunRecord(self, metadata, self.recovery_root / meta_rel)
        with self.lock:
            self.records[run_id] = record
        record.write_metadata()
        record.append_event("session_created", {"devices": devices})
        return record

    def load(self, run_id: str) -> dict:
        for path in self.recovery_root.rglob("*.meta.json"):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if metadata.get("run_id") == run_id:
                return metadata
        raise ServiceError("run_not_found", "Run was not found", 404, run_id=run_id)

    def sync_all(self) -> list[dict]:
        results = []
        with self.lock:
            for path in self.recovery_root.rglob("*.meta.json"):
                try:
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                    if metadata.get("sync", {}).get("state") == "synced":
                        continue
                    record = self.records.get(metadata.get("run_id")) or RunRecord(self, metadata, path)
                    results.append(record.sync())
                except (OSError, ValueError) as exc:
                    results.append({"metadata_path": str(path), "state": "error", "error": str(exc)})
        return results

    def forget(self, run_id: str) -> None:
        with self.lock:
            self.records.pop(run_id, None)

    def recover_interrupted(self) -> None:
        """A fresh process cannot resume an old session; preserve and mirror its files."""
        with self.lock:
            for path in self.recovery_root.rglob("*.meta.json"):
                try:
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if metadata.get("status") == "active":
                    metadata["status"] = "interrupted"
                    metadata["closed_at"] = datetime.now(timezone.utc).isoformat()
                    metadata.setdefault("warnings", []).append("Server stopped before session release")
                    metadata["sync"] = {"state": "pending_sync", "last_error": None}
                    atomic_json(path, metadata)


class RunRecord:
    def __init__(self, store: RunStore, metadata: dict, meta_path: Path):
        self.store = store
        self.metadata = metadata
        self.meta_path = meta_path
        self.lock = threading.RLock()

    @property
    def run_id(self) -> str:
        return self.metadata["run_id"]

    @property
    def data_path(self) -> Path:
        return self.store.recovery_root / self.metadata["files"]["data"]

    @property
    def event_path(self) -> Path:
        return self.store.recovery_root / self.metadata["files"]["events"]

    def write_metadata(self) -> None:
        with self.lock:
            atomic_json(self.meta_path, self.metadata)

    def update(self, **changes: Any) -> None:
        with self.lock:
            self.metadata.update(changes)
            self.metadata["sync"] = {"state": "pending_sync", "last_error": None}
            self.write_metadata()

    def append_event(self, event: str, details: dict) -> None:
        with self.lock:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                       "event": event, "details": details}
            with self.event_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, allow_nan=False, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.metadata["sync"] = {"state": "pending_sync", "last_error": None}
            self.write_metadata()

    def append_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self.lock:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            needs_header = not self.data_path.exists() or self.data_path.stat().st_size == 0
            with self.data_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=DATA_COLUMNS, extrasaction="ignore")
                if needs_header:
                    writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
            self.metadata["measurement_count"] = int(self.metadata["measurement_count"]) + len(rows)
            self.metadata["sync"] = {"state": "pending_sync", "last_error": None}
            self.write_metadata()

    def plot_sweep(self, job_id: str, rows: list[dict]) -> list[str]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        points = [(row.get("voltage_v"), row.get("current_a")) for row in rows]
        points = [(float(v), float(i)) for v, i in points if v is not None and i is not None]
        if not points:
            return []
        plot_rel_base = Path(self.metadata["files"]["metadata"]).parent.parent / "plots"
        filenames = []
        with PLOT_LOCK:
            for log_scale in (False, True):
                suffix = "_logscaleY" if log_scale else ""
                rel = plot_rel_base / f"Sweep_{job_id[:8]}{suffix}.png"
                path = self.store.recovery_root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(8, 5))
                try:
                    ax.plot([point[0] for point in points],
                            [abs(point[1]) if log_scale else point[1] for point in points], ".-")
                    ax.set_xlabel("Voltage (V)")
                    ax.set_ylabel("|Current| (A)" if log_scale else "Current (A)")
                    if log_scale:
                        ax.set_yscale("log")
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2e}"))
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(path, dpi=200)
                    filenames.append(rel.as_posix())
                finally:
                    plt.close(fig)
        with self.lock:
            self.metadata["files"]["plots"].extend(filenames)
            self.metadata["sync"] = {"state": "pending_sync", "last_error": None}
            self.write_metadata()
        return filenames

    def sync(self) -> dict:
        with self.lock:
            files = self.metadata["files"]
            relative_paths = [files["data"], files["events"], *files["plots"]]
            try:
                for relative in relative_paths:
                    source = self.store.recovery_root / relative
                    if not source.is_file():
                        continue
                    destination = self.store.dropbox_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(destination.name + ".pending")
                    shutil.copy2(source, temporary)
                    replace_with_retry(temporary, destination)
                self.metadata["sync"] = {"state": "synced", "last_error": None,
                                         "synced_at": datetime.now(timezone.utc).isoformat()}
                self.write_metadata()
                destination = self.store.dropbox_root / files["metadata"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".pending")
                shutil.copy2(self.meta_path, temporary)
                replace_with_retry(temporary, destination)
            except OSError as exc:
                self.metadata["sync"] = {"state": "pending_sync", "last_error": str(exc)}
                self.write_metadata()
            return {"run_id": self.run_id, "state": self.metadata["sync"]["state"],
                    "local_path": str(self.data_path),
                    "dropbox_path": str(self.store.dropbox_root / files["data"]),
                    "error": self.metadata["sync"].get("last_error")}
