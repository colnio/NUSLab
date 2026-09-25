"""
Writing measurements, per-device metadata, and the campaign summary.

Two habits are deliberate:

* **Flush after every row.** A stress run can be killed at any moment -- by the
  operator, by a crash, by the device shorting. Whatever was measured up to that
  point must already be on disk. This follows ``cv_cf_recipe_ui.py:1793``.
* **Reject unknown columns.** ``csv.DictWriter`` can silently drop a field it
  does not recognise. In a destructive experiment a lost column cannot be
  re-measured, so a typo should fail loudly instead.

Files are ``.csv`` (as in ``CV_MAP/``) rather than the ``.data`` extension the
KeithleyGUI apps use, because these get opened directly in analysis notebooks.
"""

from __future__ import annotations

import csv
import json
import os
import os.path as op
import tempfile
from typing import Any, Dict, Optional, Sequence

from . import _paths  # noqa: F401  (puts KeithleyGUI on sys.path)
from KeithleyGUI.ui_helpers import ensure_directory

#: Columns for a C(f) or C(V) sweep.
MFIA_COLUMNS: Sequence[str] = (
    "timestamp_unix",
    "timestamp_iso",
    "sweep_variable",
    "sweep_value",
    "direction",
    "segment",
    "point_index",
    "set_bias_V",
    "set_amplitude_V",
    "set_frequency_Hz",
    "measured_frequency_Hz",
    "measured_drive_V",
    "model",
    "param0",
    "param1",
    "C_F",
    "C_per_area_F_per_um2",
    "R_Ohm",
    "Z_real_Ohm",
    "Z_imag_Ohm",
    "Z_abs_Ohm",
    "Z_phase_rad",
)

#: Columns for an RVS or CVS run.
STRESS_COLUMNS: Sequence[str] = (
    "timestamp_unix",
    "elapsed_s",
    "phase",
    "point_index",
    "set_V",
    "measured_V",
    "measured_I_A",
    "current_density_A_per_um2",
    "bd_flag",
)

#: One row per device -- the table the campaign is actually analysed from.
SUMMARY_COLUMNS: Sequence[str] = (
    "date",
    "sample",
    "crosspoint_um",
    "area_um2",
    "thickness_nm",
    "device_index",
    "stress_type",
    "polarity",
    "C_at_fref_F",
    "C_per_area_F_per_um2",
    "f_ref_Hz",
    "ramp_rate_Vps",
    "achieved_rate_Vps",
    # Only ever filled from an RVS ramp, so the column can be fitted directly.
    "V_BD_V",
    "E_BD_MV_per_cm",
    "t_BD_s",
    "V_stress_V",
    # The field a CVS device was held at -- the x-axis for a lifetime model.
    "E_stress_MV_per_cm",
    "i_threshold_A",
    "compliance_A",
    "bd_detected",
    "termination_reason",
    "started_iso",
    "finished_iso",
    "data_dir",
)


class MeasurementWriter:
    """Streaming CSV writer that flushes every row.

    Usable as a context manager::

        with MeasurementWriter(path, MFIA_COLUMNS) as writer:
            writer.write_row({...})
    """

    def __init__(self, path: str, columns: Sequence[str]):
        self.path = str(path)
        self.columns = list(columns)
        self.row_count = 0
        parent = op.dirname(self.path)
        if parent:
            ensure_directory(parent, "measurement output")
        try:
            # A measurement is destructive and cannot be repeated after the
            # device fails. Never truncate an existing file, even if a path
            # allocation bug or timestamp collision reaches this final layer.
            self._fh = open(self.path, "x", newline="", encoding="utf-8")
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite existing measurement file: {self.path}"
            ) from exc
        self._writer = csv.DictWriter(
            self._fh, fieldnames=self.columns, extrasaction="raise"
        )
        self._writer.writeheader()
        self._fh.flush()

    def write_row(self, row: Dict[str, Any]) -> None:
        unknown = set(row) - set(self.columns)
        if unknown:
            raise ValueError(
                f"Unknown column(s) {sorted(unknown)} for {op.basename(self.path)}."
            )
        self._writer.writerow(row)
        self._fh.flush()
        self.row_count += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "MeasurementWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SummaryWriter:
    """Append-only one-row-per-device table.

    Opens and closes per append rather than holding a handle, so the file is
    complete and consistent between devices no matter how the program exits.
    """

    def __init__(self, path: str, columns: Sequence[str] = SUMMARY_COLUMNS):
        self.path = str(path)
        self.columns = list(columns)
        self._validate_existing_header()

    def _validate_existing_header(self) -> None:
        if not op.isfile(self.path) or os.path.getsize(self.path) == 0:
            return
        try:
            with open(self.path, newline="", encoding="utf-8-sig") as fh:
                header = next(csv.reader(fh), [])
        except OSError as exc:
            raise RuntimeError(f"Could not read existing summary: {self.path}") from exc
        if header != self.columns:
            raise ValueError(
                "Existing summary header is incompatible; refusing to append to "
                f"{self.path}.\nExpected: {self.columns}\nFound: {header}"
            )

    def append(self, row: Dict[str, Any]) -> None:
        unknown = set(row) - set(self.columns)
        if unknown:
            raise ValueError(f"Unknown summary column(s) {sorted(unknown)}.")
        parent = op.dirname(self.path)
        if parent:
            ensure_directory(parent, "summary output")
        is_new = not op.isfile(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.columns, extrasaction="raise")
            if is_new:
                writer.writeheader()
            writer.writerow(row)
            fh.flush()


def write_device_meta(path: str, payload: Dict[str, Any]) -> str:
    return write_json_atomic(path, payload, label="device metadata")


def write_json_atomic(path: str, payload: Dict[str, Any], label: str = "metadata") -> str:
    """Durably replace a JSON file without exposing a partially-written file."""
    path = str(path)
    parent = op.dirname(path) or "."
    ensure_directory(parent, f"{label} output")
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=parent)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
        temporary = None
    except Exception as exc:
        raise RuntimeError(f"Failed to write {label} file:\n{path}\n\n{exc}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return path


def read_summary_rows(path: str):
    """Return summary rows plus non-fatal row warnings for campaign restoration."""
    if not op.isfile(path) or os.path.getsize(path) == 0:
        return [], []
    warnings = []
    try:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != list(SUMMARY_COLUMNS):
                raise ValueError(
                    f"Existing summary header is incompatible: {reader.fieldnames}"
                )
            return list(reader), warnings
    except Exception as exc:
        warnings.append(f"Could not restore campaign history from {path}: {exc}")
        return [], warnings


def breakdown_field_MV_per_cm(
    v_bd: Optional[float], thickness_nm: Optional[float]
) -> Optional[float]:
    """Breakdown field, or None when either input is missing.

    ``E = V / d``. With V in volts and d in nanometres, dividing by 10 converts
    the result to MV/cm.
    """
    if v_bd is None or thickness_nm is None or float(thickness_nm) <= 0:
        return None
    return abs(float(v_bd)) / float(thickness_nm) * 10.0
