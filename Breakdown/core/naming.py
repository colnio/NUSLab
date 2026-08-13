"""
Output paths, filenames, and device-index scoping.

The directory layout follows the convention already used across the repo
(``<outdir>/<date>/<sample>/...``, see ``FourProbeFET.py:86-95``) with two
crosspoint-specific levels inserted::

    <outdir>/<YYYY-MM-DD>/<sample>/
        <sample>_params_<ts>.json     full parameter snapshot, reloadable
        <sample>_summary.csv          one row per device, appended live
        <sample>_<ts>.log
        <size>um/
            dev001/
                dev001.meta.json
                data/   CF_/CV_/RVS_/CVS_ csv files
                plots/  png

The device index is *derived from the directory tree* rather than held in a
counter. Scoping it under ``<size>um`` is what makes it reset when the
crosspoint size changes, and it also means the index survives a restart of the
program mid-campaign.
"""

from __future__ import annotations

import os
import os.path as op
import re
from dataclasses import dataclass
from typing import List

from . import _paths  # noqa: F401  (puts KeithleyGUI on sys.path)
from KeithleyGUI.ui_helpers import ensure_directory

_SAFE_CHARS = ("-", "_", ".", "+")

#: Matches the device folders this module creates, and nothing else.
_DEVICE_DIR_RE = re.compile(r"^dev(\d+)$")


def sanitize_filename(text: str, fallback: str = "sample") -> str:
    """Reduce arbitrary user text to something safe as a path component.

    Same rule as ``cv_cf_recipe_ui.py:250``, plus a fallback so a name made
    entirely of separators cannot collapse into an empty path component.
    """
    out = []
    for ch in str(text):
        out.append(ch if (ch.isalnum() or ch in _SAFE_CHARS) else "_")
    cleaned = "".join(out).strip("_")
    return cleaned or fallback


def size_tag(crosspoint_um: float) -> str:
    """Directory tag for a crosspoint size: ``5.0 -> '5um'``, ``2.5 -> '2.5um'``."""
    value = float(crosspoint_um)
    if value == int(value):
        return f"{int(value)}um"
    return f"{value:g}um"


def device_tag(index: int) -> str:
    """Zero-padded device tag, widening past 999 rather than truncating."""
    return f"dev{int(index):03d}"


def measurement_stem(
    sample_name: str,
    crosspoint_um: float,
    index: int,
    kind: str,
    timestamp: str,
) -> str:
    """Filename stem carrying every fact needed to identify a measurement.

    Filenames are self-describing on purpose: these files get copied out of the
    tree into analysis notebooks, where the surrounding directories are lost.
    """
    return "_".join(
        (
            str(kind),
            sanitize_filename(sample_name),
            size_tag(crosspoint_um),
            device_tag(index),
            str(timestamp),
        )
    )


@dataclass
class DeviceDirs:
    device_dir: str
    data_dir: str
    plot_dir: str
    meta_file: str


class SamplePaths:
    """Path helper scoped to one output root, date, and sample."""

    def __init__(self, output_dir: str, date: str, sample_name: str):
        if not str(output_dir).strip():
            raise ValueError("No output folder selected.")
        self.output_dir = str(output_dir)
        self.date = str(date)
        self.sample_name = str(sample_name)
        self.sample_tag = sanitize_filename(sample_name)

    # -- directories --------------------------------------------------------

    @property
    def date_dir(self) -> str:
        return op.join(self.output_dir, self.date)

    @property
    def sample_dir(self) -> str:
        return op.join(self.date_dir, self.sample_tag)

    def size_dir(self, crosspoint_um: float) -> str:
        return op.join(self.sample_dir, size_tag(crosspoint_um))

    def device_dir(self, crosspoint_um: float, index: int) -> str:
        return op.join(self.size_dir(crosspoint_um), device_tag(index))

    # -- sample-level files -------------------------------------------------

    @property
    def summary_file(self) -> str:
        return op.join(self.sample_dir, f"{self.sample_tag}_summary.csv")

    def params_file(self, timestamp: str) -> str:
        return op.join(self.sample_dir, f"{self.sample_tag}_params_{timestamp}.json")

    def log_file(self, timestamp: str) -> str:
        return op.join(self.sample_dir, f"{self.sample_tag}_{timestamp}.log")

    # -- creation -----------------------------------------------------------

    def ensure_sample_dir(self) -> str:
        ensure_directory(self.date_dir, "date")
        return ensure_directory(self.sample_dir, f"sample '{self.sample_tag}'")

    def ensure_device_dirs(self, crosspoint_um: float, index: int) -> DeviceDirs:
        self.ensure_sample_dir()
        ensure_directory(self.size_dir(crosspoint_um), "crosspoint size")
        device_dir = ensure_directory(
            self.device_dir(crosspoint_um, index), device_tag(index)
        )
        return DeviceDirs(
            device_dir=device_dir,
            data_dir=ensure_directory(op.join(device_dir, "data"), "data"),
            plot_dir=ensure_directory(op.join(device_dir, "plots"), "plots"),
            meta_file=op.join(device_dir, f"{device_tag(index)}.meta.json"),
        )

    # -- indexing -----------------------------------------------------------

    def existing_device_indices(self, crosspoint_um: float) -> List[int]:
        directory = self.size_dir(crosspoint_um)
        try:
            entries = os.listdir(directory)
        except OSError:
            return []
        found = []
        for entry in entries:
            match = _DEVICE_DIR_RE.match(entry)
            if match and op.isdir(op.join(directory, entry)):
                found.append(int(match.group(1)))
        return sorted(found)

    def next_device_index(self, crosspoint_um: float) -> int:
        """Next free index for this sample and crosspoint size.

        Continues past the highest index rather than filling gaps: a deleted
        device folder usually means a bad probe landing that still has a row in
        the summary CSV, and reusing its number would overwrite that history.
        """
        existing = self.existing_device_indices(crosspoint_um)
        return (existing[-1] + 1) if existing else 1
