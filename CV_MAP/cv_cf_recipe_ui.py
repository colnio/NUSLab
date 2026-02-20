from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
import zhinst.core as zi
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

MODEL_INFO = {
    0: ("Rp || Cp", "Rp", "Cp"),
    1: ("Rs + Cs", "Rs", "Cs"),
    2: ("Rs + Ls", "Rs", "Ls"),
    3: ("G + B", "G", "B"),
    4: ("D + Cs", "D", "Cs"),
    5: ("Q + Cs", "Q", "Cs"),
    6: ("D + Ls", "D", "Ls"),
    7: ("Q + Ls", "Q", "Ls"),
    8: ("Rp || Lp", "Rp", "Lp"),
    9: ("D + Cp", "D", "Cp"),
    10: ("Dielectric", "P0", "P1"),
}

MODEL_TAGS = {
    0: "RpCp",
    1: "RsCs",
    2: "RsLs",
    3: "GB",
    4: "DCs",
    5: "QCs",
    6: "DLs",
    7: "QLs",
    8: "RpLp",
    9: "DCp",
    10: "Dielectric",
}

QUALITY_LABELS = {
    0: "High Speed",
    1: "Medium",
    2: "High Accuracy",
    3: "Very High Accuracy",
}

BLOCK_MODE_ITEMS = [
    ("BIAS", "bias"),
    ("AMPLITUDE", "amplitude"),
    ("FREQUENCY", "frequency"),
]

SPACING_ITEMS = [("Linear", "linear"), ("Log", "log")]

ORDER_ITEMS = [
    ("0 -> MAX -> MIN -> 0", "zero_max_min_zero"),
    ("0 -> MIN -> MAX -> 0", "zero_min_max_zero"),
]

INPUT_RANGE_MODE_ITEMS = [
    ("Manual", 0),
    ("Auto", 1),
    ("Zone", 2),
]

CURRENT_RANGE_GRID_A = [
    1e-12,
    1e-11,
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
]
IV_START_RANGE_A = 1e-11  # 10 pA

CSV_FIELDS = [
    "timestamp_unix",
    "timestamp_iso",
    "sample_timestamp",
    "sample_clockbase",
    "block_index",
    "block_name",
    "mode",
    "sweep_variable",
    "sweep_value",
    "point_index_block",
    "point_index_total",
    "set_bias_V",
    "set_amplitude_V",
    "set_frequency_Hz",
    "measured_drive_V",
    "measured_frequency_Hz",
    "model",
    "model_name",
    "param0_name",
    "param1_name",
    "param0",
    "param1",
    "C_F",
    "R_Ohm",
    "demod_R",
    "Z_real_Ohm",
    "Z_imag_Ohm",
    "Z_abs_Ohm",
    "Z_phase_rad",
    "imps_current_range_A",
    "currin_index",
    "currin_range_A",
    "currin_norm_min",
    "currin_norm_max",
    "currin_est_A",
    "currin_peak_A",
    "iv_autorange_mode",
]

PLAN_CSV_FIELDS = [
    "name",
    "mode",
    "min_value",
    "max_value",
    "spacing",
    "use_points",
    "step",
    "points",
    "order",
    "fixed_bias",
    "fixed_amplitude",
    "fixed_frequency",
    "settle_s",
]


@dataclass
class SweepBlock:
    name: str
    mode: str
    min_value: float
    max_value: float
    spacing: str
    use_points: bool
    step: float
    points: int
    order: str
    fixed_bias: float
    fixed_amplitude: float
    fixed_frequency: float
    settle_s: float


@dataclass
class RecipeConfig:
    host: str
    port: int
    device: str
    imps: int
    model: int
    quality: int
    auto_bw: bool
    inputrange_mode: int
    manual_current_range: float
    demod_order: int
    demod_timeconstant: float
    demod_rate: float
    demod_sinc: bool
    ramp_step: float
    ramp_wait: float
    outdir: str
    sample: str
    blocks: List[SweepBlock]


@dataclass
class StateSnapshot:
    int_nodes: Dict[str, int]
    double_nodes: Dict[str, float]


def build_model_items() -> List[Tuple[str, int]]:
    return [
        ("0  Rp || Cp", 0),
        ("1  Rs + Cs", 1),
        ("2  Rs + Ls", 2),
        ("3  G  B (admittance)", 3),
        ("4  D  Cs", 4),
        ("5  Q  Cs", 5),
        ("6  D  Ls", 6),
        ("7  Q  Ls", 7),
        ("8  Rp || Lp", 8),
        ("9  D  Cp", 9),
        ("10 Dielectric", 10),
    ]


def build_quality_items() -> List[Tuple[str, int]]:
    return [
        ("High Speed", 0),
        ("Medium", 1),
        ("High Accuracy", 2),
        ("Very High Accuracy", 3),
    ]


def sanitize_filename(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", ".", "+"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"cv_cf_recipe_{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger


def to_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def try_get_double(daq: zi.ziDAQServer, path: str) -> Optional[float]:
    try:
        return float(daq.getDouble(path))
    except Exception:
        return None


def try_get_int(daq: zi.ziDAQServer, path: str) -> Optional[int]:
    try:
        return int(daq.getInt(path))
    except Exception:
        return None


def try_set_double(daq: zi.ziDAQServer, paths: List[str], value: float) -> Optional[str]:
    for path in paths:
        try:
            daq.setDouble(path, float(value))
            return path
        except Exception:
            continue
    return None


def try_set_int(daq: zi.ziDAQServer, paths: List[str], value: int) -> Optional[str]:
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
    if math.isclose(current, target, rel_tol=0.0, abs_tol=1e-12):
        return [target]
    delta = target - current
    n_steps = max(1, int(abs(delta) / max(step, 1e-12)))
    return [float(v) for v in np.linspace(current, target, n_steps + 1)[1:]]


def ramp_set_double(
    daq: zi.ziDAQServer, path: str, target: float, step: float, wait_s: float
) -> None:
    try:
        current = daq.getDouble(path)
    except Exception:
        daq.setDouble(path, float(target))
        if wait_s > 0:
            time.sleep(wait_s)
        return
    for v in ramp_values(float(current), float(target), float(step)):
        daq.setDouble(path, float(v))
        if wait_s > 0:
            time.sleep(wait_s)


def find_mfia_device(host: str, port: int, preferred: Optional[str]) -> Tuple[str, str]:
    disc = zi.ziDiscovery()
    devices = disc.findAll()
    if not devices:
        raise RuntimeError("No devices found via discovery.")

    if preferred:
        dev = preferred.upper()
        info = disc.get(dev)
        if info.get("devicetype") != "MFIA":
            raise RuntimeError(f"Device {dev} is not an MFIA: {info}")
        return dev, info.get("connected", "")

    for dev in devices:
        info = disc.get(dev)
        if info.get("devicetype") == "MFIA":
            return dev, info.get("connected", "")
    raise RuntimeError("No MFIA device found.")


def set_drive_amplitude(daq: zi.ziDAQServer, dev: str, imps: int, value: float) -> List[str]:
    base = f"/{dev}/imps/{imps}"
    set_paths: List[str] = []

    candidates = [
        f"{base}/drive",
        f"{base}/drive/level",
        f"{base}/drive/amplitude",
        f"{base}/osc/amp",
        f"{base}/osc/amplitude",
    ]
    for p in candidates:
        if try_set_double(daq, [p], value):
            set_paths.append(p)

    sigout_index = None
    osc_index = None
    for p in (f"{base}/sigout", f"{base}/outputselect"):
        v = try_get_int(daq, p)
        if v is not None:
            sigout_index = v
            break
    for p in (f"{base}/oscselect", f"{base}/oscindex"):
        v = try_get_int(daq, p)
        if v is not None:
            osc_index = v
            break
    if sigout_index is not None and osc_index is not None:
        amp_path = f"/{dev}/sigouts/{sigout_index}/amplitudes/{osc_index}"
        en_path = f"/{dev}/sigouts/{sigout_index}/enables/{osc_index}"
        if try_set_double(daq, [amp_path], value):
            set_paths.append(amp_path)
        if try_set_int(daq, [en_path], 1):
            set_paths.append(en_path)
    return set_paths


def compute_model_values(
    model: int, param0: float, param1: float, freq_hz: Optional[float]
) -> Tuple[float, float]:
    cap = float("nan")
    res = float("nan")

    if model in (0, 1, 4, 5, 9):
        cap = float(param1)
    elif model == 3 and freq_hz and freq_hz > 0:
        try:
            cap = float(param1) / (2.0 * math.pi * float(freq_hz))
        except Exception:
            cap = float("nan")

    if model in (0, 1, 2, 8):
        res = float(param0)
    elif model == 3:
        try:
            p0 = float(param0)
            res = (1.0 / p0) if p0 != 0 else float("nan")
        except Exception:
            res = float("nan")

    return cap, res


def extract_demod_r(sample: Dict, demod_sample: Optional[Dict], z_abs: float) -> float:
    # Prefer explicit demodulator sample fields first.
    sources = [demod_sample or {}, sample]
    for src in sources:
        for key in ("r", "R", "demodR", "demod_r"):
            val = to_float(src.get(key))
            if np.isfinite(val):
                return val

    # Fallback to demod vector magnitude.
    for src in sources:
        x = to_float(src.get("x"))
        y = to_float(src.get("y"))
        if np.isfinite(x) and np.isfinite(y):
            return float(math.hypot(x, y))

    if np.isfinite(z_abs):
        return float(z_abs)
    return float("nan")


def build_segment_values(start: float, stop: float, block: SweepBlock) -> List[float]:
    if math.isclose(start, stop, rel_tol=0.0, abs_tol=1e-15):
        return [float(start)]

    if block.use_points:
        n_points = max(2, int(block.points))
    else:
        if block.step <= 0:
            raise ValueError("Step must be > 0.")
        n_points = max(2, int(math.ceil(abs(stop - start) / block.step)) + 1)

    if block.spacing == "log":
        if start == 0 and stop != 0:
            return build_log_from_zero(float(stop), n_points, block)
        if start != 0 and stop == 0:
            return list(reversed(build_log_from_zero(float(start), n_points, block)))
        if (start > 0) != (stop > 0):
            return [float(v) for v in np.linspace(start, stop, n_points)]

        sign = 1.0 if start > 0 else -1.0
        a = abs(start)
        b = abs(stop)
        vals = np.logspace(np.log10(a), np.log10(b), n_points)
        return [float(sign * v) for v in vals]

    return [float(v) for v in np.linspace(start, stop, n_points)]


def build_log_from_zero(stop: float, n_points: int, block: SweepBlock) -> List[float]:
    if n_points <= 1:
        return [float(stop)]
    if math.isclose(stop, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return [0.0]
    if n_points == 2:
        return [0.0, float(stop)]

    sign = 1.0 if stop > 0 else -1.0
    stop_abs = abs(stop)

    hint = min(abs(block.min_value), abs(block.max_value))
    if not np.isfinite(hint) or hint <= 0:
        hint = max(stop_abs / (10.0 ** max(1, n_points - 2)), 1e-12)

    if hint >= stop_abs:
        return [0.0, float(stop)]

    vals = np.logspace(np.log10(hint), np.log10(stop_abs), n_points - 1)
    return [0.0] + [float(sign * v) for v in vals]


def append_no_duplicate(values: List[float], segment: List[float]) -> None:
    for v in segment:
        if not values or not math.isclose(values[-1], v, rel_tol=0.0, abs_tol=1e-12):
            values.append(float(v))


def build_block_path(block: SweepBlock) -> List[float]:
    v_min = min(block.min_value, block.max_value)
    v_max = max(block.min_value, block.max_value)
    if block.order == "zero_max_min_zero":
        segments = [(0.0, v_max), (v_max, v_min), (v_min, 0.0)]
    else:
        segments = [(0.0, v_min), (v_min, v_max), (v_max, 0.0)]

    out: List[float] = []
    for a, b in segments:
        append_no_duplicate(out, build_segment_values(float(a), float(b), block))
    return out


def estimate_block_points(block: SweepBlock) -> int:
    try:
        return len(build_block_path(block))
    except Exception:
        return 0


def capture_state(daq: zi.ziDAQServer, dev: str, imps: int) -> StateSnapshot:
    base = f"/{dev}/imps/{imps}"
    int_nodes: Dict[str, int] = {}
    double_nodes: Dict[str, float] = {}

    int_paths = [
        f"{base}/enable",
        f"{base}/model",
        f"{base}/bias/enable",
        f"{base}/auto/bw",
        f"{base}/auto/inputrange",
        f"{base}/demod/order",
        f"{base}/demod/sinc",
    ]
    double_paths = [
        f"{base}/bias/value",
        f"{base}/current/range",
        f"{base}/demod/timeconstant",
        f"{base}/demod/rate",
        f"{base}/freq",
        f"{base}/drive",
    ]

    for p in int_paths:
        v = try_get_int(daq, p)
        if v is not None:
            int_nodes[p] = v
    for p in double_paths:
        v = try_get_double(daq, p)
        if v is not None:
            double_nodes[p] = v

    for p in (f"{base}/precision", f"{base}/accuracy", f"{base}/quality"):
        v = try_get_int(daq, p)
        if v is not None:
            int_nodes[p] = v
            break

    sigout = None
    osc = None
    for p in (f"{base}/sigout", f"{base}/outputselect"):
        v = try_get_int(daq, p)
        if v is not None:
            sigout = v
            int_nodes[p] = v
            break
    for p in (f"{base}/oscselect", f"{base}/oscindex"):
        v = try_get_int(daq, p)
        if v is not None:
            osc = v
            int_nodes[p] = v
            break
    if sigout is not None and osc is not None:
        amp_path = f"/{dev}/sigouts/{sigout}/amplitudes/{osc}"
        en_path = f"/{dev}/sigouts/{sigout}/enables/{osc}"
        amp = try_get_double(daq, amp_path)
        en = try_get_int(daq, en_path)
        if amp is not None:
            double_nodes[amp_path] = amp
        if en is not None:
            int_nodes[en_path] = en

    input_select = try_get_int(daq, f"{base}/current/inputselect")
    if input_select is not None and 0 <= input_select <= 3:
        cbase = f"/{dev}/currins/{input_select}"
        for p in (f"{cbase}/on", f"{cbase}/autorange"):
            v = try_get_int(daq, p)
            if v is not None:
                int_nodes[p] = v
        for p in (f"{cbase}/range", f"{cbase}/scaling"):
            v = try_get_double(daq, p)
            if v is not None:
                double_nodes[p] = v

    return StateSnapshot(int_nodes=int_nodes, double_nodes=double_nodes)


def restore_state(daq: zi.ziDAQServer, snapshot: StateSnapshot) -> None:
    for path, value in snapshot.int_nodes.items():
        try:
            daq.setInt(path, int(value))
        except Exception:
            continue
    for path, value in snapshot.double_nodes.items():
        try:
            daq.setDouble(path, float(value))
        except Exception:
            continue


def get_latest_sample(daq: zi.ziDAQServer, path: str, timeout_s: float = 1.0) -> Dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        data = daq.poll(0.1, 10, 0, True)
        if path in data and data[path]:
            sample = data[path]
            out: Dict[str, object] = {}
            for k, v in sample.items():
                try:
                    out[k] = v[-1]
                except Exception:
                    out[k] = v
            return out
    raise TimeoutError(f"No data for {path} within {timeout_s:.3f}s")


def combo_value(combo: QComboBox) -> object:
    data = combo.currentData()
    if data is not None:
        return data
    return combo.currentText()


def parse_bool(text: object) -> bool:
    t = str(text).strip().lower()
    if t in ("1", "true", "yes", "y", "on"):
        return True
    if t in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {text}")


def normalize_order(text: object) -> str:
    t = str(text).strip().lower()
    if t in ("zero_max_min_zero", "0->max->min->0", "zmaxminz"):
        return "zero_max_min_zero"
    if t in ("zero_min_max_zero", "0->min->max->0", "zminmaxz"):
        return "zero_min_max_zero"
    raise ValueError(f"Invalid order: {text}")


class SweepWorker(QObject):
    log = pyqtSignal(str)
    run_paths = pyqtSignal(str, str, str)
    block_started = pyqtSignal(int, str, int)
    block_finished = pyqtSignal(int, str)
    progress = pyqtSignal(int, int, int, int)
    point = pyqtSignal(object)
    finished = pyqtSignal(int)

    def __init__(self, cfg: RecipeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._stop = False

    @pyqtSlot()
    def request_stop(self) -> None:
        self._stop = True

    def _log(self, logger: Optional[logging.Logger], message: str) -> None:
        if logger is not None:
            logger.info(message)
        self.log.emit(message)

    def _resolve_currin_index(
        self, daq: zi.ziDAQServer, dev: str, imps: int
    ) -> Optional[int]:
        base = f"/{dev}/imps/{imps}"
        sel = try_get_int(daq, f"{base}/current/inputselect")
        if sel is None:
            return 0
        if 0 <= sel <= 3:
            return int(sel)
        return None

    def _resolve_demod_index(self, daq: zi.ziDAQServer, dev: str, imps: int) -> int:
        base = f"/{dev}/imps/{imps}"
        for p in (
            f"{base}/demod",
            f"{base}/demodselect",
            f"{base}/demodindex",
        ):
            v = try_get_int(daq, p)
            if v is not None and v >= 0:
                return int(v)
        return 0

    def _set_common_settings(self, daq: zi.ziDAQServer, dev: str, logger: logging.Logger) -> None:
        base = f"/{dev}/imps/{self.cfg.imps}"
        daq.setInt(f"{base}/enable", 1)
        daq.setInt(f"{base}/model", int(self.cfg.model))
        daq.setInt(f"{base}/bias/enable", 1)

        if try_set_int(daq, [f"{base}/auto/bw"], int(self.cfg.auto_bw)) is None:
            self._log(logger, "Warning: failed to configure /auto/bw")
        if try_set_int(daq, [f"{base}/auto/inputrange"], int(self.cfg.inputrange_mode)) is None:
            self._log(logger, "Warning: failed to configure /auto/inputrange")
        if self.cfg.inputrange_mode == 0:
            try_set_double(daq, [f"{base}/current/range"], float(self.cfg.manual_current_range))

        if try_set_int(
            daq, [f"{base}/precision", f"{base}/accuracy", f"{base}/quality"], int(self.cfg.quality)
        ) is None:
            self._log(logger, "Warning: failed to configure quality node")

        try_set_int(daq, [f"{base}/demod/order"], int(self.cfg.demod_order))
        try_set_double(daq, [f"{base}/demod/timeconstant"], float(self.cfg.demod_timeconstant))
        try_set_double(daq, [f"{base}/demod/rate"], float(self.cfg.demod_rate))
        try_set_int(daq, [f"{base}/demod/sinc"], int(self.cfg.demod_sinc))

    def _set_setpoint(
        self,
        daq: zi.ziDAQServer,
        dev: str,
        imps: int,
        kind: str,
        target: float,
        last_value: Optional[float],
    ) -> Optional[float]:
        if last_value is not None and math.isclose(last_value, target, rel_tol=0.0, abs_tol=1e-15):
            return last_value

        base = f"/{dev}/imps/{imps}"
        if kind == "bias":
            ramp_set_double(
                daq, f"{base}/bias/value", float(target), self.cfg.ramp_step, self.cfg.ramp_wait
            )
            return float(target)
        if kind == "amplitude":
            set_paths = set_drive_amplitude(daq, dev, imps, float(target))
            if not set_paths:
                self.log.emit("Warning: no known drive node accepted amplitude write.")
            return float(target)
        if kind == "frequency":
            try_set_double(daq, [f"{base}/freq", f"{base}/osc/freq"], float(target))
            return float(target)
        return last_value

    def _iv_setup(
        self, daq: zi.ziDAQServer, dev: str, imps: int, logger: logging.Logger
    ) -> Dict[str, object]:
        base = f"/{dev}/imps/{imps}"
        curr_idx = self._resolve_currin_index(daq, dev, imps)
        mode = "manual_ladder"

        # Disable auto-range and drive range explicitly in IV mode.
        try_set_int(daq, [f"{base}/auto/inputrange"], 0)
        if curr_idx is not None:
            cbase = f"/{dev}/currins/{curr_idx}"
            try_set_int(daq, [f"{cbase}/on"], 1)
            try_set_int(daq, [f"{cbase}/autorange"], 0)

        ranges = self._detect_supported_ranges(daq, dev, imps, curr_idx)
        if not ranges:
            ranges = list(CURRENT_RANGE_GRID_A)
            mode = "manual_ladder_fallback"

        range_idx = self._select_iv_start_index(ranges, IV_START_RANGE_A)
        target = float(ranges[range_idx])
        self._set_current_range(daq, dev, imps, curr_idx, target)

        exact = any(
            abs(float(r) - IV_START_RANGE_A) <= max(1e-15, 0.02 * IV_START_RANGE_A)
            for r in ranges
        )
        if not exact:
            self._log(
                logger,
                f"Warning: exact 10 pA range not detected; starting from nearest supported {target:.3e} A",
            )

        self._log(
            logger,
            f"IV range mode: {mode}, supported ranges={len(ranges)}, start={target:.3e} A (target 1.000e-11 A)",
        )
        return {
            "mode": mode,
            "curr_idx": curr_idx,
            "ranges": ranges,
            "range_index": range_idx,
        }

    def _set_current_range(
        self, daq: zi.ziDAQServer, dev: str, imps: int, curr_idx: Optional[int], value_a: float
    ) -> None:
        base = f"/{dev}/imps/{imps}"
        try_set_double(daq, [f"{base}/current/range"], float(value_a))
        if curr_idx is not None:
            try_set_double(daq, [f"/{dev}/currins/{curr_idx}/range"], float(value_a))

    def _detect_supported_ranges(
        self, daq: zi.ziDAQServer, dev: str, imps: int, curr_idx: Optional[int]
    ) -> List[float]:
        original = self._read_range_state(daq, dev, imps, curr_idx)
        found: List[float] = []

        for cand in CURRENT_RANGE_GRID_A:
            self._set_current_range(daq, dev, imps, curr_idx, float(cand))
            time.sleep(0.01)
            st = self._read_range_state(daq, dev, imps, curr_idx)
            actual = st[0] if np.isfinite(st[0]) else st[1]
            if not np.isfinite(actual):
                continue
            if not any(abs(actual - x) <= max(1e-15, 0.01 * max(abs(actual), abs(x))) for x in found):
                found.append(float(actual))

        if np.isfinite(original[0]) or np.isfinite(original[1]):
            restore_val = original[0] if np.isfinite(original[0]) else original[1]
            self._set_current_range(daq, dev, imps, curr_idx, float(restore_val))

        found = sorted(found)
        return found

    def _select_iv_start_index(self, ranges: List[float], preferred_a: float = IV_START_RANGE_A) -> int:
        if not ranges:
            return 0
        # Start from 10 pA if available; otherwise use the nearest supported range above it.
        for i, r in enumerate(ranges):
            if float(r) >= float(preferred_a):
                return i
        # If all available ranges are below preferred, use the largest available.
        return len(ranges) - 1

    def _iv_manual_range_step(
        self,
        daq: zi.ziDAQServer,
        dev: str,
        imps: int,
        iv_state: Dict[str, object],
        est_current_a: float,
        logger: Optional[logging.Logger] = None,
    ) -> bool:
        ranges = iv_state.get("ranges")
        if not isinstance(ranges, list) or len(ranges) == 0:
            return False
        if not np.isfinite(est_current_a):
            return False

        i_abs = abs(float(est_current_a))
        curr_idx = int(iv_state.get("range_index", len(ranges) - 1))
        curr_idx = max(0, min(curr_idx, len(ranges) - 1))
        curr_range = float(ranges[curr_idx])
        if not np.isfinite(curr_range) or curr_range <= 0:
            return False

        # Keep signal around ~65% of range for sensitivity without clipping.
        target_fraction = 0.65
        need = max(i_abs / target_fraction, 1e-15)
        new_idx = 0
        while new_idx < len(ranges) - 1 and float(ranges[new_idx]) < need:
            new_idx += 1

        hi = 0.90 * curr_range
        lo = 0.25 * curr_range
        if new_idx == curr_idx:
            return False
        if i_abs <= hi and i_abs >= lo:
            # Inside hysteresis band; avoid unnecessary switching.
            return False

        new_range = float(ranges[new_idx])
        currin_idx = iv_state.get("curr_idx")
        if isinstance(currin_idx, int):
            currin_use = currin_idx
        else:
            currin_use = None
        self._set_current_range(daq, dev, imps, currin_use, new_range)
        iv_state["range_index"] = new_idx

        if logger is not None:
            self._log(logger, f"IV range -> {new_range:.3e} A (|I|={i_abs:.3e} A)")
        return True

    def _iv_autorange_step(
        self,
        daq: zi.ziDAQServer,
        dev: str,
        imps: int,
        iv_state: Dict[str, object],
        curr_min: float,
        curr_max: float,
        curr_range: float,
    ) -> None:
        curr_idx = iv_state.get("curr_idx")
        if curr_idx is not None:
            try_set_int(daq, [f"/{dev}/currins/{curr_idx}/rangestep/trigger"], 1)

        if "manual_fallback" not in str(iv_state.get("mode", "")):
            return
        if not np.isfinite(curr_min) or not np.isfinite(curr_max) or not np.isfinite(curr_range):
            return

        abs_norm = max(abs(curr_min), abs(curr_max))
        idx = int(np.argmin(np.abs(np.array(CURRENT_RANGE_GRID_A) - curr_range)))
        if abs_norm > 0.9 and idx < len(CURRENT_RANGE_GRID_A) - 1:
            idx += 1
        elif abs_norm < 0.15 and idx > 0:
            idx -= 1
        else:
            return

        target = float(CURRENT_RANGE_GRID_A[idx])
        base = f"/{dev}/imps/{imps}"
        try_set_double(daq, [f"{base}/current/range"], target)
        if curr_idx is not None:
            try_set_double(daq, [f"/{dev}/currins/{curr_idx}/range"], target)

    def _read_range_state(
        self, daq: zi.ziDAQServer, dev: str, imps: int, curr_idx: Optional[int]
    ) -> Tuple[float, float]:
        base = f"/{dev}/imps/{imps}"
        imps_range = to_float(try_get_double(daq, f"{base}/current/range"))
        currin_range = float("nan")
        if curr_idx is not None:
            currin_range = to_float(try_get_double(daq, f"/{dev}/currins/{curr_idx}/range"))
        return imps_range, currin_range

    def _range_states_close(self, a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        def close_val(x: float, y: float) -> bool:
            x_ok = np.isfinite(x)
            y_ok = np.isfinite(y)
            if x_ok and y_ok:
                tol = max(1e-15, 0.02 * max(abs(x), abs(y)))
                return abs(x - y) <= tol
            return (not x_ok) and (not y_ok)

        return close_val(a[0], b[0]) and close_val(a[1], b[1])

    def _acquire_stable_sample(
        self,
        daq: zi.ziDAQServer,
        sample_path: str,
        dev: str,
        imps: int,
        curr_idx: Optional[int],
        settle_s: float,
        max_attempts: int = 4,
    ) -> Tuple[Dict, Tuple[float, float], int]:
        timeout_s = max(1.0, settle_s * 5.0)
        wait_s = max(0.02, min(0.25, settle_s if settle_s > 0 else 0.05))

        prev_state: Optional[Tuple[float, float]] = None
        sample: Dict = {}
        state = (float("nan"), float("nan"))

        for attempt in range(1, max_attempts + 1):
            sample = get_latest_sample(daq, sample_path, timeout_s=timeout_s)
            state = self._read_range_state(daq, dev, imps, curr_idx)
            if prev_state is not None and self._range_states_close(prev_state, state):
                return sample, state, attempt
            prev_state = state
            if attempt < max_attempts:
                time.sleep(wait_s)

        return sample, state, max_attempts

    @pyqtSlot()
    def run(self) -> None:
        rc = 0
        logger: Optional[logging.Logger] = None
        daq = None
        dev = None
        sample_path = None
        demod_path = None
        snapshot = None

        try:
            date = str(dt.date.today())
            now_tag = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            sample_name = self.cfg.sample.strip() or "sample"
            sample_tag = sanitize_filename(sample_name)
            sample_dir = Path(self.cfg.outdir) / date / sample_name
            run_dir = sample_dir / f"{sample_tag}_run_{now_tag}"
            data_dir = run_dir / "data"
            plot_dir = run_dir / "plots"
            data_dir.mkdir(parents=True, exist_ok=True)
            plot_dir.mkdir(parents=True, exist_ok=True)
            self.run_paths.emit(sample_tag, str(run_dir), str(plot_dir))

            log_path = str(run_dir / f"{sample_tag}_run_{now_tag}.log")
            logger = setup_logger(log_path)
            self._log(logger, f"Run folder: {run_dir}")

            recipe_json = run_dir / f"{sample_tag}_plan_{now_tag}.json"
            payload = {
                "created": dt.datetime.now().isoformat(timespec="seconds"),
                "config": {
                    "host": self.cfg.host,
                    "port": self.cfg.port,
                    "device": self.cfg.device,
                    "imps": self.cfg.imps,
                    "model": self.cfg.model,
                    "quality": self.cfg.quality,
                    "auto_bw": self.cfg.auto_bw,
                    "inputrange_mode": self.cfg.inputrange_mode,
                    "manual_current_range": self.cfg.manual_current_range,
                    "demod_order": self.cfg.demod_order,
                    "demod_timeconstant": self.cfg.demod_timeconstant,
                    "demod_rate": self.cfg.demod_rate,
                    "demod_sinc": self.cfg.demod_sinc,
                    "ramp_step": self.cfg.ramp_step,
                    "ramp_wait": self.cfg.ramp_wait,
                },
                "blocks": [asdict(b) for b in self.cfg.blocks],
            }
            recipe_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._log(logger, f"Saved recipe: {recipe_json}")

            daq = zi.ziDAQServer(self.cfg.host, self.cfg.port, 6)
            dev_id, interface = find_mfia_device(
                self.cfg.host, self.cfg.port, self.cfg.device if self.cfg.device else None
            )
            dev = dev_id.lower()
            if not interface:
                interface = "PCIe"
            daq.connectDevice(dev, interface)
            self._log(logger, f"Connected to {dev} via {interface}")

            base = f"/{dev}/imps/{self.cfg.imps}"
            sample_path = f"{base}/sample"
            snapshot = capture_state(daq, dev, self.cfg.imps)
            demod_idx = self._resolve_demod_index(daq, dev, self.cfg.imps)
            demod_path = f"/{dev}/demods/{demod_idx}/sample"

            self._set_common_settings(daq, dev, logger)
            daq.subscribe(sample_path)
            try:
                daq.subscribe(demod_path)
                self._log(logger, f"Subscribed demod sample: {demod_path}")
            except Exception as exc:
                demod_path = None
                self._log(logger, f"Warning: could not subscribe demod sample node: {exc}")
            daq.sync()
            time.sleep(0.1)

            overall_done = 0
            last_bias = None
            last_amp = None
            last_freq = None
            n_blocks = len(self.cfg.blocks)
            demod_warned = False

            for block_idx, block in enumerate(self.cfg.blocks, start=1):
                if self._stop:
                    break

                values = build_block_path(block)
                block_points = len(values)
                if block_points == 0:
                    continue

                self.block_started.emit(block_idx, block.name, block_points)
                self._log(
                    logger,
                    f"Block {block_idx}/{n_blocks}: {block.name} mode={block.mode}, points={block_points}",
                )

                iv_mode = block.mode == "bias" and math.isclose(
                    block.fixed_amplitude, 0.0, rel_tol=0.0, abs_tol=1e-12
                )
                iv_state = {"mode": "off", "curr_idx": None}
                if iv_mode:
                    iv_state = self._iv_setup(daq, dev, self.cfg.imps, logger)
                else:
                    try_set_int(daq, [f"{base}/auto/inputrange"], int(self.cfg.inputrange_mode))
                meas_curr_idx = self._resolve_currin_index(daq, dev, self.cfg.imps)
                # Heavy stabilization is kept only for IV mode.
                use_range_stabilization = iv_mode
                # For non-IV C(V)/C(F) with auto/zone input range, use a light guard:
                # if range switched, discard first sample and take one settled reread.
                use_range_guard = (not iv_mode) and (self.cfg.inputrange_mode != 0)
                prev_point_range_state: Optional[Tuple[float, float]] = None

                order_tag = "zmaxminz" if block.order == "zero_max_min_zero" else "zminmaxz"
                block_file = data_dir / (
                    f"{sample_tag}_{sanitize_filename(block.name)}_"
                    f"block{block_idx:02d}_{block.mode}_{block.spacing}_{order_tag}_{now_tag}.csv"
                )

                with block_file.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                    writer.writeheader()

                    prev_curr_min = float("nan")
                    prev_curr_max = float("nan")
                    prev_curr_range = float("nan")
                    prev_current_est = float("nan")
                    range_stabilize_warned = False
                    range_guard_warned = False

                    for idx, sweep_val in enumerate(values, start=1):
                        if self._stop:
                            break

                        bias = block.fixed_bias
                        amp = block.fixed_amplitude
                        freq = block.fixed_frequency
                        if block.mode == "bias":
                            bias = sweep_val
                        elif block.mode == "amplitude":
                            amp = sweep_val
                        else:
                            freq = sweep_val

                        if freq < 0:
                            raise ValueError(f"Frequency cannot be negative: {freq}")
                        if amp < 0:
                            amp = abs(amp)

                        last_freq = self._set_setpoint(
                            daq, dev, self.cfg.imps, "frequency", float(freq), last_freq
                        )
                        last_amp = self._set_setpoint(
                            daq, dev, self.cfg.imps, "amplitude", float(amp), last_amp
                        )
                        last_bias = self._set_setpoint(
                            daq, dev, self.cfg.imps, "bias", float(bias), last_bias
                        )

                        if block.settle_s > 0:
                            time.sleep(block.settle_s)

                        if iv_mode:
                            changed = self._iv_manual_range_step(
                                daq,
                                dev,
                                self.cfg.imps,
                                iv_state,
                                prev_current_est,
                                logger,
                            )
                            if changed:
                                time.sleep(max(0.03, block.settle_s))

                        if use_range_stabilization:
                            sample, range_state, attempts = self._acquire_stable_sample(
                                daq=daq,
                                sample_path=sample_path,
                                dev=dev,
                                imps=self.cfg.imps,
                                curr_idx=meas_curr_idx,
                                settle_s=block.settle_s,
                                max_attempts=4,
                            )
                            if attempts >= 3 and not range_stabilize_warned:
                                self._log(
                                    logger,
                                    "Range switching is active; adding extra settling to stabilize traces.",
                                )
                                range_stabilize_warned = True
                        else:
                            sample = get_latest_sample(
                                daq, sample_path, timeout_s=max(1.0, block.settle_s * 5.0)
                            )
                            range_state = self._read_range_state(
                                daq, dev, self.cfg.imps, meas_curr_idx
                            )

                        if (
                            use_range_guard
                            and prev_point_range_state is not None
                            and (not self._range_states_close(prev_point_range_state, range_state))
                        ):
                            guard_wait = max(0.015, min(0.08, 0.5 * max(0.0, block.settle_s)))
                            if guard_wait > 0:
                                time.sleep(guard_wait)
                            sample = get_latest_sample(
                                daq, sample_path, timeout_s=max(1.0, block.settle_s * 5.0)
                            )
                            range_state = self._read_range_state(
                                daq, dev, self.cfg.imps, meas_curr_idx
                            )
                            if not range_guard_warned:
                                self._log(
                                    logger,
                                    "Range changed in C(V)/C(F): using one-shot settle guard to suppress transients.",
                                )
                                range_guard_warned = True

                        # If the selected current range changed versus previous point,
                        # wait and re-read once so demod output is settled on the new range.
                        if (
                            use_range_stabilization
                            and prev_point_range_state is not None
                            and (not self._range_states_close(prev_point_range_state, range_state))
                        ):
                            extra_wait = max(0.03, block.settle_s)
                            time.sleep(extra_wait)
                            sample, range_state, _ = self._acquire_stable_sample(
                                daq=daq,
                                sample_path=sample_path,
                                dev=dev,
                                imps=self.cfg.imps,
                                curr_idx=meas_curr_idx,
                                settle_s=block.settle_s + extra_wait,
                                max_attempts=4,
                            )
                        prev_point_range_state = range_state

                        demod_sample = None
                        if demod_path is not None:
                            try:
                                demod_sample = get_latest_sample(
                                    daq, demod_path, timeout_s=max(0.3, block.settle_s * 2.0)
                                )
                            except Exception:
                                if not demod_warned:
                                    self._log(
                                        logger,
                                        "Warning: demod sample read timed out; falling back to impedance stream.",
                                    )
                                    demod_warned = True

                        param0 = to_float(sample.get("param0"))
                        param1 = to_float(sample.get("param1"))
                        meas_freq = to_float(sample.get("frequency"), default=freq)
                        meas_drive = to_float(sample.get("drive"), default=amp)

                        z_val = sample.get("z", complex(float("nan"), float("nan")))
                        try:
                            z_real = float(np.real(z_val))
                            z_imag = float(np.imag(z_val))
                            z_abs = float(np.abs(z_val))
                            z_phase = float(np.angle(z_val))
                        except Exception:
                            z_real = float("nan")
                            z_imag = float("nan")
                            z_abs = float("nan")
                            z_phase = float("nan")

                        cap_f, res_ohm = compute_model_values(self.cfg.model, param0, param1, meas_freq)
                        demod_r = extract_demod_r(sample, demod_sample, z_abs)
                        model_name, p0_name, p1_name = MODEL_INFO.get(
                            self.cfg.model, ("Unknown", "param0", "param1")
                        )

                        curr_idx = meas_curr_idx
                        currin_range = range_state[1]
                        currin_min = float("nan")
                        currin_max = float("nan")
                        currin_est = float("nan")
                        currin_peak = float("nan")

                        if curr_idx is not None:
                            cbase = f"/{dev}/currins/{curr_idx}"
                            currin_min = to_float(try_get_double(daq, f"{cbase}/min"))
                            currin_max = to_float(try_get_double(daq, f"{cbase}/max"))
                            if np.isfinite(currin_range) and np.isfinite(currin_min) and np.isfinite(currin_max):
                                currin_est = 0.5 * (currin_min + currin_max) * currin_range
                                currin_peak = max(abs(currin_min), abs(currin_max)) * currin_range

                        imps_curr_range = range_state[0]
                        prev_curr_min = currin_min
                        prev_curr_max = currin_max
                        prev_curr_range = currin_range if np.isfinite(currin_range) else imps_curr_range
                        if np.isfinite(demod_r):
                            prev_current_est = abs(float(demod_r))
                        elif np.isfinite(currin_peak):
                            prev_current_est = abs(float(currin_peak))
                        elif np.isfinite(currin_est):
                            prev_current_est = abs(float(currin_est))
                        else:
                            prev_current_est = float("nan")

                        ts = time.time()
                        row = {
                            "timestamp_unix": ts,
                            "timestamp_iso": dt.datetime.fromtimestamp(ts).isoformat(
                                timespec="milliseconds"
                            ),
                            "sample_timestamp": sample.get("timestamp", ""),
                            "sample_clockbase": sample.get("clockbase", ""),
                            "block_index": block_idx,
                            "block_name": block.name,
                            "mode": block.mode,
                            "sweep_variable": block.mode,
                            "sweep_value": float(sweep_val),
                            "point_index_block": idx,
                            "point_index_total": overall_done + 1,
                            "set_bias_V": float(bias),
                            "set_amplitude_V": float(amp),
                            "set_frequency_Hz": float(freq),
                            "measured_drive_V": meas_drive,
                            "measured_frequency_Hz": meas_freq,
                            "model": int(self.cfg.model),
                            "model_name": model_name,
                            "param0_name": p0_name,
                            "param1_name": p1_name,
                            "param0": param0,
                            "param1": param1,
                            "C_F": cap_f,
                            "R_Ohm": res_ohm,
                            "demod_R": demod_r,
                            "Z_real_Ohm": z_real,
                            "Z_imag_Ohm": z_imag,
                            "Z_abs_Ohm": z_abs,
                            "Z_phase_rad": z_phase,
                            "imps_current_range_A": imps_curr_range,
                            "currin_index": curr_idx if curr_idx is not None else "",
                            "currin_range_A": currin_range,
                            "currin_norm_min": currin_min,
                            "currin_norm_max": currin_max,
                            "currin_est_A": currin_est,
                            "currin_peak_A": currin_peak,
                            "iv_autorange_mode": str(iv_state.get("mode", "off")),
                        }
                        writer.writerow(row)
                        f.flush()

                        overall_done += 1
                        self.progress.emit(block_idx, idx, block_points, overall_done)
                        self.point.emit(row)

                self._log(logger, f"Saved block data: {block_file}")
                self.block_finished.emit(block_idx, block.name)

            self._log(logger, "Sweep finished.")

        except Exception as exc:
            rc = 1
            if logger is not None:
                logger.exception("Fatal error")
            self.log.emit(f"Fatal error: {exc}")
        finally:
            try:
                if daq is not None and sample_path:
                    daq.unsubscribe(sample_path)
            except Exception:
                pass
            try:
                if daq is not None and demod_path:
                    daq.unsubscribe(demod_path)
            except Exception:
                pass
            try:
                if daq is not None and dev is not None:
                    base = f"/{dev}/imps/{self.cfg.imps}"
                    daq.setDouble(f"{base}/bias/value", 0.0)
                    daq.setInt(f"{base}/bias/enable", 0)
                    set_drive_amplitude(daq, dev, self.cfg.imps, 0.0)
            except Exception:
                pass
            try:
                if daq is not None and snapshot is not None:
                    restore_state(daq, snapshot)
            except Exception:
                pass
            self.finished.emit(rc)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MFIA C(V) / C(F) Recipe Sweep")

        self.blocks: List[SweepBlock] = []
        self.thread: Optional[QThread] = None
        self.worker: Optional[SweepWorker] = None
        self.total_points_expected = 0
        self.current_sample_tag = ""
        self.current_plot_dir: Optional[Path] = None
        self.current_block_idx = 0
        self.current_block_name = ""
        self.current_block_spacing = "linear"
        self.run_started_mono: Optional[float] = None

        self._create_widgets()
        self._build_layout()
        self._wire_signals()
        self._refresh_plan_table()

    def _create_widgets(self) -> None:
        self.host_edit = QLineEdit("192.168.121.162")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(8004)
        self.device_edit = QLineEdit("")
        self.imps_spin = QSpinBox()
        self.imps_spin.setRange(0, 3)
        self.imps_spin.setValue(0)

        self.model_combo = QComboBox()
        for label, value in build_model_items():
            self.model_combo.addItem(label, value)

        self.quality_combo = QComboBox()
        for label, value in build_quality_items():
            self.quality_combo.addItem(label, value)
        self.quality_combo.setCurrentIndex(2)

        self.auto_bw = QCheckBox("Enable Auto Bandwidth")
        self.auto_bw.setChecked(True)

        self.inputrange_combo = QComboBox()
        for label, value in INPUT_RANGE_MODE_ITEMS:
            self.inputrange_combo.addItem(label, value)
        self.inputrange_combo.setCurrentIndex(1)

        self.manual_current_range = QDoubleSpinBox()
        self.manual_current_range.setDecimals(12)
        self.manual_current_range.setRange(1e-12, 1e-1)
        self.manual_current_range.setSingleStep(1e-9)
        self.manual_current_range.setValue(1e-6)
        self.manual_current_range.setSuffix(" A")

        self.demod_order = QSpinBox()
        self.demod_order.setRange(1, 8)
        self.demod_order.setValue(4)

        self.demod_timeconst = QDoubleSpinBox()
        self.demod_timeconst.setDecimals(6)
        self.demod_timeconst.setRange(1e-6, 100.0)
        self.demod_timeconst.setSingleStep(0.001)
        self.demod_timeconst.setValue(0.01)
        self.demod_timeconst.setSuffix(" s")

        self.demod_rate = QDoubleSpinBox()
        self.demod_rate.setDecimals(3)
        self.demod_rate.setRange(0.1, 10_000.0)
        self.demod_rate.setSingleStep(10.0)
        self.demod_rate.setValue(200.0)
        self.demod_rate.setSuffix(" Sa/s")

        self.demod_sinc = QCheckBox("Enable Sinc Filter")
        self.demod_sinc.setChecked(False)

        self.ramp_step = QDoubleSpinBox()
        self.ramp_step.setDecimals(4)
        self.ramp_step.setRange(0.0, 10.0)
        self.ramp_step.setSingleStep(0.01)
        self.ramp_step.setValue(0.05)
        self.ramp_step.setSuffix(" V")

        self.ramp_wait = QDoubleSpinBox()
        self.ramp_wait.setDecimals(4)
        self.ramp_wait.setRange(0.0, 2.0)
        self.ramp_wait.setSingleStep(0.01)
        self.ramp_wait.setValue(0.01)
        self.ramp_wait.setSuffix(" s")

        self.outdir_edit = QLineEdit(os.getcwd())
        self.outdir_btn = QPushButton("Browse")
        self.sample_edit = QLineEdit("sample")

        self.block_name = QLineEdit("block_1")
        self.block_mode = QComboBox()
        for label, value in BLOCK_MODE_ITEMS:
            self.block_mode.addItem(label, value)

        self.fixed_bias = QDoubleSpinBox()
        self.fixed_bias.setDecimals(5)
        self.fixed_bias.setRange(-20.0, 20.0)
        self.fixed_bias.setSingleStep(0.1)
        self.fixed_bias.setValue(0.0)
        self.fixed_bias.setSuffix(" V")

        self.fixed_amp = QDoubleSpinBox()
        self.fixed_amp.setDecimals(6)
        self.fixed_amp.setRange(0.0, 10.0)
        self.fixed_amp.setSingleStep(0.01)
        self.fixed_amp.setValue(0.1)
        self.fixed_amp.setSuffix(" V")

        self.fixed_freq = QDoubleSpinBox()
        self.fixed_freq.setDecimals(3)
        self.fixed_freq.setRange(0.0, 5_000_000.0)
        self.fixed_freq.setSingleStep(100.0)
        self.fixed_freq.setValue(1000.0)
        self.fixed_freq.setSuffix(" Hz")

        self.sweep_min = QDoubleSpinBox()
        self.sweep_min.setDecimals(6)
        self.sweep_min.setRange(-10_000_000.0, 10_000_000.0)
        self.sweep_min.setSingleStep(0.1)
        self.sweep_min.setValue(-1.0)

        self.sweep_max = QDoubleSpinBox()
        self.sweep_max.setDecimals(6)
        self.sweep_max.setRange(-10_000_000.0, 10_000_000.0)
        self.sweep_max.setSingleStep(0.1)
        self.sweep_max.setValue(1.0)

        self.sweep_step = QDoubleSpinBox()
        self.sweep_step.setDecimals(6)
        self.sweep_step.setRange(1e-12, 10_000_000.0)
        self.sweep_step.setSingleStep(0.01)
        self.sweep_step.setValue(0.1)

        self.sweep_points = QSpinBox()
        self.sweep_points.setRange(2, 200000)
        self.sweep_points.setValue(51)

        self.use_points = QCheckBox("Use points (unchecked = use step)")
        self.use_points.setChecked(True)

        self.spacing_combo = QComboBox()
        for label, value in SPACING_ITEMS:
            self.spacing_combo.addItem(label, value)

        self.order_combo = QComboBox()
        for label, value in ORDER_ITEMS:
            self.order_combo.addItem(label, value)

        self.block_settle = QDoubleSpinBox()
        self.block_settle.setDecimals(4)
        self.block_settle.setRange(0.0, 10.0)
        self.block_settle.setSingleStep(0.01)
        self.block_settle.setValue(0.1)
        self.block_settle.setSuffix(" s")

        self.add_block_btn = QPushButton("Add Block")
        self.update_block_btn = QPushButton("Update Selected")
        self.clear_editor_btn = QPushButton("Clear Editor")

        self.plan_table = QTableWidget(0, 12)
        self.plan_table.setHorizontalHeaderLabels(
            [
                "#",
                "Name",
                "Mode",
                "Sweep",
                "Spacing",
                "Order",
                "Fixed Bias (V)",
                "Fixed Amp (V)",
                "Fixed Freq (Hz)",
                "Settle (s)",
                "By",
                "Points",
            ]
        )
        self.plan_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.plan_table.setSelectionMode(QTableWidget.SingleSelection)
        self.plan_table.verticalHeader().setVisible(False)
        hh = self.plan_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in range(2, 12):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)

        self.remove_block_btn = QPushButton("Remove")
        self.move_up_btn = QPushButton("Move Up")
        self.move_down_btn = QPushButton("Move Down")
        self.duplicate_btn = QPushButton("Duplicate")
        self.load_selected_btn = QPushButton("Load To Editor")
        self.save_plan_btn = QPushButton("Save PLAN CSV")
        self.load_plan_btn = QPushButton("Load PLAN CSV")
        self.points_label = QLabel("Total points: 0")

        self.run_btn = QPushButton("Run")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        self.status_label = QLabel("Idle")
        self.last_label = QLabel("Last: --")
        self.block_label = QLabel("Block: --")
        block_font = QFont(self.block_label.font())
        block_font.setPointSize(max(14, block_font.pointSize() + 4))
        block_font.setBold(True)
        self.block_label.setFont(block_font)
        self.eta_label = QLabel("ETA PLAN: --")
        eta_font = QFont(self.eta_label.font())
        eta_font.setPointSize(max(13, eta_font.pointSize() + 3))
        eta_font.setBold(True)
        self.eta_label.setFont(eta_font)

        self.overall_progress = QProgressBar()
        self.overall_progress.setValue(0)
        self.block_progress = QProgressBar()
        self.block_progress.setValue(0)

        self.plot_c = pg.PlotWidget()
        self.plot_c.setBackground("w")
        self.plot_c.setLabel("left", "C (F)")
        self.plot_c.setLabel("bottom", "Sweep variable")
        self.plot_c.showGrid(x=True, y=True, alpha=0.2)
        self.plot_c_curve = self.plot_c.plot(
            pen=pg.mkPen(color=(20, 90, 160), width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(20, 90, 160),
        )

        self.plot_r = pg.PlotWidget()
        self.plot_r.setBackground("w")
        self.plot_r.setLabel("left", "R (Ohm)")
        self.plot_r.setLabel("bottom", "Sweep variable")
        self.plot_r.showGrid(x=True, y=True, alpha=0.2)
        self.plot_r.plotItem.setLogMode(x=False, y=True)
        self.plot_r_curve = self.plot_r.plot(
            pen=pg.mkPen(color=(140, 70, 20), width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(140, 70, 20),
        )

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.NoWrap)
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        self.log.setFont(font)

        self.plot_x: List[float] = []
        self.plot_y_c: List[float] = []
        self.plot_y_r: List[float] = []

        self.tabs = QTabWidget()
        self.params_tab = QWidget()
        self.plan_tab = QWidget()
        self.progress_tab = QWidget()
        self.tabs.addTab(self.params_tab, "PARAMS")
        self.tabs.addTab(self.plan_tab, "PLAN")
        self.tabs.addTab(self.progress_tab, "PROGRESS")

    def _build_layout(self) -> None:
        conn_group = QGroupBox("Connection")
        conn_form = QFormLayout()
        conn_form.addRow("Host", self.host_edit)
        conn_form.addRow("Port", self.port_spin)
        conn_form.addRow("Device (optional)", self.device_edit)
        conn_form.addRow("Imps index", self.imps_spin)
        conn_group.setLayout(conn_form)

        mfia_group = QGroupBox("MFIA Measurement")
        mfia_grid = QGridLayout()
        mfia_grid.addWidget(QLabel("Model"), 0, 0)
        mfia_grid.addWidget(self.model_combo, 0, 1)
        mfia_grid.addWidget(QLabel("Quality"), 0, 2)
        mfia_grid.addWidget(self.quality_combo, 0, 3)
        mfia_grid.addWidget(self.auto_bw, 1, 0, 1, 2)
        mfia_grid.addWidget(QLabel("Input Range Mode"), 1, 2)
        mfia_grid.addWidget(self.inputrange_combo, 1, 3)
        mfia_grid.addWidget(QLabel("Manual Current Range"), 2, 0)
        mfia_grid.addWidget(self.manual_current_range, 2, 1)
        mfia_group.setLayout(mfia_grid)

        integ_group = QGroupBox("Integration / Demod")
        integ_grid = QGridLayout()
        integ_grid.addWidget(QLabel("Demod order"), 0, 0)
        integ_grid.addWidget(self.demod_order, 0, 1)
        integ_grid.addWidget(QLabel("Time constant"), 0, 2)
        integ_grid.addWidget(self.demod_timeconst, 0, 3)
        integ_grid.addWidget(QLabel("Rate"), 1, 0)
        integ_grid.addWidget(self.demod_rate, 1, 1)
        integ_grid.addWidget(self.demod_sinc, 1, 2, 1, 2)
        integ_grid.addWidget(QLabel("Bias ramp step"), 2, 0)
        integ_grid.addWidget(self.ramp_step, 2, 1)
        integ_grid.addWidget(QLabel("Bias ramp wait"), 2, 2)
        integ_grid.addWidget(self.ramp_wait, 2, 3)
        integ_group.setLayout(integ_grid)

        output_group = QGroupBox("Output")
        output_form = QFormLayout()
        outdir_row = QHBoxLayout()
        outdir_row.addWidget(self.outdir_edit)
        outdir_row.addWidget(self.outdir_btn)
        output_form.addRow("Sample name", self.sample_edit)
        output_form.addRow("Output directory", outdir_row)
        output_group.setLayout(output_form)

        block_group = QGroupBox("Block Editor")
        block_grid = QGridLayout()
        block_grid.addWidget(QLabel("Name"), 0, 0)
        block_grid.addWidget(self.block_name, 0, 1)
        block_grid.addWidget(QLabel("Mode"), 0, 2)
        block_grid.addWidget(self.block_mode, 0, 3)
        block_grid.addWidget(QLabel("Fixed bias"), 1, 0)
        block_grid.addWidget(self.fixed_bias, 1, 1)
        block_grid.addWidget(QLabel("Fixed amplitude"), 1, 2)
        block_grid.addWidget(self.fixed_amp, 1, 3)
        block_grid.addWidget(QLabel("Fixed frequency"), 2, 0)
        block_grid.addWidget(self.fixed_freq, 2, 1)
        block_grid.addWidget(QLabel("Sweep min"), 2, 2)
        block_grid.addWidget(self.sweep_min, 2, 3)
        block_grid.addWidget(QLabel("Sweep max"), 3, 0)
        block_grid.addWidget(self.sweep_max, 3, 1)
        block_grid.addWidget(QLabel("Step"), 3, 2)
        block_grid.addWidget(self.sweep_step, 3, 3)
        block_grid.addWidget(QLabel("Points"), 4, 0)
        block_grid.addWidget(self.sweep_points, 4, 1)
        block_grid.addWidget(self.use_points, 4, 2, 1, 2)
        block_grid.addWidget(QLabel("Spacing"), 5, 0)
        block_grid.addWidget(self.spacing_combo, 5, 1)
        block_grid.addWidget(QLabel("Order"), 5, 2)
        block_grid.addWidget(self.order_combo, 5, 3)
        block_grid.addWidget(QLabel("Settle"), 6, 0)
        block_grid.addWidget(self.block_settle, 6, 1)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.add_block_btn)
        btn_row.addWidget(self.update_block_btn)
        btn_row.addWidget(self.clear_editor_btn)
        block_grid.addLayout(btn_row, 7, 0, 1, 4)
        block_group.setLayout(block_grid)

        params_layout = QVBoxLayout()
        params_layout.addWidget(conn_group)
        params_layout.addWidget(mfia_group)
        params_layout.addWidget(integ_group)
        params_layout.addWidget(output_group)
        params_layout.addWidget(block_group)
        params_layout.addStretch(1)
        self.params_tab.setLayout(params_layout)

        plan_layout = QVBoxLayout()
        plan_layout.addWidget(self.plan_table)
        plan_btns = QHBoxLayout()
        plan_btns.addWidget(self.save_plan_btn)
        plan_btns.addWidget(self.load_plan_btn)
        plan_btns.addWidget(self.load_selected_btn)
        plan_btns.addWidget(self.duplicate_btn)
        plan_btns.addWidget(self.move_up_btn)
        plan_btns.addWidget(self.move_down_btn)
        plan_btns.addWidget(self.remove_block_btn)
        plan_btns.addStretch(1)
        plan_btns.addWidget(self.points_label)
        plan_layout.addLayout(plan_btns)
        self.plan_tab.setLayout(plan_layout)

        progress_layout = QVBoxLayout()
        top_row = QHBoxLayout()
        top_row.addWidget(self.run_btn)
        top_row.addWidget(self.stop_btn)
        top_row.addStretch(1)
        top_row.addWidget(self.status_label)
        progress_layout.addLayout(top_row)
        progress_layout.addWidget(self.block_label)
        progress_layout.addWidget(self.eta_label)
        progress_layout.addWidget(self.last_label)
        progress_layout.addWidget(QLabel("Block progress"))
        progress_layout.addWidget(self.block_progress)
        progress_layout.addWidget(QLabel("Overall progress"))
        progress_layout.addWidget(self.overall_progress)

        plots_group = QGroupBox("Live Plots")
        plots_layout = QGridLayout()
        plots_layout.addWidget(self.plot_c, 0, 0)
        plots_layout.addWidget(self.plot_r, 0, 1)
        plots_group.setLayout(plots_layout)
        progress_layout.addWidget(plots_group)
        progress_layout.addWidget(self.log)
        self.progress_tab.setLayout(progress_layout)

        root = QWidget()
        root_layout = QVBoxLayout()
        root_layout.addWidget(self.tabs)
        root.setLayout(root_layout)
        self.setCentralWidget(root)
        self.resize(1200, 900)

    def _wire_signals(self) -> None:
        self.outdir_btn.clicked.connect(self._on_browse_outdir)
        self.add_block_btn.clicked.connect(self._on_add_block)
        self.update_block_btn.clicked.connect(self._on_update_block)
        self.clear_editor_btn.clicked.connect(self._clear_block_editor)
        self.remove_block_btn.clicked.connect(self._on_remove_block)
        self.move_up_btn.clicked.connect(self._on_move_up)
        self.move_down_btn.clicked.connect(self._on_move_down)
        self.duplicate_btn.clicked.connect(self._on_duplicate_block)
        self.load_selected_btn.clicked.connect(self._on_load_selected_to_editor)
        self.save_plan_btn.clicked.connect(self._on_save_plan_csv)
        self.load_plan_btn.clicked.connect(self._on_load_plan_csv)
        self.plan_table.itemSelectionChanged.connect(self._on_plan_selection_changed)
        self.block_mode.currentIndexChanged.connect(self._on_block_mode_changed)
        self.inputrange_combo.currentIndexChanged.connect(self._update_inputrange_ui)
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self._on_stop)
        self._on_block_mode_changed()
        self._update_inputrange_ui()

    def _on_browse_outdir(self) -> None:
        dirname = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.outdir_edit.text()
        )
        if dirname:
            self.outdir_edit.setText(dirname)

    def _update_inputrange_ui(self) -> None:
        self.manual_current_range.setEnabled(combo_value(self.inputrange_combo) == 0)

    def _on_block_mode_changed(self) -> None:
        mode = str(combo_value(self.block_mode))
        if mode == "bias":
            self.fixed_amp.setEnabled(True)
            self.fixed_freq.setEnabled(True)
        elif mode == "amplitude":
            self.fixed_bias.setEnabled(True)
            self.fixed_freq.setEnabled(True)
        else:
            self.fixed_bias.setEnabled(True)
            self.fixed_amp.setEnabled(True)

    def _block_from_editor(self) -> Optional[SweepBlock]:
        name = self.block_name.text().strip() or f"block_{len(self.blocks) + 1}"
        mode = str(combo_value(self.block_mode))
        spacing = str(combo_value(self.spacing_combo))
        order = str(combo_value(self.order_combo))
        step = float(self.sweep_step.value())
        points = int(self.sweep_points.value())
        use_points = bool(self.use_points.isChecked())

        block = SweepBlock(
            name=name,
            mode=mode,
            min_value=float(self.sweep_min.value()),
            max_value=float(self.sweep_max.value()),
            spacing=spacing,
            use_points=use_points,
            step=step,
            points=points,
            order=order,
            fixed_bias=float(self.fixed_bias.value()),
            fixed_amplitude=float(self.fixed_amp.value()),
            fixed_frequency=float(self.fixed_freq.value()),
            settle_s=float(self.block_settle.value()),
        )

        err = self._validate_block(block)
        if err:
            QMessageBox.warning(self, "Invalid block", err)
            return None
        return block

    def _validate_block(self, block: SweepBlock) -> Optional[str]:
        if block.mode not in ("bias", "amplitude", "frequency"):
            return f"Unknown mode: {block.mode}"
        if block.spacing not in ("linear", "log"):
            return f"Unknown spacing: {block.spacing}"
        if block.order not in ("zero_max_min_zero", "zero_min_max_zero"):
            return f"Unknown order: {block.order}"
        if block.use_points and block.points < 2:
            return "Points must be >= 2."
        if (not block.use_points) and block.step <= 0:
            return "Step must be > 0."
        if block.mode == "frequency":
            if block.min_value < 0 or block.max_value < 0:
                return "Frequency sweep cannot include negative values."
            if block.fixed_frequency < 0:
                return "Fixed frequency cannot be negative."
        if block.mode == "amplitude":
            if block.min_value < 0 or block.max_value < 0:
                return "Amplitude sweep cannot include negative values."
        if block.fixed_frequency < 0:
            return "Fixed frequency cannot be negative."
        if block.fixed_amplitude < 0:
            return "Fixed amplitude cannot be negative."
        try:
            _ = build_block_path(block)
        except Exception as exc:
            return str(exc)
        return None

    def _refresh_plan_table(self) -> None:
        self.plan_table.setRowCount(len(self.blocks))
        total_points = 0
        for idx, block in enumerate(self.blocks):
            n_pts = estimate_block_points(block)
            total_points += n_pts
            by = f"{block.points} pts" if block.use_points else f"{block.step:g} step"
            sweep_txt = f"{block.min_value:g} -> {block.max_value:g}"
            order_txt = "0->MAX->MIN->0" if block.order == "zero_max_min_zero" else "0->MIN->MAX->0"

            values = [
                str(idx + 1),
                block.name,
                block.mode.upper(),
                sweep_txt,
                block.spacing,
                order_txt,
                f"{block.fixed_bias:g}",
                f"{block.fixed_amplitude:g}",
                f"{block.fixed_frequency:g}",
                f"{block.settle_s:g}",
                by,
                str(n_pts),
            ]
            for c, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.plan_table.setItem(idx, c, item)

        self.total_points_expected = total_points
        self.points_label.setText(f"Total points: {total_points}")

    def _clear_block_editor(self) -> None:
        self.block_name.setText(f"block_{len(self.blocks) + 1}")
        self.block_mode.setCurrentIndex(0)
        self.sweep_min.setValue(-1.0)
        self.sweep_max.setValue(1.0)
        self.sweep_step.setValue(0.1)
        self.sweep_points.setValue(51)
        self.use_points.setChecked(True)
        self.spacing_combo.setCurrentIndex(0)
        self.order_combo.setCurrentIndex(0)
        self.fixed_bias.setValue(0.0)
        self.fixed_amp.setValue(0.1)
        self.fixed_freq.setValue(1000.0)
        self.block_settle.setValue(0.1)

    def _on_add_block(self) -> None:
        block = self._block_from_editor()
        if block is None:
            return
        self.blocks.append(block)
        self._refresh_plan_table()

    def _selected_row(self) -> Optional[int]:
        rows = self.plan_table.selectionModel().selectedRows()
        if not rows:
            return None
        return int(rows[0].row())

    def _on_update_block(self) -> None:
        row = self._selected_row()
        if row is None:
            QMessageBox.information(self, "Update block", "Select a block in PLAN tab first.")
            return
        block = self._block_from_editor()
        if block is None:
            return
        self.blocks[row] = block
        self._refresh_plan_table()
        self.plan_table.selectRow(row)

    def _on_remove_block(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        self.blocks.pop(row)
        self._refresh_plan_table()

    def _on_move_up(self) -> None:
        row = self._selected_row()
        if row is None or row <= 0:
            return
        self.blocks[row - 1], self.blocks[row] = self.blocks[row], self.blocks[row - 1]
        self._refresh_plan_table()
        self.plan_table.selectRow(row - 1)

    def _on_move_down(self) -> None:
        row = self._selected_row()
        if row is None or row >= len(self.blocks) - 1:
            return
        self.blocks[row + 1], self.blocks[row] = self.blocks[row], self.blocks[row + 1]
        self._refresh_plan_table()
        self.plan_table.selectRow(row + 1)

    def _on_duplicate_block(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        blk = self.blocks[row]
        clone = SweepBlock(**asdict(blk))
        clone.name = f"{blk.name}_copy"
        self.blocks.insert(row + 1, clone)
        self._refresh_plan_table()
        self.plan_table.selectRow(row + 1)

    def _on_plan_selection_changed(self) -> None:
        row = self._selected_row()
        self.load_selected_btn.setEnabled(row is not None)

    def _on_load_selected_to_editor(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        block = self.blocks[row]
        self.block_name.setText(block.name)
        self._set_combo_data(self.block_mode, block.mode)
        self.sweep_min.setValue(block.min_value)
        self.sweep_max.setValue(block.max_value)
        self.sweep_step.setValue(max(block.step, self.sweep_step.minimum()))
        self.sweep_points.setValue(max(2, block.points))
        self.use_points.setChecked(block.use_points)
        self._set_combo_data(self.spacing_combo, block.spacing)
        self._set_combo_data(self.order_combo, block.order)
        self.fixed_bias.setValue(block.fixed_bias)
        self.fixed_amp.setValue(block.fixed_amplitude)
        self.fixed_freq.setValue(block.fixed_frequency)
        self.block_settle.setValue(block.settle_s)
        self.tabs.setCurrentWidget(self.params_tab)

    def _set_combo_data(self, combo: QComboBox, data: object) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i)
                return

    def _block_to_plan_row(self, block: SweepBlock) -> Dict[str, object]:
        return {
            "name": block.name,
            "mode": block.mode,
            "min_value": block.min_value,
            "max_value": block.max_value,
            "spacing": block.spacing,
            "use_points": int(block.use_points),
            "step": block.step,
            "points": block.points,
            "order": block.order,
            "fixed_bias": block.fixed_bias,
            "fixed_amplitude": block.fixed_amplitude,
            "fixed_frequency": block.fixed_frequency,
            "settle_s": block.settle_s,
        }

    def _plan_row_to_block(self, row: Dict[str, object], line_no: int) -> SweepBlock:
        try:
            block = SweepBlock(
                name=str(row["name"]).strip(),
                mode=str(row["mode"]).strip().lower(),
                min_value=float(row["min_value"]),
                max_value=float(row["max_value"]),
                spacing=str(row["spacing"]).strip().lower(),
                use_points=parse_bool(row["use_points"]),
                step=float(row["step"]),
                points=int(float(row["points"])),
                order=normalize_order(row["order"]),
                fixed_bias=float(row["fixed_bias"]),
                fixed_amplitude=float(row["fixed_amplitude"]),
                fixed_frequency=float(row["fixed_frequency"]),
                settle_s=float(row["settle_s"]),
            )
        except Exception as exc:
            raise ValueError(f"Invalid values at CSV line {line_no}: {exc}") from exc

        if not block.name:
            raise ValueError(f"Missing block name at CSV line {line_no}")

        err = self._validate_block(block)
        if err:
            raise ValueError(f"Invalid block at CSV line {line_no}: {err}")
        return block

    def _on_save_plan_csv(self) -> None:
        if not self.blocks:
            QMessageBox.information(self, "Save PLAN", "No blocks to save.")
            return

        sample_tag = sanitize_filename(self.sample_edit.text().strip() or "sample")
        default_path = str(Path(self.outdir_edit.text().strip() or os.getcwd()) / f"{sample_tag}_plan.csv")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save PLAN as CSV", default_path, "CSV Files (*.csv)"
        )
        if not filename:
            return

        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=PLAN_CSV_FIELDS)
                writer.writeheader()
                for block in self.blocks:
                    writer.writerow(self._block_to_plan_row(block))
            self.log.appendPlainText(f"Saved PLAN CSV: {filename}")
        except Exception as exc:
            QMessageBox.critical(self, "Save PLAN", f"Failed to save CSV:\n{exc}")

    def _on_load_plan_csv(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load PLAN CSV", self.outdir_edit.text().strip() or os.getcwd(), "CSV Files (*.csv)"
        )
        if not filename:
            return

        try:
            with open(filename, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError("CSV file is missing a header.")
                missing = [k for k in PLAN_CSV_FIELDS if k not in reader.fieldnames]
                if missing:
                    raise ValueError(f"Missing required columns: {', '.join(missing)}")

                loaded: List[SweepBlock] = []
                for i, row in enumerate(reader, start=2):
                    if all(str(v).strip() == "" for v in row.values()):
                        continue
                    loaded.append(self._plan_row_to_block(row, i))

            if not loaded:
                raise ValueError("No valid blocks found in CSV.")

            self.blocks = loaded
            self._refresh_plan_table()
            self.log.appendPlainText(f"Loaded PLAN CSV: {filename} ({len(loaded)} blocks)")
            self.tabs.setCurrentWidget(self.plan_tab)
        except Exception as exc:
            QMessageBox.critical(self, "Load PLAN", f"Failed to load CSV:\n{exc}")

    def _build_config(self) -> Optional[RecipeConfig]:
        outdir = self.outdir_edit.text().strip()
        sample = self.sample_edit.text().strip() or "sample"
        if not outdir:
            QMessageBox.warning(self, "Output directory", "Please set output directory.")
            return None
        if not self.blocks:
            QMessageBox.warning(self, "Recipe", "Add at least one block in the PLAN tab.")
            return None
        if not Path(outdir).exists():
            QMessageBox.warning(self, "Output directory", f"Directory does not exist: {outdir}")
            return None

        for idx, block in enumerate(self.blocks, start=1):
            err = self._validate_block(block)
            if err:
                QMessageBox.warning(self, "Block validation", f"Block {idx} '{block.name}': {err}")
                return None

        return RecipeConfig(
            host=self.host_edit.text().strip(),
            port=int(self.port_spin.value()),
            device=self.device_edit.text().strip(),
            imps=int(self.imps_spin.value()),
            model=int(combo_value(self.model_combo)),
            quality=int(combo_value(self.quality_combo)),
            auto_bw=bool(self.auto_bw.isChecked()),
            inputrange_mode=int(combo_value(self.inputrange_combo)),
            manual_current_range=float(self.manual_current_range.value()),
            demod_order=int(self.demod_order.value()),
            demod_timeconstant=float(self.demod_timeconst.value()),
            demod_rate=float(self.demod_rate.value()),
            demod_sinc=bool(self.demod_sinc.isChecked()),
            ramp_step=float(self.ramp_step.value()),
            ramp_wait=float(self.ramp_wait.value()),
            outdir=outdir,
            sample=sample,
            blocks=[SweepBlock(**asdict(b)) for b in self.blocks],
        )

    def _prepare_progress(self) -> None:
        self.plot_x = []
        self.plot_y_c = []
        self.plot_y_r = []
        self.plot_c_curve.setData([], [])
        self.plot_r_curve.setData([], [])
        self.plot_c.plotItem.setLogMode(x=False, y=False)
        self.plot_r.plotItem.setLogMode(x=False, y=True)
        self.plot_r.setLabel("left", "R (Ohm)")
        self.overall_progress.setRange(0, max(1, self.total_points_expected))
        self.overall_progress.setValue(0)
        self.block_progress.setRange(0, 1)
        self.block_progress.setValue(0)
        self.last_label.setText("Last: --")
        self.block_label.setText("Block: --")
        self.eta_label.setText("ETA PLAN: --")
        self.current_block_idx = 0
        self.current_block_name = ""
        self.current_block_spacing = "linear"
        self.run_started_mono = None

    def _on_run(self) -> None:
        if self.worker is not None:
            return
        cfg = self._build_config()
        if cfg is None:
            return

        self._refresh_plan_table()
        self._prepare_progress()
        self.current_plot_dir = None
        self.current_sample_tag = sanitize_filename(cfg.sample)
        self.run_started_mono = time.perf_counter()
        self.tabs.setCurrentWidget(self.progress_tab)
        self.log.appendPlainText("Starting recipe sweep...")
        self.status_label.setText("Running")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.thread = QThread(self)
        self.worker = SweepWorker(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._on_log)
        self.worker.run_paths.connect(self._on_run_paths)
        self.worker.block_started.connect(self._on_block_started)
        self.worker.block_finished.connect(self._on_block_finished)
        self.worker.progress.connect(self._on_progress)
        self.worker.point.connect(self._on_point)
        self.worker.finished.connect(self._on_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_stop(self) -> None:
        if self.worker is None:
            return
        self.log.appendPlainText("Stop requested...")
        self.worker.request_stop()

    @pyqtSlot(str)
    def _on_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    @pyqtSlot(str, str, str)
    def _on_run_paths(self, sample_tag: str, run_dir: str, plot_dir: str) -> None:
        self.current_sample_tag = sample_tag
        self.current_plot_dir = Path(plot_dir)
        self.log.appendPlainText(f"Run directory: {run_dir}")

    @pyqtSlot(int, str, int)
    def _on_block_started(self, block_idx: int, name: str, n_points: int) -> None:
        self.current_block_idx = block_idx
        self.current_block_name = name
        spacing = "linear"
        if 1 <= block_idx <= len(self.blocks):
            spacing = str(self.blocks[block_idx - 1].spacing)
        self.current_block_spacing = spacing

        x_log = spacing == "log"
        self.plot_c.plotItem.setLogMode(x=x_log, y=False)
        self.plot_r.plotItem.setLogMode(x=x_log, y=True)
        self.block_label.setText(f"Block {block_idx}: {name} ({n_points} points)")
        self.block_progress.setRange(0, max(1, n_points))
        self.block_progress.setValue(0)
        self.plot_x = []
        self.plot_y_c = []
        self.plot_y_r = []
        self.plot_c_curve.setData([], [])
        self.plot_r_curve.setData([], [])

    @pyqtSlot(int, str)
    def _on_block_finished(self, block_idx: int, block_name: str) -> None:
        self._export_current_plots(
            reason="block_done", block_idx=block_idx, block_name=block_name
        )

    @pyqtSlot(int, int, int, int)
    def _on_progress(self, block_idx: int, block_done: int, block_total: int, overall_done: int) -> None:
        self.block_progress.setRange(0, max(1, block_total))
        self.block_progress.setValue(block_done)
        self.overall_progress.setValue(overall_done)
        if self.run_started_mono is not None and overall_done > 0 and self.total_points_expected > 0:
            elapsed = max(0.0, time.perf_counter() - self.run_started_mono)
            avg = elapsed / overall_done
            rem_points = max(0, self.total_points_expected - overall_done)
            rem_s = rem_points * avg
            self.eta_label.setText(f"ETA PLAN: {self._format_seconds(rem_s)}")

    @pyqtSlot(object)
    def _on_point(self, row: object) -> None:
        if not isinstance(row, dict):
            return
        mode = str(row.get("mode", ""))
        x = to_float(row.get("sweep_value"))
        c = to_float(row.get("C_F"))
        model_r = to_float(row.get("R_Ohm"))
        demod_r = to_float(row.get("demod_R"))
        bias = to_float(row.get("set_bias_V"))
        amp = to_float(row.get("set_amplitude_V"))
        freq = to_float(row.get("set_frequency_Hz"))
        curr_range = to_float(row.get("currin_range_A"))

        # IV mode: when frequency is 0, plot demodR instead of model-derived R.
        use_demod_r = math.isclose(freq, 0.0, rel_tol=0.0, abs_tol=1e-15)
        r_plot = demod_r if use_demod_r else model_r
        r_label = "demodR" if use_demod_r else "R"
        r_units = "A" if use_demod_r else "Ohm"

        self.plot_x.append(x)
        self.plot_y_c.append(c)
        self.plot_y_r.append(r_plot)

        x_arr = np.array(self.plot_x, dtype=float)
        c_arr = np.array(self.plot_y_c, dtype=float)
        r_arr = np.array(self.plot_y_r, dtype=float)

        x_log = self.current_block_spacing == "log"
        if x_log:
            # Log axes cannot display non-positive values (e.g. forced endpoint at 0).
            x_mask = x_arr > 0.0
        else:
            x_mask = np.ones_like(x_arr, dtype=bool)

        c_mask = x_mask & np.isfinite(c_arr)
        r_mask = x_mask & np.isfinite(r_arr) & (r_arr > 0.0)

        self.plot_c_curve.setData(x_arr[c_mask], c_arr[c_mask])
        self.plot_r_curve.setData(x_arr[r_mask], r_arr[r_mask])

        x_label = {
            "bias": "Bias (V)",
            "amplitude": "Amplitude (V)",
            "frequency": "Frequency (Hz)",
        }.get(mode, "Sweep variable")
        self.plot_c.setLabel("bottom", x_label)
        self.plot_r.setLabel("bottom", x_label)
        self.plot_r.setLabel("left", f"{r_label} ({r_units})")

        self.last_label.setText(
            "Last: "
            f"mode={mode}, bias={bias:.4g} V, amp={amp:.4g} V, freq={freq:.4g} Hz, "
            f"C={c:.3e} F, {r_label}={r_plot:.3e} {r_units}, curr_range={curr_range:.3e} A"
        )

    @pyqtSlot(int)
    def _on_finished(self, code: int) -> None:
        reason = "completed" if code == 0 else "stopped"
        self._export_current_plots(reason=reason, block_idx=self.current_block_idx, block_name=self.current_block_name)
        QApplication.beep()
        self.status_label.setText("Idle")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None
        self.thread = None
        self.log.appendPlainText(f"Finished with exit code {code}.")

    def _export_current_plots(
        self, reason: str, block_idx: Optional[int] = None, block_name: str = ""
    ) -> None:
        if self.current_plot_dir is None:
            return
        try:
            self.current_plot_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        tag = self.current_sample_tag or "sample"
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        block_tag = "summary"
        if block_idx is not None and block_idx > 0:
            safe_block = sanitize_filename(block_name or f"block_{block_idx}")
            block_tag = f"block{block_idx:02d}_{safe_block}"

        c_path = self.current_plot_dir / f"{tag}_{block_tag}_C_{reason}_{ts}.png"
        r_path = self.current_plot_dir / f"{tag}_{block_tag}_R_{reason}_{ts}.png"

        try:
            c_exporter = ImageExporter(self.plot_c.plotItem)
            c_exporter.parameters()["width"] = 1600
            c_exporter.export(str(c_path))
        except Exception as exc:
            self.log.appendPlainText(f"Warning: failed to save C plot: {exc}")
        try:
            r_exporter = ImageExporter(self.plot_r.plotItem)
            r_exporter.parameters()["width"] = 1600
            r_exporter.export(str(r_path))
        except Exception as exc:
            self.log.appendPlainText(f"Warning: failed to save R plot: {exc}")

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.request_stop()
        super().closeEvent(event)

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        s = int(max(0.0, seconds))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
