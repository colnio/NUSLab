
import os
import csv
import re
import sys
import time
import math
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
import zhinst.core as zi
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QGridLayout,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parents[1]
KEITHLEY_PATH = ROOT / "KeithleyGUI"
if str(KEITHLEY_PATH) not in sys.path:
    sys.path.insert(0, str(KEITHLEY_PATH))

try:
    import keithley  # noqa: E402
except Exception:
    keithley = None

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


@dataclass
class SweepConfig:
    host: str
    port: int
    device: str
    imps: int
    model: int
    freq: float
    drive: float
    quality: int
    bias_start: float
    bias_stop: float
    bias_step: float
    bias_settle: float
    gate_start: float
    gate_stop: float
    gate_step: float
    gate_settle: float
    gate_source: str
    gate_device: str
    gate_current_range: str
    gate_compliance: Optional[float]
    gate_nplc: float
    aux: int
    ramp_step: float
    ramp_wait: float
    outdir: str
    sample: str


@dataclass
class SavedState:
    imps_enable: int
    imps_model: int
    bias_enable: int
    bias_value: float
    drive_path: Optional[str]
    drive_value: Optional[float]
    freq_path: Optional[str]
    freq_value: Optional[float]
    quality_path: Optional[str]
    quality_value: Optional[int]
    sigout_index: Optional[int]
    osc_index: Optional[int]
    sigout_amp_path: Optional[str]
    sigout_amp_value: Optional[float]
    sigout_enable_path: Optional[str]
    sigout_enable_value: Optional[int]
    aux_outputselect: Optional[int]
    aux_preoffset: Optional[float]
    aux_scale: Optional[float]
    aux_offset: Optional[float]
    aux_limitlower: Optional[float]
    aux_limitupper: Optional[float]


def default_python_path() -> str:
    here = Path(__file__).resolve().parent
    candidate = here / ".venv310" / "Scripts" / "python.exe"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def build_model_items() -> list[tuple[str, int]]:
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


def build_quality_items() -> list[tuple[str, int]]:
    return [
        ("High Speed", 0),
        ("Medium", 1),
        ("High Accuracy", 2),
        ("Very High Accuracy", 3),
    ]


def list_keithley_devices() -> list[str]:
    if keithley is None:
        return ["Mock"]
    try:
        devices = [" - ".join(i) for i in keithley.get_devices_list()]
    except Exception:
        devices = []
    devices.append("Mock")
    return devices


def build_range(start: float, stop: float, step: float) -> List[float]:
    if step == 0:
        raise ValueError("Step cannot be zero")
    if math.isclose(start, stop, rel_tol=0, abs_tol=1e-12):
        return [float(start)]
    step = abs(step) if stop >= start else -abs(step)
    values: List[float] = []
    v = start
    tol = abs(step) / 2.0 + 1e-12
    max_iter = 1_000_000
    while True:
        if step > 0 and v > stop + tol:
            break
        if step < 0 and v < stop - tol:
            break
        values.append(float(v))
        if len(values) > max_iter:
            raise RuntimeError("Too many steps; check start/stop/step")
        v = v + step
    if not math.isclose(values[-1], stop, rel_tol=0, abs_tol=tol):
        values.append(float(stop))
    return values


def ramp_values(current: float, target: float, step: float) -> List[float]:
    if step <= 0:
        return [target]
    if math.isclose(current, target, rel_tol=0, abs_tol=1e-9):
        return [target]
    delta = target - current
    n_steps = max(1, int(abs(delta) / max(step, 1e-9)))
    return [float(v) for v in np.linspace(current, target, n_steps + 1)[1:]]


def ramp_set_double(daq: zi.ziDAQServer, path: str, target: float, step: float, wait: float) -> None:
    try:
        current = daq.getDouble(path)
    except Exception:
        daq.setDouble(path, target)
        time.sleep(wait)
        return

    for v in ramp_values(current, target, step):
        daq.setDouble(path, float(v))
        time.sleep(wait)


def find_mfia_device(host: str, port: int, preferred: Optional[str]) -> Tuple[str, str]:
    disc = zi.ziDiscovery()
    devices = disc.findAll()
    if not devices:
        raise RuntimeError("No devices found by discovery.")

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

def save_state(daq: zi.ziDAQServer, dev: str, imps: int, aux: int) -> SavedState:
    base = f"/{dev}/imps/{imps}"
    auxbase = f"/{dev}/auxouts/{aux}"

    drive_path = f"{base}/drive"
    freq_path = f"{base}/freq"

    drive_value = try_get_double(daq, drive_path)
    freq_value = try_get_double(daq, freq_path)

    quality_path = None
    quality_value = None
    for candidate in (f"{base}/precision", f"{base}/accuracy", f"{base}/quality"):
        q = try_get_int(daq, candidate)
        if q is not None:
            quality_path = candidate
            quality_value = q
            break

    sigout_index = None
    osc_index = None
    sigout_amp_path = None
    sigout_amp_value = None
    sigout_enable_path = None
    sigout_enable_value = None

    for candidate in (f"{base}/sigout", f"{base}/outputselect"):
        s = try_get_int(daq, candidate)
        if s is not None:
            sigout_index = s
            break

    for candidate in (f"{base}/oscselect", f"{base}/oscindex"):
        o = try_get_int(daq, candidate)
        if o is not None:
            osc_index = o
            break

    if sigout_index is not None and osc_index is not None:
        amp_path = f"/{dev}/sigouts/{sigout_index}/amplitudes/{osc_index}"
        sigout_amp_value = try_get_double(daq, amp_path)
        if sigout_amp_value is not None:
            sigout_amp_path = amp_path

        en_path = f"/{dev}/sigouts/{sigout_index}/enables/{osc_index}"
        sigout_enable_value = try_get_int(daq, en_path)
        if sigout_enable_value is not None:
            sigout_enable_path = en_path

    return SavedState(
        imps_enable=daq.getInt(f"{base}/enable"),
        imps_model=daq.getInt(f"{base}/model"),
        bias_enable=daq.getInt(f"{base}/bias/enable"),
        bias_value=daq.getDouble(f"{base}/bias/value"),
        drive_path=drive_path if drive_value is not None else None,
        drive_value=drive_value,
        freq_path=freq_path if freq_value is not None else None,
        freq_value=freq_value,
        quality_path=quality_path,
        quality_value=quality_value,
        sigout_index=sigout_index,
        osc_index=osc_index,
        sigout_amp_path=sigout_amp_path,
        sigout_amp_value=sigout_amp_value,
        sigout_enable_path=sigout_enable_path,
        sigout_enable_value=sigout_enable_value,
        aux_outputselect=try_get_int(daq, f"{auxbase}/outputselect"),
        aux_preoffset=try_get_double(daq, f"{auxbase}/preoffset"),
        aux_scale=try_get_double(daq, f"{auxbase}/scale"),
        aux_offset=try_get_double(daq, f"{auxbase}/offset"),
        aux_limitlower=try_get_double(daq, f"{auxbase}/limitlower"),
        aux_limitupper=try_get_double(daq, f"{auxbase}/limitupper"),
    )


def restore_state(daq: zi.ziDAQServer, dev: str, imps: int, aux: int, s: SavedState) -> None:
    base = f"/{dev}/imps/{imps}"
    auxbase = f"/{dev}/auxouts/{aux}"

    daq.setInt(f"{base}/enable", s.imps_enable)
    daq.setInt(f"{base}/model", s.imps_model)
    daq.setInt(f"{base}/bias/enable", s.bias_enable)
    daq.setDouble(f"{base}/bias/value", s.bias_value)

    if s.drive_path and s.drive_value is not None:
        try:
            daq.setDouble(s.drive_path, s.drive_value)
        except Exception:
            pass
    if s.freq_path and s.freq_value is not None:
        try:
            daq.setDouble(s.freq_path, s.freq_value)
        except Exception:
            pass
    if s.quality_path and s.quality_value is not None:
        try:
            daq.setInt(s.quality_path, s.quality_value)
        except Exception:
            pass

    if s.sigout_amp_path and s.sigout_amp_value is not None:
        try:
            daq.setDouble(s.sigout_amp_path, s.sigout_amp_value)
        except Exception:
            pass

    if s.sigout_enable_path and s.sigout_enable_value is not None:
        try:
            daq.setInt(s.sigout_enable_path, s.sigout_enable_value)
        except Exception:
            pass

    if s.aux_outputselect is not None:
        daq.setInt(f"{auxbase}/outputselect", s.aux_outputselect)
    if s.aux_preoffset is not None:
        daq.setDouble(f"{auxbase}/preoffset", s.aux_preoffset)
    if s.aux_scale is not None:
        daq.setDouble(f"{auxbase}/scale", s.aux_scale)
    if s.aux_offset is not None:
        daq.setDouble(f"{auxbase}/offset", s.aux_offset)
    if s.aux_limitlower is not None:
        daq.setDouble(f"{auxbase}/limitlower", s.aux_limitlower)
    if s.aux_limitupper is not None:
        daq.setDouble(f"{auxbase}/limitupper", s.aux_limitupper)


def get_latest_sample(daq: zi.ziDAQServer, path: str, timeout: float = 1.0) -> Dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
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
    raise TimeoutError(f"No data received for {path} within {timeout}s")


def try_set_double(daq: zi.ziDAQServer, paths: List[str], value: float) -> Optional[str]:
    for p in paths:
        try:
            daq.setDouble(p, float(value))
            return p
        except Exception:
            continue
    return None


def try_set_int(daq: zi.ziDAQServer, paths: List[str], value: int) -> Optional[str]:
    for p in paths:
        try:
            daq.setInt(p, int(value))
            return p
        except Exception:
            continue
    return None


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
    for candidate in (f"{base}/sigout", f"{base}/outputselect"):
        s = try_get_int(daq, candidate)
        if s is not None:
            sigout_index = s
            break
    for candidate in (f"{base}/oscselect", f"{base}/oscindex"):
        o = try_get_int(daq, candidate)
        if o is not None:
            osc_index = o
            break

    if sigout_index is not None and osc_index is not None:
        amp_path = f"/{dev}/sigouts/{sigout_index}/amplitudes/{osc_index}"
        if try_set_double(daq, [amp_path], value):
            set_paths.append(amp_path)

        enable_path = f"/{dev}/sigouts/{sigout_index}/enables/{osc_index}"
        if try_set_int(daq, [enable_path], 1):
            set_paths.append(enable_path)

    return set_paths


def compute_model_values(model: int, param0: float, param1: float, freq: Optional[float]) -> Tuple[float, float]:
    cap = float("nan")
    res = float("nan")

    if model in (0, 1, 4, 5, 9):
        cap = float(param1)
    elif model == 3 and freq and freq > 0:
        try:
            cap = float(param1) / (2.0 * math.pi * float(freq))
        except Exception:
            cap = float("nan")

    if model in (0, 1, 2, 8):
        res = float(param0)
    elif model == 3:
        try:
            res = 1.0 / float(param0) if float(param0) != 0.0 else float("nan")
        except Exception:
            res = float("nan")

    return cap, res


def sanitize_filename(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", ".", "+"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"cv_gate_map_{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger


def safe_stop_outputs(cfg: SweepConfig, log: Optional[logging.Logger] = None) -> None:
    # MFIA stop (independent)
    try:
        daq = zi.ziDAQServer(cfg.host, cfg.port, 6)
        dev, interface = find_mfia_device(cfg.host, cfg.port, cfg.device if cfg.device else None)
        dev = dev.lower()
        if not interface:
            interface = "PCIe"
        daq.connectDevice(dev, interface)

        base = f"/{dev}/imps/{cfg.imps}"
        daq.setDouble(f"{base}/bias/value", 0.0)
        daq.setInt(f"{base}/bias/enable", 0)

        if cfg.gate_source == "mfia-aux":
            auxbase = f"/{dev}/auxouts/{cfg.aux}"
            daq.setDouble(f"{auxbase}/offset", 0.0)
    except Exception as exc:
        if log:
            log.warning("MFIA safe stop failed: %s", exc)

    # Keithley stop (independent)
    gate_device = None
    try:
        if cfg.gate_device and keithley is not None:
            gate_device = keithley.get_device(cfg.gate_device, nplc=cfg.gate_nplc)
        if gate_device is not None:
            try:
                gate_device.set_voltage(0)
            except Exception:
                pass
            try:
                gate_device.disable_output()
            except Exception:
                pass
    except Exception as exc:
        if log:
            log.warning("Keithley safe stop failed: %s", exc)

    if log:
        log.info("Safe stop completed")

class SweepWorker(QObject):
    log = pyqtSignal(str)
    point = pyqtSignal(float, float, float, float, float, float)
    finished = pyqtSignal(int)

    def __init__(self, cfg: SweepConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._stop = False

    @pyqtSlot()
    def request_stop(self) -> None:
        self._stop = True

    def _log(self, logger: logging.Logger, message: str) -> None:
        logger.info(message)
        self.log.emit(message)

    @pyqtSlot()
    def run(self) -> None:
        rc = 0
        daq = None
        gate_device = None
        state = None
        dev = None
        path_sample = None
        logger = None

        try:
            bias_values = build_range(self.cfg.bias_start, self.cfg.bias_stop, self.cfg.bias_step)
            gate_values = build_range(self.cfg.gate_start, self.cfg.gate_stop, self.cfg.gate_step)

            date = str(dt.date.today())
            start_time = dt.datetime.now().strftime("%Y-%m-%d %H-%M-%S")

            sample_name = self.cfg.sample.strip() or "sample"
            sample_dir = os.path.join(self.cfg.outdir, date, sample_name)
            data_dir = os.path.join(sample_dir, "data")
            plot_dir = os.path.join(sample_dir, "plots")
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(plot_dir, exist_ok=True)

            map_folder = sanitize_filename(f"CV_map_{sample_name}_{start_time}")
            map_dir = os.path.join(data_dir, map_folder)
            os.makedirs(map_dir, exist_ok=True)

            log_path = os.path.join(sample_dir, f"CVGATE_{sample_name}_{start_time}.log")
            logger = setup_logger(log_path)
            self._log(logger, f"Starting C(Vbias) sweep with gate source {self.cfg.gate_source}")
            self._log(logger, f"Output map folder: {map_dir}")

            daq = zi.ziDAQServer(self.cfg.host, self.cfg.port, 6)
            dev, interface = find_mfia_device(self.cfg.host, self.cfg.port, self.cfg.device if self.cfg.device else None)
            dev = dev.lower()
            if not interface:
                interface = "PCIe"
            daq.connectDevice(dev, interface)

            imps = self.cfg.imps
            aux = self.cfg.aux
            base = f"/{dev}/imps/{imps}"
            auxbase = f"/{dev}/auxouts/{aux}"

            if self.cfg.gate_source == "mfia-aux":
                if any(abs(v) > 10.0 for v in gate_values):
                    raise ValueError("Gate values exceed +/-10 V (MFIA AUX limit)")

            mode = daq.getInt(f"{base}/mode")
            if mode == 0:
                if any(abs(v) > 3.0 for v in bias_values):
                    raise ValueError("Bias exceeds +/-3 V in 4-terminal mode")
            else:
                if any(abs(v) > 10.0 for v in bias_values):
                    raise ValueError("Bias exceeds +/-10 V in 2-terminal mode")

            state = save_state(daq, dev, imps, aux)

            daq.setInt(f"{base}/enable", 1)
            daq.setInt(f"{base}/model", int(self.cfg.model))
            daq.setInt(f"{base}/bias/enable", 1)

            if self.cfg.freq is not None:
                try_set_double(daq, [f"{base}/freq", f"{base}/osc/freq"], self.cfg.freq)

            if self.cfg.drive is not None:
                set_paths = set_drive_amplitude(daq, dev, imps, self.cfg.drive)
                if not set_paths:
                    self._log(logger, "Warning: unable to set drive amplitude on known nodes")

            if self.cfg.quality is not None:
                qpath = try_set_int(daq, [f"{base}/precision", f"{base}/accuracy", f"{base}/quality"], self.cfg.quality)
                if not qpath:
                    self._log(logger, "Warning: unable to set measurement quality on known nodes")

            if self.cfg.gate_source == "mfia-aux":
                daq.setInt(f"{auxbase}/outputselect", -1)
                daq.setDouble(f"{auxbase}/preoffset", 0.0)
                daq.setDouble(f"{auxbase}/scale", 1.0)
                lower = max(-10.0, min(gate_values) - 0.1)
                upper = min(10.0, max(gate_values) + 0.1)
                daq.setDouble(f"{auxbase}/limitlower", lower)
                daq.setDouble(f"{auxbase}/limitupper", upper)
            else:
                if not self.cfg.gate_device:
                    raise ValueError("Gate device is required when gate source is Keithley")
                if keithley is None:
                    raise RuntimeError("Keithley module is not available")
                gate_device = keithley.get_device(self.cfg.gate_device, nplc=self.cfg.gate_nplc)
                if gate_device is None:
                    raise RuntimeError(f"Could not open gate device: {self.cfg.gate_device}")
                if self.cfg.gate_compliance is not None and hasattr(gate_device, "set_complicance_current"):
                    try:
                        gate_device.set_complicance_current(float(self.cfg.gate_compliance))
                    except Exception as exc:
                        self._log(logger, f"Warning: failed to set gate compliance current: {exc}")
                if self.cfg.gate_current_range != "auto" and hasattr(gate_device, "set_current_range"):
                    try:
                        gate_device.set_current_range(float(eval(str(self.cfg.gate_current_range))))
                    except Exception as exc:
                        self._log(logger, f"Warning: failed to set gate current range: {exc}")

            path_sample = f"{base}/sample"
            daq.subscribe(path_sample)
            daq.sync()

            fieldnames = [
                "timestamp",
                "gate_V",
                "bias_V",
                "drive_V",
                "freq_Hz",
                "model",
                "model_name",
                "param0",
                "param1",
                "param0_name",
                "param1_name",
                "C_F",
                "R_Ohm",
                "Z_real",
                "Z_imag",
                "Z_abs",
                "leakage_A",
                "gate_source",
            ]

            last_gate = gate_values[0] if gate_values else 0.0
            time.sleep(max(0.0, self.cfg.gate_settle))

            for gate in gate_values:
                if self._stop:
                    break
                if self.cfg.gate_source == "mfia-aux":
                    ramp_set_double(daq, f"{auxbase}/offset", gate, self.cfg.ramp_step, self.cfg.ramp_wait)
                else:
                    for v in ramp_values(last_gate, gate, self.cfg.ramp_step):
                        if self._stop:
                            break
                        gate_device.set_voltage(v)
                        time.sleep(self.cfg.ramp_wait)
                last_gate = gate
                if self._stop:
                    break
                time.sleep(max(0.0, self.cfg.gate_settle))

                model_tag = MODEL_TAGS.get(self.cfg.model, f"model{self.cfg.model}")
                sweep_name = (
                    f"CV_map_{sample_name}_Vg_{gate:.4f}V_"
                    f"bias_{self.cfg.bias_start:.4f}to{self.cfg.bias_stop:.4f}V_"
                    f"ST_{self.cfg.bias_settle:.3f}s_{model_tag}"
                )
                sweep_file = os.path.join(map_dir, f"{sanitize_filename(sweep_name)}_{start_time}.data")

                with open(sweep_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for bias in bias_values:
                        if self._stop:
                            break
                        ramp_set_double(daq, f"{base}/bias/value", bias, self.cfg.ramp_step, self.cfg.ramp_wait)
                        time.sleep(max(0.0, self.cfg.bias_settle))

                        sample = get_latest_sample(daq, path_sample, timeout=max(1.0, self.cfg.bias_settle * 5))

                        param0 = float(sample.get("param0", float("nan")))
                        param1 = float(sample.get("param1", float("nan")))
                        z = sample.get("z", complex(float("nan"), float("nan")))
                        freq = float(sample.get("frequency", self.cfg.freq if self.cfg.freq is not None else float("nan")))
                        drive = float(sample.get("drive", self.cfg.drive if self.cfg.drive is not None else float("nan")))

                        try:
                            z_real = float(np.real(z))
                            z_imag = float(np.imag(z))
                            z_abs = float(np.abs(z))
                        except Exception:
                            z_real = float("nan")
                            z_imag = float("nan")
                            z_abs = float("nan")

                        cap, res = compute_model_values(self.cfg.model, param0, param1, freq)

                        leakage = float("nan")
                        if gate_device is not None:
                            try:
                                if self.cfg.gate_current_range == "auto":
                                    leakage = float(keithley.auto_range(gate_device, p1=None, compl=self.cfg.gate_compliance or 1))
                                else:
                                    leakage = float(gate_device.read_current(autorange=False))
                            except Exception as exc:
                                self._log(logger, f"Warning: gate leakage read failed: {exc}")

                        model_name, p0_name, p1_name = MODEL_INFO.get(self.cfg.model, ("Unknown", "param0", "param1"))

                        row = {
                            "timestamp": time.time(),
                            "gate_V": gate,
                            "bias_V": bias,
                            "drive_V": drive,
                            "freq_Hz": freq,
                            "model": self.cfg.model,
                            "model_name": model_name,
                            "param0": param0,
                            "param1": param1,
                            "param0_name": p0_name,
                            "param1_name": p1_name,
                            "C_F": cap,
                            "R_Ohm": res,
                            "Z_real": z_real,
                            "Z_imag": z_imag,
                            "Z_abs": z_abs,
                            "leakage_A": leakage,
                            "gate_source": self.cfg.gate_source,
                        }

                        writer.writerow(row)
                        f.flush()

                        self.point.emit(bias, gate, cap, res, z_abs, leakage)

                self._log(logger, f"Saved sweep file: {sweep_file}")

            self._log(logger, "Sweep finished")
        except Exception as exc:
            rc = 1
            if logger:
                logger.exception("Fatal error")
            self.log.emit(f"Fatal error: {exc}")
        finally:
            try:
                if path_sample and daq:
                    daq.unsubscribe(path_sample)
            except Exception:
                pass
            try:
                if gate_device is not None:
                    try:
                        gate_device.set_voltage(0)
                    except Exception:
                        pass
                    try:
                        gate_device.disable_output()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if daq and state and dev:
                    restore_state(daq, dev, self.cfg.imps, self.cfg.aux, state)
            except Exception:
                pass
            try:
                if daq and dev:
                    base = f"/{dev}/imps/{self.cfg.imps}"
                    daq.setDouble(f"{base}/bias/value", 0.0)
                    daq.setInt(f"{base}/bias/enable", 0)
                    if self.cfg.gate_source == "mfia-aux":
                        auxbase = f"/{dev}/auxouts/{self.cfg.aux}"
                        daq.setDouble(f"{auxbase}/offset", 0.0)
            except Exception:
                pass
            self.finished.emit(rc)

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MFIA C(Vbias) + Gate Sweep")

        self.thread: Optional[QThread] = None
        self.worker: Optional[SweepWorker] = None

        self.status_label = QLabel("Idle")
        self.last_label = QLabel("Last: --")

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

        self.freq_hz = QDoubleSpinBox()
        self.freq_hz.setRange(0.0, 5_000_000.0)
        self.freq_hz.setDecimals(3)
        self.freq_hz.setSingleStep(100.0)
        self.freq_hz.setValue(1000.0)
        self.freq_hz.setSuffix(" Hz")

        self.drive_v = QDoubleSpinBox()
        self.drive_v.setRange(0.0, 10.0)
        self.drive_v.setDecimals(4)
        self.drive_v.setSingleStep(0.01)
        self.drive_v.setValue(0.1)
        self.drive_v.setSuffix(" V")

        self.bias_start = QDoubleSpinBox()
        self.bias_stop = QDoubleSpinBox()
        self.bias_step = QDoubleSpinBox()
        for sb in [self.bias_start, self.bias_stop]:
            sb.setRange(-20.0, 20.0)
            sb.setDecimals(4)
            sb.setSingleStep(0.1)
        self.bias_start.setValue(0.0)
        self.bias_stop.setValue(0.0)

        self.bias_step.setRange(0.0001, 20.0)
        self.bias_step.setDecimals(4)
        self.bias_step.setSingleStep(0.01)
        self.bias_step.setValue(0.1)

        self.bias_settle = QDoubleSpinBox()
        self.bias_settle.setRange(0.0, 10.0)
        self.bias_settle.setDecimals(3)
        self.bias_settle.setSingleStep(0.05)
        self.bias_settle.setValue(0.1)
        self.bias_settle.setSuffix(" s")

        self.gate_start = QDoubleSpinBox()
        self.gate_stop = QDoubleSpinBox()
        self.gate_step = QDoubleSpinBox()
        for sb in [self.gate_start, self.gate_stop]:
            sb.setRange(-200.0, 200.0)
            sb.setDecimals(4)
            sb.setSingleStep(0.1)
        self.gate_start.setValue(0.0)
        self.gate_stop.setValue(0.0)

        self.gate_step.setRange(0.0001, 200.0)
        self.gate_step.setDecimals(4)
        self.gate_step.setSingleStep(0.1)
        self.gate_step.setValue(1.0)

        self.gate_settle = QDoubleSpinBox()
        self.gate_settle.setRange(0.0, 30.0)
        self.gate_settle.setDecimals(3)
        self.gate_settle.setSingleStep(0.1)
        self.gate_settle.setValue(0.2)
        self.gate_settle.setSuffix(" s")

        self.ramp_step = QDoubleSpinBox()
        self.ramp_step.setRange(0.0, 10.0)
        self.ramp_step.setDecimals(4)
        self.ramp_step.setSingleStep(0.01)
        self.ramp_step.setValue(0.05)
        self.ramp_step.setSuffix(" V")

        self.ramp_wait = QDoubleSpinBox()
        self.ramp_wait.setRange(0.0, 2.0)
        self.ramp_wait.setDecimals(3)
        self.ramp_wait.setSingleStep(0.01)
        self.ramp_wait.setValue(0.02)
        self.ramp_wait.setSuffix(" s")

        self.gate_source_combo = QComboBox()
        self.gate_source_combo.addItems(["Keithley 6430/2400", "MFIA AUX"])
        self.gate_source_combo.currentIndexChanged.connect(self.update_gate_source_ui)

        self.aux_spin = QSpinBox()
        self.aux_spin.setRange(0, 3)
        self.aux_spin.setValue(0)

        self.gate_device_combo = QComboBox()
        self.gate_device_combo.addItems(list_keithley_devices())

        self.gate_current_range = QComboBox()
        try:
            ranges = list((2 * 10.0 ** np.arange(-12.0, -4.0, 1.0)).astype(str))
        except Exception:
            ranges = ["1e-12", "1e-11", "1e-10", "1e-9", "1e-8", "1e-7", "1e-6", "1e-5"]
        self.gate_current_range.addItems(ranges + ["Auto-range"])
        self.gate_current_range.setCurrentText("Auto-range")

        self.gate_compliance = QLineEdit("1e-6")

        self.gate_nplc = QComboBox()
        self.gate_nplc.addItems(["0.01", "0.1", "1", "10"])
        self.gate_nplc.setCurrentText("1")

        self.sample_name = QLineEdit("sample")
        self.outdir_edit = QLineEdit(os.getcwd())
        self.outdir_btn = QPushButton("Browse")
        self.outdir_btn.clicked.connect(self.on_browse_outdir)

        self.run_btn = QPushButton("Run")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.on_run)
        self.stop_btn.clicked.connect(self.on_stop)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.setLabel("bottom", "Bias (V)")
        self.plot_widget.setLabel("left", "C (F)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setMouseEnabled(False, False)
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color=(20, 90, 160), width=2))
        self.scatter = self.plot_widget.plot(
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush(20, 90, 160),
            symbolPen=None,
        )

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.NoWrap)
        font = QFont("Consolas")
        font.setStyleHint(QFont.Monospace)
        self.log.setFont(font)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress_label = QLabel("0/0 points, now Gate, Bias = (--, --) V")

        self.bias_values = np.array([])
        self.gate_values = np.array([])
        self.curve_x = np.array([])
        self.curve_y = np.array([])
        self.current_gate = None
        self.seen = np.zeros((0, 0), dtype=bool)
        self.total_points = 0
        self.point_count = 0

        self.build_layout()
        self.update_gate_source_ui()

    def build_layout(self) -> None:
        def build_two_col_grid(items: list[tuple[str, QWidget]]) -> QGridLayout:
            grid = QGridLayout()
            for idx, (label, widget) in enumerate(items):
                row = idx // 2
                col = (idx % 2) * 2
                grid.addWidget(QLabel(label), row, col)
                grid.addWidget(widget, row, col + 1)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)
            return grid

        conn = QGroupBox("Connection")
        conn_form = QFormLayout()
        conn_form.addRow("Host", self.host_edit)
        conn_form.addRow("Port", self.port_spin)
        conn_form.addRow("Device (optional)", self.device_edit)
        conn_form.addRow("Imps index", self.imps_spin)
        conn.setLayout(conn_form)

        osc = QGroupBox("Oscillator / Model")
        osc_grid = build_two_col_grid(
            [
                ("Frequency", self.freq_hz),
                ("Amplitude", self.drive_v),
                ("Model", self.model_combo),
                ("Quality", self.quality_combo),
            ]
        )
        osc.setLayout(osc_grid)

        bias = QGroupBox("Bias Sweep (MFIA)")
        bias_grid = build_two_col_grid(
            [
                ("Bias start (V)", self.bias_start),
                ("Bias stop (V)", self.bias_stop),
                ("Bias step (V)", self.bias_step),
                ("Bias settle (s)", self.bias_settle),
            ]
        )
        bias.setLayout(bias_grid)

        gate = QGroupBox("Gate Sweep")
        gate_grid = build_two_col_grid(
            [
                ("Gate source", self.gate_source_combo),
                ("Gate start (V)", self.gate_start),
                ("Gate stop (V)", self.gate_stop),
                ("Gate step (V)", self.gate_step),
                ("Gate settle (s)", self.gate_settle),
                ("Ramp step (V)", self.ramp_step),
                ("Ramp wait (s)", self.ramp_wait),
                ("", QWidget()),
            ]
        )
        gate.setLayout(gate_grid)

        gate_aux = QGroupBox("MFIA AUX Gate")
        gate_aux_form = QFormLayout()
        gate_aux_form.addRow("Aux index", self.aux_spin)
        gate_aux.setLayout(gate_aux_form)
        self.gate_aux_group = gate_aux

        gate_keithley = QGroupBox("Keithley Gate")
        gate_k_grid = build_two_col_grid(
            [
                ("Gate device", self.gate_device_combo),
                ("Compliance (A)", self.gate_compliance),
                ("Current range", self.gate_current_range),
                ("NPLC", self.gate_nplc),
            ]
        )
        gate_keithley.setLayout(gate_k_grid)
        self.gate_keithley_group = gate_keithley

        out = QGroupBox("Output")
        out_layout = QFormLayout()
        out_layout.addRow("Sample name", self.sample_name)
        outdir_row = QHBoxLayout()
        outdir_row.addWidget(self.outdir_edit)
        outdir_row.addWidget(self.outdir_btn)
        out_layout.addRow("Output directory", outdir_row)
        out.setLayout(out_layout)

        plot_box = QGroupBox("Realtime C vs Bias")
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self.plot_widget, 1)
        plot_box.setLayout(plot_layout)

        btns = QHBoxLayout()
        btns.addWidget(self.run_btn)
        btns.addWidget(self.stop_btn)
        btns.addStretch(1)
        btns.addWidget(self.status_label)
        btns.addSpacing(10)
        btns.addWidget(self.last_label)

        main = QVBoxLayout()
        main.addWidget(conn)
        main.addWidget(osc)
        main.addWidget(bias)
        main.addWidget(gate)
        main.addWidget(self.gate_aux_group)
        main.addWidget(self.gate_keithley_group)
        main.addWidget(out)
        main.addWidget(plot_box)
        main.addWidget(self.progress)
        main.addWidget(self.progress_label)
        main.addLayout(btns)
        main.addWidget(self.log)

        root = QWidget()
        root.setLayout(main)
        self.setCentralWidget(root)
        self.resize(900, 900)

    def update_gate_source_ui(self) -> None:
        use_keithley = self.gate_source_combo.currentIndex() == 0
        self.gate_keithley_group.setVisible(use_keithley)
        self.gate_aux_group.setVisible(not use_keithley)

    def on_browse_outdir(self) -> None:
        dirname = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.outdir_edit.text())
        if dirname:
            self.outdir_edit.setText(dirname)

    def build_config(self) -> Optional[SweepConfig]:
        outdir = self.outdir_edit.text().strip()
        sample = self.sample_name.text().strip() or "sample"
        if not outdir:
            QMessageBox.warning(self, "Missing output", "Please set an output directory.")
            return None

        gate_source = "keithley" if self.gate_source_combo.currentIndex() == 0 else "mfia-aux"
        gate_device = self.gate_device_combo.currentText().strip()

        gate_compliance = None
        comp_text = self.gate_compliance.text().strip()
        if comp_text:
            try:
                gate_compliance = float(eval(comp_text))
            except Exception:
                QMessageBox.warning(self, "Compliance", f"Invalid compliance: {comp_text}")
                return None

        gate_current_range = self.gate_current_range.currentText()
        gate_current_range = "auto" if gate_current_range == "Auto-range" else gate_current_range

        return SweepConfig(
            host=self.host_edit.text().strip(),
            port=int(self.port_spin.value()),
            device=self.device_edit.text().strip(),
            imps=int(self.imps_spin.value()),
            model=int(self.model_combo.currentData()),
            freq=float(self.freq_hz.value()),
            drive=float(self.drive_v.value()),
            quality=int(self.quality_combo.currentData()),
            bias_start=float(self.bias_start.value()),
            bias_stop=float(self.bias_stop.value()),
            bias_step=float(self.bias_step.value()),
            bias_settle=float(self.bias_settle.value()),
            gate_start=float(self.gate_start.value()),
            gate_stop=float(self.gate_stop.value()),
            gate_step=float(self.gate_step.value()),
            gate_settle=float(self.gate_settle.value()),
            gate_source=gate_source,
            gate_device=gate_device,
            gate_current_range=gate_current_range,
            gate_compliance=gate_compliance,
            gate_nplc=float(self.gate_nplc.currentText()),
            aux=int(self.aux_spin.value()),
            ramp_step=float(self.ramp_step.value()),
            ramp_wait=float(self.ramp_wait.value()),
            outdir=outdir,
            sample=sample,
        )

    def on_run(self) -> None:
        if self.worker is not None:
            return

        cfg = self.build_config()
        if cfg is None:
            return

        self.log.appendPlainText("Starting sweep...")
        self.status_label.setText("Running")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.prepare_trace()

        self.thread = QThread(self)
        self.worker = SweepWorker(cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.on_log)
        self.worker.point.connect(self.on_point)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_stop(self) -> None:
        if self.worker is None:
            return
        self.log.appendPlainText("Stopping...")
        self.worker.request_stop()
        cfg = self.build_config()
        if cfg is not None:
            safe_stop_outputs(cfg)

    @pyqtSlot(str)
    def on_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    @pyqtSlot(float, float, float, float, float, float)
    def on_point(self, bias: float, gate: float, cap: float, res: float, zabs: float, leak: float) -> None:
        self.last_label.setText(
            f"Last: gate={gate:.3f} V, bias={bias:.3f} V, C={cap:.3e} F, leak={leak:.3e} A"
        )
        self.update_trace(bias, gate, cap)

    @pyqtSlot(int)
    def on_worker_finished(self, exit_code: int) -> None:
        self.status_label.setText("Idle")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.worker = None
        self.thread = None
        self.log.appendPlainText(f"Finished (exit code {exit_code}).")

    def prepare_trace(self) -> None:
        self.bias_values = np.array(build_range(self.bias_start.value(), self.bias_stop.value(), self.bias_step.value()))
        self.gate_values = np.array(build_range(self.gate_start.value(), self.gate_stop.value(), self.gate_step.value()))

        self.curve_x = self.bias_values.copy()
        self.curve_y = np.full(len(self.bias_values), np.nan, dtype=float)
        self.current_gate = None
        self.seen = np.zeros((len(self.bias_values), len(self.gate_values)), dtype=bool)
        self.total_points = int(self.seen.size)
        self.point_count = 0
        self.progress.setRange(0, max(1, self.total_points))
        self.progress.setValue(0)
        self.progress_label.setText(f"0/{self.total_points} points, now Gate, Bias = (--, --) V")
        self.curve.setData(self.curve_x, self.curve_y)
        self.scatter.setData([], [])
        self.plot_widget.setTitle("Gate V = -- V")

    def find_index(self, values: np.ndarray, target: float) -> Optional[int]:
        if values.size == 0:
            return None
        idx = int(np.argmin(np.abs(values - target)))
        if values.size == 1:
            return idx
        step = abs(values[1] - values[0])
        tol = max(1e-6, step / 2.0 + 1e-12)
        if abs(values[idx] - target) <= tol:
            return idx
        return None

    def update_trace(self, bias: float, gate: float, cap: float) -> None:
        i = self.find_index(self.bias_values, bias)
        j = self.find_index(self.gate_values, gate)
        if i is None or j is None:
            return

        if not self.seen[i, j]:
            self.seen[i, j] = True
            self.point_count += 1
            self.progress.setValue(self.point_count)
        self.progress_label.setText(
            f"{self.point_count}/{self.total_points} points, now Gate, Bias = ({gate:.3f}, {bias:.3f}) V"
        )

        if self.current_gate is None or not np.isclose(self.current_gate, gate, atol=1e-9):
            self.current_gate = gate
            self.curve_y[:] = np.nan
            self.plot_widget.setTitle(f"Gate V = {gate:.3f} V")

        if np.isfinite(cap):
            self.curve_y[i] = cap
            self.curve.setData(self.curve_x, self.curve_y)
            mask = np.isfinite(self.curve_y)
            self.scatter.setData(self.curve_x[mask], self.curve_y[mask])

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            cfg = self.build_config()
            if cfg is not None:
                safe_stop_outputs(cfg)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
