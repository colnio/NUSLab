import time
from typing import Iterable, List, Optional
import ast
import operator
import json
import os
import os.path as op


def build_device_display_list(keithley_module, include_mock: bool = True) -> List[str]:
    devices = [' - '.join(item) for item in keithley_module.get_devices_list()]
    if include_mock:
        devices.append('Mock')
    return devices


def refresh_device_combos(
    keithley_module,
    combos: Iterable,
    include_mock: bool = True,
    preserve_selection: bool = True,
) -> List[str]:
    devices = build_device_display_list(keithley_module, include_mock=include_mock)
    for combo in combos:
        prev = combo.currentText() if preserve_selection else ""
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(devices)
        if preserve_selection and prev:
            idx = combo.findText(prev)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)
    return devices


class ProgressEta:
    def __init__(self, progress_bar, eta_label, progress_label: Optional[object] = None):
        self.progress_bar = progress_bar
        self.eta_label = eta_label
        self.progress_label = progress_label
        self.total = 0
        self.done = 0
        self.started = None

    def start(self, total_steps: int):
        self.total = max(1, int(total_steps))
        self.done = 0
        self.started = time.perf_counter()
        self.progress_bar.setRange(0, self.total)
        self.progress_bar.setValue(0)
        self.eta_label.setText("ETA: --")
        if self.progress_label is not None:
            self.progress_label.setText(f"Progress: 0/{self.total}")

    def step(self, done_steps: int, extra_text: str = ""):
        self.done = min(max(0, int(done_steps)), self.total)
        self.progress_bar.setValue(self.done)
        if self.progress_label is not None:
            suffix = f" | {extra_text}" if extra_text else ""
            self.progress_label.setText(f"Progress: {self.done}/{self.total}{suffix}")
        if self.started is None or self.done <= 0:
            self.eta_label.setText("ETA: --")
            return
        elapsed = max(0.0, time.perf_counter() - self.started)
        rem = (self.total - self.done) * (elapsed / self.done)
        self.eta_label.setText(f"ETA: {self._fmt_seconds(rem)}")

    @staticmethod
    def _fmt_seconds(seconds: float) -> str:
        s = int(max(0.0, seconds))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"


def apply_standard_window_style(widget):
    """
    Apply a consistent visual language across Keithley GUIs.
    """
    widget.setStyleSheet("""
        QWidget { font-size: 12px; }
        QGroupBox {
            font-weight: 600;
            border: 1px solid #b8bec6;
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 10px;
            background: #f8fafc;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        QLabel { color: #0f172a; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            min-height: 24px;
            border: 1px solid #b8bec6;
            border-radius: 4px;
            padding: 2px 6px;
            background: #ffffff;
        }
        QPushButton {
            min-height: 26px;
            border: 1px solid #9ca3af;
            border-radius: 4px;
            background: #eef2f7;
            padding: 2px 10px;
        }
        QPushButton:hover { background: #e2e8f0; }
        QPushButton:disabled { color: #9ca3af; background: #f1f5f9; }
        QPushButton#StartButton {
            background: #d1fae5;
            border-color: #22c55e;
            font-weight: 600;
        }
        QPushButton#StartButton:hover { background: #bbf7d0; }
        QPushButton#StopButton {
            background: #fee2e2;
            border-color: #ef4444;
            font-weight: 600;
        }
        QPushButton#StopButton:hover { background: #fecaca; }
        QProgressBar {
            border: 1px solid #b8bec6;
            border-radius: 4px;
            text-align: right;
        }
        QProgressBar::chunk { background-color: #16a34a; }
    """)


def ensure_directory(path: str, label: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create {label} directory:\n{path}\n\n{exc}") from exc
    if not op.isdir(path):
        raise RuntimeError(f"{label} directory was not created:\n{path}")
    return path


def build_device_metadata(keithley_module, selection_text: str, runtime_device=None) -> dict:
    selection = str(selection_text or "").strip()
    metadata = {
        'selection_text': selection,
        'resource': keithley_module._extract_resource_token(selection) if selection else '',
        'reported_model': '',
        'reported_driver_class': '',
        'reported_idn': '',
        'vendor': '',
        'model': '',
        'serial': '',
        'firmware': '',
        'runtime_class': type(runtime_device).__name__ if runtime_device is not None else '',
        'runtime_gpib_address': getattr(runtime_device, 'gpib_address', ''),
        'runtime_resource_name': getattr(getattr(runtime_device, 'device', None), 'resource_name', ''),
    }
    if selection == 'Mock':
        metadata['reported_model'] = 'Mock'
        metadata['reported_driver_class'] = 'Mock'
        metadata['reported_idn'] = 'Mock'
    elif selection:
        parts = selection.split(' - ', 3)
        if len(parts) > 1:
            metadata['reported_model'] = parts[1].strip()
        if len(parts) > 2:
            metadata['reported_driver_class'] = parts[2].strip()
        if len(parts) > 3:
            metadata['reported_idn'] = parts[3].strip()
    vendor, model, serial, firmware = keithley_module._parse_idn(metadata['reported_idn'])
    metadata['vendor'] = vendor
    metadata['model'] = model or metadata['reported_model']
    metadata['serial'] = serial
    metadata['firmware'] = firmware
    return metadata


def write_json_file(path: str, payload: dict, label: str = 'metadata') -> str:
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2)
    except Exception as exc:
        raise RuntimeError(f"Failed to write {label} file:\n{path}\n\n{exc}") from exc
    if not op.isfile(path):
        raise RuntimeError(f"{label.capitalize()} file was not created:\n{path}")
    return path


_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_numeric_expr(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_eval_numeric_expr(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        left = _eval_numeric_expr(node.left)
        right = _eval_numeric_expr(node.right)
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)
    raise ValueError("Unsupported expression")


def parse_numeric_text(text: str, field_name: str) -> float:
    """
    Parse a user-entered numeric value safely.
    Supports plain floats and simple arithmetic expressions like '1e-3' or '1/3'.
    """
    raw = str(text).strip()
    if not raw:
        raise ValueError(f"{field_name} is empty.")
    try:
        return float(raw)
    except Exception:
        pass

    try:
        expr = ast.parse(raw, mode="eval")
        return float(_eval_numeric_expr(expr.body))
    except Exception as exc:
        raise ValueError(f"{field_name} is invalid: {raw}") from exc
