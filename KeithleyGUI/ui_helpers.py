import time
from typing import Iterable, List, Optional


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
