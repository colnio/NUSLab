import os
import re
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QProgressBar,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MFIA C(Vg, Bias) Sweep")
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self.on_proc_output)
        self.proc.finished.connect(self.on_proc_finished)
        self.proc.setWorkingDirectory(str(Path(__file__).resolve().parent))

        self.last_line_regex = re.compile(
            r"bias=([+-]?[0-9.]+)\sV,\s+vg=([+-]?[0-9.]+)\sV,\s+param1\(C\?\)=([+-]?[0-9.eE+-]+)"
        )

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

        self.aux_spin = QSpinBox()
        self.aux_spin.setRange(0, 3)
        self.aux_spin.setValue(0)

        self.bias_start = QDoubleSpinBox()
        self.bias_stop = QDoubleSpinBox()
        self.bias_steps = QSpinBox()
        self.vg_start = QDoubleSpinBox()
        self.vg_stop = QDoubleSpinBox()
        self.vg_steps = QSpinBox()
        self.freq_hz = QDoubleSpinBox()
        self.settle = QDoubleSpinBox()
        self.ramp = QDoubleSpinBox()

        for sb in [self.bias_start, self.bias_stop, self.vg_start, self.vg_stop]:
            sb.setRange(-20.0, 20.0)
            sb.setDecimals(4)
            sb.setSingleStep(0.1)

        self.bias_steps.setRange(1, 100000)
        self.vg_steps.setRange(1, 100000)
        self.bias_steps.setValue(1)
        self.vg_steps.setValue(1)

        self.freq_hz.setRange(0.0, 5_000_000.0)
        self.freq_hz.setDecimals(3)
        self.freq_hz.setSingleStep(100.0)
        self.freq_hz.setValue(1000.0)
        self.freq_hz.setSuffix(" Hz")

        self.settle.setRange(0.0, 10.0)
        self.settle.setDecimals(3)
        self.settle.setSingleStep(0.05)
        self.settle.setValue(0.1)

        self.ramp.setRange(0.0, 10.0)
        self.ramp.setDecimals(3)
        self.ramp.setSingleStep(0.01)
        self.ramp.setValue(0.05)

        self.model_combo = QComboBox()
        for label, value in build_model_items():
            self.model_combo.addItem(label, value)

        self.outfile_edit = QLineEdit("cv_map.csv")
        self.outfile_btn = QPushButton("Browse")
        self.outfile_btn.clicked.connect(self.on_browse)

        self.run_btn = QPushButton("Run")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.on_run)
        self.stop_btn.clicked.connect(self.on_stop)

        self.bias_values = np.array([])
        self.vg_values = np.array([])
        self.curve_x = np.array([])
        self.curve_y = np.array([])
        self.current_vg = None
        self.seen = np.zeros((0, 0), dtype=bool)

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
        self.progress_label = QLabel("0/0 points, now Vg, Vsd = (--, --) V")

        self.build_layout()

    def build_layout(self) -> None:
        conn = QGroupBox("Connection")
        conn_form = QFormLayout()
        conn_form.addRow("Host", self.host_edit)
        conn_form.addRow("Port", self.port_spin)
        conn_form.addRow("Device (optional)", self.device_edit)
        conn_form.addRow("Imps index", self.imps_spin)
        conn_form.addRow("Aux index (Vg)", self.aux_spin)
        conn.setLayout(conn_form)

        sweep = QGroupBox("Sweep")
        sweep_form = QFormLayout()
        sweep_form.addRow("Bias start (V)", self.bias_start)
        sweep_form.addRow("Bias stop (V)", self.bias_stop)
        sweep_form.addRow("Bias steps", self.bias_steps)
        sweep_form.addRow("Vg start (V)", self.vg_start)
        sweep_form.addRow("Vg stop (V)", self.vg_stop)
        sweep_form.addRow("Vg steps", self.vg_steps)
        sweep_form.addRow("Osc frequency", self.freq_hz)
        sweep_form.addRow("Settling (s)", self.settle)
        sweep_form.addRow("Ramp step (V)", self.ramp)
        sweep_form.addRow("Impedance model", self.model_combo)
        sweep.setLayout(sweep_form)

        out = QGroupBox("Output")
        out_layout = QHBoxLayout()
        out_layout.addWidget(self.outfile_edit)
        out_layout.addWidget(self.outfile_btn)
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
        main.addWidget(sweep)
        main.addWidget(out)
        main.addWidget(plot_box)
        main.addWidget(self.progress)
        main.addWidget(self.progress_label)
        main.addLayout(btns)
        main.addWidget(self.log)

        root = QWidget()
        root.setLayout(main)
        self.setCentralWidget(root)
        self.resize(820, 720)

    def on_browse(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV", self.outfile_edit.text(), "CSV Files (*.csv)")
        if filename:
            self.outfile_edit.setText(filename)

    def on_run(self) -> None:
        if self.proc.state() != QProcess.NotRunning:
            return

        python_path = default_python_path()
        if not Path(python_path).exists():
            QMessageBox.critical(self, "Python not found", f"Python not found: {python_path}")
            return

        args = self.build_args()
        if not args:
            return

        self.log.appendPlainText("Starting sweep...")
        self.status_label.setText("Running")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.prepare_trace()
        self.proc.start(python_path, args)

    def on_stop(self) -> None:
        if self.proc.state() == QProcess.NotRunning:
            return
        self.proc.terminate()
        if not self.proc.waitForFinished(2000):
            self.proc.kill()
        self.log.appendPlainText("Stopped by user.")

    def on_proc_output(self) -> None:
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.log.appendPlainText(text.rstrip("\n"))
            for line in text.splitlines():
                m = self.last_line_regex.search(line)
                if m:
                    self.last_label.setText(f"Last: bias={m.group(1)} V, vg={m.group(2)} V, Cp={m.group(3)}")
                    self.update_trace(float(m.group(1)), float(m.group(2)), float(m.group(3)))

    def on_proc_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self.status_label.setText("Idle")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if exit_status == QProcess.CrashExit:
            self.log.appendPlainText("Process crashed.")
        else:
            self.log.appendPlainText(f"Finished (exit code {exit_code}).")

    def build_args(self) -> list[str]:
        outfile = self.outfile_edit.text().strip()
        if not outfile:
            QMessageBox.warning(self, "Missing output", "Please set an output file.")
            return []

        args = [
            "cv_map.py",
            "--host",
            self.host_edit.text().strip(),
            "--port",
            str(self.port_spin.value()),
            "--imps",
            str(self.imps_spin.value()),
            "--aux",
            str(self.aux_spin.value()),
            "--bias-start",
            f"{self.bias_start.value():.6f}",
            "--bias-stop",
            f"{self.bias_stop.value():.6f}",
            "--bias-steps",
            str(self.bias_steps.value()),
            "--vg-start",
            f"{self.vg_start.value():.6f}",
            "--vg-stop",
            f"{self.vg_stop.value():.6f}",
            "--vg-steps",
            str(self.vg_steps.value()),
            "--freq",
            f"{self.freq_hz.value():.6f}",
            "--settle",
            f"{self.settle.value():.6f}",
            "--ramp-step",
            f"{self.ramp.value():.6f}",
            "--model",
            str(self.model_combo.currentData()),
            "--outfile",
            outfile,
        ]

        device = self.device_edit.text().strip()
        if device:
            args.extend(["--device", device])

        return args

    def prepare_trace(self) -> None:
        bias_start = self.bias_start.value()
        bias_stop = self.bias_stop.value()
        bias_steps = self.bias_steps.value()
        vg_start = self.vg_start.value()
        vg_stop = self.vg_stop.value()
        vg_steps = self.vg_steps.value()

        if bias_steps <= 1:
            self.bias_values = np.array([bias_start], dtype=float)
        else:
            self.bias_values = np.linspace(bias_start, bias_stop, bias_steps)

        if vg_steps <= 1:
            self.vg_values = np.array([vg_start], dtype=float)
        else:
            self.vg_values = np.linspace(vg_start, vg_stop, vg_steps)

        self.curve_x = self.bias_values.copy()
        self.curve_y = np.full(len(self.bias_values), np.nan, dtype=float)
        self.current_vg = None
        self.seen = np.zeros((len(self.bias_values), len(self.vg_values)), dtype=bool)
        self.total_points = int(self.seen.size)
        self.point_count = 0
        self.progress.setRange(0, max(1, self.total_points))
        self.progress.setValue(0)
        self.progress_label.setText(f"0/{self.total_points} points, now Vg, Bias = (--, --) V")
        self.curve.setData(self.curve_x, self.curve_y)
        self.scatter.setData([], [])
        self.plot_widget.setTitle("Vg = -- V")

    def find_index(self, values: np.ndarray, target: float) -> int | None:
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

    def update_trace(self, bias: float, vg: float, cp: float) -> None:
        i = self.find_index(self.bias_values, bias)
        j = self.find_index(self.vg_values, vg)
        if i is None or j is None:
            return

        if not self.seen[i, j]:
            self.seen[i, j] = True
            self.point_count += 1
            self.progress.setValue(self.point_count)
        self.progress_label.setText(
            f"{self.point_count}/{self.total_points} points, now Vg, Bias = ({vg:.3f}, {bias:.3f}) V"
        )
        if self.current_vg is None or not np.isclose(self.current_vg, vg, atol=1e-9):
            self.current_vg = vg
            self.curve_y[:] = np.nan
            self.plot_widget.setTitle(f"Vg = {vg:.3f} V")

        self.curve_y[i] = cp
        self.curve.setData(self.curve_x, self.curve_y)
        mask = np.isfinite(self.curve_y)
        self.scatter.setData(self.curve_x[mask], self.curve_y[mask])

    def closeEvent(self, event) -> None:
        if self.proc.state() != QProcess.NotRunning:
            self.proc.terminate()
            if not self.proc.waitForFinished(2000):
                self.proc.kill()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
