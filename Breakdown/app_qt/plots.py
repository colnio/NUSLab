"""
Live plots: four views, one per measurement phase.

Points arrive far faster than a screen can usefully redraw -- an RVS at NPLC 0.1
produces tens per second -- so the window buffers incoming rows and calls
:meth:`PlotPanel.refresh` on a timer instead of redrawing per point. Repainting
per point would make the UI fight the measurement loop for the GIL.

Earlier devices stay on the axes dimmed, so the current one can be compared
against the campaign so far at a glance -- the trick ``FastVAC.py:776`` uses for
its cycles.
"""

from __future__ import annotations

from typing import Dict, List

import pyqtgraph as pg
from PyQt5.QtWidgets import QGridLayout, QWidget

from .theme import BORDER, DANGER, FOCUS, SURFACE, TEXT_DIM, TEXT_FAINT

CURRENT_PEN = pg.mkPen(color=FOCUS, width=2)
PREVIOUS_PEN = pg.mkPen(color=(94, 107, 124, 110), width=1)
BREAKDOWN_BRUSH = pg.mkBrush(DANGER)


class PlotPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        pg.setConfigOptions(antialias=True, background=SURFACE, foreground=TEXT_DIM)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.cf_plot = self._make_plot("C(f)", "Frequency (Hz)", "C (F)",
                                       log_x=True, log_y=True)
        self.cv_plot = self._make_plot("C(V)", "Bias (V)", "C (F)")
        self.rvs_plot = self._make_plot("RVS", "Voltage (V)", "|I| (A)", log_y=True)
        self.cvs_plot = self._make_plot("CVS", "Time (s)", "|I| (A)", log_y=True)
        layout.addWidget(self.cf_plot, 0, 0)
        layout.addWidget(self.cv_plot, 0, 1)
        layout.addWidget(self.rvs_plot, 1, 0)
        layout.addWidget(self.cvs_plot, 1, 1)

        self._current: Dict[str, List[list]] = {}
        self._curves: Dict[str, object] = {}
        self._markers: Dict[str, object] = {}
        self.reset_device()

    def _make_plot(self, title, x_label, y_label, log_x=False, log_y=False):
        widget = pg.PlotWidget()
        widget.setTitle(title, color=TEXT_DIM, size="10pt")
        widget.setLabel("bottom", x_label, color=TEXT_FAINT)
        widget.setLabel("left", y_label, color=TEXT_FAINT)
        widget.setLogMode(x=log_x, y=log_y)
        widget.showGrid(x=True, y=True, alpha=0.14)
        widget.setStyleSheet(f"border: 1px solid {BORDER}; border-radius: 6px;")
        for axis in ("bottom", "left"):
            widget.getAxis(axis).setPen(pg.mkPen(BORDER))
            widget.getAxis(axis).setTextPen(pg.mkPen(TEXT_FAINT))
        return widget

    # -- lifecycle ----------------------------------------------------------

    def clear_all(self) -> None:
        for plot in (self.cf_plot, self.cv_plot, self.rvs_plot, self.cvs_plot):
            plot.clear()
        self._curves.clear()
        self._markers.clear()
        self.reset_device()

    def reset_device(self) -> None:
        """Fade the finished device's traces and start fresh curves."""
        for curve in self._curves.values():
            curve.setPen(PREVIOUS_PEN)
            curve.setZValue(-1)
        self._curves = {}
        self._markers = {}
        self._current = {"CF": [[], []], "CV": [[], []],
                         "RVS": [[], []], "CVS": [[], []]}

    # -- data ---------------------------------------------------------------

    def add_point(self, phase: str, row: dict) -> None:
        buffer = self._current.get(phase)
        if buffer is None:
            return
        if phase == "CF":
            x, y = row.get("set_frequency_Hz"), row.get("C_F")
        elif phase == "CV":
            x, y = row.get("sweep_value"), row.get("C_F")
        elif phase == "RVS":
            x, y = row.get("set_V"), abs(row.get("measured_I_A") or 0.0)
        else:
            x, y = row.get("elapsed_s"), abs(row.get("measured_I_A") or 0.0)

        if x is None or y is None:
            return
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return
        # Log axes cannot show zero or negative values.
        if y <= 0 and phase in ("CF", "RVS", "CVS"):
            return
        buffer[0].append(x)
        buffer[1].append(y)

        if row.get("bd_flag"):
            self._mark_breakdown(phase, x, y)

    def _mark_breakdown(self, phase: str, x: float, y: float) -> None:
        if phase in self._markers:
            return
        plot = {"RVS": self.rvs_plot, "CVS": self.cvs_plot}.get(phase)
        if plot is None:
            return
        scatter = pg.ScatterPlotItem([x], [y], size=13, symbol="x",
                                     brush=BREAKDOWN_BRUSH, pen=None)
        plot.addItem(scatter)
        self._markers[phase] = scatter

    def refresh(self) -> None:
        """Redraw every phase that has new data. Called on a timer."""
        plots = {"CF": self.cf_plot, "CV": self.cv_plot,
                 "RVS": self.rvs_plot, "CVS": self.cvs_plot}
        for phase, (xs, ys) in self._current.items():
            if not xs:
                continue
            curve = self._curves.get(phase)
            if curve is None:
                curve = plots[phase].plot([], [], pen=CURRENT_PEN)
                self._curves[phase] = curve
            curve.setData(xs, ys)
