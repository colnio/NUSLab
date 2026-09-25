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

from .theme import current_theme, normalize_theme, theme_colors


class PlotPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        pg.setConfigOptions(antialias=True)
        self._theme_name = current_theme()
        self._plot_specs = {}

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
        self._plot_specs[widget] = (title, x_label, y_label)
        widget.setLogMode(x=log_x, y=log_y)
        widget.showGrid(x=True, y=True, alpha=0.14)
        self._style_plot(widget)
        return widget

    def _style_plot(self, widget) -> None:
        title, x_label, y_label = self._plot_specs[widget]
        colours = theme_colors(self._theme_name)
        widget.setBackground(colours.surface)
        widget.setTitle(title, color=colours.text_dim, size="10pt")
        widget.setLabel("bottom", x_label, color=colours.text_faint)
        widget.setLabel("left", y_label, color=colours.text_faint)
        widget.setStyleSheet(
            f"border: 1px solid {colours.border}; border-radius: 6px;"
        )
        for axis in ("bottom", "left"):
            widget.getAxis(axis).setPen(pg.mkPen(colours.border))
            widget.getAxis(axis).setTextPen(pg.mkPen(colours.text_faint))

    def apply_theme(self, theme_name: str) -> None:
        """Restyle existing axes, curves, and markers without losing data."""
        self._theme_name = normalize_theme(theme_name)
        current_curves = set(self._curves.values())
        markers = set(self._markers.values())
        for plot in self._plot_specs:
            self._style_plot(plot)
            for item in plot.listDataItems():
                if item in current_curves:
                    item.setPen(self._current_pen())
                elif item not in markers:
                    item.setPen(self._previous_pen())
        for marker in markers:
            marker.setBrush(self._breakdown_brush())

    def _current_pen(self):
        return pg.mkPen(color=theme_colors(self._theme_name).focus, width=2)

    def _previous_pen(self):
        return pg.mkPen(
            color=theme_colors(self._theme_name).plot_previous, width=1
        )

    def _breakdown_brush(self):
        return pg.mkBrush(theme_colors(self._theme_name).danger)

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
            curve.setPen(self._previous_pen())
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
                                     brush=self._breakdown_brush(), pen=None)
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
                curve = plots[phase].plot([], [], pen=self._current_pen())
                self._curves[phase] = curve
            curve.setData(xs, ys)
