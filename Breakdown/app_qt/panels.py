"""
Parameter editing, driven by a declarative field table.

Every widget is generated from :data:`SECTIONS`, so adding a parameter means one
line here and one field in :mod:`Breakdown.core.params` -- there is no third
place to forget. Reading and writing go through the same table, which is what
makes Save/Load round-trip reliably.

Units live in their own column rather than inside the label text: it keeps the
labels scannable and puts every unit on the same vertical line.

Numeric entry uses ``ui_helpers.parse_numeric_text``, the AST-based parser the
other apps use, so ``1e-4`` and ``1/3`` both work without ``eval``.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Dict, List, NamedTuple

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from Breakdown.core.models import MODEL_INFO, QUALITY_LABELS
from Breakdown.core.params import STATISTICS, BreakdownParams
from KeithleyGUI.ui_helpers import parse_numeric_text

from .theme import Card, field_row, monospace

NUMERIC_WIDTH = 150
TEXT_WIDTH = 240


class Field(NamedTuple):
    section: str
    name: str
    label: str
    kind: str = "float"          # float | int | str | bool | choice
    unit: str = ""
    choices: tuple = ()
    tooltip: str = ""


MODE_LABELS = (
    ("Alternating RVS / CVS", "alternating"),
    ("RVS only", "rvs_only"),
    ("CVS only (first device still ramps)", "cvs_only"),
    ("Manual - ask per device", "manual"),
)

#: title -> (subtitle, fields). Order here is the order on screen.
SECTIONS: Dict[str, tuple] = {
    "Sample": ("what is under the probes", [
        Field("sample", "sample_name", "Sample name", "str"),
        Field("sample", "crosspoint_um", "Crosspoint size", "float", "µm",
              tooltip="Devices are square. Area = size², used for C/A and J."),
        Field("sample", "thickness_nm", "Dielectric thickness", "float", "nm",
              tooltip="Optional. Fills in the breakdown field E_BD."),
        Field("sample", "operator", "Operator", "str"),
        Field("sample", "notes", "Notes", "str"),
    ]),
    "Run": ("sequence and limits", [
        Field("run", "mode", "Mode", "choice", "", MODE_LABELS),
        Field("run", "enable_cf", "Measure C(f)", "bool"),
        Field("run", "enable_cv", "Measure C(V)", "bool"),
        Field("run", "abs_max_voltage_V", "Voltage ceiling", "float", "V",
              tooltip="Hard cap on any sourced voltage, checked before starting."),
    ]),
    "C(f)": ("log sweep, f_min → f_max → f_min, fixed bias", [
        Field("cf", "f_min", "Start frequency", "float", "Hz"),
        Field("cf", "f_max", "Stop frequency", "float", "Hz"),
        Field("cf", "points", "Points per direction", "int"),
        Field("cf", "bias_V", "DC bias", "float", "V"),
        Field("cf", "amplitude_V", "Drive amplitude", "float", "V"),
        Field("cf", "settle_s", "Settle time", "float", "s"),
    ]),
    "C(V)": ("linear loop, 0 → Vmax → Vmin → 0, fixed frequency", [
        Field("cv", "v_min", "Minimum bias", "float", "V"),
        Field("cv", "v_max", "Maximum bias", "float", "V"),
        Field("cv", "points", "Points per segment", "int"),
        Field("cv", "frequency_Hz", "Frequency", "float", "Hz"),
        Field("cv", "amplitude_V", "Drive amplitude", "float", "V"),
        Field("cv", "settle_s", "Settle time", "float", "s"),
    ]),
    "RVS": ("ramped stress — measures V_BD", [
        Field("rvs", "polarity", "Polarity", "choice", "",
              (("Positive  +", 1), ("Negative  −", -1))),
        Field("rvs", "v_start", "Start voltage", "float", "V"),
        Field("rvs", "v_max", "Ceiling", "float", "V"),
        Field("rvs", "ramp_rate_Vps", "Ramp rate", "float", "V/s",
              tooltip="The achieved rate is measured and recorded — V_BD "
                      "depends on it."),
        Field("rvs", "max_step_V", "Maximum step", "float", "V",
              tooltip="Caps how coarse the ramp may become at high rates."),
        Field("rvs", "nplc", "NPLC", "float"),
        Field("rvs", "compliance_A", "Compliance", "float", "A"),
        Field("rvs", "i_bd_A", "Breakdown threshold", "float", "A",
              tooltip="Must sit below compliance, or breakdown can never trigger."),
        Field("rvs", "source_delay_s", "Source delay", "float", "s"),
        Field("rvs", "current_autorange", "Current autorange", "bool"),
    ]),
    "CVS": ("constant stress — measures t_BD", [
        Field("cvs", "pre_ramp_rate_Vps", "Pre-ramp rate", "float", "V/s",
              tooltip="Fast on purpose: charge injected on the way up would "
                      "contaminate t_BD."),
        Field("cvs", "sample_interval_s", "Sample interval", "float", "s"),
        Field("cvs", "max_duration_s", "Maximum duration", "float", "s"),
        Field("cvs", "nplc", "NPLC", "float"),
        Field("cvs", "compliance_A", "Compliance", "float", "A"),
        Field("cvs", "i_bd_A", "Breakdown threshold", "float", "A"),
        Field("cvs", "current_autorange", "Current autorange", "bool"),
    ]),
    "Stress level advisor": ("how the CVS suggestion is derived", [
        Field("advisor", "k_fraction", "k   (V_CVS = k × V_BD)", "float", "",
              tooltip="0.80–0.92 puts t_BD in a practical window. Vary it across "
                      "the batch to get at least three field levels."),
        Field("advisor", "statistic", "Statistic", "choice", "",
              tuple((s.capitalize(), s) for s in STATISTICS)),
    ]),
    "MFIA": ("impedance analyser — rarely changed", [
        Field("mfia", "host", "Data server host", "str"),
        Field("mfia", "port", "Port", "int"),
        Field("mfia", "device_id", "Device ID", "str",
              tooltip="Blank uses the first MFIA that discovery finds."),
        Field("mfia", "imps", "Impedance module", "int"),
        Field("mfia", "model", "Model", "choice", "",
              tuple((info[0], key) for key, info in sorted(MODEL_INFO.items()))),
        Field("mfia", "quality", "Quality", "choice", "",
              tuple((label, key) for key, label in sorted(QUALITY_LABELS.items()))),
        Field("mfia", "auto_bw", "Auto bandwidth", "bool"),
        Field("mfia", "inputrange_mode", "Input range", "choice", "",
              (("Manual", 0), ("Auto", 1), ("Zone", 2))),
        Field("mfia", "manual_current_range", "Manual current range", "float", "A"),
        Field("mfia", "four_terminal", "4-terminal wiring", "bool",
              tooltip="4-terminal limits DC bias to ±3 V; 2-terminal to ±10 V."),
        Field("mfia", "demod_order", "Demod order", "int"),
        Field("mfia", "demod_timeconstant", "Demod time constant", "float", "s"),
        Field("mfia", "demod_rate", "Demod rate", "float", "Sa/s"),
        Field("mfia", "demod_sinc", "Sinc filter", "bool"),
        Field("mfia", "ramp_step", "Bias ramp step", "float", "V"),
        Field("mfia", "ramp_wait", "Bias ramp wait", "float", "s"),
    ]),
}

#: Two columns: what to measure on the left, how to stress it on the right.
#: MFIA sits last because it is set once and then left alone.
LEFT_COLUMN = ("Sample", "Run", "C(f)", "C(V)")
RIGHT_COLUMN = ("RVS", "CVS", "Stress level advisor", "MFIA")


class ParamsPanel(QWidget):
    """Builds every parameter widget from SECTIONS and syncs both ways."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.widgets: Dict[str, QWidget] = {}

        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)

        for column, titles in enumerate((LEFT_COLUMN, RIGHT_COLUMN)):
            holder = QWidget()
            stack = QVBoxLayout(holder)
            stack.setContentsMargins(0, 0, 0, 0)
            stack.setSpacing(16)
            for title in titles:
                stack.addWidget(self._build_card(title))
            stack.addStretch(1)
            grid.addWidget(holder, 0, column)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

    def _build_card(self, title: str) -> Card:
        note, specs = SECTIONS[title]
        card = Card(title, note)
        for spec in specs:
            widget = self._make_widget(spec)
            if spec.tooltip:
                widget.setToolTip(spec.tooltip)
            self.widgets[f"{spec.section}.{spec.name}"] = widget
            card.add(field_row(spec.label, widget, spec.unit))
        return card

    def _make_widget(self, spec: Field) -> QWidget:
        if spec.kind == "bool":
            return QCheckBox()
        if spec.kind == "choice":
            combo = QComboBox()
            for label, value in spec.choices:
                combo.addItem(label, value)
            combo.setFixedWidth(TEXT_WIDTH)
            return combo
        edit = QLineEdit()
        if spec.kind == "str":
            edit.setFixedWidth(TEXT_WIDTH)
        else:
            edit.setFixedWidth(NUMERIC_WIDTH)
            monospace(edit)
        return edit

    # -- syncing ------------------------------------------------------------

    def load(self, params: BreakdownParams) -> None:
        """Populate every widget from ``params``.

        Signals stay blocked for the whole pass. Otherwise filling the first
        text field fires ``textChanged``, whose handler reads *all* the widgets
        back into ``params`` -- including the ones this loop has not reached
        yet, overwriting them with whatever they happened to contain.
        """
        for key, widget in self.widgets.items():
            section, name = key.split(".", 1)
            value = getattr(getattr(params, section), name)
            was_blocked = widget.blockSignals(True)
            try:
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findData(value)
                    widget.setCurrentIndex(index if index >= 0 else 0)
                else:
                    widget.setText("" if value is None else str(value))
            finally:
                widget.blockSignals(was_blocked)

    def apply_to(self, params: BreakdownParams) -> List[str]:
        """Write the widgets back. Returns messages for anything unreadable."""
        problems: List[str] = []
        specs = {f"{f.section}.{f.name}": f
                 for _note, fields in SECTIONS.values() for f in fields}

        for key, widget in self.widgets.items():
            section_name, name = key.split(".", 1)
            section = getattr(params, section_name)
            spec = specs[key]
            try:
                if isinstance(widget, QCheckBox):
                    setattr(section, name, widget.isChecked())
                elif isinstance(widget, QComboBox):
                    setattr(section, name, widget.currentData())
                elif spec.kind == "str":
                    setattr(section, name, widget.text().strip())
                else:
                    text = widget.text().strip()
                    if not text:
                        if _is_optional(section, name):
                            setattr(section, name, None)
                            _mark(widget, valid=True)
                            continue
                        problems.append(f"{spec.label} is empty.")
                        _mark(widget, valid=False)
                        continue
                    value = parse_numeric_text(text, spec.label)
                    setattr(section, name,
                            int(round(value)) if spec.kind == "int" else value)
                    _mark(widget, valid=True)
            except Exception as exc:
                problems.append(str(exc))
                _mark(widget, valid=False)
        return problems

    def mark_invalid(self, keys) -> None:
        """Outline the named fields in red -- used to point at validation errors."""
        for key in keys:
            widget = self.widgets.get(key)
            if widget is not None:
                _mark(widget, valid=False)


def _mark(widget: QWidget, valid: bool) -> None:
    if not isinstance(widget, QLineEdit):
        return
    if bool(widget.property("invalid")) == (not valid):
        return
    widget.setProperty("invalid", not valid)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def _is_optional(section, name: str) -> bool:
    for spec in dataclass_fields(section):
        if spec.name == name:
            return "Optional" in str(spec.type) or "None" in str(spec.type)
    return False
