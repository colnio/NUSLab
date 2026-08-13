"""
The three blocking operator prompts.

Each is modal and deliberately hard to dismiss by accident: answering one wrong
either destroys a device or leaves an instrument driving a probe the operator is
about to touch.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
)

from Breakdown.core.events import CvsVoltageContext, Instrument, StressType

from .theme import TEXT_DIM, TEXT_FAINT, WAIT, Card

CVS_GUIDANCE = """\
How to choose a constant-voltage stress level

Aim for a time-to-breakdown between roughly 10 s and 10,000 s. Much faster and
the device dies during the pre-ramp, so you have measured the ramp rather than
the hold; much slower and the campaign stops being practical.

The usual rule is V_CVS = k x median(V_BD), taking V_BD from RVS on sister
devices of the SAME sample and the SAME crosspoint size, with k between 0.80
and 0.92. The median is used rather than the mean so that one bad device -- a
poor probe landing, a particle -- does not drag the whole campaign.

To extrapolate a lifetime at operating field you need at least three different
k values, so the batch spans a range of fields. Fit all three standard models
and quote the most conservative:

  E-model (thermochemical)     ln t_BD  proportional to  -gamma * E
  1/E-model (anode hole inj.)  ln t_BD  proportional to  G / E
  Power law                    t_BD     proportional to  V^-n

Weibull statistics need 5-10 devices per level. The shape parameter beta is
also what lets you area-scale results between crosspoint sizes, via
t63(A1)/t63(A2) = (A2/A1)^(1/beta).
"""


class CableSwapDialog(QDialog):
    """Blocks until the operator confirms the probes are on the right instrument."""

    def __init__(self, target: Instrument, device_index: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reconnect the probes")
        self.setModal(True)
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(14)

        headline = QLabel(f"Connect the probes to the {target.value}")
        headline.setStyleSheet(
            f"font-size: 19px; font-weight: 700; color: {WAIT};"
        )
        headline.setWordWrap(True)
        layout.addWidget(headline)

        detail = QLabel(
            f"Device {device_index}.\n\n"
            f"Both instruments have already been driven to 0 V and their outputs "
            f"opened, so it is safe to change the wiring now.\n\n"
            f"Leave the probe needles where they are — only the instrument cables "
            f"move."
        )
        detail.setWordWrap(True)
        layout.addWidget(detail)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        layout.addWidget(line)

        buttons = QDialogButtonBox()
        self.confirm = buttons.addButton(
            f"{target.value} is connected — continue", QDialogButtonBox.AcceptRole
        )
        self.confirm.setObjectName("StartButton")
        abort = buttons.addButton("Abort run", QDialogButtonBox.RejectRole)
        abort.setObjectName("StopButton")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.confirm.setDefault(True)
        self.confirm.setFocus()

    def keyPressEvent(self, event):
        # Escape would otherwise abort a campaign on a stray keystroke.
        if event.key() == Qt.Key_Escape:
            return
        super().keyPressEvent(event)


class StressTypeDialog(QDialog):
    """Manual mode: which stress does this device get?"""

    def __init__(self, device_index: int, can_run_cvs: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Device {device_index}: choose the stress")
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"<b>Device {device_index}</b>"))

        self.rvs = QRadioButton("Ramped voltage stress (RVS) — measures V_BD")
        self.cvs = QRadioButton("Constant voltage stress (CVS) — measures t_BD")
        self.rvs.setChecked(True)
        layout.addWidget(self.rvs)
        layout.addWidget(self.cvs)

        if not can_run_cvs:
            self.cvs.setEnabled(False)
            note = QLabel(
                "CVS needs at least one measured breakdown voltage at this "
                "crosspoint size. Run an RVS device first."
            )
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {WAIT};")
            layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected(self) -> StressType:
        return StressType.CVS if self.cvs.isChecked() else StressType.RVS


class CvsVoltageDialog(QDialog):
    """Confirm the stress level, offering three ways to arrive at it."""

    def __init__(self, ctx: CvsVoltageContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle(f"Device {ctx.device_index}: constant voltage stress")
        self.setModal(True)
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        header = QLabel(
            f"<b>Device {ctx.device_index}</b> &nbsp;|&nbsp; "
            f"{ctx.crosspoint_um:g} µm crosspoint"
        )
        layout.addWidget(header)

        choices = Card("Stress voltage", "pick one")
        choice_layout = choices.body
        self.group = QButtonGroup(self)

        # 1. what was used last time
        self.previous_button = QRadioButton()
        if ctx.previous_voltage is not None:
            self.previous_button.setText(
                f"Same as the previous CVS device:  {ctx.previous_voltage:.4g} V"
            )
        else:
            self.previous_button.setText("Same as the previous CVS device (none yet)")
            self.previous_button.setEnabled(False)
        choice_layout.addWidget(self.previous_button)

        # 2. derived from the ramps measured so far
        self.recommended_button = QRadioButton()
        recommended = ctx.recommendation.voltage
        if recommended is not None:
            self.recommended_button.setText(f"Recommended:  {recommended:.4g} V")
        else:
            self.recommended_button.setText("Recommended (not available)")
            self.recommended_button.setEnabled(False)
        choice_layout.addWidget(self.recommended_button)

        basis = QLabel(ctx.recommendation.basis_text)
        basis.setWordWrap(True)
        basis.setStyleSheet(f"color: {TEXT_FAINT}; margin-left: 24px;")
        choice_layout.addWidget(basis)

        # 3. type one
        custom_row = QHBoxLayout()
        self.custom_button = QRadioButton("Use this value:")
        self.custom_edit = QLineEdit()
        self.custom_edit.setPlaceholderText("e.g. 3.6")
        self.custom_edit.setMaximumWidth(140)
        custom_row.addWidget(self.custom_button)
        custom_row.addWidget(self.custom_edit)
        custom_row.addWidget(QLabel("V"))
        custom_row.addStretch(1)
        choice_layout.addLayout(custom_row)

        for button in (self.previous_button, self.recommended_button,
                       self.custom_button):
            self.group.addButton(button)
        layout.addWidget(choices)

        if ctx.recommendation.warnings:
            warning = QLabel("\n".join(f"- {w}" for w in ctx.recommendation.warnings))
            warning.setWordWrap(True)
            warning.setStyleSheet(
                f"color: {WAIT}; background: #2A2107; border: 1px solid #4A3A0C;"
                f" border-radius: 5px; padding: 8px 10px;"
            )
            layout.addWidget(warning)

        if ctx.records:
            measured = ", ".join(f"{r.v_bd:.3g}" for r in ctx.records)
            summary = QLabel(f"Measured V_BD so far ({len(ctx.records)}): {measured} V")
            summary.setWordWrap(True)
            summary.setStyleSheet(f"color: {TEXT_DIM};")
            layout.addWidget(summary)

        self.guidance = QTextEdit()
        self.guidance.setReadOnly(True)
        self.guidance.setPlainText(CVS_GUIDANCE)
        self.guidance.setVisible(False)
        self.guidance.setMinimumHeight(240)
        self.help_button = QPushButton("How to choose this ...")
        self.help_button.setCheckable(True)
        self.help_button.toggled.connect(self.guidance.setVisible)
        self.help_button.toggled.connect(self._resize_for_guidance)
        layout.addWidget(self.help_button)
        layout.addWidget(self.guidance)

        buttons = QDialogButtonBox()
        self.ok = buttons.addButton("Start stress", QDialogButtonBox.AcceptRole)
        self.ok.setObjectName("StartButton")
        cancel = buttons.addButton("Abort run", QDialogButtonBox.RejectRole)
        cancel.setObjectName("StopButton")
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Prefer the recommendation, fall back to the previous value.
        if self.recommended_button.isEnabled():
            self.recommended_button.setChecked(True)
        elif self.previous_button.isEnabled():
            self.previous_button.setChecked(True)
        else:
            self.custom_button.setChecked(True)
            self.custom_edit.setFocus()

    def _resize_for_guidance(self, shown: bool) -> None:
        self.adjustSize()

    def _accept_if_valid(self) -> None:
        if self.selected_voltage() is not None:
            self.accept()

    def selected_voltage(self) -> Optional[float]:
        if self.previous_button.isChecked():
            return self.ctx.previous_voltage
        if self.recommended_button.isChecked():
            return self.ctx.recommendation.voltage
        try:
            from KeithleyGUI.ui_helpers import parse_numeric_text

            value = abs(parse_numeric_text(self.custom_edit.text(), "Stress voltage"))
        except Exception:
            self.custom_edit.setProperty("invalid", True)
            self.custom_edit.style().unpolish(self.custom_edit)
            self.custom_edit.style().polish(self.custom_edit)
            return None
        if value <= 0:
            self.custom_edit.setProperty("invalid", True)
            self.custom_edit.style().unpolish(self.custom_edit)
            self.custom_edit.style().polish(self.custom_edit)
            return None
        return value

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            return
        super().keyPressEvent(event)
