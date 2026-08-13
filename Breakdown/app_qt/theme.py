"""
Visual language: an instrument faceplate.

This app is read in a dim probe-station room, in glances, between cable swaps.
Its content is numbers. So it commits to one dark theme rather than following the
OS -- which also removes a whole class of bug, since a stylesheet that only
specifies foregrounds turns unreadable the moment macOS flips to dark.

Three rules the rest of the UI follows:

* **Numbers are monospace and tabular.** You scan columns of voltages and
  currents; proportional digits make that harder than it needs to be.
* **Colour means something.** Green is "safe to proceed", amber is "the run is
  waiting for you", red is "energised or destroyed". Nothing decorative is
  allowed to use those three.
* **Card titles are real widgets, not border decorations.** Qt's QGroupBox draws
  its title into the frame, where it clips at exactly the font sizes we want.

``apply_standard_window_style`` from ``KeithleyGUI/ui_helpers.py`` is
deliberately not used here; it assumes a light desktop. The other apps keep it.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

# --- tokens ----------------------------------------------------------------

BG = "#0E1116"          # window, deepest layer
SURFACE = "#161B22"     # cards
RAISED = "#1C2430"      # inputs, tabs
BORDER = "#2A3441"
BORDER_STRONG = "#3A4757"

TEXT = "#E6EDF3"
TEXT_DIM = "#8B98A9"
TEXT_FAINT = "#5E6B7C"

GO = "#3FB950"          # safe to proceed
WAIT = "#D29922"        # the run needs you
DANGER = "#F85149"      # energised, or destroyed
FOCUS = "#58A6FF"

MONO_CANDIDATES = ("SF Mono", "Menlo", "JetBrains Mono", "Consolas",
                   "DejaVu Sans Mono", "Courier New")


def mono_family() -> str:
    available = set(QFontDatabase().families())
    for name in MONO_CANDIDATES:
        if name in available:
            return name
    return "monospace"


def _stylesheet(mono: str) -> str:
    return f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-size: 13px;
}}

/* ---- tabs ---- */
QTabWidget::pane {{
    border: none;
    background: {BG};
}}
QTabBar {{ qproperty-drawBase: 0; background: {BG}; }}
/* Spacing comes from padding, not margin: a margin shifts a tab's rect without
   shifting the text Qt draws into it, and the labels clip. The font is set in
   code (see BreakdownWindow._build_ui) rather than here, because Fusion sizes
   tabs with the regular face and then paints the selected one bold -- which
   clips the widest label unless both passes use the same metrics. */
QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    padding: 10px 18px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom: 2px solid {FOCUS};
}}

/* ---- cards ---- */
QFrame#Card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QLabel#CardTitle {{
    color: {TEXT};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    background: transparent;
}}
QLabel#CardNote {{
    color: {TEXT_FAINT};
    font-size: 11px;
    background: transparent;
}}
QFrame#CardRule {{
    background: {BORDER};
    max-height: 1px;
    border: none;
}}

/* ---- form text ---- */
QLabel {{ background: transparent; }}
QLabel#FieldLabel {{ color: {TEXT_DIM}; }}
QLabel#FieldUnit {{
    color: {TEXT_FAINT};
    font-family: "{mono}";
    font-size: 12px;
}}

/* ---- inputs ---- */
QLineEdit, QComboBox, QSpinBox, QAbstractSpinBox {{
    background: {RAISED};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 9px;
    min-height: 20px;
    color: {TEXT};
    selection-background-color: {FOCUS};
    selection-color: {BG};
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover {{ border-color: {BORDER_STRONG}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
    border-color: {FOCUS};
    background: #202A38;
}}
QLineEdit[numeric="true"], QSpinBox {{
    font-family: "{mono}";
    font-size: 12.5px;
}}
QLineEdit[invalid="true"] {{ border-color: {DANGER}; background: #2A1A1C; }}
QLineEdit:disabled, QComboBox:disabled {{ color: {TEXT_FAINT}; background: #131820; }}

/* The drop-down indicator is left entirely to Fusion. Styling ::drop-down
   suppresses the stock arrow, and a hand-drawn CSS triangle renders as a
   filled box on some Qt builds -- both are worse than the default chevron. */
QComboBox QAbstractItemView {{
    background: {RAISED};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {FOCUS};
    selection-color: {BG};
    outline: none;
    padding: 3px;
}}

QCheckBox {{ background: transparent; spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER_STRONG};
    border-radius: 4px;
    background: {RAISED};
}}
QCheckBox::indicator:hover {{ border-color: {FOCUS}; }}
QCheckBox::indicator:checked {{ background: {FOCUS}; border-color: {FOCUS}; }}

/* ---- buttons ---- */
QPushButton {{
    background: {RAISED};
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    padding: 6px 14px;
    min-height: 20px;
    font-weight: 600;
}}
QPushButton:hover {{ background: #243040; border-color: {TEXT_FAINT}; }}
QPushButton:pressed {{ background: #1A2230; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: #131820;
                        border-color: {BORDER}; }}
QPushButton:checked {{ border-color: {FOCUS}; color: {FOCUS}; }}

QPushButton#StartButton {{
    background: {GO}; color: #06210C; border: none; padding: 8px 22px;
    font-size: 13px; font-weight: 700; letter-spacing: 0.3px;
}}
QPushButton#StartButton:hover {{ background: #4CC85E; }}
QPushButton#StartButton:disabled {{ background: #1D2A20; color: {TEXT_FAINT}; }}

QPushButton#StopButton {{
    background: transparent; color: {DANGER};
    border: 1px solid {DANGER}; padding: 8px 22px; font-weight: 700;
}}
QPushButton#StopButton:hover {{ background: #2A1518; }}
QPushButton#StopButton:disabled {{ color: #5A2E30; border-color: #3A2224;
                                   background: transparent; }}

/* ---- readouts ---- */
QLabel#StateChip {{
    font-size: 11px; font-weight: 700; letter-spacing: 1.4px;
    padding: 5px 12px; border-radius: 4px;
    background: {RAISED}; color: {TEXT_DIM};
}}
QLabel#HeaderValue {{ font-family: "{mono}"; font-size: 13px; color: {TEXT}; }}
QLabel#HeaderKey {{ font-size: 10px; font-weight: 700; letter-spacing: 1.1px;
                    color: {TEXT_FAINT}; }}

QTextEdit {{
    background: #0B0E13;
    border: 1px solid {BORDER};
    border-radius: 6px;
    font-family: "{mono}";
    font-size: 12px;
    color: {TEXT_DIM};
    padding: 6px;
}}

QProgressBar {{
    background: {RAISED}; border: none; border-radius: 3px;
    height: 5px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {FOCUS}; border-radius: 3px; }}

/* ---- scrolling ---- */
QScrollArea {{ border: none; background: {BG}; }}
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER_STRONG}; border-radius: 5px;
                               min-height: 32px; }}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; background: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

/* ---- dialogs ---- */
QDialog {{ background: {SURFACE}; }}
QRadioButton {{ background: transparent; spacing: 8px; padding: 3px 0; }}
QRadioButton::indicator {{
    width: 15px; height: 15px; border-radius: 8px;
    border: 1px solid {BORDER_STRONG}; background: {RAISED};
}}
QRadioButton::indicator:checked {{ background: {FOCUS}; border-color: {FOCUS}; }}
QRadioButton:disabled {{ color: {TEXT_FAINT}; }}
QToolTip {{
    background: #050709; color: {TEXT}; border: 1px solid {BORDER_STRONG};
    padding: 5px 8px; border-radius: 4px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """Fusion + an explicit palette + the stylesheet.

    All three are needed. Fusion because the native macOS style ignores much of
    the stylesheet; the palette because a few widgets (menus, tooltips, dialog
    chrome) paint from it rather than from CSS, and would otherwise stay light.
    """
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor(RAISED))
    palette.setColor(QPalette.AlternateBase, QColor(SURFACE))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(RAISED))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(FOCUS))
    palette.setColor(QPalette.HighlightedText, QColor(BG))
    palette.setColor(QPalette.ToolTipBase, QColor(SURFACE))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(TEXT_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(TEXT_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(TEXT_FAINT))
    app.setPalette(palette)

    app.setStyleSheet(_stylesheet(mono_family()))


class Card(QFrame):
    """A titled panel.

    The title is a label above a hairline rule, not text drawn into a border, so
    it cannot clip and the spacing is ours to control.
    """

    def __init__(self, title: str, note: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 13, 16, 15)
        outer.setSpacing(0)

        header = QHBoxLayout()
        header.setSpacing(10)
        label = QLabel(title.upper())
        label.setObjectName("CardTitle")
        header.addWidget(label)
        if note:
            note_label = QLabel(note)
            note_label.setObjectName("CardNote")
            header.addWidget(note_label)
        header.addStretch(1)
        outer.addLayout(header)

        rule = QFrame()
        rule.setObjectName("CardRule")
        rule.setFixedHeight(1)
        outer.addSpacing(9)
        outer.addWidget(rule)
        outer.addSpacing(12)

        self.body = QVBoxLayout()
        self.body.setSpacing(7)
        outer.addLayout(self.body)

    def add(self, widget_or_layout) -> None:
        if isinstance(widget_or_layout, QWidget):
            self.body.addWidget(widget_or_layout)
        else:
            self.body.addLayout(widget_or_layout)


def field_row(label_text: str, widget: QWidget, unit: str = "",
              label_width: int = 172) -> QHBoxLayout:
    """One left-aligned label, one control, an optional unit.

    Left-aligned rather than right-aligned: the labels are the thing you scan
    down, so a ragged right edge beats a ragged left one.
    """
    row = QHBoxLayout()
    row.setSpacing(10)

    label = QLabel(label_text)
    label.setObjectName("FieldLabel")
    label.setFixedWidth(label_width)
    label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    row.addWidget(label)
    row.addWidget(widget)

    if unit:
        unit_label = QLabel(unit)
        unit_label.setObjectName("FieldUnit")
        unit_label.setFixedWidth(52)
        row.addWidget(unit_label)
    else:
        row.addSpacing(52)

    row.addStretch(1)
    return row


def monospace(widget: QWidget) -> QWidget:
    """Mark a widget as carrying a number, so the stylesheet gives it tabular digits."""
    widget.setProperty("numeric", True)
    return widget


def state_chip_style(kind: str) -> str:
    """Inline colours for the run-state chip. See the colour rule at the top."""
    return {
        "idle": f"background: {RAISED}; color: {TEXT_DIM};",
        "measuring": f"background: #12304A; color: {FOCUS};",
        "waiting": f"background: #33280A; color: {WAIT};",
        "stressing": f"background: #3A1417; color: {DANGER};",
        "done": f"background: #12301A; color: {GO};",
    }.get(kind, f"background: {RAISED}; color: {TEXT_DIM};")
