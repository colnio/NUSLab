"""Theme support and shared widgets for the Breakdown application.

The interface is designed like an instrument faceplate: numbers use a
monospace face, and green/amber/red are reserved for operational state. Both
themes keep that visual language while allowing the operator to choose the
contrast that suits the room.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFontDatabase, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

DARK_MODE = "dark"
LIGHT_MODE = "light"


@dataclass(frozen=True)
class ThemeColors:
    bg: str
    surface: str
    raised: str
    border: str
    border_strong: str
    text: str
    text_dim: str
    text_faint: str
    go: str
    wait: str
    danger: str
    focus: str
    plot_previous: tuple


DARK = ThemeColors(
    bg="#0E1116", surface="#161B22", raised="#1C2430",
    border="#2A3441", border_strong="#3A4757", text="#E6EDF3",
    text_dim="#8B98A9", text_faint="#6E7B8C", go="#3FB950",
    wait="#D29922", danger="#F85149", focus="#58A6FF",
    plot_previous=(94, 107, 124, 110),
)

LIGHT = ThemeColors(
    bg="#F3F5F7", surface="#FFFFFF", raised="#EAEFF4",
    border="#D0D7DE", border_strong="#A8B3BF", text="#1F2328",
    text_dim="#57606A", text_faint="#6E7781", go="#1A7F37",
    wait="#9A6700", danger="#CF222E", focus="#0969DA",
    plot_previous=(110, 119, 129, 125),
)

THEMES = {DARK_MODE: DARK, LIGHT_MODE: LIGHT}

# Backwards-compatible dark aliases. Theme-aware code should call
# ``theme_colors()`` at the moment it paints.
BG = DARK.bg
SURFACE = DARK.surface
RAISED = DARK.raised
BORDER = DARK.border
BORDER_STRONG = DARK.border_strong
TEXT = DARK.text
TEXT_DIM = DARK.text_dim
TEXT_FAINT = DARK.text_faint
GO = DARK.go
WAIT = DARK.wait
DANGER = DARK.danger
FOCUS = DARK.focus

MONO_CANDIDATES = (
    "SF Mono", "Menlo", "JetBrains Mono", "Consolas",
    "DejaVu Sans Mono", "Courier New",
)


def normalize_theme(theme_name: object) -> str:
    """Return a supported theme name, falling back safely to dark mode."""
    name = str(theme_name or "").strip().lower()
    return name if name in THEMES else DARK_MODE


def current_theme(app: Optional[QApplication] = None) -> str:
    app = app or QApplication.instance()
    return normalize_theme(app.property("colorTheme") if app is not None else None)


def theme_colors(theme_name: Optional[str] = None) -> ThemeColors:
    return THEMES[normalize_theme(theme_name) if theme_name else current_theme()]


def mono_family() -> str:
    available = set(QFontDatabase().families())
    for name in MONO_CANDIDATES:
        if name in available:
            return name
    return "monospace"


def _stylesheet(mono: str, c: ThemeColors, theme_name: str) -> str:
    light = theme_name == LIGHT_MODE
    input_focus = "#FFFFFF" if light else "#202A38"
    invalid_bg = "#FFF0F0" if light else "#2A1A1C"
    disabled_bg = "#E3E8ED" if light else "#131820"
    button_hover = "#DFE7EF" if light else "#243040"
    button_pressed = "#D2DBE5" if light else "#1A2230"
    start_text = "#FFFFFF" if light else "#06210C"
    start_hover = "#16833A" if light else "#4CC85E"
    start_disabled = "#D8E6DC" if light else "#1D2A20"
    stop_hover = "#FFF0F0" if light else "#2A1518"
    stop_disabled_text = "#B88B8F" if light else "#5A2E30"
    stop_disabled_border = "#DFC7C9" if light else "#3A2224"
    text_edit_bg = "#FFFFFF" if light else "#0B0E13"
    tooltip_bg = "#1F2328" if light else "#050709"
    tooltip_text = "#FFFFFF" if light else c.text
    warning_bg = "#FFF8C5" if light else "#2A2107"
    warning_border = "#D4A72C" if light else "#4A3A0C"

    return f"""
QWidget {{
    background: {c.bg};
    color: {c.text};
    font-size: 13px;
}}

/* ---- tabs ---- */
QTabWidget::pane {{ border: none; background: {c.bg}; }}
QTabBar {{ qproperty-drawBase: 0; background: {c.bg}; }}
QTabBar::tab {{
    background: transparent;
    color: {c.text_dim};
    padding: 10px 18px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{ color: {c.text}; }}
QTabBar::tab:selected {{ color: {c.text}; border-bottom: 2px solid {c.focus}; }}

/* ---- cards and header ---- */
QWidget#HeaderBar {{
    background: {c.surface};
    border-bottom: 1px solid {c.border};
}}
QFrame#Card {{
    background: {c.surface};
    border: 1px solid {c.border};
    border-radius: 8px;
}}
QLabel#CardTitle {{
    color: {c.text};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    background: transparent;
}}
QLabel#CardNote {{ color: {c.text_faint}; font-size: 11px; background: transparent; }}
QFrame#CardRule {{ background: {c.border}; max-height: 1px; border: none; }}

/* ---- form and semantic text ---- */
QLabel {{ background: transparent; }}
QLabel#FieldLabel, QLabel#MutedText {{ color: {c.text_dim}; }}
QLabel#FaintText {{ color: {c.text_faint}; }}
QLabel#WarningText {{ color: {c.wait}; }}
QLabel#WarningPanel {{
    color: {c.wait};
    background: {warning_bg};
    border: 1px solid {warning_border};
    border-radius: 5px;
    padding: 8px 10px;
}}
QLabel#FieldUnit {{
    color: {c.text_faint};
    font-family: "{mono}";
    font-size: 12px;
}}

/* ---- inputs ---- */
QLineEdit, QComboBox, QSpinBox, QAbstractSpinBox {{
    background: {c.raised};
    border: 1px solid {c.border};
    border-radius: 5px;
    padding: 5px 9px;
    min-height: 20px;
    color: {c.text};
    selection-background-color: {c.focus};
    selection-color: {c.bg};
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover {{ border-color: {c.border_strong}; }}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
    border-color: {c.focus};
    background: {input_focus};
}}
QLineEdit[numeric="true"], QSpinBox {{ font-family: "{mono}"; font-size: 12.5px; }}
QLineEdit[invalid="true"] {{ border-color: {c.danger}; background: {invalid_bg}; }}
QLineEdit:disabled, QComboBox:disabled {{ color: {c.text_faint}; background: {disabled_bg}; }}

QComboBox QAbstractItemView {{
    background: {c.raised};
    border: 1px solid {c.border_strong};
    selection-background-color: {c.focus};
    selection-color: {c.bg};
    outline: none;
    padding: 3px;
}}

QCheckBox {{ background: transparent; spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {c.border_strong};
    border-radius: 4px;
    background: {c.raised};
}}
QCheckBox::indicator:hover {{ border-color: {c.focus}; }}
QCheckBox::indicator:checked {{ background: {c.focus}; border-color: {c.focus}; }}

/* ---- buttons ---- */
QPushButton {{
    background: {c.raised};
    color: {c.text};
    border: 1px solid {c.border_strong};
    border-radius: 5px;
    padding: 6px 14px;
    min-height: 20px;
    font-weight: 600;
}}
QPushButton:hover {{ background: {button_hover}; border-color: {c.text_faint}; }}
QPushButton:pressed {{ background: {button_pressed}; }}
QPushButton:disabled {{ color: {c.text_faint}; background: {disabled_bg}; border-color: {c.border}; }}
QPushButton:checked {{ border-color: {c.focus}; color: {c.focus}; }}

QPushButton#StartButton {{
    background: {c.go}; color: {start_text}; border: none; padding: 8px 22px;
    font-size: 13px; font-weight: 700; letter-spacing: 0.3px;
}}
QPushButton#StartButton:hover {{ background: {start_hover}; }}
QPushButton#StartButton:disabled {{ background: {start_disabled}; color: {c.text_faint}; }}

QPushButton#StopButton {{
    background: transparent; color: {c.danger};
    border: 1px solid {c.danger}; padding: 8px 22px; font-weight: 700;
}}
QPushButton#StopButton:hover {{ background: {stop_hover}; }}
QPushButton#StopButton:disabled {{
    color: {stop_disabled_text}; border-color: {stop_disabled_border}; background: transparent;
}}

/* ---- readouts ---- */
QLabel#StateChip {{
    font-size: 11px; font-weight: 700; letter-spacing: 1.4px;
    padding: 5px 12px; border-radius: 4px;
    background: {c.raised}; color: {c.text_dim};
}}
QLabel#HeaderValue {{ font-family: "{mono}"; font-size: 13px; color: {c.text}; }}
QLabel#HeaderKey {{
    font-size: 10px; font-weight: 700; letter-spacing: 1.1px; color: {c.text_faint};
}}

QTextEdit {{
    background: {text_edit_bg};
    border: 1px solid {c.border};
    border-radius: 6px;
    font-family: "{mono}";
    font-size: 12px;
    color: {c.text_dim};
    padding: 6px;
}}

QProgressBar {{
    background: {c.raised}; border: none; border-radius: 3px;
    height: 5px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {c.focus}; border-radius: 3px; }}

/* ---- scrolling ---- */
QScrollArea {{ border: none; background: {c.bg}; }}
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {c.border_strong}; border-radius: 5px; min-height: 32px;
}}
QScrollBar::handle:vertical:hover {{ background: {c.text_faint}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; background: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

/* ---- dialogs ---- */
QDialog {{ background: {c.surface}; }}
QRadioButton {{ background: transparent; spacing: 8px; padding: 3px 0; }}
QRadioButton::indicator {{
    width: 15px; height: 15px; border-radius: 8px;
    border: 1px solid {c.border_strong}; background: {c.raised};
}}
QRadioButton::indicator:checked {{ background: {c.focus}; border-color: {c.focus}; }}
QRadioButton:disabled {{ color: {c.text_faint}; }}
QToolTip {{
    background: {tooltip_bg}; color: {tooltip_text}; border: 1px solid {c.border_strong};
    padding: 5px 8px; border-radius: 4px;
}}
"""


def apply_theme(app: QApplication, theme_name: str = DARK_MODE) -> str:
    """Apply a complete light or dark Fusion palette and return its name."""
    name = normalize_theme(theme_name)
    c = THEMES[name]
    app.setStyle("Fusion")
    app.setProperty("colorTheme", name)

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(c.bg))
    palette.setColor(QPalette.WindowText, QColor(c.text))
    palette.setColor(QPalette.Base, QColor(c.raised))
    palette.setColor(QPalette.AlternateBase, QColor(c.surface))
    palette.setColor(QPalette.Text, QColor(c.text))
    palette.setColor(QPalette.Button, QColor(c.raised))
    palette.setColor(QPalette.ButtonText, QColor(c.text))
    palette.setColor(QPalette.Highlight, QColor(c.focus))
    palette.setColor(QPalette.HighlightedText, QColor(c.bg))
    palette.setColor(QPalette.ToolTipBase, QColor(c.surface))
    palette.setColor(QPalette.ToolTipText, QColor(c.text))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(c.text_faint))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(c.text_faint))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(c.text_faint))
    app.setPalette(palette)
    app.setStyleSheet(_stylesheet(mono_family(), c, name))
    return name


class Card(QFrame):
    """A titled panel whose title cannot be clipped by Qt's frame drawing."""

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
    """One left-aligned label, one control, and an optional unit."""
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
    """Mark a numeric widget so the stylesheet gives it tabular digits."""
    widget.setProperty("numeric", True)
    return widget


def state_chip_style(kind: str, theme_name: Optional[str] = None) -> str:
    """Return theme-aware operational colours for the run-state chip."""
    name = normalize_theme(theme_name) if theme_name else current_theme()
    c = THEMES[name]
    backgrounds = {
        DARK_MODE: {
            "measuring": "#12304A", "waiting": "#33280A",
            "stressing": "#3A1417", "done": "#12301A",
        },
        LIGHT_MODE: {
            "measuring": "#DDF4FF", "waiting": "#FFF8C5",
            "stressing": "#FFEBE9", "done": "#DAFBE1",
        },
    }[name]
    foregrounds = {
        "measuring": c.focus,
        "waiting": c.wait,
        "stressing": c.danger,
        "done": c.go,
    }
    if kind not in backgrounds:
        return f"background: {c.raised}; color: {c.text_dim};"
    return f"background: {backgrounds[kind]}; color: {foregrounds[kind]};"
