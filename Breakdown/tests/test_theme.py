"""Light/dark theme switching and persistence wiring."""

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyqtgraph")

from PyQt5.QtGui import QPalette  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from Breakdown.app_qt.theme import (  # noqa: E402
    DARK,
    LIGHT,
    apply_theme,
    current_theme,
)
from Breakdown.app_qt.wizard import BreakdownWindow  # noqa: E402


class MemorySettings:
    def __init__(self):
        self.values = {}
        self.synced = False

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        self.synced = True


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    apply_theme(app, "dark")
    return app


def test_light_theme_updates_the_application_palette(qt_app):
    assert apply_theme(qt_app, "light") == "light"

    assert current_theme(qt_app) == "light"
    assert qt_app.palette().color(QPalette.Window).name().upper() == LIGHT.bg
    assert qt_app.palette().color(QPalette.WindowText).name().upper() == LIGHT.text

    apply_theme(qt_app, "dark")


def test_unknown_theme_falls_back_to_dark(qt_app):
    assert apply_theme(qt_app, "not-a-theme") == "dark"
    assert qt_app.palette().color(QPalette.Window).name().upper() == DARK.bg


def test_window_toggle_updates_dynamic_colours_and_saves_choice(qt_app):
    apply_theme(qt_app, "dark")
    settings = MemorySettings()
    window = BreakdownWindow(settings=settings)
    window._set_state("waiting", "Waiting")
    window.plots.add_point(
        "RVS", {"set_V": 1.0, "measured_I_A": 1e-8, "bd_flag": True}
    )
    window.plots.refresh()

    assert window.theme_button.text() == "Light mode"

    window.theme_button.click()

    assert current_theme(qt_app) == "light"
    assert window.theme_button.text() == "Dark mode"
    assert "#FFF8C5" in window.state_chip.styleSheet()
    assert window.plots._theme_name == "light"
    assert settings.values["appearance/theme"] == "light"
    assert settings.synced

    window.close()
    apply_theme(qt_app, "dark")
