"""Parameter widgets round-tripping without clobbering each other."""

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from Breakdown.app_qt.theme import apply_theme  # noqa: E402
from Breakdown.app_qt.wizard import BreakdownWindow  # noqa: E402
from Breakdown.core.params import BreakdownParams  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


def test_the_capacitance_sweeps_are_enabled_on_a_fresh_window(qt_app):
    # Regression: filling a text field during load fires textChanged, which read
    # every widget back into params -- including checkboxes not yet loaded. That
    # silently disabled C(f) and C(V) at startup.
    window = BreakdownWindow()

    assert window.params.run.enable_cf is True
    assert window.params.run.enable_cv is True


def test_loading_does_not_disturb_values_loaded_after_it(qt_app):
    window = BreakdownWindow()
    params = BreakdownParams()
    params.sample.sample_name = "triggersTextChanged"
    params.run.enable_cf = True
    params.run.enable_cv = False
    params.mfia.four_terminal = True

    window.params_panel.load(params)
    read_back = BreakdownParams()
    window.params_panel.apply_to(read_back)

    assert read_back.run.enable_cf is True
    assert read_back.run.enable_cv is False
    assert read_back.mfia.four_terminal is True


def test_every_parameter_survives_a_load_then_read_back(qt_app):
    window = BreakdownWindow()
    original = BreakdownParams()
    original.sample.sample_name = "waferB"
    original.sample.crosspoint_um = 20.0
    original.sample.thickness_nm = 8.5
    original.rvs.polarity = -1
    original.rvs.ramp_rate_Vps = 2.5
    original.cvs.max_duration_s = 7200.0
    original.run.mode = "cvs_only"
    original.advisor.k_fraction = 0.9
    original.mfia.model = 3

    window.params_panel.load(original)
    read_back = BreakdownParams()
    problems = window.params_panel.apply_to(read_back)

    assert problems == []
    assert read_back == original


def test_a_blank_optional_field_reads_back_as_absent(qt_app):
    window = BreakdownWindow()
    window.params_panel.widgets["sample.thickness_nm"].setText("")
    params = BreakdownParams()

    assert window.params_panel.apply_to(params) == []
    assert params.sample.thickness_nm is None


def test_an_unreadable_number_is_reported_rather_than_swallowed(qt_app):
    window = BreakdownWindow()
    window.params_panel.widgets["rvs.ramp_rate_Vps"].setText("not a number")

    problems = window.params_panel.apply_to(BreakdownParams())

    assert any("Ramp rate" in p for p in problems)


def test_active_device_updates_the_next_device_header(qt_app):
    window = BreakdownWindow()
    window.header_device.setText("6")

    window._on_device_started(7, 5.0)

    assert window.progress_label.text() == "Device 7  (5 um)"
    assert window.header_device.text() == "8"
