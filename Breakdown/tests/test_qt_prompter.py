"""The blocking-prompt bridge between the worker thread and the GUI thread."""

import threading

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from Breakdown.app_qt.wizard import QtPrompter  # noqa: E402
from Breakdown.app_qt.dialogs import CvsVoltageDialog  # noqa: E402
from Breakdown.core.advisor import Recommendation  # noqa: E402
from Breakdown.core.events import (  # noqa: E402
    CvsVoltageContext, Instrument, RateLimitContext, StressType,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


class Caller:
    """Runs one prompter call on a background thread, like the session does."""

    def __init__(self, prompter, method, *args):
        self.result = "unset"
        self.returned = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(prompter, method, args), daemon=True
        )

    def _run(self, prompter, method, args):
        self.result = getattr(prompter, method)(*args)
        self.returned.set()

    def start(self):
        self._thread.start()
        return self

    def wait(self, timeout=2.0):
        assert self.returned.wait(timeout), "prompter call never returned"
        return self.result


def context(previous=None, recommended=3.5):
    return CvsVoltageContext(
        device_index=2, crosspoint_um=5.0, previous_voltage=previous,
        recommendation=Recommendation(voltage=recommended, basis_text="basis"),
    )


def test_a_prompt_blocks_until_an_answer_arrives(qt_app):
    prompter = QtPrompter(should_stop=lambda: False)
    caller = Caller(prompter, "confirm_cable_swap", Instrument.SMU, 1).start()

    assert not caller.returned.wait(0.2)   # still waiting

    prompter.deliver(True)
    assert caller.wait() is True


def test_declining_a_cable_swap_comes_back_as_false(qt_app):
    prompter = QtPrompter(should_stop=lambda: False)
    caller = Caller(prompter, "confirm_cable_swap", Instrument.MFIA, 1).start()

    prompter.deliver(None)

    assert caller.wait() is False


def test_a_stop_releases_a_waiting_prompt(qt_app):
    # Without this, pressing Stop while a dialog is open would hang the run.
    stop = False
    prompter = QtPrompter(should_stop=lambda: stop)
    caller = Caller(prompter, "confirm_cable_swap", Instrument.SMU, 1).start()

    assert not caller.returned.wait(0.2)
    stop = True

    assert caller.wait() is False


def test_a_chosen_stress_voltage_is_returned_as_a_number(qt_app):
    prompter = QtPrompter(should_stop=lambda: False)
    caller = Caller(prompter, "resolve_cvs_voltage", context()).start()

    prompter.deliver(3.42)

    assert caller.wait() == pytest.approx(3.42)


def test_refusing_to_choose_a_stress_voltage_returns_nothing(qt_app):
    prompter = QtPrompter(should_stop=lambda: False)
    caller = Caller(prompter, "resolve_cvs_voltage", context()).start()

    prompter.deliver(None)

    assert caller.wait() is None


def test_a_stop_releases_a_waiting_voltage_prompt(qt_app):
    stop = False
    prompter = QtPrompter(should_stop=lambda: stop)
    caller = Caller(prompter, "resolve_cvs_voltage", context()).start()

    assert not caller.returned.wait(0.2)
    stop = True

    assert caller.wait() is None


def test_the_chosen_stress_type_is_returned(qt_app):
    prompter = QtPrompter(should_stop=lambda: False)
    caller = Caller(prompter, "choose_stress_type", 3).start()

    prompter.deliver(StressType.CVS)

    assert caller.wait() is StressType.CVS


def test_rate_limit_confirmation_is_returned(qt_app):
    prompter = QtPrompter(should_stop=lambda: False)
    ctx = RateLimitContext(
        device_index=2, stress_type=StressType.CVS,
        requested_rate_Vps=100.0, achievable_rate_Vps=4.0,
        point_period_s=0.1, max_step_V=0.4,
    )
    caller = Caller(prompter, "confirm_rate_limit", ctx).start()

    prompter.deliver(True)

    assert caller.wait() is True


def test_cvs_dialog_rejects_a_custom_voltage_above_the_ceiling(qt_app):
    ctx = CvsVoltageContext(
        device_index=2, crosspoint_um=5.0, previous_voltage=None,
        recommendation=Recommendation(voltage=None, basis_text="none"),
        max_voltage_V=10.0,
    )
    dialog = CvsVoltageDialog(ctx)
    dialog.custom_button.setChecked(True)
    dialog.custom_edit.setText("10.1")

    assert dialog.selected_voltage() is None


def test_consecutive_prompts_do_not_reuse_the_previous_answer(qt_app):
    # A stale answer would silently auto-confirm the next cable swap.
    prompter = QtPrompter(should_stop=lambda: False)
    first = Caller(prompter, "confirm_cable_swap", Instrument.SMU, 1).start()
    prompter.deliver(True)
    assert first.wait() is True

    second = Caller(prompter, "confirm_cable_swap", Instrument.MFIA, 1).start()

    assert not second.returned.wait(0.2)
    prompter.deliver(True)
    assert second.wait() is True
