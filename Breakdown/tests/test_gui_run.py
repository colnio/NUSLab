"""
End-to-end run through the GUI's worker thread with simulated instruments.

This is the piece the core tests cannot reach: the QThread wiring, the queued
signals, and the prompt bridge all working together against a real event loop.
Dialogs are replaced with stubs that answer immediately, so the run proceeds
without a human.
"""

import csv
import json
import time

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from Breakdown.app_qt import wizard as W  # noqa: E402
from Breakdown.core.events import StressType  # noqa: E402

DEVICE_TARGET = 2
TIMEOUT_S = 90.0

#: These drive a real event loop in real time. Skip with -m "not slow".
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


class StubDialog:
    Accepted = 1
    Rejected = 0

    def __init__(self, *args, **kwargs):
        pass

    def exec_(self):
        return self.Accepted

    def reject(self):
        pass


class StubCvsDialog(StubDialog):
    #: Close to the simulated V_BD of 4.2 V so the hold fails quickly.
    voltage = 4.0

    def selected_voltage(self):
        return self.voltage


class StubStressTypeDialog(StubDialog):
    def selected(self):
        return StressType.RVS


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    monkeypatch.setattr(W, "CableSwapDialog", StubDialog)
    monkeypatch.setattr(W, "CvsVoltageDialog", StubCvsDialog)
    monkeypatch.setattr(W, "StressTypeDialog", StubStressTypeDialog)

    win = W.BreakdownWindow()
    win.mfia_combo.setCurrentText("Mock")
    win.smu_combo.setCurrentText("Mock")

    p = win.params
    p.run.output_dir = str(tmp_path)
    p.run.mode = "alternating"
    p.sample.sample_name = "guiSample"
    p.sample.crosspoint_um = 5.0
    p.sample.thickness_nm = 8.0
    # Keep a real-time run short: few points, no settle, a brisk ramp.
    p.cf.points = 3
    p.cf.settle_s = 0.0
    p.cv.points = 3
    p.cv.settle_s = 0.0
    p.rvs.v_max = 5.0
    p.rvs.ramp_rate_Vps = 5.0
    p.rvs.max_step_V = 0.1
    p.rvs.nplc = 0.1
    p.cvs.nplc = 0.1
    p.cvs.pre_ramp_rate_Vps = 5.0
    p.cvs.sample_interval_s = 0.02
    p.cvs.max_duration_s = 30.0
    win.params_panel.load(p)

    yield win

    if win.thread is not None and win.thread.isRunning():
        win.worker.request_stop()
        win.thread.quit()
        win.thread.wait(5000)


def run_devices(qt_app, window, count=DEVICE_TARGET):
    """Start the run, stop it after ``count`` devices, return the summary rows."""
    done = []
    window.worker_finished = False

    def on_device(index, row):
        done.append(row)
        if len(done) >= count:
            window._stop()

    def on_finished(*_args):
        window.worker_finished = True

    window._start()
    window.worker.device_finished.connect(on_device)
    window.worker.finished.connect(on_finished)
    window.worker.failed.connect(on_finished)

    deadline = time.time() + TIMEOUT_S
    while not window.worker_finished and time.time() < deadline:
        qt_app.processEvents()
        time.sleep(0.005)

    assert window.worker_finished, f"run did not finish within {TIMEOUT_S}s"
    return done


def read_summary(tmp_path, date):
    path = tmp_path / date / "guiSample" / "guiSample_summary.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_a_full_run_completes_the_requested_devices(qt_app, window):
    rows = run_devices(qt_app, window)

    assert len(rows) == DEVICE_TARGET


def test_a_full_run_alternates_ramp_and_hold(qt_app, window):
    rows = run_devices(qt_app, window)

    assert [r["stress_type"] for r in rows] == ["RVS", "CVS"]


def test_the_ramped_device_reports_a_breakdown_voltage(qt_app, window):
    rows = run_devices(qt_app, window)

    assert rows[0]["V_BD_V"] is not None and float(rows[0]["V_BD_V"]) > 0


def test_the_held_device_uses_the_voltage_chosen_in_the_dialog(qt_app, window):
    rows = run_devices(qt_app, window)

    assert float(rows[1]["V_stress_V"]) == pytest.approx(StubCvsDialog.voltage)


def test_the_run_writes_the_expected_folder_tree(qt_app, window, tmp_path):
    rows = run_devices(qt_app, window)
    date = rows[0]["date"]

    for index, kind in ((1, "RVS"), (2, "CVS")):
        data_dir = tmp_path / date / "guiSample" / "5um" / f"dev{index:03d}" / "data"
        kinds = sorted(f.name.split("_")[0] for f in data_dir.iterdir())
        assert kinds == sorted(["CF", "CV", kind])


def test_each_device_folder_gets_its_metadata(qt_app, window, tmp_path):
    rows = run_devices(qt_app, window)
    date = rows[0]["date"]
    meta = tmp_path / date / "guiSample" / "5um" / "dev001" / "dev001.meta.json"

    assert json.loads(meta.read_text())["stress_type"] == "RVS"


def test_the_summary_file_matches_what_the_ui_reported(qt_app, window, tmp_path):
    rows = run_devices(qt_app, window)

    on_disk = read_summary(tmp_path, rows[0]["date"])
    assert [r["stress_type"] for r in on_disk] == ["RVS", "CVS"]


def test_the_parameters_are_snapshotted_for_the_run(qt_app, window, tmp_path):
    rows = run_devices(qt_app, window)
    sample_dir = tmp_path / rows[0]["date"] / "guiSample"

    snapshots = list(sample_dir.glob("guiSample_params_*.json"))
    assert len(snapshots) == 1
    assert json.loads(snapshots[0].read_text())["run"]["mode"] == "alternating"


def test_the_run_log_is_written_alongside_the_data(qt_app, window, tmp_path):
    rows = run_devices(qt_app, window)
    sample_dir = tmp_path / rows[0]["date"] / "guiSample"

    logs = list(sample_dir.glob("guiSample_*.log"))
    assert logs and "Device 1" in logs[0].read_text()


def test_point_data_reaches_the_gui_for_every_phase(qt_app, window):
    # Asserting on the plot buffers would be racy: the session starts the next
    # device (clearing them) before the stop request lands. What matters is that
    # every phase's points crossed the thread boundary.
    seen = set()
    window._start()
    window.worker.point.connect(lambda phase, _row: seen.add(phase))
    done = []

    def on_device(index, row):
        done.append(row)
        if len(done) >= DEVICE_TARGET:
            window._stop()

    finished = []
    window.worker.device_finished.connect(on_device)
    window.worker.finished.connect(lambda *_a: finished.append(True))
    window.worker.failed.connect(lambda *_a: finished.append(True))

    deadline = time.time() + TIMEOUT_S
    while not finished and time.time() < deadline:
        qt_app.processEvents()
        time.sleep(0.005)

    assert {"CF", "CV", "RVS", "CVS"} <= seen


def test_the_controls_return_to_idle_after_the_run(qt_app, window):
    run_devices(qt_app, window)
    deadline = time.time() + 5.0
    while window.thread is not None and time.time() < deadline:
        qt_app.processEvents()
        time.sleep(0.005)

    assert window.start_button.isEnabled()
    assert not window.stop_button.isEnabled()

# Instrument safe-off is asserted in test_session.py, where the mocks stay
# reachable after the run; here the worker releases them in its finally block.
