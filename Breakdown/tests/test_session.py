import csv
import json
import os.path as op

import pytest

from Breakdown.core import params as P
from Breakdown.core import session as SE
from Breakdown.core.events import Instrument, StressType
from Breakdown.core.mock import MockImpedanceAnalyzer, MockSourceMeter


class FakeClock:
    """Virtual time, shared by the session and the simulated device.

    Without it a CVS hold would run for its real wall-clock limit, which is
    measured in hours.
    """

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


class RecordingPrompter:
    """Answers every prompt and records what was asked, and in what order."""

    def __init__(self, smu=None, mfia=None, stress_type=None, cvs_voltage=3.4,
                 accept_swaps=True):
        self.smu = smu
        self.mfia = mfia
        self.forced_stress_type = stress_type
        self.cvs_voltage = cvs_voltage
        self.accept_swaps = accept_swaps

        self.calls = []
        self.swap_targets = []
        self.cvs_contexts = []
        #: (mfia_safe_offs, smu_safe_offs) observed at each swap prompt.
        self.safe_off_at_swap = []

    def confirm_cable_swap(self, target, device_index):
        self.calls.append(("swap", target, device_index))
        self.swap_targets.append(target)
        self.safe_off_at_swap.append(
            (getattr(self.mfia, "safe_off_calls", 0),
             getattr(self.smu, "safe_off_calls", 0))
        )
        return self.accept_swaps

    def resolve_cvs_voltage(self, ctx):
        self.calls.append(("cvs_voltage", ctx.device_index))
        self.cvs_contexts.append(ctx)
        return self.cvs_voltage

    def choose_stress_type(self, device_index):
        self.calls.append(("stress_type", device_index))
        return self.forced_stress_type


def build(tmp_path, mode="alternating", **param_overrides):
    params = P.BreakdownParams()
    params.run.mode = mode
    params.run.output_dir = str(tmp_path)
    params.sample.sample_name = "waferB"
    params.sample.crosspoint_um = 5.0
    params.sample.thickness_nm = 8.0
    # Small sweeps and a low ceiling keep the simulated runs quick.
    params.cf.points = 3
    params.cv.points = 3
    params.rvs.v_max = 6.0
    params.cvs.max_duration_s = 5.0
    for key, value in param_overrides.items():
        section, _, field = key.partition("__")
        setattr(getattr(params, section), field, value)

    clock = FakeClock()
    smu = MockSourceMeter(v_bd=4.2, renew_on_configure=True, clock=clock)
    mfia = MockImpedanceAnalyzer()
    prompter = RecordingPrompter(smu=smu, mfia=mfia)
    session = SE.BreakdownSession(
        params=params, mfia=mfia, smu=smu, prompter=prompter,
        date="2026-08-13", timestamp="2026-08-13_14-30-00",
        clock=clock, sleep=clock.sleep,
    )
    return session, prompter, smu, mfia, params


def stress_types(prompter, smu, session):
    return [row["stress_type"] for row in session.summary_rows]


def read_summary(tmp_path):
    path = tmp_path / "2026-08-13" / "waferB" / "waferB_summary.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- stress type sequencing ------------------------------------------------

def test_rvs_only_mode_ramps_every_device(tmp_path):
    session, prompter, smu, _, _ = build(tmp_path, mode="rvs_only")

    session.run(device_limit=3)

    assert [r["stress_type"] for r in session.summary_rows] == ["RVS"] * 3


def test_alternating_mode_swaps_stress_type_each_device(tmp_path):
    session, _, _, _, _ = build(tmp_path, mode="alternating")

    session.run(device_limit=4)

    assert [r["stress_type"] for r in session.summary_rows] == \
        ["RVS", "CVS", "RVS", "CVS"]


def test_cvs_only_mode_still_ramps_the_first_device(tmp_path):
    # A CVS level cannot be chosen without at least one measured V_BD.
    session, _, _, _, _ = build(tmp_path, mode="cvs_only")

    session.run(device_limit=3)

    assert [r["stress_type"] for r in session.summary_rows] == \
        ["RVS", "CVS", "CVS"]


def test_manual_mode_asks_before_every_device(tmp_path):
    session, prompter, _, _, _ = build(tmp_path, mode="manual")
    prompter.forced_stress_type = StressType.RVS

    session.run(device_limit=2)

    assert [c for c in prompter.calls if c[0] == "stress_type"] == \
        [("stress_type", 1), ("stress_type", 2)]


def test_manual_mode_honours_the_choice_that_was_made(tmp_path):
    session, prompter, _, _, _ = build(tmp_path, mode="manual")
    prompter.forced_stress_type = StressType.CVS

    session.run(device_limit=1)

    assert session.summary_rows[0]["stress_type"] == "CVS"


# --- cable swap prompts ----------------------------------------------------

def test_each_device_prompts_for_both_cable_swaps(tmp_path):
    session, prompter, _, _, _ = build(tmp_path, mode="rvs_only")

    session.run(device_limit=2)

    assert prompter.swap_targets == [
        Instrument.SMU, Instrument.MFIA, Instrument.SMU, Instrument.MFIA,
    ]


def test_the_mfia_is_switched_off_before_asking_for_the_keithley(tmp_path):
    # The operator is about to touch the wiring; nothing may still be driving.
    session, prompter, _, _, _ = build(tmp_path, mode="rvs_only")

    session.run(device_limit=1)
    mfia_safe_offs_at_first_swap, _ = prompter.safe_off_at_swap[0]

    assert mfia_safe_offs_at_first_swap >= 1


def test_the_smu_is_switched_off_before_asking_for_the_mfia(tmp_path):
    session, prompter, _, _, _ = build(tmp_path, mode="rvs_only")

    session.run(device_limit=1)
    _, smu_safe_offs_at_second_swap = prompter.safe_off_at_swap[1]

    assert smu_safe_offs_at_second_swap >= 1


def test_declining_a_cable_swap_aborts_the_run(tmp_path):
    session, prompter, _, _, _ = build(tmp_path, mode="rvs_only")
    prompter.accept_swaps = False

    result = session.run(device_limit=3)

    assert result.reason == SE.RunOutcome.ABORTED
    assert result.devices_completed == 0


def test_no_cable_swap_is_requested_when_no_capacitance_is_measured(tmp_path):
    # With both MFIA sweeps off the cables never need to move.
    session, prompter, _, _, _ = build(tmp_path, mode="rvs_only",
                                       run__enable_cf=False, run__enable_cv=False)

    session.run(device_limit=2)

    assert prompter.swap_targets == []


# --- CVS voltage resolution ------------------------------------------------

def test_the_operator_is_asked_for_the_stress_voltage_before_every_cvs(tmp_path):
    session, prompter, _, _, _ = build(tmp_path, mode="alternating")

    session.run(device_limit=4)

    assert [c[1] for c in prompter.calls if c[0] == "cvs_voltage"] == [2, 4]


def test_the_chosen_stress_voltage_is_the_one_applied(tmp_path):
    session, prompter, _, _, _ = build(tmp_path, mode="alternating")
    prompter.cvs_voltage = 3.1

    session.run(device_limit=2)

    assert float(session.summary_rows[1]["V_stress_V"]) == pytest.approx(3.1)


def test_the_stress_voltage_prompt_carries_a_recommendation_from_prior_ramps(tmp_path):
    session, prompter, _, _, params = build(tmp_path, mode="alternating")

    session.run(device_limit=2)
    ctx = prompter.cvs_contexts[0]

    assert ctx.recommendation.voltage is not None
    assert "median" in ctx.recommendation.basis_text


def test_the_recommendation_reflects_the_measured_breakdown_voltages(tmp_path):
    session, prompter, _, _, params = build(tmp_path, mode="alternating")

    session.run(device_limit=2)
    ctx = prompter.cvs_contexts[0]
    measured = float(session.summary_rows[0]["V_BD_V"])

    assert ctx.recommendation.voltage == pytest.approx(
        params.advisor.k_fraction * abs(measured)
    )


def test_the_previous_stress_voltage_is_offered_as_the_default(tmp_path):
    session, prompter, _, _, _ = build(tmp_path, mode="alternating")
    prompter.cvs_voltage = 3.3

    session.run(device_limit=4)

    assert prompter.cvs_contexts[0].previous_voltage is None
    assert prompter.cvs_contexts[1].previous_voltage == pytest.approx(3.3)


def test_declining_to_pick_a_stress_voltage_aborts_the_run(tmp_path):
    session, prompter, _, _, _ = build(tmp_path, mode="alternating")
    prompter.cvs_voltage = None

    result = session.run(device_limit=2)

    assert result.reason == SE.RunOutcome.ABORTED
    assert result.devices_completed == 1


def test_the_stress_prompt_only_offers_records_of_the_matching_size(tmp_path):
    session, prompter, _, _, _ = build(tmp_path, mode="alternating")

    session.run(device_limit=2)

    assert all(r.crosspoint_um == 5.0 for r in prompter.cvs_contexts[0].records)


# --- stopping --------------------------------------------------------------

def test_a_stop_request_ends_the_run_between_devices(tmp_path):
    session, _, _, _, _ = build(tmp_path, mode="rvs_only")
    original = session._on_device_complete

    def stop_after_first(*args, **kwargs):
        row = original(*args, **kwargs)
        session.request_stop()
        return row

    session._on_device_complete = stop_after_first
    result = session.run(device_limit=5)

    assert result.reason == SE.RunOutcome.STOPPED
    assert result.devices_completed == 1


def test_stopping_leaves_both_instruments_safe(tmp_path):
    session, _, smu, mfia, _ = build(tmp_path, mode="rvs_only")
    session.request_stop()

    session.run(device_limit=2)

    assert smu.safe_off_calls >= 1 and mfia.safe_off_calls >= 1


# --- indexing --------------------------------------------------------------

def test_devices_are_numbered_from_one(tmp_path):
    session, _, _, _, _ = build(tmp_path, mode="rvs_only")

    session.run(device_limit=3)

    assert [int(r["device_index"]) for r in session.summary_rows] == [1, 2, 3]


def test_a_second_run_continues_the_numbering(tmp_path):
    first, _, _, _, _ = build(tmp_path, mode="rvs_only")
    first.run(device_limit=2)
    second, _, _, _, _ = build(tmp_path, mode="rvs_only")

    second.run(device_limit=1)

    assert int(second.summary_rows[0]["device_index"]) == 3


def test_a_manual_starting_index_is_respected(tmp_path):
    session, _, _, _, _ = build(tmp_path, mode="rvs_only")
    session.set_next_index(7)

    session.run(device_limit=2)

    assert [int(r["device_index"]) for r in session.summary_rows] == [7, 8]


# --- output files ----------------------------------------------------------

def test_every_measurement_is_written_to_its_own_file(tmp_path):
    session, _, _, _, _ = build(tmp_path, mode="rvs_only")

    session.run(device_limit=1)
    data_dir = tmp_path / "2026-08-13" / "waferB" / "5um" / "dev001" / "data"
    kinds = sorted(f.name.split("_")[0] for f in data_dir.iterdir())

    assert kinds == ["CF", "CV", "RVS"]


def test_the_device_folder_carries_its_own_metadata(tmp_path):
    session, _, _, _, _ = build(tmp_path, mode="rvs_only")

    session.run(device_limit=1)
    meta_path = tmp_path / "2026-08-13" / "waferB" / "5um" / "dev001" / "dev001.meta.json"
    payload = json.loads(meta_path.read_text())

    assert payload["device_index"] == 1
    assert payload["stress_type"] == "RVS"
    assert payload["parameters"]["sample"]["sample_name"] == "waferB"


def test_the_summary_gains_a_row_per_device(tmp_path):
    session, _, _, _, _ = build(tmp_path, mode="alternating")

    session.run(device_limit=2)

    assert [r["stress_type"] for r in read_summary(tmp_path)] == ["RVS", "CVS"]


def test_the_summary_row_records_the_breakdown_voltage_and_field(tmp_path):
    session, _, _, _, _ = build(tmp_path, mode="rvs_only")

    session.run(device_limit=1)
    row = read_summary(tmp_path)[0]

    assert float(row["V_BD_V"]) > 0
    assert float(row["E_BD_MV_per_cm"]) == pytest.approx(
        abs(float(row["V_BD_V"])) / 8.0 * 10.0
    )


def test_the_summary_row_records_the_achieved_ramp_rate(tmp_path):
    session, _, _, _, _ = build(tmp_path, mode="rvs_only")

    session.run(device_limit=1)
    row = read_summary(tmp_path)[0]

    assert float(row["achieved_rate_Vps"]) > 0
    assert float(row["ramp_rate_Vps"]) > 0


def test_the_summary_row_records_the_time_to_breakdown_for_a_hold(tmp_path):
    session, prompter, _, _, _ = build(tmp_path, mode="alternating")
    prompter.cvs_voltage = 4.0   # close enough to V_BD to fail inside the limit

    session.run(device_limit=2)
    row = read_summary(tmp_path)[1]

    assert float(row["t_BD_s"]) > 0
    assert row["bd_detected"] == "True"


def test_a_hold_that_breaks_down_does_not_report_a_breakdown_voltage(tmp_path):
    # A CVS failure happens at the stress level by construction. Recording that
    # as V_BD would poison any Weibull fit over the V_BD column.
    session, prompter, _, _, _ = build(tmp_path, mode="alternating")
    prompter.cvs_voltage = 4.0

    session.run(device_limit=2)
    row = read_summary(tmp_path)[1]

    assert row["bd_detected"] == "True"
    assert row["V_BD_V"] == ""
    assert row["E_BD_MV_per_cm"] == ""


def test_a_hold_reports_the_field_it_was_stressed_at(tmp_path):
    # Fitting a lifetime model means plotting t_BD against field, so the stress
    # field belongs in the summary next to t_BD.
    session, prompter, _, _, _ = build(tmp_path, mode="alternating")
    prompter.cvs_voltage = 4.0

    session.run(device_limit=2)
    row = read_summary(tmp_path)[1]

    assert float(row["E_stress_MV_per_cm"]) == pytest.approx(4.0 / 8.0 * 10.0)


def test_a_ramp_reports_no_stress_field(tmp_path):
    session, _, _, _, _ = build(tmp_path, mode="rvs_only")

    session.run(device_limit=1)

    assert read_summary(tmp_path)[0]["E_stress_MV_per_cm"] == ""


def test_a_hold_that_survives_records_no_time_to_breakdown(tmp_path):
    session, prompter, _, _, _ = build(tmp_path, mode="alternating")
    prompter.cvs_voltage = 2.0   # far below V_BD; outlives the duration limit

    session.run(device_limit=2)
    row = read_summary(tmp_path)[1]

    assert row["t_BD_s"] == ""
    assert row["termination_reason"] == "max_duration"


def test_the_summary_row_records_the_capacitance(tmp_path):
    session, _, _, _, _ = build(tmp_path, mode="rvs_only")

    session.run(device_limit=1)
    row = read_summary(tmp_path)[0]

    assert float(row["C_at_fref_F"]) > 0
    assert float(row["area_um2"]) == pytest.approx(25.0)


def test_the_parameters_are_snapshotted_next_to_the_data(tmp_path):
    session, _, _, _, _ = build(tmp_path, mode="rvs_only")

    session.run(device_limit=1)
    snapshot = tmp_path / "2026-08-13" / "waferB" / \
        "waferB_params_2026-08-13_14-30-00.json"

    assert json.loads(snapshot.read_text())["run"]["mode"] == "rvs_only"


def test_disabled_sweeps_produce_no_capacitance_files(tmp_path):
    session, _, _, _, _ = build(tmp_path, mode="rvs_only", run__enable_cv=False)

    session.run(device_limit=1)
    data_dir = tmp_path / "2026-08-13" / "waferB" / "5um" / "dev001" / "data"

    assert sorted(f.name.split("_")[0] for f in data_dir.iterdir()) == ["CF", "RVS"]


# --- validation ------------------------------------------------------------

def test_a_run_refuses_to_start_with_invalid_parameters(tmp_path):
    session, _, _, _, params = build(tmp_path, mode="rvs_only")
    params.rvs.i_bd_A = params.rvs.compliance_A   # can never trigger

    with pytest.raises(ValueError, match="i_bd_A"):
        session.run(device_limit=1)


def test_a_run_refuses_to_start_without_an_output_folder(tmp_path):
    session, _, _, _, params = build(tmp_path, mode="rvs_only")
    params.run.output_dir = ""

    with pytest.raises(ValueError):
        session.run(device_limit=1)
