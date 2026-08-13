import pytest

from Breakdown.core import params as P
from Breakdown.core import stress as ST
from Breakdown.core.instruments import Termination
from Breakdown.core.mock import MockSourceMeter


class FakeClock:
    """Virtual time, so the tests run instantly and deterministically.

    The mock instrument reads this same clock, so simulated stress damage
    accumulates in step with the algorithm's own view of time.
    """

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


def rvs_params(**overrides):
    p = P.RVSParams(v_max=6.0, ramp_rate_Vps=1.0, max_step_V=0.05, nplc=1.0,
                    compliance_A=1e-3, i_bd_A=1e-4)
    for key, value in overrides.items():
        setattr(p, key, value)
    return p


def cvs_params(**overrides):
    p = P.CVSParams(pre_ramp_rate_Vps=100.0, sample_interval_s=0.05,
                    max_duration_s=600.0, nplc=1.0, compliance_A=1e-3,
                    i_bd_A=1e-4)
    for key, value in overrides.items():
        setattr(p, key, value)
    return p


def run_rvs(smu, params, **kwargs):
    clock = FakeClock()
    smu._clock = clock
    return ST.run_rvs(smu, params, clock=clock, sleep=clock.sleep, **kwargs)


def run_cvs(smu, params, v_stress, **kwargs):
    clock = FakeClock()
    smu._clock = clock
    return ST.run_cvs(smu, params, v_stress, clock=clock, sleep=clock.sleep, **kwargs)


# --- RVS -------------------------------------------------------------------

def test_rvs_finds_the_breakdown_voltage():
    # The ramp itself accumulates damage, so a device fails at or a little below
    # its nominal V_BD -- never above it.
    smu = MockSourceMeter(v_bd=4.2)

    result = run_rvs(smu, rvs_params())

    assert result.bd_detected
    assert 3.5 < result.v_bd <= 4.2 + rvs_params().max_step_V


def test_a_faster_ramp_gives_a_higher_breakdown_voltage():
    # Less time under stress on the way up means less cumulative damage. This
    # is exactly why the achieved rate has to be reported with V_BD.
    slow = run_rvs(MockSourceMeter(v_bd=4.2), rvs_params(ramp_rate_Vps=0.5))
    fast = run_rvs(MockSourceMeter(v_bd=4.2), rvs_params(ramp_rate_Vps=2.0))

    assert fast.v_bd > slow.v_bd


def test_rvs_reports_breakdown_as_the_reason_it_stopped():
    result = run_rvs(MockSourceMeter(v_bd=4.2), rvs_params())

    assert result.termination_reason == Termination.BREAKDOWN


def test_rvs_leaves_the_instrument_safe_after_a_breakdown():
    smu = MockSourceMeter(v_bd=4.2)

    run_rvs(smu, rvs_params())

    assert smu.voltage == 0.0
    assert not smu.output_enabled


def test_rvs_stops_at_the_ceiling_when_the_device_survives():
    smu = MockSourceMeter(v_bd=50.0)

    result = run_rvs(smu, rvs_params(v_max=6.0))

    assert not result.bd_detected
    assert result.termination_reason == Termination.CEILING_REACHED
    assert result.v_bd is None


def test_rvs_leaves_the_instrument_safe_after_surviving_the_ceiling():
    smu = MockSourceMeter(v_bd=50.0)

    run_rvs(smu, rvs_params(v_max=6.0))

    assert smu.voltage == 0.0
    assert not smu.output_enabled


def test_rvs_never_exceeds_the_requested_ceiling():
    smu = MockSourceMeter(v_bd=50.0)
    seen = []

    run_rvs(smu, rvs_params(v_max=6.0), on_point=lambda row: seen.append(row["set_V"]))

    assert max(seen) == pytest.approx(6.0)


def test_rvs_applies_negative_polarity():
    smu = MockSourceMeter(v_bd=4.2)
    seen = []

    result = run_rvs(smu, rvs_params(polarity=-1),
                     on_point=lambda row: seen.append(row["set_V"]))

    assert min(seen) < 0
    assert result.v_bd < 0


def test_negative_polarity_breakdown_is_found_at_the_same_magnitude():
    positive = run_rvs(MockSourceMeter(v_bd=4.2), rvs_params(polarity=1))
    negative = run_rvs(MockSourceMeter(v_bd=4.2), rvs_params(polarity=-1))

    assert abs(negative.v_bd) == pytest.approx(abs(positive.v_bd))


def test_rvs_records_the_rate_it_actually_achieved():
    result = run_rvs(MockSourceMeter(v_bd=50.0), rvs_params(v_max=3.0,
                                                            ramp_rate_Vps=1.0))

    assert result.achieved_rate_Vps == pytest.approx(1.0, rel=0.05)


def test_rvs_reports_the_requested_rate_alongside_the_achieved_one():
    result = run_rvs(MockSourceMeter(v_bd=50.0), rvs_params(ramp_rate_Vps=1.0))

    assert result.requested_rate_Vps == pytest.approx(1.0)


def test_an_unachievable_rate_is_noted_on_the_result():
    # 100 V/s cannot be reached with a 0.05 V step at NPLC 1.
    result = run_rvs(MockSourceMeter(v_bd=50.0),
                     rvs_params(v_max=3.0, ramp_rate_Vps=100.0))

    assert any("not achievable" in note for note in result.notes)


def test_rvs_emits_one_row_per_measured_point():
    rows = []

    result = run_rvs(MockSourceMeter(v_bd=50.0), rvs_params(v_max=1.0),
                     on_point=rows.append)

    assert len(rows) == result.point_count > 0


def test_rvs_rows_carry_the_measured_voltage_and_current():
    rows = []

    run_rvs(MockSourceMeter(v_bd=50.0), rvs_params(v_max=0.5), on_point=rows.append)

    assert "measured_V" in rows[0] and "measured_I_A" in rows[0]


def test_rvs_rows_are_labelled_as_a_ramp():
    rows = []

    run_rvs(MockSourceMeter(v_bd=50.0), rvs_params(v_max=0.5), on_point=rows.append)

    assert rows[0]["phase"] == "ramp"


def test_rvs_flags_the_row_where_breakdown_was_detected():
    rows = []

    run_rvs(MockSourceMeter(v_bd=2.0), rvs_params(), on_point=rows.append)

    assert rows[-1]["bd_flag"]
    assert not rows[0]["bd_flag"]


def test_rvs_normalises_current_by_the_crosspoint_area():
    rows = []

    run_rvs(MockSourceMeter(v_bd=50.0), rvs_params(v_max=0.5), area_um2=25.0,
            on_point=rows.append)

    assert rows[-1]["current_density_A_per_um2"] == pytest.approx(
        rows[-1]["measured_I_A"] / 25.0
    )


def test_rvs_stops_when_asked_to():
    rows = []

    result = run_rvs(MockSourceMeter(v_bd=50.0), rvs_params(v_max=6.0),
                     on_point=rows.append,
                     should_stop=lambda: len(rows) >= 5)

    assert result.termination_reason == Termination.STOPPED
    assert len(rows) == 5


def test_a_stopped_rvs_still_leaves_the_instrument_safe():
    smu = MockSourceMeter(v_bd=50.0)

    run_rvs(smu, rvs_params(), should_stop=lambda: True)

    assert smu.voltage == 0.0 and not smu.output_enabled


def test_rvs_configures_compliance_and_integration_before_sourcing():
    smu = MockSourceMeter(v_bd=50.0)

    run_rvs(smu, rvs_params(v_max=0.2, compliance_A=2e-3, nplc=0.5))

    assert smu.compliance_A == pytest.approx(2e-3)
    assert smu.nplc == pytest.approx(0.5)


def test_the_instrument_is_left_safe_even_if_a_reading_raises():
    class ExplodingSmu(MockSourceMeter):
        def read_vi(self):
            raise RuntimeError("GPIB timeout")

    smu = ExplodingSmu(v_bd=4.2)

    with pytest.raises(RuntimeError):
        run_rvs(smu, rvs_params())

    assert smu.voltage == 0.0 and not smu.output_enabled


# --- CVS -------------------------------------------------------------------

def test_cvs_holds_until_the_device_breaks_down():
    smu = MockSourceMeter(v_bd=4.2)

    result = run_cvs(smu, cvs_params(), v_stress=3.6)

    assert result.bd_detected
    assert result.termination_reason == Termination.BREAKDOWN


def test_cvs_measures_time_to_breakdown_from_the_top_of_the_pre_ramp():
    # The pre-ramp is deliberately fast; its duration must not inflate t_BD.
    smu = MockSourceMeter(v_bd=4.2)

    result = run_cvs(smu, cvs_params(pre_ramp_rate_Vps=100.0), v_stress=3.6)

    assert result.t_bd is not None
    assert result.t_bd < result.duration_s


def test_a_lower_stress_level_takes_longer_to_break_down():
    hard = run_cvs(MockSourceMeter(v_bd=4.2), cvs_params(), v_stress=4.0)
    gentle = run_cvs(MockSourceMeter(v_bd=4.2), cvs_params(), v_stress=3.4)

    assert gentle.t_bd > hard.t_bd


def test_cvs_gives_up_at_the_time_limit_on_a_device_that_survives():
    result = run_cvs(MockSourceMeter(v_bd=50.0),
                     cvs_params(max_duration_s=2.0), v_stress=3.0)

    assert not result.bd_detected
    assert result.termination_reason == Termination.DURATION_REACHED


def test_cvs_records_the_stress_level_it_used():
    result = run_cvs(MockSourceMeter(v_bd=4.2), cvs_params(), v_stress=3.6)

    assert result.v_stress == pytest.approx(3.6)


def test_cvs_applies_the_requested_polarity():
    smu = MockSourceMeter(v_bd=4.2)
    rows = []

    run_cvs(smu, cvs_params(), v_stress=3.6, polarity=-1, on_point=rows.append)

    assert min(row["set_V"] for row in rows) == pytest.approx(-3.6)


def test_cvs_labels_the_pre_ramp_separately_from_the_hold():
    rows = []

    run_cvs(MockSourceMeter(v_bd=4.2), cvs_params(), v_stress=3.6,
            on_point=rows.append)
    phases = {row["phase"] for row in rows}

    assert phases == {"preramp", "hold"}


def test_the_pre_ramp_climbs_to_the_stress_level():
    rows = []

    run_cvs(MockSourceMeter(v_bd=4.2), cvs_params(), v_stress=3.6,
            on_point=rows.append)
    preramp = [row["set_V"] for row in rows if row["phase"] == "preramp"]

    assert preramp[-1] == pytest.approx(3.6)
    assert preramp == sorted(preramp)


def test_the_hold_stays_at_the_stress_level():
    rows = []

    run_cvs(MockSourceMeter(v_bd=4.2), cvs_params(), v_stress=3.6,
            on_point=rows.append)
    hold = [row["set_V"] for row in rows if row["phase"] == "hold"]

    assert all(v == pytest.approx(3.6) for v in hold)


def test_cvs_leaves_the_instrument_safe_after_breakdown():
    smu = MockSourceMeter(v_bd=4.2)

    run_cvs(smu, cvs_params(), v_stress=3.6)

    assert smu.voltage == 0.0 and not smu.output_enabled


def test_cvs_stops_when_asked_to():
    rows = []

    result = run_cvs(MockSourceMeter(v_bd=50.0), cvs_params(max_duration_s=1e6),
                     v_stress=3.0, on_point=rows.append,
                     should_stop=lambda: len(rows) >= 6)

    assert result.termination_reason == Termination.STOPPED


def test_cvs_rejects_a_stress_level_of_zero():
    with pytest.raises(ValueError):
        run_cvs(MockSourceMeter(), cvs_params(), v_stress=0.0)


def test_cvs_elapsed_time_advances_across_the_hold():
    rows = []

    run_cvs(MockSourceMeter(v_bd=4.2), cvs_params(), v_stress=3.6,
            on_point=rows.append)
    hold = [row["elapsed_s"] for row in rows if row["phase"] == "hold"]

    assert hold == sorted(hold)
    assert hold[-1] > hold[0]
