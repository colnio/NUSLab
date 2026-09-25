import pytest

from Breakdown.core import ramp as R


# --- period floor ----------------------------------------------------------

def test_point_period_covers_integration_source_delay_and_bus_overhead():
    # NPLC 1 at 50 Hz = 20 ms integration, + 5 ms source delay, + 10 ms bus.
    period = R.point_period_s(nplc=1.0, source_delay_s=0.005,
                              line_freq_hz=50.0, bus_overhead_s=0.010)

    assert period == pytest.approx(0.035)


def test_sixty_hertz_mains_integrates_faster_than_fifty():
    at_50 = R.point_period_s(nplc=1.0, line_freq_hz=50.0)
    at_60 = R.point_period_s(nplc=1.0, line_freq_hz=60.0)

    assert at_60 < at_50


# --- ramp planning ---------------------------------------------------------

def test_an_achievable_rate_is_delivered_exactly():
    plan = R.plan_ramp(rate_Vps=1.0, max_step_V=0.05, nplc=1.0,
                       source_delay_s=0.0, bus_overhead_s=0.010)

    assert plan.achievable_rate_Vps == pytest.approx(1.0)
    assert not plan.rate_limited


def test_an_achievable_rate_uses_the_shortest_valid_dwell():
    plan = R.plan_ramp(rate_Vps=1.0, max_step_V=0.05, nplc=1.0,
                       source_delay_s=0.0, bus_overhead_s=0.010)

    assert plan.dwell_s == pytest.approx(0.030)
    assert plan.step_V == pytest.approx(0.030)


def test_a_rate_needing_a_coarser_step_than_allowed_is_capped_and_flagged():
    # 10 V/s would need a 0.3 V step at a 30 ms period; max_step_V caps it.
    plan = R.plan_ramp(rate_Vps=10.0, max_step_V=0.05, nplc=1.0,
                       source_delay_s=0.0, bus_overhead_s=0.010)

    assert plan.step_V == pytest.approx(0.05)
    assert plan.achievable_rate_Vps == pytest.approx(0.05 / 0.030)
    assert plan.rate_limited


def test_a_capped_plan_explains_the_shortfall():
    plan = R.plan_ramp(rate_Vps=10.0, max_step_V=0.05, nplc=1.0,
                       source_delay_s=0.0, bus_overhead_s=0.010)

    assert "10" in plan.note and "max_step_V" in plan.note


def test_lowering_the_integration_time_raises_the_achievable_rate():
    slow = R.plan_ramp(rate_Vps=10.0, max_step_V=0.05, nplc=1.0)
    fast = R.plan_ramp(rate_Vps=10.0, max_step_V=0.05, nplc=0.1)

    assert fast.achievable_rate_Vps > slow.achievable_rate_Vps


def test_a_very_slow_rate_waits_longer_instead_of_using_a_meaningless_step():
    plan = R.plan_ramp(rate_Vps=0.001, max_step_V=0.05, nplc=1.0,
                       min_step_V=1e-4, source_delay_s=0.0, bus_overhead_s=0.010)

    assert plan.step_V == pytest.approx(1e-4)
    assert plan.dwell_s == pytest.approx(0.1)
    assert plan.achievable_rate_Vps == pytest.approx(0.001)
    assert not plan.rate_limited


def test_planning_rejects_a_non_positive_rate():
    with pytest.raises(ValueError):
        R.plan_ramp(rate_Vps=0.0, max_step_V=0.05, nplc=1.0)


def test_planning_rejects_a_non_positive_max_step():
    with pytest.raises(ValueError):
        R.plan_ramp(rate_Vps=1.0, max_step_V=0.0, nplc=1.0)


# --- voltage list ----------------------------------------------------------

def test_ramp_values_start_after_the_origin_and_land_on_the_target():
    values = R.build_ramp_values(0.0, 1.0, step=0.25)

    assert values == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_ramp_values_always_include_the_exact_endpoint():
    values = R.build_ramp_values(0.0, 1.0, step=0.3)

    assert values[-1] == pytest.approx(1.0)


def test_ramp_values_never_overshoot_the_target():
    values = R.build_ramp_values(0.0, 1.0, step=0.3)

    assert max(values) == pytest.approx(1.0)


def test_ramp_values_descend_for_a_negative_target():
    values = R.build_ramp_values(0.0, -1.0, step=0.5)

    assert values == pytest.approx([-0.5, -1.0])


def test_ramp_values_from_a_nonzero_start():
    values = R.build_ramp_values(1.0, 2.0, step=0.5)

    assert values == pytest.approx([1.5, 2.0])


def test_ramp_to_the_current_value_produces_no_steps():
    assert R.build_ramp_values(1.0, 1.0, step=0.5) == []


def test_estimated_duration_matches_the_point_count():
    plan = R.plan_ramp(rate_Vps=1.0, max_step_V=0.05, nplc=1.0,
                       source_delay_s=0.0, bus_overhead_s=0.010)

    # 0 -> 3 V at a 0.03 V step = 100 points at 30 ms.
    assert R.estimate_duration_s(plan, 0.0, 3.0) == pytest.approx(3.0, rel=0.02)
