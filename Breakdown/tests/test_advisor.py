import pytest

from Breakdown.core import advisor as A


def stats_with(*voltages, size=5.0):
    stats = A.VbdStatistics()
    for i, v in enumerate(voltages, start=1):
        stats.add(A.VbdRecord(device_index=i, crosspoint_um=size, v_bd=v,
                              ramp_rate_Vps=1.0))
    return stats


# --- statistics ------------------------------------------------------------

def test_median_of_an_odd_sample():
    assert stats_with(4.0, 4.2, 4.4).statistic(5.0) == pytest.approx(4.2)


def test_median_of_an_even_sample_averages_the_middle_pair():
    assert stats_with(4.0, 4.2, 4.4, 4.6).statistic(5.0) == pytest.approx(4.3)


def test_median_ignores_the_order_records_arrive_in():
    assert stats_with(4.4, 4.0, 4.2).statistic(5.0) == pytest.approx(4.2)


def test_median_resists_a_single_outlier():
    # The reason the default statistic is the median rather than the mean.
    assert stats_with(4.0, 4.2, 4.4, 40.0).statistic(5.0) == pytest.approx(4.3)


def test_mean_is_available_when_asked_for():
    assert stats_with(4.0, 5.0).statistic(5.0, kind="mean") == pytest.approx(4.5)


def test_breakdown_voltages_are_stored_as_magnitudes():
    # A negative-polarity campaign yields negative V_BD; statistics work on size.
    assert stats_with(-4.0, -4.2, -4.4).statistic(5.0) == pytest.approx(4.2)


def test_records_are_kept_separate_per_crosspoint_size():
    stats = stats_with(4.0, 4.2, 4.4, size=5.0)
    stats.add(A.VbdRecord(device_index=9, crosspoint_um=20.0, v_bd=1.0,
                          ramp_rate_Vps=1.0))

    assert stats.statistic(5.0) == pytest.approx(4.2)
    assert stats.count(5.0) == 3
    assert stats.count(20.0) == 1


def test_an_unseen_size_has_no_statistic():
    assert stats_with(4.0).statistic(20.0) is None


def test_count_of_an_unseen_size_is_zero():
    assert stats_with(4.0).count(20.0) == 0


# --- recommendation --------------------------------------------------------

def test_recommendation_is_the_fraction_of_the_median():
    rec = A.recommend(stats_with(4.0, 4.2, 4.4), crosspoint_um=5.0, k=0.85)

    assert rec.voltage == pytest.approx(0.85 * 4.2)


def test_recommendation_explains_where_the_number_came_from():
    rec = A.recommend(stats_with(4.0, 4.2, 4.4), crosspoint_um=5.0, k=0.85)

    assert "median" in rec.basis_text
    assert "3" in rec.basis_text          # device count
    assert "0.85" in rec.basis_text


def test_no_prior_breakdowns_yields_no_recommendation():
    rec = A.recommend(A.VbdStatistics(), crosspoint_um=5.0, k=0.85)

    assert rec.voltage is None
    assert any("no rvs" in w.lower() or "no breakdown" in w.lower()
               for w in rec.warnings)


def test_a_thin_sample_is_flagged_as_a_weak_basis():
    rec = A.recommend(stats_with(4.0, 4.2), crosspoint_um=5.0, k=0.85)

    assert rec.voltage is not None
    assert any("weak" in w.lower() or "only 2" in w.lower() for w in rec.warnings)


def test_an_adequate_sample_is_not_flagged_as_weak():
    rec = A.recommend(stats_with(4.0, 4.1, 4.2, 4.3), crosspoint_um=5.0, k=0.85)

    assert not any("weak" in w.lower() for w in rec.warnings)


def test_a_stress_level_near_the_breakdown_voltage_is_flagged():
    rec = A.recommend(stats_with(4.0, 4.1, 4.2), crosspoint_um=5.0, k=0.97)

    assert any("pre-ramp" in w.lower() or "immediate" in w.lower()
               for w in rec.warnings)


def test_a_very_low_stress_level_is_flagged_as_impractically_slow():
    rec = A.recommend(stats_with(4.0, 4.1, 4.2), crosspoint_um=5.0, k=0.5)

    assert any("window" in w.lower() or "long" in w.lower() for w in rec.warnings)


def test_a_stress_level_in_the_useful_band_raises_no_level_warning():
    rec = A.recommend(stats_with(4.0, 4.1, 4.2, 4.3), crosspoint_um=5.0, k=0.85)

    assert rec.warnings == []


def test_recommendation_only_uses_records_of_the_matching_size():
    stats = stats_with(4.0, 4.2, 4.4, size=5.0)
    stats.add(A.VbdRecord(device_index=9, crosspoint_um=20.0, v_bd=100.0,
                          ramp_rate_Vps=1.0))

    rec = A.recommend(stats, crosspoint_um=5.0, k=0.85)

    assert rec.voltage == pytest.approx(0.85 * 4.2)


def test_a_spread_of_ramp_rates_is_flagged_as_not_comparable():
    # V_BD is ramp-rate dependent; pooling rates makes the median meaningless.
    stats = A.VbdStatistics()
    stats.add(A.VbdRecord(1, 5.0, 4.0, ramp_rate_Vps=1.0))
    stats.add(A.VbdRecord(2, 5.0, 4.2, ramp_rate_Vps=1.0))
    stats.add(A.VbdRecord(3, 5.0, 5.5, ramp_rate_Vps=50.0))

    rec = A.recommend(stats, crosspoint_um=5.0, k=0.85)

    assert any("ramp rate" in w.lower() for w in rec.warnings)
