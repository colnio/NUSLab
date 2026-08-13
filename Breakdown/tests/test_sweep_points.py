import pytest

from Breakdown.core import sweep_points as S


# --- C(F): log, f_min -> f_max -> f_min ------------------------------------

def test_cf_sweep_goes_out_and_back():
    pts = S.build_cf_points(100.0, 10000.0, points=3)
    values = [p.value for p in pts]

    assert values[0] == pytest.approx(100.0)
    assert max(values) == pytest.approx(10000.0)
    assert values[-1] == pytest.approx(100.0)


def test_cf_sweep_measures_the_turning_point_once():
    pts = S.build_cf_points(100.0, 10000.0, points=3)
    at_max = [p for p in pts if p.value == pytest.approx(10000.0)]

    assert len(at_max) == 1


def test_cf_sweep_length_is_two_passes_minus_the_shared_turning_point():
    pts = S.build_cf_points(100.0, 10000.0, points=5)

    assert len(pts) == 2 * 5 - 1


def test_cf_sweep_is_log_spaced():
    pts = S.build_cf_points(100.0, 100000.0, points=4)
    forward = [p.value for p in pts if p.segment == 0]

    ratios = [b / a for a, b in zip(forward, forward[1:])]
    assert all(r == pytest.approx(ratios[0]) for r in ratios)


def test_cf_sweep_labels_the_two_directions():
    pts = S.build_cf_points(100.0, 10000.0, points=3)

    assert pts[0].direction == "fwd"
    assert pts[-1].direction == "rev"


def test_cf_sweep_numbers_its_two_segments():
    pts = S.build_cf_points(100.0, 10000.0, points=3)

    assert sorted({p.segment for p in pts}) == [0, 1]


def test_cf_sweep_indices_are_contiguous_from_zero():
    pts = S.build_cf_points(100.0, 10000.0, points=4)

    assert [p.index for p in pts] == list(range(len(pts)))


def test_cf_sweep_with_two_points_still_returns_there_and_back():
    pts = S.build_cf_points(100.0, 10000.0, points=2)

    assert [p.value for p in pts] == pytest.approx([100.0, 10000.0, 100.0])


def test_cf_sweep_rejects_a_non_positive_start_frequency():
    with pytest.raises(ValueError):
        S.build_cf_points(0.0, 1000.0, points=5)


# --- C(V): linear, 0 -> v_max -> v_min -> 0 --------------------------------

def test_cv_sweep_traces_the_full_hysteresis_loop():
    pts = S.build_cv_points(-2.0, 2.0, points=3)
    values = [p.value for p in pts]

    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(0.0)
    assert max(values) == pytest.approx(2.0)
    assert min(values) == pytest.approx(-2.0)


def test_cv_sweep_reaches_the_maximum_before_the_minimum():
    pts = S.build_cv_points(-2.0, 2.0, points=3)
    values = [p.value for p in pts]

    assert values.index(pytest.approx(2.0)) < values.index(pytest.approx(-2.0))


def test_cv_sweep_has_no_repeated_consecutive_points():
    pts = S.build_cv_points(-2.0, 2.0, points=5)
    values = [p.value for p in pts]

    assert all(a != pytest.approx(b) for a, b in zip(values, values[1:]))


def test_cv_sweep_is_linearly_spaced_within_a_segment():
    pts = S.build_cv_points(-2.0, 2.0, points=5)
    rising = [p.value for p in pts if p.segment == 0]

    gaps = [b - a for a, b in zip(rising, rising[1:])]
    assert all(g == pytest.approx(gaps[0]) for g in gaps)


def test_cv_sweep_labels_rising_and_falling_branches():
    pts = S.build_cv_points(-2.0, 2.0, points=3)
    by_segment = {p.segment: p.direction for p in pts}

    assert by_segment[0] == "fwd"   # 0 -> v_max, increasing
    assert by_segment[1] == "rev"   # v_max -> v_min, decreasing
    assert by_segment[2] == "fwd"   # v_min -> 0, increasing again


def test_cv_sweep_numbers_its_three_segments():
    pts = S.build_cv_points(-2.0, 2.0, points=3)

    assert sorted({p.segment for p in pts}) == [0, 1, 2]


def test_cv_sweep_handles_a_range_that_does_not_straddle_zero():
    # 0 -> 3 -> 1 -> 0 : still a valid loop, just one-sided.
    pts = S.build_cv_points(1.0, 3.0, points=3)
    values = [p.value for p in pts]

    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(0.0)
    assert max(values) == pytest.approx(3.0)


def test_cv_sweep_indices_are_contiguous_from_zero():
    pts = S.build_cv_points(-2.0, 2.0, points=4)

    assert [p.index for p in pts] == list(range(len(pts)))


def test_cv_sweep_rejects_an_inverted_range():
    with pytest.raises(ValueError):
        S.build_cv_points(2.0, -2.0, points=5)
