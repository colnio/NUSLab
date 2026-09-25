import math

import pytest

from Breakdown.core import detect as D


def feed(detector, currents, start_v=1.0):
    """Push a list of currents through the detector, returning the first event."""
    for i, current in enumerate(currents):
        event = detector.update(index=i, voltage=start_v + i, current=current,
                                elapsed_s=0.1 * i)
        if event is not None:
            return event
    return None


def make(threshold=1e-4, consecutive=2):
    return D.BreakdownDetector(
        D.BreakdownCriterion(i_threshold_A=threshold, consecutive=consecutive)
    )


def test_currents_below_the_threshold_never_trigger():
    assert feed(make(), [1e-9, 1e-8, 1e-7, 1e-6]) is None


def test_a_single_spike_does_not_trigger():
    # One noisy sample must not end a device.
    assert feed(make(consecutive=2), [1e-9, 1e-3, 1e-9, 1e-9]) is None


def test_two_consecutive_over_threshold_points_trigger():
    assert feed(make(consecutive=2), [1e-9, 1e-3, 1e-3]) is not None


def test_a_broken_run_of_excursions_does_not_trigger():
    assert feed(make(consecutive=2), [1e-3, 1e-9, 1e-3, 1e-9]) is None


def test_the_event_reports_the_first_point_of_the_run_as_the_breakdown():
    # Breakdown happened when the current first jumped, not when it was confirmed.
    detector = make(consecutive=2)
    event = feed(detector, [1e-9, 1e-3, 1e-3], start_v=1.0)

    assert event.voltage == pytest.approx(2.0)   # index 1 -> 1.0 + 1
    assert event.index == 1


def test_the_event_also_records_where_it_was_confirmed():
    event = feed(make(consecutive=2), [1e-9, 1e-3, 1e-3])

    assert event.confirmed_index == 2


def test_the_event_carries_the_current_and_elapsed_time():
    event = feed(make(consecutive=2), [1e-9, 2e-3, 3e-3])

    assert event.current == pytest.approx(2e-3)
    assert event.elapsed_s == pytest.approx(0.1)


def test_a_threshold_of_one_triggers_on_the_first_excursion():
    event = feed(make(consecutive=1), [1e-9, 1e-3])

    assert event is not None and event.index == 1


def test_a_current_exactly_at_the_threshold_counts_as_an_excursion():
    assert feed(make(threshold=1e-4, consecutive=2), [1e-4, 1e-4]) is not None


def test_negative_currents_are_judged_by_magnitude():
    # A negative-polarity ramp produces negative currents; the device does not
    # care about the sign.
    assert feed(make(consecutive=2), [-1e-9, -1e-3, -1e-3]) is not None


def test_a_dropped_reading_breaks_a_consecutive_run():
    # The stress runner retries and aborts repeated invalid reads. A single NaN
    # must not join two separated excursions into a false breakdown event.
    assert feed(make(consecutive=2), [1e-9, 1e-3, math.nan, 1e-3]) is None


def test_a_dropped_reading_alone_does_not_trigger():
    assert feed(make(consecutive=2), [math.nan, math.nan, math.nan]) is None


def test_resetting_clears_a_partial_run():
    detector = make(consecutive=2)
    detector.update(index=0, voltage=1.0, current=1e-3, elapsed_s=0.0)
    detector.reset()

    assert detector.update(index=1, voltage=2.0, current=1e-3, elapsed_s=0.1) is None


def test_the_detector_latches_and_keeps_reporting_the_same_event():
    detector = make(consecutive=2)
    first = feed(detector, [1e-3, 1e-3])
    again = detector.update(index=9, voltage=99.0, current=1e-3, elapsed_s=9.9)

    assert again is first


def test_a_criterion_needing_zero_points_is_rejected():
    with pytest.raises(ValueError):
        D.BreakdownDetector(D.BreakdownCriterion(i_threshold_A=1e-4, consecutive=0))


def test_a_non_positive_threshold_is_rejected():
    with pytest.raises(ValueError):
        D.BreakdownDetector(D.BreakdownCriterion(i_threshold_A=0.0, consecutive=2))
