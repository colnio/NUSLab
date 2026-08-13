import math

import pytest

from Breakdown.core.smu import OVERFLOW_SENTINEL, _parse_reading


def test_a_voltage_current_pair_is_split_in_order():
    # :FORM:ELEM VOLT,CURR puts the sourced voltage first.
    voltage, current = _parse_reading("2.000000E+00,1.234500E-06")

    assert voltage == pytest.approx(2.0)
    assert current == pytest.approx(1.2345e-6)


def test_extra_trailing_elements_are_ignored():
    voltage, current = _parse_reading("2.0,1e-6,0.5,1.0,0")

    assert voltage == pytest.approx(2.0)
    assert current == pytest.approx(1e-6)


def test_an_overflowed_current_becomes_not_a_number():
    # 9.9e37 is the 2400's "invalid reading" sentinel, not a real current.
    _, current = _parse_reading(f"2.0,{OVERFLOW_SENTINEL}")

    assert math.isnan(current)


def test_an_overflowed_voltage_becomes_not_a_number():
    voltage, _ = _parse_reading(f"{OVERFLOW_SENTINEL},1e-6")

    assert math.isnan(voltage)


def test_a_lone_reading_is_taken_as_the_current():
    voltage, current = _parse_reading("1.5e-6")

    assert math.isnan(voltage)
    assert current == pytest.approx(1.5e-6)


def test_an_empty_reply_yields_no_numbers():
    voltage, current = _parse_reading("")

    assert math.isnan(voltage) and math.isnan(current)


def test_garbled_tokens_are_skipped_rather_than_raising():
    # A truncated GPIB reply must not crash a destructive run mid-ramp.
    voltage, current = _parse_reading("2.0,NAN?,1e-6")

    assert voltage == pytest.approx(2.0)
    assert current == pytest.approx(1e-6)


def test_surrounding_whitespace_and_newlines_are_tolerated():
    voltage, current = _parse_reading("  2.0 , 1e-6 \n")

    assert voltage == pytest.approx(2.0)
    assert current == pytest.approx(1e-6)


def test_a_negative_current_keeps_its_sign():
    _, current = _parse_reading("-2.0,-1.5e-6")

    assert current == pytest.approx(-1.5e-6)
