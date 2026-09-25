import math

import pytest

from Breakdown.core.smu import OVERFLOW_SENTINEL, SmuSession, _parse_reading


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


def test_safe_off_attempts_output_off_even_when_zeroing_fails():
    class Device:
        def __init__(self):
            self.disabled = False

        def set_voltage(self, _value):
            raise RuntimeError("write failed")

        def disable_output(self):
            self.disabled = True

    session = SmuSession("test")
    session.device = Device()

    with pytest.raises(RuntimeError, match="zero failed"):
        session.safe_off()

    assert session.device.disabled


def test_critical_smu_setting_is_retried_after_bad_readback():
    class Resource:
        def __init__(self):
            self.queries = 0

        def write(self, command):
            if command.endswith("?"):
                self.queries += 1

        def read(self):
            return "0" if self.queries == 1 else "1"

    session = SmuSession("test")
    session.resource = Resource()
    actions = []

    session._apply_verified_number(
        lambda: actions.append(True), ":TEST?", 1.0, "test setting"
    )

    assert len(actions) == 2
