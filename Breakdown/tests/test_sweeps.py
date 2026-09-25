import math

import pytest

from Breakdown.core import params as P
from Breakdown.core import sweeps as SW
from Breakdown.core.instruments import Termination
from Breakdown.core.mock import MockImpedanceAnalyzer


def no_sleep(_seconds):
    pass


def cf_params(**overrides):
    p = P.CFParams(f_min=100.0, f_max=10000.0, points=4, bias_V=0.0,
                   amplitude_V=0.05, settle_s=0.0)
    for key, value in overrides.items():
        setattr(p, key, value)
    return p


def cv_params(**overrides):
    p = P.CVParams(v_min=-2.0, v_max=2.0, points=4, frequency_Hz=1000.0,
                   amplitude_V=0.05, settle_s=0.0)
    for key, value in overrides.items():
        setattr(p, key, value)
    return p


def run_cf(analyzer=None, params=None, **kwargs):
    analyzer = analyzer or MockImpedanceAnalyzer()
    rows = kwargs.pop("rows", [])
    result = SW.run_cf_sweep(analyzer, params or cf_params(), P.MfiaParams(),
                             on_point=rows.append, sleep=no_sleep, **kwargs)
    return analyzer, rows, result


def run_cv(analyzer=None, params=None, **kwargs):
    analyzer = analyzer or MockImpedanceAnalyzer()
    rows = kwargs.pop("rows", [])
    result = SW.run_cv_sweep(analyzer, params or cv_params(), P.MfiaParams(),
                             on_point=rows.append, sleep=no_sleep, **kwargs)
    return analyzer, rows, result


# --- C(F) ------------------------------------------------------------------

def test_cf_sweep_measures_every_point_of_the_out_and_back():
    _, rows, result = run_cf(params=cf_params(points=4))

    assert len(rows) == 2 * 4 - 1 == result.point_count


def test_cf_sweep_records_frequency_as_the_swept_variable():
    _, rows, _ = run_cf()

    assert rows[0]["sweep_variable"] == "frequency"


def test_cf_sweep_actually_sets_each_frequency_on_the_instrument():
    analyzer, rows, _ = run_cf()

    assert analyzer.frequency == pytest.approx(rows[-1]["set_frequency_Hz"])


def test_cf_sweep_holds_the_bias_fixed_throughout():
    _, rows, _ = run_cf(params=cf_params(bias_V=0.5))

    assert all(row["set_bias_V"] == pytest.approx(0.5) for row in rows)


def test_cf_sweep_carries_the_direction_of_each_branch():
    _, rows, _ = run_cf()

    assert rows[0]["direction"] == "fwd"
    assert rows[-1]["direction"] == "rev"


def test_cf_sweep_numbers_its_segments():
    _, rows, _ = run_cf()

    assert sorted({row["segment"] for row in rows}) == [0, 1]


def test_cf_sweep_reports_capacitance_from_the_impedance_model():
    _, rows, _ = run_cf()

    assert rows[0]["C_F"] > 0
    assert math.isfinite(rows[0]["C_F"])


def test_cf_sweep_normalises_capacitance_by_area():
    _, rows, _ = run_cf(area_um2=25.0)

    assert rows[0]["C_per_area_F_per_um2"] == pytest.approx(rows[0]["C_F"] / 25.0)


def test_cf_sweep_decomposes_the_impedance():
    _, rows, _ = run_cf()

    assert math.isfinite(rows[0]["Z_abs_Ohm"])
    assert math.isfinite(rows[0]["Z_phase_rad"])


def test_cf_sweep_reports_the_capacitance_at_the_reference_frequency():
    _, rows, result = run_cf(reference_hz=1000.0)
    at_reference = [r["C_F"] for r in rows
                    if r["set_frequency_Hz"] == pytest.approx(result.f_reference_Hz)]

    assert result.c_reference_F == pytest.approx(at_reference[0])


def test_the_reference_frequency_is_the_nearest_one_actually_measured():
    _, _, result = run_cf(params=cf_params(f_min=100.0, f_max=10000.0, points=3),
                          reference_hz=900.0)

    # Points are 100, 1000, 10000 Hz; 1000 is nearest to the requested 900.
    assert result.f_reference_Hz == pytest.approx(1000.0)


def test_cf_sweep_configures_the_instrument_before_measuring():
    analyzer, _, _ = run_cf()

    assert analyzer.configured


def test_cf_sweep_returns_the_bias_to_zero_when_it_finishes():
    analyzer, _, _ = run_cf(params=cf_params(bias_V=1.0))

    assert analyzer.bias == pytest.approx(0.0)


def test_cf_sweep_stops_when_asked_to():
    rows = []
    _, _, result = run_cf(rows=rows, should_stop=lambda: len(rows) >= 3)

    assert result.termination_reason == Termination.STOPPED
    assert len(rows) == 3


def test_cf_sweep_reports_completion_when_it_runs_to_the_end():
    _, _, result = run_cf()

    assert result.termination_reason == Termination.COMPLETED


def test_the_bias_is_returned_to_zero_even_if_a_read_raises():
    class ExplodingAnalyzer(MockImpedanceAnalyzer):
        def read_sample(self):
            raise RuntimeError("poll timeout")

    analyzer = ExplodingAnalyzer()

    with pytest.raises(RuntimeError):
        run_cf(analyzer=analyzer, params=cf_params(bias_V=1.0))

    assert analyzer.bias == pytest.approx(0.0)


def test_an_invalid_mfia_payload_is_retried_once():
    class FlakyAnalyzer(MockImpedanceAnalyzer):
        def __init__(self):
            super().__init__()
            self.reads = 0

        def read_sample(self):
            self.reads += 1
            if self.reads == 1:
                return {"param0": float("nan")}
            return super().read_sample()

    analyzer = FlakyAnalyzer()
    _, rows, _ = run_cf(analyzer=analyzer)

    assert rows
    assert analyzer.reads == len(rows) + 1


def test_two_invalid_mfia_payloads_abort_the_sweep_safely():
    class InvalidAnalyzer(MockImpedanceAnalyzer):
        def read_sample(self):
            return {"param0": float("nan")}

    analyzer = InvalidAnalyzer()

    with pytest.raises(RuntimeError, match="validation twice"):
        run_cf(analyzer=analyzer, params=cf_params(bias_V=1.0))

    assert analyzer.bias == pytest.approx(0.0)


# --- C(V) ------------------------------------------------------------------

def test_cv_sweep_measures_the_whole_hysteresis_loop():
    _, rows, result = run_cv(params=cv_params(points=4))
    biases = [row["sweep_value"] for row in rows]

    assert biases[0] == pytest.approx(0.0)
    assert biases[-1] == pytest.approx(0.0)
    assert max(biases) == pytest.approx(2.0)
    assert min(biases) == pytest.approx(-2.0)
    assert result.point_count == len(rows)


def test_cv_sweep_records_bias_as_the_swept_variable():
    _, rows, _ = run_cv()

    assert rows[0]["sweep_variable"] == "bias"


def test_cv_sweep_holds_the_frequency_fixed_throughout():
    _, rows, _ = run_cv(params=cv_params(frequency_Hz=5000.0))

    assert all(row["set_frequency_Hz"] == pytest.approx(5000.0) for row in rows)


def test_cv_sweep_has_three_segments():
    _, rows, _ = run_cv()

    assert sorted({row["segment"] for row in rows}) == [0, 1, 2]


def test_cv_sweep_shows_hysteresis_between_the_rising_and_falling_branches():
    # The two branches must be distinguishable, or the loop was pointless.
    # Absolute tolerance has to be pinned off, or picofarad values all compare
    # equal under pytest.approx's 1e-12 default.
    _, rows, _ = run_cv()
    rising = [r["C_F"] for r in rows if r["segment"] == 0]
    falling = [r["C_F"] for r in rows if r["segment"] == 1]

    assert rising[1] != pytest.approx(falling[-2], rel=1e-3, abs=0.0)


def test_cv_sweep_reports_the_zero_bias_capacitance_as_its_reference():
    _, rows, result = run_cv()
    at_zero = [r["C_F"] for r in rows if r["sweep_value"] == pytest.approx(0.0)]

    assert result.c_reference_F == pytest.approx(at_zero[0])


def test_cv_sweep_reference_frequency_is_the_measurement_frequency():
    _, _, result = run_cv(params=cv_params(frequency_Hz=5000.0))

    assert result.f_reference_Hz == pytest.approx(5000.0)


def test_cv_sweep_ends_with_the_bias_back_at_zero():
    analyzer, _, _ = run_cv()

    assert analyzer.bias == pytest.approx(0.0)


def test_cv_sweep_stops_when_asked_to():
    rows = []
    _, _, result = run_cv(rows=rows, should_stop=lambda: len(rows) >= 2)

    assert result.termination_reason == Termination.STOPPED


def test_cv_sweep_sets_the_drive_amplitude():
    analyzer, _, _ = run_cv(params=cv_params(amplitude_V=0.02))

    assert analyzer.amplitude == pytest.approx(0.02)


def test_cv_bias_tolerance_is_half_the_actual_local_point_spacing():
    analyzer, _, _ = run_cv(
        params=cv_params(v_min=-0.3, v_max=0.3, points=51)
    )
    measured = list(zip(
        analyzer.bias_history[:-1],
        analyzer.bias_readback_tolerances[:-1],
    ))
    _, tolerance = min(measured, key=lambda item: abs(item[0] - 0.078))

    assert tolerance == pytest.approx(0.003)
    assert analyzer.bias_readback_tolerances[-1] is None  # cleanup zero
