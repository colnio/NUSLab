import math

import pytest

from Breakdown.core import models as M


def test_the_default_model_is_parallel_rp_cp():
    assert M.MODEL_INFO[0] == ("Rp || Cp", "Rp", "Cp")


def test_every_model_has_a_short_tag_for_filenames():
    assert set(M.MODEL_TAGS) == set(M.MODEL_INFO)


def test_parallel_rp_cp_reads_capacitance_from_param1():
    cap, _ = M.compute_model_values(model=0, param0=1e9, param1=3.3e-12, freq_hz=1000.0)

    assert cap == pytest.approx(3.3e-12)


def test_parallel_rp_cp_reads_resistance_from_param0():
    _, res = M.compute_model_values(model=0, param0=1e9, param1=3.3e-12, freq_hz=1000.0)

    assert res == pytest.approx(1e9)


def test_series_rs_cs_also_reads_capacitance_from_param1():
    cap, res = M.compute_model_values(model=1, param0=50.0, param1=1e-9, freq_hz=1000.0)

    assert cap == pytest.approx(1e-9)
    assert res == pytest.approx(50.0)


def test_conductance_susceptance_converts_susceptance_to_capacitance():
    # B = 2*pi*f*C, so C = B / (2*pi*f)
    freq = 1000.0
    susceptance = 2 * math.pi * freq * 4.7e-12

    cap, _ = M.compute_model_values(model=3, param0=1e-9, param1=susceptance,
                                    freq_hz=freq)

    assert cap == pytest.approx(4.7e-12)


def test_conductance_susceptance_inverts_conductance_to_resistance():
    _, res = M.compute_model_values(model=3, param0=1e-6, param1=0.0, freq_hz=1000.0)

    assert res == pytest.approx(1e6)


def test_zero_conductance_gives_no_resistance_rather_than_dividing_by_zero():
    _, res = M.compute_model_values(model=3, param0=0.0, param1=0.0, freq_hz=1000.0)

    assert math.isnan(res)


def test_susceptance_without_a_frequency_gives_no_capacitance():
    cap, _ = M.compute_model_values(model=3, param0=1e-6, param1=1.0, freq_hz=0.0)

    assert math.isnan(cap)


def test_an_inductive_model_reports_no_capacitance():
    cap, res = M.compute_model_values(model=2, param0=50.0, param1=1e-6,
                                      freq_hz=1000.0)

    assert math.isnan(cap)
    assert res == pytest.approx(50.0)


def test_an_unknown_model_index_reports_neither_quantity():
    cap, res = M.compute_model_values(model=99, param0=1.0, param1=2.0,
                                      freq_hz=1000.0)

    assert math.isnan(cap) and math.isnan(res)


def test_non_numeric_parameters_do_not_raise():
    cap, res = M.compute_model_values(model=0, param0=None, param1="oops",
                                      freq_hz=1000.0)

    assert math.isnan(cap) and math.isnan(res)


# --- impedance decomposition ----------------------------------------------

def test_impedance_is_split_into_real_and_imaginary_parts():
    parts = M.decompose_impedance(complex(3.0, -4.0))

    assert parts["Z_real_Ohm"] == pytest.approx(3.0)
    assert parts["Z_imag_Ohm"] == pytest.approx(-4.0)


def test_impedance_magnitude_and_phase_are_derived():
    parts = M.decompose_impedance(complex(3.0, -4.0))

    assert parts["Z_abs_Ohm"] == pytest.approx(5.0)
    assert parts["Z_phase_rad"] == pytest.approx(math.atan2(-4.0, 3.0))


def test_a_missing_impedance_yields_nan_parts():
    parts = M.decompose_impedance(None)

    assert all(math.isnan(v) for v in parts.values())


# --- area normalisation ----------------------------------------------------

def test_capacitance_per_area_divides_by_the_crosspoint_area():
    assert M.per_area(2.5e-12, area_um2=25.0) == pytest.approx(1e-13)


def test_per_area_of_a_missing_value_is_nan():
    assert math.isnan(M.per_area(float("nan"), area_um2=25.0))


def test_per_area_with_no_area_is_nan():
    assert math.isnan(M.per_area(1e-12, area_um2=0.0))
