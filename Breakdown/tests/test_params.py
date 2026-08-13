import json

import pytest

from Breakdown.core import params as P


def test_area_is_derived_from_crosspoint_size():
    sample = P.SampleParams(sample_name="A1", crosspoint_um=5.0)
    assert sample.area_um2 == pytest.approx(25.0)


def test_save_then_load_round_trips_every_field(tmp_path):
    original = P.BreakdownParams()
    original.sample.sample_name = "waferB"
    original.sample.crosspoint_um = 20.0
    original.sample.thickness_nm = 8.5
    original.rvs.ramp_rate_Vps = 2.5
    original.cvs.max_duration_s = 7200.0
    original.run.mode = "alternating"

    path = tmp_path / "p.json"
    P.save_params(str(path), original)
    loaded = P.load_params(str(path))

    assert loaded.warnings == []
    assert loaded.params == original


def test_saved_file_records_the_schema_version(tmp_path):
    path = tmp_path / "p.json"
    P.save_params(str(path), P.BreakdownParams())

    payload = json.loads(path.read_text())

    assert payload["schema_version"] == P.SCHEMA_VERSION


def test_unknown_keys_are_warned_about_not_raised(tmp_path):
    path = tmp_path / "p.json"
    payload = P.to_dict(P.BreakdownParams())
    payload["rvs"]["some_retired_field"] = 123
    path.write_text(json.dumps(payload))

    loaded = P.load_params(str(path))

    assert loaded.params.rvs.ramp_rate_Vps == P.RVSParams().ramp_rate_Vps
    assert any("some_retired_field" in w for w in loaded.warnings)


def test_missing_keys_fall_back_to_defaults(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"schema_version": P.SCHEMA_VERSION,
                                "sample": {"sample_name": "only_this"}}))

    loaded = P.load_params(str(path))

    assert loaded.params.sample.sample_name == "only_this"
    assert loaded.params.cf.f_min == P.CFParams().f_min
    assert loaded.params.run.mode == P.RunParams().mode


def test_a_newer_schema_version_is_warned_about(tmp_path):
    path = tmp_path / "p.json"
    payload = P.to_dict(P.BreakdownParams())
    payload["schema_version"] = P.SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload))

    loaded = P.load_params(str(path))

    assert any("schema_version" in w for w in loaded.warnings)


def test_default_params_validate_clean():
    assert P.validate(P.BreakdownParams()) == []


def test_rvs_breakdown_threshold_at_or_above_compliance_is_rejected():
    p = P.BreakdownParams()
    p.rvs.compliance_A = 1e-3
    p.rvs.i_bd_A = 1e-3

    errors = P.validate(p)

    assert any("i_bd_A" in e and "compliance" in e for e in errors)


def test_cvs_breakdown_threshold_at_or_above_compliance_is_rejected():
    p = P.BreakdownParams()
    p.cvs.compliance_A = 1e-3
    p.cvs.i_bd_A = 5e-3

    errors = P.validate(p)

    assert any("CVS" in e and "i_bd_A" in e for e in errors)


def test_ramp_beyond_the_absolute_voltage_ceiling_is_rejected():
    p = P.BreakdownParams()
    p.run.abs_max_voltage_V = 20.0
    p.rvs.v_max = 25.0

    errors = P.validate(p)

    assert any("v_max" in e for e in errors)


def test_non_positive_start_frequency_is_rejected():
    p = P.BreakdownParams()
    p.cf.f_min = 0.0

    errors = P.validate(p)

    assert any("f_min" in e for e in errors)


def test_inverted_frequency_range_is_rejected():
    p = P.BreakdownParams()
    p.cf.f_min = 1e6
    p.cf.f_max = 1e3

    errors = P.validate(p)

    assert any("f_min" in e and "f_max" in e for e in errors)


def test_fewer_than_two_sweep_points_is_rejected():
    p = P.BreakdownParams()
    p.cv.points = 1

    errors = P.validate(p)

    assert any("points" in e for e in errors)


def test_non_positive_cvs_sample_interval_is_rejected():
    p = P.BreakdownParams()
    p.cvs.sample_interval_s = 0.0

    errors = P.validate(p)

    assert any("sample_interval_s" in e for e in errors)


def test_mfia_bias_beyond_the_four_terminal_limit_is_rejected():
    p = P.BreakdownParams()
    p.mfia.four_terminal = True   # -> +/- 3 V ceiling
    p.cv.v_max = 5.0

    errors = P.validate(p)

    assert any("3" in e and "bias" in e.lower() for e in errors)


def test_mfia_bias_within_the_two_terminal_limit_is_accepted():
    p = P.BreakdownParams()
    p.mfia.four_terminal = False  # -> +/- 10 V ceiling
    p.cv.v_max = 5.0
    p.cv.v_min = -5.0

    assert P.validate(p) == []


def test_polarity_must_be_plus_or_minus_one():
    p = P.BreakdownParams()
    p.rvs.polarity = 0

    errors = P.validate(p)

    assert any("polarity" in e for e in errors)


def test_unknown_run_mode_is_rejected():
    p = P.BreakdownParams()
    p.run.mode = "sideways"

    errors = P.validate(p)

    assert any("mode" in e for e in errors)
