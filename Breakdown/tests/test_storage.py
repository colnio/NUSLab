import csv
import json

import pytest

from Breakdown.core import storage as S


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- MeasurementWriter -----------------------------------------------------

def test_writer_creates_the_file_with_a_header(tmp_path):
    path = tmp_path / "cf.csv"
    with S.MeasurementWriter(str(path), S.MFIA_COLUMNS):
        pass

    with open(path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == list(S.MFIA_COLUMNS)


def test_writer_appends_rows_in_order(tmp_path):
    path = tmp_path / "cf.csv"
    with S.MeasurementWriter(str(path), S.MFIA_COLUMNS) as writer:
        writer.write_row({"point_index": 0, "C_F": 1e-12})
        writer.write_row({"point_index": 1, "C_F": 2e-12})

    rows = read_rows(path)
    assert [r["point_index"] for r in rows] == ["0", "1"]


def test_rows_are_on_disk_before_the_writer_is_closed(tmp_path):
    # A stress run can be killed at any moment. Data written so far must survive.
    path = tmp_path / "rvs.csv"
    with S.MeasurementWriter(str(path), S.STRESS_COLUMNS) as writer:
        writer.write_row({"point_index": 0, "measured_I_A": 1e-9})

        rows = read_rows(path)
        assert len(rows) == 1


def test_missing_fields_are_written_blank(tmp_path):
    path = tmp_path / "cf.csv"
    with S.MeasurementWriter(str(path), S.MFIA_COLUMNS) as writer:
        writer.write_row({"point_index": 0})

    assert read_rows(path)[0]["C_F"] == ""


def test_an_unexpected_column_is_rejected_rather_than_dropped(tmp_path):
    # Silently discarding a field would lose data with no trace.
    path = tmp_path / "cf.csv"
    with S.MeasurementWriter(str(path), S.MFIA_COLUMNS) as writer:
        with pytest.raises(ValueError):
            writer.write_row({"not_a_real_column": 1})


def test_writer_counts_what_it_has_written(tmp_path):
    path = tmp_path / "cf.csv"
    with S.MeasurementWriter(str(path), S.MFIA_COLUMNS) as writer:
        writer.write_row({"point_index": 0})
        writer.write_row({"point_index": 1})

        assert writer.row_count == 2


def test_writer_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "deep" / "deeper" / "cf.csv"
    with S.MeasurementWriter(str(path), S.MFIA_COLUMNS) as writer:
        writer.write_row({"point_index": 0})

    assert path.is_file()


# --- SummaryWriter ---------------------------------------------------------

def test_summary_writes_a_header_then_the_row(tmp_path):
    path = tmp_path / "summary.csv"
    S.SummaryWriter(str(path)).append({"device_index": 1, "V_BD_V": 4.2})

    rows = read_rows(path)
    assert rows[0]["device_index"] == "1"
    assert rows[0]["V_BD_V"] == "4.2"


def test_summary_appends_without_repeating_the_header(tmp_path):
    path = tmp_path / "summary.csv"
    S.SummaryWriter(str(path)).append({"device_index": 1})
    S.SummaryWriter(str(path)).append({"device_index": 2})

    rows = read_rows(path)
    assert [r["device_index"] for r in rows] == ["1", "2"]


def test_summary_survives_the_program_being_restarted(tmp_path):
    # Each append opens and closes, so a crash between devices loses nothing.
    path = tmp_path / "summary.csv"
    writer = S.SummaryWriter(str(path))
    writer.append({"device_index": 1})

    assert len(read_rows(path)) == 1


def test_summary_rejects_an_unknown_column(tmp_path):
    path = tmp_path / "summary.csv"
    with pytest.raises(ValueError):
        S.SummaryWriter(str(path)).append({"nonsense": 1})


# --- device metadata -------------------------------------------------------

def test_device_metadata_is_written_as_readable_json(tmp_path):
    path = tmp_path / "dev001.meta.json"
    S.write_device_meta(str(path), {"device_index": 1, "stress_type": "RVS"})

    payload = json.loads(path.read_text())
    assert payload["stress_type"] == "RVS"


# --- derived quantities ----------------------------------------------------

def test_breakdown_field_is_reported_in_MV_per_cm():
    # 4 V across 8 nm = 5e8 V/m = 5 MV/cm
    assert S.breakdown_field_MV_per_cm(4.0, 8.0) == pytest.approx(5.0)


def test_breakdown_field_is_unavailable_without_a_thickness():
    assert S.breakdown_field_MV_per_cm(4.0, None) is None


def test_breakdown_field_uses_the_magnitude_of_a_negative_ramp():
    assert S.breakdown_field_MV_per_cm(-4.0, 8.0) == pytest.approx(5.0)


def test_breakdown_field_is_unavailable_without_a_breakdown():
    assert S.breakdown_field_MV_per_cm(None, 8.0) is None
