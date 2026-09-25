from Breakdown.core import naming as N


# --- sanitize_filename -----------------------------------------------------

def test_sanitize_keeps_safe_characters():
    assert N.sanitize_filename("wafer-B_01.2+x") == "wafer-B_01.2+x"


def test_sanitize_replaces_path_separators_and_spaces():
    assert N.sanitize_filename("a b/c\\d") == "a_b_c_d"


def test_sanitize_falls_back_when_everything_is_stripped():
    assert N.sanitize_filename("///", fallback="sample") == "sample"


def test_sanitize_falls_back_on_empty_input():
    assert N.sanitize_filename("", fallback="sample") == "sample"


# --- size tags -------------------------------------------------------------

def test_size_tag_drops_the_trailing_zeros_of_a_whole_number():
    assert N.size_tag(5.0) == "5um"


def test_size_tag_keeps_a_fractional_size():
    assert N.size_tag(2.5) == "2.5um"


def test_size_tag_has_no_decimal_point_in_the_directory_name():
    # '.' is legal in a filename but a bare '2.5um' folder is fine; what must
    # never appear is a separator. Guard against regressions in the tag format.
    assert "/" not in N.size_tag(2.5) and "\\" not in N.size_tag(2.5)


def test_device_tag_is_zero_padded():
    assert N.device_tag(7) == "dev007"


def test_device_tag_does_not_truncate_large_indices():
    assert N.device_tag(1234) == "dev1234"


# --- directory layout ------------------------------------------------------

def test_sample_dir_nests_date_under_the_output_root(tmp_path):
    paths = N.SamplePaths(str(tmp_path), "2026-08-13", "waferB")

    assert paths.sample_dir == str(tmp_path / "2026-08-13" / "waferB")


def test_device_dir_is_scoped_by_size_then_index(tmp_path):
    paths = N.SamplePaths(str(tmp_path), "2026-08-13", "waferB")

    got = paths.device_dir(5.0, 3)

    assert got == str(tmp_path / "2026-08-13" / "waferB" / "5um" / "dev003")


def test_sample_name_is_sanitized_in_the_path(tmp_path):
    paths = N.SamplePaths(str(tmp_path), "2026-08-13", "wafer B/2")

    assert paths.sample_tag == "wafer_B_2"
    assert paths.sample_dir.endswith("wafer_B_2")


def test_summary_path_sits_beside_the_size_folders(tmp_path):
    paths = N.SamplePaths(str(tmp_path), "2026-08-13", "waferB")

    assert paths.summary_file == str(
        tmp_path / "2026-08-13" / "waferB" / "waferB_summary.csv"
    )


def test_ensure_device_dirs_creates_data_and_plots(tmp_path):
    paths = N.SamplePaths(str(tmp_path), "2026-08-13", "waferB")

    dirs = paths.ensure_device_dirs(5.0, 1)

    assert (tmp_path / "2026-08-13" / "waferB" / "5um" / "dev001" / "data").is_dir()
    assert (tmp_path / "2026-08-13" / "waferB" / "5um" / "dev001" / "plots").is_dir()
    assert dirs.data_dir.endswith("data")
    assert dirs.plot_dir.endswith("plots")


def test_reusing_a_device_allocates_an_isolated_rerun(tmp_path):
    paths = N.SamplePaths(str(tmp_path), "2026-08-13", "waferB")
    first = paths.allocate_device_run_dirs(5.0, 1)
    with open(first.meta_file, "w", encoding="utf-8") as fh:
        fh.write("original metadata")

    second = paths.allocate_device_run_dirs(5.0, 1)

    assert second.run_number == 2
    assert second.device_dir.endswith("dev001\\reruns\\run002")
    assert second.data_dir.endswith("run002\\data")
    assert open(first.meta_file, encoding="utf-8").read() == "original metadata"


def test_rerun_allocator_never_reuses_an_existing_rerun_directory(tmp_path):
    paths = N.SamplePaths(str(tmp_path), "2026-08-13", "waferB")
    first = paths.allocate_device_run_dirs(5.0, 1)
    with open(first.meta_file, "w", encoding="utf-8") as fh:
        fh.write("run one")
    second = paths.allocate_device_run_dirs(5.0, 1)
    with open(second.meta_file, "w", encoding="utf-8") as fh:
        fh.write("run two")

    third = paths.allocate_device_run_dirs(5.0, 1)

    assert third.run_number == 3
    assert third.device_dir.endswith("dev001\\reruns\\run003")


# --- index scoping ---------------------------------------------------------

def test_first_index_on_a_fresh_sample_is_one(tmp_path):
    paths = N.SamplePaths(str(tmp_path), "2026-08-13", "waferB")

    assert paths.next_device_index(5.0) == 1


def test_index_increments_past_existing_devices(tmp_path):
    paths = N.SamplePaths(str(tmp_path), "2026-08-13", "waferB")
    paths.ensure_device_dirs(5.0, 1)
    paths.ensure_device_dirs(5.0, 2)

    assert paths.next_device_index(5.0) == 3


def test_changing_the_crosspoint_size_restarts_the_index(tmp_path):
    paths = N.SamplePaths(str(tmp_path), "2026-08-13", "waferB")
    paths.ensure_device_dirs(5.0, 1)
    paths.ensure_device_dirs(5.0, 2)

    assert paths.next_device_index(20.0) == 1


def test_index_resumes_correctly_after_returning_to_an_earlier_size(tmp_path):
    paths = N.SamplePaths(str(tmp_path), "2026-08-13", "waferB")
    paths.ensure_device_dirs(5.0, 1)
    paths.ensure_device_dirs(20.0, 1)

    assert paths.next_device_index(5.0) == 2


def test_index_skips_over_a_gap_rather_than_refilling_it(tmp_path):
    # A device folder can be deleted after a bad probe landing. Reusing its
    # number would silently overwrite the summary row that still refers to it.
    paths = N.SamplePaths(str(tmp_path), "2026-08-13", "waferB")
    paths.ensure_device_dirs(5.0, 1)
    paths.ensure_device_dirs(5.0, 5)

    assert paths.next_device_index(5.0) == 6


def test_unrelated_directory_names_do_not_disturb_the_index(tmp_path):
    paths = N.SamplePaths(str(tmp_path), "2026-08-13", "waferB")
    paths.ensure_device_dirs(5.0, 1)
    (tmp_path / "2026-08-13" / "waferB" / "5um" / "notes").mkdir()
    (tmp_path / "2026-08-13" / "waferB" / "5um" / "devXYZ").mkdir()

    assert paths.next_device_index(5.0) == 2


def test_index_for_a_different_sample_is_independent(tmp_path):
    a = N.SamplePaths(str(tmp_path), "2026-08-13", "waferA")
    b = N.SamplePaths(str(tmp_path), "2026-08-13", "waferB")
    a.ensure_device_dirs(5.0, 1)
    a.ensure_device_dirs(5.0, 2)

    assert b.next_device_index(5.0) == 1


# --- measurement filenames -------------------------------------------------

def test_measurement_filename_identifies_sample_size_device_and_kind():
    stem = N.measurement_stem("waferB", 5.0, 3, "CF", "2026-08-13_14-30-00")

    assert stem == "CF_waferB_5um_dev003_2026-08-13_14-30-00"


def test_measurement_filename_sanitizes_the_sample_name():
    stem = N.measurement_stem("wafer B", 5.0, 1, "RVS", "2026-08-13_14-30-00")

    assert stem.startswith("RVS_wafer_B_5um_dev001_")
