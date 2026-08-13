import pytest

from Breakdown.core import mfia_nodes as N

SAMPLE_PATH = "/dev3519/imps/0/sample"


# --- path normalisation ----------------------------------------------------

def test_paths_are_lowercased_and_given_a_leading_slash():
    assert N.normalize_node_path("DEV3519/IMPS/0/Sample") == "/dev3519/imps/0/sample"


def test_redundant_separators_are_collapsed():
    assert N.normalize_node_path("//dev3519//imps/0/") == "/dev3519/imps/0"


def test_an_empty_path_normalises_to_the_root():
    assert N.normalize_node_path("") == "/"


def test_the_device_prefix_is_stripped_to_compare_across_serial_numbers():
    assert N.node_path_suffix("/dev3519/imps/0/sample") == "/imps/0/sample"


def test_a_path_without_a_device_prefix_is_left_alone():
    assert N.node_path_suffix("/imps/0/sample") == "/imps/0/sample"


# --- reducing a field to its latest value ----------------------------------

def test_a_list_of_readings_reduces_to_the_most_recent():
    assert N.last_value([1.0, 2.0, 3.0]) == 3.0


def test_a_scalar_reading_is_returned_unchanged():
    assert N.last_value(2.5) == 2.5


def test_an_empty_list_of_readings_yields_nothing():
    assert N.last_value([]) is None


def test_a_numpy_array_reduces_to_its_last_element():
    np = pytest.importorskip("numpy")

    assert N.last_value(np.array([1.0, 2.0, 7.0])) == pytest.approx(7.0)


def test_every_field_of_a_sample_is_reduced():
    payload = {"param0": [1.0, 2.0], "param1": [3.0, 4.0], "frequency": 1000.0}

    assert N.sample_payload_to_last_values(payload) == {
        "param0": 2.0, "param1": 4.0, "frequency": 1000.0,
    }


def test_a_structured_numpy_sample_is_reduced_field_by_field():
    np = pytest.importorskip("numpy")
    payload = np.array([(1.0, 3.0), (2.0, 4.0)],
                       dtype=[("param0", "f8"), ("param1", "f8")])

    values = N.sample_payload_to_last_values(payload)

    assert values["param0"] == pytest.approx(2.0)
    assert values["param1"] == pytest.approx(4.0)


# --- flattening what poll() returns ----------------------------------------

def test_slash_separated_field_keys_are_grouped_into_one_sample():
    data = {
        "/dev3519/imps/0/sample/param0": [1.0],
        "/dev3519/imps/0/sample/param1": [2.0],
    }

    flattened = N.flatten_poll_sample_nodes(data)

    assert set(flattened) == {SAMPLE_PATH}
    assert set(flattened[SAMPLE_PATH]) == {"param0", "param1"}


def test_dot_separated_field_keys_are_grouped_the_same_way():
    data = {
        "/dev3519/imps/0/sample.param0": [1.0],
        "/dev3519/imps/0/sample.param1": [2.0],
    }

    flattened = N.flatten_poll_sample_nodes(data)

    assert set(flattened[SAMPLE_PATH]) == {"param0", "param1"}


def test_a_whole_sample_delivered_under_one_key_is_kept():
    data = {"/dev3519/imps/0/sample": {"param0": [1.0], "param1": [2.0]}}

    flattened = N.flatten_poll_sample_nodes(data)

    assert SAMPLE_PATH in flattened


def test_a_nested_dictionary_tree_is_walked():
    data = {"dev3519": {"imps": {"0": {"sample": {"param0": [1.0]}}}}}

    flattened = N.flatten_poll_sample_nodes(data)

    assert SAMPLE_PATH in flattened


def test_an_empty_sample_is_not_reported_as_data():
    data = {"/dev3519/imps/0/sample": {}}

    assert N.flatten_poll_sample_nodes(data) == {}


def test_a_non_dictionary_payload_flattens_to_nothing():
    assert N.flatten_poll_sample_nodes([1, 2, 3]) == {}


# --- extracting the sample we asked for ------------------------------------

def test_the_requested_sample_is_returned_with_its_latest_values():
    data = {"/dev3519/imps/0/sample/param0": [1.0, 5.0]}

    sample, _ = N.extract_sample_payload(data, SAMPLE_PATH)

    assert sample["param0"] == pytest.approx(5.0)


def test_the_request_matches_regardless_of_letter_case():
    data = {"/DEV3519/IMPS/0/SAMPLE/param0": [5.0]}

    sample, _ = N.extract_sample_payload(data, SAMPLE_PATH)

    assert sample["param0"] == pytest.approx(5.0)


def test_a_sample_from_a_differently_named_device_still_matches_by_suffix():
    # Discovery and the data server do not always agree on the device id.
    data = {"/dev9999/imps/0/sample/param0": [5.0]}

    sample, _ = N.extract_sample_payload(data, SAMPLE_PATH)

    assert sample["param0"] == pytest.approx(5.0)


def test_a_sample_for_a_different_module_is_not_mistaken_for_ours():
    data = {"/dev3519/imps/1/sample/param0": [5.0]}

    sample, _ = N.extract_sample_payload(data, SAMPLE_PATH)

    assert sample is None


def test_nothing_to_extract_reports_no_sample():
    sample, seen = N.extract_sample_payload({}, SAMPLE_PATH)

    assert sample is None and seen == []


def test_the_node_paths_that_were_seen_are_reported_for_diagnostics():
    # When the requested node yields nothing, knowing what *did* arrive is the
    # difference between a useful error and a bare timeout.
    data = {"/dev3519/demods/0/sample/x": [1.0]}

    sample, seen = N.extract_sample_payload(data, SAMPLE_PATH)

    assert sample is None
    assert "/dev3519/demods/0/sample" in seen
