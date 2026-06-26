from health_misinfo.datasets.labels import coaid_harmonized_label, fakehealth_harmonized_label, native_label_dictionary


def test_coaid_label_mapping_is_explicit():
    assert coaid_harmonized_label("NewsFakeCOVID-19.csv") == ("fake", "unreliable")
    assert coaid_harmonized_label("ClaimRealCOVID-19.csv") == ("real", "reliable")


def test_fakehealth_rating_mapping_is_explicit():
    assert fakehealth_harmonized_label(2) == ("rating_2", "unreliable")
    assert fakehealth_harmonized_label(3) == ("rating_3", "reliable")


def test_native_dictionary_has_no_empty_harmonized_labels():
    labels = native_label_dictionary()
    assert labels["harmonized_label"].isin(["reliable", "unreliable"]).all()
    assert {"coaid", "fakehealth"}.issubset(set(labels["dataset"]))

