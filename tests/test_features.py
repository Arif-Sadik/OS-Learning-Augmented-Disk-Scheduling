from src.features import FEATURE_NAMES, extract_features


def test_extract_features_for_empty_workload():
    features = extract_features([], initial_head=50)

    assert set(features) == set(FEATURE_NAMES)
    assert all(value == 0.0 for value in features.values())


def test_extract_features_for_single_request():
    features = extract_features([80], initial_head=50)

    assert features["mean_position"] == 80.0
    assert features["std_position"] == 0.0
    assert features["request_range"] == 0.0
    assert features["head_to_mean_distance"] == 30.0
    assert features["mean_abs_step"] == 0.0


def test_extract_features_for_directional_sequence():
    features = extract_features([10, 20, 30, 25], initial_head=50)

    assert features["mean_position"] == 21.25
    assert features["request_range"] == 20.0
    assert features["mean_abs_step"] == (10 + 10 + 5) / 3
    assert features["median_abs_step"] == 10.0
    assert features["increasing_ratio"] == 2 / 3
    assert features["decreasing_ratio"] == 1 / 3
    assert features["direction_change_ratio"] == 0.5
    assert features["head_to_mean_distance"] == 28.75
