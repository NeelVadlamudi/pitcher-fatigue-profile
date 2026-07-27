import pytest

from pitcher_fatigue.features import (
    build_features,
    compute_game_pitch_count,
)
from pitcher_fatigue.quality import validate_and_clean_statcast


def test_game_pitch_count_continues_across_plate_appearances(mixed_pitch_frame):
    result = compute_game_pitch_count(mixed_pitch_frame)
    for _, game in result.groupby("game_pk"):
        assert game["game_pitch_count"].tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_baseline_is_separate_for_each_game_and_pitch_type(
    mixed_pitch_frame, small_config
):
    cleaned, _ = validate_and_clean_statcast(mixed_pitch_frame, small_config)
    features = build_features(cleaned, small_config)

    game_one = features[features["game_pk"].eq(1)]
    fastball = game_one[game_one["pitch_type"].eq("FF")]
    slider = game_one[game_one["pitch_type"].eq("SL")]

    assert fastball["baseline_speed"].unique().tolist() == [95.5]
    assert slider["baseline_speed"].unique().tolist() == [85.5]
    assert fastball.iloc[-1]["speed_delta"] == pytest.approx(-1.5)
    assert slider.iloc[-1]["speed_delta"] == pytest.approx(-1.5)


def test_pitch_without_two_early_observations_has_no_baseline(
    mixed_pitch_frame, small_config
):
    extra = mixed_pitch_frame.iloc[[0]].copy()
    extra["pitch_type"] = "CU"
    extra["release_speed"] = 80.0
    extra["at_bat_number"] = 3
    extra["pitch_number"] = 2
    combined = mixed_pitch_frame._append(extra, ignore_index=True)

    cleaned, _ = validate_and_clean_statcast(combined, small_config)
    features = build_features(cleaned, small_config)
    curveball = features[features["pitch_type"].eq("CU")]
    assert not curveball["baseline_available"].any()
    assert curveball["speed_delta"].isna().all()
