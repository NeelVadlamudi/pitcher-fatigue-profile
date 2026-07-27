import pandas as pd
import pytest

from pitcher_fatigue.quality import (
    DataQualityError,
    field_coverage,
    mnar_risk,
    validate_and_clean_statcast,
)


def test_missing_required_columns_block_analysis():
    with pytest.raises(DataQualityError) as error:
        validate_and_clean_statcast(pd.DataFrame({"game_pk": [1]}))
    assert error.value.report.status == "blocked"


def test_duplicate_pitch_key_is_removed_and_reported(
    mixed_pitch_frame, small_config
):
    duplicated = pd.concat(
        [mixed_pitch_frame, mixed_pitch_frame.iloc[[0]]],
        ignore_index=True,
    )
    cleaned, report = validate_and_clean_statcast(duplicated, small_config)
    assert len(cleaned) == len(mixed_pitch_frame)
    assert report.metrics["exact_duplicates_removed"] == 1
    assert not cleaned.duplicated(
        ["game_pk", "at_bat_number", "pitch_number"]
    ).any()


def test_invalid_velocity_becomes_missing(mixed_pitch_frame, small_config):
    mixed_pitch_frame.loc[0, "release_speed"] = 999
    cleaned, report = validate_and_clean_statcast(
        mixed_pitch_frame, small_config
    )
    assert pd.isna(cleaned.loc[0, "release_speed"])
    assert any(issue.check == "release_speed_range" for issue in report.issues)


def test_field_coverage_reports_optional_fields_explicitly(mixed_pitch_frame):
    coverage = field_coverage(mixed_pitch_frame)
    assert coverage["velocity"] == 1.0
    assert coverage["spin_rate"] == 1.0
    assert coverage["horizontal_break"] == 1.0

    without_spin = field_coverage(
        mixed_pitch_frame.drop(columns=["release_spin_rate"])
    )
    assert without_spin["spin_rate"] == 0.0


def test_mnar_diagnostic_uses_total_rows_and_minimum_window_guard():
    frame = pd.DataFrame(
        {
            "game_pitch_count": list(range(1, 31)) * 2
            + list(range(80, 110)) * 2,
            "release_spin_rate": [2200.0] * 60
            + [2200.0] * 45
            + [None] * 15,
        }
    )
    assert mnar_risk(frame, min_pitches=50) == pytest.approx(0.25)
    assert mnar_risk(frame.iloc[:90], min_pitches=50) is None
