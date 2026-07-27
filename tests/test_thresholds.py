import numpy as np
import pandas as pd

from pitcher_fatigue.config import AnalysisConfig
from pitcher_fatigue.thresholds import find_fatigue_threshold


def threshold_features(n_games: int, declining: bool) -> pd.DataFrame:
    rows = []
    for game_pk in range(n_games):
        for pitch_count in range(1, 101):
            if declining:
                delta = -0.05 * max(0, pitch_count - 60)
            else:
                delta = 0.05 * np.sin(pitch_count)
            bucket_start = ((pitch_count - 1) // 10) * 10 + 1
            rows.append(
                {
                    "game_pk": game_pk,
                    "game_pitch_count": pitch_count,
                    "pitch_type": "FF",
                    "baseline_available": True,
                    "speed_delta": delta,
                    "pitch_count_bucket": f"{bucket_start}–{bucket_start + 9}",
                    "pitch_count_bucket_start": bucket_start,
                    "pitch_count_bucket_end": bucket_start + 9,
                    "pitch_count_bucket_mid": bucket_start + 4.5,
                }
            )
    return pd.DataFrame(rows)


def threshold_config(min_starts: int = 10) -> AnalysisConfig:
    return AnalysisConfig(
        min_starts=min_starts,
        threshold_min_games=5,
        threshold_min_coverage=0.30,
        threshold_consecutive_buckets=2,
        bootstrap_iterations=100,
    )


def test_threshold_requires_sustained_supported_decline():
    threshold, curve = find_fatigue_threshold(
        threshold_features(12, declining=True),
        config=threshold_config(),
    )
    assert threshold.status == "established"
    assert threshold.threshold_pitch == 81
    assert curve["crossing_candidate"].sum() >= 2


def test_flat_profile_reports_not_reached():
    threshold, _ = find_fatigue_threshold(
        threshold_features(12, declining=False),
        config=threshold_config(),
    )
    assert threshold.status == "not_reached"
    assert threshold.threshold_pitch is None


def test_small_sample_reports_insufficient_data():
    threshold, _ = find_fatigue_threshold(
        threshold_features(4, declining=True),
        config=threshold_config(),
    )
    assert threshold.status == "insufficient_data"

