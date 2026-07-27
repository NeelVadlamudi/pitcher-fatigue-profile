from __future__ import annotations

import pandas as pd
import pytest

from pitcher_fatigue.config import AnalysisConfig


@pytest.fixture
def small_config() -> AnalysisConfig:
    return AnalysisConfig(
        baseline_window=5,
        baseline_min_pitches_per_type=2,
        min_pitches_per_start=5,
        min_starts=1,
        threshold_min_games=1,
        threshold_min_coverage=0.25,
        threshold_consecutive_buckets=1,
        bootstrap_iterations=100,
    )


@pytest.fixture
def mixed_pitch_frame() -> pd.DataFrame:
    rows = []
    pitches = [
        ("FF", 95.0),
        ("SL", 85.0),
        ("FF", 96.0),
        ("SL", 86.0),
        ("CH", 88.0),
        ("FF", 94.0),
        ("SL", 84.0),
    ]
    for game_pk, date_value in [(1, "2024-04-01"), (2, "2024-04-07")]:
        for index, (pitch_type, speed) in enumerate(pitches, start=1):
            rows.append(
                {
                    "game_date": date_value,
                    "game_pk": game_pk,
                    "inning": 1,
                    "at_bat_number": (index - 1) // 3 + 1,
                    "pitch_number": (index - 1) % 3 + 1,
                    "pitch_type": pitch_type,
                    "release_speed": speed + (0.2 if game_pk == 2 else 0),
                    "release_spin_rate": 2_400 if pitch_type == "FF" else 2_600,
                    "release_extension": 6.2,
                    "pfx_x": -0.5 if pitch_type == "FF" else 0.4,
                    "pfx_z": 1.2 if pitch_type == "FF" else 0.3,
                    "game_type": "R",
                    "n_thruorder_pitcher": 1,
                    "pitcher_days_since_prev_game": 5,
                    "stand": "R",
                }
            )
    return pd.DataFrame(rows)
