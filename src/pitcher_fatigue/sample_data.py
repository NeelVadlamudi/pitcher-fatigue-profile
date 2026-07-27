"""Generate a deterministic synthetic pitcher for demos and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_SPEED = {"FF": 96.0, "SI": 94.5, "SL": 86.0, "CH": 88.0}
BASE_SPIN = {"FF": 2_450.0, "SI": 2_250.0, "SL": 2_600.0, "CH": 1_850.0}
BASE_PFX_X = {"FF": -0.45, "SI": -1.05, "SL": 0.45, "CH": -0.80}
BASE_PFX_Z = {"FF": 1.30, "SI": 0.75, "SL": 0.25, "CH": 0.55}
FATIGUE_SLOPE = {"FF": -0.042, "SI": -0.032, "SL": -0.022, "CH": -0.014}


def make_synthetic_pitcher(
    *,
    n_games: int = 16,
    season: int = 2024,
    seed: int = 42,
) -> pd.DataFrame:
    """Return clearly labeled synthetic Statcast-shaped data.

    The four-seam fastball begins a sustained decline near pitch 65. The data
    are useful for exercising thresholds and charts, but never represent an
    actual player.
    """

    if n_games < 4:
        raise ValueError("n_games must be at least 4")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    dates = pd.date_range(f"{season}-04-02", periods=n_games, freq="6D")
    opening_sequence = ["FF", "SL", "SI", "CH", "FF", "SL", "SI", "CH"]
    choices = np.array(["FF", "SI", "SL", "CH"])
    probabilities = np.array([0.46, 0.17, 0.23, 0.14])

    for game_index, game_date in enumerate(dates):
        total_pitches = int(rng.integers(98, 111))
        game_form = float(rng.normal(0.0, 0.35))
        fatigue_onset = int(rng.integers(61, 69))
        game_pk = 900_000 + game_index

        pitch_types = opening_sequence + list(
            rng.choice(
                choices,
                size=total_pitches - len(opening_sequence),
                p=probabilities,
            )
        )

        for game_pitch_count, pitch_type in enumerate(pitch_types, start=1):
            fatigue_load = max(0, game_pitch_count - fatigue_onset)
            speed_delta = FATIGUE_SLOPE[pitch_type] * fatigue_load
            spin_delta = -2.6 * fatigue_load if pitch_type in {"FF", "SI"} else -1.4 * fatigue_load
            extension_delta = -0.0025 * fatigue_load
            movement_scale = max(0.86, 1.0 - 0.0014 * fatigue_load)

            at_bat_number = (game_pitch_count - 1) // 4 + 1
            pitch_number = (game_pitch_count - 1) % 4 + 1
            inning = min(9, (game_pitch_count - 1) // 15 + 1)

            release_spin_rate: float | None = float(
                BASE_SPIN[pitch_type] + spin_delta + rng.normal(0, 65)
            )
            if rng.random() < 0.008:
                release_spin_rate = None

            rows.append(
                {
                    "pitch_type": pitch_type,
                    "game_date": game_date.date().isoformat(),
                    "release_speed": float(
                        BASE_SPEED[pitch_type]
                        + game_form
                        + speed_delta
                        + rng.normal(0, 0.48)
                    ),
                    "release_spin_rate": release_spin_rate,
                    "release_extension": float(
                        6.35 + extension_delta + rng.normal(0, 0.08)
                    ),
                    "pfx_x": float(
                        BASE_PFX_X[pitch_type] * movement_scale + rng.normal(0, 0.06)
                    ),
                    "pfx_z": float(
                        BASE_PFX_Z[pitch_type] * movement_scale + rng.normal(0, 0.06)
                    ),
                    "game_pk": game_pk,
                    "inning": inning,
                    "inning_topbot": "Top",
                    "at_bat_number": at_bat_number,
                    "pitch_number": pitch_number,
                    "game_type": "R",
                    "n_thruorder_pitcher": min(4, (at_bat_number - 1) // 9 + 1),
                    "pitcher_days_since_prev_game": 5 if game_index else np.nan,
                    "p_throws": "R",
                    "stand": "L" if rng.random() < 0.48 else "R",
                    "player_name": "Demonstration, Synthetic",
                }
            )

    return pd.DataFrame(rows)


def write_synthetic_sample(path: str | Path = "data/sample/sample_pitcher.csv") -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_pitcher().to_csv(output, index=False)
    return output

