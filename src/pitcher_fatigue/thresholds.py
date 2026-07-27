"""Conservative fatigue-threshold estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

from .config import FASTBALL_TYPES, AnalysisConfig
from .features import aggregate_decay_curve


ThresholdStatus = Literal["established", "not_reached", "insufficient_data"]


@dataclass(frozen=True)
class FatigueThreshold:
    status: ThresholdStatus
    threshold_pitch: int | None
    threshold_range: str | None
    drop_threshold_mph: float
    games_analyzed: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def find_fatigue_threshold(
    features: pd.DataFrame,
    *,
    pitch_types: set[str] | frozenset[str] | None = None,
    config: AnalysisConfig | None = None,
) -> tuple[FatigueThreshold, pd.DataFrame]:
    """Find the first sustained, supported velocity-decline bucket.

    A bucket must:

    - average at least ``fatigue_drop_mph`` below its within-game baseline,
    - have a bootstrap upper confidence bound below zero,
    - include the configured minimum number and share of starts, and
    - remain below the threshold for the configured number of consecutive
      pitch-count buckets.

    The result is a pitch-count range, not a claim that one exact pitch causes
    fatigue.
    """

    config = config or AnalysisConfig()
    selected_types = set(pitch_types or FASTBALL_TYPES)
    curve = aggregate_decay_curve(
        features,
        pitch_types=selected_types,
        metric="speed_delta",
        config=config,
    ).copy()
    games = int(features["game_pk"].nunique())

    if games < config.min_starts:
        result = FatigueThreshold(
            status="insufficient_data",
            threshold_pitch=None,
            threshold_range=None,
            drop_threshold_mph=config.fatigue_drop_mph,
            games_analyzed=games,
            reason=(
                f"{games} eligible starts are available; at least "
                f"{config.min_starts} are required."
            ),
        )
        return result, curve

    if curve.empty:
        result = FatigueThreshold(
            status="insufficient_data",
            threshold_pitch=None,
            threshold_range=None,
            drop_threshold_mph=config.fatigue_drop_mph,
            games_analyzed=games,
            reason="No pitch-count buckets have a usable within-game baseline.",
        )
        return result, curve

    curve["eligible"] = (
        curve["games"].ge(config.threshold_min_games)
        & curve["coverage"].ge(config.threshold_min_coverage)
    )
    curve["below_drop"] = curve["mean_delta"].le(-config.fatigue_drop_mph)
    curve["statistically_below_fresh"] = curve["ci_upper"].lt(0)
    curve["crossing_candidate"] = (
        curve["eligible"]
        & curve["below_drop"]
        & curve["statistically_below_fresh"]
    )

    required = config.threshold_consecutive_buckets
    candidate_rows = curve.index[curve["crossing_candidate"]].tolist()
    threshold_index: int | None = None
    for index in candidate_rows:
        positions = list(range(index, index + required))
        if positions[-1] >= len(curve):
            continue
        window = curve.loc[positions]
        starts = window["pitch_count_bucket_start"].tolist()
        expected = [
            starts[0] + offset * config.pitch_bucket_size
            for offset in range(required)
        ]
        if window["crossing_candidate"].all() and starts == expected:
            threshold_index = index
            break

    if threshold_index is not None:
        row = curve.loc[threshold_index]
        result = FatigueThreshold(
            status="established",
            threshold_pitch=int(row["pitch_count_bucket_start"]),
            threshold_range=str(row["pitch_count_bucket"]),
            drop_threshold_mph=config.fatigue_drop_mph,
            games_analyzed=games,
            reason=(
                "First sustained bucket meeting the velocity-drop, uncertainty, "
                "sample-size, and coverage criteria."
            ),
        )
        return result, curve

    eligible_late = curve[
        curve["eligible"]
        & curve["pitch_count_bucket_start"].ge(71)
    ]
    if eligible_late.empty:
        result = FatigueThreshold(
            status="insufficient_data",
            threshold_pitch=None,
            threshold_range=None,
            drop_threshold_mph=config.fatigue_drop_mph,
            games_analyzed=games,
            reason=(
                "Too few starts reach late-game buckets with the selected pitch "
                "types to estimate a threshold without severe survivor bias."
            ),
        )
    else:
        result = FatigueThreshold(
            status="not_reached",
            threshold_pitch=None,
            threshold_range=None,
            drop_threshold_mph=config.fatigue_drop_mph,
            games_analyzed=games,
            reason=(
                "No sustained late-game bucket met both the configured velocity "
                "drop and statistical-support criteria."
            ),
        )
    return result, curve

