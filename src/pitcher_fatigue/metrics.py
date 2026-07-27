"""Named experimental metrics derived from validated fatigue features."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import FASTBALL_TYPES, AnalysisConfig
from .thresholds import FatigueThreshold


@dataclass(frozen=True)
class ArmStaminaIndex:
    score: float | None
    threshold_component: float | None
    late_retention_component: float | None
    consistency_component: float | None
    late_game_mean_delta: float | None
    late_game_sd: float | None
    late_games: int
    components: tuple[str, ...]
    status: str
    caveat: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_arm_stamina_index(
    features: pd.DataFrame,
    threshold: FatigueThreshold,
    *,
    coverage: dict[str, float | int | None] | None = None,
    config: AnalysisConfig | None = None,
) -> ArmStaminaIndex:
    """Compute the explicitly experimental 0-100 Arm Stamina Index.

    The component scales are transparent but not population-calibrated:

    - fatigue-threshold timing: 50 points
    - mean fastball retention at pitch 80+: 30 points
    - between-game late-velocity consistency: 20 points
    """

    config = config or AnalysisConfig()
    games = int(features["game_pk"].nunique())
    component_names = (
        "threshold timing",
        "fastball velocity retention",
        "between-game consistency",
    )
    caveat = (
        f"Experimental. Components: {', '.join(component_names)}. "
        "The formula is not population-calibrated and is not an injury metric."
    )
    if games < config.min_starts or threshold.status == "insufficient_data":
        return ArmStaminaIndex(
            score=None,
            threshold_component=None,
            late_retention_component=None,
            consistency_component=None,
            late_game_mean_delta=None,
            late_game_sd=None,
            late_games=0,
            components=(),
            status="insufficient_data",
            caveat=caveat,
        )

    velocity_coverage = float(
        (coverage or {}).get(
            "velocity_coverage",
            features["release_speed"].notna().mean()
            if "release_speed" in features and len(features)
            else 0.0,
        )
        or 0.0
    )
    if velocity_coverage < config.asi_min_velocity_coverage:
        return ArmStaminaIndex(
            score=None,
            threshold_component=None,
            late_retention_component=None,
            consistency_component=None,
            late_game_mean_delta=None,
            late_game_sd=None,
            late_games=0,
            components=(),
            status="insufficient_velocity_coverage",
            caveat=(
                f"{caveat} Overall velocity coverage "
                f"({velocity_coverage:.1%}) is below the "
                f"{config.asi_min_velocity_coverage:.0%} scoring requirement."
            ),
        )

    fastballs = features[
        features["pitch_type"].isin(FASTBALL_TYPES)
        & features["baseline_available"]
        & features["speed_delta"].notna()
        & features["game_pitch_count"].ge(80)
    ]
    per_game = fastballs.groupby("game_pk", observed=True)["speed_delta"].mean()
    late_games = int(len(per_game))
    if late_games < config.threshold_min_games:
        return ArmStaminaIndex(
            score=None,
            threshold_component=None,
            late_retention_component=None,
            consistency_component=None,
            late_game_mean_delta=None,
            late_game_sd=None,
            late_games=late_games,
            components=(),
            status="insufficient_late_game_coverage",
            caveat=caveat,
        )

    if threshold.status == "not_reached":
        threshold_pitch = config.max_pitch_count
    else:
        threshold_pitch = int(threshold.threshold_pitch or 0)
    threshold_component = float(
        np.clip(threshold_pitch / config.max_pitch_count, 0, 1) * 50.0
    )

    late_mean = float(per_game.mean())
    late_sd = float(per_game.std(ddof=1)) if late_games > 1 else 0.0
    late_retention_component = float(np.clip((late_mean + 2.0) / 2.0, 0, 1) * 30)
    consistency_component = float(np.clip(1.0 - late_sd / 2.0, 0, 1) * 20)
    score = round(
        threshold_component + late_retention_component + consistency_component,
        1,
    )

    return ArmStaminaIndex(
        score=score,
        threshold_component=round(threshold_component, 1),
        late_retention_component=round(late_retention_component, 1),
        consistency_component=round(consistency_component, 1),
        late_game_mean_delta=round(late_mean, 3),
        late_game_sd=round(late_sd, 3),
        late_games=late_games,
        components=component_names,
        status="experimental",
        caveat=caveat,
    )
