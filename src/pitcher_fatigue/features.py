"""Feature engineering for within-game pitch-quality degradation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import (
    FASTBALL_TYPES,
    PITCH_TYPE_NAMES,
    VALID_PITCH_TYPES,
    AnalysisConfig,
)


BASELINE_METRICS = {
    "release_speed": "baseline_speed",
    "release_spin_rate": "baseline_spin_rate",
    "release_extension": "baseline_extension",
    "pfx_x": "baseline_pfx_x",
    "pfx_z": "baseline_pfx_z",
    "movement_magnitude_in": "baseline_movement_magnitude_in",
}


def compute_game_pitch_count(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign a one-indexed cumulative pitch count within each game.

    ``at_bat_number`` is the plate-appearance number of the game and
    ``pitch_number`` is the pitch number within that plate appearance. Counting
    happens before pitch-type or tracking-quality filters so an untracked pitch
    does not make every later count one pitch too low.
    """

    required = {"game_pk", "at_bat_number", "pitch_number"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot order pitches; missing: {sorted(missing)}")

    sort_columns = [
        column
        for column in ["game_date", "game_pk", "at_bat_number", "pitch_number"]
        if column in frame.columns
    ]
    data = frame.sort_values(sort_columns, kind="mergesort").copy()
    data["game_pitch_count"] = data.groupby("game_pk", sort=False).cumcount() + 1
    data["appearance_pitch_count"] = data.groupby("game_pk", sort=False)[
        "game_pitch_count"
    ].transform("max")
    return data


def filter_eligible_starts(
    frame: pd.DataFrame,
    config: AnalysisConfig | None = None,
) -> pd.DataFrame:
    """Keep traditional starter appearances with enough observed pitches."""

    config = config or AnalysisConfig()
    data = (
        frame
        if "game_pitch_count" in frame
        else compute_game_pitch_count(frame)
    ).copy()

    game_profile = data.groupby("game_pk", observed=True).agg(
        first_inning=("inning", "min"),
        pitch_count=("game_pitch_count", "max"),
    )
    eligible_games = game_profile.index[
        game_profile["first_inning"].eq(1)
        & game_profile["pitch_count"].ge(config.min_pitches_per_start)
    ]
    data = data[data["game_pk"].isin(eligible_games)].copy()
    data = data[data["inning"].le(config.max_inning)].copy()
    return data


def _pitch_family(pitch_type: str) -> str:
    if pitch_type in FASTBALL_TYPES:
        return "Fastball"
    if pitch_type in {"SL", "ST", "SV", "CU", "KC", "CS"}:
        return "Breaking"
    if pitch_type in {"CH", "FS", "FO"}:
        return "Offspeed"
    return "Other"


def _add_pitch_buckets(data: pd.DataFrame, bucket_size: int) -> pd.DataFrame:
    result = data.copy()
    start = ((result["game_pitch_count"] - 1) // bucket_size) * bucket_size + 1
    result["pitch_count_bucket_start"] = start.astype(int)
    result["pitch_count_bucket_end"] = (start + bucket_size - 1).astype(int)
    result["pitch_count_bucket_mid"] = (
        result["pitch_count_bucket_start"] + result["pitch_count_bucket_end"]
    ) / 2
    result["pitch_count_bucket"] = (
        result["pitch_count_bucket_start"].astype(str)
        + "–"
        + result["pitch_count_bucket_end"].astype(str)
    )
    return result


def build_features(
    frame: pd.DataFrame,
    config: AnalysisConfig | None = None,
) -> pd.DataFrame:
    """Build pitch-type-specific within-game baselines and degradation features.

    A pitch is compared only with the same pitch type in the same game's early
    window. Rows without enough early-game observations of that pitch type are
    retained but marked ``baseline_available=False`` and excluded from
    baseline-dependent analysis.
    """

    config = config or AnalysisConfig()
    config.validate()
    data = compute_game_pitch_count(frame)
    data = filter_eligible_starts(data, config)
    data = data[data["pitch_type"].isin(VALID_PITCH_TYPES)].copy()

    data["movement_magnitude_in"] = (
        np.hypot(data.get("pfx_x"), data.get("pfx_z")) * 12.0
        if {"pfx_x", "pfx_z"}.issubset(data.columns)
        else np.nan
    )

    early = data[data["game_pitch_count"].le(config.baseline_window)].copy()
    available_metrics = [
        metric for metric in BASELINE_METRICS if metric in early.columns
    ]
    aggregation: dict[str, tuple[str, str]] = {
        BASELINE_METRICS[metric]: (metric, "mean") for metric in available_metrics
    }
    aggregation["baseline_pitch_count"] = ("release_speed", "count")

    baselines = (
        early.groupby(["game_pk", "pitch_type"], observed=True)
        .agg(**aggregation)
        .reset_index()
    )
    eligible_baseline = baselines["baseline_pitch_count"].ge(
        config.baseline_min_pitches_per_type
    )
    for baseline_column in BASELINE_METRICS.values():
        if baseline_column in baselines:
            baselines.loc[~eligible_baseline, baseline_column] = np.nan
    baselines["baseline_available"] = eligible_baseline

    data = data.merge(
        baselines,
        on=["game_pk", "pitch_type"],
        how="left",
        validate="many_to_one",
    )
    data["baseline_available"] = data["baseline_available"].eq(True)
    data["baseline_pitch_count"] = data["baseline_pitch_count"].fillna(0).astype(int)

    if "baseline_speed" in data:
        data["speed_delta"] = data["release_speed"] - data["baseline_speed"]
    if "baseline_spin_rate" in data:
        data["spin_delta_rpm"] = (
            data["release_spin_rate"] - data["baseline_spin_rate"]
        )
    if "baseline_extension" in data:
        data["extension_delta_ft"] = (
            data["release_extension"] - data["baseline_extension"]
        )
    if "baseline_pfx_x" in data:
        data["pfx_x_delta_in"] = (data["pfx_x"] - data["baseline_pfx_x"]) * 12.0
    if "baseline_pfx_z" in data:
        data["pfx_z_delta_in"] = (data["pfx_z"] - data["baseline_pfx_z"]) * 12.0
    if "baseline_movement_magnitude_in" in data:
        data["movement_delta_in"] = (
            data["movement_magnitude_in"] - data["baseline_movement_magnitude_in"]
        )

    data["pitch_type_name"] = data["pitch_type"].map(PITCH_TYPE_NAMES).fillna(
        data["pitch_type"]
    )
    data["pitch_family"] = data["pitch_type"].map(_pitch_family)
    data["is_fastball"] = data["pitch_type"].isin(FASTBALL_TYPES).astype(int)
    data = _add_pitch_buckets(data, config.pitch_bucket_size)
    return data.reset_index(drop=True)


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, clean.size, size=(iterations, clean.size))
    means = clean[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def aggregate_decay_curve(
    features: pd.DataFrame,
    *,
    pitch_types: Iterable[str] | None = None,
    metric: str = "speed_delta",
    config: AnalysisConfig | None = None,
) -> pd.DataFrame:
    """Aggregate degradation by pitch-count bucket with equal game weights.

    Confidence intervals resample game-level bucket means, not individual
    pitches, so games with more pitches of a type do not dominate uncertainty.
    """

    config = config or AnalysisConfig()
    if metric not in features:
        raise ValueError(f"Metric {metric!r} is unavailable")
    selected_types = set(pitch_types or FASTBALL_TYPES)
    data = features[
        features["pitch_type"].isin(selected_types)
        & features["baseline_available"]
        & features[metric].notna()
        & features["game_pitch_count"].le(config.max_pitch_count)
    ].copy()
    if data.empty:
        return pd.DataFrame(
            columns=[
                "pitch_count_bucket",
                "pitch_count_bucket_start",
                "pitch_count_bucket_end",
                "pitch_count_bucket_mid",
                "mean_delta",
                "median_delta",
                "ci_lower",
                "ci_upper",
                "games",
                "pitches",
                "coverage",
            ]
        )

    total_games = int(features["game_pk"].nunique())
    per_game = (
        data.groupby(
            [
                "game_pk",
                "pitch_count_bucket",
                "pitch_count_bucket_start",
                "pitch_count_bucket_end",
                "pitch_count_bucket_mid",
            ],
            observed=True,
        )[metric]
        .agg(game_mean="mean", pitches="size")
        .reset_index()
    )

    rows: list[dict[str, float | int | str]] = []
    for bucket_start, bucket in per_game.groupby(
        "pitch_count_bucket_start", observed=True, sort=True
    ):
        values = bucket["game_mean"].to_numpy(dtype=float)
        ci_lower, ci_upper = _bootstrap_mean_interval(
            values,
            iterations=config.bootstrap_iterations,
            confidence=config.bootstrap_confidence,
            seed=config.random_state + int(bucket_start),
        )
        rows.append(
            {
                "pitch_count_bucket": str(bucket["pitch_count_bucket"].iloc[0]),
                "pitch_count_bucket_start": int(bucket_start),
                "pitch_count_bucket_end": int(
                    bucket["pitch_count_bucket_end"].iloc[0]
                ),
                "pitch_count_bucket_mid": float(
                    bucket["pitch_count_bucket_mid"].iloc[0]
                ),
                "mean_delta": float(np.mean(values)),
                "median_delta": float(np.median(values)),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "games": int(bucket["game_pk"].nunique()),
                "pitches": int(bucket["pitches"].sum()),
                "coverage": (
                    float(bucket["game_pk"].nunique() / total_games)
                    if total_games
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def pitch_type_degradation(
    features: pd.DataFrame,
    *,
    metric: str = "speed_delta",
    min_pitches_per_game_type: int = 6,
    min_pitch_span: int = 30,
    min_games: int = 5,
    config: AnalysisConfig | None = None,
) -> pd.DataFrame:
    """Estimate pitch-type slopes from per-game slopes.

    Each game contributes at most one slope per pitch type, limiting the
    influence of high-usage games. Results are expressed per ten pitches.
    """

    config = config or AnalysisConfig()
    data = features[
        features["baseline_available"]
        & features[metric].notna()
        & features["game_pitch_count"].le(config.max_pitch_count)
    ].copy()

    game_slopes: list[dict[str, float | int | str]] = []
    for (game_pk, pitch_type), group in data.groupby(
        ["game_pk", "pitch_type"], observed=True
    ):
        if len(group) < min_pitches_per_game_type:
            continue
        span = int(group["game_pitch_count"].max() - group["game_pitch_count"].min())
        if span < min_pitch_span:
            continue
        slope = float(
            np.polyfit(
                group["game_pitch_count"].to_numpy(dtype=float),
                group[metric].to_numpy(dtype=float),
                1,
            )[0]
        )
        late = group.loc[group["game_pitch_count"].ge(80), metric]
        game_slopes.append(
            {
                "game_pk": int(game_pk),
                "pitch_type": str(pitch_type),
                "slope_per_pitch": slope,
                "late_delta": float(late.mean()) if not late.empty else np.nan,
                "pitches": int(len(group)),
            }
        )

    slopes = pd.DataFrame(game_slopes)
    if slopes.empty:
        return pd.DataFrame(
            columns=[
                "pitch_type",
                "pitch_type_name",
                "slope_mph_per_10",
                "ci_lower",
                "ci_upper",
                "games",
                "pitches",
                "late_delta",
                "eligible",
            ]
        )

    rows: list[dict[str, float | int | str | bool]] = []
    for index, (pitch_type, group) in enumerate(
        slopes.groupby("pitch_type", observed=True, sort=True)
    ):
        values = group["slope_per_pitch"].to_numpy(dtype=float) * 10.0
        ci_lower, ci_upper = _bootstrap_mean_interval(
            values,
            iterations=config.bootstrap_iterations,
            confidence=config.bootstrap_confidence,
            seed=config.random_state + 1_000 + index,
        )
        rows.append(
            {
                "pitch_type": str(pitch_type),
                "pitch_type_name": PITCH_TYPE_NAMES.get(
                    str(pitch_type), str(pitch_type)
                ),
                "slope_mph_per_10": float(np.mean(values)),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "games": int(group["game_pk"].nunique()),
                "pitches": int(group["pitches"].sum()),
                "late_delta": float(group["late_delta"].mean()),
                "eligible": bool(group["game_pk"].nunique() >= min_games),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["eligible", "slope_mph_per_10"],
        ascending=[False, True],
    )


def feature_coverage_summary(features: pd.DataFrame) -> dict[str, float | int]:
    """Return compact coverage metrics for the methodology panel."""

    rows = len(features)
    baseline_rows = int(features["baseline_available"].sum())
    return {
        "rows": int(rows),
        "games": int(features["game_pk"].nunique()),
        "pitch_types": int(features["pitch_type"].nunique()),
        "baseline_rows": baseline_rows,
        "baseline_coverage": baseline_rows / rows if rows else 0.0,
        "late_games_80_plus": int(
            features.loc[features["game_pitch_count"].ge(80), "game_pk"].nunique()
        ),
    }
