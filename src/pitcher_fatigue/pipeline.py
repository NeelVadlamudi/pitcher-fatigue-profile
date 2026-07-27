"""End-to-end orchestration for the app, notebooks, and exports."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import FASTBALL_TYPES, AnalysisConfig
from .features import (
    aggregate_decay_curve,
    build_features,
    feature_coverage_summary,
    pitch_type_degradation,
)
from .metrics import ArmStaminaIndex, compute_arm_stamina_index
from .model import (
    ModelTrainingError,
    VelocityModelResult,
    compute_feature_importance,
    train_velocity_model,
)
from .quality import (
    DataQualityReport,
    field_coverage,
    mnar_risk,
    validate_and_clean_statcast,
)
from .thresholds import FatigueThreshold, find_fatigue_threshold


@dataclass
class AnalysisBundle:
    provenance: dict[str, object]
    config: AnalysisConfig
    quality_report: DataQualityReport
    cleaned: pd.DataFrame
    features: pd.DataFrame
    coverage: dict[str, float | int | None]
    velocity_curve: pd.DataFrame
    spin_curve: pd.DataFrame
    movement_curve: pd.DataFrame
    threshold: FatigueThreshold
    threshold_curve: pd.DataFrame
    pitch_type_slopes: pd.DataFrame
    arm_stamina_index: ArmStaminaIndex
    pre_pitch_model: VelocityModelResult | None = None
    descriptive_model: VelocityModelResult | None = None
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance_method: str | None = None
    model_error: str | None = None


def analyze_pitcher_frame(
    frame: pd.DataFrame,
    *,
    provenance: dict[str, object] | None = None,
    config: AnalysisConfig | None = None,
    train_models: bool = True,
    model_backend: str = "auto",
) -> AnalysisBundle:
    """Run validation, features, uncertainty, models, and named metrics."""

    config = config or AnalysisConfig()
    config.validate()
    cleaned, quality_report = validate_and_clean_statcast(frame, config)
    if not quality_report.is_usable:
        raise ValueError("The Statcast data failed critical quality checks")

    features = build_features(cleaned, config)
    coverage = feature_coverage_summary(features)
    coverage.update(
        {
            f"{label}_coverage": value
            for label, value in field_coverage(features).items()
        }
    )
    spin_mnar_shift = mnar_risk(
        features,
        min_pitches=config.mnar_min_window_pitches,
    )
    coverage["spin_rate_mnar_shift"] = spin_mnar_shift
    quality_report.metrics["spin_rate_mnar_shift"] = spin_mnar_shift
    threshold, threshold_curve = find_fatigue_threshold(
        features,
        pitch_types=FASTBALL_TYPES,
        config=config,
    )

    velocity_curve = aggregate_decay_curve(
        features,
        pitch_types=FASTBALL_TYPES,
        metric="speed_delta",
        config=config,
    )
    spin_curve = (
        aggregate_decay_curve(
            features,
            pitch_types=FASTBALL_TYPES,
            metric="spin_delta_rpm",
            config=config,
        )
        if "spin_delta_rpm" in features
        else pd.DataFrame()
    )
    movement_curve = (
        aggregate_decay_curve(
            features,
            pitch_types=FASTBALL_TYPES,
            metric="movement_delta_in",
            config=config,
        )
        if "movement_delta_in" in features
        else pd.DataFrame()
    )
    pitch_type_slopes = pitch_type_degradation(
        features,
        metric="speed_delta",
        min_games=config.threshold_min_games,
        config=config,
    )
    arm_stamina_index = compute_arm_stamina_index(
        features,
        threshold,
        coverage=coverage,
        config=config,
    )

    pre_pitch_model: VelocityModelResult | None = None
    descriptive_model: VelocityModelResult | None = None
    importance = pd.DataFrame()
    importance_method: str | None = None
    model_error: str | None = None

    if train_models:
        try:
            pre_pitch_model = train_velocity_model(
                features,
                feature_mode="pre_pitch",
                backend=model_backend,
                split_strategy="chronological",
                cross_validate=True,
                config=config,
            )
            descriptive_model = train_velocity_model(
                features,
                feature_mode="descriptive_quality",
                backend=model_backend,
                split_strategy="chronological",
                cross_validate=False,
                config=config,
            )
            importance, importance_method = compute_feature_importance(
                descriptive_model,
                prefer_shap=True,
                random_state=config.random_state,
            )
        except (ModelTrainingError, ValueError) as exc:
            model_error = str(exc)

    return AnalysisBundle(
        provenance=provenance or {},
        config=config,
        quality_report=quality_report,
        cleaned=cleaned,
        features=features,
        coverage=coverage,
        velocity_curve=velocity_curve,
        spin_curve=spin_curve,
        movement_curve=movement_curve,
        threshold=threshold,
        threshold_curve=threshold_curve,
        pitch_type_slopes=pitch_type_slopes,
        arm_stamina_index=arm_stamina_index,
        pre_pitch_model=pre_pitch_model,
        descriptive_model=descriptive_model,
        feature_importance=importance,
        importance_method=importance_method,
        model_error=model_error,
    )
