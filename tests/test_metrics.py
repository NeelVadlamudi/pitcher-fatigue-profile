from pitcher_fatigue.config import AnalysisConfig
from pitcher_fatigue.features import build_features
from pitcher_fatigue.metrics import compute_arm_stamina_index
from pitcher_fatigue.quality import validate_and_clean_statcast
from pitcher_fatigue.sample_data import make_synthetic_pitcher
from pitcher_fatigue.thresholds import find_fatigue_threshold


def _synthetic_features():
    config = AnalysisConfig(min_starts=8, bootstrap_iterations=100)
    cleaned, _ = validate_and_clean_statcast(
        make_synthetic_pitcher(n_games=10),
        config,
    )
    features = build_features(cleaned, config)
    threshold, _ = find_fatigue_threshold(features, config=config)
    return features, threshold, config


def test_asi_names_all_components_when_score_is_available():
    features, threshold, config = _synthetic_features()
    result = compute_arm_stamina_index(
        features,
        threshold,
        coverage={"velocity_coverage": 1.0},
        config=config,
    )
    assert result.score is not None
    assert result.components == (
        "threshold timing",
        "fastball velocity retention",
        "between-game consistency",
    )


def test_asi_is_unavailable_when_velocity_coverage_is_too_low():
    features, threshold, config = _synthetic_features()
    result = compute_arm_stamina_index(
        features,
        threshold,
        coverage={"velocity_coverage": 0.75},
        config=config,
    )
    assert result.score is None
    assert result.status == "insufficient_velocity_coverage"
    assert result.components == ()
