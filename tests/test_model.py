import numpy as np

from pitcher_fatigue.config import AnalysisConfig
from pitcher_fatigue.features import build_features
from pitcher_fatigue.model import split_by_game, train_velocity_model
from pitcher_fatigue.quality import validate_and_clean_statcast
from pitcher_fatigue.sample_data import make_synthetic_pitcher


def model_features():
    config = AnalysisConfig(
        min_starts=8,
        bootstrap_iterations=100,
    )
    raw = make_synthetic_pitcher(n_games=10)
    cleaned, _ = validate_and_clean_statcast(raw, config)
    return build_features(cleaned, config), config


def test_chronological_split_has_no_game_overlap():
    features, config = model_features()
    modeling = features[
        features["baseline_available"] & features["speed_delta"].notna()
    ].reset_index(drop=True)
    split = split_by_game(
        modeling,
        strategy="chronological",
        test_size=config.model_test_size,
    )
    assert not split.has_overlap
    assert set(split.train_games).isdisjoint(split.test_games)

    latest_train = modeling[
        modeling["game_pk"].isin(split.train_games)
    ]["game_date"].max()
    earliest_test = modeling[
        modeling["game_pk"].isin(split.test_games)
    ]["game_date"].min()
    assert latest_train < earliest_test


def test_sklearn_model_trains_only_on_complete_games():
    features, config = model_features()
    result = train_velocity_model(
        features,
        backend="sklearn",
        split_strategy="chronological",
        cross_validate=False,
        config=config,
    )
    assert not result.split.has_overlap
    assert np.isfinite(result.metrics["mae"])
    assert result.metrics["test_games"] == len(result.split.test_games)
    assert set(result.test_game_ids.unique()) == set(result.split.test_games)

