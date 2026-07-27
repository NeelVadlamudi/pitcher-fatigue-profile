"""Game-isolated modeling and non-causal feature attribution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import AnalysisConfig


FeatureMode = Literal["pre_pitch", "descriptive_quality"]
SplitStrategy = Literal["chronological", "random_group"]

PRE_PITCH_NUMERIC = [
    "game_pitch_count",
    "inning",
    "n_thruorder_pitcher",
    "pitcher_days_since_prev_game",
]
PRE_PITCH_CATEGORICAL = ["pitch_type", "stand"]
DESCRIPTIVE_NUMERIC = [
    "spin_delta_rpm",
    "extension_delta_ft",
    "pfx_x_delta_in",
    "pfx_z_delta_in",
    "movement_delta_in",
]


class ModelTrainingError(RuntimeError):
    """Raised when the modeling sample cannot support a valid split."""


@dataclass(frozen=True)
class GameSplit:
    strategy: SplitStrategy
    train_games: tuple[int, ...]
    test_games: tuple[int, ...]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]

    @property
    def has_overlap(self) -> bool:
        return bool(set(self.train_games) & set(self.test_games))


@dataclass
class VelocityModelResult:
    pipeline: Pipeline
    backend: str
    feature_mode: FeatureMode
    numeric_features: list[str]
    categorical_features: list[str]
    split: GameSplit
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_game_ids: pd.Series
    predictions: np.ndarray
    metrics: dict[str, float | int | str]

    def prediction_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "game_pk": self.test_game_ids.to_numpy(),
                "actual_speed_delta": self.y_test.to_numpy(),
                "predicted_speed_delta": self.predictions,
                "absolute_error": np.abs(
                    self.y_test.to_numpy() - self.predictions
                ),
            },
            index=self.y_test.index,
        )


def split_by_game(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.20,
    strategy: SplitStrategy = "chronological",
    random_state: int = 42,
) -> GameSplit:
    """Split complete games, never individual pitch rows."""

    if "game_pk" not in frame:
        raise ValueError("game_pk is required for a leakage-resistant split")
    games = frame[["game_pk"]].drop_duplicates().copy()
    if len(games) < 5:
        raise ModelTrainingError("At least five games are required for modeling")

    if strategy == "chronological":
        if "game_date" not in frame:
            raise ValueError("game_date is required for a chronological split")
        game_dates = (
            frame.groupby("game_pk", observed=True)["game_date"]
            .min()
            .pipe(pd.to_datetime)
            .sort_values(kind="mergesort")
        )
        ordered_games = game_dates.index.to_numpy()
        n_test = max(1, int(np.ceil(len(ordered_games) * test_size)))
        n_test = min(n_test, len(ordered_games) - 2)
        train_games = ordered_games[:-n_test]
        test_games = ordered_games[-n_test:]
    elif strategy == "random_group":
        train_games, test_games = train_test_split(
            games["game_pk"].to_numpy(),
            test_size=test_size,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    train_mask = frame["game_pk"].isin(train_games)
    test_mask = frame["game_pk"].isin(test_games)
    split = GameSplit(
        strategy=strategy,
        train_games=tuple(int(value) for value in train_games),
        test_games=tuple(int(value) for value in test_games),
        train_indices=tuple(int(value) for value in frame.index[train_mask]),
        test_indices=tuple(int(value) for value in frame.index[test_mask]),
    )
    if split.has_overlap:
        raise AssertionError("A game appeared in both train and test sets")
    return split


def _select_feature_columns(
    data: pd.DataFrame,
    mode: FeatureMode,
) -> tuple[list[str], list[str]]:
    numeric_candidates = PRE_PITCH_NUMERIC.copy()
    if mode == "descriptive_quality":
        numeric_candidates += DESCRIPTIVE_NUMERIC
    numeric = [
        column
        for column in numeric_candidates
        if column in data and data[column].notna().any()
    ]
    categorical = [
        column
        for column in PRE_PITCH_CATEGORICAL
        if column in data and data[column].notna().any()
    ]
    if not numeric:
        raise ModelTrainingError("No usable numeric model features are available")
    return numeric, categorical


def _make_estimator(
    backend: str,
    config: AnalysisConfig,
):
    if backend not in {"auto", "xgboost", "sklearn"}:
        raise ValueError("backend must be auto, xgboost, or sklearn")

    if backend in {"auto", "xgboost"}:
        try:
            from xgboost import XGBRegressor

            return (
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=350,
                    max_depth=3,
                    learning_rate=0.035,
                    min_child_weight=8,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.05,
                    reg_lambda=1.0,
                    random_state=config.random_state,
                    n_jobs=2,
                    verbosity=0,
                ),
                "xgboost",
            )
        except Exception as exc:
            if backend == "xgboost":
                raise ModelTrainingError(
                    "XGBoost was requested but its native runtime could not load"
                ) from exc

    return (
        HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=config.random_state,
        ),
        "sklearn_hist_gradient_boosting",
    )


def _cluster_mae_interval(
    actual: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    scores = np.empty(iterations, dtype=float)
    grouped_indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    for iteration in range(iterations):
        sampled_groups = rng.choice(
            unique_groups, size=unique_groups.size, replace=True
        )
        sampled_indices = np.concatenate(
            [grouped_indices[group] for group in sampled_groups]
        )
        scores[iteration] = mean_absolute_error(
            actual[sampled_indices], predicted[sampled_indices]
        )
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(scores, alpha)),
        float(np.quantile(scores, 1.0 - alpha)),
    )


def train_velocity_model(
    features: pd.DataFrame,
    *,
    feature_mode: FeatureMode = "pre_pitch",
    backend: str = "auto",
    split_strategy: SplitStrategy = "chronological",
    cross_validate: bool = True,
    config: AnalysisConfig | None = None,
) -> VelocityModelResult:
    """Train and evaluate a velocity-delta model on wholly held-out games."""

    config = config or AnalysisConfig()
    data = features[
        features["baseline_available"] & features["speed_delta"].notna()
    ].copy()
    data["game_date"] = pd.to_datetime(data["game_date"])
    data = data.reset_index(drop=True)
    if data["game_pk"].nunique() < 5:
        raise ModelTrainingError("At least five games with baselines are required")

    numeric, categorical = _select_feature_columns(data, feature_mode)
    feature_columns = numeric + categorical
    split = split_by_game(
        data,
        test_size=config.model_test_size,
        strategy=split_strategy,
        random_state=config.random_state,
    )

    train_rows = list(split.train_indices)
    test_rows = list(split.test_indices)
    X = data[feature_columns]
    y = data["speed_delta"].astype(float)
    X_train, X_test = X.loc[train_rows], X.loc[test_rows]
    y_train, y_test = y.loc[train_rows], y.loc[test_rows]

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median"))]
    )
    transformers: list[tuple[str, Pipeline, list[str]]] = [
        ("numeric", numeric_pipeline, numeric)
    ]
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    estimator, resolved_backend = _make_estimator(backend, config)
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("regressor", estimator)]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions)) if len(y_test) > 1 else np.nan
    naive_predictions = np.full(len(y_test), float(y_train.mean()))
    naive_mae = float(mean_absolute_error(y_test, naive_predictions))
    test_groups = data.loc[test_rows, "game_pk"].astype(int)
    mae_ci_lower, mae_ci_upper = _cluster_mae_interval(
        y_test.to_numpy(),
        predictions,
        test_groups.to_numpy(),
        iterations=config.bootstrap_iterations,
        confidence=config.bootstrap_confidence,
        seed=config.random_state + 2_000,
    )

    metrics: dict[str, float | int | str] = {
        "mae": mae,
        "mae_ci_lower": mae_ci_lower,
        "mae_ci_upper": mae_ci_upper,
        "rmse": rmse,
        "r2": r2,
        "naive_mae": naive_mae,
        "mae_improvement_pct": (
            float((naive_mae - mae) / naive_mae * 100)
            if naive_mae > 0
            else 0.0
        ),
        "train_games": len(split.train_games),
        "test_games": len(split.test_games),
        "train_pitches": len(X_train),
        "test_pitches": len(X_test),
        "split_strategy": split.strategy,
    }

    if cross_validate:
        game_count = int(data["game_pk"].nunique())
        folds = min(5, game_count)
        if folds >= 3:
            cv = GroupKFold(n_splits=folds)
            scores = -cross_val_score(
                clone(pipeline),
                X,
                y,
                groups=data["game_pk"],
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=None,
            )
            metrics["group_cv_folds"] = folds
            metrics["group_cv_mae_mean"] = float(scores.mean())
            metrics["group_cv_mae_sd"] = float(scores.std(ddof=1))

    return VelocityModelResult(
        pipeline=pipeline,
        backend=resolved_backend,
        feature_mode=feature_mode,
        numeric_features=numeric,
        categorical_features=categorical,
        split=split,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_game_ids=test_groups,
        predictions=np.asarray(predictions),
        metrics=metrics,
    )


def _raw_feature_from_transformed(
    transformed_name: str,
    numeric: list[str],
    categorical: list[str],
) -> str:
    suffix = transformed_name.split("__", 1)[-1]
    if suffix in numeric:
        return suffix
    for column in categorical:
        if suffix == column or suffix.startswith(f"{column}_"):
            return column
    return suffix


def compute_feature_importance(
    result: VelocityModelResult,
    *,
    prefer_shap: bool = True,
    max_samples: int = 600,
    random_state: int = 42,
) -> tuple[pd.DataFrame, str]:
    """Compute SHAP importance when supported, otherwise permutation importance.

    Both outputs are associative model explanations, not causal effects.
    """

    sample = result.X_test.sample(
        n=min(max_samples, len(result.X_test)),
        random_state=random_state,
    )

    if prefer_shap and result.backend == "xgboost":
        try:
            import shap

            preprocessor = result.pipeline.named_steps["preprocessor"]
            estimator = result.pipeline.named_steps["regressor"]
            transformed = preprocessor.transform(sample)
            feature_names = list(preprocessor.get_feature_names_out())
            explainer = shap.TreeExplainer(estimator)
            shap_values = np.asarray(explainer.shap_values(transformed))
            mean_abs = np.abs(shap_values).mean(axis=0)
            detail = pd.DataFrame(
                {
                    "transformed_feature": feature_names,
                    "importance": mean_abs,
                }
            )
            detail["feature"] = detail["transformed_feature"].map(
                lambda name: _raw_feature_from_transformed(
                    name,
                    result.numeric_features,
                    result.categorical_features,
                )
            )
            importance = (
                detail.groupby("feature", observed=True)["importance"]
                .sum()
                .reset_index()
                .sort_values("importance", ascending=False)
            )
            total = importance["importance"].sum()
            importance["share"] = (
                importance["importance"] / total if total > 0 else 0.0
            )
            return importance.reset_index(drop=True), "mean_absolute_shap"
        except Exception:
            pass

    repeats = 10 if len(sample) >= 50 else 5
    permutation = permutation_importance(
        result.pipeline,
        sample,
        result.y_test.loc[sample.index],
        scoring="neg_mean_absolute_error",
        n_repeats=repeats,
        random_state=random_state,
    )
    importance = pd.DataFrame(
        {
            "feature": sample.columns,
            "importance": np.maximum(permutation.importances_mean, 0.0),
            "importance_sd": permutation.importances_std,
        }
    ).sort_values("importance", ascending=False)
    total = importance["importance"].sum()
    importance["share"] = (
        importance["importance"] / total if total > 0 else 0.0
    )
    return importance.reset_index(drop=True), "permutation_mae"


def save_model(
    result: VelocityModelResult,
    path: str | Path,
) -> tuple[Path, Path]:
    """Persist the fitted pipeline and audit metadata."""

    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, model_path)
    metadata_path = model_path.with_suffix(".json")
    metadata = {
        "backend": result.backend,
        "feature_mode": result.feature_mode,
        "numeric_features": result.numeric_features,
        "categorical_features": result.categorical_features,
        "split": asdict(result.split),
        "metrics": result.metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(Path(path))

