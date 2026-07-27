"""Build the two reproducible project notebooks with nbformat."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def write_notebook(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    nbf.write(notebook, NOTEBOOK_DIR / name)


def build_feature_notebook() -> None:
    cells = [
        markdown(
            """
# Data Quality and Feature Engineering

## tl;dr

- The deterministic demonstration source contains 1,661 pitch rows across 16 eligible starts.
- Every retained pitch has a same-game, same-pitch-type early baseline in this controlled sample.
- The conservative fastball procedure establishes its first sustained threshold at pitches 91–100.
- These values validate the software path only; they are synthetic and are not an MLB finding.
"""
        ),
        markdown(
            """
## Context & Methods

This companion notebook audits the raw-to-feature path used by the Streamlit
application. It reconstructs cumulative game pitch count before filtering,
checks pitch-level grain, and calculates each early-game baseline separately by
`game_pk` and `pitch_type`.

### Key Assumptions

- A starter appearance begins in inning 1 and contains at least 40 observed pitches.
- The fresh baseline is the mean of at least two same-type pitches among the first 25 game pitches.
- Ten-pitch bucket estimates give every game equal weight.
- Confidence intervals resample games, not individual pitches.
"""
        ),
        markdown("## Data\n\n### 1. Load the deterministic sample"),
        code(
            """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from pitcher_fatigue.config import AnalysisConfig
from pitcher_fatigue.pipeline import analyze_pitcher_frame

sample_path = ROOT / "data" / "sample" / "sample_pitcher.csv"
raw = pd.read_csv(sample_path, low_memory=False)
raw.shape
"""
        ),
        markdown("### 2. Validate source grain and physical ranges"),
        code(
            """
config = AnalysisConfig()
bundle = analyze_pitcher_frame(
    raw,
    provenance={"source": "deterministic_synthetic_demo"},
    config=config,
    train_models=False,
)

quality_summary = pd.Series(bundle.quality_report.to_dict()).drop("issues")
quality_summary
"""
        ),
        code(
            """
bundle.quality_report.to_frame()
"""
        ),
        markdown("## Results\n\n### 3. Verify pitch-type-specific baselines"),
        code(
            """
baseline_audit = (
    bundle.features[
        [
            "game_pk",
            "pitch_type",
            "baseline_speed",
            "baseline_pitch_count",
            "baseline_available",
        ]
    ]
    .drop_duplicates()
    .sort_values(["game_pk", "pitch_type"])
)
baseline_audit.head(12)
"""
        ),
        code(
            """
assert not bundle.features.duplicated(
    ["game_pk", "at_bat_number", "pitch_number"]
).any()
assert bundle.features.groupby("game_pk")["game_pitch_count"].min().eq(1).all()
assert bundle.features.loc[
    bundle.features["baseline_available"], "baseline_speed"
].notna().all()
bundle.coverage
"""
        ),
        markdown("### 4. Plot the game-weighted fastball fatigue curve"),
        code(
            """
curve = bundle.velocity_curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(
    curve["pitch_count_bucket_mid"],
    curve["ci_lower"],
    curve["ci_upper"],
    color="#2F6690",
    alpha=0.16,
    label="95% game-cluster interval",
)
ax.plot(
    curve["pitch_count_bucket_mid"],
    curve["mean_delta"],
    color="#2F6690",
    marker="o",
    label="Equal-game mean",
)
ax.axhline(0, color="#667085", linestyle="--", linewidth=1)
ax.axhline(-1, color="#D4A017", linestyle=":", linewidth=1.3)
ax.set(
    title="Fastball velocity delta by game pitch count",
    xlabel="Game pitch count",
    ylabel="Velocity delta from same-type baseline (mph)",
)
ax.legend(frameon=False)
ax.grid(axis="y", color="#D9DEE7", linewidth=0.6)
plt.show()
"""
        ),
        code(
            """
pd.DataFrame(
    {
        "status": [bundle.threshold.status],
        "threshold_range": [bundle.threshold.threshold_range],
        "reason": [bundle.threshold.reason],
    }
)
"""
        ),
        markdown(
            """
## Takeaways

The deterministic sample passes the intended feature-engineering checks. Pitch
count is reconstructed before tracking filters, baselines do not mix pitch
types, and threshold uncertainty respects game-level dependence. A real
pitcher-season must pass the same coverage and data-quality gates before its
results can be shared.
"""
        ),
    ]
    write_notebook("01_data_quality_and_features.ipynb", cells)


def build_model_notebook() -> None:
    cells = [
        markdown(
            """
# Game-Isolated Model Validation

## tl;dr

- The primary demonstration model holds out the latest complete games, never individual pitch rows.
- Using the deterministic sample and scikit-learn backend, holdout MAE is approximately 0.41 mph.
- The model improves on a train-mean baseline by approximately 21%, with holdout R² near 0.42.
- Feature importance is associative and must not be described as causal.
"""
        ),
        markdown(
            """
## Context & Methods

This notebook validates the model boundary that matters most: every pitch from a
test game is unseen during training. The primary model uses only pre-pitch
context. A separate descriptive-quality model supplies feature importance and
may include same-pitch tracking measurements.

### Key Assumptions

- The latest 20% of games form the chronological holdout.
- Mean absolute error is the primary accuracy measure because its unit is mph.
- A train-mean predictor is the explicit naive benchmark.
- Same-game pitch rows are dependent, so MAE uncertainty resamples games.
"""
        ),
        markdown("## Data\n\n### 1. Build the validated feature table"),
        code(
            """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from pitcher_fatigue.config import AnalysisConfig
from pitcher_fatigue.features import build_features
from pitcher_fatigue.model import (
    compute_feature_importance,
    train_velocity_model,
)
from pitcher_fatigue.quality import validate_and_clean_statcast

raw = pd.read_csv(ROOT / "data" / "sample" / "sample_pitcher.csv", low_memory=False)
config = AnalysisConfig()
cleaned, quality = validate_and_clean_statcast(raw, config)
features = build_features(cleaned, config)
features.shape
"""
        ),
        markdown("## Results\n\n### 2. Train on complete earlier games"),
        code(
            """
pre_pitch_model = train_velocity_model(
    features,
    feature_mode="pre_pitch",
    backend="sklearn",
    split_strategy="chronological",
    cross_validate=True,
    config=config,
)
pd.Series(pre_pitch_model.metrics)
"""
        ),
        markdown("### 3. Audit the game boundary"),
        code(
            """
split_audit = pd.DataFrame(
    {
        "partition": (
            ["train"] * len(pre_pitch_model.split.train_games)
            + ["test"] * len(pre_pitch_model.split.test_games)
        ),
        "game_pk": (
            list(pre_pitch_model.split.train_games)
            + list(pre_pitch_model.split.test_games)
        ),
    }
)
assert set(pre_pitch_model.split.train_games).isdisjoint(
    pre_pitch_model.split.test_games
)
split_audit
"""
        ),
        markdown("### 4. Inspect held-out predictions"),
        code(
            """
prediction = pre_pitch_model.prediction_frame()
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(
    prediction["actual_speed_delta"],
    prediction["predicted_speed_delta"],
    color="#2F6690",
    alpha=0.45,
    s=18,
)
lower = min(
    prediction["actual_speed_delta"].min(),
    prediction["predicted_speed_delta"].min(),
)
upper = max(
    prediction["actual_speed_delta"].max(),
    prediction["predicted_speed_delta"].max(),
)
ax.plot([lower, upper], [lower, upper], color="#667085", linestyle="--")
ax.set(
    title="Held-out-game predictions",
    xlabel="Observed velocity delta (mph)",
    ylabel="Predicted velocity delta (mph)",
)
ax.grid(color="#D9DEE7", linewidth=0.6)
plt.show()
"""
        ),
        markdown("### 5. Explain the descriptive-quality model"),
        code(
            """
descriptive_model = train_velocity_model(
    features,
    feature_mode="descriptive_quality",
    backend="sklearn",
    split_strategy="chronological",
    cross_validate=False,
    config=config,
)
importance, method = compute_feature_importance(
    descriptive_model,
    prefer_shap=True,
    random_state=config.random_state,
)
importance
"""
        ),
        code(
            """
plot_data = importance.head(10).sort_values("importance")
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(
    plot_data["feature"].str.replace("_", " ").str.title(),
    plot_data["importance"],
    color="#16324F",
)
ax.set(
    title=f"Associative model feature importance ({method.replace('_', ' ')})",
    xlabel="Contribution to model accuracy",
)
ax.grid(axis="x", color="#D9DEE7", linewidth=0.6)
plt.show()
"""
        ),
        markdown(
            """
## Takeaways

The train/test boundary is clean at the game level and the test period is later
than the training period. The synthetic benchmark is useful for regression
testing, not as a claim about MLB performance. Model importance describes what
the fitted model used; it does not show that a feature caused fatigue.
"""
        ),
    ]
    write_notebook("02_model_validation.ipynb", cells)


if __name__ == "__main__":
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    build_feature_notebook()
    build_model_notebook()
    print(NOTEBOOK_DIR / "01_data_quality_and_features.ipynb")
    print(NOTEBOOK_DIR / "02_model_validation.ipynb")

