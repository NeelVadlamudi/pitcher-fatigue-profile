# GitHub Publishing Checklist

Use these settings when this project is published as a public portfolio
repository.

## Repository profile

**Repository name**

`pitcher-fatigue-profile`

**Description**

> Auditable Statcast analysis of within-game pitcher velocity and pitch-quality
> decline, with game-isolated validation and a Streamlit dashboard.

**Topics**

`baseball-analytics`, `statcast`, `sports-analytics`, `streamlit`,
`machine-learning`, `xgboost`, `data-visualization`, `python`

**Social preview**

Use `outputs/figures/synthetic_demo_summary.png`. It shows the actual analytical
output and does not rely on team marks or an invented product logo.

## Before the first push

1. Choose the repository visibility.
2. Choose and add a software license. No license is included because licensing
   changes reuse rights and should be an explicit owner decision.
3. Confirm that `data/raw/`, `data/processed/`, fitted models, local
   environments, and credentials are absent from the commit.
4. Run `python -m pytest -q`.
5. Execute both notebooks and confirm that their outputs match the committed
   validation report.
6. Confirm that the GitHub Actions workflow passes on Python 3.11.
7. Add the deployed Hugging Face Space URL to the repository **About** section
   after the Space is live.

## Recommended first release

Use a short release title such as `v0.1.0 — validated portfolio release`.

The release notes should name the supported scope:

- same-game, same-pitch-type baselines;
- equal-game weighting and game-cluster uncertainty;
- chronological held-out-game model evaluation;
- synthetic offline demonstration;
- Logan Webb 2024 validation example; and
- Docker deployment for Hugging Face Spaces.

Avoid claims that the project detects physiological fatigue, predicts injury,
or prescribes bullpen decisions.
