---
title: Pitcher Fatigue Profile
emoji: ⚾
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Pitcher Fatigue Profile

Historical Statcast analysis of how a starting pitcher's velocity and pitch
quality change within a game.

Built by **Neel Vadlamudi** as an auditable baseball analytics case study.

![Pitcher Fatigue Profile summary](outputs/figures/synthetic_demo_summary.png)

> This is a postgame decision-support tool. It does not predict injury, measure
> physiological fatigue directly, or tell a coach when to remove a pitcher.

## The question

A late-game velocity drop can look obvious while still being statistically
fragile. High-pitch-count outings are a selected group, different pitch types
operate at different velocity bands, and a pitch-level train/test split can put
the same game on both sides of model evaluation.

I built this project to answer a narrower question:

**At what pitch-count range does a starter show a sustained, coverage-qualified
decline relative to his own early-game baseline?**

The dashboard shows the evidence behind the answer: game coverage, uncertainty,
pitch-type slopes, data quality, and held-out model performance.

## Case study: Logan Webb, 2024

The full pipeline was run on Logan Webb's 2024 regular-season Statcast data.

| Result | Observed value |
|---|---:|
| Eligible starts | 33 |
| Supported 1 mph threshold | Not established |
| Fastball delta, pitches 71–80 | −0.902 mph |
| Fastball delta, pitches 81–90 | −0.882 mph |
| Chronological holdout MAE | 0.752 mph |
| Chronological holdout R² | 0.020 |
| Experimental ASI | 83.0 / 100 |

The useful finding is the absence of a clean breaking point. Webb's fastball
velocity was roughly 0.9 mph below his same-type early baseline late in games,
but no pitch-count window met every threshold rule. The low holdout R² also
shows that the available pre-pitch context explains little of the variation in
an individual pitch.

See the [validation record](outputs/validation/logan_webb_2024_validation.json),
[summary sheet](outputs/validation/logan_webb_2024_summary.png), and
[full validation notes](docs/validation_report.md).

## What the app includes

- An equal-game-weighted fastball decay curve with clustered uncertainty
- A sustained 1 mph threshold with explicit coverage requirements
- Pitch-type velocity slopes with game-level bootstrap intervals
- Spin, movement, and velocity retention views
- Chronological held-out-game model evaluation
- Associative feature importance with a clear non-causal label
- Field-level coverage checks and late-game missingness warnings
- Web PNG, print PNG, and feature-level CSV exports

The default view uses a deterministic synthetic pitcher, so the complete app
can be reviewed without a network request.

## Method choices that matter

| Common failure | Implementation here |
|---|---|
| Mixing fastballs and breaking balls in one baseline | Baselines are computed for each `game_pk × pitch_type` |
| Letting one game appear in both training and testing | Complete games are isolated; the latest games form the main holdout |
| Allowing long outings to dominate the mean | Pitches are averaged within each game and bucket before games are averaged |
| Hiding survivor bias in late innings | Every bucket carries contributing-game and start-coverage counts |
| Calling one noisy crossing a threshold | The decline must persist, meet coverage rules, and have a bootstrap upper bound below zero |
| Treating feature importance as causation | Importance is explicitly labeled as a model association |
| Silently dropping unavailable ASI components | The score is unavailable when required velocity coverage is inadequate |

The exact estimands and equations are documented in
[docs/methodology.md](docs/methodology.md).

## Run locally

Python 3.11 is the supported runtime.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
streamlit run app/streamlit_app.py
```

Choose **Live Statcast** in the sidebar to retrieve a pitcher-season. The
downloaded CSV is cached under `data/raw/` and excluded from version control.

## Reproduce the analysis

Run the automated checks:

```bash
python -m pytest -q
```

Execute both notebooks from top to bottom:

```bash
python scripts/build_notebooks.py
python -m nbconvert --execute --to notebook --inplace \
  notebooks/01_data_quality_and_features.ipynb \
  notebooks/02_model_validation.ipynb \
  --ExecutePreprocessor.timeout=300
```

Build a real-player validation record and summary:

```bash
python scripts/validate_real_pitcher.py Logan Webb 2024
```

The committed release has 23 passing tests, 75% branch-aware package coverage,
two fully executed notebooks, and exception-free dashboard checks for both the
synthetic sample and the cached Logan Webb season.

## Repository map

```text
.
├── app/                    Streamlit interface
├── data/sample/            Deterministic offline demonstration
├── docs/                   Methodology, validation, and chart contracts
├── notebooks/              Executed analysis walkthroughs
├── outputs/                Shareable figures and validation records
├── scripts/                Reproduction and export commands
├── src/pitcher_fatigue/    Analysis package
├── tests/                  Unit and integration checks
├── Dockerfile              Hugging Face Docker Space runtime
└── pyproject.toml          Package and test configuration
```

## Documentation

- [Methodology](docs/methodology.md)
- [Validation report](docs/validation_report.md)
- [Model card](docs/model_card.md)
- [Chart contracts](docs/chart_contracts.md)
- [Pre-deployment review](docs/predeploy_review.md)
- [GitHub publishing checklist](docs/github_publishing.md)

## Modeling boundary

The primary model predicts velocity delta from information available before a
pitch and evaluates on the latest complete games. A separate descriptive model
may use spin, extension, and movement measured on the same pitch to explain
associations. That second model is not a real-time forecast.

The Arm Stamina Index is an experimental summary of threshold timing,
late-fastball retention, and between-game consistency. It is transparent but
not population-calibrated, and it has no injury-risk interpretation.

## Deployment

The repository is ready for a Hugging Face Docker Space. The `Dockerfile`
installs the required Linux runtime and starts Streamlit on port 7860.

1. Create a Space with the **Docker** SDK.
2. Push this repository to the Space.
3. Keep `sdk: docker` and `app_port: 7860` in the README metadata.
4. Treat live Baseball Savant access as optional; the bundled sample provides a
   reliable first view.

## Limitations

- The estimates are historical associations, not direct physiological measures.
- The early baseline uses the first 25 game pitches and is not known pregame.
- Statcast pitch labels and historical records can be revised.
- Warm-up pitches and other workload outside Statcast are not observed.
- Pitchers who reach high counts are selected, even when coverage is reported.
- Weather, health status, between-game workload, opponent, catcher, and intent
  are incomplete or absent.
- A single season can still be too small for some pitch types.

## Data and credits

Pitch-level data come from
[Baseball Savant](https://baseballsavant.mlb.com) through
[pybaseball](https://github.com/jldbc/pybaseball). Field definitions follow the
official [Statcast CSV documentation](https://baseballsavant.mlb.com/csv-docs).

The pitch palette is adapted from Thomas Nestico's
[pitching_summary](https://github.com/tnestico/pitching_summary). Red Sox colors
are used only as interface accents; pitch identity keeps its own established
color mapping.

Workload context includes Bradbury and Forman's 2012 study,
[PMID 22344048](https://pubmed.ncbi.nlm.nih.gov/22344048/), and a
[systematic review of pitcher fatigue](https://pmc.ncbi.nlm.nih.gov/articles/PMC6673423/).

This independent portfolio project is not affiliated with MLB, the Boston Red
Sox, Baseball Savant, or the cited visualization authors.
