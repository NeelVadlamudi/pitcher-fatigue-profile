# Validation Report

## Overall assessment: Share with caveats

The project is methodologically ready to share as a historical, descriptive
decision-support tool. The source-backed Logan Webb 2024 analysis passed the
automated data-quality checks, the architectural invariants are covered by
23 passing tests with 75% branch-aware package coverage, both notebooks execute
top-to-bottom, and the Streamlit app renders its main charts and downloads
without an exception in both the synthetic and cached real-data paths.

The predictive model is weak on the real chronological holdout. That is a
result to disclose, not hide. XGBoost/SHAP was not executed locally because the
available macOS runtime lacks `libomp`; the tested app used its documented
scikit-learn/permutation fallback. The Docker image installs Linux `libgomp1`,
but Docker is unavailable on this machine for a local image build.

## Question and source

- Question: how does one starter's tracked pitch quality change as game pitch
  count rises?
- Real-data validation: Logan Webb, 2024 regular season.
- Source: Baseball Savant via pybaseball 2.2.7.
- Pitcher MLBAM ID: 657277.
- Source coverage through: September 24, 2024.
- Validation executed: July 27, 2026.

## Methodology review

- Cumulative pitch count is assigned before pitch-type and tracking filters.
- Baselines are separate by game and pitch type.
- Bucket estimates weight games equally.
- Confidence intervals resample games as clusters.
- Thresholds require magnitude, uncertainty, sample-size, coverage, and
  persistence criteria.
- Complete games are isolated between train and test.
- The primary split holds out the latest games.
- Same-pitch measurements appear only in the descriptive-quality explanation
  model and are explicitly non-causal.
- Field coverage is reported separately and the late-vs-early spin-missingness
  diagnostic requires at least 50 pitch rows in both windows.
- ASI requires all three named components, 90% velocity coverage, and at least
  five late-game starts; otherwise the score is unavailable.

## Data-quality findings

| Check | Result |
|---|---:|
| Raw rows | 3,346 |
| Non-regular-season rows removed | 137 |
| Clean regular-season rows | 3,209 |
| Duplicate pitch keys | 0 |
| Eligible starts | 33 |
| Baseline-dependent feature rows | 3,095 of 3,200 |
| Baseline coverage | 96.7% |
| Starts reaching pitch 80 | 33 |
| Release-speed null rate | 0.25% |
| Velocity coverage | 99.75% |
| Spin-rate coverage | 99.56% |
| Extension coverage | 99.72% |
| Horizontal/vertical break coverage | 99.75% / 99.75% |
| Late-minus-early spin missingness | +0.50 percentage points |

Overall source status: `ready`.

## Calculation spot-checks

- Row reconciliation: 3,346 − 137 = 3,209 clean regular-season rows.
- Baseline coverage: 3,095 / 3,200 = 96.72%.
- Threshold: no eligible fastball bucket averaged at or below −1 mph.
- Lowest eligible late means were −0.902 mph at pitches 71–80 and −0.882
  mph at pitches 81–90.
- The pitch 101–110 bucket had only nine games and 27.3% coverage, below the
  configured 30% threshold-eligibility rule.
- Experimental ASI used 32 games with pitch-80-plus fastballs; late mean delta
  was −0.773 mph with 0.539 mph between-game standard deviation. Velocity
  coverage exceeded the 90% scoring requirement and all three components were
  included.

## Model validation

| Metric | Result |
|---|---:|
| Training games | 26 |
| Chronological test games | 7 |
| Test pitches | 636 |
| MAE | 0.752 mph |
| Game-cluster MAE interval | 0.709–0.800 mph |
| RMSE | 0.942 mph |
| R² | 0.020 |
| Naive train-mean MAE | 0.759 mph |
| MAE improvement | 0.97% |
| Five-fold grouped-CV MAE | 0.721 ± 0.035 mph |

The model only narrowly improves on the naive baseline. It is acceptable as an
audited portfolio component and diagnostic layer, but not as a strong
individual-pitch forecast.

## Visualization review

- The velocity curve includes a neutral zero line, −1 mph reference, 95%
  intervals, exact units, and sample coverage in the interactive hover state.
- Pitch-type ranking uses horizontal bars and game-bootstrap intervals.
- Pitch-type identity uses an explicit, attributed palette; the aggregate
  fastball curve remains a single blue series.
- Held-out prediction scatter uses an identity line and equal axis scaling.
- Feature importance is labeled associative and names the explanation method.
- The white static summary sheet replaces model diagnostics with coach-facing
  findings, exposes the supported threshold and coverage, and passed visual
  inspection without clipping or label collisions at both 150 and 300 DPI.
- The application exposes three successful downloads: web PNG, print PNG, and
  feature CSV.

## Release checks

- 23 automated tests passed.
- Branch-aware package coverage: 75%.
- Both notebooks executed top-to-bottom in the declared development
  environment.
- Both synthetic and cached real-data Streamlit paths rendered five Plotly
  charts, one detail table, and three downloads without an exception.
- All 115 installed packages passed the dependency compatibility check.
- Python compilation, local README links, and credential/placeholder scanning
  passed.

## Required caveats

- The analysis is conditional on the pitcher remaining in the game.
- A same-game baseline is observed during the first 25 pitches and is not known
  before the game.
- Within-game change does not establish physiological fatigue.
- The 1 mph rule is a transparent decision threshold, not a clinical cutoff.
- ASI is experimental and not population-calibrated.
- The local validation used the scikit-learn/permutation fallback because
  XGBoost's macOS OpenMP runtime was unavailable.

## Incomplete external handoff

No public GitHub repository or Hugging Face Space was created from this local
workspace. Publishing requires the user's destination account/repository and
authorization to create public external resources.
