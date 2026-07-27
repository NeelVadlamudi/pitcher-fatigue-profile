# Methodology

## Analytical question

For one starting pitcher and regular season, estimate how velocity and other
tracked pitch characteristics change as game pitch count rises. The primary
output is a historical profile with uncertainty, not a real-time fatigue
diagnosis.

## Population and unit of analysis

The raw unit is one Statcast pitch row. An eligible appearance:

- begins in inning 1;
- contains at least 40 observed pitch rows; and
- occurs in a regular-season game.

Pitch count is reconstructed before pitch-type and tracking-quality filters.
Within a pitcher-specific extract, rows are ordered by `game_pk`,
`at_bat_number`, and `pitch_number`. MLB defines `at_bat_number` as the plate
appearance number of the game and `pitch_number` as the pitch number of the
plate appearance.

## Data-quality contract

Required fields are:

- `game_date`
- `game_pk`
- `inning`
- `at_bat_number`
- `pitch_number`
- `pitch_type`
- `release_speed`

The composite pitch key is
`(game_pk, at_bat_number, pitch_number)`. Exact duplicates are removed.
Conflicting duplicate keys are retained once and reported as high severity.
Unplaceable rows are excluded.

Plausibility ranges are diagnostic, not winsorization rules:

| Field | Valid range |
|---|---:|
| Release speed | 50–110 mph |
| Spin rate | 0–4,000 rpm |
| Extension | 3–9 ft |
| Horizontal/vertical movement | −5–5 ft |

Values outside these ranges become missing. They are not clipped to a boundary.

Coverage is reported separately for velocity, spin rate, extension, horizontal
break, and vertical break. Spin rate receives an additional missingness
diagnostic: after each window contains at least 50 pitch rows, the project
compares its missing rate during pitches 1–30 with its missing rate at pitch 80
and later. A difference above five percentage points is surfaced in the app.
This diagnostic flags a representation risk; it does not explain why tracking
is missing or determine the direction of any resulting bias.

## Same-type fresh baseline

The core baseline is calculated separately for each:

```text
game_pk × pitch_type
```

For a metric \(m\), pitch type \(t\), and game \(g\):

$$
\text{baseline}_{g,t,m}
=
\operatorname{mean}\left(
m_i \mid game_i=g,\ type_i=t,\ count_i\leq25
\right)
$$

At least two tracked same-type pitches must occur in the first 25 game pitches.
Otherwise that game/pitch-type pair has no baseline-dependent result. There is
no silent fallback to a season mean.

Derived deltas include:

- velocity delta in mph;
- spin-rate delta in rpm;
- extension delta in feet;
- horizontal and vertical movement delta in inches; and
- total movement-magnitude delta in inches.

This design prevents normal repertoire differences from being misclassified as
fatigue.

## Curve aggregation and uncertainty

Pitches are assigned to 10-pitch game-count buckets. Aggregation happens in two
steps:

1. calculate the mean delta within each game and bucket;
2. average those game means across games.

Each game therefore contributes at most one equally weighted value per bucket.
The 95% interval resamples game-level bucket means with replacement. It does not
resample individual pitches as if they were independent.

Every bucket reports:

- contributing games;
- contributing pitches; and
- contributing games divided by all eligible starts.

Coverage is a guardrail against obscuring late-game survivor bias. It does not
eliminate the bias.

## Fatigue-threshold rule

The primary threshold uses the fastball family (`FF`, `SI`, `FC`). Because the
fresh baselines are already pitch-type-specific, their deltas can be combined
without mixing raw velocity levels.

A bucket is a crossing candidate when:

- mean velocity delta is at most −1 mph;
- the bootstrap upper bound is below zero;
- at least five games contribute; and
- at least 30% of eligible starts contribute.

The first candidate must be followed by another consecutive candidate bucket.
The reported result is a range such as `81–90`, not a precise causal pitch.

If adequate late-game buckets exist but no sustained crossing occurs, the
status is `not_reached`. If late-game evidence is inadequate, the status is
`insufficient_data`.

## Pitch-type degradation ranking

For every game and pitch type with at least six pitches spanning at least 30
game pitches, a linear slope is fitted:

$$
\Delta v = \alpha_{g,t} + \beta_{g,t}\times count
$$

Reported pitch-type slopes are means of the per-game slopes, expressed as mph
per 10 game pitches. Confidence intervals bootstrap games. A pitch type needs
at least five game-level slopes to be marked eligible.

The slope summarizes association across the outing. It is not a physiological
decay constant.

## Train/test boundary

All pitches from one `game_pk` belong to exactly one partition. The primary
holdout sorts games chronologically and assigns the latest 20% of games to the
test set.

The pre-pitch model can use:

- game pitch count;
- inning;
- times through the order, when available;
- days since the previous game, when available;
- pitch type; and
- batter side.

Evaluation reports MAE, a game-cluster bootstrap MAE interval, RMSE, R², a
train-mean naive MAE, and game-grouped cross-validation MAE.

## Explanation boundary

A separate descriptive-quality model may add same-pitch:

- spin delta;
- extension delta; and
- movement deltas.

SHAP is computed only when the XGBoost backend and SHAP runtime are available.
Otherwise the project uses held-out permutation importance. Both describe model
associations. Neither establishes that a feature caused velocity loss.

## Experimental Arm Stamina Index

ASI is a transparent portfolio metric with three components:

| Component | Points | Mapping |
|---|---:|---|
| Threshold timing | 50 | Threshold pitch divided by 120; “not reached” receives 120 |
| Late fastball retention | 30 | Linear scale from −2 mph = 0 to 0 mph = 30 |
| Late-game consistency | 20 | Linear scale from 2 mph between-game SD = 0 to 0 SD = 20 |

At least 10 eligible starts, 90% overall velocity coverage, and five games
reaching pitch 80 are required. All three named components must be available;
the project does not silently drop a component and still report a score out of
100. The formula is not population-calibrated, has no validated category
labels, and must not be used as an injury-risk measure.

## Interpretation limits

The design estimates within-game tracking changes conditional on a pitcher
remaining in the game. It cannot separate fatigue from intent, changing pitch
mix, opponent quality, leverage, mechanics, or measurement error. Pitch count
also omits warm-ups and non-game workload. Conclusions should use “associated
with,” “historically observed,” or “supported threshold,” never causal or
clinical language.
