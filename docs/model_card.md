# Model Card

## Intended use

Historical analysis of starting-pitcher velocity retention and model
generalization across games. Appropriate audiences include analysts, coaches,
and portfolio reviewers who understand the documented limitations.

## Out-of-scope use

- injury prediction;
- medical or workload clearance;
- live automated pitching changes;
- causal attribution;
- reliever stamina;
- pitchers with inadequate starter or late-game coverage.

## Target

`speed_delta`: release velocity minus the same-game, same-pitch-type mean among
the first 25 game pitches.

## Model variants

### Pre-pitch context model

Used for headline holdout evaluation. It excludes same-pitch quality
measurements and is evaluated on the latest complete games.

### Descriptive-quality model

Used for associative feature importance. It can include same-pitch spin,
extension, and movement deltas, so its evaluation must not be presented as a
pregame or pre-pitch forecast.

## Algorithms

- preferred: XGBoost regressor;
- fallback: scikit-learn histogram gradient boosting;
- explanation: mean absolute SHAP for XGBoost, otherwise held-out permutation
  importance.

## Evaluation

- complete-game chronological holdout;
- game-grouped cross-validation;
- mean absolute error in mph;
- cluster-bootstrap MAE interval;
- RMSE and R²;
- improvement over a train-mean baseline.

## Known risks

- test-game target normalization uses its first-25-pitch baseline;
- selection into late pitch counts;
- temporal changes within a season;
- missing rest or pitch-quality fields;
- differential early/late tracking missingness;
- changing Statcast pitch classifications;
- small samples by pitch type;
- same-pitch association being misread as causal.

## Communication requirements

Every user-facing result must disclose:

- the data season and as-of date;
- eligible starts and late-game coverage;
- field-specific measurement coverage and material missingness shifts;
- threshold status and criteria;
- game-level holdout design;
- explanation method; and
- experimental ASI status and included component names.
