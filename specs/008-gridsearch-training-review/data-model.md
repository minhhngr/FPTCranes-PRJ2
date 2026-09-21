# Data model: GridSearchCV evidence

## `GridSearchContext`

- `context_id`: `outer-1` through `outer-5` or `final-train`
- `parent_population_id`, `outer_fold_id`, `inner_fold_ids`
- `train_positions`, `excluded_positions` (producer-only; never render raw positions)
- `scoring`: declared metric and direction
- `seed`, `status`, `reason`

## `GridSearchCandidateEvidence`

One row per context and Cartesian parameter candidate:

- `method_version`, `context_id`, `candidate_id`, `rank`, `params`, `evidence_ref`
- `n_estimators`, `max_depth`
- frozen `min_samples_leaf`, `max_features`
- `mean_test_MAE`, `std_test_MAE`, fold-level MAE values
- `mean_test_R2`, `std_test_R2`, fold-level R² values, or an explicit non-finite reason
- `mean_fit_time_s`, `std_fit_time_s`, `status`, `failure_reason`
- `inner_fold_ids`, `evidence_ref`

## `TemporalFoldEvidence`

- fold/context identity, parent identity, ordinal
- train/validation month bounds and counts
- train/validation/not-yet-used row counts
- holdout/reserve exclusions
- chronology, row-overlap, and expanding-history checks
- `evidence_ref`

## `TrainingStepEvidence`

- ordered step ID, stage, status, timestamp when raw
- context/fold/candidate identity
- human-readable action and outcome
- `source_kind`: `raw_event` or `evidence_derived`
- `evidence_ref`

## `TrainingExecutionLog`

- deterministic English evidence-derived text written as `training.log`
- 5W1H plus ordered partition → feature engineering → training → evaluation → status stages
- requested and actual whole-month split counts/shares
- fold/model/tuning/runtime/fit/feature/holdout/residual/subgroup/q90 evidence from persisted values
- final scientific and operational status with safe next action

## `EvidenceCompatibilityState`

- `status`: `valid`, `invalidated`, `unavailable`, `incompatible`
- `reason_code`, safe message, next action
- producer/schema/source/policy identities
- must contain no fallback metrics.
