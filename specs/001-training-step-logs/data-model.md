# Data Model: Training Audit Evidence

All entities below are in-memory/session state or additive log records, not database tables. Existing model artifacts and return schemas are unchanged.

## Run

- `schema_version`: integer, initially 1.
- `audit_run_id`: generated UUID-based identity, unique across executions.
- `pipeline_run_id`: existing run ID, nullable until known; never replaced.
- `source`: allowlisted input identifier, SHA-256, byte size; unavailable fields carry a reason.
- `configuration`: relevant seed, requested temporal folds, preferred locked period, feature/target lists and configuration hash.
- `environment`: Python and relevant installed package versions; Git revision and dirty-worktree flag (unknown explicitly represented).
- `log_path`, `started_at`, `ended_at`, `status`, `log_complete`.

State transitions: initializing -> running -> succeeded / failed / cancelled. Logging completeness is independent: a scientifically successful run can have `log_complete=false`. A failed run never becomes succeeded because error reporting succeeded. Previously written records remain append-only.

## Event

- Common required fields: `schema_version`, `timestamp` (UTC ISO 8601), `audit_run_id`, `sequence` (positive, monotonically increasing within run), `level`, `event`, `message`, `step_id`, `operation`, `status`.
- Context where applicable: `pipeline_run_id`, `phase`, `model_id`, `model_name`, `trial_id`, `split_id`, `fold_id`, `partition`.
- Completion: non-negative elapsed seconds; counts/shapes and computed metrics as applicable.
- Error: exception class and sanitized English context; no arbitrary raw exception payload or local-variable dump. Preserve the original raised exception independently of logging.

A logical operation has `started` then one terminal state: `completed`, `failed`, or `cancelled`. `skipped` is a standalone terminal state with an actual reason. Abrupt process termination can leave an incomplete log; absence of a terminal event is never interpreted as success.

## Dataset / Split Definition

- `dataset_id`: content/order digest scoped to the prepared DEV dataset; linked to source hash and existing development artifact path.
- Row positions are zero-based offsets in that dataset, not DataFrame index labels or predictive features.
- `split_id`: deterministic digest of dataset identity, effective splitting policy, and ordered memberships.
- Policy: sorting columns, stable order rule, requested/effective folds, configured/effective block size, remainder rule.
- Per fold: `fold_id`, ordered `train_positions` and `validation_positions`, actual counts, train/validation min/max periods, row-intersection count, shared periods.

One definition per unique split per run; every evaluation references it. Reuse across folds is not a within-fold overlap violation. Bounds and intersection observations do not modify or reject existing partitions; invalid pre-existing behavior is recorded and retains its original failure path.

## Candidate Evaluation

- Stable phase/model/trial identity; actual feature set and primitive estimator parameter values, seed, consumed split ID.
- Per fold: MAE/RMSE/R²/MedAE and durations from operations already performed.
- Aggregate: existing mean metrics and MAE population SD; additional logged R² population SD (`ddof=0`), labeled as supplementary.
- Selection: ranking metric/direction, selected row/source table, fixed overrides and actual final parameters recorded separately.
- Metric status: finite numeric value, or `null` with `non_finite`, `not_computed`, or `unavailable`. No added predictions to fill absent metrics.

## Artifact Reference / Finding

- Artifact: existing path relative to workspace, format, producing operation and success/failure. Reference existing charts/outputs; do not embed whole files in logs.
- Finding: identifier, observed behavior, source function and artifact/run reference, verification category (`static_inspection`, `historical_artifact`, `runtime_verified`), implications, deferred status.
- Findings are authored in `docs/TRAINING_AUDIT.md`; logging observations are not an automated scientific verdict.

## Validation Boundaries

No arbitrary user-authored IDs or destination strings are introduced. The generated log filename must be exclusive and remain inside the resolved workspace log directory; reject symlinked log destination components. Validate audit payload types and serialize only allowlisted primitives/arrays. A payload or file failure results in visible evidence degradation, not model-data mutation. Secrets, raw dataset contents, and process environment variables are excluded.
