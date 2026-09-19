# Data Model: Training Observation and Evidence

These are event/file contracts, not database tables or a mandate to add classes. Reuse dictionaries and the existing session. All additions are observation-only.

## TrainingRun

- `audit_run_id`: unique generated identifier; independent of existing `pipeline_run_id`.
- `source_path`, `source_sha256`, `config_sha256`, `seed`, `target`, `versions`, `code_revision`, `dirty_worktree`: reproducibility facts; unavailable values have explicit reasons, not fake hashes.
- `debuglog`: presentation flag only; must not filter persisted events.
- `log_path`, `evidence_directory`: unique owned destinations within the workspace.
- `training_status`: started → completed | failed | cancelled. No end event after a forced kill means incomplete/unknown, not completed.
- `log_complete`: monotonic true → false on structured event/export failure.
- `console_complete`: independent true → false on terminal failure.
- `coverage_complete`: false until expected executed stages have real terminal states; skipped branches count only with reasons.
- Relationships: run owns many observations, evaluations and exports.

## Observation

Existing envelope retained: `schema_version=1`, UTC timestamp, audit/pipeline IDs, increasing `sequence`, severity `level`, `event`, `message`, `step_id`, `operation`, `status`.

Add optional fields: `detail_level` (integer 1–4), `stage_id`, `parent_id`, `evaluation_id`, `trial_id`, `fold_id`, `split_id`, `dataset_id`, `evidence_refs`, `elapsed_s`, `counts`, `units`, `missing_reason`.

- `level` remains INFO/WARNING/ERROR; it is not indentation.
- Status remains one of started/completed/skipped/failed/cancelled for events. CLI `UNKNOWN` is the absence of a trustworthy terminal stage event, not a synthetic success event.
- Unique step IDs include invocation context; basename and model name alone are not identity.
- Do not allow payload fields to overwrite envelope identity/sequence/status.
- Every start has one end unless process termination prevents it. Failures close active child operations before parent failure. A skipped operation must have a reason and no start/fit call.
- Unavailable metrics serialize as null plus reason; non-finite values must not produce invalid JSON tokens.

## DatasetIdentity and Partition

- Source fingerprint ties the run to exact source bytes.
- `dataset_id`: digest of the actual DEV content and order, recorded with fingerprint method/version, columns and row count. Compute once per observed DEV object/context; do not mutate it.
- `dataset_row_order_sha256`: existing legacy positional digest retained, explicitly documented as insufficient for content identity.
- Partition role: development | fold_train | fold_validation | locked_test.
- Actual period bounds, selected test rule, count, reference to parent dataset and split.
- Membership positions: zero-based in the original DEV frame consumed by the splitter. Preserve order and duplicate-index behavior; do not use nonunique dataframe labels as keys.
- Count invariants: raw − invalid-category − subsequent duplicates = clean; DEV + test = prepared; each membership table's row count equals its declared role count. No invariant assumes equal fold sizes or disjoint months.

## SplitDefinition

- `split_id`: existing membership-derived split key, scoped with `dataset_id` and run.
- Requested/effective folds, sort policy, actual train/validation positions and time bounds, row overlap and shared months.
- Deduplication key: dataset identity + split identity, not row count alone.
- Many evaluations reference one definition; logging does not call the split function again.

## ModelEvaluation

- `evaluation_id`, parent stage, model display name/family, variant label, optional trial ID.
- Target, actual ordered raw features, full effective safe estimator parameters/seed, fitted encoded-feature count where available.
- Fold results: partition=validation, fold/split reference, row counts, MAE/RMSE/MedAE in USD, R² unitless, observed fit/predict durations.
- Aggregates: arithmetic mean and population SD (`ddof=0`) for observed finite fold scores, contributing fold count and missing count per metric. Do not silently exclude a missing fold without showing it. Existing training summary/ranking is not changed.
- Train metrics: unavailable when not computed. Locked-test results exist only for the actual final evaluation; do not attach that result to a different baseline/tuned candidate as if they were the same fit.
- `fit_assessment`: insufficient_evidence for current production evidence; missing training scores and rule are named. Reserved contract labels overfitting/good_fit/underfitting require explicit rule ID/thresholds, baseline and paired evidence before any future use.

## TuningTrial and Selection

- Trial identity includes group and ordinal; explicit planned/started/completed/failed/skipped status and effective parameter values.
- 26 current RF trials across groups of 4/6/6/4/6, but display progress from actual loop inputs rather than hardcoding totals.
- Trial references evaluation/folds; group winner references its actual result row and ranking metric/direction.
- Final selection records family choice, tuning table source, applied parameters and differences versus later group recommendations. Null max_depth means unbounded when it is a valid parameter; it does not mean a missing metric.
- Non-RF path has explicit tuning skipped; no fake zero-trial winner.

## EvidenceExport

Manifest entry fields: `evidence_id`, `kind`, `path` (relative to run directory), `status` (complete | failed), `rows`, `columns`, `sha256`, source event sequence and stage/model/trial/split references as applicable. Snapshot entries also carry original artifact path/hash. Failed entries have a reason and no claim of a valid complete file.

Kinds: data_count_ledger, feature_inventory, fold_membership, model_comparison, tuning_trials, metric_details, segmentation_summary.

- CSVs contain headers, full precision and explicit identity/status/reason fields. Display rounding never changes saved values.
- Membership CSV: one row per fold/role/position; no raw feature/target values.
- Feature CSV: one row per evaluation/feature (or fitted feature name), with ordering and policy/scope.
- Metrics/tuning: one row per evaluation/trial/fold/partition/metric in long form where needed; summary table may be wide as specified in the contract.
- Run manifest is finalized on success/failure/cancellation when writable; exclusive filenames prevent overwriting another run. Missing manifest after abrupt kill means incomplete evidence, even if CSV fragments exist.
- Retain/export only one table at a time. Session keeps identities and manifest metadata, not every dataframe.
