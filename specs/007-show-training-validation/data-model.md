# Data Model: Training-Validation UI Consumer

## 1. TrainingValidationPackDescriptor

Represents one discovered immutable final pack.

| Field | Type | Rules |
| --- | --- | --- |
| `run_id` | string | `^tv-[0-9a-f]{32}$`; matches directory and manifest |
| `generated_at` | UTC datetime/string | strict ISO-8601 ending `Z`; required for selection |
| `manifest_path` | workspace-relative path | fixed run namespace; no symlink/escape |
| `manifest_sha256` | 64-char hex | calculated after validation for display/cache identity |
| `execution_status` | enum | must be `complete` |
| `schema_version` | string | `training-validation/v1` |
| `experiment_sha256` | 64-char hex | displayed as provenance, never used as chronology |
| `selected_family` | string/null | reconciles selection and conclusions |
| `scientific_outcome` | structured value | separate from operational assessment |
| `invalid_candidates` | bounded list | safe reason code/run ID only; max 20 displayed |

Selection ordering is `(generated_at, run_id)` descending among fully validated descriptors.

## 2. TrainingValidationUnavailable

| Field | Type | Meaning |
| --- | --- | --- |
| `available` | false | stable state discriminator |
| `reason_code` | enum | `ROOT_MISSING`, `PACK_ROOT_MISSING`, `NO_FINAL_PACK`, `NO_VALID_PACK`, `PACK_LIMIT_EXCEEDED`, `PACK_INCOMPATIBLE` |
| `message` | string | bounded English; no raw artifact content/path outside workspace |
| `invalid_candidate_count` | integer | 0–100 |
| `inspect_command` | string | fixed read-only command using established workspace |
| `next_action` | string | eligible data/approval guidance; never starts run |
| `fallback_used` | false | invariant |

## 3. TrainingValidationView

Top-level consumer projection.

| Field | Type | Rules |
| --- | --- | --- |
| `available` | bool | true only after complete pack and table validation |
| `descriptor` | PackDescriptor/null | required when available |
| `page04` | Page04TrainingView | independent section states |
| `page05` | Page05TrainingView | independent section states |
| `downloads` | list | manifest-listed source file only; path/checksum already validated |
| `warnings` | bounded list | scientific/availability warnings, not raw exceptions |

## 4. Page04TrainingView

- `partition`: counts, actual shares, approved boundaries, reserve exposure label.
- `outer_folds`: exactly five outer rows with periods, parent denominator, train/validation/later rows, added history and leakage checks.
- `inner_folds`: exactly 18 rows (five outer parents plus final TRAIN, each three) when RF nested plan evidence is present; parent/search labels retained even when tuning skipped.
- `monthly_counts`: one row per observed month/partition.
- `candidate_summary`: exactly five declared candidate roles; approved reader-facing metrics only.
- `selection`: lowest MAE model, selected family, overlap/simplicity reason.
- `performance`: bias/tail/Dummy skill/Pareto/runtime summary where available.
- `conclusions`: exactly five candidate records, preserving status, decision, limitation, action and evidence references.
- `section_states`: local availability/reason for optional runtime/importance/ablation detail.

## 5. Page05TrainingView

- `tuning_contexts`: six contexts with status, skip reason or winner, and actual fit count.
- `tuning_trials`: bounded preview plus complete download; only when completed.
- `variant_folds`: paired `full`/`top2` rows per outer fold.
- `holdout_metrics`: seven declared roles, partition exactly `EVALUATION_HOLDOUT`.
- `holdout_prediction_summary`: aggregate only (row count, residual center/tail); raw row payload is download-only.
- `encoded_importance`: bounded Top-N per final role with explicit method.
- `permutation_importance`: repeat-aggregated raw-feature/family MAE increase; raw repeats download-only.
- `subgroups`: bounded support/error table with `small_sample` retained.
- `uncertainty`: Full and Top-2 q90/coverage, method and `same_holdout_descriptive_not_calibrated` basis.
- `outcome`: scientific result and separately nested operational assessment.
- `conclusions`: exactly two final-role records plus overall selected-family conclusion.

## 6. TrainingTranscriptRow

A deterministic projection over authoritative evidence. It is always labelled `evidence-derived transcript`, never `raw execution event`.

| Field | Type | Rules |
| --- | --- | --- |
| `sequence` | positive integer | deterministic method order, not an invented timestamp |
| `stage` | enum | `partition`, `outer_fold`, `candidate_fold`, `selection`, `inner_fold`, `tuning_trial`, `variant_fold`, `final_fit`, `holdout`, `explainability`, `subgroup`, `uncertainty`, `conclusion`, `publication` |
| `step` | string | stable reader-facing operation name |
| `model_role_id` | string/null | one of five candidate or two final roles where applicable |
| `configuration_id` | string/null | copied from evidence; never inferred from display rounding |
| `fold_id` | string/null | outer/inner identity |
| `parent_fold_id` | string/null | required for inner folds |
| `search_id` / `trial_id` | string/null | tuning identity |
| `partition` | enum/null | TRAIN, OUTER_VALIDATION or EVALUATION_HOLDOUT only; never reserve metrics |
| `train_periods` / `validation_periods` | string/list/null | copied from fold evidence |
| `parent_rows` | integer/null | explicit denominator |
| `train_rows` / `validation_rows` / `not_used_yet_rows` | integer/null | exact source counts |
| `added_train_rows` | integer/null | explicit expansion delta/reason |
| `holdout_rows_excluded` / `reserve_rows_excluded` | integer/null | required for fold explanations |
| `parameters` | JSON-safe object/null | copied tuning/final settings |
| `metrics` | JSON-safe object/null | copied source metrics with units/scope metadata |
| `status` / `reason` | string/null | completed/reused/skipped/failed/unavailable semantics |
| `finding` / `decision` / `limitation` / `next_action` | string/null | copied conclusion fields |
| `source_file` / `evidence_ref` | relative string | mandatory authoritative provenance |
| `record_kind` | constant | `evidence_derived`, preventing confusion with runtime event rows |

The complete transcript includes every applicable row. UI previews are capped at 200 rows per table; the generated CSV download is deterministic and untruncated.

## 7. RawLogView

- `training_log_preview`: bounded UTF-8 lines from `training.log`, labelled short human log.
- `event_preview`: bounded validated JSONL lifecycle rows from `events.jsonl`, labelled raw structured events.
- `report_preview`: bounded Markdown excerpt from `report.md` without unsafe HTML execution.
- Original bytes for `training.log`, `events.jsonl`, `report.md`, `agent_summary.json` and `model_conclusions.json` remain byte-identical downloads.
- Any disagreement with authoritative transcript sources becomes a warning and does not rewrite original bytes.

## 8. DownloadDescriptor

| Field | Type | Rules |
| --- | --- | --- |
| `label` | string | reader-facing |
| `filename` | basename | no path components |
| `mime` | enum | CSV, JSON, Markdown or text only for UI download |
| `content` | bytes | loaded only from validated manifest-listed file |
| `source_path` | relative string | never displayed as external absolute path |
| `sha256` | string | matches manifest |

Model bundles and HTML chart files are not offered as UI downloads by this feature. One additional generated descriptor named `training-validation-transcript.csv` contains the full in-memory transcript; it has no source path because it is not written to the workspace, and records its selected run/manifest identity in download metadata.

## 9. HistoricalTrainingReport

- `source_kind`: constant `historical_primary_pipeline_and_supplemental`.
- `audit_run_id`, `pipeline_run_id`, `evidence_id`, source SHA-256 and validation status.
- `exposure_label`: required statement that locked-test evidence is historically exposed and does not restore unseen-test purity.
- `branch_flow`: common prepared feature base → Branch A analytical completion → Branch B salary comparison/tuning/evaluation; cluster labels are not model inputs.
- `candidate_records`: exactly five `Model5W1HRecord` values.
- `final_records`: selected-family plus Full/Top-2 records where supported.
- `event_preview`: bounded `HistoricalAuditEvent` rows.
- `downloads`: checksum-verified original manifest, complete JSONL and relevant exports.

## 10. Model5W1HRecord

| Field | Rules |
| --- | --- |
| `model_role_id` / `configuration_id` | validated role/config identity |
| `who` | audit/pipeline identity and model role; never a fabricated person |
| `what` | regression target plus evidence-backed feature/config scope |
| `when` | source fold/evaluation periods |
| `where` | DEV temporal validation or historical locked-test scope |
| `why` | declared role/decision from evidence, not causal speculation |
| `how` | fold-local preprocessing/fit/evaluation and configuration |
| `result` | source metrics with explicit units/scope |
| `limitation` / `next_action` | mandatory bounded interpretation |
| `evidence_refs` | one or more validated relative references |

Page 04 has exactly five candidate records. Page 05 has selected-family and supported Full/Top-2 final records. Missing evidence produces explicit unavailable fields, never invented text.

## 11. HistoricalAuditEvent

- audit/pipeline run IDs, sequence/timestamp when present, depth, operation, status, model/fold/trial identity, bounded message and relative evidence reference;
- preview includes only model comparison, tuning, final-estimator and evaluation events needed by Pages 04–05;
- raw fields are not interpreted as strict `training-validation/v1` events;
- complete original JSONL bytes remain downloadable.

## Invariants

1. No view exists without successful complete-pack validation.
2. No record from another run/evidence source is merged into a view.
3. No `INFERENCE_RESERVE` prediction/metric row is accepted.
4. Missing/null is never converted to zero or favorable language.
5. Display rounding occurs after validation/selection and never drives decisions.
6. The UI does not deserialize bundles or call model operations.
7. Unavailable state always has `fallback_used=false`.
8. Raw logs, raw events and evidence-derived transcript are visibly and structurally distinct.
9. Every transcript row has an authoritative source file/reference; no absent step or value is invented.
10. All 25 candidate-fold rows and all seven model-role conclusions are preserved in the complete transcript/download.
