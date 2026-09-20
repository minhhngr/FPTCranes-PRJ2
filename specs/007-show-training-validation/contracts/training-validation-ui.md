# Contract: Read-Only Training-Validation UI v1

**Status**: Proposed for task approval  
**Producer**: Existing offline `ai_job_market.training_validation run`  
**Consumers**: Streamlit Pages 04 and 05 only

## Source selection

The consumer searches only:

```text
<workspace>/outputs/training_validation/tv-[0-9a-f]{32}/manifest.json
```

It ignores hidden staging directories, `holdout_access`, arbitrary paths and symlinks. At most 100 final run directories are considered. Each candidate must:

1. have a bounded regular JSON manifest;
2. declare `schema_version=training-validation/v1`, matching run ID and `execution_status=complete`;
3. have a strict UTC `generated_at` ending `Z`;
4. pass `validate_complete_pack(workspace, run_id)`, including all evidence and bundle metadata/hash/size checks without bundle deserialization.

Select max `(generated_at, run_id)` among valid candidates. A newer invalid candidate does not suppress an older valid pack. If no valid candidate remains, return unavailable with bounded reason metadata.

## Read contract

Only paths listed in the selected manifest may be read. Every path is re-resolved under the workspace without symlink traversal. JSON/CSV/Markdown/text downloads are allowed; joblib and generated HTML are never loaded or offered by this UI consumer.

Limits:

- candidate run directories scanned: 100;
- manifest/control JSON: existing 1 MiB boundary;
- individual display CSV: 50 MiB and 100,000 rows maximum, then fail the affected pack/section according to requiredness;
- displayed table preview: 200 rows;
- invalid-candidate reasons displayed: 20;
- downloadable bytes must match manifest size/hash and are not transformed.

Required Page 04 files:

```text
partition_declaration.json
partition_membership.csv
fold_summary.csv
monthly_row_counts.csv
fold_explanation.md
candidate_summary.csv
candidate_fold_metrics.csv
fit_diagnostics.csv
runtime_summary.csv
performance_comparison.csv
accuracy_runtime_tradeoff.csv
family_selection.json
model_conclusions.json
agent_summary.json
training.log
events.jsonl
report.md
```

Required Page 05 files:

```text
tuning_trials.csv
tuning_summary.json
variant_fold_metrics.csv
evaluation_declaration.json
holdout_metrics.csv
holdout_predictions.csv
encoded_importance.csv
permutation_importance.csv
subgroup_metrics.csv
uncertainty.json
operational_assessment.json
conclusion.json
model_conclusions.json
training.log
events.jsonl
report.md
agent_summary.json
```

The complete-pack validator already requires the full producer inventory. UI section parsing additionally validates the fields used for display.

## Always-visible trust overview

Pages 04 and 05 render a compact `Training & validation evidence` container before the collapsed details and before existing supplemental-source early returns. It is additive and MUST NOT modify or replace existing page sections.

When a valid pack exists, the overview shows verified offline status, run ID/generated time, evidence-derived counts appropriate to the page, and the TRAIN-only tuning / `EVALUATION_HOLDOUT` / unevaluated `INFERENCE_RESERVE` trust boundary. It states that the display is read-only evidence, not live training or model activation.

When no valid pack exists, the overview may describe the supported offline workflow but MUST state that no verified active-workspace evidence exists and that capability text is not a completed or trusted run. It shows no fallback metrics. Detailed reasons and the read-only inspect command remain in the collapsed unavailable content.

The overview is intentionally isolated in one helper plus one call per page renderer so it can be removed after stakeholder verification without changing existing Page 04/05 analytics.

## Page 04 expander contract

Label: `Latest training validation: folds and candidate models`.

When available, show in order:

1. run ID, generated time, complete status and source distinction;
2. partition count/share/boundary summary;
3. outer fold table (five rows) with periods and exact train/validation/later/added counts;
4. inner fold table (18 rows) with parent/search IDs and exact counts;
5. candidate table with MAE, RMSE, MedAE, R², fit time and fold stability;
6. explicit lowest-CV-MAE, selected-family and fastest-fit identities;
7. five candidate conclusions with finding, decision, limitation and next action;
8. evidence-derived transcript preview containing all five outer split explanations and 25 candidate-fold evaluation rows;
9. one scoped conclusion for each of five candidate roles;
10. approved original log/report/summary downloads plus complete generated transcript CSV.

No value from existing `ui_evidence` may populate this expander.

## Page 05 expander contract

Labels:

- `Latest training validation: tuning and Full/Top-2 variants`
- `Latest training validation: holdout, explainability and uncertainty`

The first shows tuning status/reason/winner for six contexts, trial count/fit count, parent-labelled fold references and paired Full/Top-2 fold metrics. Its evidence-derived transcript follows method order and includes every context/trial status, parameters, inner folds, reuse/skip reason, applied settings and variant fold step. Non-RF skip is explicit and not treated as failure or successful tuning.

The second shows `EVALUATION_HOLDOUT` metrics, aggregate residual evidence, encoded importance method, repeat-aggregated permutation MAE increase, subgroup support/small-sample flags, uncertainty basis/method/coverage, scientific outcome, operational status and Full/Top-2 conclusions. Its transcript covers final fit declaration, one-time holdout scoring, explainability, subgroup, uncertainty and conclusion steps. It always states that the inference reserve has no metrics/predictions and that holdout ranking does not alter frozen selection.

## Log and transcript contract

The UI presents three distinct evidence types:

1. **Short human execution log** — a bounded preview of original `training.log` plus byte-identical download.
2. **Raw structured lifecycle events** — validated bounded preview of `events.jsonl` plus byte-identical download.
3. **Evidence-derived detailed transcript** — deterministic rows reshaped from authoritative CSV/JSON evidence, visibly labelled as derived and downloadable completely as `training-validation-transcript.csv` without workspace writes.

The derived transcript MUST include:

- partition declaration and monthly allocation;
- five outer fold explanations with parent denominator, train/validation periods and exact rows, later-unused/added rows, holdout/reserve exclusions and leakage checks;
- 25 candidate fold evaluations and five candidate conclusions;
- 18 parent-labelled inner fold descriptors;
- every tuning context/trial or explicit skip/reuse reason;
- ten matched Full/Top-2 outer-fold evaluations;
- final fit/freeze declaration;
- seven holdout-role metric summaries and aggregate residual evidence;
- encoded/permutation importance method steps, subgroup support warnings and uncertainty basis;
- two final-role conclusions, scientific outcome, operational assessment and publication identity.

Each row has `record_kind=evidence_derived`, deterministic sequence, source file and evidence reference. The sequence represents method order unless a real event timestamp exists; the UI MUST NOT invent timestamps. Missing values retain explicit null/reason semantics. No displayed or exported metric may be recomputed from rounded values.

Original downloads required on both relevant pages are `training.log`, `events.jsonl`, `report.md`, `agent_summary.json` and `model_conclusions.json`. The complete generated transcript is an additional in-memory CSV download. UI previews may cap rows/lines, but downloads are never silently truncated.

## Historical Branch B report contract

This report is additive and independent of strict `training-validation/v1`. Its source badge is `Historical primary-pipeline + supplemental evidence`, with pipeline/audit/evidence IDs and the known-exposure limitation visible before metrics.

Page 04 shows exactly five compact candidate 5W1H records. Page 05 shows the selected-family and supported Full/Top-2 records plus inherited tuning/final evidence. Every record contains Who, What, When, Where, Why and How, followed by result, limitation, next action and source reference. `Why` is a role/decision explanation, not an invented causal diagnosis.

A collapsed `Historical training activity log` provides bounded model-relevant raw-event rows and distinguishes them from any evidence-derived report/table. Downloads include the checksum-verified original manifest, complete JSONL and relevant fold/model/tuning exports already accepted by `load_compatible_training_audit()`.

The branch-flow explanation states that the common prepared feature base supports both branches and that Branch B runs after upstream preparation/analysis. It MUST NOT claim that Branch A cluster assignments are Branch B training features unless a future approved producer contract actually adds them.

No historical value may fill strict-pack identity, folds, holdout/reserve assertions or scientific outcome. Strict unavailable content remains visible when no valid strict pack exists.

## Unavailable contract

The expander remains present and collapsed. Its content includes:

```text
Training validation unavailable
Reason: <safe bounded English>
No fallback evidence is displayed.
Read-only check: .venv/bin/python training_validation.py inspect --workspace <workspace>
Next action: obtain eligible future prepared data and exact approval before an offline run.
```

Do not include raw exception traces, arbitrary artifact content, credentials or an executable `run` button.

## Streamlit operation prohibition

During import, page render, expander render and rerun, the consumer MUST NOT call:

- `fit`, training comparison/evaluation/search/finalization;
- `predict`, permutation benchmark or model deserialization;
- `training_validation.run_workspace` or CLI `main`;
- `publish_staged_pack`, `claim_holdout_access`, `complete_holdout_access`, or writes;
- `pipeline.py` or supplemental evidence producer.

Only safe path resolution, hashing, JSON/CSV/text reads, validation, pure aggregation/formatting and Streamlit rendering are allowed.

## Existing source independence

Training-validation rendering happens before existing `ui_evidence` load/early return. The states form a matrix:

| Training validation | Existing supplemental source | Required behavior |
| --- | --- | --- |
| available | available | show new expanders and existing page |
| unavailable | available | show unavailable expander and existing page |
| available | unavailable | show available expander, then existing source error |
| unavailable | unavailable | show unavailable expander, then existing source error |

No source supplies fallback fields for the other.
