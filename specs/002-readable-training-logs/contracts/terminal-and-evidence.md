# CLI, Observation and Evidence Contract

**Status**: Proposed implementation contract; the new flag/arguments do not exist yet.

## 1. CLI and callable compatibility

Existing command: `python pipeline.py [--data PATH] [--heartbeat SECONDS] [--no-heartbeat]`.

Add `--debuglog` as an argparse `store_true` option, default false; no explicit value accepted. Preserve existing flags and exit semantics: 0 scientific pipeline return, 1 training exception, 2 argument/preflight error, 130 interruption. A logging-only failure reports incomplete evidence without turning completed training into a failed experiment. Do not silently convert exceptions to successful return values.

Proposed callable:

```python
run_pipeline(raw_path, root=None, workspace_root=None, *, debuglog=False, on_audit_event=None)
```

- Existing positional calls and returned summary keys remain compatible.
- Debug affects only terminal depth.
- Optional observer consumes a normalized event dictionary for CLI progress/timing, must not mutate training state, and has no result used by ML logic. Pass an independent event view/copy so observer mutation cannot corrupt saved evidence.
- Observer exceptions produce an observation warning; session/context restoration is guaranteed.
- Preserve `emit_event` callers by making new depth/context arguments optional. Existing event names retain their meaning and file envelope fields/types.

## 2. Terminal behavior

| Depth | Normal | Debug | Meaning |
|---|---|---|---|
| 1, 0 spaces | yes | yes | Stage/model action and completion status |
| 2, 2 spaces | yes | yes | Counts, progress, candidate/tuning summary, units and export locations |
| 3, 4 spaces | no | yes | Individual folds/trials and segmentation candidate breakdowns |
| 4, 6 spaces | no | yes | Exact parameter/feature/count derivations, membership previews and complete-evidence links |

Severity warning/error bypasses depth filtering. Use explicit labels rather than color alone. Group writes under one lock shared with CLI heartbeats; no live redraw or carriage-return-only status. Narrow output wraps text without dumping dictionaries or losing hierarchy. A broken output stream never suppresses file evidence.

Illustrative shape only (values are placeholders, not results):

```text
[RUNNING] B4 — Compare salary models using development CV
  Source: <source> | DEV: <count> rows | Inputs: <count> features
[RUNNING] Random Forest — fit and validate candidate
  Progress: <done>/<total> folds | seed=<seed>
    Fold <id>: train=<count>, validation=<count>; period=<period>
      Fit scope: fold training only | encoded features=<count>
  CV: MAE=<value> USD | RMSE=<value> USD | R²=<value> (unitless) | MedAE=<value> USD
  Fit assessment: insufficient evidence — training scores and diagnostic rule unavailable
  Evidence: <run-relative CSV path> (<total> rows; <omitted> omitted from preview)
[COMPLETED] B4 — comparison and its required outputs completed
```

No raw JSON event or Python dict repr is printed, including error/degraded paths. Third-party warnings are not reclassified as depth-4 noise. Normal output still has every model's summary; limiting a table preview does not omit model identities from the complete sequence of summaries.

## 3. Stage and operation events

New event families: `stage_started`, `stage_completed`, `stage_failed`, `stage_cancelled`, `stage_skipped`, `data_summary`, `preprocessing_summary`, `evaluation_started`, `evaluation_completed`, `trial_started`, `trial_completed`, `trial_failed`, `tuning_skipped`, `evidence_exported`, `evidence_export_failed`.

Reuse existing split/operation/selection/artifact/run event families. Every production family must have a tested depth and readable formatter. Add severity WARNING to logging-degraded records without changing valid scientific status values.

Stage IDs follow the current CLI order: C1, C2, C3, C4, C5, B1–B3, A1–A8, B4, B5–B6, B7, I1, F. Persist stable ASCII IDs (`B1-B3`, `A1-A8`, `B5-B6`); display labels may retain typographic dashes. Parent IDs preserve the actual nesting, not inferred filename order.

Stage completed means all operations and existing writes within its [coverage.md](../coverage.md) boundary returned. A marker alone is not completion. Optional tuning skipped does not skip final evaluation. Failed trials abort exactly where the existing code aborts; this feature does not add retries or continue training.

If core returns but expected stage observations are missing, show `training completed; observation coverage incomplete`, mark missing stages UNKNOWN and never fabricate stage durations or PASS. Audit-only tables/manifests may report this separately without rewriting existing scientific status artifacts.

## 4. File layout and envelope

```text
<workspace>/outputs/08_full_pipeline/logs/
  training-<UTC>-<audit_run_id>.logs       # strict JSONL, all depths
  training-<UTC>-<audit_run_id>/
    manifest.json
    data-counts.csv
    features-<evaluation-id>.csv
    membership-<dataset-id>-<split-id>.csv
    comparison.csv
    tuning-trials.csv
    <safe-evidence-id>.csv
```

Evidence files are generated as applicable; no empty placeholder table may imply a computation occurred. Basename uniqueness comes from generated identifiers, never arbitrary user strings. Preserve existing `.logs` envelope and schema_version 1; new optional metadata is additive. Legacy readers that ignore unknown keys remain supported. No existing model/artifact format migrates.

Manifest has `schema_version=1`, audit/pipeline run IDs, source/config identity, training status, `log_complete`, `console_complete`, `coverage_complete`, and `exports` (entries defined in data-model.md). Do not list a failed/missing file as a successful export. Record fingerprint methods. Manifest and terminal identities must match the current run.

## 5. CSV and table semantics

- UTF-8, header row, comma-separated with standard CSV quoting; no formatted currency symbols in numeric columns.
- Maximum 20 displayed **data rows per table**; print total, shown, omitted and complete export path. Apply this to feature/parameter/membership lists normalized as tables too.
- Persist full safe table in both verbosity modes, including tables not printed normally. Reuse immutable same-run evidence; otherwise snapshot relevant mutable source evidence after its successful write.
- `data-counts.csv`: stage_id, source_ref, input_rows, output_rows, excluded_rows, reason, formula, event_sequence. One exclusion reason per row where relevant; denominators explicit for percentages.
- Feature files: evaluation_id, scope, feature_order, feature_name, role, policy_reason. Distinguish raw model input, encoded fitted name and experimental variant.
- Membership: dataset_id, split_id, fold_id, role, position. One row per actual role membership; positions must be in range for the recorded DEV frame. Full arrays may also remain in JSON; do not filter them by debug mode.
- `comparison.csv`: evaluation_id, model, variant, partition, MAE, RMSE, R2, MedAE, metric_status, missing_reason, fit_assessment, fit_reason; fold aggregates use explicitly named mean/SD columns or separate metric-detail rows.
- `tuning-trials.csv`: stage_id, trial_id, evaluation_id, ordinal, status, seed, four actual tuned parameter columns, CV metrics, ranking_metric, ranking_direction, selected_in_group. Preserve valid unbounded max_depth distinctly from unavailable numerical metrics through parameter status/representation.
- Missing numerical values have empty numeric cells and explicit status/reason; they are never zero-filled. Original floating-point values are saved without display rounding. Parameters and aggregation conventions are recorded separately from measured metrics.
- Current fit assessment always says insufficient evidence; three reserved labels must not be produced from legacy rationale strings, a single R² or CV–test difference.

## 6. Failures, completeness and concurrency

- Attempt structured write independently from console write; one failure does not gate the other.
- Logging initialization, serialization, renderer, observer, export, manifest, flush and close errors produce best-effort readable warnings and accurate completeness. Preserve actual training exceptions and exits.
- Fail closed for unsafe destinations; ordinary logging I/O failures can continue in degraded mode. No new auth/network controls.
- Exclusive run files and scoped context prevent cross-run overwrite/leakage. Nested sessions restore the parent; repeated calls do not accumulate handlers/observers. Threaded heartbeat writes share the output lock; estimator workers do not open independent audit files.
- Context tracking/extraction must remain no-op without an active session. No audit-only estimator clone/fit/predict or extra split call.
- If both sinks fail or the process is forcibly killed, delivery is not guaranteed; open operations/missing end/manifest must be interpreted as incomplete, never success.

## 7. Contract acceptance

Tests must assert depth/indentation, per-model summaries, ordered lifecycle, file precision/counts, explicit omissions, evidence path identity, missing-score honesty, preservation of existing call counts and safe failure behavior. Test normal/debug using identical inputs and compare normalized file evidence, excluding presentation flag, IDs, timestamps, durations and destination paths. The flag must not change which scientific records are retained.
