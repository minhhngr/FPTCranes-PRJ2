# Training Log Contract v1

Status: proposed contract; not implemented.

## Compatibility Surface

- Preserve `run_pipeline(raw_path, root=None, workspace_root=None)` and all existing core/facade signatures and return values.
- Preserve CLI arguments `--data`, `--heartbeat`, `--no-heartbeat` and exit statuses 0 (success), 1 (training error), 2 (preflight rejection), 130 (cancellation).
- Core logging starts when `run_pipeline` is invoked. A CLI preflight rejection before that point retains the existing English terminal error and creates no training log; do not call it a started training run.
- Do not alter existing metadata run IDs, result CSV/JSON schemas, model bundles, or stage summaries.
- New file: `<workspace>/outputs/08_full_pipeline/logs/training-<UTC timestamp>-<UUID>.logs`, exclusively created, UTF-8, newline-delimited JSON. No ANSI codes in owned audit events.
- Terminal and file receive identical audit event objects. Existing human progress/heartbeat text is outside this contract. Owned audit lines have `schema_version=1` and `audit_run_id`, allowing a reader to distinguish them from progress text.

## Event Envelope

Required on every owned event: `schema_version`, `timestamp`, `audit_run_id`, `sequence`, `level`, `event`, `message`, `step_id`, `operation`, `status`. `pipeline_run_id` may be null before the core creates it. English names/messages; numeric values retain machine precision rather than terminal rounding.

Allowed levels: INFO, WARNING, ERROR. Status: started, completed, skipped, failed, cancelled. An informational definition uses completed status. Required contextual fields are defined in [data-model.md](../data-model.md).

Illustrative shape only (values below are placeholders for shape, not measured evidence):

```json
{"schema_version":1,"timestamp":"2026-09-18T10:00:00Z","audit_run_id":"example-run","pipeline_run_id":null,"sequence":12,"level":"INFO","event":"operation_started","message":"Starting preprocessing and model fit.","step_id":"comparison.rf.fold1.fit","operation":"preprocess_and_fit","status":"started","phase":"comparison","model_id":"rf","trial_id":null,"split_id":"example-split","fold_id":1,"train_rows":200,"validation_rows":200}
```

## Required Event Families

| Family | Mandatory additional information |
|---|---|
| `run_started` | input identity/hash or explicit unavailable reason, configuration/environment, log location, `log_complete` |
| `run_context` | existing pipeline ID when known; seed/target/features and resolved partition identity |
| `split_defined` | dataset identity, actual policy, full ordered membership, bounds/counts, row and period overlap observations |
| `split_used` | phase/model/trial and split reference; never imply newly generated independent folds |
| `operation_started` / `operation_completed` | operation context; completion duration, shapes/counts/metrics where computed |
| `candidate_completed` | actual parameters, per-fold references, computed aggregate metrics, aggregate methods |
| `selection_recorded` | real ranking basis, source table/row, fixed values and actual applied estimator parameters |
| `operation_skipped` | existing branch condition/reason; no fabricated evaluation |
| `artifact_written` | successful existing artifact path/type, producing context and duration |
| `operation_failed` / `operation_cancelled` | last active context, sanitized error class/context, duration if available |
| `logging_degraded` | evidence failure category, `log_complete=false`, surviving destination |
| `run_completed` / `run_failed` / `run_cancelled` | final training state, duration, log completeness, evidence locations |

On artifact operations emit a start before the existing write and completion only after it succeeds; `artifact_written` is the terminal completion event for that step. Avoid duplicate terminal completion records for the same operation.

## Metrics and Fold Details

- MAE, RMSE, MedAE: USD; R²: unitless. Durations: seconds.
- Preserve existing CV aggregation; label additional R² SD as population SD over the actually computed folds. No new output columns or selection effects.
- Non-finite values serialize as `null` with a companion status; use strict JSON, not NaN/Infinity tokens.
- Fold position lists identify rows in the recorded DEV ordering. A digest alone does not replace the first full membership definition. Later trials may reference that definition.
- Importance-only fits report `metrics_status=not_computed`; never add prediction calls for audit completeness.
- `preprocess_and_fit` reflects a combined sklearn pipeline call. Do not invent preprocessing-only or per-tree timings.
- Existing locked-test transforms, final metrics, permutation-importance diagnostics and reload predictions have distinct labels. “One final metric report” does not mean “one test access.”

## Safety and Failure

Generate IDs internally. Resolve log location under the existing workspace and reject symlinked log destinations/escape before creating files. Exclusive creation prevents same-second overwrites. No root logger changes, duplicated handlers, or lingering file handles after invocation. Flush each complete line; never log secrets, raw data, environment dumps, or arbitrary estimator representations.

File initialization/write failures produce an English terminal `logging_degraded` warning where possible and mark incomplete evidence. They must not replace the original training exception or change estimator operations. A broken terminal cannot be guaranteed to show a warning; retain the surviving file event and do not recursively retry failed sinks indefinitely. Process kill/power loss may leave no final event; docs must explain this limitation.

## Acceptance Checks

Tests must reconcile event objects across destinations, verify lifecycle/order/context and English owned templates, compare fold definitions to consumed arrays, prove no handler/resource leaks, induce file failures/cancellation, preserve CLI statuses, and assert pre/post scientific equivalence. These assertions—not visual inspection of a progress bar—establish compliance.
