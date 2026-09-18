# Training Audit Guide

## Scope

The offline pipeline emits two complementary views without changing model logic:

1. **Human terminal output**: readable actions, counts and results. Normal mode shows levels 1–2; `--debuglog` adds levels 3–4.
2. **Complete structured evidence**: strict JSON Lines plus safe CSV tables under the run's log directory in both modes.

This is observation only. It does not change model selection, temporal folds, preprocessing, fit/predict calls, hyperparameter searches, final evaluation or Streamlit behavior. R² 0.85 is informational, not a retraining gate. Existing scientific differences are reported rather than silently repaired.

## Running Normal and Debug Modes

```bash
python pipeline.py                    # levels 1–2
python pipeline.py --debuglog         # levels 1–4
python pipeline.py --debuglog --no-heartbeat
```

`run_pipeline(...)` also accepts keyword-only `debuglog=False` and `on_audit_event=None`. Existing positional arguments and return fields remain compatible. The observer is presentation-only and cannot supply values to model logic.

### Four information levels

- **Level 1** (no indentation): stage/model action and real completion state.
- **Level 2** (two spaces): counts, model summaries, tuning progress, units and evidence paths.
- **Level 3** (four spaces, debug): fold/trial and operation detail.
- **Level 4** (six spaces, debug): full split, parameter, feature and artifact detail or a complete CSV link.

Warnings and errors remain visible in either mode. Raw JSON is never printed to the terminal. Output does not rely on color and remains readable when redirected.

## How to Find Logs

Each `run_pipeline()` execution owns unique paths:

```text
<workspace>/outputs/08_full_pipeline/logs/
  training-<UTC timestamp>-<audit_run_id>.logs
  training-<UTC timestamp>-<audit_run_id>/
    manifest.json
    *.csv
```

The `.logs` file is strict JSONL containing every depth regardless of `--debuglog`. The sibling manifest identifies the run, source fingerprint, training/evidence status and complete CSV exports. Existing logs are not overwritten.

Open a large table without rerunning training:

```python
import json
import pandas as pd
from pathlib import Path

log = next(Path("outputs/08_full_pipeline/logs").glob("training-*.logs"))
manifest = json.loads((log.with_suffix("") / "manifest.json").read_text())
for export in manifest["exports"]:
    if export["status"] == "complete":
        table = pd.read_csv(log.with_suffix("") / export["path"])
        print(export["kind"], table.shape)
```

Terminal table previews show no more than 20 data rows and state the total/omitted counts. CSV files retain full numerical precision.

## Log Event Contract

Every structured event retains `schema_version`, timestamp, audit/pipeline run IDs, sequence, severity `level`, event, message, step ID, operation and status. Additive fields include `detail_level`, stage/evaluation/trial/fold/split/dataset identities and evidence references.

Severity and information depth are different: a level-4 warning is still shown normally. Non-finite measurements become JSON `null`; unavailable measurements also carry a status/reason. Event fields cannot overwrite run identity or sequence fields.

Run status and evidence completeness are independent. A scientific run can complete while a file/terminal sink is incomplete; the manifest reports both. Missing stage completion is `UNKNOWN`/incomplete, never inferred PASS from an output filename. A forced process kill can leave open events and no manifest; that means incomplete evidence.

## Data, Feature and Count Evidence

The audit reports the actual source fingerprint, raw/clean/prepared/DEV/locked-test counts and their derivations. It lists the target, actual feature variants, blocked-feature reasons, fitted encoded-feature counts and preprocessing scope. It does not duplicate raw rows into new audit CSVs.

`skill_count` is described as the count of unique normalized tokens in `required_skills`. Cleaning count arithmetic follows the actual operation order: raw rows minus invalid-category rows, then duplicate removal, equals clean rows.

Branch A reports its DEV-fitted encoder, O1 R0–R4 representations, official O1 evaluation, O2 correlation-selected space, candidate evidence and final option selection at caller boundaries. It does not claim per-tree, solver-iteration or individual resample tracing.

## Fold Evidence

Fold definitions come from the same arrays consumed by model fitting; logging does not call the splitter again. Evidence includes:

- source and DEV content/order identity;
- requested/effective folds and stable chronological sort policy;
- zero-based DEV-relative train/validation positions and counts;
- actual time bounds, row overlap and shared calendar periods;
- a complete membership CSV.

The legacy positional digest remains labeled positions-only; a separate content/order digest distinguishes same-sized datasets. Shared months and shared rows are different facts. The logger records both and does not repair folds.

## Model Comparison, Fit Assessment and Tuning

Every executed salary candidate reports available validation MAE, RMSE and MedAE in USD, R² as unitless, population fold variability and the aggregation convention. Missing train or candidate locked-test scores are explicitly `not_computed`/`not_evaluated`.

Current code does not compute paired training scores and has no approved diagnostic rule for `overfitting`, `good fit` or `underfitting`. Therefore current candidate entries say:

```text
Fit assessment: insufficient evidence — training scores not computed;
diagnostic rule not defined.
```

A high R², CV–test difference or legacy narrative is not converted into a fit diagnosis.

Random Forest tuning records all actual groups and trials, seeds, effective parameters, fold results, ranking metric/direction and winners.

## Current Behavior Findings

1. Model-family comparison ranks by mean validation MAE.
2. Manual Random Forest tuning ranks by descending CV R².
3. Later tuning fixes `max_depth=20` where current code does so.
4. The final Random Forest is created from the first row of the provided initial tuning table (`provided_tuning_table_first_row`), not silently replaced with a later recommendation.
5. Only the selected final model is evaluated on locked test; other candidate test fields remain `not evaluated`.
6. Importance-only fold fits report `metrics_status=not_computed` because they do not predict validation rows.

These are traceable inherited behaviors, not endorsements or changes.

## Failures and Recovery

A training exception remains a training failure and is re-raised to the existing CLI exit handling. Cancellation remains distinct. File, terminal, observer and manifest failures are independently contained where possible and mark evidence incomplete; they never turn failed training into success or alter model decisions.

Unsafe workspace log destinations remain rejected. For ordinary evidence I/O failures, use the warning's path/reason, fix permissions or space, and rerun if a complete audit record is required. If both terminal and file sinks fail, delivery cannot be guaranteed.

## Historical Metrics

Historical pre-feature artifact values include Random Forest mean CV R² `0.8224307902351375` and selected locked-test R² `0.81269038067487`. These are historical artifact evidence, not a newly reproduced result merely because documentation changed. New-run claims must cite that run's workspace, JSONL and manifest.

## Validation Commands

```bash
.venv/bin/python -m pytest -q \
  tests/test_training_behavior.py tests/test_training_audit.py \
  tests/test_training_audit_integration.py tests/test_training_fold_audit.py \
  tests/test_training_console.py tests/test_training_stage_coverage.py \
  tests/test_training_evaluation_logs.py tests/test_training_evidence_exports.py \
  tests/test_training_cli.py tests/test_training_audit_docs.py
.venv/bin/python -m pytest -q
git diff --check
```

Real normal/debug validation must run in clean copied projects because the root CLI writes release outputs. Exact commands and parity exclusions are in `specs/002-readable-training-logs/quickstart.md`. Do not claim validation until commands, exits and asserted results are recorded in `verification.md`.

## Limitations

- Timestamps/durations are runtime facts and will differ across runs.
- Existing locked-test exposure cannot be made unseen by logging.
- The current experiment has documented candidate-test, nested-tuning and selection-policy governance differences; readable output does not resolve them.
- Fit labels remain unavailable without additional approved scientific measurements/rules.
- Abrupt process termination and simultaneous sink failure can prevent complete evidence.
- Branch A coverage is caller/result-table level, not every delegated estimator iteration.

## Source and Spec References

- `specs/002-readable-training-logs/spec.md`
- `specs/002-readable-training-logs/plan.md`
- `specs/002-readable-training-logs/coverage.md`
- `specs/002-readable-training-logs/contracts/terminal-and-evidence.md`
- Model comparison charts: `outputs/04_model_comparison/`
- Locked-test diagnostics and importance: `outputs/05_best_model/`
