# Quickstart: Training-Only Validation

**Status**: Commands are implemented and fixture-verified after explicit task approval. Do not run real training until eligible future data and its exact partition/exposure approval exist.

## 1. Review the scope

Read `spec.md`, `plan.md`, `research.md` and `contracts/training-evidence.md`. Confirm the proposed fixed two-feature pair, tie/diagnostic policies, bounded search and [runtime/logging protocol](contracts/runtime-observability.md). Clarification answers are already recorded; task approval is a separate gate.

- All tuning and CV: TRAIN only (~80%).
- Final evaluation: strictly later whole months (~19%), opened after freeze.
- Inference reserve: strictly later whole months (~1%), no evaluation and no predictions in this feature.
- Current UI and primary pipeline remain unchanged. Do not use `python pipeline.py` to exercise this feature.

## 2. After implementation: inspect existing data without fitting

Use the established virtual environment; do not install or update dependencies merely for review.

```bash
PYTHONPATH=src .venv/bin/python -m ai_job_market.training_validation inspect --workspace .
```

Expect JSON on stdout and a readable explanation on stderr. On the current historical snapshot, expect exit 3/readiness blocked: the latest month was already evaluated and cannot be an unseen reserve. Counts and ratio deviations are still reportable. This expected block is not a test failure or a reason to sample rows from March.

The current saved last-two-month allocation would be 895/306/298 (about 59.71/20.41/19.88), not an approved 80/19/1 result. No dates or target values are altered to force the percentages.

## 3. Provide eligible data and freeze approval

A data custodian must supply genuinely future prepared records via the existing preparation workflow outside this feature. Use an isolated workspace with the unchanged cleaned-raw schema. Inspect that workspace first.

Place an approval JSON inside that workspace with the exact source/policy/partition fingerprints and actual month boundaries/counts/shares from inspection, reviewer/time, exposure evidence and any predeclared business MAE/RMSE limits. See the contract for fields. Do not mark unknown lineage as unseen to pass a gate. No approval example with invented hashes or historical “fresh” rows is provided.

Actual ratios must be reviewed even if they differ substantially from the targets. Do not adjust boundaries based on salary metrics. If suitable future data remains unavailable, record real-run verification as blocked and rely on isolated fixture tests for implementation acceptance, not on invented real results.

## 4. Run only after task and partition approval

```bash
# Replace /absolute/eligible-workspace with a real, inspected local workspace.
PYTHONPATH=src .venv/bin/python -m ai_job_market.training_validation run \
  --workspace /absolute/eligible-workspace \
  --approval /absolute/eligible-workspace/training-approval.json
```

Expected stages: input/partition validation → five-model monthly CV → runtime and family decision → ablation/importance → conditional bounded nested RF tuning → paired full/Top-2 CV and final TRAIN fits → immutable evaluation lock → final holdout diagnostics → English report/publication.

A full RF-winning run has an upper bound of 540 fits before reuse. The added TRAIN-only runtime microbenchmark performs at most 1,784 predict calls and no extra fits; 30 measured calls per workload follow three warmups. This overhead is reported separately, not hidden inside model-fit speed. Keep production search bounds fixed; tiny test fixtures may inject test estimators/search bounds without exposing an unreviewed production override. No reserve fit, predict, score, uncertainty calculation or target export is permitted.

An interrupted holdout evaluation leaves an exposure record. Do not delete that record or change the run ID to retry as a pristine test. Review partial evidence and obtain a fresh future holdout when needed.

## 5. Validate and read results

Use the emitted run ID, not the placeholder below:

```bash
PYTHONPATH=src .venv/bin/python -m ai_job_market.training_validation check \
  --workspace /absolute/eligible-workspace --run-id tv-<32-lowercase-hex-characters>
```

Read `outputs/training_validation/<run_id>/report.md` and `manifest.json`. Download/parse CSV and JSON evidence offline. Read-only `check` must not deserialize model bundles, fit or predict. An unchanged repeated `run` reuses the validated complete pack without fitting or reopening final evaluation.

Additional reader views:

- Humans: `training.log` for chronological detail; `report.md` for explanations and conclusions.
- Agents: `agent_summary.json` + `metric_catalog.json`, with stable metric IDs, status/reason codes, evidence links and approval-aware next actions. No English parsing or automatic command execution.
- Future UI: `ui_summary.json` describes Page 04/05 KPIs/tables/downloads with raw values, units, source IDs and explicit unavailable states.
- Runtime audit: `runtime.csv`, `runtime_samples.csv`, `runtime_summary.csv`, `performance_comparison.csv`, `accuracy_runtime_tradeoff.csv` and `operational_assessment.json`.

For the fold-splitting request, open `fold_explanation.md` and `fold_summary.csv`: each fold must show training/validation months and exact row counts, its parent population, added history, later-unused rows and leakage checks. Reconcile with `monthly_row_counts.csv` and `fold_membership.csv`. Page 04's future snapshot includes the five outer folds; Page 05's includes parent-labelled inner tuning folds. These are mandatory details, not just download links without explanation.

Open `model_conclusions.json`: expect one conclusion/status for each of the five candidates plus separate final full and Top-2 roles. Each explains accuracy, fit/stability, runtime, selection/comparison reason, limitation and next action. Missing evidence must be stated, not replaced by a generic favorable conclusion.

Check that lowest CV MAE, selected family and fastest measured model are labelled separately. Inspect batch size/sample count/thread/hardware context before comparing speed. Do not treat process-lifetime RSS as per-model memory, first in-process load as cold server startup, or the local microbenchmark as live UI latency. Missing operational limits are not assessed; they do not default to passed. Actual budget values require approval rather than copying an invented SLA.

Existing Pages 04–06 still use their historical artifacts. Do not replace them manually or claim new exports are already visible in the UI.

## 6. Implementation verification commands

Proposed files must exist after implementation before running these tests:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_training_partitions_v2.py \
  tests/test_training_evidence_v2.py \
  tests/test_training_evaluation_v2.py \
  tests/test_training_search_v2.py \
  tests/test_training_runtime_v2.py \
  tests/test_training_report_v2.py \
  tests/test_training_validation_cli.py

PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check src/ai_job_market/training_*.py tests/test_training_*.py
 git diff --check
```

Tests must use temporary workspaces; no root pipeline or artifact-generating real-data test may mutate the checkout. Before broad pytest execution, inspect fixture/workspace isolation and report inherited failures separately. Small estimator fits on synthetic fixtures are permitted after implementation approval. Byte-hash compare protected source/preparation outputs/UI and existing bundles before/after testing. Preserve inherited `uv.lock`.

Required checks: strict month boundaries, nine-month readiness, five outer/three inner expanding folds, zero reserve access, target loader gate, undefined R², MAE/simplicity tie policy, bounded search/carried winners, holdout locking across changed run IDs/concurrency/crash, provenance rejection, paired variants, chart/evidence consistency, no-fit reuse and readable failure/conclusion output.

Also validate raw latency quantiles/throughput with fake clocks, fit/benchmark call bounds, thread comparability, no additional fits or protected targets, reuse timing attribution, OOF tail/baseline metric arithmetic and English/agent/UI numeric parity. Use a structured-data-only agent fixture to identify selected/fastest models, fold 3's exact months/counts, the reason for expanding history, every model's decision and safe next actions; do not replace the independent human comprehension check with that fixture. Real smoke tests assert schema/finite durations, not flaky millisecond performance thresholds.

Record exact commands/results in `specs/004-validate-training-split/verification.md`; do not mark pending real-data, reader-walkthrough or UI integration as passed. Graph refresh is required after code changes unless a documented exception applies.
