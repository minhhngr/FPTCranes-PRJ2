# Verification Evidence: English Training Step Logs

Status: implementation approved by user on 2026-09-18.

## Approval and Governance Disposition

- User approved option 1: implement logging and isolated validation.
- Scope remains observation-only: preserve existing training, fold splitting, model selection, tuning order and output schemas.
- Inherited findings are documented, not waived or fixed:
  - Model family comparison currently ranks by mean MAE.
  - Manual Random Forest tuning currently ranks candidates by CV R².
  - Manual tuning fixes depth at 20 in later stages.
  - Final Random Forest estimator currently consumes the initial grid table, not the final staged tuning table.
  - Existing final evaluation reports the selected model on locked test; this feature does not add candidate-level locked-test scoring.
- R² = 0.85 remains informational only. No task may retrain repeatedly or alter fold/model logic to reach that value.
- Isolated validation runs are authorized if they do not overwrite root `outputs/` or `artifacts/`.

## Baseline Notes

- Outstanding user edit: `requirements.txt` is already modified before implementation and is preserved.
- No implementation tests or training runs have been completed yet.

## T002 Source/Environment Evidence

Graphify query: `run_pipeline evaluate_model_cv temporal_cv_splits tune_random_forest_manual_steps final_model_from_selection pipeline TerminalProgress`. Source inspection still takes precedence over stale graph line numbers.

```text
Traversal: BFS depth=2 | Start: ['run_pipeline()', 'TerminalProgress', 'evaluate_model_cv()', 'final_model_from_selection()', 'Pipeline', 'temporal_cv_splits()', 'tune_random_forest_manual_steps()'] | 144 nodes found

[!] TRUNCATED: showing 37 of 144 nodes (~1000-token budget). The answer may be among the 107 cut nodes — raise the token budget (CLI: --budget) or narrow the query (e.g. context_filter=['call'], or get_node for a specific symbol).

NODE run_pipeline() [src=src/ai_job_market/core.py loc=L2235 community=core.py]
NODE TerminalProgress [src=pipeline.py loc=L416 community=pipeline.py]
NODE evaluate_model_cv() [src=src/ai_job_market/core.py loc=L561 community=core.py]
NODE final_model_from_selection() [src=src/ai_job_market/core.py loc=L2150 community=core.py]
NODE Pipeline [src= loc= community=core.py]
NODE temporal_cv_splits() [src=src/ai_job_market/core.py loc=L513 community=core.py]
NODE tune_random_forest_manual_steps() [src=src/ai_job_market/core.py loc=L2075 community=core.py]
NODE core.py [src=src/ai_job_market/core.py loc=L1 community=core.py]
NODE pipeline.py [src=pipeline.py loc=L1 community=pipeline.py]
NODE DataFrame [src= loc= community=core.py]
NODE run_segmentation() [src=src/ai_job_market/core.py loc=L1122 community=run_segmentation]
NODE Any [src= loc= community=core.py]
NODE main() [src=pipeline.py loc=L729 community=pipeline.py]
NODE finalize_salary_model() [src=src/ai_job_market/core.py loc=L2170 community=core.py]
NODE _print_final_summary() [src=pipeline.py loc=L657 community=pipeline.py]
NODE Path [src= loc= community=core.py]
NODE make_model_pipeline() [src=src/ai_job_market/core.py loc=L509 community=core.py]
NODE random_forest_importance_by_fold() [src=src/ai_job_market/core.py loc=L637 community=core.py]
NODE data_source.py [src=src/c
```

Installed/runtime versions observed before code instrumentation:

```text
python=3.13.12
numpy=2.5.3
pandas=2.3.3
scikit-learn=1.9.1
pytest=9.1.1
joblib=1.6.0
```

Baseline file hashes:

```text
7fa4b10fcecfbdd23823c0b0234786b67e0d7f28097c0e8d7faac41361c5cc45  data/raw/ai_jobs_market_2025_2026.csv
da5d23253b328f2594d345509c48931f5993a3d503588c10690e95f323f116a3  config/project.yaml
7277a1d897362acfeb660339e13e89f2901c74f0fe923e970a2120ba3cad4d82  src/ai_job_market/core.py
1eaf6475927312ef942c22eb37fe3cb2d2ed7eabe7ee7cc08467cf2c2374b944  pipeline.py
```

## T003-T004 Baseline Characterization

- Added `tests/test_training_behavior.py` to preserve current split, fit/predict call count, R²-ranked tuning and final estimator table behavior.
- First draft exposed current small-input `evaluate_model_cv` aggregation failure when no default 200-row fold exists; this is recorded as existing behavior, not fixed.
- Verification command: `.venv/bin/python -m pytest tests/test_training_behavior.py -q` -> 4 passed.

## US1 Partial Implementation Evidence

- Added `src/ai_job_market/training_audit.py` with scoped JSON-line audit sessions, exclusive `.logs` files, terminal mirroring, safe log-directory checks and no-op emission outside an active session.
- Wrapped `core.run_pipeline` without changing its public signature or return value; the original implementation is now `_run_pipeline_impl`.
- Added audit events for `evaluate_model_cv` split definition, combined preprocessing/model fit, validation prediction and fold metrics.
- Added CSV/JSON artifact write start/completion audit events while preserving existing `save_csv` and `save_json` signatures for terminal progress hooks.
- Verification command: `.venv/bin/python -m pytest tests/test_training_behavior.py tests/test_training_audit.py tests/test_training_audit_integration.py -q` -> 9 passed.
- Remaining US1 gaps: tuning/final-selection-specific summary events, final DEV fit/test diagnostics labels, cancellation/write-failure depth, and full operation inventory are not yet complete.

## Current Verification Status

- `.venv/bin/python -m pytest tests/test_training_behavior.py tests/test_training_audit.py tests/test_training_audit_integration.py -q` -> 9 passed after latest code changes.
- `git diff --check` -> passed.
- `.venv/bin/python -m pytest tests/test_release_contract.py -q` -> 15 passed, 1 failed because existing `outputs/validation_report.json` is missing. No artifact was generated to hide this pre-existing validation gap.

## T009-T011 and T014-T018 Evidence

- Added audit events for importance-only Random Forest fold fits, manual Random Forest tuning stages, final estimator selection, final development fit, locked-test prediction/scoring, permutation importance and encoded importance extraction.
- Added fold audit payload with `split_id`, dataset row-order digest, sort policy, ordered row positions, period bounds, row-overlap count and shared periods.
- Verification command: `.venv/bin/python -m pytest tests/test_training_fold_audit.py tests/test_training_behavior.py tests/test_training_audit.py tests/test_training_audit_integration.py tests/test_release_contract.py -q` -> 27 passed.
- T012 remains open for deeper degraded-write/cancellation coverage.

## T012 and T019-T021 Evidence

- Added degraded file-sink and cancellation tests for the audit session; original exceptions remain visible to callers.
- Added `docs/TRAINING_AUDIT.md` and README link with English scope, log contract, fold evidence, current behavior findings, historical metrics, validation commands and limitations.
- Verification commands: `.venv/bin/python -m pytest tests/test_training_audit.py -q` -> 5 passed; `.venv/bin/python -m pytest tests/test_training_audit_docs.py -q` -> 2 passed.

## T022 Full Test Evidence

- Verification command: `.venv/bin/python -m pytest -q` -> 44 passed.
- Release contract is included in the full suite and now passes after the user fix.

## T023 Isolated Runtime Validation Attempt

- Attempted approved isolated core run in `/tmp/training-audit-core-faz29a5q`; root `outputs/` and `artifacts/` were not targeted.
- Command timed out after 900 seconds before completion. Last captured audit event was around `08_training_readiness.json`; no `run_completed` evidence was produced.
- Log path started: `/tmp/training-audit-core-faz29a5q/outputs/08_full_pipeline/logs/training-20260917T183801Z-43c759c59da84b1e8b85496065bab194.logs`.
- T023 remains open because full isolated run/CLI parity evidence is incomplete.

## T023 Completed Runtime Validation

- Core isolated run completed in `/tmp/training-audit-core-sle163mb`; root `outputs/` and `artifacts/` were not targeted.
- Core log: `/tmp/training-audit-core-sle163mb/outputs/08_full_pipeline/logs/training-20260918T023333Z-31f78221d02d4309b2b1a6ad33e2040f.logs`.
- CLI copied-project run completed in `/tmp/training-audit-cli-88t5spoe` with exit status 0; root `outputs/` and `artifacts/` were not targeted.
- CLI log: `/tmp/training-audit-cli-88t5spoe/outputs/08_full_pipeline/logs/training-20260918T025236Z-c39c026574d0474da8b659129a5503fa.logs`.
- Both logs parsed as strict JSON, each with 1,310 events, `run_started` first, `run_completed` last, one full `split_defined`, 39 `split_used`, and 195 `candidate_fold_scored` events.
- Core log size: 769,461 bytes. CLI log size: 769,193 bytes. CLI terminal capture size: 778,295 bytes.
- Runtime evidence: `best_model=Random Forest`, `locked_test_r2=0.81269038067487`, `locked_test_mae=14735.12606698571`. These match historical locked-test metrics and do not meet or target 0.85.
- CLI terminal summary reported total elapsed `17:23.8`. No repeated training was performed to chase scores.

## T024 Final Review Evidence

- Verification command: `.venv/bin/python -m pytest -q` -> 45 passed.
- `git diff --check` -> passed.
- Source/consumer review: existing core return signatures, CLI arguments, output artifact schemas and Streamlit boundary are preserved. New audit logging is additive and writes to workspace `outputs/08_full_pipeline/logs/*.logs`.
- Secret/raw-data scan found only expected code identifiers such as `contextvars.Token`, skill tokens and pre-existing artifact names; no new credential logging or raw dataset dumps were identified.
- Runtime docs updated in `docs/TRAINING_AUDIT.md` with isolated core/CLI paths, event counts, locked-test metrics and elapsed time.

## T025 Graphify and Spec Sync Evidence

- `graphify update .` completed: rebuilt 614 nodes, 1110 edges and 39 communities; `graphify-out/graph.json`, `graphify-out/graph.html` and `graphify-out/GRAPH_REPORT.md` updated.
- Graphify warning: 3 source files produced zero nodes (`RELEASE_MANIFEST.json`, `FPTCranes-PRJ2-main_branch_b_manifest.json`, `pyproject.toml`) and community labels may need refresh. This does not block code verification.
- Spec, plan, tasks, docs and verification notes reflect the delivered additive audit logging implementation.
