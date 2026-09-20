# Verification Record: Readable Training Logs

## Approval and scientific disposition

- **Approval received**: 2026-09-18. The user explicitly requested implementation of the approved spec/plan/tasks, granted required local execution permissions, and required preservation of all training logic.
- **Disposition of inherited VI.2/VI.3/VI.4 findings**: observation-only implementation was authorized. Existing selected-model-only locked-test evaluation, non-nested manual tuning, R²-ranked tuning, fixed-depth behavior and s1 final parameter source are retained and exposed honestly. This is a bounded implementation decision, **not** a declaration that inherited behavior complies with Constitution VI.2–VI.4 and not a waiver of leakage, validation or evidence requirements.
- No data, split, preprocessing, fit/predict call count, search space, ranking or selection behavior was intentionally changed.
- No login or external credential was required.

## Planning baseline

- Branch: `004-readable-training-logs`
- Planning HEAD: `4a02508dc88f6c9abe52360e0b12a89e5975d4d7`
- The working tree contained inherited uncommitted training-audit work. It was preserved; no unrelated changes were reverted or committed.
- Optional `before_implement` and `after_implement` `speckit.git.commit` hooks were deliberately skipped because a commit would mix inherited user changes with this feature. Required hooks: none.

## Isolated pre-change baseline (T002)

- Path: `/tmp/readable-training-baseline-WvwyEv`
- Command: copied working-tree `src/`, `config/`, `data/`, `pipeline.py`, `requirements.txt`, `pyproject.toml`; ran copied `pipeline.py --no-heartbeat` with the project venv and copied source on `PYTHONPATH`.
- Exit: 0; elapsed: 1,037 seconds; 129 output files; 10 artifacts.
- Source SHA-256: `7fa4b10fcecfbdd23823c0b0234786b67e0d7f28097c0e8d7faac41361c5cc45`.
- Config SHA-256: `da5d23253b328f2594d345509c48931f5993a3d503588c10690e95f323f116a3`.
- Result: raw 1,500×25; clean 1,499×24; DEV 1,201; locked test 298; Random Forest; MAE 14735.12606698571, RMSE 29111.29612088309, R² 0.81269038067487, MedAE 4346.63809712512, empirical absolute-error q90 40067.74765859205.
- Audit baseline: 1,310 JSON events, 770,743 bytes. Terminal included raw JSON, confirming the requested behavior gap.
- Root `outputs/` and `artifacts/` were not used by the copied process.

## Test-first and static evidence

Tests were added before or with each implementation slice for console depth/filtering, strict JSONL, manifests/CSVs, file and observer failures, stage lifecycle, data identity, split membership, evaluation/fit evidence, CLI compatibility and scientific call-count parity.

Final commands:

```bash
.venv/bin/python -m pytest -q
uvx ruff check pipeline.py src/ai_job_market/core.py \
  src/ai_job_market/training_audit.py src/ai_job_market/training_console.py \
  tests/test_training_*.py
git diff --check
.venv/bin/python -m compileall -q pipeline.py src/ai_job_market tests
```

Results:

- pytest: **69 passed**.
- Explicit release/segmentation subset (`test_release_contract.py`, `test_segmentation_v3.py`, `test_segmentation_representation_robustness.py`): **29 passed**.
- Ruff: **all checks passed**. Ruff was executed through `uvx`; it was not added as a project dependency.
- `git diff --check`: pass.
- `compileall`: pass.
- Targeted scientific parity tests prove normal/debug observer modes do not change fit/predict/split call counts or returned metrics.
- CLI error-path tests prove preflight=2, training failure=1 and cancellation=130.

## Final isolated runtime evidence

All final runs used copied workspaces or an explicit non-root workspace. Root release artifacts were not overwritten.

### Normal CLI

- Path: `/tmp/readable-training-final-WzjUJw`
- Command equivalent: copied project inputs/source, then `pipeline.py --no-heartbeat` with copied source on `PYTHONPATH`.
- Exit 0; elapsed 1,162 seconds.
- 12/12 real lifecycle stages recorded `PASS` in `pipeline_terminal_timing.csv`; no filename-based inferred PASS.
- 1,608 strict JSONL events; final event `run_completed`.
- 90 complete CSV exports; every manifest row count reconciled to its CSV.
- Manifest: `training_status=completed`, `coverage_complete=true`, `log_complete=true`.
- Terminal: 22,891 bytes, zero JSON-looking lines, no fold/split level-3/4 detail.

### Debug CLI

- Path: `/tmp/readable-training-final-debug-4Kc54N`
- Command equivalent: copied project inputs/source, then `pipeline.py --debuglog --no-heartbeat` with copied source on `PYTHONPATH`.
- Exit 0; elapsed 1,186 seconds.
- 12/12 lifecycle stages `PASS`.
- Same 1,608 complete JSONL events and 90 reconciled CSV exports as normal mode.
- Manifest: `training_status=completed`, `coverage_complete=true`, `log_complete=true`.
- Terminal: 136,017 bytes, zero JSON-looking lines, with fold and temporal-split level-3/4 detail.
- Normal/debug dataset and split identities match. Both contain exactly 26 completed RF tuning trials.

### Direct core call without observer

- Path: `/tmp/readable-training-core-I99403`
- Call: `run_pipeline(raw_path, root=project, workspace_root=isolated_path, debuglog=False)` with no CLI observer.
- Exit 0; elapsed 1,216 seconds.
- 1,608 events, 90 exports, complete manifest and no JSON terminal output.
- Confirms core-owned audit lifecycle works independently of `pipeline.py`.

A discarded parallel normal/debug attempt exceeded the 2,400-second harness timeout because both CPU-heavy pipelines contended for resources. It was not counted as validation evidence; both processes were gone and their incomplete records were left only under `/tmp`.

## Scientific parity evidence

Baseline, final normal and final debug were compared with `rtol=1e-9`, `atol=1e-9` after excluding runtime-only columns/metadata:

- raw/clean/DEV/locked-test counts: identical;
- split definitions and dataset identities: identical between final normal/debug;
- model family ordering and numerical metrics: identical;
- all tuning rows, parameters and numerical rankings: identical;
- selected model and final parameter source: identical;
- locked-test metrics and empirical q90: identical;
- final locked-test metric CSV SHA-256 in all runs: `cb8e2ea323bdcf1ce6c5829f38525bba405549997a18104a5b527ef299f8cb9c`;
- `model_bundle.joblib` SHA-256 in baseline/normal/debug: `53b99e3d472a8d8627fc88eadc3b911d08304eafbf3e1d56704e404553f7830a`;
- `preprocessor_ml_ready.joblib` SHA-256 in baseline/normal/debug: `116885d9ce9090da4eecb30bd8042c16838391ffd639db5c8b5c611aa2ffd078`.

Runtime timing columns and run timestamps were intentionally excluded. Cluster bundle bytes differ because existing run metadata includes runtime identity; its selected representation, algorithm, K, silhouette and stability values were compared semantically and match.

## Coverage and failure evidence

- All planned C1–F stages emit real start/completion events; `coverage_complete` requires all 12 completion IDs.
- Unseen CLI stages finalize as `UNKNOWN`, never fabricated `PASS`.
- CV membership uses the exact consumed split arrays; membership exports and manifest counts reconcile.
- Model candidates explicitly report train metrics `not_computed`, candidate locked test `not_evaluated`, and fit assessment `insufficient_evidence` with reason.
- Importance-only CV fits explicitly report `metrics_status=not_computed`.
- Final runs include six joblib writes, one joblib read, one serialization-equivalence prediction and one locked-test segment assignment, each with started/completed events.
- Audit file, observer and training exceptions have regression coverage. Audit delivery failures mark evidence incomplete without changing model logic; training exceptions remain failures and cancellation remains distinct.

## Review disposition

A five-axis review (correctness, readability, architecture, security/path safety, performance) found and resolved two blocking implementation issues:

1. Three CLI stage constants used Unicode en dashes while core emitted ASCII IDs, causing `UNKNOWN` timing rows. IDs are now stable ASCII and a 12-stage regression test enforces the contract.
2. JSONL and evidence-directory initialization were coupled, and manifest writing preceded JSONL close. The sinks are now initialized independently and the manifest is written after close so evidence status reflects close failures.

No new dependency was added. CSV destinations are constrained beneath the owned evidence directory, names are sanitized, structured writes are locked, and terminal tables remain bounded.

## Five-minute reviewer exercise and success criteria

Using the final debug transcript and its sibling manifest/CSVs, a reviewer located source identity/hash (`run_started`), raw/clean/DEV/test counts (`data_summary`, `preprocessing_summary`), fold membership (`fold_membership` CSV), model metrics/units (`candidate_metrics` and `model_comparison` CSVs), all tuning settings/winners (`tuning_trials` CSVs), applied final settings (`selection_recorded`) and selected locked-test metrics in under five minutes. No rerun, raw JSON terminal parsing or source-code inspection was needed.

- **SC-001**: normal/debug transcripts and console tests prove four stable levels with two-space indentation and labeled states.
- **SC-002**: all large-table records report `shown_rows <= 20`, total/omitted counts and complete CSV paths; manifest reconciliation passed for 90 exports per run.
- **SC-003**: five model candidates and 26 RF trials reconcile to full-precision evidence; missing train/candidate-test evidence is explicit, never invented.
- **SC-004**: reviewer exercise above passed.
- **SC-005**: baseline/final metric, split, ranking, selection, call-count and deterministic bundle parity passed at declared tolerance.
- **SC-006**: tested training, cancellation, sink, observer and incomplete-stage paths preserve honest, distinct states.

## Graphify refresh

- `graphify . --update --no-viz` correctly detected 51 changed code files and 44 changed semantic files, but stopped because no LLM API key was configured. No login/key was requested.
- `graphify . --update --code-only --no-viz` then succeeded: 51 code files re-extracted; final graph had 679 nodes, 1,268 edges and 42 communities. It warned that three non-code/config files produced zero nodes.
- `graphify cluster-only .` succeeded and refreshed `GRAPH_REPORT.md`, `graph.json` and `graph.html`; community-set drift was reported honestly (39 saved labels vs 42 communities).
- A bounded query found `TrainingAuditSession`, `run_pipeline()`, `_run_pipeline_impl()`, `start_training_audit()`, `active_audit()` and `ConsoleRenderer` at their final source locations, confirming the new code is indexed. Documentation semantic refresh remains unavailable without an already-configured supported key.

## Final requirement evidence

- Normal mode presents levels 1–2; debug adds levels 3–4. Warnings/errors bypass depth filtering.
- Raw JSON remains file-only; complete JSONL/CSV evidence exists in both modes.
- Source hash/path, cleaning arithmetic, feature governance, DEV/test counts, temporal folds, metrics/units, all tuning trials, selections, final evaluation, Branch A caller-level evidence and artifact lifecycle are present.
- Terminal previews report at most 20 rows and omitted totals; full-precision CSVs and manifest references retain the complete tables.
- README and `docs/TRAINING_AUDIT.md` document usage, semantics, current inherited behavior, historical-versus-reproduced claims, limitations and recovery.
- Validation evidence above satisfies the quickstart gates without changing scientific behavior.
- Final reconciliation: 10 required feature artifacts present, 35/35 task IDs checked, referenced concrete paths and `.specify/feature.json` pointer valid.
- Final `git status` showed no staged files. Existing inherited modifications/untracked files remain in place; no root output/artifact path appeared as changed.
