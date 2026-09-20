# Technical Research: English Training Step Logs

Date: 2026-09-18. Status: planning evidence, not implementation validation.

Assessment: `.specify/assessments/training-step-logs/research.md` (local, ignored by Git). This tracked document preserves the decisions and source findings needed for implementation.

## 1. Instrument existing operations, not a new training runner

**Decision**: Add a small standard-library audit module and surgical calls around existing operations in `src/ai_job_market/core.py`. Establish a scoped audit session at `run_pipeline`; use a context-local active session so nested core calls share run context without changing public function signatures or adding a logging argument through every helper. No active session means no-op audit calls for standalone helper use.

**Rationale**: The terminal runner already calls the core; compatibility facades re-export core functions. A CLI-only implementation would miss direct offline core executions. A thin session wrapper/decorator can own setup/teardown without reindenting the entire long pipeline body. Preserve callable signature, return values, and original exceptions.

**Alternatives considered**: Monkeypatching estimator `fit` globally obscures operation identity and risks scientific behavior; instrumenting only artifact saves misses in-progress failures; creating a new runner duplicates existing workflow.

Sources: `pipeline.py:main`, `src/ai_job_market/core.py:run_pipeline`, `src/training/temporal_cv.py`.

## 2. One event record, two destinations

**Decision**: Use standard-library logging with one English JSON object per line in both terminal and a UTF-8 `.logs` file. Include a short English `message` plus structured fields. Use an isolated logger/session, no root logger configuration, no propagation duplication. Keep existing stage/heartbeat output untouched.

**Rationale**: JSON lines remain readable text while allowing exact event-parity assertions. No dependencies or remote telemetry are needed. Flush each event. Record fold definitions once per unique split and reference them on every use.

**Alternatives considered**: Free-text-only records make membership comparison fragile; two independently formatted payloads risk loss of detail; dashboards, distributed tracing, and external collectors are out of scope.

## 3. Location, provenance, and lifecycle

**Decision**: Write under `<workspace>/outputs/08_full_pipeline/logs/training-<UTC timestamp>-<generated UUID>.logs`. Create exclusively; never overwrite or delete previous logs automatically. Generate `audit_run_id` independently, then attach the existing `pipeline_run_id` when it becomes available. Do not replace the existing pipeline ID or metadata schema.

**Rationale**: `outputs/` is already ignored and workspace isolation exists. No new CLI switch or user-supplied run ID is needed. Validate the generated log path remains in the selected workspace, reject symlinked log destinations, and preserve existing workspace behavior. Failure to log is explicitly reported to the terminal, not allowed to alter fitting decisions or suppress original exceptions.

Provenance includes input SHA-256, source/config identity, Git HEAD and dirty status, relevant package/runtime versions, seed, target/features, requested and resolved split policy. Capture only allowlisted fields; never environment dumps, dataset rows, arbitrary estimator reprs, or unsanitized exception payloads. A missing Git command produces explicit unavailable provenance, not training failure.

Sources: `src/ai_job_market/core.py:run_pipeline`, `.gitignore`, `pipeline.py:main`.

## 4. Fold identity and true operation boundaries

**Decision**: Inspect arrays returned by the real splitter, without calling it again for logging. Give every split a digest over development-dataset identity, ordering policy, and actual ordered memberships. Store zero-based positional membership relative to the identified development artifact. Record actual min/max periods for each partition, row intersection count, and shared months. Hash/reference the development row order so reset or duplicated DataFrame indices cannot be mistaken for identity.

**Rationale**: `temporal_split` resets indices; `temporal_cv_splits` sorts positional indices stably, uses 200-row blocks, and puts the remainder into the last validation fold. Do not change its small-input behavior or claim strictly disjoint months.

Combined `Pipeline.fit` is logged as `preprocess_and_fit`, not separate invented durations. Existing standalone preprocessor fits can be separately labeled. `random_forest_importance_by_fold` fits but does not score validation predictions: log importance extraction and metrics-not-computed, never add scoring for completeness. Fold R² population SD (`ddof=0`) may be added to logs only; preserve existing summary tables and ranking.

Sources: `core.py:temporal_split`, `temporal_cv_splits`, `evaluate_model_cv`, `random_forest_importance_by_fold`.

## 5. Audit actual decisions without correcting them

**Decision**: Log trial configurations, actual ranking metric, applied/fixed values, and final estimator parameters independently. Document these inherited observations:

1. Family comparison sorts `MAE_mean`; manual tuning sorts `CV_R2`.
2. Manual tuning fixes `best_d_val = 20`, not the top depth row.
3. `run_pipeline` sets `tuning = s1`, so `final_model_from_selection` consumes the initial grid winner, not the final staged winner.
4. Existing `tuning_rationale` strings include fixed values/claims; they are not newly measured audit facts.
5. Existing test transformation precedes final selection, and final evaluation includes permutation importance and reload predictions. Preserve that call order and distinguish transformation, scoring, diagnostics, and reload checks; do not claim “test accessed exactly once.”

**Rationale**: Correcting these would violate approved scope. Record them as inherited issues, not a new constitutional waiver. Maintainer disposition is an implementation gate.

Sources: `core.py:tune_random_forest_manual_steps`, `final_model_from_selection`, `finalize_salary_model`, `run_pipeline`; constitution VI.2/VI.4.

## 6. Verify semantic parity without overwriting release evidence

**Decision**: Characterize current behavior before implementation using fixed fixtures and call spies. Write failing logging tests next, then implement. Compare pre/post split arrays and parameters exactly; numeric prediction/metric comparisons use initial `rtol=1e-9, atol=1e-9` with justified deviations documented rather than silently widened. Ignore timings and generated run IDs in semantic comparisons. Compare deterministic model outputs rather than requiring byte-identical joblib bundles.

Use core `workspace_root` for an approved full offline validation. Exercise the actual CLI entrypoint in a temporary copied project, not against baseline `outputs/` and `artifacts/`. Tests load root `pipeline.py` by file location to avoid collision with the `src/pipeline/` package identified in `tests/conftest.py`. Bounded unit tests may use fake/lightweight estimators; production candidate searches stay unchanged. Real full-run validation must not mock the training path.

No full training or behavioral tests were executed during planning. Only source/artifact inspection and package metadata reads were performed.

## Observed environment and limits

- Local `.venv`: Python 3.13.12; NumPy 2.5.3; pandas 2.3.3; scikit-learn 1.9.1; pytest 9.1.1; joblib 1.6.0 (metadata queried 2026-09-18).
- Ruff targets Python 3.12. `requirements.txt` uses version ranges, not a reproducible lock; preserve the user's outstanding edit and capture installed versions during validation. This plan adds no dependency changes.
- Historical file values: RF mean CV R² 0.8224307902351375; selected locked-test R² 0.81269038067487. These are not new measurements and 0.85 is not a gate.
- Existing Graphify query identified core/facade dependencies but line numbers are stale. Source inspection takes precedence. Future code changes require refresh or documented justified alternative.
