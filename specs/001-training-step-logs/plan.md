# Implementation Plan: English Training Step Logs and Audit Evidence

**Branch**: `003-training-step-logs` | **Date**: 2026-09-18 | **Spec**: [spec.md](spec.md)  
**Input**: `specs/001-training-step-logs/spec.md`  
**Status**: Design complete for review; implementation blocked pending explicit approval and disposition of inherited governance findings.

## Summary

Add English, per-operation audit events to the existing offline salary-training path, mirrored to the terminal and unique `.logs` files. Record actual fold memberships, parameters, measured metrics, final applied choices, and failures. Add an evidence-backed English audit guide. Preserve every existing model decision, fit/predict call, output schema, CLI behavior, and UI boundary. R² 0.85 remains informational.

Use one small standard-library audit module, a scoped session around `run_pipeline`, and explicit events at existing operation boundaries in `core.py`. Do not create a replacement trainer or estimator wrapper. Existing `pipeline.py` stage/heartbeat output continues unchanged and receives new audit lines through the core call; production CLI edits are not anticipated.

## Technical Context

**Language/Version**: Python; local interpreter 3.13.12, Ruff target 3.12.  
**Primary Dependencies**: Existing pandas/NumPy/scikit-learn/joblib/PyYAML; only standard-library logging, JSON, hashing, UUID, context management, and metadata facilities added.  
**Storage**: Existing output/artifact packs unchanged; additive `outputs/08_full_pipeline/logs/training-<timestamp>-<uuid>.logs` in the selected workspace.  
**Testing**: pytest, capsys/caplog or captured streams, temporary workspaces, fixed-data characterization, estimator spies, actual offline integration after approval.  
**Target Platform**: Existing local Linux/Python offline workflow.  
**Project Type**: Importable ML core plus CLI and read-only reporting consumers.  
**Performance Goals**: No extra fit/predict/split calls; serialize membership once per unique split; report logging size/overhead during validation, without an invented speed SLA.  
**Constraints**: No dependency upgrades, training fixes, schema migrations, UI changes, new searches, metric gates, root logger mutation, or release artifact overwrites during validation.  
**Scale/Scope**: Existing salary evaluation path; historical development set is 1,201 rows and five folds. Do not hardcode these sizes. No per-tree internals or segmentation instrumentation expansion.

## Constitution Check

Pre-research review and post-design review have the same outcome: additive design respects the boundaries below, but inherited VI.2/VI.4 gaps remain an explicit implementation gate, not an assertion of full compliance.

| Gate | Design / evidence | Status |
|---|---|---|
| Data integrity | Preserve target `annual_salary_usd`, actual feature lists (including experimental ablations), existing latest/preferred locked period resolution, train-only fits, arrays and call order. Audit same-month boundaries rather than calling them strict separation. | Design pass; runtime proof pending |
| Reproducibility | Input/config hashes, code revision/dirty state, installed versions, seed, split membership and existing artifact references. No existing schema changes. | Design pass |
| Verification | Failing-first log tests and pre-change characterization; isolated core/CLI execution and semantic parity assertions. | Design pass; not executed |
| Streamlit | No UI code changes or new UI fitting path. Audit lifecycle applies to existing core invocations, without broadening who triggers training. | Design pass |
| Input validation | No new CLI inputs. Generated log basename, exclusive create, resolved containment in workspace, no symlinked log destination. Existing data/workspace validation preserved. | Design pass |
| Scientific/security claims | English allowlisted structured fields, no secrets/raw rows, CV/test labels, 0.85 informational, sanitized errors. No extra held-out scoring. | Design pass |
| Documentation | English audit guide, source findings, log schema, current scientific caveats, existing chart/artifact links, provenance and verification notes. | Design pass |
| Graphify/Karpathy | Prior graph query plus source verification; one small module and surgical calls; refresh after implementation. | Design pass |
| Constitution VI.2 / VI.4 | Current selected-model-only final reporting and R²-ranked RF tuning differ from stated candidate test reporting / MAE policy. | Inherited gaps; maintainer disposition required |

No exception may waive leakage safety, input validation, or evidence requirements. If approval requires changing model logic or additional locked-test evaluation, stop and revise scope rather than implement those changes under this feature.

## Project Structure

### Documentation (this feature)

```text
specs/001-training-step-logs/
├── spec.md
├── checklists/requirements.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/training-log.md
└── tasks.md
.specify/assessments/training-step-logs/research.md  # local assessment, Git-ignored
```

### Source Code (repository root)

```text
src/ai_job_market/
├── core.py                         # surgical instrumentation only
└── training_audit.py               # new session/events/safe dual output
pipeline.py                        # existing CLI, compatibility test only
src/training/temporal_cv.py         # unchanged re-export facade
src/pipeline/model_training_comparison.py  # unchanged facade
config/project.yaml                # unchanged
requirements.txt                   # untouched user edit

tests/
├── test_training_behavior.py       # pre-change characterization/parity
├── test_training_audit.py          # sink/lifecycle/failure/English contract
├── test_training_fold_audit.py     # membership/consistency/edge cases
├── test_training_audit_integration.py  # core + actual CLI compatibility
└── test_training_audit_docs.py     # required sections/valid references

docs/TRAINING_AUDIT.md              # new English operating/audit guide
README.md                          # short guide link only
```

**Structure Decision**: Preserve the shared core and facades. Session-owned audit state is context-local; never configure a process-global root logger or patch estimator behavior. No generalized event bus, new service, database, or packaged framework.

## Phase 0 — Research Outcomes

See [research.md](research.md). Source evidence confirms a shared splitter but not strict month separation. Existing terminal progress observes output writes, not every fit. Tuning depth is fixed at 20 in later stages, and final RF parameters are taken from the initial grid table. These are findings to expose, not repairs to perform.

## Phase 1 — Design

### Lifecycle and outputs

- A thin signature-preserving wrapper around `run_pipeline` owns an audit session; it resolves workspace from the existing arguments and closes only its own resources in `finally`.
- Generate an independent unique `audit_run_id`; record the original pipeline run ID later without changing it.
- Dedicated session logger emits the same complete JSON event to terminal and exclusive UTF-8 `.logs` file. Flush each event; disable propagation. Existing heartbeats remain presentation-only.
- No active session makes instrumentation no-op. A repeated/nested invocation restores the prior session and cannot leak handlers or state. No multiprocess logging is introduced; events surround existing parallel estimator calls in the parent process.
- Logging-only errors degrade visibly to available terminal output; status includes `log_complete=false`. Never swallow a training exception or turn failed training into success. Capture interruption context and re-raise to the existing CLI exit handling.

### Coverage inventory

| Existing operation | Event evidence | Scientific preservation check |
|---|---|---|
| Input/config load and prepared split | source/config identity, dimensions, seed, preferred/resolved period, partition references | No new cleaning, splitting, or held-out summaries |
| `evaluate_model_cv` used by ablations/comparison/tuning | model/features/parameters, split ref, fold start, combined preprocessing+fit, predict, score, aggregate | Same clone/fit/predict counts, arrays, returns and rankings |
| `encoded_correlations_train` and DEV readiness preprocessing | actual fit scope, shape, start/end; early test transform labeled transform-only | No new analysis or test scores |
| `random_forest_importance_by_fold` | actual split ref, fit and importance extraction | No added validation prediction or metrics |
| `tune_random_forest_manual_steps` | trial/group identities, candidate parameters, rankings, fixed values, stage results | Fixed depth, spaces, order and rationale strings untouched |
| Family/final selection | ranking basis, source table, final actual `get_params` allowlist | `s1` remains final input; do not substitute `s5` |
| `finalize_salary_model` | DEV fit, locked prediction/scoring, interval and importance diagnostics | Same existing diagnostic calls; no new test evaluation |
| Existing CSV/JSON/joblib writes and reload equivalence | scoped artifact path/type, start/end/error, reload delta | No output schema/path changes; no duplicate writes |
| Non-salary segmentation interval | top-level boundary if needed for context only | No new internal segmentation logging or computation |

Instrument core CSV/JSON save boundaries with a no-op-outside-session emitter if needed; they retain their signatures so the CLI's existing save wrappers continue to work. Wrap only current joblib write/load call sites, not joblib globally. Define stable step/model/trial IDs separately from display names so repeated ablation names cannot collide.

### Split and metric evidence

Membership positions reference the actual DEV row ordering and a dataset digest; complete definitions are emitted once per unique split. Every consumer emits a split reference. Shared-month and row intersections are distinct. Do not sort/reindex the training data for logging. R² mean and population SD are supplemental log summaries derived from computed fold results only; retain original `MAE_std` convention and output tables. Undefined values use `null` plus status, not nonstandard JSON NaN.

### Documentation

`docs/TRAINING_AUDIT.md` contains operation/function map, split illustration from actual evidence, historical metrics with citations, inherited issues, how to find/filter logs, field meanings, failure recovery, approved reproduction commands, and validation results. It links existing diagnostic/chart outputs where present, without regenerating them in this phase. New run evidence must cite its own workspace and not imply an unseen test or causality.

## Phase 2 — Verification and Delivery

1. Record approval and governance disposition before changes.
2. Characterize current split/selection behavior and call counts on fixed fixtures before instrumentation. Preserve expected outputs in tests/evidence, including fixed depth and initial-grid final selection.
3. Write failing audit contracts, implement the smallest session/emitter, then instrument one complete CV path (US1 MVP).
4. Complete executed training coverage and fold evidence (US2), then documentation (US3).
5. Test errors/cancellation, file initialization/write failure, repeated/nested sessions, duplicate indices, remainder/small datasets, non-finite metrics, skipped tuning, and no extra model calls.
6. Run the relevant and full pytest suites, then approved isolated offline core and actual CLI integration. Compare pre-change baseline evidence to instrumented results, excluding timing/IDs; retain release artifacts untouched. Initial numeric tolerance is `rtol=1e-9, atol=1e-9`.
7. Refresh Graphify after code verification or document a justified inability; update tasks/spec verification notes. Do not mark unexecuted validation complete.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Inherited VI.4 tuning policy mismatch | User explicitly requests observability without selection changes; disclose exact behavior. Maintainer disposition remains required before implementation. | Silently switching to MAE would alter the experiment and break scope/parity. |
| Inherited VI.2 candidate-level locked-test reporting gap | Keep the existing final-evaluation path and document its limitation; no blanket waiver is claimed. | Adding candidate test scoring expands held-out exposure and changes scope/call counts. |

## Risks and Review Decisions

- **Approval gate**: Confirm observation-only treatment of inherited findings; any required governance amendment is separate work, not implied by this plan.
- **Runtime overhead**: Detailed membership is sizable; deduplicate definitions and measure bytes/time. Do not suppress required evidence or claim a speed target without measurement.
- **Terminal coexistence**: Existing heartbeat threads can interleave presentation lines; complete audit lines and per-session sequence numbers must remain parseable.
- **Reproducibility**: Requirements are ranged, environment differs from historical artifact provenance, and test data has prior exposure. Report these limits rather than promising identical historic metrics.
- **Approval scope**: Plan/tasks approval can authorize bounded tests and isolated validation, never silent baseline artifact replacement or repeated tuning to reach 0.85.
