# Tasks: Readable Training Logs and Evaluation Evidence

**Input**: Design documents in `specs/002-readable-training-logs/`, including `spec.md`, `plan.md`, `research.md`, `coverage.md`, `data-model.md`, `contracts/terminal-and-evidence.md`, `quickstart.md`; post-plan research at `.specify/assessments/readable-training-logs/research.md`.  
**Status**: 35 prepared tasks; none executed. User authorized automatic preparation through tasks, not implementation.  
**Blocking gate**: T001 must close before source changes or training runs. No security expansion is scheduled; preserve existing validation and minimal file safety.

## Format

`- [X] Tnnn [P?] [US?] Description with exact file path`

[P] means independent files within the stated dependency phase. Tests must fail for the intended missing behavior before implementation; capture expected failure and later pass in `verification.md`. Do not weaken scientific characterization to make presentation tests pass.

## Phase 1: Setup and Approval Gate

- [X] T001 Resolve the documented constitution VI.2/VI.3/VI.4 and historical-exposure findings, and record explicit user approval of implementation tasks in `specs/002-readable-training-logs/plan.md` and `specs/002-readable-training-logs/verification.md`; if scientific behavior must change, STOP and revise `specs/002-readable-training-logs/spec.md` before proceeding. Acceptance: named disposition and approval, not an assumed waiver; no leakage/evidence/validation waiver.
- [X] T002 Preserve the inherited dirty working-tree baseline and refresh source/Graphify coverage against `specs/002-readable-training-logs/research.md` and `specs/002-readable-training-logs/coverage.md`; run the isolated pre-change workflow from `specs/002-readable-training-logs/quickstart.md` only after T001, recording source hashes, command, exit, source data/config identity, semantic artifact evidence and source/output snapshot locations in `specs/002-readable-training-logs/verification.md`. Acceptance: baseline includes uncommitted audit code, not HEAD alone; root outputs untouched.

**Checkpoint**: Approval and reproducible baseline exist. Differences from the researched tree must be reconciled before tests or code edits.

## Phase 2: Foundational Observation and Export Primitives

- [X] T003 Extend `tests/test_training_behavior.py` with fixed-input no-session/normal/debug parity spies for fit, predict, split inputs/counts and actual feature/parameter selection; retain current RF R²/fixed-depth/s1 characterization and add scope-aware baseline fixtures. Acceptance: new debug path fails before implementation, existing scientific expectations stay unchanged. Depends on T002.
- [X] T004 [P] Write sink/lifecycle regression tests in `tests/test_training_audit.py` for independent file/console writes, initialization/flush/close failures, strict JSON/non-finite reasons, exception preservation and nested/repeated session cleanup. Move the old stdout JSON equality expectation to explicit file contract plus human-output assertions. Acceptance: broken console does not prevent file writes; broken file does not hide training failure. Depends on T002.
- [X] T005 [P] Write event-context and stage-coverage fixtures in `tests/test_training_stage_coverage.py` for unique run/evaluation/trial/operation IDs, protected envelope fields, paired starts/ends, failure/cancellation propagation and no-op outside session. Acceptance: no basename/display-name collisions or synthetic completed stages. Depends on T002.
- [X] T006 Extend `src/ai_job_market/training_audit.py` with optional depth/context metadata, scoped lifecycle bookkeeping, separately guarded sinks/observer, guaranteed context restoration and best-effort completeness reporting; retain the existing file envelope and no-active-session behavior. Acceptance: T004–T005 pass; do not wrap estimators or swallow training exceptions. Depends on T004–T005.
- [X] T007 Write minimal table/export contract tests in `tests/test_training_evidence_exports.py` for UTF-8 quoting, precision, 0/20/21 rows, complete save before path announcement, same-run identity and safe generated names. Acceptance: truncation affects presentation only; existing containment tests remain. Depends on T006.
- [X] T008 Add the minimal session-owned table export/preview primitive to `src/ai_job_market/training_audit.py` without routing its writes through `core.save_csv`; retain one table/preview at a time and metadata only in the session. Acceptance: T007 passes, no recursive audit and no overwrite of another run. Depends on T007.

**Verification checkpoint**: Run `tests/test_training_audit.py`, `tests/test_training_stage_coverage.py`, `tests/test_training_evidence_exports.py`; document expected pending higher-level failures separately. Foundations are not completion of user stories.

## Phase 3: User Story 1 — Follow Training Without Raw Logs (P1, MVP)

**Independent acceptance**: Normal-mode fixture run exposes every executed workflow stage's action/status and applicable numbers, no JSON payloads, and no false PASS. Tables use the export primitive if needed.

### Test-first implementation slices

- [X] T009 [P] Add `tests/test_training_console.py` covering depth 1–4 formatting, 0/2/4/6-space indentation, normal filtering, unconditional warning/error visibility, missing-value labels, table limits, units, no raw dict/JSON fallback, narrow/no-color/redirected output and synchronized multiline writes. Acceptance: tests fail against current raw JSON console. Depends on T008.
- [X] T010 Implement `src/ai_job_market/training_console.py` and connect it in `src/ai_job_market/training_audit.py`: explicit event/depth formatters, readable fallback, bounded tables and shared locked output blocks; structured payloads remain file-only even on degraded paths. Acceptance: T009 passes, severity is not confused with depth, no new package. Depends on T009.
- [X] T011 [P] Add `tests/test_training_cli.py` and extend `tests/test_training_stage_coverage.py` with parser/preflight/heartbeat/observer and 12-stage execution fixtures, including Branch A caller substages, missing marker/events, failure after an artifact write, cancellation and failing equivalence evidence. Acceptance: missing observations never become PASS; existing exit codes preserved. Depends on T008; independent of T009–T010.
- [X] T012 Add keyword-only `debuglog`/`on_audit_event` plumbing to `src/ai_job_market/core.py` and `--debuglog` plus event-driven `TerminalProgress` integration to `pipeline.py`; retire marker hooks from the main path and stop `finalize_unobserved` fabricating PASS. Preserve callable returns, other CLI flags, observer-independent direct-core behavior, heartbeat/timing filenames and actual training exit semantics. Acceptance: parser/progress portions of T011 pass. Depends on T010–T011.
- [X] T013 Add real C1–C5, B1–B3, B4, B5–B6, B7, I1 and F stage boundaries in `src/ai_job_market/core.py`; emit safe source/config/version/count/feature-policy summaries from existing values and sequential cleaning arithmetic without recomputing preprocessing/splits. Acceptance: T011 stage fixtures match `specs/002-readable-training-logs/coverage.md`, including failures before/after writes. Depends on T012.
- [X] T014 Add A1–A8 stage/substage observations at existing call boundaries in `src/ai_job_market/core.py:run_segmentation` and its caller: encoder, representations, each evaluation invocation, O1/O2 choice, evidence construction and cluster-bundle completion. Report actual returned dimensions/metrics and bounded table references; leave helper algorithms and per-resample loops untouched. Acceptance: Branch A tests from T011 pass, no added algorithm calls or salary-fit claims. Depends on T013.
- [X] T015 Verify US1's complete normal-mode path with `tests/test_training_console.py`, `tests/test_training_cli.py` and `tests/test_training_stage_coverage.py`; record each covered stage, remaining unsupported internal-trace boundary and zero raw-event dumps in `specs/002-readable-training-logs/verification.md`. Acceptance: every expected executed stage has a real terminal status; no MVP completion claim based only on a final banner. Depends on T014.

**Checkpoint**: Default human-readable training is independently demonstrable. Candidate/trial scientific-detail completion remains US3, not implied by this checkpoint.

## Phase 4: User Story 2 — Inspect Detailed Data and Fold Evidence (P1)

**Independent acceptance**: Fixed DEV frames with shared months, duplicate indices and unequal folds produce correct membership/count/feature evidence in debug mode, with identical science and complete file evidence in normal mode.

- [X] T016 Extend `tests/test_training_fold_audit.py` with same-sized different-content DEV identities, position-vs-index distinction, split reuse scoped by dataset, last-fold remainder, empty/small-data behavior and normal/debug evidence parity; migrate all structured assertions to file reads. Acceptance: actual arrays reconcile with CSV and no second splitter call occurs. Depends on T015.
- [X] T017 Add a once-per-DEV-context content/order identity and safe feature/membership export metadata to `src/ai_job_market/training_audit.py`; keep the legacy positional digest meaning explicit. Acceptance: changed data/order changes new fingerprint and current schema remains readable. Depends on T016.
- [X] T018 Enrich `_temporal_fold_audit_payload`, its callers and the DEV/test split observations in `src/ai_job_market/core.py` using existing arrays/results: stable IDs, actual boundaries, requested/effective folds, train/validation counts, row overlap versus shared months and complete membership refs. Acceptance: T016 checks pass without mutation/resplitting; bind fold events to the correct dataset/evaluation. Depends on T017.
- [X] T019 Instrument existing correlation/readiness fit/transform and salary fold fit/predict/score/extract boundaries in `src/ai_job_market/core.py`; record actual feature variants, training-only fit scope, fitted encoded counts, derivations and explicit failed/cancelled operations. Preserve importance-only `not_computed` metrics and current combined fit_transform semantics. Acceptance: T003 parity and T016 evidence checks pass; no additional transform or prediction for logging. Depends on T018.
- [X] T020 Verify US2 using `tests/test_training_fold_audit.py`, `tests/test_training_behavior.py` and the debug integration fixtures in `tests/test_training_audit_integration.py`; record fixture/count/identity reconciliation in `specs/002-readable-training-logs/verification.md`. Acceptance: normal/debug filtering differs only in presentation and all required detail remains persisted. Depends on T019.

## Phase 5: User Story 3 — Understand Comparison and Tuning (P1)

**Independent acceptance**: Known candidate/trial fixtures, including failed trials, non-RF selection, undefined scores and later-stage parameter disagreement, render the exact available evidence with no invented fit label or extra evaluation.

- [X] T021 Add `tests/test_training_evaluation_logs.py` for all five candidate summaries, per-metric aggregation/population SD, partition/units, unavailable train/test metrics, insufficient-evidence labels, legacy narrative rejection, all 26 RF trial identities/progress, non-RF skip and actual s1 final settings. Include failed-trial propagation, log-only aggregation with missing values, and no additional scorer calls. Acceptance: red before new reporting. Depends on T020.
- [X] T022 Emit per-invocation evaluation start/completion and log-only aggregates for primary candidates and both ablation families in `src/ai_job_market/core.py`; render/model-export those summaries in `src/ai_job_market/training_console.py`. Acceptance: each actual candidate has labeled metrics and baseline differences where available, unchanged returned scientific tables, and `insufficient_evidence` with concrete reasons. Depends on T021.
- [X] T023 Add explicit tuning group/trial context, seed/effective parameters, progress and group-winner evidence around existing loops in `src/ai_job_market/core.py:tune_random_forest_manual_steps`, plus explicit skipped tuning in its caller. Acceptance: 4/6/6/4/6 fixture trials trace to their folds and true R² rankings; no retries or parameter changes. Depends on T022.
- [X] T024 Add actual family/final selection, applied-versus-recommended settings, selected-only locked-test metrics, empirical q90, importance and subgroup/error-report evidence in `src/ai_job_market/core.py` and readable summaries in `src/ai_job_market/training_console.py`. Acceptance: missing candidate test entries are not evaluated; no new test exposure, fit classifier or changed s1 selection. Depends on T023.
- [X] T025 Verify US3 with `tests/test_training_evaluation_logs.py` and `tests/test_training_behavior.py`, tracing each fixture's displayed numbers and settings to the saved evidence; record metrics/selection parity and limitations in `specs/002-readable-training-logs/verification.md`. Acceptance: failure reasons and unsupported fit assessments are explicit for every applicable candidate. Depends on T024.

## Phase 6: User Story 4 — Retrieve Full Evidence Without Terminal Flooding (P2)

**Independent acceptance**: Large synthetic safe evidence tables can be inspected through current-run CSVs with matching counts, hashes and numerical precision, while terminal previews remain bounded and failed exports remain visibly incomplete.

- [X] T026 Extend `tests/test_training_evidence_exports.py` and `tests/test_training_audit_integration.py` for manifest identity/hash/schema, mutable-source snapshots, large/wide tables, failed/missing/corrupt exports, callback failure, repeated/nested runs and file-only evidence parity. Add joblib/reload/integrated-operation expectations without creating extra model calls. Acceptance: no stale cross-run evidence fills missing current results. Depends on T025.
- [X] T027 Complete the remaining observation-only joblib write/load, existing reload-check predictions, prediction summaries and integrated segment prediction/grouping events in `src/ai_job_market/core.py`; close final stages only after their actual operations/writes return. Acceptance: all six existing dumps, one load, two reload predictions and existing segment prediction are observed once at caller level; preserve their call order and output schemas. Depends on T026.
- [X] T028 Finalize run-scoped evidence manifests, safe CSV snapshots and per-table provenance in `src/ai_job_market/training_audit.py`, adding required evidence references at `src/ai_job_market/core.py` result boundaries. Acceptance: complete safe data/features/folds/comparison/tuning/segmentation evidence in both modes, full precision and no recursive core saves or raw-record replication. Depends on T027.
- [X] T029 Finish failure-path behavior in `src/ai_job_market/training_audit.py` and `pipeline.py`: independent sink/observer/export/close/timing warnings, accurate completeness and run status, guaranteed cleanup and no JSON fallback. Acceptance: T004/T026 failure matrix passes; unsafe paths remain rejected; training exceptions/exits are never masked by log cleanup. Depends on T028.

**Checkpoint**: All user stories have executable acceptance evidence; not yet release-verified until the final phase.

## Phase 7: Documentation and Cross-Cutting Verification

- [X] T030 [P] Update `docs/TRAINING_AUDIT.md`, `README.md` and `tests/test_training_audit_docs.py` with four-level examples, new flag/optional callable arguments, exact source/count/feature interpretation, CSV inspection, fit limitations, honest completion states and existing chart links. Preserve historical evidence labeling and explain the intentional terminal JSON contract change. Acceptance: documentation tests pass and no planned/unexecuted run is claimed as completed. Depends on T029.
- [X] T031 Add/finish true CLI subprocess and core integration assertions in `tests/test_training_cli.py` and `tests/test_training_audit_integration.py` for normal/debug output, invalid args/preflight exit 2, runtime failure 1, interruption 130, observer-free core, heartbeat coexistence and UNKNOWN rather than fabricated PASS. Acceptance: real parser/entrypoint exercised with bounded fixtures; no root output writes. Depends on T029; may run in parallel with T030.
- [X] T032 Execute the isolated baseline/normal/debug/core workflows in `specs/002-readable-training-logs/quickstart.md` and record results in `specs/002-readable-training-logs/verification.md`; use parity assertions in `tests/test_training_behavior.py` and integration tests for memberships/features/settings/call counts/metric outputs at declared tolerances. Acceptance: actual command exits, normalized comparisons, log/CSV reconciliation and timing/size evidence recorded; root outputs/artifacts unchanged. Depends on T030–T031.
- [X] T033 Run the focused and full pytest suites, including `tests/test_release_contract.py`, `tests/test_segmentation_v3.py` and `tests/test_segmentation_representation_robustness.py`; record exact results and `git diff --check` in `specs/002-readable-training-logs/verification.md`. Acceptance: artifact consumers still work; unrelated inherited failures are reported rather than silently fixed. Depends on T032.
- [X] T034 Complete the five-minute transcript-plus-CSV reviewer exercise and a final requirement-to-evidence audit in `specs/002-readable-training-logs/verification.md` and `specs/002-readable-training-logs/checklists/requirements.md`. Acceptance: SC-001–SC-006 each has actual proof or explicit remaining failure; document Branch A internal-trace and missing train-score limits. Depends on T033.
- [X] T035 Refresh Graphify after verified code changes and record the command/result or justified inability in `specs/002-readable-training-logs/verification.md`; reconcile `specs/002-readable-training-logs/{spec,plan,coverage,tasks}.md` against the final diff. Acceptance: all completed tasks cite proof, no unsupported completion/security/scientific claims, no unrelated user changes staged or reverted. Depends on T034.

## Dependencies & Execution Order

- T001 → T002 → foundation. T003/T004/T005 may proceed in parallel after T002 because their test files differ; T006 waits for T004/T005; T007 → T008.
- US1: T009 and T011 are independent red-test work; T010 depends on T009; T012 waits for both T010 and T011; T013 → T014 → T015.
- US2: T016 → T017 → T018 → T019 → T020.
- US3: T021 → T022 → T023 → T024 → T025.
- US4: T026 → T027 → T028 → T029.
- Final: T030 and T031 can run in parallel, then T032 → T033 → T034 → T035.
- Production edits to `core.py` or `training_audit.py` must be serialized. Do not assign independent agents to modify these shared files concurrently.
- Every test phase has a red → minimal implementation → green checkpoint. A unit-fixture pass does not substitute for final real pipeline parity.

## Requirement Traceability

| Requirements | Tasks |
|---|---|
| FR-001–003 verbosity/readability | T004, T009–T012, T015, T031 |
| FR-004–005 counts/features | T013–T014, T016–T020 |
| FR-006 split evidence | T016–T020 |
| FR-007–010 model/tuning/scientific preservation | T003, T021–T025, T032 |
| FR-011–013 complete exports/provenance | T005–T008, T017–T018, T026–T028 |
| FR-014 failure honesty | T004–T006, T011–T012, T026–T029, T031 |
| FR-015–016 minimal validation/safe evidence | T004, T007–T008, T026, T028–T029 |
| FR-017 observation-only scope | T001–T003, T019, T023–T024, T027, T032–T035 |
| FR-018 documentation | T030, T034–T035 |
| SC-001–002 coverage/presentation | T015, T020, T028, T031–T034 |
| SC-003 accurate scores/counts | T020, T025, T032–T034 |
| SC-004 reviewer usability | T030, T034 |
| SC-005 scientific parity | T003, T019, T025, T032–T033 |
| SC-006 failure status | T004–T006, T011, T026–T029, T031 |

## Implementation Strategy and Stop Point

MVP is foundation + US1, with a complete normal-mode lifecycle rather than only a prettier individual line. Subsequent slices add debug data traceability, model/tuning explanations, and durable full-run exports. All tasks remain unchecked at planning completion.

**STOP FOR REVIEW**: This file is the automatically generated task plan the user requested. Before `/speckit.implement`, obtain explicit task approval and resolve T001. User may approve, request revisions, or edit artifacts manually. No implementation is authorized merely because task generation was automatic.
