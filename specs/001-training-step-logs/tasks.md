# Tasks: English Training Step Logs and Audit Evidence

**Input**: `specs/001-training-step-logs/`  
**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `contracts/training-log.md`, `quickstart.md`  
**Status**: Proposed; all implementation tasks are unstarted. STOP for user review.

**Tests**: Required. Characterize existing behavior before instrumentation; logging tests must fail before implementation and pass afterward. No performance-score gate. Do not modify existing scientific behavior to make tests pass.

**Format**: `- [ ] TNNN [P?] [US?] Description with file path`. `[P]` means an explicitly independent file/task after prerequisites are met.

## Phase 1: Setup and Approval

- [X] T001 Record explicit implementation/isolated-validation approval and maintainer disposition of inherited VI.2/VI.4 findings in `specs/001-training-step-logs/plan.md`; if disposition requires modeling changes or governance amendment, stop and request a separate scope instead of changing training.
- [X] T002 Recheck Graphify/source relationships and exact operation coverage in `specs/001-training-step-logs/research.md`; record source/config/data identity, installed versions and baseline verification strategy in `specs/001-training-step-logs/verification.md`, preserving the outstanding `requirements.txt` edit.

**Checkpoint**: No implementation before T001. T002 must distinguish current measurements, static findings, and historical artifacts; do not claim full historical reproducibility.

## Phase 2: Foundational Characterization

- [X] T003 Add fixed-data characterization in `tests/test_training_behavior.py` for actual split membership/order, remainders/small inputs, clone/fit/predict call counts, metrics and train-only preprocessing; run against original source before instrumentation and record results in `specs/001-training-step-logs/verification.md`.
- [X] T004 Extend `tests/test_training_behavior.py` to freeze family-MAE ranking, tuning-R² ranking, fixed depth 20, initial-grid final estimator selection and existing final diagnostic/reload calls; use bounded deterministic fakes for search orchestration and save expected scientific results before code changes.

**Checkpoint**: Baseline characterization passes without fixing current behavior. Any failure is investigated and recorded, not silently treated as a new expected value.

## Phase 3: User Story 1 — Live and Persistent Training Logs (P1 / MVP)

**Goal**: Follow executed training operations in English on the terminal and replay the same events from a per-run file.

**Independent test**: A bounded run/fixture produces matching terminal/file events with context, real start/end timing, metrics, and preserved failure/return semantics.

### Tests first

- [X] T005 [P] [US1] Add failing session/contract tests in `tests/test_training_audit.py` for strict JSON, required fields, identical dual-destination events, unique exclusive paths, English owned templates, flush/close, no root logger mutation, repeat/nested sessions and no-op emission outside a session.
- [X] T006 [P] [US1] Add failing core/CLI integration tests in `tests/test_training_audit_integration.py` for operation context, combined fit labeling, return/signature compatibility, stage-hook coexistence, preflight-without-run and exit codes 0/1/2/130; load root `pipeline.py` by file location and use isolated copied roots.

### Implementation

- [X] T007 [US1] Implement minimal scoped session, standard-library dual-output logger, strict allowlisted serialization, provenance capture, exclusive safe workspace path creation and English failure reporting in `src/ai_job_market/training_audit.py`; satisfy T005 without new dependencies.
- [X] T008 [US1] Integrate signature-preserving run lifecycle and contextual model/trial/fold events into `src/ai_job_market/core.py:run_pipeline` and `evaluate_model_cv`; preserve original fit/predict/timing/return computations and record input/config/seed/features plus existing pipeline ID.
- [X] T009 [US1] Instrument existing ablation contexts, correlation/readiness preprocessing and importance-only fits in `src/ai_job_market/core.py`; report fit scopes and not-computed validation metrics without adding any model calls.
- [X] T010 [US1] Instrument all five manual tuning stages and actual family/final selection in `src/ai_job_market/core.py`; emit real parameters, ranking basis, fixed values, skipped tuning reasons and initial-grid final source, leaving search order, rationales and selected values untouched.
- [X] T011 [US1] Instrument final DEV fit, locked-test prediction/metrics, interval/importance diagnostics, existing CSV/JSON/joblib writes and reload equivalence in `src/ai_job_market/core.py`; retain save signatures used by CLI hooks and do not duplicate computations or artifact writes.
- [X] T012 [US1] Add induced initialization/write/serialization failure, cancellation and sanitized-error tests in `tests/test_training_audit.py`, observe their failures, then minimally complete failure isolation in `src/ai_job_market/training_audit.py`; assert visible degraded evidence and unchanged original training exceptions.
- [X] T013 [US1] Run `tests/test_training_behavior.py`, `tests/test_training_audit.py` and `tests/test_training_audit_integration.py`; record MVP evidence and remaining gaps in `specs/001-training-step-logs/verification.md` before proceeding.

**Checkpoint**: Live/persistent logging works across the executed salary path; baseline call counts and selection tests still pass. No claim that detailed fold audit or docs are complete yet.

## Phase 4: User Story 2 — Verifiable Fold Evidence (P2)

**Goal**: Verify the exact folds actually consumed by comparison, tuning and existing auxiliary salary fits.

**Independent test**: Reconstruct logged memberships from a fixture and compare them with consumed arrays, without an additional splitter call.

### Tests first

- [X] T014 [US2] Add failing tests in `tests/test_training_fold_audit.py` for ordered membership identity, 200/201 remainder behavior, small/empty fold behavior, non-contiguous/duplicate indices, same-month versus row overlap, repeated split references, skipped branches and no extra splitter/fit/predict calls.

### Implementation

- [X] T015 [US2] Add dataset/order identity and split definitions/references in `src/ai_job_market/core.py`; emit full actual positional lists and observed counts/period overlaps from consumed splits without changing input data.
- [X] T016 [US2] Connect split evidence to the arrays already consumed by `evaluate_model_cv` and `random_forest_importance_by_fold` in `src/ai_job_market/core.py`, including requested/effective folds and block/remainder rules; retain all splitter return values and failure paths.
- [X] T017 [US2] Add failing aggregate/non-finite metric tests in `tests/test_training_fold_audit.py`, then add supplementary log-only R² mean/population SD and strict unavailable statuses through `src/ai_job_market/core.py` and `src/ai_job_market/training_audit.py`; preserve existing result schemas and ranking metrics.
- [X] T018 [US2] Run `tests/test_training_fold_audit.py` with the behavior and US1 suites; reconcile candidate split references with consumed memberships and record results in `specs/001-training-step-logs/verification.md`.

**Checkpoint**: Fold identities and metric evidence are traceable; same-month boundaries are disclosed rather than repaired. US1 remains valid.

## Phase 5: User Story 3 — Audit Documentation (P3)

**Goal**: Explain current training and log interpretation with traceable English evidence.

**Independent test**: Follow documented source/output references and reconcile every performance claim with its labeled historical or new-run evidence.

### Tests first

- [X] T019 [US3] Add failing guide contract checks in `tests/test_training_audit_docs.py` for required workflow/log/failure/limits sections, valid repository links, distinct CV/test and historical/runtime labels, and explicit non-gating treatment of 0.85.

### Documentation

- [X] T020 [US3] Write `docs/TRAINING_AUDIT.md` in English with operation/function map, actual folds and parameters, log schema/examples, commands, provenance, existing diagnostic artifact/chart references and historical metrics; document R²/MAE mismatch, fixed depth and initial-grid final selection as uncorrected findings, not validated rationale.
- [X] T021 [US3] Link the guide from `README.md` and document actual validation run locations/results from `specs/001-training-step-logs/verification.md` in `docs/TRAINING_AUDIT.md`; run `tests/test_training_audit_docs.py` and leave pending runtime evidence explicitly pending until final validation.

**Checkpoint**: Documentation is usable from historical evidence alone and does not falsely claim a new run; actual new evidence is appended only after execution.

## Phase 6: Cross-Cutting Verification and Review

- [X] T022 Run all new tests and the existing full pytest suite using `specs/001-training-step-logs/quickstart.md`; record exact results in `specs/001-training-step-logs/verification.md`, investigating any scientific behavior difference before continuing.
- [X] T023 Execute the approved real core run in an isolated workspace and the actual CLI in a temporary copied project per `specs/001-training-step-logs/quickstart.md`; record strict event parity, original-root artifact preservation, output-schema/decision/prediction equivalence, measured log size and overhead methodology/limits in `specs/001-training-step-logs/verification.md` (no repeated training to chase scores).
- [X] T024 Update runtime evidence and caveats in `docs/TRAINING_AUDIT.md`, review changed source for English-only additions/no modeling changes/no raw data or secrets, run `git diff --check`, and record source/consumer-contract review results in `specs/001-training-step-logs/verification.md`; no UI refitting or browser feature is introduced.
- [X] T025 Refresh `graphify-out/graph.json` with `graphify update .` after code verification (or record a justified inability in `specs/001-training-step-logs/verification.md`), reconcile `spec.md`/`plan.md`/`tasks.md` with actual delivery, and present completed evidence and remaining limitations for review.

## Dependencies & Execution Order

- T001 -> T002 -> T003 -> T004 before any instrumentation.
- T005 and T006 may run in parallel after T004: different test files and shared contract already defined.
- T007 depends on T005; T008 follows T007 and T006; T009–T013 sequential because production changes share `core.py` and verification notes.
- US2 follows US1 in this implementation to reuse its audit session. T014 -> T015 -> T016 -> T017 -> T018. Its membership assertions remain independently runnable with a bounded fixture.
- US3 static guide/test drafting can start after the contract is approved, but T021 completion depends on US1/US2 field names being verified; T019 -> T020 -> T021.
- T022 -> T023 -> T024 -> T025 after all three stories. T023 requires explicit isolated real-run approval captured in T001; otherwise mark blocked, not complete.

## Parallel Examples

- **US1**: T005 session-contract test author and T006 integration-test author can work concurrently; do not edit shared fixtures without coordination.
- **US2**: Keep production tasks sequential because they share the audit module/core. Membership test drafting can be reviewed independently, but implementation waits for failing tests.
- **US3**: T020 static documentation drafting can run alongside US2 code work after T019, since files differ; do not finalize runtime claims before validation.

## Implementation Strategy

1. MVP = characterization plus US1: genuine English step events in terminal/file, preserved training behavior.
2. Add exact fold evidence with US2, verify no extra computations.
3. Add evidence guide with US3; use historical-only labels until approved real validation.
4. Finish parity/failure/full-suite checks and isolated real-run evidence. Do not “clean up” adjacent modeling code.
5. Mark `[X]` only after the named verification passes; keep unresolved governance/runtimes visible.

## Traceability

| Requirements | Primary tasks |
|---|---|
| FR-001, FR-015 (preservation) | T003, T004, T013, T018, T022–T024 |
| FR-002 (English) | T005, T007–T012, T019–T021, T024 |
| FR-003–FR-005 (dual output/context/provenance) | T005–T008, T012, T023 |
| FR-006 (operation coverage) | T008–T011, T013 |
| FR-007 (fold audit) | T014–T018 |
| FR-008 (metrics/actual selection) | T010, T011, T017, T020 |
| FR-009 (CV/test separation) | T004, T011, T020, T023 |
| FR-010 (failures/skips) | T006, T010, T012, T014 |
| FR-011–FR-012 (docs/non-gating reference) | T019–T021, T024 |
| FR-013–FR-014 (safety/boundaries) | T005–T007, T012, T024 |

## Mandatory User Review Gate

25 tasks total: setup 2; foundation 2; US1 9; US2 5; US3 3; final verification 4. All remain unchecked.

Choose: **approve implementation**, **revise with agent**, or **revise manually**. Approval must address the inherited governance findings and whether real isolated validation runs are authorized. No code changes or training are performed by generating this task list.
