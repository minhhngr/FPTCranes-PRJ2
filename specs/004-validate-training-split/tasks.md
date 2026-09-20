# Tasks: Training-Only Split and Validation Evidence

**Input**: `specs/004-validate-training-split/` spec, plan, research, data model, quickstart and contract.  
**Status**: APPROVED 2026-09-19; implementation in progress. Real-data execution remains separately gated by eligible unseen reserve data and partition approval.  
**Scope**: Offline training only; no UI/preprocessing/preparation changes; no real training without eligible data and separate partition approval.

## Format and validation policy

`- [ ] Tnnn [P?] [USn?] Description with exact paths.` `[P]` means independent files/no dependencies within that phase. All behavior tests must fail for the missing behavior before implementation; record failure/pass evidence in `specs/004-validate-training-split/verification.md`. No test result is claimed by this document.

## Phase 1 — Approval and protected baseline

- [X] T001 Record explicit user task approval, reviewed policy choices (Top-2 pair, split-ranking objective, tie/fit rules, 540-fit bound, runtime protocol and 1,784 additional predict-call ceiling) and execution/data gates in `specs/004-validate-training-split/verification.md`; stop if scope differs.
- [X] T002 Inventory hashes for `src/ai_job_market/core.py`, `pipeline.py`, `config/project.yaml`, preparation modules, `src/pages/`, `streamlit.py`, existing `outputs/`/`artifacts/`, dependencies and inherited `uv.lock`; record exclusions for only the new namespace in `specs/004-validate-training-split/verification.md`.
- [X] T003 Refresh the targeted Graphify query and inspect reused helper/audit/import boundaries; record confirmed functions and unchanged contracts in `specs/004-validate-training-split/research.md` before code changes.

**Checkpoint**: Scope approved; no training/preparation or baseline mutation. T001 is a hard dependency for every implementation task.

## Phase 2 — Foundational evidence/input contract

- [X] T004 Add failing tests in `tests/test_training_evidence_v2.py` for schema versions, finite/null metrics, fixed input locations, path/symlink escapes, resource limits, run IDs, policy/approval parsing and provenance identities defined by `contracts/training-evidence.md`.
- [X] T005 Implement minimal typed records/readers/hash and boundary validators in `src/ai_job_market/training_evidence_io.py`; make T004 pass without loading joblib, fitting or modifying legacy artifacts.
- [X] T006 Add the bounded training-only policy in `config/training_validation.json` and policy-fingerprint tests in `tests/test_training_evidence_v2.py`; preserve `config/project.yaml` and validate exact feature/search/fold defaults against the contract.
- [X] T007 Test then implement new-namespace ordered JSONL lifecycle events in `tests/test_training_evidence_v2.py` and `src/ai_job_market/training_evidence_io.py`; verify completion/failure/cancellation/reuse and no writes to primary-pipeline logs.

**Checkpoint**: `test_training_evidence_v2.py` passes; invalid inputs fail before writes/fits. Existing audit lifecycle unchanged.

## Phase 3 — US1: Auditable temporal partitions and folds (P1, MVP)

**Independent value**: Read-only inspection explains exactly why a dataset is ready or blocked without training it.

- [X] T008 [US1] Create synthetic whole-month fixtures under `tests/fixtures/training_validation/` and failing tests in `tests/test_training_partitions_v2.py` for contiguous approximate allocation, proposal ranking/ties, complete disjoint membership, missing/invalid dates and stale approvals.
- [X] T009 [US1] Implement target-independent metadata projection, stable row identity, approved month cutoffs and actual-share/deviation reporting in `src/ai_job_market/training_partitions.py`; make T008 pass with no target statistics or preparation calls.
- [X] T010 [US1] Extend `tests/test_training_partitions_v2.py` with historical/unknown reserve, changed-file known-row fingerprint, ambiguous duplicate, custodian attestation and holdout-exposure cases; assert known exposure cannot be overridden by an “unseen” label.
- [X] T011 [US1] Implement exposure reconciliation and actionable readiness failures in `src/ai_job_market/training_partitions.py`; current exposed reserve must block complete execution, not become a same-month or sampled reserve.
- [X] T012 [US1] Add failing five-outer/three-inner expanding-fold tests in `tests/test_training_partitions_v2.py`, including nine-month readiness, unequal monthly row counts/gaps, shared periods, expansion, nested parents, later-unused rows, explicit percentage denominators and independent monthly-total/membership reconciliation; reject count mutations.
- [X] T013 [US1] Implement outer/inner fold declarations and summaries in `src/ai_job_market/training_partitions.py`; export `fold_summary.csv`, `monthly_row_counts.csv` and memberships and emit `fold_plan_defined`/`fold_plan_used` via `src/ai_job_market/training_evidence_io.py`. Make T012 pass; show exact counts/added history/coverage and no fallback to sliding blocks or random folds.
- [X] T014 [US1] Add inspect CLI tests in `tests/test_training_validation_cli.py`, then implement `inspect` in `src/ai_job_market/training_validation.py`; prove JSON stdout/English stderr, readiness exit codes and zero fits/predicts/writes via spies.

**Checkpoint**: US1 tests pass independently. Run only read-only current-workspace inspect after implementation; record expected unavailable unseen reserve and actual ratios. This does not authorize training on that dataset.

## Phase 4 — US2: Five-family comparison and explanations (P1)

**Independent value**: Training-only fixture comparison explains candidate performance/runtime/selection without opening holdout targets.

- [X] T015 [US2] Add failing evaluator tests in `tests/test_training_evaluation_v2.py` for 25 identical-membership candidate folds, cloned fold-local preprocessing, fresh vocabularies, paired metrics, constant/single-row R² and no holdout/reserve target access.
- [X] T016 [US2] Implement the explicit-fold paired evaluator in `src/ai_job_market/training_evaluation.py` using only existing model/preprocessor factories; make T015 pass and record measured fit/predict durations and dimensions.
- [X] T017 [US2] Add then satisfy arithmetic/tie/fit-diagnostic tests in `tests/test_training_evaluation_v2.py` and `src/ai_job_market/training_evaluation.py` for equal-fold mean/SD, separately pooled OOF scores, raw MAE rank, interval-overlap simplicity choice, zero denominators and mixed-fit labels.
- [X] T018 [US2] Test then implement six approved-family removal experiments and baseline-matched deltas in `tests/test_training_evaluation_v2.py` and `src/ai_job_market/training_evaluation.py`; reject prohibited `experience_level` and any adaptive feature selection.
- [X] T019 [US2] Test then capture RF encoded/raw-family importance in the existing comparison fits, vocabulary-presence alignment and adjacent-fold half-L1 drift in `tests/test_training_evaluation_v2.py` and `src/ai_job_market/training_evaluation.py`; assert zero extra importance-only fits.
- [X] T020 [US2] Add comparison-stage orchestration/family-freeze tests in `tests/test_training_validation_cli.py`, then wire training-only comparison stages into `src/ai_job_market/training_validation.py`; incomplete/failed candidates cannot produce an approved selection.

**Checkpoint**: US2 fixture metrics can be independently recomputed; compare R²/MAE before runtime and decision. No final evaluation yet. Fit budget and all selection inputs are logged.

## Phase 5 — US3: Bounded tuning, frozen evaluation and diagnostics (P2)

**Independent test approach**: Supply frozen selection/partition fixtures to this stage; integration later consumes US2 outputs.

- [X] T021 [US3] Add failing search tests in `tests/test_training_search_v2.py` for five outer-training plus final TRAIN inner searches, 26 slots/search, three inner folds, carried incumbent settings, deterministic MAE ties, exact-identity reuse and non-RF skip.
- [X] T022 [US3] Implement the bounded stepwise RF search in `src/ai_job_market/training_search.py`; make T021 pass and log every trial's actual parameters/folds/seed/result, including failed/reused slots and final applied settings.
- [X] T023 [US3] Add then satisfy full/Top-2 matched outer-fold and final TRAIN-only fit tests in `tests/test_training_search_v2.py`, `src/ai_job_market/training_search.py` and `src/ai_job_market/training_evaluation.py`; use fixed category/experience inputs and no separate Top-2 search or holdout-driven promotion.
- [X] T024 [US3] Add failing evaluation-ledger tests in `tests/test_training_evidence_v2.py` for population-keyed identity, new run-ID bypass, known fingerprints under changed containers, concurrency, changed declarations, started/failed locks and unchanged completed reuse.
- [X] T025 [US3] Implement immutable evaluation declaration and exclusive persistent access claim in `src/ai_job_market/training_evidence_io.py`; make T024 pass and gate partition-scoped holdout target loading on the recorded lock.
- [X] T026 [US3] Add failing final-evaluation tests in `tests/test_training_evaluation_v2.py` for five frozen factory candidates plus full/Top-2, residual sign, exact MAE/RMSE/R²/MedAE, no reserve rows and unchanged frozen selection.
- [X] T027 [US3] Implement final holdout scoring/prediction evidence in `src/ai_job_market/training_evaluation.py`; make T026 pass without refitting any estimator on holdout or reserve.
- [X] T028 [US3] Test then implement raw-column and joint-family permutation MAE increase (12 repeats), method-labelled encoded importance, support-flagged subgroup slices and linear-method q90/bands in `tests/test_training_evaluation_v2.py` and `src/ai_job_market/training_evaluation.py`; assert same-population coverage caveat and zero reserve analysis.
- [X] T029 [US3] Wire conditional search, final bundles, pre-target freeze/lock and final diagnostics in `src/ai_job_market/training_validation.py` with `tests/test_training_validation_cli.py`; verify 540-fit upper bound, stage failures and target-access ordering using spies.

**Checkpoint**: US3 controlled tests pass; changed model/config cannot gain another pristine evaluation of consumed rows. Tiny estimators prove real arithmetic on isolated fixtures; no real-data generation until lineage gates pass.

## Phase 6 — US4: English reports and offline consumer evidence (P2)

**Independent value**: Render and validate English evidence from known fixture packs without training.

- [X] T030 [P] [US4] Add report/glossary/conclusion tests in `tests/test_training_report_v2.py` for all six 5W1H questions, exact numerical evidence references, fit-vs-outcome separation, missing thresholds, historical exposure and educational-only classification.
- [X] T031 [P] [US4] Add publication/read-only consumer tests in `tests/test_training_evidence_v2.py` for incomplete/corrupt/mismatched packs, bundle metadata/hash integrity, staging failures, unknown schema and zero-fit/predict/deserialization check/reuse paths.
- [X] T032 [US4] Implement pure English report, metric glossary, section conclusions and Good/Bad/Blocked/Inconclusive rule evaluation in `src/ai_job_market/training_report.py`; make T030 pass using full-precision evidence and explicit missing-data reasons.
- [X] T033 [US4] Test then generate offline Plotly comparison, holdout-diagnostic and feature-importance HTML from authoritative tables in `tests/test_training_report_v2.py` and `src/ai_job_market/training_report.py`; no UI/shared chart-producer edits.
- [X] T034 [US4] Implement validated staged publication and read-only pack loading/checking in `src/ai_job_market/training_evidence_io.py`, then finish `run`/`check` in `src/ai_job_market/training_validation.py`; make T031 and CLI tests pass without changing any existing UI pointer/bundle.
- [X] T035 [US4] Add `docs/TRAINING_VALIDATION.md` and update only the command/guide section of `README.md`; explain chronological split/folds, source artifacts, nested tuning, outputs, statuses, educational classification, exposure limitations and future monitoring/retraining handoffs.

**Checkpoint**: Produced pack is consumed by its actual offline validator/report; current UI integration is explicitly absent, not claimed complete.

## Phase 7 — Runtime and human/agent/UI evidence amendment (US2/US4)

**Contract**: `contracts/runtime-observability.md`. Extend existing producer wiring before any real pack is generated. No extra fits or UI changes.

- [X] T036 [US2] Add failing fake-clock/raw-sample/quantile/throughput tests in `tests/test_training_runtime_v2.py` for batch groups, warmups, first-load scope, CPU/RSS units, parent-child timing accounting and exact workload/call bounds.
- [X] T037 [US2] Implement minimal measurement contexts and bounded TRAIN-only microbenchmarks in `src/ai_job_market/training_runtime.py`; make T036 pass, retain samples, verify effective thread/environment conditions and expose unavailable reasons without dependency installs.
- [X] T038 [US2] Add then satisfy benchmark integration spies in `tests/test_training_validation_cli.py`, `src/ai_job_market/training_evaluation.py` and `src/ai_job_market/training_validation.py`; measure 25 candidate-fold and two final variants, first-load observations, no extra fits, no holdout/reserve access and no benchmark on reuse.
- [X] T039 [US2] Add reference tests and derive bias/tail/Dummy-skill/gap/worst-fold/tuning/Top-2 deltas plus OOF row exports in `tests/test_training_evaluation_v2.py` and `src/ai_job_market/training_evaluation.py`; verify null semantics and unchanged primary selection.
- [X] T040 [US2] Test then implement runtime summaries, comparable fastest-model findings, accuracy/fit-cost Pareto data and operational budget verdicts in `tests/test_training_runtime_v2.py` and `src/ai_job_market/training_runtime.py`; separate scientific winner, operational readiness and deployment-review eligibility.
- [X] T041 [US4] Test then extend event schema/renderer in `tests/test_training_evidence_v2.py` and `src/ai_job_market/training_evidence_io.py` with invocation/operation IDs, severity/reason enums, bounded payloads, honest progress, redaction and failure-safe telemetry; no writes to immutable histories on reuse.
- [X] T042 [US4] Add failing parity/consumer tests in `tests/test_training_report_v2.py` for metric catalog, English transcript, agent summary and UI snapshot; require exact fold-count/month/parent explanations and all seven model-role conclusions. Cover losing/failed models, undefined R², zero Dummy error, unavailable runtime, missing/mismatched source references and count mutations without fabricated verdicts.
- [X] T043 [US4] Produce metric catalog, transcript/agent/UI views, `fold_explanation.md`, `model_conclusions.json` and accuracy-runtime HTML in `src/ai_job_market/training_report.py`; make T042 pass using authoritative evidence. Put outer-fold method/count descriptors before comparisons and parent-labelled inner folds before tuning; explain each candidate/full/Top-2 role's decision, limitation and safe action without hidden recalculation.
- [X] T044 [US4] Extend `src/ai_job_market/training_evidence_io.py` and `tests/test_training_evidence_v2.py` for all new manifest files, runtime policy/operational-limit validation, snapshot descriptor/hash bounds and no-model-load check/reuse; update `config/training_validation.json` defaults to the reviewed protocol.
- [X] T045 [US4] Update `docs/TRAINING_VALIDATION.md` and CLI fixture checks in `tests/test_training_validation_cli.py` for performance questions, local-microbenchmark limitations, safe agent next actions, complete-pack UI exports and separate operational budgets; verify agent answers directly from structured fixture evidence.

**Checkpoint**: All views agree before rounding; raw samples reproduce summaries; benchmark costs do not change scientific selection, 540-fit ceiling, reserve protection or holdout locking. UI integration remains separately scoped.

## Phase 8 — Verification and release review

- [X] T046 Execute the full new CLI against a temporary chronological fixture workspace, including failure/reuse paths, via `tests/test_training_validation_cli.py`; record timings, fit/benchmark counts and zero protected-path changes in `specs/004-validate-training-split/verification.md`.
- [X] T047 Run read-only preflight on current prepared data through `src/ai_job_market/training_validation.py`; record real-input readiness, ratio deviations and unavailable new-experiment artifacts in `specs/004-validate-training-split/verification.md`. If blocked, do not train or mark real scientific validation passed.
- [ ] T048 Only if separately approved eligible future inputs exist, generate required new-namespace evidence via `src/ai_job_market/training_validation.py`, run `check` and prove unchanged zero-fit/predict/load reuse with spies; otherwise explicitly defer this task/data-dependent acceptance in `specs/004-validate-training-split/verification.md` without modifying old outputs.
- [X] T049 Run relevant existing/new pytest suites, full isolated regression tests, Ruff and diff checks from `quickstart.md`; compare T002 protected hashes and record exact results/inherited failures in `specs/004-validate-training-split/verification.md`.
- [ ] T050 Conduct the two-reader English comprehension walkthrough and inspect actual report/chart references using `docs/TRAINING_VALIDATION.md`, including fastest/selected distinction, fold 3's exact months/counts, why training expands, protected-population exclusions, every model's decision and missing evidence; record results or explicit pending status in `specs/004-validate-training-split/verification.md`, never fabricate human acceptance.
- [X] T051 Update Graphify after source changes (or justify a valid exception), review the final diff/contracts, and synchronize `specs/004-validate-training-split/spec.md`, `plan.md`, `tasks.md` and `verification.md`; leave all unresolved real-data/reader acceptance visible before merge review.

## Dependencies and execution order

- T001–T003 precede all implementation. T004–T007 establish the evidence boundary.
- US1 T008–T014 precedes candidate fitting; metadata/exposure checks run before any scientific stage.
- US2 T015–T020 depends on verified partitions and can be tested without final evaluation.
- US3 T021–T029 depends on fold/evidence contracts and consumes the frozen US2 family decision. Search tests can use an explicit fixture decision. T024–T025 must pass before T026–T029 permit any holdout target load.
- US4 T030 and T031 may run in parallel because they edit different tests; neither starts implementation until its failing tests are captured. T032–T034 implement the corresponding contracts, then T035 documents them.
- T036–T045 extend US2/US4 before release validation; T036 precedes T037–T038 and T040, T041 precedes T042–T044; T039 supplies derived metrics for T043. T044 finishes after T040–T043. No real pack may be generated with only the older partial contract.
- T046–T051 follow all desired stories/amendment. T048 is conditional on future data and additional partition approval, not automatically unlocked by code completion. Report deferred rather than checking it complete if data is absent.
- Every test-before-implementation task contains a fail-then-pass checkpoint; reuse tests cannot rely only on printed values.

## Parallel example

After US3 interfaces are stable, T030 (`tests/test_training_report_v2.py`) and T031 (`tests/test_training_evidence_v2.py`) can be written independently. Do not parallelize tasks that both edit `training_validation.py` or `training_evaluation.py`; those files are shared stage integration points.

## MVP and incremental delivery

1. US1 delivers safe data inspection/fold audit and a truthful blocked result for the current snapshot.
2. US2 adds TRAIN-only comparison with fixture evidence; holdout is still closed.
3. US3 adds conditional nested tuning and locked final diagnostics under strict data gates.
4. US4 completes human-readable evidence and offline exports; no page changes.
5. Full implementation acceptance may be separate from real future-data scientific acceptance. Document that distinction; do not infer approval for acquiring data, editing preparation or relaxing the unseen-reserve requirement.

## Requirement traceability

| Requirements | Tasks |
| --- | --- |
| FR-001–006, FR-022; SC-001–002 | T004–T014 |
| FR-007–010; SC-003 | T015–T020 |
| FR-011–015; SC-004 | T021–T029 |
| FR-016–019, FR-023; SC-005–006 | T030, T032–T033, T035, T050 |
| FR-020–021; SC-007–008 | T002, T024–T025, T031, T034, T046–T049 |
| FR-024–027; SC-009, SC-012 | T036–T040 |
| FR-028–031; SC-010–012 | T041–T045, T050 |
| FR-032–033; SC-013–014 | T012–T013, T041–T045, T050 |
| Governance / living spec | T001, T003, T051 |

**Progress**: 49 complete, 2 data/human-acceptance gated (T048 and T050). Implementation was explicitly approved; unresolved integration, eligible-data, and reader-acceptance work remains visible above.
