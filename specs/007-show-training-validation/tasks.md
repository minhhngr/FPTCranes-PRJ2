# Tasks: Show Offline Training Validation on Pages 04–05

**Input**: `specs/007-show-training-validation/` specification, plan, research, data model, quickstart and UI contract.  
**Status**: Approved and implemented. Browser/pixel and two-reader acceptance remain pending.  
**Scope**: Read-only Page 04/05 expanders; no pipeline, producer, Page 06, training, artifact generation or fallback changes.

## Phase 1 — Approval and protected baseline

- [X] T001 Record explicit approval of this task package and confirmed boundaries (offline-only producer, Pages 04–05 expanders, unavailable state, no fallback, no Page 06/pipeline/model activation) in `specs/007-show-training-validation/verification.md`; stop if scope differs.
- [X] T002 Capture hashes/status for `src/ai_job_market/training_*.py`, `pipeline.py`, `src/pages/page06_prediction.py`, `src/ai_job_market/ui_evidence_io.py`, `config/`, existing `outputs/`/`artifacts/`, dependency files and inherited `uv.lock` in `specs/007-show-training-validation/verification.md`; preserve the dirty work inherited from feature 004.
- [X] T003 Refresh the targeted Graphify query and inspect Page 04/05 render order, workspace routing, existing evidence loaders and AppTest fixtures; record the confirmed narrow dependency path in `specs/007-show-training-validation/research.md`.

**Checkpoint**: Approved scope and immutable producer/artifact baseline recorded. No training or UI code changed.

## Phase 2 — Foundational read-only consumer contract

- [X] T004 Add RED discovery tests in `tests/test_training_validation_views.py` for missing root, empty root, staging/ledger ignore, malformed IDs, symlink/path escape, strict UTC timestamps, bounded 100-run scan, newest-valid selection, deterministic ties and newer-invalid/older-valid behavior.
- [X] T005 Implement minimal pack discovery in new `src/pages/training_validation_presentation.py` using fixed workspace paths and `training_evidence_io.validate_complete_pack()`; make T004 pass without importing joblib or training orchestration.
- [X] T006 Add RED contract/projection tests in `tests/test_training_validation_views.py` for required Page 04/05 files/columns, five candidate and seven role identities, five outer/18 inner folds, source/run reconciliation, finite/null metrics, holdout-only predictions, reserve exclusion, raw log/event/report validation, table/download bounds and local optional-section failures.
- [X] T007 Implement validated `TrainingValidationView`, section states, bounded table projections, safe JSON/CSV/JSONL/Markdown/text readers and byte-identical original log/report/summary download descriptors in `src/pages/training_validation_presentation.py`; make T006 pass with no fallback or writes.
- [X] T008 Add no-operation spies in `tests/test_training_validation_views.py` covering fit/search/predict/benchmark, `run_workspace`, pipeline, publication, holdout access and joblib deserialization during discovery/loading; prove all remain zero.

**Checkpoint**: Pure consumer tests select and project complete packs while invalid/missing packs fail closed. No Streamlit page edit yet.

## Phase 3 — US1: Page 04 fold and candidate expanders (P1)

**Goal**: Readers can understand exact fold construction and five-model evidence without opening offline files.

- [X] T009 [US1] Add RED transcript/render-data tests in `tests/test_training_validation_views.py` for Page 04 partition summary, every outer parent/train/validation/later/added/protected row count, exact outer fold 3 months/counts, 18 parent-labelled inner rows, all 25 candidate-fold evaluations, candidate units, selected/lowest/fastest distinctions, five scoped conclusions, source references and deterministic ordering.
- [X] T010 [US1] Implement Page 04 view formatting, evidence-derived transcript rows/complete CSV bytes and `render_page04_training_validation()` in `src/pages/training_validation_presentation.py` using one native collapsed expander, reader-facing bounded previews, explicit derived-vs-raw labels, limitations and authoritative/original/generated downloads.
- [X] T011 [US1] Add RED real-entrypoint AppTest cases in `tests/test_model_ui_pages.py` proving the Page 04 expander appears before existing-source failure, remains collapsed by default, shows exact fixture evidence when opened/rendered, and preserves all existing Page 04 behavior.
- [X] T012 [US1] Insert the minimal Page 04 renderer call after title/caption and before `load_evidence()` in `src/pages/page04_model_comparison.py`; make T011 pass without changing existing charts/tabs/conclusions.
- [X] T013 [US1] Add Page 04 no-fit/predict/load/write AppTest spies, byte/hash assertions for `training.log`/`events.jsonl`/`report.md`/agent/conclusions downloads and deterministic untruncated transcript CSV assertions in `tests/test_model_ui_pages.py` and `tests/test_training_validation_views.py`; ensure previews do not send raw prediction rows or hidden sensitive columns to the frontend.
- [X] T014 [US1] Independently verify Page 04 with valid, no-pack, corrupt-newer/valid-older and existing-supplemental-missing fixtures; record results in `specs/007-show-training-validation/verification.md`.

**Checkpoint**: Page 04 independently delivers fold/candidate evidence or truthful unavailable status.

## Phase 4 — US2: Page 05 tuning and final-evidence expanders (P1)

**Goal**: Readers can understand nested tuning, Full/Top-2 evidence and one-time holdout diagnostics without blending historical supplemental results.

- [X] T015 [US2] Add RED transcript/render-data tests in `tests/test_training_validation_views.py` for six tuning contexts, every trial/status/parameter/parent inner fold, completed/skipped/reused semantics, matched Full/Top-2 fold steps, final fit declaration, seven holdout roles, residual summary, encoded/permutation methods, subgroup support flags, q90 basis, scientific/operational separation, two final conclusions, source references and deterministic method order.
- [X] T016 [US2] Implement Page 05 tuning/variant and holdout/explainability transcript/view formatting plus two native collapsed renderers in `src/pages/training_validation_presentation.py`; aggregate raw predictions/permutation repeats only for method-labelled display while retaining complete validated original and transcript downloads.
- [X] T017 [US2] Add RED real-entrypoint AppTest cases in `tests/test_model_ui_pages.py` proving both Page 05 expanders appear before existing-source failure, show run/skip and exact fixture evidence, visibly exclude reserve results, preserve non-promotion language and retain existing Page 05 behavior.
- [X] T018 [US2] Insert minimal Page 05 renderer calls after title/caption and before `load_evidence()` in `src/pages/page05_best_model.py`; make T017 pass without changing existing tabs/charts/historical evidence.
- [X] T019 [US2] Add Page 05 no-fit/predict/load/write AppTest spies, original-log/report byte/hash checks and complete transcript row/source parity assertions; verify historical `ui_evidence` metrics never populate training-validation fields and vice versa.
- [X] T020 [US2] Test optional-section failure behavior for missing runtime limits, unavailable R², skipped tuning, small subgroups and unavailable estimator importance; show local reasons without favorable defaults or hiding other sections.
- [X] T021 [US2] Independently verify Page 05 with valid and unavailable fixtures and record exact source/value parity in `specs/007-show-training-validation/verification.md`.

**Checkpoint**: Page 05 independently delivers tuning/final evidence or truthful unavailable status.

## Phase 5 — US3: Truthful real-workspace unavailable state (P1)

**Goal**: The current blocked workspace explains absence without hiding existing page content or suggesting unsafe generation.

- [X] T022 [US3] Add RED AppTest/pure tests for the current no-pack state: exact `Training validation unavailable` heading, safe reason, `fallback_used=false`, read-only `inspect` command, eligible-data/approval next action and no executable run button.
- [X] T023 [US3] Implement shared unavailable rendering and safe exception-to-reason mapping in `src/pages/training_validation_presentation.py`; never render tracebacks, raw artifact content, credentials or external absolute paths.
- [X] T024 [US3] Test the four-source-state matrix (training validation available/unavailable × existing supplemental evidence available/unavailable) through Page 04 and Page 05; ensure one source never suppresses or fills the other.
- [X] T025 [US3] Exercise real `streamlit.py` against the current workspace and record that both pages render the unavailable expanders with zero generated packs/files in `specs/007-show-training-validation/verification.md`.

**Checkpoint**: The confirmed initial production state is honest, actionable and non-blocking for existing page content.

## Phase 6 — Documentation, regression and review

- [X] T026 [P] Update `docs/MODEL_UI.md` and the concise README guide section with the second evidence source, latest-valid selection, exact fold/model/tuning/final transcript contents, raw-log-versus-derived distinction, original/generated downloads, unavailable behavior, offline inspect/run/check boundary, scientific labels and no model activation.
- [X] T027 Run focused pure/AppTest suites, existing Page 04–06 tests, full pytest, Ruff and `git diff --check`; record exact outcomes/inherited failures in `specs/007-show-training-validation/verification.md`.
- [X] T028 Compare all T002 protected hashes and prove no training output/artifact, producer, pipeline, Page 06, config, dependency, pointer or `uv.lock` content changed; do not regenerate a pack for review.
- [ ] T029 Perform browser review at 1280×800 and 1440×900 through the real entrypoint if tooling is available, checking expander labels, table readability, no horizontal overflow and existing page layout; otherwise record pixel acceptance pending rather than infer success.
- [ ] T030 Conduct the five-minute two-reader Vietnamese/English walkthrough for run/unavailable identity, exact fold 3 periods/rows, expansion/protected counts, all 25 candidate-fold steps, selected/fastest distinction, five candidate decisions, raw-vs-derived log identity, tuning context/trial status, holdout/reserve scope, Full/Top-2 conclusions and safe action; never substitute automated tests for human acceptance.
- [X] T031 Run `graphify . --update --no-viz --code-only` (or document a valid exception), inspect the final diff across correctness/readability/architecture/security/performance, and synchronize `spec.md`, `plan.md`, contracts, checklist, tasks and verification.

## Phase 7 — Additive third-party trust overview

- [X] T032 Record the user's explicit approval to add an always-visible trust overview while preserving the existing Page 04/05 layout and keeping detailed evidence collapsed.
- [X] T033 Add failing real-entrypoint AppTest assertions for valid and unavailable overview states, verified evidence counts, and unchanged existing page behavior.
- [X] T034 Implement one removable native Streamlit overview helper in `src/pages/training_validation_presentation.py` and invoke it inside the existing Page 04/05 training-validation renderers without changing existing analytical sections.
- [X] T035 Synchronize the UI contract, specification, documentation, tests, and verification evidence; retain the no-training/no-fallback/non-activation boundaries.

## Phase 8 — Source-checkout CLI reliability

- [X] T036 Reproduce the reported `ModuleNotFoundError` when the module CLI is run without `PYTHONPATH` and preserve the blocked-inspection evidence.
- [X] T037 Add a failing subprocess regression test and a repository-local `training_validation.py` wrapper that inserts `src/` only for this CLI, then update the UI and operator documentation to use the wrapper.
- [X] T038 Verify the exact displayed command from the repository root without `PYTHONPATH`; require structured `RESERVE_KNOWN_EXPOSED`, exit code 3, and zero fits/predictions rather than an import failure.

## Phase 9 — Historical Branch B 5W1H report and logs (awaiting approval)

- [X] T039 Record explicit approval of this amended Phase 9 package and capture a fresh protected baseline for current outputs/artifacts, producer/pipeline/Page 06/config/dependencies before implementation.
- [X] T040 Add RED loader tests for the newest complete source-compatible primary audit, bounded model-operation event projection, run/source reconciliation, malformed event rejection, checksum/path containment, and byte-identical JSONL/manifest/export downloads in `tests/test_model_training_presentation.py`.
- [X] T041 Extend `src/pages/model_training_presentation.py` minimally to return validated audit-event previews and source/run descriptors without fitting, predicting, deserializing, writing, or weakening existing compatibility checks.
- [X] T042 Add RED pure tests for exactly five Page 04 candidate `Model5W1HRecord` rows, evidence-backed Who/What/When/Where/Why/How, metrics/units/roles/limitations/source parity, and rejection of unsupported causal/pristine claims in `tests/test_training_validation_views.py`.
- [X] T043 Implement deterministic historical candidate 5W1H builders and a compact highlighted Page 04 report/log renderer in `src/pages/training_validation_presentation.py`, sourcing active supplemental evidence plus the compatible audit while keeping strict validation independent.
- [X] T044 Add RED pure tests for Page 05 selected-family and Full/Top-2 5W1H records, inherited tuning settings/status, historical locked-test exposure, explainability/q90 limitations, non-promotion language, and exact source parity.
- [X] T045 Implement the additive Page 05 historical 5W1H/tuning/final report and bounded training-activity log using native containers/expanders and the visual hierarchy from `docs/spec-imporve-ui.md`; do not move or change existing sections.
- [X] T046 Extend real-entrypoint AppTest for the current workspace: historical report visible despite strict-pack unavailability, five candidate records, required final records, raw-vs-derived labels, complete downloads, unchanged existing charts/tabs, and zero fit/predict/run/load/write calls.
- [X] T047 Update `docs/MODEL_UI.md`, README and feature verification with the common-foundation/Branch A → Branch B sequence, historical/exposed evidence label, 5W1H semantics, log provenance, no cluster-feature claim, and no-retraining reuse decision.
- [X] T048 Run focused/full pytest, Ruff, `git diff --check`, protected-hash comparison and Graphify refresh; record browser/two-reader acceptance as pending unless actually completed.

## Dependencies and execution order

- T001–T003 precede all implementation.
- T004–T008 establish the only pack-discovery/validation path and block page work.
- US1 (T009–T014) and US2 (T015–T021) share the presentation module and should be implemented sequentially to avoid conflicting edits; each remains independently testable.
- US3 (T022–T025) validates the current real state after both page calls exist.
- T026 may proceed in parallel with final tests after UI wording stabilizes.
- T027–T031 are final gates. T029/T030 may remain explicitly pending if browser/human reviewers are unavailable; they cannot be fabricated.
- No task authorizes a real training-validation run. If a producer/schema change becomes necessary, stop and revise/approve the spec package before implementation.
- T039 gates T040–T048. Phase 9 reuses complete historical evidence and MUST NOT run the root pipeline or regenerate artifacts merely to populate the report.

## Requirement traceability

| Requirements | Tasks |
| --- | --- |
| FR-001–005, FR-010, FR-019 | T004–T008, T022–T024 |
| FR-006–007, FR-011–012 | T009–T014 |
| FR-008–009, FR-011–012 | T015–T021 |
| FR-013–015 | T008, T022–T025 |
| FR-016–018 | T002, T011–T014, T017–T019, T027–T028 |
| FR-020 | T026, T030–T031 |
| FR-021–022, FR-024–025 | T006–T007, T009–T014, T026, T030 |
| FR-023–025 | T006–T007, T015–T021, T026, T030 |
| FR-027–034 | T039–T048 |

**Status summary**: 46 completed, 2 pending. Phase 9 is implemented; only T029 browser/pixel review and T030 independent two-reader acceptance remain external gates.
