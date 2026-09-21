# Tasks: English/Vietnamese internationalization

**Input**: Design documents from `/specs/009-add-i18n-support/`  
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/i18n-resource-and-output-localization.md`  
**Tests**: Write tests first for every resolver, mapping, state, and rendering behavior. No offline training is needed; existing outputs are reused read-only.

## Phase 1: Setup and inventory

- [ ] T001 Record the inherited dirty-worktree boundary and create a complete static-string/controlled-output inventory for `streamlit.py`, `src/components/`, and `src/pages/` in `specs/009-add-i18n-support/` task notes or a linked inventory document.
- [ ] T002 [P] Define the complete EN key namespace and controlled output header/value allowlist in `config/language/en.json` according to `contracts/i18n-resource-and-output-localization.md`.
- [ ] T003 [P] Create the matching Vietnamese resource in `config/language/vi.json` with identical required key paths and Vietnamese display values.
- [X] T004 [P] Add failing tests for resource discovery, EN/VI key parity, malformed resource reporting, named interpolation, and invalid-language fallback in `tests/test_i18n.py`.
- [X] T005 [P] Add failing tests for known DataFrame/JSON output header/value mapping and unknown/free-form passthrough in `tests/test_i18n.py`.

## Phase 2: Foundational i18n boundary

- [X] T006 Implement the smallest importable resource loader, language validator, static translator, interpolation safety, and controlled output resolver in `src/ai_job_market/i18n.py` to satisfy T004–T005.
- [X] T007 Implement display-only DataFrame/JSON localization adapters in `src/pages/common.py`; preserve raw frames for filtering/calculation/downloads and raw artifact data on disk.
- [X] T008 Add failing `AppTest` coverage for default EN, validated selector state, invalid state fallback, and language persistence across page navigation in `tests/test_model_ui_pages.py` (or a focused new Streamlit UI test module).
- [X] T009 Initialize the validated shared language session state and add the keyed native selector in the sidebar/settings region of `streamlit.py` before authentication/navigation; localize sidebar title, caption, role navigation labels, and active-workspace label.
- [X] T010 Run focused `tests/test_i18n.py` and AppTests; demonstrate tests were RED before T006/T009 and pass afterward.

## Phase 3: User Story 1 - Switch the application language (Priority: P1) 🎯 MVP

**Goal**: An authenticated user selects EN/VI once and receives a consistent shared sidebar/navigation experience throughout the active session.

- [X] T011 [US1] Migrate static authentication/status/form/error text in `src/components/auth.py` to translation keys and preserve the current role/credential validation behavior.
- [X] T012 [US1] Migrate workspace/upload/schema-gate/progress/error text in `src/components/data_source.py` to translation keys; do not change pipeline invocation, upload validation, or workspace behavior.
- [X] T013 [US1] Extend AppTests in `tests/test_model_ui_pages.py` to authenticate, change language, navigate, and assert EN/VI shared controls plus persistent session state.
- [ ] T014 [US1] Run focused Streamlit AppTests for shared selector/auth/workspace behavior and manually verify both languages without invoking a pipeline run.

## Phase 4: User Story 2 - Read every application page in the selected language (Priority: P1)

**Goal**: Every static text element available to each role is resource-driven in EN and VI.

- [ ] T015 [P] [US2] Migrate static titles, controls, messages, tables, and chart labels in `src/pages/page01_data_basic_clean.py` and `src/pages/page02_data_ready.py` to translation keys and use display-only localization adapters.
- [ ] T016 [US2] Migrate the high-volume segmentation UI in `src/pages/page03_segmentation.py` to translation keys and localized display adapters, preserving raw filter/data columns and calculations.
- [ ] T017 [P] [US2] Migrate model evidence/shared narrative and training presentation text in `src/pages/model_evidence.py` and `src/pages/model_training_presentation.py` to translation keys.
- [ ] T018 [US2] Migrate model comparison and best-model UI text in `src/pages/page04_model_comparison.py` and `src/pages/page05_best_model.py`, including chart/table/download labels.
- [ ] T019 [US2] Migrate prediction and integrated-insight UI text in `src/pages/page06_prediction.py` and `src/pages/page07_integrated.py`, preserving input validation, queue state, and inference behavior.
- [ ] T020 [US2] Migrate full-pipeline and compatible training-validation presentation text in `src/pages/page08_full_pipeline.py` and `src/pages/training_validation_presentation.py`; preserve unavailable/evidence semantics and raw downloads.
- [ ] T021 [US2] Add/adjust EN- and VI-derived assertions for every role-available page in `tests/test_model_ui_pages.py` and any affected page-specific test modules; update pre-existing literal-English tests only as necessary.
- [ ] T022 [US2] Add an automated static-string guard or documented inventory-based test covering `streamlit.py`, `src/components/`, and `src/pages/`, allowing only approved non-display/raw-evidence exceptions.
- [ ] T023 [US2] Run all affected AppTests and inspect each page in both languages for localized title, control, message, chart/table/status text and no Streamlit exception.

## Phase 5: User Story 3 - Localize known evidence/output values safely (Priority: P2)

**Goal**: Known generated values render in EN/VI without mutating outputs or translating unknown content.

- [ ] T024 [US3] Apply controlled output localization to generic readers/table/chart render paths in `src/pages/common.py` and `src/pages/model_evidence.py`; add fixture coverage for known headers/statuses and unknown pass-through in `tests/test_i18n.py`.
- [ ] T025 [US3] Apply the explicit mapping boundary to specialized compatible training/audit readers in `src/pages/model_training_presentation.py` and `src/pages/training_validation_presentation.py`; preserve raw manifest paths, schema validation, evidence references, and download bytes.
- [ ] T026 [US3] Add integration coverage using representative existing `outputs/` fixtures/workspaces to prove known labels localize while raw source frames/files and unknown values stay unchanged in the relevant `tests/test_*` modules.
- [ ] T027 [US3] Run the output-reader, audit, model-page, and prediction regression tests; verify no training-producing module, generated output, or artifact was changed.

## Phase 6: User Story 4 - Maintain translations centrally and extend them safely (Priority: P3)

**Goal**: Maintainers can add a language/resource term centrally with parity protection and no page-specific language branching.

- [X] T028 [US4] Add a resource-validation/add-language test case in `tests/test_i18n.py` proving a complete registered test language resolves without a page-module edit.
- [X] T029 [US4] Document resource key conventions, controlled output mappings, fallback behavior, raw-value preservation, and add-language steps in `docs/STREAMLIT_CONTENT_MAP.md` and/or a dedicated i18n maintenance document under `docs/`.
- [X] T030 [US4] Add concise module/function documentation in `src/ai_job_market/i18n.py` and `src/pages/common.py` describing inputs, output invariants, fallback, and failure behavior.

## Phase 7: Polish and verification

- [ ] T031 Run the focused i18n tests, all affected page/component tests, and the relevant full pytest suite; record commands/results in `specs/009-add-i18n-support/quickstart.md` or verification notes.
- [ ] T032 Run Ruff and `git diff --check`; inspect the diff to ensure inherited feature-008 changes, raw outputs, training producers, schemas, and downloads were not included or altered.
- [ ] T033 Perform the manual EN/VI Streamlit smoke test in `specs/009-add-i18n-support/quickstart.md`, including navigation, output mappings, unknown values, and no pipeline execution.
- [X] T034 Run `graphify update .` after code changes and record the affected-file/relationship result in feature verification notes.
- [ ] T035 Update `specs/009-add-i18n-support/spec.md`, `plan.md`, and this `tasks.md` if implementation changes scope, mapping coverage, resource contract, or validation evidence; do not modify inherited `AGENTS.md` work without maintainer direction.

## Increment status — 2026-09-21

User approved implementation. The first increment covers the core and shared UI, **not
full i18n acceptance**. See `verification.md`. `text-inventory.csv` lists 5,539 literal
candidates (including technical/non-display strings) across 18 source files; it is not
an allowlist of approved untranslated UI. T001–T003 remain open until page/output terms
are classified and the full EN/VI catalog is populated. Resource tests and extensions
were validated against the initial shared-UI vocabulary; parity alone does not prove
full application coverage. T007's pure frame/metadata adapters live on `Translator`
and are consumed by common helpers (documented design refinement).

Tests: 228 passed, 1 skipped, 2 inherited release-contract failures. No browser tool
was available, so manual smoke testing remains unchecked. Do not mark T015–T027 or
feature-wide acceptance complete based on the shared UI tests.

## Dependencies and execution order

- T001–T005 establish the inventory/resources/tests; T006–T010 are the blocking shared foundation.
- User Story 1 depends on T006–T010.
- User Story 2 depends on the shared resolver/adapters and can migrate file groups incrementally; Page 03 must be serialized with any edits that touch its shared helpers.
- User Story 3 follows the stable display adapters and can be validated alongside the associated page migrations.
- User Story 4 follows resource validation and is independent of output-reader migration after the shared API is stable.
- Phase 7 follows all desired user stories.

## Parallel opportunities

- T002–T005 may proceed in parallel after the inventory contract is agreed.
- T015 and T017 can proceed in parallel on disjoint files; T018–T020 proceed after shared helper/API stabilization and must avoid overlapping file edits.
- Resource tests/documentation may progress in parallel with independent page migrations once T006–T007 are complete.

## Implementation strategy

1. Deliver the shared EN/VI selector and resource resolver first (US1) and validate it through the real entrypoint.
2. Migrate page text in small, test-backed file groups (US2), keeping raw data logic untouched.
3. Add the allowlisted output-mapping layer and fixtures (US3), then document extension operations (US4).
4. Do not train or regenerate outputs: this feature reuses them read-only. Stop immediately for user review before beginning any task.
