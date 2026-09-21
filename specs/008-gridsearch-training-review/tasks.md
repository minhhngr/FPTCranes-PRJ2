# Tasks: Review Branch B training and expose GridSearchCV evidence

**Input**: `specs/008-gridsearch-training-review/`  
**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `contracts/`  
**Status**: Approved. Offline training/validation scope implemented; additional UI tasks are deferred by the latest user boundary.

**Latest scope note**: The user explicitly requested no additional UI work. Existing inherited Page 04/05 edits are preserved. T011–T016 and browser-only T021 are therefore not implementation requirements for this training-only pass.

## Phase 1 — Baseline and contract

- [X] T001 Record inherited dirty changes and verification status in `specs/008-gridsearch-training-review/verification.md`; do not alter `docs/legacy_reference/*` or `src/constants/*`.
- [X] T002 Add contract tests in `tests/test_training_partitions_v2.py` for a GridSearchCV temporal splitter: context-only positional indices, chronology, disjointness, and protected-partition exclusion.
- [X] T003 Add contract tests in `tests/test_training_search_v2.py` requiring `GridSearchCV`, `n_estimators` × `max_depth` candidates, declared scorer/ranking, fold metrics/timing/status, and conditional skip behavior.
- [X] T004 Update `src/ai_job_market/training_partitions.py` with the smallest validated temporal CV adapter that exposes existing `FoldDeclaration` objects to scikit-learn without random splitting; make T002 pass.
- [X] T005 Replace manual sequential search in `src/ai_job_market/training_search.py` with a fold-local pipeline `GridSearchCV` implementation and result-to-evidence mapper; make T003 pass.

**Checkpoint**: Tuning uses declared chronological folds and GridSearchCV without protected-row access.

## Phase 2 — Offline producer, leakage gates, and artifacts

- [X] T006 Extend `tests/test_training_validation_cli.py` for GridSearchCV event/grid evidence and fail-closed known-exposed reserve behavior.
- [X] T007 Update `src/ai_job_market/training_validation.py` to supply context-local frames/folds to GridSearchCV and emit raw search/event evidence; existing chronology, approval, and exposure gates remain fail-closed.
- [X] T008 Add revised-schema validation tests in `tests/test_training_evidence_v2.py` and `tests/test_training_validation_views.py` for GridSearchCV method identity, required evidence fields and invalidation state.
- [X] T009 Update `src/ai_job_market/training_evidence_io.py` and `config/training_validation.json` for the GridSearchCV policy contract.
- [X] T010 Validate synthetic eligible fixture packs in tests and record the current real workspace’s exposed-reserve block in `specs/008-gridsearch-training-review/verification.md`; no real trusted evaluation was run.
- [X] T010a Emit a deterministic detailed English `training.log` and structured stage events covering 5W1H, requested/actual split, folds, five-model metrics/runtime/fit status, feature evidence, conditional GridSearchCV, variants, holdout/residuals, explainability, subgroup/q90 evidence, and final status with source-consistent numbers.

**Checkpoint**: The offline producer has a tested revised evidence contract, and the real workspace’s limitation is explicit.

## Phase 3 — Page 04: split/fold/candidate report (US1)

- [ ] T011 [US1] Add failing pure/AppTests in `tests/test_training_validation_views.py` and `tests/test_model_ui_pages.py` for Page 04 outer/inner GridSearchCV fold periods/counts, exclusion/chronology evidence, candidate R²/MAE/RMSE/MedAE, charts, raw/derived step logs, downloads, and invalidated state.
- [ ] T012 [US1] Update `src/pages/training_validation_presentation.py` to render Page 04’s compact visual summary and end-of-page collapsed split/fold/candidate/log evidence from the revised contract; make T011 pass without UI model operations.
- [ ] T013 [US1] Change `src/pages/page04_model_comparison.py` only if required to place the new detail section after existing content; preserve existing charts/KPIs/tabs.

## Phase 4 — Page 05: GridSearchCV/final report (US2)

- [ ] T014 [US2] Add failing pure/AppTests in `tests/test_training_validation_views.py` and `tests/test_model_ui_pages.py` for scorer/grid/candidate rank, inner-fold allocation, winner `n_estimators`/`max_depth`, R²/final scope labels, charts, raw/derived step logs, downloads, and invalidated state.
- [ ] T015 [US2] Update `src/pages/training_validation_presentation.py` to render Page 05’s visual search summary and end-of-page collapsed GridSearchCV/final/log evidence; make T014 pass without UI model operations.
- [ ] T016 [US2] Change `src/pages/page05_best_model.py` only if required to place the new detail section after existing content; preserve existing charts/KPIs/tabs.

## Phase 5 — Serving safety (US3)

- [ ] T017 [US3] Add Page 06 regression tests in `tests/test_salary_inference.py` and `tests/test_model_ui_pages.py` proving the existing serving contract remains compatible and incompatible metadata/bundles fail closed.
- [ ] T018 [US3] Modify `src/pages/page06_prediction.py` and/or serving reader only if T017 exposes a required revised-contract compatibility gap; otherwise record no code change.

## Phase 6 — Documentation, verification, and review

- [X] T019 Update `docs/BRANCH_B_COMBINATION.md`, `docs/TRAINING_VALIDATION.md`, and `docs/MODEL_UI.md` with GridSearchCV, chronological nesting, current exposure blocker, UI labels, and Page 06 compatibility boundary.
- [X] T020 Run focused tests, full pytest, scoped Ruff, `git diff --check`, inspection, and a code-only Graphify refresh; record exact outcomes and inherited failures in `specs/008-gridsearch-training-review/verification.md`.
- [ ] T021 Use browser testing at 1280×800 and 1440×900 for Pages 04–06; verify chart labels, collapsed evidence placement, table readability, no overflow, unavailable state, and console errors. Record results.

## Phase 7 — Approved historical full-pipeline verification

- [X] T022 Record the user's approval to run `pipeline.py --debuglog` against `data/raw/ai_jobs_market_2025_2026.csv` as historical/exposed evidence; do not relabel its locked test as pristine or bypass strict reserve validation.
- [X] T023 Capture pre-run output/artifact hashes, execute the full pipeline, and retain the completed audit run and regenerated model artifacts.
- [X] T024 Regenerate stale supplemental UI evidence, add the missing source-backed training-period projection, and validate the active evidence pointer.
- [X] T025 Verify Pages 04–05 with AppTest, including latest run identity, exposure warning, five fold rows, model/final 5W1H sections, and training-log downloads.

## Dependencies

- T001–T005 block producer and UI work.
- T006–T010 block Page 04/05 consumption because the evidence schema changes.
- T011–T013 and T014–T016 can proceed after T010 but share a presentation module and should be sequential in one worktree.
- T017–T018 can run after T010 and is independent of Page 04/05 rendering.
- T019–T021 follow completed implementation.

## Artifact decision

T010 is mandatory: changed producer/consumer logic invalidates manual-search evidence. It explicitly prohibits retraining the current known-exposed real workspace. A fresh approved eligible dataset is required before a real immutable evidence pack can be published.
