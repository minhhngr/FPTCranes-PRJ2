# Implementation Plan: Review Branch B training and expose GridSearchCV evidence

**Branch**: `feat/verify-008` | **Date**: 2026-09-20 | **Spec**: [spec.md](spec.md)  
**Status**: Offline training/validation implementation approved and completed; additional UI work is outside the latest scope.

## Summary

Replace the offline `training-validation/v1` manual sequential Random Forest search with a deterministic `GridSearchCV` adapter over declared chronological inner folds. Version/invalidate affected evidence and emit detailed English 5W1H, split, fold, metric, tuning, explainability and conclusion logs for later read-only consumption. Preserve inherited Page 04/05 edits without adding UI work. Do not activate training from Streamlit or claim a fresh locked-test result from the current known-exposed population.

## Technical Context

**Language/Version**: Python 3.13; Ruff target py312  
**Dependencies**: Existing pandas, NumPy, scikit-learn (includes `GridSearchCV`), Streamlit, Plotly  
**Storage**: Versioned `outputs/training_validation/`, existing `outputs/04_model_comparison/`, `outputs/05_best_model/`, `artifacts/`  
**Testing**: pytest and Streamlit `AppTest`  
**Platform**: Offline Linux workspace and Streamlit UI  
**Constraints**: Training only offline; chronological target-safe folds; no UI fit/search/predict; only Pages 04–06 and training/business code in scope; preserve inherited dirty files.

## Research decisions

1. `training_search.py` uses `stepwise_rf_search`, a hand-written anchor/coordinate sweep; it does not use `GridSearchCV`. This fails the requested method requirement.
2. `training_partitions.py` already creates chronological expanding fold declarations and validates direct fold overlap/chronology. A custom scikit-learn CV iterable can expose these declared train/validation positions to GridSearchCV without using random K-fold behavior.
3. GridSearchCV must receive a fold-local pipeline and a negative-MAE scorer (or explicit scorer documented in evidence). Its parameter names must target the pipeline’s Random Forest estimator. The final exact parameter grid will be the policy’s `n_estimators` × `max_depth` combinations; `min_samples_leaf` and `max_features` stay frozen only if explicitly recorded.
4. Nested tuning contexts remain five outer-training contexts plus final TRAIN. Family comparison remains outer-CV-only; no evaluation-holdout/reserve row is passed into search.
5. The current `inspect_workspace()` reports `RESERVE_KNOWN_EXPOSED`, because committed historical locked-test artifacts overlap the proposed reserve. This is a hard evidence blocker: do not execute a new trusted run against this population. Code/schema changes invalidate existing results for claims under the revised method.
6. Existing Page 04/05 already have read-only training-validation renderers. Extend the reader schema/render projections rather than add another UI pipeline. Place new details at the end of existing analytical content, preserving the requested report-first layout.
7. Page 06 currently consumes a separate Top-2 serving evidence/bundle. Leave it untouched unless the final artifact schema changes; test compatibility instead.

## Constitution Check

| Gate | Design disposition |
| --- | --- |
| Data integrity | Pass by using explicit temporal fold positions and fold-local pipelines. New trusted holdout metrics are blocked without fresh eligible data. |
| Reproducibility | Seed, grid, scorer, fold IDs, ranks, timings, source hashes, schema version, and invalidation reason are emitted. |
| Verification | RED tests precede implementation; synthetic offline fixture covers full producer contract; AppTests cover readers. |
| Streamlit boundary | UI is read-only. No run button or training invocation. |
| Input validation | Offline approval/policy/data/grid/schema/path values validate fail-closed. |
| Scientific claims | CV, historical test, and new holdout labels remain separate; no new untouched-test claim on current data. |
| Documentation | Training, Branch B, and UI evidence docs update with method and invalidation decision. |
| Graphify/Karpathy | Existing graph query identifies a surgical training-search → producer → reader path. |

## Project Structure

```text
src/ai_job_market/
├── training_partitions.py          # add GridSearchCV-compatible temporal split adapter/validation
├── training_search.py              # replace manual sweep with GridSearchCV and evidence mapping
├── training_validation.py          # pass context-local data/folds; write versioned evidence/events
├── training_evidence_io.py         # revised schema validation/invalidation handling if required
└── training_report.py              # method/report wording if required
src/pages/
├── training_validation_presentation.py  # render revised grid/fold/log projections
├── page04_model_comparison.py           # only placement adjustment if needed
├── page05_best_model.py                 # only placement adjustment if needed
└── page06_prediction.py                 # unchanged unless serving schema changes
tests/
├── test_training_search_v2.py
├── test_training_partitions_v2.py
├── test_training_validation_cli.py
├── test_training_validation_views.py
├── test_model_ui_pages.py
└── test_salary_inference.py
config/training_validation.json          # declared grid/schema only if required
specs/008-gridsearch-training-review/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/training-gridsearch-evidence.md
└── tasks.md
```

**Structure decision**: Keep the existing producer/consumer boundaries. Add one narrow CV adapter and one evidence mapper; do not add a new pipeline, UI component framework, or model-serving route.

## Priority amendment — visible current-pipeline evidence

The visible primary UI is the latest source-compatible Branch B pipeline evidence already produced under the active supplemental/audit contract. Pages 04–05 present this first as `Latest pipeline training validation`, retain its historical/exposed locked-test warning, and expose the model and raw step evidence in collapsed expanders. Strict `training-validation/v1` remains separate and truthfully unavailable until eligible data exists; it does not suppress the visible current-pipeline report.

## Artifact reuse / invalidation decision

Training producer behavior and the evidence consumer contract change. Therefore existing training-validation and any generated UI evidence representing the old manual-tuning method cannot be reused as evidence for this feature. Mark them invalid for revised-method claims. Generate a new immutable pack only after an eligible data source, non-exposure proof, and fresh exact approval are available. Existing Page 06 serving artifacts may be reused only if their schema and contract remain unchanged.

## Verification plan

1. Record protected hashes/status and inherited dirty files; do not alter the deleted archives or untracked constants.
2. Add RED unit tests for temporal GridSearchCV split positions, parameter grid, scorer direction, pipeline-local preprocessors, deterministic ranks, failed candidates, and no protected-row access.
3. Implement the smallest adapter/search mapping; run focused tests.
4. Add RED producer-contract tests for evidence version, complete candidate/fold/log fields, invalidation state, and fail-closed current-data inspection.
5. Implement producer/report/schema changes; run synthetic eligible-fixture generation and validate the pack.
6. Add RED Page 04/05 AppTests and pure view tests for fold/grid/step projections, charts, downloads, and unavailable/invalidation state; implement reader changes.
7. Add Page 06 regression/compatibility tests; modify Page 06 only if schema migration makes this necessary.
8. Run focused tests, full pytest, Ruff, `git diff --check`, artifact hash/invalidation checks, and browser review on Pages 04–06.
9. Run `graphify . --update --no-viz` after code changes and record results.

## Complexity Tracking

No exception is proposed. A custom CV iterable is necessary because standard random/default fold splitters would violate temporal fold declarations; it is smaller and safer than replacing the established partition system.
