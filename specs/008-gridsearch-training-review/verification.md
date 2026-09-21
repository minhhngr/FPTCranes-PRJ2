# Verification record

## Scope and inherited state

- The user approved implementation on 2026-09-20 and explicitly limited the latest pass to training/validation rather than additional UI work.
- Inherited, untouched changes: deleted `docs/legacy_reference/FPTCranes-PRJ2-main.rar`, deleted `docs/legacy_reference/FPTCranes-PRJ2_full(3).7z`, and untracked `src/constants/`.
- No root pipeline run or real artifact regeneration was performed.

## Implemented evidence

- `GridSearchCV` now tunes Random Forest over the declared `n_estimators` × `max_depth` grid using a validated adapter over declared chronological inner folds.
- The adapter rejects missing identities, empty fold sides, chronology failures, and train/validation overlap. It only maps positions represented by the tuning context’s declared inner folds.
- The search records method identity, parameter values, negative-MAE ranking, fold-level and mean/standard-deviation MAE/R², timing, rank, inner-fold IDs, status, failure reason, and evidence reference. The offline writer emits GridSearchCV context and candidate events.
- Complete manifests require `training_method_version=gridsearchcv-temporal/v1`; prior manual-search packs fail validation for revised-method claims.
- `training.log` is now a deterministic detailed English method record sourced from the same persisted values. It answers 5W1H and records requested/actual partition shares, every outer/inner fold, all 25 candidate-fold evaluations, runtime and underfit/overfit/good-fit indications, selection, ablation/importance/drift, each GridSearchCV candidate or skip, Full/Top-2 evidence, all holdout roles, residuals, encoded/permutation importance, subgroup slices, q90 bands, and final scientific/operational status.
- Raw lifecycle events now include partition validation, five-model comparison, GridSearchCV candidate/skip activity, and frozen final evaluation. The temporal splitter explicitly rejects protected-partition rows and row/month declaration mismatches.
- Page 04/05 training-validation details are rendered after normal analytical content (or before the existing error return). Page 05 suppresses old/manual tuning tables as revised-method evidence and renders a native GridSearchCV MAE chart/table only when the revised fields are present.
- Page 06 source code was not changed; existing regression tests passed.

## Current workspace training decision

Command:

```bash
PYTHONPATH=src .venv/bin/python training_validation.py inspect --workspace . --policy config/training_validation.json
```

Result: `RESERVE_KNOWN_EXPOSED`; source `outputs/02_data_ready_for_ml/locked_test_raw.csv`; zero fits and predictions. This integrity guard remains intentionally active: administrator permission does not make an exposed population newly unseen. Existing manual-search evidence must not be presented as revised-method training evidence.

## Automated verification

Passed:

```text
71 passed
PYTHONPATH=src .venv/bin/pytest tests/test_training_partitions_v2.py tests/test_training_search_v2.py tests/test_training_evidence_v2.py tests/test_training_validation_cli.py tests/test_training_validation_views.py tests/test_model_ui_pages.py tests/test_salary_inference.py -q
```

Passed:

```text
PYTHONPATH=src .venv/bin/pytest -q
205 passed, 1 skipped; two failures were unrelated inherited workspace defects:
- `tests/test_release_contract.py::test_legacy_main_archive_manifest_packaged` expects the inherited-deleted RAR.
- `tests/test_page05_branchb_fold_log.py::test_page05_shows_branch_b_fold_method_before_tuning_detail` expects historical `train_period` data absent from the active supplemental `candidate_fold_metrics.csv` artifact.
```

Passed for changed files:

```text
uv run ruff check <changed training, page, and test files>
git diff --check
```

Full-repository Ruff still reports four pre-existing unrelated import-format errors in `src/pipeline/best_model_selection_feature_importance_review.py`, `src/training/temporal_cv.py`, `src/training/tuning.py`, and `tests/test_segmentation_representation_robustness.py`; they were not changed.

Graph refresh: the compliant code-only refresh succeeded with `graphify . --update --no-viz --code-only` (1,593 nodes, 3,211 edges). Graphify warned that four configuration/manifest files produced zero nodes; no source extraction failure occurred.

## Approved historical full-pipeline run and UI check

The user explicitly approved using `data/raw/ai_jobs_market_2025_2026.csv` as historical, no-longer-future input for a full-pipeline verification. This does not waive or bypass the separate strict future-reserve guard.

Command:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python pipeline.py \
  --data data/raw/ai_jobs_market_2025_2026.csv --debuglog
```

Result:

- completed run: `20260920T161210Z`
- raw/clean rows: 1500 / 1499
- development / historically scored locked test: 1201 / 298
- selected family: Random Forest
- historical locked-test MAE: 14,735.126 USD/year
- historical locked-test RMSE: 29,111.296 USD/year
- historical locked-test R²: 0.812690
- historical locked-test MedAE: 4,346.638 USD/year
- elapsed: approximately 3 minutes 9 seconds
- audit directory: `outputs/08_full_pipeline/logs/training-20260920T161210Z-f2d4b5791be34632915eeeb591c0c95e/`

The historical full pipeline retains its inherited manual sequential tuning semantics. These outputs are not labelled as the revised strict GridSearchCV pack.

The source change made the supplemental UI evidence stale as expected. It was regenerated as `ui-8748556ffa87b29d`; `ui_evidence --check` reports `valid: true`. The producer now persists each fold's source-derived `train_period`, allowing Page 05 to display the complete five-row train → validation method table.

UI evidence:

```text
17 passed
PYTHONPATH=src .venv/bin/pytest tests/test_model_ui_pages.py tests/test_page05_branchb_fold_log.py -q
```

Direct AppTest inspection confirmed both Pages 04 and 05 render `Latest pipeline training validation`, the historical/exposed warning, run metric `20260920T161210Z`, and their model/final evidence and activity-log expanders. Page 05 also renders `Branch B · How the temporal DEV folds are split` with five source-backed rows.

`validate_release.py` completed 33/36 checks. The three failures are inherited: two deleted legacy archives and the validator's static interpretation-card detector for Pages 04–06. Because it wrote `outputs/validation_report.json` with `overall=FAIL`, the full suite now reports 205 passed, 1 skipped and two release-contract failures: the inherited missing RAR and the expected non-PASS validation report. Focused UI and changed-file Ruff checks pass.

## Pending external validation

- Browser/pixel review was not run because no Streamlit process was started without a separate request.
- A real immutable revised pack requires new eligible unexposed data and a fresh approval; it cannot be generated from the current workspace.
