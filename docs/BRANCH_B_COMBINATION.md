# Branch B combination — FPTCranes-PRJ2-main + Technical Design

The uploaded `FPTCranes-PRJ2-main.rar` is retained under `docs/legacy_reference/`. The runtime could index the RAR5 archive directory but did not have a RAR5 decompressor available for reliable line-by-line extraction of compressed source files. Therefore this release does not pretend that every legacy implementation line was copied. Instead, the archive manifest was reviewed against the Technical Design and the Branch B structure/output contract was reproduced and combined in the current codebase.

The complete archive inventory is stored in:

- `docs/FPTCranes-PRJ2-main_manifest.csv`
- `docs/FPTCranes-PRJ2-main_branch_b_manifest.json`

The archive contains the following Branch B structure, among others:

- `src/training/ablation_study.py`
- `src/training/build_model_pipeline.py`
- `src/training/build_preprocessor.py`
- `src/training/model_catalog.py`
- `src/training/temporal_cv.py`
- `src/training/tuning.py`
- `src/pipeline/train_test_split.py`
- `src/pipeline/model_training_comparison.py`
- `src/pipeline/best_model_selection_feature_importance_review.py`
- `src/pages/_model_comparison.py`
- `src/pages/_best_model.py`
- `src/pages/_prediction.py`

## Combined behaviour in this release

The current release keeps one source of truth in `src/ai_job_market/core.py`, then provides thin compatibility facades under `src/training/` and `src/pipeline/`. This avoids duplicated modelling logic while preserving the clearer module organization of FPTCranes-PRJ2-main.

Branch B now generates the richer evidence contract visible in the archive manifest:

### B1–B3 — temporal split and ML readiness

- `08_before_after_processing.csv`
- `08_monthly_distribution.csv`
- `08_split_summary.csv`
- `08_training_readiness.json`
- encoded Train / locked-Test matrices and targets

### B4 — model comparison

- Dummy Median + Linear + Ridge + Random Forest + Gradient Boosting
- 5 expanding monthly folds
- fold-level MAE/RMSE/R²/MedAE
- runtime evidence
- feature-family ablation
- feature importance by fold and importance drift
- `09_*` compatibility outputs

### B5–B6 — tuning, locked test and explainability

- bounded `GridSearchCV` Random-Forest tuning after the family is frozen, using the declared chronological inner folds and `n_estimators` × `max_depth` grid
- one-time locked future evaluation
- actual vs predicted and residual evidence
- raw permutation importance
- encoded impurity importance
- subgroup error slices
- empirical q90 absolute-error band
- `10_*` compatibility outputs

### B7 — deployment contract

- end-to-end `model_bundle.joblib`
- train-fitted preprocessing artifacts
- metadata and exact feature order
- reload equivalence evidence
- `11_deployment_artifact_manifest.csv`
- `12_locked_test_prediction_examples.csv`
- `12_prediction_summary.csv`

The Streamlit Model Comparison and Best Model pages read these output artifacts rather than fitting models inside the presentation layer.
