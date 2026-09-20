# Read-Only Evidence Review: Model UI Improvement

**Date**: 2026-09-19  
**Input**: `docs/spec-imporve-ui.md`  
**Purpose**: Distinguish design intent from verified evidence before approving a technical plan. This is specification-phase due diligence, not a completed `/speckit.assess.research` phase or runtime validation.

## Inspection method

- Queried the existing Graphify graph using vocabulary-backed terms `comparison prediction metadata bundle temporal tuning`; its 1,400-token response was truncated (51 of 119 nodes shown), so direct source inspection followed. Graph locations are navigation hints, not current evidence provenance.
- Read `src/pages/page04_model_comparison.py`, `src/pages/page05_best_model.py`, `src/pages/page06_prediction.py`, the current constitution, previous training-log plan, metadata and relevant CSV/JSON artifacts.
- Read `src/ai_job_market/core.py` evaluation logic: the inspected `evaluate_model_cv` path fits on training rows, predicts validation rows and returns validation-only fold/summary metrics. It does not produce the `Train_*` fields now present in comparison CSVs.
- Counted CSV rows/columns and inspected title/category/experience associations using read-only standard-library CSV processing. No fitting, inference, artifact regeneration, dependency installation, application launch or test-suite execution occurred.

## Findings and required disposition

| Topic | Stakeholder document | Current evidence | Specification treatment |
| --- | --- | --- | --- |
| Candidate validation ranking | RF $15,662; GB $16,421; Ridge $28,057; Dummy $54,011; Linear $59,006 | Rounded values match `outputs/04_model_comparison/model_comparison.csv` | Derive ranking and values from approved artifacts, never literals |
| Train scores | Complete train MAE/R² and fold scores | `Train_*` columns exist in both named and numbered comparison/fold CSVs, but the inspected current producer does not emit them | Presence alone does not establish reproducibility; establish run/source provenance before use, otherwise unavailable. Q1 determines whether missing provenance can trigger a separately approved artifact workstream |
| Final fold size | Common description: 200 rows per block | Fold CSVs have 200 training rows and 201 validation rows in fold 5 | Report actual fold sizes and avoid strict nonoverlapping-month claims |
| Tuned CV | MAE $15,594; R² 0.824; MedAE $7,280 | These values appear in `outputs/05_best_model/manual_tuning_step1_gridsearch.csv` for candidate 4 | Match applied configuration to trial before contrasting with test; this review does not establish every added tuning field's producer provenance |
| Test population | 295 benchmark rows plus three pristine reserved records | `artifacts/metadata.json` says 298 test rows; `outputs/05_best_model/locked_test_predictions.csv` has 298 rows; `outputs/06_salary_prediction/12_prediction_summary.csv` also says 298 | Do not label existing metrics as a 295-row benchmark or invent an excluded subset |
| Full-model test metrics | MAE $14,735; R² 0.813; MedAE $4,347; RMSE $29,111 | Match rounded values in `outputs/05_best_model/locked_test_metrics.json`, but for current 298-row evidence | Preserve values and correct population identity |
| Uncertainty | Full q90 ±$40,068; Top-2 ±$34,101 | Full q90 is 40067.74765859205 in metadata/metrics; `core.py` computes q90 from test absolute errors | Label full-model band as historical test-error q90, not independent coverage validation or a guaranteed future interval; no verified Top-2 band found |
| Raw permutation reliance | job category +$44,120; experience +$22,350; country +$4,890 | `outputs/05_best_model/raw_permutation_importance.csv`: 49598.14149, 12378.99919 and 8.53856 USD respectively | Use recorded values and variability, preserve even negative values, prohibit copied claims |
| Encoded feature count | 93 encoded columns | `outputs/05_best_model/encoded_importance.csv` has 193 rows | Scope count to fitted configuration/partition; do not reuse a global 93 literal or infer fold dimensionality from final-model count |
| Fold importance | Category 48.2%, experience 35.4% | Drift file is encoded-feature-level; leading entry is `nominal__job_category_AI Engineering` at 0.616207 mean; experience at 0.242634 | Do not call encoded-feature importance a raw-family weight or sum marginal standard deviations to infer family drift |
| Top-2 serving | Two-feature model and independent metrics | `artifacts/feature_contract.json` and metadata list 13 features; `artifacts/model_bundle_top2.joblib` absent; referenced Top-2 comparison/summary files absent | Block incompatible Top-2 serving, do not silently feed two columns to full bundle; Q1 required |
| Pristine records | Three named examples, excluded from training/tuning/scoring | Referenced `reserved_3_test_records_for_salary.csv` and Top-2/full reserved prediction files are absent from Pages 05/06 output packs | No synthetic replacements and no retroactive purity claim; Q1 required |
| Canonical categories and titles | Example categories AI Research/Machine Learning/Data Architecture; Principal Data Architect title | Metadata uses `Research`, `ML Operations`, `Architecture` among 12 categories and 25 titles; Principal Data Architect is absent | Metadata spellings govern categorical choices; examples are not approved enum values |
| Experience guards | Scientist bounds 4–12; quick-load scientist 3 years; curves 0–15 | Full basic-clean data reproduces the four illustrated title-level ranges/medians; metadata global minimum is 1; scientist example violates its own guard | Explicit Q2 for uniform bounds versus declared benchmark/extrapolation exceptions; no silent clamp of audited records |
| Role mapping | Category-filtered valid titles | No mapping in metadata. Basic-clean scientist rows span nine categories, not simply Research; Agent Developer and Compliance Manager rows each span all 12 categories | Explicit Q3 for observed associations versus approved curated semantics; do not infer business taxonomy from names |
| Fit diagnoses | Good fit, catastrophic overfit and mathematical/causal root causes asserted as proven | Error contrast is descriptive; no reviewed matrix-condition evidence, unbiasedness test or external validation supports those conclusions | Evidence-qualified interpretation only; do not claim plotted trajectories prove no memorization, no bias or production fitness |
| Serving page today | Metadata-driven two-feature safety | Page 06 contains static Top-2 R²/band/model labels and a full-bundle fallback; Page 05 contains hardcoded 295-row/Top-2 narratives | Existing UI prose is not scientific evidence; redesign must remove or qualify unsupported claims |
| Export/error semantics | Signed variance and enterprise-schema prediction export | Current page computes absolute percentage error; separate queue export uses source columns, while result export has derived fields | Explicit signed variance and separate source/audit contracts needed; compatibility migration must be planned, not silently reinterpret existing `error_pct` |

## Clarification choices

### Q1 — Artifact scope

**Recommended A**: Preserve the no-training-change boundary. Plan an artifact-backed UI with explicit unavailable sections and disabled Top-2 inference where prerequisites are absent or unverified. Supply/reconcile provenance for available train-score evidence before reliance.

**B**: Provide compatible pre-existing Top-2, benchmark and train-score evidence, with model/partition/exclusion provenance, for inspection before planning.

**C**: Expand scope with a separately approved offline-artifact workstream. This requires revised experiment/contracts, producer-consumer tests and regeneration/invalidation. Already exposed test records cannot become pristine merely by selecting three afterward; new untouched evidence or honest relabelling is required.

### Q2 — Bounds and exceptions

**Recommended A**: Apply the same role-specific guard to builder, quick-load and curves; reject incompatible benchmark rows without alteration and restrict growth curves to supported experience values.

**B**: Permit explicitly labelled benchmark/extrapolation paths with separately defined validation. Manual scenarios remain strictly bounded. Unsupported examples must never receive the normal supported-input badge.

### Q3 — Title/category relationship

**A**: Use observed pairs from a declared target-independent reference population and label them observed, not logically validated. Preserves data semantics but offers limited filtering because associations are inconsistent.

**B**: Use a stakeholder-approved curated mapping. Better matches the requested logical role guard but requires an explicit mapping and rules for conflicting existing benchmark records; this must not rewrite training data or imply that the model was trained on the curated taxonomy.

### User disposition — 2026-09-19

- **Q1: C approved for planning/tasks.** Generate missing evidence offline, with new functionality in new `.py` files rather than the existing god module. This is not approval to execute implementation or training yet.
- **Q2: Uniform bounds by default, labelled exceptions allowed.** Plan an explicit acknowledgement for experience-only benchmark/curve exceptions; never bypass enum, observed-pair or model checks.
- **Q3: A.** Use observed dataset pairs. The plan chooses DEV as the target-independent reference population to avoid learning serving constraints from held-out data.
- Historical test exposure is irreversible: generated examples will be labelled historical benchmark records and retain inclusion in the 298-row benchmark, not presented as newly pristine or excluded from prior scoring. No new data collection or split change is scheduled.

The options above are retained as the decision history; all three product clarifications are now resolved.

## Governance and artifact reuse

The prior `specs/002-readable-training-logs/plan.md` records inherited constitutional issues: candidate-level test coverage, non-nested tuning and differing family/tuning selection policies. The UI cannot repair or waive them. Current evidence differs from that plan in containing extra train-score fields, making a provenance check especially important.

Specification work does not change training-producing code, data, configuration, locks or consumer contracts. Existing outputs remain untouched; no retraining is justified for this review. Future UI-only reuse requires verifying the packs, and any approved contract/producer change requires regeneration or explicit invalidation before reliance.

During the initial specify phase, the original source document, earlier specs, `uv.lock`, application source and training artifacts were not edited. Auto-commit is disabled by repository configuration. The subsequent planning phase updates only this feature's documentation, the assessment and the `AGENTS.md` plan link; no application/training implementation is authorized yet.

## Additional planning inspection — 2026-09-19

- `candidate_models` in `src/ai_job_market/core.py` actually uses RF 180 trees / leaf 2 / max_features 0.8 and GB 180 trees / depth 2 / learning_rate 0.04 / Huber loss. Source-document model chips (RF 100 trees and GB depth 3) cannot be copied. Fresh supplemental results may differ from existing illustrative CSVs; label configuration and evidence run explicitly.
- `make_model_pipeline(model, features)` already accepts feature subsets, enabling Top-2 preprocessing reuse without editing the monolith. `temporal_cv_splits` exposes the shared chronological membership, and `regression_metrics` can score both train and validation predictions in a new evaluator.
- `tune_random_forest_manual_steps` emits CV_MedAE today; candidate `evaluate_model_cv` still does not emit Train_* fields. The recorded tuning history may be imported with explicit inherited provenance; no new hyperparameter search is needed for a frozen Top-2 experiment.
- `SOURCE_COLUMNS` contains `AI Engineering`, not `job_category`. A source-schema export must map category back to that original column. The current queue exporter sets an extra key that is dropped when constructing the source-column DataFrame.
- Read-only DEV inspection: 1,201 rows, 298 test rows, no `job_id` in prepared partitions; Scientist DEV bounds are 4–12 (median 7), Architect DEV bounds are 3–14 (median 7.5). Retaining fractional medians avoids an arbitrary rounding policy.
- Read-only feature-policy inspection found 272 historical test rows satisfying observed DEV pairs and title-level experience bounds. A deterministic target-independent choice of three examples is feasible without exemptions or invented records.
- The app routes through `streamlit.py` to `render(st, root, role)` page functions and passes a selected workspace root. Preserve that interface; caches/state must be scoped to workspace and evidence identity. The existing global upload handler directly calls `run_pipeline`; this inherited issue is outside this feature and is not evidence that the new workflow may train in the UI.
- Local versions inspected without installing packages: Python 3.13.12, Streamlit 1.64.0, Plotly 7.1.0, pandas 2.3.3, NumPy 2.5.3, scikit-learn 1.9.1, pytest 9.1.1, joblib 1.6.0. New APIs must be checked against this environment; no upgrade is planned.
