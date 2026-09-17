# Legacy Feature Merge Notes

This combined release keeps the strongest project behaviours requested across the earlier FPTCranes-PRJ2_full iterations while moving them into the technical-design order.

## Preserved and expanded

- Pastel, wide Streamlit layout with authenticated Admin/User roles.
- Eight-page presentation order rather than a duplicated 12-stage navigation.
- Full data-quality story beyond null/duplicate checks: categorical corruption, semantic contradictions, leakage and redundancy.
- Multiple complementary charts per major integrity issue rather than one summary visual only.
- Numeric labels and exact hover values on analytical charts.
- TRAIN-only correlation / encoding / scaling evidence and encoded-to-original feature mapping.
- 93-skill multi-hot logic, `skill_count`, top-skill analysis, sparse-skill review and TRAIN-only skill-target correlation.
- AI Job Market Segmentation EDA by six feature families: Job Domain, Skills, Experience, Company, Geography, Demand / Benefits.
- Geographic salary/segment views at country level plus city-level analytical view without requiring online geocoding.
- Dummy baseline, linear models, nonlinear candidates, 5-fold temporal evidence and runtime comparison.
- Locked-test actual-vs-predicted, residual-tail analysis, encoded importance, raw permutation importance and subgroup error diagnostics.
- Metadata-driven Salary Prediction page with validation queue, batch prediction, practical error band and original-schema CSV export.
- Full three-pipeline visual reference (overall, segmentation and prediction).
- New-data schema preflight and adaptive latest-period holdout in the offline pipeline.

## Architecture rule retained

Streamlit is presentation/inference only. All model fitting, preprocessing fitting, feature construction, clustering, tuning and output generation remain in the offline pipeline. This keeps the UI reproducible and prevents hidden leakage.

## Source provenance

The user-supplied legacy `.7z` is retained under `docs/legacy_reference/` as an audit reference. The active source code in this release is the consolidated implementation under `src/`.
