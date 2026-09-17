# Merge Notes — Full Combined Release

This release merges the September-2026 Technical Specification workflow, the three supplied pipeline diagrams, the current `ai_jobs_market_2025_2026.csv`, and the richer dashboard requirements carried forward from the earlier FPTCranes-PRJ2_full work.

## What changed from the previous release

- Rebuilt the Streamlit evidence pages around **interactive Plotly charts** instead of mostly static Matplotlib figures.
- Added category/cluster/country/model filters and chart selectors so users can focus on specific segments and inspect exact values.
- Added output tables required for dynamic analysis: row-level integrity flags, encoding-to-original feature map, numeric scaling summary, skill-token summary/correlation, model runtime, and locked-test subgroup error tables.
- Restored the full approved regression ladder including the **Dummy Median** floor.
- Expanded data-quality analysis to multiple complementary views for experience contradictions, salary-range inconsistency and salary-tier inconsistency.
- Expanded segmentation EDA across all six feature families and added country/city market views.
- Added model-runtime evidence, subgroup diagnostics and interactive locked-test filtering.
- Added dynamic batch-prediction visualization without changing the metadata-driven validation queue.
- Added a Full Pipeline page with all three supplied diagrams and a schema-only new-data preflight.

## Architecture retained

The offline pipeline owns data cleaning, feature engineering, preprocessing fitting, clustering, CV, tuning, final evaluation and serialization. Streamlit is report/inference only. Segment labels are never fed into the salary predictor; integration occurs only after both branches complete.
