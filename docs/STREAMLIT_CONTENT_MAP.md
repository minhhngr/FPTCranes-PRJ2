# Streamlit Content Map — technical order, dynamic charts and output-driven comments

All eight pages read the active run workspace. Each major chart/table is followed by a **data-driven interpretation card** built from the currently displayed output/filter: observed evidence → interpretation → recommended next action. The UI never fits a model when the user changes a chart control.

| Page | Technical-design order | Core content | Interactive focus / commentary |
|---|---|---|---|
| 1. Data Basic Clean | Stage 1→3 | target profile, schema/cardinality, missing/hidden missing, raw-vs-clean, ranked integrity findings, experience contradiction, 3 salary-range views, 3 salary-tier views | category/country/experience/remote filters, chart selector, Top-N; each analysis block has computed interpretation |
| 2. Data Ready for ML | Stage 4→5 + B1–B3 | leakage policy, governance mix, ablation, temporal split, TRAIN-only encoding/scaling, encoded mapping, Pearson/family correlations, 93-skill evidence | correlation direction, Top-N, source feature, numeric distribution and skill metric controls; comments change with selection |
| 3. AI Job Market Segmentation | A1→A8 | six families, family balance, PCA, KMeans/GMM K=2…8, multi-seed ARI stability, balance gate, Silhouette, CH/DB, inertia, BIC/AIC, profiles and family EDA | **How K is Selected** tab, alternative algorithm/K explorer, cluster/category/country/family filters; K-decision rationale comes from `k_selection_rationale.json` |
| 4. Model Comparison | B4 | five-model ladder, 5 temporal folds, ranking, fold stability, runtime, feature-family ablation, importance drift | metric/model/fold controls; dynamic winner/stability/runtime/ablation commentary |
| 5. Best Model & Importance | B5–B6 | tuning table, locked metrics, actual-vs-predicted, residuals, q90 band, raw and encoded importance, subgroup errors | subgroup/filter/importance Top-N/diagnostic selector; comments quantify CV-to-test degradation, tails and reliance |
| 6. Salary Prediction | B7 / serving | metadata-driven form, logical validation, queue, batch inference, empirical error band, CSV export | queue and batch controls plus dynamic summary of active scenarios and prediction spread |
| 7. Integrated Insight | post-branch | cluster salary summaries, cluster prediction error, market structure, geography | cluster/country/category and mean/median controls; commentary links structural segments to predictive evidence |
| 8. Full Pipeline | overall | three supplied pipeline diagrams, active-run status, artifact/provenance review and processing contract | explains current workspace and schema/process status; global upload/process control lives in sidebar |

## Global new-data workflow

Admin sidebar: **Upload & process a new dataset**.

1. Upload CSV.
2. Parse file.
3. Validate the source schema and key types.
4. Show PASS/FAIL, issues and expected-vs-detected column table.
5. Keep the process button disabled while any blocking ERROR exists.
6. If valid, click **Process full pipeline on this dataset**.
7. Run the same Common Foundation + Branch A + Branch B workflow in `runs/<timestamp_hash>/`.
8. Baseline outputs are not overwritten.
9. Automatically switch all Streamlit pages to the new run workspace.

The full-run smoke test in `outputs/new_data_validation.json` confirms this architecture with a future April-2026 locked period and a previously unseen locked-period job category.
