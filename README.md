# FPTCranes-PRJ2 — Full Project v2
## AI Job Market Segmentation + Salary Prediction

This release combines the technical workflow in `Document_QD Project KHDL&AI(8)` with the richer analytical structure of the previous full project and the Branch B organization/evidence contract indexed from `FPTCranes-PRJ2-main.rar`.

The central architecture is **outputs first**: the offline pipeline creates data/model/evidence artifacts; Streamlit reads the active run artifacts and does not refit models when charts or filters change.

Training audit logs and interpretation notes are documented in `docs/TRAINING_AUDIT.md`.

## End-to-end workflow

### Common Foundation
1. Project Scope & Raw Data Ingestion
2. Basic Clean
3. Contradictory-Feature Investigation
4. Feature Importance / Leakage Prevention
5. Feature Selection & Shared Prepared Feature Base

### Branch A — AI Job Market Segmentation
A1 Inputs → A2 six feature families → A3 encoding → A4 scaling/family balancing → A5 PCA → A6 PC1+PC2 → A7 KMeans/GMM K=2…8 → A8 cluster evaluation.

`annual_salary_usd` is excluded from clustering. Salary is reintroduced only after segment assignment for descriptive profiling.

### Branch B — Salary Prediction
B1 target definition → B2 temporal locked split → B3 TRAIN-only preprocessing → B4 five-model expanding temporal CV → B5 bounded tuning + explainability → B6 one-time locked-test evaluation → B7 saved preprocessing+model bundle and metadata-driven inference.

The release also exposes `src/training/` and `src/pipeline/` Branch-B facades matching the clearer organization found in `FPTCranes-PRJ2-main`, while keeping the implementation source of truth in `src/ai_job_market/core.py`.

## How K is selected in unsupervised learning

K is **not** selected by eyeballing one plot.

For K-Means and GMM, K=2…8 are evaluated with:

- Silhouette Score — primary separation metric;
- multi-seed Adjusted Rand Index (ARI) — stability gate;
- minimum cluster share — anti-tiny-cluster gate;
- Calinski-Harabasz and Davies-Bouldin — supporting quality evidence;
- K-Means inertia/elbow;
- GMM BIC/AIC.

Selection policy:

1. ARI stability ≥ 0.90;
2. every cluster ≥ 5% of records;
3. maximize Silhouette;
4. candidates within 0.01 Silhouette are considered practically tied;
5. among near-best candidates, prefer smaller K, then K-Means for simpler interpretation.

### Baseline result

Current source data selects **K-Means, K=2**:

- selected Silhouette ≈ **0.388**;
- ARI stability = **1.000**;
- smallest cluster ≈ **40.4%**;
- quality band = **moderate separation**.

The raw best Silhouette is GMM K=2 ≈ 0.393. The gap is only ≈0.005, below the 0.01 practical-tie tolerance, so K-Means K=2 is selected for parsimony. See `docs/UNSUPERVISED_K_SELECTION.md` and the Streamlit **How K is Selected** tab.

## Streamlit — 8-page order

1. **Data Basic Clean** — Stage 1→3
2. **Data Ready for ML** — Stage 4→5 + B1→B3
3. **AI Job Market Segmentation** — A1→A8
4. **Model Comparison** — B4
5. **Best Model & Importance** — B5→B6
6. **Salary Prediction** — B7 / serving
7. **Integrated Market Insight**
8. **Full Pipeline**

Every page contains data-driven interpretation cards. The comments are calculated from the active output tables and the current filter/metric selection rather than written as fixed conclusions.

## Dynamic Streamlit charts

The dashboard uses Plotly for the main analytical charts. Depending on the page, users can:

- hover to see exact values;
- filter job category, experience, country, remote mode and cluster;
- choose chart/metric and Top-N;
- compare alternative K/algorithm partitions without fitting in the UI;
- compare temporal folds and model runtime;
- inspect feature-family ablation and feature-importance drift;
- drill down locked-test errors by subgroup;
- select cluster/category/country in integrated analysis;
- inspect queued salary scenarios and prediction bands.

See `docs/STREAMLIT_CONTENT_MAP.md`.

## Upload and process a completely new dataset

Admin users now have a global sidebar section: **Upload & process a new dataset**.

1. Upload a CSV.
2. The app parses the file and validates the expected 25-column source contract.
3. It shows schema PASS/FAIL, errors/warnings and an expected-vs-detected field table.
4. If any blocking schema/type/time error exists, **Process full pipeline** stays disabled.
5. If valid, click **Process full pipeline on this dataset**.
6. The entire Common Foundation + Branch A + Branch B pipeline runs in an isolated `runs/<timestamp_hash>/` workspace.
7. The baseline release is not overwritten.
8. Streamlit automatically switches every page to the new run's outputs/artifacts.
9. Previous validated runs can be selected from the workspace dropdown.

Accepted raw alias: `job_category` can substitute for raw `AI Engineering`; it is canonicalized before processing. Unknown category values are allowed; fitted encoders use safe unknown handling.

A full new-data smoke test is recorded in `outputs/new_data_validation.json`: the future holdout was moved to April-2026 and one unseen job category was introduced in that locked period; the full pipeline completed successfully.

## Branch B combination from FPTCranes-PRJ2-main

The uploaded RAR is packaged under `docs/legacy_reference/FPTCranes-PRJ2-main.rar`. The runtime could inspect its RAR5 directory/manifest but did not have a RAR5 decompressor for reliable line-by-line source extraction. To stay transparent, this release includes the archive inventory and reproduces/combines the Branch B module/output contract supported by the manifest and the Technical Design rather than claiming byte-for-byte code merging.

Added/retained Branch B evidence includes:

- 08 readiness/split/before-after outputs;
- 09 fold metrics, temporal model comparison, runtime, feature-family ablation, feature importance by fold and drift;
- 10 bounded tuning, locked-test predictions, subgroup slices, raw/encoded importance and final metrics;
- 11 deployment artifact manifest and reload-equivalence evidence;
- 12 prediction summary and representative prediction examples.

See `docs/BRANCH_B_COMBINATION.md` and `docs/FPTCranes-PRJ2-main_manifest.csv`.

## Project structure

```text
FPTCranes-PRJ2_full_v2/
├── config/project.yaml
├── data/raw/ai_jobs_market_2025_2026.csv
├── docs/
│   ├── Document_QD Project KHDL&AI(8).docx
│   ├── UNSUPERVISED_K_SELECTION.md
│   ├── BRANCH_B_COMBINATION.md
│   ├── STREAMLIT_CONTENT_MAP.md
│   ├── FPTCranes-PRJ2-main_manifest.csv
│   ├── pipelines/                         # 3 supplied pipeline images
│   └── legacy_reference/                  # original comparison archives
├── src/
│   ├── ai_job_market/core.py              # one computational source of truth
│   ├── training/                           # Branch-B compatibility facades
│   ├── pipeline/                           # stage-oriented Branch-B facades
│   ├── components/
│   │   ├── auth.py
│   │   └── data_source.py                 # upload → schema gate → full run
│   └── pages/page01...page08*.py
├── outputs/
│   ├── 01_data_basic_clean/
│   ├── 02_data_ready_for_ml/
│   ├── 03_ai_job_market_segmentation/
│   ├── 04_model_comparison/
│   ├── 05_best_model/
│   ├── 06_salary_prediction/
│   ├── 07_integrated_insight/
│   └── 08_full_pipeline/
├── artifacts/
├── runs/                                  # created by Streamlit uploads
├── tests/
├── pipeline.py
├── validate_release.py
├── streamlit.py
└── requirements.txt
```

## Install and run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
python pipeline.py
# Add fold/trial/split/feature detail while keeping structured JSON file-only:
python pipeline.py --debuglog
python validate_release.py
python -m pytest -q
streamlit run streamlit.py
```

Demo credentials:

- Admin: `admin / AIJob2026!`
- User: `user / user123`

Replace demo credentials before shared deployment.

## Validation status

Current release checks:

- offline baseline pipeline: PASS;
- release validator: **33/33 PASS**;
- pytest: **16/16 PASS**;
- saved-bundle reload equivalence: PASS (`max_abs_diff = 0`);
- new-dataset full-flow smoke test: PASS;
- Streamlit presentation modules compile and contain no model `.fit()` calls.

The validation runtime does not have the Streamlit package installed, so browser-server launch is not claimed as executed here. Installing `requirements.txt` enables the local UI.

## Scientific boundary

The supplied dataset contains strong logical/synthetic-looking artifacts. Correlation, clustering, feature importance and predictive metrics describe **this dataset's behaviour**, not causal labour-market economics. The displayed salary uncertainty band is an empirical held-out absolute-error band, not a formal confidence interval or compensation guarantee.

## Branch A v3 scientific-stability revision

The current revision adds row-subsample ARI stability, selects the PCA clustering dimension from a DEV-only cumulative-variance threshold, separates the multi-PC clustering space from the PC1/PC2 display projection, and moves Branch-A tokenization/profile derivation fully into the offline pipeline. See `docs/SEGMENTATION_V3_CHANGE_GUIDE.md`.
