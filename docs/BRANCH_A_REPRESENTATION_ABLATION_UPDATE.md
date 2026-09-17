# Branch A Representation Robustness Update

This update implements the requested four-representation PCA ablation study and makes the Branch-A input contract data-driven.

## Files to add / replace

### 1. ADD NEW FILE
`src/ai_job_market/segmentation_robustness.py`

Purpose:
- build R0/R1/R2/R3 representations;
- run the same KMeans/GMM K=2..8 protocol on every representation;
- calculate seed/subsample stability;
- calculate cross-representation ARI;
- choose the official representation;
- transform future rows through the frozen selected representation.

### 2. REPLACE FILE
`src/ai_job_market/core.py`

Purpose of update:
- invokes the four-representation study;
- freezes the selected raw-input contract;
- uses selected representation for the official clustering model;
- persists representation-ablation evidence;
- uses `transform_to_cluster_space()` / `predict_segments()` for future/test inference;
- prevents the previous 199-vs-15 dimension mismatch.

### 3. REPLACE FILE
`src/pages/page03_segmentation.py`

Purpose of update:
- Streamlit reads only persisted evidence;
- adds Representation Robustness tab;
- shows R0/R1/R2/R3 comparison;
- shows pairwise ARI heatmap;
- shows selected Branch-A input feature contract;
- shows PCA/latent-space loadings and dynamic run-specific interpretation.

### 4. REPLACE FILE
`config/project.yaml`

Adds:
```yaml
segmentation:
  representation_robustness:
    enabled: true
    silhouette_tolerance: 0.02
    family_pca_caps:
      Job Domain: 3
      Experience & Education: 3
      Company & Work Mode: 3
      Geography: 3
      Demand / Benefits: 3
      Skills: 5
```

### 5. ADD TEST FILE
`tests/test_segmentation_representation_robustness.py`

Checks:
- all four representations are persisted;
- exactly one official representation is selected;
- selected input contract follows the ablation result;
- pairwise ARI matrix is complete;
- PCA loadings are created offline;
- Streamlit does not fit PCA/KMeans/GMM or tokenize skills.

## Four representations

- `R0_GLOBAL_PCA`: baseline global family-balanced PCA.
- `R1_FAMILYWISE_PCA`: separate PCA per family, capped latent dimensions, then concatenate.
- `R2_NO_JOB_CATEGORY`: global PCA after removing encoded `job_category` columns.
- `R3_NO_YEARS_EXPERIENCE`: global PCA after removing `years_of_experience`.

## Selection logic

1. Within each representation:
   - KMeans and GMM, K=2..8;
   - minimum cluster-share gate;
   - seed-stability ARI gate;
   - subsample-stability ARI gate;
   - silhouette as primary separation metric;
   - parsimonious K/algorithm tie-break.

2. Across representations:
   - keep representations within configured silhouette tolerance of the best;
   - prefer higher mean cross-representation ARI;
   - then higher subsample stability;
   - then lower latent/raw-input complexity.

3. Freeze the winning representation's raw-input feature contract and use it for the official clustering model.

## Current dataset result

On `ai_jobs_market_2025_2026.csv`, the current run selects:

- Representation: `R2_NO_JOB_CATEGORY`
- Official clustering: `KMeans, K=2`
- `job_category` is removed from the Branch-A input contract for this run.
- R0 vs R2 assignment ARI = `1.000`, meaning removal of `job_category` did not change the baseline partition.
- R0 vs R3 assignment ARI ≈ `0.990`, so `years_of_experience` also has low structural dependence under this representation study.

This decision is not hard-coded. A new dataset may select R0, R1, R2 or R3.

## New offline outputs

`outputs/03_ai_job_market_segmentation/`

- `representation_summary.csv`
- `representation_candidate_metrics.csv`
- `representation_candidate_assignments.csv`
- `representation_pairwise_ari.csv`
- `representation_pca_variance.csv`
- `representation_top_loadings.csv`
- `representation_component_summary.csv`
- `representation_component_loadings.csv`
- `representation_resample_stability_runs.csv`
- `representation_selected_assignments.csv`
- `feature_dependency_summary.csv`
- `segmentation_metadata.json`
- `segmentation_insights.json`

## Run order

After replacing the files, from project root:

```powershell
python pipeline.py
python -m pytest -q
streamlit run streamlit.py
```

Check the Streamlit page:

`3. AI Job Market Segmentation -> A5 Representation Robustness`

The selected representation and the final feature input list are shown dynamically from pipeline outputs.
