# Branch A — 5-Representation PCA Robustness Update

This patch updates Branch A from four to five representation studies and tightens the PCA contract.

## Representation study

- R0_GLOBAL_PCA: full approved Branch-A feature set, family-balanced global PCA.
- R1_FAMILYWISE_PCA: PCA is fitted separately inside each feature family, family-specific PC caps are applied, then retained latent blocks are rebalanced and concatenated.
- R2_NO_JOB_CATEGORY: remove job_category before the global PCA geometry is built.
- R3_NO_YEARS_EXPERIENCE: remove years_of_experience before the global PCA geometry is built.
- R4_NO_JOB_CATEGORY_NO_YEARS: remove both dominant features before the global PCA geometry is built.

## PCA contract

- All preprocessing/PCA fitting is DEV-only.
- R0/R2/R3/R4 each form their own ablated/rebalanced encoded geometry before fitting global PCA.
- Global PCA retains the smallest number of PCs reaching the configured variance target (default 85%), with a safety cap of 25 PCs.
- R1 retains PCs family-by-family at the same variance target, subject to family caps. Family latent blocks are then weighted by 1/sqrt(k_f) before concatenation.
- PCA whitening is not used.
- PC1/PC2 are display coordinates only; clustering uses the retained latent space.
- 80%/85%/90% PCA sensitivity is persisted as a diagnostic only. It does not select the official representation or optimize silhouette.

## Representation selection

Every representation runs the same KMeans/GMM K search and is assessed using separation, optimizer-seed stability, row-subsample stability, minimum cluster share and cross-representation ARI. Only near-best separation candidates are compared for robustness/parsimony. Selection is data-driven and the final raw input contract is frozen in the segmentation metadata/artifact.

## Files to update

Copy these files to the same relative location in the project and replace the existing file when one already exists:

1. `src/ai_job_market/segmentation_robustness.py` — REPLACE
2. `src/ai_job_market/core.py` — REPLACE
3. `src/pages/page03_segmentation.py` — REPLACE
4. `config/project.yaml` — REPLACE
5. `tests/test_segmentation_representation_robustness.py` — REPLACE (or ADD if not present)

No other source file is required by this patch.

## New offline evidence

The updated pipeline persists, among other existing Branch-A outputs:

- `representation_summary.csv`
- `representation_pairwise_ari.csv` (5 x 5 = 25 comparisons)
- `representation_pca_variance.csv`
- `representation_component_loadings.csv`
- `representation_pca_sensitivity.csv`
- `feature_dependency_summary.csv` including the R4 joint-ablation row
- `segmentation_metadata.json` with the selected representation and final feature contract

## Run after replacement

From project root:

```powershell
python pipeline.py
python -m pytest -q
streamlit run streamlit.py
```

Do not copy old output CSVs from a previous release over the new project. Re-run `pipeline.py` so Streamlit reads evidence created by the updated code.
