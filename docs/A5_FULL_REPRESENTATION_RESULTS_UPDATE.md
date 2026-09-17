# A5 Full Representation Results Update

Replace only:

`src/pages/page03_segmentation.py`

with the file in this patch.

No pipeline/model refit logic is added to Streamlit. The page reads the already-persisted files:
- representation_summary.csv
- representation_candidate_metrics.csv
- representation_pairwise_ari.csv
- feature_dependency_summary.csv
- representation_pca_sensitivity.csv

What changes in A5:
1. Always shows all R0-R4 in a full comparison table.
2. Adds categorical metric charts for all five representations: silhouette, seed/subsample ARI, minimum cluster share, and complexity.
3. Keeps a separation-vs-stability scatter but explicitly controls axis ranges and warns if any representation lacks a required metric.
4. Shows the 5x5 pairwise ARI heatmap.
5. Shows feature-dependence evidence.
6. Adds candidate-level KMeans/GMM K=2..8 results for each representation via a selector.
7. Keeps the selected input contract and data-driven narrative.

After replacement, restart Streamlit. Re-run `python pipeline.py` only if the evidence CSV/JSON outputs are stale or missing.
