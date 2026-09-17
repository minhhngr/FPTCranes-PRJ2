# Branch A — How the project selects an effective K

The clustering UI does **not** choose K from one score alone. For every candidate from K=2…8 and for both K-Means and GMM, the offline pipeline writes a row to `outputs/03_ai_job_market_segmentation/cluster_evaluation.csv`.

## Decision policy

1. **Stability gate** — rerun the candidate across several random seeds and compute pairwise Adjusted Rand Index (ARI). A candidate must have mean ARI ≥ 0.90.
2. **Minimum-size gate** — reject partitions with any cluster smaller than 5% of records. This avoids selecting a mathematically attractive K that creates tiny, difficult-to-interpret segments.
3. **Primary separation metric** — among eligible candidates, maximize Silhouette Score.
4. **Practical-tie rule** — differences of less than 0.01 Silhouette are treated as practically tied rather than over-interpreted.
5. **Parsimony tie-break** — among near-best candidates, prefer the smaller K; for the same K, prefer K-Means over GMM because it is easier to explain and operationalize.
6. **Supporting diagnostics** — Calinski-Harabasz (higher is better), Davies-Bouldin (lower is better), K-Means inertia/elbow, and GMM BIC/AIC are shown but do not silently overrule the primary policy.

## Current baseline result

The source dataset produces:

- Raw best Silhouette: **GMM K=2 ≈ 0.393**
- Selected: **K-Means K=2 ≈ 0.388**
- Stability ARI: **1.000**
- Smallest-cluster share: **≈40.4%**
- Interpretation: **moderate separation**

The Silhouette difference is only about 0.005, which is below the configured 0.01 practical-tie tolerance. Both candidates choose K=2, and the simpler K-Means partition is therefore selected.

The Streamlit page also stores the cluster assignment of every candidate K/algorithm. Users can select any alternative K and inspect the same PC1/PC2 projection, cluster balance, and profile evidence **without fitting anything in the UI**.

## Why the project does not automatically select K=6

For this data, K=6 has stronger Calinski-Harabasz / Davies-Bouldin evidence in some views, but its Silhouette is materially lower than the K=2 candidates. Those supporting scores answer different questions and can favour more partitions. The selection rule is explicit so K is not chosen post hoc to make a chart look more interesting.
