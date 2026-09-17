# Branch A — A5 Complete Representation Robustness Evidence

## Why A5 had missing values
The R0-R4 screening pass previously used staged robustness checks to reduce runtime. Separation/balance were calculated for every Algorithm x K candidate, but seed ARI and subsample ARI were calculated only for the strongest candidates. As a result, representation summaries and candidate tables could contain missing stability values, and the A5 separation-vs-stability scatter could only plot representations with non-null subsample ARI.

## Change
`src/ai_job_market/core.py` now calls `evaluate_candidates(..., full_stability=True)` for every one of R0-R4. Therefore every valid KMeans/GMM K=2..8 candidate in every representation receives:
- Silhouette
- Seed stability ARI
- Subsample stability mean/std/P10/min ARI
- Minimum cluster share
- Balance entropy
- Calinski-Harabasz
- Davies-Bouldin
- KMeans inertia or GMM BIC/AIC
- gate/eligibility flags

The selected representation is still rerun afterward to create the official downstream model/evidence bundle.

## Replace
Replace only:
`src/ai_job_market/core.py`

No Streamlit page change is required if the latest A5/A7 page patch is already installed.

## Mandatory rerun
The old CSVs contain the missing values, so rerun the offline pipeline:

```powershell
python pipeline.py
streamlit run streamlit.py
```

Expected result: the A5 full-comparison table and decision scatter have complete stability evidence for R0-R4, and the candidate-level selector shows complete seed/subsample metrics for all valid KMeans/GMM K=2..8 candidates.
