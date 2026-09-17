# Branch A — A7/A8 Full K Evidence Update

## Why the charts were incomplete
The five-representation robustness screening intentionally used staged stability testing to save runtime. Separation and balance were computed for every Algorithm×K candidate, but seed ARI and subsample ARI were only computed for the strongest candidates until an eligible candidate was found. As a result, the Streamlit A7/A8 charts contained gaps.

## What changed
1. `segmentation_robustness.py`
   - adds `full_stability=True` mode to `evaluate_candidates()`;
   - in full mode, every valid KMeans/GMM K=2..8 candidate receives seed ARI and subsample ARI, even when a gate fails;
   - eligibility still requires balance + seed stability + subsample stability gates.

2. `core.py`
   - R0-R4 representation screening remains staged for efficiency;
   - after the official representation is selected, the pipeline reruns only that representation with `full_stability=True`;
   - `cluster_evaluation.csv` and `resample_stability_runs.csv` therefore contain complete official K-search evidence;
   - the final official K/algorithm/model/assignments are taken from this exhaustive pass.

3. `page03_segmentation.py`
   - adds coverage counters for candidate rows, seed ARI, and subsample ARI;
   - warns when old/incomplete outputs are loaded;
   - adds a downloadable official KMeans/GMM K=2-8 evidence table;
   - keeps all K values visible on the x-axis.

## Files to replace
- `src/ai_job_market/segmentation_robustness.py`
- `src/ai_job_market/core.py`
- `src/pages/page03_segmentation.py`

## Required rerun
After replacing the files, run:

```powershell
python pipeline.py
streamlit run streamlit.py
```

Old CSV outputs will still show gaps until `pipeline.py` is rerun.
