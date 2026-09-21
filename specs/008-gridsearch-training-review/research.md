# Research: GridSearchCV training review

## Sources reviewed

- `docs/pipelines/Pipeline_AI-salary-overall.png` and `Pipeline_AI-salary-prediction.png` (pipeline progression reference)
- `docs/BRANCH_B_COMBINATION.md`
- `docs/spec-imporve-ui.md`
- `src/ai_job_market/training_partitions.py`
- `src/ai_job_market/training_search.py`
- `src/ai_job_market/training_validation.py`
- `config/training_validation.json`
- Existing Page 04–06 and training-validation presentation code
- Existing Graphify graph query

## Findings

| Finding | Evidence | Decision |
| --- | --- | --- |
| Current RF tuning is manual | `stepwise_rf_search()` iterates anchors then coordinate sweeps | Replace with GridSearchCV. |
| Existing folds are chronological expanding monthly folds | `build_expanding_monthly_folds()` emits ordered fold declarations | Reuse exact fold declarations through a CV adapter; do not use default KFold. |
| Current policy includes n_estimators and max_depth candidates | `config/training_validation.json` `rf_search` | Make their Cartesian grid explicit and persist every candidate result. |
| Training-validation current reserve is exposed | `inspect_workspace()` detects `outputs/02_data_ready_for_ml/locked_test_raw.csv` overlap | No fresh trusted evaluation run on current data. |
| UI already supports read-only evidence renderers | `training_validation_presentation.py` is called before Page 04/05 supplemental reader | Evolve its contract; do not train in UI. |
| Page 06 has a separate serving consumer | `page06_prediction.py` loads active supplemental Top-2 bundle | Leave unchanged unless metadata/schema changes. |

## Leakage controls required

- Fit preprocessing only inside the estimator pipeline supplied to GridSearchCV.
- Convert only the context-local parent membership to positional indices.
- Assert each supplied inner train index predates and is disjoint from validation indices.
- Assert outer validation, evaluation holdout, and inference reserve positions are absent from each search context.
- Freeze candidate family from outer CV before tuning; never rank families using tuned/holdout results.
- Reject stale approval, changed source/policy/producer identity, known reserve exposure, missing folds, and non-finite selection scores.

## UI decisions

- Preserve existing report charts and KPI sections.
- Add a compact method/status visual summary, then place all full split/search/log tables in collapsed native expanders at page end.
- Prefer existing Plotly evidence colors and direct labels; no CSS or custom components.
- Label raw events separately from derived step records and offer complete downloads.

## Open operational blocker

A code change will invalidate prior manual-search evidence. The current workspace does not satisfy the non-exposure condition for a new trusted pack. Implementation must make that state explicit; actual retraining awaits fresh eligible data plus new approval.
