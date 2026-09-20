# Model Comparison, Diagnostics and Controlled Salary Scenarios

Pages 04–06 consume a versioned supplemental evidence pack. Pages 04–05 also expose a separate immutable `training-validation/v1` pack when one exists. They do not fit, tune, regenerate, deserialize, publish or silently substitute models during normal page interaction.

## Offline training-validation evidence on Pages 04–05

A compact always-visible `Training & validation evidence` overview gives third-party reviewers immediate verified-run status and trust boundaries; the detailed tables remain in collapsed expanders. This additive block does not replace or restructure existing Page 04/05 analytics and is isolated for easy removal after verification. Both overview and expanders are independent of `outputs/ui_evidence/current.json`. They scan at most 100 final directories under `outputs/training_validation/tv-<32 lowercase hex>/`, validate every candidate with the producer's complete-pack validator, and select the greatest strict UTC `generated_at` (run ID breaks ties). A corrupt newer candidate does not hide an older valid pack. Staging directories and the holdout-access ledger are never candidates.

Page 04 reports the approved chronological partition, five expanding monthly outer folds, 18 parent-labelled inner folds, all 25 candidate/fold evaluations, fit/stability/runtime context, the lowest-CV-MAE versus selected-family versus fastest-fit distinction, and five scoped candidate conclusions. Page 05 separates nested tuning and matched Full/Top-2 evidence from one-time `EVALUATION_HOLDOUT`, explainability, subgroup, descriptive q90 uncertainty, scientific outcome, operational status and two final-role conclusions. `INFERENCE_RESERVE` never has displayed metrics or predictions, and holdout results cannot silently promote a model.

Three log forms remain visibly distinct:

1. original short `training.log`;
2. original raw lifecycle `events.jsonl`;
3. a deterministic **evidence-derived** method transcript reconstructed from authoritative CSV/JSON rows, with source references and no invented timestamps.

Original log/report/summary downloads are byte-identical. The complete `training-validation-transcript.csv` is generated in memory and is not written to the workspace; previews may be bounded. If no valid pack exists, each expander says `Training validation unavailable`, uses no fixture/historical/hardcoded fallback, and offers only the read-only command:

```bash
.venv/bin/python training_validation.py inspect --workspace . --policy config/training_validation.json
```

A future approved operator may run and then check training **offline** as documented in `docs/TRAINING_VALIDATION.md`; Streamlit provides no run button. Scientific status is evidence, not activation, deployment approval or serving configuration.

### Historical Branch B report and 5W1H logs

The current workspace also has a complete source-compatible primary-pipeline audit and active supplemental evidence. Pages 04–05 therefore add a separately labelled **Historical Branch B training report** even while strict future-reserve validation remains unavailable. It never fills strict-pack fields or claims that historically scored locked-test rows are unseen.

The evidence flow is shown as common prepared feature base → Branch A analytical workflow → Branch B salary comparison/tuning/evaluation. Both branches share upstream preparation, but Branch A cluster assignments are not silently added as Branch B salary features. Existing outputs and logs are reused; the root pipeline is not rerun merely to populate the UI.

Page 04 provides five source-bound 5W1H records—Dummy Median, Linear Regression, Ridge Regression, Random Forest and Gradient Boosting. Page 05 provides selected-family, Full 13 and fixed Top-2 records. Each record identifies Who, What, When, Where, Why and How, then reports observed metrics, limitation, next action and relative source references. `Why` means evidence-backed comparison role/decision; it does not invent memorization, singularity, causal economics or pristine-test claims.

A collapsed historical activity-log section shows at most 200 model-relevant raw JSONL event projections and downloads the complete original audit log, manifest and relevant checksum-verified exports. The generated Markdown 5W1H report is created in memory. Raw events, derived historical report rows and strict training-validation transcripts remain visibly distinct.

## Generate or validate evidence offline

From the repository root:

```bash
# Validate an existing pack without fitting, predicting or writing.
PYTHONPATH=src .venv/bin/python -m ai_job_market.ui_evidence --workspace "$PWD" --check

# Generate only missing/invalidated supplemental components.
PYTHONPATH=src .venv/bin/python -m ai_job_market.ui_evidence --workspace "$PWD"
```

A valid unchanged invocation is reused. The producer writes immutable reports under `outputs/ui_evidence/<evidence-id>/`, the fixed Top-2 bundle under `artifacts/ui_evidence/<evidence-id>/`, and atomically updates `outputs/ui_evidence/current.json`. It never overwrites the primary output packs or `artifacts/model_bundle.joblib`, and does not run `pipeline.py`.

The active evidence ID is resolved from `outputs/ui_evidence/current.json`. Its authoritative tables include:

- `candidate_fold_metrics.csv` and `candidate_summary.csv`: paired train/validation metrics from the same fold fits for five frozen candidate configurations;
- `candidate_test_metrics.csv`: retrospective diagnostics on the already exposed historical test set; never used to rerank the family winner;
- `variant_metrics.csv` and `variant_test_predictions.csv`: matched full/Top-2 fixed-configuration evidence;
- `rf_fold_importance.csv`, `rf_importance_drift.csv`, `variant_encoded_importance.csv` and `variant_permutation_importance.csv`: encoded, family and raw-feature reliance evidence;
- `scenario_policy.json`: DEV-only observed title/category pairs and title experience limits;
- `benchmark_examples.csv`: deterministic historical examples selected without targets/errors;
- `charts/*.html`: static references generated from the same tables as the UI.

The producer records exact file hashes, model parameters, source partitions, folds, dependency versions and limitations in `manifest.json`. A checksum establishes file identity, not external scientific validity.

## Visual hierarchy and semantic colors

Pages 04–06 use role-based colors rather than Plotly trace-order defaults:

- blue/teal columns: temporal validation values or salary predictions;
- orange line/markers: training or DEV-CV comparison evidence;
- red columns: historical-test dollar error;
- dashed dark-red line: Dummy/failure threshold reference;
- amber whiskers: empirical uncertainty;
- red diamond markers: known historical actual salary;
- gray: neutral identity/reference evidence.

Legend text, direct labels, marker shapes and line styles repeat the meaning so color is not the only encoding. Red historical-test columns emphasize consequential out-of-time error; red does not automatically mean the model failed.

Charts also have explicit visual priority. P1 decision charts render first and may occupy the full main-content width. P2 supporting charts use an approximately 820-pixel container or a balanced two-column row. P3 diagnostics use an approximately 680-pixel container, paired layout, expander or on-request view. `width="stretch"` stretches a Plotly chart only inside its assigned container. Native Streamlit stacking remains available on narrow screens, and a supporting chart may be promoted when direct labels would otherwise clip.

Current mapping:

| Page | P1 | P2 | P3 |
| --- | --- | --- | --- |
| 04 | Overall train/validation comparison; primary deep-dive chart within each tab | Error ratio/R² | Fold timeline and reliance drift |
| 05 | Dollar-error generalization, actual/predicted, raw permutation reliance | Separate R² and stacked tuning sensitivities | Residual histogram, encoded importance and empirical CDF |
| 06 | Prediction bars with uncertainty and actual markers | Requested growth curve | Detailed queue/table evidence |

## Reading path and conclusions

Each page now starts with a report card containing the question, evidence scope, current evidence-derived takeaway and principal limitation. `How to read these metrics` defines only the terms used on that page. Each major section states its question before the primary visual and follows it with a four-part evidence conclusion:

1. **Finding** — observed values and direction;
2. **Why it matters** — bounded practical meaning;
3. **Limit** — the most important scientific/provenance qualification;
4. **Decision/use** — optional safe use, never automatic model promotion or compensation advice.

Conclusions are built from the active validated tables or current Page 06 result snapshot. Missing required fields produce a named unavailable conclusion rather than stale prose. Adding or clearing a Page 06 queue row removes the old result and its conclusion.

## Chart readability

Figure builders own their geometry. The shared renderer no longer overwrites explicit heights/margins. Categorical axes use automargins and ordinary comma-grouped numeric ticks. Long-label horizontal charts reserve larger left margins and disclose their visible Top N.

Direct labels are selective:

- primary bars retain visible values;
- compact comparison lines label key extrema/optima while every value remains in hover and tables;
- dense scatter points use hover rather than hundreds of rendered text labels;
- Page 04 drift labels only the three most relied-on visible families;
- Page 06 uses short scenario IDs on the chart and full profiles in hover/table.

Supporting tuning charts are stacked in readable 820-pixel containers instead of squeezed into half-width columns. Reader-facing tables label currencies, units and identities; audit/source downloads retain the complete machine-readable contracts.

Pages 04–05 use one two-tier table grammar. Small decision summaries use Markdown-capable static tables: **bold means the governing winner, consequential comparison or saved applied value**, while *italics mean a baseline, development reference or contextual scope*. Wide/detail evidence remains interactive with plain cells, reader-facing headers and typed formatting—never literal Markdown syntax. Highlights are calculated from unrounded evidence; displayed USD is whole dollars, R² has three decimals, percentages one decimal, durations three decimals and counts are integers. Equal best values remain tied rather than being separated by rounded display values.

Pixel-level screenshots, DOM label-intersection checks, console review and a fresh independent comprehension walkthrough remain release-acceptance steps when Chrome DevTools tooling is available.

## Page 04 — Model comparison

The page has five evidence-driven KPIs, a collapsed priority guide to temporal validation and four tabs. The guide documents the implemented splitter rather than generic K-fold: DEV rows are stably sorted in time, divided into contiguous row blocks, and each fold trains on one block and validates on the immediately following block. Preprocessing and the candidate estimator are fitted inside each training fold. The design is neither randomized K-fold nor an expanding window; adjacent blocks can share month labels while retaining disjoint row identities. Active `candidate_fold_metrics.csv` and `fold_membership.csv` provide the displayed fold counts, periods and downloadable evidence.

The four tabs are:

1. **Overall comparison**: train/validation MAE, Dummy reference, MAE ratio/R², executive table and actual fold timeline.
2. **Random Forest**: fold errors and raw-family reliance drift.
3. **Gradient Boosting**: identical-fold comparison and median-error evidence.
4. **Baseline controls**: Linear, Ridge, Dummy and the lowest-CV-MAE candidate.

`Below baseline` means validation MAE is lower than the matching Dummy MAE. It is not a universal good-fit classifier. Error gaps do not by themselves prove matrix singularity, memorization or causal model behavior. Candidate ranking is separate from the saved selected full model.

## Page 05 — Saved-model diagnostics

The page compares the saved full model's frozen configuration on DEV CV and all 298 historically scored March-2026 test rows. Dollar errors and R² use separate scales. The tuning section includes a collapsed priority guide describing the actual inherited process:

1. evaluate four Random Forest anchor configurations;
2. rank trials by temporal DEV CV R², higher is better;
3. sweep 6 `n_estimators`, 6 `max_depth`, 4 `min_samples_leaf` and 6 `max_features` values sequentially while carrying/fixing prior values;
4. create the final estimator from the first row of the R²-sorted initial tuning table; treat later sweeps as sensitivity evidence rather than an automatic replacement.

Each trial uses the recorded five temporal folds. This differs from Page 04 family ranking, which uses mean validation MAE, lower is better. The search is inherited and non-nested, so the same DEV fold system supports tuning and reporting; the locked test did not select settings but was historically exposed to final scoring. Applied settings come from the saved bundle. Legacy `tuning_rationale` prose is not displayed because its causal/overfit explanations are unsupported by the recorded trial metrics.

Full and Top-2 use the same historical test row identities. Top-2 reuses the full model's frozen estimator settings and was not independently tuned. Current supplemental results are read from the pack rather than copied into source code.

Permutation and encoded importance measure fitted-model reliance, not causal wage effects, fairness or production readiness. Empirical q90 is computed from absolute error on the same historical test population; displayed coverage is descriptive, not an independently calibrated guarantee.

The three examples are **historical benchmark records**. They remain included in the 298-row metrics, were already exposed to scoring, and are explicitly marked `pristine=false`.

## Page 06 — Scenario policy and queue

Inputs are controlled selectboxes plus a finite numeric experience control; there is no free-text scenario input. Allowed title/category pairs are observed in DEV and are labelled as observed associations—not a curated business taxonomy. Experience min/max/median is title-level DEV evidence; fractional medians are retained.

The strict policy is common to manual scenarios, quick-load examples and default growth curves. Manual scenarios cannot bypass it. Extended growth points and immutable benchmark rows may use an explicit experience-only acknowledgement within 0–15 years; category/title/model checks remain mandatory. Exceptions are labelled `experience_exception`, carry `experience_outside_observed_dev_range`, and do not inherit a claim of validated historical-q90 coverage.

The queue is session-local and capped at 100. Adding or clearing a row invalidates previous results. Workspace, evidence, policy or model changes clear all queue/results/acknowledgements. Duplicate profiles retain unique scenario IDs. The whole batch is validated before one prediction call; invalid batches produce no partial new results.

Result charts display ten scenarios at a time, while tables and downloads retain the entire queue. Known historical actual markers remain at their true values even outside whiskers. Lower bounds are clipped at zero.

## Downloads

Page 04 offers byte-exact active-pack candidate summary, fold metrics and fold membership. Page 05 offers the tuning summary plus every available anchor/sweep table. When a complete source-compatible training audit exists under `outputs/08_full_pipeline/logs/`, both pages additionally show the pipeline/audit run IDs and offer its manifest, relevant fold/tuning CSVs and complete JSONL trace. The reader requires completed log/console/coverage status, a matching raw-source SHA-256, contained nonsymlinked paths and valid export checksums. Audit traces are labelled historical primary-pipeline evidence and never replace or blend with active supplemental calculations. Missing/incompatible logs remain optional unavailable states and never trigger training.

### `salary_prediction_audit_v1.csv`

Includes evidence/model/policy IDs, scenario inputs, validation mode/exception reason, prediction, bounds, empirical q90 basis, optional actual, absolute error and **signed variance**:

```text
100 × (predicted − actual) / actual
```

The older generic `error_pct` field is not reused. Unknown actuals have blank errors; zero actual has absolute error but blank percentage.

### `salary_scenarios_source_schema_v1.csv`

Uses the exact original 25-column enterprise order. Category maps back to `AI Engineering`. Unknown manual attributes—including observed salary and posting dates—remain blank. Predictions never replace observed salary fields. `SCENARIO-*` IDs are visibly synthetic; this interchange file is not a completed training dataset.

## Missing or incompatible evidence

Each page fails locally with an actionable offline command. Page 06 never passes two columns to the 13-feature bundle and never uses static fallback metrics. Optional missing tuning history does not disable valid inference.

The existing administrator upload sidebar can run the full pipeline in Streamlit; that inherited workflow is outside this feature. It is not used by Pages 04–06 or by the supplemental evidence command.

## Scientific limitations

- The data contains contradictory and synthetic-looking structure and does not establish real labor-market causality.
- Historical candidate test diagnostics do not restore unseen-test purity or repair previous test exposure.
- Existing tuning is not outer-nested, and family/tuning ranking metrics differ.
- Top-2 is a controlled fixed-feature experiment, not proof that two inputs are universally sufficient.
- Scenario predictions and curves are academic model outputs, not salary guarantees or causal seniority effects.
